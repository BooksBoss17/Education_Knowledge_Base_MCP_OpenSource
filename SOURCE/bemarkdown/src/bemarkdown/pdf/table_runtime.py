"""Lazy, registry-backed adapter from installed Paddle table models to TableEngine."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ..pdf_document_ir import document_asset_source_ref
from ..pdf_table_engine import assess_detector_grid, assess_structure, normalize_table_classification


class RegistryBackedTableEvidenceProvider:
    """Use the validated wired/wireless model family with no Developer dependencies."""

    def __init__(self, work_root, *, models_root=None, config_path=None, mcp_root=None):
        self.work_root = Path(work_root).resolve()
        self.registry_options = {
            "models_root": models_root,
            "config_path": config_path,
            "mcp_root": mcp_root,
        }
        self._models = {}
        self._fingerprints = {}
        self._ocr = None
        self._paddle = None
        self._counts = {"tables": 0, "runtime_failures": 0, "model_loads": 0}

    def __call__(self, block, asset):
        path = Path(document_asset_source_ref(asset) or "").resolve(strict=True)
        path.relative_to(self.work_root)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != asset["content_sha256"]:
            raise RuntimeError("TABLE_SOURCE_CROP_HASH_MISMATCH")
        with Image.open(path) as image:
            width, height = image.size
        evidence = {
            "table_id": block["node_id"],
            "document_node_id": block["node_id"],
            "document_id": block["document_id"],
            "page_index": block["page_index"],
            "source_crop_ref": str(path),
            "source_crop_sha256": digest,
            "source_bbox_pdf_pt": list(block["bbox_pdf_pt"]),
            "crop_width": width,
            "crop_height": height,
            "source_truth": None,
            "classification": {
                "normalized_label": "UNKNOWN",
                "selected_branch": "WIRED",
                "score": None,
            },
            "structure": {"tokens": [], "locations_1000": []},
            "cell_detection": {"boxes": []},
            "ocr": {"boxes": []},
        }
        self._counts["tables"] += 1
        try:
            classification = self._predict(
                "pp-lcnet-x1-0-table-cls", path, ("label_names", "scores")
            )
            raw = classification["raw"]
            normalized = normalize_table_classification(
                raw.get("label_names", []), raw.get("scores", [])
            )
            evidence["classification"] = {**classification, **normalized}
            branch = normalized["selected_branch"].lower()
            structure = self._predict(
                f"slanext-{branch}", path, ("structure", "bbox", "structure_score")
            )
            evidence["structure"] = {
                **structure,
                "tokens": structure["raw"].get("structure", []),
                "locations_1000": [],
                "geometry_source": "DETECTOR_GRID",
            }
            cells = self._predict(
                f"rt-detr-l-{branch}-table-cell-det", path, ("boxes",)
            )
            evidence["cell_detection"] = {
                **cells,
                "boxes": cells["raw"].get("boxes", []),
            }
            if self._ocr is None:
                from ..pdf_content_router import PaddleXPdfOcrRuntime

                self._ocr = PaddleXPdfOcrRuntime(**self.registry_options)
            bbox = block["bbox_pdf_pt"]
            lines = self._ocr.recognize(
                {
                    "path": str(path),
                    "content_sha256": digest,
                    "width": width,
                    "height": height,
                    "bbox_pdf_pt": bbox,
                    "scale_x": width / (bbox[2] - bbox[0]),
                    "scale_y": height / (bbox[3] - bbox[1]),
                }
            )
            identity = self._ocr.fingerprint()
            evidence["ocr"] = {
                "boxes": [
                    {
                        "bbox": line["bbox_local_px"],
                        "text": line["text"],
                        "confidence": line.get("confidence"),
                    }
                    for line in lines
                ],
                **{
                    key: identity[key]
                    for key in (
                        "det_model_id",
                        "det_model_fingerprint",
                        "rec_model_id",
                        "rec_model_fingerprint",
                    )
                },
            }
            refined, diagnostics = _refine_grid_geometry(
                path, evidence["cell_detection"]["boxes"], evidence["ocr"]["boxes"],
                evidence["structure"]["tokens"],
            )
            evidence["cell_detection"]["boxes"] = refined
            evidence["cell_detection"]["source_border_refinement"] = diagnostics
            from .source_table_grid import recover_source_table_grid

            recovered, source_grid = recover_source_table_grid(
                path, refined, evidence["structure"]["tokens"],
            )
            evidence["structure"]["source_grid_recovery"] = source_grid
            if recovered is not None:
                evidence["structure"]["original_model_tokens"] = evidence["structure"]["tokens"]
                evidence["structure"]["tokens"] = recovered
                excluded = set(source_grid.get("excluded_source_void_indices", []))
                if excluded:
                    evidence["cell_detection"]["source_void_boxes"] = [
                        {"index": index, "box": box}
                        for index, box in enumerate(refined) if index in excluded
                    ]
                    evidence["cell_detection"]["boxes"] = [
                        box for index, box in enumerate(refined) if index not in excluded
                    ]
            final_grid = assess_detector_grid(
                evidence["structure"]["tokens"], evidence["cell_detection"]["boxes"],
                (width, height), diagnostics.get("excluded_blank_separator_bboxes", []),
            )
            if final_grid["status"] == "STRUCTURE_VALID":
                from .table_source_markers import recover_dashed_cell_markers

                model_box_count = len(evidence["ocr"]["boxes"])
                markers = recover_dashed_cell_markers(
                    path, evidence["cell_detection"]["boxes"], evidence["ocr"]["boxes"]
                )
                evidence["ocr"]["source_marker_recovery"] = {
                    "policy": "observed-dashed-cell-markers-v1",
                    "source_crop_sha256": digest,
                    "model_ocr_box_count": model_box_count,
                    "recovered_marker_count": len(markers),
                }
                evidence["ocr"]["boxes"].extend(markers)
        except Exception as exc:  # noqa: BLE001 - preserve table image and expose failed inference
            self._counts["runtime_failures"] += 1
            # Partial model output must not become a clean structured table.
            evidence["structure"] = {
                "tokens": [],
                "locations_1000": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            evidence["runtime_failure"] = True
        diagnostics = self.work_root / "table_evidence"
        diagnostics.mkdir(parents=True, exist_ok=True)
        evidence_id = hashlib.sha256(block["node_id"].encode("utf-8")).hexdigest()[:24]
        (diagnostics / f"{evidence_id}.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return evidence

    def _predict(self, model_id, path, fields):
        if model_id not in self._models:
            self._load_model(model_id)
        started = time.perf_counter()
        results = list(self._models[model_id].predict(str(path), batch_size=1))
        if len(results) != 1:
            raise RuntimeError(f"TABLE_MODEL_RESULT_CARDINALITY:{model_id}")
        result = results[0]
        raw = result.json.get("res", {}) if hasattr(result, "json") else result
        return {
            "raw": {key: _plain(raw[key]) for key in fields if key in raw},
            "model_id": model_id,
            "model_fingerprint": self._fingerprints[model_id],
            "latency_seconds": time.perf_counter() - started,
            "error": None,
        }

    def _load_model(self, model_id):
        from ..model_registry import PADDLE_MODEL_CATALOG, ModelRegistry
        from ..model_runtime import paddle_compatible_model_dir
        from ..paddlex_runtime import import_paddlex_for_paddle_provider

        os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] = "True"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        paddle, _paddlex, create_model = import_paddlex_for_paddle_provider()
        if (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.cuda.device_count() < 1
        ):
            raise RuntimeError("TABLE_MODELS_REQUIRE_GPU")
        paddle.set_device("gpu:0")
        self._paddle = paddle
        resolution = ModelRegistry(**self.registry_options).resolve(model_id, deep=True)
        self._models[model_id] = create_model(
            PADDLE_MODEL_CATALOG[model_id].directory,
            model_dir=str(paddle_compatible_model_dir(resolution.model_root)),
            device="gpu:0",
        )
        self._fingerprints[model_id] = resolution.manifest["model_fingerprint"]
        self._counts["model_loads"] += 1

    def metrics(self):
        return {
            **copy.deepcopy(self._counts),
            "models": copy.deepcopy(self._fingerprints),
            "device": "gpu:0",
            "cpu_fallback": False,
        }

    def close(self):
        if self._ocr is not None:
            self._ocr.close()
            self._ocr = None
        self._models.clear()
        gc.collect()
        if self._paddle is not None:
            self._paddle.device.cuda.empty_cache()


def _refine_grid_geometry(path, cells, ocr, tokens):
    """Snap detector edges only to long dark rules measured in the source crop.

    Blank strips between double horizontal rules are excluded only when they
    contain no OCR or ink and exactly explain the surplus against topology.
    Raw model boxes remain in evidence['cell_detection']['raw'].
    """
    import numpy as np

    with Image.open(path) as image:
        dark = np.asarray(image.convert("L")) < 200
    height, width = dark.shape
    refined = copy.deepcopy(cells)
    snaps = []

    def snap(box, side):
        axis = side % 2
        other = 1 - axis
        limit = width if axis == 0 else height
        cross_limit = height if axis == 0 else width
        start = max(0, min(cross_limit, int(box[other]) + 1))
        end = max(start, min(cross_limit, int(box[other + 2]) - 1))
        if end - start < 8:
            return None
        radius = max(2, (box[axis + 2] - box[axis]) * 0.18)
        lo = max(0, int(box[side] - radius))
        hi = min(limit, int(box[side] + radius) + 1)
        if lo >= hi:
            return None
        strip = dark[start:end, lo:hi] if axis == 0 else dark[lo:hi, start:end]
        scores = strip.mean(axis=0 if axis == 0 else 1)
        indices = np.flatnonzero(scores >= 0.85)
        if not len(indices):
            return None
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        centers = [float(group.mean()) + lo for group in groups]
        return min(centers, key=lambda value: abs(value - box[side]))

    for index, cell in enumerate(refined):
        original = cell.get("coordinate")
        if not isinstance(original, list) or len(original) != 4:
            continue
        box = list(original)
        for side in (0, 2, 1, 3):
            edge = snap(box, side)
            if edge is not None:
                box[side] = edge
        if box[0] < box[2] and box[1] < box[3] and box != original:
            cell["coordinate"] = box
            snaps.append({"cell_index": index, "before": original, "after": box})
    heights = [cell["coordinate"][3] - cell["coordinate"][1] for cell in refined
               if isinstance(cell.get("coordinate"), list) and len(cell["coordinate"]) == 4]
    median_height = float(np.median(heights)) if heights else 0
    separators = []
    for index, cell in enumerate(refined):
        box = cell.get("coordinate", [])
        if len(box) != 4 or not (2 < box[3] - box[1] < median_height * 0.2):
            continue
        if box[2] - box[0] < 5 * (box[3] - box[1]):
            continue
        if any(box[0] <= (row["bbox"][0] + row["bbox"][2]) / 2 <= box[2]
               and box[1] <= (row["bbox"][1] + row["bbox"][3]) / 2 <= box[3]
               for row in ocr if str(row.get("text") or "").strip()):
            continue
        x0, y0 = max(0, int(box[0]) + 2), max(0, int(box[1]) + 2)
        x1, y1 = min(width, int(box[2]) - 1), min(height, int(box[3]) - 1)
        interior = dark[y0:y1, x0:x1]
        if interior.size and interior.mean() < 0.03:
            top, bottom = round(box[1]), round(box[3])
            if (0 <= top < bottom < height and x1 > x0
                    and dark[top, x0:x1].mean() >= 0.85
                    and dark[bottom, x0:x1].mean() >= 0.85):
                separators.append(index)
    expected = len(assess_structure(tokens, [], (width, height))["cells"])
    removed = separators if expected > 0 and len(refined) - len(separators) == expected else []
    candidate = [cell for index, cell in enumerate(refined) if index not in removed]
    diagnostics = {
        "policy": "source-rules-and-blank-double-rule-strips-v1",
        "edge_snaps": snaps, "excluded_blank_separator_indices": removed,
        "excluded_blank_separator_bboxes": [refined[index]["coordinate"] for index in removed],
        "raw_cells_preserved": True, "ocr_boxes_removed": 0,
    }
    if snaps and assess_detector_grid(tokens, cells, (width, height))["status"] == "STRUCTURE_VALID":
        assessed = assess_detector_grid(
            tokens, candidate, (width, height), diagnostics["excluded_blank_separator_bboxes"]
        )
        if assessed["status"] != "STRUCTURE_VALID":
            # A rule visible only along the header can move one edge away from
            # otherwise agreeing rows. Keep the corroborated original grid.
            return copy.deepcopy(cells), {
                "policy": diagnostics["policy"],
                "edge_snaps": [],
                "excluded_blank_separator_indices": [],
                "excluded_blank_separator_bboxes": [],
                "raw_cells_preserved": True,
                "ocr_boxes_removed": 0,
                "selection": "RAW_DETECTOR_GRID_PRESERVED",
                "rejected_refinement": diagnostics,
                "rejection_reason_codes": assessed["reason_codes"],
            }
    return candidate, diagnostics


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value
