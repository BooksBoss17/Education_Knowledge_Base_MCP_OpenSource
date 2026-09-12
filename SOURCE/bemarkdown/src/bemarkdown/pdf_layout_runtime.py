"""Production-owned PP-DocLayout runtime and page-render contracts."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from .model_registry import ModelRegistry
from .pdf_region_ir import (
    LAYOUT_LABEL_MAPPING,
    PageRenderTransform,
    raw_detection_id,
)

LAYOUT_MODEL_ID = "pp-doclayout-plus-l"
LAYOUT_MODEL_FINGERPRINT = (
    "4cbd1c9582e8a738539fce3a28e355ed5627fddeea424cdb461e3b9bcdfd138d"
)
CAPTURE_FLOOR = 0.3
LAYOUT_RENDER_DPI = 200
LAYOUT_RENDER_ENCODING = "JPEG"
LAYOUT_RENDER_JPEG_QUALITY = 92


@dataclass(frozen=True, slots=True)
class LayoutPageRender:
    """One immutable rendered page ready for layout inference and RegionIR."""

    source_path: str
    source_sha256: str
    page_index: int
    page_number: int
    image_bytes: bytes
    render_sha256: str
    transform: PageRenderTransform
    structural_geometry: dict[str, list[list[float]]]
    schema: str = "bemarkdown-production-layout-page-render-v1"
    encoding: str = LAYOUT_RENDER_ENCODING
    jpeg_quality: int = LAYOUT_RENDER_JPEG_QUALITY
    jpeg_optimize: bool = True
    annotations: bool = True

    @property
    def render_bytes(self) -> int:
        return len(self.image_bytes)

    def to_page_record(self, *, include_image_bytes: bool = False) -> dict[str, Any]:
        """Return a serializable record while keeping image bytes opt-in."""

        record: dict[str, Any] = {
            "schema": self.schema,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "page_render_sha256": self.render_sha256,
            "render_bytes": self.render_bytes,
            "render_transform": self.transform.to_dict(),
            "structural_geometry": self.structural_geometry,
            "render_contract": {
                "dpi": self.transform.dpi,
                "color_space": self.transform.color_space,
                "alpha": self.transform.alpha,
                "annotations": self.annotations,
                "encoding": self.encoding,
                "jpeg_quality": self.jpeg_quality,
                "jpeg_optimize": self.jpeg_optimize,
            },
        }
        if include_image_bytes:
            record["image_bytes"] = self.image_bytes
        return record


class ProductionPdfPageRenderer:
    """Render raw PDF pages with the frozen layout-authority contract."""

    def render(
        self,
        source_pdf: str | Path,
        page_indices: Sequence[int] | None = None,
    ) -> list[LayoutPageRender]:
        import fitz
        from PIL import Image

        source = Path(source_pdf).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        source_sha256 = _sha256_file(source)
        with fitz.open(source) as document:
            if document.needs_pass:
                raise ValueError("ENCRYPTED_PDF_REQUIRES_PASSWORD")
            indices = (
                list(range(document.page_count))
                if page_indices is None
                else [int(index) for index in page_indices]
            )
            if len(indices) != len(set(indices)):
                raise ValueError("DUPLICATE_LAYOUT_RENDER_PAGE_INDEX")
            ordered = sorted(indices)
            for page_index in ordered:
                if page_index < 0 or page_index >= document.page_count:
                    raise IndexError(
                        f"page {page_index} outside 0..{document.page_count - 1}"
                    )
            rows = []
            for page_index in ordered:
                page = document[page_index]
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(
                        LAYOUT_RENDER_DPI / 72.0,
                        LAYOUT_RENDER_DPI / 72.0,
                    ),
                    colorspace=fitz.csRGB,
                    alpha=False,
                    annots=True,
                )
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                stream = BytesIO()
                image.save(
                    stream,
                    format=LAYOUT_RENDER_ENCODING,
                    quality=LAYOUT_RENDER_JPEG_QUALITY,
                    optimize=True,
                )
                payload = stream.getvalue()
                transform = PageRenderTransform.create(
                    page_width_pt=page.rect.width,
                    page_height_pt=page.rect.height,
                    render_width=pixmap.width,
                    render_height=pixmap.height,
                    dpi=LAYOUT_RENDER_DPI,
                    rotation=page.rotation,
                )
                rows.append(
                    LayoutPageRender(
                        source_path=str(source),
                        source_sha256=source_sha256,
                        page_index=page_index,
                        page_number=page_index + 1,
                        image_bytes=payload,
                        render_sha256=hashlib.sha256(payload).hexdigest(),
                        transform=transform,
                        structural_geometry=_page_structural_geometry(page),
                    )
                )
        return rows


def load_layout_label_inventory(model_root: str | Path) -> dict[str, Any]:
    """Read and normalize the frozen layout model label inventory."""

    model_root = Path(model_root).resolve()
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    labels = []
    for index, raw_label in enumerate(config.get("label_list", [])):
        mapping = LAYOUT_LABEL_MAPPING.get(str(raw_label))
        if mapping is None:
            labels.append(
                {
                    "class_id": index,
                    "raw_label": str(raw_label),
                    "handling_status": "UNKNOWN_REVIEW_REQUIRED",
                    "semantic_type": "UNKNOWN",
                    "semantic_subtype": None,
                    "routing_intent": "REVIEW_REQUIRED",
                    "rationale": "Raw model label is not mapped in bemarkdown-region-v0.",
                }
            )
        else:
            labels.append(
                {
                    "class_id": index,
                    "raw_label": str(raw_label),
                    "handling_status": "MAPPED",
                    **mapping,
                }
            )
    return {
        "schema": "bemarkdown-layout-label-inventory-v1",
        "model_id": LAYOUT_MODEL_ID,
        "model_name": config.get("Global", {}).get(
            "model_name", "PP-DocLayout_plus-L"
        ),
        "model_default_threshold": float(config.get("draw_threshold", 0.5)),
        "label_count": len(labels),
        "labels": labels,
        "source": "formal model config.json label_list",
    }


class FormalPaddleLayoutRuntime:
    """Lazy, GPU-only boundary for the formal MCP PP-DocLayout model."""

    def __init__(
        self,
        *,
        models_root: str | Path | None = None,
        config_path: str | Path | None = None,
        mcp_root: str | Path | None = None,
        capture_floor: float = CAPTURE_FLOOR,
    ) -> None:
        self.models_root = (
            Path(models_root).resolve() if models_root is not None else None
        )
        self.config_path = (
            Path(config_path).resolve() if config_path is not None else None
        )
        self.mcp_root = Path(mcp_root).resolve() if mcp_root is not None else None
        self.capture_floor = float(capture_floor)
        self._model: Any | None = None
        self._paddle: Any | None = None
        self._identity: dict[str, Any] | None = None
        self._model_identity: str | None = None
        self.load_seconds: float | None = None

    def load(self) -> dict[str, Any]:
        if self._model is not None:
            assert self._identity is not None
            return dict(self._identity)
        os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] = "True"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from .paddlex_runtime import import_paddlex_for_paddle_provider

        paddle, paddlex, create_model = import_paddlex_for_paddle_provider()

        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("PP_DOCLAYOUT_REQUIRES_PADDLE_GPU")
        if paddle.device.cuda.device_count() < 1:
            raise RuntimeError("PP_DOCLAYOUT_REQUIRES_GPU_0")
        paddle.set_device("gpu:0")
        if paddle.device.get_device() != "gpu:0":
            raise RuntimeError("CPU fallback is forbidden for Phase 6B-1")
        resolution = ModelRegistry(
            models_root=self.models_root,
            config_path=self.config_path,
            mcp_root=self.mcp_root,
        ).resolve(LAYOUT_MODEL_ID, deep=True)
        fingerprint = resolution.manifest["model_fingerprint"]
        if fingerprint != LAYOUT_MODEL_FINGERPRINT:
            raise RuntimeError("Formal PP-DocLayout fingerprint mismatch")
        inventory = load_layout_label_inventory(resolution.model_root)
        started = time.perf_counter()
        self._model = create_model(
            "PP-DocLayout_plus-L",
            model_dir=str(resolution.model_root),
            device="gpu:0",
        )
        self.load_seconds = time.perf_counter() - started
        self._paddle = paddle
        self._model_identity = f"{LAYOUT_MODEL_ID}@{fingerprint}"
        self._identity = {
            "model_id": LAYOUT_MODEL_ID,
            "display_name": "PP-DocLayout_plus-L",
            "model_root": str(resolution.model_root),
            "model_fingerprint": fingerprint,
            "manifest_path": str(resolution.model_root / "MODEL_MANIFEST.json"),
            "resolution_source": resolution.resolution_source,
            "fingerprint_verified": resolution.fingerprint_verified,
            "integration_status": resolution.readiness,
            "raw_labels": [row["raw_label"] for row in inventory["labels"]],
            "model_default_threshold": inventory["model_default_threshold"],
            "capture_floor": self.capture_floor,
            "device": paddle.device.get_device(),
            "precision": "fp32",
            "python": platform.python_version(),
            "paddlepaddle_gpu": paddle.__version__,
            "paddlex": paddlex.__version__,
            "cuda_runtime": paddle.version.cuda(),
            "cudnn_compiled": paddle.version.cudnn(),
            "cudnn_package": _package_version("nvidia-cudnn-cu12"),
            "load_seconds": round(self.load_seconds, 6),
            "network_model_download": False,
            "cpu_fallback": False,
        }
        return dict(self._identity)

    def predict_path(
        self,
        path: str | Path,
        *,
        document_id: str = "integration-smoke",
        page_index: int = 0,
        page_render_identity: str | None = None,
    ) -> list[dict[str, Any]]:
        path = Path(path).resolve()
        page_render_identity = page_render_identity or _sha256_file(path)
        return self._predict(
            str(path),
            document_id=document_id,
            page_index=page_index,
            page_render_identity=page_render_identity,
        )

    def predict_image(
        self,
        image: Any,
        *,
        document_id: str,
        page_index: int,
        page_render_identity: str,
    ) -> tuple[list[dict[str, Any]], float]:
        from PIL import Image

        if isinstance(image, Image.Image):
            import numpy as np

            image = np.asarray(image.convert("RGB"))
        started = time.perf_counter()
        rows = self._predict(
            image,
            document_id=document_id,
            page_index=page_index,
            page_render_identity=page_render_identity,
        )
        return rows, time.perf_counter() - started

    def _predict(
        self,
        value: Any,
        *,
        document_id: str,
        page_index: int,
        page_render_identity: str,
    ) -> list[dict[str, Any]]:
        if self._model is None:
            self.load()
        assert self._model is not None
        assert self._model_identity is not None
        results = list(
            self._model.predict(
                value,
                batch_size=1,
                threshold=self.capture_floor,
            )
        )
        if len(results) != 1:
            raise RuntimeError(
                f"PP-DocLayout returned {len(results)} results for one page"
            )
        rows = []
        for box in results[0].get("boxes") or []:
            label = str(box["label"])
            score = float(box["score"])
            bbox = [round(float(item), 6) for item in box["coordinate"]]
            rows.append(
                {
                    "raw_detection_id": raw_detection_id(
                        document_id=document_id,
                        page_index=page_index,
                        raw_label=label,
                        score=score,
                        bbox_render_px=bbox,
                        model_identity=self._model_identity,
                        page_render_identity=page_render_identity,
                    ),
                    "raw_class_id": int(box["cls_id"]),
                    "raw_label": label,
                    "raw_score": round(score, 8),
                    "raw_bbox_render_px": bbox,
                    "model_identity": self._model_identity,
                    "page_render_identity": page_render_identity,
                }
            )
        rows.sort(
            key=lambda row: (
                row["raw_bbox_render_px"][1],
                row["raw_bbox_render_px"][0],
                row["raw_label"],
                -row["raw_score"],
                row["raw_detection_id"],
            )
        )
        return rows

    def peak_gpu_memory_bytes(self) -> int | None:
        if self._paddle is None:
            return None
        try:
            return int(self._paddle.device.cuda.max_memory_allocated())
        except (AttributeError, RuntimeError, ValueError):
            return None

    def reset_peak_gpu_memory(self) -> None:
        if self._paddle is None:
            return
        try:
            self._paddle.device.cuda.reset_max_memory_allocated()
        except (AttributeError, RuntimeError, ValueError):
            pass

    def unload(self) -> dict[str, Any]:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        status = "UNLOAD_PASS"
        allocation = None
        if self._paddle is not None:
            try:
                self._paddle.device.cuda.empty_cache()
                allocation = int(self._paddle.device.cuda.memory_allocated())
            except (AttributeError, RuntimeError, ValueError):
                status = "UNLOAD_RECORDED_WITHOUT_COUNTER"
        return {
            "unload_status": status,
            "gpu_allocation_after_unload_bytes": allocation,
        }


def _page_structural_geometry(page: Any) -> dict[str, list[list[float]]]:
    text_payload = page.get_text("dict", sort=False)
    text_boxes = [
        _clip_rect(block.get("bbox"), page.rect)
        for block in text_payload.get("blocks", [])
        if block.get("type") == 0
    ]
    try:
        image_boxes = [
            _clip_rect(row.get("bbox"), page.rect)
            for row in page.get_image_info(hashes=False, xrefs=True)
        ]
    except Exception:  # noqa: BLE001 - structural diagnostic isolation
        image_boxes = []
    try:
        vector_boxes = [
            _clip_rect(row.get("rect"), page.rect) for row in page.get_drawings()
        ]
    except Exception:  # noqa: BLE001 - structural diagnostic isolation
        vector_boxes = []
    return {
        "native_text_bboxes_pdf_pt": [row for row in text_boxes if row is not None],
        "native_image_bboxes_pdf_pt": [row for row in image_boxes if row is not None],
        "native_vector_bboxes_pdf_pt": [row for row in vector_boxes if row is not None],
    }


def _clip_rect(value: Any, page_rect: Any) -> list[float] | None:
    import fitz

    try:
        rect = fitz.Rect(value) & page_rect
    except (TypeError, ValueError):
        return None
    if rect.is_empty or rect.is_infinite:
        return None
    return [
        round(float(item), 6)
        for item in (rect.x0, rect.y0, rect.x1, rect.y1)
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
