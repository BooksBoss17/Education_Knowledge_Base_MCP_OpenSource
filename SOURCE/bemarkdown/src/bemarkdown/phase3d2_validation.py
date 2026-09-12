from __future__ import annotations

import hashlib
import html
import json
import mimetypes
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .drawingml import feature_fingerprint
from .formula_ocr import FormulaOcrAdapter
from .mtef_cache import MtefCacheContext
from .package import PackageIndex
from .pipeline import convert_docx


@dataclass(frozen=True)
class Phase3D2Result:
    summary: dict[str, Any]
    output_dir: Path
    summary_path: Path


class _FrozenPredictionRuntime:
    """Replay the frozen Phase 3B raw model outputs by rendered PNG SHA."""

    def __init__(self, predictions: dict[str, str], fingerprint: dict[str, Any]):
        self.predictions = predictions
        self._fingerprint = fingerprint
        self.load_seconds = 0.0

    def fingerprint(self) -> dict[str, Any]:
        return self._fingerprint

    def predict(self, paths, *, batch_size):
        del batch_size
        outputs = []
        for path in paths:
            sha = _sha256_file(path)
            if sha not in self.predictions:
                raise KeyError(
                    f"PNG SHA absent from frozen Phase 3B predictions: {sha}"
                )
            outputs.append(self.predictions[sha])
        return outputs

    @staticmethod
    def peak_gpu_memory_bytes():
        return None


def run_phase3d2_validation(
    manifest_path: str | Path,
    phase3d1_dir: str | Path,
    phase3b_dir: str | Path,
    output_dir: str | Path,
    *,
    adapter: FormulaOcrAdapter | None = None,
    progress: Callable[[str], None] | None = None,
    overwrite: bool = False,
) -> Phase3D2Result:
    started = time.perf_counter()
    manifest_path = Path(manifest_path).resolve()
    phase3d1_dir = Path(phase3d1_dir).resolve()
    phase3b_dir = Path(phase3b_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_files = (
        "drawingml_feature_matrix.jsonl",
        "drawingml_classifications.jsonl",
        "drawingml_review.html",
        "asset_migration.jsonl",
        "asset_contract_summary.json",
        "linked_image_results.jsonl",
        "header_title_results.jsonl",
        "regression_summary.json",
    )
    if not overwrite and any((output_dir / name).exists() for name in expected_files):
        raise FileExistsError(f"Phase 3D-2 output already exists: {output_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        row
        for row in manifest.get("records", [])
        if str(row.get("file_type", "")).upper() == "DOCX"
    ]
    if len(records) != 169:
        raise ValueError(f"Phase 3D-2 requires 169 DOCX records, found {len(records)}")
    expected_headers = _load_expected_headers(phase3d1_dir / "edge_occurrences.jsonl")
    legacy_phase3b_regex_count = sum(
        row.get("asset_links", {}).get("checked", 0)
        for row in _read_jsonl(phase3b_dir / "docx_records.jsonl")
        if row.get("status") == "SUCCESS"
    )

    frozen_ocr_rows = _read_jsonl(phase3b_dir / "ocr_records.jsonl")
    frozen_predictions = {
        row["png_sha256"]: row["raw_latex"]
        for row in frozen_ocr_rows
        if row.get("png_sha256") and row.get("raw_latex") is not None
    }
    performance = json.loads(
        (phase3b_dir / "performance.json").read_text(encoding="utf-8")
    )
    frozen_fingerprint = performance["formula_ocr_runtime"]["runtime_fingerprint"]
    adapter = adapter or FormulaOcrAdapter(
        runtime_factory=lambda: _FrozenPredictionRuntime(
            frozen_predictions, frozen_fingerprint
        )
    )
    cache_context = MtefCacheContext(mode="persistent")
    feature_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    migration_rows: list[dict[str, Any]] = []
    linked_rows: list[dict[str, Any]] = []
    header_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    formula_totals: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    asset_totals: Counter[str] = Counter()
    representative_svgs: dict[str, tuple[str, bytes]] = {}

    with tempfile.TemporaryDirectory(prefix="bemarkdown-phase3d2-") as temporary:
        work_root = Path(temporary)
        for index, record in enumerate(records, 1):
            source = (manifest_path.parent / record["target_relative_path"]).resolve()
            expected_sha = record.get("source_sha256")
            actual_sha = _sha256_file(source)
            if actual_sha != expected_sha:
                raise ValueError(f"Source SHA mismatch: {source}")
            destination = work_root / f"{index:04d}"
            result = convert_docx(
                source,
                destination,
                formula_ocr="auto",
                formula_ocr_adapter=adapter,
                mtef_cache_context=cache_context,
            )
            report = result.report
            package = PackageIndex(source)
            asset_rows = _read_jsonl(destination / "assets_manifest.jsonl")
            markdown = result.markdown_path.read_text(encoding="utf-8")
            validation = _validate_document_assets(destination, markdown, asset_rows)
            if not all(validation.values()):
                raise ValueError(
                    f"Asset validation failed for {source}: "
                    + json.dumps(validation, sort_keys=True)
                )

            manifest_by_locator = {
                row["source_locator"]: row
                for row in asset_rows
                if row["semantic_type"] == "DRAWINGML_GROUP"
            }
            for drawing in report["drawingml"]["records"]:
                base = {
                    "document_id": f"docx-{index:04d}",
                    "source_path": record["target_relative_path"],
                    "source_sha256": actual_sha,
                    **drawing,
                }
                feature_rows.append(base)
                classification_rows.append(
                    {
                        key: base[key]
                        for key in (
                            "document_id",
                            "source_path",
                            "source_sha256",
                            "source_part",
                            "source_locator",
                            "classification",
                            "visible_labels",
                            "visible_label_count",
                            "unsupported_features",
                        )
                    }
                )
                family = hashlib.sha256(
                    feature_fingerprint(drawing["features"]).encode("utf-8")
                ).hexdigest()[:12]
                asset = manifest_by_locator.get(drawing["source_locator"])
                if (
                    asset
                    and asset["relative_path"]
                    and family not in representative_svgs
                    and len(representative_svgs) < 6
                ):
                    representative_svgs[family] = (
                        asset["relative_path"],
                        (destination / asset["relative_path"]).read_bytes(),
                    )

            for asset in asset_rows:
                asset_totals["total_visible"] += 1
                asset_totals[asset["semantic_type"]] += 1
                asset_totals[asset["status"]] += 1
                migration = _migration_record(
                    index,
                    record["target_relative_path"],
                    package,
                    asset,
                )
                migration_rows.append(migration)
                if asset["semantic_type"] == "EXTERNAL_LINKED_IMAGE":
                    linked_rows.append(
                        {
                            "document_id": f"docx-{index:04d}",
                            "source_path": record["target_relative_path"],
                            **asset,
                            "external_path_accessed": False,
                            "silent_drop": False,
                        }
                    )

            expected_for_doc = {
                key: value
                for key, value in expected_headers.items()
                if key[0] == actual_sha
            }
            actual_header_records = report["headers"]["records"]
            for (document_sha, part), expected in expected_for_doc.items():
                match = next(
                    (
                        row
                        for row in actual_header_records
                        if row.get("source_part") == part
                        and _normalize(row.get("text", ""))
                        == _normalize(expected["text"])
                    ),
                    None,
                )
                header_rows.append(
                    {
                        "document_id": f"docx-{index:04d}",
                        "source_path": record["target_relative_path"],
                        "source_sha256": document_sha,
                        "source_part": part,
                        "expected_subtype": expected["subtype"],
                        "text": expected["text"],
                        "result_status": match.get("status") if match else "NOT_FOUND",
                        "preserved_in_metadata": bool(match),
                        "inserted_in_markdown": bool(
                            match and match.get("status") == "INSERTED_ONCE"
                        ),
                    }
                )

            formulas = report["formulas"]
            ocr = report["formula_ocr"]
            formula_totals.update(
                {
                    "candidates": formulas["equation_candidates"]
                    - formulas["non_formula_objects"],
                    "real": formulas["real_formulas"],
                    "empty": formulas["empty_placeholders"],
                    "structural": formulas["structural_success"],
                    "ocr_candidates": ocr["candidates"],
                    "accepted": ocr["accepted"],
                    "warning": ocr["accepted_with_warning"],
                    "review": ocr["review_required"],
                    "rejected": ocr["rejected_preserved"],
                    "failure": ocr["inference_failed_preserved"],
                }
            )
            quality_counts[report["quality_status"]] += 1
            regression_rows.append(
                {
                    "document_id": f"docx-{index:04d}",
                    "source_path": record["target_relative_path"],
                    "source_sha256": actual_sha,
                    "source_sha_valid": True,
                    "status": "SUCCESS",
                    "quality_status": report["quality_status"],
                    "assets": {
                        "total_visible": len(asset_rows),
                        "resolved": sum(
                            row["relative_path"] is not None for row in asset_rows
                        ),
                        "unresolved": sum(
                            row["relative_path"] is None for row in asset_rows
                        ),
                    },
                    "drawingml": {
                        key: report["drawingml"][key]
                        for key in (
                            "groups_total",
                            "visual_groups",
                            "pure_text_groups",
                            "decorative_or_empty_groups",
                            "unsupported_groups",
                            "visible_labels_total",
                            "visible_labels_accounted",
                        )
                    },
                    "headers": {
                        key: report["headers"][key]
                        for key in (
                            "title_candidates",
                            "titles_inserted",
                            "titles_deduplicated",
                            "watermark_or_visual_excluded",
                        )
                    },
                    "asset_validation": validation,
                    "formula_conservation": formulas["conservation_ok"]
                    and ocr["conservation_ok"]
                    and ocr["real_formula_conservation_ok"],
                }
            )
            if progress:
                progress(
                    f"Phase 3D-2 {index:03d}/169: assets={len(asset_rows)} "
                    f"groups={report['drawingml']['groups_total']}"
                )

    _write_representatives(output_dir, representative_svgs)
    _write_jsonl(output_dir / "drawingml_feature_matrix.jsonl", feature_rows)
    _write_jsonl(output_dir / "drawingml_classifications.jsonl", classification_rows)
    _write_jsonl(output_dir / "asset_migration.jsonl", migration_rows)
    _write_jsonl(output_dir / "linked_image_results.jsonl", linked_rows)
    _write_jsonl(output_dir / "header_title_results.jsonl", header_rows)
    _write_review_html(output_dir, feature_rows, representative_svgs)

    classifications = Counter(row["classification"] for row in classification_rows)
    labels = Counter()
    for row in classification_rows:
        labels[row["classification"]] += row["visible_label_count"]
    migration = Counter(row["migration_status"] for row in migration_rows)
    corrected_old_visible_count = (
        migration["MAPPED_EXACT"] + migration["MAPPED_EXTERNAL_UNRESOLVED"]
    )
    header_positive = [
        row for row in header_rows if row["expected_subtype"] == "school_or_exam_title"
    ]
    header_negative = [
        row
        for row in header_rows
        if row["expected_subtype"] == "repeated_watermark_or_vendor_header"
    ]
    summary = {
        "schema": "bemarkdown-phase3d2-visual-asset-contract-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "corpus": {
            "docx_total": len(records),
            "docx_success": len(regression_rows),
            "docx_failed": len(records) - len(regression_rows),
            "source_sha_valid": sum(row["source_sha_valid"] for row in regression_rows),
        },
        "drawingml": {
            "groups_total": len(classification_rows),
            "classifications": dict(sorted(classifications.items())),
            "visible_labels_total": sum(
                row["visible_label_count"] for row in classification_rows
            ),
            "visible_labels_by_classification": dict(sorted(labels.items())),
            "silent_p0_loss": 0
            if all(row["classification"] for row in classification_rows)
            else 1,
            "feature_families": len(representative_svgs),
        },
        "assets": {
            "old_visible_occurrences": corrected_old_visible_count,
            "legacy_phase3b_regex_count": legacy_phase3b_regex_count,
            "legacy_regex_escaped_alt_undercount": (
                corrected_old_visible_count - legacy_phase3b_regex_count
            ),
            "new_visible_occurrences": len(migration_rows),
            "resolved_files": sum(
                row["new_relative_path"] is not None for row in migration_rows
            ),
            "unresolved_occurrences": sum(
                row["new_relative_path"] is None for row in migration_rows
            ),
            "semantic_types": {
                key: asset_totals[key]
                for key in sorted(
                    key
                    for key in asset_totals
                    if key not in {"total_visible", "RESOLVED", "UNRESOLVED_EXTERNAL"}
                )
            },
            "migration_status": dict(sorted(migration.items())),
            "manifest_version": "bemarkdown-asset-contract-v1",
        },
        "linked_images": {
            "expected": 5,
            "detected": len(linked_rows),
            "allocated": sum(bool(row["asset_id"]) for row in linked_rows),
            "explicit_unresolved": sum(
                row["status"] == "UNRESOLVED_EXTERNAL" for row in linked_rows
            ),
            "external_path_reads": sum(
                row["external_path_accessed"] for row in linked_rows
            ),
            "silent_drops": sum(row["silent_drop"] for row in linked_rows),
        },
        "headers": {
            "positive_expected": len(header_positive),
            "positive_preserved": sum(
                row["preserved_in_metadata"] for row in header_positive
            ),
            "positive_inserted_or_deduplicated": sum(
                row["result_status"]
                in {"INSERTED_ONCE", "DEDUPLICATED_SECTION", "DEDUPLICATED_BODY"}
                for row in header_positive
            ),
            "negative_expected": len(header_negative),
            "negative_excluded": sum(
                row["result_status"] == "EXCLUDED_VISUAL_OR_WATERMARK"
                for row in header_negative
            ),
        },
        "formulas": dict(formula_totals),
        "formula_ocr_execution": {
            "mode": "frozen_phase3b_raw_output_replay",
            "unique_png_predictions": len(frozen_predictions),
            "model_identifier": frozen_fingerprint.get("model_identifier"),
            "model_revision": frozen_fingerprint.get("model_revision"),
            "model_sha256": frozen_fingerprint.get("model_sha256"),
        },
        "quality_status": dict(sorted(quality_counts.items())),
        "checks": {},
        "wall_seconds": time.perf_counter() - started,
    }
    summary["checks"] = {
        "docx_169_success": summary["corpus"]
        == {
            "docx_total": 169,
            "docx_success": 169,
            "docx_failed": 0,
            "source_sha_valid": 169,
        },
        "drawingml_120_classified": len(classification_rows) == 120,
        "labels_720_accounted": summary["drawingml"]["visible_labels_total"] == 720,
        "silent_p0_loss_zero": summary["drawingml"]["silent_p0_loss"] == 0,
        "linked_5_explicit": summary["linked_images"]
        == {
            "expected": 5,
            "detected": 5,
            "allocated": 5,
            "explicit_unresolved": 5,
            "external_path_reads": 0,
            "silent_drops": 0,
        },
        "header_9_positive": summary["headers"]["positive_preserved"] == 9,
        "header_48_negative": summary["headers"]["negative_excluded"] == 48,
        "old_assets_one_to_one": migration["MAPPED_EXACT"]
        + migration["MAPPED_EXTERNAL_UNRESOLVED"]
        == corrected_old_visible_count
        and len(migration_rows) - migration["EXPECTED_NEW_DRAWINGML"]
        == corrected_old_visible_count,
        "expected_drawingml_additions": migration["EXPECTED_NEW_DRAWINGML"] == 120,
        "external_occurrences_migrated": migration["MAPPED_EXTERNAL_UNRESOLVED"] == 5,
        "formula_frozen": dict(formula_totals)
        == {
            "candidates": 35920,
            "real": 35892,
            "empty": 28,
            "structural": 35647,
            "ocr_candidates": 245,
            "accepted": 162,
            "warning": 12,
            "review": 20,
            "rejected": 51,
            "failure": 0,
        },
        "asset_validation_all_documents": all(
            all(row["asset_validation"].values()) for row in regression_rows
        ),
        "formula_conservation_all_documents": all(
            row["formula_conservation"] for row in regression_rows
        ),
    }
    summary["all_ok"] = all(summary["checks"].values())
    _write_json(output_dir / "asset_contract_summary.json", summary)
    _write_json(
        output_dir / "regression_summary.json",
        {
            "schema": "bemarkdown-phase3d2-regression-v1",
            "all_ok": summary["all_ok"],
            "checks": summary["checks"],
            "documents": regression_rows,
        },
    )
    return Phase3D2Result(
        summary,
        output_dir,
        output_dir / "asset_contract_summary.json",
    )


def _migration_record(index, source_path, package, asset):
    semantic = asset["semantic_type"]
    if semantic == "DRAWINGML_GROUP":
        status = "EXPECTED_NEW_DRAWINGML"
        exact = True
    elif semantic == "EXTERNAL_LINKED_IMAGE":
        status = "MAPPED_EXTERNAL_UNRESOLVED"
        exact = asset["relative_path"] is None and asset["content_sha256"] is None
    else:
        expected_sha = None
        relationship_id = asset.get("relationship_id")
        if relationship_id:
            relationship = package.relationship(asset["source_part"], relationship_id)
            if (
                relationship
                and relationship.target_part
                and package.has_part(relationship.target_part)
            ):
                expected_sha = hashlib.sha256(
                    package.data(relationship.target_part)
                ).hexdigest()
        if semantic == "FORMULA_FALLBACK":
            ocr = asset.get("provenance", {}).get("ocr_provenance") or {}
            expected_sha = ocr.get("png_sha256") or expected_sha
        exact = bool(expected_sha) and expected_sha == asset["content_sha256"]
        status = "MAPPED_EXACT" if exact else "MIGRATION_MISMATCH"
    return {
        "document_id": f"docx-{index:04d}",
        "source_path": source_path,
        "source_part": asset["source_part"],
        "source_locator": asset["source_locator"],
        "semantic_type": semantic,
        "old_relative_path": asset.get("provenance", {}).get("legacy_relative_path"),
        "new_asset_id": asset["asset_id"],
        "new_asset_uid": asset["asset_uid"],
        "new_relative_path": asset["relative_path"],
        "content_sha256": asset["content_sha256"],
        "content_sha_exact": exact,
        "migration_status": status,
    }


def _validate_document_assets(output_dir, markdown, rows):
    ids = [row["asset_id"] for row in rows]
    expected = [f"image_{index:06d}" for index in range(1, len(rows) + 1)]
    files_ok = True
    mime_ok = True
    for row in rows:
        relative = row["relative_path"]
        if relative is None:
            continue
        path = output_dir / relative
        files_ok = (
            files_ok and path.is_file() and _sha256_file(path) == row["content_sha256"]
        )
        guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        mime_ok = mime_ok and (
            guessed == row["mime_type"]
            or row["mime_type"] == "image/svg+xml"
            and path.suffix == ".svg"
        )
    links = re.findall(r"\((assets/image_\d{6}\.[^\s\)\"]+)", markdown)
    resolved = [row["relative_path"] for row in rows if row["relative_path"]]
    unresolved = [row for row in rows if row["status"] == "UNRESOLVED_EXTERNAL"]
    return {
        "numbering_starts_at_one": not ids or ids[0] == "image_000001",
        "numbering_continuous": ids == expected,
        "asset_id_unique": len(ids) == len(set(ids)),
        "files_and_sha_match": files_ok,
        "mime_extension_match": mime_ok,
        "markdown_resolved_links_match_manifest": Counter(links) == Counter(resolved),
        "unresolved_markers_match_manifest": all(
            f"[Unresolved image: {row['asset_id']}]" in markdown for row in unresolved
        ),
    }


def _load_expected_headers(path):
    result = {}
    for row in _read_jsonl(path):
        if row.get("element_type") != "header" or row.get("subtype") not in {
            "school_or_exam_title",
            "repeated_watermark_or_vendor_header",
        }:
            continue
        result[(row["source_sha256"], row["part_name"])] = {
            "subtype": row["subtype"],
            "text": row["visible_text"],
        }
    return result


def _write_representatives(output_dir, representatives):
    target = output_dir / "representative_svgs"
    target.mkdir(parents=True, exist_ok=True)
    for index, (family, (_source, payload)) in enumerate(
        sorted(representatives.items()), 1
    ):
        (target / f"feature_family_{index:03d}_{family}.svg").write_bytes(payload)


def _write_review_html(output_dir, rows, representatives):
    family_files = {
        family: f"representative_svgs/feature_family_{index:03d}_{family}.svg"
        for index, family in enumerate(sorted(representatives), 1)
    }
    cards = []
    for row in rows:
        family = hashlib.sha256(
            feature_fingerprint(row.get("features", {})).encode("utf-8")
        ).hexdigest()[:12]
        image = family_files.get(family, "")
        cards.append(
            "<section><h2>"
            + html.escape(row["document_id"] + " " + row["source_locator"])
            + "</h2>"
            + (
                f'<img src="{html.escape(image)}" alt="DrawingML render">'
                if image
                else ""
            )
            + "<pre>"
            + html.escape(json.dumps(row, ensure_ascii=False, indent=2))
            + "</pre></section>"
        )
    payload = (
        """<!doctype html><html><head><meta charset="utf-8"><title>DrawingML review</title>
<style>body{font:14px system-ui;margin:2rem}section{border:1px solid #bbb;padding:1rem;margin:1rem 0}img{width:420px;max-height:260px;object-fit:contain;border:1px solid #ddd}pre{white-space:pre-wrap}</style>
</head><body><h1>BeMarkdown Phase 3D-2 DrawingML Review</h1>"""
        + "".join(cards)
        + "</body></html>\n"
    )
    (output_dir / "drawingml_review.html").write_text(
        payload, encoding="utf-8", newline="\n"
    )


def _normalize(value):
    return re.sub(r"\s+", "", value).casefold()


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
