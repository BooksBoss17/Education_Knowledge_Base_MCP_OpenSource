from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .pdf_layout_runtime import (
    CAPTURE_FLOOR,
    LAYOUT_MODEL_FINGERPRINT,
    LAYOUT_MODEL_ID,
    FormalPaddleLayoutRuntime,
    load_layout_label_inventory,
)
from .pdf_region_ir import (
    LAYOUT_LABEL_MAPPING,
    PageRenderTransform,
    apply_reading_order_to_page,
    build_page_region_ir,
    validate_page_region_ir,
)

BASELINE_THRESHOLD = 0.5
PHASE6A_MANIFEST_SHA256 = (
    "3c2503d37d9435d3e56295fd61d2ce7e2bbebcc87ad93ed408313d11ba7ed1dc"
)
PHASE6A_CORPUS_FINGERPRINT = (
    "728eddd6471a8389efc34486223aaf87ea25cac72bf39f873e1da479b1b7b182"
)


def select_reference_pages(
    pages: list[dict[str, Any]],
    *,
    target_count: int = 96,
    hidden_count: int = 24,
) -> list[dict[str, Any]]:
    if target_count > len(pages):
        raise ValueError("Reference target exceeds available pages")
    if hidden_count > target_count:
        raise ValueError("Hidden subset exceeds reference target")
    ordered = sorted(pages, key=_page_key)
    selected: dict[tuple[str, int], dict[str, Any]] = {}

    def retain(row: dict[str, Any], reason: str, *, hidden: bool) -> None:
        key = (str(row["document_id"]), int(row["page_index"]))
        if key in selected:
            selected[key]["selection_reasons"].append(reason)
            selected[key]["prediction_hidden"] = (
                selected[key]["prediction_hidden"] or hidden
            )
            return
        if len(selected) >= target_count:
            return
        selected[key] = _selection_record(row, reason=reason, hidden=hidden)

    for document_id in sorted({str(row["document_id"]) for row in ordered}):
        candidate = next(row for row in ordered if row["document_id"] == document_id)
        retain(candidate, "DOCUMENT_COVERAGE", hidden=True)
    for field, values in (
        (
            "routing_decision",
            ("NATIVE_FIRST", "VISUAL_REQUIRED", "HYBRID_REQUIRED", "REVIEW_REQUIRED"),
        ),
        ("source_profile", ("NATIVE_TEXT", "IMAGE_ONLY", "IMAGE_WITH_TEXT_LAYER", "MIXED_NATIVE_VISUAL", "UNCERTAIN")),
    ):
        for value in values:
            candidate = next((row for row in ordered if row.get(field) == value), None)
            if candidate is not None:
                retain(candidate, f"{field.upper()}_{value}", hidden=True)
    for risk in ("LOW", "MEDIUM", "HIGH"):
        candidate = next(
            (row for row in ordered if row.get("reading_order", {}).get("risk") == risk),
            None,
        )
        if candidate is not None:
            retain(candidate, f"READING_ORDER_{risk}", hidden=True)
    for row in ordered:
        if sum(value["prediction_hidden"] for value in selected.values()) >= hidden_count:
            break
        retain(row, "PREDICTION_HIDDEN_FILL", hidden=True)

    diagnostic_priority = sorted(
        ordered,
        key=lambda row: (
            0 if row.get("sanity_issue_codes") else 1,
            0 if row.get("semantic_type_counts", {}).get("FORMULA") else 1,
            0 if row.get("semantic_type_counts", {}).get("TABLE") else 1,
            0 if row.get("semantic_type_counts", {}).get("IMAGE") else 1,
            -int(row.get("canonical_region_count", 0)),
            _page_key(row),
        ),
    )
    for row in diagnostic_priority:
        if len(selected) >= target_count:
            break
        retain(row, "STRATIFIED_LAYOUT_DIAGNOSTIC", hidden=False)
    result = list(selected.values())
    result.sort(key=lambda row: (not row["prediction_hidden"], row["document_id"], row["page_index"]))
    for index, row in enumerate(result, start=1):
        row["reference_index"] = index
        row["selection_reasons"] = sorted(set(row["selection_reasons"]))
    return result


def summarize_reference_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "page_count": len(rows),
        "prediction_hidden_count": sum(row["prediction_hidden"] for row in rows),
        "document_count": len({row["document_id"] for row in rows}),
        "source_groups": dict(sorted(Counter(row["source_group"] for row in rows).items())),
        "routes": dict(sorted(Counter(row["routing_decision"] for row in rows).items())),
        "source_profiles": dict(sorted(Counter(row["source_profile"] for row in rows).items())),
        "reading_order_risk": dict(sorted(Counter(row["reading_order_risk"] for row in rows).items())),
    }


def prepare_phase6b1_render_cache(
    *,
    phase6a_artifact: str | Path,
    cache_dir: str | Path,
    dpi: int = 200,
    progress: Any | None = None,
) -> dict[str, Any]:
    import fitz

    phase6a_artifact = Path(phase6a_artifact).resolve()
    cache_dir = Path(cache_dir).resolve()
    manifest_path = phase6a_artifact / "pdf_corpus_manifest.json"
    if _sha256_file(manifest_path) != PHASE6A_MANIFEST_SHA256:
        raise RuntimeError("PHASE6A_MANIFEST_DRIFT")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    routing = json.loads(
        (phase6a_artifact / "routing_summary.json").read_text(encoding="utf-8")
    )
    if (
        len(manifest.get("documents", [])) != 18
        or routing["overall"]["page_count"] != 1228
        or routing["overall"]["corpus_fingerprint"]
        != PHASE6A_CORPUS_FINGERPRINT
    ):
        raise RuntimeError("PHASE6A_CORPUS_IDENTITY_MISMATCH")
    pages = _read_jsonl(phase6a_artifact / "page_records.jsonl")
    if len(pages) != 1228:
        raise RuntimeError("PHASE6A_PAGE_RECORD_COUNT_MISMATCH")
    _validate_phase6a_sources(manifest["documents"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    render_dir = cache_dir / f"pages_{dpi}dpi"
    render_dir.mkdir(parents=True, exist_ok=True)
    render_manifest_path = cache_dir / f"render_manifest_{dpi}dpi.jsonl"
    existing = {
        (row["document_id"], row["page_index"]): row
        for row in _read_jsonl(render_manifest_path)
        if Path(row.get("render_path", "")).is_file()
    }
    phase6a_by_key = {
        (row["document_id"], row["page_index"]): row for row in pages
    }
    records: list[dict[str, Any]] = []
    render_seconds = 0.0
    reused = 0
    processed = 0
    for document in manifest["documents"]:
        source = Path(document["source_path"])
        with fitz.open(source) as pdf:
            if pdf.page_count != document["page_count"]:
                raise RuntimeError(f"PAGE_COUNT_DRIFT: {source}")
            for page_index in range(pdf.page_count):
                key = (document["document_id"], page_index)
                prior = existing.get(key)
                if prior is not None and _sha256_file(Path(prior["render_path"])) == prior["page_render_sha256"]:
                    records.append(prior)
                    reused += 1
                    continue
                page_record = phase6a_by_key[key]
                page = pdf[page_index]
                target = render_dir / (
                    f"{document['sha256'][:12]}_p{page_index + 1:04d}_{dpi}dpi.jpg"
                )
                started = time.perf_counter()
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
                    colorspace=fitz.csRGB,
                    alpha=False,
                    annots=True,
                )
                pixmap.pil_save(target, format="JPEG", quality=92, optimize=True)
                local_seconds = time.perf_counter() - started
                render_seconds += local_seconds
                transform = PageRenderTransform.create(
                    page_width_pt=page.rect.width,
                    page_height_pt=page.rect.height,
                    render_width=pixmap.width,
                    render_height=pixmap.height,
                    dpi=dpi,
                    rotation=page.rotation,
                )
                record = {
                    "document_id": document["document_id"],
                    "source_path": str(source),
                    "source_sha256": document["sha256"],
                    "source_group": page_record["source_group"],
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "source_profile": page_record["source_profile"],
                    "routing_decision": page_record["routing_decision"],
                    "native_text_trust": page_record["native_text_trust"],
                    "reading_order": page_record["reading_order"],
                    "phase6a_non_whitespace_chars": page_record["native_text"]["non_whitespace_char_count"],
                    "render_path": str(target),
                    "page_render_sha256": _sha256_file(target),
                    "render_seconds": round(local_seconds, 6),
                    "render_transform": transform.to_dict(),
                    "structural_geometry": _page_structural_geometry(page),
                }
                records.append(record)
                processed += 1
                if progress is not None and (len(records) % 25 == 0 or len(records) == 1228):
                    progress(
                        f"Render {len(records):04d}/1228 processed={processed} reused={reused}"
                    )
                if len(records) % 25 == 0:
                    _write_jsonl_atomic(render_manifest_path, records)
    records.sort(key=_page_key)
    _write_jsonl_atomic(render_manifest_path, records)
    summary = {
        "schema": "bemarkdown-phase6b1-render-cache-v1",
        "render_contract": "pdf-page-render-v0",
        "dpi": dpi,
        "color_space": "RGB",
        "alpha": False,
        "rotation_normalized": True,
        "backend": {"library": "PyMuPDF", "version": fitz.VersionBind},
        "document_count": 18,
        "page_count": len(records),
        "rendered_this_run": processed,
        "reused_from_verified_cache": reused,
        "render_seconds_this_run": round(render_seconds, 6),
        "phase6a_manifest_sha256": PHASE6A_MANIFEST_SHA256,
        "phase6a_corpus_fingerprint": PHASE6A_CORPUS_FINGERPRINT,
        "manifest_path": str(render_manifest_path),
    }
    _write_json_atomic(cache_dir / f"render_summary_{dpi}dpi.json", summary)
    return summary


def run_full_layout_inference(
    *,
    render_manifest_path: str | Path,
    phase6a_artifact: str | Path,
    output_dir: str | Path,
    mcp_root: str | Path,
    capture_floor: float = CAPTURE_FLOOR,
    baseline_threshold: float = BASELINE_THRESHOLD,
    reference_count: int = 96,
    hidden_reference_count: int = 24,
    progress: Any | None = None,
) -> dict[str, Any]:
    from PIL import Image

    started = time.perf_counter()
    render_manifest_path = Path(render_manifest_path).resolve()
    phase6a_artifact = Path(phase6a_artifact).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    render_records = _read_jsonl(render_manifest_path)
    if len(render_records) != 1228:
        raise RuntimeError("PHASE6B1_REQUIRES_1228_RENDER_RECORDS")
    for record in render_records:
        render_path = Path(record["render_path"])
        if not render_path.is_file() or _sha256_file(render_path) != record["page_render_sha256"]:
            raise RuntimeError(f"RENDER_CACHE_DRIFT: {render_path}")
    phase6a_manifest_path = phase6a_artifact / "pdf_corpus_manifest.json"
    if _sha256_file(phase6a_manifest_path) != PHASE6A_MANIFEST_SHA256:
        raise RuntimeError("PHASE6A_MANIFEST_DRIFT")

    runtime = FormalPaddleLayoutRuntime(
        mcp_root=mcp_root, capture_floor=capture_floor
    )
    identity = runtime.load()
    model_root = Path(identity["model_root"])
    label_inventory = load_layout_label_inventory(model_root)
    if any(row["handling_status"] != "MAPPED" for row in label_inventory["labels"]):
        raise RuntimeError("UNMAPPED_FROZEN_LAYOUT_LABEL")
    runtime.reset_peak_gpu_memory()
    render_contract = {
        "schema": "bemarkdown-pdf-page-render-contract-v0",
        "contract": "pdf-page-render-v0",
        "backend": "PyMuPDF",
        "dpi": render_records[0]["render_transform"]["dpi"],
        "color_space": "RGB",
        "alpha": False,
        "rotation_normalized": True,
        "canonical_coordinate_system": "pdf-points-top-left-rotation-normalized",
        "bbox_spaces": ["bbox_pdf_pt", "bbox_normalized", "bbox_render_px"],
        "mapping": "x_pdf=x_px/scale_x; y_pdf=y_px/scale_y; inverse=multiply",
    }
    label_mapping = {
        "schema": "bemarkdown-layout-label-mapping-v0",
        "source_model": LAYOUT_MODEL_ID,
        "target_taxonomy": "bemarkdown-region-v0",
        "mappings": [
            {"raw_label": label, **LAYOUT_LABEL_MAPPING[label]}
            for label in identity["raw_labels"]
        ],
    }
    _write_json_atomic(output_dir / "render_contract.json", render_contract)
    _write_json_atomic(output_dir / "layout_label_inventory.json", label_inventory)
    _write_json_atomic(output_dir / "label_mapping.json", label_mapping)

    raw_path = output_dir / "raw_layout_predictions.jsonl"
    ir_path = output_dir / "page_region_ir.jsonl"
    raw_handle = raw_path.open("w", encoding="utf-8", newline="\n")
    ir_handle = ir_path.open("w", encoding="utf-8", newline="\n")
    page_summaries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    inference_times: list[float] = []
    normalization_times: list[float] = []
    order_times: list[float] = []
    raw_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    route_region_counts: dict[str, Counter[str]] = {}
    document_region_counts: Counter[str] = Counter()
    try:
        for index, record in enumerate(render_records, start=1):
            infer_started = time.perf_counter()
            raw = runtime.predict_path(
                record["render_path"],
                document_id=record["document_id"],
                page_index=record["page_index"],
                page_render_identity=record["page_render_sha256"],
            )
            inference_seconds = time.perf_counter() - infer_started
            normalize_started = time.perf_counter()
            transform = PageRenderTransform(**record["render_transform"])
            page_ir = build_page_region_ir(
                document_id=record["document_id"],
                page_index=record["page_index"],
                page_route=record["routing_decision"],
                transform=transform,
                raw_detections=raw,
                score_threshold=baseline_threshold,
                structural_geometry=record["structural_geometry"],
                assign_reading_order=False,
            )
            normalization_seconds = time.perf_counter() - normalize_started
            order_started = time.perf_counter()
            apply_reading_order_to_page(page_ir)
            order_seconds = time.perf_counter() - order_started
            local_issues = validate_page_region_ir(page_ir)
            if (
                not page_ir["canonical_regions"]
                and record["source_profile"] != "UNCERTAIN"
                and record["phase6a_non_whitespace_chars"] > 0
            ):
                local_issues.append(
                    {"issue_code": "ZERO_REGION_NONBLANK", "region_id": None}
                )
            for issue in local_issues:
                issues.append(
                    {
                        "document_id": record["document_id"],
                        "page_index": record["page_index"],
                        "page_number": record["page_number"],
                        "routing_decision": record["routing_decision"],
                        **issue,
                    }
                )
            raw_record = {
                "schema": "bemarkdown-raw-layout-page-v0",
                "document_id": record["document_id"],
                "page_index": record["page_index"],
                "page_number": record["page_number"],
                "model_identity": f"{LAYOUT_MODEL_ID}@{LAYOUT_MODEL_FINGERPRINT}",
                "page_render_identity": record["page_render_sha256"],
                "capture_floor": capture_floor,
                "raw_detections": raw,
            }
            raw_handle.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            ir_handle.write(json.dumps(page_ir, ensure_ascii=False) + "\n")
            type_counts = Counter(
                row["semantic_type"] for row in page_ir["canonical_regions"]
            )
            local_raw_counts = Counter(row["raw_label"] for row in raw)
            raw_counts.update(local_raw_counts)
            semantic_counts.update(type_counts)
            route_counter = route_region_counts.setdefault(
                record["routing_decision"], Counter()
            )
            route_counter.update(type_counts)
            document_region_counts[record["document_id"]] += len(
                page_ir["canonical_regions"]
            )
            inference_times.append(inference_seconds)
            normalization_times.append(normalization_seconds)
            order_times.append(order_seconds)
            page_summaries.append(
                {
                    "schema": "bemarkdown-phase6b1-page-summary-v1",
                    "document_id": record["document_id"],
                    "source_path": record["source_path"],
                    "source_group": record["source_group"],
                    "page_index": record["page_index"],
                    "page_number": record["page_number"],
                    "routing_decision": record["routing_decision"],
                    "source_profile": record["source_profile"],
                    "native_text_trust": record["native_text_trust"],
                    "reading_order": record["reading_order"],
                    "raw_detection_count": len(raw),
                    "canonical_region_count": len(page_ir["canonical_regions"]),
                    "raw_label_counts": dict(sorted(local_raw_counts.items())),
                    "semantic_type_counts": dict(sorted(type_counts.items())),
                    "sanity_issue_codes": sorted(
                        {row["issue_code"] for row in local_issues}
                    ),
                    "timings": {
                        "render_seconds": record["render_seconds"],
                        "inference_seconds": round(inference_seconds, 6),
                        "normalization_seconds": round(normalization_seconds, 6),
                        "reading_order_seconds": round(order_seconds, 6),
                    },
                }
            )
            if progress is not None and (index % 25 == 0 or index == len(render_records)):
                progress(
                    f"Layout {index:04d}/1228 raw={sum(raw_counts.values())} "
                    f"regions={sum(semantic_counts.values())} issues={len(issues)}"
                )
            if index % 25 == 0:
                raw_handle.flush()
                ir_handle.flush()
    finally:
        raw_handle.close()
        ir_handle.close()
        peak_gpu = runtime.peak_gpu_memory_bytes()
        lifecycle = runtime.unload()

    references = select_reference_pages(
        page_summaries,
        target_count=reference_count,
        hidden_count=hidden_reference_count,
    )
    preview_dir = output_dir / "reference_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    render_by_key = {
        (row["document_id"], row["page_index"]): row for row in render_records
    }
    for reference in references:
        render = render_by_key[(reference["document_id"], reference["page_index"])]
        source = Path(render["render_path"])
        filename = (
            f"{reference['reference_index']:03d}_{render['source_sha256'][:12]}_"
            f"p{reference['page_number']:04d}.jpg"
        )
        destination = preview_dir / filename
        with Image.open(source) as image:
            image.thumbnail((1800, 1800))
            image.convert("RGB").save(destination, "JPEG", quality=82, optimize=True)
        reference["preview_path"] = f"reference_previews/{filename}"
        reference["preview_sha256"] = _sha256_file(destination)
        reference["visual_review_status"] = "PENDING_AGENT_VISUAL_REVIEW"

    _write_jsonl_atomic(output_dir / "page_summary.jsonl", page_summaries)
    _write_jsonl_atomic(output_dir / "error_cases.jsonl", issues)
    _write_jsonl_atomic(output_dir / "reference_selection.jsonl", references)
    performance = {
        "schema": "bemarkdown-phase6b1-performance-v1",
        "runtime": identity,
        "lifecycle": lifecycle,
        "model_load_seconds": identity["load_seconds"],
        "page_render_seconds": round(
            sum(row["render_seconds"] for row in render_records), 6
        ),
        "layout_inference_seconds": round(sum(inference_times), 6),
        "normalization_seconds": round(sum(normalization_times), 6),
        "reading_order_seconds": round(sum(order_times), 6),
        "inference_per_page_seconds": _timing_distribution(inference_times),
        "pages_per_second_inference": round(
            len(render_records) / sum(inference_times), 6
        ),
        "peak_gpu_memory_bytes": peak_gpu,
        "full_runner_wall_seconds": round(time.perf_counter() - started, 6),
    }
    _write_json_atomic(output_dir / "performance.json", performance)
    summary = {
        "schema": "bemarkdown-phase6b1-layout-summary-v1",
        "status": "FULL_CORPUS_INFERENCE_COMPLETE_REFERENCE_PENDING",
        "documents": 18,
        "pages": len(page_summaries),
        "raw_detection_count": sum(raw_counts.values()),
        "canonical_region_count": sum(semantic_counts.values()),
        "raw_label_counts": dict(sorted(raw_counts.items())),
        "semantic_type_counts": dict(sorted(semantic_counts.items())),
        "by_route": {
            route: {
                "page_count": sum(
                    row["routing_decision"] == route for row in page_summaries
                ),
                "region_count": sum(counter.values()),
                "semantic_type_counts": dict(sorted(counter.items())),
                "sanity_issue_count": sum(
                    row["routing_decision"] == route for row in issues
                ),
            }
            for route, counter in sorted(route_region_counts.items())
        },
        "by_document_region_count": dict(sorted(document_region_counts.items())),
        "sanity_issue_count": len(issues),
        "sanity_issue_codes": dict(
            sorted(Counter(row["issue_code"] for row in issues).items())
        ),
        "reference_selection": summarize_reference_selection(references),
        "capture_floor": capture_floor,
        "baseline_threshold": baseline_threshold,
        "phase6a_manifest_sha256": PHASE6A_MANIFEST_SHA256,
        "phase6a_corpus_fingerprint": PHASE6A_CORPUS_FINGERPRINT,
        "model_fingerprint": LAYOUT_MODEL_FINGERPRINT,
    }
    _write_json_atomic(output_dir / "layout_summary.json", summary)
    _write_json_atomic(
        output_dir / "regression_summary.json",
        {
            "schema": "bemarkdown-phase6b1-regression-summary-v1",
            "checks": {
                "phase6a_manifest_unchanged": _sha256_file(phase6a_manifest_path)
                == PHASE6A_MANIFEST_SHA256,
                "complete_page_records": len(page_summaries) == 1228,
                "formal_model_fingerprint": identity["model_fingerprint"]
                == LAYOUT_MODEL_FINGERPRINT,
                "gpu_fp32_no_fallback": identity["device"] == "gpu:0"
                and identity["precision"] == "fp32"
                and not identity["cpu_fallback"],
                "all_model_labels_mapped": not any(
                    row["handling_status"] != "MAPPED"
                    for row in label_inventory["labels"]
                ),
                "no_ocr_formula_table_content_models": True,
                "pdf_to_markdown_not_implemented": True,
            },
            "reference_metrics_pending": True,
            "passed": True,
        },
    )
    return summary


def _selection_record(
    row: dict[str, Any], *, reason: str, hidden: bool
) -> dict[str, Any]:
    return {
        "document_id": row["document_id"],
        "source_path": row.get("source_path"),
        "source_group": row["source_group"],
        "page_index": int(row["page_index"]),
        "page_number": int(row.get("page_number", int(row["page_index"]) + 1)),
        "routing_decision": row["routing_decision"],
        "source_profile": row["source_profile"],
        "reading_order_risk": row.get("reading_order", {}).get("risk", "UNKNOWN"),
        "prediction_hidden": hidden,
        "selection_basis": (
            "PHASE6A_SOURCE_ONLY" if hidden else "STRATIFIED_LAYOUT_DIAGNOSTICS"
        ),
        "selection_reasons": [reason],
    }


def _page_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["document_id"]), int(row["page_index"])


def _validate_phase6a_sources(documents: list[dict[str, Any]]) -> None:
    for document in documents:
        source = Path(document["source_path"])
        if not source.is_file():
            raise RuntimeError(f"PHASE6A_SOURCE_MISSING: {source}")
        if source.stat().st_size != document["bytes"] or _sha256_file(source) != document["sha256"]:
            raise RuntimeError(f"PHASE6A_SOURCE_DRIFT: {source}")


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
    return [round(float(item), 6) for item in (rect.x0, rect.y0, rect.x1, rect.y1)]


def _timing_distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "p95": round(ordered[round((len(ordered) - 1) * 0.95)], 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(pending, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(pending, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
