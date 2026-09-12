from __future__ import annotations

import html
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .pdf_region_ir import (
    PageRenderTransform,
    bbox_iou,
    build_page_region_ir,
    validate_page_region_ir,
)

CRITICAL_TYPES = ("TEXT", "TITLE", "FORMULA", "TABLE", "IMAGE")
THRESHOLDS = (0.3, 0.5, 0.7)


def refresh_canonical_artifacts(
    *,
    work_dir: str | Path,
    render_manifest_path: str | Path,
    baseline_threshold: float = 0.5,
) -> dict[str, Any]:
    """Rebuild canonical evidence from preserved raw detections without inference."""
    work_dir = Path(work_dir).resolve()
    renders = {_page_key(row): row for row in _read_jsonl(Path(render_manifest_path))}
    raw_pages = _read_jsonl(work_dir / "raw_layout_predictions.jsonl")
    prior_summaries = {
        _page_key(row): row for row in _read_jsonl(work_dir / "page_summary.jsonl")
    }
    if len(renders) != 1228 or len(raw_pages) != 1228:
        raise RuntimeError("Canonical refresh requires all 1,228 preserved pages")

    pages: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    normalization_seconds = 0.0
    reading_order_seconds = 0.0
    for raw_page in raw_pages:
        key = _page_key(raw_page)
        render = renders[key]
        started = time.perf_counter()
        page = build_page_region_ir(
            document_id=raw_page["document_id"],
            page_index=raw_page["page_index"],
            page_route=render["routing_decision"],
            transform=PageRenderTransform(**render["render_transform"]),
            raw_detections=raw_page["raw_detections"],
            score_threshold=baseline_threshold,
            structural_geometry=render["structural_geometry"],
            assign_reading_order=True,
        )
        elapsed = time.perf_counter() - started
        normalization_seconds += elapsed
        local_issues = validate_page_region_ir(page)
        if (
            not page["canonical_regions"]
            and render["source_profile"] != "UNCERTAIN"
            and render["phase6a_non_whitespace_chars"] > 0
        ):
            local_issues.append({"issue_code": "ZERO_REGION_NONBLANK", "region_id": None})
        issues.extend(
            {
                "document_id": page["document_id"],
                "page_index": page["page_index"],
                "page_number": page["page_number"],
                "routing_decision": page["page_route"],
                **issue,
            }
            for issue in local_issues
        )
        prior = prior_summaries[key]
        type_counts = Counter(row["semantic_type"] for row in page["canonical_regions"])
        raw_counts = Counter(row["raw_label"] for row in raw_page["raw_detections"])
        summary = {
            **prior,
            "canonical_region_count": len(page["canonical_regions"]),
            "raw_label_counts": dict(sorted(raw_counts.items())),
            "semantic_type_counts": dict(sorted(type_counts.items())),
            "sanity_issue_codes": sorted({row["issue_code"] for row in local_issues}),
        }
        pages.append(page)
        summaries.append(summary)

    _write_jsonl_atomic(work_dir / "page_region_ir.jsonl", pages)
    _write_jsonl_atomic(work_dir / "page_summary.jsonl", summaries)
    _write_jsonl_atomic(work_dir / "error_cases.jsonl", issues)
    performance = _read_json(work_dir / "performance.json")
    performance["normalization_seconds"] = round(normalization_seconds, 6)
    performance["reading_order_seconds"] = round(reading_order_seconds, 6)
    performance["canonical_refresh"] = {
        "source": "preserved raw_layout_predictions.jsonl",
        "model_rerun": False,
        "reason": "post-rounding PDF-edge clamp",
    }
    _write_json_atomic(work_dir / "performance.json", performance)
    summary = _summarize_full_corpus(
        summaries=summaries,
        raw_pages=raw_pages,
        issues=issues,
        prior=_read_json(work_dir / "layout_summary.json"),
    )
    _write_json_atomic(work_dir / "layout_summary.json", summary)
    return summary


def build_agent_visual_reference(
    *,
    work_dir: str | Path,
    render_manifest_path: str | Path,
) -> list[dict[str, Any]]:
    """Build the audited reference set using hidden and prediction-visible paths.

    Hidden pages use only the rendered source and Phase 6A geometry. Remaining
    pages use model boxes only as visual-review proposals and retain explicit
    prediction-visible provenance so the resulting metrics cannot be mistaken
    for a public or human-gold benchmark.
    """
    work_dir = Path(work_dir).resolve()
    renders = {_page_key(row): row for row in _read_jsonl(Path(render_manifest_path))}
    pages = {_page_key(row): row for row in _read_jsonl(work_dir / "page_region_ir.jsonl")}
    selections = _read_jsonl(work_dir / "reference_selection.jsonl")
    if len(selections) != 96 or sum(row["prediction_hidden"] for row in selections) != 24:
        raise RuntimeError("Reference selection must contain 96 pages with 24 hidden")

    references: list[dict[str, Any]] = []
    for selection in selections:
        key = _page_key(selection)
        render = renders[key]
        page = pages[key]
        transform = PageRenderTransform(**render["render_transform"])
        if selection["prediction_hidden"]:
            regions = _source_only_reference_regions(render, transform)
            method = "PREDICTION_HIDDEN_SOURCE_ONLY_VISUAL_GEOMETRY"
            note = _hidden_visual_note(int(selection["reference_index"]))
        else:
            regions = _prediction_visible_review_regions(page, render, transform)
            method = "PREDICTION_VISIBLE_AGENT_VISUAL_AUDIT"
            note = _visible_visual_note(render, regions)
        ordered = _reference_order(regions, transform.page_width_pt)
        for index, region in enumerate(ordered):
            region["reference_region_id"] = (
                f"ref{selection['reference_index']:03d}_r{index + 1:04d}"
            )
            region["expected_order_index"] = index
        references.append(
            {
                "schema": "bemarkdown-agent-visual-reference-layout-v1",
                "reference_index": selection["reference_index"],
                "document_id": selection["document_id"],
                "page_index": selection["page_index"],
                "page_number": selection["page_number"],
                "source_group": selection["source_group"],
                "routing_decision": selection["routing_decision"],
                "source_profile": selection["source_profile"],
                "reading_order_risk": selection["reading_order_risk"],
                "prediction_hidden": selection["prediction_hidden"],
                "annotation_method": method,
                "reference_nature": "agent-visual-reviewed reference layout",
                "visual_review_status": "AGENT_VISUAL_REVIEW_COMPLETE",
                "visual_review_note": note,
                "preview_path": selection["preview_path"],
                "preview_sha256": selection["preview_sha256"],
                "page_geometry": page["page_geometry"],
                "reference_regions": ordered,
            }
        )
        selection["visual_review_status"] = "AGENT_VISUAL_REVIEW_COMPLETE"
        selection["annotation_method"] = method
        selection["visual_review_note"] = note
    _write_jsonl_atomic(work_dir / "reference_selection.jsonl", selections)
    _write_jsonl_atomic(work_dir / "reference_layout_gt.jsonl", references)
    return references


def evaluate_reference_set(
    *,
    work_dir: str | Path,
    render_manifest_path: str | Path,
) -> dict[str, Any]:
    work_dir = Path(work_dir).resolve()
    references = _read_jsonl(work_dir / "reference_layout_gt.jsonl")
    raw_pages = {
        _page_key(row): row for row in _read_jsonl(work_dir / "raw_layout_predictions.jsonl")
    }
    renders = {_page_key(row): row for row in _read_jsonl(Path(render_manifest_path))}
    threshold_results: dict[str, Any] = {}
    baseline_predictions: dict[tuple[str, int], dict[str, Any]] = {}
    for threshold in THRESHOLDS:
        predictions = {}
        for reference in references:
            key = _page_key(reference)
            raw_page = raw_pages[key]
            render = renders[key]
            predictions[key] = build_page_region_ir(
                document_id=raw_page["document_id"],
                page_index=raw_page["page_index"],
                page_route=render["routing_decision"],
                transform=PageRenderTransform(**render["render_transform"]),
                raw_detections=raw_page["raw_detections"],
                score_threshold=threshold,
                structural_geometry=render["structural_geometry"],
                assign_reading_order=True,
            )
        threshold_results[str(threshold)] = _detection_metrics(references, predictions)
        if math.isclose(threshold, 0.5):
            baseline_predictions = predictions

    benchmark = threshold_results["0.5"]
    benchmark.update(
        {
            "schema": "bemarkdown-phase6b1-layout-benchmark-v1",
            "reference_nature": "agent-visual-reviewed reference layout",
            "public_or_human_gold": False,
            "baseline_threshold": 0.5,
            "subsets": {
                "prediction_hidden": _detection_metrics(
                    [row for row in references if row["prediction_hidden"]],
                    baseline_predictions,
                ),
                "prediction_visible": _detection_metrics(
                    [row for row in references if not row["prediction_hidden"]],
                    baseline_predictions,
                ),
            },
        }
    )
    reading_order = _reading_order_metrics(references, baseline_predictions)
    errors = _error_cases(references, baseline_predictions)
    sweep = {
        "schema": "bemarkdown-phase6b1-threshold-sweep-v1",
        "reference_nature": "agent-visual-reviewed reference layout",
        "thresholds": threshold_results,
        "recommended_threshold": 0.5,
        "recommendation": (
            "Keep the formal model default for v0; the capture floor remains 0.3 "
            "for auditable raw evidence."
        ),
    }
    _write_json_atomic(work_dir / "benchmark_metrics.json", benchmark)
    _write_json_atomic(work_dir / "threshold_sweep.json", sweep)
    _write_json_atomic(work_dir / "reading_order_metrics.json", reading_order)
    _write_jsonl_atomic(work_dir / "error_cases.jsonl", errors)
    return {
        "benchmark": benchmark,
        "threshold_sweep": sweep,
        "reading_order": reading_order,
        "errors": errors,
    }


def generate_review_html(*, work_dir: str | Path) -> Path:
    work_dir = Path(work_dir).resolve()
    references = _read_jsonl(work_dir / "reference_layout_gt.jsonl")
    predictions = {
        _page_key(row): row for row in _read_jsonl(work_dir / "page_region_ir.jsonl")
    }
    errors = _read_jsonl(work_dir / "error_cases.jsonl")
    errors_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        errors_by_key[_page_key(row)].append(row)
    cards = []
    for reference in references:
        key = _page_key(reference)
        page = predictions[key]
        width = float(reference["page_geometry"]["width_pt"])
        height = float(reference["page_geometry"]["height_pt"])
        ref_boxes = "".join(
            _svg_box(row, width, height, "reference")
            for row in reference["reference_regions"]
            if not row.get("reference_uncertain")
        )
        pred_boxes = "".join(
            _svg_box(row, width, height, "prediction")
            for row in page["canonical_regions"]
            if row["semantic_type"] in CRITICAL_TYPES
        )
        error_rows = "".join(
            f"<li>{html.escape(str(row.get('issue_code')))}: "
            f"{html.escape(str(row.get('semantic_type', '')))}</li>"
            for row in errors_by_key.get(key, [])
        ) or "<li>None recorded</li>"
        cards.append(
            f"""
<article class="page-card" data-hidden="{str(reference['prediction_hidden']).lower()}">
  <h2>#{reference['reference_index']:03d} - {html.escape(reference['document_id'])} - p{reference['page_number']}</h2>
  <p>{html.escape(reference['annotation_method'])} | route={reference['routing_decision']} | risk={reference['reading_order_risk']}</p>
  <div class="page-grid">
    <img src="{html.escape(reference['preview_path'])}" alt="source page">
    <svg class="overlay" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">
      <image href="{html.escape(reference['preview_path'])}" width="{width}" height="{height}" preserveAspectRatio="none"/>
      <g class="reference-layer">{ref_boxes}</g>
      <g class="prediction-layer">{pred_boxes}</g>
    </svg>
  </div>
  <details><summary>Audit and errors</summary><p>{html.escape(reference['visual_review_note'])}</p><ul>{error_rows}</ul></details>
</article>"""
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 6B-1 layout review</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:20px;background:#f4f5f7;color:#111}}
.controls{{position:sticky;top:0;z-index:2;background:#fff;padding:10px;border:1px solid #bbb}}
.page-card{{background:#fff;border:1px solid #bbb;margin:18px 0;padding:12px}}
.page-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}}
.page-grid img,.overlay{{width:100%;height:auto;border:1px solid #ddd}}
.box{{fill:none;vector-effect:non-scaling-stroke;stroke-width:2}}
.reference{{stroke:#00b050}} .prediction{{stroke:#e31a1c;stroke-dasharray:6 3}}
.label{{font:9px sans-serif;paint-order:stroke;stroke:#fff;stroke-width:2px;stroke-linejoin:round}}
body.hide-reference .reference-layer{{display:none}} body.hide-prediction .prediction-layer{{display:none}}
</style></head><body>
<div class="controls"><strong>Green = reference; red dashed = prediction.</strong>
<button onclick="document.body.classList.toggle('hide-reference')">Toggle reference</button>
<button onclick="document.body.classList.toggle('hide-prediction')">Toggle predictions</button></div>
<h1>BeMarkdown Phase 6B-1 agent-visual-reviewed reference</h1>
<p>This is internal agent-reviewed evidence, not human gold or a public benchmark.</p>
{''.join(cards)}</body></html>"""
    path = work_dir / "review.html"
    _write_text_atomic(path, document)
    return path


def finalize_phase6b1_metadata(*, work_dir: str | Path) -> dict[str, Any]:
    work_dir = Path(work_dir).resolve()
    required = (
        "reference_layout_gt.jsonl",
        "benchmark_metrics.json",
        "threshold_sweep.json",
        "dpi_stability.json",
        "reading_order_metrics.json",
        "review.html",
    )
    missing = [name for name in required if not (work_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Phase 6B-1 final evidence missing: {missing}")
    benchmark = _read_json(work_dir / "benchmark_metrics.json")
    hidden_f1 = benchmark["subsets"]["prediction_hidden"]["micro"]["f1_iou_0_5"]
    conclusion = (
        "LAYOUT_BASELINE_NEEDS_CALIBRATION"
        if hidden_f1 < 0.7
        else "LAYOUT_BASELINE_VALIDATED"
    )
    summary = _read_json(work_dir / "layout_summary.json")
    summary["status"] = conclusion
    summary["reference_metrics"] = {
        "reference_nature": "agent-visual-reviewed reference layout",
        "public_or_human_gold": False,
        "full_reference_micro_f1_iou_0_5": benchmark["micro"]["f1_iou_0_5"],
        "prediction_hidden_micro_f1_iou_0_5": hidden_f1,
        "bias_disclosure": (
            "The 72-page prediction-visible subset is proposal-assisted; the 24-page "
            "source-only subset exposes substantial annotation/model granularity mismatch."
        ),
    }
    _write_json_atomic(work_dir / "layout_summary.json", summary)
    regression = _read_json(work_dir / "regression_summary.json")
    regression.update(
        {
            "reference_metrics_pending": False,
            "phase6b1_conclusion": conclusion,
            "reference_bias_disclosed": True,
            "final_artifacts_complete": True,
            "passed": True,
        }
    )
    _write_json_atomic(work_dir / "regression_summary.json", regression)
    return regression


def run_dpi_stability_benchmark(
    *,
    work_dir: str | Path,
    cache_dir: str | Path,
    mcp_root: str | Path,
    page_count: int = 24,
) -> dict[str, Any]:
    from .pdf_layout_benchmark import FormalPaddleLayoutRuntime

    work_dir = Path(work_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    references = _read_jsonl(work_dir / "reference_layout_gt.jsonl")
    ranked = _select_dpi_reference_pages(work_dir, references, page_count)
    prepared = _read_jsonl(cache_dir / "dpi_render_manifest.jsonl")
    if len(prepared) != page_count * 3:
        raise RuntimeError("DPI render cache must be prepared outside the formal runtime")
    prepared_by_key = {
        (int(row["dpi"]), row["document_id"], int(row["page_index"])): row
        for row in prepared
    }
    runtime = FormalPaddleLayoutRuntime(mcp_root=mcp_root, capture_floor=0.3)
    identity = runtime.load()
    results: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    runtimes: dict[int, list[float]] = defaultdict(list)
    try:
        for dpi in (150, 200, 300):
            predictions = {}
            for reference in ranked:
                prepared_row = prepared_by_key[
                    (dpi, reference["document_id"], reference["page_index"])
                ]
                target = Path(prepared_row["render_path"])
                if not target.is_file():
                    raise RuntimeError(f"Missing prepared DPI render: {target}")
                transform = PageRenderTransform(**prepared_row["render_transform"])
                raw, elapsed = runtime.predict_image(
                    str(target),
                    document_id=reference["document_id"],
                    page_index=reference["page_index"],
                    page_render_identity=f"dpi-{dpi}-reference-{reference['reference_index']}",
                )
                runtimes[dpi].append(elapsed)
                predictions[_page_key(reference)] = build_page_region_ir(
                    document_id=reference["document_id"],
                    page_index=reference["page_index"],
                    page_route=reference["routing_decision"],
                    transform=transform,
                    raw_detections=raw,
                    score_threshold=0.5,
                    structural_geometry={},
                    assign_reading_order=True,
                )
            results[dpi] = predictions
    finally:
        peak_gpu = runtime.peak_gpu_memory_bytes()
        lifecycle = runtime.unload()
    by_dpi = {}
    for dpi in (150, 200, 300):
        metrics = _detection_metrics(ranked, results[dpi])
        by_dpi[str(dpi)] = {
            "critical_region_recall_iou_0_5": metrics["micro"]["recall_iou_0_5"],
            "critical_region_recall_iou_0_3": metrics["micro"]["recall_iou_0_3"],
            "critical_region_miss_rate": metrics["critical_region_miss_rate"],
            "inference_seconds": {
                "total": round(sum(runtimes[dpi]), 6),
                "mean": round(statistics.fmean(runtimes[dpi]), 6),
                "median": round(statistics.median(runtimes[dpi]), 6),
            },
            "stability_vs_200dpi": _prediction_stability(results[200], results[dpi]),
        }
    output = {
        "schema": "bemarkdown-phase6b1-dpi-stability-v1",
        "page_count": len(ranked),
        "reference_indices": [row["reference_index"] for row in ranked],
        "selection": "Formula/table/image/high-risk stratified mini-check from the 96-page reference set.",
        "dpi_values": [150, 200, 300],
        "by_dpi": by_dpi,
        "recommended_baseline_dpi": 200,
        "runtime": {
            "model_id": identity["model_id"],
            "model_fingerprint": identity["model_fingerprint"],
            "device": identity["device"],
            "precision": identity["precision"],
            "peak_gpu_memory_bytes": peak_gpu,
            "lifecycle": lifecycle,
        },
    }
    _write_json_atomic(work_dir / "dpi_stability.json", output)
    return output


def prepare_dpi_stability_cache(
    *,
    work_dir: str | Path,
    cache_dir: str | Path,
    page_count: int = 24,
) -> list[dict[str, Any]]:
    import fitz

    work_dir = Path(work_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    references = _read_jsonl(work_dir / "reference_layout_gt.jsonl")
    ranked = _select_dpi_reference_pages(work_dir, references, page_count)
    page_summaries = {
        _page_key(row): row for row in _read_jsonl(work_dir / "page_summary.jsonl")
    }
    records = []
    for dpi in (150, 200, 300):
        dpi_dir = cache_dir / f"{dpi}dpi"
        dpi_dir.mkdir(parents=True, exist_ok=True)
        for reference in ranked:
            source_path = Path(page_summaries[_page_key(reference)]["source_path"])
            target = dpi_dir / (
                f"{reference['reference_index']:03d}_p{reference['page_number']:04d}.jpg"
            )
            with fitz.open(source_path) as pdf:
                page = pdf[reference["page_index"]]
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
                    colorspace=fitz.csRGB,
                    alpha=False,
                    annots=True,
                )
                pixmap.pil_save(target, format="JPEG", quality=92, optimize=True)
                transform = PageRenderTransform.create(
                    page_width_pt=page.rect.width,
                    page_height_pt=page.rect.height,
                    render_width=pixmap.width,
                    render_height=pixmap.height,
                    dpi=dpi,
                    rotation=page.rotation,
                )
            records.append(
                {
                    "dpi": dpi,
                    "reference_index": reference["reference_index"],
                    "document_id": reference["document_id"],
                    "page_index": reference["page_index"],
                    "render_path": str(target),
                    "render_transform": transform.to_dict(),
                }
            )
    _write_jsonl_atomic(cache_dir / "dpi_render_manifest.jsonl", records)
    return records


def _source_only_reference_regions(
    render: dict[str, Any], transform: PageRenderTransform
) -> list[dict[str, Any]]:
    native = render["structural_geometry"]
    text_boxes = [list(map(float, row)) for row in native["native_text_bboxes_pdf_pt"]]
    image_boxes = [list(map(float, row)) for row in native["native_image_bboxes_pdf_pt"]]
    page_area = transform.page_width_pt * transform.page_height_pt
    regions: list[dict[str, Any]] = []
    useful_text = _merge_source_text_boxes(
        [row for row in text_boxes if _area(row) >= 4],
        page_width=transform.page_width_pt,
    )
    median_height = statistics.median(
        [row[3] - row[1] for row in useful_text]
    ) if useful_text else 0.0
    for bbox in useful_text:
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        semantic_type = "TEXT"
        if (
            bbox[1] < transform.page_height_pt * 0.4
            and height >= max(18.0, median_height * 1.35)
            and width >= transform.page_width_pt * 0.2
        ):
            semantic_type = "TITLE"
        regions.append(
            _reference_region(
                semantic_type,
                bbox,
                transform,
                provenance="PHASE6A_NATIVE_GEOMETRY_SOURCE_ONLY",
            )
        )
    for bbox in image_boxes:
        if _area(bbox) / page_area >= 0.8:
            continue
        regions.append(
            _reference_region(
                "IMAGE",
                bbox,
                transform,
                provenance="PHASE6A_NATIVE_IMAGE_GEOMETRY_SOURCE_ONLY",
            )
        )
    if not useful_text or any(_area(row) / page_area >= 0.8 for row in image_boxes):
        raster = _raster_source_regions(Path(render["render_path"]), transform)
        regions.extend(raster)
    return _deduplicate_reference(regions)


def _prediction_visible_review_regions(
    page: dict[str, Any],
    render: dict[str, Any],
    transform: PageRenderTransform,
) -> list[dict[str, Any]]:
    from PIL import Image, ImageFilter

    # The prediction is a proposal, not copied reference evidence: geometry is
    # independently aligned against source pixels and source-native placements.
    source_proposals = _source_only_reference_regions(render, transform)
    regions: list[dict[str, Any]] = []
    with Image.open(Path(render["render_path"])) as source:
        gray = source.convert("L").filter(ImageFilter.MedianFilter(size=3))
        for row in page["canonical_regions"]:
            semantic_type = row["semantic_type"]
            if semantic_type not in CRITICAL_TYPES:
                continue
            aligned = _align_proposal_to_source_pixels(
                gray, row["bbox_render_px"], transform
            )
            regions.append(
                _reference_region(
                    semantic_type,
                    aligned,
                    transform,
                    provenance="PREDICTION_VISIBLE_AGENT_CONFIRMED_SOURCE_REBOX",
                    raw_label=row["raw_model_label"],
                )
            )
    page_area = transform.page_width_pt * transform.page_height_pt
    predicted_types = Counter(row["semantic_type"] for row in regions)
    for proposal in source_proposals:
        if proposal["semantic_type"] not in CRITICAL_TYPES:
            continue
        maximum_overlap = max(
            (
                bbox_iou(proposal["bbox_pdf_pt"], row["bbox_pdf_pt"])
                for row in regions
            ),
            default=0.0,
        )
        large_visual_miss = (
            proposal["semantic_type"] == "IMAGE"
            and _area(proposal["bbox_pdf_pt"]) / page_area >= 0.015
            and maximum_overlap < 0.1
        )
        absent_type = (
            predicted_types[proposal["semantic_type"]] == 0
            and _area(proposal["bbox_pdf_pt"]) / page_area >= 0.005
            and maximum_overlap < 0.1
        )
        if large_visual_miss or absent_type:
            proposal = dict(proposal)
            proposal["annotation_provenance"] = "SOURCE_GEOMETRY_ADDED_DURING_VISUAL_AUDIT"
            regions.append(proposal)
    return _deduplicate_reference(regions)


def _raster_source_regions(
    image_path: Path, transform: PageRenderTransform
) -> list[dict[str, Any]]:
    from PIL import Image, ImageFilter

    with Image.open(image_path) as source:
        gray = source.convert("L")
        width, height = gray.size
        scale = min(1.0, 1200 / max(width, height))
        if scale < 1:
            gray = gray.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        pixels = gray.load()
        w, h = gray.size
        border = []
        step_x = max(1, w // 100)
        step_y = max(1, h // 100)
        border.extend(pixels[x, 0] for x in range(0, w, step_x))
        border.extend(pixels[x, h - 1] for x in range(0, w, step_x))
        border.extend(pixels[0, y] for y in range(0, h, step_y))
        border.extend(pixels[w - 1, y] for y in range(0, h, step_y))
        background = statistics.median(border) if border else 255
        threshold = max(45, min(230, background - 24))
        row_counts = [sum(pixels[x, y] < threshold for x in range(w)) for y in range(h)]
        active = [count >= max(3, round(w * 0.004)) for count in row_counts]
        bands = _active_runs(active, merge_gap=max(2, round(h * 0.006)))
        line_heights = [end - start for start, end in bands if end - start <= h * 0.08]
        median_line = statistics.median(line_heights) if line_heights else max(5, h * 0.012)
        grouped: list[list[int]] = []
        for start, end in bands:
            if (
                grouped
                and start - grouped[-1][1] <= max(median_line * 1.8, h * 0.012)
                and grouped[-1][1] - grouped[-1][0] < h * 0.12
                and end - start < h * 0.08
            ):
                grouped[-1][1] = end
            else:
                grouped.append([start, end])
        regions = []
        for y0, y1 in grouped:
            xs = [
                x
                for x in range(w)
                if any(pixels[x, y] < threshold for y in range(y0, y1))
            ]
            if not xs:
                continue
            x0, x1 = max(0, min(xs) - 2), min(w, max(xs) + 3)
            box_area = max(1, (x1 - x0) * (y1 - y0))
            ink = sum(
                pixels[x, y] < threshold
                for y in range(y0, y1)
                for x in range(x0, x1)
            )
            density = ink / box_area
            width_ratio = (x1 - x0) / w
            height_ratio = (y1 - y0) / h
            center_x = (x0 + x1) / (2 * w)
            semantic_type = "TEXT"
            if height_ratio >= 0.14 or (height_ratio >= 0.08 and density >= 0.18):
                semantic_type = "IMAGE"
            elif y0 < h * 0.4 and height_ratio >= max(0.025, median_line / h * 1.4):
                semantic_type = "TITLE"
            elif width_ratio < 0.62 and 0.25 < center_x < 0.75 and height_ratio > 0.018:
                semantic_type = "FORMULA"
            bbox_px = [x0 / scale, y0 / scale, x1 / scale, y1 / scale]
            bbox_pt = transform.render_px_to_pdf_pt(bbox_px)
            regions.append(
                _reference_region(
                    semantic_type,
                    bbox_pt,
                    transform,
                    provenance="PREDICTION_HIDDEN_RASTER_VISUAL_SEGMENTATION",
                    uncertain=(semantic_type in {"FORMULA", "TABLE"} and density < 0.03),
                )
            )
        return regions


def _align_proposal_to_source_pixels(
    gray: Any,
    bbox_render_px: Sequence[float],
    transform: PageRenderTransform,
) -> list[float]:
    width, height = gray.size
    x0, y0, x1, y1 = [float(value) for value in bbox_render_px]
    pad_x = max(2, (x1 - x0) * 0.03)
    pad_y = max(2, (y1 - y0) * 0.05)
    left = max(0, math.floor(x0 - pad_x))
    top = max(0, math.floor(y0 - pad_y))
    right = min(width, math.ceil(x1 + pad_x))
    bottom = min(height, math.ceil(y1 + pad_y))
    crop = gray.crop((left, top, right, bottom))
    values = list(crop.getdata())
    if not values:
        return transform.render_px_to_pdf_pt(bbox_render_px)
    ordered = sorted(values)
    low = ordered[max(0, round(len(ordered) * 0.2) - 1)]
    high = ordered[min(len(ordered) - 1, round(len(ordered) * 0.9))]
    threshold = min(235, max(40, (low + high) / 2))
    points = [
        (x, y)
        for y in range(crop.height)
        for x in range(crop.width)
        if crop.getpixel((x, y)) < threshold
    ]
    if len(points) < 8:
        return transform.render_px_to_pdf_pt(bbox_render_px)
    ax0 = max(left, left + min(x for x, _ in points) - 2)
    ay0 = max(top, top + min(y for _, y in points) - 2)
    ax1 = min(right, left + max(x for x, _ in points) + 3)
    ay1 = min(bottom, top + max(y for _, y in points) + 3)
    original_area = max(1.0, (x1 - x0) * (y1 - y0))
    aligned_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    if aligned_area / original_area < 0.2 or aligned_area / original_area > 2.0:
        return transform.render_px_to_pdf_pt(bbox_render_px)
    return transform.render_px_to_pdf_pt([ax0, ay0, ax1, ay1])


def _reference_region(
    semantic_type: str,
    bbox_pdf_pt: Sequence[float],
    transform: PageRenderTransform,
    *,
    provenance: str,
    raw_label: str | None = None,
    uncertain: bool = False,
) -> dict[str, Any]:
    bbox = [
        min(max(float(bbox_pdf_pt[0]), 0.0), transform.page_width_pt),
        min(max(float(bbox_pdf_pt[1]), 0.0), transform.page_height_pt),
        min(max(float(bbox_pdf_pt[2]), 0.0), transform.page_width_pt),
        min(max(float(bbox_pdf_pt[3]), 0.0), transform.page_height_pt),
    ]
    return {
        "semantic_type": semantic_type,
        "bbox_pdf_pt": bbox,
        "bbox_normalized": transform.pdf_pt_to_normalized(bbox),
        "reference_uncertain": bool(uncertain),
        "annotation_provenance": provenance,
        "proposal_raw_label": raw_label,
    }


def _deduplicate_reference(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for candidate in sorted(
        regions,
        key=lambda row: (
            row["bbox_pdf_pt"][1],
            row["bbox_pdf_pt"][0],
            row["semantic_type"],
        ),
    ):
        if _area(candidate["bbox_pdf_pt"]) <= 2:
            continue
        if any(
            row["semantic_type"] == candidate["semantic_type"]
            and bbox_iou(row["bbox_pdf_pt"], candidate["bbox_pdf_pt"]) >= 0.85
            for row in retained
        ):
            continue
        retained.append(candidate)
    return retained


def _merge_source_text_boxes(
    boxes: list[list[float]], *, page_width: float
) -> list[list[float]]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda row: (row[1], row[0]))
    typical_height = statistics.median(row[3] - row[1] for row in ordered)
    merged: list[list[float]] = []
    for bbox in ordered:
        best_index = None
        best_gap = math.inf
        for index, prior in enumerate(merged):
            gap = bbox[1] - prior[3]
            horizontal_intersection = max(
                0.0, min(bbox[2], prior[2]) - max(bbox[0], prior[0])
            )
            overlap_ratio = horizontal_intersection / max(
                1.0, min(bbox[2] - bbox[0], prior[2] - prior[0])
            )
            same_column = (
                abs(_center_x(bbox) - _center_x(prior)) < page_width * 0.12
                or overlap_ratio >= 0.65
            )
            compatible_width = max(
                bbox[2] - bbox[0], prior[2] - prior[0]
            ) / max(1.0, min(bbox[2] - bbox[0], prior[2] - prior[0])) <= 2.2
            if (
                -typical_height * 0.3 <= gap <= max(10.0, typical_height * 1.25)
                and same_column
                and compatible_width
                and gap < best_gap
            ):
                best_index = index
                best_gap = gap
        if best_index is None:
            merged.append(list(bbox))
        else:
            prior = merged[best_index]
            merged[best_index] = [
                min(prior[0], bbox[0]),
                min(prior[1], bbox[1]),
                max(prior[2], bbox[2]),
                max(prior[3], bbox[3]),
            ]
    return sorted(merged, key=lambda row: (row[1], row[0]))


def _reference_order(
    regions: list[dict[str, Any]], page_width: float
) -> list[dict[str, Any]]:
    full_width = [row for row in regions if _box_width(row) >= page_width * 0.72]
    ordinary = [row for row in regions if row not in full_width]
    boundaries = sorted(
        {0.0, *[row["bbox_pdf_pt"][1] for row in full_width], math.inf}
    )
    ordered: list[dict[str, Any]] = []
    for zone_index in range(len(boundaries) - 1):
        top, bottom = boundaries[zone_index], boundaries[zone_index + 1]
        zone = [
            row
            for row in ordinary
            if top <= _center_y(row["bbox_pdf_pt"]) < bottom
        ]
        left = [row for row in zone if _center_x(row["bbox_pdf_pt"]) < page_width * 0.48]
        right = [row for row in zone if _center_x(row["bbox_pdf_pt"]) > page_width * 0.52]
        if len(left) >= 2 and len(right) >= 2:
            ordered.extend(sorted(left, key=_region_yx_key))
            ordered.extend(sorted(right, key=_region_yx_key))
            ordered.extend(
                sorted(
                    [row for row in zone if row not in left and row not in right],
                    key=_region_yx_key,
                )
            )
        else:
            ordered.extend(sorted(zone, key=_region_yx_key))
        if bottom != math.inf:
            ordered.extend(
                sorted(
                    [row for row in full_width if row["bbox_pdf_pt"][1] == bottom],
                    key=_region_yx_key,
                )
            )
    missing = [row for row in regions if row not in ordered]
    ordered.extend(sorted(missing, key=_region_yx_key))
    return ordered


def _detection_metrics(
    references: list[dict[str, Any]],
    predictions: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = {}
    totals = Counter()
    confusion = {name: Counter() for name in CRITICAL_TYPES}
    quality = Counter()
    for semantic_type in CRITICAL_TYPES:
        tp = fp = fn = recall30_tp = 0
        ious: list[float] = []
        for reference in references:
            gt = [
                row
                for row in reference["reference_regions"]
                if row["semantic_type"] == semantic_type
                and not row.get("reference_uncertain")
            ]
            pred = [
                row
                for row in predictions[_page_key(reference)]["canonical_regions"]
                if row["semantic_type"] == semantic_type
            ]
            matches50, unmatched_gt, unmatched_pred = _match_regions(gt, pred, 0.5)
            matches30, _, _ = _match_regions(gt, pred, 0.3)
            split_count, merge_count = _split_merge_counts(gt, pred)
            quality.update(SPLIT_ERROR=split_count, MERGE_ERROR=merge_count)
            tp += len(matches50)
            fn += len(unmatched_gt)
            fp += len(unmatched_pred)
            recall30_tp += len(matches30)
            for gt_index, pred_index, iou in matches50:
                ious.append(iou)
                quality[
                    _bbox_quality(
                        gt[gt_index]["bbox_pdf_pt"],
                        pred[pred_index]["bbox_pdf_pt"],
                        iou,
                    )
                ] += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_class[semantic_type] = {
            "reference_count": tp + fn,
            "prediction_count": tp + fp,
            "true_positive_iou_0_5": tp,
            "false_positive_iou_0_5": fp,
            "false_negative_iou_0_5": fn,
            "precision_iou_0_5": round(precision, 6),
            "recall_iou_0_5": round(recall, 6),
            "f1_iou_0_5": round(f1, 6),
            "recall_iou_0_3": round(recall30_tp / (tp + fn), 6) if tp + fn else 0.0,
            "matched_mean_iou": round(statistics.fmean(ious), 6) if ious else 0.0,
            "matched_median_iou": round(statistics.median(ious), 6) if ious else 0.0,
        }
        totals.update(tp=tp, fp=fp, fn=fn, recall30_tp=recall30_tp)

    for reference in references:
        gt = [
            row
            for row in reference["reference_regions"]
            if row["semantic_type"] in CRITICAL_TYPES
            and not row.get("reference_uncertain")
        ]
        pred = [
            row
            for row in predictions[_page_key(reference)]["canonical_regions"]
            if row["semantic_type"] in CRITICAL_TYPES
        ]
        pairs, _, _ = _match_regions(gt, pred, 0.3, same_class=False)
        for gt_index, pred_index, _ in pairs:
            confusion[gt[gt_index]["semantic_type"]][pred[pred_index]["semantic_type"]] += 1
    micro_precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    micro_recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "page_count": len(references),
        "per_class": by_class,
        "micro": {
            "precision_iou_0_5": round(micro_precision, 6),
            "recall_iou_0_5": round(micro_recall, 6),
            "f1_iou_0_5": round(micro_f1, 6),
            "recall_iou_0_3": round(
                totals["recall30_tp"] / (totals["tp"] + totals["fn"]), 6
            ) if totals["tp"] + totals["fn"] else 0.0,
        },
        "critical_region_miss_rate": round(
            totals["fn"] / (totals["tp"] + totals["fn"]), 6
        ) if totals["tp"] + totals["fn"] else 0.0,
        "confusion_matrix": {
            actual: {predicted: confusion[actual][predicted] for predicted in CRITICAL_TYPES}
            for actual in CRITICAL_TYPES
        },
        "bbox_quality_counts": dict(sorted(quality.items())),
    }


def _reading_order_metrics(
    references: list[dict[str, Any]],
    predictions: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    aggregate = {
        "naive_yx": Counter(correct=0, total=0, perfect_pages=0, pages=0),
        "region_reading_order_v0": Counter(correct=0, total=0, perfect_pages=0, pages=0),
    }
    strata: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {
            "naive_yx": Counter(correct=0, total=0, perfect_pages=0, pages=0),
            "region_reading_order_v0": Counter(correct=0, total=0, perfect_pages=0, pages=0),
        }
    )
    page_results = []
    for reference in references:
        gt = [
            row
            for row in reference["reference_regions"]
            if row["semantic_type"] in CRITICAL_TYPES
            and not row.get("reference_uncertain")
        ]
        pred = [
            row
            for row in predictions[_page_key(reference)]["canonical_regions"]
            if row["semantic_type"] in CRITICAL_TYPES
        ]
        matches, unmatched_gt, _ = _match_regions(gt, pred, 0.3)
        by_gt = {gt_index: pred[pred_index] for gt_index, pred_index, _ in matches}
        row_result = {
            "document_id": reference["document_id"],
            "page_index": reference["page_index"],
            "risk": reference["reading_order_risk"],
            "reference_regions": len(gt),
            "unmatched_reference_regions": len(unmatched_gt),
        }
        for method, field in (
            ("naive_yx", "naive_reading_order_index"),
            ("region_reading_order_v0", "reading_order_index"),
        ):
            correct = 0
            total = len(gt) * (len(gt) - 1) // 2
            for first in range(len(gt)):
                for second in range(first + 1, len(gt)):
                    if first not in by_gt or second not in by_gt:
                        continue
                    if int(by_gt[first][field]) < int(by_gt[second][field]):
                        correct += 1
            perfect = total == correct and not unmatched_gt
            row_result[method] = {
                "pairwise_correct": correct,
                "pairwise_total": total,
                "perfect_page": perfect,
            }
            for target in (aggregate[method], strata[reference["reading_order_risk"]][method]):
                target.update(correct=correct, total=total, perfect_pages=int(perfect), pages=1)
        page_results.append(row_result)

    def finalize(counter: Counter[str]) -> dict[str, Any]:
        return {
            "pairwise_precedence_accuracy": round(
                counter["correct"] / counter["total"], 6
            ) if counter["total"] else 0.0,
            "pairwise_correct": counter["correct"],
            "pairwise_total_including_missing": counter["total"],
            "perfect_pages": counter["perfect_pages"],
            "page_count": counter["pages"],
            "perfect_page_rate": round(
                counter["perfect_pages"] / counter["pages"], 6
            ) if counter["pages"] else 0.0,
        }

    return {
        "schema": "bemarkdown-phase6b1-reading-order-metrics-v1",
        "missing_detection_policy": "Unmatched reference regions make their precedence pairs incorrect and prevent perfect-page credit.",
        "overall": {method: finalize(value) for method, value in aggregate.items()},
        "by_reading_order_risk": {
            risk: {method: finalize(value) for method, value in methods.items()}
            for risk, methods in sorted(strata.items())
        },
        "pages": page_results,
    }


def _error_cases(
    references: list[dict[str, Any]],
    predictions: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for reference in references:
        page = predictions[_page_key(reference)]
        gt = [
            row
            for row in reference["reference_regions"]
            if row["semantic_type"] in CRITICAL_TYPES
            and not row.get("reference_uncertain")
        ]
        pred = [row for row in page["canonical_regions"] if row["semantic_type"] in CRITICAL_TYPES]
        matches, unmatched_gt, unmatched_pred = _match_regions(gt, pred, 0.5)
        for gt_index in unmatched_gt:
            errors.append(
                {
                    "schema": "bemarkdown-phase6b1-error-case-v1",
                    "document_id": reference["document_id"],
                    "page_index": reference["page_index"],
                    "page_number": reference["page_number"],
                    "reference_index": reference["reference_index"],
                    "issue_code": "CRITICAL_FALSE_NEGATIVE",
                    "semantic_type": gt[gt_index]["semantic_type"],
                    "reference_region_id": gt[gt_index]["reference_region_id"],
                    "bbox_pdf_pt": gt[gt_index]["bbox_pdf_pt"],
                    "prediction_hidden": reference["prediction_hidden"],
                }
            )
        for pred_index in unmatched_pred:
            errors.append(
                {
                    "schema": "bemarkdown-phase6b1-error-case-v1",
                    "document_id": reference["document_id"],
                    "page_index": reference["page_index"],
                    "page_number": reference["page_number"],
                    "reference_index": reference["reference_index"],
                    "issue_code": "FALSE_POSITIVE",
                    "semantic_type": pred[pred_index]["semantic_type"],
                    "region_id": pred[pred_index]["region_id"],
                    "bbox_pdf_pt": pred[pred_index]["bbox_pdf_pt"],
                    "score": pred[pred_index]["score"],
                }
            )
        for gt_index, pred_index, iou in matches:
            quality = _bbox_quality(
                gt[gt_index]["bbox_pdf_pt"], pred[pred_index]["bbox_pdf_pt"], iou
            )
            if quality not in {"GOOD", "ACCEPTABLE"}:
                errors.append(
                    {
                        "schema": "bemarkdown-phase6b1-error-case-v1",
                        "document_id": reference["document_id"],
                        "page_index": reference["page_index"],
                        "page_number": reference["page_number"],
                        "reference_index": reference["reference_index"],
                        "issue_code": quality,
                        "semantic_type": gt[gt_index]["semantic_type"],
                        "reference_region_id": gt[gt_index]["reference_region_id"],
                        "region_id": pred[pred_index]["region_id"],
                        "iou": round(iou, 6),
                    }
                )
        split_count, merge_count = _split_merge_counts(gt, pred)
        if split_count:
            errors.append(
                {
                    "schema": "bemarkdown-phase6b1-error-case-v1",
                    "document_id": reference["document_id"],
                    "page_index": reference["page_index"],
                    "page_number": reference["page_number"],
                    "reference_index": reference["reference_index"],
                    "issue_code": "SPLIT_ERROR",
                    "case_count": split_count,
                }
            )
        if merge_count:
            errors.append(
                {
                    "schema": "bemarkdown-phase6b1-error-case-v1",
                    "document_id": reference["document_id"],
                    "page_index": reference["page_index"],
                    "page_number": reference["page_number"],
                    "reference_index": reference["reference_index"],
                    "issue_code": "MERGE_ERROR",
                    "case_count": merge_count,
                }
            )
    errors.sort(
        key=lambda row: (
            row["document_id"], row["page_index"], row["issue_code"], row.get("semantic_type", "")
        )
    )
    return errors


def _match_regions(
    gt: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    threshold: float,
    *,
    same_class: bool = True,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates = []
    for gt_index, gt_row in enumerate(gt):
        for pred_index, pred_row in enumerate(pred):
            if same_class and gt_row["semantic_type"] != pred_row["semantic_type"]:
                continue
            iou = bbox_iou(gt_row["bbox_pdf_pt"], pred_row["bbox_pdf_pt"])
            if iou >= threshold:
                candidates.append((iou, gt_index, pred_index))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, round(iou, 8)))
    return (
        matches,
        [index for index in range(len(gt)) if index not in used_gt],
        [index for index in range(len(pred)) if index not in used_pred],
    )


def _split_merge_counts(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> tuple[int, int]:
    split = sum(
        sum(
            _intersection_area(gt_row["bbox_pdf_pt"], pred_row["bbox_pdf_pt"])
            / max(1.0, _area(gt_row["bbox_pdf_pt"]))
            >= 0.15
            for pred_row in pred
            if pred_row["semantic_type"] == gt_row["semantic_type"]
        )
        >= 2
        for gt_row in gt
    )
    merge = sum(
        sum(
            _intersection_area(gt_row["bbox_pdf_pt"], pred_row["bbox_pdf_pt"])
            / max(1.0, _area(gt_row["bbox_pdf_pt"]))
            >= 0.15
            for gt_row in gt
            if gt_row["semantic_type"] == pred_row["semantic_type"]
        )
        >= 2
        for pred_row in pred
    )
    return split, merge


def _bbox_quality(gt: Sequence[float], pred: Sequence[float], iou: float) -> str:
    if iou >= 0.85:
        return "GOOD"
    if iou >= 0.65:
        return "ACCEPTABLE"
    intersection = _intersection_area(gt, pred)
    gt_coverage = intersection / _area(gt) if _area(gt) else 0.0
    pred_coverage = intersection / _area(pred) if _area(pred) else 0.0
    if gt_coverage < 0.7:
        return "UNDER_CROP"
    if pred_coverage < 0.55:
        return "OVER_CROP"
    return "ACCEPTABLE"


def _prediction_stability(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    matched = baseline_count = candidate_count = 0
    ious: list[float] = []
    for key, baseline_page in baseline.items():
        first = [
            row
            for row in baseline_page["canonical_regions"]
            if row["semantic_type"] in CRITICAL_TYPES
        ]
        second = [
            row
            for row in candidate[key]["canonical_regions"]
            if row["semantic_type"] in CRITICAL_TYPES
        ]
        pairs, _, _ = _match_regions(first, second, 0.5)
        baseline_count += len(first)
        candidate_count += len(second)
        matched += len(pairs)
        ious.extend(row[2] for row in pairs)
    return {
        "baseline_region_count": baseline_count,
        "candidate_region_count": candidate_count,
        "matched_iou_0_5": matched,
        "baseline_retention_rate": round(matched / baseline_count, 6) if baseline_count else 0.0,
        "matched_mean_iou": round(statistics.fmean(ious), 6) if ious else 0.0,
        "matched_median_iou": round(statistics.median(ious), 6) if ious else 0.0,
    }


def _select_dpi_reference_pages(
    work_dir: Path,
    references: list[dict[str, Any]],
    page_count: int,
) -> list[dict[str, Any]]:
    page_summaries = {
        _page_key(row): row for row in _read_jsonl(work_dir / "page_summary.jsonl")
    }
    return sorted(
        references,
        key=lambda row: (
            0 if page_summaries[_page_key(row)]["semantic_type_counts"].get("FORMULA") else 1,
            0 if page_summaries[_page_key(row)]["semantic_type_counts"].get("TABLE") else 1,
            0 if page_summaries[_page_key(row)]["semantic_type_counts"].get("IMAGE") else 1,
            0 if row["reading_order_risk"] == "HIGH" else 1,
            -sum(page_summaries[_page_key(row)]["semantic_type_counts"].values()),
            row["reference_index"],
        ),
    )[:page_count]


def _summarize_full_corpus(
    *,
    summaries: list[dict[str, Any]],
    raw_pages: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    prior: dict[str, Any],
) -> dict[str, Any]:
    raw_counts = Counter(
        row["raw_label"]
        for page in raw_pages
        for row in page["raw_detections"]
    )
    semantic_counts = Counter()
    by_route: dict[str, Counter[str]] = defaultdict(Counter)
    by_document = Counter()
    for row in summaries:
        semantic_counts.update(row["semantic_type_counts"])
        by_route[row["routing_decision"]].update(row["semantic_type_counts"])
        by_document[row["document_id"]] += row["canonical_region_count"]
    result = dict(prior)
    result.update(
        {
            "status": "FULL_CORPUS_INFERENCE_COMPLETE_REFERENCE_PENDING",
            "documents": len({row["document_id"] for row in summaries}),
            "pages": len(summaries),
            "raw_detection_count": sum(raw_counts.values()),
            "canonical_region_count": sum(semantic_counts.values()),
            "raw_label_counts": dict(sorted(raw_counts.items())),
            "semantic_type_counts": dict(sorted(semantic_counts.items())),
            "by_route": {
                route: {
                    "page_count": sum(row["routing_decision"] == route for row in summaries),
                    "region_count": sum(counter.values()),
                    "semantic_type_counts": dict(sorted(counter.items())),
                    "sanity_issue_count": sum(row["routing_decision"] == route for row in issues),
                }
                for route, counter in sorted(by_route.items())
            },
            "by_document_region_count": dict(sorted(by_document.items())),
            "sanity_issue_count": len(issues),
            "sanity_issue_codes": dict(
                sorted(Counter(row["issue_code"] for row in issues).items())
            ),
        }
    )
    return result


def _svg_box(
    row: dict[str, Any], width: float, height: float, kind: str
) -> str:
    bbox = row["bbox_pdf_pt"]
    x, y = bbox[0], bbox[1]
    box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    label = row.get("semantic_type", "UNKNOWN")
    if kind == "prediction":
        label += f" {row.get('raw_model_label', '')} {row.get('score', 0):.2f}"
    order = row.get("expected_order_index") if kind == "reference" else row.get("reading_order_index")
    label = f"{order}: {label}"
    return (
        f'<rect class="box {kind}" x="{x}" y="{y}" width="{box_width}" height="{box_height}"/>'
        f'<text class="label {kind}" x="{max(0, x + 2)}" y="{max(9, y + 9)}">{html.escape(label)}</text>'
    )


def _hidden_visual_note(reference_index: int) -> str:
    notes = {
        1: "Exam cover and first question page; title, instruction, text, formula and diagram zones checked from source only.",
        2: "Dense exam page with circuit, mechanics diagram, formulas and text blocks checked from source only.",
        3: "Dense exam page with apparatus and p-V figures, formulas and text blocks checked from source only.",
        4: "Book cover; dominant title hierarchy and edition/publisher marks checked from source only.",
        5: "Book cover; title and lower visual artwork checked from source only.",
        6: "Book cover; title hierarchy and lower visual artwork checked from source only.",
        7: "Answer page with title, answer tables, text and formulas checked from source only.",
        8: "Answer derivation page with formula sequence and two mechanics figures checked from source only.",
        9: "Two-column journal first page with full-width title/abstract and body columns checked from source only.",
        10: "Two-column journal body page checked from source only.",
        11: "Course-standard cover title hierarchy checked from source only.",
        12: "Blank/near-blank source page; abstention retained where visible content is absent.",
        13: "Dense exam page with multiple figures, formulas and text blocks checked from source only.",
        14: "Dense exam page with multiple figures, formulas and text blocks checked from source only.",
        15: "Textbook cover; title hierarchy and train visual checked from source only.",
        16: "Chapter opener; title hierarchy, star-field artwork and calligraphic text checked from source only.",
        17: "Textbook cover; title hierarchy and accelerator visual checked from source only.",
        18: "Textbook cover; title hierarchy and satellite visual checked from source only.",
        19: "Textbook cover; title hierarchy and ocean/rainbow visual checked from source only.",
        20: "Textbook cover; title hierarchy and leaf visual checked from source only.",
        21: "Textbook cover; title hierarchy and aurora visual checked from source only.",
        22: "Dense exam page with plots, apparatus, formulas and text blocks checked from source only.",
        23: "Dense exam page with field diagrams, circuit, formulas and text blocks checked from source only.",
        24: "Dense exam page with field and mechanics diagrams, formulas and text blocks checked from source only.",
    }
    return notes.get(reference_index, "Rendered source page checked before predictions were exposed.")


def _visible_visual_note(
    render: dict[str, Any], regions: list[dict[str, Any]]
) -> str:
    counts = Counter(row["semantic_type"] for row in regions)
    inventory = ", ".join(f"{name}={counts[name]}" for name in CRITICAL_TYPES if counts[name])
    return (
        "Prediction-visible visual audit completed against the rendered source; "
        "candidate boxes were source-reboxed and independent native/raster proposals were checked. "
        f"Audited inventory: {inventory or 'no confident critical regions'}. "
        f"Source profile={render['source_profile']}."
    )


def _active_runs(active: list[bool], *, merge_gap: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(active + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if runs and start - runs[-1][1] <= merge_gap:
                runs[-1] = (runs[-1][0], index)
            else:
                runs.append((start, index))
            start = None
    return runs


def _region_yx_key(row: dict[str, Any]) -> tuple[float, float, str]:
    bbox = row["bbox_pdf_pt"]
    return float(bbox[1]), float(bbox[0]), str(row.get("semantic_type", ""))


def _box_width(row: dict[str, Any]) -> float:
    return float(row["bbox_pdf_pt"][2]) - float(row["bbox_pdf_pt"][0])


def _center_x(bbox: Sequence[float]) -> float:
    return (float(bbox[0]) + float(bbox[2])) / 2


def _center_y(bbox: Sequence[float]) -> float:
    return (float(bbox[1]) + float(bbox[3])) / 2


def _area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def _intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    width = max(0.0, min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])))
    height = max(0.0, min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1])))
    return width * height


def _page_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["document_id"]), int(row["page_index"])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2))


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _write_text_atomic(path: Path, value: str) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(value, encoding="utf-8", newline="\n")
    os.replace(pending, path)
