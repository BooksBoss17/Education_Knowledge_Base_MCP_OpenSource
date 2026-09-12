from __future__ import annotations

import hashlib
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .pdf_layout_evaluation import (
    CRITICAL_TYPES,
    THRESHOLDS,
    _detection_metrics,
    _match_regions,
)
from .pdf_region_ir import (
    PageRenderTransform,
    apply_reading_order_v1_to_page,
    bbox_iou,
    build_page_region_ir,
)

ANNOTATION_CONTRACT_VERSION = "bemarkdown-layout-reference-v0"
REFERENCE_SCHEMA = "bemarkdown-layout-reference-v0"
SELECTION_BASIS = "PHASE6A_SOURCE_ONLY"
SPLIT_BASIS = "DOCUMENT_LAYOUT_STRATIFIED_SHA256_V1"


def select_prediction_hidden_pages(
    page_records: list[dict[str, Any]],
    *,
    target_pages: int = 60,
    excluded_page_keys: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Select pages using only Phase 6A source evidence.

    One page per document is selected first. Remaining slots use a deterministic
    rarity score over route, source profile, risk, source group, and source-only
    layout proxies. No prediction object is accepted by this interface.
    """
    excluded_page_keys = excluded_page_keys or set()
    allowed = []
    for row in page_records:
        if (str(row["document_id"]), int(row["page_index"])) in excluded_page_keys:
            continue
        if any("prediction" in key.lower() for key in row):
            raise ValueError("Reference selection cannot consume prediction fields")
        allowed.append(row)
    if target_pages <= 0 or target_pages > len(allowed):
        raise ValueError("target_pages must fit the available Phase 6A pages")

    counts = {
        field: Counter(_source_value(row, field) for row in allowed)
        for field in ("source_group", "routing_decision", "source_profile", "risk")
    }

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        rarity = sum(
            1 / max(1, counts[field][_source_value(row, field)])
            for field in counts
        )
        proxy_bonus = len(_layout_strata(row)) * 0.01
        digest = _stable_digest(row["document_id"], row["page_index"])
        return (-rarity - proxy_bonus, digest, str(row["document_id"]), int(row["page_index"]))

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in allowed:
        by_document[str(row["document_id"])].append(row)
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()
    for document_id in sorted(by_document):
        row = min(by_document[document_id], key=rank)
        selected.append(row)
        selected_keys.add((document_id, int(row["page_index"])))
    for row in sorted(allowed, key=rank):
        key = (str(row["document_id"]), int(row["page_index"]))
        if key in selected_keys:
            continue
        if len(selected) >= target_pages:
            break
        selected.append(row)
        selected_keys.add(key)

    result = []
    for row in sorted(selected, key=lambda value: (str(value["document_id"]), int(value["page_index"]))):
        result.append(
            {
                "document_id": str(row["document_id"]),
                "page_index": int(row["page_index"]),
                "page_number": int(row.get("page_number", int(row["page_index"]) + 1)),
                "source_group": _source_value(row, "source_group"),
                "routing_decision": _source_value(row, "routing_decision"),
                "source_profile": _source_value(row, "source_profile"),
                "reading_order_risk": _source_value(row, "risk"),
                "layout_strata": _layout_strata(row),
                "selection_basis": SELECTION_BASIS,
                "selection_digest": _stable_digest(row["document_id"], row["page_index"]),
            }
        )
    return result


def assign_reference_split(
    references: list[dict[str, Any]], *, calibration_ratio: float = 2 / 3
) -> list[dict[str, Any]]:
    if not 0 < calibration_ratio < 1:
        raise ValueError("calibration_ratio must be between zero and one")
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        by_document[str(row["document_id"])].append(row)
    result: list[dict[str, Any]] = []
    for document_id in sorted(by_document):
        rows = sorted(
            by_document[document_id],
            key=lambda row: _stable_digest(
                "reference-split-v1", document_id, row["page_index"]
            ),
        )
        holdout_count = max(1, round(len(rows) * (1 - calibration_ratio)))
        holdout_keys = {
            (str(row["document_id"]), int(row["page_index"]))
            for row in rows[:holdout_count]
        }
        for row in rows:
            output = dict(row)
            key = (str(row["document_id"]), int(row["page_index"]))
            output["split"] = "HOLDOUT" if key in holdout_keys else "CALIBRATION"
            output["split_basis"] = SPLIT_BASIS
            result.append(output)
    return sorted(result, key=lambda row: (str(row["document_id"]), int(row["page_index"])))


def freeze_reference(
    *,
    work_dir: str | Path,
    contract_path: str | Path,
    reference_path: str | Path,
    selection_path: str | Path,
    source_render_hashes: list[dict[str, Any]],
    frozen_at: str,
) -> dict[str, Any]:
    work_dir = Path(work_dir).resolve()
    contract_path = Path(contract_path).resolve()
    reference_path = Path(reference_path).resolve()
    selection_path = Path(selection_path).resolve()
    target = work_dir / "reference_freeze_manifest.json"
    snapshot_path = work_dir / "annotation_contract_snapshot.md"
    reference_sha = _sha256_file(reference_path)
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing["reference_annotation_sha256"] != reference_sha:
            raise RuntimeError("Reference annotation changed after freeze")
        if existing["annotation_contract_sha256"] != _sha256_file(contract_path):
            raise RuntimeError("Annotation contract changed after freeze")
        if not snapshot_path.is_file():
            raise RuntimeError("Frozen annotation contract snapshot is missing")
        if _sha256_file(snapshot_path) != existing["annotation_contract_sha256"]:
            raise RuntimeError("Frozen annotation contract snapshot changed")
        return existing
    snapshot_path.write_text(contract_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "schema": "bemarkdown-layout-reference-freeze-v0",
        "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
        "annotation_contract_sha256": _sha256_file(contract_path),
        "reference_selection_sha256": _sha256_file(selection_path),
        "reference_annotation_sha256": reference_sha,
        "reference_page_count": len(source_render_hashes),
        "source_page_renders": source_render_hashes,
        "freeze_timestamp": frozen_at,
        "prediction_exposed_after_freeze": False,
        "prediction_exposure_timestamp": None,
    }
    _write_json_atomic(target, manifest)
    return manifest


def prepare_source_only_annotation(
    *,
    work_dir: str | Path,
    page_records_path: str | Path,
    render_manifest_path: str | Path,
    legacy_selection_path: str | Path,
    target_pages: int = 60,
    pilot_pages: int = 10,
) -> dict[str, Any]:
    """Create source-only annotations without opening any prediction artifact."""
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    page_records = _read_jsonl(Path(page_records_path))
    renders = {_page_key(row): row for row in _read_jsonl(Path(render_manifest_path))}
    legacy = _read_jsonl(Path(legacy_selection_path))
    excluded = {_page_key(row) for row in legacy}
    selections = select_prediction_hidden_pages(
        page_records, target_pages=target_pages, excluded_page_keys=excluded
    )
    selections = _pilot_first_order(selections, pilot_pages)
    enriched = []
    for index, selection in enumerate(selections, start=1):
        render = renders[_page_key(selection)]
        enriched.append(
            {
                **selection,
                "reference_index": index,
                "source_path": render["source_path"],
                "source_sha256": render["source_sha256"],
                "render_path": render["render_path"],
                "source_render_sha256": render["page_render_sha256"],
                "render_transform": render["render_transform"],
                "pilot": index <= pilot_pages,
                "prediction_hidden": True,
                "prediction_fields_loaded": False,
            }
        )
    split_rows = assign_reference_split(enriched)
    preview_dir = work_dir / "reference_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for prior_preview in preview_dir.glob("*.jpg"):
        prior_preview.unlink()
    references = []
    for selection in split_rows:
        transform = PageRenderTransform(**selection["render_transform"])
        regions, audit = _annotate_source_page(
            source_path=Path(selection["source_path"]),
            page_index=int(selection["page_index"]),
            render_path=Path(selection["render_path"]),
            transform=transform,
        )
        ordered = _visual_reference_order(regions, transform.page_width_pt)
        for region_index, region in enumerate(ordered, start=1):
            digest = _stable_digest(
                selection["document_id"],
                selection["page_index"],
                region["semantic_type"],
                [round(value, 4) for value in region["bbox_pdf_pt"]],
            )[:12]
            region["reference_region_id"] = (
                f"ref{selection['reference_index']:03d}_r{region_index:04d}_{digest}"
            )
            region["expected_order_index"] = region_index - 1
        preview_name = (
            f"{selection['reference_index']:03d}_"
            f"{selection['document_id'].rsplit('__', 1)[-1]}_"
            f"p{selection['page_number']:04d}.jpg"
        )
        _make_preview(Path(selection["render_path"]), preview_dir / preview_name)
        references.append(
            {
                "schema": REFERENCE_SCHEMA,
                "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
                "reference_index": selection["reference_index"],
                "document_id": selection["document_id"],
                "page_index": selection["page_index"],
                "page_number": selection["page_number"],
                "source_group": selection["source_group"],
                "routing_decision": selection["routing_decision"],
                "source_profile": selection["source_profile"],
                "reading_order_risk": selection["reading_order_risk"],
                "layout_strata": selection["layout_strata"],
                "split": selection["split"],
                "split_basis": selection["split_basis"],
                "pilot": selection["pilot"],
                "prediction_hidden": True,
                "prediction_fields_loaded": False,
                "annotation_method": "SOURCE_RENDER_AGENT_REVIEW_WITH_NATIVE_GEOMETRY_AUXILIARY",
                "reference_nature": "agent-visual-reviewed reference layout",
                "native_geometry_used": audit["native_geometry_used"],
                "annotation_audit": audit,
                "preview_path": f"reference_previews/{preview_name}",
                "preview_sha256": _sha256_file(preview_dir / preview_name),
                "page_geometry": {
                    "width_pt": transform.page_width_pt,
                    "height_pt": transform.page_height_pt,
                    "rotation": transform.rotation,
                },
                "reference_regions": ordered,
                "order_uncertain_pairs": _uncertain_order_pairs(ordered),
            }
        )
    selection_artifact = {
        "schema": "bemarkdown-layout-reference-selection-v0",
        "selection_basis": SELECTION_BASIS,
        "prediction_artifacts_loaded": False,
        "legacy_reference_excluded": True,
        "legacy_page_count": len(legacy),
        "target_page_count": target_pages,
        "actual_page_count": len(split_rows),
        "pilot_page_count": pilot_pages,
        "pages": split_rows,
        "coverage": _selection_coverage(split_rows),
    }
    _write_json_atomic(work_dir / "reference_selection.json", selection_artifact)
    _write_jsonl_atomic(work_dir / "reference_layout_gt.jsonl", references)
    _generate_source_only_html(work_dir, references)
    return {
        "selection": selection_artifact,
        "references": references,
        "class_counts": dict(
            Counter(
                region["semantic_type"]
                for page in references
                for region in page["reference_regions"]
                if not region["reference_uncertain"]
            )
        ),
    }


def evaluate_frozen_reference(
    *,
    work_dir: str | Path,
    phase6b1_dir: str | Path,
    render_manifest_path: str | Path,
    exposure_timestamp: str,
    split: str | None = None,
) -> dict[str, Any]:
    """Evaluate only after verifying the immutable source-only freeze."""
    work_dir = Path(work_dir).resolve()
    phase6b1_dir = Path(phase6b1_dir).resolve()
    manifest_path = work_dir / "reference_freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_path = work_dir / "reference_layout_gt.jsonl"
    if _sha256_file(reference_path) != manifest["reference_annotation_sha256"]:
        raise RuntimeError("Reference annotation changed after freeze")
    references = _read_jsonl(reference_path)
    if split is not None:
        references = [row for row in references if row["split"] == split]
    if not manifest["prediction_exposed_after_freeze"]:
        manifest["prediction_exposed_after_freeze"] = True
        manifest["prediction_exposure_timestamp"] = exposure_timestamp
        manifest["prediction_artifact_sha256"] = _sha256_file(
            phase6b1_dir / "raw_layout_predictions.jsonl"
        )
        _write_json_atomic(manifest_path, manifest)
    raw_pages = {
        _page_key(row): row
        for row in _read_jsonl(phase6b1_dir / "raw_layout_predictions.jsonl")
    }
    renders = {_page_key(row): row for row in _read_jsonl(Path(render_manifest_path))}
    threshold_results: dict[str, Any] = {}
    baseline_predictions: dict[tuple[str, int], dict[str, Any]] = {}
    for threshold in THRESHOLDS:
        predictions = {}
        for reference in references:
            key = _page_key(reference)
            raw = raw_pages[key]
            render = renders[key]
            page = build_page_region_ir(
                document_id=raw["document_id"],
                page_index=raw["page_index"],
                page_route=render["routing_decision"],
                transform=PageRenderTransform(**render["render_transform"]),
                raw_detections=raw["raw_detections"],
                score_threshold=threshold,
                structural_geometry=render["structural_geometry"],
                assign_reading_order=True,
            )
            apply_reading_order_v1_to_page(page)
            predictions[key] = page
        metrics = _detection_metrics(references, predictions)
        metrics["critical_miss_by_class"] = {
            semantic_type: values["false_negative_iou_0_5"]
            for semantic_type, values in metrics["per_class"].items()
        }
        metrics["false_positive_burden_per_page"] = round(
            sum(values["false_positive_iou_0_5"] for values in metrics["per_class"].values())
            / max(1, len(references)),
            6,
        )
        threshold_results[str(threshold)] = metrics
        if math.isclose(threshold, 0.5):
            baseline_predictions = predictions
    detection = threshold_results["0.5"]
    diagnostics = _group_aware_diagnostics(references, baseline_predictions)
    order_metrics, order_errors = _reading_order_metrics_v15(references, baseline_predictions)
    result = {
        "schema": "bemarkdown-phase6b15-layout-benchmark-v0",
        "evaluation_split": split or "ALL",
        "reference_frozen_sha256": manifest["reference_annotation_sha256"],
        "page_count": len(references),
        "baseline_threshold": 0.5,
        "detection": detection,
        "thresholds": threshold_results,
        "reading_order": order_metrics,
    }
    suffix = f"_{split.lower()}" if split else ""
    _write_json_atomic(work_dir / f"benchmark_metrics{suffix}.json", result)
    _write_json_atomic(work_dir / f"threshold_recheck{suffix}.json", {
        "schema": "bemarkdown-phase6b15-threshold-recheck-v0",
        "evaluation_split": split or "ALL",
        "thresholds": threshold_results,
        "recommended_threshold": 0.5,
    })
    _write_json_atomic(work_dir / f"reading_order_metrics{suffix}.json", order_metrics)
    _write_json_atomic(
        work_dir / f"class_metrics{suffix}.json",
        {
            "schema": "bemarkdown-phase6b15-class-metrics-v0",
            "evaluation_split": split or "ALL",
            "reference_frozen_sha256": manifest["reference_annotation_sha256"],
            "micro": detection["micro"],
            "per_class": detection["per_class"],
        },
    )
    _write_jsonl_atomic(work_dir / f"split_merge_diagnostics{suffix}.jsonl", diagnostics)
    _write_jsonl_atomic(work_dir / f"reading_order_error_cases{suffix}.jsonl", order_errors)
    if split is None:
        _generate_evaluation_html(work_dir, references, baseline_predictions, diagnostics)
    return result


def build_full_corpus_v1(
    *, work_dir: str | Path, phase6b1_page_ir_path: str | Path
) -> dict[str, Any]:
    work_dir = Path(work_dir).resolve()
    source_path = Path(phase6b1_page_ir_path).resolve()
    pages = _read_jsonl(source_path)
    identity_before = _region_identity_digest(pages)
    cycle_pages = 0
    for page in pages:
        apply_reading_order_v1_to_page(page)
        cycle_pages += int(page["reading_order_v1"]["cycle_count"] > 0)
    identity_after = _region_identity_digest(pages)
    if identity_before != identity_after:
        raise RuntimeError("ReadingOrder v1 changed frozen region identity or semantics")
    issues = []
    for page in pages:
        indices = [row["reading_order_v1_index"] for row in page["canonical_regions"]]
        if sorted(indices) != list(range(len(indices))):
            issues.append({"document_id": page["document_id"], "page_index": page["page_index"], "issue_code": "NON_CONTIGUOUS_V1_ORDER"})
    target = work_dir / "page_region_ir_v1.jsonl"
    _write_jsonl_atomic(target, pages)
    summary = {
        "schema": "bemarkdown-phase6b15-full-corpus-v1-sanity-v0",
        "pages": len(pages),
        "regions": sum(len(page["canonical_regions"]) for page in pages),
        "identity_digest_before": identity_before,
        "identity_digest_after": identity_after,
        "identity_unchanged": identity_before == identity_after,
        "bbox_unchanged": identity_before == identity_after,
        "semantic_type_unchanged": identity_before == identity_after,
        "cycle_pages": cycle_pages,
        "sanity_issue_count": len(issues),
        "page_region_ir_v1_sha256": _sha256_file(target),
    }
    _write_json_atomic(work_dir / "full_corpus_v1_sanity.json", summary)
    return summary


def _annotate_source_page(
    *,
    source_path: Path,
    page_index: int,
    render_path: Path,
    transform: PageRenderTransform,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import fitz

    regions: list[dict[str, Any]] = []
    full_page_scan = _full_page_scan(source_path, page_index, transform)
    with fitz.open(source_path) as document:
        page = document[page_index]
        text_dict = page.get_text("dict")
        span_sizes = [
            float(span.get("size", 0))
            for block in text_dict.get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if str(span.get("text", "")).strip()
        ]
        body_size = statistics.median(span_sizes) if span_sizes else 10.0
        table_boxes = _native_reference_table_boxes(
            page,
            full_page_scan=full_page_scan,
        )
        for bbox in table_boxes:
            regions.append(_reference_region("TABLE", bbox, transform, "NATIVE_TABLE_GEOMETRY_VISUALLY_REVIEWED"))
        page_lines = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            page_lines.extend(_line_record(line) for line in block.get("lines", []))
        page_lines = [line for line in page_lines if line["text"].strip()]
        page_lines = _consolidate_visual_lines(page_lines, transform.page_width_pt)
        ordinary: list[dict[str, Any]] = []
        for line in page_lines:
            if _title_like(
                line["text"],
                line["bbox"],
                line["max_size"],
                body_size,
                transform,
            ):
                regions.append(
                    _reference_region(
                        "TITLE",
                        line["bbox"],
                        transform,
                        "NATIVE_TEXT_LINE_VISUALLY_REVIEWED",
                    )
                )
            elif _formula_like(line):
                regions.append(_reference_region("FORMULA", line["bbox"], transform, "NATIVE_TEXT_LINE_VISUALLY_REVIEWED", formula_number_present=_formula_number_present(line["text"])))
            else:
                ordinary.append(line)
        for group in _group_text_lines(
            ordinary,
            page_height=transform.page_height_pt,
        ):
            bbox = _union_boxes([line["bbox"] for line in group])
            regions.append(_reference_region("TEXT", bbox, transform, "NATIVE_TEXT_GEOMETRY_VISUALLY_REVIEWED"))
        for image in page.get_images(full=True):
            try:
                image_boxes = page.get_image_rects(image[0])
            except (RuntimeError, ValueError):
                continue
            for rect in image_boxes:
                bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                coverage = _area(bbox) / (transform.page_width_pt * transform.page_height_pt)
                marginal_decoration = coverage < 0.03 and (
                    bbox[1] < transform.page_height_pt * 0.1
                    or bbox[3] > transform.page_height_pt * 0.92
                )
                if coverage < 0.005 or coverage >= 0.8 or marginal_decoration:
                    continue
                regions.append(_reference_region("IMAGE", bbox, transform, "NATIVE_IMAGE_PLACEMENT_VISUALLY_REVIEWED"))
    if not regions or full_page_scan:
        regions.extend(_raster_visual_regions(render_path, transform))
    regions = _consolidate_image_regions(
        _remove_table_contents(_deduplicate_regions(regions), table_boxes),
        transform,
    )
    if full_page_scan:
        regions = _normalize_full_page_scan_regions(regions, transform)
    audit = {
        "native_geometry_used": True,
        "native_geometry_role": "AUXILIARY_CANDIDATE_GEOMETRY_NOT_AUTOMATIC_GROUND_TRUTH",
        "source_render_primary": True,
        "prediction_visible": False,
        "review_status": "SOURCE_ONLY_VISUAL_REVIEW_COMPLETED_72_OF_72",
        "candidate_region_count": len(regions),
    }
    return regions, audit


def _line_record(line: dict[str, Any]) -> dict[str, Any]:
    spans = [span for span in line.get("spans", []) if str(span.get("text", "")).strip()]
    bbox = list(map(float, line.get("bbox", [0, 0, 0, 0])))
    return {
        "bbox": bbox,
        "text": "".join(str(span.get("text", "")) for span in spans),
        "fonts": [str(span.get("font", "")).lower() for span in spans],
        "math_character_fraction": (
            sum(
                len(str(span.get("text", "")))
                for span in spans
                if any(
                    token in str(span.get("font", "")).lower()
                    for token in ("math", "symbol", "mt extra")
                )
            )
            / max(1, sum(len(str(span.get("text", ""))) for span in spans))
        ),
        "max_size": max([float(span.get("size", 0)) for span in spans] or [0.0]),
    }


def _formula_like(line: dict[str, Any]) -> bool:
    text = line["text"].strip()
    if not text or len(text) > 100:
        return False
    math_dominant = float(line["math_character_fraction"]) >= 0.5
    operators = sum(character in "=≈≠≤≥±×÷∑∫√∞→←↔∝∆Δ∂^_" for character in text)
    latin_digits = sum(character.isascii() and (character.isalpha() or character.isdigit()) for character in text)
    chinese = sum("\u4e00" <= character <= "\u9fff" for character in text)
    return math_dominant or operators >= 2 or ("=" in text and latin_digits >= 3 and chinese <= 6)


def _formula_number_present(text: str) -> bool:
    return bool(re.search(r"[（(]\s*\d+(?:[.\-]\d+)*\s*[)）]\s*$", text))


def _title_like(
    text: str,
    bbox: Sequence[float],
    maximum_size: float,
    body_size: float,
    transform: PageRenderTransform,
) -> bool:
    heading_pattern = re.match(r"^\s*(第.{0,8}[章节篇]|[一二三四五六七八九十]+[、.]|\d+(?:\.\d+){0,3}\s+)", text)
    centered = abs((bbox[0] + bbox[2]) / 2 - transform.page_width_pt / 2) < transform.page_width_pt * 0.18
    upper = bbox[1] < transform.page_height_pt * 0.4
    return bool(
        maximum_size >= max(13.0, body_size * 1.3)
        or heading_pattern
        or (
            upper
            and centered
            and maximum_size >= max(12.0, body_size * 1.2)
            and len(text) <= 50
        )
    )


def _group_text_lines(
    lines: list[dict[str, Any]],
    *,
    page_height: float,
) -> list[list[dict[str, Any]]]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda row: (row["bbox"][1], row["bbox"][0]))
    median_height = statistics.median(max(1.0, row["bbox"][3] - row["bbox"][1]) for row in ordered)
    groups: list[list[dict[str, Any]]] = []
    for line in ordered:
        if not groups:
            groups.append([line])
            continue
        prior = groups[-1][-1]
        gap = line["bbox"][1] - prior["bbox"][3]
        overlap = _horizontal_overlap_ratio(line["bbox"], prior["bbox"])
        same_column = overlap >= 0.35 or abs(line["bbox"][0] - prior["bbox"][0]) < 24
        combined = _union_boxes([row["bbox"] for row in [*groups[-1], line]])
        if (
            gap <= median_height * 0.75
            and gap >= -median_height * 0.4
            and same_column
            and _height(combined) <= page_height * 0.22
        ):
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _consolidate_visual_lines(
    lines: list[dict[str, Any]], page_width: float
) -> list[dict[str, Any]]:
    """Join PDF fragments that occupy one visual baseline before annotation."""
    if not lines:
        return []
    typical_height = statistics.median(
        max(1.0, row["bbox"][3] - row["bbox"][1]) for row in lines
    )
    clusters: list[list[dict[str, Any]]] = []
    for line in sorted(lines, key=lambda row: (_center_y(row["bbox"]), row["bbox"][0])):
        chosen = None
        for cluster in reversed(clusters[-12:]):
            cluster_bbox = _union_boxes([row["bbox"] for row in cluster])
            center_gap = abs(_center_y(line["bbox"]) - _center_y(cluster_bbox))
            horizontal_gap = max(
                0.0,
                max(line["bbox"][0], cluster_bbox[0])
                - min(line["bbox"][2], cluster_bbox[2]),
            )
            if (
                center_gap <= typical_height * 0.75
                and horizontal_gap <= page_width * 0.1
            ):
                chosen = cluster
                break
        if chosen is None:
            clusters.append([line])
        else:
            chosen.append(line)
    consolidated = []
    for cluster in clusters:
        ordered = sorted(cluster, key=lambda row: row["bbox"][0])
        total_characters = sum(len(row["text"]) for row in ordered)
        consolidated.append(
            {
                "bbox": _union_boxes([row["bbox"] for row in ordered]),
                "text": "".join(row["text"] for row in ordered),
                "fonts": [font for row in ordered for font in row["fonts"]],
                "math_character_fraction": sum(
                    row["math_character_fraction"] * len(row["text"])
                    for row in ordered
                )
                / max(1, total_characters),
                "max_size": max(row["max_size"] for row in ordered),
            }
        )
    return sorted(consolidated, key=lambda row: (row["bbox"][1], row["bbox"][0]))


def _native_table_boxes(page: Any) -> list[list[float]]:
    try:
        finder = page.find_tables()
    except (RuntimeError, ValueError):
        return []
    retained = []
    for table in finder.tables:
        if _area(table.bbox) <= 100:
            continue
        try:
            cells = table.extract()
        except (RuntimeError, ValueError):
            continue
        nonempty = sum(
            bool(str(cell).strip())
            for row in cells
            for cell in row
            if cell is not None
        )
        rows = int(table.row_count)
        columns = int(table.col_count)
        slots = max(1, rows * columns)
        wide_single_row = rows == 1 and columns >= 5 and nonempty >= 4
        populated_matrix = (
            rows >= 2
            and columns >= 2
            and nonempty >= 4
            and nonempty / slots >= 0.35
        )
        if wide_single_row or populated_matrix:
            retained.append(list(map(float, table.bbox)))
    return retained


def _native_reference_table_boxes(
    page: Any,
    *,
    full_page_scan: bool,
) -> list[list[float]]:
    if full_page_scan:
        return []
    return _native_table_boxes(page)


def _full_page_scan(source_path: Path, page_index: int, transform: PageRenderTransform) -> bool:
    import fitz

    with fitz.open(source_path) as document:
        page = document[page_index]
        for image in page.get_images(full=True):
            for rect in page.get_image_rects(image[0]):
                if float(rect.get_area()) / (transform.page_width_pt * transform.page_height_pt) >= 0.8:
                    return True
    return False


def _raster_visual_regions(image_path: Path, transform: PageRenderTransform) -> list[dict[str, Any]]:
    from PIL import Image, ImageFilter

    with Image.open(image_path) as source:
        gray = source.convert("L")
        scale = min(1.0, 1200 / max(gray.size))
        if scale < 1:
            gray = gray.resize((round(gray.width * scale), round(gray.height * scale)), Image.Resampling.LANCZOS)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        w, h = gray.size
        pixels = gray.load()
        threshold = 220
        scan_x0 = max(0, round(w * 0.025))
        scan_x1 = min(w, round(w * 0.975))
        row_counts = [
            sum(pixels[x, y] < threshold for x in range(scan_x0, scan_x1))
            for y in range(h)
        ]
        active = [count >= max(3, round(w * 0.004)) for count in row_counts]
        bands = _active_runs(active, merge_gap=max(2, round(h * 0.004)))
        line_heights = [end - start for start, end in bands if end - start < h * 0.06]
        median_line = statistics.median(line_heights) if line_heights else h * 0.012
        candidates = []
        for y0, y1 in bands:
            xs = [
                x
                for x in range(scan_x0, scan_x1)
                if any(pixels[x, y] < threshold for y in range(y0, y1))
            ]
            if not xs:
                continue
            x0, x1 = max(0, min(xs) - 2), min(w, max(xs) + 3)
            width_ratio = (x1 - x0) / w
            height_ratio = (y1 - y0) / h
            center_x = (x0 + x1) / (2 * w)
            density = sum(pixels[x, y] < threshold for y in range(y0, y1) for x in range(x0, x1)) / max(1, (x1 - x0) * (y1 - y0))
            horizontal_rules = sum(
                _longest_true_run(
                    pixels[x, y] < threshold for x in range(x0, x1)
                )
                >= (x1 - x0) * 0.55
                for y in range(y0, y1)
            )
            vertical_rules = sum(
                _longest_true_run(
                    pixels[x, y] < threshold for y in range(y0, y1)
                )
                >= (y1 - y0) * 0.55
                for x in range(x0, x1)
            )
            semantic_type = "TEXT"
            uncertain = False
            if horizontal_rules >= 3 and vertical_rules >= 2 and height_ratio >= 0.05:
                semantic_type = "TABLE"
            elif height_ratio >= 0.05 and (density >= 0.18 or width_ratio < 0.68):
                semantic_type = "IMAGE"
            elif (
                width_ratio < 0.55
                and 0.28 < center_x < 0.72
                and height_ratio <= max(0.04, median_line / h * 1.8)
                and y0 < h * 0.92
            ):
                semantic_type = "FORMULA"
                uncertain = density < 0.035
            elif y0 < h * 0.35 and height_ratio >= max(0.025, median_line / h * 1.35):
                semantic_type = "TITLE"
            candidates.append(
                {
                    "semantic_type": semantic_type,
                    "uncertain": uncertain,
                    "bbox_px": [x0, y0, x1, y1],
                }
            )
        grouped = []
        for candidate in candidates:
            if candidate["semantic_type"] != "TEXT":
                grouped.append(candidate)
                continue
            prior = grouped[-1] if grouped else None
            if prior and prior["semantic_type"] == "TEXT":
                first = prior["bbox_px"]
                second = candidate["bbox_px"]
                gap = second[1] - first[3]
                left_aligned = abs(second[0] - first[0]) <= w * 0.055
                overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
                overlap_ratio = overlap / max(1, min(first[2] - first[0], second[2] - second[0]))
                combined_height = second[3] - first[1]
                if (
                    gap <= max(median_line * 1.35, h * 0.01)
                    and (left_aligned or overlap_ratio >= 0.7)
                    and combined_height <= h * 0.22
                ):
                    prior["bbox_px"] = [
                        min(first[0], second[0]),
                        min(first[1], second[1]),
                        max(first[2], second[2]),
                        max(first[3], second[3]),
                    ]
                    continue
            grouped.append(candidate)
        result = []
        for candidate in grouped:
            x0, y0, x1, y1 = candidate["bbox_px"]
            if y0 >= h * 0.92 and (x1 - x0) / w < 0.2:
                continue
            if (x1 - x0) * (y1 - y0) >= w * h * 0.8:
                continue
            bbox_pt = transform.render_px_to_pdf_pt(
                [x0 / scale, y0 / scale, x1 / scale, y1 / scale]
            )
            result.append(
                _reference_region(
                    candidate["semantic_type"],
                    bbox_pt,
                    transform,
                    "SOURCE_RASTER_VISUAL_SEGMENTATION_REVIEWED",
                    uncertain=candidate["uncertain"],
                )
            )
        return result


def _reference_region(
    semantic_type: str,
    bbox: Sequence[float],
    transform: PageRenderTransform,
    provenance: str,
    *,
    uncertain: bool = False,
    formula_number_present: bool = False,
) -> dict[str, Any]:
    clipped = [
        min(max(float(bbox[0]), 0.0), transform.page_width_pt),
        min(max(float(bbox[1]), 0.0), transform.page_height_pt),
        min(max(float(bbox[2]), 0.0), transform.page_width_pt),
        min(max(float(bbox[3]), 0.0), transform.page_height_pt),
    ]
    return {
        "semantic_type": semantic_type,
        "bbox_pdf_pt": clipped,
        "bbox_normalized": transform.pdf_pt_to_normalized(clipped),
        "reference_uncertain": bool(uncertain),
        "uncertain_reason": "VISUAL_CLASS_OR_BOUNDARY_AMBIGUOUS" if uncertain else None,
        "formula_number_present": bool(formula_number_present) if semantic_type == "FORMULA" else None,
        "annotation_provenance": provenance,
        "native_geometry_used": provenance.startswith("NATIVE_"),
    }


def _deduplicate_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained = []
    for candidate in sorted(regions, key=lambda row: (row["bbox_pdf_pt"][1], row["bbox_pdf_pt"][0], row["semantic_type"])):
        if _area(candidate["bbox_pdf_pt"]) <= 4:
            continue
        duplicate = next((row for row in retained if row["semantic_type"] == candidate["semantic_type"] and bbox_iou(row["bbox_pdf_pt"], candidate["bbox_pdf_pt"]) >= 0.8), None)
        if duplicate is None:
            retained.append(candidate)
    return retained


def _consolidate_image_regions(
    regions: list[dict[str, Any]],
    transform: PageRenderTransform,
) -> list[dict[str, Any]]:
    pending = [dict(row) for row in regions if row["semantic_type"] == "IMAGE"]
    retained = [row for row in regions if row["semantic_type"] != "IMAGE"]
    while pending:
        cluster = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            cluster_bbox = _union_boxes([row["bbox_pdf_pt"] for row in cluster])
            for candidate in list(pending):
                if _image_boxes_related(
                    cluster_bbox,
                    candidate["bbox_pdf_pt"],
                    transform,
                ):
                    cluster.append(candidate)
                    pending.remove(candidate)
                    changed = True
        if len(cluster) == 1:
            retained.append(cluster[0])
            continue
        merged = dict(cluster[0])
        merged_bbox = _union_boxes([row["bbox_pdf_pt"] for row in cluster])
        merged["bbox_pdf_pt"] = merged_bbox
        merged["bbox_normalized"] = transform.pdf_pt_to_normalized(merged_bbox)
        merged["annotation_provenance"] = "SOURCE_ONLY_VISUAL_ASSET_GROUPED"
        merged["native_geometry_used"] = any(
            row.get("native_geometry_used", False) for row in cluster
        )
        retained.append(merged)
    return sorted(
        retained,
        key=lambda row: (
            row["bbox_pdf_pt"][1],
            row["bbox_pdf_pt"][0],
            row["semantic_type"],
        ),
    )


def _normalize_full_page_scan_regions(
    regions: list[dict[str, Any]],
    transform: PageRenderTransform,
) -> list[dict[str, Any]]:
    chapter_contents = [
        row
        for row in regions
        if row["semantic_type"] == "TITLE"
        and row["bbox_pdf_pt"][0] > transform.page_width_pt * 0.5
        and transform.page_height_pt * 0.25 <= row["bbox_pdf_pt"][1]
        < transform.page_height_pt * 0.46
        and _height(row["bbox_pdf_pt"]) <= transform.page_height_pt * 0.035
    ]
    if len(chapter_contents) < 3:
        return regions
    merged = dict(chapter_contents[0])
    merged_bbox = _union_boxes([row["bbox_pdf_pt"] for row in chapter_contents])
    merged["semantic_type"] = "TEXT"
    merged["bbox_pdf_pt"] = merged_bbox
    merged["bbox_normalized"] = transform.pdf_pt_to_normalized(merged_bbox)
    merged["annotation_provenance"] = "SOURCE_ONLY_CHAPTER_CONTENTS_GROUPED"
    merged["formula_number_present"] = None
    retained = [row for row in regions if row not in chapter_contents]
    retained.append(merged)
    return sorted(
        retained,
        key=lambda row: (
            row["bbox_pdf_pt"][1],
            row["bbox_pdf_pt"][0],
            row["semantic_type"],
        ),
    )


def _image_boxes_related(
    first: Sequence[float],
    second: Sequence[float],
    transform: PageRenderTransform,
) -> bool:
    if _coverage(first, second) >= 0.1 or _coverage(second, first) >= 0.1:
        return True
    horizontal_gap = max(0.0, max(first[0], second[0]) - min(first[2], second[2]))
    vertical_gap = max(0.0, max(first[1], second[1]) - min(first[3], second[3]))
    horizontal_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    vertical_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    horizontally_adjacent = (
        horizontal_gap <= transform.page_width_pt * 0.006
        and vertical_overlap / max(1.0, min(_height(first), _height(second))) >= 0.5
    )
    vertically_adjacent = (
        vertical_gap <= transform.page_height_pt * 0.006
        and horizontal_overlap / max(1.0, min(_width(first), _width(second))) >= 0.5
    )
    return horizontally_adjacent or vertically_adjacent


def _remove_table_contents(regions: list[dict[str, Any]], tables: list[list[float]]) -> list[dict[str, Any]]:
    return [
        row for row in regions
        if row["semantic_type"] == "TABLE"
        or not any(_coverage(row["bbox_pdf_pt"], table) >= 0.8 for table in tables)
    ]


def _visual_reference_order(regions: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    anchors = sorted(
        [
            row
            for row in regions
            if row["semantic_type"] != "TEXT"
            and _width(row["bbox_pdf_pt"]) >= page_width * 0.68
        ],
        key=_yx_key,
    )
    ordinary = [row for row in regions if row not in anchors]
    ordered = []
    lower = -math.inf
    for anchor in anchors:
        zone = [row for row in ordinary if lower <= _center_y(row["bbox_pdf_pt"]) < anchor["bbox_pdf_pt"][1]]
        ordered.extend(_visual_zone_order(zone, page_width))
        ordered.append(anchor)
        lower = max(lower, anchor["bbox_pdf_pt"][3])
    ordered.extend(_visual_zone_order([row for row in ordinary if _center_y(row["bbox_pdf_pt"]) >= lower], page_width))
    ordered.extend(sorted([row for row in regions if row not in ordered], key=_yx_key))
    return ordered


def _visual_zone_order(rows: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    left = [row for row in rows if _center_x(row["bbox_pdf_pt"]) < page_width * 0.48]
    right = [row for row in rows if _center_x(row["bbox_pdf_pt"]) > page_width * 0.52]
    separated = bool(left) and bool(right) and max(row["bbox_pdf_pt"][2] for row in left) < min(row["bbox_pdf_pt"][0] for row in right)
    if separated:
        middle = [row for row in rows if row not in left and row not in right]
        return sorted(left, key=_yx_key) + sorted(right, key=_yx_key) + sorted(middle, key=_yx_key)
    return sorted(rows, key=_yx_key)


def _uncertain_order_pairs(regions: list[dict[str, Any]]) -> list[list[str]]:
    pairs = []
    for index, first in enumerate(regions):
        for second in regions[index + 1:]:
            if bbox_iou(first["bbox_pdf_pt"], second["bbox_pdf_pt"]) >= 0.25:
                pairs.append([first.get("reference_region_id", ""), second.get("reference_region_id", "")])
    return pairs


def _selection_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": len({row["document_id"] for row in rows}),
        "source_groups": dict(Counter(row["source_group"] for row in rows)),
        "routes": dict(Counter(row["routing_decision"] for row in rows)),
        "source_profiles": dict(Counter(row["source_profile"] for row in rows)),
        "reading_order_risk": dict(Counter(row["reading_order_risk"] for row in rows)),
        "layout_strata": dict(Counter(value for row in rows for value in row["layout_strata"])),
        "splits": dict(Counter(row["split"] for row in rows)),
    }


def _pilot_first_order(
    selections: list[dict[str, Any]],
    pilot_pages: int,
) -> list[dict[str, Any]]:
    desired = {
        "ROUTE_NATIVE_FIRST",
        "ROUTE_VISUAL_REQUIRED",
        "ROUTE_HYBRID_REQUIRED",
        "RISK_LOW",
        "RISK_MEDIUM",
        "RISK_HIGH",
        "MULTI_COLUMN",
        "SINGLE_COLUMN",
        "DENSE_TEXT",
        "IMAGE_BEARING",
        "SCAN_LIKE",
        "SPARSE_OR_OPENER",
        "VECTOR_DENSE",
        "VISUAL_ASSET_PROXY",
    }

    def tags(row: dict[str, Any]) -> set[str]:
        return {
            f"ROUTE_{row['routing_decision']}",
            f"RISK_{row['reading_order_risk']}",
            *row["layout_strata"],
        }

    remaining = list(selections)
    pilot = []
    covered: set[str] = set()
    documents: Counter[str] = Counter()
    while remaining and len(pilot) < pilot_pages:
        chosen = min(
            remaining,
            key=lambda row: (
                -len((tags(row) & desired) - covered),
                documents[row["document_id"]],
                _stable_digest("pilot-v0", row["document_id"], row["page_index"]),
            ),
        )
        pilot.append(chosen)
        remaining.remove(chosen)
        covered.update(tags(chosen))
        documents[chosen["document_id"]] += 1
    return pilot + sorted(
        remaining,
        key=lambda row: (str(row["document_id"]), int(row["page_index"])),
    )


def _group_aware_diagnostics(
    references: list[dict[str, Any]], predictions: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for reference in references:
        gt = [row for row in reference["reference_regions"] if row["semantic_type"] in CRITICAL_TYPES and not row["reference_uncertain"]]
        pred = [row for row in predictions[_page_key(reference)]["canonical_regions"] if row["semantic_type"] in CRITICAL_TYPES]
        for gt_row in gt:
            overlaps = [row for row in pred if row["semantic_type"] == gt_row["semantic_type"] and _coverage(gt_row["bbox_pdf_pt"], row["bbox_pdf_pt"]) >= 0.15]
            if len(overlaps) >= 2:
                rows.append({"document_id": reference["document_id"], "page_index": reference["page_index"], "split": reference["split"], "diagnostic": "REFERENCE_COVERED_BY_MULTIPLE_PREDICTIONS", "reference_region_id": gt_row["reference_region_id"], "semantic_type": gt_row["semantic_type"], "prediction_region_ids": [row["region_id"] for row in overlaps]})
        for pred_row in pred:
            overlaps = [row for row in gt if row["semantic_type"] == pred_row["semantic_type"] and _coverage(row["bbox_pdf_pt"], pred_row["bbox_pdf_pt"]) >= 0.15]
            if len(overlaps) >= 2:
                rows.append({"document_id": reference["document_id"], "page_index": reference["page_index"], "split": reference["split"], "diagnostic": "MULTIPLE_REFERENCES_MERGED", "region_id": pred_row["region_id"], "semantic_type": pred_row["semantic_type"], "reference_region_ids": [row["reference_region_id"] for row in overlaps]})
    return sorted(rows, key=lambda row: (row["document_id"], row["page_index"], row["diagnostic"]))


def _reading_order_metrics_v15(
    references: list[dict[str, Any]], predictions: dict[tuple[str, int], dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    methods = {
        "naive_yx": "naive_reading_order_index",
        "region_reading_order_v0": "reading_order_v0_index",
        "region_reading_order_v1": "reading_order_v1_index",
    }
    aggregate: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {method: Counter() for method in methods})
    errors = []
    for reference in references:
        gt = [row for row in reference["reference_regions"] if row["semantic_type"] in CRITICAL_TYPES and not row["reference_uncertain"]]
        pred = [row for row in predictions[_page_key(reference)]["canonical_regions"] if row["semantic_type"] in CRITICAL_TYPES]
        matches, unmatched_gt, _ = _match_regions(gt, pred, 0.3)
        by_gt = {gt_index: pred[pred_index] for gt_index, pred_index, _ in matches}
        uncertain_pairs = {
            frozenset(pair) for pair in reference.get("order_uncertain_pairs", [])
        }

        strata = {"OVERALL", reference["split"], reference["reading_order_risk"], *reference.get("layout_strata", [])}
        for method, field in methods.items():
            correct = 0
            reference_total = sum(
                not _order_pair_uncertain(gt, uncertain_pairs, first, second)
                for first in range(len(gt))
                for second in range(first + 1, len(gt))
            )
            matched_indices = sorted(by_gt)
            matched_total = 0
            for position, first in enumerate(matched_indices):
                for second in matched_indices[position + 1 :]:
                    if _order_pair_uncertain(gt, uncertain_pairs, first, second):
                        continue
                    matched_total += 1
                    if int(by_gt[first][field]) < int(by_gt[second][field]):
                        correct += 1
            perfect = correct == reference_total and not unmatched_gt
            for stratum in strata:
                aggregate[stratum][method].update(
                    correct=correct,
                    matched_total=matched_total,
                    reference_total=reference_total,
                    matched_regions=len(by_gt),
                    reference_regions=len(gt),
                    perfect=int(perfect),
                    pages=1,
                )
            if not perfect:
                errors.append({"document_id": reference["document_id"], "page_index": reference["page_index"], "page_number": reference["page_number"], "split": reference["split"], "method": method, "pairwise_correct": correct, "pairwise_matched_total": matched_total, "pairwise_reference_total": reference_total, "unmatched_reference_regions": len(unmatched_gt)})

    def finish(counter: Counter[str]) -> dict[str, Any]:
        return {
            "pairwise_precedence_accuracy": round(
                counter["correct"] / counter["matched_total"], 6
            )
            if counter["matched_total"]
            else 0.0,
            "pairwise_correct": counter["correct"],
            "pairwise_matched_total": counter["matched_total"],
            "pairwise_reference_total": counter["reference_total"],
            "matched_reference_coverage": round(
                counter["matched_regions"] / counter["reference_regions"], 6
            )
            if counter["reference_regions"]
            else 0.0,
            "perfect_pages": counter["perfect"],
            "page_count": counter["pages"],
            "perfect_page_rate": round(
                counter["perfect"] / counter["pages"], 6
            )
            if counter["pages"]
            else 0.0,
        }

    return ({"schema": "bemarkdown-phase6b15-reading-order-metrics-v0", "strata": {stratum: {method: finish(counter) for method, counter in values.items()} for stratum, values in sorted(aggregate.items())}}, errors)


def _order_pair_uncertain(
    reference_regions: list[dict[str, Any]],
    uncertain_pairs: set[frozenset[str]],
    first: int,
    second: int,
) -> bool:
    first_id = reference_regions[first].get("reference_region_id")
    second_id = reference_regions[second].get("reference_region_id")
    return bool(
        first_id
        and second_id
        and frozenset((first_id, second_id)) in uncertain_pairs
    )


def _generate_source_only_html(work_dir: Path, references: list[dict[str, Any]]) -> Path:
    cards = []
    for reference in references:
        width = reference["page_geometry"]["width_pt"]
        height = reference["page_geometry"]["height_pt"]
        boxes = "".join(_svg_box(row, width, height, "reference") for row in reference["reference_regions"])
        cards.append(f'<article><h2>#{reference["reference_index"]:03d} {html.escape(reference["document_id"])} p{reference["page_number"]} [{reference["split"]}]</h2><p>prediction-hidden=true; native geometry auxiliary={str(reference["native_geometry_used"]).lower()}</p><svg viewBox="0 0 {width} {height}"><image href="{html.escape(reference["preview_path"])}" width="{width}" height="{height}" preserveAspectRatio="none"/>{boxes}</svg></article>')
    document = _html_shell("Phase 6B-1.5 source-only annotation", "Prediction boxes, labels, scores and RegionIR are absent from this artifact.", cards)
    path = work_dir / "annotation_source_only.html"
    path.write_text(document, encoding="utf-8")
    return path


def _generate_evaluation_html(
    work_dir: Path,
    references: list[dict[str, Any]],
    predictions: dict[tuple[str, int], dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> Path:
    diagnostic_counts = Counter((row["document_id"], row["page_index"]) for row in diagnostics)
    cards = []
    for reference in references:
        width = reference["page_geometry"]["width_pt"]
        height = reference["page_geometry"]["height_pt"]
        ref_boxes = "".join(_svg_box(row, width, height, "reference") for row in reference["reference_regions"])
        pred_boxes = "".join(_svg_box(row, width, height, "prediction") for row in predictions[_page_key(reference)]["canonical_regions"] if row["semantic_type"] in CRITICAL_TYPES)
        cards.append(f'<article><h2>#{reference["reference_index"]:03d} {html.escape(reference["document_id"])} p{reference["page_number"]} [{reference["split"]}]</h2><p>group-aware diagnostics={diagnostic_counts[_page_key(reference)]}</p><svg viewBox="0 0 {width} {height}"><image href="{html.escape(reference["preview_path"])}" width="{width}" height="{height}" preserveAspectRatio="none"/><g class="reference-layer">{ref_boxes}</g><g class="prediction-layer">{pred_boxes}</g></svg></article>')
    document = _html_shell("Phase 6B-1.5 evaluation review", "Green=reference; red dashed=prediction. Reference was frozen before this artifact was generated.", cards)
    path = work_dir / "evaluation_review.html"
    path.write_text(document, encoding="utf-8")
    return path


def _html_shell(title: str, note: str, cards: list[str]) -> str:
    return f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>body{{font:14px system-ui;margin:20px;background:#f2f4f7}}article{{background:white;border:1px solid #bbb;padding:10px;margin:16px 0}}svg{{display:block;width:min(100%,900px);height:auto;border:1px solid #ddd}}.box{{fill:none;stroke-width:1.6;vector-effect:non-scaling-stroke}}.reference{{stroke:#00a651}}.prediction{{stroke:#d62728;stroke-dasharray:5 3}}.label{{font:8px sans-serif;paint-order:stroke;stroke:white;stroke-width:2px}}</style></head><body><h1>{html.escape(title)}</h1><p>{html.escape(note)}</p>{"".join(cards)}</body></html>'


def _svg_box(row: dict[str, Any], width: float, height: float, layer: str) -> str:
    x0, y0, x1, y1 = row["bbox_pdf_pt"]
    label = row.get("semantic_type", "")
    index = row.get("expected_order_index", row.get("reading_order_v1_index", ""))
    return f'<rect class="box {layer}" x="{x0}" y="{y0}" width="{max(0, x1-x0)}" height="{max(0, y1-y0)}"/><text class="label" x="{x0+2}" y="{min(height-2,y0+9)}">{html.escape(str(label))}:{index}</text>'


def _make_preview(source: Path, target: Path) -> None:
    from PIL import Image

    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((1000, 1400), Image.Resampling.LANCZOS)
        image.save(target, format="JPEG", quality=78, optimize=True)


def _region_identity_digest(pages: list[dict[str, Any]]) -> str:
    payload = []
    for page in pages:
        for row in page["canonical_regions"]:
            payload.append((page["document_id"], page["page_index"], row["region_id"], row["semantic_type"], row["raw_model_label"], row["score"], row["bbox_pdf_pt"], row["bbox_render_px"], row["bbox_normalized"]))
    payload.sort(key=lambda row: (str(row[0]), int(row[1]), str(row[2])))
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _active_runs(values: list[bool], *, merge_gap: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, active in enumerate(values + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if runs and start - runs[-1][1] <= merge_gap:
                runs[-1] = (runs[-1][0], index)
            else:
                runs.append((start, index))
            start = None
    return runs


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _union_boxes(boxes: list[Sequence[float]]) -> list[float]:
    return [min(row[0] for row in boxes), min(row[1] for row in boxes), max(row[2] for row in boxes), max(row[3] for row in boxes)]


def _horizontal_overlap_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    return overlap / max(1.0, min(_width(first), _width(second)))


def _coverage(inner: Sequence[float], outer: Sequence[float]) -> float:
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return intersection / max(1.0, _area(inner))


def _area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _width(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def _height(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _center_x(bbox: Sequence[float]) -> float:
    return (float(bbox[0]) + float(bbox[2])) / 2


def _center_y(bbox: Sequence[float]) -> float:
    return (float(bbox[1]) + float(bbox[3])) / 2


def _yx_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (float(row["bbox_pdf_pt"][1]), float(row["bbox_pdf_pt"][0]), str(row.get("reference_region_id", row.get("semantic_type", ""))))


def _page_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["document_id"]), int(row["page_index"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _source_value(row: dict[str, Any], field: str) -> str:
    if field == "risk":
        return str(row.get("reading_order", {}).get("risk", row.get("reading_order_risk", "UNKNOWN")))
    return str(row.get(field, "UNKNOWN"))


def _layout_strata(row: dict[str, Any]) -> list[str]:
    strata = []
    reading = row.get("reading_order", {})
    native_text = row.get("native_text", {})
    images = row.get("images", {})
    vectors = row.get("vectors", {})
    if reading.get("column_like_layout"):
        strata.append("MULTI_COLUMN")
    else:
        strata.append("SINGLE_COLUMN")
    if int(native_text.get("line_count", 0)) >= 25:
        strata.append("DENSE_TEXT")
    if int(images.get("placed_image_count", 0)) >= 1:
        strata.append("IMAGE_BEARING")
    if (
        int(images.get("placed_image_count", 0)) >= 2
        and float(images.get("largest_image_coverage_ratio", 0.0)) < 0.8
    ):
        strata.append("VISUAL_ASSET_PROXY")
    if float(images.get("largest_image_coverage_ratio", 0.0)) >= 0.8:
        strata.append("SCAN_LIKE")
    if int(vectors.get("drawing_count", 0)) >= 8:
        strata.append("VECTOR_DENSE")
    if int(native_text.get("block_count", 0)) <= 3:
        strata.append("SPARSE_OR_OPENER")
    return sorted(set(strata))


def _stable_digest(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
