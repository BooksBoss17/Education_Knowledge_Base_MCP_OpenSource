from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

FUSION_SCHEMA = "bemarkdown-page-region-fusion-ir-v0.4"
LEGACY_FUSION_SCHEMA = "bemarkdown-page-region-fusion-ir-v0.3"
FRESH_SELECTION_BASIS = "PHASE6A_SOURCE_ONLY"
FALLBACK_KINDS = frozenset(
    {
        "MODEL_LOW_SCORE_FALLBACK",
        "NATIVE_TEXT_FALLBACK",
        "NATIVE_IMAGE_FALLBACK",
        "VECTOR_VISUAL_FALLBACK",
    }
)
PAGE_ESCALATION_STATUSES = frozenset(
    {"NONE", "VISUAL_PAGE_REQUIRED", "VISUAL_PAGE_REVIEW"}
)
DEFAULT_FUSION_STRATEGY: dict[str, Any] = {
    "strategy_id": "region-fusion-v0",
    "canonical_claim_coverage_threshold": 0.9,
    "partial_text_horizontal_overlap_threshold": 0.1,
    "low_score_minimum": 0.3,
    "low_score_maximum": 0.5,
    "low_score_source_alignment_iou": 0.25,
    "fallback_dedup_iou": 0.85,
    "image_group_iou": 0.7,
    "image_group_containment": 0.85,
    "max_fallback_candidates_per_page": 64,
    "reading_order": "NAIVE_Y_X_V0",
    "enable_native_text": True,
    "enable_native_image": True,
    "enable_low_score": True,
    "enable_vector": True,
    "enable_page_visual": True,
}
FINE_GRAINED_FUSION_STRATEGY: dict[str, Any] = {
    **DEFAULT_FUSION_STRATEGY,
    "strategy_id": "fine-grained-residual-v0.1",
    "native_text_unit_mode": "LINE_RESIDUAL",
    "canonical_claim_coverage_threshold": 0.9,
    "formula_claim_coverage_threshold": 0.95,
    "partial_claim_minimum_ratio": 0.05,
    "formula_residual_subtract_threshold": 0.8,
    "residual_min_width_pt": 2.0,
    "residual_min_area_pt2": 4.0,
    "residual_min_estimated_chars": 2,
    "residual_max_vertical_gap_pt": 12.0,
    "residual_vertical_gap_height_factor": 1.5,
    "residual_horizontal_alignment_tolerance_pt": 12.0,
    "residual_min_horizontal_overlap_ratio": 0.3,
    "residual_candidate_horizontal_padding_pt": 0.0,
    "residual_candidate_vertical_padding_pt": 0.0,
    "residual_use_source_line_band": False,
    "residual_expand_unclaimed_to_block_width": False,
    "residual_preserve_partial_line_envelope": False,
    "max_fallback_candidates_per_page": 96,
}
PAGE_ESCALATION_FUSION_STRATEGY: dict[str, Any] = {
    **FINE_GRAINED_FUSION_STRATEGY,
    "strategy_id": "page-escalation-v0",
    "enable_page_visual": False,
    "page_escalation_decision_version": "page-escalation-v0",
    "full_page_raster_threshold": 0.85,
    "major_visual_load_threshold": 0.8,
    "localized_visual_coverage_threshold": 0.5,
}
LOCAL_OWNERSHIP_FUSION_STRATEGY: dict[str, Any] = {
    **PAGE_ESCALATION_FUSION_STRATEGY,
    "strategy_id": "local-source-unit-ownership-v0",
    "native_text_unit_mode": "SOURCE_UNIT_OWNERSHIP",
    "source_unit_ownership_version": "source-unit-ownership-v0",
    "local_grouping_version": "local-grouping-v0",
    "ownership_claim_area_threshold": 0.97,
    "ownership_claim_horizontal_threshold": 0.97,
    "ownership_claim_vertical_threshold": 0.75,
    "ownership_formula_area_threshold": 0.99,
    "ownership_formula_horizontal_threshold": 0.99,
    "ownership_partial_minimum_ratio": 0.05,
    "ownership_group_max_vertical_gap_pt": 12.0,
    "ownership_group_vertical_gap_height_factor": 1.5,
    "ownership_group_horizontal_alignment_tolerance_pt": 12.0,
    "ownership_group_min_horizontal_overlap_ratio": 0.3,
    "ownership_group_same_baseline_max_gap_pt": 36.0,
    "ownership_group_same_baseline_gap_height_factor": 4.0,
    "max_fallback_candidates_per_page": 512,
}
ADAPTIVE_LOCAL_GROUPING_STRATEGY: dict[str, Any] = {
    **LOCAL_OWNERSHIP_FUSION_STRATEGY,
    "strategy_id": "adaptive-local-grouping-v1",
    "local_grouping_version": "adaptive-local-grouping-v1",
    "adaptive_grouping_profile_version": "adaptive-local-profile-v1",
    "adaptive_dense_line_count": 36,
    "adaptive_dense_gap_height_ratio": 0.6,
    "adaptive_fragmented_unit_ratio": 0.3,
    "adaptive_formula_region_ratio": 0.12,
    "adaptive_formula_area_ratio": 0.03,
    "adaptive_same_block_gap_height_factor": 2.0,
    "adaptive_cross_block_gap_height_factor": 1.6,
    "adaptive_same_block_spacing_factor": 1.25,
    "adaptive_cross_block_spacing_factor": 1.0,
    "adaptive_max_vertical_gap_pt": 20.0,
    "adaptive_max_group_height_lines": 32.0,
    "adaptive_max_group_page_height_ratio": 0.5,
    "adaptive_min_group_height_cap_pt": 144.0,
    "adaptive_inline_formula_max_height_factor": 3.0,
    "adaptive_inline_formula_max_page_area_ratio": 0.03,
    "adaptive_column_margin_ratio": 0.02,
}


def build_page_fusion_ir(
    *,
    page_ir: dict[str, Any],
    raw_page: dict[str, Any],
    page_record: dict[str, Any],
    source_evidence: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    """Build a conservative, deterministic union of model and source evidence."""
    _validate_page_identity(page_ir, raw_page, page_record)
    rules = {**DEFAULT_FUSION_STRATEGY, **copy.deepcopy(strategy)}
    width = float(page_ir["page_geometry"]["width_pt"])
    height = float(page_ir["page_geometry"]["height_pt"])
    canonical_regions = copy.deepcopy(page_ir.get("canonical_regions", []))
    candidates = [
        _canonical_candidate(row, width=width, height=height)
        for row in canonical_regions
    ]
    canonical_boxes = [row["bbox_pdf_pt"] for row in candidates]

    text_candidates, text_diagnostics = (
        _native_text_candidates(
            source_evidence.get("native_text", []),
            canonical_regions,
            source_evidence.get("native_images", []),
            width=width,
            height=height,
            rules=rules,
        )
        if rules["enable_native_text"]
        else ([], _empty_text_diagnostics())
    )
    image_candidates = (
        _native_image_candidates(
            source_evidence.get("native_images", []),
            canonical_boxes,
            width=width,
            height=height,
            rules=rules,
        )
        if rules["enable_native_image"]
        else []
    )
    vector_candidates = (
        _native_vector_candidates(
            source_evidence.get("native_vectors", []),
            canonical_boxes,
            width=width,
            height=height,
            rules=rules,
        )
        if rules["enable_vector"]
        else []
    )
    source_rows = [
        *source_evidence.get("native_text", []),
        *source_evidence.get("native_images", []),
        *source_evidence.get("native_vectors", []),
    ]
    low_score_candidates = (
        _low_score_candidates(
            raw_page,
            page_ir,
            source_rows,
            canonical_boxes,
            width=width,
            height=height,
            rules=rules,
        )
        if rules["enable_low_score"]
        else []
    )

    fallback = _deduplicate_fallbacks(
        [
            *low_score_candidates,
            *text_candidates,
            *image_candidates,
            *vector_candidates,
        ],
        float(rules["fallback_dedup_iou"]),
    )
    fallback = sorted(fallback, key=_candidate_sort_key)
    limit = int(rules["max_fallback_candidates_per_page"])
    guard_triggered = len(fallback) > limit
    if guard_triggered:
        fallback = _apply_fallback_limit(fallback, limit)
    candidates.extend(fallback)
    evidence_sources = {
        "MODEL_CANONICAL_REGION": ["PP_DOCLAYOUT_CANONICAL"],
        "MODEL_LOW_SCORE_FALLBACK": ["PP_DOCLAYOUT_RAW", "NATIVE_PDF_GEOMETRY"],
        "NATIVE_TEXT_FALLBACK": ["NATIVE_PDF_TEXT"],
        "NATIVE_IMAGE_FALLBACK": ["NATIVE_PDF_IMAGE_PLACEMENT"],
        "VECTOR_VISUAL_FALLBACK": ["NATIVE_PDF_VECTOR_GEOMETRY"],
    }
    for candidate in candidates:
        canonical = candidate.get("canonical_region", {})
        candidate.update(
            {
                "source_region_id": canonical.get("region_id"),
                "evidence_sources": evidence_sources[candidate["candidate_kind"]],
                "coverage_status": "PRIMARY"
                if candidate["candidate_kind"] == "MODEL_CANONICAL_REGION"
                else "SUPPLEMENTAL",
                "page_route": page_ir.get("page_route"),
                "reading_order_method": "naive_yx",
                "provenance": {
                    "strategy_id": str(rules["strategy_id"]),
                    "source_evidence_mutated": False,
                },
            }
        )
        if canonical:
            candidate.update(
                {
                    "semantic_type": canonical.get("semantic_type"),
                    "region_id": canonical.get("region_id"),
                    "score": canonical.get("score"),
                    "raw_provenance": copy.deepcopy(canonical.get("provenance", {})),
                }
            )
    candidates.sort(key=_candidate_sort_key)
    for index, candidate in enumerate(candidates):
        candidate["reading_order_index"] = index

    page_escalation = decide_page_escalation(
        page_record=page_record,
        source_evidence=source_evidence,
        local_candidates=candidates,
        width=width,
        height=height,
        strategy=rules,
    )

    source_unit_ownership = text_diagnostics.pop("source_unit_ownership", None)
    source_unit_ownership_version = text_diagnostics.pop(
        "source_unit_ownership_version", None
    )
    output = {
        "schema": (
            FUSION_SCHEMA
            if str(rules.get("local_grouping_version", ""))
            == "adaptive-local-grouping-v1"
            else LEGACY_FUSION_SCHEMA
        ),
        "strategy_id": str(rules["strategy_id"]),
        "document_id": str(page_ir["document_id"]),
        "page_index": int(page_ir["page_index"]),
        "page_number": int(page_ir.get("page_number", int(page_ir["page_index"]) + 1)),
        "page_route": page_ir.get("page_route"),
        "page_geometry": copy.deepcopy(page_ir["page_geometry"]),
        "render_transform": copy.deepcopy(page_ir.get("render_transform", {})),
        "fusion_candidates": candidates,
        "page_escalation": page_escalation,
        "diagnostics": {
            "canonical_candidate_count": len(canonical_regions),
            "fallback_candidate_count": len(fallback),
            "candidate_count": len(candidates),
            "candidate_explosion_guard_triggered": guard_triggered,
            "temporary_reading_order": str(rules["reading_order"]),
            **text_diagnostics,
            "evidence_coverage": _evidence_coverage_diagnostics(
                source_evidence,
                canonical_boxes,
                [row["bbox_pdf_pt"] for row in candidates],
            ),
        },
    }
    if source_unit_ownership is not None:
        output["source_unit_ownership_version"] = source_unit_ownership_version
        output["source_unit_ownership"] = source_unit_ownership
    return output


def validate_page_fusion_ir(page: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(page.get("fusion_candidates", [])):
        candidate_id = str(candidate.get("candidate_id", ""))
        if candidate_id in seen:
            issues.append(
                {
                    "issue_code": "DUPLICATE_CANDIDATE_ID",
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                }
            )
        seen.add(candidate_id)
        if not _valid_bbox(candidate.get("bbox_pdf_pt")):
            issues.append(
                {
                    "issue_code": "INVALID_CANDIDATE_BBOX",
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                }
            )
        if candidate.get("candidate_kind") == "PAGE_VISUAL_FALLBACK":
            issues.append(
                {
                    "issue_code": "DEPRECATED_PAGE_VISUAL_REGION_CANDIDATE",
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                }
            )
    escalation = page.get("page_escalation", {})
    if escalation.get("status") not in PAGE_ESCALATION_STATUSES:
        issues.append({"issue_code": "INVALID_PAGE_ESCALATION_STATUS"})
    if not escalation.get("reason_codes"):
        issues.append({"issue_code": "MISSING_PAGE_ESCALATION_REASON"})
    return issues


def discover_source_table_candidates(
    *,
    document_id: str,
    page_index: int,
    page_width: float,
    page_height: float,
    text_lines: Sequence[dict[str, Any]],
    vector_lines: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    """Discover conservative source-only logical-table candidates.

    This is a benchmark census helper, not a production table detector. It requires
    a rectangular vector lattice plus text distributed across multiple cells so
    coordinate grids, diagrams, and decorative rules are not promoted by geometry
    alone.
    """
    vertical = []
    horizontal = []
    for value in vector_lines:
        if len(value) != 4:
            continue
        x0, y0, x1, y1 = (float(item) for item in value)
        if abs(x1 - x0) <= 1.5 and abs(y1 - y0) >= 8.0:
            vertical.append([x0, min(y0, y1), x1, max(y0, y1)])
        elif abs(y1 - y0) <= 1.5 and abs(x1 - x0) >= 8.0:
            horizontal.append([min(x0, x1), y0, max(x0, x1), y1])
    if len(vertical) < 3 or len(horizontal) < 3:
        return []

    x_values = _cluster_axis_values([row[0] for row in vertical], tolerance=2.0)
    y_values = _cluster_axis_values([row[1] for row in horizontal], tolerance=2.0)
    if len(x_values) < 3 or len(y_values) < 3:
        return []
    bbox = [min(x_values), min(y_values), max(x_values), max(y_values)]
    if not _valid_bbox(bbox):
        return []
    intersections = 0
    for x in x_values:
        for y in y_values:
            vertical_hit = any(
                abs(row[0] - x) <= 2.0 and row[1] - 2.0 <= y <= row[3] + 2.0
                for row in vertical
            )
            horizontal_hit = any(
                abs(row[1] - y) <= 2.0 and row[0] - 2.0 <= x <= row[2] + 2.0
                for row in horizontal
            )
            intersections += int(vertical_hit and horizontal_hit)
    minimum_intersections = max(6, math.ceil(len(x_values) * len(y_values) * 0.6))
    if intersections < minimum_intersections:
        return []

    in_grid_text = [
        row
        for row in text_lines
        if _valid_bbox(row.get("bbox_pdf_pt"))
        and _bbox_intersection(bbox, row["bbox_pdf_pt"]) is not None
    ]
    occupied_rows = {
        index
        for row in in_grid_text
        for index in [_axis_bin(_bbox_center(row["bbox_pdf_pt"])[1], y_values)]
        if index is not None
    }
    occupied_columns = {
        index
        for row in in_grid_text
        for index in [_axis_bin(_bbox_center(row["bbox_pdf_pt"])[0], x_values)]
        if index is not None
    }
    if len(in_grid_text) < 4 or len(occupied_rows) < 2 or len(occupied_columns) < 2:
        return []

    normalized = _normalized_bbox(bbox, float(page_width), float(page_height))
    identity = {
        "document_id": str(document_id),
        "page_index": int(page_index),
        "bbox_pdf_pt": normalized,
        "grid_intersection_count": intersections,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:16]
    return [
        {
            "candidate_id": f"table-source-{digest}",
            "document_id": str(document_id),
            "page_index": int(page_index),
            "bbox_pdf_pt": normalized,
            "bbox_normalized": [
                round(normalized[0] / float(page_width), 8),
                round(normalized[1] / float(page_height), 8),
                round(normalized[2] / float(page_width), 8),
                round(normalized[3] / float(page_height), 8),
            ],
            "source_only_signals": {
                "vertical_rule_count": len(x_values),
                "horizontal_rule_count": len(y_values),
                "grid_intersection_count": intersections,
                "text_line_count": len(in_grid_text),
                "occupied_row_band_count": len(occupied_rows),
                "occupied_column_band_count": len(occupied_columns),
            },
            "reason_codes": ["RECTANGULAR_GRID_WITH_CELL_TEXT"],
            "prediction_artifacts_loaded": False,
        }
    ]


def _cluster_axis_values(values: Sequence[float], *, tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or value - statistics.mean(groups[-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [round(statistics.mean(group), 6) for group in groups]


def _bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2,
        (float(bbox[1]) + float(bbox[3])) / 2,
    )


def _axis_bin(value: float, boundaries: Sequence[float]) -> int | None:
    for index, (start, end) in enumerate(pairwise(boundaries)):
        if start <= value <= end:
            return index
    return None


def select_fresh_general_pages(
    page_records: list[dict[str, Any]],
    *,
    excluded_page_keys: set[tuple[str, int]],
    target_pages: int,
) -> list[dict[str, Any]]:
    allowed = _prediction_free_rows(page_records, excluded_page_keys)
    if target_pages <= 0 or target_pages > len(allowed):
        raise ValueError("target_pages must fit available prediction-free pages")
    fields = (
        "source_group",
        "routing_decision",
        "source_profile",
        "reading_order_risk",
    )
    counts = {
        field: Counter(_source_value(row, field) for row in allowed) for field in fields
    }

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        rarity = sum(
            1 / max(1, counts[field][_source_value(row, field)]) for field in fields
        )
        return (
            -rarity,
            _stable_digest("fresh-general-v1", row["document_id"], row["page_index"]),
            str(row["document_id"]),
            int(row["page_index"]),
        )

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in allowed:
        by_document[str(row["document_id"])].append(row)
    representatives = [min(rows, key=rank) for rows in by_document.values()]
    selected = sorted(representatives, key=rank)[:target_pages]
    selected_keys = {_page_key(row) for row in selected}
    if len(selected) < target_pages:
        for row in sorted(allowed, key=rank):
            if _page_key(row) in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(_page_key(row))
            if len(selected) == target_pages:
                break
    return [
        _selection_row(row, "FRESH_GENERAL_HOLDOUT") for row in _page_order(selected)
    ]


def select_fresh_support_pages(
    page_records: list[dict[str, Any]],
    *,
    excluded_page_keys: set[tuple[str, int]],
    target_pages: int,
) -> list[dict[str, Any]]:
    allowed = _prediction_free_rows(page_records, excluded_page_keys)
    scored: list[tuple[tuple[Any, ...], dict[str, Any], list[str]]] = []
    for row in allowed:
        signals = _support_signals(row)
        if not signals:
            continue
        score = sum(_support_signal_weight(signal) for signal in signals)
        rank = (
            -score,
            _stable_digest("fresh-support-v1", row["document_id"], row["page_index"]),
            str(row["document_id"]),
            int(row["page_index"]),
        )
        scored.append((rank, row, signals))
    if target_pages <= 0 or target_pages > len(scored):
        raise ValueError("target_pages must fit source-feature-targeted support pages")
    result = []
    for _, row, signals in sorted(scored)[:target_pages]:
        selected = _selection_row(row, "FRESH_SUPPORT_DIAGNOSTIC")
        selected["support_selection_signals"] = signals
        result.append(selected)
    return _page_order(result)


def freeze_fresh_reference(
    *,
    work_dir: str | Path,
    contract_path: str | Path,
    general_selection_path: str | Path,
    support_selection_path: str | Path,
    reference_path: str | Path,
    source_render_hashes: list[dict[str, Any]],
    frozen_at: str,
) -> dict[str, Any]:
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "annotation_contract": Path(contract_path).resolve(),
        "general_selection": Path(general_selection_path).resolve(),
        "support_selection": Path(support_selection_path).resolve(),
        "reference_annotation": Path(reference_path).resolve(),
    }
    target = work_dir / "fresh_reference_freeze_manifest.json"
    snapshot = work_dir / "fresh_annotation_contract_snapshot.md"
    expected = {f"{name}_sha256": _sha256_file(path) for name, path in paths.items()}
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Fresh reference inputs changed after freeze")
        return existing
    snapshot.write_text(
        paths["annotation_contract"].read_text(encoding="utf-8"), encoding="utf-8"
    )
    manifest = {
        "schema": "bemarkdown-fresh-reference-freeze-v0",
        **expected,
        "source_page_renders": copy.deepcopy(source_render_hashes),
        "reference_page_count": len(source_render_hashes),
        "freeze_timestamp": frozen_at,
        "prediction_exposed_after_freeze": False,
        "prediction_exposure_timestamp": None,
        "fresh_holdout_evaluation_count": 0,
    }
    _write_json_atomic(target, manifest)
    return manifest


def freeze_fusion_strategy(
    *, work_dir: str | Path, strategy: dict[str, Any], frozen_at: str
) -> dict[str, Any]:
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = work_dir / "frozen_fusion_strategy.json"
    manifest_path = work_dir / "fusion_strategy_freeze_manifest.json"
    canonical_strategy = {**DEFAULT_FUSION_STRATEGY, **copy.deepcopy(strategy)}
    if strategy_path.is_file() or manifest_path.is_file():
        if not strategy_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Fusion strategy freeze is incomplete")
        existing_strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        if existing_strategy != canonical_strategy:
            raise RuntimeError("Fusion strategy changed after freeze")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    _write_json_atomic(strategy_path, canonical_strategy)
    manifest = {
        "schema": "bemarkdown-fusion-strategy-freeze-v0",
        "strategy_id": canonical_strategy["strategy_id"],
        "strategy_sha256": _sha256_file(strategy_path),
        "freeze_timestamp": frozen_at,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def mark_fresh_prediction_exposure(
    *,
    work_dir: str | Path,
    prediction_sha256: str,
    page_ir_sha256: str,
    exposed_at: str,
) -> dict[str, Any]:
    work_dir = Path(work_dir).resolve()
    reference_path = work_dir / "fresh_reference_freeze_manifest.json"
    strategy_path = work_dir / "fusion_strategy_freeze_manifest.json"
    if not reference_path.is_file() or not strategy_path.is_file():
        raise RuntimeError("Fresh reference and fusion strategy must be frozen first")
    manifest = json.loads(reference_path.read_text(encoding="utf-8"))
    if manifest.get("prediction_exposed_after_freeze"):
        raise RuntimeError("Fresh predictions were already exposed")
    manifest.update(
        {
            "prediction_exposed_after_freeze": True,
            "prediction_exposure_timestamp": exposed_at,
            "prediction_sha256": prediction_sha256,
            "page_ir_sha256": page_ir_sha256,
            "fresh_holdout_evaluation_count": 1,
        }
    )
    _write_json_atomic(reference_path, manifest)
    _write_json_atomic(work_dir / "fresh_prediction_exposure.json", manifest)
    return manifest


def content_preservation_metrics(
    references: Sequence[dict[str, Any]],
    canonical_pages: dict[tuple[str, int], dict[str, Any]],
    fusion_pages: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    canonical_coverages: list[float] = []
    fusion_coverages: list[float] = []
    rescued = 0
    reference_count = 0
    by_class: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"canonical_only": [], "fusion": []}
    )
    for reference in references:
        key = _page_key(reference)
        canonical_boxes = [
            row["bbox_pdf_pt"]
            for row in canonical_pages.get(key, {}).get("canonical_regions", [])
            if _valid_bbox(row.get("bbox_pdf_pt"))
        ]
        fusion_boxes = [
            row["bbox_pdf_pt"]
            for row in fusion_pages.get(key, {}).get("fusion_candidates", [])
            if _valid_bbox(row.get("bbox_pdf_pt"))
        ]
        for region in reference.get("reference_regions", []):
            if region.get("reference_uncertain") or not _valid_bbox(
                region.get("bbox_pdf_pt")
            ):
                continue
            reference_count += 1
            canonical_coverage = _union_coverage(region["bbox_pdf_pt"], canonical_boxes)
            fusion_coverage = _union_coverage(region["bbox_pdf_pt"], fusion_boxes)
            canonical_coverages.append(canonical_coverage)
            fusion_coverages.append(fusion_coverage)
            semantic_type = str(region.get("semantic_type", "UNKNOWN"))
            by_class[semantic_type]["canonical_only"].append(canonical_coverage)
            by_class[semantic_type]["fusion"].append(fusion_coverage)
            if canonical_coverage < 0.9 <= fusion_coverage:
                rescued += 1
    canonical_summary = _coverage_summary(canonical_coverages)
    fusion_summary = _coverage_summary(fusion_coverages)
    canonical_uncovered = sum(value < 0.9 for value in canonical_coverages)
    fusion_uncovered = sum(value < 0.9 for value in fusion_coverages)
    return {
        "reference_region_count": reference_count,
        "canonical_only": canonical_summary,
        "fusion": fusion_summary,
        "by_semantic_type": {
            semantic_type: {
                "canonical_only": _coverage_summary(values["canonical_only"]),
                "fusion": _coverage_summary(values["fusion"]),
            }
            for semantic_type, values in sorted(by_class.items())
        },
        "fallback_rescue": {
            "rescued_reference_regions": rescued,
            "rescue_rate": round(rescued / canonical_uncovered, 6)
            if canonical_uncovered
            else 0.0,
        },
        "critical_uncovered": {
            "canonical_only": canonical_uncovered,
            "fusion": fusion_uncovered,
            "absolute_reduction": canonical_uncovered - fusion_uncovered,
        },
    }


def source_unit_preservation_metrics(
    pages: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Verify that every content-bearing source unit has an output owner.

    Canonically owned units are accounted for only when at least one recorded
    owner is still present as a canonical fusion candidate. Units requiring
    fallback are accounted for only when their deterministic source-unit ID is
    carried by an emitted fallback candidate. This deliberately verifies
    the emitted IR instead of trusting the ownership decision alone.
    """
    ownership_counts: Counter[str] = Counter()
    expected_count = 0
    preserved_count = 0
    unaccounted_ids: list[str] = []
    duplicate_fallback_assignments = 0

    for page in pages:
        canonical_region_ids = {
            str(
                candidate.get("canonical_region", {}).get(
                    "region_id", candidate.get("evidence_ids", [""])[0]
                )
            )
            for candidate in page.get("fusion_candidates", [])
            if candidate.get("candidate_kind") == "MODEL_CANONICAL_REGION"
        }
        fallback_membership: Counter[str] = Counter(
            str(source_unit_id)
            for candidate in page.get("fusion_candidates", [])
            if candidate.get("candidate_kind") in FALLBACK_KINDS
            for source_unit_id in candidate.get("source_unit_ids", [])
        )
        duplicate_fallback_assignments += sum(
            count - 1 for count in fallback_membership.values() if count > 1
        )

        for unit in page.get("source_unit_ownership", []):
            status = str(unit.get("ownership_status", ""))
            ownership_counts[status] += 1
            if status == "IGNORED_NONCONTENT":
                continue
            expected_count += 1
            source_unit_id = str(unit.get("source_unit_id", ""))
            if status == "CANONICAL_OWNED":
                accounted = bool(
                    canonical_region_ids.intersection(
                        str(value) for value in unit.get("owner_region_ids", [])
                    )
                )
            else:
                accounted = fallback_membership[source_unit_id] > 0
            if accounted:
                preserved_count += 1
            else:
                unaccounted_ids.append(source_unit_id)

    return {
        "page_count": len(pages),
        "source_unit_total_count": sum(ownership_counts.values()),
        "source_unit_expected_count": expected_count,
        "source_unit_preserved_count": preserved_count,
        "source_unit_preserved_rate": round(preserved_count / expected_count, 6)
        if expected_count
        else 1.0,
        "source_unit_unaccounted_count": len(unaccounted_ids),
        "unaccounted_source_unit_ids": sorted(unaccounted_ids),
        "duplicate_fallback_source_unit_assignment_count": (
            duplicate_fallback_assignments
        ),
        "ownership_status_counts": {
            status: int(ownership_counts[status])
            for status in (
                "CANONICAL_OWNED",
                "SHARED_OR_UNCERTAIN",
                "FALLBACK_REQUIRED",
                "IGNORED_NONCONTENT",
            )
        },
    }


def summarize_fallback_burden(
    pages: Sequence[dict[str, Any]], *, reference_region_count: int
) -> dict[str, Any]:
    counts = [
        sum(
            row.get("candidate_kind") in FALLBACK_KINDS
            for row in page.get("fusion_candidates", [])
        )
        for page in pages
    ]
    area_ratios = []
    candidate_area_ratios = []
    candidate_width_ratios = []
    candidate_height_ratios = []
    duplicate_overlaps = 0
    duplicate_area = 0.0
    fallback_area = 0.0
    low_score_count = 0
    tiny_count = 0
    mergeable_fragment_pairs = 0
    consumability: Counter[str] = Counter()
    for page in pages:
        fallback = [
            row
            for row in page.get("fusion_candidates", [])
            if row.get("candidate_kind") in FALLBACK_KINDS
        ]
        geometry = page.get("page_geometry", {})
        page_box = [
            0.0,
            0.0,
            float(geometry.get("width_pt", 0.0)),
            float(geometry.get("height_pt", 0.0)),
        ]
        canonical_boxes = [
            row["bbox_pdf_pt"]
            for row in page.get("fusion_candidates", [])
            if row.get("candidate_kind") == "MODEL_CANONICAL_REGION"
        ]
        page_width = max(1e-9, page_box[2] - page_box[0])
        page_height = max(1e-9, page_box[3] - page_box[1])
        page_area = page_width * page_height
        area_ratios.append(
            _union_coverage(page_box, [row["bbox_pdf_pt"] for row in fallback])
            if _valid_bbox(page_box)
            else 0.0
        )
        low_score_count += sum(
            row.get("candidate_kind") == "MODEL_LOW_SCORE_FALLBACK" for row in fallback
        )
        for candidate in fallback:
            box = candidate["bbox_pdf_pt"]
            area = _bbox_area(box)
            area_ratio = area / page_area
            candidate_area_ratios.append(area_ratio)
            candidate_width_ratios.append((box[2] - box[0]) / page_width)
            candidate_height_ratios.append((box[3] - box[1]) / page_height)
            fallback_area += area
            duplicate_area += area * _union_coverage(box, canonical_boxes)
            char_count = int(
                candidate.get("subdivision_provenance", {}).get("native_char_count", 0)
            )
            is_tiny = area_ratio < 0.0005 or (char_count and char_count <= 1)
            tiny_count += int(is_tiny)
            if is_tiny:
                consumability["TOO_FRAGMENTED"] += 1
            elif area_ratio > 0.5:
                consumability["TOO_LARGE"] += 1
            elif candidate.get("semantic_hint") in {"VISUAL_UNKNOWN", "PAGE_VISUAL"}:
                consumability["AMBIGUOUS"] += 1
            else:
                consumability["CONSUMABLE"] += 1
        for index, first in enumerate(fallback):
            duplicate_overlaps += sum(
                _bbox_iou(first["bbox_pdf_pt"], second["bbox_pdf_pt"]) >= 0.85
                for second in fallback[index + 1 :]
            )
            mergeable_fragment_pairs += sum(
                _residual_units_can_group(
                    first["bbox_pdf_pt"],
                    second["bbox_pdf_pt"],
                    barriers=[],
                    rules=FINE_GRAINED_FUSION_STRATEGY,
                )
                for second in fallback[index + 1 :]
                if first.get("candidate_kind") == "NATIVE_TEXT_FALLBACK"
                and second.get("candidate_kind") == "NATIVE_TEXT_FALLBACK"
            )
    total_candidates = sum(counts)
    return {
        "page_count": len(pages),
        "fallback_candidate_count": sum(counts),
        "fallback_candidates_per_page": {
            "mean": round(statistics.fmean(counts), 6) if counts else 0.0,
            "median": float(statistics.median(counts)) if counts else 0.0,
            "p75": _percentile(counts, 0.75),
            "p90": _percentile(counts, 0.9),
            "p95": _percentile(counts, 0.95),
            "maximum": max(counts, default=0),
        },
        "fallback_area_ratio_per_page": {
            "mean": round(statistics.fmean(area_ratios), 6) if area_ratios else 0.0,
            "median": float(statistics.median(area_ratios)) if area_ratios else 0.0,
            "p75": _percentile_float(area_ratios, 0.75),
            "p90": _percentile_float(area_ratios, 0.9),
            "p95": _percentile_float(area_ratios, 0.95),
            "maximum": max(area_ratios, default=0.0),
        },
        "near_duplicate_fallback_pair_count": duplicate_overlaps,
        "duplicate_coverage_ratio": round(duplicate_area / fallback_area, 6)
        if fallback_area
        else 0.0,
        "candidate_size_distribution": {
            "area_ratio": _float_distribution(candidate_area_ratios),
            "width_ratio": _float_distribution(candidate_width_ratios),
            "height_ratio": _float_distribution(candidate_height_ratios),
        },
        "fragmentation": {
            "tiny_candidate_count": tiny_count,
            "mergeable_continuous_fragment_pair_count": mergeable_fragment_pairs,
        },
        "downstream_consumability_proxy": {
            key: int(consumability[key])
            for key in ("CONSUMABLE", "TOO_LARGE", "TOO_FRAGMENTED", "AMBIGUOUS")
        },
        "low_score_model_fallback_count": low_score_count,
        "page_escalation_excluded": True,
        "candidate_to_reference_ratio": round(
            total_candidates / reference_region_count, 6
        )
        if reference_region_count
        else None,
    }


def page_escalation_metrics(
    references: Sequence[dict[str, Any]],
    fusion_pages: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate FULL_PAGE_VISUAL_REQUIRED as the positive page-level class."""
    tp = fp = fn = tn = 0
    uncertain_reference_count = 0
    review_output_count = 0
    errors: list[dict[str, Any]] = []
    for reference in references:
        label = str(reference.get("page_escalation_reference", ""))
        if label == "PAGE_ESCALATION_UNCERTAIN":
            uncertain_reference_count += 1
            continue
        key = _page_key(reference)
        output = fusion_pages.get(key, {}).get("page_escalation", {})
        status = str(output.get("status", "VISUAL_PAGE_REVIEW"))
        review_output_count += int(status == "VISUAL_PAGE_REVIEW")
        actual_positive = label == "FULL_PAGE_VISUAL_REQUIRED"
        predicted_positive = status == "VISUAL_PAGE_REQUIRED"
        if actual_positive and predicted_positive:
            tp += 1
        elif actual_positive:
            fn += 1
            errors.append(
                {
                    "document_id": key[0],
                    "page_index": key[1],
                    "error_type": "MISSED_PAGE_ESCALATION",
                    "reference": label,
                    "prediction": status,
                }
            )
        elif predicted_positive:
            fp += 1
            errors.append(
                {
                    "document_id": key[0],
                    "page_index": key[1],
                    "error_type": "UNNECESSARY_PAGE_ESCALATION",
                    "reference": label,
                    "prediction": status,
                }
            )
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "positive_class": "FULL_PAGE_VISUAL_REQUIRED",
        "evaluated_page_count": tp + fp + fn + tn,
        "true_positive_count": tp,
        "true_negative_count": tn,
        "false_negative_count": fn,
        "false_positive_count": fp,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "uncertain_reference_count": uncertain_reference_count,
        "visual_page_review_output_count": review_output_count,
        "error_cases": errors,
    }


def hierarchical_content_preservation_metrics(
    references: Sequence[dict[str, Any]],
    canonical_pages: dict[tuple[str, int], dict[str, Any]],
    fusion_pages: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Combine page-level carrier preservation with local region coverage."""
    page_level_preserved = 0
    region_coverages: list[float] = []
    critical_unpreserved = 0
    uncertain_pages = 0
    page_results: list[dict[str, Any]] = []
    for reference in references:
        key = _page_key(reference)
        label = str(reference.get("page_escalation_reference", ""))
        fusion_page = fusion_pages.get(key, {})
        status = str(fusion_page.get("page_escalation", {}).get("status", ""))
        if label == "PAGE_ESCALATION_UNCERTAIN":
            uncertain_pages += 1
            continue
        if label == "FULL_PAGE_VISUAL_REQUIRED":
            preserved = status == "VISUAL_PAGE_REQUIRED"
            page_level_preserved += int(preserved)
            critical_unpreserved += int(not preserved)
            page_results.append(
                {
                    "document_id": key[0],
                    "page_index": key[1],
                    "mode": "PAGE",
                    "preserved": preserved,
                }
            )
            continue
        boxes = [
            row["bbox_pdf_pt"]
            for row in fusion_page.get("fusion_candidates", [])
            if _valid_bbox(row.get("bbox_pdf_pt"))
        ]
        page_coverages = []
        for region in reference.get("reference_regions", []):
            if region.get("reference_uncertain") or not _valid_bbox(
                region.get("bbox_pdf_pt")
            ):
                continue
            coverage = _union_coverage(region["bbox_pdf_pt"], boxes)
            region_coverages.append(coverage)
            page_coverages.append(coverage)
            critical_unpreserved += int(coverage < 0.9)
        page_results.append(
            {
                "document_id": key[0],
                "page_index": key[1],
                "mode": "REGION",
                "preserved": bool(page_coverages)
                and all(value >= 0.9 for value in page_coverages),
                "reference_region_count": len(page_coverages),
            }
        )
    return {
        "page_level_preserved_pages": page_level_preserved,
        "region_level_reference_region_count": len(region_coverages),
        "region_level_gte_0_90_count": sum(value >= 0.9 for value in region_coverages),
        "region_level_gte_0_90_rate": round(
            sum(value >= 0.9 for value in region_coverages) / len(region_coverages), 6
        )
        if region_coverages
        else 0.0,
        "overall_critical_unpreserved_count": critical_unpreserved,
        "uncertain_page_count": uncertain_pages,
        "page_results": page_results,
        "page_escalation_mixed_into_region_union": False,
    }


def _float_distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6) if values else 0.0,
        "median": round(float(statistics.median(values)), 6) if values else 0.0,
        "p75": round(_percentile_float(values, 0.75), 6),
        "p90": round(_percentile_float(values, 0.9), 6),
        "p95": round(_percentile_float(values, 0.95), 6),
        "maximum": round(max(values, default=0.0), 6),
    }


def _canonical_candidate(
    region: dict[str, Any], *, width: float, height: float
) -> dict[str, Any]:
    bbox = _normalized_bbox(region["bbox_pdf_pt"], width, height)
    return _make_candidate(
        kind="MODEL_CANONICAL_REGION",
        semantic_hint=str(region.get("semantic_type", "VISUAL_UNKNOWN")),
        bbox=bbox,
        width=width,
        height=height,
        evidence_ids=[str(region.get("region_id", region.get("raw_detection_id", "")))],
        reason_codes=["CANONICAL_MODEL_REGION_PRESERVED"],
        payload={"canonical_region": copy.deepcopy(region)},
    )


def _native_text_candidates(
    rows: Iterable[dict[str, Any]],
    canonical_regions: Sequence[dict[str, Any]],
    native_images: Sequence[dict[str, Any]],
    *,
    width: float,
    height: float,
    rules: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(rows)
    canonical_boxes = [row["bbox_pdf_pt"] for row in canonical_regions]
    unit_mode = str(rules.get("native_text_unit_mode", "BLOCK"))
    if unit_mode == "SOURCE_UNIT_OWNERSHIP":
        return _source_unit_ownership_candidates(
            rows,
            canonical_regions,
            native_images,
            width=width,
            height=height,
            rules=rules,
        )
    if unit_mode != "LINE_RESIDUAL":
        return (
            _legacy_native_text_candidates(
                rows,
                canonical_boxes,
                width=width,
                height=height,
                rules=rules,
            ),
            _empty_text_diagnostics(),
        )

    claimable = [
        row
        for row in canonical_regions
        if str(row.get("semantic_type", "")).upper()
        in {"TEXT", "TITLE", "FORMULA", "CAPTION"}
        and _valid_bbox(row.get("bbox_pdf_pt"))
    ]
    units: list[dict[str, Any]] = []
    claim_states: Counter[str] = Counter()
    tiny_count = 0
    for index, row in enumerate(rows):
        bbox = _clip_source_bbox(row.get("bbox_pdf_pt", []), width, height)
        if bbox is None:
            continue
        area_coverage = _union_coverage(
            bbox, [candidate["bbox_pdf_pt"] for candidate in claimable]
        )
        formula_coverage = _union_coverage(
            bbox,
            [
                candidate["bbox_pdf_pt"]
                for candidate in claimable
                if str(candidate.get("semantic_type", "")).upper() == "FORMULA"
            ],
        )
        threshold = (
            float(rules["formula_claim_coverage_threshold"])
            if formula_coverage > 0
            else float(rules["canonical_claim_coverage_threshold"])
        )
        if area_coverage >= threshold:
            claim_state = "CLAIMED"
        elif area_coverage >= float(rules["partial_claim_minimum_ratio"]):
            claim_state = "PARTIALLY_CLAIMED"
        else:
            claim_state = "UNCLAIMED"
        claim_states[claim_state] += 1
        if claim_state == "CLAIMED":
            continue

        subtract_boxes = []
        for candidate in claimable:
            semantic_type = str(candidate.get("semantic_type", "")).upper()
            if semantic_type == "FORMULA" and formula_coverage < float(
                rules["formula_residual_subtract_threshold"]
            ):
                continue
            subtract_boxes.append(candidate["bbox_pdf_pt"])
        segments = [bbox]
        if claim_state == "PARTIALLY_CLAIMED" and not rules.get(
            "residual_preserve_partial_line_envelope"
        ):
            segments = _horizontal_residual_segments(bbox, subtract_boxes)
        char_count = max(0, int(row.get("native_char_count", 0)))
        for segment_index, segment in enumerate(segments):
            residual_ratio = _bbox_area(segment) / max(_bbox_area(bbox), 1e-9)
            candidate_segment = segment
            band = row.get("source_line_band_bbox_pdf_pt")
            if rules.get("residual_use_source_line_band") and _valid_bbox(band):
                candidate_segment = [
                    float(band[0])
                    if claim_state == "UNCLAIMED"
                    and rules.get("residual_expand_unclaimed_to_block_width")
                    else float(segment[0]),
                    float(band[1]),
                    float(band[2])
                    if claim_state == "UNCLAIMED"
                    and rules.get("residual_expand_unclaimed_to_block_width")
                    else float(segment[2]),
                    float(band[3]),
                ]
            estimated_chars = (
                max(1, round(char_count * residual_ratio)) if char_count else 0
            )
            if (
                segment[2] - segment[0] < float(rules["residual_min_width_pt"])
                or _bbox_area(segment) < float(rules["residual_min_area_pt2"])
                or (
                    char_count
                    and estimated_chars < int(rules["residual_min_estimated_chars"])
                )
            ):
                tiny_count += 1
                continue
            evidence_id = str(row.get("evidence_id", f"native-line-{index:04d}"))
            units.append(
                {
                    "bbox_pdf_pt": candidate_segment,
                    "evidence_id": evidence_id,
                    "source_block_id": str(
                        row.get("source_block_id", f"block-{index:04d}")
                    ),
                    "source_line_span_ids": [
                        str(value)
                        for value in row.get("source_line_span_ids", [evidence_id])
                    ],
                    "native_char_count": estimated_chars or char_count,
                    "claim_state": claim_state,
                    "claimed_ratio": round(area_coverage, 6),
                    "residual_ratio": round(residual_ratio, 6),
                    "segment_index": segment_index,
                }
            )

    barriers = [
        row["bbox_pdf_pt"]
        for row in canonical_regions
        if str(row.get("semantic_type", "")).upper() in {"FORMULA", "TITLE"}
        and _valid_bbox(row.get("bbox_pdf_pt"))
    ]
    barriers.extend(
        row["bbox_pdf_pt"]
        for row in native_images
        if _valid_bbox(row.get("bbox_pdf_pt"))
    )
    groups = _group_residual_text_units(units, barriers=barriers, rules=rules)
    result = []
    for group in groups:
        bbox = _bbox_union([row["bbox_pdf_pt"] for row in group])
        bbox = _clip_source_bbox(
            [
                bbox[0]
                - float(rules.get("residual_candidate_horizontal_padding_pt", 0.0)),
                bbox[1]
                - float(rules.get("residual_candidate_vertical_padding_pt", 0.0)),
                bbox[2]
                + float(rules.get("residual_candidate_horizontal_padding_pt", 0.0)),
                bbox[3]
                + float(rules.get("residual_candidate_vertical_padding_pt", 0.0)),
            ],
            width,
            height,
        )
        if bbox is None:
            continue
        states = {row["claim_state"] for row in group}
        aggregate_state = (
            "PARTIALLY_CLAIMED" if "PARTIALLY_CLAIMED" in states else "UNCLAIMED"
        )
        native_chars = sum(int(row["native_char_count"]) for row in group)
        weighted_claim = sum(
            float(row["claimed_ratio"]) * max(1, int(row["native_char_count"]))
            for row in group
        ) / max(1, native_chars)
        result.append(
            _make_candidate(
                kind="NATIVE_TEXT_FALLBACK",
                semantic_hint="TEXT_LIKE",
                bbox=bbox,
                width=width,
                height=height,
                evidence_ids=sorted({str(row["evidence_id"]) for row in group}),
                reason_codes=[
                    "PARTIAL_NATIVE_TEXT_COVERAGE"
                    if aggregate_state == "PARTIALLY_CLAIMED"
                    else "UNCLAIMED_NATIVE_TEXT",
                    "LINE_LEVEL_RESIDUAL_SEGMENTATION",
                ],
                payload={
                    "subdivision_provenance": {
                        "source_unit": "LINE_OR_SPAN_GROUP",
                        "source_block_ids": sorted(
                            {str(row["source_block_id"]) for row in group}
                        ),
                        "source_line_span_ids": sorted(
                            {
                                value
                                for row in group
                                for value in row["source_line_span_ids"]
                            }
                        ),
                        "native_char_count": native_chars,
                        "claim_state": aggregate_state,
                        "claimed_ratio": round(weighted_claim, 6),
                        "residual_ratio": round(1.0 - weighted_claim, 6),
                        "residual_unit_count": len(group),
                    }
                },
            )
        )
    diagnostics = {
        "native_text_claim_states": {
            state: int(claim_states[state])
            for state in ("CLAIMED", "PARTIALLY_CLAIMED", "UNCLAIMED")
        },
        "native_text_residual_unit_count": len(units),
        "tiny_residual_fragment_count": tiny_count,
    }
    return result, diagnostics


def _source_unit_ownership_candidates(
    rows: Sequence[dict[str, Any]],
    canonical_regions: Sequence[dict[str, Any]],
    native_images: Sequence[dict[str, Any]],
    *,
    width: float,
    height: float,
    rules: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    claimable_types = {
        "TEXT",
        "TITLE",
        "FORMULA",
        "CAPTION",
        "HEADER_FOOTER",
        "PAGE_NUMBER",
    }
    claimable = [
        row
        for row in canonical_regions
        if str(row.get("semantic_type", "")).upper() in claimable_types
        and _valid_bbox(row.get("bbox_pdf_pt"))
    ]
    ownership_records: list[dict[str, Any]] = []
    unusable_source_units = 0
    for index, row in enumerate(rows):
        bbox = _clip_source_bbox(row.get("bbox_pdf_pt", []), width, height)
        if bbox is None:
            unusable_source_units += 1
            continue
        source_line_id = str(row.get("evidence_id", f"native-line-{index:04d}"))
        source_block_id = str(row.get("source_block_id", f"block-{index:04d}"))
        source_span_ids = sorted(
            str(value)
            for value in row.get("source_line_span_ids", [source_line_id])
        )
        native_char_count = max(0, int(row.get("native_char_count", 0)))
        text = str(row.get("text", ""))
        printable_text = "".join(character for character in text if character.isprintable())
        is_noncontent = native_char_count == 0 and not printable_text.strip()
        source_identity = {
            "source_kind": "NATIVE_TEXT_LINE",
            "bbox_pdf_pt": _normalized_bbox(bbox, width, height),
            "source_block_id": source_block_id,
            "source_line_id": source_line_id,
            "source_span_ids": source_span_ids,
            "native_char_count": native_char_count,
        }
        source_unit_id = "source-unit-" + hashlib.sha256(
            _canonical_json(source_identity).encode("utf-8")
        ).hexdigest()[:16]
        owners = []
        for candidate in claimable:
            intersection = _bbox_intersection(bbox, candidate["bbox_pdf_pt"])
            if intersection is None or _bbox_area(intersection) <= 0:
                continue
            owners.append(candidate)
        owner_boxes = [row["bbox_pdf_pt"] for row in owners]
        area_coverage = _union_coverage(bbox, owner_boxes)
        horizontal_coverage = _axis_union_coverage(bbox, owner_boxes, axis="x")
        vertical_coverage = _axis_union_coverage(bbox, owner_boxes, axis="y")
        owner_region_ids = sorted(str(row["region_id"]) for row in owners)
        owner_semantic_types = sorted(
            {str(row.get("semantic_type", "UNKNOWN")).upper() for row in owners}
        )
        formula_claim = "FORMULA" in owner_semantic_types
        area_threshold = float(
            rules[
                "ownership_formula_area_threshold"
                if formula_claim
                else "ownership_claim_area_threshold"
            ]
        )
        horizontal_threshold = float(
            rules[
                "ownership_formula_horizontal_threshold"
                if formula_claim
                else "ownership_claim_horizontal_threshold"
            ]
        )
        vertical_threshold = float(rules["ownership_claim_vertical_threshold"])
        partial_threshold = float(rules["ownership_partial_minimum_ratio"])
        ownership_score = min(
            area_coverage, horizontal_coverage, vertical_coverage
        )
        reason_codes: list[str] = []
        if is_noncontent:
            ownership_status = "IGNORED_NONCONTENT"
            reason_codes = ["EMPTY_OR_CONTROL_ONLY_SOURCE_UNIT"]
        elif not owners:
            ownership_status = "FALLBACK_REQUIRED"
            reason_codes = ["NO_CANONICAL_CLAIM"]
        elif len(owners) == 1 and (
            area_coverage >= area_threshold
            and horizontal_coverage >= horizontal_threshold
            and vertical_coverage >= vertical_threshold
        ):
            ownership_status = "CANONICAL_OWNED"
            reason_codes = ["COMPLETE_CANONICAL_CLAIM"]
        elif area_coverage >= partial_threshold:
            ownership_status = "SHARED_OR_UNCERTAIN"
            reason_codes.append("PARTIAL_CANONICAL_CLAIM")
            if len(owners) > 1:
                reason_codes.extend(
                    ["MULTIPLE_CANONICAL_CLAIMS", "CANONICAL_GEOMETRY_FRAGMENTED"]
                )
            if len(owner_semantic_types) > 1:
                reason_codes.append("SEMANTIC_CLAIM_AMBIGUOUS")
            if horizontal_coverage < horizontal_threshold:
                reason_codes.append("SOURCE_LINE_EXTENDS_BEYOND_CLAIM")
        else:
            ownership_status = "FALLBACK_REQUIRED"
            reason_codes = ["INSUFFICIENT_CANONICAL_COVERAGE"]
            if horizontal_coverage < horizontal_threshold:
                reason_codes.append("SOURCE_LINE_EXTENDS_BEYOND_CLAIM")
        ownership_records.append(
            {
                "source_unit_id": source_unit_id,
                "source_kind": "NATIVE_TEXT_LINE",
                "bbox_pdf_pt": bbox,
                "source_block_id": source_block_id,
                "source_line_id": source_line_id,
                "source_span_ids": source_span_ids,
                "native_char_count": native_char_count,
                "ownership_status": ownership_status,
                "owner_region_ids": owner_region_ids,
                "owner_semantic_types": owner_semantic_types,
                "ownership_score": round(ownership_score, 6),
                "claim_evidence": {
                    "area_coverage": round(area_coverage, 6),
                    "horizontal_coverage": round(horizontal_coverage, 6),
                    "vertical_coverage": round(vertical_coverage, 6),
                    "claiming_region_count": len(owners),
                },
                "reason_codes": reason_codes,
                "provenance": {
                    "source_evidence_id": source_line_id,
                    "source_unit": str(row.get("source_unit", "LINE")),
                    "line_geometry_reliable": True,
                    "full_source_unit_bbox_preserved": ownership_status
                    in {"SHARED_OR_UNCERTAIN", "FALLBACK_REQUIRED"},
                },
            }
        )
    ownership_records.sort(
        key=lambda row: (
            row["bbox_pdf_pt"][1],
            row["bbox_pdf_pt"][0],
            row["source_line_id"],
            row["source_unit_id"],
        )
    )
    fallback_units = [
        row
        for row in ownership_records
        if row["ownership_status"] in {"SHARED_OR_UNCERTAIN", "FALLBACK_REQUIRED"}
    ]
    barrier_records = [
        {
            "barrier_kind": str(row.get("semantic_type", "")).upper(),
            "bbox_pdf_pt": row["bbox_pdf_pt"],
        }
        for row in canonical_regions
        if str(row.get("semantic_type", "")).upper() in {"FORMULA", "TITLE"}
        and _valid_bbox(row.get("bbox_pdf_pt"))
    ]
    barrier_records.extend(
        {"barrier_kind": "IMAGE", "bbox_pdf_pt": row["bbox_pdf_pt"]}
        for row in native_images
        if _valid_bbox(row.get("bbox_pdf_pt"))
    )
    grouping_context = _adaptive_grouping_context(
        ownership_records,
        barrier_records=barrier_records,
        width=width,
        height=height,
        rules=rules,
    )
    barriers = [row["bbox_pdf_pt"] for row in grouping_context["strict_barriers"]]
    groups = _group_source_ownership_units(
        fallback_units,
        barriers=barriers,
        rules=rules,
        context=grouping_context,
    )
    candidates = []
    for group in groups:
        bbox = _clip_source_bbox(
            _bbox_union([row["bbox_pdf_pt"] for row in group]), width, height
        )
        if bbox is None:
            continue
        status_counts = Counter(row["ownership_status"] for row in group)
        native_char_count = sum(int(row["native_char_count"]) for row in group)
        source_unit_ids = sorted(str(row["source_unit_id"]) for row in group)
        source_line_ids = sorted(str(row["source_line_id"]) for row in group)
        source_span_ids = sorted(
            {value for row in group for value in row["source_span_ids"]}
        )
        owner_region_ids = sorted(
            {value for row in group for value in row["owner_region_ids"]}
        )
        ownership_statuses = sorted(status_counts)
        adaptive_enabled = grouping_context["adaptive_enabled"]
        grouping_reason = (
            "SINGLE_SOURCE_UNIT"
            if len(group) == 1
            else (
                "ADAPTIVE_LOCAL_CONTINUITY"
                if adaptive_enabled
                else "SAME_COLUMN_LOCAL_CONTINUITY"
            )
        )
        grouping_decision_reasons = (
            _grouping_decision_reasons(
                group,
                barriers=barriers,
                rules=rules,
                context=grouping_context,
            )
            if adaptive_enabled
            else []
        )
        grouping_provenance = {
            "grouping_version": str(rules["local_grouping_version"]),
            "grouping_reason": grouping_reason,
            "source_unit_count": len(group),
            "barrier_checked": True,
            "full_source_units_preserved": True,
        }
        if adaptive_enabled:
            grouping_provenance.update(
                {
                    "grouping_profile": grouping_context["profile"],
                    "local_spacing_stats": grouping_context["local_spacing_stats"],
                    "grouping_decision_reasons": grouping_decision_reasons,
                    "relaxed_inline_formula_barrier_count": len(
                        grouping_context["relaxed_inline_formula_barriers"]
                    ),
                }
            )
        candidates.append(
            _make_candidate(
                kind="NATIVE_TEXT_FALLBACK",
                semantic_hint="TEXT_LIKE",
                bbox=bbox,
                width=width,
                height=height,
                evidence_ids=source_line_ids,
                reason_codes=sorted(
                    {
                        "FULL_SOURCE_UNIT_PRESERVATION",
                        "LOCAL_SOURCE_UNIT_GROUPING",
                        *(
                            reason
                            for row in group
                            for reason in row["reason_codes"]
                        ),
                    }
                ),
                payload={
                    "source_unit_ids": source_unit_ids,
                    "source_block_ids": sorted(
                        {str(row["source_block_id"]) for row in group}
                    ),
                    "source_line_ids": source_line_ids,
                    "source_span_ids": source_span_ids,
                    "ownership_statuses": ownership_statuses,
                    "owner_region_ids": owner_region_ids,
                    "ownership_summary": {
                        "status_counts": {
                            status: int(status_counts[status])
                            for status in (
                                "CANONICAL_OWNED",
                                "SHARED_OR_UNCERTAIN",
                                "FALLBACK_REQUIRED",
                                "IGNORED_NONCONTENT",
                            )
                            if status_counts[status]
                        },
                        "mean_ownership_score": round(
                            statistics.fmean(
                                float(row["ownership_score"]) for row in group
                            ),
                            6,
                        ),
                    },
                    "grouping_provenance": grouping_provenance,
                    "subdivision_provenance": {
                        "source_unit": "NORMALIZED_NATIVE_LINE_GROUP",
                        "source_block_ids": sorted(
                            {str(row["source_block_id"]) for row in group}
                        ),
                        "source_line_span_ids": source_span_ids,
                        "native_char_count": native_char_count,
                        "claim_state": "SOURCE_UNIT_OWNERSHIP",
                        "claimed_ratio": round(
                            statistics.fmean(
                                float(row["ownership_score"]) for row in group
                            ),
                            6,
                        ),
                        "residual_ratio": 1.0,
                        "residual_unit_count": len(group),
                    },
                },
            )
        )
    ownership_counts = Counter(
        row["ownership_status"] for row in ownership_records
    )
    diagnostics = {
        "source_unit_ownership_version": str(
            rules["source_unit_ownership_version"]
        ),
        "source_unit_ownership": ownership_records,
        "source_unit_ownership_counts": {
            status: int(ownership_counts[status])
            for status in (
                "CANONICAL_OWNED",
                "SHARED_OR_UNCERTAIN",
                "FALLBACK_REQUIRED",
                "IGNORED_NONCONTENT",
            )
        },
        "native_text_claim_states": {
            "CLAIMED": int(ownership_counts["CANONICAL_OWNED"]),
            "PARTIALLY_CLAIMED": int(ownership_counts["SHARED_OR_UNCERTAIN"]),
            "UNCLAIMED": int(ownership_counts["FALLBACK_REQUIRED"]),
        },
        "native_text_residual_unit_count": len(fallback_units),
        "tiny_residual_fragment_count": 0,
        "source_unit_unusable_geometry_count": unusable_source_units,
        "uncertain_fallback_count": int(ownership_counts["SHARED_OR_UNCERTAIN"]),
        "multi_owner_source_unit_count": sum(
            len(row["owner_region_ids"]) > 1 for row in ownership_records
        ),
    }
    if grouping_context["adaptive_enabled"]:
        diagnostics["adaptive_local_grouping"] = {
            "version": str(rules["local_grouping_version"]),
            "profile_version": str(rules["adaptive_grouping_profile_version"]),
            "profile": grouping_context["profile"],
            "profile_reasons": grouping_context["profile_reasons"],
            "local_spacing_stats": grouping_context["local_spacing_stats"],
            "strict_barrier_count": len(grouping_context["strict_barriers"]),
            "relaxed_inline_formula_barrier_count": len(
                grouping_context["relaxed_inline_formula_barriers"]
            ),
        }
    return candidates, diagnostics


def _group_source_ownership_units(
    units: Sequence[dict[str, Any]],
    *,
    barriers: Sequence[Sequence[float]],
    rules: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for unit in sorted(
        units,
        key=lambda row: (
            row["bbox_pdf_pt"][1],
            row["bbox_pdf_pt"][0],
            row["source_line_id"],
            row["source_unit_id"],
        ),
    ):
        match = None
        for group in groups:
            if _source_ownership_units_can_group(
                group[-1],
                unit,
                barriers=barriers,
                rules=rules,
                context=context,
            ):
                if context and context["adaptive_enabled"]:
                    group_bbox = _bbox_union(
                        [row["bbox_pdf_pt"] for row in [*group, unit]]
                    )
                    group_height = float(group_bbox[3]) - float(group_bbox[1])
                    if group_height > float(context["maximum_group_height_pt"]):
                        continue
                match = group
                break
        if match is None:
            groups.append([unit])
        else:
            match.append(unit)
    return groups


def _source_ownership_units_can_group(
    first_unit: dict[str, Any],
    second_unit: dict[str, Any],
    *,
    barriers: Sequence[Sequence[float]],
    rules: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> bool:
    if context and context["adaptive_enabled"]:
        decision, _ = _adaptive_source_ownership_grouping_decision(
            first_unit,
            second_unit,
            barriers=barriers,
            rules=rules,
            context=context,
        )
        return decision
    first = first_unit["bbox_pdf_pt"]
    second = second_unit["bbox_pdf_pt"]
    first_height = float(first[3]) - float(first[1])
    second_height = float(second[3]) - float(second[1])
    vertical_overlap = max(
        0.0,
        min(float(first[3]), float(second[3]))
        - max(float(first[1]), float(second[1])),
    )
    baseline_overlap = vertical_overlap / max(
        1e-9, min(first_height, second_height)
    )
    first_center_y = (float(first[1]) + float(first[3])) / 2
    second_center_y = (float(second[1]) + float(second[3])) / 2
    same_inline_cluster = baseline_overlap >= 0.25 or abs(
        first_center_y - second_center_y
    ) <= max(first_height, second_height)
    horizontal_gap = max(
        0.0,
        max(float(first[0]), float(second[0]))
        - min(float(first[2]), float(second[2])),
    )
    same_source_block = str(first_unit["source_block_id"]) == str(
        second_unit["source_block_id"]
    )
    fragmented_glyph_run = min(
        int(first_unit["native_char_count"]), int(second_unit["native_char_count"])
    ) <= 4
    maximum_inline_gap = min(
        float(rules["ownership_group_same_baseline_max_gap_pt"]),
        max(first_height, second_height)
        * float(rules["ownership_group_same_baseline_gap_height_factor"]),
    )
    if (
        (same_source_block or fragmented_glyph_run)
        and same_inline_cluster
        and horizontal_gap <= maximum_inline_gap
    ):
        # PyMuPDF commonly exposes formula glyph runs as multiple one-character
        # lines, sometimes split across source blocks. Reconstructing a tightly
        # adjacent inline cluster does not bridge a column or a vertical barrier.
        return True
    translated_rules = {
        "residual_max_vertical_gap_pt": rules["ownership_group_max_vertical_gap_pt"],
        "residual_vertical_gap_height_factor": rules[
            "ownership_group_vertical_gap_height_factor"
        ],
        "residual_horizontal_alignment_tolerance_pt": rules[
            "ownership_group_horizontal_alignment_tolerance_pt"
        ],
        "residual_min_horizontal_overlap_ratio": rules[
            "ownership_group_min_horizontal_overlap_ratio"
        ],
    }
    return _residual_units_can_group(
        first, second, barriers=barriers, rules=translated_rules
    )


def _adaptive_grouping_context(
    units: Sequence[dict[str, Any]],
    *,
    barrier_records: Sequence[dict[str, Any]],
    width: float,
    height: float,
    rules: dict[str, Any],
) -> dict[str, Any]:
    adaptive_enabled = (
        str(rules.get("local_grouping_version", ""))
        == "adaptive-local-grouping-v1"
    )
    valid_units = [row for row in units if _valid_bbox(row.get("bbox_pdf_pt"))]
    line_heights = [
        float(row["bbox_pdf_pt"][3]) - float(row["bbox_pdf_pt"][1])
        for row in valid_units
    ]
    median_line_height = statistics.median(line_heights) if line_heights else 0.0
    block_gaps: list[float] = []
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_units:
        by_block[str(row["source_block_id"])].append(row)
    for block_units in by_block.values():
        ordered = sorted(
            block_units,
            key=lambda row: (
                row["bbox_pdf_pt"][1],
                row["bbox_pdf_pt"][0],
                row["source_line_id"],
            ),
        )
        for first_unit, second_unit in pairwise(ordered):
            first = first_unit["bbox_pdf_pt"]
            second = second_unit["bbox_pdf_pt"]
            gap = float(second[1]) - float(first[3])
            overlap = max(
                0.0,
                min(float(first[2]), float(second[2]))
                - max(float(first[0]), float(second[0])),
            )
            minimum_width = max(
                1e-9,
                min(float(first[2]) - float(first[0]), float(second[2]) - float(second[0])),
            )
            if gap >= 0 and overlap / minimum_width >= 0.3:
                block_gaps.append(gap)
    median_block_gap = statistics.median(block_gaps) if block_gaps else 0.0
    fragmented_count = sum(
        int(row.get("native_char_count", 0)) <= 4 for row in valid_units
    )
    fragmented_ratio = fragmented_count / len(valid_units) if valid_units else 0.0
    formula_records = [
        row for row in barrier_records if row["barrier_kind"] == "FORMULA"
    ]
    page_area = max(1e-9, width * height)
    formula_area_ratio = sum(
        _bbox_area(row["bbox_pdf_pt"]) for row in formula_records
    ) / page_area
    formula_region_ratio = len(formula_records) / max(1, len(valid_units))
    column_like = _adaptive_column_like(valid_units, width=width, height=height)
    profile_reasons: list[str] = []
    formula_dense = (
        formula_region_ratio >= float(rules.get("adaptive_formula_region_ratio", 1.0))
        or formula_area_ratio >= float(rules.get("adaptive_formula_area_ratio", 1.0))
    )
    source_fragmented = fragmented_ratio >= float(
        rules.get("adaptive_fragmented_unit_ratio", 1.0)
    )
    if column_like:
        profile = "MULTI_COLUMN"
        profile_reasons.append("TWO_SOURCE_COLUMN_BANDS")
    elif formula_dense:
        profile = "FORMULA_DENSE"
        profile_reasons.append("FORMULA_GEOMETRY_DENSE")
    elif source_fragmented:
        profile = "SOURCE_FRAGMENTED"
        profile_reasons.append("SHORT_SOURCE_UNIT_RATIO_HIGH")
    elif len(valid_units) >= int(rules.get("adaptive_dense_line_count", 10**9)) or (
        len(valid_units) >= 20
        and median_line_height > 0
        and median_block_gap / median_line_height
        <= float(rules.get("adaptive_dense_gap_height_ratio", 0.0))
    ):
        profile = "DENSE_TEXT"
        profile_reasons.append("SOURCE_LINE_DENSITY_HIGH")
    else:
        profile = "STANDARD_TEXT"
        profile_reasons.append("STANDARD_SOURCE_GEOMETRY")
    relax_inline_formula = formula_dense or source_fragmented
    strict_barriers: list[dict[str, Any]] = []
    relaxed_barriers: list[dict[str, Any]] = []
    for record in barrier_records:
        bbox = record["bbox_pdf_pt"]
        barrier_height = float(bbox[3]) - float(bbox[1])
        inline_formula = (
            adaptive_enabled
            and relax_inline_formula
            and record["barrier_kind"] == "FORMULA"
            and median_line_height > 0
            and barrier_height
            <= median_line_height
            * float(rules.get("adaptive_inline_formula_max_height_factor", 0.0))
            and _bbox_area(bbox) / page_area
            <= float(rules.get("adaptive_inline_formula_max_page_area_ratio", 0.0))
        )
        (relaxed_barriers if inline_formula else strict_barriers).append(record)
    maximum_group_height = max(
        float(rules.get("adaptive_min_group_height_cap_pt", 72.0)),
        min(
            height * float(rules.get("adaptive_max_group_page_height_ratio", 0.35)),
            median_line_height
            * float(rules.get("adaptive_max_group_height_lines", 14.0)),
        ),
    )
    return {
        "adaptive_enabled": adaptive_enabled,
        "profile": profile,
        "profile_reasons": profile_reasons,
        "page_width_pt": round(width, 6),
        "page_height_pt": round(height, 6),
        "maximum_group_height_pt": round(maximum_group_height, 6),
        "strict_barriers": strict_barriers,
        "relaxed_inline_formula_barriers": relaxed_barriers,
        "local_spacing_stats": {
            "source_line_count": len(valid_units),
            "source_block_count": len(by_block),
            "median_line_height_pt": round(median_line_height, 6),
            "median_same_block_gap_pt": round(median_block_gap, 6),
            "fragmented_source_unit_ratio": round(fragmented_ratio, 6),
            "formula_region_ratio": round(formula_region_ratio, 6),
            "formula_area_ratio": round(formula_area_ratio, 6),
            "column_like": column_like,
        },
    }


def _adaptive_column_like(
    units: Sequence[dict[str, Any]], *, width: float, height: float
) -> bool:
    if len(units) < 4 or width <= 0 or height <= 0:
        return False
    middle = width / 2
    margin = width * 0.02
    left = [row for row in units if float(row["bbox_pdf_pt"][2]) <= middle - margin]
    right = [row for row in units if float(row["bbox_pdf_pt"][0]) >= middle + margin]
    minimum_band_count = max(2, math.ceil(len(units) * 0.2))
    if len(left) < minimum_band_count or len(right) < minimum_band_count:
        return False
    left_extent = [
        min(float(row["bbox_pdf_pt"][1]) for row in left),
        max(float(row["bbox_pdf_pt"][3]) for row in left),
    ]
    right_extent = [
        min(float(row["bbox_pdf_pt"][1]) for row in right),
        max(float(row["bbox_pdf_pt"][3]) for row in right),
    ]
    vertical_overlap = max(
        0.0,
        min(left_extent[1], right_extent[1]) - max(left_extent[0], right_extent[0]),
    )
    minimum_extent = max(
        1e-9,
        min(left_extent[1] - left_extent[0], right_extent[1] - right_extent[0]),
    )
    return vertical_overlap / minimum_extent >= 0.25


def _adaptive_source_ownership_grouping_decision(
    first_unit: dict[str, Any],
    second_unit: dict[str, Any],
    *,
    barriers: Sequence[Sequence[float]],
    rules: dict[str, Any],
    context: dict[str, Any],
) -> tuple[bool, str]:
    first = first_unit["bbox_pdf_pt"]
    second = second_unit["bbox_pdf_pt"]
    if float(second[1]) < float(first[1]):
        first_unit, second_unit = second_unit, first_unit
        first, second = second, first
    first_height = float(first[3]) - float(first[1])
    second_height = float(second[3]) - float(second[1])
    height = max(first_height, second_height)
    page_width = float(context["page_width_pt"])
    middle = page_width / 2
    margin = page_width * float(rules.get("adaptive_column_margin_ratio", 0.02))
    opposite_columns = (
        context["profile"] == "MULTI_COLUMN"
        and (
            (
                float(first[2]) <= middle - margin
                and float(second[0]) >= middle + margin
            )
            or (
                float(second[2]) <= middle - margin
                and float(first[0]) >= middle + margin
            )
        )
    )
    if opposite_columns:
        return False, "MULTI_COLUMN_STRICT_BOUNDARY"
    vertical_overlap = max(
        0.0,
        min(float(first[3]), float(second[3]))
        - max(float(first[1]), float(second[1])),
    )
    baseline_overlap = vertical_overlap / max(1e-9, min(first_height, second_height))
    first_center_y = (float(first[1]) + float(first[3])) / 2
    second_center_y = (float(second[1]) + float(second[3])) / 2
    same_inline_cluster = baseline_overlap >= 0.25 or abs(
        first_center_y - second_center_y
    ) <= height
    horizontal_gap = max(
        0.0,
        max(float(first[0]), float(second[0]))
        - min(float(first[2]), float(second[2])),
    )
    same_source_block = str(first_unit["source_block_id"]) == str(
        second_unit["source_block_id"]
    )
    fragmented_glyph_run = min(
        int(first_unit["native_char_count"]), int(second_unit["native_char_count"])
    ) <= 4
    maximum_inline_gap = min(
        float(rules["ownership_group_same_baseline_max_gap_pt"]),
        height * float(rules["ownership_group_same_baseline_gap_height_factor"]),
    )
    if (
        (same_source_block or fragmented_glyph_run)
        and same_inline_cluster
        and horizontal_gap <= maximum_inline_gap
    ):
        if _strict_barrier_separates_units(first, second, context=context):
            return False, "STRICT_BARRIER"
        return True, (
            "SAME_BLOCK_INLINE_CONTINUITY"
            if same_source_block
            else "FRAGMENTED_GLYPH_INLINE_CONTINUITY"
        )
    gap = float(second[1]) - float(first[3])
    spacing = float(context["local_spacing_stats"]["median_same_block_gap_pt"])
    if same_source_block:
        maximum_gap = max(
            height * float(rules["adaptive_same_block_gap_height_factor"]),
            spacing * float(rules["adaptive_same_block_spacing_factor"]),
        )
        accepted_reason = "SAME_BLOCK_ADAPTIVE_GAP"
    else:
        maximum_gap = max(
            height * float(rules["adaptive_cross_block_gap_height_factor"]),
            spacing * float(rules["adaptive_cross_block_spacing_factor"]),
        )
        accepted_reason = "PARAGRAPH_GEOMETRY_CONTINUITY"
    maximum_gap = min(float(rules["adaptive_max_vertical_gap_pt"]), maximum_gap)
    if gap < -height * 0.5 or gap > maximum_gap:
        return False, "VERTICAL_GAP_EXCEEDS_PROFILE"
    overlap = max(
        0.0,
        min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])),
    )
    minimum_width = max(
        1e-9,
        min(float(first[2]) - float(first[0]), float(second[2]) - float(second[0])),
    )
    aligned = abs(float(first[0]) - float(second[0])) <= max(
        float(rules["ownership_group_horizontal_alignment_tolerance_pt"]),
        height,
    )
    if not aligned and overlap / minimum_width < float(
        rules["ownership_group_min_horizontal_overlap_ratio"]
    ):
        return False, "HORIZONTAL_ALIGNMENT_REJECTED"
    if _strict_barrier_separates_units(first, second, context=context):
        return False, "STRICT_BARRIER"
    return True, accepted_reason


def _strict_barrier_separates_units(
    first: Sequence[float],
    second: Sequence[float],
    *,
    context: dict[str, Any],
) -> bool:
    corridor = _source_unit_corridor(first, second)
    for record in context["strict_barriers"]:
        barrier = record["bbox_pdf_pt"]
        if not _bbox_intersection(corridor, barrier):
            continue
        first_inside = _bbox_intersection(first, barrier) is not None
        second_inside = _bbox_intersection(second, barrier) is not None
        if record["barrier_kind"] == "IMAGE":
            # A shared raster backdrop is not geometrically between the units.
            # A raster intersecting only one side remains a strict separator.
            if not (first_inside and second_inside):
                return True
            continue
        if not (first_inside and second_inside):
            return True
    return False


def _source_unit_corridor(
    first: Sequence[float], second: Sequence[float]
) -> list[float]:
    if float(second[1]) < float(first[1]):
        first, second = second, first
    if float(second[1]) <= float(first[3]):
        return [
            min(float(first[2]), float(second[2])),
            max(float(first[1]), float(second[1])),
            max(float(first[0]), float(second[0])),
            min(float(first[3]), float(second[3])),
        ]
    return [
        min(float(first[0]), float(second[0])),
        float(first[3]),
        max(float(first[2]), float(second[2])),
        float(second[1]),
    ]


def _grouping_decision_reasons(
    group: Sequence[dict[str, Any]],
    *,
    barriers: Sequence[Sequence[float]],
    rules: dict[str, Any],
    context: dict[str, Any],
) -> list[str]:
    if len(group) == 1:
        return ["SINGLE_SOURCE_UNIT"]
    reasons = []
    for first_unit, second_unit in pairwise(group):
        accepted, reason = _adaptive_source_ownership_grouping_decision(
            first_unit,
            second_unit,
            barriers=barriers,
            rules=rules,
            context=context,
        )
        if accepted:
            reasons.append(reason)
    return sorted(set(reasons)) or ["DETERMINISTIC_LOCAL_CONTINUITY"]


def _legacy_native_text_candidates(
    rows: Iterable[dict[str, Any]],
    canonical_boxes: Sequence[Sequence[float]],
    *,
    width: float,
    height: float,
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for group in _group_near_identical(rows, float(rules["fallback_dedup_iou"])):
        bbox = _clip_source_bbox(
            _bbox_union([row["bbox_pdf_pt"] for row in group]), width, height
        )
        if bbox is None:
            continue
        area_coverage = _union_coverage(bbox, canonical_boxes)
        horizontal_coverage = max(
            (_horizontal_coverage(bbox, candidate) for candidate in canonical_boxes),
            default=0.0,
        )
        if area_coverage >= float(rules["canonical_claim_coverage_threshold"]):
            continue
        reason = (
            "PARTIAL_NATIVE_TEXT_COVERAGE"
            if area_coverage > 0
            or horizontal_coverage
            >= float(rules["partial_text_horizontal_overlap_threshold"])
            else "UNCLAIMED_NATIVE_TEXT"
        )
        result.append(
            _make_candidate(
                kind="NATIVE_TEXT_FALLBACK",
                semantic_hint="TEXT_LIKE",
                bbox=bbox,
                width=width,
                height=height,
                evidence_ids=sorted(str(row["evidence_id"]) for row in group),
                reason_codes=[reason],
            )
        )
    return result


def _empty_text_diagnostics() -> dict[str, Any]:
    return {
        "native_text_claim_states": {
            "CLAIMED": 0,
            "PARTIALLY_CLAIMED": 0,
            "UNCLAIMED": 0,
        },
        "native_text_residual_unit_count": 0,
        "tiny_residual_fragment_count": 0,
    }


def _horizontal_residual_segments(
    bbox: Sequence[float], claim_boxes: Sequence[Sequence[float]]
) -> list[list[float]]:
    intervals = []
    line_height = max(1e-9, float(bbox[3]) - float(bbox[1]))
    for claim in claim_boxes:
        intersection = _bbox_intersection(bbox, claim)
        if intersection is None:
            continue
        vertical_ratio = (intersection[3] - intersection[1]) / line_height
        if vertical_ratio >= 0.5:
            intervals.append([intersection[0], intersection[2]])
    if not intervals:
        return [[float(value) for value in bbox]]
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    cursor = float(bbox[0])
    result = []
    for start, end in merged:
        if start > cursor:
            result.append([cursor, float(bbox[1]), start, float(bbox[3])])
        cursor = max(cursor, end)
    if cursor < float(bbox[2]):
        result.append([cursor, float(bbox[1]), float(bbox[2]), float(bbox[3])])
    return result


def _group_residual_text_units(
    units: Sequence[dict[str, Any]],
    *,
    barriers: Sequence[Sequence[float]],
    rules: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for unit in sorted(
        units,
        key=lambda row: (
            row["bbox_pdf_pt"][1],
            row["bbox_pdf_pt"][0],
            row["evidence_id"],
            row["segment_index"],
        ),
    ):
        match = None
        for group in groups:
            previous = group[-1]
            if _residual_units_can_group(
                previous["bbox_pdf_pt"],
                unit["bbox_pdf_pt"],
                barriers=barriers,
                rules=rules,
            ):
                match = group
                break
        if match is None:
            groups.append([unit])
        else:
            match.append(unit)
    return groups


def _residual_units_can_group(
    first: Sequence[float],
    second: Sequence[float],
    *,
    barriers: Sequence[Sequence[float]],
    rules: dict[str, Any],
) -> bool:
    if float(second[1]) < float(first[1]):
        first, second = second, first
    gap = float(second[1]) - float(first[3])
    height = max(float(first[3]) - float(first[1]), float(second[3]) - float(second[1]))
    maximum_gap = min(
        float(rules["residual_max_vertical_gap_pt"]),
        height * float(rules["residual_vertical_gap_height_factor"]),
    )
    if gap < -height * 0.5 or gap > maximum_gap:
        return False
    overlap = max(
        0.0,
        min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])),
    )
    minimum_width = max(
        1e-9,
        min(float(first[2]) - float(first[0]), float(second[2]) - float(second[0])),
    )
    aligned = abs(float(first[0]) - float(second[0])) <= float(
        rules["residual_horizontal_alignment_tolerance_pt"]
    )
    if not aligned and overlap / minimum_width < float(
        rules["residual_min_horizontal_overlap_ratio"]
    ):
        return False
    corridor = [
        min(float(first[0]), float(second[0])),
        min(float(first[3]), float(second[1])),
        max(float(first[2]), float(second[2])),
        max(float(first[3]), float(second[1])),
    ]
    return not any(_bbox_intersection(corridor, barrier) for barrier in barriers)


def _native_image_candidates(
    rows: Iterable[dict[str, Any]],
    canonical_boxes: Sequence[Sequence[float]],
    *,
    width: float,
    height: float,
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    groups = _group_images(
        rows,
        iou_threshold=float(rules["image_group_iou"]),
        containment_threshold=float(rules["image_group_containment"]),
    )
    for group in groups:
        bbox = _clip_source_bbox(
            _bbox_union([row["bbox_pdf_pt"] for row in group]), width, height
        )
        if bbox is None:
            continue
        if _union_coverage(bbox, canonical_boxes) >= float(
            rules["canonical_claim_coverage_threshold"]
        ):
            continue
        reasons = ["UNCLAIMED_NATIVE_IMAGE"]
        if len(group) > 1:
            reasons.append("GROUPED_NATIVE_IMAGE_PLACEMENT")
        result.append(
            _make_candidate(
                kind="NATIVE_IMAGE_FALLBACK",
                semantic_hint="IMAGE_LIKE",
                bbox=bbox,
                width=width,
                height=height,
                evidence_ids=sorted(str(row["evidence_id"]) for row in group),
                reason_codes=reasons,
            )
        )
    return result


def _native_vector_candidates(
    rows: Iterable[dict[str, Any]],
    canonical_boxes: Sequence[Sequence[float]],
    *,
    width: float,
    height: float,
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for group in _group_near_identical(rows, float(rules["fallback_dedup_iou"])):
        bbox = _clip_source_bbox(
            _bbox_union([row["bbox_pdf_pt"] for row in group]), width, height
        )
        if bbox is None:
            continue
        if _union_coverage(bbox, canonical_boxes) >= float(
            rules["canonical_claim_coverage_threshold"]
        ):
            continue
        result.append(
            _make_candidate(
                kind="VECTOR_VISUAL_FALLBACK",
                semantic_hint="VISUAL_UNKNOWN",
                bbox=bbox,
                width=width,
                height=height,
                evidence_ids=sorted(str(row["evidence_id"]) for row in group),
                reason_codes=["UNCLAIMED_NATIVE_VECTOR"],
            )
        )
    return result


def _low_score_candidates(
    raw_page: dict[str, Any],
    page_ir: dict[str, Any],
    source_rows: Sequence[dict[str, Any]],
    canonical_boxes: Sequence[Sequence[float]],
    *,
    width: float,
    height: float,
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    transform = page_ir.get("render_transform", {})
    scale_x = float(transform.get("scale_x", 1.0)) or 1.0
    scale_y = float(transform.get("scale_y", 1.0)) or 1.0
    minimum = max(
        float(raw_page.get("capture_floor", 0.3)), float(rules["low_score_minimum"])
    )
    maximum = min(
        float(page_ir.get("score_threshold", 0.5)), float(rules["low_score_maximum"])
    )
    result = []
    for raw in raw_page.get("raw_detections", []):
        score = float(raw.get("raw_score", 0.0))
        if score < minimum or score >= maximum:
            continue
        render_bbox = raw.get("raw_bbox_render_px")
        if not _valid_bbox(render_bbox):
            continue
        bbox = _clip_source_bbox(
            [
                float(render_bbox[0]) / scale_x,
                float(render_bbox[1]) / scale_y,
                float(render_bbox[2]) / scale_x,
                float(render_bbox[3]) / scale_y,
            ],
            width,
            height,
        )
        if bbox is None:
            continue
        if _union_coverage(bbox, canonical_boxes) > 0:
            continue
        aligned = [
            row
            for row in source_rows
            if _valid_bbox(row.get("bbox_pdf_pt"))
            and _bbox_iou(bbox, row["bbox_pdf_pt"])
            >= float(rules["low_score_source_alignment_iou"])
        ]
        if not aligned:
            continue
        raw_id = str(raw.get("raw_detection_id", ""))
        label = str(raw.get("raw_label", "")).lower()
        semantic_hint = (
            "TEXT_LIKE"
            if "text" in label
            else "IMAGE_LIKE"
            if "image" in label
            else "VISUAL_UNKNOWN"
        )
        result.append(
            _make_candidate(
                kind="MODEL_LOW_SCORE_FALLBACK",
                semantic_hint=semantic_hint,
                bbox=bbox,
                width=width,
                height=height,
                evidence_ids=[
                    raw_id,
                    *sorted(str(row["evidence_id"]) for row in aligned),
                ],
                reason_codes=["LOW_SCORE_SOURCE_ALIGNED"],
                payload={"raw_model_label": raw.get("raw_label"), "raw_score": score},
            )
        )
    return result


def decide_page_escalation(
    *,
    page_record: dict[str, Any],
    source_evidence: dict[str, Any],
    local_candidates: Sequence[dict[str, Any]],
    width: float,
    height: float,
    strategy: dict[str, Any],
) -> dict[str, Any]:
    """Return a deterministic page-level processing control signal.

    The result deliberately has no bbox and never participates in region ordering or
    fallback burden. It uses only frozen source/native geometry and local Fusion
    diagnostics; no recognized content or prediction label is introduced here.
    """
    profile = str(page_record.get("source_profile", "UNCERTAIN"))
    route = str(page_record.get("routing_decision", "REVIEW_REQUIRED"))
    text_trust = str(page_record.get("native_text_trust", "NONE"))
    images = page_record.get("images", {})
    native_text = page_record.get("native_text", {})
    vectors = page_record.get("vectors", {})
    largest_raster = float(images.get("largest_image_coverage_ratio", 0.0))
    raster_union = float(images.get("page_coverage_ratio", largest_raster))
    full_page_threshold = float(strategy.get("full_page_raster_threshold", 0.85))
    major_visual_threshold = float(strategy.get("major_visual_load_threshold", 0.8))
    localized_threshold = float(
        strategy.get("localized_visual_coverage_threshold", 0.5)
    )
    page_box = [0.0, 0.0, width, height]
    local_visual_boxes = [
        row["bbox_pdf_pt"]
        for row in local_candidates
        if row.get("candidate_kind")
        in {"NATIVE_IMAGE_FALLBACK", "VECTOR_VISUAL_FALLBACK"}
        or row.get("semantic_hint") in {"IMAGE", "TABLE", "VISUAL_UNKNOWN"}
    ]
    localized_visual_coverage = _union_coverage(page_box, local_visual_boxes)
    full_page_raster = largest_raster >= full_page_threshold

    if profile == "UNCERTAIN" or route == "REVIEW_REQUIRED":
        status = "VISUAL_PAGE_REVIEW"
        reason_codes = ["AMBIGUOUS_PAGE_STRUCTURE"]
    elif profile in {"IMAGE_ONLY", "SCAN", "SCANNED"} and text_trust == "NONE":
        status = "VISUAL_PAGE_REQUIRED"
        reason_codes = ["IMAGE_ONLY_SOURCE"]
        if full_page_raster:
            reason_codes.insert(0, "FULL_PAGE_RASTER_CARRIER")
    elif (
        profile == "IMAGE_WITH_TEXT_LAYER"
        and text_trust in {"LOW", "NONE"}
        and full_page_raster
    ):
        status = "VISUAL_PAGE_REQUIRED"
        reason_codes = [
            "FULL_PAGE_RASTER_CARRIER",
            "IMAGE_WITH_LOW_TRUST_TEXT_LAYER",
            "SCAN_LIKE_PAGE",
        ]
    elif (
        route == "HYBRID_REQUIRED"
        and max(largest_raster, raster_union) >= major_visual_threshold
        and localized_visual_coverage < localized_threshold
    ):
        status = "VISUAL_PAGE_REQUIRED"
        reason_codes = [
            "UNLOCALIZED_MAJOR_VISUAL_LOAD",
            "SOURCE_STRUCTURE_INSUFFICIENT",
        ]
    else:
        status = "NONE"
        reason_codes = ["LOCAL_REGIONS_SUFFICIENT"]

    return {
        "status": status,
        "reason_codes": reason_codes,
        "source_signals": {
            "source_profile": profile,
            "page_route": route,
            "native_text_trust": text_trust,
            "largest_raster_coverage": round(largest_raster, 6),
            "raster_union_coverage": round(raster_union, 6),
            "full_page_raster_signal": full_page_raster,
            "native_text_non_whitespace_char_count": int(
                native_text.get("non_whitespace_char_count", 0)
            ),
            "native_image_placement_count": int(images.get("placed_image_count", 0)),
            "native_vector_coverage": round(
                float(vectors.get("drawing_coverage_ratio", 0.0)), 6
            ),
            "localized_visual_coverage": round(localized_visual_coverage, 6),
            "page_visual_bbox_present": _valid_bbox(
                source_evidence.get("page_visual_bbox")
            ),
        },
        "decision_version": str(
            strategy.get("page_escalation_decision_version", "page-escalation-v0")
        ),
        "provenance": {
            "strategy_id": str(strategy.get("strategy_id", "unknown")),
            "source_only_and_local_fusion_signals": True,
            "ocr_or_model_classifier_used": False,
        },
    }


def _make_candidate(
    *,
    kind: str,
    semantic_hint: str,
    bbox: Sequence[float],
    width: float,
    height: float,
    evidence_ids: list[str],
    reason_codes: list[str],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_bbox = _normalized_bbox(bbox, width, height)
    identity = {
        "candidate_kind": kind,
        "semantic_hint": semantic_hint,
        "bbox_pdf_pt": normalized_bbox,
        "evidence_ids": evidence_ids,
        "reason_codes": reason_codes,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:16]
    candidate = {
        "candidate_id": f"fusion-{digest}",
        **identity,
        "bbox_normalized": [
            round(normalized_bbox[0] / width, 8),
            round(normalized_bbox[1] / height, 8),
            round(normalized_bbox[2] / width, 8),
            round(normalized_bbox[3] / height, 8),
        ],
    }
    if payload:
        candidate.update(payload)
    return candidate


def _deduplicate_fallbacks(
    candidates: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    priorities = {
        "MODEL_LOW_SCORE_FALLBACK": 0,
        "NATIVE_TEXT_FALLBACK": 1,
        "NATIVE_IMAGE_FALLBACK": 2,
        "VECTOR_VISUAL_FALLBACK": 3,
    }
    for candidate in sorted(
        candidates,
        key=lambda row: (priorities[row["candidate_kind"]], *_candidate_sort_key(row)),
    ):
        duplicate = next(
            (
                row
                for row in kept
                if row["semantic_hint"] == candidate["semantic_hint"]
                and _bbox_iou(row["bbox_pdf_pt"], candidate["bbox_pdf_pt"]) >= threshold
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
            continue
        duplicate["evidence_ids"] = sorted(
            set(duplicate["evidence_ids"] + candidate["evidence_ids"])
        )
        duplicate["reason_codes"] = sorted(
            set(duplicate["reason_codes"] + candidate["reason_codes"])
        )
        for field in (
            "source_unit_ids",
            "source_block_ids",
            "source_line_ids",
            "source_span_ids",
            "ownership_statuses",
            "owner_region_ids",
        ):
            if candidate.get(field):
                duplicate[field] = sorted(
                    {
                        str(value)
                        for value in [
                            *duplicate.get(field, []),
                            *candidate.get(field, []),
                        ]
                    }
                )
        if candidate.get("grouping_provenance"):
            duplicate.setdefault("deduplicated_source_unit_groups", []).append(
                copy.deepcopy(candidate["grouping_provenance"])
            )
        if candidate.get("ownership_summary") and not duplicate.get(
            "ownership_summary"
        ):
            duplicate["ownership_summary"] = copy.deepcopy(
                candidate["ownership_summary"]
            )
        if candidate.get("subdivision_provenance") and not duplicate.get(
            "subdivision_provenance"
        ):
            duplicate["subdivision_provenance"] = copy.deepcopy(
                candidate["subdivision_provenance"]
            )
    return kept


def _apply_fallback_limit(
    candidates: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("max_fallback_candidates_per_page cannot be negative")
    return candidates[:limit]


def _prediction_free_rows(
    rows: Iterable[dict[str, Any]], excluded: set[tuple[str, int]]
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if _page_key(row) in excluded:
            continue
        if any("prediction" in str(key).lower() for key in row):
            raise ValueError("Fresh selection cannot consume prediction fields")
        result.append(copy.deepcopy(row))
    return result


def _selection_row(row: dict[str, Any], reference_set: str) -> dict[str, Any]:
    return {
        "document_id": str(row["document_id"]),
        "page_index": int(row["page_index"]),
        "page_number": int(row.get("page_number", int(row["page_index"]) + 1)),
        "source_group": _source_value(row, "source_group"),
        "source_path": row.get("source_path"),
        "source_sha256": row.get("source_sha256"),
        "routing_decision": _source_value(row, "routing_decision"),
        "source_profile": _source_value(row, "source_profile"),
        "reading_order_risk": _source_value(row, "reading_order_risk"),
        "reference_set": reference_set,
        "selection_basis": FRESH_SELECTION_BASIS,
        "selection_digest": _stable_digest(
            reference_set, row["document_id"], row["page_index"]
        ),
    }


def _support_signals(row: dict[str, Any]) -> list[str]:
    signals = []
    images = row.get("images", {})
    vectors = row.get("vectors", {})
    if int(images.get("placed_image_count", 0)) > 0:
        signals.append("NATIVE_IMAGE_PLACEMENT")
    if float(images.get("largest_image_coverage_ratio", 0.0)) >= 0.5:
        signals.append("LARGE_IMAGE_COVERAGE")
    if int(vectors.get("drawing_count", 0)) > 0:
        signals.append("NATIVE_VECTOR_DRAWINGS")
    if float(vectors.get("drawing_coverage_ratio", 0.0)) >= 0.15:
        signals.append("VECTOR_COVERAGE")
    table_signals = row.get("table_signals", row.get("tables", {}))
    if any(
        float(value or 0) > 0
        for value in table_signals.values()
        if isinstance(value, (int, float))
    ):
        signals.append("SOURCE_GRID_OR_TABLE_SIGNAL")
    return sorted(set(signals))


def _support_signal_weight(signal: str) -> int:
    return {
        "SOURCE_GRID_OR_TABLE_SIGNAL": 5,
        "LARGE_IMAGE_COVERAGE": 4,
        "VECTOR_COVERAGE": 3,
        "NATIVE_IMAGE_PLACEMENT": 2,
        "NATIVE_VECTOR_DRAWINGS": 1,
    }[signal]


def _coverage_summary(values: list[float]) -> dict[str, Any]:
    return {
        "region_count": len(values),
        "mean_union_coverage": round(statistics.fmean(values), 6) if values else 0.0,
        "coverage_gte_0_9_count": sum(value >= 0.9 for value in values),
        "coverage_gte_0_9_rate": round(
            sum(value >= 0.9 for value in values) / len(values), 6
        )
        if values
        else 0.0,
        "coverage_gte_0_5_count": sum(value >= 0.5 for value in values),
        "coverage_gte_0_5_rate": round(
            sum(value >= 0.5 for value in values) / len(values), 6
        )
        if values
        else 0.0,
    }


def _evidence_coverage_diagnostics(
    source_evidence: dict[str, Any],
    canonical_boxes: Sequence[Sequence[float]],
    fusion_boxes: Sequence[Sequence[float]],
) -> dict[str, Any]:
    evidence_sets = {
        "native_text_evidence": source_evidence.get("native_text", []),
        "native_image_evidence": source_evidence.get("native_images", []),
        "native_vector_evidence": source_evidence.get("native_vectors", []),
    }
    page_visual_bbox = source_evidence.get("page_visual_bbox")
    if _valid_bbox(page_visual_bbox):
        evidence_sets["page_raster_visual_evidence"] = [
            {"evidence_id": "page-visual", "bbox_pdf_pt": page_visual_bbox}
        ]
    result = {}
    for name, rows in evidence_sets.items():
        boxes = [
            row["bbox_pdf_pt"] for row in rows if _valid_bbox(row.get("bbox_pdf_pt"))
        ]
        canonical = [_union_coverage(box, canonical_boxes) for box in boxes]
        fusion = [_union_coverage(box, fusion_boxes) for box in boxes]
        result[name] = {
            "evidence_count": len(boxes),
            "canonical_only": _coverage_summary(canonical),
            "fusion": _coverage_summary(fusion),
        }
    return result


def _union_coverage(target: Sequence[float], boxes: Sequence[Sequence[float]]) -> float:
    if not _valid_bbox(target):
        return 0.0
    clipped = [_bbox_intersection(target, box) for box in boxes if _valid_bbox(box)]
    clipped = [box for box in clipped if box is not None]
    if not clipped:
        return 0.0
    xs = sorted({float(box[0]) for box in clipped} | {float(box[2]) for box in clipped})
    area = 0.0
    for left, right in pairwise(xs):
        if right <= left:
            continue
        intervals = sorted(
            (float(box[1]), float(box[3]))
            for box in clipped
            if box[0] < right and box[2] > left
        )
        covered_y = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered_y += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered_y += end - start
        area += (right - left) * covered_y
    return min(1.0, area / _bbox_area(target))


def _axis_union_coverage(
    target: Sequence[float], boxes: Sequence[Sequence[float]], *, axis: str
) -> float:
    """Return union coverage of a target's x or y projection."""
    if not _valid_bbox(target) or axis not in {"x", "y"}:
        return 0.0
    start_index, end_index = (0, 2) if axis == "x" else (1, 3)
    target_start = float(target[start_index])
    target_end = float(target[end_index])
    intervals = []
    for box in boxes:
        intersection = _bbox_intersection(target, box) if _valid_bbox(box) else None
        if intersection is not None:
            intervals.append(
                (
                    float(intersection[start_index]),
                    float(intersection[end_index]),
                )
            )
    if not intervals:
        return 0.0
    ordered_intervals = sorted(intervals)
    covered = 0.0
    current_start, current_end = ordered_intervals[0]
    for next_start, next_end in ordered_intervals[1:]:
        if next_start > current_end:
            covered += current_end - current_start
            current_start, current_end = next_start, next_end
        else:
            current_end = max(current_end, next_end)
    covered += current_end - current_start
    return min(1.0, covered / max(1e-9, target_end - target_start))


def _group_near_identical(
    rows: Iterable[dict[str, Any]], threshold: float
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in sorted(
        (copy.deepcopy(row) for row in rows if _valid_bbox(row.get("bbox_pdf_pt"))),
        key=lambda value: (
            value["bbox_pdf_pt"][1],
            value["bbox_pdf_pt"][0],
            str(value.get("evidence_id", "")),
        ),
    ):
        group = next(
            (
                group
                for group in groups
                if _bbox_iou(group[0]["bbox_pdf_pt"], row["bbox_pdf_pt"]) >= threshold
            ),
            None,
        )
        if group is None:
            groups.append([row])
        else:
            group.append(row)
    return groups


def _group_images(
    rows: Iterable[dict[str, Any]],
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    ordered = sorted(
        (copy.deepcopy(row) for row in rows if _valid_bbox(row.get("bbox_pdf_pt"))),
        key=lambda row: (
            row["bbox_pdf_pt"][1],
            row["bbox_pdf_pt"][0],
            str(row.get("evidence_id", "")),
        ),
    )
    for row in ordered:
        group = next(
            (
                group
                for group in groups
                if _bbox_iou(group[0]["bbox_pdf_pt"], row["bbox_pdf_pt"])
                >= iou_threshold
                or _bbox_containment(group[0]["bbox_pdf_pt"], row["bbox_pdf_pt"])
                >= containment_threshold
            ),
            None,
        )
        if group is None:
            groups.append([row])
        else:
            group.append(row)
    return groups


def _normalized_bbox(bbox: Sequence[float], width: float, height: float) -> list[float]:
    values = [float(value) for value in bbox]
    return [
        round(max(0.0, min(width, values[0])), 6),
        round(max(0.0, min(height, values[1])), 6),
        round(max(0.0, min(width, values[2])), 6),
        round(max(0.0, min(height, values[3])), 6),
    ]


def _clip_source_bbox(
    bbox: Sequence[float], width: float, height: float
) -> list[float] | None:
    clipped = _normalized_bbox(bbox, width, height)
    return clipped if _valid_bbox(clipped) else None


def _bbox_union(boxes: Sequence[Sequence[float]]) -> list[float]:
    return [
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    ]


def _bbox_intersection(
    first: Sequence[float], second: Sequence[float]
) -> list[float] | None:
    box = [
        max(float(first[0]), float(second[0])),
        max(float(first[1]), float(second[1])),
        min(float(first[2]), float(second[2])),
        min(float(first[3]), float(second[3])),
    ]
    return box if _valid_bbox(box) else None


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = _bbox_intersection(first, second)
    if intersection is None:
        return 0.0
    area = _bbox_area(intersection)
    return area / (_bbox_area(first) + _bbox_area(second) - area)


def _bbox_containment(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = _bbox_intersection(first, second)
    if intersection is None:
        return 0.0
    return _bbox_area(intersection) / min(_bbox_area(first), _bbox_area(second))


def _horizontal_coverage(first: Sequence[float], second: Sequence[float]) -> float:
    overlap = max(
        0.0,
        min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])),
    )
    return overlap / max(1e-9, float(first[2]) - float(first[0]))


def _bbox_area(box: Sequence[float]) -> float:
    return (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in value
        )
        and float(value[0]) < float(value[2])
        and float(value[1]) < float(value[3])
    )


def _candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    bbox = row["bbox_pdf_pt"]
    return (
        float(bbox[1]),
        float(bbox[0]),
        float(bbox[3]),
        float(bbox[2]),
        str(row["candidate_id"]),
    )


def _page_order(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows, key=lambda row: (str(row["document_id"]), int(row["page_index"]))
    )


def _page_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["document_id"]), int(row["page_index"])


def _source_value(row: dict[str, Any], field: str) -> str:
    if field in {"risk", "reading_order_risk"}:
        direct = row.get("reading_order_risk")
        if isinstance(direct, dict):
            return str(direct.get("risk", "UNKNOWN"))
        if direct is not None:
            return str(direct)
        return str(row.get("reading_order", {}).get("risk", "UNKNOWN"))
    return str(row.get(field, "UNKNOWN"))


def _validate_page_identity(*rows: dict[str, Any]) -> None:
    identities = {(str(row["document_id"]), int(row["page_index"])) for row in rows}
    if len(identities) != 1:
        raise ValueError("Fusion inputs must describe the same page")


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _percentile_float(values: Sequence[float], fraction: float) -> float:
    return _percentile([float(value) for value in values], fraction)


def _stable_digest(*parts: Any) -> str:
    return hashlib.sha256(
        "\0".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
