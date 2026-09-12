from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

PAGE_REGION_IR_SCHEMA = "bemarkdown-page-region-ir-v0"
REGION_TAXONOMY = "bemarkdown-region-v0"
READING_ORDER_METHOD = "region-reading-order-v0"
READING_ORDER_V1_METHOD = "region-reading-order-v1"
NAIVE_READING_ORDER_METHOD = "naive-yx"

MODEL_LABELS = (
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart",
    "formula_number",
    "aside_text",
    "reference_content",
)


def _mapping(
    semantic_type: str,
    semantic_subtype: str | None,
    routing_intent: str,
    rationale: str,
) -> dict[str, str | None]:
    return {
        "semantic_type": semantic_type,
        "semantic_subtype": semantic_subtype,
        "routing_intent": routing_intent,
        "rationale": rationale,
    }


LAYOUT_LABEL_MAPPING = {
    "paragraph_title": _mapping("TITLE", "PARAGRAPH_TITLE", "TEXT_CONTENT", "Section or paragraph heading."),
    "image": _mapping("IMAGE", "FIGURE", "VISUAL_ASSET", "Generic visual asset candidate."),
    "text": _mapping("TEXT", "BODY", "TEXT_CONTENT", "Body text region."),
    "number": _mapping("PAGE_NUMBER", None, "METADATA", "Page-number-like region."),
    "abstract": _mapping("TEXT", "ABSTRACT", "TEXT_CONTENT", "Abstract text remains textual content."),
    "content": _mapping("TEXT", "CONTENTS", "TEXT_CONTENT", "Table-of-contents content is textual."),
    "figure_title": _mapping("CAPTION", "FIGURE", "TEXT_CONTENT", "Figure caption or title."),
    "formula": _mapping("FORMULA", None, "FORMULA_CONTENT", "Formula location for a future FormulaNet adapter."),
    "table": _mapping("TABLE", None, "TABLE_CONTENT", "Table location for a future table adapter."),
    "reference": _mapping("TITLE", "REFERENCE_HEADING", "TEXT_CONTENT", "Reference-section heading."),
    "doc_title": _mapping("TITLE", "DOCUMENT_TITLE", "TEXT_CONTENT", "Document-level title."),
    "footnote": _mapping("TEXT", "FOOTNOTE", "TEXT_CONTENT", "Footnote remains textual content."),
    "header": _mapping("HEADER_FOOTER", "HEADER", "METADATA", "Repeated page header."),
    "algorithm": _mapping("OTHER", "ALGORITHM", "REVIEWABLE_CONTENT", "Algorithm block has no stable v0 content adapter."),
    "footer": _mapping("HEADER_FOOTER", "FOOTER", "METADATA", "Repeated page footer."),
    "seal": _mapping("IMAGE", "SEAL", "VISUAL_ASSET", "Seal is preserved as a visual asset candidate."),
    "chart": _mapping("IMAGE", "CHART", "VISUAL_ASSET", "Chart is a visual asset candidate in v0."),
    "formula_number": _mapping("FORMULA", "NUMBER", "FORMULA_CONTENT", "Formula number stays attached to formula routing."),
    "aside_text": _mapping("TEXT", "ASIDE", "TEXT_CONTENT", "Sidebar or aside text."),
    "reference_content": _mapping("TEXT", "REFERENCE", "TEXT_CONTENT", "Reference-list content."),
}


@dataclass(frozen=True)
class PageRenderTransform:
    contract: str
    page_width_pt: float
    page_height_pt: float
    render_width: int
    render_height: int
    dpi: int
    scale_x: float
    scale_y: float
    rotation: int
    coordinate_system: str = "pdf-points-top-left-rotation-normalized"
    color_space: str = "RGB"
    alpha: bool = False

    @classmethod
    def create(
        cls,
        *,
        page_width_pt: float,
        page_height_pt: float,
        render_width: int,
        render_height: int,
        dpi: int,
        rotation: int,
    ) -> PageRenderTransform:
        if page_width_pt <= 0 or page_height_pt <= 0:
            raise ValueError("PDF page dimensions must be positive")
        if render_width <= 0 or render_height <= 0 or dpi <= 0:
            raise ValueError("Render dimensions and DPI must be positive")
        return cls(
            contract="pdf-page-render-v0",
            page_width_pt=float(page_width_pt),
            page_height_pt=float(page_height_pt),
            render_width=int(render_width),
            render_height=int(render_height),
            dpi=int(dpi),
            scale_x=float(render_width) / float(page_width_pt),
            scale_y=float(render_height) / float(page_height_pt),
            rotation=int(rotation) % 360,
        )

    def render_px_to_pdf_pt(self, bbox: Sequence[float]) -> list[float]:
        _require_bbox(bbox)
        rounded = _round_bbox(
            [
                float(bbox[0]) / self.scale_x,
                float(bbox[1]) / self.scale_y,
                float(bbox[2]) / self.scale_x,
                float(bbox[3]) / self.scale_y,
            ]
        )
        # Rounding a box that lands exactly on the render edge can move the
        # canonical PDF coordinate a few sub-micro-points beyond the page.
        # Clamp after rounding so the stored RegionIR remains exactly valid;
        # raw render-space evidence is preserved separately.
        return [
            min(max(rounded[0], 0.0), self.page_width_pt),
            min(max(rounded[1], 0.0), self.page_height_pt),
            min(max(rounded[2], 0.0), self.page_width_pt),
            min(max(rounded[3], 0.0), self.page_height_pt),
        ]

    def pdf_pt_to_render_px(self, bbox: Sequence[float]) -> list[float]:
        _require_bbox(bbox)
        return _round_bbox(
            [
                float(bbox[0]) * self.scale_x,
                float(bbox[1]) * self.scale_y,
                float(bbox[2]) * self.scale_x,
                float(bbox[3]) * self.scale_y,
            ]
        )

    def pdf_pt_to_normalized(self, bbox: Sequence[float]) -> list[float]:
        _require_bbox(bbox)
        return _round_bbox(
            [
                float(bbox[0]) / self.page_width_pt,
                float(bbox[1]) / self.page_height_pt,
                float(bbox[2]) / self.page_width_pt,
                float(bbox[3]) / self.page_height_pt,
            ],
            digits=8,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_page_region_ir(
    *,
    document_id: str,
    page_index: int,
    page_route: str,
    transform: PageRenderTransform,
    raw_detections: list[dict[str, Any]],
    score_threshold: float,
    structural_geometry: dict[str, list[list[float]]] | None = None,
    assign_reading_order: bool = True,
) -> dict[str, Any]:
    structural_geometry = structural_geometry or {}
    candidates: list[dict[str, Any]] = []
    for raw in raw_detections:
        if float(raw["raw_score"]) < score_threshold:
            continue
        bbox_render = _clamp_bbox(
            raw["raw_bbox_render_px"],
            width=transform.render_width,
            height=transform.render_height,
        )
        if _bbox_area(bbox_render) <= 0:
            continue
        bbox_pdf = transform.render_px_to_pdf_pt(bbox_render)
        mapping = LAYOUT_LABEL_MAPPING.get(
            str(raw["raw_label"]),
            _mapping("UNKNOWN", None, "REVIEW_REQUIRED", "Raw model label is not mapped in bemarkdown-region-v0."),
        )
        candidates.append(
            {
                "semantic_type": mapping["semantic_type"],
                "semantic_subtype": mapping["semantic_subtype"],
                "bbox_pdf_pt": bbox_pdf,
                "bbox_normalized": transform.pdf_pt_to_normalized(bbox_pdf),
                "bbox_render_px": bbox_render,
                "score": round(float(raw["raw_score"]), 8),
                "raw_model_label": str(raw["raw_label"]),
                "raw_detection_id": str(raw["raw_detection_id"]),
                "reading_order_index": None,
                "reading_order_method": None,
                "naive_reading_order_index": None,
                "parent_region_id": None,
                "page_route": page_route,
                "routing_intent": mapping["routing_intent"],
                "normalization_reason": "DIRECT_LABEL_MAPPING",
                "alignment_diagnostics": _alignment_diagnostics(
                    bbox_pdf, structural_geometry
                ),
                "provenance": {
                    "model_identity": raw["model_identity"],
                    "page_render_identity": raw["page_render_identity"],
                    "region_taxonomy": REGION_TAXONOMY,
                    "score_threshold": score_threshold,
                },
            }
        )
    candidates.sort(key=_canonical_sort_key)
    retained, suppressed = _deduplicate_same_label(candidates)
    for region in retained:
        region["region_id"] = _region_id(document_id, page_index, region)
    _assign_nested_parents(retained)
    page = {
        "schema": PAGE_REGION_IR_SCHEMA,
        "document_id": document_id,
        "page_index": int(page_index),
        "page_number": int(page_index) + 1,
        "page_route": page_route,
        "page_geometry": {
            "width_pt": transform.page_width_pt,
            "height_pt": transform.page_height_pt,
            "rotation": transform.rotation,
        },
        "render_transform": transform.to_dict(),
        "score_threshold": score_threshold,
        "raw_detection_count": len(raw_detections),
        "canonical_regions": retained,
        "suppressed_detections": suppressed,
    }
    if assign_reading_order:
        apply_reading_order_to_page(page)
    return page


def apply_reading_order_to_page(page: dict[str, Any]) -> dict[str, Any]:
    regions = page["canonical_regions"]
    naive = sorted(
        regions,
        key=lambda row: (_bbox(row)[1], _bbox(row)[0], row["region_id"]),
    )
    naive_rank = {row["region_id"]: index for index, row in enumerate(naive)}
    geometry = page["page_geometry"]
    ordered = order_regions(
        regions,
        page_width_pt=float(geometry["width_pt"]),
        page_height_pt=float(geometry["height_pt"]),
    )
    for region in ordered:
        region["naive_reading_order_index"] = naive_rank[region["region_id"]]
    page["canonical_regions"] = ordered
    return page


def apply_reading_order_v1_to_page(page: dict[str, Any]) -> dict[str, Any]:
    """Add v1 order without discarding the frozen naive and v0 evidence."""
    geometry = page["page_geometry"]
    for region in page["canonical_regions"]:
        region["reading_order_v0_index"] = region["reading_order_index"]
    ordered, diagnostics = order_regions_v1(
        page["canonical_regions"],
        page_width_pt=float(geometry["width_pt"]),
        page_height_pt=float(geometry["height_pt"]),
    )
    for index, region in enumerate(ordered):
        region["reading_order_v1_index"] = index
        region["reading_order_v1_status"] = diagnostics["status"]
    page["canonical_regions"] = ordered
    page["reading_order_v1"] = diagnostics
    return page


def order_regions_v1(
    regions: Iterable[dict[str, Any]],
    *,
    page_width_pt: float,
    page_height_pt: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Order regions through anchors, column groups and a precedence graph.

    The graph intentionally stays small and geometry-only. If constraints form
    a cycle, Kahn's algorithm emits the acyclic prefix and appends the
    conflicted component in stable naive y/x order.
    """
    rows = [dict(row) for row in regions]
    if not rows:
        return [], {
            "method": READING_ORDER_V1_METHOD,
            "status": "OK",
            "edge_count": 0,
            "cycle_count": 0,
            "fallback": None,
        }
    by_id = {str(row["region_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("RegionReadingOrder v1 requires unique region IDs")
    edges = _reading_order_v1_edges(
        rows, page_width_pt=page_width_pt, page_height_pt=page_height_pt
    )
    indegree = {region_id: 0 for region_id in by_id}
    for source, targets in edges.items():
        for target in targets:
            if source != target:
                indegree[target] += 1
    ready: list[tuple[tuple[Any, ...], str]] = []
    for region_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (_v1_stable_key(by_id[region_id]), region_id))
    emitted: list[str] = []
    while ready:
        _, region_id = heapq.heappop(ready)
        emitted.append(region_id)
        for target in sorted(edges.get(region_id, set())):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, (_v1_stable_key(by_id[target]), target))
    remaining = [region_id for region_id in by_id if region_id not in set(emitted)]
    status = "OK"
    fallback = None
    cycle_count = 0
    if remaining:
        status = "READING_ORDER_CONSTRAINT_CYCLE"
        fallback = "NAIVE_YX_WITHIN_CONFLICTED_COMPONENT"
        cycle_count = 1
        emitted.extend(sorted(remaining, key=lambda value: _v1_stable_key(by_id[value])))
    ordered = [by_id[region_id] for region_id in emitted]
    return ordered, {
        "method": READING_ORDER_V1_METHOD,
        "status": status,
        "edge_count": sum(len(targets) for targets in edges.values()),
        "cycle_count": cycle_count,
        "fallback": fallback,
    }


def _reading_order_v1_edges(
    rows: list[dict[str, Any]],
    *,
    page_width_pt: float,
    page_height_pt: float,
) -> dict[str, set[str]]:
    del page_height_pt
    edges = {str(row["region_id"]): set() for row in rows}
    anchors = sorted(
        [row for row in rows if _bbox_width(row) / page_width_pt >= 0.68],
        key=_v1_stable_key,
    )
    ordinary = [row for row in rows if row not in anchors]
    for first, second in pairwise(anchors):
        edges[str(first["region_id"])].add(str(second["region_id"]))
    for anchor in anchors:
        anchor_id = str(anchor["region_id"])
        anchor_center = _center_y(anchor)
        for row in ordinary:
            row_id = str(row["region_id"])
            if _bbox(row)[3] <= anchor_center:
                edges[row_id].add(anchor_id)
            elif _bbox(row)[1] >= anchor_center:
                edges[anchor_id].add(row_id)

    zones: dict[int, list[dict[str, Any]]] = {}
    anchor_centers = [_center_y(row) for row in anchors]
    for row in ordinary:
        zone = sum(center < _center_y(row) for center in anchor_centers)
        zones.setdefault(zone, []).append(row)
    for zone_rows in zones.values():
        _add_zone_edges(edges, zone_rows, page_width_pt)

    # Explicit containment and caption relations are weak but deterministic.
    by_id = {str(row["region_id"]): row for row in rows}
    for row in rows:
        parent_id = row.get("parent_region_id")
        if parent_id in by_id and parent_id != row["region_id"]:
            parent = by_id[str(parent_id)]
            if row.get("semantic_type") == "CAPTION" or _center_y(parent) <= _center_y(row):
                edges[str(parent_id)].add(str(row["region_id"]))
    return edges


def _add_zone_edges(
    edges: dict[str, set[str]], rows: list[dict[str, Any]], page_width: float
) -> None:
    if len(rows) < 2:
        return
    left = [row for row in rows if _center_x(row) < page_width * 0.48]
    right = [row for row in rows if _center_x(row) > page_width * 0.52]
    separated = (
        bool(left)
        and bool(right)
        and max(_bbox(row)[2] for row in left) < min(_bbox(row)[0] for row in right)
    )
    groups: list[list[dict[str, Any]]]
    if separated:
        middle = [row for row in rows if row not in left and row not in right]
        groups = [left, right, middle]
        for first in left:
            for second in right:
                edges[str(first["region_id"])].add(str(second["region_id"]))
    else:
        groups = [rows]
    for group in groups:
        ordered = sorted(group, key=_v1_stable_key)
        for first, second in pairwise(ordered):
            edges[str(first["region_id"])].add(str(second["region_id"]))


def _v1_stable_key(row: dict[str, Any]) -> tuple[Any, ...]:
    semantic_priority = {
        "HEADER_FOOTER": 0,
        "TITLE": 1,
        "TEXT": 2,
        "FORMULA": 3,
        "TABLE": 4,
        "IMAGE": 5,
        "CAPTION": 6,
        "PAGE_NUMBER": 8,
    }
    return (
        round(_bbox(row)[1], 6),
        round(_bbox(row)[0], 6),
        semantic_priority.get(str(row.get("semantic_type")), 7),
        str(row.get("region_id", "")),
    )


def order_regions(
    regions: Iterable[dict[str, Any]],
    *,
    page_width_pt: float,
    page_height_pt: float,
) -> list[dict[str, Any]]:
    del page_height_pt
    rows = [dict(row) for row in regions]
    full_width = [row for row in rows if _bbox_width(row) / page_width_pt >= 0.72]
    narrow = [row for row in rows if row not in full_width]
    anchors = sorted(full_width, key=lambda row: (_bbox(row)[1], _bbox(row)[0], row.get("region_id", "")))
    ordered: list[dict[str, Any]] = []
    lower = float("-inf")
    for anchor in anchors:
        anchor_top = _bbox(anchor)[1]
        zone = [row for row in narrow if lower <= _center_y(row) < anchor_top]
        ordered.extend(_order_vertical_zone(zone, page_width_pt))
        ordered.append(anchor)
        lower = max(lower, _bbox(anchor)[3])
    ordered.extend(_order_vertical_zone([row for row in narrow if _center_y(row) >= lower], page_width_pt))
    missing = [row for row in rows if row not in ordered]
    ordered.extend(sorted(missing, key=lambda row: (_bbox(row)[1], _bbox(row)[0], row.get("region_id", ""))))
    for index, row in enumerate(ordered):
        row["reading_order_index"] = index
        row["reading_order_method"] = READING_ORDER_METHOD
    return ordered


def validate_page_region_ir(page: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    width = float(page["page_geometry"]["width_pt"])
    height = float(page["page_geometry"]["height_pt"])
    seen: set[str] = set()
    for region in page.get("canonical_regions", []):
        region_id = str(region.get("region_id", ""))
        if region_id in seen:
            issues.append({"issue_code": "DUPLICATE_REGION_ID", "region_id": region_id})
        seen.add(region_id)
        bbox = region.get("bbox_pdf_pt")
        if not _is_bbox(bbox) or _bbox_area(bbox) <= 0:
            issues.append({"issue_code": "ZERO_OR_INVALID_BBOX", "region_id": region_id})
            continue
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
            issues.append({"issue_code": "BBOX_OUT_OF_PAGE", "region_id": region_id})
        if region.get("semantic_type") == "UNKNOWN":
            issues.append({"issue_code": "UNKNOWN_LABEL", "region_id": region_id})
    order = [row.get("reading_order_index") for row in page.get("canonical_regions", [])]
    if sorted(order) != list(range(len(order))):
        issues.append({"issue_code": "NON_DETERMINISTIC_ORDER", "region_id": None})
    if len(page.get("canonical_regions", [])) > 300:
        issues.append({"issue_code": "REGION_COUNT_EXPLOSION", "region_id": None})
    return issues


def raw_detection_id(
    *,
    document_id: str,
    page_index: int,
    raw_label: str,
    score: float,
    bbox_render_px: Sequence[float],
    model_identity: str,
    page_render_identity: str,
) -> str:
    payload = {
        "document_id": document_id,
        "page_index": page_index,
        "raw_label": raw_label,
        "score": round(float(score), 8),
        "bbox_render_px": _round_bbox(bbox_render_px),
        "model_identity": model_identity,
        "page_render_identity": page_render_identity,
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()[:16]
    return f"p{page_index + 1:06d}_d{digest}"


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = _intersection_area(first, second)
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _order_vertical_zone(rows: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    left = [row for row in rows if _center_x(row) < page_width * 0.48]
    right = [row for row in rows if _center_x(row) > page_width * 0.52]
    separated = (
        len(left) >= 2
        and len(right) >= 2
        and max(_bbox(row)[2] for row in left) < min(_bbox(row)[0] for row in right)
    )
    key = lambda row: (_bbox(row)[1], _bbox(row)[0], row.get("region_id", ""))
    if separated:
        middle = [row for row in rows if row not in left and row not in right]
        return sorted(left, key=key) + sorted(right, key=key) + sorted(middle, key=key)
    return sorted(rows, key=key)


def _deduplicate_same_label(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: (-row["score"],) + _canonical_sort_key(row)):
        duplicate = next(
            (
                row
                for row in retained
                if row["raw_model_label"] == candidate["raw_model_label"]
                and bbox_iou(row["bbox_pdf_pt"], candidate["bbox_pdf_pt"]) >= 0.95
            ),
            None,
        )
        if duplicate is None:
            retained.append(candidate)
        else:
            suppressed.append(
                {
                    "raw_detection_id": candidate["raw_detection_id"],
                    "retained_raw_detection_id": duplicate["raw_detection_id"],
                    "normalization_reason": "SAME_RAW_LABEL_NEAR_DUPLICATE",
                }
            )
    retained.sort(key=_canonical_sort_key)
    suppressed.sort(key=lambda row: row["raw_detection_id"])
    return retained, suppressed


def _assign_nested_parents(regions: list[dict[str, Any]]) -> None:
    for child in regions:
        containers = [
            parent
            for parent in regions
            if parent is not child
            and parent["semantic_type"] != child["semantic_type"]
            and _containment_ratio(child["bbox_pdf_pt"], parent["bbox_pdf_pt"]) >= 0.95
            and _bbox_area(parent["bbox_pdf_pt"]) > _bbox_area(child["bbox_pdf_pt"])
        ]
        if containers:
            parent = min(containers, key=lambda row: _bbox_area(row["bbox_pdf_pt"]))
            child["parent_region_id"] = parent["region_id"]


def _alignment_diagnostics(
    region_bbox: Sequence[float],
    structural_geometry: dict[str, list[list[float]]],
) -> dict[str, float]:
    return {
        "native_text_overlap_ratio": _overlap_ratio(
            region_bbox, structural_geometry.get("native_text_bboxes_pdf_pt", [])
        ),
        "native_image_overlap_ratio": _overlap_ratio(
            region_bbox, structural_geometry.get("native_image_bboxes_pdf_pt", [])
        ),
        "native_vector_overlap_ratio": _overlap_ratio(
            region_bbox, structural_geometry.get("native_vector_bboxes_pdf_pt", [])
        ),
    }


def _overlap_ratio(region_bbox: Sequence[float], boxes: list[list[float]]) -> float:
    region_area = _bbox_area(region_bbox)
    if region_area <= 0:
        return 0.0
    intersections = [_intersection_box(region_bbox, bbox) for bbox in boxes]
    valid = [bbox for bbox in intersections if bbox is not None]
    return round(min(1.0, _rect_union_area(valid) / region_area), 8)


def _region_id(document_id: str, page_index: int, region: dict[str, Any]) -> str:
    payload = {
        "document_id": document_id,
        "page_index": page_index,
        "semantic_type": region["semantic_type"],
        "semantic_subtype": region["semantic_subtype"],
        "bbox_pdf_pt": region["bbox_pdf_pt"],
        "score": region["score"],
        "raw_model_label": region["raw_model_label"],
        "raw_detection_id": region["raw_detection_id"],
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()[:12]
    return f"p{page_index + 1:06d}_r{digest}"


def _canonical_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    bbox = row["bbox_pdf_pt"]
    return (
        round(float(bbox[1]), 4),
        round(float(bbox[0]), 4),
        str(row["semantic_type"]),
        str(row["raw_model_label"]),
        -round(float(row["score"]), 8),
        str(row["raw_detection_id"]),
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _bbox(row: dict[str, Any]) -> list[float]:
    return row["bbox_pdf_pt"]


def _bbox_width(row: dict[str, Any]) -> float:
    bbox = _bbox(row)
    return float(bbox[2]) - float(bbox[0])


def _center_x(row: dict[str, Any]) -> float:
    bbox = _bbox(row)
    return (float(bbox[0]) + float(bbox[2])) / 2


def _center_y(row: dict[str, Any]) -> float:
    bbox = _bbox(row)
    return (float(bbox[1]) + float(bbox[3])) / 2


def _clamp_bbox(bbox: Sequence[float], *, width: float, height: float) -> list[float]:
    _require_bbox(bbox)
    return _round_bbox(
        [
            min(max(float(bbox[0]), 0.0), width),
            min(max(float(bbox[1]), 0.0), height),
            min(max(float(bbox[2]), 0.0), width),
            min(max(float(bbox[3]), 0.0), height),
        ]
    )


def _require_bbox(bbox: Sequence[float]) -> None:
    if not _is_bbox(bbox):
        raise ValueError("bbox must be a finite [x0, y0, x1, y1] sequence")


def _is_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in values)


def _round_bbox(bbox: Sequence[float], *, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in bbox]


def _bbox_area(bbox: Sequence[float]) -> float:
    if not _is_bbox(bbox):
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def _intersection_box(first: Sequence[float], second: Sequence[float]) -> list[float] | None:
    result = [
        max(float(first[0]), float(second[0])),
        max(float(first[1]), float(second[1])),
        min(float(first[2]), float(second[2])),
        min(float(first[3]), float(second[3])),
    ]
    return result if _bbox_area(result) > 0 else None


def _intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = _intersection_box(first, second)
    return _bbox_area(intersection) if intersection is not None else 0.0


def _containment_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    area = _bbox_area(inner)
    return _intersection_area(inner, outer) / area if area > 0 else 0.0


def _rect_union_area(rectangles: list[list[float]]) -> float:
    if not rectangles:
        return 0.0
    x_values = sorted({value for rect in rectangles for value in (rect[0], rect[2])})
    area = 0.0
    for left, right in pairwise(x_values):
        if right <= left:
            continue
        intervals = sorted(
            (rect[1], rect[3])
            for rect in rectangles
            if rect[0] < right and rect[2] > left
        )
        if not intervals:
            continue
        start, end = intervals[0]
        covered = 0.0
        for local_start, local_end in intervals[1:]:
            if local_start > end:
                covered += max(0.0, end - start)
                start, end = local_start, local_end
            else:
                end = max(end, local_end)
        covered += max(0.0, end - start)
        area += (right - left) * covered
    return area
