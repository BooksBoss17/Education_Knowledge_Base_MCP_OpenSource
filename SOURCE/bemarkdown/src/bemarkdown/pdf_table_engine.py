from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import re
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any

TABLE_IR_SCHEMA = "bemarkdown-table-ir-v0"
TABLE_CELL_IR_SCHEMA = "bemarkdown-table-cell-ir-v0"
TABLE_MATCHER_SCHEMA = "table-text-cell-matcher-v0"
TABLE_FUSION_SCHEMA = "table-structure-cell-fusion-v0"
TABLE_PLAUSIBILITY_SCHEMA = "table-structure-plausibility-v0"
TABLE_SERIALIZER_SCHEMA = "bemarkdown-table-serializer-v0"
TABLE_QUALITY_POLICY_SCHEMA = "bemarkdown-table-quality-policy-v0"

TABLE_STATUSES = {
    "STRUCTURED_TABLE",
    "PARTIAL_TABLE",
    "TABLE_REVIEW_REQUIRED",
    "NOT_A_TABLE_REVIEW",
}


def normalize_table_classification(
    label_names: list[str], scores: list[float], *, threshold: float = 0.6
) -> dict[str, Any]:
    pairs = [(str(name), float(score)) for name, score in zip(label_names, scores, strict=False)]
    top_name, top_score = max(pairs, default=("", 0.0), key=lambda value: value[1])
    raw_branch = {
        "wired_table": "WIRED",
        "wireless_table": "WIRELESS",
    }.get(top_name, "UNKNOWN")
    return {
        "normalized_label": raw_branch if top_score >= threshold else "UNKNOWN",
        "selected_branch": raw_branch if raw_branch != "UNKNOWN" else "WIRED",
        "score": top_score if pairs else None,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, value: Any, length: int = 24) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _intersection(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(first: list[float], second: list[float]) -> float:
    intersection = _intersection(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union else 0.0


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _contains(bbox: list[float], point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[dict[str, Any]] = []
        self.row = -1
        self._in_row = False
        self._occupied: set[tuple[int, int]] = set()
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.row += 1
            self._in_row = True
            return
        if tag not in {"td", "th"}:
            return
        if not self._in_row or self.row < 0:
            self.errors.append("CELL_OUTSIDE_ROW")
            return
        values = {key.lower(): value for key, value in attrs}
        rowspan = _span(values.get("rowspan"))
        colspan = _span(values.get("colspan"))
        col = 0
        while (self.row, col) in self._occupied:
            col += 1
        self.cells.append(
            {
                "ordinal": len(self.cells),
                "row": self.row,
                "col": col,
                "rowspan": rowspan,
                "colspan": colspan,
                "header": tag == "th",
            }
        )
        if isinstance(rowspan, int) and isinstance(colspan, int):
            for row_offset in range(rowspan):
                for col_offset in range(colspan):
                    self._occupied.add((self.row + row_offset, col + col_offset))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "tr":
            self._in_row = False


def _span(value: str | None) -> int | str:
    if value is None or value == "":
        return 1
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    return parsed if parsed > 0 else "UNKNOWN"


def _tag_balance(markup: str) -> list[str]:
    stack: list[str] = []
    errors: list[str] = []
    for match in re.finditer(r"</?(?:table|tr|td|th)\b[^>]*>", markup, re.IGNORECASE):
        token = match.group(0)
        closing = token.startswith("</")
        tag_match = re.match(r"</?([a-z]+)", token, re.IGNORECASE)
        if tag_match is None:
            continue
        tag = tag_match.group(1).lower()
        if not closing:
            stack.append(tag)
            continue
        if not stack or stack[-1] != tag:
            errors.append(f"UNBALANCED_{tag.upper()}")
            continue
        stack.pop()
    errors.extend(f"UNCLOSED_{tag.upper()}" for tag in reversed(stack))
    return errors


def _location_bbox(location: Any, crop_size: tuple[int, int]) -> list[float] | None:
    if not isinstance(location, (list, tuple)) or len(location) not in {4, 8}:
        return None
    try:
        values = [float(value) for value in location]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if len(values) == 8:
        xs, ys = values[0::2], values[1::2]
        if any(value < 0 or value > 1000 for value in values):
            return None
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        bbox = [
            bbox[0] * crop_size[0] / 1000,
            bbox[1] * crop_size[1] / 1000,
            bbox[2] * crop_size[0] / 1000,
            bbox[3] * crop_size[1] / 1000,
        ]
    else:
        bbox = values
    if not _valid_bbox(bbox):
        return None
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > crop_size[0] or bbox[3] > crop_size[1]:
        return None
    return [round(value, 6) for value in bbox]


def assess_structure(
    tokens: list[str], locations: list[Any], crop_size: tuple[int, int]
) -> dict[str, Any]:
    markup = "".join(str(token) for token in tokens)
    parser = _StructureParser()
    parser.feed(markup)
    balance_errors = _tag_balance(markup)
    reason_codes = [*balance_errors, *parser.errors]
    if "<table" not in markup.lower():
        reason_codes.append("TABLE_TAG_MISSING")
    if not parser.cells:
        reason_codes.append("EMPTY_STRUCTURE")
    bboxes = [_location_bbox(location, crop_size) for location in locations]
    invalid_locations = [index for index, bbox in enumerate(bboxes) if bbox is None]
    if invalid_locations:
        reason_codes.append("INVALID_STRUCTURE_BBOX")
    if len(locations) != len(parser.cells):
        reason_codes.append("STRUCTURE_LOCATION_COUNT_MISMATCH")
    cells = []
    for index, parsed in enumerate(parser.cells):
        cells.append({**parsed, "bbox_crop_px": bboxes[index] if index < len(bboxes) else None})
    rows = max((cell["row"] for cell in cells), default=-1) + 1
    cols = max(
        (
            cell["col"] + (cell["colspan"] if isinstance(cell["colspan"], int) else 1)
            for cell in cells
        ),
        default=0,
    )
    invalid_reasons = {
        "TABLE_TAG_MISSING",
        "EMPTY_STRUCTURE",
        "INVALID_STRUCTURE_BBOX",
        "CELL_OUTSIDE_ROW",
    }
    if any(reason in invalid_reasons or reason.startswith(("UNBALANCED_", "UNCLOSED_")) for reason in reason_codes):
        status = "STRUCTURE_INVALID"
    elif reason_codes:
        status = "STRUCTURE_SUSPICIOUS"
    else:
        status = "STRUCTURE_VALID"
    return {
        "schema": TABLE_PLAUSIBILITY_SCHEMA,
        "status": status,
        "reason_codes": sorted(set(reason_codes)),
        "rows": rows,
        "cols": cols,
        "cells": cells,
        "markup": markup,
        "tag_balance_valid": not balance_errors,
        "invalid_location_indices": invalid_locations,
    }


def _normalize_detector_boxes(
    rows: list[dict[str, Any]], crop_size: tuple[int, int]
) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        bbox = source.get("coordinate", source.get("bbox"))
        if not _valid_bbox(bbox):
            invalid.append({"index": index, "reason": "INVALID_BBOX", "raw": copy.deepcopy(source)})
            continue
        raw_bbox = [float(value) for value in bbox]
        clipped = [
            max(0.0, raw_bbox[0]),
            max(0.0, raw_bbox[1]),
            min(float(crop_size[0]), raw_bbox[2]),
            min(float(crop_size[1]), raw_bbox[3]),
        ]
        if not _valid_bbox(clipped):
            invalid.append({"index": index, "reason": "OUTSIDE_CROP", "raw": copy.deepcopy(source)})
            continue
        valid.append(
            {
                "detector_id": _stable_id(
                    "det-cell-", {"index": index, "bbox": raw_bbox, "score": source.get("score")}
                ),
                "bbox_crop_px": [round(value, 6) for value in clipped],
                "raw_bbox": raw_bbox,
                "score": float(source["score"]) if source.get("score") is not None else None,
                "clipped": clipped != raw_bbox,
                "raw": copy.deepcopy(source),
            }
        )
    ordered = sorted(
        valid,
        key=lambda row: (
            -(row["score"] if row["score"] is not None else -1.0),
            row["bbox_crop_px"],
            row["detector_id"],
        ),
    )
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in ordered:
        match = next(
            (existing for existing in kept if _iou(row["bbox_crop_px"], existing["bbox_crop_px"]) >= 0.9),
            None,
        )
        if match is None:
            kept.append(row)
        else:
            duplicates.append(
                {
                    "suppressed_detector_id": row["detector_id"],
                    "primary_detector_id": match["detector_id"],
                    "reason": "DETERMINISTIC_IOU_DUPLICATE",
                    "raw": row,
                }
            )
    kept.sort(key=lambda row: (row["bbox_crop_px"][1], row["bbox_crop_px"][0], row["detector_id"]))
    return {"valid": kept, "invalid": invalid, "duplicates": duplicates}


def _fuse_structure_cells(
    structure_cells: list[dict[str, Any]], detector_cells: list[dict[str, Any]]
) -> dict[str, Any]:
    used: set[str] = set()
    mapped = []
    unmapped = []
    for source in structure_cells:
        bbox = source.get("bbox_crop_px")
        if not _valid_bbox(bbox):
            unmapped.append(source["ordinal"])
            mapped.append({**source, "detector_cell": None})
            continue
        candidates = []
        for detector in detector_cells:
            if detector["detector_id"] in used:
                continue
            detector_bbox = detector["bbox_crop_px"]
            overlap = _iou(bbox, detector_bbox)
            center_bonus = 1 if _contains(bbox, _center(detector_bbox)) else 0
            candidates.append((center_bonus, overlap, detector["score"] or -1.0, detector["detector_id"], detector))
        best = max(candidates, default=None, key=lambda row: row[:4])
        if best is None or (best[0] == 0 and best[1] < 0.1):
            unmapped.append(source["ordinal"])
            mapped.append({**source, "detector_cell": None})
            continue
        detector = best[4]
        used.add(detector["detector_id"])
        mapped.append({**source, "detector_cell": detector})
    extras = [row for row in detector_cells if row["detector_id"] not in used]
    return {
        "schema": TABLE_FUSION_SCHEMA,
        "structure_expected_cells": len(structure_cells),
        "detected_cells": len(detector_cells),
        "mapped_cells": len(structure_cells) - len(unmapped),
        "unmapped_structure_cells": unmapped,
        "extra_detector_cells": extras,
        "cells": mapped,
    }


def match_ocr_to_cells(
    ocr_boxes: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    *,
    crop_size: tuple[int, int],
) -> dict[str, Any]:
    assignments: dict[str, str] = {}
    orphan_text = []
    diag = math.hypot(*crop_size) or 1.0
    for index, source in enumerate(ocr_boxes):
        row = copy.deepcopy(source)
        row.setdefault(
            "ocr_id",
            _stable_id(
                "ocr-box-",
                {"index": index, "bbox": row.get("bbox"), "text": row.get("text")},
            ),
        )
        bbox = row.get("bbox")
        candidates = []
        if _valid_bbox(bbox):
            center = _center(bbox)
            for cell in cells:
                cell_bbox = cell.get("bbox_crop_px")
                if not _valid_bbox(cell_bbox):
                    continue
                intersection = _intersection(bbox, cell_bbox)
                coverage = intersection / _area(bbox) if _area(bbox) else 0.0
                overlap = _iou(bbox, cell_bbox)
                distance = math.dist(center, _center(cell_bbox)) / diag
                center_contained = _contains(cell_bbox, center)
                acceptable = center_contained or coverage >= 0.5 or overlap > 0 or distance <= 0.08
                if acceptable:
                    candidates.append(
                        (
                            int(center_contained),
                            coverage,
                            overlap,
                            -distance,
                            str(cell["cell_id"]),
                        )
                    )
        if candidates:
            assignments[row["ocr_id"]] = max(candidates)[4]
        else:
            orphan_text.append(row)
    accounting = {
        "all": len(ocr_boxes),
        "assigned": len(assignments),
        "orphan": len(orphan_text),
    }
    if accounting["all"] != accounting["assigned"] + accounting["orphan"]:
        raise ValueError("OCR accounting invariant failed")
    return {
        "schema": TABLE_MATCHER_SCHEMA,
        "assignments": assignments,
        "orphan_text": orphan_text,
        "accounting": accounting,
    }


def _crop_to_pdf_bbox(
    crop_bbox: list[float] | None,
    source_bbox: list[float],
    crop_size: tuple[int, int],
) -> list[float] | None:
    if not _valid_bbox(crop_bbox) or not _valid_bbox(source_bbox):
        return None
    width = source_bbox[2] - source_bbox[0]
    height = source_bbox[3] - source_bbox[1]
    return [
        round(source_bbox[0] + crop_bbox[0] / crop_size[0] * width, 6),
        round(source_bbox[1] + crop_bbox[1] / crop_size[1] * height, 6),
        round(source_bbox[0] + crop_bbox[2] / crop_size[0] * width, 6),
        round(source_bbox[1] + crop_bbox[3] / crop_size[1] * height, 6),
    ]


_FORMULA_LIKE_RE = re.compile(r"(?:[=<>±×÷√∑∫]|\d\s*[+\-*/^]\s*\d|[A-Za-z]\s*=)")


def _content_type(text: str) -> str:
    return "FORMULA_LIKE_CELL" if _FORMULA_LIKE_RE.search(text) else "TEXT"


def _retain_formula_region_reviews(cells, ownership):
    """Attach layout evidence without asserting that OCR recovered the formula."""
    unassigned = []
    for region in ownership.get("children", []):
        box = region.get("bbox_pdf_pt")
        owners = []
        if _valid_bbox(box):
            area = (box[2] - box[0]) * (box[3] - box[1])
            for cell in cells:
                target = cell["bbox_pdf_pt"]
                intersection = max(0, min(box[2], target[2]) - max(box[0], target[0])) * max(
                    0, min(box[3], target[3]) - max(box[1], target[1])
                )
                if intersection > area * 0.5:
                    owners.append(cell)
        if len(owners) != 1:
            unassigned.append(copy.deepcopy(region))
            continue
        cell = owners[0]
        cell.setdefault("formula_region_evidence", []).append(copy.deepcopy(region))
        cell["content_type"] = "FORMULA_LIKE_CELL"
        cell["review_state"] = "REVIEW"
    return unassigned


def _quality(
    evidence: dict[str, Any],
    structure: dict[str, Any],
    detectors: dict[str, Any],
    fusion: dict[str, Any],
    matcher: dict[str, Any],
) -> dict[str, Any]:
    reasons = set(structure["reason_codes"])
    truth = evidence.get("source_truth") or {}
    if truth.get("classification") == "NOT_TABLE":
        return {
            "status": "NOT_A_TABLE_REVIEW",
            "reason_codes": ["HISTORICAL_SOURCE_ONLY_NOT_TABLE"],
        }
    if structure["status"] == "STRUCTURE_INVALID" or not fusion["structure_expected_cells"]:
        reasons.add("STRUCTURE_NOT_USABLE")
        return {"status": "TABLE_REVIEW_REQUIRED", "reason_codes": sorted(reasons)}
    if detectors["invalid"]:
        reasons.add("INVALID_DETECTOR_CELL_RETAINED_IN_DIAGNOSTICS")
    if detectors["duplicates"]:
        reasons.add("DUPLICATE_DETECTOR_CELL_RECORDED")
    if fusion["unmapped_structure_cells"]:
        reasons.add("UNMAPPED_STRUCTURE_CELL")
    if fusion["extra_detector_cells"]:
        reasons.add("EXTRA_DETECTOR_CELL")
    if matcher["orphan_text"]:
        reasons.add("ORPHAN_OCR_TEXT")
    if any(
        cell[span] == "UNKNOWN"
        for cell in structure["cells"]
        for span in ("rowspan", "colspan")
    ):
        reasons.add("SPAN_UNKNOWN")
    complete = (
        structure["status"] == "STRUCTURE_VALID"
        and not detectors["invalid"]
        and not fusion["unmapped_structure_cells"]
        and not fusion["extra_detector_cells"]
        and not matcher["orphan_text"]
        and "SPAN_UNKNOWN" not in reasons
    )
    return {
        "status": "STRUCTURED_TABLE" if complete else "PARTIAL_TABLE",
        "reason_codes": sorted(reasons),
    }


def assess_detector_grid(tokens, boxes, crop_size, separator_bands=()):
    """Assess model topology against detector geometry without assigning OCR text."""
    return _assess_detector_grid(
        tokens, _normalize_detector_boxes(boxes, crop_size), crop_size, separator_bands
    )


def _assess_detector_grid(tokens, detectors, crop_size, separator_bands=()):
    """Bind SLANeXt topology to detector edges, including row/column spans.

    SLANeXt positions are not a calibrated 0..1000 coordinate grid. A table
    is eligible only when every detected cell agrees with the independent
    structure tokens on its row, column and spans. No cell is synthesized.
    """
    topology = assess_structure(tokens, [], crop_size)
    cells, boxes = topology["cells"], detectors["valid"]
    rows, cols = topology["rows"], topology["cols"]
    complete = (
        rows > 0 and cols > 0 and len(cells) == len(boxes) > 0
        and not detectors["invalid"]
        and all(isinstance(cell[span], int) and cell[span] > 0
                for cell in cells for span in ("rowspan", "colspan"))
    )
    indices = {}
    for axis, count in ((0, cols), (1, rows)):
        if not complete:
            break
        extents = [cell["bbox_crop_px"][axis + 2] - cell["bbox_crop_px"][axis]
                   for cell in boxes]
        tolerance = max(1.0, min(extents) * 0.08)
        def logical_edge(value, axis=axis):
            if axis == 1:
                for band in separator_bands:
                    if band[1] <= value <= band[3]:
                        return (band[1] + band[3]) / 2
            return value

        edges = sorted((logical_edge(cell["bbox_crop_px"][side]), index, side)
                       for index, cell in enumerate(boxes)
                       for side in (axis, axis + 2))
        clusters = []
        for edge in edges:
            if not clusters or edge[0] - clusters[-1][0][0] > tolerance:
                clusters.append([])
            clusters[-1].append(edge)
        complete = len(clusters) == count + 1 and all(
            clusters[index + 1][0][0] - cluster[-1][0] > tolerance
            for index, cluster in enumerate(clusters[:-1])
        )
        for boundary, cluster in enumerate(clusters):
            for _coordinate, index, side in cluster:
                indices[index, side] = boundary
    mapped = {}
    if complete:
        for index, cell in enumerate(boxes):
            x0, y0, x1, y1 = [indices[index, side] for side in range(4)]
            key = (y0, x0, y1 - y0, x1 - x0)
            if key in mapped or x1 <= x0 or y1 <= y0:
                complete = False
            mapped[key] = cell["bbox_crop_px"]
        expected = [(cell["row"], cell["col"], cell["rowspan"], cell["colspan"])
                    for cell in cells]
        complete = complete and len(set(expected)) == len(expected) and set(expected) == set(mapped)
    if not complete:
        topology["status"] = "STRUCTURE_INVALID"
        topology["reason_codes"].append("DETECTOR_GRID_TOPOLOGY_MISMATCH")
        return topology
    structure = assess_structure(tokens, [mapped[key] for key in expected], crop_size)
    structure["geometry_source"] = "DETECTOR_GRID"
    return structure


class TableEngine:
    """Deep, deterministic seam from raw table-model evidence to TableIR."""

    def build(self, evidence: dict[str, Any]) -> dict[str, Any]:
        width = int(evidence["crop_width"])
        height = int(evidence["crop_height"])
        if width <= 0 or height <= 0:
            raise ValueError("Table crop dimensions must be positive")
        crop_size = (width, height)
        structure = assess_structure(
            list(evidence.get("structure", {}).get("tokens", [])),
            list(evidence.get("structure", {}).get("locations_1000", [])),
            crop_size,
        )
        detectors = _normalize_detector_boxes(
            list(evidence.get("cell_detection", {}).get("boxes", [])), crop_size
        )
        if evidence.get("structure", {}).get("geometry_source") == "DETECTOR_GRID":
            structure = _assess_detector_grid(
                list(evidence.get("structure", {}).get("tokens", [])), detectors, crop_size,
                evidence.get("cell_detection", {}).get("source_border_refinement", {}).get(
                    "excluded_blank_separator_bboxes", []
                ),
            )
        fusion = _fuse_structure_cells(structure["cells"], detectors["valid"])
        provisional_cells = []
        for fused in fusion["cells"]:
            detector = fused["detector_cell"]
            bbox = detector["bbox_crop_px"] if detector else fused.get("bbox_crop_px")
            if not _valid_bbox(bbox):
                # Keep the unmapped ordinal in fusion diagnostics. OCR without
                # valid geometry remains orphaned instead of inventing a cell.
                continue
            cell_id = _stable_id(
                "table-cell-",
                {
                    "table_id": evidence["table_id"],
                    "ordinal": fused["ordinal"],
                    "row": fused["row"],
                    "col": fused["col"],
                    "rowspan": fused["rowspan"],
                    "colspan": fused["colspan"],
                    "schema": TABLE_CELL_IR_SCHEMA,
                },
            )
            provisional_cells.append({"cell_id": cell_id, "bbox_crop_px": bbox, "fused": fused})
        ocr_rows = []
        for index, source in enumerate(evidence.get("ocr", {}).get("boxes", [])):
            row = copy.deepcopy(source)
            row.setdefault(
                "ocr_id",
                _stable_id(
                    "ocr-box-",
                    {"table_id": evidence["table_id"], "index": index, "bbox": row.get("bbox")},
                ),
            )
            ocr_rows.append(row)
        matcher = match_ocr_to_cells(ocr_rows, provisional_cells, crop_size=crop_size)
        assigned: dict[str, list[dict[str, Any]]] = defaultdict(list)
        ocr_by_id = {row["ocr_id"]: row for row in ocr_rows}
        for ocr_id, cell_id in matcher["assignments"].items():
            assigned[cell_id].append(ocr_by_id[ocr_id])
        cells = []
        for provisional in provisional_cells:
            fused = provisional["fused"]
            ocr = sorted(
                assigned[provisional["cell_id"]],
                key=lambda row: (
                    row.get("bbox", [0, 0, 0, 0])[1],
                    row.get("bbox", [0, 0, 0, 0])[0],
                    row["ocr_id"],
                ),
            )
            text = "\n".join(str(row.get("text") or "") for row in ocr)
            bbox = provisional["bbox_crop_px"]
            pdf_bbox = _crop_to_pdf_bbox(bbox, evidence["source_bbox_pdf_pt"], crop_size)
            from .pdf.source_table_math import recover_scientific_cell

            text, native_recovery = recover_scientific_cell(text, pdf_bbox, evidence)
            cells.append(
                {
                    "schema": TABLE_CELL_IR_SCHEMA,
                    "cell_id": provisional["cell_id"],
                    "row": fused["row"],
                    "col": fused["col"],
                    "rowspan": fused["rowspan"],
                    "colspan": fused["colspan"],
                    "bbox_crop_px": bbox,
                    "bbox_pdf_pt": pdf_bbox,
                    "text": text,
                    "source_ocr_boxes": ocr,
                    "content_type": _content_type(text),
                    "review_state": "REVIEW" if _content_type(text) == "FORMULA_LIKE_CELL" else "NONE",
                    "header": fused["header"],
                    "structure_ordinal": fused["ordinal"],
                    "detector_cell_id": (
                        fused["detector_cell"]["detector_id"] if fused["detector_cell"] else None
                    ),
                }
            )
            if native_recovery is not None:
                cells[-1]["native_script_recovery"] = native_recovery
        unassigned_formula_regions = _retain_formula_region_reviews(
            cells, evidence.get("table_content_ownership") or {}
        )
        quality = _quality(evidence, structure, detectors, fusion, matcher)
        quality["content_review_cell_ids"] = [
            cell["cell_id"] for cell in cells if cell["review_state"] != "NONE"
        ]
        content_reasons = set(quality["reason_codes"])
        if quality["content_review_cell_ids"]:
            content_reasons.add("TABLE_CELL_CONTENT_REVIEW_REQUIRED")
        if unassigned_formula_regions:
            content_reasons.add("TABLE_FORMULA_REGION_UNASSIGNED")
        quality["reason_codes"] = sorted(content_reasons)
        table = {
            "schema": TABLE_IR_SCHEMA,
            "table_id": str(evidence["table_id"]),
            "document_node_id": str(evidence["document_node_id"]),
            "document_id": str(evidence["document_id"]),
            "page_index": int(evidence["page_index"]),
            "source": {
                "crop_ref": str(evidence["source_crop_ref"]),
                "crop_sha256": str(evidence["source_crop_sha256"]),
                "bbox_pdf_pt": [float(value) for value in evidence["source_bbox_pdf_pt"]],
                "crop_width": width,
                "crop_height": height,
                "normalized_inference_crop_ref": evidence.get("normalized_inference_crop_ref"),
                "normalized_transform": copy.deepcopy(evidence.get("normalized_transform")),
            },
            "table_type": evidence.get("classification", {}).get("normalized_label", "UNKNOWN"),
            "classification_provenance": copy.deepcopy(evidence.get("classification", {})),
            "structure_provenance": copy.deepcopy(evidence.get("structure", {})),
            "cell_detection_provenance": copy.deepcopy(evidence.get("cell_detection", {})),
            "ocr_provenance": copy.deepcopy(evidence.get("ocr", {})),
            "structure_plausibility": structure,
            "fusion": {key: value for key, value in fusion.items() if key != "cells"},
            "detector_conflicts": {
                "invalid": detectors["invalid"],
                "duplicates": detectors["duplicates"],
            },
            "rows": structure["rows"],
            "cols": structure["cols"],
            "cells": cells,
            "unassigned_formula_regions": unassigned_formula_regions,
            "orphan_text": matcher["orphan_text"],
            "ocr_accounting": matcher["accounting"],
            "quality": quality,
            "source_truth": copy.deepcopy(evidence.get("source_truth")),
            "policy_contract": TABLE_QUALITY_POLICY_SCHEMA,
        }
        serialization = serialize_table_ir(table, evidence["source_crop_ref"])
        if quality["status"] == "STRUCTURED_TABLE" and serialization["format"] not in {
            "MARKDOWN",
            "HTML",
        }:
            table["quality"] = {
                **quality,
                "status": "PARTIAL_TABLE",
                "reason_codes": sorted({*quality["reason_codes"], "SERIALIZER_NOT_STRUCTURED"}),
            }
            serialization = serialize_table_ir(table, evidence["source_crop_ref"])
        validate_table_ir(table)
        return {
            "table_ir": table,
            "serialization": serialization,
            "diagnostics": {
                "structure": structure,
                "detectors": detectors,
                "fusion": fusion,
                "matcher": matcher,
            },
        }


def _escape_markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "\n").replace(
        "\r", "\n"
    ).replace("\n", "<br>")


def _cell_grid(table: dict[str, Any]) -> list[list[dict[str, Any] | None]]:
    rows, cols = int(table["rows"]), int(table["cols"])
    grid: list[list[dict[str, Any] | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    for cell in table["cells"]:
        row, col = int(cell["row"]), int(cell["col"])
        if 0 <= row < rows and 0 <= col < cols:
            grid[row][col] = cell
    return grid


def _serialize_markdown(table: dict[str, Any]) -> str:
    grid = _cell_grid(table)
    rendered = [
        "| " + " | ".join(_escape_markdown_cell(str(cell.get("text") or "")) if cell else "" for cell in row) + " |"
        for row in grid
    ]
    if not rendered:
        raise ValueError("Cannot serialize an empty table")
    separator = "| " + " | ".join("---" for _ in range(table["cols"])) + " |"
    return "\n".join([rendered[0], separator, *rendered[1:]])


def _serialize_html(table: dict[str, Any]) -> str:
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in table["cells"]:
        by_row[int(cell["row"])].append(cell)
    lines = ["<table>"]
    for row in range(int(table["rows"])):
        lines.append("  <tr>")
        for cell in sorted(by_row[row], key=lambda value: (int(value["col"]), value["cell_id"])):
            tag = "th" if cell.get("header") else "td"
            attrs = []
            if isinstance(cell["rowspan"], int) and cell["rowspan"] > 1:
                attrs.append(f'rowspan="{cell["rowspan"]}"')
            if isinstance(cell["colspan"], int) and cell["colspan"] > 1:
                attrs.append(f'colspan="{cell["colspan"]}"')
            attr_text = " " + " ".join(attrs) if attrs else ""
            body = html.escape(str(cell.get("text") or "")).replace("\r\n", "\n").replace(
                "\r", "\n"
            ).replace("\n", "<br>")
            lines.append(f"    <{tag}{attr_text}>{body}</{tag}>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def serialize_table_ir(table: dict[str, Any], asset_ref: str) -> dict[str, Any]:
    content_body = ""
    status = table["quality"]["status"]
    has_span = any(
        cell.get("rowspan") != 1 or cell.get("colspan") != 1 for cell in table.get("cells", [])
    )
    multiline = any("\n" in str(cell.get("text") or "") for cell in table.get("cells", []))
    if status == "STRUCTURED_TABLE":
        if not has_span and not multiline:
            body, output_format = _serialize_markdown(table), "MARKDOWN"
        else:
            body, output_format = _serialize_html(table), "HTML"
        content_body = body
        body += (
            "\n\n<details><summary>表格原图</summary>\n\n"
            f"![表格原图]({asset_ref})\n\n</details>"
        )
    elif status == "PARTIAL_TABLE":
        preview = _serialize_html(table) if table.get("cells") else ""
        content_body = preview
        body = "\n".join(
            value
            for value in (
                '<!-- TABLE_PARTIAL_REVIEW_REQUIRED source_crop_retained="true" -->',
                preview,
                f"![表格原图]({asset_ref})",
            )
            if value
        )
        output_format = "PARTIAL_WITH_IMAGE"
    else:
        body = "\n".join(
            (
                '<!-- TABLE_REVIEW_REQUIRED source_crop_retained="true" -->',
                f"![表格待确认]({asset_ref})",
            )
        )
        output_format = "IMAGE_FALLBACK"
    return {
        "schema": TABLE_SERIALIZER_SCHEMA,
        "format": output_format,
        "body": body,
        "content_body": content_body,
        "source_crop_retained": True,
        "determinism_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def validate_table_ir(table: dict[str, Any]) -> None:
    if table.get("schema") != TABLE_IR_SCHEMA:
        raise ValueError("Unsupported TableIR schema")
    if table.get("quality", {}).get("status") not in TABLE_STATUSES:
        raise ValueError("Unsupported table quality status")
    cells = table.get("cells", [])
    ids = [cell.get("cell_id") for cell in cells]
    if any(not cell_id for cell_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("TableIR cell IDs must be present and unique")
    for cell in cells:
        if cell.get("schema") != TABLE_CELL_IR_SCHEMA:
            raise ValueError("Unsupported TableCellIR schema")
        if not _valid_bbox(cell.get("bbox_crop_px")):
            raise ValueError("TableCellIR crop bbox is invalid")
        if cell.get("bbox_pdf_pt") is not None and not _valid_bbox(cell["bbox_pdf_pt"]):
            raise ValueError("TableCellIR PDF bbox is invalid")
    accounting = table.get("ocr_accounting", {})
    if accounting.get("all") != accounting.get("assigned", 0) + accounting.get("orphan", 0):
        raise ValueError("TableIR OCR accounting is invalid")
    if accounting.get("orphan") != len(table.get("orphan_text", [])):
        raise ValueError("TableIR OCR accounting orphan count is invalid")
    assigned = sum(len(cell.get("source_ocr_boxes", [])) for cell in cells)
    if accounting.get("assigned") != assigned:
        raise ValueError("TableIR OCR accounting assigned count is invalid")
    if table["quality"]["status"] == "STRUCTURED_TABLE":
        if table["structure_plausibility"]["status"] != "STRUCTURE_VALID":
            raise ValueError("Structured table requires valid structure")
        if table["fusion"]["unmapped_structure_cells"] or table["fusion"][
            "extra_detector_cells"
        ]:
            raise ValueError("Structured table requires complete cell fusion")
        serialized = serialize_table_ir(table, table["source"]["crop_ref"])
        if serialized["format"] not in {"MARKDOWN", "HTML"}:
            raise ValueError("Structured table requires structured serialization")


def apply_table_results(document: dict[str, Any], results: dict[str, dict[str, Any]]) -> None:
    table_nodes = {block["node_id"] for block in document["blocks"] if block.get("kind") == "TABLE"}
    if set(results) - table_nodes:
        raise ValueError("Table result references an unknown DocumentIR TABLE node")
    preserved_reviews = [
        row for row in document.get("review_items", []) if row.get("node_id") not in table_nodes
    ]
    table_reviews = []
    for block in document["blocks"]:
        if block.get("kind") != "TABLE" or block["node_id"] not in results:
            continue
        result = copy.deepcopy(results[block["node_id"]])
        table = result["table_ir"]
        validate_table_ir(table)
        if table["document_node_id"] != block["node_id"]:
            raise ValueError("TableIR DocumentIR identity mismatch")
        status = table["quality"]["status"]
        review_state = {
            "STRUCTURED_TABLE": "NONE",
            "PARTIAL_TABLE": "WARNING",
            "TABLE_REVIEW_REQUIRED": "REVIEW_REQUIRED",
            "NOT_A_TABLE_REVIEW": "REVIEW_REQUIRED",
        }[status]
        cell_reviews = [
            cell["cell_id"] for cell in table.get("cells", [])
            if cell.get("review_state", "NONE") != "NONE"
        ]
        boundary_reviews = block.get('provenance', {}).get('route_provenance', {}).get('boundary_reviews', [])
        if cell_reviews or table.get("unassigned_formula_regions") or boundary_reviews:
            review_state = "REVIEW_REQUIRED"
        review_reasons = sorted(set(table['quality']['reason_codes']) | {
            review['kind'] + '_REVIEW_REQUIRED' for review in boundary_reviews})
        block["content"]["table_ir"] = table
        block["content"]["table_serialization"] = result["serialization"]
        block["content"]["source_status"] = status
        block["review_state"] = review_state
        block["provenance"]["table_engine"] = {
            "schema": TABLE_IR_SCHEMA,
            "quality_status": status,
            "reason_codes": table["quality"]["reason_codes"],
            "source_crop_retained": True,
        }
        route_provenance = block["provenance"].get("route_provenance")
        if isinstance(route_provenance, dict) and "table_engine_called" in route_provenance:
            route_provenance["table_engine_called"] = True
        block["provenance"]["review_reasons"] = review_reasons
        if review_state != "NONE":
            table_reviews.append(
                {
                    "node_id": block["node_id"],
                    "page_index": block["page_index"],
                    "kind": "TABLE",
                    "cell_ids": cell_reviews,
                    "review_state": review_state,
                    "asset_uid": block["content"].get("asset_uid"),
                    "reason_codes": review_reasons,
                }
            )
    document["review_items"] = sorted(
        [*preserved_reviews, *table_reviews], key=lambda row: (row.get("page_index", -1), row["node_id"])
    )
    document["accounting"]["deferred_nodes"] = sum(
        block.get("review_state") == "DEFERRED" for block in document["blocks"]
    )
    document["accounting"]["review_nodes"] = len(document["review_items"])
    document.setdefault("provenance", {})["table_engine"] = {
        "schema": TABLE_IR_SCHEMA,
        "table_nodes_available": len(results),
        "global_reading_order_recomputed": False,
    }


def render_table_block(
    block: dict[str, Any], *, asset_ref: str | None = None
) -> str:
    serialized = block.get("content", {}).get("table_serialization")
    if serialized and serialized.get("schema") == TABLE_SERIALIZER_SCHEMA:
        table = block.get("content", {}).get("table_ir")
        if table and asset_ref:
            return serialize_table_ir(table, asset_ref)["body"]
        return str(serialized["body"])
    asset_ref = asset_ref or block.get("content", {}).get("asset_ref")
    return f"![表格待结构化]({asset_ref})"


class TableAuditEvidenceAccessor:
    schema = "bemarkdown-table-audit-evidence-accessor-v0"

    def get(self, block: dict[str, Any]) -> dict[str, Any]:
        if block.get("kind") != "TABLE":
            raise ValueError("Table audit evidence requires a TABLE block")
        content = block.get("content", {})
        table = content.get("table_ir")
        return {
            "schema": self.schema,
            "stable_node_id": block["node_id"],
            "source_crop": content.get("asset_ref"),
            "table_ir": table,
            "rendered_preview": content.get("table_serialization", {}).get("body"),
            "cell_bboxes": [cell.get("bbox_crop_px") for cell in (table or {}).get("cells", [])],
            "ocr": [
                row
                for cell in (table or {}).get("cells", [])
                for row in cell.get("source_ocr_boxes", [])
            ],
            "orphan_text": (table or {}).get("orphan_text", []),
            "quality": (table or {}).get("quality"),
        }
