"""Production-owned native PDF evidence extraction.

This module is the package-local form of the frozen Phase 6B-1.6 source
evidence contract.  It reads only the caller supplied PDF and page records;
it never reaches into Developer scripts, artifacts, or replay fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def extract_production_pdf_source_evidence(
    page_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Extract immutable line, image, vector, and page-visual evidence."""

    import fitz

    by_source: dict[Path, list[Mapping[str, Any]]] = defaultdict(list)
    for record in page_records:
        by_source[Path(str(record["source_path"])).resolve()].append(record)

    result: dict[tuple[str, int], dict[str, Any]] = {}
    for source_path, records in sorted(
        by_source.items(), key=lambda item: str(item[0])
    ):
        with fitz.open(source_path) as document:
            font_unicode_cache: dict[int, bool] = {}
            for record in sorted(records, key=lambda row: int(row["page_index"])):
                page = document[int(record["page_index"])]
                from .native_visibility import hidden_native_characters

                hidden = hidden_native_characters(page)
                native_text = _native_text_lines(
                    page, hidden_characters=hidden, font_unicode_cache=font_unicode_cache
                )
                native_images = [
                    {
                        "evidence_id": f"image-{index:04d}-xref-{placement['xref']}",
                        "bbox_pdf_pt": [float(value) for value in placement["bbox"]],
                        "xref": placement["xref"],
                        "object_identity": placement["object_identity"],
                        "directly_extractable": placement.get("directly_extractable"),
                        "extension": placement.get("extension"),
                        "native_appearance": placement.get("native_appearance", {}),
                    }
                    for index, placement in enumerate(
                        record.get("images", {}).get("placements", [])
                    )
                    if _valid_bbox(placement.get("bbox"))
                ]
                profile = str(record.get("source_profile") or "UNCERTAIN")
                route = str(record.get("routing_decision") or "REVIEW_REQUIRED")
                largest_raster = float(
                    record.get("images", {}).get("largest_image_coverage_ratio", 0.0)
                )
                page_visual_bbox = None
                if (
                    profile in {"IMAGE_ONLY", "IMAGE_WITH_TEXT_LAYER"}
                    or route == "VISUAL_REQUIRED"
                    or largest_raster >= 0.8
                ):
                    geometry = record["geometry"]
                    page_visual_bbox = [
                        0.0,
                        0.0,
                        float(geometry["width_pt"]),
                        float(geometry["height_pt"]),
                    ]
                key = str(record["document_id"]), int(record["page_index"])
                result[key] = {
                    "native_text": native_text,
                    "native_images": native_images,
                    "native_vectors": _vector_clusters(page),
                    "native_text_visibility": {
                        "version": "native-paint-order-visibility-v1",
                        "excluded_characters": list(hidden.values()),
                    },
                    "page_visual_bbox": page_visual_bbox,
                }
                if record.get("native_text_trust") in {
                    "HIGH",
                    "MEDIUM",
                } and profile not in {"IMAGE_ONLY", "IMAGE_WITH_TEXT_LAYER"}:
                    from .native_math import propose_native_math

                    result[key]["native_math_geometry"] = propose_native_math(page, hidden_characters=hidden)
    return result


def _native_text_lines(
    page: Any, *, hidden_characters=None, font_unicode_cache=None
) -> list[dict[str, Any]]:
    from .native_font_unicode import native_font_unicode_issues, normalized_font_name
    from .native_glyph_bounds import character_key, page_native_ink_map
    from .native_visibility import hidden_native_characters

    if hidden_characters is None:
        hidden_characters = hidden_native_characters(page)
    ink_map = page_native_ink_map(page) if hasattr(page, "get_fonts") else {}
    font_issues = native_font_unicode_issues(
        page, font_unicode_cache if font_unicode_cache is not None else {}
    ) if hasattr(page, "get_fonts") else {}
    rows: list[dict[str, Any]] = []
    blocks = page.get_text("rawdict", sort=False).get("blocks", [])
    for block_index, block in enumerate(blocks):
        if block.get("type") != 0 or not _valid_bbox(block.get("bbox")):
            continue
        valid_lines = []
        for line_index, line in enumerate(block.get("lines", [])):
            for span in line.get("spans", []):
                characters = span.get("chars", [])
                characters = [char for char in characters if not char.get("origin")
                              or character_key(char.get("c", ""), char["origin"]) not in hidden_characters]
                span["chars"] = characters
                span["text"] = "".join(str(char.get("c", "")) for char in characters)
                ink_boxes = [
                    char["bbox"]
                    for char in characters
                    if str(char.get("c", "")).strip() and _valid_bbox(char.get("bbox"))
                ]
                if ink_boxes:
                    span["content_bbox"] = _union_boxes(ink_boxes)
            spans = [
                span for span in line.get("spans", [])
                if _valid_bbox(span.get("bbox")) and str(span.get("text", "")).strip()
            ]
            if not spans:
                continue
            bbox = line.get("bbox")
            if not _valid_bbox(bbox):
                bbox = _union_boxes([span["bbox"] for span in spans])
            valid_lines.append((line_index, bbox, spans))
        for local_index, (line_index, bbox, spans) in enumerate(valid_lines):
            line_id = f"text-block-{block_index:04d}-line-{line_index:04d}"
            previous = valid_lines[local_index - 1][1] if local_index else block["bbox"]
            following = (
                valid_lines[local_index + 1][1]
                if local_index + 1 < len(valid_lines)
                else block["bbox"]
            )
            band_top = (
                float(block["bbox"][1])
                if local_index == 0
                else (float(previous[3]) + float(bbox[1])) / 2
            )
            band_bottom = (
                float(block["bbox"][3])
                if local_index + 1 == len(valid_lines)
                else (float(bbox[3]) + float(following[1])) / 2
            )
            span_rows = [
                {
                    "span_id": f"{line_id}-span-{span_index:04d}",
                    "bbox_pdf_pt": [
                        float(value) for value in span.get("content_bbox", span["bbox"])
                    ],
                    "storage_bbox_pdf_pt": [float(value) for value in span["bbox"]],
                    "punctuation_characters": [
                        {"text": char["c"], "bbox_pdf_pt": list(char["bbox"])}
                        for char in span.get("chars", [])
                        if char.get("c") in "。，；：、"
                        and _valid_bbox(char.get("bbox"))
                    ],
                    "text": str(span.get("text", "")),
                }
                for span_index, span in enumerate(spans)
            ]
            for source_span, span_row in zip(spans, span_rows, strict=True):
                issue = font_issues.get(normalized_font_name(str(source_span.get("font", ""))))
                if issue:
                    span_row["unicode_mapping_issue"] = dict(issue)
                chars = [c for c in source_span.get("chars", []) if c.get("c", "").strip()]
                bounds = [ink_map.get(character_key(c["c"], c["origin"]))
                          if c.get("origin") else None for c in chars]
                if bounds and all(b is not None for b in bounds):
                    span_row["ink_bbox_pdf_pt"] = _union_boxes(bounds)
                    span_row["ink_bbox_basis"] = "EMBEDDED_SOURCE_GLYPH_OUTLINES"
            rows.append(
                {
                    "evidence_id": line_id,
                    "source_block_id": f"text-block-{block_index:04d}",
                    "source_line_span_ids": [row["span_id"] for row in span_rows],
                    "spans": span_rows,
                    "bbox_pdf_pt": [float(value) for value in bbox],
                    "source_block_bbox_pdf_pt": [
                        float(value) for value in block["bbox"]
                    ],
                    "source_line_band_bbox_pdf_pt": [
                        float(block["bbox"][0]),
                        band_top,
                        float(block["bbox"][2]),
                        band_bottom,
                    ],
                    "native_char_count": sum(
                        len(str(span.get("text", ""))) for span in spans
                    ),
                    "text": "".join(str(span.get("text", "")) for span in spans),
                    "source_unit": "LINE",
                }
            )
    from .native_drawn_blanks import recover_drawn_answer_blanks

    return recover_drawn_answer_blanks(page, rows)


def _vector_clusters(page: Any) -> list[dict[str, Any]]:
    from .visible_vectors import visible_vector_boxes

    boxes: list[list[float]] = []
    try:
        drawing_boxes = list(visible_vector_boxes(page))
    except (RuntimeError, ValueError):
        drawing_boxes = []
    page_area = max(1.0, float(page.rect.width * page.rect.height))
    for bbox in drawing_boxes:
        area = _bbox_area(bbox)
        if _valid_bbox(bbox) and 4.0 <= area < page_area * 0.98:
            boxes.append(bbox)
    groups: list[list[list[float]]] = []
    for bbox in sorted(
        boxes, key=lambda value: (value[1], value[0], value[3], value[2])
    ):
        group = next(
            (
                current
                for current in groups
                if _boxes_touch(_union_boxes(current), bbox, margin=6.0)
            ),
            None,
        )
        if group is None:
            groups.append([bbox])
        else:
            group.append(bbox)
    return [
        {
            "evidence_id": f"vector-cluster-{index:04d}",
            "bbox_pdf_pt": _union_boxes(group),
            "drawing_count": len(group),
        }
        for index, group in enumerate(groups)
        if _bbox_area(_union_boxes(group)) >= 16.0
    ]


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _bbox_area(value: Sequence[float]) -> float:
    return max(0.0, float(value[2]) - float(value[0])) * max(
        0.0, float(value[3]) - float(value[1])
    )


def _union_boxes(values: Sequence[Sequence[float]]) -> list[float]:
    return [
        min(float(value[0]) for value in values),
        min(float(value[1]) for value in values),
        max(float(value[2]) for value in values),
        max(float(value[3]) for value in values),
    ]


def _boxes_touch(
    first: Sequence[float], second: Sequence[float], *, margin: float
) -> bool:
    return not (
        float(first[2]) + margin < float(second[0])
        or float(second[2]) + margin < float(first[0])
        or float(first[3]) + margin < float(second[1])
        or float(second[3]) + margin < float(first[1])
    )
