from __future__ import annotations

import hashlib
import time
import unicodedata
from collections import Counter
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any

import fitz

from .production import build_document_id

ROUTING_BASELINE = "pdf-routing-v0"
ROUTING_STATUS = "BASELINE"
ROUTING_THRESHOLDS = {
    "substantial_text_non_whitespace_chars": 20,
    "high_raster_coverage_ratio": 0.5,
    "scan_like_raster_coverage_ratio": 0.85,
    "significant_raster_coverage_ratio": 0.05,
    "significant_vector_coverage_ratio": 0.05,
    "healthy_printable_ratio": 0.85,
    "maximum_tolerated_replacement_char_ratio": 0.05,
    "suspicious_very_short_span_ratio": 0.85,
}


class PdfInspectionError(RuntimeError):
    """Explicit, source-bound failure from deterministic PDF inspection."""

    def __init__(self, code: str, source: Path, detail: str):
        super().__init__(f"{code}: {source}: {detail}")
        self.code = code
        self.source = source
        self.detail = detail


class PdfSourceInspector:
    """Collect deterministic page-source features without OCR or rendering."""

    def inspect(self, source: str | Path) -> dict[str, Any]:
        started = time.perf_counter()
        source = Path(source).resolve()
        if not source.is_file():
            raise PdfInspectionError("SOURCE_NOT_FOUND", source, "file is missing")
        source_sha256 = _sha256_file(source)
        size_bytes = source.stat().st_size
        try:
            document = fitz.open(source)
        except (fitz.EmptyFileError, fitz.FileDataError) as exc:
            raise PdfInspectionError("MALFORMED_PDF", source, str(exc)) from exc
        with document:
            if document.needs_pass:
                raise PdfInspectionError(
                    "ENCRYPTED_PDF", source, "password is required"
                )
            if document.page_count <= 0:
                raise PdfInspectionError(
                    "ZERO_PAGE_PDF", source, "document has no pages"
                )
            extraction_cache: dict[int, dict[str, Any] | None] = {}
            try:
                pages = [
                    _inspect_page(document[index], document, extraction_cache)
                    for index in range(document.page_count)
                ]
            except (RuntimeError, ValueError) as exc:
                raise PdfInspectionError(
                    "BROKEN_PAGE_OBJECT", source, str(exc)
                ) from exc
            identities = Counter(
                placement["object_identity"]
                for page in pages
                for placement in page["images"]["placements"]
            )
            for page in pages:
                for placement in page["images"]["placements"]:
                    placement["reused_object"] = identities[
                        placement["object_identity"]
                    ] > 1
        wall_seconds = time.perf_counter() - started
        return {
            "document_id": build_document_id(source, source_sha256),
            "source_path": str(source),
            "sha256": source_sha256,
            "bytes": size_bytes,
            "page_count": len(pages),
            "backend": {
                "library": "PyMuPDF",
                "version": fitz.VersionBind,
            },
            "inspection_wall_seconds": round(wall_seconds, 6),
            "pages": pages,
        }


def apply_routing_baseline(document: dict[str, Any]) -> dict[str, Any]:
    """Apply the corpus-derived v0 page router without reading source names."""
    result = deepcopy(document)
    for page in result.get("pages", []):
        page.update(_classify_page(page))
    result["routing_baseline"] = ROUTING_BASELINE
    result["routing_status"] = ROUTING_STATUS
    result["routing_thresholds"] = dict(ROUTING_THRESHOLDS)
    result["document_profile"] = _document_profile(result.get("pages", []))
    return result


def _classify_page(page: dict[str, Any]) -> dict[str, Any]:
    text = page["native_text"]
    images = page["images"]
    vectors = page["vectors"]
    chars = int(text["non_whitespace_char_count"])
    largest_raster = float(images["largest_image_coverage_ratio"])
    total_raster = float(images["page_coverage_ratio"])
    vector_coverage = float(vectors["drawing_coverage_ratio"])
    has_text = chars > 0
    substantial_text = chars >= ROUTING_THRESHOLDS[
        "substantial_text_non_whitespace_chars"
    ]
    scan_like_raster = largest_raster >= ROUTING_THRESHOLDS[
        "scan_like_raster_coverage_ratio"
    ]
    significant_raster = total_raster >= ROUTING_THRESHOLDS[
        "significant_raster_coverage_ratio"
    ]
    significant_vector = vector_coverage >= ROUTING_THRESHOLDS[
        "significant_vector_coverage_ratio"
    ]
    trust = _native_text_trust(text, scan_like_raster=scan_like_raster)
    reasons = []
    if largest_raster >= 0.9:
        reasons.append("FULL_PAGE_RASTER")
    elif largest_raster >= ROUTING_THRESHOLDS["high_raster_coverage_ratio"]:
        reasons.append("HIGH_RASTER_COVERAGE")
    if not has_text:
        reasons.append("NO_NATIVE_TEXT")
    if substantial_text:
        reasons.append("SUBSTANTIAL_NATIVE_TEXT")
    if scan_like_raster and has_text:
        reasons.extend(["TEXT_LAYER_PRESENT", "TEXT_LAYER_SUSPICIOUS"])
    if significant_vector:
        reasons.append("SIGNIFICANT_NATIVE_VECTOR")
    if has_text and significant_raster and not scan_like_raster:
        reasons.append("MIXED_RASTER_AND_NATIVE")
    if any(
        placement.get("extraction_error") not in {None, "NO_XREF"}
        for placement in images["placements"]
    ):
        reasons.append("BROKEN_IMAGE_EXTRACTION")

    if page.get("structural_warnings") and "VERY_LARGE_PAGE_DIMENSIONS" in page[
        "structural_warnings"
    ]:
        source_profile = "UNCERTAIN"
        route = "REVIEW_REQUIRED"
    elif scan_like_raster and has_text:
        source_profile = "IMAGE_WITH_TEXT_LAYER"
        route = "VISUAL_REQUIRED"
    elif not has_text and images["placed_image_count"]:
        source_profile = "IMAGE_ONLY"
        route = "VISUAL_REQUIRED"
    elif not has_text or trust == "LOW" or not substantial_text:
        source_profile = "UNCERTAIN"
        route = "REVIEW_REQUIRED"
    elif significant_raster or significant_vector:
        source_profile = "MIXED_NATIVE_VISUAL"
        route = "HYBRID_REQUIRED"
    else:
        source_profile = "NATIVE_TEXT"
        route = "NATIVE_FIRST"
    if source_profile == "UNCERTAIN" or route == "REVIEW_REQUIRED":
        reasons.append("AMBIGUOUS_STRUCTURE")
    return {
        "source_profile": source_profile,
        "native_text_trust": trust,
        "routing_decision": route,
        "reason_codes": sorted(set(reasons)),
    }


def _native_text_trust(text: dict[str, Any], *, scan_like_raster: bool) -> str:
    chars = int(text["non_whitespace_char_count"])
    if chars == 0:
        return "NONE"
    healthy = (
        float(text["printable_ratio"])
        >= ROUTING_THRESHOLDS["healthy_printable_ratio"]
        and _ratio(float(text["replacement_char_count"]), float(max(chars, 1)))
        <= ROUTING_THRESHOLDS["maximum_tolerated_replacement_char_ratio"]
    )
    fragmented = float(text["very_short_span_ratio"]) >= ROUTING_THRESHOLDS[
        "suspicious_very_short_span_ratio"
    ]
    degraded = (
        fragmented
        or int(text["control_char_count"]) > 0
        or int(text["replacement_char_count"]) > 0
    )
    if scan_like_raster:
        return "LOW"
    if (
        healthy
        and not degraded
        and chars >= 200
        and float(text["text_bbox_vertical_span_ratio"]) >= 0.2
    ):
        return "HIGH"
    if healthy and chars >= ROUTING_THRESHOLDS[
        "substantial_text_non_whitespace_chars"
    ]:
        return "MEDIUM"
    return "LOW"


def _document_profile(pages: list[dict[str, Any]]) -> str:
    if not pages:
        return "UNCERTAIN"
    profiles = Counter(page["source_profile"] for page in pages)
    if profiles["IMAGE_ONLY"] == len(pages):
        return "IMAGE_ONLY"
    if profiles["NATIVE_TEXT"] / len(pages) >= 0.8 and not any(
        profiles[name]
        for name in ("IMAGE_ONLY", "IMAGE_WITH_TEXT_LAYER", "MIXED_NATIVE_VISUAL")
    ):
        return "NATIVE_DOMINANT"
    if profiles["UNCERTAIN"] == len(pages):
        return "UNCERTAIN"
    return "MIXED"


def _inspect_page(
    page: fitz.Page,
    document: fitz.Document,
    extraction_cache: dict[int, dict[str, Any] | None],
) -> dict[str, Any]:
    page_rect = page.rect
    page_area = max(0.0, page_rect.width * page_rect.height)
    text = _text_features(page, page_area)
    images = _image_features(page, document, extraction_cache, page_area)
    structural_warnings = []
    if max(page_rect.width, page_rect.height) > 14_400:
        structural_warnings.append("VERY_LARGE_PAGE_DIMENSIONS")
    if any(
        placement["extraction_error"] not in {None, "NO_XREF"}
        for placement in images["placements"]
    ):
        structural_warnings.append("BROKEN_IMAGE_EXTRACTION")
    if images["inspection_error"] is not None:
        structural_warnings.append("BROKEN_IMAGE_OBJECT")
    return {
        "page_index": page.number,
        "page_number": page.number + 1,
        "geometry": {
            "width_pt": round(page_rect.width, 4),
            "height_pt": round(page_rect.height, 4),
            "rotation": page.rotation,
        },
        "native_text": text,
        "images": images,
        "vectors": _vector_features(page, page_area),
        "reading_order": _reading_order_features(text["blocks"], page_rect),
        "structural_warnings": structural_warnings,
    }


def _reading_order_features(
    blocks: list[dict[str, Any]], page_rect: fitz.Rect
) -> dict[str, Any]:
    rectangles = [fitz.Rect(block["bbox"]) for block in blocks]
    overlaps = 0
    for index, first in enumerate(rectangles):
        for second in rectangles[index + 1 :]:
            intersection = first & second
            intersection_area = max(0.0, intersection.width * intersection.height)
            smaller_area = min(first.width * first.height, second.width * second.height)
            if smaller_area and intersection_area / smaller_area >= 0.05:
                overlaps += 1

    midpoint = page_rect.x0 + page_rect.width / 2
    narrow = [rect for rect in rectangles if rect.width <= page_rect.width * 0.6]
    left = [rect for rect in narrow if (rect.x0 + rect.x1) / 2 < midpoint * 0.9]
    right = [rect for rect in narrow if (rect.x0 + rect.x1) / 2 > midpoint * 1.1]
    column_like = len(left) >= 2 and len(right) >= 2
    indexed = list(enumerate(rectangles))
    if column_like:
        expected = sorted(
            indexed,
            key=lambda row: (
                0 if (row[1].x0 + row[1].x1) / 2 < midpoint else 1,
                row[1].y0,
                row[1].x0,
            ),
        )
    else:
        expected = sorted(indexed, key=lambda row: (row[1].y0, row[1].x0))
    expected_rank = {original_index: rank for rank, (original_index, _) in enumerate(expected)}
    displacement = _ratio(
        sum(abs(index - expected_rank[index]) for index in range(len(rectangles))),
        len(rectangles) ** 2,
    )
    if overlaps or displacement >= 0.3:
        risk = "HIGH"
    elif column_like or displacement >= 0.15:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {
        "block_order": [_rect_list(rect) for rect in rectangles],
        "column_like_layout": column_like,
        "overlapping_text_bbox_count": overlaps,
        "object_order_displacement_ratio": displacement,
        "risk": risk,
    }


def _vector_features(page: fitz.Page, page_area: float) -> dict[str, Any]:
    records = []
    clipped_rectangles: list[fitz.Rect] = []
    for drawing in page.get_drawings():
        rect = _valid_rect(drawing.get("rect"))
        if rect is None:
            continue
        clipped = rect & page.rect
        if not clipped.is_empty:
            clipped_rectangles.append(clipped)
        records.append(
            {
                "bbox": _rect_list(rect),
                "path_item_count": len(drawing.get("items", [])),
                "has_fill": drawing.get("fill") is not None,
                "has_stroke": drawing.get("color") is not None,
            }
        )
    return {
        "drawing_count": len(records),
        "drawing_coverage_ratio": _ratio(
            _rect_union_area(clipped_rectangles), page_area
        ),
        "drawing_bbox": (
            _rect_list(bounds) if (bounds := _bounding_rect(clipped_rectangles)) else None
        ),
        "drawings": records,
    }


def _image_features(
    page: fitz.Page,
    document: fitz.Document,
    extraction_cache: dict[int, dict[str, Any] | None],
    page_area: float,
) -> dict[str, Any]:
    xobjects = page.get_images(full=True)
    xrefs = {int(row[0]) for row in xobjects if int(row[0]) > 0}
    smasks = {int(row[0]): int(row[1]) for row in xobjects if int(row[0]) > 0}
    try:
        occurrences = page.get_image_info(hashes=True, xrefs=True)
    except Exception as exc:  # noqa: BLE001 - isolate one broken page image table
        return {
            "image_xobject_count": len(xrefs),
            "placed_image_count": 0,
            "placement_area": 0.0,
            "page_coverage_ratio": 0.0,
            "largest_image_coverage_ratio": 0.0,
            "directly_extractable_occurrence_count": 0,
            "inspection_error": f"{type(exc).__name__}: {exc}",
            "placements": [],
        }
    placements = []
    clipped_rectangles: list[fitz.Rect] = []
    for occurrence in occurrences:
        rect = _valid_rect(occurrence.get("bbox"))
        if rect is None:
            continue
        clipped = rect & page.rect
        clipped_area = max(0.0, clipped.width * clipped.height)
        if clipped_area:
            clipped_rectangles.append(clipped)
        xref = int(occurrence.get("xref") or 0)
        digest = occurrence.get("digest")
        digest_hex = digest.hex() if isinstance(digest, bytes) else str(digest or "")
        identity = f"xref:{xref}" if xref > 0 else f"digest:{digest_hex}"
        extracted = _extract_image(document, xref, extraction_cache)
        extractable = extracted is not None and extracted["error"] is None
        from .pdf.native_image_appearance import verify_native_image_appearance

        appearance = verify_native_image_appearance(page, document, xref, rect)
        placements.append(
            {
                "xref": xref or None,
                "object_identity": identity,
                "bbox": _rect_list(rect),
                "pixel_width": int(occurrence.get("width") or 0),
                "pixel_height": int(occurrence.get("height") or 0),
                "placement_area": round(clipped_area, 4),
                "coverage_ratio": _ratio(clipped_area, page_area),
                "directly_extractable": extractable,
                "extractable_bytes": extracted["bytes"] if extractable else 0,
                "extension": extracted["ext"] if extractable else None,
                "extraction_error": (
                    extracted["error"] if extracted is not None else "NO_XREF"
                ),
                "has_alpha_or_mask": bool(
                    occurrence.get("has-mask")
                    or smasks.get(xref, 0) > 0
                    or (extracted is not None and extracted["smask"] > 0)
                ),
                "reused_object": False,
                "native_appearance": appearance,
            }
        )
    placement_area = sum(row["placement_area"] for row in placements)
    return {
        "image_xobject_count": len(xrefs),
        "placed_image_count": len(placements),
        "placement_area": round(placement_area, 4),
        "page_coverage_ratio": _ratio(_rect_union_area(clipped_rectangles), page_area),
        "largest_image_coverage_ratio": max(
            (row["coverage_ratio"] for row in placements), default=0.0
        ),
        "directly_extractable_occurrence_count": sum(
            row["directly_extractable"] for row in placements
        ),
        "inspection_error": None,
        "placements": placements,
    }


def _extract_image(
    document: fitz.Document,
    xref: int,
    cache: dict[int, dict[str, Any] | None],
) -> dict[str, Any] | None:
    if xref <= 0:
        return None
    if xref not in cache:
        try:
            value = document.extract_image(xref)
            image = value.get("image")
            cache[xref] = (
                {
                    "bytes": len(image),
                    "ext": value.get("ext"),
                    "smask": int(value.get("smask") or 0),
                    "error": None,
                }
                if image
                else None
            )
        except Exception as exc:  # noqa: BLE001 - isolate one broken image object
            cache[xref] = {
                "bytes": 0,
                "ext": None,
                "smask": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return cache[xref]


def _text_features(page: fitz.Page, page_area: float) -> dict[str, Any]:
    payload = page.get_text("dict", sort=False)
    blocks = [block for block in payload.get("blocks", []) if block.get("type") == 0]
    lines = [line for block in blocks for line in block.get("lines", [])]
    spans = [span for line in lines for span in line.get("spans", [])]
    span_texts = [str(span.get("text", "")) for span in spans]
    text = "".join(span_texts)
    block_rects = [_valid_rect(block.get("bbox")) for block in blocks]
    block_rects = [rect for rect in block_rects if rect is not None]
    bounds = _bounding_rect(block_rects)
    char_count = len(text)
    nonempty_spans = [value for value in span_texts if value.strip()]
    return {
        "char_count": char_count,
        "non_whitespace_char_count": sum(not char.isspace() for char in text),
        "block_count": len(blocks),
        "line_count": len(lines),
        "span_count": len(spans),
        "font_count": len(
            {str(span.get("font", "")) for span in spans if span.get("font")}
        ),
        "text_bbox_union_area_ratio": _ratio(_rect_union_area(block_rects), page_area),
        "text_bbox_vertical_span_ratio": _ratio(
            bounds.height if bounds is not None else 0.0, page.rect.height
        ),
        "text_bbox_horizontal_span_ratio": _ratio(
            bounds.width if bounds is not None else 0.0, page.rect.width
        ),
        "printable_ratio": _ratio(sum(char.isprintable() for char in text), char_count),
        "whitespace_ratio": _ratio(sum(char.isspace() for char in text), char_count),
        "control_char_count": sum(
            unicodedata.category(char) == "Cc" for char in text
        ),
        "replacement_char_count": text.count("\ufffd"),
        "very_short_span_ratio": _ratio(
            sum(len(value.strip()) <= 2 for value in nonempty_spans),
            len(nonempty_spans),
        ),
        "blocks": [
            {
                "bbox": _rect_list(rect),
                "text_char_count": sum(
                    len(str(span.get("text", "")))
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ),
            }
            for block, rect in zip(blocks, block_rects, strict=False)
        ],
    }


def _valid_rect(value: Any) -> fitz.Rect | None:
    try:
        rect = fitz.Rect(value)
    except (TypeError, ValueError):
        return None
    if rect.is_empty or rect.is_infinite:
        return None
    return rect


def _bounding_rect(rectangles: list[fitz.Rect]) -> fitz.Rect | None:
    if not rectangles:
        return None
    result = fitz.Rect(rectangles[0])
    for rect in rectangles[1:]:
        result.include_rect(rect)
    return result


def _rect_union_area(rectangles: list[fitz.Rect]) -> float:
    if not rectangles:
        return 0.0
    x_values = sorted({value for rect in rectangles for value in (rect.x0, rect.x1)})
    area = 0.0
    for left, right in pairwise(x_values):
        if right <= left:
            continue
        intervals = sorted(
            (rect.y0, rect.y1)
            for rect in rectangles
            if rect.x0 < right and rect.x1 > left
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for local_start, local_end in intervals[1:]:
                if local_start > end:
                    covered += max(0.0, end - start)
                    start, end = local_start, local_end
                else:
                    end = max(end, local_end)
            covered += max(0.0, end - start)
        area += (right - left) * covered
    return area


def _rect_list(rect: fitz.Rect) -> list[float]:
    return [round(value, 4) for value in (rect.x0, rect.y0, rect.x1, rect.y1)]


def _ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
