"""Deterministic page-level PDF object profiling.

No learned confidence model is used.  Each profile exposes the measurements
and reasons that determine whether native extraction may be authoritative.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

PDF_OBJECT_PROFILE_SCHEMA = "bemarkdown-pdf-object-profile-v1"


class ProfileState(str, Enum):
    NATIVE_RELIABLE = "NATIVE_RELIABLE"
    MIXED = "MIXED"
    VISUAL_REQUIRED = "VISUAL_REQUIRED"


@dataclass(frozen=True, slots=True)
class PDFObjectProfile:
    schema: str
    document_id: str
    page_id: str
    page_index: int
    source_pdf_sha256: str
    has_text_objects: bool
    has_font_objects: bool
    has_image_objects: bool
    full_page_raster_likelihood: float
    native_text_char_count: int
    native_text_block_count: int
    native_text_bbox_coverage: float
    unicode_valid_ratio: float
    replacement_char_ratio: float
    control_char_ratio: float
    suspicious_glyph_ratio: float
    native_image_count: int
    extraction_order_available: bool
    geometry_available: bool
    profile_state: ProfileState
    reliability_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["profile_state"] = self.profile_state.value
        result["reliability_reasons"] = list(self.reliability_reasons)
        return result


def _ratio(count: int, total: int, *, empty: float = 0.0) -> float:
    return round(count / total, 8) if total else empty


def _text_quality(text: str) -> tuple[float, float, float, float]:
    total = len(text)
    if not total:
        return 0.0, 0.0, 0.0, 0.0
    replacement = text.count("\ufffd")
    controls = sum(
        unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in text
    )
    invalid = sum(0xD800 <= ord(char) <= 0xDFFF for char in text)
    private_use = sum(unicodedata.category(char) == "Co" for char in text)
    suspicious_fragments = ("Ã", "Â", "â€", "ï¿½")
    mojibake = sum(text.count(fragment) for fragment in suspicious_fragments)
    suspicious = private_use + mojibake + invalid
    return (
        _ratio(total - invalid - replacement, total),
        _ratio(replacement, total),
        _ratio(controls, total),
        _ratio(suspicious, total),
    )


def _profile_state(
    *,
    text_chars: int,
    geometry_available: bool,
    unicode_valid_ratio: float,
    replacement_char_ratio: float,
    control_char_ratio: float,
    suspicious_glyph_ratio: float,
    full_page_raster_likelihood: float,
) -> tuple[ProfileState, tuple[str, ...]]:
    reasons: list[str] = []
    if text_chars == 0:
        reasons.append("NO_NATIVE_TEXT_OBJECTS")
    elif text_chars < 16:
        reasons.append("NATIVE_TEXT_TOO_SPARSE")
    if not geometry_available:
        reasons.append("NATIVE_TEXT_GEOMETRY_UNAVAILABLE")
    if unicode_valid_ratio < 0.98:
        reasons.append("UNICODE_VALID_RATIO_LOW")
    if replacement_char_ratio > 0.01:
        reasons.append("REPLACEMENT_CHAR_RATIO_HIGH")
    if control_char_ratio > 0.01:
        reasons.append("CONTROL_CHAR_RATIO_HIGH")
    if suspicious_glyph_ratio > 0.03:
        reasons.append("SUSPICIOUS_GLYPH_RATIO_HIGH")
    if full_page_raster_likelihood >= 0.8:
        reasons.append("FULL_PAGE_RASTER_PRESENT")

    text_reliable = (
        text_chars >= 16
        and geometry_available
        and unicode_valid_ratio >= 0.98
        and replacement_char_ratio <= 0.01
        and control_char_ratio <= 0.01
        and suspicious_glyph_ratio <= 0.03
    )
    if text_reliable and full_page_raster_likelihood < 0.8:
        return ProfileState.NATIVE_RELIABLE, tuple(reasons or ["NATIVE_TEXT_RELIABLE"])
    if text_chars > 0:
        return ProfileState.MIXED, tuple(
            reasons or ["NATIVE_AND_VISUAL_EVIDENCE_PRESENT"]
        )
    return ProfileState.VISUAL_REQUIRED, tuple(reasons)


class PDFObjectProfiler:
    """Profile every page through one stable, deterministic interface."""

    @staticmethod
    def classify_measurements(
        *,
        text_chars: int,
        geometry_available: bool,
        unicode_valid_ratio: float,
        replacement_char_ratio: float,
        control_char_ratio: float,
        suspicious_glyph_ratio: float,
        full_page_raster_likelihood: float,
    ) -> tuple[ProfileState, tuple[str, ...]]:
        return _profile_state(
            text_chars=text_chars,
            geometry_available=geometry_available,
            unicode_valid_ratio=unicode_valid_ratio,
            replacement_char_ratio=replacement_char_ratio,
            control_char_ratio=control_char_ratio,
            suspicious_glyph_ratio=suspicious_glyph_ratio,
            full_page_raster_likelihood=full_page_raster_likelihood,
        )

    def profile(
        self, source_pdf: Path | str, *, document_id: str
    ) -> list[PDFObjectProfile]:
        import fitz

        path = Path(source_pdf)
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        document = fitz.open(path)
        profiles: list[PDFObjectProfile] = []
        try:
            for page_index, page in enumerate(document):
                raw = page.get_text("dict", sort=False)
                text_blocks = [
                    block for block in raw.get("blocks", []) if block.get("type") == 0
                ]
                text_spans = [
                    span
                    for block in text_blocks
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ]
                text = "".join(str(span.get("text", "")) for span in text_spans)
                text_bbox_area = sum(
                    max(0.0, float(block["bbox"][2]) - float(block["bbox"][0]))
                    * max(0.0, float(block["bbox"][3]) - float(block["bbox"][1]))
                    for block in text_blocks
                    if len(block.get("bbox", [])) == 4
                )
                page_area = max(1.0, float(page.rect.width * page.rect.height))
                images = page.get_images(full=True)
                image_coverage = 0.0
                for image in images:
                    for rect in page.get_image_rects(image[0]):
                        image_coverage = max(
                            image_coverage,
                            max(0.0, float(rect.width * rect.height)) / page_area,
                        )
                quality = _text_quality(text)
                geometry = bool(text_blocks) and all(
                    len(block.get("bbox", [])) == 4 for block in text_blocks
                )
                state, reasons = _profile_state(
                    text_chars=len(text),
                    geometry_available=geometry,
                    unicode_valid_ratio=quality[0],
                    replacement_char_ratio=quality[1],
                    control_char_ratio=quality[2],
                    suspicious_glyph_ratio=quality[3],
                    full_page_raster_likelihood=image_coverage,
                )
                profiles.append(
                    PDFObjectProfile(
                        schema=PDF_OBJECT_PROFILE_SCHEMA,
                        document_id=document_id,
                        page_id=f"{document_id}:{page_index}",
                        page_index=page_index,
                        source_pdf_sha256=source_sha,
                        has_text_objects=bool(text_blocks),
                        has_font_objects=any(span.get("font") for span in text_spans),
                        has_image_objects=bool(images),
                        full_page_raster_likelihood=round(image_coverage, 8),
                        native_text_char_count=len(text),
                        native_text_block_count=len(text_blocks),
                        native_text_bbox_coverage=round(
                            min(text_bbox_area / page_area, 1.0), 8
                        ),
                        unicode_valid_ratio=quality[0],
                        replacement_char_ratio=quality[1],
                        control_char_ratio=quality[2],
                        suspicious_glyph_ratio=quality[3],
                        native_image_count=len(images),
                        extraction_order_available=bool(text_blocks),
                        geometry_available=geometry,
                        profile_state=state,
                        reliability_reasons=reasons,
                    )
                )
        finally:
            document.close()
        return profiles
