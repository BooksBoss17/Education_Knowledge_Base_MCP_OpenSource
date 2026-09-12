"""Dual Paddle evidence contracts for the unpublished BeMarkdown PDF pipeline.

The integrated PaddleOCR-VL result remains the draft-content candidate while
specialist Paddle models provide independent evidence.  Disagreements are
preserved and escalated; this module never performs majority-vote rewriting.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

from .pdf.pipeline_mode import (  # noqa: F401 - frozen public import surface
    PDFPipelineMode,
    resolve_pdf_pipeline_mode,
)

PADDLE_VL_PAGE_SCHEMA = "bemarkdown-paddle-vl-page-ir-v1"
PADDLE_VL_BLOCK_SCHEMA = "bemarkdown-paddle-vl-block-ir-v1"
SPECIALIST_EVIDENCE_SCHEMA = "bemarkdown-specialist-evidence-ir-v1"
ALIGNED_EVIDENCE_SCHEMA = "bemarkdown-aligned-evidence-set-v1"
RISK_MANIFEST_SCHEMA = "bemarkdown-risk-manifest-v1"
PADDLE_UNIFIED_DOCUMENT_SCHEMA = "bemarkdown-paddle-unified-document-ir-v1"
DUAL_EVIDENCE_HANDOFF_SCHEMA = "bemarkdown-dual-evidence-handoff-v1"

EVIDENCE_TYPES = {
    "LAYOUT_EVIDENCE",
    "TEXT_EVIDENCE",
    "FORMULA_EVIDENCE",
    "TABLE_EVIDENCE",
}
KNOWN_VL_LABELS = {
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vision_footnote",
}
TEXT_LABELS = {
    "abstract",
    "algorithm",
    "aside_text",
    "content",
    "doc_title",
    "figure_title",
    "footer",
    "footnote",
    "header",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "text",
    "vision_footnote",
}
FORMULA_LABELS = {"display_formula", "inline_formula", "formula", "formula_number"}
TABLE_LABELS = {"table"}
VISUAL_ASSET_LABELS = {"image", "chart", "seal", "header_image", "footer_image"}
SEVERITY_RANK = {"SAFE": 0, "WATCH": 1, "HIGH": 2, "CRITICAL": 3}

class EvidenceRunMode(str, Enum):
    FULL_EVIDENCE = "FULL_EVIDENCE"
    ADAPTIVE_EVIDENCE = "ADAPTIVE_EVIDENCE"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: Any, length: int = 24) -> str:
    return f"{prefix}-{semantic_sha256(value)[:length]}"


def _require_source_authority(source: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "document_id",
        "page_id",
        "page_index",
        "source_pdf_sha256",
        "canonical_render_sha256",
        "canonical_render_ref",
        "page_width_pt",
        "page_height_pt",
    }
    missing = sorted(required.difference(source))
    if missing:
        raise ValueError(f"SOURCE_AUTHORITY_FIELDS_MISSING:{missing}")
    if len(str(source["source_pdf_sha256"])) != 64:
        raise ValueError("SOURCE_PDF_SHA256_INVALID")
    if len(str(source["canonical_render_sha256"])) != 64:
        raise ValueError("CANONICAL_RENDER_SHA256_INVALID")
    return {key: source[key] for key in sorted(required)}


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _bbox(value: Any) -> list[float]:
    if not _valid_bbox(value):
        raise ValueError(f"INVALID_EVIDENCE_BBOX:{value!r}")
    return [round(float(item), 6) for item in value]


def _intersection(first: Sequence[float], second: Sequence[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _area(value: Sequence[float]) -> float:
    return max(0.0, value[2] - value[0]) * max(0.0, value[3] - value[1])


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    overlap = _intersection(first, second)
    union = _area(first) + _area(second) - overlap
    return overlap / union if union else 0.0


def bbox_overlap_smaller(first: Sequence[float], second: Sequence[float]) -> float:
    overlap = _intersection(first, second)
    smaller = min(_area(first), _area(second))
    return overlap / smaller if smaller else 0.0


def official_paddle_model_inventory() -> dict[str, Any]:
    """Declare the verified Paddle 3.7 architecture without conflating roles."""

    return {
        "schema": "bemarkdown-paddle-model-authority-v1",
        "integrated_pipeline": "PaddleOCR-VL-1.6",
        "orientation_correction": False,
        "document_unwarping": False,
        "official_markdown_is_bemarkdown_authority": False,
        "models": [
            {
                "model_name": "PaddleOCR-VL-1.6",
                "role": "INTEGRATED_FULL_PIPELINE",
                "input": "PDF_OR_IMAGE",
                "output": "PaddleOCRVLResult_WITH_ParsingResult",
                "confidence": "LAYOUT_ONLY",
                "bbox": True,
                "reading_order": True,
            },
            {
                "model_name": "PP-DocLayoutV3",
                "role": "INTEGRATED_LAYOUT",
                "input": "PAGE_IMAGE",
                "output": "LAYOUT_BLOCKS",
                "confidence": True,
                "bbox": True,
                "reading_order": "PIPELINE_DERIVED",
            },
            {
                "model_name": "PaddleOCR-VL-1.6-0.9B",
                "role": "INTEGRATED_REGION_RECOGNITION",
                "input": "LAYOUT_REGION_IMAGE",
                "output": "STRUCTURED_BLOCK_CONTENT",
                "confidence": False,
                "bbox": "INHERITED_FROM_LAYOUT",
                "reading_order": "INHERITED_FROM_PIPELINE",
            },
            {
                "model_name": "PP-DocLayout_plus-L",
                "role": "SPECIALIST_LAYOUT_EVIDENCE",
                "input": "CANONICAL_PAGE_IMAGE",
                "output": "LAYOUT_EVIDENCE",
                "confidence": True,
                "bbox": True,
                "reading_order": "BEMARKDOWN_DERIVED",
            },
            {
                "model_name": "PP-OCRv6 medium det/rec",
                "role": "SPECIALIST_TEXT_EVIDENCE",
                "input": "PAGE_OR_REGION_IMAGE",
                "output": "TEXT_EVIDENCE",
                "confidence": "DET_AND_REC",
                "bbox": True,
                "reading_order": False,
            },
            {
                "model_name": "PP-FormulaNet_plus-L",
                "role": "SPECIALIST_FORMULA_EVIDENCE",
                "input": "FORMULA_REGION_IMAGE",
                "output": "LATEX_EVIDENCE",
                "confidence": False,
                "bbox": "SOURCE_REGION",
                "reading_order": False,
            },
            {
                "model_name": "Table recognition v2 / TableEngine",
                "role": "SPECIALIST_TABLE_EVIDENCE",
                "input": "TABLE_REGION_IMAGE_AND_OCR",
                "output": "TABLE_STRUCTURE_EVIDENCE",
                "confidence": "COMPONENT_DEPENDENT",
                "bbox": True,
                "reading_order": False,
            },
        ],
    }


def validate_paddlevl_pipeline_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the official full 1.6 pipeline contract is present."""

    submodules = config.get("SubModules", {})
    layout = submodules.get("LayoutDetection", {})
    recognition = submodules.get("VLRecognition", {})
    errors = []
    if config.get("use_doc_preprocessor") is not False:
        errors.append("DUPLICATE_DOCUMENT_PREPROCESSOR_NOT_DISABLED")
    if config.get("use_layout_detection") is not True:
        errors.append("FULL_PIPELINE_LAYOUT_NOT_ENABLED")
    if layout.get("model_name") != "PP-DocLayoutV3":
        errors.append("INTEGRATED_LAYOUT_MODEL_MISMATCH")
    if float(layout.get("threshold", -1)) != 0.3:
        errors.append("OFFICIAL_LAYOUT_THRESHOLD_MISMATCH")
    if recognition.get("model_name") != "PaddleOCR-VL-1.6-0.9B":
        errors.append("INTEGRATED_VL_MODEL_MISMATCH")
    backend = recognition.get("genai_config", {}).get("backend")
    if backend != "native":
        errors.append("INTEGRATED_VL_BACKEND_MISMATCH")
    if errors:
        raise ValueError(f"PADDLEOCR_VL_1_6_AUTHORITY_INVALID:{errors}")
    return {
        "pipeline": "PaddleOCR-VL-1.6",
        "layout_model": layout["model_name"],
        "layout_threshold": float(layout["threshold"]),
        "recognition_model": recognition["model_name"],
        "backend": backend,
        "orientation_correction": False,
        "document_unwarping": False,
        "full_pipeline": True,
    }


def build_source_asset_ref(
    *, asset_sha256: str, source_ref: str, source_bbox_pdf_pt: Sequence[float]
) -> dict[str, Any]:
    """Build a source-owned visual reference; descriptions never replace it."""

    if not re.fullmatch(r"[0-9a-f]{64}", asset_sha256):
        raise ValueError("ASSET_SHA256_INVALID")
    return {
        "asset_uid": f"asset-sha256-{asset_sha256}",
        "asset_sha256": asset_sha256,
        "source_ref": str(source_ref),
        "source_bbox_pdf_pt": _bbox(source_bbox_pdf_pt),
        "vl_description_is_authoritative": False,
    }


def _result_body(official_result: Mapping[str, Any]) -> Mapping[str, Any]:
    body = official_result.get("res", official_result)
    if not isinstance(body, Mapping):
        raise TypeError("PADDLEOCR_VL_STRUCTURED_RESULT_REQUIRED")
    if not isinstance(body.get("parsing_res_list"), list):
        raise TypeError("PADDLEOCR_VL_PARSING_RESULT_REQUIRED")
    return body


def _layout_score(block: Mapping[str, Any], layout_result: Any) -> float | None:
    if not isinstance(layout_result, Mapping):
        return None
    boxes = layout_result.get("boxes")
    if not isinstance(boxes, list):
        return None
    target_bbox = block.get("block_bbox")
    target_label = str(block.get("block_label") or "")
    candidates = []
    for row in boxes:
        if not isinstance(row, Mapping):
            continue
        coordinate = row.get("coordinate") or row.get("bbox")
        if not _valid_bbox(coordinate):
            continue
        score = row.get("score")
        if not isinstance(score, (int, float)):
            continue
        label_match = str(row.get("label") or "") == target_label
        overlap = (
            bbox_iou(_bbox(target_bbox), _bbox(coordinate))
            if _valid_bbox(target_bbox)
            else 0
        )
        candidates.append((label_match, overlap, float(score)))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return round(best[2], 8)


def build_paddle_vl_page_ir(
    *,
    source_authority: Mapping[str, Any],
    official_result: Mapping[str, Any],
    raw_result_ref: str,
    pipeline_fingerprint: str,
    layout_model_fingerprint: str,
    vl_model_fingerprint: str,
) -> dict[str, Any]:
    source = _require_source_authority(source_authority)
    body = _result_body(official_result)
    render_width = float(body.get("width") or 0)
    render_height = float(body.get("height") or 0)
    if render_width <= 0 or render_height <= 0:
        raise ValueError("PADDLEOCR_VL_RESULT_GEOMETRY_REQUIRED")
    scale_x = float(source["page_width_pt"]) / render_width
    scale_y = float(source["page_height_pt"]) / render_height
    blocks = []
    for raw_index, raw in enumerate(body["parsing_res_list"]):
        if not isinstance(raw, Mapping):
            raise TypeError(f"PADDLEOCR_VL_BLOCK_NOT_MAPPING:{raw_index}")
        render_bbox = _bbox(raw.get("block_bbox"))
        pdf_bbox = [
            round(render_bbox[0] * scale_x, 6),
            round(render_bbox[1] * scale_y, 6),
            round(render_bbox[2] * scale_x, 6),
            round(render_bbox[3] * scale_y, 6),
        ]
        label = str(raw.get("block_label") or "unknown")
        identity = {
            "schema": PADDLE_VL_BLOCK_SCHEMA,
            "document_id": source["document_id"],
            "page_id": source["page_id"],
            "source_bbox_pdf_pt": pdf_bbox,
            "label": label,
            "official_block_id": raw.get("block_id", raw_index),
            "source_occurrence_index": raw_index,
        }
        blocks.append(
            {
                "schema": PADDLE_VL_BLOCK_SCHEMA,
                "document_id": source["document_id"],
                "page_id": source["page_id"],
                "vl_block_id": _stable_id("paddlevl-block", identity),
                "bbox_render_px": render_bbox,
                "bbox_pdf_pt": pdf_bbox,
                "label": label,
                "content": raw.get("block_content"),
                "block_order": raw.get("block_order"),
                "official_block_id": raw.get("block_id", raw_index),
                "official_group_id": raw.get("group_id"),
                "source_occurrence_index": raw_index,
                "layout_score": _layout_score(raw, body.get("layout_det_res")),
                "source_refs": [source["canonical_render_ref"], raw_result_ref],
                "pipeline_fingerprint": pipeline_fingerprint,
                "layout_model_fingerprint": layout_model_fingerprint,
                "vl_model_fingerprint": vl_model_fingerprint,
                "unknown_label_preserved": label not in KNOWN_VL_LABELS,
                "raw_block": dict(raw),
            }
        )
    return {
        "schema": PADDLE_VL_PAGE_SCHEMA,
        "document_id": source["document_id"],
        "page_id": source["page_id"],
        "page_index": int(source["page_index"]),
        "source_authority": source,
        "blocks": blocks,
        "model_settings": dict(body.get("model_settings") or {}),
        "pipeline_fingerprints": {
            "PaddleOCR-VL-1.6": pipeline_fingerprint,
            "PP-DocLayoutV3": layout_model_fingerprint,
            "PaddleOCR-VL-1.6-0.9B": vl_model_fingerprint,
        },
        "provenance": {
            "raw_result_ref": raw_result_ref,
            "official_structured_result_used": True,
            "official_markdown_used_as_authority": False,
            "orientation_correction": False,
            "document_unwarping": False,
        },
    }


def build_specialist_evidence_ir(
    *,
    source_authority: Mapping[str, Any],
    evidence_type: str,
    bbox: Sequence[float],
    content: Any,
    confidence: float | Mapping[str, float] | None,
    model_name: str,
    model_fingerprint: str,
    source_refs: Sequence[str],
    validator_state: str | Mapping[str, Any],
    reading_order: int | None = None,
    source_occurrence_id: str | int | None = None,
) -> dict[str, Any]:
    source = _require_source_authority(source_authority)
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"UNKNOWN_SPECIALIST_EVIDENCE_TYPE:{evidence_type}")
    source_bbox = _bbox(bbox)
    identity = {
        "schema": SPECIALIST_EVIDENCE_SCHEMA,
        "document_id": source["document_id"],
        "page_id": source["page_id"],
        "evidence_type": evidence_type,
        "bbox_pdf_pt": source_bbox,
        "model_name": model_name,
        "model_fingerprint": model_fingerprint,
        "source_refs": sorted(str(value) for value in source_refs),
        "source_occurrence_id": (
            str(source_occurrence_id) if source_occurrence_id is not None else None
        ),
        "content": content,
        "reading_order": reading_order,
    }
    return {
        "schema": SPECIALIST_EVIDENCE_SCHEMA,
        "document_id": source["document_id"],
        "page_id": source["page_id"],
        "page_index": int(source["page_index"]),
        "evidence_id": _stable_id("specialist-evidence", identity),
        "evidence_type": evidence_type,
        "bbox_pdf_pt": source_bbox,
        "content": content,
        "confidence": confidence,
        "model_name": model_name,
        "model_fingerprint": model_fingerprint,
        "source_refs": [str(value) for value in source_refs],
        "validator_state": validator_state,
        "reading_order": reading_order,
        "source_occurrence_id": (
            str(source_occurrence_id) if source_occurrence_id is not None else None
        ),
        "source_authority_refs": {
            "source_pdf_sha256": source["source_pdf_sha256"],
            "canonical_render_sha256": source["canonical_render_sha256"],
        },
    }


_PUNCTUATION_TRANSLATION = str.maketrans(
    {"，": ",", "。": ".", "：": ":", "；": ";", "（": "(", "）": ")"}
)


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(_PUNCTUATION_TRANSLATION)
    return " ".join(text.split()).strip()


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", _normalized_text(value))


def _numeric_tokens(value: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", value)


def _unit_tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:/[A-Za-z]+)(?:\^?\d+|[²³])?", value)


def _confidence_low(value: Any, threshold: float = 0.6) -> bool:
    if isinstance(value, (int, float)):
        return float(value) < threshold
    if isinstance(value, Mapping):
        numbers = [
            float(item) for item in value.values() if isinstance(item, (int, float))
        ]
        return bool(numbers) and min(numbers) < threshold
    return False


def _text_granularity_containment(first: str, second: str) -> bool:
    first_compact = _compact_text(first)
    second_compact = _compact_text(second)
    short, long = sorted((first_compact, second_compact), key=len)
    if not short or short == long:
        return False
    critical_neighbor = re.compile(r"[A-Za-z0-9²³₀-₉/^_]")
    start = 0
    while (position := long.find(short, start)) >= 0:
        before = long[position - 1] if position else ""
        after_position = position + len(short)
        after = long[after_position] if after_position < len(long) else ""
        if not critical_neighbor.fullmatch(before) and not critical_neighbor.fullmatch(
            after
        ):
            return True
        start = position + 1
    return False


def compare_text_evidence(
    vl_content: Any, specialist_content: Any, *, specialist_confidence: Any
) -> dict[str, Any]:
    vl = _normalized_text(vl_content)
    specialist = _normalized_text(specialist_content)
    conflicts: list[str] = []
    if not vl or not specialist:
        state = "TEXT_MISSING"
        conflicts.append(state)
    elif vl == specialist:
        state = "TEXT_MATCH"
    elif _compact_text(vl) == _compact_text(specialist):
        state = "TEXT_FORMAT_VARIATION"
        conflicts.append(state)
    elif _text_granularity_containment(vl, specialist):
        state = "TEXT_GRANULARITY_VARIATION"
        conflicts.append(state)
    elif _numeric_tokens(vl) != _numeric_tokens(specialist):
        state = "TEXT_NUMBER_CONFLICT"
        conflicts.append(state)
    elif _unit_tokens(vl) != _unit_tokens(specialist):
        state = "TEXT_UNIT_CONFLICT"
        conflicts.append(state)
    elif re.sub(r"[\w\s]", "", vl) != re.sub(r"[\w\s]", "", specialist):
        state = "TEXT_SYMBOL_CONFLICT"
        conflicts.append(state)
    else:
        state = "TEXT_CHARACTER_CONFLICT"
        conflicts.append(state)
    if _confidence_low(specialist_confidence):
        conflicts.append("TEXT_LOW_CONFIDENCE")
    return {
        "state": state,
        "conflict_types": list(dict.fromkeys(conflicts)),
        "vl_normalized": vl,
        "specialist_normalized": specialist,
        "authoritative_rewrite": None,
    }


def _formula_compact(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))


def _formula_minor(value: str) -> str:
    return (
        value.replace(r"\left", "")
        .replace(r"\right", "")
        .replace("{", "")
        .replace("}", "")
    )


def _validator_passed(value: Any) -> bool:
    if isinstance(value, str):
        return value == "PASS"
    if isinstance(value, Mapping):
        states = [item for item in value.values() if isinstance(item, str)]
        return bool(states) and all(item == "PASS" for item in states)
    return False


def compare_formula_evidence(
    vl_content: Any, specialist_content: Any, *, validator_state: Any
) -> dict[str, Any]:
    vl = _formula_compact(vl_content)
    specialist = _formula_compact(specialist_content)
    conflicts: list[str] = []
    if not vl or not specialist:
        state = "FORMULA_MISSING"
        conflicts.append(state)
    elif vl == specialist:
        state = "FORMULA_MATCH"
    elif _formula_minor(vl) == _formula_minor(specialist):
        state = "FORMULA_MINOR_SERIALIZATION_VARIATION"
        conflicts.append(state)
    else:
        vl_sub = re.findall(r"_(?:\{([^}]*)\}|([A-Za-z0-9ν]+))", vl)
        sp_sub = re.findall(r"_(?:\{([^}]*)\}|([A-Za-z0-9ν]+))", specialist)
        vl_super = re.findall(r"\^(?:\{([^}]*)\}|([A-Za-z0-9]+))", vl)
        sp_super = re.findall(r"\^(?:\{([^}]*)\}|([A-Za-z0-9]+))", specialist)
        if vl_sub != sp_sub:
            state = "FORMULA_SUBSCRIPT_CONFLICT"
        elif vl_super != sp_super:
            state = "FORMULA_SUPERSCRIPT_CONFLICT"
        elif any(
            (left in vl and right in specialist) or (right in vl and left in specialist)
            for left, right in (("v", "ν"), ("0", "O"), ("1", "l"))
        ):
            state = "FORMULA_GLYPH_CONFLICT"
        else:
            state = "FORMULA_TOKEN_CONFLICT"
        conflicts.append(state)
    if not _validator_passed(validator_state):
        conflicts.append("FORMULA_VALIDATION_FAILURE")
    return {
        "state": state,
        "conflict_types": list(dict.fromkeys(conflicts)),
        "severity_floor": "HIGH" if conflicts else "SAFE",
        "authoritative_rewrite": None,
    }


def compare_embedded_formula_evidence(
    vl_content: Any, specialist_content: Any, *, validator_state: Any
) -> dict[str, Any]:
    vl = _formula_minor(_formula_compact(vl_content))
    specialist = _formula_minor(_formula_compact(specialist_content))
    if vl and specialist and specialist in vl:
        conflicts = []
        if not _validator_passed(validator_state):
            conflicts.append("FORMULA_VALIDATION_FAILURE")
        return {
            "state": "FORMULA_MATCH",
            "conflict_types": conflicts,
            "severity_floor": "HIGH" if conflicts else "SAFE",
            "embedded_in_text_block": True,
            "authoritative_rewrite": None,
        }
    result = compare_formula_evidence(
        vl_content, specialist_content, validator_state=validator_state
    )
    result["embedded_in_text_block"] = True
    return result


def _flatten_table_cells(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    flattened = []
    for row in value:
        if isinstance(row, (list, tuple)):
            flattened.extend(_normalized_text(cell) for cell in row)
        else:
            flattened.append(_normalized_text(row))
    return flattened


def _table_from_html(value: str) -> dict[str, Any] | None:
    row_html = re.findall(
        r"<tr\b[^>]*>(.*?)</tr>", value, flags=re.IGNORECASE | re.DOTALL
    )
    if not row_html:
        return None
    row_cells = [
        re.findall(
            r"<t[dh]\b[^>]*>(.*?)</t[dh]>",
            row,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for row in row_html
    ]
    columns = max((len(row) for row in row_cells), default=0)
    if columns == 0:
        return None
    cells = [
        _normalized_text(html.unescape(re.sub(r"<[^>]+>", "", cell)))
        for row in row_cells
        for cell in row
    ]
    merged_cells = len(
        re.findall(
            r"\b(?:rowspan|colspan)\s*=\s*['\"]?(?!1\b)\d+",
            value,
            flags=re.IGNORECASE,
        )
    )
    return {
        "rows": len(row_cells),
        "columns": columns,
        "merged_cells": merged_cells,
        "cell_count": len(cells),
        "cells": cells,
    }


def _normalized_table(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        return _table_from_html(value)
    if not isinstance(value, Mapping):
        return None
    rows = value.get("rows")
    columns = value.get("columns")
    merged_cells = value.get("merged_cells")
    cells = _flatten_table_cells(value.get("cells"))
    cell_count = value.get("cell_count")
    if cell_count is None and cells:
        cell_count = len(cells)
    if rows is not None and columns is not None:
        return {
            "rows": rows,
            "columns": columns,
            "merged_cells": merged_cells,
            "cell_count": cell_count,
            "cells": cells,
        }
    structure = value.get("structure")
    if isinstance(structure, Mapping):
        structure = structure.get("structure")
    if isinstance(structure, (list, tuple)):
        parsed = _table_from_html("".join(str(token) for token in structure))
        if parsed is not None:
            if cell_count is not None:
                parsed["cell_count"] = cell_count
            if cells:
                parsed["cells"] = cells
            return parsed
    return None


def _table_shape(value: Any) -> tuple[Any, Any, Any, Any] | None:
    normalized = _normalized_table(value)
    if normalized is None:
        return None
    return (
        normalized.get("rows"),
        normalized.get("columns"),
        normalized.get("merged_cells"),
        normalized.get("cell_count"),
    )


def compare_table_evidence(
    vl_content: Any, specialist_content: Any, *, validator_state: Any
) -> dict[str, Any]:
    from .pdf.table_evidence_v2 import compare_table_evidence_v2

    conflicts: list[str] = []
    vl_table = _normalized_table(vl_content)
    specialist_table = _normalized_table(specialist_content)
    if not vl_content or not specialist_content:
        state = "TABLE_MISSING"
    elif (
        vl_table is None
        or specialist_table is None
        or _table_shape(vl_table) != _table_shape(specialist_table)
    ):
        state = "TABLE_STRUCTURE_CONFLICT"
    elif vl_table["cells"] and specialist_table["cells"]:
        v2_state = compare_table_evidence_v2(vl_content, specialist_content)["state"]
        if v2_state == "TABLE_CELL_POSITION_CONFLICT":
            state = v2_state
        elif v2_state == "TABLE_MATCH":
            state = "TABLE_MATCH"
        else:
            state = "TABLE_CELL_CONFLICT"
    else:
        state = "TABLE_MATCH"
    if state != "TABLE_MATCH":
        conflicts.append(state)
    if not _validator_passed(validator_state):
        conflicts.append("TABLE_VALIDATION_FAILURE")
    return {
        "state": state,
        "conflict_types": list(dict.fromkeys(conflicts)),
        "authoritative_rewrite": None,
    }


def _expected_type(label: str) -> str | None:
    if label in TEXT_LABELS:
        return "TEXT_EVIDENCE"
    if label in FORMULA_LABELS:
        return "FORMULA_EVIDENCE"
    if label in TABLE_LABELS:
        return "TABLE_EVIDENCE"
    return None


def _labels_compatible(first: str, second: str) -> bool:
    if first == second:
        return True
    formula_aliases = {"display_formula", "inline_formula", "formula"}
    return first in formula_aliases and second in formula_aliases


def _granularity_variation(first: str, second: str, overlap_smaller: float) -> bool:
    if overlap_smaller < 0.8:
        return False
    embedded_formula = {first, second} & FORMULA_LABELS
    containing_text = {first, second} & TEXT_LABELS
    return bool(embedded_formula and containing_text)


class MultiEvidenceAligner:
    def __init__(self, *, minimum_iou: float = 0.1, bbox_conflict_iou: float = 0.5):
        if not 0 < minimum_iou <= bbox_conflict_iou <= 1:
            raise ValueError("INVALID_ALIGNMENT_THRESHOLDS")
        self.minimum_iou = minimum_iou
        self.bbox_conflict_iou = bbox_conflict_iou

    def align(
        self, page: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        blocks = [dict(row) for row in page.get("blocks", [])]
        specialists = [dict(row) for row in evidence]
        block_id_occurrences = [str(row["vl_block_id"]) for row in blocks]
        if len(set(block_id_occurrences)) != len(block_id_occurrences):
            raise ValueError("DUPLICATE_VL_BLOCK_ID")
        specialist_id_occurrences = [str(row["evidence_id"]) for row in specialists]
        if len(set(specialist_id_occurrences)) != len(specialist_id_occurrences):
            raise ValueError("DUPLICATE_SPECIALIST_EVIDENCE_ID")
        if any(row.get("page_id") != page.get("page_id") for row in specialists):
            raise ValueError("CROSS_PAGE_EVIDENCE_FORBIDDEN")
        assigned: dict[str, list[dict[str, Any]]] = {
            str(block["vl_block_id"]): [] for block in blocks
        }
        unmatched = []
        for specialist in specialists:
            ranked = sorted(
                (
                    (
                        max(
                            bbox_iou(block["bbox_pdf_pt"], specialist["bbox_pdf_pt"]),
                            bbox_overlap_smaller(
                                block["bbox_pdf_pt"], specialist["bbox_pdf_pt"]
                            ),
                        ),
                        bbox_iou(block["bbox_pdf_pt"], specialist["bbox_pdf_pt"]),
                        bbox_overlap_smaller(
                            block["bbox_pdf_pt"], specialist["bbox_pdf_pt"]
                        ),
                        block,
                    )
                    for block in blocks
                ),
                key=lambda item: (item[0], item[1], str(item[3]["vl_block_id"])),
                reverse=True,
            )
            if ranked and ranked[0][0] >= self.minimum_iou:
                assigned[str(ranked[0][3]["vl_block_id"])].append(
                    {
                        **specialist,
                        "alignment_iou": round(ranked[0][1], 8),
                        "alignment_overlap_smaller": round(ranked[0][2], 8),
                    }
                )
            else:
                unmatched.append(specialist)

        units = [
            self._unit(page, block, assigned[str(block["vl_block_id"])])
            for block in blocks
        ]
        for specialist in unmatched:
            units.append(self._specialist_only_unit(page, specialist))
        self._apply_reading_order_conflicts(units)
        output_id_occurrences = [
            str(row["evidence_id"])
            for unit in units
            for row in unit["specialist_evidence"]
        ]
        if len(set(output_id_occurrences)) != len(output_id_occurrences):
            raise ValueError("DUPLICATE_OUTPUT_SPECIALIST_EVIDENCE_ID")
        output_ids = set(output_id_occurrences)
        input_ids = set(specialist_id_occurrences)
        vl_output_id_occurrences = [
            str(unit["vl_block"]["vl_block_id"])
            for unit in units
            if unit["vl_block"] is not None
        ]
        if len(set(vl_output_id_occurrences)) != len(vl_output_id_occurrences):
            raise ValueError("DUPLICATE_OUTPUT_VL_BLOCK_ID")
        vl_output_ids = set(vl_output_id_occurrences)
        vl_input_ids = set(block_id_occurrences)
        return {
            "schema": ALIGNED_EVIDENCE_SCHEMA,
            "document_id": page["document_id"],
            "page_id": page["page_id"],
            "page_index": page["page_index"],
            "source_authority": dict(page["source_authority"]),
            "aligned_units": units,
            "input_accounting": {
                "vl_block_ids": sorted(vl_input_ids),
                "specialist_evidence_ids": sorted(input_ids),
                "output_vl_block_ids": sorted(vl_output_ids),
                "output_specialist_evidence_ids": sorted(output_ids),
                "input_vl_block_count": len(block_id_occurrences),
                "input_specialist_evidence_count": len(specialist_id_occurrences),
                "output_vl_block_count": len(vl_output_id_occurrences),
                "output_specialist_evidence_count": len(output_id_occurrences),
                "silent_drop_count": len(vl_input_ids - vl_output_ids)
                + len(input_ids - output_ids),
            },
            "provenance": {
                "winner_selection": "FORBIDDEN",
                "conflicts_preserved": True,
            },
        }

    def _unit(
        self,
        page: Mapping[str, Any],
        block: Mapping[str, Any],
        specialists: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        conflicts: list[str] = []
        comparisons = []
        expected = _expected_type(str(block["label"]))
        content_matches = [
            row for row in specialists if row["evidence_type"] == expected
        ]
        layout_matches = [
            row for row in specialists if row["evidence_type"] == "LAYOUT_EVIDENCE"
        ]
        for row in specialists:
            if (
                float(row["alignment_iou"]) < self.bbox_conflict_iou
                and float(row["alignment_overlap_smaller"]) < 0.8
            ):
                conflicts.append("LAYOUT_BBOX_CONFLICT")
            if row["evidence_type"] == "LAYOUT_EVIDENCE":
                specialist_label = (
                    row["content"].get("label")
                    if isinstance(row.get("content"), Mapping)
                    else row.get("content")
                )
                if _labels_compatible(str(block["label"]), str(specialist_label)):
                    state = "LAYOUT_MATCH"
                elif _granularity_variation(
                    str(block["label"]),
                    str(specialist_label),
                    float(row["alignment_overlap_smaller"]),
                ):
                    state = "LAYOUT_GRANULARITY_VARIATION"
                else:
                    state = "LAYOUT_LABEL_CONFLICT"
                if state != "LAYOUT_MATCH":
                    conflicts.append(state)
                comparisons.append({"evidence_id": row["evidence_id"], "state": state})
            elif (
                row["evidence_type"] == "TEXT_EVIDENCE" and expected == "TEXT_EVIDENCE"
            ):
                comparison = compare_text_evidence(
                    block.get("content"),
                    row.get("content"),
                    specialist_confidence=row.get("confidence"),
                )
                conflicts.extend(comparison["conflict_types"])
                comparisons.append({"evidence_id": row["evidence_id"], **comparison})
            elif row["evidence_type"] == "FORMULA_EVIDENCE" and expected in {
                "FORMULA_EVIDENCE",
                "TEXT_EVIDENCE",
            }:
                comparator = (
                    compare_formula_evidence
                    if expected == "FORMULA_EVIDENCE"
                    else compare_embedded_formula_evidence
                )
                comparison = comparator(
                    block.get("content"),
                    row.get("content"),
                    validator_state=row.get("validator_state"),
                )
                conflicts.extend(comparison["conflict_types"])
                comparisons.append({"evidence_id": row["evidence_id"], **comparison})
            elif (
                row["evidence_type"] == "TABLE_EVIDENCE"
                and expected == "TABLE_EVIDENCE"
            ):
                comparison = compare_table_evidence(
                    block.get("content"),
                    row.get("content"),
                    validator_state=row.get("validator_state"),
                )
                conflicts.extend(comparison["conflict_types"])
                comparisons.append({"evidence_id": row["evidence_id"], **comparison})
        coverage = "COVERAGE_COMPLETE"
        if not layout_matches:
            coverage = "POSSIBLE_MISSING_REGION_MAIN"
            conflicts.append(coverage)
        if expected is not None and not content_matches:
            coverage = "POSSIBLE_MISSING_REGION_MAIN"
            conflicts.append(coverage)
        conflicts = list(dict.fromkeys(conflicts))
        identity = {
            "page_id": page["page_id"],
            "vl_block_id": block["vl_block_id"],
            "specialist_evidence_ids": sorted(
                row["evidence_id"] for row in specialists
            ),
        }
        return {
            "aligned_unit_id": _stable_id("aligned-unit", identity),
            "page_id": page["page_id"],
            "candidate_source_bbox": list(block["bbox_pdf_pt"]),
            "vl_block": dict(block),
            "specialist_evidence": [dict(row) for row in specialists],
            "agreement_state": "CONFLICT" if conflicts else "AGREEMENT",
            "conflict_types": conflicts,
            "confidence_summary": [row.get("confidence") for row in specialists],
            "validator_summary": [row.get("validator_state") for row in specialists],
            "coverage_state": coverage,
            "comparisons": comparisons,
            "automatic_winner_selected": False,
        }

    def _specialist_only_unit(
        self, page: Mapping[str, Any], specialist: Mapping[str, Any]
    ) -> dict[str, Any]:
        identity = {
            "page_id": page["page_id"],
            "specialist_evidence_id": specialist["evidence_id"],
            "vl_block_id": None,
        }
        return {
            "aligned_unit_id": _stable_id("aligned-unit", identity),
            "page_id": page["page_id"],
            "candidate_source_bbox": list(specialist["bbox_pdf_pt"]),
            "vl_block": None,
            "specialist_evidence": [dict(specialist)],
            "agreement_state": "CONFLICT",
            "conflict_types": ["POSSIBLE_MISSING_REGION_VL"],
            "confidence_summary": [specialist.get("confidence")],
            "validator_summary": [specialist.get("validator_state")],
            "coverage_state": "POSSIBLE_MISSING_REGION_VL",
            "comparisons": [],
            "automatic_winner_selected": False,
        }

    @staticmethod
    def _apply_reading_order_conflicts(units: list[dict[str, Any]]) -> None:
        ordered = []
        for unit in units:
            block = unit["vl_block"]
            evidence = [
                row
                for row in unit["specialist_evidence"]
                if row.get("reading_order") is not None
            ]
            if block is not None and block.get("block_order") is not None and evidence:
                ordered.append(
                    (unit, int(block["block_order"]), int(evidence[0]["reading_order"]))
                )
        if len(ordered) < 2:
            return
        vl_sequence = [
            unit[0]["aligned_unit_id"]
            for unit in sorted(ordered, key=lambda row: row[1])
        ]
        specialist_sequence = [
            unit[0]["aligned_unit_id"]
            for unit in sorted(ordered, key=lambda row: row[2])
        ]
        if vl_sequence != specialist_sequence:
            vl_position = {unit_id: index for index, unit_id in enumerate(vl_sequence)}
            specialist_position = {
                unit_id: index for index, unit_id in enumerate(specialist_sequence)
            }
            for unit, _, _ in ordered:
                unit_id = unit["aligned_unit_id"]
                if vl_position[unit_id] == specialist_position[unit_id]:
                    continue
                unit["conflict_types"] = list(
                    dict.fromkeys([*unit["conflict_types"], "READING_ORDER_CONFLICT"])
                )
                unit["agreement_state"] = "CONFLICT"


def _risk_severity(unit: Mapping[str, Any]) -> str:
    conflicts = set(unit.get("conflict_types", []))
    critical = {
        "POSSIBLE_MISSING_REGION_MAIN",
        "POSSIBLE_MISSING_REGION_VL",
        "FORMULA_VALIDATION_FAILURE",
        "TABLE_VALIDATION_FAILURE",
        "TEXT_MISSING",
        "FORMULA_MISSING",
        "TABLE_MISSING",
    }
    high = {
        "TEXT_CHARACTER_CONFLICT",
        "TEXT_NUMBER_CONFLICT",
        "TEXT_UNIT_CONFLICT",
        "TEXT_SYMBOL_CONFLICT",
        "TEXT_LOW_CONFIDENCE",
        "FORMULA_TOKEN_CONFLICT",
        "FORMULA_SUBSCRIPT_CONFLICT",
        "FORMULA_SUPERSCRIPT_CONFLICT",
        "FORMULA_GLYPH_CONFLICT",
        "TABLE_STRUCTURE_CONFLICT",
        "TABLE_CELL_CONFLICT",
        "LAYOUT_LABEL_CONFLICT",
        "READING_ORDER_CONFLICT",
    }
    watch = {
        "TEXT_FORMAT_VARIATION",
        "TEXT_GRANULARITY_VARIATION",
        "FORMULA_MINOR_SERIALIZATION_VARIATION",
        "LAYOUT_BBOX_CONFLICT",
        "LAYOUT_GRANULARITY_VARIATION",
    }
    if conflicts & critical:
        return "CRITICAL"
    if conflicts & high:
        return "HIGH"
    if conflicts & watch:
        return "WATCH"
    return "SAFE"


_RISK_PRIORITY = [
    "POSSIBLE_MISSING_REGION_MAIN",
    "POSSIBLE_MISSING_REGION_VL",
    "TEXT_MISSING",
    "FORMULA_MISSING",
    "TABLE_MISSING",
    "FORMULA_VALIDATION_FAILURE",
    "TABLE_VALIDATION_FAILURE",
    "TEXT_NUMBER_CONFLICT",
    "TEXT_UNIT_CONFLICT",
    "FORMULA_SUBSCRIPT_CONFLICT",
    "FORMULA_SUPERSCRIPT_CONFLICT",
    "FORMULA_GLYPH_CONFLICT",
    "FORMULA_TOKEN_CONFLICT",
    "TABLE_STRUCTURE_CONFLICT",
    "TABLE_CELL_CONFLICT",
    "READING_ORDER_CONFLICT",
    "LAYOUT_LABEL_CONFLICT",
    "TEXT_CHARACTER_CONFLICT",
    "TEXT_SYMBOL_CONFLICT",
    "TEXT_LOW_CONFIDENCE",
    "LAYOUT_BBOX_CONFLICT",
    "LAYOUT_GRANULARITY_VARIATION",
    "TEXT_FORMAT_VARIATION",
    "TEXT_GRANULARITY_VARIATION",
    "FORMULA_MINOR_SERIALIZATION_VARIATION",
]


def _primary_risk(conflicts: Iterable[str], severity: str) -> str:
    observed = set(conflicts)
    for risk in _RISK_PRIORITY:
        if risk in observed:
            return risk
    return "EVIDENCE_AGREEMENT" if severity == "SAFE" else "UNCLASSIFIED_EVIDENCE_RISK"


class RiskManifestBuilder:
    def build(self, aligned: Mapping[str, Any]) -> dict[str, Any]:
        risks = []
        for unit in aligned.get("aligned_units", []):
            severity = _risk_severity(unit)
            risk_type = _primary_risk(unit.get("conflict_types", []), severity)
            vl = unit.get("vl_block")
            specialists = list(unit.get("specialist_evidence", []))
            specialist_result: Any = [row.get("content") for row in specialists]
            if len(specialist_result) == 1:
                specialist_result = specialist_result[0]
            identity = {
                "schema": RISK_MANIFEST_SCHEMA,
                "document_id": aligned["document_id"],
                "page_id": aligned["page_id"],
                "aligned_unit_id": unit["aligned_unit_id"],
                "risk_type": risk_type,
            }
            risks.append(
                {
                    "risk_id": _stable_id("risk", identity),
                    "document_id": aligned["document_id"],
                    "page_id": aligned["page_id"],
                    "target": unit["aligned_unit_id"],
                    "risk_type": risk_type,
                    "severity": severity,
                    "all_conflict_types": list(unit.get("conflict_types", [])),
                    "source_crop_ref": unit.get("source_crop_ref")
                    or aligned["source_authority"]["canonical_render_ref"],
                    "source_crop_sha256": unit.get("source_crop_sha256"),
                    "source_crop_is_exact": bool(unit.get("source_crop_is_exact")),
                    "source_bbox_pdf_pt": list(unit["candidate_source_bbox"]),
                    "vl_result": vl.get("content") if vl else None,
                    "vl_model_fingerprint": vl.get("vl_model_fingerprint")
                    if vl
                    else None,
                    "specialist_result": specialist_result,
                    "specialist_model_fingerprints": [
                        row.get("model_fingerprint") for row in specialists
                    ],
                    "specialist_confidence": [
                        row.get("confidence") for row in specialists
                    ],
                    "validator_state": [
                        row.get("validator_state") for row in specialists
                    ],
                    "coverage_state": unit["coverage_state"],
                    "recommended_action": (
                        "TARGETED_VISION_REVIEW_IN_PHASE_7B"
                        if severity in {"HIGH", "CRITICAL"}
                        else "LOCAL_ACCEPT"
                        if severity == "SAFE"
                        else "LOCAL_WATCH"
                    ),
                    "automatic_winner_selected": False,
                }
            )
        return {
            "schema": RISK_MANIFEST_SCHEMA,
            "document_id": aligned["document_id"],
            "page_id": aligned["page_id"],
            "risks": risks,
            "summary": {
                severity: sum(row["severity"] == severity for row in risks)
                for severity in SEVERITY_RANK
            },
            "policy": "AGREEMENT_LOWERS_RISK_DISAGREEMENT_NEVER_AUTO_REWRITES",
        }


def build_paddle_unified_document_ir(
    aligned_pages: Sequence[Mapping[str, Any]],
    risk_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not aligned_pages:
        raise ValueError("PADDLE_UNIFIED_DOCUMENT_REQUIRES_PAGES")
    document_ids = {str(row["document_id"]) for row in aligned_pages}
    if len(document_ids) != 1:
        raise ValueError("PADDLE_UNIFIED_DOCUMENT_CROSS_DOCUMENT_INPUT")
    risk_by_target = {
        risk["target"]: risk
        for manifest in risk_manifests
        for risk in manifest.get("risks", [])
    }
    nodes = []
    for page in aligned_pages:
        for unit in page["aligned_units"]:
            vl = unit.get("vl_block")
            specialists = list(unit.get("specialist_evidence", []))
            risk = risk_by_target.get(unit["aligned_unit_id"])
            specialist_layout_labels = [
                str(row["content"].get("label"))
                for row in specialists
                if row.get("evidence_type") == "LAYOUT_EVIDENCE"
                and isinstance(row.get("content"), Mapping)
                and row["content"].get("label") is not None
            ]
            visual_label = next(
                (
                    label
                    for label in [
                        str(vl.get("label")) if vl else None,
                        *specialist_layout_labels,
                    ]
                    if label in VISUAL_ASSET_LABELS
                ),
                None,
            )
            source_asset_ref = unit.get("source_asset_ref")
            if visual_label:
                if not isinstance(source_asset_ref, Mapping) or not re.fullmatch(
                    r"asset-sha256-[0-9a-f]{64}",
                    str(source_asset_ref.get("asset_uid") or ""),
                ):
                    raise ValueError(
                        f"VISUAL_SOURCE_ASSET_REQUIRED:{unit['aligned_unit_id']}"
                    )
                authoritative_content: Any = {
                    "asset_uid": str(source_asset_ref["asset_uid"])
                }
            else:
                authoritative_content = vl.get("content") if vl else None
            node_identity = {
                "document_id": page["document_id"],
                "page_id": page["page_id"],
                "aligned_unit_id": unit["aligned_unit_id"],
            }
            nodes.append(
                {
                    "node_id": _stable_id("paddle-doc-node", node_identity),
                    "page_id": page["page_id"],
                    "page_index": int(page["page_index"]),
                    "bbox_pdf_pt": list(unit["candidate_source_bbox"]),
                    "content_kind": str(
                        visual_label
                        or (vl.get("label") if vl else specialists[0]["evidence_type"])
                    ),
                    "authoritative_content": authoritative_content,
                    "non_authoritative_vl_description": (
                        vl.get("content") if visual_label and vl else None
                    ),
                    "supporting_evidence": specialists,
                    "risk_state": risk["severity"] if risk else "CRITICAL",
                    "review_reasons": list(unit.get("conflict_types", [])),
                    "reading_order_evidence": {
                        "vl_block_order": vl.get("block_order") if vl else None,
                        "specialist_orders": [
                            row["reading_order"]
                            for row in specialists
                            if row.get("reading_order") is not None
                        ],
                        "automatic_reorder_applied": False,
                    },
                    "provenance": {
                        "aligned_unit_id": unit["aligned_unit_id"],
                        "vl_block_id": vl.get("vl_block_id") if vl else None,
                        "specialist_evidence_ids": [
                            row["evidence_id"] for row in specialists
                        ],
                        "automatic_winner_selected": False,
                        "source_pdf_sha256": page["source_authority"][
                            "source_pdf_sha256"
                        ],
                        "asset_uid": (
                            str(source_asset_ref["asset_uid"]) if visual_label else None
                        ),
                    },
                }
            )
    nodes.sort(
        key=lambda row: (
            row["page_index"],
            (
                row["reading_order_evidence"]["vl_block_order"]
                if row["reading_order_evidence"]["vl_block_order"] is not None
                else float("inf")
            ),
            row["bbox_pdf_pt"][1],
            row["node_id"],
        )
    )
    return {
        "schema": PADDLE_UNIFIED_DOCUMENT_SCHEMA,
        "document_id": next(iter(document_ids)),
        "nodes": nodes,
        "source_page_ids": [
            str(row["page_id"])
            for row in sorted(aligned_pages, key=lambda value: value["page_index"])
        ],
        "authority_policy": {
            "draft_candidate": "PaddleOCR-VL-1.6 structured block content",
            "specialist_role": "SUPPORTING_EVIDENCE_AND_RISK",
            "conflict_policy": "PRESERVE_DRAFT_AND_ESCALATE_NO_AUTO_OVERWRITE",
        },
    }


def build_dual_evidence_handoff(
    document: Mapping[str, Any], risk_manifests: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    candidates = [
        risk
        for manifest in risk_manifests
        for risk in manifest.get("risks", [])
        if risk.get("severity") in {"HIGH", "CRITICAL"}
    ]
    return {
        "schema": DUAL_EVIDENCE_HANDOFF_SCHEMA,
        "document_id": document["document_id"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "vision_agent_invoked": False,
        "phase": "PHASE_7A_R1_LOCAL_EVIDENCE_ONLY",
    }


class ModelResidencyPlanner:
    """Produce a sequential GPU residency plan for 8-16 GB target devices."""

    def __init__(self, *, vram_budget_mib: int):
        if vram_budget_mib < 8_000:
            raise ValueError("PHASE7A_REQUIRES_AT_LEAST_8GB_VRAM")
        self.vram_budget_mib = int(vram_budget_mib)

    def plan(self, mode: EvidenceRunMode | str) -> dict[str, Any]:
        selected = EvidenceRunMode(mode)
        if selected is EvidenceRunMode.ADAPTIVE_EVIDENCE:
            return {
                "mode": selected.value,
                "status": "READY",
                "vram_budget_mib": self.vram_budget_mib,
                "stages": [
                    {
                        "stage": stage,
                        "resident_models": models,
                        "unload_before_next": True,
                        "device": "gpu:0",
                        "activation": activation,
                    }
                    for stage, models, activation in [
                        (
                            "SPECIALIST_LAYOUT",
                            ["PP-DocLayout_plus-L"],
                            "CHEAP_ALWAYS_ON",
                        ),
                        (
                            "SELECTIVE_INTEGRATED_VL",
                            ["PP-DocLayoutV3", "PaddleOCR-VL-1.6-0.9B"],
                            "VISUAL_REQUIRED_PAGES",
                        ),
                        (
                            "SPECIALIST_TEXT",
                            ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"],
                            "TRIGGERED_TEXT_ONLY",
                        ),
                        (
                            "SPECIALIST_FORMULA",
                            ["PP-FormulaNet_plus-L"],
                            "TRIGGERED_FORMULAS_ONLY",
                        ),
                        (
                            "SPECIALIST_TABLE",
                            ["Table recognition v2 / TableEngine"],
                            "TABLES_HIGH_COVERAGE",
                        ),
                    ]
                ],
                "all_models_simultaneously_resident": False,
            }
        stages = [
            ("INTEGRATED_VL", ["PP-DocLayoutV3", "PaddleOCR-VL-1.6-0.9B"]),
            ("SPECIALIST_LAYOUT", ["PP-DocLayout_plus-L"]),
            ("SPECIALIST_TEXT", ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"]),
            ("SPECIALIST_FORMULA", ["PP-FormulaNet_plus-L"]),
            ("SPECIALIST_TABLE", ["Table recognition v2 / TableEngine"]),
        ]
        return {
            "mode": selected.value,
            "status": "READY",
            "vram_budget_mib": self.vram_budget_mib,
            "stages": [
                {
                    "stage": stage,
                    "resident_models": models,
                    "unload_before_next": True,
                    "device": "gpu:0",
                }
                for stage, models in stages
            ],
            "all_models_simultaneously_resident": False,
        }


class GpuRuntimeGuard:
    @staticmethod
    def validate(
        *, requested_device: str, compiled_with_cuda: bool, cuda_device_count: int
    ) -> dict[str, Any]:
        if not requested_device.startswith("gpu:"):
            raise RuntimeError("SILENT_CPU_FALLBACK_FORBIDDEN")
        if not compiled_with_cuda or cuda_device_count < 1:
            raise RuntimeError("GPU_RUNTIME_UNAVAILABLE")
        return {
            "device": requested_device,
            "compiled_with_cuda": True,
            "cuda_device_count": int(cuda_device_count),
            "silent_cpu_fallback": False,
        }
