"""Deterministic contracts for the PP OCR error root-cause audit.

This module contains selection, crop-window, accounting, and scoring logic only.
It does not participate in the production OCR pipeline.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .ocr_ensemble_benchmark import normalize_text

ROOT_CAUSES = {
    "CROP_LEFT_CUTOFF",
    "CROP_RIGHT_CUTOFF",
    "CROP_TOP_CUTOFF",
    "CROP_BOTTOM_CUTOFF",
    "CROP_TOO_TIGHT_NO_VISIBLE_CUTOFF",
    "LOW_RESOLUTION",
    "LOW_CHARACTER_HEIGHT",
    "NEIGHBOR_TEXT_CONTAMINATION",
    "MULTILINE_OR_LAYOUT_CONTAMINATION",
    "MIXED_TEXT_FORMULA",
    "SPECIAL_SYMBOL_OR_UNIT",
    "TRUE_PP_REC_ERROR",
    "SOURCE_AMBIGUOUS",
    "OTHER",
}
VISUAL_ADEQUACY = {
    "ORIGINAL_VISUALLY_COMPLETE",
    "ORIGINAL_VISUALLY_INCOMPLETE",
    "ORIGINAL_VISUALLY_AMBIGUOUS",
}
VARIANTS = ("ORIGINAL", "PAD_SMALL", "PAD_MEDIUM")
MODELS = ("P", "G", "S")
PADDING_FACTORS = {
    "PAD_SMALL": {"horizontal": 0.10, "vertical": 0.05},
    "PAD_MEDIUM": {"horizontal": 0.25, "vertical": 0.10},
}


def select_pp_error_ids(
    gold_rows: Sequence[Mapping[str, Any]],
    pp_outputs: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = 13,
) -> list[str]:
    """Return the frozen normalized-exact PP errors in stable ID order."""

    gold = _unique(gold_rows, "crop_id")
    outputs = _unique(pp_outputs, "sample_id")
    native_ids = {
        crop_id for crop_id, row in gold.items() if row.get("status") == "NATIVE_GOLD"
    }
    if not native_ids.issubset(outputs):
        raise ValueError("PP_OUTPUT_NATIVE_GOLD_SAMPLE_SET_MISMATCH")
    errors = sorted(
        crop_id
        for crop_id in native_ids
        if normalize_text(gold[crop_id].get("text"))
        != normalize_text(outputs[crop_id].get("raw_output"))
    )
    if len(errors) != expected_count:
        raise ValueError(
            f"PP_ERROR_INVENTORY_MISMATCH:{len(errors)}:{expected_count}"
        )
    return errors


def padding_bbox(
    bbox: Sequence[float], source_size: Sequence[int], variant: str
) -> dict[str, Any]:
    """Expand a canonical line bbox by height-relative padding and clamp it."""

    if variant not in PADDING_FACTORS:
        raise ValueError(f"UNKNOWN_PADDING_VARIANT:{variant}")
    if len(bbox) != 4 or len(source_size) != 2:
        raise ValueError("INVALID_BBOX_OR_SOURCE_SIZE")
    source_width, source_height = (int(source_size[0]), int(source_size[1]))
    left, top, right, bottom = (float(value) for value in bbox)
    if not (0 <= left < right <= source_width and 0 <= top < bottom <= source_height):
        raise ValueError("BBOX_OUTSIDE_SOURCE")
    height = bottom - top
    factors = PADDING_FACTORS[variant]
    horizontal = _round_pixels(height * factors["horizontal"])
    vertical = _round_pixels(height * factors["vertical"])
    expanded = [
        max(0, math.floor(left) - horizontal),
        max(0, math.floor(top) - vertical),
        min(source_width, math.ceil(right) + horizontal),
        min(source_height, math.ceil(bottom) + vertical),
    ]
    original = [math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)]
    return {
        "variant": variant,
        "bbox_px": expanded,
        "requested_padding_px": {
            "left_px": horizontal,
            "right_px": horizontal,
            "top_px": vertical,
            "bottom_px": vertical,
        },
        "actual_padding_px": {
            "left_px": original[0] - expanded[0],
            "right_px": expanded[2] - original[2],
            "top_px": original[1] - expanded[1],
            "bottom_px": expanded[3] - original[3],
        },
        "clamped": expanded
        != [
            original[0] - horizontal,
            original[1] - vertical,
            original[2] + horizontal,
            original[3] + vertical,
        ],
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_variant_crop_contract(
    manifest: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    expected = _unique(manifest, "variant_id")
    if set(outputs) != set(MODELS):
        raise ValueError("P_G_S_OUTPUT_SET_REQUIRED")
    for model in MODELS:
        actual = _unique(outputs[model], "sample_id")
        if set(actual) != set(expected):
            raise ValueError(f"MODEL_{model}_OUTPUT_SAMPLE_SET_MISMATCH")
        for variant_id, row in expected.items():
            actual_sha = str(
                actual[variant_id].get("crop_sha256")
                or actual[variant_id].get("source_crop_sha256")
                or ""
            )
            if actual_sha != str(row["crop_sha256"]):
                raise ValueError(f"MODEL_{model}_CROP_SHA_MISMATCH:{variant_id}")
    return {
        "schema": "bemarkdown-pp-error-padding-crop-contract-v1",
        "variant_count": len(expected),
        "models": list(MODELS),
        "gate": "PASS",
    }


def score_recoverability(
    inventory: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    inventory_by_error = _unique(inventory, "error_id")
    variant_by_id = _unique(variants, "variant_id")
    output_maps = {model: _unique(outputs[model], "sample_id") for model in MODELS}
    if any(set(rows) != set(variant_by_id) for rows in output_maps.values()):
        raise ValueError("RECOVERABILITY_OUTPUT_SAMPLE_SET_MISMATCH")
    rows = []
    summary = {
        variant: {model: 0 for model in MODELS} for variant in VARIANTS
    }
    for variant_id in sorted(variant_by_id):
        variant = variant_by_id[variant_id]
        error_id = str(variant["error_id"])
        gold = normalize_text(inventory_by_error[error_id]["gold_raw"])
        correct = {}
        normalized = {}
        for model in MODELS:
            value = normalize_text(output_maps[model][variant_id].get("raw_output"))
            normalized[model] = value
            correct[model] = value == gold
            summary[str(variant["variant"])][model] += int(correct[model])
        rows.append(
            {
                "variant_id": variant_id,
                "error_id": error_id,
                "variant": str(variant["variant"]),
                "gold_normalized": gold,
                "normalized_outputs": normalized,
                "exact_correct": correct,
            }
        )
    return {
        "schema": "bemarkdown-pp-error-recoverability-v1",
        "rows": rows,
        "summary": summary,
    }


def root_cause_accounting(
    audits: Sequence[Mapping[str, Any]], *, expected_count: int = 13
) -> dict[str, Any]:
    rows = _unique(audits, "error_id")
    if len(rows) != expected_count:
        raise ValueError(f"AUDIT_ACCOUNTING_MISMATCH:{len(rows)}:{expected_count}")
    causes = Counter()
    adequacy = Counter()
    for error_id, row in rows.items():
        cause = str(row.get("primary_root_cause") or "")
        visual = str(row.get("visual_adequacy") or "")
        if cause not in ROOT_CAUSES:
            raise ValueError(f"INVALID_ROOT_CAUSE:{error_id}:{cause}")
        if visual not in VISUAL_ADEQUACY:
            raise ValueError(f"INVALID_VISUAL_ADEQUACY:{error_id}:{visual}")
        causes[cause] += 1
        adequacy[visual] += 1
    return {
        "schema": "bemarkdown-pp-error-root-cause-accounting-v1",
        "count": len(rows),
        "root_causes": dict(sorted(causes.items())),
        "adequacy": dict(sorted(adequacy.items())),
        "gate": "PASS",
    }


def _unique(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = str(row.get(field) or "")
        if not identity or identity in result:
            raise ValueError(f"INVALID_OR_DUPLICATE_{field.upper()}:{identity}")
        result[identity] = row
    return result


def _round_pixels(value: float) -> int:
    return math.floor(value + 0.5)
