"""Frozen Round 2 helpers for OCR reserve-candidate evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .ocr_ensemble_benchmark import (
    CONSENSUS_COMPARATOR_STATUS,
    normalize_text,
    score_ensemble,
)

ROUND2_BASELINE_COMMIT = "13c31c90241ea5d8ec719641a124f59ffe122174"
ROUND1_A_OUTPUT_SHA256 = (
    "a1e0bc6ec195b92cbc55def85e98fa9bc7930b8c88f7ba82d3ac3a5d16a2d17c"
)
ROUND1_TRUTH_FINGERPRINT = (
    "7ef6e76384b7ea6d0fdd94b2c4cb2b9ee5a563771b8724eb5cd54f23e9405820"
)
ROUND1_CROP_FINGERPRINT = (
    "ae95c6face305c0cc317ceb0409be6040f740d8ab3960dc0a79491c71edaa5a8"
)
FROZEN_NORMALIZATION_SHA256 = (
    "6b5211b67df1d38f1c4bc3322dbc92209efd42cc916700b961c50cf735a427b3"
)
EXTRACTION_RULE = "PROCESSOR_DECODE_SKIP_SPECIAL_TOKENS_TRUE_IDENTITY_TEXT"
EXTRACTION_RULE_SHA256 = hashlib.sha256(EXTRACTION_RULE.encode("utf-8")).hexdigest()

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "B2": {
        "formal_name": "GLM-OCR",
        "model_id": "zai-org/GLM-OCR",
        "revision": "ca5d8b3e287e52589e37c28385d9655ee4372f9d",
        "weight_sha256": (
            "a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815"
        ),
        "weight_bytes": 2_650_579_464,
        "parameters": 1_325_258_240,
        "license": "MIT",
        "official_task": "Text Recognition:",
        "official_max_new_tokens": 8192,
    },
    "C2": {
        "formal_name": "GOT-OCR2.0",
        "model_id": "stepfun-ai/GOT-OCR-2.0-hf",
        "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
        "weight_sha256": (
            "6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4"
        ),
        "weight_bytes": 1_121_114_488,
        "parameters": 560_528_640,
        "license": "Apache-2.0",
        "official_task": "PLAIN_TEXT_OCR_PROCESSOR_DEFAULT",
        "official_max_new_tokens": 4096,
    },
}

CONTRACT_STATUSES = (
    "PASS",
    "EMPTY",
    "TRUNCATED",
    "DUPLICATED",
    "ANALYSIS_LEAKAGE",
    "MARKUP_LEAKAGE",
    "FORMAT_VIOLATION",
    "INFERENCE_FAILURE",
)


def select_generation_budget(max_truth_char_length: int, official_limit: int) -> int:
    """Choose a crop budget with 3x character headroom, rounded to 128 tokens."""

    if max_truth_char_length <= 0 or official_limit <= 0:
        raise ValueError("GENERATION_BUDGET_REQUIRES_POSITIVE_LIMITS")
    target = max(512, max_truth_char_length * 3 + 64)
    rounded = int(math.ceil(target / 128) * 128)
    return min(official_limit, rounded)


def select_stratified_preflight(
    rows: Sequence[Mapping[str, Any]], *, per_stratum: int = 4
) -> list[Mapping[str, Any]]:
    """Select a deterministic equal-sized preflight cohort from every stratum."""

    if per_stratum <= 0:
        raise ValueError("PREFLIGHT_PER_STRATUM_MUST_BE_POSITIVE")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stratum") or "")].append(row)
    if not grouped or "" in grouped:
        raise ValueError("PREFLIGHT_REQUIRES_NAMED_STRATA")
    selected: list[Mapping[str, Any]] = []
    for stratum in sorted(grouped):
        available = sorted(grouped[stratum], key=lambda row: str(row["sample_id"]))
        if len(available) < per_stratum:
            raise ValueError(f"PREFLIGHT_STRATUM_TOO_SMALL:{stratum}")
        selected.extend(available[:per_stratum])
    return selected


def assess_output_contract(
    raw_output: str,
    extracted_text: str,
    *,
    generated_tokens: int,
    max_new_tokens: int,
    inference_error: str | None = None,
    extraction_failed: bool = False,
) -> str:
    """Classify output without repairing or semantically cleaning OCR text."""

    if inference_error:
        return "INFERENCE_FAILURE"
    if extraction_failed:
        return "FORMAT_VIOLATION"
    value = extracted_text.strip()
    if not value:
        return "EMPTY"
    if generated_tokens >= max_new_tokens:
        return "TRUNCATED"
    combined = f"{raw_output}\n{value}"
    if re.search(
        r"</?think\b|(?:^|\n)\s*(?:analysis|reasoning)\s*:",
        combined,
        re.IGNORECASE,
    ):
        return "ANALYSIS_LEAKAGE"
    if re.search(
        r"```|</?(?:html|body|table|img|div|span)\b", value, re.IGNORECASE
    ):
        return "MARKUP_LEAKAGE"
    if _is_duplicated(value):
        return "DUPLICATED"
    return "PASS"


def contract_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("output_contract_status") or "FORMAT_VIOLATION") for row in rows)
    for status in CONTRACT_STATUSES:
        counts.setdefault(status, 0)
    failures = len(rows) - counts["PASS"]
    runaway = sum(bool(row.get("hit_generation_limit")) for row in rows)
    return {
        "population": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "pass_rate": _ratio(counts["PASS"], len(rows)),
        "contract_failure_rate": _ratio(failures, len(rows)),
        "generation_runaway_rate": _ratio(runaway, len(rows)),
        "early_stop_contract_failure": bool(rows) and failures / len(rows) >= 0.80,
        "early_stop_generation_runaway": bool(rows) and runaway / len(rows) >= 0.80,
    }


def score_round2(
    truth_rows: Sequence[Mapping[str, Any]],
    a_outputs: Sequence[Mapping[str, Any]],
    b2_outputs: Sequence[Mapping[str, Any]],
    c2_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reuse the frozen scorer and rename B/C evidence to B2/C2."""

    mapped = score_ensemble(
        truth_rows,
        {
            "A": a_outputs,
            "B": _extracted_scoring_rows(b2_outputs),
            "C": _extracted_scoring_rows(c2_outputs),
        },
    )
    accuracy = dict(mapped["accuracy_metrics"])
    accuracy["models"] = _rename_model_mapping(accuracy["models"])
    sensitive = dict(mapped["sensitive_token_metrics"])
    sensitive["models"] = _rename_model_mapping(sensitive["models"])
    taxonomy = dict(mapped["error_taxonomy_metrics"])
    taxonomy["models"] = _rename_model_mapping(taxonomy["models"])

    old_recovery = mapped["recovery_metrics"]
    recovery = {
        "schema": "bemarkdown-ocr-round2-recovery-on-a-error-v1",
        "a_error_count": old_recovery["a_error_count"],
        "b2_recovery_count": old_recovery["b_recovery_count"],
        "b2_recovery_on_a_error": old_recovery["b_recovery_on_a_error"],
        "c2_recovery_count": old_recovery["c_recovery_count"],
        "c2_recovery_on_a_error": old_recovery["c_recovery_on_a_error"],
        "b2c2_recovery_count": old_recovery["bc_recovery_count"],
        "b2c2_recovery_on_a_error": old_recovery["bc_recovery_on_a_error"],
        "all_three_wrong_count": old_recovery["all_three_wrong_count"],
        "all_equal_but_wrong_count": old_recovery["all_equal_but_wrong_count"],
        "all_three_wrong_sample_ids": old_recovery["all_three_wrong_sample_ids"],
        "all_equal_but_wrong_sample_ids": old_recovery[
            "all_equal_but_wrong_sample_ids"
        ],
    }

    old_correlation = mapped["pairwise_error_correlation"]
    pairs = {
        "A-B2": _rename_pair(old_correlation["pairs"]["A-B"], "A", "B2"),
        "A-C2": _rename_pair(old_correlation["pairs"]["A-C"], "A", "C2"),
        "B2-C2": _rename_pair(old_correlation["pairs"]["B-C"], "B2", "C2"),
    }
    correlation = {
        "schema": "bemarkdown-ocr-round2-error-correlation-v1",
        "pairs": pairs,
        "triple_error_count": old_correlation["triple_error_count"],
        "triple_error_rate": old_correlation["triple_error_rate"],
        "triple_error_sample_ids": old_correlation["triple_error_sample_ids"],
        "p_b2_or_c2_correct_given_a_wrong": old_correlation[
            "p_b_or_c_correct_given_a_wrong"
        ],
    }
    disagreement = _round2_disagreement(
        truth_rows,
        {"A": a_outputs, "B2": b2_outputs, "C2": c2_outputs},
        mapped["disagreement_patterns"],
    )
    return {
        "accuracy_metrics": accuracy,
        "sensitive_token_metrics": sensitive,
        "error_taxonomy_metrics": taxonomy,
        "recovery_metrics": recovery,
        "error_correlation": correlation,
        "disagreement_patterns": disagreement,
        "consensus_comparator": CONSENSUS_COMPARATOR_STATUS,
    }


def candidate_selection_gate(
    accuracy_metrics: Mapping[str, Any],
    recovery_metrics: Mapping[str, Any],
    correlation_metrics: Mapping[str, Any],
    sensitive_token_metrics: Mapping[str, Any],
    performance_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen minimum complement gate without implementing consensus."""

    accuracy = accuracy_metrics["models"]
    sensitive = sensitive_token_metrics["models"]
    performance = performance_metrics["models"]
    a_material = float(accuracy["A"]["material_error_rate"])
    candidates: dict[str, Any] = {}
    for name in ("B2", "C2"):
        lower = name.lower()
        model_accuracy = accuracy[name]
        model_performance = performance[name]
        recovery = float(recovery_metrics[f"{lower}_recovery_on_a_error"])
        conditional = float(
            correlation_metrics["pairs"][f"A-{name}"][
                f"p_{lower}_wrong_given_a_wrong"
            ]
        )
        numbers_ratio = _relative_accuracy(
            sensitive[name]["numbers"]["accuracy"],
            sensitive["A"]["numbers"]["accuracy"],
        )
        units_ratio = _relative_accuracy(
            sensitive[name]["units"]["accuracy"],
            sensitive["A"]["units"]["accuracy"],
        )
        checks = {
            "full_500_completed": int(model_performance["population"]) == 500,
            "rtx4070_12gb_no_oom": (
                int(model_performance["oom_count"]) == 0
                and int(model_performance["peak_vram_bytes"]) <= 12_282 * 1024 * 1024
            ),
            "no_cpu_fallback": int(model_performance["cpu_fallback_count"]) == 0,
            "format_violation_rate_le_0_02": (
                float(model_accuracy["output_format_violation_rate"]) <= 0.02
            ),
            "recovery_on_a_error_ge_0_15": recovery >= 0.15,
            "numbers_not_collapsed_vs_a": numbers_ratio >= 0.80,
            "units_not_collapsed_vs_a": units_ratio >= 0.80,
            "license_suitable_for_project": True,
        }
        confirmed = all(checks.values())
        candidates[name] = {
            "model": MODEL_SPECS[name]["formal_name"],
            "confirmed": confirmed,
            "status": (
                f"OCR_MODEL_{name}_{'GLM_OCR' if name == 'B2' else 'GOT_OCR2'}_CONFIRMED"
                if confirmed
                else f"MODEL_{name}_NOT_CONFIRMED"
            ),
            "checks": checks,
            "diagnostics": {
                "material_error_rate": float(model_accuracy["material_error_rate"]),
                "material_error_delta_vs_a": float(
                    model_accuracy["material_error_rate"]
                )
                - a_material,
                "recovery_on_a_error": recovery,
                "strong_recovery_ge_0_25": recovery >= 0.25,
                "p_candidate_wrong_given_a_wrong": conditional,
                "preferred_error_conditional_lt_0_80": conditional < 0.80,
                "numbers_accuracy_ratio_vs_a": numbers_ratio,
                "units_accuracy_ratio_vs_a": units_ratio,
            },
        }
    confirmed = [name for name in ("B2", "C2") if candidates[name]["confirmed"]]
    confirmed.sort(
        key=lambda name: (
            float(accuracy[name]["material_error_rate"]),
            -float(recovery_metrics[f"{name.lower()}_recovery_on_a_error"]),
            float(
                correlation_metrics["pairs"][f"A-{name}"][
                    f"p_{name.lower()}_wrong_given_a_wrong"
                ]
            ),
        )
    )
    if len(confirmed) == 2:
        final_gate = [
            "OCR_MODEL_A_PP_OCRV6_CONFIRMED",
            "OCR_MODEL_B2_GLM_OCR_CONFIRMED",
            "OCR_MODEL_C2_GOT_OCR2_CONFIRMED",
            "OCR_THREE_MODEL_SET_SELECTED",
            "UNBLOCK_OCR_2PLUS1_CONSENSUS_DESIGN",
        ]
    elif len(confirmed) == 1:
        final_gate = ["OCR_ONE_COMPLEMENT_CONFIRMED", "THIRD_MODEL_STILL_REQUIRED"]
    else:
        final_gate = [
            "OCR_RESERVE_ROUND2_FAILED",
            "OCR_2PLUS1_MODEL_SEARCH_RECONSIDERATION_REQUIRED",
        ]
    return {
        "schema": "bemarkdown-ocr-ensemble-reserve-round2-selection-v1",
        "model_a_status": "OCR_MODEL_A_PP_OCRV6_CONFIRMED",
        "candidates": candidates,
        "selected_second_model": confirmed[0] if confirmed else None,
        "selected_third_model": confirmed[1] if len(confirmed) > 1 else None,
        "three_model_set_selected": len(confirmed) == 2,
        "next_phase_unblocked": len(confirmed) == 2,
        "final_gate": final_gate,
        "consensus_comparator": CONSENSUS_COMPARATOR_STATUS,
        "automatic_winner_selection_implemented": False,
    }


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _round2_disagreement(
    truth_rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    old_disagreement: Mapping[str, Any],
) -> dict[str, Any]:
    truth = {
        str(row["sample_id"]): normalize_text(str(row["truth_text"]))
        for row in truth_rows
        if row.get("truth_status") != "UNVERIFIABLE" and row.get("truth_text") is not None
    }
    by_model = {
        name: {str(row["sample_id"]): row for row in rows} for name, rows in outputs.items()
    }
    pattern_ids: dict[str, list[str]] = defaultdict(list)
    for sample_id in truth:
        a, b2, c2 = (
            normalize_text(
                str(
                    by_model[name][sample_id].get("normalized_output")
                    or by_model[name][sample_id].get("extracted_text")
                    or by_model[name][sample_id].get("raw_output")
                    or ""
                )
            )
            for name in ("A", "B2", "C2")
        )
        if a == b2 == c2:
            pattern = "A=B2=C2"
        elif a == b2:
            pattern = "A=B2!=C2"
        elif a == c2:
            pattern = "A=C2!=B2"
        elif b2 == c2:
            pattern = "B2=C2!=A"
        else:
            pattern = "A!=B2!=C2"
        pattern_ids[pattern].append(sample_id)

    old_by_new = {
        "A=B2=C2": "A=B=C",
        "A=B2!=C2": "A=B!=C",
        "A=C2!=B2": "A=C!=B",
        "B2=C2!=A": "B=C!=A",
        "A!=B2!=C2": "A!=B!=C",
    }
    patterns = {}
    for pattern, old_pattern in old_by_new.items():
        ids = pattern_ids.get(pattern, [])
        correct = {
            name: sum(
                normalize_text(
                    str(
                        by_model[name][sample_id].get("normalized_output")
                        or by_model[name][sample_id].get("extracted_text")
                        or by_model[name][sample_id].get("raw_output")
                        or ""
                    )
                )
                == truth[sample_id]
                for sample_id in ids
            )
            for name in ("A", "B2", "C2")
        }
        none_correct = sum(
            all(
                normalize_text(
                    str(
                        by_model[name][sample_id].get("normalized_output")
                        or by_model[name][sample_id].get("extracted_text")
                        or by_model[name][sample_id].get("raw_output")
                        or ""
                    )
                )
                != truth[sample_id]
                for name in ("A", "B2", "C2")
            )
            for sample_id in ids
        )
        patterns[pattern] = {
            "sample_count": len(ids),
            "a_correct_count": correct["A"],
            "b2_correct_count": correct["B2"],
            "c2_correct_count": correct["C2"],
            "none_correct_count": none_correct,
            "material_error_probability": old_disagreement["patterns"][old_pattern][
                "material_error_probability"
            ],
            "sample_ids": sorted(ids),
        }
    return {
        "schema": "bemarkdown-ocr-round2-disagreement-patterns-v1",
        "patterns": patterns,
        "winner_selection_implemented": False,
    }


def _rename_model_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"A": value["A"], "B2": value["B"], "C2": value["C"]}


def _extracted_scoring_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scoring_rows = []
    for row in rows:
        effective = str(row.get("extracted_text") or "")
        scoring_row = dict(row)
        scoring_row["raw_output"] = effective
        scoring_row["normalized_output"] = normalize_text(effective)
        scoring_rows.append(scoring_row)
    return scoring_rows


def _rename_pair(value: Mapping[str, Any], left: str, right: str) -> dict[str, Any]:
    result = {
        key: item
        for key, item in value.items()
        if not str(key).startswith("p_")
    }
    result["pair"] = f"{left}-{right}"
    conditional = next(item for key, item in value.items() if str(key).startswith("p_"))
    result[f"p_{right.lower()}_wrong_given_{left.lower()}_wrong"] = conditional
    return result


def _is_duplicated(value: str) -> bool:
    lines = [line.strip() for line in normalize_text(value).splitlines() if line.strip()]
    if len(lines) >= 3 and len(set(lines)) <= len(lines) // 2:
        return True
    return bool(re.search(r"(.{8,128})(?:\s*\1){2,}", value, re.DOTALL))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _relative_accuracy(value: float, baseline: float) -> float:
    return float(value / baseline) if baseline else (1.0 if not value else 0.0)
