"""Frozen Round 3 helpers for the LightOnOCR third-model evaluation."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .ocr_ensemble_benchmark import (
    CONSENSUS_COMPARATOR_STATUS,
    classify_error,
    normalize_text,
    score_ensemble,
)
from .ocr_ensemble_round2 import assess_output_contract, select_stratified_preflight

ROUND3_BASELINE_COMMIT = "a11a52157ecc2db1f0ebc6cbcaa1f09488e2c5b7"
TRUTH_FINGERPRINT = "7ef6e76384b7ea6d0fdd94b2c4cb2b9ee5a563771b8724eb5cd54f23e9405820"
CROP_FINGERPRINT = "ae95c6face305c0cc317ceb0409be6040f740d8ab3960dc0a79491c71edaa5a8"
A_OUTPUT_SHA256 = "a1e0bc6ec195b92cbc55def85e98fa9bc7930b8c88f7ba82d3ac3a5d16a2d17c"
GOT_OUTPUT_SHA256 = "e5d8fb052fcdf6fdf7dbc5e63997925d29af6637824ca17f78e728d813534151"
NORMALIZATION_SHA256 = (
    "6b5211b67df1d38f1c4bc3322dbc92209efd42cc916700b961c50cf735a427b3"
)
EXTRACTION_RULE = "PROCESSOR_DECODE_SKIP_SPECIAL_TOKENS_TRUE_IDENTITY_TEXT"
EXTRACTION_RULE_SHA256 = hashlib.sha256(EXTRACTION_RULE.encode()).hexdigest()
GENERATION_BUDGET = 1024
MODEL_SPEC: dict[str, Any] = {
    "formal_name": "LightOnOCR-2-1B",
    "model_id": "lightonai/LightOnOCR-2-1B",
    "revision": "c97bd377f04481830395218fa8951df9deaba756",
    "weight_sha256": "cbe12d0831cca119facce91268fc7c5fb72babdf919a6e352e3131bda85c8fbb",
    "weight_bytes": 2_011_367_489,
    "parameters": 1_005_647_872,
    "license": "Apache-2.0",
}


def score_round3(
    truth_rows: Sequence[Mapping[str, Any]],
    a_outputs: Sequence[Mapping[str, Any]],
    got_outputs: Sequence[Mapping[str, Any]],
    c3_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score frozen A/GOT and C3 without selecting or patching an OCR answer."""

    truth = _unique(truth_rows)
    outputs = {
        "A": _unique(a_outputs),
        "GOT": _unique(got_outputs),
        "C3": _unique(c3_outputs),
    }
    sample_ids = set(truth)
    if any(set(rows) != sample_ids for rows in outputs.values()):
        raise ValueError("ROUND3_OUTPUT_SAMPLE_SET_MISMATCH")

    mapped = score_ensemble(
        truth_rows,
        {
            "A": list(a_outputs),
            "B": _extracted_scoring_rows(got_outputs),
            "C": _extracted_scoring_rows(c3_outputs),
        },
    )
    accuracy = dict(mapped["accuracy_metrics"])
    accuracy["models"] = _rename_models(accuracy["models"])
    sensitive = dict(mapped["sensitive_token_metrics"])
    sensitive["models"] = _rename_models(sensitive["models"])
    taxonomy = dict(mapped["error_taxonomy_metrics"])
    taxonomy["models"] = _rename_models(taxonomy["models"])

    correct = {
        name: {
            sample_id: normalize_text(_text(row))
            == normalize_text(str(truth[sample_id].get("truth_text") or ""))
            for sample_id, row in rows.items()
        }
        for name, rows in outputs.items()
    }
    joint = {
        sample_id
        for sample_id in sample_ids
        if not correct["A"][sample_id] and not correct["GOT"][sample_id]
    }
    c3_recovered = {sample_id for sample_id in joint if correct["C3"][sample_id]}
    triple = joint - c3_recovered
    a_errors = {sample_id for sample_id in sample_ids if not correct["A"][sample_id]}
    total_recovered = {
        sample_id
        for sample_id in a_errors
        if correct["GOT"][sample_id] or correct["C3"][sample_id]
    }
    differs_and_correct = {
        sample_id
        for sample_id in joint
        if _normalized(outputs["C3"][sample_id])
        not in {
            _normalized(outputs["A"][sample_id]),
            _normalized(outputs["GOT"][sample_id]),
        }
        and correct["C3"][sample_id]
    }
    unanimous_wrong = {
        sample_id
        for sample_id in triple
        if len({_normalized(outputs[name][sample_id]) for name in outputs}) == 1
    }

    return {
        "accuracy_metrics": accuracy,
        "sensitive_token_metrics": sensitive,
        "error_taxonomy": taxonomy,
        "joint_error_set": _joint_error_set(joint, truth, outputs),
        "joint_error_recovery": {
            "schema": "bemarkdown-ocr-round3-joint-error-recovery-v1",
            "a_got_joint_error_count": len(joint),
            "c3_recovered_count": len(c3_recovered),
            "c3_recovery_rate": _ratio(len(c3_recovered), len(joint)),
            "c3_recovered_sample_ids": sorted(c3_recovered),
            "differs_from_both_and_correct_count": len(differs_and_correct),
            "differs_from_both_and_correct_sample_ids": sorted(differs_and_correct),
            "engineering_band": _recovery_band(_ratio(len(c3_recovered), len(joint))),
        },
        "triple_error_metrics": {
            "schema": "bemarkdown-ocr-round3-triple-error-v1",
            "triple_error_count": len(triple),
            "triple_error_rate": _ratio(len(triple), len(sample_ids)),
            "triple_error_sample_ids": sorted(triple),
            "a_error_count": len(a_errors),
            "total_recovery_on_a_error_count": len(total_recovered),
            "total_recovery_on_a_error": _ratio(len(total_recovered), len(a_errors)),
        },
        "unanimous_wrong_metrics": _unanimous_wrong(
            unanimous_wrong, truth, outputs["C3"], len(sample_ids)
        ),
        "a_got_adjudication": _a_got_adjudication(truth, outputs, correct),
        "joint_new_answer_value": _joint_new_answer(joint, outputs, correct),
        "error_correlation": _correlations(correct),
        "disagreement_patterns": _disagreement_patterns(truth, outputs, correct),
        "consensus_comparator": CONSENSUS_COMPARATOR_STATUS,
    }


def candidate_selection_gate(
    accuracy_metrics: Mapping[str, Any],
    joint_recovery: Mapping[str, Any],
    correlation: Mapping[str, Any],
    performance: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the Round 3 candidate gate without implementing consensus."""

    format_rate = float(
        accuracy_metrics["models"]["C3"]["output_format_violation_rate"]
    )
    recovery = float(joint_recovery["c3_recovery_rate"])
    conditional = float(correlation["p_c3_wrong_given_a_and_got_wrong"])
    checks = {
        "full_500_completed": int(performance["population"]) == 500,
        "rtx4070_12gb_no_oom": int(performance["oom_count"]) == 0
        and int(performance["peak_vram_bytes"]) <= 12_282 * 1024 * 1024,
        "no_cpu_fallback": int(performance["cpu_fallback_count"]) == 0,
        "format_violation_rate_le_0_02": format_rate <= 0.02,
        "joint_error_recovery_ge_0_15": recovery >= 0.15,
    }
    confirmed = all(checks.values())
    return {
        "schema": "bemarkdown-ocr-round3-selection-v1",
        "candidate": "C3",
        "model": MODEL_SPEC["formal_name"],
        "checks": checks,
        "diagnostics": {
            "joint_error_recovery": recovery,
            "preferred_joint_recovery_ge_0_20": recovery >= 0.20,
            "strong_joint_recovery_ge_0_30": recovery >= 0.30,
            "p_c3_wrong_given_a_and_got_wrong": conditional,
            "preferred_conditional_lt_0_85": conditional < 0.85,
            "format_violation_rate": format_rate,
        },
        "confirmed": confirmed,
        "three_model_set_selected": confirmed,
        "consensus_comparator": CONSENSUS_COMPARATOR_STATUS,
        "automatic_winner_selection_implemented": False,
        "final_gate": (
            [
                "OCR_MODEL_C3_LIGHTONOCR_CONFIRMED",
                "OCR_THREE_MODEL_SET_SELECTED",
                "UNBLOCK_OCR_2PLUS1_CONSENSUS_DESIGN",
            ]
            if confirmed
            else [
                "OCR_MODEL_C3_NOT_CONFIRMED",
                "OCR_THREE_MODEL_SEARCH_RECONSIDERATION_REQUIRED",
                "OCR_2PLUS1_CONSENSUS_DESIGN_BLOCKED",
            ]
        ),
    }


def project_cost(
    a_outputs: Sequence[Mapping[str, Any]],
    got_outputs: Sequence[Mapping[str, Any]],
    c3_performance: Mapping[str, Any],
    *,
    candidate_confirmed: bool,
    routes_per_page: float = 2.7,
) -> dict[str, Any]:
    """Project, but never implement, a third-model disagreement trigger."""

    if not candidate_confirmed:
        return {
            "schema": "bemarkdown-ocr-round3-production-cost-projection-v1",
            "status": "NOT_CALCULATED_CANDIDATE_GATE_FAILED",
            "projection_only": True,
            "consensus_implemented": False,
        }
    a = _unique(a_outputs)
    got = _unique(got_outputs)
    if set(a) != set(got):
        raise ValueError("ROUND3_PROJECTION_SAMPLE_SET_MISMATCH")
    triggers = {
        sample_id
        for sample_id in a
        if _normalized(a[sample_id]) != _normalized(got[sample_id])
    }
    trigger_rate = _ratio(len(triggers), len(a))
    calls_per_page = trigger_rate * routes_per_page
    extra_seconds = calls_per_page * float(c3_performance["mean_latency_seconds"])
    return {
        "schema": "bemarkdown-ocr-round3-production-cost-projection-v1",
        "status": "PROJECTION_ONLY",
        "projection_only": True,
        "trigger_definition": "A_GOT_NORMALIZED_DISAGREEMENT",
        "trigger_count": len(triggers),
        "c3_trigger_rate": trigger_rate,
        "modular_text_routes_per_page": routes_per_page,
        "projected_lighton_calls_per_page": calls_per_page,
        "projected_additional_seconds_per_page": extra_seconds,
        "projected_residency": "SEQUENTIAL_A_THEN_GOT_THEN_OPTIONAL_C3",
        "consensus_implemented": False,
    }


def _joint_error_set(
    sample_ids: set[str],
    truth: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for sample_id in sorted(sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "track": truth[sample_id].get("track"),
                "stratum": truth[sample_id].get("stratum"),
                "truth_text": truth[sample_id].get("truth_text"),
                "a_output": _text(outputs["A"][sample_id]),
                "got_output": _text(outputs["GOT"][sample_id]),
            }
        )
    return {
        "schema": "bemarkdown-ocr-round3-a-got-joint-error-set-v1",
        "definition": "A_WRONG_AND_GOT_WRONG",
        "count": len(rows),
        "rows": rows,
    }


def _unanimous_wrong(
    sample_ids: set[str],
    truth: Mapping[str, Mapping[str, Any]],
    c3_outputs: Mapping[str, Mapping[str, Any]],
    population: int,
) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    rows = []
    for sample_id in sorted(sample_ids):
        error = classify_error(
            str(truth[sample_id].get("truth_text") or ""), _text(c3_outputs[sample_id])
        )
        category = _unanimous_category(error)
        categories[category] += 1
        rows.append({"sample_id": sample_id, "category": category, "error_type": error})
    for name in ("数字", "单位", "上下标", "字符", "标点", "截断"):
        categories.setdefault(name, 0)
    return {
        "schema": "bemarkdown-ocr-round3-unanimous-wrong-v1",
        "definition": "A_EQUALS_GOT_EQUALS_C3_AND_TRUTH_DIFFERS",
        "unanimous_wrong_count": len(sample_ids),
        "unanimous_wrong_rate": _ratio(len(sample_ids), population),
        "category_counts": dict(categories),
        "rows": rows,
    }


def _a_got_adjudication(
    truth: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    correct: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    sample_ids = []
    for sample_id in truth:
        a = _normalized(outputs["A"][sample_id])
        got = _normalized(outputs["GOT"][sample_id])
        c3 = _normalized(outputs["C3"][sample_id])
        if a == got:
            continue
        sample_ids.append(sample_id)
        if c3 == a:
            key = (
                "c3_equals_a_a_correct"
                if correct["A"][sample_id]
                else "c3_equals_a_a_wrong"
            )
        elif c3 == got:
            key = (
                "c3_equals_got_got_correct"
                if correct["GOT"][sample_id]
                else "c3_equals_got_got_wrong"
            )
        else:
            key = (
                "c3_differs_from_both_c3_correct"
                if correct["C3"][sample_id]
                else "c3_differs_from_both_c3_wrong"
            )
        counts[key] += 1
    keys = (
        "c3_equals_a_a_correct",
        "c3_equals_a_a_wrong",
        "c3_equals_got_got_correct",
        "c3_equals_got_got_wrong",
        "c3_differs_from_both_c3_correct",
        "c3_differs_from_both_c3_wrong",
    )
    return {
        "schema": "bemarkdown-ocr-round3-a-got-adjudication-v1",
        "population": len(sample_ids),
        "counts": {key: counts[key] for key in keys},
        "sample_ids": sorted(sample_ids),
    }


def _joint_new_answer(
    joint: set[str],
    outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    correct: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for sample_id in joint:
        a = _normalized(outputs["A"][sample_id])
        got = _normalized(outputs["GOT"][sample_id])
        c3 = _normalized(outputs["C3"][sample_id])
        if c3 == a:
            counts["c3_equals_a"] += 1
        elif c3 == got:
            counts["c3_equals_got"] += 1
        else:
            counts["c3_differs_from_both"] += 1
            counts[
                "c3_differs_from_both_and_correct"
                if correct["C3"][sample_id]
                else "c3_differs_from_both_and_wrong"
            ] += 1
    keys = (
        "c3_equals_a",
        "c3_equals_got",
        "c3_differs_from_both",
        "c3_differs_from_both_and_correct",
        "c3_differs_from_both_and_wrong",
    )
    return {
        "schema": "bemarkdown-ocr-round3-joint-new-answer-value-v1",
        "population": len(joint),
        "counts": {key: counts[key] for key in keys},
    }


def _correlations(correct: Mapping[str, Mapping[str, bool]]) -> dict[str, Any]:
    errors = {
        name: {sample_id for sample_id, value in rows.items() if not value}
        for name, rows in correct.items()
    }
    pairs: dict[str, Any] = {}
    for left, right in (("A", "C3"), ("GOT", "C3"), ("A", "GOT")):
        overlap = errors[left] & errors[right]
        union = errors[left] | errors[right]
        pairs[f"{left}-{right}"] = {
            "left_error_count": len(errors[left]),
            "right_error_count": len(errors[right]),
            "overlap_count": len(overlap),
            "jaccard": _ratio(len(overlap), len(union)),
            f"p_{right.lower()}_wrong_given_{left.lower()}_wrong": _ratio(
                len(overlap), len(errors[left])
            ),
            "overlap_sample_ids": sorted(overlap),
        }
    joint = errors["A"] & errors["GOT"]
    triple = joint & errors["C3"]
    return {
        "schema": "bemarkdown-ocr-round3-error-correlation-v1",
        "pairs": pairs,
        "triple_overlap_count": len(triple),
        "p_c3_wrong_given_a_wrong": _ratio(
            len(errors["C3"] & errors["A"]), len(errors["A"])
        ),
        "p_c3_wrong_given_got_wrong": _ratio(
            len(errors["C3"] & errors["GOT"]), len(errors["GOT"])
        ),
        "p_c3_wrong_given_a_and_got_wrong": _ratio(len(triple), len(joint)),
    }


def _disagreement_patterns(
    truth: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    correct: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in truth:
        a, got, c3 = (
            _normalized(outputs[name][sample_id]) for name in ("A", "GOT", "C3")
        )
        if a == got == c3:
            pattern = "A=GOT=C3"
        elif a == got:
            pattern = "A=GOT!=C3"
        elif a == c3:
            pattern = "A=C3!=GOT"
        elif got == c3:
            pattern = "GOT=C3!=A"
        else:
            pattern = "A!=GOT!=C3"
        groups[pattern].append(sample_id)
    result = {}
    for pattern in ("A=GOT=C3", "A=GOT!=C3", "A=C3!=GOT", "GOT=C3!=A", "A!=GOT!=C3"):
        ids = groups.get(pattern, [])
        a_correct = sum(correct["A"][sample_id] for sample_id in ids)
        got_correct = sum(correct["GOT"][sample_id] for sample_id in ids)
        c3_correct = sum(correct["C3"][sample_id] for sample_id in ids)
        none = sum(
            not any(correct[name][sample_id] for name in correct) for sample_id in ids
        )
        result[pattern] = {
            "count": len(ids),
            "a_correct": a_correct,
            "got_correct": got_correct,
            "c3_correct": c3_correct,
            "none_correct": none,
            "material_error_probability": _ratio(none, len(ids)),
            "sample_ids": sorted(ids),
        }
    return {
        "schema": "bemarkdown-ocr-round3-disagreement-patterns-v1",
        "patterns": result,
        "majority_vote_implemented": False,
    }


def _extracted_scoring_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        mapped = dict(row)
        extracted = _text(row)
        mapped["raw_output"] = extracted
        mapped["normalized_output"] = normalize_text(extracted)
        result.append(mapped)
    return result


def _rename_models(models: Mapping[str, Any]) -> dict[str, Any]:
    return {"A": models["A"], "GOT": models["B"], "C3": models["C"]}


def _unique(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("ROUND3_DUPLICATE_SAMPLE_ID")
    return result


def _text(row: Mapping[str, Any]) -> str:
    return str(
        row.get("extracted_text")
        if row.get("extracted_text") is not None
        else row.get("raw_output") or ""
    )


def _normalized(row: Mapping[str, Any]) -> str:
    return normalize_text(_text(row))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _recovery_band(value: float) -> str:
    if value >= 0.30:
        return "STRONG_THIRD_MODEL_CANDIDATE"
    if value >= 0.20:
        return "CLEAR_COMPLEMENTARY_VALUE"
    if value >= 0.10:
        return "LIMITED_COMPLEMENTARY_VALUE"
    return "LOW_COMPLEMENTARY_VALUE"


def _unanimous_category(error_type: str) -> str:
    if error_type in {"NUMBER_ERROR", "DECIMAL_ERROR", "SIGN_ERROR"}:
        return "数字"
    if error_type == "UNIT_ERROR":
        return "单位"
    if error_type in {"SUBSCRIPT_ERROR", "SUPERSCRIPT_ERROR"}:
        return "上下标"
    if error_type == "PUNCTUATION_ONLY":
        return "标点"
    if error_type == "TRUNCATION":
        return "截断"
    return "字符"


__all__ = [
    "A_OUTPUT_SHA256",
    "CROP_FINGERPRINT",
    "EXTRACTION_RULE",
    "EXTRACTION_RULE_SHA256",
    "GENERATION_BUDGET",
    "GOT_OUTPUT_SHA256",
    "MODEL_SPEC",
    "NORMALIZATION_SHA256",
    "ROUND3_BASELINE_COMMIT",
    "TRUTH_FINGERPRINT",
    "assess_output_contract",
    "candidate_selection_gate",
    "project_cost",
    "score_round3",
    "select_stratified_preflight",
]
