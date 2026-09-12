"""Deterministic, truth-side scoring for text-only OCR ensemble benchmarks."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .text_recognition_contract import (
    NORMALIZATION_VERSION,
    extract_sensitive_tokens,
    normalize_text,
)

ERROR_TAXONOMY_VERSION = "ocr-error-taxonomy-v1"
CONSENSUS_COMPARATOR_STATUS = "NOT_IMPLEMENTED"

ERROR_TYPES = (
    "CHAR_SUBSTITUTION",
    "CHAR_DELETION",
    "CHAR_INSERTION",
    "NUMBER_ERROR",
    "DECIMAL_ERROR",
    "SIGN_ERROR",
    "UNIT_ERROR",
    "GREEK_ERROR",
    "SUBSCRIPT_ERROR",
    "SUPERSCRIPT_ERROR",
    "OPTION_LABEL_ERROR",
    "PUNCTUATION_ONLY",
    "WHITESPACE_ONLY",
    "DUPLICATION",
    "TRUNCATION",
    "HALLUCINATION",
    "EMPTY",
)

TOKEN_KINDS = (
    "numbers",
    "decimal_points",
    "signs",
    "units",
    "Greek_letters",
    "subscripts",
    "superscripts",
    "option_labels",
    "equation_like_inline_tokens",
    "vector_notation",
)

_PUNCTUATION = re.compile(r"[^\w\u3400-\u9fff\u0370-\u03ff]+", re.UNICODE)
_NUMBER = re.compile(r"[+\-]?\d+(?:\.\d+)?")
_UNIT = re.compile(
    r"(?<![A-Za-z])(?:m/s(?:²|2)?|m·s[⁻-]?\d*|kg|mol|Hz|Pa|J|W|V|A|Ω|N|C|T|s|m|g|K|%)(?![A-Za-z])"
)
_GREEK = re.compile(r"[\u0370-\u03ff]")
_SUBSCRIPT = re.compile(r"[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ]")
_SUPERSCRIPT = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿ]")
_OPTION = re.compile(r"(?<![A-Za-z])[A-D](?:[.、)])?(?![A-Za-z])")
_INLINE_EQUATION = re.compile(r"[^\s，。；;]{0,16}[=+\-×÷][^\s，。；;]{0,16}")
_VECTOR = re.compile(r"(?:[⃗→]|\\vec\s*\{?\w+\}?)")


def classify_error(truth: str, prediction: str) -> str:
    """Return one primary deterministic error class, with material tokens first."""

    truth_n = normalize_text(truth)
    pred_n = normalize_text(prediction)
    if truth_n == pred_n:
        return "WHITESPACE_ONLY" if truth != prediction else "PUNCTUATION_ONLY"
    if not pred_n:
        return "EMPTY"
    if not truth_n:
        return "HALLUCINATION"
    if _strip_whitespace(truth_n) == _strip_whitespace(pred_n):
        return "WHITESPACE_ONLY"
    if _decimal_signature(truth_n) != _decimal_signature(pred_n):
        return "DECIMAL_ERROR"
    if _signs(truth_n) != _signs(pred_n):
        return "SIGN_ERROR"
    if _token_values(_SUBSCRIPT, truth_n) != _token_values(_SUBSCRIPT, pred_n):
        return "SUBSCRIPT_ERROR"
    if _token_values(_SUPERSCRIPT, truth_n) != _token_values(_SUPERSCRIPT, pred_n):
        return "SUPERSCRIPT_ERROR"
    if _token_values(_GREEK, truth_n) != _token_values(_GREEK, pred_n):
        return "GREEK_ERROR"
    if _token_values(_UNIT, truth_n) != _token_values(_UNIT, pred_n):
        return "UNIT_ERROR"
    if _token_values(_OPTION, truth_n) != _token_values(_OPTION, pred_n):
        return "OPTION_LABEL_ERROR"
    if _token_values(_NUMBER, truth_n) != _token_values(_NUMBER, pred_n):
        return "NUMBER_ERROR"
    if _strip_punctuation(truth_n) == _strip_punctuation(pred_n):
        return "PUNCTUATION_ONLY"
    if _is_duplicate_output(pred_n):
        return "DUPLICATION"
    if len(pred_n) <= max(1, int(len(truth_n) * 0.55)):
        return "TRUNCATION"
    if len(pred_n) >= max(len(truth_n) + 8, int(len(truth_n) * 1.75)):
        return "HALLUCINATION"
    operations = _edit_operations(truth_n, pred_n)
    counts = Counter(kind for kind, _left, _right in operations)
    if counts["substitute"]:
        return "CHAR_SUBSTITUTION"
    if counts["delete"]:
        return "CHAR_DELETION"
    return "CHAR_INSERTION"


def score_ensemble(
    truth_rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Score A/B/C independently and expose complementarity without choosing a winner."""

    required_models = {"A", "B", "C"}
    if set(outputs) != required_models:
        raise ValueError("OCR ensemble scoring requires exactly models A, B, and C")
    truth_by_id = _unique_by_id(truth_rows)
    output_by_model = {name: _unique_by_id(rows) for name, rows in outputs.items()}
    sample_ids = set(truth_by_id)
    for name, rows in output_by_model.items():
        if set(rows) != sample_ids:
            raise ValueError(f"MODEL_{name}_OUTPUT_SAMPLE_SET_MISMATCH")

    verified = {
        sample_id: row
        for sample_id, row in truth_by_id.items()
        if row.get("truth_status") != "UNVERIFIABLE" and row.get("truth_text") is not None
    }
    scored: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for name in sorted(required_models):
        for sample_id, truth in verified.items():
            scored[name][sample_id] = _score_sample(
                str(truth["truth_text"]), output_by_model[name][sample_id]
            )

    accuracy_models = {}
    for name in sorted(required_models):
        accuracy_models[name] = _accuracy_summary(
            truth_by_id,
            output_by_model[name],
            scored[name],
        )
    accuracy = {
        "schema": "bemarkdown-ocr-ensemble-accuracy-v1",
        "normalization_version": NORMALIZATION_VERSION,
        "population": len(truth_rows),
        "verified_denominator": len(verified),
        "unverifiable_count": len(truth_rows) - len(verified),
        "unverifiable_rate": _ratio(len(truth_rows) - len(verified), len(truth_rows)),
        "models": accuracy_models,
    }

    sensitive = {
        "schema": "bemarkdown-ocr-sensitive-token-metrics-v1",
        "models": {
            name: _sensitive_token_summary(verified, output_by_model[name])
            for name in sorted(required_models)
        },
    }
    taxonomy = {
        "schema": "bemarkdown-ocr-error-taxonomy-metrics-v1",
        "taxonomy_version": ERROR_TAXONOMY_VERSION,
        "models": {
            name: _taxonomy_summary(scored[name]) for name in sorted(required_models)
        },
    }
    correctness = {
        name: {
            sample_id: not result["normalized_error"]
            for sample_id, result in scored[name].items()
        }
        for name in sorted(required_models)
    }
    recovery = _recovery_metrics(correctness, output_by_model)
    correlations = _correlation_metrics(correctness)
    disagreements = _disagreement_metrics(
        verified,
        output_by_model,
        scored,
        correctness,
    )
    return {
        "accuracy_metrics": accuracy,
        "sensitive_token_metrics": sensitive,
        "error_taxonomy_metrics": taxonomy,
        "recovery_metrics": recovery,
        "pairwise_error_correlation": correlations,
        "disagreement_patterns": disagreements,
        "consensus_comparator": CONSENSUS_COMPARATOR_STATUS,
    }


def score_model(
    truth_rows: Sequence[Mapping[str, Any]],
    output_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score one model, including a valid intentionally truncated/blocked cohort."""

    truth_by_id = _unique_by_id(truth_rows)
    output_by_id = _unique_by_id(output_rows)
    if set(output_by_id) != set(truth_by_id):
        raise ValueError("SINGLE_MODEL_OUTPUT_SAMPLE_SET_MISMATCH")
    verified = {
        sample_id: row
        for sample_id, row in truth_by_id.items()
        if row.get("truth_status") != "UNVERIFIABLE" and row.get("truth_text") is not None
    }
    scored = {
        sample_id: _score_sample(str(row["truth_text"]), output_by_id[sample_id])
        for sample_id, row in verified.items()
    }
    return {
        "accuracy": _accuracy_summary(truth_by_id, output_by_id, scored),
        "sensitive_tokens": _sensitive_token_summary(verified, output_by_id),
        "error_taxonomy": _taxonomy_summary(scored),
        "attempted_population": len(output_rows),
        "verified_denominator": len(verified),
    }


def _score_sample(truth: str, output: Mapping[str, Any]) -> dict[str, Any]:
    raw = str(output.get("raw_output") or "")
    normalized = str(output.get("normalized_output") or normalize_text(raw))
    truth_normalized = normalize_text(truth)
    distance = _edit_distance(truth_normalized, normalized)
    normalized_error = normalized != truth_normalized
    material = _material_error(truth_normalized, normalized)
    return {
        "raw_exact": raw == truth,
        "normalized_exact": not normalized_error,
        "normalized_error": normalized_error,
        "material_error": material,
        "edit_distance": distance,
        "truth_characters": len(truth_normalized),
        "error_type": classify_error(truth, raw) if normalized_error else None,
    }


def _accuracy_summary(
    truth: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
    scored: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    verified_rows = [scored[sample_id] for sample_id in scored]
    denominator = len(verified_rows)
    raw_exact = sum(bool(row["raw_exact"]) for row in verified_rows)
    normalized_exact = sum(bool(row["normalized_exact"]) for row in verified_rows)
    edits = sum(int(row["edit_distance"]) for row in verified_rows)
    characters = sum(int(row["truth_characters"]) for row in verified_rows)
    material = sum(bool(row["material_error"]) for row in verified_rows)
    empty = sum(not str(row.get("raw_output") or "").strip() for row in outputs.values())
    duplicate = sum(_is_duplicate_output(str(row.get("raw_output") or "")) for row in outputs.values())
    format_violations = sum(bool(row.get("output_format_violation")) for row in outputs.values())
    result = {
        "raw_exact_match_count": raw_exact,
        "raw_exact_match_rate": _ratio(raw_exact, denominator),
        "normalized_exact_match_count": normalized_exact,
        "normalized_exact_match_rate": _ratio(normalized_exact, denominator),
        "cer": _ratio(edits, characters),
        "character_accuracy": max(0.0, 1.0 - _ratio(edits, characters)),
        "material_error_count": material,
        "material_error_rate": _ratio(material, denominator),
        "empty_output_count": empty,
        "empty_output_rate": _ratio(empty, len(outputs)),
        "duplicate_output_count": duplicate,
        "duplicate_output_rate": _ratio(duplicate, len(outputs)),
        "output_format_violation_count": format_violations,
        "output_format_violation_rate": _ratio(format_violations, len(outputs)),
    }
    result["by_track"] = _grouped_accuracy(truth, scored, "track")
    result["by_stratum"] = _grouped_accuracy(truth, scored, "stratum")
    return result


def _grouped_accuracy(
    truth: Mapping[str, Mapping[str, Any]],
    scored: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample_id, score in scored.items():
        groups[str(truth[sample_id].get(field) or "UNKNOWN")].append(score)
    result = {}
    for name, rows in sorted(groups.items()):
        denominator = len(rows)
        exact = sum(bool(row["normalized_exact"]) for row in rows)
        material = sum(bool(row["material_error"]) for row in rows)
        edits = sum(int(row["edit_distance"]) for row in rows)
        chars = sum(int(row["truth_characters"]) for row in rows)
        result[name] = {
            "count": denominator,
            "normalized_exact_match_rate": _ratio(exact, denominator),
            "cer": _ratio(edits, chars),
            "material_error_rate": _ratio(material, denominator),
        }
    return result


def _sensitive_token_summary(
    verified: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    totals = {kind: Counter() for kind in TOKEN_KINDS}
    for sample_id, truth in verified.items():
        expected = extract_sensitive_tokens(str(truth["truth_text"]))
        actual = extract_sensitive_tokens(str(outputs[sample_id].get("normalized_output") or ""))
        for kind in TOKEN_KINDS:
            expected_counter = Counter(expected[kind])
            actual_counter = Counter(actual[kind])
            correct = sum((expected_counter & actual_counter).values())
            totals[kind]["correct"] += correct
            totals[kind]["missed"] += sum(expected_counter.values()) - correct
            totals[kind]["wrong"] += sum(actual_counter.values()) - correct
    return {
        kind: {
            "correct": totals[kind]["correct"],
            "wrong": totals[kind]["wrong"],
            "missed": totals[kind]["missed"],
            "accuracy": _ratio(
                totals[kind]["correct"],
                totals[kind]["correct"] + totals[kind]["missed"],
            ),
        }
        for kind in TOKEN_KINDS
    }


def _taxonomy_summary(scored: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        str(row["error_type"])
        for row in scored.values()
        if row.get("normalized_error") and row.get("error_type")
    )
    for error_type in ERROR_TYPES:
        counts.setdefault(error_type, 0)
    return {
        "counts": dict(sorted(counts.items())),
        "wrong_sample_count": sum(bool(row["normalized_error"]) for row in scored.values()),
        "semantic_correction_used": False,
    }


def _recovery_metrics(
    correctness: Mapping[str, Mapping[str, bool]],
    outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    a_errors = {sample_id for sample_id, correct in correctness["A"].items() if not correct}
    b_recovered = {sample_id for sample_id in a_errors if correctness["B"][sample_id]}
    c_recovered = {sample_id for sample_id in a_errors if correctness["C"][sample_id]}
    all_wrong = {
        sample_id
        for sample_id in a_errors
        if not correctness["B"][sample_id] and not correctness["C"][sample_id]
    }
    all_equal_wrong = {
        sample_id
        for sample_id in all_wrong
        if len(
            {
                normalize_text(str(outputs[name][sample_id].get("raw_output") or ""))
                for name in ("A", "B", "C")
            }
        )
        == 1
    }
    return {
        "schema": "bemarkdown-ocr-recovery-on-a-error-v1",
        "a_error_count": len(a_errors),
        "b_recovery_count": len(b_recovered),
        "b_recovery_on_a_error": _ratio(len(b_recovered), len(a_errors)),
        "c_recovery_count": len(c_recovered),
        "c_recovery_on_a_error": _ratio(len(c_recovered), len(a_errors)),
        "bc_recovery_count": len(b_recovered | c_recovered),
        "bc_recovery_on_a_error": _ratio(len(b_recovered | c_recovered), len(a_errors)),
        "all_three_wrong_count": len(all_wrong),
        "all_equal_but_wrong_count": len(all_equal_wrong),
        "all_three_wrong_sample_ids": sorted(all_wrong),
        "all_equal_but_wrong_sample_ids": sorted(all_equal_wrong),
    }


def _correlation_metrics(correctness: Mapping[str, Mapping[str, bool]]) -> dict[str, Any]:
    errors = {
        name: {sample_id for sample_id, correct in values.items() if not correct}
        for name, values in correctness.items()
    }
    pairs = {}
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
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
    triple = errors["A"] & errors["B"] & errors["C"]
    return {
        "schema": "bemarkdown-ocr-pairwise-error-correlation-v1",
        "pairs": pairs,
        "triple_error_count": len(triple),
        "triple_error_rate": _ratio(len(triple), len(correctness["A"])),
        "triple_error_sample_ids": sorted(triple),
        "p_b_or_c_correct_given_a_wrong": _ratio(
            len(errors["A"] - (errors["B"] & errors["C"])), len(errors["A"])
        ),
    }


def _disagreement_metrics(
    verified: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    scored: Mapping[str, Mapping[str, Mapping[str, Any]]],
    correctness: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    pattern_rows: dict[str, list[str]] = defaultdict(list)
    for sample_id in verified:
        a, b, c = (
            normalize_text(str(outputs[name][sample_id].get("raw_output") or ""))
            for name in ("A", "B", "C")
        )
        if a == b == c:
            pattern = "A=B=C"
        elif a == b:
            pattern = "A=B!=C"
        elif a == c:
            pattern = "A=C!=B"
        elif b == c:
            pattern = "B=C!=A"
        else:
            pattern = "A!=B!=C"
        pattern_rows[pattern].append(sample_id)
    patterns = {}
    for pattern in ("A=B=C", "A=B!=C", "A=C!=B", "B=C!=A", "A!=B!=C"):
        sample_ids = pattern_rows.get(pattern, [])
        truth_counts = Counter(
            sum(correctness[name][sample_id] for name in ("A", "B", "C"))
            for sample_id in sample_ids
        )
        any_material = sum(
            any(scored[name][sample_id]["material_error"] for name in ("A", "B", "C"))
            for sample_id in sample_ids
        )
        patterns[pattern] = {
            "sample_count": len(sample_ids),
            "truth_distribution": {
                f"correct_model_count_{count}": truth_counts[count] for count in range(4)
            },
            "material_error_probability": _ratio(any_material, len(sample_ids)),
            "sample_ids": sorted(sample_ids),
        }
    return {
        "schema": "bemarkdown-ocr-disagreement-patterns-v1",
        "patterns": patterns,
        "winner_selection_implemented": False,
    }


def _unique_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("DUPLICATE_OCR_BENCHMARK_SAMPLE_ID")
    return result


def _material_error(truth: str, prediction: str) -> bool:
    if truth == prediction:
        return False
    if _strip_whitespace(truth) == _strip_whitespace(prediction):
        return False
    return _strip_punctuation(truth) != _strip_punctuation(prediction)


def _strip_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _strip_punctuation(value: str) -> str:
    return _PUNCTUATION.sub("", value)


def _decimal_signature(value: str) -> tuple[str, ...]:
    return tuple(token for token in _NUMBER.findall(value) if "." in token)


def _signs(value: str) -> list[str]:
    return re.findall(r"(?<!\w)[+\-](?=\d|\w)", value)


def _token_values(pattern: re.Pattern[str], value: str) -> list[str]:
    return pattern.findall(value)


def _is_duplicate_output(value: str) -> bool:
    lines = [line.strip() for line in normalize_text(value).splitlines() if line.strip()]
    if len(lines) >= 3 and len(set(lines)) <= len(lines) // 2:
        return True
    return bool(re.search(r"(.{8,128})(?:\s*\1){2,}", value, re.DOTALL))


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _edit_operations(left: str, right: str) -> list[tuple[str, str, str]]:
    rows = len(left) + 1
    columns = len(right) + 1
    distance = [[0] * columns for _ in range(rows)]
    for index in range(rows):
        distance[index][0] = index
    for index in range(columns):
        distance[0][index] = index
    for i in range(1, rows):
        for j in range(1, columns):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (left[i - 1] != right[j - 1]),
            )
    operations = []
    i, j = len(left), len(right)
    while i or j:
        if i and j and distance[i][j] == distance[i - 1][j - 1] + (left[i - 1] != right[j - 1]):
            if left[i - 1] != right[j - 1]:
                operations.append(("substitute", left[i - 1], right[j - 1]))
            i -= 1
            j -= 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            operations.append(("delete", left[i - 1], ""))
            i -= 1
        else:
            operations.append(("insert", "", right[j - 1]))
            j -= 1
    operations.reverse()
    return operations


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
