"""Deterministic evaluation primitives for Formula quality benchmarks.

This module deliberately evaluates frozen predictions.  It never invokes a
formula model, changes a crop, or repairs LaTeX.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .formula_gt_evaluation import normalize_latex_for_evaluation


class FormulaBenchmarkError(RuntimeError):
    """Raised when a benchmark contract would be violated."""


_BLINDNESS_FORBIDDEN_KEYS = {
    "expected_error_category",
    "formula_issue",
    "formula_issue_count",
    "formula_net_output",
    "model_output",
    "old_formula_issue",
    "old_formula_issues",
    "prediction",
    "prediction_latex",
    "production_latex",
    "raw_prediction",
    "raw_latex",
}
_BLINDNESS_FORBIDDEN_KEY_FRAGMENTS = {
    "expected_error_category",
    "formula_issue",
    "formula_net",
    "formulanet",
    "model_output",
    "old_formula_issue",
    "prediction",
    "production_latex",
    "raw_latex",
}

GEOMETRY_MATCH_CONTRACT = {
    "schema": "bemarkdown-formula-geometry-match-contract-v1",
    "candidate_edge": (
        "positive intersection and (IoU >= 0.1 or either-box coverage >= 0.5 "
        "or either center contained)"
    ),
    "component_rule": "connected components in one document/page bipartite graph",
    "proximity_only_edges": False,
    "latex_similarity_used": False,
}

RECOGNITION_ERROR_CATEGORIES = {
    "DIGIT_LETTER_CONFUSION",
    "GREEK_LATIN_CONFUSION",
    "SUBSCRIPT_ERROR",
    "SUPERSCRIPT_ERROR",
    "MISSING_SYMBOL",
    "EXTRA_SYMBOL",
    "OPERATOR_ERROR",
    "RELATION_ERROR",
    "FRACTION_STRUCTURE_ERROR",
    "ROOT_STRUCTURE_ERROR",
    "PARENTHESIS_ERROR",
    "LATEX_SYNTAX_ERROR",
    "RENDER_FAILURE",
    "MULTI_TOKEN_STRUCTURE_ERROR",
    "OTHER_RECOGNITION_ERROR",
    "SOURCE_AMBIGUOUS",
}
RECOGNITION_CONFUSION_SUBTYPES = {
    "ONE_L_I",
    "ZERO_O",
    "NU_V",
    "RHO_P",
    "OTHER_DIGIT_LETTER",
    "OTHER_GREEK_LATIN",
}
DIGIT_LETTER_CONFUSION_SUBTYPES = {
    "ONE_L_I",
    "ZERO_O",
    "OTHER_DIGIT_LETTER",
}
GREEK_LATIN_CONFUSION_SUBTYPES = {
    "NU_V",
    "RHO_P",
    "OTHER_GREEK_LATIN",
}
SEVERITIES = {"CRITICAL", "MAJOR", "MINOR"}
DENOMINATOR_BINDING_VERDICTS = {
    "BINDING_VALID",
    "TRUTH_BBOX_GT_MISMATCH",
    "GEOMETRY_MATCH_MISMATCH",
    "PRODUCTION_CROP_BINDING_MISMATCH",
    "SOURCE_AMBIGUOUS",
}
DENOMINATOR_BINDING_FIELDS = (
    "formula_content_id",
    "source_page_sha256",
    "truth_row_semantic_sha256",
    "crop_sha256",
    "task_semantic_sha256",
)


def normalize_formula_latex(value: str) -> str:
    """Reuse the repository's frozen Formula evaluation normalization."""

    return normalize_latex_for_evaluation(value)


def levenshtein_distance(left: str, right: str) -> int:
    """Return deterministic character-level Levenshtein distance."""

    if left == right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_char in enumerate(right, 1):
        current = [right_index]
        for left_index, left_char in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(gt_latex: str, prediction_latex: str) -> tuple[int, int, float]:
    """Measure CER on already-normalized LaTeX strings."""

    distance = levenshtein_distance(gt_latex, prediction_latex)
    characters = len(gt_latex)
    if characters == 0:
        cer = 0.0 if not prediction_latex else 1.0
    else:
        cer = distance / characters
    return distance, characters, cer


def validate_recognition_confusion_subtypes(
    source_formula_id: str,
    categories: Sequence[str],
    confusion_subtypes: Sequence[str],
) -> None:
    if len(confusion_subtypes) != len(set(confusion_subtypes)):
        raise FormulaBenchmarkError(
            f"DUPLICATE_RECOGNITION_CONFUSION_SUBTYPE:{source_formula_id}"
        )
    unknown = sorted(
        set(confusion_subtypes).difference(RECOGNITION_CONFUSION_SUBTYPES)
    )
    if unknown:
        raise FormulaBenchmarkError(
            f"UNKNOWN_RECOGNITION_CONFUSION_SUBTYPE:{source_formula_id}:{unknown}"
        )
    category_set = set(categories)
    subtype_set = set(confusion_subtypes)
    for category, allowed_subtypes in (
        ("DIGIT_LETTER_CONFUSION", DIGIT_LETTER_CONFUSION_SUBTYPES),
        ("GREEK_LATIN_CONFUSION", GREEK_LATIN_CONFUSION_SUBTYPES),
    ):
        bound = subtype_set.intersection(allowed_subtypes)
        if category in category_set and not bound:
            raise FormulaBenchmarkError(
                f"CONFUSION_SUBTYPE_REQUIRED:{source_formula_id}:{category}"
            )
        if category not in category_set and bound:
            raise FormulaBenchmarkError(
                f"CONFUSION_CATEGORY_REQUIRED:{source_formula_id}:{sorted(bound)}"
            )


def evaluate_recognition_sample(
    *,
    source_formula_id: str,
    formula_content_id: str,
    gt_latex: str,
    prediction_latex: str,
    renderable: bool,
    render_error: str | None,
    syntax_verdict: str,
    annotation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one strict-denominator sample without guessing materiality."""

    normalized_gt = normalize_formula_latex(gt_latex)
    normalized_prediction = normalize_formula_latex(prediction_latex)
    raw_exact = prediction_latex == gt_latex
    normalized_exact = normalized_prediction == normalized_gt
    distance, characters, cer = character_error_rate(
        normalized_gt, normalized_prediction
    )
    if normalized_exact:
        material_error = False
        categories: list[str] = []
        confusion_subtypes: list[str] = []
        severity = "NONE"
        adjudication_notes = "Normalized strings are identical."
        adjudicator = "DETERMINISTIC_NORMALIZED_EXACT"
    else:
        if annotation is None:
            raise FormulaBenchmarkError(
                f"RECOGNITION_ADJUDICATION_REQUIRED:{source_formula_id}"
            )
        material_error = annotation.get("material_error")
        if not isinstance(material_error, bool):
            raise FormulaBenchmarkError(
                f"INVALID_MATERIAL_ERROR_ANNOTATION:{source_formula_id}"
            )
        categories = [str(value) for value in annotation.get("error_categories", [])]
        confusion_subtypes = [
            str(value) for value in annotation.get("confusion_subtypes", [])
        ]
        if len(categories) != len(set(categories)):
            raise FormulaBenchmarkError(
                f"DUPLICATE_RECOGNITION_ERROR_CATEGORY:{source_formula_id}"
            )
        unknown = sorted(set(categories).difference(RECOGNITION_ERROR_CATEGORIES))
        if unknown:
            raise FormulaBenchmarkError(
                f"UNKNOWN_RECOGNITION_ERROR_CATEGORY:{source_formula_id}:{unknown}"
            )
        validate_recognition_confusion_subtypes(
            source_formula_id, categories, confusion_subtypes
        )
        severity = annotation.get("severity")
        if material_error and (not categories or severity not in SEVERITIES):
            raise FormulaBenchmarkError(
                f"INVALID_RECOGNITION_SEVERITY:{source_formula_id}:{severity!r}"
            )
        if not material_error and (
            categories or severity not in {None, "NONE", "MINOR"}
        ):
            raise FormulaBenchmarkError(
                f"NON_MATERIAL_SEVERITY_INVALID:{source_formula_id}:{severity!r}"
            )
        if severity is None:
            severity = "NONE"
        adjudication_notes = str(annotation.get("notes") or "")
        adjudicator = str(annotation.get("adjudicator") or "")
        if not adjudicator:
            raise FormulaBenchmarkError(
                f"RECOGNITION_ADJUDICATOR_REQUIRED:{source_formula_id}"
            )
    if not renderable and "RENDER_FAILURE" not in categories:
        categories.append("RENDER_FAILURE")
    return {
        "source_formula_id": source_formula_id,
        "formula_content_id": formula_content_id,
        "gt_latex": gt_latex,
        "prediction_latex": prediction_latex,
        "normalized_gt_latex": normalized_gt,
        "normalized_prediction_latex": normalized_prediction,
        "raw_exact": raw_exact,
        "normalized_exact": normalized_exact,
        "edit_distance": distance,
        "gt_character_count": characters,
        "cer": cer,
        "syntax_verdict": syntax_verdict,
        "renderable": renderable,
        "render_error": render_error,
        "material_error": material_error,
        "error_categories": sorted(set(categories)),
        "confusion_subtypes": sorted(confusion_subtypes),
        "severity": severity,
        "adjudication_notes": adjudication_notes,
        "adjudicator": adjudicator,
    }


def summarize_recognition_results(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute Tier-B aggregates from exact bound recognition rows.

    Derived row fields are checked against the frozen GT and prediction strings
    before they are aggregated.  This prevents a stale or edited summary (or a
    row with edited metrics) from being accepted merely because membership and
    row count still match.
    """

    rows = [dict(row) for row in results]
    if not rows:
        raise FormulaBenchmarkError("STRICT_RECOGNITION_DENOMINATOR_EMPTY")

    source_ids: set[str] = set()
    content_ids: set[str] = set()
    for row in rows:
        source_formula_id = str(row.get("source_formula_id") or "").strip()
        formula_content_id = str(row.get("formula_content_id") or "").strip()
        if not source_formula_id or source_formula_id in source_ids:
            raise FormulaBenchmarkError(
                "RECOGNITION_SUMMARY_SOURCE_ID_INVALID_OR_DUPLICATE:"
                + source_formula_id
            )
        if not formula_content_id or formula_content_id in content_ids:
            raise FormulaBenchmarkError(
                "RECOGNITION_SUMMARY_CONTENT_ID_INVALID_OR_DUPLICATE:"
                + formula_content_id
            )
        source_ids.add(source_formula_id)
        content_ids.add(formula_content_id)

        gt_latex = str(row.get("gt_latex") or "")
        prediction_latex = str(row.get("prediction_latex") or "")
        normalized_gt = normalize_formula_latex(gt_latex)
        normalized_prediction = normalize_formula_latex(prediction_latex)
        distance, characters, cer = character_error_rate(
            normalized_gt, normalized_prediction
        )
        if characters == 0:
            raise FormulaBenchmarkError(
                "STRICT_RECOGNITION_GT_EMPTY:" + source_formula_id
            )
        expected_booleans = {
            "raw_exact": gt_latex == prediction_latex,
            "normalized_exact": normalized_gt == normalized_prediction,
        }
        expected_strings = {
            "normalized_gt_latex": normalized_gt,
            "normalized_prediction_latex": normalized_prediction,
        }
        expected_integers = {
            "edit_distance": distance,
            "gt_character_count": characters,
        }
        mismatched = [
            field
            for field, value in expected_booleans.items()
            if row.get(field) is not value
        ]
        mismatched.extend(
            field
            for field, value in expected_strings.items()
            if not isinstance(row.get(field), str) or row.get(field) != value
        )
        mismatched.extend(
            field
            for field, value in expected_integers.items()
            if type(row.get(field)) is not int or row.get(field) != value
        )
        observed_cer = row.get("cer")
        if (
            isinstance(observed_cer, bool)
            or not isinstance(observed_cer, (int, float))
            or not math.isfinite(float(observed_cer))
            or not math.isclose(float(observed_cer), cer, rel_tol=0.0, abs_tol=1e-12)
        ):
            mismatched.append("cer")
        if mismatched:
            raise FormulaBenchmarkError(
                "RECOGNITION_RESULT_DERIVATION_MISMATCH:"
                + source_formula_id
                + ":"
                + repr(sorted(mismatched))
            )
        if not isinstance(row.get("renderable"), bool):
            raise FormulaBenchmarkError(
                "RECOGNITION_RENDERABLE_BOOL_REQUIRED:" + source_formula_id
            )
        if not isinstance(row.get("material_error"), bool):
            raise FormulaBenchmarkError(
                "RECOGNITION_MATERIAL_ERROR_BOOL_REQUIRED:" + source_formula_id
            )
        raw_categories = row.get("error_categories")
        if not isinstance(raw_categories, Sequence) or isinstance(
            raw_categories, (str, bytes, bytearray)
        ):
            raise FormulaBenchmarkError(
                "RECOGNITION_ERROR_CATEGORIES_LIST_REQUIRED:" + source_formula_id
            )
        categories = [str(value) for value in raw_categories]
        if len(categories) != len(set(categories)):
            raise FormulaBenchmarkError(
                "RECOGNITION_ERROR_CATEGORY_DUPLICATE:" + source_formula_id
            )
        unknown = sorted(set(categories).difference(RECOGNITION_ERROR_CATEGORIES))
        if unknown:
            raise FormulaBenchmarkError(
                "UNKNOWN_RECOGNITION_ERROR_CATEGORY:"
                + source_formula_id
                + ":"
                + repr(unknown)
            )
        raw_confusion_subtypes = row.get("confusion_subtypes", [])
        if not isinstance(raw_confusion_subtypes, Sequence) or isinstance(
            raw_confusion_subtypes, (str, bytes, bytearray)
        ):
            raise FormulaBenchmarkError(
                "RECOGNITION_CONFUSION_SUBTYPES_LIST_REQUIRED:"
                + source_formula_id
            )
        confusion_subtypes = [str(value) for value in raw_confusion_subtypes]
        validate_recognition_confusion_subtypes(
            source_formula_id, categories, confusion_subtypes
        )
        material_error = row["material_error"]
        severity = row.get("severity")
        substantive_categories = set(categories) - {
            "LATEX_SYNTAX_ERROR",
            "RENDER_FAILURE",
        }
        materiality_inconsistent = (
            bool(row["normalized_exact"]) and material_error
        ) or (
            material_error
            and (not categories or severity not in SEVERITIES)
        ) or (
            not material_error
            and (
                bool(substantive_categories)
                or severity not in {None, "NONE", "MINOR"}
            )
        )
        if materiality_inconsistent:
            raise FormulaBenchmarkError(
                "RECOGNITION_RESULT_MATERIALITY_INCONSISTENT:"
                + source_formula_id
            )
        if not row["renderable"] and "RENDER_FAILURE" not in categories:
            raise FormulaBenchmarkError(
                "RENDER_FAILURE_CATEGORY_REQUIRED:" + source_formula_id
            )

    denominator = len(rows)
    total_distance = sum(int(row["edit_distance"]) for row in rows)
    total_characters = sum(int(row["gt_character_count"]) for row in rows)
    category_counts = Counter(
        category for row in rows for category in row["error_categories"]
    )
    confusion_subtype_counts = Counter(
        subtype for row in rows for subtype in row.get("confusion_subtypes", [])
    )
    return {
        "schema": "bemarkdown-formula-recognition-summary-v1",
        "strict_denominator_contract": (
            "ONE_TO_ONE + CROP_COMPLETE + SOURCE_LEGIBLE"
        ),
        "strict_recognition_denominator": denominator,
        "raw_exact_match_count": sum(bool(row["raw_exact"]) for row in rows),
        "raw_exact_match_rate": sum(bool(row["raw_exact"]) for row in rows)
        / denominator,
        "normalized_exact_match_count": sum(
            bool(row["normalized_exact"]) for row in rows
        ),
        "normalized_exact_match_rate": sum(
            bool(row["normalized_exact"]) for row in rows
        )
        / denominator,
        "total_edit_distance": total_distance,
        "total_gt_characters": total_characters,
        "cer": total_distance / total_characters,
        "mean_sample_cer": sum(float(row["cer"]) for row in rows) / denominator,
        "renderable_count": sum(bool(row["renderable"]) for row in rows),
        "renderability_rate": sum(bool(row["renderable"]) for row in rows)
        / denominator,
        "material_error_count": sum(bool(row["material_error"]) for row in rows),
        "material_error_rate": sum(bool(row["material_error"]) for row in rows)
        / denominator,
        "error_category_counts": {
            category: category_counts[category]
            for category in sorted(RECOGNITION_ERROR_CATEGORIES)
        },
        "confusion_subtype_counts": {
            subtype: confusion_subtype_counts[subtype]
            for subtype in sorted(RECOGNITION_CONFUSION_SUBTYPES)
        },
        "quality_status": "MEASURED",
    }


def recognition_eligible(row: Mapping[str, Any]) -> bool:
    """Apply the strict Tier-B denominator contract."""

    return (
        row.get("match_class") == "ONE_TO_ONE"
        and row.get("crop_completeness") == "CROP_COMPLETE"
        and row.get("truth_status") == "SOURCE_LEGIBLE"
    )


def validate_denominator_binding_audit(
    expected_samples: Iterable[Mapping[str, Any]],
    annotations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless every current Tier-B sample has a valid visual binding.

    The visual adjudication itself is external evidence.  This function makes
    that evidence deterministic and non-reusable after truth, page, crop, or
    task inputs change.
    """

    expected_by_id: dict[str, dict[str, str]] = {}
    for raw_row in expected_samples:
        source_formula_id = str(raw_row.get("source_formula_id") or "").strip()
        if not source_formula_id or source_formula_id in expected_by_id:
            raise FormulaBenchmarkError(
                "DENOMINATOR_BINDING_EXPECTED_ID_INVALID_OR_DUPLICATE:"
                + source_formula_id
            )
        bindings = {
            field: str(raw_row.get(field) or "").strip()
            for field in DENOMINATOR_BINDING_FIELDS
        }
        if any(not value for value in bindings.values()):
            raise FormulaBenchmarkError(
                "DENOMINATOR_BINDING_EXPECTED_FIELDS_REQUIRED:" + source_formula_id
            )
        expected_by_id[source_formula_id] = bindings

    annotation_by_id: dict[str, dict[str, Any]] = {}
    for raw_row in annotations:
        source_formula_id = str(raw_row.get("source_formula_id") or "").strip()
        if not source_formula_id or source_formula_id in annotation_by_id:
            raise FormulaBenchmarkError(
                "DENOMINATOR_BINDING_ANNOTATION_ID_INVALID_OR_DUPLICATE:"
                + source_formula_id
            )
        annotation_by_id[source_formula_id] = dict(raw_row)

    expected_ids = set(expected_by_id)
    annotation_ids = set(annotation_by_id)
    missing_ids = sorted(expected_ids - annotation_ids)
    extra_ids = sorted(annotation_ids - expected_ids)
    stale_ids: list[str] = []
    malformed_ids: list[str] = []
    contaminated_ids: list[str] = []
    valid_ids: list[str] = []
    verdict_counts: dict[str, int] = {}

    for source_formula_id in sorted(expected_ids & annotation_ids):
        annotation = annotation_by_id[source_formula_id]
        verdict = str(annotation.get("verdict") or "").strip().upper()
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        contaminated = any(
            key.strip().lower() in _BLINDNESS_FORBIDDEN_KEYS
            for _, key in _walk_mapping_keys(annotation)
        )
        if contaminated:
            contaminated_ids.append(source_formula_id)
            malformed_ids.append(source_formula_id)
            continue
        if (
            verdict not in DENOMINATOR_BINDING_VERDICTS
            or not str(annotation.get("adjudicator") or "").strip()
            or annotation.get("prediction_visible_to_adjudicator") is not False
        ):
            malformed_ids.append(source_formula_id)
            continue
        observed_bindings = {
            field: str(annotation.get(field) or "").strip()
            for field in DENOMINATOR_BINDING_FIELDS
        }
        if observed_bindings != expected_by_id[source_formula_id]:
            stale_ids.append(source_formula_id)
            continue
        if verdict == "BINDING_VALID":
            valid_ids.append(source_formula_id)

    empty_expected_samples = not expected_by_id
    passed = (
        not empty_expected_samples
        and not missing_ids
        and not extra_ids
        and not stale_ids
        and not malformed_ids
        and len(valid_ids) == len(expected_by_id)
    )
    return {
        "schema": "bemarkdown-formula-denominator-binding-guard-v1",
        "expected_sample_count": len(expected_by_id),
        "annotation_count": len(annotation_by_id),
        "binding_valid_count": len(valid_ids),
        "empty_expected_samples": empty_expected_samples,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "missing_source_formula_ids": missing_ids,
        "extra_source_formula_ids": extra_ids,
        "stale_source_formula_ids": stale_ids,
        "malformed_source_formula_ids": malformed_ids,
        "contaminated_source_formula_ids": contaminated_ids,
        "passed": passed,
        "gate": (
            "DENOMINATOR_BINDING_AUDIT_PASS"
            if passed
            else "DENOMINATOR_BINDING_AUDIT_FAIL"
        ),
    }


def adjudicate_denominator_geometry_disputes(
    annotations: Iterable[Mapping[str, Any]],
    independent_review_rounds: Iterable[Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Resolve blind geometry disputes only after two exact source-only reviews."""

    rows = [dict(row) for row in annotations]
    rows_by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_formula_id = str(row.get("source_formula_id") or "").strip()
        if not source_formula_id or source_formula_id in rows_by_source:
            raise FormulaBenchmarkError(
                "DENOMINATOR_GEOMETRY_ANNOTATION_ID_INVALID_OR_DUPLICATE:"
                + source_formula_id
            )
        rows_by_source[source_formula_id] = row
    disputed_ids = sorted(
        source_formula_id
        for source_formula_id, row in rows_by_source.items()
        if row.get("verdict") == "GEOMETRY_MATCH_MISMATCH"
    )
    rounds = [list(round_rows) for round_rows in independent_review_rounds]
    if len(rounds) != 2:
        raise FormulaBenchmarkError(
            f"DENOMINATOR_GEOMETRY_TWO_REVIEWS_REQUIRED:{len(rounds)}"
        )
    normalized_rounds = []
    for round_index, raw_round in enumerate(rounds, 1):
        review_by_source: dict[str, dict[str, Any]] = {}
        for raw_review in raw_round:
            review = dict(raw_review)
            source_formula_id = str(
                review.get("source_formula_id") or ""
            ).strip()
            if not source_formula_id or source_formula_id in review_by_source:
                raise FormulaBenchmarkError(
                    "DENOMINATOR_GEOMETRY_REVIEW_ID_INVALID_OR_DUPLICATE:"
                    f"{round_index}:{source_formula_id}"
                )
            review_by_source[source_formula_id] = review
        if set(review_by_source) != set(disputed_ids):
            raise FormulaBenchmarkError(
                "DENOMINATOR_GEOMETRY_REVIEW_MEMBERSHIP_MISMATCH:"
                f"{round_index}"
            )
        for source_formula_id in disputed_ids:
            review = review_by_source[source_formula_id]
            annotation = rows_by_source[source_formula_id]
            if (
                review.get("decision") != "CURRENT_GEOMETRY_MATCH_VALID"
                or str(review.get("current_formula_content_id") or "")
                != str(annotation.get("formula_content_id") or "")
                or review.get("recommended_formula_content_id") is not None
                or review.get("recommended_visual_bbox_px") is not None
            ):
                raise FormulaBenchmarkError(
                    "DENOMINATOR_GEOMETRY_DISPUTE_NOT_RESOLVED:"
                    f"{round_index}:{source_formula_id}"
                )
        normalized_rounds.append(
            [review_by_source[source_formula_id] for source_formula_id in disputed_ids]
        )
    effective = []
    for source_formula_id in sorted(rows_by_source):
        row = rows_by_source[source_formula_id]
        if source_formula_id in disputed_ids:
            effective.append(
                {
                    **row,
                    "blind_verdict": row["verdict"],
                    "verdict": "BINDING_VALID",
                    "geometry_adjudication": (
                        "TWO_INDEPENDENT_SOURCE_ONLY_REVIEWS_"
                        "CURRENT_MATCH_VALID"
                    ),
                }
            )
        else:
            effective.append(row)
    original_counts = Counter(str(row.get("verdict")) for row in rows)
    effective_counts = Counter(str(row.get("verdict")) for row in effective)
    return {
        "schema": "bemarkdown-formula-denominator-geometry-adjudication-v1",
        "required_independent_review_round_count": 2,
        "disputed_source_formula_ids": disputed_ids,
        "adjudicated_source_formula_ids": disputed_ids,
        "original_verdict_counts": dict(sorted(original_counts.items())),
        "effective_verdict_counts": dict(sorted(effective_counts.items())),
        "independent_review_rounds": normalized_rounds,
        "effective_annotations": effective,
        "gate": "DENOMINATOR_GEOMETRY_ADJUDICATION_PASS",
    }


def validate_recognition_result_membership(
    expected_samples: Iterable[Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind recognition rows to the exact current Tier-B member and strings."""

    fields = (
        "formula_content_id",
        "gt_latex",
        "prediction_latex",
        "task_semantic_sha256",
    )
    expected_by_id: dict[str, dict[str, str]] = {}
    expected_content_ids: list[str] = []
    for raw_row in expected_samples:
        source_formula_id = str(raw_row.get("source_formula_id") or "").strip()
        if not source_formula_id or source_formula_id in expected_by_id:
            raise FormulaBenchmarkError(
                "RECOGNITION_EXPECTED_ID_INVALID_OR_DUPLICATE:" + source_formula_id
            )
        values = {field: str(raw_row.get(field) or "") for field in fields}
        if not values["formula_content_id"] or not values["task_semantic_sha256"]:
            raise FormulaBenchmarkError(
                "RECOGNITION_EXPECTED_BINDING_FIELDS_REQUIRED:" + source_formula_id
            )
        expected_by_id[source_formula_id] = values
        expected_content_ids.append(values["formula_content_id"])
    duplicate_expected_content_ids = sorted(
        content_id
        for content_id, count in Counter(expected_content_ids).items()
        if count > 1
    )
    if duplicate_expected_content_ids:
        raise FormulaBenchmarkError(
            "RECOGNITION_EXPECTED_CONTENT_ID_DUPLICATE:"
            + repr(duplicate_expected_content_ids)
        )

    result_by_id: dict[str, dict[str, Any]] = {}
    result_content_ids: list[str] = []
    for raw_row in results:
        source_formula_id = str(raw_row.get("source_formula_id") or "").strip()
        if not source_formula_id or source_formula_id in result_by_id:
            raise FormulaBenchmarkError(
                "RECOGNITION_RESULT_ID_INVALID_OR_DUPLICATE:" + source_formula_id
            )
        result_by_id[source_formula_id] = dict(raw_row)
        result_content_ids.append(str(raw_row.get("formula_content_id") or ""))

    expected_ids = set(expected_by_id)
    result_ids = set(result_by_id)
    missing_ids = sorted(expected_ids - result_ids)
    extra_ids = sorted(result_ids - expected_ids)
    stale_ids = []
    malformed_ids = []
    for source_formula_id in sorted(expected_ids & result_ids):
        observed = {
            field: str(result_by_id[source_formula_id].get(field) or "")
            for field in fields
        }
        if not observed["formula_content_id"] or not observed["task_semantic_sha256"]:
            malformed_ids.append(source_formula_id)
        elif observed != expected_by_id[source_formula_id]:
            stale_ids.append(source_formula_id)
    duplicate_result_content_ids = sorted(
        content_id
        for content_id, count in Counter(result_content_ids).items()
        if content_id and count > 1
    )
    empty_expected_samples = not expected_by_id
    passed = not empty_expected_samples and not any(
        (
            missing_ids,
            extra_ids,
            stale_ids,
            malformed_ids,
            duplicate_result_content_ids,
        )
    )
    return {
        "schema": "bemarkdown-formula-recognition-membership-guard-v1",
        "expected_sample_count": len(expected_by_id),
        "result_count": len(result_by_id),
        "empty_expected_samples": empty_expected_samples,
        "missing_source_formula_ids": missing_ids,
        "extra_source_formula_ids": extra_ids,
        "stale_source_formula_ids": stale_ids,
        "malformed_source_formula_ids": malformed_ids,
        "duplicate_formula_content_ids": duplicate_result_content_ids,
        "passed": passed,
        "gate": (
            "RECOGNITION_RESULT_MEMBERSHIP_PASS"
            if passed
            else "RECOGNITION_RESULT_MEMBERSHIP_FAIL"
        ),
    }


def _walk_mapping_keys(value: Any, prefix: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            yield child_path, str(key)
            yield from _walk_mapping_keys(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk_mapping_keys(child, f"{prefix}[{index}]")


def validate_truth_blindness(
    payload: Mapping[str, Any], *, allowed_keys: Iterable[str] | None = None
) -> dict[str, Any]:
    """Reject truth-stage payloads that expose prediction-side fields."""

    if allowed_keys is not None:
        allowed = {str(key) for key in allowed_keys}
        forbidden_fields = sorted(str(key) for key in payload if str(key) not in allowed)
        if forbidden_fields:
            raise FormulaBenchmarkError(
                "TRUTH_PAYLOAD_FIELDS_FORBIDDEN:" + repr(forbidden_fields)
            )
    contaminated = []
    for path, key in _walk_mapping_keys(payload):
        canonical = key.strip().lower()
        if canonical in _BLINDNESS_FORBIDDEN_KEYS or any(
            fragment in canonical for fragment in _BLINDNESS_FORBIDDEN_KEY_FRAGMENTS
        ):
            contaminated.append({"path": path, "key": key})
    if contaminated:
        raise FormulaBenchmarkError(
            "BENCHMARK_TRUTH_CONTAMINATED:" + ",".join(row["path"] for row in contaminated)
        )
    return {
        "schema": "bemarkdown-formula-truth-blindness-check-v1",
        "forbidden_keys": sorted(_BLINDNESS_FORBIDDEN_KEYS),
        "forbidden_key_fragments": sorted(_BLINDNESS_FORBIDDEN_KEY_FRAGMENTS),
        "allowed_keys_enforced": allowed_keys is not None,
        "contaminated_fields": [],
        "gate": "PASS",
    }


def _bbox(value: Sequence[float]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise FormulaBenchmarkError(f"INVALID_BBOX_LENGTH:{value!r}")
    left, top, right, bottom = (float(part) for part in value)
    if not all(math.isfinite(part) for part in (left, top, right, bottom)):
        raise FormulaBenchmarkError(f"INVALID_BBOX_NON_FINITE:{value!r}")
    if right <= left or bottom <= top:
        raise FormulaBenchmarkError(f"INVALID_BBOX_GEOMETRY:{value!r}")
    return left, top, right, bottom


def _area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _contains(
    box: tuple[float, float, float, float], point: tuple[float, float]
) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _geometry_edge(
    source: tuple[float, float, float, float],
    production: tuple[float, float, float, float],
) -> tuple[bool, dict[str, Any]]:
    intersection = _intersection_area(source, production)
    source_area = _area(source)
    production_area = _area(production)
    union = source_area + production_area - intersection
    iou = intersection / union if union else 0.0
    source_coverage = intersection / source_area if source_area else 0.0
    production_coverage = intersection / production_area if production_area else 0.0
    production_center_in_source = _contains(source, _center(production))
    source_center_in_production = _contains(production, _center(source))
    matched = (
        intersection > 0
        and (
            production_center_in_source
            or source_center_in_production
            or iou >= 0.1
            or source_coverage >= 0.5
            or production_coverage >= 0.5
        )
    )
    return matched, {
        "intersection_area": round(intersection, 9),
        "iou": round(iou, 9),
        "source_coverage": round(source_coverage, 9),
        "production_coverage": round(production_coverage, 9),
        "production_center_in_source": production_center_in_source,
        "source_center_in_production": source_center_in_production,
    }


@dataclass(frozen=True)
class GeometryMatchResult:
    source_matches: tuple[dict[str, Any], ...]
    false_positive_formula_content_ids: tuple[str, ...]


def classify_geometry_matches(
    source_rows: Sequence[Mapping[str, Any]],
    production_rows: Sequence[Mapping[str, Any]],
) -> GeometryMatchResult:
    """Classify a page's source-to-production geometry as a bipartite graph."""

    supplied_page_keys = {
        (str(row["document_id"]), int(row["page_index"]))
        for row in (*source_rows, *production_rows)
        if row.get("document_id") is not None and row.get("page_index") is not None
    }
    if len(supplied_page_keys) > 1:
        raise FormulaBenchmarkError(
            "GEOMETRY_MATCH_REQUIRES_ONE_DOCUMENT_PAGE:"
            + repr(sorted(supplied_page_keys))
        )

    source_by_id = {str(row["source_formula_id"]): row for row in source_rows}
    production_by_id = {
        str(row["formula_content_id"]): row for row in production_rows
    }
    if len(source_by_id) != len(source_rows):
        raise FormulaBenchmarkError("DUPLICATE_SOURCE_FORMULA_ID")
    if len(production_by_id) != len(production_rows):
        raise FormulaBenchmarkError("DUPLICATE_FORMULA_CONTENT_ID")

    source_edges: dict[str, set[str]] = defaultdict(set)
    production_edges: dict[str, set[str]] = defaultdict(set)
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for source_id, source_row in source_by_id.items():
        source_bbox = _bbox(source_row["visual_bbox_pdf_pt"])
        for content_id, production_row in production_by_id.items():
            production_bbox = _bbox(production_row["bbox_pdf_pt"])
            matched, row_evidence = _geometry_edge(source_bbox, production_bbox)
            if matched:
                source_edges[source_id].add(content_id)
                production_edges[content_id].add(source_id)
                evidence[(source_id, content_id)] = row_evidence

    component_by_source: dict[str, tuple[set[str], set[str]]] = {}
    visited_sources: set[str] = set()
    for initial_source in source_by_id:
        if initial_source in visited_sources or not source_edges[initial_source]:
            continue
        component_sources: set[str] = set()
        component_productions: set[str] = set()
        queue: deque[tuple[str, str]] = deque((("source", initial_source),))
        while queue:
            kind, identifier = queue.popleft()
            if kind == "source":
                if identifier in component_sources:
                    continue
                component_sources.add(identifier)
                visited_sources.add(identifier)
                queue.extend(("production", value) for value in source_edges[identifier])
            else:
                if identifier in component_productions:
                    continue
                component_productions.add(identifier)
                queue.extend(("source", value) for value in production_edges[identifier])
        for source_id in component_sources:
            component_by_source[source_id] = (
                component_sources,
                component_productions,
            )

    source_matches = []
    for source_id in source_by_id:
        component = component_by_source.get(source_id)
        if component is None:
            match_class = "MISSING"
            matched_ids: list[str] = []
            component_source_ids = [source_id]
        else:
            component_sources, component_productions = component
            matched_ids = sorted(component_productions)
            component_source_ids = sorted(component_sources)
            counts = (len(component_sources), len(component_productions))
            if counts == (1, 1):
                match_class = "ONE_TO_ONE"
            elif counts[0] == 1 and counts[1] > 1:
                match_class = "SPLIT"
            elif counts[0] > 1 and counts[1] == 1:
                match_class = "MERGED"
            else:
                match_class = "GEOMETRY_AMBIGUOUS"
        source_matches.append(
            {
                "source_formula_id": source_id,
                "matched_formula_content_ids": matched_ids,
                "match_class": match_class,
                "geometry_evidence": [
                    {
                        "formula_content_id": content_id,
                        **evidence[(source_id, content_id)],
                    }
                    for content_id in matched_ids
                    if (source_id, content_id) in evidence
                ],
                "component_source_formula_ids": component_source_ids,
            }
        )

    false_positives = tuple(
        sorted(
            content_id
            for content_id in production_by_id
            if not production_edges[content_id]
        )
    )
    return GeometryMatchResult(tuple(source_matches), false_positives)


def classify_crop_completeness(
    source_bbox: Sequence[float],
    crop_bbox: Sequence[float],
    *,
    tolerance_pt: float = 1.5,
    excessive_area_ratio: float = 4.0,
) -> str:
    """Classify crop geometry without looking at prediction text."""

    source = _bbox(source_bbox)
    crop = _bbox(crop_bbox)
    covers = (
        crop[0] <= source[0] + tolerance_pt
        and crop[1] <= source[1] + tolerance_pt
        and crop[2] >= source[2] - tolerance_pt
        and crop[3] >= source[3] - tolerance_pt
    )
    if not covers:
        return "CROP_TRUNCATED"
    if _area(crop) / _area(source) > excessive_area_ratio:
        return "CROP_EXCESSIVE"
    return "CROP_COMPLETE"
