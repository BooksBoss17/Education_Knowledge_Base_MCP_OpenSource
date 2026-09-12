from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .blind_vision_formula import build_vision_formula_resegmentation_ir
from .formulanet_runtime import FormulaOcrOutputValidator, OcrVerdict
from .vision_evaluation_context import quality_gate_context_decision

FINAL_RESOLUTIONS = {
    "CONFIRMED_CURRENT",
    "REPLACED",
    "RECOVERED_MISSING",
    "RESEGMENTED_FORMULA",
    "NO_FORMULA_VISIBLE",
    "UNRESOLVED",
}

PROTOCOL_FREEZE_COMPONENTS = (
    "blind_request_schema_sha256",
    "blind_instruction_sha256",
    "blind_image_layout_sha256",
    "blind_importer_contract_sha256",
    "comparator_contract_sha256",
    "resegmentation_contract_sha256",
    "pass_b_request_schema_sha256",
    "pass_b_instruction_sha256",
    "candidate_renderer_sha256",
    "candidate_randomization_sha256",
    "origin_firewall_sha256",
    "pass_b_parser_contract_sha256",
    "final_resolution_policy_sha256",
    "human_fallback_policy_sha256",
    "context_isolation_contract_sha256",
)


class ReferenceTruthLocked(RuntimeError):
    """Raised before the recorded final-resolution gate authorizes truth access."""


def validate_project_runtime_identity(
    executable: str | Path,
    *,
    expected_runtime: str | Path,
    developer_root: str | Path,
    python_version: tuple[int, int],
    implementation: str,
) -> dict[str, Any]:
    resolved = Path(executable).resolve()
    expected = Path(expected_runtime).resolve()
    developer = str(Path(developer_root).resolve()).casefold()
    lowered = str(resolved).casefold()
    checks = {
        "exact_project_runtime": resolved == expected,
        "cpython_3_11": implementation == "CPython" and python_version == (3, 11),
        "developer_owned": lowered.startswith(developer),
        "not_hermes": "hermes" not in lowered,
        "not_codex_bundled": ".codex" not in lowered,
        "not_python_3_13": python_version != (3, 13),
    }
    return {
        "schema": "bemarkdown-project-runtime-identity-v0",
        "executable": str(resolved),
        "expected_runtime": str(expected),
        "checks": checks,
        "verified": all(checks.values()),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    material = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_reference_truth_unlock_record(
    path: str | Path,
    *,
    completeness: dict[str, Any],
    pass_a_context: dict[str, Any],
    pass_b_context: dict[str, Any],
    runtime_verified: bool,
    pass_a_provenance_eligible: bool,
    pass_b_provenance_eligible: bool,
    validation_sealed: bool,
    validation_consumed: bool,
    final_resolution_sha256: str,
    reference_truth_path: str | Path,
) -> dict[str, Any]:
    context = quality_gate_context_decision(
        evaluation_role="REFERENCE_QUALITY_EVALUATION",
        pass_a_context=pass_a_context,
        pass_b_context=pass_b_context,
        pass_b_required=True,
        truth_loaded_before_final_resolution=False,
    )
    zero_fields = (
        "pending_pass_b",
        "unmapped_pass_b",
        "invalid_final_latex",
        "duplicate_final_resolution",
        "missing_final_resolution",
    )
    checks = {
        "runtime_correction_passed": runtime_verified is True,
        "pass_a_provenance_eligible": pass_a_provenance_eligible is True,
        "pass_b_provenance_eligible": pass_b_provenance_eligible is True,
        "context_isolation_eligible": context["quality_gate_eligible"] is True,
        "final_resolution_complete": completeness.get("complete") is True,
        "expected_166_rows": completeness.get("expected_formula_count") == 166,
        "resolved_166_unique_rows": completeness.get("final_resolution_count") == 166
        and completeness.get("unique_final_resolution_count") == 166,
        "no_pending_or_invalid_rows": all(
            completeness.get(field) == 0 for field in zero_fields
        ),
        "validation_sealed": validation_sealed is True,
        "validation_not_consumed": validation_consumed is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ReferenceTruthLocked(
            "REFERENCE_TRUTH_ACCESS_FORBIDDEN_BEFORE_FINAL_RESOLUTION: "
            + ",".join(failed)
        )
    truth = Path(reference_truth_path).resolve()
    if "validation" in truth.name.lower():
        raise ReferenceTruthLocked("Validation truth can never be unlocked by Reference gate")
    record = {
        "schema": "bemarkdown-reference-truth-unlock-record-v0",
        "status": "REFERENCE_TRUTH_UNLOCKED_AFTER_RESOLUTION",
        "reference_truth_access_allowed": True,
        "checks": checks,
        "context_decision": context,
        "final_resolution_sha256": final_resolution_sha256,
        "reference_truth_integrity": {
            "path": str(truth),
            "sha256": _sha256_file(truth),
            "bytes": truth.stat().st_size,
            "content_parsed_at_record_creation": False,
        },
        "validation_truth_access_allowed": False,
    }
    _write_json(Path(path), record)
    return record


def load_reference_truth_after_unlock(
    truth_path: str | Path, unlock_record_path: str | Path
) -> list[dict[str, Any]]:
    truth = Path(truth_path).resolve()
    if "validation" in truth.name.lower():
        raise ReferenceTruthLocked("Validation truth remains sealed")
    unlock = Path(unlock_record_path)
    if not unlock.is_file():
        raise ReferenceTruthLocked("Reference truth unlock record is absent")
    record = json.loads(unlock.read_text(encoding="utf-8"))
    expected = record.get("reference_truth_integrity", {})
    if (
        record.get("status") != "REFERENCE_TRUTH_UNLOCKED_AFTER_RESOLUTION"
        or record.get("reference_truth_access_allowed") is not True
        or record.get("validation_truth_access_allowed") is not False
        or str(truth) != expected.get("path")
        or _sha256_file(truth) != expected.get("sha256")
        or truth.stat().st_size != expected.get("bytes")
    ):
        raise ReferenceTruthLocked("Reference truth unlock record does not bind this file")
    return [
        json.loads(line)
        for line in truth.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_reference_quality(
    final_rows: list[dict[str, Any]],
    reference_truth_rows: list[dict[str, Any]],
    quality_reviews: dict[str, dict[str, Any]],
    *,
    expected_certain: int | None = None,
    expected_material: int | None = None,
    expected_acceptable: int | None = None,
) -> dict[str, Any]:
    """Score the frozen certain universe from explicit final-quality reviews."""

    final_by_id = _unique_index(final_rows, "formula_id", label="Final resolution")
    truth_by_id = _unique_index(reference_truth_rows, "content_id", label="Reference truth")
    acceptable_labels = {"CORRECT", "MINOR_ERROR"}
    material_labels = {"MAJOR_ERROR", "UNUSABLE"}
    supported_labels = acceptable_labels | material_labels
    rows: list[dict[str, Any]] = []
    for formula_id, truth in sorted(truth_by_id.items()):
        original_quality = truth.get("formulanet_label")
        final = final_by_id.get(formula_id)
        if final is None:
            raise ValueError(f"Reference truth formula has no final resolution: {formula_id}")
        resolution = str(final["final_resolution"])
        review = quality_reviews.get(formula_id)
        if original_quality == "REFERENCE_UNCERTAIN":
            final_quality = "REFERENCE_UNCERTAIN"
        elif original_quality not in supported_labels:
            final_quality = "NO_FROZEN_ACCURACY_TRUTH"
        elif resolution == "CONFIRMED_CURRENT":
            final_quality = str(original_quality)
        elif resolution == "UNRESOLVED":
            final_quality = "UNRESOLVED"
        elif resolution == "NO_FORMULA_VISIBLE":
            final_quality = "UNUSABLE"
        else:
            if not isinstance(review, dict):
                raise ValueError(
                    f"Changed frozen-truth formula requires a visual quality review: {formula_id}"
                )
            if review.get("reviewed_final_latex") != final.get("final_latex"):
                raise ValueError(
                    f"Visual quality review does not bind final LaTeX: {formula_id}"
                )
            final_quality = str(review.get("final_quality"))
            if final_quality not in supported_labels:
                raise ValueError("Visual quality review has an unsupported final quality")
        rows.append(
            {
                **final,
                "reference_id": truth.get("reference_id"),
                "truth_label": original_quality,
                "source_truth_latex": truth.get("source_truth_latex"),
                "truth_crop_path": truth.get("crop_path"),
                "truth_crop_sha256": truth.get("crop_sha256"),
                "final_quality": final_quality,
                "quality_review": review,
            }
        )

    certain = [row for row in rows if row["truth_label"] in supported_labels]
    material = [row for row in certain if row["truth_label"] in material_labels]
    acceptable = [row for row in certain if row["truth_label"] in acceptable_labels]
    expected_checks = {
        "certain_count": expected_certain is None or len(certain) == expected_certain,
        "material_count": expected_material is None or len(material) == expected_material,
        "acceptable_count": expected_acceptable is None
        or len(acceptable) == expected_acceptable,
    }
    if not all(expected_checks.values()):
        raise ValueError("Frozen Reference accuracy universe does not match expected counts")

    material_acceptable = sum(
        row["final_quality"] in acceptable_labels for row in material
    )
    acceptable_damage = [
        row for row in acceptable if row["final_quality"] in material_labels
    ]
    unnecessary_fallback = [
        row for row in acceptable if row["final_quality"] == "UNRESOLVED"
    ]
    gate_passed = (
        material_acceptable == len(material)
        and len(acceptable_damage) <= 1
        and len(unnecessary_fallback) <= 1
    )
    metrics = {
        "schema": "bemarkdown-vision-reference-quality-metrics-v2",
        "truth_cohort_consumed": "REFERENCE_ONLY",
        "accuracy_denominator": len(certain),
        "uncertain_excluded": sum(
            row["truth_label"] == "REFERENCE_UNCERTAIN" for row in rows
        ),
        "no_truth_rows_excluded": sum(
            row["final_quality"] == "NO_FROZEN_ACCURACY_TRUTH" for row in rows
        ),
        "reference_material_total": len(material),
        "reference_material_final_acceptable": material_acceptable,
        "reference_material_final_material": sum(
            row["final_quality"] in material_labels for row in material
        ),
        "reference_material_unresolved": sum(
            row["final_quality"] == "UNRESOLVED" for row in material
        ),
        "reference_acceptable_total": len(acceptable),
        "reference_acceptable_final_acceptable": sum(
            row["final_quality"] in acceptable_labels for row in acceptable
        ),
        "reference_acceptable_materially_damaged": len(acceptable_damage),
        "reference_acceptable_unresolved": len(unnecessary_fallback),
        "unnecessary_human_fallback": len(unnecessary_fallback),
        "gate_passed": gate_passed,
        "status": (
            "VISION_REFERENCE_QUALITY_SIGNAL_PASSED"
            if gate_passed
            else "VISION_REFERENCE_QUALITY_SIGNAL_FAILED"
        ),
    }
    return {
        "metrics": metrics,
        "rows": rows,
        "material_cases": material,
        "acceptable_damage_cases": acceptable_damage,
        "human_fallback_cases": unnecessary_fallback,
    }


def build_protocol_freeze(
    quality_metrics: dict[str, Any], components: dict[str, str]
) -> dict[str, Any]:
    if quality_metrics.get("gate_passed") is not True:
        raise ValueError("Reference quality gate has not passed")
    missing = sorted(set(PROTOCOL_FREEZE_COMPONENTS) - set(components))
    extra = sorted(set(components) - set(PROTOCOL_FREEZE_COMPONENTS))
    if missing or extra:
        raise ValueError(f"Protocol freeze component mismatch: missing={missing}, extra={extra}")
    if any(not isinstance(components[name], str) or not components[name] for name in components):
        raise ValueError("Protocol freeze component fingerprints must be non-empty strings")
    frozen = {
        "schema": "bemarkdown-vision-reference-protocol-freeze-v0",
        "status": "VISION_REFERENCE_PROTOCOL_FROZEN",
        "quality_gate_status": quality_metrics.get("status"),
        "components": {name: components[name] for name in sorted(components)},
        "policies": {
            "both_equivalent": "PRESERVE_CURRENT",
            "candidate_current": "CONFIRMED_CURRENT",
            "candidate_blind": "REPLACED",
            "unresolved": "HUMAN_REVIEW_PENDING",
            "validation_membership": "FIXED_12_PAGES_46_CERTAIN_44_ACCEPTABLE_2_MATERIAL",
        },
    }
    frozen["protocol_fingerprint_sha256"] = _canonical_sha256(frozen)
    return frozen


def _unique_index(rows: list[dict[str, Any]], key: str, *, label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in index:
            raise ValueError(f"{label} requires unique non-empty {key}")
        index[value] = row
    return index


def _machine_validation(
    latex: str | None,
    *,
    render_validator: Callable[[str], bool],
) -> dict[str, Any]:
    syntax = FormulaOcrOutputValidator().validate(latex)
    syntax_valid = syntax.verdict in {OcrVerdict.VALID, OcrVerdict.VALID_WITH_WARNING}
    project_format = bool(latex and "```" not in latex and "formula-review" not in latex)
    renderable = bool(latex and render_validator(latex))
    return {
        "syntax": syntax.to_dict(),
        "syntax_valid": syntax_valid,
        "renderable": renderable,
        "project_format": project_format,
        "valid": syntax_valid and renderable and project_format,
    }


def _pass_b_resolution(
    result: dict[str, Any],
    private: dict[str, Any],
    current_latex: str,
) -> tuple[str, str | None, str | None, str | None]:
    decision = str(result["decision"])
    if str(private.get("formula_id")) != str(result["formula_id"]):
        raise ValueError("PASS B private mapping formula ID mismatch")
    if decision == "BOTH_EQUIVALENT":
        return "CONFIRMED_CURRENT", current_latex, None, None
    if decision in {"CANDIDATE_A", "CANDIDATE_B"}:
        suffix = "a" if decision == "CANDIDATE_A" else "b"
        origin = str(private.get(f"candidate_{suffix}_origin", ""))
        latex = private.get(f"candidate_{suffix}_latex")
        if origin == "FORMULANET_CURRENT":
            return "CONFIRMED_CURRENT", current_latex, origin, str(latex)
        if origin == "PASS_A_VISUAL":
            return "REPLACED", str(latex), origin, str(latex)
        raise ValueError("PASS B private mapping has an unsupported candidate origin")
    if decision == "NEITHER_SOURCE_LATEX":
        return "REPLACED", str(result["latex"]), "PASS_B_SOURCE_LATEX", None
    if decision == "REGION_NOT_SINGLE_FORMULA":
        return "RESEGMENTED_FORMULA", None, "PASS_B_RESEGMENTATION", None
    if decision == "UNRESOLVED":
        return "UNRESOLVED", None, None, None
    raise ValueError("PASS B result has an unsupported decision")


def resolve_reference_final_set(
    partial_rows: list[dict[str, Any]],
    pass_b_rows: list[dict[str, Any]],
    private_origin_cases: list[dict[str, Any]],
    formula_index: dict[str, dict[str, Any]],
    *,
    render_validator: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Merge truth-free PASS A/PASS B evidence into one final row per formula."""

    render = render_validator or (lambda value: bool(value.strip()))
    partial_by_id = _unique_index(partial_rows, "formula_id", label="Partial resolution")
    pass_b_by_formula = _unique_index(pass_b_rows, "formula_id", label="PASS B result")
    pass_b_by_case = _unique_index(pass_b_rows, "case_id", label="PASS B result")
    private_by_case = _unique_index(private_origin_cases, "case_id", label="Private origin map")
    pending_ids = {
        formula_id
        for formula_id, row in partial_by_id.items()
        if row.get("resolution_state") == "PENDING_PASS_B"
    }
    if set(pass_b_by_formula) != pending_ids:
        raise ValueError("PASS B formula set must equal pending disagreement set")
    if set(pass_b_by_case) != set(private_by_case):
        raise ValueError("PASS B result and private mapping case sets differ")
    if set(partial_by_id) != set(formula_index):
        raise ValueError("Formula metadata set must equal partial resolution set")

    rows: list[dict[str, Any]] = []
    invalid_final_latex = 0
    for formula_id in sorted(partial_by_id):
        partial = partial_by_id[formula_id]
        formula = formula_index[formula_id]
        current_latex = formula.get("current_latex")
        partial_state = str(partial.get("resolution_state"))
        pass_b_result = pass_b_by_formula.get(formula_id)
        pass_b_case_id: str | None = None
        pass_b_decision: str | None = None
        chosen_origin: str | None = None
        blind_latex: str | None = None
        resegmentation_ir = partial.get("resegmentation_ir")

        if partial_state == "PENDING_PASS_B":
            assert pass_b_result is not None
            pass_b_case_id = str(pass_b_result["case_id"])
            pass_b_decision = str(pass_b_result["decision"])
            final_resolution, final_latex, chosen_origin, chosen_latex = _pass_b_resolution(
                pass_b_result,
                private_by_case[pass_b_case_id],
                str(current_latex),
            )
            private = private_by_case[pass_b_case_id]
            if "candidate_a_latex" in private and "candidate_b_latex" in private:
                blind_latex = (
                    str(private["candidate_a_latex"])
                    if private.get("candidate_a_origin") == "PASS_A_VISUAL"
                    else str(private["candidate_b_latex"])
                )
            if chosen_latex is not None and chosen_origin == "PASS_A_VISUAL":
                blind_latex = chosen_latex
            if final_resolution == "RESEGMENTED_FORMULA":
                try:
                    resegmentation_ir = build_vision_formula_resegmentation_ir(
                        formula_id=formula_id,
                        audit_id=str(partial["audit_id"]),
                        source_bbox_pdf_pt=formula["bbox_pdf_pt"],
                        formula_subregions=pass_b_result["formula_subregions"],
                        render_validator=render,
                    )
                    final_latex = resegmentation_ir["primary_formula_resolution"]
                except (KeyError, TypeError, ValueError) as exc:
                    final_resolution = "UNRESOLVED"
                    final_latex = None
                    resegmentation_ir = {
                        "schema": "bemarkdown-pass-b-resegmentation-fallback-v0",
                        "formula_id": formula_id,
                        "reason": str(exc),
                    }
        elif partial_state == "BLIND_INDEPENDENT_MATCH":
            final_resolution = "CONFIRMED_CURRENT"
            final_latex = str(partial["resolved_latex"])
            blind_latex = final_latex
        elif partial_state == "RECOVERED_MISSING":
            final_resolution = "RECOVERED_MISSING"
            final_latex = str(partial["resolved_latex"])
            blind_latex = final_latex
        elif partial_state == "RESEGMENTED_FORMULA":
            final_resolution = "RESEGMENTED_FORMULA"
            final_latex = resegmentation_ir.get("primary_formula_resolution") if isinstance(resegmentation_ir, dict) else None
        elif partial_state == "NO_FORMULA_VISIBLE":
            final_resolution = "NO_FORMULA_VISIBLE"
            final_latex = None
        elif partial_state == "HUMAN_FALLBACK_CANDIDATE":
            final_resolution = "UNRESOLVED"
            final_latex = None
        else:
            raise ValueError(f"Unsupported partial resolution state: {partial_state}")

        validation: dict[str, Any] | None = None
        if final_resolution in {"CONFIRMED_CURRENT", "REPLACED", "RECOVERED_MISSING"}:
            validation = _machine_validation(final_latex, render_validator=render)
            if not validation["valid"]:
                invalid_final_latex += 1
        elif final_resolution == "RESEGMENTED_FORMULA":
            valid_resegmentation = bool(
                isinstance(resegmentation_ir, dict)
                and resegmentation_ir.get("formula_id") == formula_id
                and (
                    resegmentation_ir.get("status") == "VISION_RESEGMENTED"
                )
            )
            validation = {
                "valid": valid_resegmentation,
                "resegmentation_ir_valid": valid_resegmentation,
            }
            if not valid_resegmentation:
                invalid_final_latex += 1

        rows.append(
            {
                "schema": "bemarkdown-vision-reference-final-resolution-v0",
                "formula_id": formula_id,
                "audit_id": str(partial["audit_id"]),
                "document_id": formula.get("document_id"),
                "page_index": formula.get("page_index"),
                "bbox_normalized": formula.get("bbox_normalized"),
                "bbox_pdf_pt": formula.get("bbox_pdf_pt"),
                "current_latex_state": str(partial["current_latex_state"]),
                "current_latex": current_latex,
                "blind_latex": blind_latex,
                "pass_b_case_id": pass_b_case_id,
                "pass_b_decision": pass_b_decision,
                "pass_b_chosen_origin": chosen_origin,
                "final_resolution": final_resolution,
                "final_latex": final_latex,
                "resegmentation_ir": resegmentation_ir,
                "human_fallback": final_resolution == "UNRESOLVED",
                "machine_validation": validation,
                "final_resolution_complete": True,
                "truth_loaded": False,
            }
        )

    counts = Counter(row["final_resolution"] for row in rows)
    missing = sorted(set(formula_index) - {row["formula_id"] for row in rows})
    duplicates = len(rows) - len({row["formula_id"] for row in rows})
    completeness = {
        "schema": "bemarkdown-vision-reference-resolution-completeness-v0",
        "expected_formula_count": len(formula_index),
        "final_resolution_count": len(rows),
        "unique_final_resolution_count": len({row["formula_id"] for row in rows}),
        "pending_pass_b": 0,
        "unmapped_pass_b": 0,
        "invalid_final_latex": invalid_final_latex,
        "duplicate_final_resolution": duplicates,
        "missing_final_resolution": len(missing),
        "missing_formula_ids": missing,
        "complete": not any((invalid_final_latex, duplicates, len(missing))),
    }
    return {
        "rows": rows,
        "completeness": completeness,
        "behavior_metrics": {
            "schema": "bemarkdown-full-reference-behavior-metrics-v0",
            "formula_count": len(rows),
            "final_resolution_counts": dict(sorted(counts.items())),
            "pass_b_both_equivalent": sum(
                row["pass_b_decision"] == "BOTH_EQUIVALENT" for row in rows
            ),
            "pass_b_chose_current": sum(
                row["pass_b_chosen_origin"] == "FORMULANET_CURRENT" for row in rows
            ),
            "pass_b_chose_blind": sum(
                row["pass_b_chosen_origin"] == "PASS_A_VISUAL" for row in rows
            ),
            "accuracy_claim": False,
        },
    }
