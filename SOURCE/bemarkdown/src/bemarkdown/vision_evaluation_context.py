from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

EVALUATION_CONTEXT_SCHEMA = "bemarkdown-vision-evaluation-context-isolation-v0"
EVALUATION_ROLE_DIAGNOSTIC = "DEVELOPMENT_DIAGNOSTIC_ONLY"
FRESH_CONTEXT_ROLE = "FRESH_ISOLATED"
CONTEXT_ROLES = {FRESH_CONTEXT_ROLE, "CONTAMINATED_DIAGNOSTIC", "UNKNOWN"}
EVALUATION_STAGES = {"PASS_A", "PASS_B"}

EVALUATION_CONTEXT_ISOLATION_CONTRACT = {
    "schema": EVALUATION_CONTEXT_SCHEMA,
    "context_roles": sorted(CONTEXT_ROLES),
    "quality_gate_requires": {
        "pass_a_context_role": FRESH_CONTEXT_ROLE,
        "pass_b_context_role_when_required": FRESH_CONTEXT_ROLE,
        "pass_b_context_independent_from_pass_a": True,
        "prior_project_history_exposed": False,
        "prior_candidate_history_exposed": False,
        "pass_b_prior_pass_a_raw_result_exposed": False,
        "pass_b_candidate_origins_exposed": False,
        "operator_attestation": True,
        "truth_loaded_before_final_resolution": False,
    },
    "fail_closed": True,
}

FRESH_EXTERNAL_RESULT_INTAKE_CONTRACT = {
    "schema": "bemarkdown-fresh-external-result-intake-v0",
    "modes": [
        "fresh-reference-pass-a",
        "fresh-reference-pass-b",
    ],
    "required_operator_attestations": [
        "fresh_isolated_context",
        "no_prior_project_history",
        "canonical_bundle_sha256_verified",
    ],
    "pass_b_additional_attestations": [
        "different_context_from_pass_a",
        "pass_a_raw_result_not_exposed",
        "candidate_origins_not_exposed",
    ],
    "unknown_or_missing_attestation": "QUALITY_GATE_INELIGIBLE",
}

NO_TRUTH_SCORE_GUARD = {
    "schema": "bemarkdown-no-truth-score-guard-v0",
    "diagnostic_role": EVALUATION_ROLE_DIAGNOSTIC,
    "diagnostic_accuracy_evaluation": "REJECT",
    "truth_before_full_resolution": "REJECT",
    "ineligible_context": "REJECT",
    "diagnostic_output": "DISTRIBUTION_ONLY_NO_ACCURACY_CLAIM",
}

ADJUDICATION_EXTERNAL_FORBIDDEN_FRAGMENTS = (
    "formulanet",
    "blind",
    "current_latex",
    "blind_latex",
    "candidate_origin",
    "material",
    "acceptable",
    "truth",
    "expected",
)


class DiagnosticEvidenceNotEligibleForQualityGate(RuntimeError):
    """Raised before truth is consumed for diagnostic-only evidence."""


class EvaluationContextNotEligibleForQualityGate(RuntimeError):
    """Raised before truth is consumed when context isolation is insufficient."""


def build_evaluation_context_isolation_ir(
    *,
    stage: str,
    context_role: str,
    context_id: str,
    prior_project_history_exposed: bool,
    prior_candidate_history_exposed: bool,
    operator_attestation: bool,
    prior_pass_a_raw_result_exposed: bool = False,
    candidate_origins_exposed: bool = False,
) -> dict[str, Any]:
    normalized_stage = str(stage).upper()
    normalized_role = str(context_role).upper()
    if normalized_stage not in EVALUATION_STAGES:
        raise ValueError("Evaluation context stage must be PASS_A or PASS_B")
    if normalized_role not in CONTEXT_ROLES:
        raise ValueError("Unsupported evaluation context role")
    if not isinstance(context_id, str) or not context_id.strip():
        raise ValueError("Evaluation context ID is required")
    if normalized_stage == "PASS_A" and (
        prior_pass_a_raw_result_exposed or candidate_origins_exposed
    ):
        raise ValueError("PASS A cannot carry PASS B-only exposure fields")
    return {
        "schema": EVALUATION_CONTEXT_SCHEMA,
        "stage": normalized_stage,
        "context_role": normalized_role,
        "context_id": context_id.strip(),
        "prior_project_history_exposed": bool(prior_project_history_exposed),
        "prior_candidate_history_exposed": bool(prior_candidate_history_exposed),
        "prior_pass_a_raw_result_exposed": bool(prior_pass_a_raw_result_exposed),
        "candidate_origins_exposed": bool(candidate_origins_exposed),
        "operator_attestation": operator_attestation is True,
    }


def quality_gate_context_decision(
    *,
    evaluation_role: str,
    pass_a_context: dict[str, Any] | None,
    pass_b_context: dict[str, Any] | None = None,
    pass_b_required: bool = False,
    truth_loaded_before_final_resolution: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    if str(evaluation_role).upper() == EVALUATION_ROLE_DIAGNOSTIC:
        reasons.append("DIAGNOSTIC_EVIDENCE_ROLE")
    if truth_loaded_before_final_resolution:
        reasons.append("TRUTH_LOADED_BEFORE_FINAL_RESOLUTION")
    reasons.extend(_context_reasons(pass_a_context, stage="PASS_A"))
    if pass_b_required:
        reasons.extend(_context_reasons(pass_b_context, stage="PASS_B"))
        if pass_a_context and pass_b_context:
            if pass_a_context.get("context_id") == pass_b_context.get("context_id"):
                reasons.append("PASS_B_CONTEXT_NOT_INDEPENDENT")
            if pass_b_context.get("prior_pass_a_raw_result_exposed") is not False:
                reasons.append("PASS_B_RAW_PASS_A_EXPOSED_OR_UNKNOWN")
            if pass_b_context.get("candidate_origins_exposed") is not False:
                reasons.append("PASS_B_CANDIDATE_ORIGINS_EXPOSED_OR_UNKNOWN")
    return {
        "schema": "bemarkdown-vision-quality-gate-context-decision-v0",
        "quality_gate_eligible": not reasons,
        "status": (
            "QUALITY_GATE_CONTEXT_ELIGIBLE"
            if not reasons
            else "REFERENCE_GATE_INELIGIBLE_CONTEXT_CONTAMINATION"
        ),
        "reasons": sorted(set(reasons)),
        "pass_b_required": bool(pass_b_required),
        "truth_loaded_before_final_resolution": bool(
            truth_loaded_before_final_resolution
        ),
    }


def assert_quality_gate_eligible(
    *,
    evaluation_role: str,
    pass_a_context: dict[str, Any] | None,
    pass_b_context: dict[str, Any] | None = None,
    pass_b_required: bool = False,
    truth_loaded_before_final_resolution: bool = False,
) -> dict[str, Any]:
    if str(evaluation_role).upper() == EVALUATION_ROLE_DIAGNOSTIC:
        raise DiagnosticEvidenceNotEligibleForQualityGate(
            "Development diagnostic evidence cannot enter a quality evaluator"
        )
    decision = quality_gate_context_decision(
        evaluation_role=evaluation_role,
        pass_a_context=pass_a_context,
        pass_b_context=pass_b_context,
        pass_b_required=pass_b_required,
        truth_loaded_before_final_resolution=truth_loaded_before_final_resolution,
    )
    if not decision["quality_gate_eligible"]:
        raise EvaluationContextNotEligibleForQualityGate(
            ",".join(decision["reasons"])
        )
    return decision


def scan_adjudication_origin_leakage(bundle_root: str | Path) -> dict[str, Any]:
    root = Path(bundle_root).resolve()
    findings: list[dict[str, str]] = []
    scanned_text_files = 0
    scanned_image_metadata = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        lowered_name = relative.lower()
        for fragment in ADJUDICATION_EXTERNAL_FORBIDDEN_FRAGMENTS:
            if fragment in lowered_name:
                findings.append(
                    {"path": relative, "surface": "FILENAME", "fragment": fragment}
                )
        if path.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".html", ".svg"}:
            scanned_text_files += 1
            lowered = path.read_text(encoding="utf-8").lower()
            for fragment in ADJUDICATION_EXTERNAL_FORBIDDEN_FRAGMENTS:
                if fragment in lowered:
                    findings.append(
                        {"path": relative, "surface": "TEXT", "fragment": fragment}
                    )
        elif path.suffix.lower() == ".png":
            scanned_image_metadata += 1
            with Image.open(path) as image:
                metadata = json.dumps(image.info, sort_keys=True).lower()
            for fragment in ADJUDICATION_EXTERNAL_FORBIDDEN_FRAGMENTS:
                if fragment in metadata:
                    findings.append(
                        {
                            "path": relative,
                            "surface": "PNG_METADATA",
                            "fragment": fragment,
                        }
                    )
    return {
        "schema": "bemarkdown-adjudication-origin-leakage-scan-v0",
        "status": (
            "ADJUDICATION_ORIGIN_LEAKAGE_ZERO"
            if not findings
            else "ADJUDICATION_ORIGIN_LEAKAGE_FOUND"
        ),
        "leakage_count": len(findings),
        "findings": findings,
        "scanned_text_files": scanned_text_files,
        "scanned_image_metadata": scanned_image_metadata,
    }


def _context_reasons(
    context: dict[str, Any] | None, *, stage: str
) -> list[str]:
    if not isinstance(context, dict):
        return [f"{stage}_CONTEXT_MISSING"]
    reasons = []
    if context.get("schema") != EVALUATION_CONTEXT_SCHEMA:
        reasons.append(f"{stage}_CONTEXT_SCHEMA_INVALID")
    if context.get("stage") != stage:
        reasons.append(f"{stage}_CONTEXT_STAGE_INVALID")
    if context.get("context_role") != FRESH_CONTEXT_ROLE:
        reasons.append(f"{stage}_CONTEXT_NOT_FRESH")
    if context.get("operator_attestation") is not True:
        reasons.append(f"{stage}_OPERATOR_ATTESTATION_MISSING")
    if context.get("prior_project_history_exposed") is not False:
        reasons.append(f"{stage}_PROJECT_HISTORY_EXPOSED_OR_UNKNOWN")
    if context.get("prior_candidate_history_exposed") is not False:
        reasons.append(f"{stage}_CANDIDATE_HISTORY_EXPOSED_OR_UNKNOWN")
    if not isinstance(context.get("context_id"), str) or not context["context_id"].strip():
        reasons.append(f"{stage}_CONTEXT_ID_MISSING")
    return reasons
