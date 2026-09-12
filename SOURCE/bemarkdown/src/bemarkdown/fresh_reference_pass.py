from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .blind_vision_formula import ADJUDICATION_CONTRACT
from .vision_evaluation_context import quality_gate_context_decision

FRESH_PASS_A_ROLE = "FRESH_REFERENCE_PASS_A"
FRESH_PASS_B_ROLE = "FRESH_REFERENCE_PASS_B"
FRESH_PASS_B_PENDING_ROLE = "FRESH_REFERENCE_PASS_B_PENDING"

_ORIGIN_FORBIDDEN_FRAGMENTS = (
    "formulanet",
    "blind",
    "current_latex",
    "blind_latex",
    "current candidate",
    "vision candidate",
    "candidate_origin",
    "risk priority",
    "risk_priority",
    "pass a decision",
    "pass_a_decision",
    "primary",
    "secondary",
)
_TRUTH_FORBIDDEN_FRAGMENTS = (
    "reference truth",
    "reference_truth",
    "truth label",
    "truth_label",
    "material",
    "acceptable",
    "expected correction",
    "expected_correction",
)
_TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".html", ".svg"}
_ROOT_FILES = {
    "README.md",
    "VISION_ADJUDICATION_REFERENCE_HANDOFF.md",
    "result_schema.json",
    "bundle_manifest.json",
}
_ROOT_DIRECTORIES = {"requests", "images"}


class ReferenceTruthAccessForbidden(RuntimeError):
    """Raised before any Reference truth content may be supplied to an evaluator."""


def normalize_fresh_pass_a_payloads(
    payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt the public handoff names to the frozen internal IR without changing values."""

    normalized = copy.deepcopy(payloads)
    observation_alias_rows = 0
    bbox_alias_rows = 0
    for payload in normalized:
        for observation in payload.get("observations", []):
            if "subregions" in observation:
                if "formula_subregions" in observation:
                    raise ValueError("Fresh PASS A contains both subregion field names")
                observation["formula_subregions"] = observation.pop("subregions")
                observation_alias_rows += 1
            for subregion in observation.get("formula_subregions", []):
                if "bbox_normalized" in subregion:
                    if "bbox_crop_1000" in subregion:
                        raise ValueError("Fresh PASS A contains both bbox field names")
                    subregion["bbox_crop_1000"] = subregion.pop("bbox_normalized")
                    bbox_alias_rows += 1
    return normalized, {
        "schema": "bemarkdown-fresh-pass-a-structural-adapter-metrics-v0",
        "adapter": "PUBLIC_HANDOFF_SUBREGION_FIELD_ALIAS_ONLY",
        "observation_alias_rows": observation_alias_rows,
        "bbox_alias_rows": bbox_alias_rows,
        "latex_values_rewritten": 0,
        "semantic_rewrites": 0,
    }


def assert_reference_truth_access_allowed(
    *,
    pass_a_context: dict[str, Any] | None,
    pass_b_context: dict[str, Any] | None,
    pass_b_required: bool,
    region_handling_complete: bool,
    pending_disagreement_count: int,
) -> dict[str, Any]:
    context = quality_gate_context_decision(
        evaluation_role="REFERENCE_QUALITY_EVALUATION",
        pass_a_context=pass_a_context,
        pass_b_context=pass_b_context,
        pass_b_required=pass_b_required,
        truth_loaded_before_final_resolution=False,
    )
    reasons = list(context["reasons"])
    if not region_handling_complete:
        reasons.append("REGION_HANDLING_INCOMPLETE")
    if pending_disagreement_count:
        reasons.append("PENDING_CANDIDATE_DISAGREEMENT")
    decision = {
        "schema": "bemarkdown-reference-truth-access-decision-v0",
        "reference_truth_access_allowed": not reasons,
        "status": (
            "REFERENCE_TRUTH_ACCESS_ALLOWED_AFTER_FINAL_RESOLUTION"
            if not reasons
            else "REFERENCE_TRUTH_ACCESS_FORBIDDEN_BEFORE_FINAL_RESOLUTION"
        ),
        "reasons": sorted(set(reasons)),
        "pass_b_required": bool(pass_b_required),
        "region_handling_complete": bool(region_handling_complete),
        "pending_disagreement_count": int(pending_disagreement_count),
    }
    if reasons:
        raise ReferenceTruthAccessForbidden(
            "REFERENCE_TRUTH_ACCESS_FORBIDDEN_BEFORE_FINAL_RESOLUTION: "
            + ",".join(decision["reasons"])
        )
    return decision


def validate_fresh_pass_b_root(bundle_root: str | Path) -> dict[str, Any]:
    root = Path(bundle_root).resolve()
    present_files = {path.name for path in root.iterdir() if path.is_file()}
    present_directories = {path.name for path in root.iterdir() if path.is_dir()}
    missing_files = sorted(_ROOT_FILES - present_files)
    missing_directories = sorted(_ROOT_DIRECTORIES - present_directories)
    return {
        "schema": "bemarkdown-fresh-pass-b-root-contract-v0",
        "required_root_files": sorted(_ROOT_FILES),
        "required_root_directories": sorted(_ROOT_DIRECTORIES),
        "present_root_files": sorted(present_files),
        "present_root_directories": sorted(present_directories),
        "missing_root_files": missing_files,
        "missing_root_directories": missing_directories,
        "required_root_files_exact_complete": not missing_files,
        "required_root_directories_exact_complete": not missing_directories,
        "passed": not missing_files and not missing_directories,
    }


def scan_external_bundle_leakage(
    bundle_root: str | Path, *, category: str
) -> dict[str, Any]:
    root = Path(bundle_root).resolve()
    normalized_category = category.lower()
    if normalized_category == "origin":
        fragments = _ORIGIN_FORBIDDEN_FRAGMENTS
        zero_status = "ADJUDICATION_ORIGIN_LEAKAGE_ZERO"
        found_status = "ADJUDICATION_ORIGIN_LEAKAGE_FOUND"
    elif normalized_category == "truth":
        fragments = _TRUTH_FORBIDDEN_FRAGMENTS
        zero_status = "REFERENCE_TRUTH_LEAKAGE_ZERO"
        found_status = "REFERENCE_TRUTH_LEAKAGE_FOUND"
    else:
        raise ValueError("Leakage category must be origin or truth")

    findings: list[dict[str, str]] = []
    scanned_text_files = 0
    scanned_image_metadata = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        lowered_name = relative.lower()
        for fragment in fragments:
            if fragment in lowered_name:
                findings.append(
                    {"path": relative, "surface": "FILENAME", "fragment": fragment}
                )
        if path.suffix.lower() in _TEXT_SUFFIXES:
            scanned_text_files += 1
            lowered = path.read_text(encoding="utf-8").lower()
            for fragment in fragments:
                if fragment in lowered:
                    findings.append(
                        {"path": relative, "surface": "TEXT", "fragment": fragment}
                    )
        elif path.suffix.lower() == ".png":
            scanned_image_metadata += 1
            with Image.open(path) as image:
                metadata = json.dumps(image.info, sort_keys=True).lower()
            for fragment in fragments:
                if fragment in metadata:
                    findings.append(
                        {
                            "path": relative,
                            "surface": "PNG_METADATA",
                            "fragment": fragment,
                        }
                    )
    return {
        "schema": f"bemarkdown-fresh-pass-b-{normalized_category}-leakage-scan-v0",
        "status": zero_status if not findings else found_status,
        "leakage_count": len(findings),
        "findings": findings,
        "scanned_text_files": scanned_text_files,
        "scanned_image_metadata": scanned_image_metadata,
    }


class FreshPassBResultImporter:
    """Import full-row PASS B results only from an independent fresh context."""

    def __init__(
        self,
        expected_requests: list[dict[str, Any]],
        *,
        source_pass_a_context_id: str,
        canonical_bundle_sha256: str,
    ) -> None:
        self._requests = {
            str(row["case_id"]): str(row["formula_id"]) for row in expected_requests
        }
        if len(self._requests) != len(expected_requests):
            raise ValueError("PASS B case IDs must be unique")
        self.source_pass_a_context_id = source_pass_a_context_id
        self.canonical_bundle_sha256 = canonical_bundle_sha256

    def import_payloads(
        self, payloads: list[dict[str, Any]], *, provenance: dict[str, Any]
    ) -> dict[str, Any]:
        provenance_error = self._validate_provenance(provenance)
        if provenance_error:
            return self._rejected(provenance_error)
        if not isinstance(payloads, list):
            return self._rejected("PASS_B_RESULT_NOT_ROWS")
        case_ids = [str(row.get("case_id", "")) for row in payloads]
        if len(case_ids) != len(set(case_ids)):
            return self._rejected("DUPLICATE_PASS_B_RESULT_ROW")
        if set(case_ids) != set(self._requests):
            return self._rejected("PASS_B_FULL_ROW_COVERAGE_REQUIRED")
        allowed = set(ADJUDICATION_CONTRACT["decisions"])
        for row in payloads:
            case_id = str(row.get("case_id", ""))
            if row.get("schema") != "bemarkdown-vision-formula-adjudication-result-v0":
                return self._rejected("PASS_B_RESULT_SCHEMA_MISMATCH")
            if str(row.get("formula_id", "")) != self._requests[case_id]:
                return self._rejected("PASS_B_FORMULA_ID_MISMATCH")
            decision = row.get("decision")
            if decision not in allowed:
                return self._rejected("PASS_B_DECISION_INVALID")
            latex = row.get("latex")
            subregions = row.get("formula_subregions")
            if not isinstance(subregions, list):
                return self._rejected("PASS_B_SUBREGIONS_INVALID")
            if decision == "NEITHER_SOURCE_LATEX" and not (
                isinstance(latex, str) and latex.strip()
            ):
                return self._rejected("PASS_B_LATEX_REQUIRED")
            if decision != "NEITHER_SOURCE_LATEX" and latex is not None:
                return self._rejected("PASS_B_LATEX_FORBIDDEN")
            if decision != "REGION_NOT_SINGLE_FORMULA" and subregions:
                return self._rejected("PASS_B_SUBREGIONS_FORBIDDEN")
            if decision == "REGION_NOT_SINGLE_FORMULA" and any(
                not self._valid_subregion(item) for item in subregions
            ):
                return self._rejected("PASS_B_SUBREGIONS_INVALID")
        return {
            "schema": "bemarkdown-fresh-pass-b-result-import-v0",
            "status": "FRESH_PASS_B_RESULTS_IMPORTED",
            "response_count": len(payloads),
            "full_row_coverage": True,
            "responses": payloads,
            "context_id": provenance["context_id"],
            "source_pass_a_context_id": self.source_pass_a_context_id,
            "error_codes": [],
        }

    def _validate_provenance(self, provenance: dict[str, Any]) -> str | None:
        if not isinstance(provenance, dict):
            return "PASS_B_ATTESTATION_INCOMPLETE"
        if provenance.get("context_id") == self.source_pass_a_context_id:
            return "PASS_B_CONTEXT_NOT_INDEPENDENT"
        required_false = (
            "prior_project_history_exposed",
            "prior_candidate_history_exposed",
            "prior_pass_a_raw_result_exposed",
            "candidate_origins_exposed",
            "reference_truth_exposed",
        )
        if (
            provenance.get("evaluation_role") != FRESH_PASS_B_ROLE
            or provenance.get("context_isolation") != "FRESH_ISOLATED"
            or not isinstance(provenance.get("context_id"), str)
            or not provenance["context_id"].strip()
            or provenance.get("source_pass_a_context_id")
            != self.source_pass_a_context_id
            or provenance.get("operator_attestation") is not True
            or any(provenance.get(field) is not False for field in required_false)
        ):
            return "PASS_B_ATTESTATION_INCOMPLETE"
        if provenance.get("source_bundle_sha256") != self.canonical_bundle_sha256:
            return "PASS_B_BUNDLE_SHA_MISMATCH"
        return None

    @staticmethod
    def _valid_subregion(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        bbox = value.get("bbox_crop_1000")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(item, int | float) for item in bbox)
            or not (0 <= bbox[0] < bbox[2] <= 1000)
            or not (0 <= bbox[1] < bbox[3] <= 1000)
        ):
            return False
        visibility = value.get("visibility")
        latex = value.get("latex")
        if visibility == "CLEAR":
            return isinstance(latex, str) and bool(latex.strip())
        if visibility == "AMBIGUOUS":
            return latex is None or (isinstance(latex, str) and bool(latex.strip()))
        return False

    @staticmethod
    def _rejected(code: str) -> dict[str, Any]:
        return {
            "schema": "bemarkdown-fresh-pass-b-result-import-v0",
            "status": "FRESH_PASS_B_RESULT_IMPORT_REJECTED",
            "response_count": 0,
            "full_row_coverage": False,
            "responses": [],
            "error_codes": [code],
        }
