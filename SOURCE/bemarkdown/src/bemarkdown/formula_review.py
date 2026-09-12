"""Formula risk escalation and vision-first review contracts.

This module is post-route and model-agnostic.  It never repairs FormulaNet output
or guesses whether the caller has vision capability.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from .formulanet_runtime import FormulaOcrOutputValidator, OcrVerdict

FORMULA_RISK_SCHEMA = "bemarkdown-formula-risk-ir-v0"
FORMULA_RISK_FEATURE_VERSION = "formula-risk-features-v0"
FORMULA_RISK_POLICY_VERSION = "formula-risk-policy-v0"
FORMULA_REVIEW_TASK_SCHEMA = "bemarkdown-formula-review-task-v0"
VISION_REQUEST_SCHEMA = "bemarkdown-vision-formula-resolution-request-v0"
VISION_RESULT_SCHEMA = "bemarkdown-vision-formula-resolution-result-v0"
FORMULA_RESOLUTION_SCHEMA = "bemarkdown-formula-resolution-v0"

ACCEPTABLE_LABELS = {"CORRECT", "MINOR_ERROR"}
MATERIAL_LABELS = {"MAJOR_ERROR", "UNUSABLE"}
TERMINAL_STATES = {
    "AUTO_ACCEPTED",
    "AGENT_VISION_RESOLVED",
    "HUMAN_RESOLVED",
    "UNRESOLVED_PRESERVE_INPUT",
    "FAILED_PRESERVE_INPUT",
}

RISK_POLICY_CONTRACT = {
    "schema": "bemarkdown-formula-risk-policy-contract-v0",
    "policy_version": FORMULA_RISK_POLICY_VERSION,
    "policy_family": "R2_SHALLOW_INTERPRETABLE_RULE_LIST",
    "calibration_source": "FORMULA_RISK_REFERENCE_SET_ONLY",
    "native_model_confidence_used": False,
    "rules": [
        {
            "id": "R_STRONG_INDEPENDENT_CONTRADICTION",
            "conditions": {
                "multiview_status": "MULTIVIEW_STRONG_DISAGREEMENT",
                "ocr_status": "STRONG_CONTRADICTION",
            },
            "risk_score": 1.0,
        },
        {
            "id": "R_COMPLEX_CONTAMINATION_COMPOUND",
            "conditions": {
                "visual_status": "VISUAL_REVIEW",
                "multiview_status": "MULTIVIEW_WEAK_DISAGREEMENT",
                "output_length_min": 40,
            },
            "risk_score": 0.85,
        },
        {
            "id": "R_TINY_WEAK_COMPOUND",
            "conditions": {
                "crop_width_max": 40,
                "output_length_max": 8,
                "multiview_status": "MULTIVIEW_WEAK_DISAGREEMENT",
                "consensus_risk_min": 0.30,
            },
            "risk_score": 0.80,
        },
        {
            "id": "R_TINY_STABLE_VISUAL_WARNING",
            "conditions": {
                "crop_width_max": 32,
                "output_length_max": 3,
                "multiview_status": "MULTIVIEW_STABLE",
                "visual_status": "VISUAL_WARNING",
            },
            "risk_score": 0.70,
        },
    ],
    "forbidden_features": [
        "formula_id",
        "reference_id",
        "document_id",
        "page_index",
        "raw_latex_value",
    ],
}


class FormulaRiskMarker:
    """Rank already-recognized Formula routes without claiming probability."""

    def mark(self, formula_id: str, row: dict[str, Any]) -> dict[str, Any]:
        features = extract_formula_risk_features(row)
        reasons: list[str] = []
        matched_scores = []
        hard_failure = features["existing_gate"] not in {"ACCEPT", "ACCEPT_WITH_WARNING"}
        hard_failure |= features["evidence_unavailable"]
        if hard_failure:
            reasons.append("FORMULA_RISK_HARD_FAILURE")
            decision = "HARD_FAILURE_REVIEW"
            risk_score = 1.0
        else:
            for rule in RISK_POLICY_CONTRACT["rules"]:
                if _matches_risk_rule(features, rule["conditions"]):
                    reasons.append(rule["id"])
                    matched_scores.append(rule["risk_score"])
            decision = "REVIEW_REQUIRED" if matched_scores else "AUTO_ACCEPT"
            risk_score = max(matched_scores, default=0.0)
        return {
            "schema": FORMULA_RISK_SCHEMA,
            "formula_id": formula_id,
            "decision": decision,
            "risk_score": risk_score,
            "score_semantics": "CALIBRATED_REVIEW_RANKING_NOT_PROBABILITY",
            "risk_level": "LOW_TRUST" if decision != "AUTO_ACCEPT" else "ACCEPTABLE_TRUST",
            "reason_codes": sorted(reasons),
            "feature_version": FORMULA_RISK_FEATURE_VERSION,
            "policy_version": FORMULA_RISK_POLICY_VERSION,
            "native_model_confidence": None,
            "features": features,
            "provenance": {
                "source_route_id": row.get("route_id"),
                "semantic_repair_used": False,
                "identity_features_used": False,
            },
        }


class OneShotValidationLedger:
    """In-memory guard used by the artifact runner before persisting validation."""

    def __init__(self, policy_sha256: str, split_sha256: str):
        self.policy_sha256 = policy_sha256
        self.split_sha256 = split_sha256
        self._evaluated = False

    def evaluate_once(self, decisions: Iterable[dict[str, Any]]) -> dict[str, Any]:
        if self._evaluated:
            raise RuntimeError("Validation was already evaluated")
        self._evaluated = True
        decisions = list(decisions)
        return {
            "policy_sha256": self.policy_sha256,
            "split_sha256": self.split_sha256,
            "evaluated_once": True,
            "decision_counts": dict(sorted(Counter(row["decision"] for row in decisions).items())),
        }


def extract_formula_risk_features(row: dict[str, Any]) -> dict[str, Any]:
    multiview = row.get("multiview") or {}
    ocr = row.get("ocr_crosscheck") or {}
    visual = row.get("visual_evidence") or {}
    crop = row.get("crop") or {}
    c3 = (row.get("strategies") or {}).get("C3_E0_E1_E2_E3") or {}
    raw_latex = str(row.get("raw_latex") or row.get("formulanet_raw_latex") or "")
    return {
        "existing_gate": (row.get("existing_gate") or {}).get("verdict"),
        "evidence_unavailable": bool(multiview.get("evidence_unavailable"))
        or bool(ocr.get("pipeline_failure"))
        or visual.get("status") == "VERIFICATION_UNAVAILABLE",
        "multiview_status": multiview.get("stability_status"),
        "max_normalized_edit_distance": multiview.get("max_normalized_edit_distance"),
        "ocr_status": ocr.get("evidence_status"),
        "ocr_normalized_edit_distance": ocr.get("normalized_edit_distance"),
        "visual_status": visual.get("status"),
        "visual_score": visual.get("score"),
        "consensus_status": c3.get("consensus_status"),
        "consensus_risk": float(c3.get("risk_score") or 0.0),
        "crop_width": int(crop.get("width") or row.get("crop_width") or 0),
        "crop_height": int(crop.get("height") or row.get("crop_height") or 0),
        "output_length": len(raw_latex),
        "formula_type_proxy": row.get("formula_type_proxy"),
    }


def _matches_risk_rule(features: dict[str, Any], conditions: dict[str, Any]) -> bool:
    for name, expected in conditions.items():
        if name.endswith("_min"):
            actual = features[name.removesuffix("_min")]
            if actual is None or actual < expected:
                return False
        elif name.endswith("_max"):
            actual = features[name.removesuffix("_max")]
            if actual is None or actual > expected:
                return False
        elif features[name] != expected:
            return False
    return True


def freeze_formula_risk_split(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Freeze a deterministic 70/45/5 split using labels and document IDs only."""

    rows = [copy.deepcopy(row) for row in rows]
    if len(rows) != 120:
        raise ValueError(f"Formula risk split requires 120 rows, got {len(rows)}")
    ids = [row["reference_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Formula risk split requires unique reference IDs")

    acceptable = [row for row in rows if row["formulanet_label"] in ACCEPTABLE_LABELS]
    material = [row for row in rows if row["formulanet_label"] in MATERIAL_LABELS]
    uncertain = [row for row in rows if row["formulanet_label"] == "REFERENCE_UNCERTAIN"]
    if (len(acceptable), len(material), len(uncertain)) != (109, 6, 5):
        raise ValueError("Frozen Formula truth must be 109 acceptable, 6 material, 5 uncertain")

    anchor = next(
        (row for row in material if row["reference_id"] == "formula-reference-0015"),
        None,
    )
    if anchor is None:
        raise ValueError("Validation anchor formula-reference-0015 is missing")
    alternatives = [row for row in material if row["document_id"] != anchor["document_id"]]
    if not alternatives:
        raise ValueError("A document-separated second material row is required")
    # Pre-recorded rule: prefer a different failure class (UNUSABLE), then stable ID.
    second = min(
        alternatives,
        key=lambda row: (
            0 if row["formulanet_label"] == "UNUSABLE" else 1,
            row["reference_id"],
        ),
    )
    validation_material = [anchor, second]
    validation_documents = {row["document_id"] for row in validation_material}

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in acceptable:
        by_document[row["document_id"]].append(row)
    validation_acceptable = [
        row for row in acceptable if row["document_id"] in validation_documents
    ]
    remaining = 43 - len(validation_acceptable)
    if remaining < 0:
        raise ValueError("Material validation documents exceed acceptable holdout target")
    available_groups = {
        document: group
        for document, group in by_document.items()
        if document not in validation_documents
    }
    chosen = _choose_whole_document_groups(available_groups, remaining)
    validation_documents.update(chosen)
    validation_acceptable = [
        row for row in acceptable if row["document_id"] in validation_documents
    ]

    validation_ids = {
        row["reference_id"] for row in validation_acceptable + validation_material
    }
    validation = [row for row in rows if row["reference_id"] in validation_ids]
    reference = [
        row
        for row in rows
        if row["formulanet_label"] in ACCEPTABLE_LABELS | MATERIAL_LABELS
        and row["reference_id"] not in validation_ids
    ]
    if (len(reference), len(validation), len(uncertain)) != (70, 45, 5):
        raise ValueError("Unable to produce the required 70/45/5 split")

    payload = {
        "schema": "bemarkdown-formula-risk-split-v0",
        "selection_inputs": ["formulanet_label", "document_id", "reference_id", "crop_sha256"],
        "validation_material_rule": (
            "FIX_0015_THEN_DIFFERENT_DOCUMENT_PREFER_UNUSABLE_THEN_REFERENCE_ID"
        ),
        "acceptable_rule": "WHOLE_DOCUMENT_SUBSET_SUM_MIN_DOCUMENTS_THEN_SHA256",
        "truth_features_consumed_before_freeze": False,
        "counts": {"reference": 70, "validation": 45, "uncertain": 5},
        "label_counts": {
            "reference": _label_group_counts(reference),
            "validation": _label_group_counts(validation),
            "uncertain": {"reference_uncertain": len(uncertain)},
        },
        "reference_document_ids": sorted({row["document_id"] for row in reference}),
        "validation_document_ids": sorted(validation_documents),
        "acceptable_reference_document_ids": sorted(
            {
                row["document_id"]
                for row in reference
                if row["formulanet_label"] in ACCEPTABLE_LABELS
            }
        ),
        "acceptable_document_overlap": sorted(
            {
                row["document_id"]
                for row in reference
                if row["formulanet_label"] in ACCEPTABLE_LABELS
            }
            & validation_documents
        ),
        "forced_material_document_overlap": sorted(
            {row["document_id"] for row in reference if row["formulanet_label"] in MATERIAL_LABELS}
            & validation_documents
        ),
        "reference_reference_ids": sorted(row["reference_id"] for row in reference),
        "validation_reference_ids": sorted(row["reference_id"] for row in validation),
        "uncertain_reference_ids": sorted(row["reference_id"] for row in uncertain),
        "rows": [
            {
                "reference_id": row["reference_id"],
                "document_id": row["document_id"],
                "crop_sha256": row["crop_sha256"],
                "label_group": _label_group(row["formulanet_label"]),
                "split": (
                    "FORMULA_RISK_VALIDATION_SET"
                    if row["reference_id"] in validation_ids
                    else "FORMULA_RISK_UNCERTAIN_DIAGNOSTIC_SET"
                    if row["formulanet_label"] == "REFERENCE_UNCERTAIN"
                    else "FORMULA_RISK_REFERENCE_SET"
                ),
            }
            for row in sorted(rows, key=lambda item: item["reference_id"])
        ],
    }
    payload["split_sha256"] = _sha256_json(payload)
    return payload


def _choose_whole_document_groups(
    groups: dict[str, list[dict[str, Any]]], target: int
) -> tuple[str, ...]:
    if target == 0:
        return ()
    ordered = sorted(groups, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for document in ordered:
        count = len(groups[document])
        for total, selected in sorted(choices.items(), reverse=True):
            candidate_total = total + count
            if candidate_total > target:
                continue
            candidate = selected + (document,)
            current = choices.get(candidate_total)
            if current is None or (len(candidate), candidate) < (len(current), current):
                choices[candidate_total] = candidate
    if target not in choices:
        raise ValueError(f"No whole-document acceptable subset totals {target}")
    return choices[target]


def build_formula_review_task(
    row: dict[str, Any], *, decision: str, risk: dict[str, Any] | None = None
) -> dict[str, Any]:
    if decision not in {"REVIEW_REQUIRED", "HARD_FAILURE_REVIEW"}:
        raise ValueError("Only review decisions create FormulaReviewTask")
    formula_id = str(row.get("content_id") or row.get("reference_id"))
    return {
        "schema": FORMULA_REVIEW_TASK_SCHEMA,
        "formula_id": formula_id,
        "document_id": row.get("document_id"),
        "page_index": row.get("page_index"),
        "bbox_pdf_pt": row.get("bbox_pdf_pt") or [],
        "source_crop_ref": row.get("crop_path"),
        "page_context_crop_ref": row.get("page_context_crop_ref"),
        "current_latex": row.get("formulanet_raw_latex") or row.get("latex") or "",
        "current_render_ref": row.get("current_render_ref"),
        "risk": risk
        or {
            "decision": decision,
            "risk_score": 1.0,
            "reason_codes": ["FORMULA_REVIEW_REQUIRED"],
        },
        "context": {
            "before_text": row.get("before_text"),
            "after_text": row.get("after_text"),
        },
        "resolution": {"status": "PENDING", "resolver": None},
        "provenance": {
            "original_crop_sha256": row.get("crop_sha256"),
            "original_latex_sha256": hashlib.sha256(
                (row.get("formulanet_raw_latex") or row.get("latex") or "").encode()
            ).hexdigest(),
            "route_id": row.get("route_id"),
        },
    }


def build_vision_resolution_request(task: dict[str, Any]) -> dict[str, Any]:
    if task.get("schema") != FORMULA_REVIEW_TASK_SCHEMA:
        raise ValueError("Vision request requires FormulaReviewTask v0")
    return {
        "schema": VISION_REQUEST_SCHEMA,
        "formula_id": task["formula_id"],
        "source_formula_crop": task["source_crop_ref"],
        "page_context_crop": task.get("page_context_crop_ref"),
        "before_text": task["context"].get("before_text"),
        "after_text": task["context"].get("after_text"),
        "current_latex": task["current_latex"],
        "current_render_ref": task.get("current_render_ref"),
        "risk_reasons": task["risk"].get("reason_codes") or [],
        "instruction": [
            "先独立观察原公式图像。",
            "忠实转写，不根据物理常识改写。",
            "再与 current_latex 对比。",
            "若 current_latex 已正确，原样返回 CONFIRMED_CURRENT。",
            "若有差异，返回原图公式的 LaTeX 和 REPLACED。",
            "无法辨认时返回 UNRESOLVED。",
        ],
        "provider_binding": None,
    }


def parse_vision_resolution_result(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if status not in {"CONFIRMED_CURRENT", "REPLACED", "UNRESOLVED", "FAILED"}:
        raise ValueError("Unknown vision resolution status")
    if status == "REPLACED" and not str(payload.get("latex") or "").strip():
        raise ValueError("REPLACED vision result requires LaTeX")
    if status in {"UNRESOLVED", "FAILED"} and payload.get("latex") is not None:
        raise ValueError("Unresolved vision result cannot provide replacement LaTeX")
    return {
        "schema": VISION_RESULT_SCHEMA,
        "status": status,
        "latex": payload.get("latex"),
        "resolver_id": payload.get("resolver_id"),
        "technical_error": payload.get("technical_error") if status == "FAILED" else None,
    }


class FormulaReviewWorkflow:
    """Drive model-agnostic vision-first review through one state interface."""

    def __init__(self, *, render_validator: Callable[[str], bool]):
        self._render_validator = render_validator
        self._syntax_validator = FormulaOcrOutputValidator()

    def dispatch(self, task: dict[str, Any], capability: dict[str, Any]) -> dict[str, Any]:
        state = copy.deepcopy(task)
        state["state"] = (
            "VISION_REVIEW_PENDING"
            if capability.get("vision_input_available") is True
            else "HUMAN_REVIEW_PENDING"
        )
        state["capability"] = {
            "vision_input_available": capability.get("vision_input_available") is True
        }
        return state

    def apply_vision_result(
        self, state: dict[str, Any], result: dict[str, Any], *, attempt: int = 1
    ) -> dict[str, Any]:
        if state.get("state") != "VISION_REVIEW_PENDING":
            raise ValueError("Vision result requires VISION_REVIEW_PENDING")
        status = result.get("status")
        if status not in {
            "CONFIRMED_CURRENT",
            "REPLACED",
            "RECOVERED_MISSING",
            "UNRESOLVED",
            "FAILED",
        }:
            raise ValueError("Unknown vision resolution status")
        current_state = state.get("current_latex_state") or (
            "PRESENT" if state.get("current_latex") else "MISSING"
        )
        if current_state == "MISSING" and status in {
            "CONFIRMED_CURRENT",
            "REPLACED",
        }:
            raise ValueError("Missing current LaTeX requires recovery or unresolved")
        if current_state == "PRESENT" and status == "RECOVERED_MISSING":
            raise ValueError("RECOVERED_MISSING requires missing current LaTeX")
        latex = state["current_latex"] if status == "CONFIRMED_CURRENT" else result.get("latex")
        validation = self._validate(latex)
        if status in {
            "CONFIRMED_CURRENT",
            "REPLACED",
            "RECOVERED_MISSING",
        } and validation["all_passed"]:
            return self._resolved(
                state,
                latex,
                resolver_kind="VISION_AGENT",
                resolver_id=result.get("resolver_id"),
                action=status,
                terminal_state="AGENT_VISION_RESOLVED",
                validation=validation,
            )
        output = copy.deepcopy(state)
        output["vision_attempt"] = attempt
        output["vision_validation"] = validation
        output["vision_result_status"] = status
        output["state"] = "VISION_REVIEW_PENDING" if attempt < 2 else "HUMAN_REVIEW_PENDING"
        return output

    def send_to_human(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("state") in TERMINAL_STATES:
            raise ValueError("A terminal review state cannot enter human review")
        output = copy.deepcopy(state)
        output["state"] = "HUMAN_REVIEW_PENDING"
        return output

    def apply_human_result(
        self, state: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        if state.get("state") != "HUMAN_REVIEW_PENDING":
            raise ValueError("Human result requires HUMAN_REVIEW_PENDING")
        if result.get("action") == "UNRESOLVED":
            output = copy.deepcopy(state)
            output["state"] = "UNRESOLVED_PRESERVE_INPUT"
            return output
        latex = result.get("latex")
        validation = self._validate(latex)
        if not validation["all_passed"]:
            output = copy.deepcopy(state)
            output["human_validation"] = validation
            return output
        return self._resolved(
            state,
            latex,
            resolver_kind="HUMAN",
            resolver_id=result.get("resolver_id"),
            action=result.get("action") or "EDITED",
            terminal_state="HUMAN_RESOLVED",
            validation=validation,
            math_json=result.get("math_json"),
        )

    def _validate(self, latex: str | None) -> dict[str, Any]:
        syntax = self._syntax_validator.validate(latex)
        project_format = bool(latex and "```" not in latex and "formula-review" not in latex)
        renderable = bool(latex and self._render_validator(latex))
        syntax_ok = syntax.verdict in {OcrVerdict.VALID, OcrVerdict.VALID_WITH_WARNING}
        return {
            "syntax": syntax.to_dict(),
            "syntax_passed": syntax_ok,
            "renderability_passed": renderable,
            "project_format_passed": project_format,
            "all_passed": syntax_ok and renderable and project_format,
        }

    @staticmethod
    def _resolved(
        state: dict[str, Any],
        latex: str,
        *,
        resolver_kind: str,
        resolver_id: str | None,
        action: str,
        terminal_state: str,
        validation: dict[str, Any],
        math_json: Any = None,
    ) -> dict[str, Any]:
        output = copy.deepcopy(state)
        output["state"] = terminal_state
        output["resolution"] = {
            "schema": FORMULA_RESOLUTION_SCHEMA,
            "formula_id": state["formula_id"],
            "status": "RESOLVED",
            "resolver_kind": resolver_kind,
            "resolver_id": resolver_id,
            "latex": latex,
            "math_json": math_json,
            "action": action,
            "validator": validation,
        }
        output["audit"] = {
            "original_latex": state["current_latex"],
            "original_crop_sha256": state.get("provenance", {}).get(
                "original_crop_sha256"
            ),
            "risk_decision": state["risk"]["decision"],
            "resolved_latex": latex,
            "resolver_kind": resolver_kind,
            "resolver_id": resolver_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "validator": validation,
        }
        if action == "RECOVERED_MISSING":
            output["audit"].update(
                {
                    "original_current_latex": None,
                    "current_latex_state": "MISSING",
                    "recovered_latex": latex,
                    "resolver": "VISION_AGENT",
                }
            )
        return output


def render_review_markdown(
    formula_nodes: Iterable[dict[str, Any]], pending_formula_ids: set[str]
) -> str:
    lines = []
    for node in formula_nodes:
        formula_id = node["formula_id"]
        line = f"${node['latex']}$"
        if formula_id in pending_formula_ids:
            line += f'<!-- bemarkdown:formula-review id="{formula_id}" -->'
        lines.append(line)
    return "\n\n".join(lines) + "\n"


def apply_resolution_by_id(
    formula_nodes: Iterable[dict[str, Any]], formula_id: str, resolved: dict[str, Any]
) -> list[dict[str, Any]]:
    output = copy.deepcopy(list(formula_nodes))
    matches = [node for node in output if node["formula_id"] == formula_id]
    if len(matches) != 1:
        raise ValueError("Formula resolution requires exactly one matching stable ID")
    matches[0]["latex"] = resolved["resolution"]["latex"]
    matches[0]["resolution_audit"] = copy.deepcopy(resolved["audit"])
    return output


def _label_group(label: str) -> str:
    if label in ACCEPTABLE_LABELS:
        return "acceptable"
    if label in MATERIAL_LABELS:
        return "material"
    return "reference_uncertain"


def _label_group_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(_label_group(row["formulanet_label"]) for row in rows)
    return {key: counts[key] for key in ("acceptable", "material")}


def _sha256_json(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(data).hexdigest()
