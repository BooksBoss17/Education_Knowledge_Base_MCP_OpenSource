"""Production Output Audit v1 single-task policy and orchestration seam."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .audit_batching import (
    DEFAULT_MAX_EVIDENCE_BYTES,
    DEFAULT_MAX_IMAGE_COUNT,
    DEFAULT_MAX_PAYLOAD_BYTES,
    AuditBatchManifest,
    AuditBatchPlanner,
    write_json,
)
from .pdf_document_ir import semantic_sha256
from .pdf_output_audit import (
    ISSUE_TAXONOMY,
    PATCH_OPERATIONS,
    RESULT_SCHEMA,
    RESULT_STATUSES,
    LeanHandoffPackager,
    OutputAuditPatchPipeline,
    _patch_node_ids,
    validate_audited_document,
)
from .reference_patch_planning import (
    ReferencePatchDependencyAnalyzer,
    ReferencePatchPlanner,
)

PRODUCTION_RESULT_SCHEMA = "bemarkdown-output-audit-result-v1"
PRODUCTION_POLICY_SCHEMA = "bemarkdown-output-audit-production-policy-v1"
PRODUCTION_CORRECTION_BASES = {
    "VISUAL_DIRECT",
    "VISUAL_CONTEXT_DISAMBIGUATION",
    "STRUCTURE_VISUAL",
}
PRODUCTION_SELF_CHECK = "SOURCE_MATCH_CONFIRMED"
PRODUCTION_TERMINAL_STATUSES = {
    "AUDITED",
    "AUDITED_WITH_UNRESOLVED",
    "TECHNICAL_FAILURE",
}
STRUCTURAL_OPERATIONS = {
    "DELETE_DUPLICATE",
    "MOVE_BLOCK",
    "INSERT_BLOCK",
    "SPLIT_BLOCK",
    "MERGE_BLOCK",
    "CHANGE_HEADING_LEVEL",
    "CHANGE_BLOCK_KIND",
    "UPDATE_CAPTION_RELATION",
    "REPLACE_TABLE",
}
_RESULT_FIELDS = {
    "schema",
    "audit_page_id",
    "status",
    "issues",
    "patches",
    "unresolved",
}


def production_audit_result_schema() -> dict[str, Any]:
    """Return the provider-neutral v1 result contract."""

    issue_schema = {
        "type": "object",
        "required": ["issue_type"],
        "properties": {"issue_type": {"enum": sorted(ISSUE_TAXONOMY)}},
    }
    patch_schema = {
        "type": "object",
        "required": [
            "op",
            "issue_type",
            "correction_basis",
            "source_evidence_refs",
            "self_check",
        ],
        "properties": {
            "op": {"enum": sorted(PATCH_OPERATIONS)},
            "issue_type": {
                "enum": sorted(ISSUE_TAXONOMY - {"SOURCE_CONTENT_ANOMALY"})
            },
            "correction_basis": {"enum": sorted(PRODUCTION_CORRECTION_BASES)},
            "source_evidence_refs": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "self_check": {"const": PRODUCTION_SELF_CHECK},
            "affected_node_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "allOf": [
            {
                "if": {"properties": {"op": {"enum": sorted(STRUCTURAL_OPERATIONS)}}},
                "then": {"required": ["affected_node_ids"]},
            }
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": PRODUCTION_RESULT_SCHEMA,
        "type": "object",
        "required": sorted(_RESULT_FIELDS),
        "properties": {
            "schema": {"const": PRODUCTION_RESULT_SCHEMA},
            "audit_page_id": {"type": "string", "minLength": 1},
            "status": {"enum": sorted(RESULT_STATUSES)},
            "issues": {"type": "array", "items": issue_schema},
            "patches": {"type": "array", "items": patch_schema},
            "unresolved": {"type": "array", "items": issue_schema},
        },
        "additionalProperties": False,
        "statuses": sorted(RESULT_STATUSES),
        "issue_taxonomy": sorted(ISSUE_TAXONOMY),
        "patch_operations": sorted(PATCH_OPERATIONS),
        "patch": {
            "required": patch_schema["required"],
            "correction_basis": sorted(PRODUCTION_CORRECTION_BASES),
            "self_check": [PRODUCTION_SELF_CHECK],
            "structural_operations": sorted(STRUCTURAL_OPERATIONS),
            "structural_required": ["affected_node_ids"],
        },
        "complete_row_required_for_every_page": True,
        "internal_reasoning_required_in_output": False,
    }


def production_audit_policy() -> dict[str, Any]:
    """Return the frozen single-task production policy."""

    return {
        "schema": PRODUCTION_POLICY_SCHEMA,
        "content_agent_calls_per_batch": 1,
        "content_decision_retry_count": 0,
        "technical_failure_retry_max": 1,
        "agent_internal_steps": [
            "SOURCE_VS_DRAFT_CHECK",
            "PROPOSE_CANDIDATE_PATCHES",
            "SOURCE_RECHECK_EACH_PATCH",
            "DROP_UNSUPPORTED_PATCHES",
            "RETURN_VERIFIED_PATCHES_OR_UNRESOLVED",
        ],
        "chain_of_thought_output_required": False,
        "patch_self_check": PRODUCTION_SELF_CHECK,
        "correction_bases": sorted(PRODUCTION_CORRECTION_BASES),
        "source_content_anomaly_behavior": "RECORD_WITHOUT_REWRITING_SOURCE_MEANING",
        "non_applied_patch_behavior": "ROLLBACK_AND_UNRESOLVED",
        "human_fallback": "ONLY_WHEN_UNRESOLVED_AND_MUST_RESOLVE_TRUE",
        "batching": {
            "preferred_pages": [4, 6],
            "max_pages": 6,
            "payload_bytes_bounded": True,
            "image_count_bounded": True,
            "evidence_bytes_bounded": True,
            "state_authority": "DISK_CHECKPOINT",
        },
        "authority": "AUDITED_DOCUMENT_IR",
        "handoff": "LEAN_HANDOFF_FROM_DOCUMENT_IR",
        "reference_adjudication": "OPTIONAL_RESEARCH_EVIDENCE_NOT_RELEASE_GATE",
        "legacy_validation": "RETIRED_UNCONSUMED_FROM_RELEASE_GATE",
        "semantic_quality_gate": "PREREGISTERED_CORPUS_V3",
    }


class ProductionAuditResultImporter:
    """Validate complete Production v1 page results before any mutation."""

    def parse(
        self,
        rows: list[dict[str, Any]],
        page_packages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        expected = {str(row["audit_page_id"]): row for row in page_packages}
        ids = [str(row.get("audit_page_id") or "") for row in rows]
        duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"DUPLICATE_AUDIT_PAGE_ID:{duplicates}")
        unknown = sorted(set(ids) - set(expected))
        if unknown:
            raise ValueError(f"UNKNOWN_AUDIT_PAGE_ID:{unknown}")
        missing = sorted(set(expected) - set(ids))
        if missing:
            raise ValueError(f"MISSING_AUDIT_PAGE_ID:{missing}")

        parsed = []
        for row in rows:
            self._validate_row(row, expected[str(row["audit_page_id"])])
            parsed.append(copy.deepcopy(row))
        return parsed

    def _validate_row(self, row: dict[str, Any], package: dict[str, Any]) -> None:
        if not isinstance(row, dict) or set(row) != _RESULT_FIELDS:
            raise ValueError("TOP_LEVEL_FIELDS_INVALID")
        if row.get("schema") != PRODUCTION_RESULT_SCHEMA:
            raise ValueError("RESULT_SCHEMA_INVALID")
        if row.get("status") not in RESULT_STATUSES:
            raise ValueError("RESULT_STATUS_INVALID")
        for field in ("issues", "patches", "unresolved"):
            if not isinstance(row.get(field), list) or not all(
                isinstance(item, dict) for item in row[field]
            ):
                raise ValueError(f"{field.upper()}_OBJECT_ARRAY_REQUIRED")
        if row["status"] == "NO_CHANGE" and (row["patches"] or row["unresolved"]):
            raise ValueError("NO_CHANGE_HAS_CHANGES")
        if row["status"] == "PATCHED" and not row["patches"]:
            raise ValueError("PATCHED_WITHOUT_PATCHES")
        if row["status"] == "UNRESOLVED" and not row["unresolved"]:
            raise ValueError("UNRESOLVED_WITHOUT_FINDING")
        if row["status"] == "TECHNICAL_FAILURE" and row["patches"]:
            raise ValueError("TECHNICAL_FAILURE_HAS_PATCHES")
        for issue in [*row["issues"], *row["unresolved"]]:
            if issue.get("issue_type") not in ISSUE_TAXONOMY:
                raise ValueError("ISSUE_TYPE_INVALID")

        allowed_nodes = {str(value) for value in package.get("node_ids", [])}
        for patch in row["patches"]:
            self._validate_patch(patch, allowed_nodes)

    @staticmethod
    def _validate_patch(patch: dict[str, Any], allowed_nodes: set[str]) -> None:
        operation = patch.get("op")
        if operation not in PATCH_OPERATIONS:
            raise ValueError("PATCH_OPERATION_INVALID")
        if patch.get("issue_type") not in ISSUE_TAXONOMY:
            raise ValueError("PATCH_ISSUE_TYPE_INVALID")
        if patch.get("issue_type") == "SOURCE_CONTENT_ANOMALY":
            raise ValueError("SOURCE_CONTENT_ANOMALY_CANNOT_BE_PATCHED")
        if patch.get("correction_basis") not in PRODUCTION_CORRECTION_BASES:
            raise ValueError("PATCH_CORRECTION_BASIS_INVALID")
        evidence = patch.get("source_evidence_refs")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(value, str) and value.strip() for value in evidence
        ):
            raise ValueError("PATCH_SOURCE_EVIDENCE_REFS_REQUIRED")
        if patch.get("self_check") != PRODUCTION_SELF_CHECK:
            raise ValueError("PATCH_SELF_CHECK_INVALID")
        referenced = _patch_node_ids(patch)
        if not referenced:
            raise ValueError("PATCH_TARGET_NODE_IDS_REQUIRED")
        if referenced - allowed_nodes:
            raise ValueError("PATCH_NODE_REFERENCE_INVALID")
        if operation in STRUCTURAL_OPERATIONS:
            affected = patch.get("affected_node_ids")
            if not isinstance(affected, list) or not affected:
                raise ValueError("PATCH_AFFECTED_NODE_IDS_REQUIRED")
            affected_set = {str(value) for value in affected}
            mutation_targets = {
                str(value)
                for key in ("target_node_id", "caption_node_id")
                if (value := patch.get(key))
            }
            mutation_targets.update(str(value) for value in patch.get("target_node_ids", []))
            expected_affected = mutation_targets or referenced
            if affected_set - allowed_nodes or affected_set != expected_affected:
                raise ValueError("PATCH_AFFECTED_NODE_IDS_INVALID")


def _to_v0_result(result: dict[str, Any], patches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    converted = copy.deepcopy(result)
    converted["schema"] = RESULT_SCHEMA
    if patches is not None:
        converted["patches"] = copy.deepcopy(patches)
    for patch in converted["patches"]:
        patch.pop("self_check", None)
        patch.pop("affected_node_ids", None)
    if converted["patches"]:
        converted["status"] = "PATCHED"
    elif converted["unresolved"]:
        converted["status"] = "UNRESOLVED"
    elif converted["status"] != "TECHNICAL_FAILURE":
        converted["status"] = "NO_CHANGE"
    return converted


def _dependency_plan(result: dict[str, Any], package: dict[str, Any]) -> list[dict[str, Any]]:
    converted = _to_v0_result(result)
    plan = ReferencePatchPlanner().plan(
        [converted], {str(package["audit_page_id"]): package}
    )
    dependency = ReferencePatchDependencyAnalyzer().analyze(plan["patches"])
    return [
        {**status, "planned_patch": planned}
        for planned, status in zip(plan["patches"], dependency["rows"], strict=True)
    ]


def _primitive_patches(planned: dict[str, Any]) -> list[dict[str, Any]]:
    patch = copy.deepcopy(planned["normalized_patch"])
    operation = patch["op"]
    if operation == "DELETE_DUPLICATE":
        return [
            {**patch, "target_node_id": node_id}
            for node_id in patch.get("target_node_ids", [])
        ]
    if operation == "MOVE_BLOCK":
        targets = list(patch.get("target_node_ids", []))
        if patch.get("after_node_id"):
            targets.reverse()
        return [
            {
                **patch,
                "target_node_id": node_id,
                "target_node_ids": [node_id],
            }
            for node_id in targets
        ]
    return [patch]


def apply_audit_batch_result(
    document: dict[str, Any],
    page_packages: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    agent: dict[str, Any],
    timestamp: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one content-agent batch once; all non-applied changes become unresolved."""

    parsed = ProductionAuditResultImporter().parse(results, page_packages)
    packages = {str(row["audit_page_id"]): row for row in page_packages}
    current = copy.deepcopy(document)
    page_reports = []
    for result in parsed:
        audit_page_id = str(result["audit_page_id"])
        package = packages[audit_page_id]
        if result["status"] == "TECHNICAL_FAILURE":
            page_reports.append(
                {
                    "audit_page_id": audit_page_id,
                    "status": "TECHNICAL_FAILURE",
                    "patched": 0,
                    "unresolved": copy.deepcopy(result["unresolved"]),
                    "technical_failures": 1,
                }
            )
            continue

        dependencies = _dependency_plan(result, package)
        eligible: list[dict[str, Any]] = []
        unresolved = [
            {"reason": "AGENT_UNRESOLVED", **copy.deepcopy(value)}
            for value in result["unresolved"]
        ]
        for patch, dependency in zip(result["patches"], dependencies, strict=True):
            if dependency["status"] == "PATCH_DEPENDENCY_VALID":
                eligible.append(dependency["planned_patch"])
            else:
                unresolved.append(
                    {
                        "reason": dependency["status"],
                        "reason_codes": copy.deepcopy(dependency["reason_codes"]),
                        "issue_type": patch["issue_type"],
                        "target_node_ids": sorted(_patch_node_ids(patch)),
                        "patch_fingerprint": semantic_sha256(patch),
                    }
                )

        applied = 0
        for planned in eligible:
            before_patch = current
            primitives = _primitive_patches(planned)
            v0_result = _to_v0_result(result, primitives)
            v0_result["unresolved"] = []
            candidate, application = OutputAuditPatchPipeline().apply(
                current,
                package,
                v0_result,
                agent=agent,
                timestamp=timestamp,
            )
            failures = [row for row in application["patches"] if row["state"] != "APPLIED"]
            if not failures and len(application["patches"]) == len(primitives):
                current = candidate
                applied += 1
            else:
                current = before_patch
                reasons = [reason for row in failures for reason in row["reason_codes"]]
                failure_reason = (
                    failures[0]["reason"]
                    if failures
                    else "PATCH_REJECTED_MACHINE_VALIDATION"
                )
                unresolved.append(
                    {
                        "reason": failure_reason,
                        "reason_codes": reasons,
                        "issue_type": planned["issue_type"],
                        "target_node_ids": copy.deepcopy(planned["target_node_ids"]),
                        "patch_id": planned["patch_id"],
                    }
                )
        page_reports.append(
            {
                "audit_page_id": audit_page_id,
                "status": "AUDITED_WITH_UNRESOLVED" if unresolved else "AUDITED",
                "patched": applied,
                "unresolved": unresolved,
                "technical_failures": 0,
            }
        )

    technical_failures = sum(row["technical_failures"] for row in page_reports)
    unresolved_count = sum(len(row["unresolved"]) for row in page_reports)
    status = (
        "TECHNICAL_FAILURE"
        if technical_failures
        else "AUDITED_WITH_UNRESOLVED"
        if unresolved_count
        else "AUDITED"
    )
    return current, {
        "schema": "bemarkdown-output-audit-production-batch-summary-v1",
        "status": status,
        "pages": page_reports,
        "patched": sum(row["patched"] for row in page_reports),
        "unresolved": unresolved_count,
        "technical_failures": technical_failures,
        "content_agent_calls": 1,
        "recursive_content_audit_calls": 0,
    }


class ProductionAuditBatchController:
    """Terminate content decisions; permit at most one technical retry."""

    def __init__(
        self, batch_ids: list[str], checkpoint_path: str | Path | None = None
    ):
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("BATCH_IDS_MUST_BE_UNIQUE")
        self._state = {
            batch_id: {
                "status": "PENDING",
                "technical_retry_count": 0,
                "content_retry_count": 0,
            }
            for batch_id in batch_ids
        }
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint()

    def record(self, batch_id: str, status: str) -> str:
        row = self._state[batch_id]
        if row["status"].startswith("TERMINATED_"):
            raise RuntimeError("BATCH_ALREADY_TERMINATED")
        if status == "TECHNICAL_FAILURE":
            if row["technical_retry_count"] < 1:
                row["technical_retry_count"] += 1
                row["status"] = "RETRY_PENDING"
                self.checkpoint()
                return "RETRY_TECHNICAL_ONCE"
            row["status"] = "TERMINATED_TECHNICAL_FAILURE"
            self.checkpoint()
            return "TERMINATE_TECHNICAL_FAILURE"
        if status not in {"AUDITED", "AUDITED_WITH_UNRESOLVED"}:
            raise ValueError("PRODUCTION_BATCH_STATUS_INVALID")
        row["status"] = f"TERMINATED_{status}"
        self.checkpoint()
        return "TERMINATE_CONTENT"

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._state)

    def checkpoint(self) -> None:
        if self._checkpoint_path is not None:
            write_json(
                self._checkpoint_path,
                {
                    "schema": "bemarkdown-output-audit-production-batch-state-v1",
                    "batches": self._state,
                },
            )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> ProductionAuditBatchController:
        checkpoint_path = Path(path)
        value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if value.get("schema") != "bemarkdown-output-audit-production-batch-state-v1":
            raise ValueError("PRODUCTION_BATCH_CHECKPOINT_SCHEMA_INVALID")
        controller = cls(list(value["batches"]))
        controller._checkpoint_path = checkpoint_path
        controller._state = copy.deepcopy(value["batches"])
        return controller


def prepare_audit_batches(
    manifest_rows: list[dict[str, Any]],
    page_features: dict[str, dict[str, Any]],
    *,
    max_pages_per_batch: int = 6,
    max_payload_bytes: int | None = DEFAULT_MAX_PAYLOAD_BYTES,
    max_image_count: int | None = DEFAULT_MAX_IMAGE_COUNT,
    max_evidence_bytes: int | None = DEFAULT_MAX_EVIDENCE_BYTES,
) -> AuditBatchManifest:
    if max_pages_per_batch > 6:
        raise ValueError("PRODUCTION_AUDIT_MAX_PAGES_IS_SIX")
    return AuditBatchPlanner(
        max_pages_per_batch=max_pages_per_batch,
        max_payload_bytes=max_payload_bytes,
        max_image_count=max_image_count,
        max_evidence_bytes=max_evidence_bytes,
        batch_id_prefix="production-audit-batch",
    ).plan(manifest_rows, page_features)


def finalize_audit(
    document: dict[str, Any], page_reports: list[dict[str, Any]]
) -> dict[str, Any]:
    validate_audited_document(document)
    patched = sum(int(row.get("patched", 0)) for row in page_reports)
    unresolved = sum(
        len(value) if isinstance(value := row.get("unresolved", []), list) else int(value)
        for row in page_reports
    )
    technical = sum(int(row.get("technical_failures", 0)) for row in page_reports)
    status = (
        "TECHNICAL_FAILURE"
        if technical
        else "AUDITED_WITH_UNRESOLVED"
        if unresolved
        else "AUDITED"
    )
    return {
        "status": status,
        "pages": len(page_reports),
        "patched": patched,
        "unresolved": unresolved,
        "technical_failures": technical,
    }


def render_handoff(
    document: dict[str, Any],
    source_package: str | Path,
    target: str | Path,
    audit_summary: dict[str, Any],
) -> dict[str, Any]:
    if audit_summary.get("status") not in PRODUCTION_TERMINAL_STATUSES:
        raise ValueError("AUDIT_SUMMARY_NOT_TERMINAL")
    return LeanHandoffPackager().write(
        document,
        source_package,
        target,
        audit_status=str(audit_summary["status"]),
    )


def classify_legacy_reference_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay v0 patches against v1 eligibility without asserting visual quality."""

    packages = {
        str(row["audit_page_id"]): {
            "audit_page_id": str(row["audit_page_id"]),
            "document_id": "legacy-reference",
            "page_index": index,
            "node_ids": [],
        }
        for index, row in enumerate(rows)
    }
    plan = ReferencePatchPlanner().plan(rows, packages)
    counts = Counter()
    for patch in plan["patches"]:
        normalized = patch["normalized_patch"]
        if patch["normalization_errors"]:
            counts["v1_invalid"] += 1
        elif patch["operation"] in STRUCTURAL_OPERATIONS and not normalized.get(
            "affected_node_ids"
        ):
            counts["v1_unresolved_due_structure"] += 1
        elif (
            normalized.get("correction_basis") not in PRODUCTION_CORRECTION_BASES
            or not normalized.get("source_evidence_refs")
            or normalized.get("self_check") != PRODUCTION_SELF_CHECK
            or not patch.get("issue_type")
        ):
            counts["v1_unresolved_due_evidence"] += 1
        else:
            counts["v1_auto_eligible"] += 1
    metrics = {
        "schema": "bemarkdown-output-audit-legacy-reference-replay-v1",
        "v0_total_patches": plan["patch_count"],
        "v1_auto_eligible": counts["v1_auto_eligible"],
        "v1_unresolved_due_evidence": counts["v1_unresolved_due_evidence"],
        "v1_unresolved_due_structure": counts["v1_unresolved_due_structure"],
        "v1_invalid": counts["v1_invalid"],
        "visual_correctness_claimed": False,
        "interpretation": "CONSERVATIVENESS_AND_OPERATIONAL_IMPACT_ONLY",
    }
    metrics["fingerprint"] = semantic_sha256(metrics)
    return metrics


def legacy_validation_retirement_record() -> dict[str, Any]:
    return {
        "schema": "bemarkdown-output-audit-legacy-validation-retirement-v1",
        "validation_page_count": 29,
        "validation_status": "NOT_CONSUMED",
        "release_gate_status": "RETIRED_FROM_RELEASE_GATE",
        "packages_opened": False,
        "deleted": False,
        "fresh_validation_claim_preserved": False,
        "future_use": "DIAGNOSTIC_OR_REGRESSION_SAMPLE_ONLY",
        "reason": "COMPLETE_PRE_FROZEN_STABLE_TARGET_TRUTH_UNAVAILABLE",
    }


def single_task_termination_policy() -> dict[str, Any]:
    return {
        "schema": "bemarkdown-output-audit-single-task-termination-v1",
        "content_agent_calls_per_batch": 1,
        "content_retry_count": 0,
        "technical_retry_max": 1,
        "terminal_statuses": sorted(PRODUCTION_TERMINAL_STATUSES),
        "fail_closed_mapping": {
            "SOURCE_FIDELITY_GUARD_REJECT": "UNRESOLVED",
            "MACHINE_VALIDATION_REJECT": "UNRESOLVED",
            "PATCH_CONFLICT": "UNRESOLVED",
            "PATCH_BLOCKED_BY_DEPENDENCY": "UNRESOLVED",
        },
        "recursive_content_agent_allowed": False,
        "human_review_condition": "UNRESOLVED_AND_MUST_RESOLVE_TRUE",
    }


def corpus_v3_quality_preregistration() -> dict[str, Any]:
    """Freeze the protocol template; no Corpus v3 documents are selected or run here."""

    value = {
        "schema": "bemarkdown-corpus-v3-quality-preregistration-v1",
        "status": "PROTOCOL_PREREGISTERED_CORPUS_INSTANCE_PENDING",
        "execution_started": False,
        "model_output_observed": False,
        "required_pre_output_freezes": [
            "DOCUMENT_LIST_FINGERPRINT",
            "EVALUATION_SUBSET",
            "SOURCE_ONLY_TRUTH_AND_ADJUDICATION_PROTOCOL",
            "METRICS",
            "PASS_FAIL_THRESHOLDS",
        ],
        "sampling": {
            "end_to_end_page_sample": True,
            "stratified_patch_target_sample": True,
            "critical_content_sample": True,
            "required_categories": [
                "TEXT",
                "FORMULA",
                "TABLE",
                "IMAGE",
                "READING_ORDER",
                "HEADING_STRUCTURE",
                "MISSING_CONTENT",
            ],
        },
        "metrics": [
            "CONTENT_PRESERVATION",
            "TRUE_CORRECTION",
            "FALSE_CORRECTION",
            "MATERIAL_FALSE_CORRECTION",
            "SILENT_MATERIAL_ERROR",
            "UNRESOLVED_RATE",
            "HUMAN_FALLBACK_RATE",
            "END_TO_END_HANDOFF_COMPLETENESS",
        ],
        "default_hard_thresholds": {
            "machine_invalid_applied_patch_count": 0,
            "material_false_correction_count": 0,
            "silent_material_error_count": 0,
            "critical_content_preservation_rate": 1.0,
            "end_to_end_handoff_completeness_rate": 1.0,
        },
        "threshold_freeze_rule": (
            "CORPUS_SPECIFIC_NUMERIC_THRESHOLDS_AND_SAMPLE_COUNTS_MUST_BE_FINGERPRINTED_"
            "BEFORE_ANY_MODEL_OUTPUT"
        ),
        "post_output_threshold_change_allowed": False,
        "patch_volume_is_quality_metric": False,
    }
    value["protocol_fingerprint"] = semantic_sha256(value)
    return value
