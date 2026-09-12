from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .audit_batching import AuditBatchPlanner
from .pdf_document_ir import OutputAuditPatchEngine, semantic_sha256
from .pdf_output_audit import SourceFidelityGuard, validate_audited_document
from .reference_patch_planning import (
    DEPENDENCY_SCHEMA,
    PATCH_PLAN_SCHEMA,
    ReferencePatchDependencyAnalyzer,
    ReferencePatchPlanner,
)

__all__ = [
    "DEPENDENCY_SCHEMA",
    "PATCH_PLAN_SCHEMA",
    "ReferencePatchDependencyAnalyzer",
    "ReferencePatchPlanner",
]

EXECUTION_SCHEMA = "bemarkdown-reference-patch-execution-v0"
TRUTH_COVERAGE_SCHEMA = "bemarkdown-reference-truth-coverage-v0"


def _page_snapshot(document: dict[str, Any], page_index: int) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(block)
        for block in [*document.get("blocks", []), *document.get("suppressed_blocks", [])]
        if int(block["page_index"]) == int(page_index)
    ]


def _meaningful_node_mutations(
    before: dict[str, Any], after: dict[str, Any], allowed: set[str]
) -> list[str]:
    fields = (
        "kind",
        "subtype",
        "content",
        "relations",
        "source_content_ids",
        "source_candidate_ids",
        "source_region_ids",
        "source_unit_ids",
        "bbox_pdf_pt",
        "review_state",
        "visibility",
    )
    left = {
        block["node_id"]: block
        for block in [*before.get("blocks", []), *before.get("suppressed_blocks", [])]
    }
    right = {
        block["node_id"]: block
        for block in [*after.get("blocks", []), *after.get("suppressed_blocks", [])]
    }
    return [
        node_id
        for node_id in sorted(set(left) & set(right) - allowed)
        if {field: left[node_id].get(field) for field in fields}
        != {field: right[node_id].get(field) for field in fields}
    ]


class ReferencePatchExecutor:
    def execute(
        self,
        documents: dict[str, dict[str, Any]],
        manifest_rows: list[dict[str, Any]],
        page_packages: dict[str, dict[str, Any]],
        results: list[dict[str, Any]],
        plan_rows: list[dict[str, Any]],
        dependency_report: dict[str, Any],
        evidence_root: str | Path,
        *,
        mode: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        evidence_root = Path(evidence_root).resolve()
        current = copy.deepcopy(documents)
        dependency = {row["patch_id"]: row for row in dependency_report["rows"]}
        plans_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in plan_rows:
            plans_by_page[row["audit_page_id"]].append(row)
        results_by_page = {str(row["audit_page_id"]): row for row in results}
        aliases: dict[str, str] = {}
        patch_rows = []
        pages = []
        for manifest in manifest_rows:
            audit_page_id = str(manifest["audit_page_id"])
            package = page_packages[audit_page_id]
            document = current[str(manifest["document_id"])]
            page_index = int(manifest["page_index"])
            before_page = _page_snapshot(document, page_index)
            page_patch_rows = []
            for plan in plans_by_page.get(audit_page_id, []):
                row = self._execute_one(
                    document,
                    package,
                    plan,
                    dependency[plan["patch_id"]],
                    aliases,
                    evidence_root,
                )
                document = row.pop("document")
                current[str(manifest["document_id"])] = document
                page_patch_rows.append(row)
                patch_rows.append(row)
            after_page = _page_snapshot(document, page_index)
            unresolved = list(results_by_page[audit_page_id].get("unresolved", []))
            pages.append(
                {
                    "audit_page_id": audit_page_id,
                    "document_id": str(manifest["document_id"]),
                    "page_index": page_index,
                    "before_sha256": semantic_sha256(before_page),
                    "after_sha256": semantic_sha256(after_page),
                    "applied_patch_ids": [
                        row["patch_id"] for row in page_patch_rows if row["state"] == "APPLIED"
                    ],
                    "rejected_patch_ids": [
                        row["patch_id"]
                        for row in page_patch_rows
                        if row["state"].startswith("REJECTED_")
                    ],
                    "blocked_patch_ids": [
                        row["patch_id"]
                        for row in page_patch_rows
                        if row["state"].startswith("BLOCKED_")
                    ],
                    "unresolved_count": len(unresolved),
                    "unresolved": copy.deepcopy(unresolved),
                }
            )
        validation = {
            document_id: validate_audited_document(document)
            for document_id, document in sorted(current.items())
        }
        state_counts = Counter(row["state"] for row in patch_rows)
        unresolved_count = sum(page["unresolved_count"] for page in pages)
        report = {
            "schema": EXECUTION_SCHEMA,
            "mode": mode,
            "patch_count": len(patch_rows),
            "page_count": len(pages),
            "state_counts": dict(sorted(state_counts.items())),
            "original_unresolved_count": unresolved_count,
            "silent_drop_count": len(plan_rows) - len(patch_rows),
            "pages": pages,
            "patches": patch_rows,
            "machine_validation": validation,
            "machine_validation_passed": len(validation) == len(documents),
        }
        report["fingerprint"] = semantic_sha256(report)
        return current, report

    def _execute_one(
        self,
        document: dict[str, Any],
        page_package: dict[str, Any],
        plan: dict[str, Any],
        dependency: dict[str, Any],
        aliases: dict[str, str],
        evidence_root: Path,
    ) -> dict[str, Any]:
        base = {
            "patch_id": plan["patch_id"],
            "audit_page_id": plan["audit_page_id"],
            "operation": plan["operation"],
            "issue_type": plan["issue_type"],
            "correction_basis": plan["correction_basis"],
            "original_result_fingerprint": plan["original_result_fingerprint"],
            "original_patch_fingerprint": plan["original_patch_fingerprint"],
        }
        if dependency["status"] == "PATCH_CONFLICT":
            return {
                **base,
                "state": "BLOCKED_CONFLICT",
                "reason_codes": dependency["reason_codes"],
                "document": document,
            }
        if dependency["status"] == "PATCH_BLOCKED_BY_DEPENDENCY":
            return {
                **base,
                "state": "BLOCKED_DEPENDENCY",
                "reason_codes": dependency["reason_codes"],
                "document": document,
            }
        normalized = copy.deepcopy(plan["normalized_patch"])
        if normalized.pop("after_node_alias", None):
            alias = plan["required_aliases"][0]
            if alias not in aliases:
                return {
                    **base,
                    "state": "BLOCKED_DEPENDENCY",
                    "reason_codes": [f"CREATED_ALIAS_UNAVAILABLE:{alias}"],
                    "document": document,
                }
            normalized["after_node_id"] = aliases[alias]
        dynamic_package = copy.deepcopy(page_package)
        dynamic_package["node_ids"] = [
            block["node_id"]
            for block in document.get("blocks", [])
            if int(block["page_index"]) == int(page_package["page_index"])
        ]
        guard_reasons = SourceFidelityGuard().validate(
            document, dynamic_package, normalized
        )
        guard_reasons.extend(
            self._evidence_reasons(
                normalized.get("source_evidence_refs", []),
                evidence_root,
                plan["audit_page_id"],
            )
        )
        if guard_reasons:
            return {
                **base,
                "state": "REJECTED_GUARD",
                "reason_codes": sorted(set(guard_reasons)),
                "document": document,
            }
        try:
            operations = self._engine_operations(normalized)
            candidate, engine_log = OutputAuditPatchEngine().apply(document, operations)
            created = [
                node_id
                for operation in engine_log["operations"]
                for node_id in operation.get("created_node_ids", [])
            ]
            allowed = set(plan["target_node_ids"]) | set(created)
            unintended = _meaningful_node_mutations(document, candidate, allowed)
            if unintended:
                raise ValueError(f"Patch modified non-target nodes: {unintended}")
            validation = validate_audited_document(candidate)
            if plan["operation"] == "INSERT_BLOCK" and plan["created_aliases"]:
                aliases[plan["created_aliases"][0]] = created[0]
            if plan["operation"] == "SPLIT_BLOCK":
                for alias, node_id in zip(plan["created_aliases"], created, strict=True):
                    aliases[alias] = node_id
            return {
                **base,
                "state": "APPLIED",
                "reason_codes": [],
                "before_sha256": semantic_sha256(document),
                "after_sha256": semantic_sha256(candidate),
                "created_node_ids": created,
                "machine_validation": validation,
                "document": candidate,
            }
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            return {
                **base,
                "state": "REJECTED_MACHINE_VALIDATION",
                "reason_codes": [type(exc).__name__, str(exc)],
                "document": document,
            }

    @staticmethod
    def _evidence_reasons(
        refs: Iterable[str], evidence_root: Path, audit_page_id: str
    ) -> list[str]:
        reasons = []
        for reference in refs:
            path_text = str(reference).split("#", 1)[0].replace("\\", "/")
            relative = Path(path_text)
            if relative.is_absolute() or ".." in relative.parts:
                reasons.append("SOURCE_EVIDENCE_REF_UNSAFE")
                continue
            if path_text.startswith("pages/"):
                candidate = evidence_root / relative
            else:
                candidate = evidence_root / "pages" / audit_page_id / relative
            resolved = candidate.resolve()
            if evidence_root not in resolved.parents or not resolved.is_file():
                reasons.append("SOURCE_EVIDENCE_NOT_FOUND")
        return reasons

    @staticmethod
    def _engine_operations(patch: dict[str, Any]) -> list[dict[str, Any]]:
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


def classify_truth_coverage(
    reference_pages: Iterable[str],
    patch_nodes: Iterable[str],
    issue_types: Iterable[str],
    eligible_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    pages = set(reference_pages)
    nodes = set(patch_nodes)
    issues = set(issue_types)
    truth_pages = {
        str(value)
        for row in eligible_records
        for value in row.get("audit_page_ids", [])
    }
    truth_nodes = {
        str(value) for row in eligible_records for value in row.get("node_ids", [])
    }
    truth_issues = {
        str(value) for row in eligible_records for value in row.get("issue_types", [])
    }
    covered_pages = pages & truth_pages
    covered_nodes = nodes & truth_nodes
    covered_issues = issues & truth_issues
    full = covered_pages == pages and covered_nodes == nodes and covered_issues == issues
    any_coverage = bool(covered_pages or covered_nodes or covered_issues)
    classification = (
        "FULL_TRUTH_COVERAGE"
        if full
        else "PARTIAL_TRUTH_COVERAGE"
        if any_coverage
        else "NO_FROZEN_TRUTH"
    )
    return {
        "schema": TRUTH_COVERAGE_SCHEMA,
        "classification": classification,
        "sufficient_for_full_quality_gate": full,
        "page_coverage": f"{len(covered_pages)}/{len(pages)}",
        "node_coverage": f"{len(covered_nodes)}/{len(nodes)}",
        "issue_type_coverage": f"{len(covered_issues)}/{len(issues)}",
        "covered_audit_page_ids": sorted(covered_pages),
        "covered_node_ids": sorted(covered_nodes),
        "covered_issue_types": sorted(covered_issues),
    }


def assess_truth_candidate(
    *,
    path: str,
    sha256: str,
    tracked: bool,
    first_commit: str | None,
    predates_fresh_results: bool,
    independent_of_fresh_results: bool,
    explicit_provenance: bool,
) -> dict[str, Any]:
    reasons = []
    if not tracked or not first_commit:
        reasons.append("IMMUTABLE_GIT_PROVENANCE_MISSING")
    if not predates_fresh_results:
        reasons.append("POST_RESULT_ARTIFACT")
    if not independent_of_fresh_results:
        reasons.append("FRESH_RESULT_DERIVED")
    if not explicit_provenance:
        reasons.append("TRUTH_PROVENANCE_AMBIGUOUS")
    return {
        "path": path,
        "sha256": sha256,
        "first_commit": first_commit,
        "eligible": not reasons,
        "status": "FROZEN_INDEPENDENT_TRUTH" if not reasons else "INELIGIBLE",
        "reason_codes": reasons,
    }


def score_reference_quality(
    comparisons: Iterable[dict[str, Any]],
    *,
    machine_invalid_applied_patch_count: int,
    source_unsupported_auto_rewrite_count: int,
) -> dict[str, Any]:
    rows = []
    for comparison in comparisons:
        before_correct = bool(comparison["before_correct"])
        after_correct = bool(comparison["after_correct"])
        unresolved = bool(comparison.get("unresolved_marked"))
        if not before_correct and after_correct:
            outcome = "TRUE_CORRECTION"
        elif before_correct and not after_correct:
            outcome = "FALSE_CORRECTION"
        elif not before_correct and not after_correct:
            outcome = "UNRESOLVED_ERROR" if unresolved else "MISSED_ERROR"
        else:
            outcome = "UNCHANGED_CORRECT"
        rows.append({**copy.deepcopy(comparison), "outcome": outcome})
    outcomes = Counter(row["outcome"] for row in rows)
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        categories[str(row.get("category") or "UNSPECIFIED")][row["outcome"]] += 1
    false_material = sum(
        row["outcome"] == "FALSE_CORRECTION" and bool(row.get("material"))
        for row in rows
    )
    acceptable_damage = sum(
        row["outcome"] == "FALSE_CORRECTION" and not bool(row.get("material"))
        for row in rows
    )
    silent_material = sum(
        not bool(row["after_correct"])
        and bool(row.get("material"))
        and not bool(row.get("unresolved_marked"))
        for row in rows
    )
    before_errors = sum(not bool(row["before_correct"]) for row in rows)
    after_errors = sum(not bool(row["after_correct"]) for row in rows)
    gate_passed = (
        machine_invalid_applied_patch_count == 0
        and source_unsupported_auto_rewrite_count == 0
        and false_material == 0
        and silent_material == 0
    )
    return {
        "schema": "bemarkdown-reference-quality-metrics-v0",
        "comparison_count": len(rows),
        "before_error_count": before_errors,
        "after_error_count": after_errors,
        "net_error_reduction": before_errors - after_errors,
        "outcome_counts": dict(sorted(outcomes.items())),
        "false_correction_count": outcomes["FALSE_CORRECTION"],
        "false_correction_material_count": false_material,
        "acceptable_content_damage_count": acceptable_damage,
        "silent_material_error_count": silent_material,
        "category_breakdown": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(categories.items())
        },
        "machine_invalid_applied_patch_count": machine_invalid_applied_patch_count,
        "source_unsupported_auto_rewrite_count": source_unsupported_auto_rewrite_count,
        "quality_gate_passed": gate_passed,
        "rows": rows,
    }


def anonymous_before_after_pair(
    before: Any, after: Any, *, seed: str
) -> tuple[dict[str, Any], dict[str, str]]:
    swap = int(semantic_sha256(seed), 16) % 2 == 1
    public = {
        "candidate_a": copy.deepcopy(after if swap else before),
        "candidate_b": copy.deepcopy(before if swap else after),
    }
    private = {
        "candidate_a_origin": "AFTER" if swap else "BEFORE",
        "candidate_b_origin": "BEFORE" if swap else "AFTER",
    }
    return public, private


def build_adjudication_shards(
    manifest_rows: list[dict[str, Any]], page_payloads: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    serialized = json.dumps(page_payloads, ensure_ascii=False).lower()
    if "validation" in serialized:
        raise ValueError("Validation leakage detected in adjudication payload")
    features = {
        audit_page_id: {
            "payload_bytes": len(
                json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ),
            "image_count": 1,
            "evidence_image_count": 0,
            "evidence_bytes": 0,
        }
        for audit_page_id, value in page_payloads.items()
    }
    planned = AuditBatchPlanner(
        max_pages_per_batch=6,
        max_payload_bytes=None,
        max_image_count=None,
        max_evidence_bytes=None,
        batch_id_prefix="reference-adjudication-shard",
    ).plan(manifest_rows, features)
    return {
        "schema": "bemarkdown-reference-adjudication-manifest-v0",
        "page_count": len(manifest_rows),
        "shard_count": len(planned.rows),
        "max_pages_per_shard": max(row["page_count"] for row in planned.rows),
        "origin_hidden_from_adjudicator": True,
        "validation_included": False,
        "shards": planned.rows,
        "fingerprint": planned.fingerprint,
    }
