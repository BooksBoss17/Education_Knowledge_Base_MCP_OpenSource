"""Crop-first residual-risk AgentTask/AgentResult and guarded patch contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

AGENT_TASK_SCHEMA = "bemarkdown-residual-risk-audit-task-v1"
AGENT_RESULT_SCHEMA = "bemarkdown-residual-risk-audit-result-v1"
FORMULA_AUTHORITY_STATE = "CALIBRATED_FROM_STRICT_SOURCE_ORDER_SUBSET"

CONTENT_JUDGMENTS = {
    "SOURCE_MATCHES_PRIMARY",
    "SOURCE_MATCHES_SECONDARY",
    "SOURCE_MATCHES_NEITHER",
    "INSUFFICIENT_VISUAL_EVIDENCE",
}
STRUCTURE_JUDGMENTS = {
    "STRUCTURE_PRIMARY_VALID",
    "STRUCTURE_PATCH_REQUIRED",
    "STRUCTURE_UNRESOLVED",
}
ALL_JUDGMENTS = CONTENT_JUDGMENTS | STRUCTURE_JUDGMENTS
CONTENT_OPS = {
    "REPLACE_TEXT",
    "REPLACE_FORMULA",
    "REPLACE_TABLE_CELL",
    "REPLACE_TABLE_STRUCTURE",
    "CHANGE_BLOCK_TYPE",
}
STRUCTURE_OPS = {
    "REORDER_NODES",
    "INSERT_MISSED_NODE",
    "MERGE_NODE",
    "SPLIT_NODE",
    "RELABEL_HEADING",
}
ALLOWED_OPS = CONTENT_OPS | STRUCTURE_OPS
KNOWN_FAMILIES = {
    "TEXT",
    "FORMULA",
    "TABLE",
    "HEADING",
    "READING_ORDER",
    "MISSING_CONTENT",
    "LAYOUT",
    "IMAGE",
}
STRUCTURE_FAMILIES = {"READING_ORDER", "MISSING_CONTENT", "LAYOUT"}
CORRECTION_BASES = {
    "VISUAL_DIRECT",
    "VISUAL_CONTEXT_DISAMBIGUATION",
    "STRUCTURE_VISUAL",
}
_FORBIDDEN_TOKENS = {
    "truth",
    "reference answer",
    "reference_answer",
    "expected output",
    "expected_output",
    "answer key",
    "answer_key",
    "scoring result",
    "scoring_result",
}

_TASK_FIELDS = {
    "schema",
    "task_id",
    "document_id",
    "page_id",
    "risk_family",
    "target_nodes",
    "source_visuals",
    "primary_evidence",
    "secondary_evidence",
    "risk_reasons",
    "validator_state",
    "typed_authority_policy_refs",
    "bounded_context",
    "allowed_judgments",
    "allowed_patch_ops",
    "source_fidelity_rules",
}
_RESULT_FIELDS = {"schema", "task_id", "targets", "executor_metadata"}
_RESULT_TARGET_FIELDS = {
    "node_id",
    "judgment",
    "issue_type",
    "source_evidence_refs",
    "self_check",
    "correction_basis",
    "replacement",
    "patch",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _risk_family(node: Mapping[str, Any]) -> str:
    family = str(node.get("content_type") or "LAYOUT").upper()
    return family if family in KNOWN_FAMILIES else "LAYOUT"


def _context_level(families: set[str]) -> tuple[str, str]:
    if families & STRUCTURE_FAMILIES:
        return (
            "L3_FULL_PAGE_STRUCTURE",
            "SPECIFIED_STRUCTURAL_RELATION_REQUIRES_CLEAN_PAGE_CONTEXT",
        )
    if families & {"TABLE", "HEADING", "IMAGE"}:
        return "L1_LOCAL_CONTEXT", "LOCAL_STRUCTURE_AROUND_TARGETS_REQUIRED"
    return "L0_EXACT_CROP", "EXACT_SOURCE_CROPS_ARE_SUFFICIENT_BY_DEFAULT"


def _op_for_family(family: str) -> str:
    return {
        "FORMULA": "REPLACE_FORMULA",
        "TABLE": "REPLACE_TABLE_CELL",
        "HEADING": "RELABEL_HEADING",
        "TEXT": "REPLACE_TEXT",
        "IMAGE": "CHANGE_BLOCK_TYPE",
    }.get(family, "REPLACE_TEXT")


def _find_forbidden(value: Any, *, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower().replace("-", "_")
            if any(token.replace(" ", "_") in key_text for token in _FORBIDDEN_TOKENS):
                findings.append(f"{path}.{key}")
            findings.extend(_find_forbidden(child, path=f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        normalized = value.lower().replace("\\", "/")
        string_tokens = _FORBIDDEN_TOKENS - {"truth"}
        explicit_truth_path = any(
            marker in normalized
            for marker in (
                "/truth/",
                "/truth_frozen/",
                "reference_truth",
                "corpus_v3_truth",
            )
        )
        if explicit_truth_path or any(token in normalized for token in string_tokens):
            findings.append(path)
    return findings


def residual_agent_task_contract() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": AGENT_TASK_SCHEMA,
        "type": "object",
        "required": sorted(_TASK_FIELDS),
        "additionalProperties": False,
        "properties": {
            "schema": {"const": AGENT_TASK_SCHEMA},
            "task_id": {"type": "string", "minLength": 1},
            "document_id": {"type": "string", "minLength": 1},
            "page_id": {"type": "string", "minLength": 1},
            "risk_family": {"type": "string", "minLength": 1},
            "target_nodes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {"type": "object"},
            },
            "source_visuals": {"type": "array", "minItems": 1},
            "primary_evidence": {"type": "array"},
            "secondary_evidence": {"type": "array"},
            "risk_reasons": {"type": "array"},
            "validator_state": {"type": "array"},
            "typed_authority_policy_refs": {"type": "array"},
            "bounded_context": {"type": "object"},
            "allowed_judgments": {"type": "array", "minItems": 1},
            "allowed_patch_ops": {"type": "array"},
            "source_fidelity_rules": {"type": "object"},
        },
    }


def residual_agent_result_contract() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": AGENT_RESULT_SCHEMA,
        "type": "object",
        "required": sorted(_RESULT_FIELDS),
        "additionalProperties": False,
        "properties": {
            "schema": {"const": AGENT_RESULT_SCHEMA},
            "task_id": {"type": "string", "minLength": 1},
            "targets": {"type": "array", "minItems": 1, "maxItems": 8},
            "executor_metadata": {"type": "object"},
        },
    }


def build_agent_tasks(
    nodes: Iterable[Mapping[str, Any]],
    *,
    clean_pages: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group residual nodes by source page, retaining a hard 8-node ceiling."""

    residual = [
        copy.deepcopy(dict(node))
        for node in nodes
        if str(node.get("residual_risk")) in {"HIGH", "CRITICAL"}
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in residual:
        grouped[(str(node["document_id"]), str(node["page_id"]))].append(node)

    tasks: list[dict[str, Any]] = []
    for (document_id, page_id), page_nodes in sorted(grouped.items()):
        ordered = sorted(page_nodes, key=lambda row: str(row["node_id"]))
        for offset in range(0, len(ordered), 8):
            chunk = ordered[offset : offset + 8]
            families = {_risk_family(node) for node in chunk}
            context_level, escalation_reason = _context_level(families)
            visuals = []
            target_nodes = []
            for node in chunk:
                crop = Path(str(node["source_crop"]))
                if not crop.is_file():
                    raise FileNotFoundError(crop)
                crop_sha = sha256_file(crop)
                family = _risk_family(node)
                evidence_ref = f"crop:{node['node_id']}:{crop_sha[:16]}"
                visuals.append(
                    {
                        "node_id": str(node["node_id"]),
                        "visual_kind": "EXACT_SOURCE_CROP",
                        "path": str(crop.resolve()),
                        "sha256": crop_sha,
                        "source_bbox": list(node["source_bbox"]),
                        "evidence_ref": evidence_ref,
                    }
                )
                target_nodes.append(
                    {
                        "node_id": str(node["node_id"]),
                        "document_id": document_id,
                        "page_id": page_id,
                        "risk_family": family,
                        "source_bbox": list(node["source_bbox"]),
                        "primary_evidence": copy.deepcopy(
                            node.get("primary_evidence")
                        ),
                        "secondary_evidence": copy.deepcopy(
                            node.get("secondary_evidence")
                        ),
                    }
                )
            if families & STRUCTURE_FAMILIES:
                page = clean_pages.get(page_id)
                if page is None:
                    raise ValueError(f"STRUCTURAL_TASK_CLEAN_PAGE_MISSING:{page_id}")
                page_path = Path(str(page["path"]))
                if not page_path.is_file():
                    raise FileNotFoundError(page_path)
                page_sha = sha256_file(page_path)
                visuals.append(
                    {
                        "node_id": None,
                        "visual_kind": "CLEAN_PAGE_STRUCTURE",
                        "path": str(page_path.resolve()),
                        "sha256": page_sha,
                        "source_bbox": None,
                        "evidence_ref": f"page:{page_id}:{page_sha[:16]}",
                    }
                )

            identity = {
                "document_id": document_id,
                "page_id": page_id,
                "node_ids": [row["node_id"] for row in target_nodes],
            }
            task_id = f"phase7c-task-{semantic_sha256(identity)[:16]}"
            all_allowed = set(CONTENT_JUDGMENTS)
            if families & STRUCTURE_FAMILIES:
                all_allowed.update(STRUCTURE_JUDGMENTS)
            task = {
                "schema": AGENT_TASK_SCHEMA,
                "task_id": task_id,
                "document_id": document_id,
                "page_id": page_id,
                "risk_family": (
                    next(iter(families)) if len(families) == 1 else "MULTI_FAMILY"
                ),
                "target_nodes": target_nodes,
                "source_visuals": visuals,
                "primary_evidence": [
                    {
                        "node_id": str(node["node_id"]),
                        "value": copy.deepcopy(node.get("primary_evidence")),
                    }
                    for node in chunk
                ],
                "secondary_evidence": [
                    {
                        "node_id": str(node["node_id"]),
                        "value": copy.deepcopy(node.get("secondary_evidence")),
                    }
                    for node in chunk
                ],
                "risk_reasons": [
                    {
                        "node_id": str(node["node_id"]),
                        "reasons": list(node.get("residual_reasons") or []),
                    }
                    for node in chunk
                ],
                "validator_state": [
                    {
                        "node_id": str(node["node_id"]),
                        "resolver_state": str(node.get("resolver_state") or "UNAVAILABLE"),
                    }
                    for node in chunk
                ],
                "typed_authority_policy_refs": [
                    {
                        "node_id": str(node["node_id"]),
                        "risk_family": _risk_family(node),
                        "calibration_state": (
                            FORMULA_AUTHORITY_STATE
                            if _risk_family(node) == "FORMULA"
                            else "TYPED_AUTHORITY_POLICY_V1"
                        ),
                        "global_authority_proven": False,
                    }
                    for node in chunk
                ],
                "bounded_context": {
                    "context_level": context_level,
                    "context_escalation_reason": escalation_reason,
                    "related_node_ids": [row["node_id"] for row in target_nodes],
                    "instruction": (
                        "Judge only the listed targets and specified structural relations; "
                        "do not perform a whole-page content audit."
                    ),
                },
                "allowed_judgments": sorted(all_allowed),
                "allowed_patch_ops": sorted(
                    CONTENT_OPS
                    | (STRUCTURE_OPS if families & STRUCTURE_FAMILIES else set())
                ),
                "source_fidelity_rules": {
                    "semantic_guess_only_forbidden": True,
                    "preserve_source_anomalies": True,
                    "required_self_check": "SOURCE_MATCH_CONFIRMED",
                    "whole_page_markdown_rewrite_forbidden": True,
                },
            }
            validate_agent_task(task)
            tasks.append(task)

    observed = [node_id for task in tasks for node_id in _target_ids(task)]
    expected = [str(node["node_id"]) for node in residual]
    if sorted(observed) != sorted(expected) or len(observed) != len(set(observed)):
        raise RuntimeError("AGENT_TASK_NODE_ACCOUNTING_INVALID")
    return tasks


def _target_ids(task: Mapping[str, Any]) -> list[str]:
    return [str(row["node_id"]) for row in task.get("target_nodes", [])]


def validate_agent_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if set(task) != _TASK_FIELDS:
        raise ValueError("AGENT_TASK_TOP_LEVEL_FIELDS_INVALID")
    if task.get("schema") != AGENT_TASK_SCHEMA:
        raise ValueError("AGENT_TASK_SCHEMA_INVALID")
    targets = list(task.get("target_nodes") or [])
    node_ids = _target_ids(task)
    if not 1 <= len(targets) <= 8 or len(node_ids) != len(set(node_ids)):
        raise ValueError("AGENT_TASK_TARGET_CARDINALITY_INVALID")
    if any(str(row.get("page_id")) != str(task["page_id"]) for row in targets):
        raise ValueError("AGENT_TASK_CROSS_PAGE_TARGET")
    if _find_forbidden(task):
        raise ValueError(f"AGENT_TASK_FORBIDDEN_CONTENT:{_find_forbidden(task)}")
    visuals = list(task.get("source_visuals") or [])
    if not visuals:
        raise ValueError("AGENT_TASK_SOURCE_VISUALS_MISSING")
    for visual in visuals:
        path = Path(str(visual.get("path") or ""))
        if not path.is_file() or sha256_file(path) != visual.get("sha256"):
            raise ValueError("AGENT_TASK_SOURCE_VISUAL_INTEGRITY_INVALID")
    crop_nodes = {
        str(row["node_id"])
        for row in visuals
        if row.get("visual_kind") == "EXACT_SOURCE_CROP"
    }
    if crop_nodes != set(node_ids):
        raise ValueError("AGENT_TASK_EXACT_CROP_COVERAGE_INVALID")
    level = str(task["bounded_context"].get("context_level"))
    if level not in {
        "L0_EXACT_CROP",
        "L1_LOCAL_CONTEXT",
        "L2_STRUCTURAL_REGION",
        "L3_FULL_PAGE_STRUCTURE",
    }:
        raise ValueError("AGENT_TASK_CONTEXT_LEVEL_INVALID")
    if level == "L3_FULL_PAGE_STRUCTURE" and not any(
        row.get("visual_kind") == "CLEAN_PAGE_STRUCTURE" for row in visuals
    ):
        raise ValueError("AGENT_TASK_L3_CLEAN_PAGE_MISSING")
    return {
        "schema": "bemarkdown-residual-risk-task-validation-v1",
        "passed": True,
        "target_count": len(node_ids),
        "context_level": level,
        "provider_api_binding_count": 0,
        "forbidden_content_count": 0,
    }


def validate_agent_result(
    task: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    validate_agent_task(task)
    if set(result) != _RESULT_FIELDS or result.get("schema") != AGENT_RESULT_SCHEMA:
        raise ValueError("AGENT_RESULT_TOP_LEVEL_FIELDS_INVALID")
    if str(result.get("task_id")) != str(task["task_id"]):
        raise ValueError("AGENT_RESULT_TASK_ID_MISMATCH")
    targets = list(result.get("targets") or [])
    ids = [str(row.get("node_id")) for row in targets]
    if ids != _target_ids(task) or len(ids) != len(set(ids)):
        raise ValueError("AGENT_RESULT_TARGET_CARDINALITY")
    allowed = set(task["allowed_judgments"])
    for row in targets:
        if set(row) != _RESULT_TARGET_FIELDS:
            raise ValueError("AGENT_RESULT_TARGET_FIELDS_INVALID")
        if row["judgment"] not in ALL_JUDGMENTS or row["judgment"] not in allowed:
            raise ValueError("AGENT_RESULT_JUDGMENT_INVALID")
        if str(row["issue_type"]) not in KNOWN_FAMILIES:
            raise ValueError("AGENT_RESULT_ISSUE_TYPE_INVALID")
        if row.get("patch") is not None and not isinstance(row["patch"], Mapping):
            raise ValueError("AGENT_RESULT_PATCH_INVALID")
    metadata = result.get("executor_metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("AGENT_RESULT_EXECUTOR_METADATA_INVALID")
    if int(metadata.get("content_retry_count", -1)) != 0:
        raise ValueError("AGENT_RESULT_CONTENT_RETRY_FORBIDDEN")
    if int(metadata.get("technical_retry_count", -1)) not in {0, 1}:
        raise ValueError("AGENT_RESULT_TECHNICAL_RETRY_POLICY_EXCEEDED")
    return {
        "schema": "bemarkdown-residual-risk-result-validation-v1",
        "passed": True,
        "target_count": len(targets),
        "technical_retry_count": int(metadata["technical_retry_count"]),
        "content_retry_count": 0,
    }


def audit_truth_blindness(
    tasks: Sequence[Mapping[str, Any]],
    *,
    executor_request: Mapping[str, Any],
    semantic_rows_read_before_result_freeze: int,
) -> dict[str, Any]:
    task_findings = [finding for task in tasks for finding in _find_forbidden(task)]
    request_findings = _find_forbidden(executor_request)
    passed = not task_findings and not request_findings and semantic_rows_read_before_result_freeze == 0
    return {
        "schema": "bemarkdown-agent-blindness-audit-v1",
        "truth_tokens_in_task_payload": len(task_findings),
        "truth_paths_in_executor_request": len(request_findings),
        "truth_semantic_rows_read_before_output_freeze": int(
            semantic_rows_read_before_result_freeze
        ),
        "passed": passed,
    }


def map_judgment_to_action(
    target_node: Mapping[str, Any], target_result: Mapping[str, Any]
) -> dict[str, Any]:
    node_id = str(target_node["node_id"])
    judgment = str(target_result["judgment"])
    if judgment in {"SOURCE_MATCHES_PRIMARY", "STRUCTURE_PRIMARY_VALID"}:
        return {"node_id": node_id, "action": "NO_CHANGE"}
    if judgment in {"INSUFFICIENT_VISUAL_EVIDENCE", "STRUCTURE_UNRESOLVED"}:
        return {"node_id": node_id, "action": "UNRESOLVED", "reason": judgment}
    family = str(target_result.get("issue_type") or target_node.get("risk_family"))
    if judgment == "SOURCE_MATCHES_SECONDARY":
        secondary = target_node.get("secondary_evidence")
        value = secondary.get("content") if isinstance(secondary, Mapping) else None
        if value in (None, "", []):
            return {
                "node_id": node_id,
                "action": "UNRESOLVED",
                "reason": "SECONDARY_EVIDENCE_UNAVAILABLE",
            }
        if family in {"TEXT", "HEADING"} and isinstance(value, list):
            value = _minimal_text_secondary_segments(value)
        patch = _patch_payload(
            target_node,
            target_result,
            op=_op_for_family(family),
            replacement=copy.deepcopy(value),
        )
        return {"node_id": node_id, "action": "PATCH", "patch": patch}
    if judgment == "SOURCE_MATCHES_NEITHER":
        replacement = target_result.get("replacement")
        grounded = (
            replacement not in (None, "", [])
            and bool(target_result.get("source_evidence_refs"))
            and target_result.get("self_check") == "SOURCE_MATCH_CONFIRMED"
            and target_result.get("correction_basis") in CORRECTION_BASES
        )
        if not grounded:
            return {
                "node_id": node_id,
                "action": "UNRESOLVED",
                "reason": "SOURCE_GROUNDED_REPLACEMENT_REQUIRED",
            }
        supplied = target_result.get("patch")
        if isinstance(supplied, Mapping):
            patch = {**copy.deepcopy(dict(supplied))}
            patch.setdefault("target_node_id", node_id)
            patch.setdefault("document_id", target_node.get("document_id"))
            patch.setdefault("issue_type", family)
            patch.setdefault(
                "correction_basis", target_result.get("correction_basis")
            )
            patch.setdefault(
                "source_evidence_refs",
                list(target_result.get("source_evidence_refs") or []),
            )
            patch.setdefault("self_check", target_result.get("self_check"))
            patch.setdefault("replacement", copy.deepcopy(replacement))
            patch.setdefault("affected_node_ids", [node_id])
            patch.setdefault("dependency_scope", [node_id])
            return {"node_id": node_id, "action": "PATCH", "patch": patch}
        patch = _patch_payload(
            target_node,
            target_result,
            op=_op_for_family(family),
            replacement=copy.deepcopy(replacement),
        )
        return {"node_id": node_id, "action": "PATCH", "patch": patch}
    if judgment == "STRUCTURE_PATCH_REQUIRED":
        supplied = target_result.get("patch")
        if not isinstance(supplied, Mapping):
            return {
                "node_id": node_id,
                "action": "UNRESOLVED",
                "reason": "BOUNDED_STRUCTURE_PATCH_REQUIRED",
            }
        patch = {**copy.deepcopy(dict(supplied))}
        patch.setdefault("target_node_id", node_id)
        patch.setdefault("document_id", target_node.get("document_id"))
        patch.setdefault("issue_type", family)
        patch.setdefault("correction_basis", "STRUCTURE_VISUAL")
        patch.setdefault("source_evidence_refs", list(target_result.get("source_evidence_refs") or []))
        patch.setdefault("self_check", target_result.get("self_check"))
        patch.setdefault("affected_node_ids", [node_id])
        patch.setdefault("dependency_scope", [node_id])
        return {"node_id": node_id, "action": "PATCH", "patch": patch}
    raise ValueError(f"UNKNOWN_AGENT_JUDGMENT:{judgment}")


def _minimal_text_secondary_segments(values: Sequence[Any]) -> str:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        normalized = "".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append((text, normalized))
    selected = [
        text
        for index, (text, normalized) in enumerate(candidates)
        if not any(
            index != other_index
            and len(other_normalized) >= 8
            and other_normalized != normalized
            and other_normalized in normalized
            for other_index, (_, other_normalized) in enumerate(candidates)
        )
    ]
    return "\n".join(selected)


def _patch_payload(
    target_node: Mapping[str, Any],
    target_result: Mapping[str, Any],
    *,
    op: str,
    replacement: Any,
) -> dict[str, Any]:
    node_id = str(target_node["node_id"])
    return {
        "op": op,
        "target_node_id": node_id,
        "document_id": str(target_node["document_id"]),
        "issue_type": str(target_result.get("issue_type") or target_node.get("risk_family")),
        "correction_basis": str(target_result.get("correction_basis") or "VISUAL_DIRECT"),
        "source_evidence_refs": list(target_result.get("source_evidence_refs") or []),
        "self_check": str(target_result.get("self_check") or ""),
        "replacement": replacement,
        "affected_node_ids": [node_id],
        "dependency_scope": [node_id],
    }


class SourceFidelityGuard:
    """Fail-closed validation before any stable-ID transaction begins."""

    def __init__(self, nodes: Mapping[str, Mapping[str, Any]]):
        self.nodes = nodes

    def validate(self, patch: Mapping[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        node_id = str(patch.get("target_node_id") or "")
        target = self.nodes.get(node_id)
        if target is None:
            reasons.append("STABLE_TARGET_MISSING")
        if patch.get("op") not in ALLOWED_OPS:
            reasons.append("PATCH_OPERATION_NOT_ALLOWED")
        if patch.get("issue_type") not in KNOWN_FAMILIES:
            reasons.append("ISSUE_TYPE_UNKNOWN")
        if patch.get("correction_basis") not in CORRECTION_BASES:
            reasons.append("CORRECTION_BASIS_INVALID")
        if not patch.get("source_evidence_refs"):
            reasons.append("SOURCE_EVIDENCE_INCOMPLETE")
        if patch.get("self_check") != "SOURCE_MATCH_CONFIRMED":
            reasons.append("SELF_CHECK_INCOMPLETE")
        affected = [str(value) for value in patch.get("affected_node_ids") or []]
        scope = [str(value) for value in patch.get("dependency_scope") or []]
        if node_id not in affected or not scope:
            reasons.append("PATCH_SCOPE_INCOMPLETE")
        if any(value not in self.nodes for value in set(affected + scope)):
            reasons.append("PATCH_SCOPE_TARGET_MISSING")
        if target is not None and str(patch.get("document_id")) != str(
            target.get("document_id")
        ):
            reasons.append("CROSS_DOCUMENT_TARGET")
        if target is not None and any(
            str(self.nodes[value].get("document_id")) != str(target.get("document_id"))
            for value in set(affected + scope)
            if value in self.nodes
        ):
            reasons.append("CROSS_DOCUMENT_SCOPE")
        if patch.get("replacement") in (None, "", []) and patch.get("op") in CONTENT_OPS:
            reasons.append("HIDDEN_SEMANTIC_REWRITE_OR_EMPTY_REPLACEMENT")
        return {
            "schema": "bemarkdown-source-fidelity-guard-decision-v1",
            "target_node_id": node_id,
            "passed": not reasons,
            "reasons": sorted(set(reasons)),
        }


def _balanced_latex(value: str) -> bool:
    depth = 0
    escaped = False
    for char in value:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and "\\begin" not in value or (
        depth == 0 and value.count("\\begin{") == value.count("\\end{")
    )


def _machine_validate(nodes: Mapping[str, Mapping[str, Any]], patch: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    node_id = str(patch["target_node_id"])
    if node_id not in nodes:
        reasons.append("TARGET_MISSING_AFTER_TRANSACTION")
    if len(nodes) != len(set(nodes)):
        reasons.append("DUPLICATE_NODE_ID")
    op = str(patch["op"])
    value = patch.get("replacement")
    if op in {"REPLACE_TEXT", "RELABEL_HEADING"}:
        if not isinstance(value, str) or not value.strip():
            reasons.append("TEXT_EMPTY_OR_ENCODING_INVALID")
        else:
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                reasons.append("TEXT_EMPTY_OR_ENCODING_INVALID")
    if op == "REPLACE_FORMULA":
        formula_values = value if isinstance(value, list) else [value]
        if not formula_values or any(
            not isinstance(item, str)
            or not item.strip()
            or not _balanced_latex(item)
            for item in formula_values
        ):
            reasons.append("FORMULA_SYNTAX_OR_RENDERABILITY_INVALID")
    if op in {"REPLACE_TABLE_CELL", "REPLACE_TABLE_STRUCTURE"} and value in (
        None,
        "",
        [],
        {},
    ):
        reasons.append("TABLE_PARSE_OR_POSITION_INVALID")
    if op in STRUCTURE_OPS:
        affected = patch.get("affected_node_ids") or []
        scope = patch.get("dependency_scope") or []
        if not affected or not scope:
            reasons.append("STRUCTURE_ACCOUNTING_OR_GRAPH_INVALID")
    return {
        "schema": "bemarkdown-residual-patch-machine-validation-v1",
        "passed": not reasons,
        "reasons": reasons,
    }


def apply_patch_transaction(
    nodes: Mapping[str, Mapping[str, Any]], patch: Mapping[str, Any]
) -> dict[str, Any]:
    original = copy.deepcopy(dict(nodes))
    guard = SourceFidelityGuard(original).validate(patch)
    if not guard["passed"]:
        return {
            "status": "GUARD_REJECTED",
            "rolled_back": True,
            "nodes": original,
            "guard": guard,
            "machine_validation": None,
        }
    updated = copy.deepcopy(original)
    node_id = str(patch["target_node_id"])
    op = str(patch["op"])
    if op == "CHANGE_BLOCK_TYPE":
        replacement = patch.get("replacement")
        if isinstance(replacement, Mapping):
            updated[node_id]["content_type"] = str(replacement["content_type"])
            if "content" in replacement:
                evidence = copy.deepcopy(
                    updated[node_id].get("primary_evidence") or {}
                )
                evidence["content"] = copy.deepcopy(replacement["content"])
                evidence["correction_basis"] = patch["correction_basis"]
                updated[node_id]["primary_evidence"] = evidence
        else:
            updated[node_id]["content_type"] = str(replacement)
    elif op in CONTENT_OPS | {"RELABEL_HEADING"}:
        evidence = copy.deepcopy(updated[node_id].get("primary_evidence") or {})
        evidence["content"] = copy.deepcopy(patch.get("replacement"))
        evidence["correction_basis"] = patch["correction_basis"]
        updated[node_id]["primary_evidence"] = evidence
        if op == "RELABEL_HEADING":
            updated[node_id]["content_type"] = "HEADING"
    else:
        updated[node_id]["structure_patch"] = copy.deepcopy(dict(patch))
    machine = _machine_validate(updated, patch)
    if not machine["passed"]:
        return {
            "status": "MACHINE_REJECTED",
            "rolled_back": True,
            "nodes": original,
            "guard": guard,
            "machine_validation": machine,
        }
    return {
        "status": "PATCH_APPLIED",
        "rolled_back": False,
        "nodes": updated,
        "guard": guard,
        "machine_validation": machine,
    }
