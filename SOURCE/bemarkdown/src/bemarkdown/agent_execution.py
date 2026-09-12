"""Vendor-neutral AgentTask/AgentExecutor/AgentResult seam for Output Audit v1."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .output_audit_production import (
    PRODUCTION_RESULT_SCHEMA,
    ProductionAuditResultImporter,
    production_audit_result_schema,
)
from .production_runtime import canonical_json

AGENT_TASK_SCHEMA = "bemarkdown-agent-audit-task-v1"
AGENT_EXECUTION_RECORD_SCHEMA = "bemarkdown-agent-execution-record-v1"
AGENT_EXECUTOR_ID = "codex-agent"
_TASK_FIELDS = {
    "schema",
    "task_id",
    "document_id",
    "page_ids",
    "protocol_version",
    "result_schema_version",
    "source_clean_images",
    "source_overlay_images",
    "current_page_markdown",
    "audit_node_views",
    "bounded_context",
    "formula_evidence_refs",
    "table_evidence_refs",
    "expected_result_path",
}
_FORBIDDEN_TASK_TOKENS = {
    "truth",
    "reference_truth",
    "corpus_v3_truth",
    "truth_frozen",
    "expected_patch",
    "known_error",
    "answer_key",
    "raw_paddle",
    "benchmark_score",
    "api_key",
    "credential",
    "provider_binding",
}
_VENDOR_IMPORTS = {
    "openai",
    "anthropic",
    "google.generativeai",
    "google.genai",
    "gemini",
}
_REQUIRED_LOCK_FIELDS = {
    "developer_commit_sha",
    "runtime_fingerprint",
    "portable_fp32_profile_sha",
    "layout_authority_v1_fingerprint",
    "output_audit_production_v1_sha",
    "agent_task_schema_sha",
    "agent_result_schema_sha",
    "codex_agent_prompt_sha",
    "agent_executor",
    "corpus_fingerprint",
    "eval_instance_sha",
    "truth_fingerprint",
    "metrics_threshold_fingerprint",
}


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def agent_task_contract() -> dict[str, Any]:
    ref = {
        "type": "object",
        "required": ["audit_page_id", "path", "sha256"],
        "properties": {
            "audit_page_id": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "additionalProperties": False,
    }
    page_value = {
        "type": "object",
        "required": ["audit_page_id"],
        "properties": {"audit_page_id": {"type": "string", "minLength": 1}},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": AGENT_TASK_SCHEMA,
        "type": "object",
        "required": sorted(_TASK_FIELDS),
        "properties": {
            "schema": {"const": AGENT_TASK_SCHEMA},
            "task_id": {"type": "string", "minLength": 1},
            "document_id": {"type": "string", "minLength": 1},
            "page_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "protocol_version": {"const": "bemarkdown-output-audit-production-policy-v1"},
            "result_schema_version": {"const": PRODUCTION_RESULT_SCHEMA},
            "source_clean_images": {"type": "array", "items": ref},
            "source_overlay_images": {"type": "array", "items": ref},
            "current_page_markdown": {"type": "array", "items": page_value},
            "audit_node_views": {"type": "array", "items": page_value},
            "bounded_context": {"type": "array", "items": page_value},
            "formula_evidence_refs": {"type": "array"},
            "table_evidence_refs": {"type": "array"},
            "expected_result_path": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
        "truth_bearing_fields_included": False,
        "provider_api_binding_included": False,
    }


def agent_result_contract() -> dict[str, Any]:
    """Return the single production wire contract; no executor-specific schema."""

    return production_audit_result_schema()


def _page_entry(page_id: str, path: Path, *, content_key: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "audit_page_id": page_id,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }
    if content_key is not None:
        if path.suffix == ".json":
            value[content_key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            value[content_key] = path.read_text(encoding="utf-8")
    return value


def build_agent_task(
    *,
    task_id: str,
    document_id: str,
    page_packages: Sequence[Mapping[str, Any]],
    package_roots: Mapping[str, str | Path],
    expected_result_path: str | Path,
) -> dict[str, Any]:
    if not 1 <= len(page_packages) <= 6:
        raise ValueError("AGENT_TASK_PAGE_COUNT_MUST_BE_1_TO_6")
    page_ids = [str(row["audit_page_id"]) for row in page_packages]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("AGENT_TASK_PAGE_IDS_MUST_BE_UNIQUE")
    if {str(row["document_id"]) for row in page_packages} != {str(document_id)}:
        raise ValueError("AGENT_TASK_DOCUMENT_ID_MISMATCH")

    clean: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    markdown: list[dict[str, Any]] = []
    node_views: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    formula_refs: list[dict[str, Any]] = []
    table_refs: list[dict[str, Any]] = []
    for package in page_packages:
        page_id = str(package["audit_page_id"])
        root = Path(package_roots[page_id]).resolve()
        files = package["files"]
        clean.append(_page_entry(page_id, root / files["clean_source"]))
        overlays.append(_page_entry(page_id, root / files["node_overlay"]))
        markdown.append(
            _page_entry(
                page_id,
                root / files["current_page_markdown"],
                content_key="markdown",
            )
        )
        node_views.append(
            _page_entry(page_id, root / files["page_nodes"], content_key="nodes")
        )
        contexts.append(
            _page_entry(page_id, root / files["context"], content_key="context")
        )
        for evidence in files.get("evidence", []):
            path = root / evidence
            value = _page_entry(page_id, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            value["stable_node_id"] = str(payload.get("stable_node_id") or "")
            if "formula" in str(payload.get("schema", "")):
                formula_refs.append(value)
            elif "table" in str(payload.get("schema", "")):
                table_refs.append(value)

    task = {
        "schema": AGENT_TASK_SCHEMA,
        "task_id": str(task_id),
        "document_id": str(document_id),
        "page_ids": page_ids,
        "protocol_version": "bemarkdown-output-audit-production-policy-v1",
        "result_schema_version": PRODUCTION_RESULT_SCHEMA,
        "source_clean_images": clean,
        "source_overlay_images": overlays,
        "current_page_markdown": markdown,
        "audit_node_views": node_views,
        "bounded_context": contexts,
        "formula_evidence_refs": formula_refs,
        "table_evidence_refs": table_refs,
        "expected_result_path": str(Path(expected_result_path).resolve()),
    }
    validate_agent_task(task)
    return task


def _find_forbidden(value: Any, *, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in _FORBIDDEN_TASK_TOKENS):
                findings.append(f"{path}.{key}")
            findings.extend(_find_forbidden(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower().replace("\\", "/")
        if any(token in lowered for token in _FORBIDDEN_TASK_TOKENS):
            findings.append(path)
    return findings


def validate_agent_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if set(task) != _TASK_FIELDS:
        raise ValueError("AGENT_TASK_TOP_LEVEL_FIELDS_INVALID")
    if task.get("schema") != AGENT_TASK_SCHEMA:
        raise ValueError("AGENT_TASK_SCHEMA_INVALID")
    page_ids = [str(value) for value in task.get("page_ids", [])]
    if not 1 <= len(page_ids) <= 6 or len(page_ids) != len(set(page_ids)):
        raise ValueError("AGENT_TASK_PAGE_IDS_INVALID")
    if task.get("result_schema_version") != PRODUCTION_RESULT_SCHEMA:
        raise ValueError("AGENT_TASK_RESULT_SCHEMA_INVALID")
    if task.get("protocol_version") != "bemarkdown-output-audit-production-policy-v1":
        raise ValueError("AGENT_TASK_PROTOCOL_INVALID")
    findings = _find_forbidden(task)
    if findings:
        raise ValueError(f"AGENT_TASK_FORBIDDEN_CONTENT:{findings}")
    for field in (
        "source_clean_images",
        "source_overlay_images",
        "current_page_markdown",
        "audit_node_views",
        "bounded_context",
    ):
        rows = task.get(field)
        if not isinstance(rows, list) or [str(row.get("audit_page_id")) for row in rows] != page_ids:
            raise ValueError(f"AGENT_TASK_{field.upper()}_COVERAGE_INVALID")
    for field in ("source_clean_images", "source_overlay_images"):
        for row in task[field]:
            path = Path(str(row["path"]))
            if not path.is_file() or sha256_file(path) != row.get("sha256"):
                raise ValueError(f"AGENT_TASK_{field.upper()}_INTEGRITY_INVALID")
    allowed_nodes = {
        str(node["node_id"])
        for row in task["audit_node_views"]
        for node in row.get("nodes", [])
    }
    evidence_nodes = {
        str(row.get("stable_node_id") or "")
        for row in [*task["formula_evidence_refs"], *task["table_evidence_refs"]]
    } - {""}
    if evidence_nodes - allowed_nodes:
        raise ValueError("AGENT_TASK_EVIDENCE_NODE_REFERENCE_INVALID")
    return {
        "schema": "bemarkdown-agent-task-validation-v1",
        "passed": True,
        "page_count": len(page_ids),
        "truth_bearing_field_count": 0,
        "provider_api_binding_count": 0,
        "stable_node_count": len(allowed_nodes),
    }


@runtime_checkable
class AgentExecutor(Protocol):
    """Upper-layer capability: execute one bounded AgentTask and return rows."""

    executor_id: str

    def execute(self, task: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class AgentResult:
    rows: list[dict[str, Any]]
    validation: dict[str, Any]

    @classmethod
    def from_rows(
        cls, task: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
    ) -> AgentResult:
        validate_agent_task(task)
        copied = [copy.deepcopy(dict(row)) for row in rows]
        packages = [
            {
                "audit_page_id": page_id,
                "node_ids": [str(node["node_id"]) for node in node_row["nodes"]],
            }
            for page_id, node_row in zip(task["page_ids"], task["audit_node_views"], strict=True)
        ]
        ProductionAuditResultImporter().parse(copied, packages)
        counts = Counter(str(row["audit_page_id"]) for row in copied)
        expected = set(task["page_ids"])
        observed = set(counts)
        validation = {
            "schema": "bemarkdown-agent-result-validation-v1",
            "passed": True,
            "page_coverage": len(observed & expected) / len(expected),
            "validated_rows": len(copied),
            "errors": {
                "duplicate": sum(count - 1 for count in counts.values() if count > 1),
                "unknown": len(observed - expected),
                "missing": len(expected - observed),
                "node_reference": 0,
                "patch_operation": 0,
                "schema": 0,
            },
        }
        return cls(rows=copied, validation=validation)


def record_agent_execution(
    task: Mapping[str, Any],
    result: AgentResult,
    *,
    executor_id: str,
    execution_environment: Mapping[str, Any],
    technical_retry_count: int = 0,
) -> dict[str, Any]:
    if technical_retry_count not in {0, 1}:
        raise ValueError("TECHNICAL_RETRY_COUNT_MUST_BE_ZERO_OR_ONE")
    if not result.validation.get("passed"):
        raise ValueError("AGENT_RESULT_MUST_BE_VALIDATED")
    return {
        "schema": AGENT_EXECUTION_RECORD_SCHEMA,
        "task_id": str(task["task_id"]),
        "task_fingerprint": semantic_sha256(task),
        "executor_id": str(executor_id),
        "execution_environment": copy.deepcopy(dict(execution_environment)),
        "result_schema": PRODUCTION_RESULT_SCHEMA,
        "result_fingerprint": semantic_sha256(result.rows),
        "validated_page_count": len(result.rows),
        "technical_retry_count": technical_retry_count,
        "content_retry_count": 0,
        "status": "VALIDATED",
    }


def provider_api_dependency_audit(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source_root = root / "src" / "bemarkdown"
    dependencies: list[dict[str, str]] = []
    for path in sorted(source_root.glob("*.py"), key=lambda value: value.name.casefold()):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            module = stripped.split()[1].split(",")[0]
            if any(module == vendor or module.startswith(vendor + ".") for vendor in _VENDOR_IMPORTS):
                dependencies.append(
                    {"path": str(path.relative_to(root)), "line": str(line_number), "module": module}
                )
    return {
        "schema": "bemarkdown-provider-api-dependency-audit-v1",
        "scope": "DEFAULT_BEMARKDOWN_RUNTIME_IMPORTS",
        "provider_api_dependency_count": len(dependencies),
        "dependencies": dependencies,
        "vendor_sdk_required": False,
        "api_key_required": False,
        "endpoint_url_required": False,
        "passed": not dependencies,
    }


def build_cold_run_lock(*, agent_executor: str, **fingerprints: str) -> dict[str, Any]:
    values = {**fingerprints, "agent_executor": agent_executor}
    missing = sorted(_REQUIRED_LOCK_FIELDS - set(values))
    unknown = sorted(set(values) - _REQUIRED_LOCK_FIELDS)
    if missing or unknown:
        raise ValueError(f"COLD_RUN_LOCK_FIELDS_INVALID:missing={missing}:unknown={unknown}")
    for key, value in values.items():
        if key == "agent_executor":
            if value != AGENT_EXECUTOR_ID:
                raise ValueError("COLD_RUN_AGENT_EXECUTOR_INVALID")
        elif key == "developer_commit_sha":
            if len(str(value)) not in {40, 64} or any(
                char not in "0123456789abcdef" for char in str(value)
            ):
                raise ValueError("COLD_RUN_GIT_COMMIT_INVALID")
        elif len(str(value)) != 64 or any(
            char not in "0123456789abcdef" for char in str(value)
        ):
            raise ValueError(f"COLD_RUN_LOCK_SHA256_INVALID:{key}")
    return {
        "schema": "bemarkdown-corpus-v3-cold-run-lock-v1",
        **values,
        "states": [
            "CORPUS_V3_VISION_AGENT_EXECUTOR_READY",
            "CORPUS_V3_COLD_RUN_LOCK_READY",
            "CORPUS_V3_MODEL_OUTPUT_NOT_STARTED",
            "FORMAL_MCP_PUBLICATION_BLOCKED",
        ],
        "post_lock_mutation_allowed": False,
        "truth_content_included": False,
    }


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    target = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"IMMUTABLE_ARTIFACT_MISMATCH:{target}")
        return hashlib.sha256(data).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return hashlib.sha256(data).hexdigest()
