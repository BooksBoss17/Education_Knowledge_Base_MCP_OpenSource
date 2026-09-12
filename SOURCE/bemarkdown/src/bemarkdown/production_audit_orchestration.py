from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .production_runtime import canonical_json

CHECKPOINT_SCHEMA = "bemarkdown-production-runtime-checkpoint-v0"
AUDIT_PROTOCOL_FINGERPRINT = "bemarkdown-output-audit-production-policy-v1"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class AuditNodeView:
    node_id: str
    kind: str
    bbox: list[float] | None
    current_content: dict[str, Any]
    asset_ref: str | None
    review_state: dict[str, Any]

    @classmethod
    def from_block(cls, block: Mapping[str, Any]) -> AuditNodeView:
        content = block.get("content") if isinstance(block.get("content"), Mapping) else {}
        allowed_content = {
            key: copy.deepcopy(content[key])
            for key in ("text", "latex", "markdown", "html", "status", "serialization")
            if key in content
        }
        review = block.get("review") if isinstance(block.get("review"), Mapping) else {}
        review_state = {
            key: copy.deepcopy(review[key])
            for key in ("status", "reason_codes", "unresolved")
            if key in review
        }
        bbox = block.get("bbox_pdf_pt") or block.get("bbox")
        asset_ref = block.get("asset_uid") or content.get("asset_uid")
        return cls(
            node_id=str(block["node_id"]),
            kind=str(block["kind"]),
            bbox=list(bbox) if bbox is not None else None,
            current_content=allowed_content,
            asset_ref=str(asset_ref) if asset_ref is not None else None,
            review_state=review_state,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "bbox": copy.deepcopy(self.bbox),
            "current_content": copy.deepcopy(self.current_content),
            "asset_ref": self.asset_ref,
            "review_state": copy.deepcopy(self.review_state),
        }


def audit_cache_key(
    source_batch_fingerprint: str,
    document_ir_semantic_sha256: str,
    protocol_fingerprint: str,
    agent_executor_identity: str | None = None,
    *,
    provider_model_identity: str | None = None,
) -> str:
    """Bind cache identity to the executor; accept the old keyword read-only."""

    if agent_executor_identity and provider_model_identity:
        raise ValueError("AGENT_EXECUTOR_IDENTITY_CONFLICTS_WITH_LEGACY_PROVIDER_ALIAS")
    identity = agent_executor_identity or provider_model_identity
    fields = {
        "source_batch_fingerprint": source_batch_fingerprint,
        "document_ir_semantic_sha256": document_ir_semantic_sha256,
        "protocol_fingerprint": protocol_fingerprint,
        "agent_executor_identity": identity,
    }
    if not all(str(value).strip() for value in fields.values()):
        raise ValueError("AUDIT_CACHE_KEY_FIELD_REQUIRED")
    return hashlib.sha256(canonical_json(fields).encode("utf-8")).hexdigest()


class ProductionCheckpoint:
    """Disk-authoritative conversion/audit/handoff recovery state."""

    def __init__(self, path: str | Path, state: Mapping[str, Any]):
        self.path = Path(path).resolve()
        self._state = copy.deepcopy(dict(state))
        self._lock = threading.RLock()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        local_window_ids: Sequence[str],
        audit_batch_ids: Sequence[str],
    ) -> ProductionCheckpoint:
        if len(local_window_ids) != len(set(local_window_ids)):
            raise ValueError("LOCAL_WINDOW_IDS_MUST_BE_UNIQUE")
        if len(audit_batch_ids) != len(set(audit_batch_ids)):
            raise ValueError("AUDIT_BATCH_IDS_MUST_BE_UNIQUE")
        state = {
            "schema": CHECKPOINT_SCHEMA,
            "local_windows": {
                window_id: {"status": "PENDING"} for window_id in local_window_ids
            },
            "audit_batches": {
                batch_id: {
                    "status": "PENDING",
                    "technical_retry_count": 0,
                    "content_retry_count": 0,
                    "result_fingerprint": None,
                }
                for batch_id in audit_batch_ids
            },
            "patch_apply_state": "PENDING",
            "handoff_state": "PENDING",
            "cache_state": {},
        }
        checkpoint = cls(path, state)
        checkpoint.flush()
        return checkpoint

    @classmethod
    def load(cls, path: str | Path) -> ProductionCheckpoint:
        path = Path(path).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("PRODUCTION_CHECKPOINT_SCHEMA_INVALID")
        required = {
            "local_windows",
            "audit_batches",
            "patch_apply_state",
            "handoff_state",
            "cache_state",
        }
        if not required.issubset(value):
            raise ValueError("PRODUCTION_CHECKPOINT_INCOMPLETE")
        return cls(path, value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def flush(self) -> None:
        with self._lock:
            _atomic_write_json(self.path, self._state)

    def mark_local_window(self, window_id: str, status: str, **fields: Any) -> None:
        with self._lock:
            if window_id not in self._state["local_windows"]:
                raise KeyError(f"UNKNOWN_LOCAL_WINDOW:{window_id}")
            self._state["local_windows"][window_id] = {
                "status": status,
                **copy.deepcopy(fields),
            }
            self.flush()

    def mark_audit_batch(self, batch_id: str, status: str, **fields: Any) -> None:
        with self._lock:
            if batch_id not in self._state["audit_batches"]:
                raise KeyError(f"UNKNOWN_AUDIT_BATCH:{batch_id}")
            current = self._state["audit_batches"][batch_id]
            current["status"] = status
            current.update(copy.deepcopy(fields))
            self.flush()

    def set_patch_apply_state(self, status: str, **fields: Any) -> None:
        with self._lock:
            self._state["patch_apply_state"] = status
            if fields:
                self._state["patch_apply_details"] = copy.deepcopy(fields)
            self.flush()

    def set_handoff_state(self, status: str, **fields: Any) -> None:
        with self._lock:
            self._state["handoff_state"] = status
            if fields:
                self._state["handoff_details"] = copy.deepcopy(fields)
            self.flush()

    def set_cache_state(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self._state["cache_state"] = copy.deepcopy(dict(state))
            self.flush()


class AuditDispatcher:
    """Bounded dispatcher preserving Production v1 retry and identity rules."""

    def __init__(
        self,
        resource_class: str,
        *,
        concurrency: int | None = None,
        checkpoint: ProductionCheckpoint,
        gpu_scheduler: Any | None = None,
    ):
        if resource_class not in {"REMOTE_AGENT", "LOCAL_GPU_AGENT"}:
            raise ValueError("AUDIT_RESOURCE_CLASS_INVALID")
        default = 2 if resource_class == "REMOTE_AGENT" else 1
        if concurrency is None:
            concurrency = default
        if resource_class == "REMOTE_AGENT" and not 1 <= concurrency <= 4:
            raise ValueError("REMOTE_AGENT_CONCURRENCY_MUST_BE_1_TO_4")
        if resource_class == "LOCAL_GPU_AGENT" and concurrency != 1:
            raise ValueError("LOCAL_GPU_AGENT_REQUIRES_SINGLE_GPU_SCHEDULER")
        if resource_class == "LOCAL_GPU_AGENT" and gpu_scheduler is None:
            raise ValueError("LOCAL_GPU_AGENT_REQUIRES_GPU_RESOURCE_SCHEDULER")
        self.resource_class = resource_class
        self.concurrency = concurrency
        self.checkpoint = checkpoint
        self.gpu_scheduler = gpu_scheduler

    @staticmethod
    def _validate_batches(batches: Sequence[Mapping[str, Any]]) -> None:
        batch_ids = [str(batch.get("batch_id") or "") for batch in batches]
        if not all(batch_ids) or len(batch_ids) != len(set(batch_ids)):
            raise ValueError("AUDIT_BATCH_IDS_INVALID")
        page_ids: list[str] = []
        for batch in batches:
            if batch.get("identity_frozen") is not True:
                raise ValueError("AUDIT_PAGE_IDENTITY_NOT_FROZEN")
            pages = batch.get("pages")
            if not isinstance(pages, list) or not pages:
                raise ValueError("AUDIT_BATCH_PAGES_REQUIRED")
            if len(pages) > 6:
                raise ValueError("AUDIT_BATCH_PAGE_CEILING_EXCEEDED")
            for page in pages:
                audit_page_id = str(page.get("audit_page_id") or "")
                if not audit_page_id or not isinstance(page.get("node_ids"), list):
                    raise ValueError("AUDIT_PAGE_IDENTITY_INCOMPLETE")
                page_ids.append(audit_page_id)
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("DUPLICATE_PAGE_AUDIT")

    def dispatch(
        self,
        batches: Sequence[Mapping[str, Any]],
        executor: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        *,
        provider: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if executor is not None and provider is not None:
            raise ValueError("AGENT_EXECUTOR_CONFLICTS_WITH_LEGACY_PROVIDER_ALIAS")
        execute = executor or provider
        if execute is None:
            raise ValueError("AGENT_EXECUTOR_REQUIRED")
        batches = [copy.deepcopy(dict(batch)) for batch in batches]
        self._validate_batches(batches)
        checkpoint_batches = self.checkpoint.snapshot()["audit_batches"]
        unknown = sorted(str(batch["batch_id"]) for batch in batches if str(batch["batch_id"]) not in checkpoint_batches)
        if unknown:
            raise ValueError(f"AUDIT_BATCH_NOT_IN_CHECKPOINT:{unknown}")

        if self.resource_class == "LOCAL_GPU_AGENT":
            results = [self._dispatch_one(batch, execute) for batch in batches]
        else:
            by_id: dict[str, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._dispatch_one, batch, execute): str(batch["batch_id"])
                    for batch in batches
                }
                for future in as_completed(futures):
                    by_id[futures[future]] = future.result()
            results = [by_id[str(batch["batch_id"])] for batch in batches]
        return results

    def _executor_call(
        self,
        batch: dict[str, Any],
        executor: Callable[[dict[str, Any]], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if self.resource_class == "LOCAL_GPU_AGENT":
            return self.gpu_scheduler.run(
                "audit-agent",
                lambda: executor(copy.deepcopy(batch)),
            )
        return executor(copy.deepcopy(batch))

    def _dispatch_one(
        self,
        batch: dict[str, Any],
        executor: Callable[[dict[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any]:
        batch_id = str(batch["batch_id"])
        existing = self.checkpoint.snapshot()["audit_batches"][batch_id]
        if existing["status"] == "VALIDATED":
            return {
                "batch_id": batch_id,
                "status": "VALIDATED_FROM_CHECKPOINT",
                "attempts": 0,
                "technical_retry_count": existing.get("technical_retry_count", 0),
                "content_retry_count": 0,
                "result": None,
            }

        last_error: str | None = None
        for attempt in (1, 2):
            self.checkpoint.mark_audit_batch(
                batch_id,
                "DISPATCHING",
                technical_retry_count=attempt - 1,
                content_retry_count=0,
            )
            try:
                result = dict(self._executor_call(batch, executor))
                status = result.get("status")
                returned_batch_id = result.get("batch_id")
                if returned_batch_id is not None and str(returned_batch_id) != batch_id:
                    raise ValueError("AUDIT_EXECUTOR_BATCH_ID_MISMATCH")
                if status == "TECHNICAL_FAILURE":
                    raise RuntimeError("AUDIT_EXECUTOR_TECHNICAL_FAILURE")
                if status not in {"AUDITED", "AUDITED_WITH_UNRESOLVED"}:
                    raise ValueError("AUDIT_EXECUTOR_STATUS_INVALID")
            except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 1:
                    self.checkpoint.mark_audit_batch(
                        batch_id,
                        "RETRY_PENDING",
                        technical_retry_count=1,
                        content_retry_count=0,
                        last_error=last_error,
                    )
                    continue
                self.checkpoint.mark_audit_batch(
                    batch_id,
                    "TECHNICAL_FAILURE",
                    technical_retry_count=1,
                    content_retry_count=0,
                    last_error=last_error,
                )
                return {
                    "batch_id": batch_id,
                    "status": "TECHNICAL_FAILURE",
                    "attempts": 2,
                    "technical_retry_count": 1,
                    "content_retry_count": 0,
                    "error": last_error,
                    "result": None,
                }
            fingerprint = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
            self.checkpoint.mark_audit_batch(
                batch_id,
                "VALIDATED",
                technical_retry_count=attempt - 1,
                content_retry_count=0,
                result_fingerprint=fingerprint,
                terminal_status=status,
            )
            return {
                "batch_id": batch_id,
                "status": status,
                "attempts": attempt,
                "technical_retry_count": attempt - 1,
                "content_retry_count": 0,
                "result": result,
            }
        raise AssertionError(last_error)
