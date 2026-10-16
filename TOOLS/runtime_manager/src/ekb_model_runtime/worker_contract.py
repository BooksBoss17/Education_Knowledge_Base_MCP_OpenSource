from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeWorkerRequest:
    request_id: str
    model_id: str
    capability: str
    operation: str
    runtime_profile: str | None = None
    input_refs: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation not in {"load", "infer", "unload", "health", "stats"}:
            raise ValueError("RUNTIME_WORKER_OPERATION_INVALID")
        if not self.request_id or not self.model_id or not self.capability:
            raise ValueError("RUNTIME_WORKER_IDENTITY_REQUIRED")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = "ekb-runtime-worker-request-v1"
        return value


@dataclass(frozen=True, slots=True)
class RuntimeWorkerResponse:
    request_id: str
    model_id: str
    runtime_id: str
    status: str
    latency_ms: float
    device: str
    result: Any = None
    result_ref: str | None = None
    peak_vram_mib: int | None = None
    warnings: tuple[str, ...] = ()
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"OK", "ERROR"} or self.latency_ms < 0:
            raise ValueError("RUNTIME_WORKER_RESPONSE_INVALID")
        if self.status == "ERROR" and self.error is None:
            raise ValueError("RUNTIME_WORKER_ERROR_REQUIRED")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = "ekb-runtime-worker-response-v1"
        return value
