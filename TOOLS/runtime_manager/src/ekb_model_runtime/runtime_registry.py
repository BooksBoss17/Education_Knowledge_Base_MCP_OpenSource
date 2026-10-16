from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ModelRuntimeError
from .path_security import require_runtime_relative, safe_relative_path

RUNTIME_REGISTRY_V1_SCHEMA = "ekb-runtime-registry-v1"
RUNTIME_KINDS = {"python_env", "native_binary"}


@dataclass(frozen=True, slots=True)
class RuntimeEntry:
    runtime_id: str
    runtime_family: str
    runtime_kind: str
    platforms: tuple[str, ...]
    env_relpath: str
    python_version: str | None
    frameworks: tuple[str, ...]
    framework_versions: dict[str, str]
    lock_ref: str
    launcher_ref: str
    cache_policy: dict[str, str]
    gpu_support: bool
    cpu_support: bool
    status: str
    lock_sha256: str | None = None


class RuntimeRegistryV1:
    def __init__(self, entries: list[RuntimeEntry]) -> None:
        ids = [entry.runtime_id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ModelRuntimeError("INVALID_RUNTIME_REGISTRY", "duplicate runtime_id")
        self.entries = {entry.runtime_id: entry for entry in entries}

    @classmethod
    def load(cls, path: str | Path) -> RuntimeRegistryV1:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelRuntimeError("INVALID_RUNTIME_REGISTRY", str(exc)) from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RuntimeRegistryV1:
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != RUNTIME_REGISTRY_V1_SCHEMA
        ):
            raise ModelRuntimeError(
                "UNKNOWN_RUNTIME_REGISTRY_SCHEMA",
                str(getattr(payload, "get", lambda *_: None)("schema")),
            )
        rows = payload.get("runtimes")
        if not isinstance(rows, list):
            raise ModelRuntimeError("INVALID_RUNTIME_REGISTRY", "runtimes must be list")
        return cls([_parse(row) for row in rows])

    def get(self, runtime_id: str) -> RuntimeEntry:
        try:
            return self.entries[runtime_id]
        except KeyError as exc:
            raise ModelRuntimeError("RUNTIME_NOT_REGISTERED", runtime_id) from exc


def _parse(row: Any) -> RuntimeEntry:
    required = (
        "runtime_id",
        "runtime_family",
        "runtime_kind",
        "platforms",
        "env_relpath",
        "python_version",
        "frameworks",
        "framework_versions",
        "lock_ref",
        "launcher_ref",
        "cache_policy",
        "gpu_support",
        "cpu_support",
        "status",
    )
    if not isinstance(row, dict) or any(key not in row for key in required):
        raise ModelRuntimeError("INVALID_RUNTIME_REGISTRY", "runtime row incomplete")
    if row["runtime_kind"] not in RUNTIME_KINDS:
        raise ModelRuntimeError("INVALID_RUNTIME_KIND", str(row["runtime_kind"]))
    require_runtime_relative(row["env_relpath"])
    require_runtime_relative(row["launcher_ref"])
    safe_relative_path(row["lock_ref"], code="UNSAFE_RUNTIME_PATH")
    if not isinstance(row["platforms"], list) or not row["platforms"]:
        raise ModelRuntimeError("INVALID_RUNTIME_REGISTRY", "platforms required")
    return RuntimeEntry(
        **{key: row[key] for key in required if key not in {"platforms", "frameworks"}},
        platforms=tuple(row["platforms"]),
        frameworks=tuple(row["frameworks"]),
        lock_sha256=row.get("lock_sha256"),
    )
