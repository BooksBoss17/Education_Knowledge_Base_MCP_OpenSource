from __future__ import annotations

import json
import platform as platform_module
import subprocess
from pathlib import Path
from typing import Any

from .errors import ModelRuntimeError
from .path_security import resolve_beneath
from .resolver import ModelRuntimeResolver


def run_doctor(
    models_root: str | Path,
    *,
    platform: str = "windows-x86_64",
    device: str = "cuda:0",
    deep: bool = False,
) -> dict[str, Any]:
    root = Path(models_root).resolve()
    result: dict[str, Any] = {
        "schema": "ekb-model-runtime-doctor-v1",
        "models_root": str(root),
        "offline_readiness": True,
        "gpu": _gpu_status(),
        "host_python": platform_module.python_version(),
        "models": {},
        "runtimes": {},
    }
    try:
        resolver = ModelRuntimeResolver.from_root(root)
        result["model_registry"] = {
            "status": "PASS",
            "schema": resolver.models.source_schema,
            "registered_model_count": len(resolver.models.entries),
        }
        result["runtime_registry"] = {
            "status": "PASS",
            "schema": "ekb-runtime-registry-v1",
            "runtime_count": len(resolver.runtimes.entries),
            "runtime_family_count": len(
                {r.runtime_family for r in resolver.runtimes.entries.values()}
            ),
        }
        for runtime_id, runtime in sorted(resolver.runtimes.entries.items()):
            env_path = resolve_beneath(
                root, runtime.env_relpath, code="UNSAFE_RUNTIME_PATH"
            )
            lock_path = resolve_beneath(
                root, runtime.lock_ref, code="UNSAFE_RUNTIME_PATH"
            )
            launcher_path = resolve_beneath(
                root, runtime.launcher_ref, code="UNSAFE_RUNTIME_PATH"
            )
            result["runtimes"][runtime_id] = {
                "runtime_family": runtime.runtime_family,
                "registry_status": runtime.status,
                "environment_present": env_path.exists(),
                "lock_present": lock_path.is_file(),
                "launcher_present": launcher_path.is_file(),
            }
        for model_id in sorted(resolver.models.entries):
            try:
                resolution = resolver.resolve(
                    model_id,
                    models_root=root,
                    platform=platform,
                    device=device,
                    verify_fingerprint=deep,
                )
                result["models"][model_id] = {
                    "status": "PASS",
                    "runtime_id": resolution.runtime_id,
                    "fingerprint_verified": deep,
                }
            except ModelRuntimeError as exc:
                result["models"][model_id] = {"status": "FAIL", "error": exc.code}
    except ModelRuntimeError as exc:
        result["model_registry"] = {"status": "FAIL", "error": exc.code}
    result["readiness"] = (
        "PASS"
        if result.get("model_registry", {}).get("status") == "PASS"
        and all(row["status"] == "PASS" for row in result["models"].values())
        else "FAIL"
    )
    return result


def _gpu_status() -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        return {"available": bool(output), "inventory": output.splitlines()}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "inventory": []}


def write_doctor_json(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
