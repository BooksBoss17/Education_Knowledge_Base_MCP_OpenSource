from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ModelRuntimeError
from .model_registry import ModelEntry, ModelRegistryV2
from .path_security import resolve_beneath
from .runtime_registry import RuntimeEntry, RuntimeRegistryV1

MODELS_ROOT_ENV = "EKB_MODELS_ROOT"


@dataclass(frozen=True, slots=True)
class ModelRuntimeResolution:
    model: ModelEntry
    runtime: RuntimeEntry
    models_root: Path
    model_asset_path: Path
    model_manifest: dict[str, Any]
    runtime_env_path: Path
    launcher_path: Path
    runtime_profile: str | None
    device: str

    @property
    def model_id(self) -> str:
        return self.model.model_id

    @property
    def runtime_id(self) -> str:
        return self.runtime.runtime_id


class ModelRuntimeResolver:
    def __init__(self, models: ModelRegistryV2, runtimes: RuntimeRegistryV1) -> None:
        self.models = models
        self.runtimes = runtimes

    @classmethod
    def from_root(cls, models_root: str | Path | None = None) -> ModelRuntimeResolver:
        root = resolve_models_root(models_root)
        return cls(
            ModelRegistryV2.load(root / "MODEL_REGISTRY.json"),
            RuntimeRegistryV1.load(root / "RUNTIME_REGISTRY.json"),
        )

    def resolve(
        self,
        model_id: str,
        *,
        models_root: str | Path,
        platform: str,
        device: str,
        runtime_profile: str | None = None,
        verify_fingerprint: bool = True,
    ) -> ModelRuntimeResolution:
        root = Path(models_root).resolve()
        model = self.models.get(model_id)
        runtime = self.runtimes.get(model.runtime_id)
        if (
            platform not in model.supported_platforms
            or platform not in runtime.platforms
        ):
            raise ModelRuntimeError("PLATFORM_UNSUPPORTED", platform)
        if device.startswith("cuda") and not runtime.gpu_support:
            raise ModelRuntimeError("RUNTIME_UNAVAILABLE", "GPU not supported")
        if device == "cpu" and not runtime.cpu_support:
            raise ModelRuntimeError("RUNTIME_UNAVAILABLE", "CPU fallback forbidden")
        if runtime.status != "READY":
            raise ModelRuntimeError("RUNTIME_UNAVAILABLE", runtime.runtime_id)
        model_path = resolve_beneath(
            root, model.artifact_relpath, code="UNSAFE_MODEL_PATH"
        )
        if not model_path.is_dir():
            raise ModelRuntimeError("MODEL_ASSET_MISSING", model.model_id)
        manifest_path = resolve_beneath(
            root, model.manifest_ref, code="UNSAFE_MODEL_PATH"
        )
        if not manifest_path.is_file():
            raise ModelRuntimeError("MODEL_ASSET_MISSING", model.manifest_ref)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelRuntimeError("MODEL_ASSET_MISSING", str(exc)) from exc
        manifest_fingerprint = manifest.get("model_fingerprint") or manifest.get(
            "fingerprint"
        )
        if manifest_fingerprint != model.fingerprint:
            raise ModelRuntimeError("MODEL_FINGERPRINT_MISMATCH", model.model_id)
        if (
            verify_fingerprint
            and directory_fingerprint(model_path) != model.fingerprint
        ):
            raise ModelRuntimeError("MODEL_FINGERPRINT_MISMATCH", model.model_id)
        env_path = resolve_beneath(
            root, runtime.env_relpath, code="UNSAFE_RUNTIME_PATH"
        )
        launcher = resolve_beneath(
            root, runtime.launcher_ref, code="UNSAFE_RUNTIME_PATH"
        )
        if not env_path.exists() or not launcher.is_file():
            raise ModelRuntimeError("RUNTIME_UNAVAILABLE", runtime.runtime_id)
        lock_path = resolve_beneath(root, runtime.lock_ref, code="UNSAFE_RUNTIME_PATH")
        if not lock_path.is_file():
            raise ModelRuntimeError("DEPENDENCY_LOCK_MISMATCH", runtime.lock_ref)
        if runtime.lock_sha256 and sha256_file(lock_path) != runtime.lock_sha256:
            raise ModelRuntimeError("DEPENDENCY_LOCK_MISMATCH", runtime.runtime_id)
        return ModelRuntimeResolution(
            model,
            runtime,
            root,
            model_path,
            manifest,
            env_path,
            launcher,
            runtime_profile,
            device,
        )


def directory_fingerprint(model_root: str | Path) -> str:
    root = Path(model_root).resolve()
    digest = hashlib.sha256()
    for path in sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if relative == "MODEL_MANIFEST.json":
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_models_root(explicit: str | Path | None = None) -> Path:
    value = explicit if explicit is not None else os.environ.get(MODELS_ROOT_ENV)
    if value is None or not str(value).strip():
        raise ModelRuntimeError(
            "MODELS_ROOT_NOT_CONFIGURED", "use an explicit root or EKB_MODELS_ROOT"
        )
    return Path(value).expanduser().resolve()
