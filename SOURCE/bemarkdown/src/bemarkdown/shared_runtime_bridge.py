from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SharedRuntimeBridgeResolution:
    model_id: str
    model_root: Path
    manifest: dict[str, Any]
    runtime_id: str
    provider_adapter: str
    shared_runtime: bool


class SharedRuntimeRegistryBridge:
    """Opt-in bridge; existing BeMarkdown registry remains the fail-closed default."""

    def __init__(self, *, shared_resolver: Any | None = None, legacy_registry: Any | None = None) -> None:
        if shared_resolver is None and legacy_registry is None:
            raise ValueError("A shared resolver or legacy registry is required")
        self.shared_resolver = shared_resolver
        self.legacy_registry = legacy_registry

    def resolve(
        self,
        model_id: str,
        *,
        models_root: str | Path | None = None,
        platform: str = "windows-x86_64",
        device: str = "cuda:0",
        deep: bool = True,
        use_shared_runtime: bool = False,
    ) -> Any:
        if not use_shared_runtime:
            if self.legacy_registry is None:
                raise RuntimeError("BEMARKDOWN_LEGACY_MODEL_PATH_UNAVAILABLE")
            return self.legacy_registry.resolve(model_id, deep=deep)
        if self.shared_resolver is None:
            raise RuntimeError("SHARED_MODEL_RUNTIME_UNAVAILABLE")
        if models_root is None:
            raise RuntimeError("SHARED_MODEL_RUNTIME_ROOT_REQUIRED")
        resolution = self.shared_resolver.resolve(
            model_id,
            models_root=models_root,
            platform=platform,
            device=device,
            verify_fingerprint=deep,
        )
        return SharedRuntimeBridgeResolution(
            resolution.model_id,
            resolution.model_asset_path,
            resolution.model_manifest,
            resolution.runtime_id,
            resolution.model.provider_adapter,
            True,
        )
