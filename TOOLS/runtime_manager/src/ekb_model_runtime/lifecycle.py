from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class RuntimeLifecycle:
    """One-model-at-a-time lifecycle primitive; orchestration remains tool-owned."""

    def __init__(
        self, factory: Callable[[], Any], unloader: Callable[[Any], None] | None = None
    ) -> None:
        self._factory = factory
        self._unloader = unloader
        self._instance: Any = None
        self._loads = 0
        self._reuses = 0
        self._unloads = 0
        self._load_ms = 0.0

    def load(self) -> Any:
        if self._instance is not None:
            self._reuses += 1
            return self._instance
        started = time.perf_counter()
        self._instance = self._factory()
        self._load_ms += (time.perf_counter() - started) * 1000
        self._loads += 1
        return self._instance

    def unload(self) -> None:
        if self._instance is None:
            return
        if self._unloader is not None:
            self._unloader(self._instance)
        self._instance = None
        self._unloads += 1

    def health(self) -> dict[str, Any]:
        return {"status": "READY", "loaded": self._instance is not None}

    def stats(self) -> dict[str, Any]:
        return {
            "loaded": self._instance is not None,
            "load_count": self._loads,
            "reuse_count": self._reuses,
            "unload_count": self._unloads,
            "total_load_ms": self._load_ms,
        }
