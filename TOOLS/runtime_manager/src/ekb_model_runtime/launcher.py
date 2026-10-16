from __future__ import annotations

import os
import subprocess

from .environment import cache_environment
from .errors import ModelRuntimeError
from .resolver import ModelRuntimeResolution


class ControlledPythonLauncher:
    """Launch a registry-controlled Python worker without a shell or fallback."""

    def __init__(self, resolution: ModelRuntimeResolution) -> None:
        if resolution.runtime.runtime_kind != "python_env":
            raise ModelRuntimeError("RUNTIME_UNAVAILABLE", "not a Python environment")
        self.resolution = resolution

    def command(self, arguments: list[str]) -> list[str]:
        env = self.resolution.runtime_env_path
        candidates = (env / "Scripts/python.exe", env / "bin/python")
        python = next((path for path in candidates if path.is_file()), None)
        if python is None:
            raise ModelRuntimeError("RUNTIME_UNAVAILABLE", "registered Python missing")
        return [str(python), str(self.resolution.launcher_path), *arguments]

    def run(
        self, arguments: list[str], *, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(cache_environment(self.resolution.models_root))
        return subprocess.run(
            self.command(arguments),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=environment,
            shell=False,
        )
