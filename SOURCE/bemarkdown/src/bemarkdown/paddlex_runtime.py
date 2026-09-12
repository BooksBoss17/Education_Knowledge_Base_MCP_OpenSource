"""Import PaddleX without loading optional Torch into a Paddle provider process."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from types import ModuleType
from typing import Any

_IMPORT_GUARD = threading.RLock()


class _CompiledYamlReader:
    """Use the equivalent LibYAML loader only inside PaddleX's config reader."""

    def __init__(self, delegate):
        self._delegate = delegate
        self.FullLoader = getattr(delegate, "CFullLoader", delegate.FullLoader)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def use_compiled_yaml_reader(reader: ModuleType) -> None:
    # Model dictionaries can contain tens of thousands of YAML entries. The
    # native loader avoids seconds of Python tokenization without changing
    # FullLoader semantics or the shared yaml module used by other libraries.
    if not isinstance(reader.yaml, _CompiledYamlReader):
        reader.yaml = _CompiledYamlReader(reader.yaml)


class _TorchBlindImportUtil:
    """Module-local proxy used by ModelScope's logger in Paddle-only workers."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def find_spec(self, name: str, package: str | None = None) -> Any:
        if name == "torch" or name.startswith("torch."):
            return None
        return self._delegate.find_spec(name, package)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def import_paddlex_for_paddle_provider(
) -> tuple[ModuleType, ModuleType, Callable[..., Any]]:
    """Load Paddle/PaddleX while keeping Torch out of the provider process.

    PaddleX 3.7.2 imports ModelScope's download hoster even for local model
    paths.  ModelScope's logger probes an installed Torch distribution directly
    and imports ``torch_utils`` despite ``USE_TORCH=0``.  In the single full
    runtime that would load Torch's CUDA 13 cuDNN after Paddle's CUDA 12.6 cuDNN.
    The logger proxy keeps that optional discovery disabled only inside the
    Paddle provider process; the GOT worker remains a normal Torch process.
    """

    if _torch_is_loaded():
        raise RuntimeError("PADDLE_PROVIDER_TORCH_ALREADY_IMPORTED")

    paddle = importlib.import_module("paddle")
    with _temporary_optional_torch_discovery_guard():
        logger_module = importlib.import_module("modelscope.utils.logger")
        logger_module.iutil = _TorchBlindImportUtil(logger_module.iutil)
        logger_module.get_logger()
    paddlex = importlib.import_module("paddlex")
    config_reader = sys.modules.get("paddlex.inference.utils.io.readers")
    if config_reader is not None:
        use_compiled_yaml_reader(config_reader)
    if _torch_is_loaded():
        raise RuntimeError("PADDLE_PROVIDER_OPTIONAL_TORCH_IMPORT_NOT_ISOLATED")
    create_model = getattr(paddlex, "create_model", None)
    if not callable(create_model):
        raise TypeError("PADDLEX_CREATE_MODEL_UNAVAILABLE")

    # Keep the module-local proxy installed for later ModelScope logger calls in
    # this Paddle-only process.  It does not modify importlib globally.
    if logger_module.iutil.find_spec("torch") is not None:
        raise RuntimeError("PADDLE_PROVIDER_TORCH_DISCOVERY_NOT_ISOLATED")
    return paddle, paddlex, create_model


@contextmanager
def _temporary_optional_torch_discovery_guard():
    """Hide installed Torch while ModelScope's parent package initializes."""

    with _IMPORT_GUARD:
        had_use_torch = "USE_TORCH" in os.environ
        previous_use_torch = os.environ.get("USE_TORCH")
        original_find_spec = importlib.util.find_spec

        def find_spec_without_torch(name: str, package: str | None = None) -> Any:
            if name == "torch" or name.startswith("torch."):
                return None
            return original_find_spec(name, package)

        os.environ["USE_TORCH"] = "0"
        importlib.util.find_spec = find_spec_without_torch
        try:
            yield
        finally:
            importlib.util.find_spec = original_find_spec
            if had_use_torch:
                assert previous_use_torch is not None
                os.environ["USE_TORCH"] = previous_use_torch
            else:
                os.environ.pop("USE_TORCH", None)


def _torch_is_loaded() -> bool:
    return any(name == "torch" or name.startswith("torch.") for name in sys.modules)
