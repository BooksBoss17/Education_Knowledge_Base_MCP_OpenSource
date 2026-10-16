from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def cache_environment(models_root: str | Path) -> dict[str, str]:
    root = Path(models_root).resolve() / "_CACHE"
    return {
        "UV_CACHE_DIR": str(root / "uv"),
        "HF_HOME": str(root / "huggingface"),
        "TORCH_HOME": str(root / "torch"),
        "TRITON_CACHE_DIR": str(root / "kernels"),
        **OFFLINE_ENVIRONMENT,
    }


@contextmanager
def offline_environment(models_root: str | Path) -> Iterator[dict[str, str]]:
    values = cache_environment(models_root)
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield values
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
