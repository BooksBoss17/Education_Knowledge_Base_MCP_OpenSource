from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def paddle_compatible_model_dir(model_dir: str | Path) -> Path:
    """Expose an immutable model through a short ASCII junction when required."""

    resolved = Path(model_dir).resolve()
    if os.name != "nt" or str(resolved).isascii():
        return resolved

    import _winapi
    import ctypes

    local_app_data = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    alias_parent = local_app_data / "BeMarkdown" / "model-aliases"
    alias_parent.mkdir(parents=True, exist_ok=True)
    get_short_path = ctypes.windll.kernel32.GetShortPathNameW
    length = get_short_path(str(alias_parent), None, 0)
    if not length:
        raise RuntimeError("PADDLE_MODEL_PATH_ALIAS_FAILED: no short host-state path")
    buffer = ctypes.create_unicode_buffer(length)
    if not get_short_path(str(alias_parent), buffer, length):
        raise RuntimeError("PADDLE_MODEL_PATH_ALIAS_FAILED: short path resolution failed")
    short_parent = Path(buffer.value)
    if not str(short_parent).isascii():
        raise RuntimeError("PADDLE_MODEL_PATH_ALIAS_FAILED: host-state path is not ASCII")

    alias_name = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    alias = short_parent / alias_name
    if not alias.exists():
        try:
            _winapi.CreateJunction(str(resolved), str(alias))
        except FileExistsError:
            pass
    if not alias.exists() or alias.resolve() != resolved:
        raise RuntimeError("PADDLE_MODEL_PATH_ALIAS_FAILED: junction target mismatch")
    return alias
