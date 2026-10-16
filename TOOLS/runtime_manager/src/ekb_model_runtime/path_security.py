from __future__ import annotations

from pathlib import Path, PurePosixPath

from .errors import ModelRuntimeError


def safe_relative_path(
    value: str, *, code: str = "UNSAFE_RELATIVE_PATH"
) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ModelRuntimeError(code, repr(value))
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ModelRuntimeError(code, value)
    if len(path.parts) and ":" in path.parts[0]:
        raise ModelRuntimeError(code, value)
    return path


def resolve_beneath(root: Path, relative: str, *, code: str) -> Path:
    rel = safe_relative_path(relative, code=code)
    base = root.resolve()
    candidate = base.joinpath(*rel.parts).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ModelRuntimeError(code, relative) from exc
    return candidate


def require_runtime_relative(relative: str) -> PurePosixPath:
    path = safe_relative_path(relative, code="UNSAFE_RUNTIME_PATH")
    if not path.parts or path.parts[0] != "_RUNTIME":
        raise ModelRuntimeError("UNSAFE_RUNTIME_PATH", relative)
    return path
