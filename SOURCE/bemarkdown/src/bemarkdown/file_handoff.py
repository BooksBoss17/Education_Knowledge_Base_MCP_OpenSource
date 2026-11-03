"""Bounded recovery of file handoffs, without repeating the producing work.

Access denied can be a persistent ACL error or a transient Windows sharing
conflict. Retry only this narrow error family and preserve the original error
when the budget expires. These functions never retry a model or conversion.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

LOGGER = logging.getLogger("bemarkdown.file_handoff")
T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 9
    timeout_seconds: float = 2.0
    initial_delay_seconds: float = 0.025
    max_delay_seconds: float = 0.4

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.timeout_seconds < 0:
            raise ValueError("File retry budget must be nonnegative with at least one attempt")
        if self.initial_delay_seconds <= 0 or self.max_delay_seconds <= 0:
            raise ValueError("File retry delays must be positive")


DEFAULT_RETRY_POLICY = RetryPolicy()


def retry_file_operation(
    action: Callable[[], T], *, operation: str, paths: tuple[Path, ...],
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
) -> T:
    """Retry one idempotent file operation; re-raise its last original error.

    Both attempt and elapsed-time bounds apply. The elapsed bound governs
    retries; it cannot interrupt a filesystem syscall already in progress.
    """
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            result = action()
        except OSError as error:
            elapsed = time.monotonic() - started
            transient = isinstance(error, PermissionError) or getattr(error, "winerror", None) in {5, 32, 33}
            remaining = policy.timeout_seconds - elapsed
            retry = transient and attempts < policy.max_attempts and remaining > 0
            record = {
                "operation": operation, "paths": [str(path) for path in paths],
                "status": "RETRY" if retry else "FAILED", "attempts": attempts,
                "elapsed_seconds": elapsed, "errno": error.errno,
                "winerror": getattr(error, "winerror", None),
                "error_type": type(error).__name__, "error": str(error),
            }
            LOGGER.warning("FILE_HANDOFF %s", json.dumps(record, ensure_ascii=False))
            if not retry:
                error.add_note("FILE_HANDOFF " + json.dumps(record, ensure_ascii=False))
                raise
            delay = min(policy.initial_delay_seconds * 2 ** (attempts - 1), policy.max_delay_seconds, remaining)
            time.sleep(delay)
        else:
            if attempts > 1:
                LOGGER.warning("FILE_HANDOFF %s", json.dumps({
                    "operation": operation, "paths": [str(path) for path in paths],
                    "status": "RECOVERED", "attempts": attempts,
                    "elapsed_seconds": time.monotonic() - started,
                }, ensure_ascii=False))
            return result


def replace_path(source: Path, destination: Path, *, policy: RetryPolicy = DEFAULT_RETRY_POLICY) -> None:
    retry_file_operation(lambda: os.replace(source, destination), operation="replace", paths=(source, destination), policy=policy)


def read_text(path: Path, *, policy: RetryPolicy = DEFAULT_RETRY_POLICY) -> str:
    return retry_file_operation(lambda: path.read_text(encoding="utf-8"), operation="read_text", paths=(path,), policy=policy)


def atomic_write_bytes(path: Path, value: bytes, *, policy: RetryPolicy = DEFAULT_RETRY_POLICY) -> None:
    retry_file_operation(lambda: path.parent.mkdir(parents=True, exist_ok=True), operation="mkdir", paths=(path.parent,), policy=policy)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        retry_file_operation(lambda: temporary.write_bytes(value), operation="write_bytes", paths=(temporary,), policy=policy)
        replace_path(temporary, path, policy=policy)
    except OSError as error:
        # Keep the temporary payload for diagnosis/recovery (a failed write
        # may be partial). Never repeat inference because publication failed.
        error.add_note(f"FILE_HANDOFF retained_temporary={temporary} destination={path}")
        raise


def atomic_write_text(path: Path, value: str, *, policy: RetryPolicy = DEFAULT_RETRY_POLICY) -> None:
    atomic_write_bytes(path, value.encode("utf-8"), policy=policy)
