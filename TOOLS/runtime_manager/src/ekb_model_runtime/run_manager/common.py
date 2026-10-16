"""Strict JSON, fingerprints and bounded local persistence helpers."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import time

SCHEMA = "ekb-run-manager-v1"
ACTIVE = ("QUEUED", "WAITING_RESOURCES", "STARTING", "RUNNING", "CANCEL_REQUESTED")
TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED", "CAPACITY_INSUFFICIENT", "WAIT_EXPIRED")


class RunManagerError(RuntimeError):
    def __init__(self, code, message=None, details=None):
        self.code, self.details = code, details or {}
        super().__init__(message or code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprint():
    folder = Path(__file__).resolve().parent
    return fingerprint({path.name: file_sha(path) for path in sorted(folder.glob("*.py"))})


def home(root=None):
    return Path(root or os.environ.get("EKB_RUN_MANAGER_HOME") or
                Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share")) / "BeMarkdown/run-manager").resolve()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    try:
        temporary.write_text(canonical(value) + "\n", encoding="utf-8")
        for attempt in range(4):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(.03 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def bounded_number(value, name, maximum=2**63 - 1):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise RunManagerError("INVALID_ARGUMENT", name)
    return value


class RotatingLog:
    """At most (backups+1)*limit bytes on disk; no growing in-memory buffer."""
    def __init__(self, path, limit=1024 * 1024, backups=2):
        self.path, self.limit, self.backups = Path(path), limit, backups
        self.tail = b""

    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8", errors="replace")
        for offset in range(0, len(data), self.limit):
            part = data[offset:offset + self.limit]
            if self.path.exists() and self.path.stat().st_size + len(part) > self.limit:
                self.path.with_name(self.path.name + f".{self.backups}").unlink(missing_ok=True)
                for number in range(self.backups - 1, 0, -1):
                    old = self.path.with_name(self.path.name + f".{number}")
                    if old.exists():
                        old.replace(self.path.with_name(self.path.name + f".{number + 1}"))
                self.path.replace(self.path.with_name(self.path.name + ".1"))
            with self.path.open("ab", buffering=0) as stream:
                stream.write(part)
            self.tail = (self.tail + part)[-4096:]

    def event(self, value):
        self.write(canonical(value) + "\n")
