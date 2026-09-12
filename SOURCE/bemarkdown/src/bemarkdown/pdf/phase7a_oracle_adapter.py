"""Read-only adapter for the frozen Phase 7A-R1 diagnostic oracle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def verify_oracle_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    verified = 0
    absent_local_payloads = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.exists():
            absent_local_payloads.append(row["path"])
            continue
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        expected = {"bytes": int(row["bytes"]), "sha256": row["sha256"]}
        if actual != expected:
            mismatches.append(
                {"path": row["path"], "expected": expected, "actual": actual}
            )
        else:
            verified += 1
    if mismatches:
        raise RuntimeError(f"PHASE7A_ORACLE_HASH_MISMATCH:{mismatches}")
    return {
        "schema": "bemarkdown-phase7a-oracle-verification-v1",
        "status": "FULL_EVIDENCE_DIAGNOSTIC_ORACLE_PRESERVED",
        "manifest_sha256": sha256_file(manifest_path),
        "verified_payloads": verified,
        "absent_local_payloads": absent_local_payloads,
        "mismatches": [],
    }
