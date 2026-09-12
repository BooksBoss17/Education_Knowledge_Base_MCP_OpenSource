from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .model_registry import (
    MODEL_CATALOG,
    MODEL_MANIFEST_NAME,
    MODEL_RUNTIME_CONTRACT,
    PADDLE_MODEL_CATALOG,
    ModelRegistry,
)
from .release import sha256_file

MODEL_SUITE_ID = "bemarkdown-paddle-document-suite-v1"
MODEL_SUITE_SCHEMA = "bemarkdown-model-suite-manifest-v1"
MODEL_SUITE_MANIFEST_NAME = "MODEL_SUITE_MANIFEST.json"
PDF_PRODUCTION_SUITE_ID = "bemarkdown-pdf-production-suite-v1"
PDF_PRODUCTION_SUITE_MANIFEST_NAME = "PDF_PRODUCTION_SUITE_MANIFEST.json"


class ModelSuiteManifestError(RuntimeError):
    pass


def build_model_suite_manifest(models_root: str | Path) -> dict[str, Any]:
    return _build_suite_manifest(
        models_root,
        catalog=PADDLE_MODEL_CATALOG,
        suite_id=MODEL_SUITE_ID,
    )


def build_pdf_production_suite_manifest(models_root: str | Path) -> dict[str, Any]:
    return _build_suite_manifest(
        models_root,
        catalog=MODEL_CATALOG,
        suite_id=PDF_PRODUCTION_SUITE_ID,
    )


def build_pdf_production_suite_manifest_from_legacy(
    legacy_suite: dict[str, Any],
    extension_models_root: str | Path,
) -> dict[str, Any]:
    """Extend a validated nine-model suite without copying its model payloads.

    The publication RC owns only the two PDF text-extension models.  The
    existing Paddle suite remains owned by Formal MCP, so its rows are copied
    verbatim after their frozen metadata and fingerprint have been checked.
    """

    legacy_rows = _validate_suite_payload(
        legacy_suite,
        catalog=PADDLE_MODEL_CATALOG,
        suite_id=MODEL_SUITE_ID,
    )
    registry = ModelRegistry(models_root=Path(extension_models_root).resolve())
    rows = deepcopy(legacy_rows)
    extension_catalog = {
        model_id: spec
        for model_id, spec in MODEL_CATALOG.items()
        if model_id not in PADDLE_MODEL_CATALOG
    }
    for model_id in sorted(extension_catalog):
        spec = extension_catalog[model_id]
        resolution = registry.resolve(model_id, deep=True)
        manifest_path = resolution.model_root / MODEL_MANIFEST_NAME
        rows.append(
            {
                "model_id": model_id,
                "path": spec.directory,
                "engine": spec.engine,
                "task": spec.task,
                "required_by": list(spec.required_by),
                "model_manifest_sha256": sha256_file(manifest_path),
                "model_fingerprint": resolution.manifest["model_fingerprint"],
                "integration_status": spec.integration_status,
            }
        )
    rows.sort(key=lambda row: row["model_id"])
    payload = {
        "schema": MODEL_SUITE_SCHEMA,
        "suite_id": PDF_PRODUCTION_SUITE_ID,
        "runtime_contract": MODEL_RUNTIME_CONTRACT,
        "model_count": len(rows),
        "models": rows,
        "suite_fingerprint": suite_fingerprint(rows),
        "suite_status": "MODEL_READY",
    }
    _validate_suite_payload(
        payload,
        catalog=MODEL_CATALOG,
        suite_id=PDF_PRODUCTION_SUITE_ID,
    )
    return payload


def _build_suite_manifest(
    models_root: str | Path,
    *,
    catalog: dict[str, Any],
    suite_id: str,
) -> dict[str, Any]:
    models_root = Path(models_root).resolve()
    registry = ModelRegistry(models_root=models_root)
    rows: list[dict[str, Any]] = []
    for model_id in sorted(catalog):
        spec = catalog[model_id]
        resolution = registry.resolve(model_id, deep=True)
        manifest_path = resolution.model_root / MODEL_MANIFEST_NAME
        rows.append(
            {
                "model_id": model_id,
                "path": spec.directory,
                "engine": spec.engine,
                "task": spec.task,
                "required_by": list(spec.required_by),
                "model_manifest_sha256": sha256_file(manifest_path),
                "model_fingerprint": resolution.manifest["model_fingerprint"],
                "integration_status": spec.integration_status,
            }
        )
    return {
        "schema": MODEL_SUITE_SCHEMA,
        "suite_id": suite_id,
        "runtime_contract": MODEL_RUNTIME_CONTRACT,
        "model_count": len(rows),
        "models": rows,
        "suite_fingerprint": suite_fingerprint(rows),
        "suite_status": "MODEL_READY",
    }


def validate_model_suite(
    models_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    deep: bool = True,
) -> dict[str, Any]:
    return _validate_suite(
        models_root,
        manifest_path=manifest_path,
        deep=deep,
        catalog=PADDLE_MODEL_CATALOG,
        suite_id=MODEL_SUITE_ID,
        default_manifest_name=MODEL_SUITE_MANIFEST_NAME,
    )


def validate_pdf_production_suite(
    models_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    deep: bool = True,
) -> dict[str, Any]:
    return _validate_suite(
        models_root,
        manifest_path=manifest_path,
        deep=deep,
        catalog=MODEL_CATALOG,
        suite_id=PDF_PRODUCTION_SUITE_ID,
        default_manifest_name=PDF_PRODUCTION_SUITE_MANIFEST_NAME,
    )


def _validate_suite(
    models_root: str | Path,
    *,
    manifest_path: str | Path | None,
    deep: bool,
    catalog: dict[str, Any],
    suite_id: str,
    default_manifest_name: str,
) -> dict[str, Any]:
    models_root = Path(models_root).resolve()
    path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else models_root / default_manifest_name
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelSuiteManifestError(f"Invalid model suite manifest: {exc}") from exc
    rows = _validate_suite_payload(payload, catalog=catalog, suite_id=suite_id)

    registry = ModelRegistry(models_root=models_root)
    for row in rows:
        model_id = row["model_id"]
        resolution = registry.resolve(model_id, deep=deep)
        manifest_path = resolution.model_root / MODEL_MANIFEST_NAME
        if sha256_file(manifest_path) != row.get("model_manifest_sha256"):
            raise ModelSuiteManifestError(f"Model manifest SHA mismatch for {model_id}")
        if resolution.manifest["model_fingerprint"] != row.get("model_fingerprint"):
            raise ModelSuiteManifestError(f"Model fingerprint mismatch for {model_id}")
    return payload


def _validate_suite_payload(
    payload: dict[str, Any],
    *,
    catalog: dict[str, Any],
    suite_id: str,
) -> list[dict[str, Any]]:
    if payload.get("schema") != MODEL_SUITE_SCHEMA or payload.get("suite_id") != suite_id:
        raise ModelSuiteManifestError("Unsupported model suite manifest identity")
    rows = payload.get("models")
    if not isinstance(rows, list) or len(rows) != len(catalog):
        raise ModelSuiteManifestError(
            f"Model suite must contain exactly {len(catalog)} models"
        )
    ids = [row.get("model_id") for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or len(ids) != len(set(ids)):
        raise ModelSuiteManifestError("Model suite contains a duplicate model_id")
    if set(ids) != set(catalog):
        raise ModelSuiteManifestError("Model suite model_id set is incomplete or unexpected")
    if ids != sorted(ids):
        raise ModelSuiteManifestError("Model suite models must use canonical model_id order")

    for row in rows:
        model_id = row["model_id"]
        spec = catalog[model_id]
        if row.get("path") != spec.directory:
            raise ModelSuiteManifestError(f"Model suite path mismatch for {model_id}")
        frozen_fields = {
            "engine": spec.engine,
            "task": spec.task,
            "required_by": list(spec.required_by),
            "integration_status": spec.integration_status,
        }
        for field, expected in frozen_fields.items():
            if row.get(field) != expected:
                raise ModelSuiteManifestError(
                    f"Model suite {field} mismatch for {model_id}"
                )
    expected_fingerprint = suite_fingerprint(rows)
    if payload.get("suite_fingerprint") != expected_fingerprint:
        raise ModelSuiteManifestError("Model suite fingerprint mismatch")
    if payload.get("model_count") != len(catalog):
        raise ModelSuiteManifestError("Model suite model_count is invalid")
    if payload.get("runtime_contract") != MODEL_RUNTIME_CONTRACT:
        raise ModelSuiteManifestError("Model suite runtime contract is invalid")
    if payload.get("suite_status") != "MODEL_READY":
        raise ModelSuiteManifestError("Model suite is not MODEL_READY")
    return rows


def suite_fingerprint(rows: list[dict[str, Any]]) -> str:
    identity = [
        {
            "model_id": row["model_id"],
            "model_manifest_sha256": row["model_manifest_sha256"],
            "model_fingerprint": row["model_fingerprint"],
        }
        for row in sorted(rows, key=lambda item: item["model_id"])
    ]
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
