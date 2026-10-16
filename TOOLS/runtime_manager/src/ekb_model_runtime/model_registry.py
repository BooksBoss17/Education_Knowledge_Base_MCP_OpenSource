from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ModelRuntimeError
from .path_security import safe_relative_path

MODEL_REGISTRY_V2_SCHEMA = "ekb-model-registry-v2"
LEGACY_SCHEMAS = {"mcp-shared-model-registry-v1", "bemarkdown-model-registry-v1"}
CAPABILITIES = {
    "layout",
    "ocr_text",
    "ocr_formula",
    "table_classifier",
    "table_structure",
    "table_cell_detection",
    "asr",
    "embedding",
}

_TASK_CAPABILITY = {
    "layout_detection": "layout",
    "text_detection": "ocr_text",
    "text_recognition": "ocr_text",
    "formula_ocr": "ocr_formula",
    "table_classification": "table_classifier",
    "table_structure_recognition": "table_structure",
    "table_cell_detection": "table_cell_detection",
}


@dataclass(frozen=True, slots=True)
class ModelEntry:
    model_id: str
    display_name: str
    capability: str
    model_family: str
    artifact_relpath: str
    manifest_ref: str
    revision: str
    fingerprint: str
    precision: str
    runtime_id: str
    provider_adapter: str
    license: Any
    source: Any
    supported_platforms: tuple[str, ...]
    minimum_vram_mib: int
    recommended_vram_mib: int
    status: str
    legacy: bool = False


class ModelRegistryV2:
    def __init__(self, entries: list[ModelEntry], *, source_schema: str) -> None:
        ids = [entry.model_id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ModelRuntimeError("INVALID_MODEL_REGISTRY", "duplicate model_id")
        self.entries = {entry.model_id: entry for entry in entries}
        self.source_schema = source_schema

    @classmethod
    def load(cls, path: str | Path) -> ModelRegistryV2:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelRuntimeError("INVALID_MODEL_REGISTRY", str(exc)) from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelRegistryV2:
        if not isinstance(payload, dict):
            raise ModelRuntimeError("INVALID_MODEL_REGISTRY", "root must be object")
        schema = payload.get("schema")
        rows = payload.get("models")
        if not isinstance(rows, list):
            raise ModelRuntimeError("INVALID_MODEL_REGISTRY", "models must be a list")
        if schema == MODEL_REGISTRY_V2_SCHEMA:
            entries = [_parse_v2(row) for row in rows]
        elif schema in LEGACY_SCHEMAS:
            entries = [_migrate_v1(row) for row in rows]
        else:
            raise ModelRuntimeError("UNKNOWN_MODEL_REGISTRY_SCHEMA", str(schema))
        return cls(entries, source_schema=str(schema))

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self.entries[model_id]
        except KeyError as exc:
            raise ModelRuntimeError("MODEL_NOT_REGISTERED", model_id) from exc


def _parse_v2(row: Any) -> ModelEntry:
    required = (
        "model_id",
        "display_name",
        "capability",
        "model_family",
        "artifact_relpath",
        "manifest_ref",
        "revision",
        "fingerprint",
        "precision",
        "runtime_id",
        "provider_adapter",
        "license",
        "source",
        "supported_platforms",
        "minimum_vram_mib",
        "recommended_vram_mib",
        "status",
    )
    if not isinstance(row, dict) or any(key not in row for key in required):
        raise ModelRuntimeError("INVALID_MODEL_REGISTRY", "v2 row is incomplete")
    safe_relative_path(row["artifact_relpath"], code="UNSAFE_MODEL_PATH")
    safe_relative_path(row["manifest_ref"], code="UNSAFE_MODEL_PATH")
    if row["capability"] not in CAPABILITIES:
        raise ModelRuntimeError("INVALID_MODEL_REGISTRY", "unknown capability")
    platforms = row["supported_platforms"]
    if not isinstance(platforms, list) or not platforms:
        raise ModelRuntimeError("INVALID_MODEL_REGISTRY", "platforms required")
    if not isinstance(row["fingerprint"], str) or len(row["fingerprint"]) != 64:
        raise ModelRuntimeError("INVALID_MODEL_REGISTRY", "fingerprint must be sha256")
    return ModelEntry(
        **{
            **{key: row[key] for key in required},
            "supported_platforms": tuple(platforms),
        }
    )


def _migrate_v1(row: Any) -> ModelEntry:
    if not isinstance(row, dict):
        raise ModelRuntimeError("INVALID_MODEL_REGISTRY", "legacy row must be object")
    model_id = row.get("model_id")
    artifact = row.get("physical_directory") or row.get("directory")
    fingerprint = row.get("model_fingerprint")
    if not all(
        isinstance(value, str) and value for value in (model_id, artifact, fingerprint)
    ):
        raise ModelRuntimeError("INVALID_MODEL_REGISTRY", "legacy identity incomplete")
    safe_relative_path(artifact, code="UNSAFE_MODEL_PATH")
    manifest = row.get("manifest_path") or f"{artifact}/MODEL_MANIFEST.json"
    safe_relative_path(manifest, code="UNSAFE_MODEL_PATH")
    capability = _TASK_CAPABILITY.get(row.get("task"), "ocr_text")
    return ModelEntry(
        model_id=model_id,
        display_name=artifact,
        capability=capability,
        model_family=str(row.get("provider", "legacy")).casefold(),
        artifact_relpath=artifact,
        manifest_ref=manifest,
        revision=str(row.get("revision", "LEGACY_MANIFEST_AUTHORITY")),
        fingerprint=fingerprint,
        precision=str(row.get("precision", "fp32")),
        runtime_id=str(row.get("runtime_id", "paddle-win-cu")),
        provider_adapter=str(row.get("provider_adapter", "bemarkdown.paddle_legacy")),
        license=row.get("license", {"status": "MANIFEST_AUTHORITY"}),
        source=row.get("source", {"provider": row.get("provider", "unknown")}),
        supported_platforms=tuple(row.get("supported_platforms", ["windows-x86_64"])),
        minimum_vram_mib=int(row.get("minimum_vram_mib", 0)),
        recommended_vram_mib=int(row.get("recommended_vram_mib", 0)),
        status=str(row.get("state", "MODEL_READY")),
        legacy=True,
    )
