from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import load_workspace_config

MODEL_ID = "pp-formulanet-plus-l"
MODEL_DIRECTORY = "PP-FormulaNet_plus-L"
MODEL_MANIFEST_NAME = "MODEL_MANIFEST.json"
MODEL_MANIFEST_SCHEMA = "bemarkdown-model-manifest-v1"
MODEL_REVISION = "0809597a77f735bfb35354edb632f2e6dff606f3"
MODEL_EXPECTED_FINGERPRINT = (
    "b215b9cfdebb18660cc6ab66951fa89a23463128be2b879a6e4f2c2d80939b4e"
)
MODEL_RUNTIME_PROFILE = "win-x64-py311-cuda126"
MODEL_RUNTIME_CONTRACT = "bemarkdown-runtime-contract-v1"
PATH_INVENTORY_FINGERPRINT_CONTRACT = (
    "sha256(path\\0file_sha256\\n), excluding MODEL_MANIFEST.json"
)
CANONICAL_JSON_INVENTORY_FINGERPRINT_CONTRACT = (
    "sha256-canonical-json-file-inventory-v1"
)
MODELS_ROOT_ENV = "BEMARKDOWN_MODELS_ROOT"
MCP_ROOT_ENV = "BEMARKDOWN_MCP_ROOT"
DEVELOPER_FALLBACK_ENV = "BEMARKDOWN_ALLOW_DEVELOPER_MODEL_FALLBACK"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    directory: str
    engine: str
    task: str
    upstream_revision: str
    required_by: tuple[str, ...]
    integration_status: str = "MODEL_READY"
    provider: str = "PaddlePaddle"
    upstream_source_override: str | None = None
    runtime_contract: str = MODEL_RUNTIME_CONTRACT
    expected_device: str = "gpu:0"
    precision_baseline: str = "fp32"
    fingerprint_contract: str = PATH_INVENTORY_FINGERPRINT_CONTRACT
    license_status: str = "VERIFIED_FROM_MODEL_CARD_METADATA"
    license_spdx: str = "Apache-2.0"
    license_source_file: str = "README.md"

    @property
    def upstream_source(self) -> str:
        return self.upstream_source_override or f"PaddlePaddle/{self.directory}"

    @property
    def license(self) -> dict[str, str]:
        return {
            "status": self.license_status,
            "spdx": self.license_spdx,
            "source_file": self.license_source_file,
        }


_PADDLE_MODEL_SPECS = (
    ModelSpec(
        "pp-doclayout-plus-l",
        "PP-DocLayout_plus-L",
        "layout",
        "layout_detection",
        "aa52b8528c84f9b1a34ac3a88fe0e576edb9d11d",
        ("PDF_LAYOUT", "PDF_TEXT", "PDF_FORMULA", "PDF_TABLE", "PDF_IMAGE"),
    ),
    ModelSpec(
        "pp-ocrv6-medium-det",
        "PP-OCRv6_medium_det",
        "text",
        "text_detection",
        "8e0f56fb2ef86b461d99cfc7ac5c137738985f61",
        ("PDF_TEXT", "PDF_TABLE"),
    ),
    ModelSpec(
        "pp-ocrv6-medium-rec",
        "PP-OCRv6_medium_rec",
        "text",
        "text_recognition",
        "e5a92bcbc5cc1b494628e458d267778f0704fd7c",
        ("PDF_TEXT", "PDF_TABLE"),
    ),
    ModelSpec(
        MODEL_ID,
        MODEL_DIRECTORY,
        "formula",
        "formula_ocr",
        MODEL_REVISION,
        ("DOCX_FORMULA", "PDF_FORMULA"),
        "INTEGRATED_VALIDATED",
    ),
    ModelSpec(
        "pp-lcnet-x1-0-table-cls",
        "PP-LCNet_x1_0_table_cls",
        "table",
        "table_classification",
        "2fa6323e7dab88fa883081db1460995f46af2922",
        ("PDF_TABLE",),
    ),
    ModelSpec(
        "slanext-wired",
        "SLANeXt_wired",
        "table",
        "table_structure_recognition",
        "763069fcda6a065f2171753205a32bf899a88d15",
        ("PDF_TABLE",),
    ),
    ModelSpec(
        "slanext-wireless",
        "SLANeXt_wireless",
        "table",
        "table_structure_recognition",
        "1b73e05c752d2e0763b5d205846dec23122e5808",
        ("PDF_TABLE",),
    ),
    ModelSpec(
        "rt-detr-l-wired-table-cell-det",
        "RT-DETR-L_wired_table_cell_det",
        "table",
        "table_cell_detection",
        "e2bd53c06b3a815d86acbf5c6779dada58819cfe",
        ("PDF_TABLE",),
    ),
    ModelSpec(
        "rt-detr-l-wireless-table-cell-det",
        "RT-DETR-L_wireless_table_cell_det",
        "table",
        "table_cell_detection",
        "25ca86356a601c877476bb0dcc5fd09153d9d64d",
        ("PDF_TABLE",),
    ),
)

PADDLE_MODEL_CATALOG = {spec.model_id: spec for spec in _PADDLE_MODEL_SPECS}

_PDF_EXTENSION_MODEL_SPECS = (
    ModelSpec(
        "ch-svtrv2-rec",
        "ch_SVTRv2_rec",
        "text",
        "text_recognition",
        "UNRESOLVED",
        ("PDF_TEXT",),
        "INTEGRATED_VALIDATED",
        fingerprint_contract=CANONICAL_JSON_INVENTORY_FINGERPRINT_CONTRACT,
        license_status="UNRESOLVED",
        license_spdx="UNRESOLVED",
        license_source_file="UNRESOLVED",
    ),
    ModelSpec(
        "got-ocr2-0",
        "GOT-OCR2.0",
        "text",
        "text_recognition",
        "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
        ("PDF_TEXT",),
        "INTEGRATED_VALIDATED",
        provider="Torch/Transformers",
        upstream_source_override="stepfun-ai/GOT-OCR-2.0-hf",
        precision_baseline="bf16",
        license_status="UNRESOLVED",
        license_spdx="UNRESOLVED",
        license_source_file="UNRESOLVED",
    ),
)

MODEL_CATALOG = {
    spec.model_id: spec for spec in (*_PADDLE_MODEL_SPECS, *_PDF_EXTENSION_MODEL_SPECS)
}


class ModelRegistryError(RuntimeError):
    pass


class ModelNotFoundError(ModelRegistryError):
    pass


class ModelManifestError(ModelRegistryError):
    pass


class ModelFingerprintMismatch(ModelManifestError):
    pass


@dataclass(frozen=True)
class ModelResolution:
    model_id: str
    models_root: Path
    model_root: Path
    manifest: dict[str, Any]
    fingerprint_verified: bool
    resolution_source: str
    readiness: str
    verified_inventory: dict[str, Any] | None = None


def resolve_models_root(
    explicit: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    mcp_root: str | Path | None = None,
    allow_developer_fallback: bool = False,
) -> Path:
    """Resolve explicit > environment > config > MCP root > opt-in dev cache."""

    root, _source = _resolve_models_root_with_source(
        explicit,
        config_path=config_path,
        mcp_root=mcp_root,
        allow_developer_fallback=allow_developer_fallback,
    )
    return root


def _resolve_models_root_with_source(
    explicit: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    mcp_root: str | Path | None = None,
    allow_developer_fallback: bool = False,
) -> tuple[Path, str]:
    if explicit is not None:
        return Path(explicit).expanduser().resolve(), "explicit"
    environment = os.environ.get(MODELS_ROOT_ENV)
    if environment:
        return Path(environment).expanduser().resolve(), "environment"

    config, payload = load_workspace_config(config_path)
    workspace = payload.get("workspace", {}) if isinstance(payload, dict) else {}
    configured = workspace.get("models_root") if isinstance(workspace, dict) else None
    if configured is not None:
        if not isinstance(configured, str) or not configured.strip():
            raise ModelRegistryError(
                f"workspace.models_root must be a non-empty string in {config}"
            )
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute() and config is not None:
            candidate = config.parent / candidate
        return candidate.resolve(), "configuration"

    resolved_mcp = mcp_root or os.environ.get(MCP_ROOT_ENV)
    if resolved_mcp:
        return (Path(resolved_mcp).expanduser() / "MODELS").resolve(), "mcp_root"

    if allow_developer_fallback and os.environ.get(DEVELOPER_FALLBACK_ENV) == "1":
        return (
            Path.home() / ".cache" / "bemarkdown" / "phase3a" / "paddlex" / "official_models"
        ).resolve(), "developer_fallback"
    raise ModelNotFoundError(
        "No models root is configured; use an explicit root, "
        "BEMARKDOWN_MODELS_ROOT, workspace.models_root, or BEMARKDOWN_MCP_ROOT"
    )


class ModelRegistry:
    """Manifest-driven, local-only model resolver for immutable MCP model assets."""

    def __init__(
        self,
        *,
        models_root: str | Path | None = None,
        config_path: str | Path | None = None,
        mcp_root: str | Path | None = None,
        allow_developer_fallback: bool = False,
    ):
        self.models_root, self.resolution_source = _resolve_models_root_with_source(
            models_root,
            config_path=config_path,
            mcp_root=mcp_root,
            allow_developer_fallback=allow_developer_fallback,
        )

    def resolve(self, model_id: str = MODEL_ID, *, deep: bool = True) -> ModelResolution:
        spec = MODEL_CATALOG.get(model_id)
        if spec is None:
            raise ModelNotFoundError(f"Unknown BeMarkdown model_id: {model_id}")
        model_root = self.models_root / spec.directory
        if not model_root.is_dir():
            raise ModelNotFoundError(
                f"Model {model_id} is missing under the configured MODELS root"
            )
        manifest_path = model_root / MODEL_MANIFEST_NAME
        if not manifest_path.is_file():
            raise ModelManifestError(f"Missing {MODEL_MANIFEST_NAME} for {model_id}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelManifestError(f"Invalid model manifest: {exc}") from exc
        _validate_manifest_shape(manifest, spec)
        _validate_quick_inventory(model_root, manifest)
        verified = False
        actual = None
        if deep:
            actual = model_inventory(model_root)
            expected = manifest["model_fingerprint"]
            actual_fingerprint = model_inventory_fingerprint(
                actual, manifest["fingerprint_contract"]
            )
            if actual_fingerprint != expected:
                raise ModelFingerprintMismatch(
                    "MODEL_FINGERPRINT_MISMATCH: actual model payload does not match "
                    f"{MODEL_MANIFEST_NAME}"
                )
            if actual["files"] != manifest["files"]:
                raise ModelFingerprintMismatch(
                    "MODEL_FINGERPRINT_MISMATCH: model file inventory differs"
                )
            verified = True
        return ModelResolution(
            model_id,
            self.models_root,
            model_root,
            manifest,
            verified,
            self.resolution_source,
            spec.integration_status,
            actual,
        )


def build_model_manifest(
    model_root: str | Path,
    *,
    model_id: str = MODEL_ID,
    upstream_revision: str | None = None,
    runtime_profile: str = MODEL_RUNTIME_PROFILE,
) -> dict[str, Any]:
    spec = MODEL_CATALOG.get(model_id)
    if spec is None:
        raise ModelNotFoundError(f"Unknown BeMarkdown model_id: {model_id}")
    upstream_revision = upstream_revision or spec.upstream_revision
    if upstream_revision != spec.upstream_revision:
        raise ModelManifestError(f"Model revision is not frozen for {model_id}")
    inventory = model_inventory(Path(model_root))
    model_fingerprint = model_inventory_fingerprint(
        inventory, spec.fingerprint_contract
    )
    return {
        "schema": MODEL_MANIFEST_SCHEMA,
        "manifest_version": MODEL_MANIFEST_SCHEMA,
        "model_id": model_id,
        "display_name": spec.directory,
        "engine": spec.engine,
        "task": spec.task,
        "provider": spec.provider,
        "upstream": spec.upstream_source,
        "upstream_source": spec.upstream_source,
        "upstream_revision": upstream_revision,
        "model_fingerprint": model_fingerprint,
        "fingerprint_contract": spec.fingerprint_contract,
        "runtime_contract": spec.runtime_contract,
        "runtime_profile": runtime_profile,
        "expected_device": spec.expected_device,
        "precision_baseline": spec.precision_baseline,
        "integration_status": spec.integration_status,
        "license": spec.license,
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "files": inventory["files"],
    }


def model_inventory(model_root: str | Path) -> dict[str, Any]:
    model_root = Path(model_root)
    if not model_root.is_dir():
        raise ModelNotFoundError(f"Model directory is missing: {model_root}")
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    paths = (item for item in model_root.rglob("*") if item.is_file())
    for path in sorted(paths, key=lambda item: _inventory_sort_key(model_root, item)):
        relative = path.relative_to(model_root).as_posix()
        if relative == MODEL_MANIFEST_NAME:
            continue
        _validate_relative_path(relative)
        if path.stat().st_size == 0:
            raise ModelManifestError(f"Model inventory contains zero-byte file: {relative}")
        file_sha = _sha256_file(path)
        record = {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha}
        records.append(record)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return {
        "directory_sha256": digest.hexdigest(),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }


def _validate_manifest_shape(manifest: dict[str, Any], spec: ModelSpec) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != MODEL_MANIFEST_SCHEMA:
        raise ModelManifestError("Unsupported model manifest schema")
    if manifest.get("model_id") != spec.model_id:
        raise ModelManifestError("Model manifest model_id is not supported")
    if manifest.get("upstream_revision") != spec.upstream_revision:
        raise ModelManifestError("Model manifest upstream revision is not frozen")
    if manifest.get("provider") != spec.provider:
        raise ModelManifestError("Model manifest provider is not frozen")
    legacy_paddle_manifest = (
        spec.provider == "PaddlePaddle" and manifest.get("manifest_version") is None
    )
    upstream_source = manifest.get("upstream_source")
    if manifest.get("upstream") != spec.upstream_source or (
        upstream_source is not None and upstream_source != spec.upstream_source
    ):
        raise ModelManifestError("Model manifest upstream source is not frozen")
    runtime_contract = manifest.get("runtime_contract")
    if runtime_contract != spec.runtime_contract and not (
        legacy_paddle_manifest and runtime_contract is None
    ):
        raise ModelManifestError("Model manifest runtime contract is not frozen")
    if manifest.get("fingerprint_contract") != spec.fingerprint_contract:
        raise ModelManifestError("Model manifest fingerprint contract is not frozen")
    runtime_profile = manifest.get("runtime_profile")
    if not isinstance(runtime_profile, str) or not runtime_profile:
        raise ModelManifestError("Model manifest runtime profile is invalid")
    expected_device = manifest.get("expected_device")
    if expected_device != spec.expected_device and not (
        legacy_paddle_manifest and expected_device is None
    ):
        raise ModelManifestError("Model manifest expected device is not frozen")
    precision_baseline = manifest.get("precision_baseline")
    if precision_baseline != spec.precision_baseline and not (
        legacy_paddle_manifest and precision_baseline is None
    ):
        raise ModelManifestError("Model manifest precision baseline is not frozen")
    if manifest.get("license") != spec.license:
        raise ModelManifestError("Model manifest license identity is not frozen")
    fingerprint = manifest.get("model_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ModelManifestError("Model manifest fingerprint is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ModelManifestError("Model manifest file inventory is empty")


def _validate_quick_inventory(model_root: Path, manifest: dict[str, Any]) -> None:
    files = manifest["files"]
    if manifest.get("file_count") != len(files):
        raise ModelManifestError("Model manifest file_count does not match inventory")
    total = 0
    paths: list[str] = []
    for row in files:
        if not isinstance(row, dict):
            raise ModelManifestError("Model manifest file row is invalid")
        relative = row.get("path")
        if not isinstance(relative, str):
            raise ModelManifestError("Model manifest file path is invalid")
        _validate_relative_path(relative)
        path = model_root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.stat().st_size != row.get("bytes"):
            raise ModelFingerprintMismatch(
                f"MODEL_FINGERPRINT_MISMATCH: missing or resized model file {relative}"
            )
        if path.stat().st_size == 0:
            raise ModelManifestError(f"Model manifest contains zero-byte file: {relative}")
        total += path.stat().st_size
        paths.append(relative)
    if paths != sorted(paths, key=_manifest_path_sort_key) or len(paths) != len(set(paths)):
        raise ModelManifestError("Model manifest paths are not unique and sorted")
    if total != manifest.get("total_bytes"):
        raise ModelFingerprintMismatch(
            "MODEL_FINGERPRINT_MISMATCH: model byte total differs"
        )


def model_inventory_fingerprint(
    inventory: dict[str, Any], fingerprint_contract: str
) -> str:
    if fingerprint_contract == PATH_INVENTORY_FINGERPRINT_CONTRACT:
        return str(inventory["directory_sha256"])
    if fingerprint_contract == CANONICAL_JSON_INVENTORY_FINGERPRINT_CONTRACT:
        canonical = json.dumps(
            inventory["files"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    raise ModelManifestError("Unsupported model fingerprint contract")


def _validate_relative_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or "\\" in relative or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ModelManifestError(f"Unsafe model file path: {relative!r}")


def _inventory_sort_key(model_root: Path, path: Path) -> tuple[str, str]:
    return _manifest_path_sort_key(path.relative_to(model_root).as_posix())


def _manifest_path_sort_key(relative: str) -> tuple[str, str]:
    """Keep inventory order portable while preserving the frozen Windows fingerprint."""

    return relative.casefold(), relative


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
