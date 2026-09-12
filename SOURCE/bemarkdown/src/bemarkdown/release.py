from __future__ import annotations

import ast
import hashlib
import json
import shutil
import zipfile
from collections import deque
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .asset_contract import ASSET_MANIFEST_VERSION
from .production import PACKAGE_CONTRACT

TOOL_MANIFEST_SCHEMA = "bemarkdown-tool-manifest-v1"
RUNTIME_CONTRACT = "bemarkdown-runtime-contract-v1"
RELEASE_CHANNEL = "rc"
PUBLICATION_RC_MANIFEST_NAME = "PUBLICATION_RC_MANIFEST.json"
PUBLICATION_RC_MANIFEST_SCHEMA = "bemarkdown-pdf-publication-rc-manifest-v1"
DEFAULT_RUNTIME_ENTRY_MODULES = (
    "bemarkdown",
    "bemarkdown.__main__",
    "bemarkdown.cli",
    "bemarkdown.production",
    "bemarkdown.pdf.workers.ch_svtrv2_rec",
    "bemarkdown.pdf.workers.got_ocr2",
)


def build_tool_manifest(
    wheel_path: str | Path,
    runtime_manifest_path: str | Path,
    core_lock_path: str | Path,
    formula_lock_path: str | Path,
    *,
    build_commit: str,
    model_suite: dict[str, Any] | None = None,
    pdf_model_suite: dict[str, Any] | None = None,
    text_ensemble_lock_path: str | Path | None = None,
    wheel_import_closure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wheel_path = Path(wheel_path)
    runtime_manifest_path = Path(runtime_manifest_path)
    core_lock_path = Path(core_lock_path)
    formula_lock_path = Path(formula_lock_path)
    if len(build_commit) != 40 or any(value not in "0123456789abcdef" for value in build_commit):
        raise ValueError("build_commit must be a lowercase full Git SHA")
    manifest = {
        "schema": TOOL_MANIFEST_SCHEMA,
        "tool_id": "bemarkdown",
        "tool_version": __version__,
        "release_channel": RELEASE_CHANNEL,
        "entrypoint": "bemarkdown",
        "python_api": ["convert_document", "convert_documents", "validate_package"],
        "cli": ["convert", "convert-batch", "validate", "cleanup-staging", "doctor"],
        "package_contract": PACKAGE_CONTRACT,
        "asset_contract": ASSET_MANIFEST_VERSION,
        "supported_sources": ["DOCX", "PDF"],
        "capabilities": {
            "docx": True,
            "pdf": {
                "publication_status": "PUBLICATION_CANDIDATE",
                "independent_quality_validation": "PENDING_ZCODE",
            },
            "wmf_render": {"platform": "win32-gdi", "required_for_full_quality": True},
            "formula_ocr": {
                "model_id": "pp-formulanet-plus-l",
                "required_for_full_quality": True,
                "cpu_fallback": False,
            },
        },
        "runtime_contract": RUNTIME_CONTRACT,
        "python_compatibility": "CPython 3.11.x",
        "validated_platform": "Windows x64",
        "build_commit": build_commit,
        "wheel": {
            "path": f"dist/{wheel_path.name}",
            "sha256": sha256_file(wheel_path),
            "bytes": wheel_path.stat().st_size,
            "content_inventory": wheel_content_inventory(wheel_path),
        },
        "runtime": {
            "manifest": {
                "path": "runtime/RUNTIME_MANIFEST.json",
                "sha256": sha256_file(runtime_manifest_path),
            },
            "core_lock": {
                "path": "runtime/requirements-core.lock",
                "sha256": sha256_file(core_lock_path),
            },
            "formula_ocr_lock": {
                "path": "runtime/requirements-formula-ocr.lock",
                "sha256": sha256_file(formula_lock_path),
            },
        },
    }
    if text_ensemble_lock_path is not None:
        text_ensemble_lock_path = Path(text_ensemble_lock_path)
        manifest["runtime"]["text_ensemble_lock"] = {
            "path": "runtime/requirements-text-ensemble.lock",
            "sha256": sha256_file(text_ensemble_lock_path),
        }
    if wheel_import_closure is not None:
        if wheel_import_closure.get("gate") != "PASS":
            raise ValueError("wheel runtime import closure is not PASS")
        manifest["wheel"]["runtime_import_closure"] = wheel_import_closure
    if model_suite is not None:
        suite_id = model_suite.get("suite_id")
        fingerprint = model_suite.get("suite_fingerprint")
        count = model_suite.get("model_count")
        if (
            suite_id != "bemarkdown-paddle-document-suite-v1"
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or count != 9
        ):
            raise ValueError("model_suite identity is invalid")
        manifest["capabilities"]["paddle_model_suite"] = {
            "suite_id": suite_id,
            "suite_fingerprint": fingerprint,
            "model_count": count,
            "status": "MODEL_READY",
            "pdf_pipeline_ready": False,
        }
    if pdf_model_suite is not None:
        suite_id = pdf_model_suite.get("suite_id")
        fingerprint = pdf_model_suite.get("suite_fingerprint")
        count = pdf_model_suite.get("model_count")
        if (
            suite_id != "bemarkdown-pdf-production-suite-v1"
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or count != 11
        ):
            raise ValueError("pdf_model_suite identity is invalid")
        if model_suite is None:
            raise ValueError("pdf_model_suite requires the legacy Paddle suite")
        manifest["capabilities"]["pdf_production_suite"] = {
            "suite_id": suite_id,
            "suite_fingerprint": fingerprint,
            "model_count": count,
            "status": "MODEL_READY",
            "pdf_pipeline_ready": True,
            "independent_quality_validation": "PENDING_ZCODE",
        }
    return manifest


def copy_exact_model_payload(
    source_root: str | Path,
    target_root: str | Path,
    *,
    required_root_files: Sequence[str],
    allowed_source_directories: Sequence[str] = (),
) -> list[str]:
    """Copy an exact frozen root-file closure and omit allowed host cache trees."""

    source_root = Path(source_root).resolve()
    target_root = Path(target_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    required = list(required_root_files)
    if len(required) != len(set(required)) or any(
        PurePosixPath(name).name != name or name in {"", ".", ".."}
        for name in required
    ):
        raise ValueError("required_root_files must be unique root file names")
    observed_files = sorted(path.name for path in source_root.iterdir() if path.is_file())
    missing = sorted(set(required) - set(observed_files))
    unexpected = sorted(set(observed_files) - set(required))
    if missing:
        raise ValueError(f"model payload is missing frozen root files: {missing}")
    if unexpected:
        raise ValueError(f"model payload has unexpected root files: {unexpected}")
    observed_directories = sorted(
        path.name for path in source_root.iterdir() if path.is_dir()
    )
    unexpected_directories = sorted(
        set(observed_directories) - set(allowed_source_directories)
    )
    if unexpected_directories:
        raise ValueError(
            "model payload has unexpected source directories: "
            f"{unexpected_directories}"
        )
    if target_root.exists():
        raise FileExistsError(target_root)
    target_root.mkdir(parents=True)
    for name in required:
        shutil.copy2(source_root / name, target_root / name)
    return required


def scan_wheel_runtime_import_closure(
    wheel_path: str | Path,
    *,
    entry_modules: Sequence[str] = DEFAULT_RUNTIME_ENTRY_MODULES,
) -> dict[str, Any]:
    """Follow eager in-wheel imports from production entry modules."""

    wheel_path = Path(wheel_path)
    sources: dict[str, str] = {}
    packages: set[str] = set()
    parse_errors: list[str] = []
    with zipfile.ZipFile(wheel_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".py"):
                continue
            module, is_package = _wheel_module_name(info.filename)
            if module is None:
                continue
            try:
                sources[module] = archive.read(info).decode("utf-8")
            except UnicodeDecodeError:
                parse_errors.append(module)
            if is_package:
                packages.add(module)

    missing_entries = sorted(set(entry_modules) - set(sources))
    queue = deque(module for module in entry_modules if module in sources)
    reachable: set[str] = set()
    edges: dict[str, list[str]] = {}
    while queue:
        module = queue.popleft()
        if module in reachable:
            continue
        reachable.add(module)
        try:
            tree = ast.parse(sources[module], filename=module)
        except SyntaxError:
            parse_errors.append(module)
            continue
        imports = sorted(_internal_module_imports(module, tree, sources, packages))
        edges[module] = imports
        queue.extend(value for value in imports if value not in reachable)

    forbidden = sorted(module for module in reachable if _is_forbidden_runtime_module(module))
    parse_errors = sorted(set(parse_errors))
    gate = "PASS" if not (missing_entries or parse_errors or forbidden) else "FAIL"
    identity = {
        "entry_modules": list(entry_modules),
        "reachable_modules": sorted(reachable),
        "forbidden_reachable_modules": forbidden,
        "missing_entry_modules": missing_entries,
        "parse_error_modules": parse_errors,
        "import_edges": edges,
    }
    return {
        "schema": "bemarkdown-wheel-runtime-import-closure-v1",
        **identity,
        "reachable_module_count": len(reachable),
        "closure_fingerprint": _semantic_sha256(identity),
        "gate": gate,
    }


def build_publication_rc_manifest(
    rc_root: str | Path,
    *,
    build_commit: str,
    expected_formal_mcp_commit: str,
    expected_formal_registry_fingerprint: str,
    managed_model_ids: Sequence[str],
) -> dict[str, Any]:
    """Build the self-excluding immutable inventory for a PDF publication RC."""

    rc_root = Path(rc_root).resolve()
    for name, value, length in (
        ("build_commit", build_commit, 40),
        ("expected_formal_mcp_commit", expected_formal_mcp_commit, 40),
        (
            "expected_formal_registry_fingerprint",
            expected_formal_registry_fingerprint,
            64,
        ),
    ):
        _require_lower_hex(name, value, length)
    managed = list(managed_model_ids)
    if managed != ["ch-svtrv2-rec", "got-ocr2-0"]:
        raise ValueError("managed_model_ids must be the frozen B/C model pair")

    tool_root = rc_root / "TOOLS" / "bemarkdown"
    tool_manifest_path = tool_root / "TOOL_MANIFEST.json"
    legacy_suite_path = tool_root / "runtime" / "model_suite.json"
    pdf_suite_path = tool_root / "runtime" / "pdf_production_suite.json"
    tool = _read_json(tool_manifest_path)
    legacy = _read_json(legacy_suite_path)
    pdf_suite = _read_json(pdf_suite_path)
    wheel_path = tool_root / str(tool["wheel"]["path"])
    if sha256_file(wheel_path) != tool["wheel"].get("sha256"):
        raise ValueError("Tool manifest wheel SHA does not match the RC wheel")
    if (
        legacy.get("suite_id") != "bemarkdown-paddle-document-suite-v1"
        or legacy.get("model_count") != 9
    ):
        raise ValueError("legacy suite identity is invalid")
    if (
        pdf_suite.get("suite_id") != "bemarkdown-pdf-production-suite-v1"
        or pdf_suite.get("model_count") != 11
    ):
        raise ValueError("PDF production suite identity is invalid")

    files = _file_inventory_excluding(rc_root, {PUBLICATION_RC_MANIFEST_NAME})
    managed_root_documents = ["README.md"] if (rc_root / "README.md").is_file() else []
    return {
        "schema": PUBLICATION_RC_MANIFEST_SCHEMA,
        "build_commit": build_commit,
        "expected_formal_mcp_commit": expected_formal_mcp_commit,
        "expected_formal_registry_fingerprint": expected_formal_registry_fingerprint,
        "tool_manifest_sha256": sha256_file(tool_manifest_path),
        "wheel_sha256": sha256_file(wheel_path),
        "legacy_suite_sha256": sha256_file(legacy_suite_path),
        "pdf_suite_sha256": sha256_file(pdf_suite_path),
        "pdf_suite_id": pdf_suite["suite_id"],
        "pdf_suite_fingerprint": pdf_suite["suite_fingerprint"],
        "pdf_suite_model_count": pdf_suite["model_count"],
        "managed_model_ids": managed,
        "managed_root_documents": managed_root_documents,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
        "inventory_fingerprint": _inventory_fingerprint(files),
        "self_entry_excluded": True,
    }


def validate_publication_rc_manifest(
    rc_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    rc_root = Path(rc_root).resolve()
    path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else rc_root / PUBLICATION_RC_MANIFEST_NAME
    )
    payload = _read_json(path)
    if payload.get("schema") != PUBLICATION_RC_MANIFEST_SCHEMA:
        raise ValueError("publication RC manifest schema is invalid")
    files = _file_inventory_excluding(rc_root, {path.relative_to(rc_root).as_posix()})
    if files != payload.get("files"):
        raise ValueError("publication RC inventory does not match immutable files")
    if _inventory_fingerprint(files) != payload.get("inventory_fingerprint"):
        raise ValueError("publication RC inventory fingerprint is invalid")
    if payload.get("file_count") != len(files) or payload.get("total_bytes") != sum(
        row["bytes"] for row in files
    ):
        raise ValueError("publication RC inventory totals are invalid")
    return payload


def _wheel_module_name(path: str) -> tuple[str | None, bool]:
    pure = PurePosixPath(path)
    if not pure.parts or pure.parts[0] != "bemarkdown" or pure.suffix != ".py":
        return None, False
    parts = list(pure.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _internal_module_imports(
    current: str,
    tree: ast.AST,
    sources: dict[str, str],
    packages: set[str],
) -> set[str]:
    imports: set[str] = set()
    current_package = current if current in packages else current.rpartition(".")[0]
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = current_package.split(".") if current_package else []
                ascend = node.level - 1
                if ascend > len(base_parts):
                    continue
                if ascend:
                    base_parts = base_parts[:-ascend]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                candidates.append(base)
            candidates.extend(
                f"{base}.{alias.name}" if base else alias.name
                for alias in node.names
                if alias.name != "*"
            )
        for candidate in candidates:
            if candidate in sources:
                imports.add(candidate)
    return imports


def _is_forbidden_runtime_module(module: str) -> bool:
    parts = module.split(".")[1:]
    forbidden_exact = {
        "artifacts",
        "docx_census",
        "large_e2e_validation",
        "reference_evaluation",
        "scripts",
        "tests",
        "tmp",
        "vision_reference_evaluation",
        "vision_reference_gate",
    }
    return any(
        part in forbidden_exact
        or "benchmark" in part
        or part.endswith("_validation")
        for part in parts
    )


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_lower_hex(name: str, value: str, length: int) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a {length}-character lowercase hex value")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON payload: {path.name}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path.name}")
    return payload


def _file_inventory_excluding(
    root: Path,
    excluded_relative_paths: Iterable[str],
) -> list[dict[str, Any]]:
    excluded = set(excluded_relative_paths)
    rows = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _inventory_fingerprint(files: list[dict[str, Any]]) -> str:
    return _semantic_sha256(files)


def wheel_content_inventory(wheel_path: str | Path) -> dict[str, Any]:
    wheel_path = Path(wheel_path)
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with zipfile.ZipFile(wheel_path) as archive:
        for info in sorted(archive.infolist(), key=lambda value: value.filename):
            if info.is_dir():
                continue
            payload = archive.read(info)
            file_sha = hashlib.sha256(payload).hexdigest()
            record = {"path": info.filename, "bytes": len(payload), "sha256": file_sha}
            records.append(record)
            digest.update(info.filename.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_sha.encode("ascii"))
            digest.update(b"\n")
    return {
        "content_sha256": digest.hexdigest(),
        "file_count": len(records),
        "total_uncompressed_bytes": sum(row["bytes"] for row in records),
        "files": records,
    }


def scan_development_path_leaks(
    root: str | Path,
    forbidden_text: str | None = None,
    *,
    wheel_runtime_modules: Iterable[str] | None = None,
) -> list[str]:
    root = Path(root)
    if forbidden_text is None:
        forbidden_text = "Education_Knowledge_Base_" + "DEVELOPER"
    needle = forbidden_text.casefold().encode("utf-8")
    runtime_modules = (
        None if wheel_runtime_modules is None else set(wheel_runtime_modules)
    )
    leaks: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".whl":
            with zipfile.ZipFile(path) as archive:
                if any(
                    needle in archive.read(info).lower()
                    for info in archive.infolist()
                    if not info.is_dir()
                    and info.file_size <= 16 * 1024 * 1024
                    and _wheel_member_is_in_runtime_scope(info.filename, runtime_modules)
                ):
                    leaks.append(relative)
            continue
        if path.stat().st_size > 16 * 1024 * 1024:
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if needle in payload.lower():
            leaks.append(relative)
    return leaks


def _wheel_member_is_in_runtime_scope(
    path: str,
    runtime_modules: set[str] | None,
) -> bool:
    if runtime_modules is None or not path.endswith(".py"):
        return True
    module, _is_package = _wheel_module_name(path)
    return module in runtime_modules


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
