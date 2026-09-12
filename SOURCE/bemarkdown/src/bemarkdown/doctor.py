from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .asset_contract import ASSET_MANIFEST_VERSION
from .config import resolve_output_root
from .model_registry import (
    MODEL_ID,
    PADDLE_MODEL_CATALOG,
    ModelRegistry,
    ModelRegistryError,
)
from .model_suite import ModelSuiteManifestError, validate_model_suite
from .mtef_cache import MtefCacheContext
from .production import PACKAGE_CONTRACT
from .release import RUNTIME_CONTRACT, sha256_file
from .wmf import WmfInspector

DOCTOR_SCHEMA = "bemarkdown-doctor-v1"
READINESS = {"READY_FULL", "READY_DEGRADED", "NOT_READY"}
MATHTYPEJX_REVISION = "7d90e7274c85cf56ac28d4d15e593044693d7e70"
CORE_DEPENDENCIES = (
    "beautifulsoup4",
    "lxml",
    "mathml2latex",
    "mathtypejx",
    "olefile",
    "Pillow",
)


@dataclass(frozen=True)
class DoctorResult:
    schema: str
    readiness: str
    capabilities: dict[str, dict[str, Any]]
    capability_readiness: dict[str, str]
    runtime: dict[str, Any]
    timings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_doctor(
    *,
    models_root: str | Path | None = None,
    output_root: str | Path | None = None,
    tool_root: str | Path | None = None,
    config_path: str | Path | None = None,
    mcp_root: str | Path | None = None,
    deep: bool = False,
    probe_formula_runtime: bool = True,
    probe_model_suite_runtime: bool = True,
) -> DoctorResult:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    capabilities: dict[str, dict[str, Any]] = {}

    capabilities["python"] = _python_check()
    capabilities["core_dependencies"] = _core_dependencies_check()
    capabilities["package_contract"] = _check(
        "PASS", PACKAGE_CONTRACT, "bemarkdown-package-v1", "Frozen package contract"
    )
    capabilities["asset_contract"] = _check(
        "PASS",
        ASSET_MANIFEST_VERSION,
        "bemarkdown-asset-contract-v1",
        "Frozen asset contract",
    )
    capabilities["mtef"] = _mtef_check(tool_root)
    capabilities["wmf_inspector"] = _wmf_inspector_check()
    capabilities["wmf_render"] = _wmf_render_check()
    capabilities["tool_payload"] = _tool_payload_check(tool_root)
    capabilities["output_root"] = _output_root_check(output_root, config_path)

    core_parts = (
        "python",
        "core_dependencies",
        "package_contract",
        "asset_contract",
        "mtef",
        "wmf_inspector",
    )
    core_failures = [
        name for name in core_parts if capabilities[name]["status"] != "PASS"
    ]
    capabilities["docx_core"] = _check(
        "FAIL" if core_failures else "PASS",
        core_failures or "all core checks passed",
        "all core checks PASS",
        "Core DOCX runtime is unavailable" if core_failures else "Core DOCX runtime is ready",
    )

    model_started = time.perf_counter()
    try:
        resolution = ModelRegistry(
            models_root=models_root,
            config_path=config_path,
            mcp_root=mcp_root,
        ).resolve(MODEL_ID, deep=deep)
        capabilities["formula_ocr_model"] = _check(
            "PASS",
            {
                "model_id": resolution.model_id,
                "directory": resolution.model_root.name,
                "fingerprint": resolution.manifest["model_fingerprint"],
                "fingerprint_verified": resolution.fingerprint_verified,
                "resolution_source": resolution.resolution_source,
            },
            {"model_id": MODEL_ID, "local_only": True},
            "Local model manifest and inventory are valid",
        )
    except ModelRegistryError as exc:
        capabilities["formula_ocr_model"] = _check(
            "FAIL", None, {"model_id": MODEL_ID, "local_only": True}, str(exc)
        )
    timings["model_check_seconds"] = time.perf_counter() - model_started

    runtime_started = time.perf_counter()
    capabilities["formula_ocr_runtime"] = (
        _formula_runtime_check()
        if probe_formula_runtime
        else _check("SKIP", None, "Paddle GPU probe", "Formula runtime probe skipped")
    )
    timings["formula_runtime_probe_seconds"] = time.perf_counter() - runtime_started

    suite_started = time.perf_counter()
    _add_model_suite_checks(
        capabilities,
        models_root=models_root,
        tool_root=tool_root,
        config_path=config_path,
        mcp_root=mcp_root,
        deep=deep,
        probe_runtime=probe_model_suite_runtime,
    )
    timings["model_suite_check_seconds"] = time.perf_counter() - suite_started

    readiness = derive_readiness(capabilities)
    capability_readiness = derive_capability_readiness(capabilities, readiness)
    timings["total_seconds"] = time.perf_counter() - started
    runtime = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "runtime_contract": RUNTIME_CONTRACT,
        "formula_cpu_fallback": False,
        "network_model_download": False,
    }
    return DoctorResult(
        DOCTOR_SCHEMA,
        readiness,
        capabilities,
        capability_readiness,
        runtime,
        timings,
    )


def derive_readiness(capabilities: dict[str, dict[str, Any]]) -> str:
    mandatory = ("docx_core", "output_root", "wmf_render")
    if any(capabilities.get(name, {}).get("status") != "PASS" for name in mandatory):
        return "NOT_READY"
    formula = ("formula_ocr_runtime", "formula_ocr_model")
    if all(capabilities.get(name, {}).get("status") == "PASS" for name in formula):
        return "READY_FULL"
    return "READY_DEGRADED"


def derive_capability_readiness(
    capabilities: dict[str, dict[str, Any]],
    tool_readiness: str,
) -> dict[str, str]:
    if tool_readiness == "READY_FULL":
        docx = "READY"
    elif tool_readiness == "READY_DEGRADED":
        docx = "READY_DEGRADED"
    else:
        docx = "NOT_READY"

    result = {"DOCX": docx}
    runtime_ready = capabilities.get("formula_ocr_runtime", {}).get("status") == "PASS"
    for capability in ("PDF_LAYOUT", "PDF_TEXT", "PDF_FORMULA", "PDF_TABLE"):
        required = [
            model_id
            for model_id, spec in PADDLE_MODEL_CATALOG.items()
            if capability in spec.required_by
        ]
        models_ready = all(
            capabilities.get(f"paddle_model:{model_id}", {}).get("status") == "PASS"
            for model_id in required
        )
        result[capability] = (
            "MODEL_READY" if runtime_ready and models_ready else "MODEL_NOT_READY"
        )
    return result


def _add_model_suite_checks(
    capabilities: dict[str, dict[str, Any]],
    *,
    models_root: str | Path | None,
    tool_root: str | Path | None,
    config_path: str | Path | None,
    mcp_root: str | Path | None,
    deep: bool,
    probe_runtime: bool,
) -> None:
    try:
        registry = ModelRegistry(
            models_root=models_root,
            config_path=config_path,
            mcp_root=mcp_root,
        )
    except ModelRegistryError as exc:
        capabilities["paddle_model_suite"] = _check(
            "FAIL", None, {"model_count": 9}, str(exc)
        )
        for model_id in PADDLE_MODEL_CATALOG:
            capabilities[f"paddle_model:{model_id}"] = _check(
                "FAIL", None, {"model_id": model_id}, str(exc)
            )
        return

    resolutions = {}
    for model_id in PADDLE_MODEL_CATALOG:
        try:
            resolution = registry.resolve(model_id, deep=deep)
            resolutions[model_id] = resolution
            capabilities[f"paddle_model:{model_id}"] = _check(
                "PASS",
                {
                    "model_id": model_id,
                    "directory": resolution.model_root.name,
                    "fingerprint": resolution.manifest["model_fingerprint"],
                    "fingerprint_verified": resolution.fingerprint_verified,
                    "integration_status": resolution.readiness,
                },
                {"model_id": model_id, "local_only": True},
                "Local model manifest and inventory are valid",
            )
        except ModelRegistryError as exc:
            capabilities[f"paddle_model:{model_id}"] = _check(
                "FAIL", None, {"model_id": model_id, "local_only": True}, str(exc)
            )
    try:
        tool_suite = (
            Path(tool_root) / "runtime" / "model_suite.json"
            if tool_root is not None
            else None
        )
        manifest_path = (
            tool_suite if tool_suite is not None and tool_suite.is_file() else None
        )
        suite = validate_model_suite(
            registry.models_root, manifest_path=manifest_path, deep=deep
        )
        suite_check = _check(
            "PASS",
            {
                "suite_id": suite["suite_id"],
                "model_count": suite["model_count"],
                "suite_fingerprint": suite["suite_fingerprint"],
                "deep_verified": deep,
                "gpu_smoke": "PENDING" if deep and probe_runtime else "NOT_RUN",
            },
            {"model_count": 9, "local_only": True},
            "Nine-model suite manifest and inventories are valid",
        )
        capabilities["paddle_model_suite"] = suite_check
        if deep and probe_runtime:
            try:
                from .model_suite_runtime import run_model_suite_smoke

                smoke = run_model_suite_smoke(
                    registry.models_root, manifest_path=manifest_path
                )
                if not smoke["all_ok"]:
                    raise RuntimeError("One or more model suite GPU smokes failed")
                suite_check["actual"]["gpu_smoke"] = "PASS"
                suite_check["actual"]["network_attempts"] = smoke["network_attempts"]
                for record in smoke["models"]:
                    model_check = capabilities[f"paddle_model:{record['model_id']}"]
                    model_check["actual"]["runtime_smoke"] = {
                        "load": record["load_status"],
                        "inference": record["inference_status"],
                        "unload": record["unload_status"],
                    }
            except Exception as exc:  # noqa: BLE001 - doctor deep probe boundary
                capabilities["paddle_model_suite"] = _check(
                    "FAIL",
                    suite_check["actual"],
                    {"model_count": 9, "gpu_smoke": "PASS"},
                    f"Model suite GPU smoke failed: {type(exc).__name__}: {exc}",
                )
                for model_id in PADDLE_MODEL_CATALOG:
                    capabilities[f"paddle_model:{model_id}"]["status"] = "FAIL"
                    capabilities[f"paddle_model:{model_id}"]["reason"] = (
                        "Deep model suite GPU smoke did not complete"
                    )
    except (ModelSuiteManifestError, ModelRegistryError) as exc:
        capabilities["paddle_model_suite"] = _check(
            "FAIL", None, {"model_count": 9, "local_only": True}, str(exc)
        )


def _python_check() -> dict[str, Any]:
    actual = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
    }
    passed = (
        platform.python_implementation() == "CPython"
        and sys.version_info[:2] == (3, 11)
        and os.name == "nt"
        and platform.machine().lower() in {"amd64", "x86_64"}
    )
    return _check(
        "PASS" if passed else "FAIL",
        actual,
        {"implementation": "CPython", "version": "3.11.x", "platform": "Windows x64"},
        "Validated release platform" if passed else "Runtime platform does not match RC profile",
    )


def _core_dependencies_check() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    missing: list[str] = []
    for name in CORE_DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            missing.append(name)
    return _check(
        "FAIL" if missing else "PASS",
        versions,
        list(CORE_DEPENDENCIES),
        f"Missing core dependencies: {', '.join(missing)}" if missing else "All core dependencies import from installed distributions",
    )


def _mtef_check(tool_root: str | Path | None) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("mathtypejx")
        direct_url = distribution.read_text("direct_url.json")
        revision = None
        if direct_url:
            payload = json.loads(direct_url)
            revision = payload.get("vcs_info", {}).get("commit_id")
        context = MtefCacheContext(mode="off")
        manifest_revision = None
        if tool_root is not None:
            runtime_path = Path(tool_root) / "runtime" / "RUNTIME_MANIFEST.json"
            if runtime_path.is_file():
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                manifest_revision = runtime.get("mathtypejx", {}).get("revision")
        passed = revision == MATHTYPEJX_REVISION or manifest_revision == MATHTYPEJX_REVISION
        return _check(
            "PASS" if passed else "FAIL",
            {
                "version": distribution.version,
                "installed_revision": revision,
                "runtime_manifest_revision": manifest_revision,
                "contract": context.contract_fingerprint,
            },
            {"revision": MATHTYPEJX_REVISION},
            "MTEF converter revision is frozen" if passed else "mathtypejx revision identity is missing or different",
        )
    except Exception as exc:  # noqa: BLE001 - doctor boundary
        return _check("FAIL", None, {"revision": MATHTYPEJX_REVISION}, f"MTEF probe failed: {type(exc).__name__}: {exc}")


def _wmf_inspector_check() -> dict[str, Any]:
    try:
        inspection = WmfInspector().inspect(b"")
        return _check(
            "PASS",
            {"component": type(inspection).__name__, "empty_input_safe": not inspection.valid},
            {"bounded_parser": True},
            "WMF inspector loaded and rejected empty input safely",
        )
    except Exception as exc:  # noqa: BLE001 - doctor boundary
        return _check("FAIL", None, {"bounded_parser": True}, f"WMF inspector failed: {exc}")


def _wmf_render_check() -> dict[str, Any]:
    try:
        if os.name != "nt":
            raise RuntimeError("Win32 GDI is unavailable on this platform")
        import ctypes

        gdi = ctypes.WinDLL("gdi32", use_last_error=True)
        passed = bool(gdi.PlayEnhMetaFile)
        return _check(
            "PASS" if passed else "FAIL",
            {"renderer": "win32_gdi", "library": "gdi32"},
            {"platform": "Windows", "renderer": "win32_gdi"},
            "Win32 GDI renderer is available" if passed else "Win32 GDI symbol is unavailable",
        )
    except Exception as exc:  # noqa: BLE001 - doctor boundary
        return _check("FAIL", None, {"renderer": "win32_gdi"}, f"WMF renderer unavailable: {exc}")


def _output_root_check(output_root, config_path) -> dict[str, Any]:
    try:
        root = resolve_output_root(output_root, config_path=config_path)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".bemarkdown-doctor-", dir=root):
            pass
        return _check(
            "PASS",
            {"configured": True, "writable": True},
            {"writable": True},
            "Output root is writable",
        )
    except Exception as exc:  # noqa: BLE001 - doctor boundary
        return _check("FAIL", None, {"writable": True}, f"Output root unavailable: {exc}")


def _tool_payload_check(tool_root: str | Path | None) -> dict[str, Any]:
    if tool_root is None:
        environment = os.environ.get("BEMARKDOWN_TOOL_ROOT")
        if not environment:
            return _check("SKIP", None, "RC TOOL_MANIFEST.json", "No RC tool root was supplied")
        tool_root = environment
    root = Path(tool_root)
    manifest_path = root / "TOOL_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        wheel = root / manifest["wheel"]["path"]
        valid = (
            manifest.get("tool_id") == "bemarkdown"
            and manifest.get("package_contract") == PACKAGE_CONTRACT
            and manifest.get("asset_contract") == ASSET_MANIFEST_VERSION
            and wheel.is_file()
            and sha256_file(wheel) == manifest["wheel"]["sha256"]
        )
        return _check(
            "PASS" if valid else "FAIL",
            {"tool_id": manifest.get("tool_id"), "wheel": wheel.name},
            {"tool_id": "bemarkdown", "wheel_sha": "valid"},
            "Immutable tool payload is valid" if valid else "Tool manifest or wheel integrity failed",
        )
    except Exception as exc:  # noqa: BLE001 - doctor boundary
        return _check("FAIL", None, "RC TOOL_MANIFEST.json", f"Tool payload validation failed: {exc}")


def _formula_runtime_check() -> dict[str, Any]:
    expected = {
        "paddlepaddle_gpu": "3.2.2",
        "paddlex": "3.7.2",
        "cuda_runtime": "12.6",
        "cudnn_compiled": "9.9.0",
        "cudnn_package": "9.9.0.52",
        "device": "gpu:0",
    }
    try:
        os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] = "True"
        from .paddlex_runtime import import_paddlex_for_paddle_provider

        paddle, paddlex, _create_model = import_paddlex_for_paddle_provider()

        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("Paddle is not compiled with CUDA")
        if paddle.device.cuda.device_count() < 1:
            raise RuntimeError("Paddle sees no CUDA device")
        paddle.set_device("gpu:0")
        tensor = paddle.matmul(
            paddle.to_tensor([[2.0]], dtype="float32"),
            paddle.to_tensor([[3.0]], dtype="float32"),
        )
        value = float(tensor.numpy()[0][0])
        actual = {
            "paddlepaddle_gpu": paddle.__version__,
            "paddlex": paddlex.__version__,
            "cuda_runtime": paddle.version.cuda(),
            "cudnn_compiled": paddle.version.cudnn(),
            "cudnn_package": _package_version("nvidia-cudnn-cu12"),
            "device": paddle.device.get_device(),
            "gpu_count": paddle.device.cuda.device_count(),
            "tensor_result": value,
        }
        passed = all(actual.get(key) == expected_value for key, expected_value in expected.items()) and value == 6.0
        return _check(
            "PASS" if passed else "FAIL",
            actual,
            expected,
            "Paddle GPU initialized and completed a real tensor operation" if passed else "Formula OCR runtime differs from the frozen profile",
        )
    except Exception as exc:  # noqa: BLE001 - safe degraded doctor boundary
        return _check("FAIL", None, expected, f"Formula OCR GPU runtime unavailable: {type(exc).__name__}: {exc}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check(status: str, actual: Any, expected: Any, reason: str) -> dict[str, Any]:
    return {"status": status, "actual": actual, "expected": expected, "reason": reason}
