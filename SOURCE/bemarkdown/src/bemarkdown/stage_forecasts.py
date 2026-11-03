"""Stdlib-only forecast identity checks before native model child creation.

The packaged envelope is calibrated evidence, not certification of unmeasured
inputs. No Torch/Paddle imports, model changes, fallback budgets or thresholds.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
FORECAST_PATH = ROOT / "runtime" / "RESOURCE_FORECASTS_6GB.json"
GiB = 1024 ** 3
MiB = 1024 ** 2
_LOCATION_KEYS = {"models_root", "mcp_root", "config_path"}
_POST_LOAD_OPTIONS = {"capture_floor", "character_positions"}


class ForecastAdmissionError(RuntimeError):
    pass


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MiB), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _normal(value):
    return value.lower().replace("_", "-").replace(".", "-")


def validate_source_bindings(bindings, root=None):
    root = ROOT if root is None else Path(root)
    for relative, expected in bindings.items():
        allowed = [expected] if isinstance(expected, str) else expected
        # Only this recorded LF/CRLF equivalence may have two raw byte hashes.
        if (not isinstance(allowed, list) or not allowed or
                (not isinstance(expected, str) and (relative != "formula_ocr.py" or len(allowed) != 2)) or
                any(not isinstance(v, str) or len(v) != 64 for v in allowed)):
            raise ForecastAdmissionError("FORECAST_SOURCE_BINDING_INVALID:" + relative)
        if _sha(root / relative) not in allowed:
            raise ForecastAdmissionError("FORECAST_SOURCE_BINDING_MISMATCH:" + relative)


def actor_recipe(config):
    kwargs = {k: v for k, v in config.get("kwargs", {}).items() if k not in _LOCATION_KEYS | _POST_LOAD_OPTIONS}
    if config.get("kind") == "ocr":
        kwargs.setdefault("device", "gpu:0")
    if config.get("kind") == "formula":
        kwargs.setdefault("deep_model_validation", True)
        kwargs.setdefault("runtime_owner", None)
    return {"kind": config.get("kind"), "args": config.get("args", []),
            "kwargs": kwargs,
            "attributes": {k: v for k, v in config.get("attributes", {}).items() if k not in _POST_LOAD_OPTIONS},
            "backend_loader": config.get("backend_loader")}


def dependency_identity(paths, required):
    """Resolve the actual child search path, then bind metadata and loader bytes."""
    found = {}
    for distribution in importlib.metadata.distributions(path=[str(p) for p in paths]):
        name = _normal(distribution.metadata.get("Name", ""))
        if name not in required or name in found:
            continue
        spec = required[name]
        meta = Path(distribution._path)
        record = meta / "RECORD"
        files = {}
        for relative in spec.get("loader_files", {}):
            path = Path(distribution.locate_file(relative))
            files[relative] = _sha(path)
        found[name] = {"version": distribution.version,
                       "metadata_sha256": _sha(meta / "METADATA"),
                       "record_sha256": _sha(record), "loader_files": files}
    if set(found) != set(required):
        raise ForecastAdmissionError("FORECAST_DEPENDENCY_MISSING:" + ",".join(sorted(set(required) - set(found))))
    if found != required:
        raise ForecastAdmissionError("FORECAST_DEPENDENCY_BINDING_MISMATCH")
    return found


def _child_paths(args, env, request=None):
    paths = []
    exe = Path(args[0]).resolve()
    # Venv redirector; its site-packages precede base-interpreter packages.
    if exe.parent.name.lower() in {"scripts", "bin"}:
        paths.append(exe.parent.parent / "Lib" / "site-packages")
    if request:
        paths.extend(Path(p) for p in request.get("dependency_paths", []))
    if "-I" not in args:
        paths.extend(Path(p) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p)
    paths.extend(Path(p) for p in sys.path if p)
    return list(dict.fromkeys(p.resolve() for p in paths if p.is_dir()))


def _resolve_models(stage, entry, request, actor):
    # Registry itself is stdlib-only. Quick inventory validates manifest and file
    # sizes; the unchanged native loader retains its existing deep verification.
    from .model_registry import ModelRegistry
    if stage in {"ovis-ocr2", "xiaomi-ocr-0"}:
        model_root = Path(request["model_root"])
        manifest = _read(model_root / "MODEL_MANIFEST.json")
        if request.get("model_fingerprint") != manifest.get("model_fingerprint"):
            raise ForecastAdmissionError("FORECAST_MODEL_BINDING_MISMATCH")
        registry = ModelRegistry(models_root=model_root.parent)
    else:
        options = actor.get("kwargs", {}) if actor else request.get("registry_options", {})
        registry = ModelRegistry(**{k: v for k, v in options.items() if k in _LOCATION_KEYS})
    for spec in entry["models"]:
        resolved = registry.resolve(spec["model_id"], deep=False)
        if (resolved.manifest.get("model_fingerprint") != spec["model_fingerprint"] or
                _sha(resolved.model_root / "MODEL_MANIFEST.json") != spec["manifest_sha256"]):
            raise ForecastAdmissionError("FORECAST_MODEL_BINDING_MISMATCH:" + spec["model_id"])


def _identify(args, stage, env, manifest):
    if not isinstance(args, (list, tuple)):
        raise ForecastAdmissionError("FORECAST_COMMAND_REQUIRES_ARGV")
    argv = [os.fspath(v) for v in args]
    actor = request = None
    if "--actor" in argv:
        actor = _read(argv[argv.index("--actor") + 1])
        expected = {"layout": "pp-doclayout", "ocr": "pp-ocrv6", "formula": "pp-formulanet"}.get(actor.get("kind"))
        if expected != stage:
            raise ForecastAdmissionError("FORECAST_STAGE_IDENTITY_MISMATCH")
        if actor_recipe(actor) not in manifest["stages"][stage]["actor_recipes"]:
            raise ForecastAdmissionError("FORECAST_INPUT_POLICY_BINDING_MISMATCH")
        worker = next((Path(a) for a in argv[1:] if Path(a).name == "isolated_native_backends.py"), None)
        if worker is None or _sha(worker) != manifest["source_bindings"]["isolated_native_backends.py"]:
            raise ForecastAdmissionError("FORECAST_WORKER_BINDING_MISMATCH")
    elif "--manifest" in argv:
        request = _read(argv[argv.index("--manifest") + 1])
        worker_name = "qwen_ocr.py" if stage in {"ovis-ocr2", "xiaomi-ocr-0"} else "table_stage.py"
        worker = next((Path(a) for a in argv[1:] if Path(a).name == worker_name), None)
        relative = "pdf/workers/" + worker_name
        if worker is None or _sha(worker) != manifest["source_bindings"][relative]:
            raise ForecastAdmissionError("FORECAST_WORKER_BINDING_MISMATCH")
        if worker_name == "qwen_ocr.py":
            recipe = {k: request.get(k, default) for k, default in (("vram_limit_mib", 2816), ("max_new_tokens", 2048))}
            recipe.update({k: env.get(k) == "1" for k in ("BEMARKDOWN_DISABLE_QWEN_ALLOCATOR_WARMUP",
                "BEMARKDOWN_QWEN_READONLY_WEIGHT_MAPS", "HF_DEACTIVATE_ASYNC_LOAD")})
            if recipe != manifest["stages"][stage]["input_recipe"]:
                raise ForecastAdmissionError("FORECAST_INPUT_POLICY_BINDING_MISMATCH")
            if _read(Path(request["model_root"]) / "MODEL_MANIFEST.json").get("model_id") != stage:
                raise ForecastAdmissionError("FORECAST_STAGE_IDENTITY_MISMATCH")
        else:
            expected = request.get("model_id")
            if (expected if expected == "pp-ocrv6" else "table:" + str(expected)) != stage:
                raise ForecastAdmissionError("FORECAST_STAGE_IDENTITY_MISMATCH")
            if request.get("schema") != "isolated-table-model-stage-v1":
                raise ForecastAdmissionError("FORECAST_INPUT_POLICY_BINDING_MISMATCH")
            if request.get("worker_sha256") != manifest["source_bindings"][relative]:
                raise ForecastAdmissionError("FORECAST_WORKER_BINDING_MISMATCH")
            if request.get("coordinator_sha256") != manifest["source_bindings"]["pdf/table_staged_runtime.py"]:
                raise ForecastAdmissionError("FORECAST_INPUT_POLICY_BINDING_MISMATCH")
    else:
        raise ForecastAdmissionError("FORECAST_HEAVY_COMMAND_UNQUALIFIED")
    return argv, actor, request


def check_projected(snapshot, bound, memory_policy=None):
    from .host_memory_policy import current_memory_policy, evaluate_admission
    decision=evaluate_admission(memory_policy or current_memory_policy(),snapshot,bound)
    if decision['decision']!='ADMIT':
        prefix='DEVICE_CAPACITY_INSUFFICIENT:' if decision['decision']=='CAPACITY_INSUFFICIENT' else ''
        error=ForecastAdmissionError(prefix+';'.join(decision['reasons']))
        error.projected=decision['projected'];error.decision=decision['decision'];error.policy=decision
        raise error
    return decision['projected']

def preflight_stage_forecast(args, stage, snapshot, env=None):
    """Automatic existing-command admission; no model modules are imported."""
    env = os.environ if env is None else env
    # Generic native commands retain the strict live resource policy and owned
    # Job semantics, but receive no heavy-stage forecast certification.
    if stage in {"native", "native-smoke"}:
        if any(str(a).endswith(("qwen_ocr.py", "table_stage.py", "isolated_native_backends.py")) for a in args):
            raise ForecastAdmissionError("FORECAST_HEAVY_STAGE_UNQUALIFIED")
        return {"status": "GENERIC_NATIVE_NO_HEAVY_FORECAST"}
    evidence = {"stage": stage, "status": "FORECAST_REFUSED"}
    try:
        manifest = _read(FORECAST_PATH)
        evidence.update(manifest_sha256=_sha(FORECAST_PATH), calibration=manifest.get("calibration"))
        if manifest.get("schema") != "bemarkdown-owned-stage-forecast-v1" or stage not in manifest.get("stages", {}):
            raise ForecastAdmissionError("FORECAST_STAGE_UNQUALIFIED:" + stage)
        if env.get("BEMARKDOWN_RESOURCE_PROFILE", "6gb") != "6gb":
            raise ForecastAdmissionError("FORECAST_RESOURCE_PROFILE_BINDING_MISMATCH")
        validate_source_bindings(manifest["source_bindings"])
        argv, actor, request = _identify(args, stage, env, manifest)
        entry = manifest["stages"][stage]
        evidence["bound"] = {k: entry[k] for k in ("extra_private_bytes", "extra_physical_bytes", "extra_gpu_mib")}
        if _sha(argv[0]) not in manifest["dependency_profiles"][entry["dependency_profile"]]["executable_sha256"]:
            raise ForecastAdmissionError("FORECAST_EXECUTABLE_BINDING_MISMATCH")
        _resolve_models(stage, entry, request, actor)
        dependencies = manifest["dependency_profiles"][entry["dependency_profile"]]["distributions"]
        dependency_identity(_child_paths(argv, env, request), dependencies)
        from .host_memory_policy import current_memory_policy
        projected = check_projected(snapshot, entry, current_memory_policy(env))
        return {**evidence, "status": manifest["status"], "projected": projected}
    except ForecastAdmissionError as exc:
        exc.forecast = evidence
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        error = ForecastAdmissionError("FORECAST_BINDING_UNAVAILABLE:" + type(exc).__name__ + ":" + str(exc))
        error.forecast = evidence
        raise error from exc
