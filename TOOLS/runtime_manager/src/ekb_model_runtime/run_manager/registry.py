"""Installation-owned worker registrations, never caller-supplied commands."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import string

from .common import RunManagerError, bounded_number, file_sha, fingerprint


def load_registry(path):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "ekb-run-manager-registry-v1" or not value.get("programs"):
        raise RunManagerError("INVALID_PROGRAM_REGISTRY")
    programs = {}
    for identifier, row in value["programs"].items():
        current = dict(row)
        python = Path(row["python"]).resolve(strict=True)
        script = Path(row["script"]).resolve(strict=True)
        if python.name.lower() not in {"python.exe", "python3.exe", "pythonw.exe"} or script.suffix.lower() != ".py":
            raise RunManagerError("UNCONTROLLED_PROGRAM", identifier)
        if row.get("installed_root") and not script.is_relative_to(Path(row["installed_root"]).resolve(strict=True)):
            raise RunManagerError("PROGRAM_OUTSIDE_INSTALLATION", identifier)
        if file_sha(script) != row["script_sha256"]:
            raise RunManagerError("PROGRAM_SOURCE_CHANGED", identifier)
        current.update(python=str(python), script=str(script), actual_python_sha256=file_sha(python))
        if row.get("python_sha256") and current["actual_python_sha256"] != row["python_sha256"]:
            raise RunManagerError("PROGRAM_INTERPRETER_CHANGED", identifier)
        arguments = row.get("arguments", ["--manager-job-id", "{job_id}", "--manager-request", "{request_path}"])
        if not isinstance(arguments, list) or any(not isinstance(argument, str) for argument in arguments):
            raise RunManagerError("INVALID_PROGRAM_ARGUMENTS", identifier)
        for argument in arguments:
            for _, field, fmt, conversion in string.Formatter().parse(argument):
                if field is not None and (field not in {"job_id", "workspace", "request_path"} or fmt or conversion):
                    raise RunManagerError("INVALID_ARGUMENT_PLACEHOLDER", identifier)
        current["arguments"] = arguments
        current["resource_kind"] = row.get("resource_kind", "gpu")
        current["gpu_key"] = row.get("gpu_key", "GPU:0")
        current["requirement"] = requirement(row.get("requirement", {}))
        current["program_fingerprint"] = fingerprint(current)
        programs[identifier] = current
    value["programs"] = programs
    value["registry_path"] = str(path)
    value["registry_file_sha256"] = file_sha(path)
    value["registry_fingerprint"] = fingerprint(value)
    return value


def requirement(row):
    allowed = {"extra_private_bytes", "extra_physical_bytes", "extra_gpu_mib"}
    if set(row) - allowed:
        raise RunManagerError("UNKNOWN_REQUIREMENT_FIELD")
    return {key: bounded_number(row.get(key, 0), key) for key in sorted(allowed)}


class ResourceAdapter:
    def __init__(self, specification):
        self.specification = specification
        path = Path(specification["module_path"]).resolve(strict=True)
        if file_sha(path) != specification["sha256"]:
            raise RunManagerError("RESOURCE_POLICY_SOURCE_CHANGED")
        spec = importlib.util.spec_from_file_location("_ekb_registered_device_policy", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def sample(self):
        return self.module.sample_device(gpu_index=self.specification.get("gpu_index", 0),
                                         gpu_uuid=self.specification.get("gpu_uuid"))

    def policy(self, observation):
        return self.module.resolve_device_policy(
            observation["total_physical_bytes"], observation["commit_limit_bytes"], observation["gpu_total_bytes"],
            gpu_uuid=observation["gpu_uuid"], gpu_index=observation.get("gpu_index", 0),
            gpu_count=observation.get("gpu_count", 1), mapping_supported=observation.get("mapping_supported", True),
            mode=self.specification.get("mode", "auto"))

    def admit(self, policy, observation, requested):
        return self.module.evaluate_admission(policy, observation, requested, reservations=[])

    def check(self, policy, observation):
        return self.module.check_current(policy, observation, raise_on_violation=False)
