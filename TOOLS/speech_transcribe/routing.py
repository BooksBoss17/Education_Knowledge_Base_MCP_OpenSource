"""Resolve installation-owned speech routing before submitting or running a job."""
import hashlib
import ipaddress
import json
import re
from pathlib import Path
from urllib.parse import urlsplit


def resolve(root, timestamps=None, *, config_path=None):
    path = config_path or root / ".local/speech-api.json"
    if not path.exists():
        path = root / "TOOLS/speech_transcribe/providers.json"
    config = json.loads(path.read_text("utf-8-sig"))
    timing = config.get("timing")
    if config.get("schema") != "speech-providers-v1" or timing not in ("none", "native", "aligner"):
        raise ValueError("Invalid speech provider schema or timing mode")
    if timestamps is True and timing == "none":
        raise ValueError("Configured speech route supports text only")
    timing = "none" if timestamps is False else timing
    resolved = {"timing": timing}
    for role in (("asr", "aligner") if timing == "aligner" else ("asr",)):
        spec = config.get(role)
        if not isinstance(spec, dict) or not isinstance(spec.get("model"), str) or not spec["model"].strip():
            raise ValueError(f"Missing {role} model")
        provider = dict(spec)
        if spec.get("type") in ("managed_http", "qwen"):
            provider["type"] = "managed_http"
            if set(spec) != {"type", "model"} or not re.fullmatch(r"[\w-][\w.-]*", spec["model"]):
                raise ValueError("Invalid local Qwen model")
            if timing == "native":
                raise ValueError("Local Qwen ASR needs a separate aligner for timestamps")
            folder = root / "MODELS" / spec["model"]
            if not folder.resolve().is_relative_to((root / "MODELS").resolve()):
                raise ValueError("Model escapes configured directory")
            provider["manifest_sha256"] = hashlib.sha256((folder / "MODEL_MANIFEST.json").read_bytes()).hexdigest()
        elif spec.get("type") == "http":
            if set(spec) - {"type","model","endpoint","revision","granularity","token_env","allow_remote"}:
                raise ValueError("Unknown HTTP speech provider setting")
            url = urlsplit(spec.get("endpoint",""))
            try:
                loopback = ipaddress.ip_address(url.hostname or "").is_loopback
            except ValueError:
                loopback = url.hostname == "localhost"
            if not url.hostname or url.username is not None or url.password is not None or url.query or url.fragment:
                raise ValueError("Use an absolute endpoint without embedded credentials or query string")
            if not (loopback and url.scheme in ("http","https") or spec.get("allow_remote") is True and url.scheme=="https"):
                raise ValueError("External speech endpoints require HTTPS and allow_remote=true")
            if not isinstance(spec.get("revision"),str) or not spec["revision"].strip():
                raise ValueError("HTTP speech model revision required")
            if (role=="aligner" or timing=="native") and spec.get("granularity") not in ("segment","word","character"):
                raise ValueError("Declare the timestamp granularity")
            if "token_env" in spec and not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", spec["token_env"]):
                raise ValueError("Invalid token environment variable")
        else:
            raise ValueError(f"Unknown {role} provider")
        resolved[role] = provider
    return resolved


def fingerprint(route):
    return hashlib.sha256(json.dumps(route,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
