"""Loopback model service. Each inference request owns a disposable process.

Built-in backends are deliberately explicit. Other model servers may implement
the same wire contract without importing this module or changing the converter.
"""
from __future__ import annotations

import argparse
import hmac
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

from .pdf.local_model_api import MAX_BYTES, MAX_ITEMS, ROLES, LEGACY_ROLES, SCHEMA, ModelApiError, decoded_image


def load_server_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or frozenset(config.get("roles", {})) not in {frozenset(ROLES), LEGACY_ROLES}:
        raise ModelApiError("MODEL_API_SERVER_CONFIG_INVALID")
    for role, spec in config["roles"].items():
        root = Path(spec["model_root"])
        spec["model_root"] = str((path.parent / root).resolve())
        if not (Path(spec["model_root"]) / "config.json").is_file():
            raise ModelApiError(f"MODEL_API_MODEL_CONFIG_MISSING:{role}")
        # Existing OCR prompts are role-specific. A new architecture/role needs
        # an adapter honoring that role, not just a renamed model directory.
        supported = {"layout": {"pp-doclayout-v3"}, "formula": {"xiaomi-ocr-0"},
                     "table": {"ovisocr3"}, "text": {"ovisocr3"}}[role]
        if spec.get("backend") not in supported:
            raise ModelApiError(f"MODEL_API_BUILTIN_BACKEND_UNSUPPORTED:{role}")
    return config


def validate_request(payload, role):
    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA
            or payload.get("role") != role or not isinstance(payload.get("request_id"), str)
            or not 1 <= len(payload["request_id"]) <= 128):
        raise ModelApiError("MODEL_API_REQUEST_BINDING_INVALID")
    items = payload.get("items")
    if (not isinstance(items, list) or not 0 < len(items) <= MAX_ITEMS
            or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in items)
            or len({row["id"] for row in items}) != len(items)):
        raise ModelApiError("MODEL_API_REQUEST_ITEMS_INVALID")


def make_backend(role, spec):
    from .pdf.visual_serial_runtime import TransformersV3LayoutRuntime, TransformersVisionOcrRuntime
    if role == "layout":
        return TransformersV3LayoutRuntime(spec["model_root"], threshold=spec.get("threshold", 0.35))
    return TransformersVisionOcrRuntime(spec["backend"], spec["model_root"],
                                       max_new_tokens=spec.get("max_new_tokens", 2048))


def infer(payload, config, *, backend_factory=make_backend):
    role = payload["role"]
    validate_request(payload, role)
    spec = config["roles"][role]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="bemarkdown-inference-") as temp:
        images = [decoded_image(item) for item in payload["items"]]
        backend = backend_factory(role, spec)
        try:
            if role == "layout":
                renders = [SimpleNamespace(image_bytes=data, page_index=i, page_number=i + 1,
                                           render_sha256=item["sha256"])
                           for i, (item, data) in enumerate(zip(payload["items"], images, strict=True))]
                detected = backend.detect(renders, output_dir=temp)
                label_map = spec.get("label_map", {})
                items = [{"id": item["id"], "regions": [
                    {"label": label_map.get(row["label"], row["label"]), "bbox": row["bbox"],
                     "reading_order": row["reading_order"], "score": row.get("metadata", {}).get("score")}
                    for row in detected if row["page_index"] == i]}
                    for i, item in enumerate(payload["items"])]
            else:
                crops = []
                for i, (item, data) in enumerate(zip(payload["items"], images, strict=True)):
                    path = Path(temp) / f"{i}.png"
                    path.write_bytes(data)
                    profile = "TABLE_HTML_V1" if role == "table" else item.get("serialization_profile", "default")
                    if profile not in ({"TABLE_HTML_V1"} if role == "table" else {"default", "UPSTREAM_TEX_MIXED_V1"}):
                        raise ModelApiError("MODEL_API_SERIALIZATION_PROFILE_UNSUPPORTED")
                    crops.append({"path": str(path), "sha256": item["sha256"], "serialization_profile": profile})
                values = backend.recognize_batch(crops)
                if len(values) != len(crops) or not all(isinstance(value, str) for value in values):
                    raise ModelApiError("MODEL_API_BACKEND_OCR_CARDINALITY_OR_TYPE_INVALID")
                items = [{"id": item["id"], "text": value}
                         for item, value in zip(payload["items"], values, strict=True)]
        finally:
            backend.close()
        identity = dict(backend.identity)
        if role == "table":
            from .pdf.visual_serial_prompts import prompt_for, prompt_fingerprint
            identity.update(prompt=prompt_for("ovisocr3-table"),
                            prompt_fingerprint=prompt_fingerprint("ovisocr3-table"))
        return {"schema": SCHEMA, "role": role, "request_id": payload["request_id"],
                "items": items, "identity": identity,
                "metrics": {**backend.metrics, "worker_seconds": time.perf_counter() - started}}


def run_owned_process(command, *, stdout, timeout):
    """Wait for inference, or terminate its whole owned process tree on timeout.

    Windows venv executables can be launchers, so killing only Popen.pid is
    insufficient. A timeout is never reported as released until tree cleanup.
    """
    child = subprocess.Popen(command, stdout=stdout, stderr=stdout,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                             start_new_session=os.name != "nt")
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if os.name == "nt":
            cleanup = subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            if cleanup.returncode:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=30)
                raise ModelApiError("MODEL_API_WORKER_TIMEOUT_TREE_EXIT_UNCONFIRMED") from exc
        else:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
        raise ModelApiError("MODEL_API_WORKER_TIMEOUT_TREE_EXIT_CONFIRMED") from exc


class ProcessBackend:
    def __init__(self, config_path, *, timeout=540):
        self.config_path = Path(config_path).resolve()
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.timeout = timeout

    def __call__(self, payload):
        # This child inherits the run manager's Windows Job and resource budget.
        if not os.environ.get("EKB_MANAGED_JOB_ID"):
            raise ModelApiError("MODEL_API_MANAGED_JOB_REQUIRED")
        if hashlib.sha256(self.config_path.read_bytes()).hexdigest() != self.config_sha256:
            raise ModelApiError("MODEL_API_SERVER_CONFIG_CHANGED")
        with tempfile.TemporaryDirectory(prefix="bemarkdown-api-") as temp:
            request, response, log = [Path(temp) / name for name in ("request.json", "response.json", "worker.log")]
            request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with log.open("w", encoding="utf-8") as stream:
                returncode = run_owned_process(
                    [sys.executable, "-m", "bemarkdown.local_model_server", "worker",
                     "--config", str(self.config_path), "--request", str(request), "--response", str(response)],
                    stdout=stream, timeout=self.timeout)
            if returncode:
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 6000))
                    detail = stream.read().decode("utf-8", errors="replace")
                raise ModelApiError(f"MODEL_API_WORKER_FAILED:{returncode}:{detail}")
            payload = json.loads(response.read_text(encoding="utf-8"))
            payload["released"] = True
            payload["metrics"]["worker_exit_code"] = returncode
            return payload


def create_server(address, backend, *, token):
    if address[0] != "127.0.0.1" or not token:
        raise ModelApiError("MODEL_API_LOOPBACK_AND_TOKEN_REQUIRED")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def send_json(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A disconnected client does not restart finished inference.

        def do_POST(self):
            role = self.path.removeprefix("/v1/")
            if self.path != f"/v1/{role}" or role not in ROLES:
                self.send_json(404, {"error": "unknown role endpoint"})
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_json(401, {"error": "unauthorized"})
                return
            if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                self.send_json(415, {"error": "JSON with Content-Length required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    raise ModelApiError("MODEL_API_REQUEST_SIZE_INVALID")
                payload = json.loads(self.rfile.read(length))
                validate_request(payload, role)
            except (ValueError, OSError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            try:
                result = backend(payload)
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
                return
            self.send_json(200, result)

    # One request at a time: no overlapping GPU model loads across roles.
    return HTTPServer(address, Handler)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("serve", "worker"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-env", default="BEMARKDOWN_MODEL_API_TOKEN")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args(argv)
    if not os.environ.get("EKB_MANAGED_JOB_ID"):
        parser.error("Start inference through the workspace resource manager")
    config = load_server_config(args.config)
    if args.mode == "worker":
        result = infer(json.loads(args.request.read_text(encoding="utf-8")), config)
        args.response.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return 0
    server = create_server(("127.0.0.1", args.port), ProcessBackend(args.config),
                           token=os.environ.get(args.token_env, ""))
    if args.ready_file:
        args.ready_file.write_text(json.dumps({"port": server.server_port, "pid": os.getpid()}), encoding="utf-8")
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
