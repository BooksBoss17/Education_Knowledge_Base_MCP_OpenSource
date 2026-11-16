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
import re
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


def infer(payload, config, *, backend_factory=make_backend, release_backend=True):
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
            if release_backend:
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
        self.session = None
        self.closed_sessions = {}

    def capabilities(self):
        from .model_session import SESSION_SCHEMA
        return {"schema": "bemarkdown-model-api-capabilities-v1", "roles": list(ROLES),
                "session_schema": SESSION_SCHEMA, "max_items": MAX_ITEMS, "max_bytes": MAX_BYTES,
                "layout_batch_items": 8, "idle_seconds": 120}

    def session_action(self, action, payload):
        from .model_session import SessionProcess, SESSION_SCHEMA
        session_id = payload.get("session_id", "")
        if payload.get("schema") != SESSION_SCHEMA or not isinstance(session_id, str) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise ModelApiError("MODEL_API_SESSION_BINDING_INVALID")
        home, job = os.environ.get("EKB_RUN_MANAGER_HOME"), os.environ.get("EKB_MANAGED_JOB_ID", "")
        journal = Path(home) / "jobs" / job / "model_api_sessions" / session_id if home and re.fullmatch(r"[a-f0-9]{32}", job) else None
        closed = self.closed_sessions.get(session_id)
        if closed is None and journal is not None and (journal / "CLOSED.json").is_file():
            closed = {"schema": SESSION_SCHEMA, "state": "CLOSED", **json.loads((journal / "CLOSED.json").read_text("utf-8"))}
        if action == "open":
            if hashlib.sha256(self.config_path.read_bytes()).hexdigest() != self.config_sha256:
                raise ModelApiError("MODEL_API_SERVER_CONFIG_CHANGED")
            role = payload.get("role")
            if role not in ROLES:
                raise ModelApiError("MODEL_API_SESSION_ROLE_INVALID")
            if closed is not None:
                raise ModelApiError("MODEL_API_SESSION_ALREADY_CLOSED")
            if self.session is not None:
                if self.session.session_id != session_id or self.session.role != role:
                    raise ModelApiError("MODEL_API_ANOTHER_SESSION_ACTIVE")
            else:
                home = os.environ.get("EKB_RUN_MANAGER_HOME")
                job = os.environ.get("EKB_MANAGED_JOB_ID")
                if not home or not job:
                    raise ModelApiError("MODEL_API_MANAGED_JOB_REQUIRED")
                journal = Path(home) / "jobs" / job / "model_api_sessions"
                self.session = SessionProcess(self.config_path, role, timeout=self.timeout,
                    idle_seconds=120, journal_root=journal, session_id=session_id)
            return {"schema": SESSION_SCHEMA, "session_id": session_id, "role": role, "state": "OPEN"}
        if self.session is None or self.session.session_id != session_id:
            if action in {"close", "status"} and closed is not None:
                result = dict(closed)
                if action == "status" and payload.get("request_id") is not None:
                    home = os.environ.get("EKB_RUN_MANAGER_HOME")
                    job = os.environ.get("EKB_MANAGED_JOB_ID")
                    key = hashlib.sha256(str(payload["request_id"]).encode()).hexdigest()
                    receipt = Path(home) / "jobs" / job / "model_api_sessions" / session_id / (key + ".json")
                    result["request_state"] = "COMPLETE" if receipt.exists() else "UNKNOWN"
                    if receipt.exists():
                        result["receipt"] = json.loads(receipt.read_text("utf-8"))
                return result
            raise ModelApiError("MODEL_API_SESSION_UNKNOWN")
        if action == "status":
            return {"schema": SESSION_SCHEMA, **self.session.status(payload.get("request_id"))}
        if action == "close":
            return self.close_session()
        if action == "infer":
            internal = {key: value for key, value in payload.items() if key != "session_id"}
            internal["schema"] = SCHEMA
            validate_request(internal, self.session.role)
            return self.session.request(internal)
        raise ModelApiError("MODEL_API_SESSION_ACTION_UNKNOWN")

    def close_session(self):
        from .model_session import SESSION_SCHEMA
        session = self.session
        if session is None:
            return None
        closed = {"schema": SESSION_SCHEMA, "state": "CLOSED", **session.close()}
        self.closed_sessions[session.session_id] = closed
        while len(self.closed_sessions) > 64:
            del self.closed_sessions[next(iter(self.closed_sessions))]
        self.session = None
        return closed

    def reap(self):
        if self.session is not None and self.session.child.poll() is not None:
            self.close_session()

    def close(self):
        self.close_session()

    def __call__(self, payload):
        self.reap()
        if self.session is not None:
            raise ModelApiError("MODEL_API_ANOTHER_SESSION_ACTIVE")
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

        def authorized(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_json(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            capability = getattr(backend, "capabilities", None)
            if self.path == "/v1/capabilities" and callable(capability):
                try:
                    result = capability()
                except Exception as exc:
                    self.send_json(500, {"error": str(exc)})
                    return
                self.send_json(200, result)
            else:
                self.send_json(404, {"error": "capabilities unsupported"})

        def do_POST(self):
            role = self.path.removeprefix("/v1/")
            action = self.path.removeprefix("/v2/session/")
            session_request = (self.path == f"/v2/session/{action}" and action in {"open", "infer", "close", "status"}
                               and callable(getattr(backend, "session_action", None)))
            if not session_request and (self.path != f"/v1/{role}" or role not in ROLES):
                self.send_json(404, {"error": "unknown role endpoint"})
                return
            if not self.authorized():
                return
            if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                self.send_json(415, {"error": "JSON with Content-Length required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    raise ModelApiError("MODEL_API_REQUEST_SIZE_INVALID")
                payload = json.loads(self.rfile.read(length))
                if session_request:
                    if not isinstance(payload, dict):
                        raise ModelApiError("MODEL_API_SESSION_OBJECT_REQUIRED")
                else:
                    validate_request(payload, role)
            except (ValueError, OSError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            try:
                result = backend.session_action(action, payload) if session_request else backend(payload)
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
                return
            self.send_json(200, result)

    class Server(HTTPServer):
        def service_actions(self):
            reap = getattr(backend, "reap", None)
            if callable(reap):
                reap()

        def server_close(self):
            try:
                close = getattr(backend, "close", None)
                if callable(close):
                    close()
            finally:
                super().server_close()

    return Server(address, Handler)


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
