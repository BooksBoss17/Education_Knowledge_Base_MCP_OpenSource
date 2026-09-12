"""Loopback-only Human Formula Review prototype server."""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .formula_review import FormulaReviewWorkflow


class FormulaReviewServer:
    """Serve a token-protected review session on a random loopback port."""

    def __init__(
        self,
        tasks: list[dict[str, Any]],
        workflow: FormulaReviewWorkflow,
        *,
        static_root: str | Path | None = None,
    ):
        self._tasks = {task["formula_id"]: task for task in tasks}
        self._workflow = workflow
        self._token = secrets.token_urlsafe(32)
        self._static_root = Path(
            static_root
            or Path(__file__).parent / "static" / "formula_review"
        ).resolve()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> dict[str, Any]:
        if self._server is not None:
            raise RuntimeError("Formula review server is already running")
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path.startswith("/api/") and not owner._authorized(parsed.query):
                    return self._json(HTTPStatus.FORBIDDEN, {"message": "会话凭据无效。"})
                if parsed.path == "/api/tasks":
                    return self._json(
                        HTTPStatus.OK,
                        {"tasks": list(owner._tasks.values())},
                    )
                if parsed.path.startswith("/api/crop/"):
                    formula_id = unquote(parsed.path.removeprefix("/api/crop/"))
                    task = owner._tasks.get(formula_id)
                    if task is None:
                        return self._json(HTTPStatus.NOT_FOUND, {"message": "公式不存在。"})
                    path = Path(str(task.get("source_crop_ref") or ""))
                    if not path.is_file():
                        return self._json(HTTPStatus.NOT_FOUND, {"message": "原公式图片不可用。"})
                    return self._bytes(
                        HTTPStatus.OK,
                        path.read_bytes(),
                        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    )
                return self._static(parsed.path)

            def do_POST(self):
                parsed = urlparse(self.path)
                if not owner._authorized(parsed.query):
                    return self._json(HTTPStatus.FORBIDDEN, {"message": "会话凭据无效。"})
                if parsed.path != "/api/human-resolution":
                    return self._json(HTTPStatus.NOT_FOUND, {"message": "入口不存在。"})
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(length))
                except (ValueError, json.JSONDecodeError):
                    return self._json(HTTPStatus.BAD_REQUEST, {"message": "提交内容无效。"})
                formula_id = payload.get("formula_id")
                task = owner._tasks.get(formula_id)
                if task is None:
                    return self._json(HTTPStatus.NOT_FOUND, {"message": "公式不存在。"})
                try:
                    result = owner._workflow.apply_human_result(task, payload)
                except ValueError:
                    return self._json(HTTPStatus.CONFLICT, {"message": "公式当前不能提交人工结果。"})
                if result.get("state") == "HUMAN_REVIEW_PENDING" and result.get(
                    "human_validation"
                ):
                    validation = result["human_validation"]
                    if not validation["syntax_passed"]:
                        message = "公式中还有一个未填写的位置，请补充后再确认。"
                    else:
                        message = "这个公式目前无法正常显示，请检查刚才修改的位置。"
                    return self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"message": message})
                owner._tasks[formula_id] = result
                return self._json(HTTPStatus.OK, result)

            def _static(self, request_path: str):
                relative = "index.html" if request_path in {"", "/"} else unquote(request_path.lstrip("/"))
                candidate = (owner._static_root / relative).resolve()
                try:
                    candidate.relative_to(owner._static_root)
                except ValueError:
                    return self._json(HTTPStatus.FORBIDDEN, {"message": "路径无效。"})
                if not candidate.is_file():
                    return self._json(HTTPStatus.NOT_FOUND, {"message": "资源不存在。"})
                return self._bytes(
                    HTTPStatus.OK,
                    candidate.read_bytes(),
                    mimetypes.guess_type(candidate.name)[0] or "application/octet-stream",
                )

            def _json(self, status: HTTPStatus, value: Any):
                return self._bytes(
                    status,
                    json.dumps(value, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )

            def _bytes(self, status: HTTPStatus, data: bytes, content_type: str):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        base_url = f"http://{host}:{port}"
        return {
            "host": host,
            "port": port,
            "token": self._token,
            "base_url": base_url,
            "review_url": f"{base_url}/?token={self._token}",
            "cdn_dependency": False,
        }

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def _authorized(self, query: str) -> bool:
        values = parse_qs(query).get("token") or []
        return len(values) == 1 and secrets.compare_digest(values[0], self._token)
