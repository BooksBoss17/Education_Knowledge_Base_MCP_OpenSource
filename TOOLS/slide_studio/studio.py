"""Local editor service, offline package IO and headless command entry."""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import secrets
import sys
import tempfile
import threading
from urllib.parse import parse_qs, quote, unquote, urlsplit
import webbrowser

from deck import Conflict, blank, check
from example import example
from portable import ROOT, filename, import_ppt, offline, read_html
from store import Store, dump
import session_link
import codex_notify
from motion import CATALOG, player_script, templates

DEFAULT_HOME = Path(os.environ.get("CLASSPAGE_STUDIO_HOME") or
                    Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ClasspageStudio")

def save_feedback_file(library, name, stream, size):
    check(bool(name) and name not in {".",".."} and not re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
          and not name.endswith((" ", ".")) and not Path(name).is_reserved(), "无效文件名")
    directory = library / "feedback-files"
    directory.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=directory))
    path = folder / name
    try:
        with path.open("xb") as dest:
            while size:
                part = stream.read(min(size, 1024*1024))
                check(part, "文件传输中断")
                dest.write(part)
                size -= len(part)
    except BaseException:
        path.unlink(missing_ok=True)
        folder.rmdir()
        raise
    return str(path.resolve())


def import_path(store, path, *, reopen=False):
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() in {".ppt", ".pptx"}:
        return import_ppt(store, path, reopen=reopen)
    check(path.suffix.lower() == ".html", "支持 PPT、PPTX 或课件 HTML")
    images = path.parent / "images"
    return read_html(
        store,
        path.read_text("utf-8-sig"),
        path.name,
        {p.name: p.read_bytes() for p in images.glob("[0-9]*.png")}
        if images.is_dir()
        else None,
        reopen=reopen,
    )


def agent_call(store, name, args):
    if name == "catalog":
        section = args.get("section", "effects")
        if section == "templates":
            return [t if args.get("name") == t["name"] else {"name": t["name"], "actions": t["actions"]}
                    for t in templates(store if args.get("name") else None) if not args.get("name") or args["name"] == t["name"]]
        if section == "effects":
            return {key: {"label": label, "group": group}
                    for group, entries in CATALOG["authoring"].items() for key, label in entries.items()}
        check(section == "transitions", "未知目录")
        return {key: {"label": item["label"], "group": item.get("group", "page")} for key, item in CATALOG[section].items()}
    if name == "library":
        return store.library()
    if name == "create":
        return {"id": store.create(blank(args["title"]))}
    if name == "import":
        return {"id": import_path(store, args["path"])}
    key = args["id"]
    if name == "edit":
        return store.transact(key, args["command"])
    if name == "package":
        from package_deck import package

        title = filename(store.state(key)["deck"]["title"])[:-5]
        dest = Path(args["directory"]).resolve(strict=True) / (title + ".zip")
        content = package(store, key)
        if dest.exists():
            # ZIP stores timestamps: compare actual members instead of archive bytes.
            import zipfile
            from io import BytesIO

            with zipfile.ZipFile(dest) as a, zipfile.ZipFile(BytesIO(content)) as b:
                check(
                    a.namelist() == b.namelist()
                    and all(a.read(n) == b.read(n) for n in a.namelist()),
                    "同名课件包已存在且内容不同",
                )
        else:
            with dest.open("xb") as f:
                f.write(content)
        return {"file": str(dest)}
    if name == "export":
        html = offline(store, key)
        dest = Path(args["directory"]).resolve(strict=True) / filename(
            store.state(key)["deck"]["title"]
        )
        if dest.exists():
            check(
                dest.read_text("utf8") == html, "导出目标已存在且内容不同，请选择新目录"
            )
        else:
            with dest.open("x", encoding="utf8") as f:
                f.write(html)
        return {"file": str(dest)}
    state = store.state(key)
    if name == "feedback":
        return {"revision": state["revision"], "feedback": state["feedback"]}
    check(name == "inspect", "未知操作")
    if args.get("slide"):
        return {
            "revision": state["revision"],
            "slide": state["deck"]["slides"][args["slide"]],
        }
    return {
        "revision": state["revision"],
        "title": state["deck"]["title"],
        "pages": [
            {"id": s, "title": state["deck"]["slides"][s]["title"]}
            for s in state["deck"]["order"]
        ],
    }


class WorkspaceHTTPServer(ThreadingHTTPServer):
    # Chromium may open several connections at once for covers and the next editor page.
    request_queue_size = 64


class HTTP(BaseHTTPRequestHandler):
    server_version = "Classpage/1"
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, *args):
        pass

    def send(self, code, body, mime="application/json; charset=utf-8", headers=None):
        if not isinstance(body, bytes):
            body = dump(body).encode()
        self.send_response(code)
        if code >= 400:
            # Some rejected requests still have an unread body.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def handle_request(self, write):
        store = self.server.store
        addr = f"127.0.0.1:{self.server.server_port}"
        check(self.headers.get("Host") == addr, "无效服务地址")
        path = urlsplit(self.path).path
        if not write and path in {
            "/",
            "/editor.js",
            "/effect-icons.js",
            "/effect-gallery.js",
            "/editor.css",
            "/home.js",
            "/home.css",
            "/workspace.js",
            "/session-dock.js",
            "/player.js",
            "/player.css",
            "/show.html",
            "/show.js",
            "/studio-mark.png",
        }:
            if path == "/":
                requested = parse_qs(urlsplit(self.path).query).get("deck")
                p = ROOT / "ui" / ("index.html" if requested else "home.html")
            else:
                p = ROOT / "ui" / path[1:]
            self.send(
                200,
                player_script().encode("utf8") if path == "/player.js" else p.read_bytes(),
                {
                    ".html": "text/html; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".js": "text/javascript; charset=utf-8",
                    ".png": "image/png",
                }[p.suffix],
            )
            return
        if write:
            check(
                self.headers.get("Origin") in {None, "http://" + addr}, "请求来源不符"
            )
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        token = self.headers.get("Authorization", "").removeprefix("Bearer ") or (
            cookie["studio"].value if "studio" in cookie else ""
        )
        if not hmac.compare_digest(token, self.server.token):
            self.send(401, {"error": "请使用启动器显示的本机链接"})
            return
        if write:
            size = int(self.headers.get("Content-Length", "0"))
            if path == "/api/feedback-file":
                check("Content-Length" in self.headers and size >= 0, "缺少文件大小")
                name = unquote(self.headers.get("X-Filename", ""))
                result = save_feedback_file(store.path.parent, name, self.rfile, size)
                self.send(201, {"path": result})
                return
            check(0 <= size <= 512 * 1024**2, "文件过大")
            data = self.rfile.read(size)
            if path == "/api/login":
                self.send(
                    200,
                    {},
                    headers={
                        "Set-Cookie": f"studio={token}; Path=/; SameSite=Strict; HttpOnly"
                    },
                )
                return
            if path == "/api/import-ppt":
                name = Path(unquote(self.headers.get("X-Filename", ""))).name
                check(name and not any(c in name for c in '<>:"/\\|?*'), "无效文件名")
                with tempfile.TemporaryDirectory(prefix="classpage-input-") as tmp:
                    src = Path(tmp) / name
                    src.write_bytes(data)
                    key = import_ppt(store, src, reopen=self.headers.get("X-Studio-Open") == "1")
                self.send(201, {"id": key})
                return
            if path == "/api/asset":
                self.send(
                    201,
                    {"asset": store.asset(self.headers.get("Content-Type", ""), data)},
                )
                return
            args = json.loads(data or b"{}")
            dock_actions = {
                "/api/session/status": session_link.status,
                "/api/session/issue": session_link.issue,
                "/api/session/disconnect": session_link.disconnect,
                "/api/session/recover": session_link.recover,
                "/api/session/send": session_link.send,
                "/api/session/wake": codex_notify.configure,
            }
            if path in dock_actions:
                self.send(200, dock_actions[path](store, **args))
                return
            if path == "/api/session/config":
                self.send(200, {"mcpServers": {"classpage": {"command": sys.executable, "args": [
                    "-B","-X","utf8",str(Path(__file__).resolve()),"--home",str(store.path.parent.resolve()),"mcp"]}}})
                return
            if path == "/api/workspace-options":
                self.send(200, store.workspace_options(args))
                return
            if path == "/api/import-html":
                key = read_html(
                    store,
                    args["html"],
                    args["name"],
                    {
                        k: base64.b64decode(v, validate=True)
                        for k, v in args.get("images", {}).items()
                    },
                    reopen=self.headers.get("X-Studio-Open") == "1",
                )
                self.send(201, {"id": key})
                return
            if path == "/api/agent":
                with self.server.worker_lock:
                    if not args.get("enabled"):
                        if self.server.worker_stop:
                            self.server.worker_stop.set()
                        self.server.worker_model = ""
                    else:
                        check(
                            not self.server.worker or not self.server.worker.is_alive(),
                            "已有模型正在处理，请等待当前调用结束",
                        )
                        endpoint = urlsplit(args["endpoint"])
                        check(
                            (
                                endpoint.scheme == "https"
                                or (
                                    endpoint.scheme == "http"
                                    and endpoint.hostname
                                    in {"localhost", "127.0.0.1", "::1"}
                                )
                            )
                            and endpoint.hostname
                            and not endpoint.username
                            and not endpoint.password,
                            "请使用 HTTPS 或本机模型地址",
                        )
                        check(
                            isinstance(args["model"], str) and args["model"].strip(),
                            "缺少模型名",
                        )
                        from agent_worker import watch

                        store.state(args["id"])
                        self.server.worker_deck = args["id"]
                        self.server.worker_stop = threading.Event()
                        self.server.worker_model = args["model"]
                        self.server.worker = threading.Thread(
                            target=watch,
                            args=(
                                store,
                                (args["endpoint"], args["model"], args.get("key")),
                                self.server.worker_stop,
                                self.server.worker_deck,
                            ),
                            daemon=True,
                        )
                        self.server.worker.start()
                self.send(
                    200,
                    {
                        "model": self.server.worker_model,
                        "deck": self.server.worker_deck,
                    },
                )
                return
            if path == "/api/create":
                self.send(201, {"id": store.create(blank(args["title"]))})
                return
            if path.startswith("/api/open/"):
                self.send(200, store.state(path.split("/")[-1], opened=True))
                return
            if path.startswith("/api/edit/"):
                self.send(200, store.transact(path.split("/")[-1], args))
                return
        elif path == "/api/agent":
            self.send(
                200,
                {"model": self.server.worker_model, "deck": self.server.worker_deck},
            )
            return
        elif path == "/api/components":
            self.send(200, templates(store))
            return
        elif path == "/api/workspace-options":
            self.send(200, store.workspace_options())
            return
        elif path == "/api/library":
            self.send(200, store.library())
            return
        elif path == "/api/home":
            self.send(200, store.home_items())
            return
        elif path.startswith("/api/cover/"):
            self.send(200, store.cover(path.split("/")[-1]))
            return
        elif path.startswith("/api/state/"):
            self.send(200, store.state(path.split("/")[-1]))
            return
        elif path.startswith("/api/revision/"):
            key = path.split("/")[-1]
            with store.db() as db:
                row = db.execute(
                    "SELECT revision FROM decks WHERE id=?", (key,)
                ).fetchone()
                check(row, "课件不存在")
            result = {"revision": row[0]}
            query = parse_qs(urlsplit(self.path).query)
            if "room" in query:
                result["session"] = session_link.status(store, key, query["room"][0], query.get("since", [None])[0])
            self.send(200, result)
            return
        elif path.startswith("/api/package/"):
            from package_deck import package

            key = path.split("/")[-1]
            name = filename(store.state(key)["deck"]["title"])[:-5] + ".zip"
            self.send(
                200,
                package(store, key),
                "application/zip",
                {"Content-Disposition": "attachment; filename*=UTF-8''" + quote(name)},
            )
            return
        elif path.startswith("/api/export/"):
            key = path.split("/")[-1]
            name = filename(store.state(key)["deck"]["title"])
            self.send(
                200,
                offline(store, key).encode(),
                "text/html; charset=utf-8",
                {"Content-Disposition": "attachment; filename*=UTF-8''" + quote(name)},
            )
            return
        elif path.startswith("/asset/"):
            self.stream_asset(store, path.split("/")[-1])
            return
        self.send(404, {"error": "找不到此入口"})

    def stream_asset(self, store, key):
        with store.db() as db:
            row = db.execute(
                "SELECT rowid,mime,length(content) FROM assets WHERE id=?", (key,)
            ).fetchone()
            check(row, "找不到素材")
            rowid, mime, size = row
            first, last = 0, size - 1
            partial = self.headers.get("Range")
            if partial:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", partial)
                if match and any(match.groups()):
                    a, b = match.groups()
                    first = int(a) if a else max(0, size - int(b))
                    last = min(size - 1, int(b)) if a and b else size - 1
                else:
                    first = size
                if first > last:
                    self.send(
                        416, b"", "text/plain", {"Content-Range": f"bytes */{size}"}
                    )
                    return
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(last - first + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "private, max-age=86400, immutable")
            if partial:
                self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
            self.end_headers()
            with db.blobopen("assets", "content", rowid, readonly=True) as blob:
                blob.seek(first)
                remaining = last - first + 1
                try:
                    while remaining:
                        chunk = blob.read(min(remaining, 262144))
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

    def route(self, write):
        try:
            self.handle_request(write)
        except Conflict as exc:
            self.send(409, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Navigation may cancel an outstanding thumbnail or asset response.
            return
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            self.send(400, {"error": str(exc)})
        except Exception as exc:
            self.send(500, {"error": str(exc)})

    def do_POST(self):
        self.route(True)

    def do_GET(self):
        self.route(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--desktop", action="store_true", help="Run for a desktop host over private stdio")
    parser.add_argument("--open", type=Path, help="Open a local course in the editor")
    parser.add_argument(
        "action",
        nargs="?",
        default="serve",
        choices=[
            "serve",
            "catalog",
            "library",
            "inspect",
            "edit",
            "feedback",
            "create",
            "export",
            "package",
            "import",
            "mcp",
            "agent",
        ],
    )
    parser.add_argument("--args", type=Path, help="UTF-8 JSON parameters")
    args = parser.parse_args()
    store = Store(args.home / "studio.sqlite")
    if args.action == "mcp":
        from mcp_stdio import serve

        serve(store)
        return
    if args.action == "agent":
        from agent_worker import watch

        watch(store)
        return
    if args.action != "serve":
        params = json.loads(args.args.read_text("utf-8-sig")) if args.args else {}
        print(dump(agent_call(store, args.action, params)))
        return
    target = import_path(store, args.open, reopen=True) if args.open else None
    if not store.library():
        store.create(example())
    server = WorkspaceHTTPServer(("127.0.0.1", args.port), HTTP)
    server.daemon_threads = True
    server.worker_deck = None
    server.worker_lock = threading.Lock()
    server.worker = None
    server.worker_model = ""
    server.worker_stop = None
    server.store = store
    wake_stop = threading.Event()
    wake_worker = threading.Thread(target=codex_notify.watch, args=(store, wake_stop), daemon=True)
    wake_worker.start()
    server.token = secrets.token_urlsafe(32)
    query = "?deck=" + target if target else ""
    url = f"http://127.0.0.1:{server.server_port}/{query}#access={server.token}"
    connection = args.home / "session.json"
    if args.desktop:
        # The owning window closes its pipe on exit, including abnormal exit.
        def wait_for_host():
            sys.stdin.buffer.read(1)
            server.shutdown()

        threading.Thread(target=wait_for_host, daemon=True).start()
        print(dump({"url": url, "deck": target}), flush=True)
    else:
        connection.write_text(dump({"url": url, "pid": os.getpid()}), "utf8")
        print("课页 Studio " + url, flush=True)
        if not args.no_browser:
            webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        wake_stop.set()
        wake_worker.join(timeout=10)
        if server.worker_stop:
            server.worker_stop.set()
        server.server_close()
        if (
            not args.desktop and connection.is_file()
            and json.loads(connection.read_text("utf8"))["pid"] == os.getpid()
        ):
            connection.unlink()


if __name__ == "__main__":
    main()
