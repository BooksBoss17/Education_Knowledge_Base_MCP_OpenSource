"""Education MCP adapter. Studio imports and its document store live in an isolated worker."""
from contextvars import ContextVar
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent
SCHEMAS = json.loads((ROOT / "tools.json").read_text("utf8"))
NAMES = frozenset(tool["name"] for tool in SCHEMAS)
stopping = ContextVar("studio_receive_stop", default=None)


def call(service, name, args):
    request = json.dumps(dict(name=name, args=args, workspace=str(service.workspace),
                              roots=[str(p) for p in service.allowed])).encode("utf8")
    stop = stopping.get()
    with subprocess.Popen(
        [sys.executable, "-I", "-B", "-X", "utf8", str(ROOT / "worker.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as child:
        try:
            while True:
                if stop is not None and stop.is_set():
                    return {"content":[{"type":"text","text":'{"status":"cancelled"}'}]}
                try:
                    output, _ = child.communicate(request, timeout=.25 if stop is not None else None)
                    break
                except subprocess.TimeoutExpired:
                    request = None
            if child.returncode:
                raise RuntimeError("课页 Studio 后端异常退出 (%s)" % child.returncode)
            return json.loads(output)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()


class Receivers:
    """Only long Studio receives leave the serial request loop."""
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, ident, invoke, reply):
        stop = threading.Event()
        def run():
            token = stopping.set(stop)
            try:
                try:
                    result = invoke()
                except Exception as exc:
                    result = {"isError":True,"content":[{"type":"text","text":str(exc)}]}
                reply({"jsonrpc":"2.0","id":ident,"result":result})
            finally:
                stopping.reset(token)
                with self.lock:
                    self.jobs.pop(ident, None)
        with self.lock:
            if ident in self.jobs or len(self.jobs) >= 16:
                raise ValueError("已有等待请求，请稍后再试")
            thread = threading.Thread(target=run, daemon=True)
            self.jobs[ident] = stop, thread
            thread.start()

    def cancel(self, ident):
        with self.lock:
            if ident in self.jobs:
                self.jobs[ident][0].set()

    def close(self):
        with self.lock:
            jobs = list(self.jobs.values())
        for stop, _ in jobs:
            stop.set()
        for _, thread in jobs:
            thread.join()
