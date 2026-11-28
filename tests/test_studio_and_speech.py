"""Exercise public Studio stdio and the HTTP speech provider with synthetic inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != "nt", reason="Exercise the Windows public installer and launch paths")
def test_studio_public_stdio(tmp_path):
    receipt = ROOT/".local/runtime-selection.json"
    if not receipt.is_file():
        pytest.skip("Run scripts/setup.py first")
    python = ROOT/json.loads(receipt.read_bytes())["python"]
    workspace = tmp_path/"workspace"
    workspace.mkdir()
    launch = ROOT/"TOOLS/education_mcp/launch.py"
    args = [str(python), "-I", "-X", "utf8", str(launch), "--mcp-root", str(ROOT),
            "--workspace", str(workspace)]
    env = dict(os.environ, CLASSPAGE_STUDIO_HOME=str(tmp_path/"studio"))
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          encoding="utf8", text=True, env=env) as proc:
        def rpc(method, params):
            proc.stdin.write(json.dumps(dict(jsonrpc="2.0", id=1, method=method, params=params))+"\n")
            proc.stdin.flush()
            result = json.loads(proc.stdout.readline())["result"]
            assert not result.get("isError"), result
            return result
        def call(name, **args):
            result = rpc("tools/call", dict(name=name, arguments=args))
            return json.loads(result["content"][0]["text"])
        try:
            assert rpc("initialize", {"protocolVersion":"2025-11-25"})["serverInfo"]["version"] == "0.15.1"
            names = [tool["name"] for tool in rpc("tools/list", {})["tools"]]
            assert len(names) == len(set(names)) == 32
            assert sum(name.startswith("studio_") for name in names) == 14
            call("studio_catalog")
            deck = call("studio_create", title="Public Studio Test")["id"]
            assert call("studio_inspect", id=deck)
            call("studio_export", id=deck, directory=str(workspace))
            exported = workspace/"Public Studio Test.html"
            assert exported.is_file()
            assert call("studio_import", path=str(exported))["id"]
        finally:
            proc.stdin.close()
            assert proc.wait(timeout=30) == 0


def test_http_speech_timestamps_and_audio_time_mapping(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"TOOLS"))
    import numpy as np
    from speech_transcribe import routing, worker
    seen = []
    class Speech(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"text":"sample","language":"English",
                               "timestamps":[{"text":"sample","start":0.1,"end":0.6}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    with ThreadingHTTPServer(("127.0.0.1", 0), Speech) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            config = tmp_path/"provider.json"
            config.write_text(json.dumps(dict(schema="speech-providers-v1", timing="native",
                asr=dict(type="http", endpoint=f"http://127.0.0.1:{server.server_port}/speech",
                         model="fixture", revision="v1", granularity="segment"))), encoding="utf8")
            route = routing.resolve(ROOT, config_path=config)
            def frames(source, timeline):
                timeline.append({"audio_start":0, "media_start":1.5})
                yield np.zeros(16000, dtype=np.float32)
            monkeypatch.setattr(worker, "windows", frames)
            summary = worker.transcribe(ROOT/"MODELS", tmp_path/"audio.wav",
                {"timestamps":True, "language":"English", "source_sha256":"fixture"}, tmp_path, route)
            row = json.loads((tmp_path/"segments.jsonl").read_text("utf8"))
            assert len(seen) == 1 and seen[0]["schema"] == "speech-api-v1"
            assert row["timed_segments"] == [{"text":"sample","start":0.1,"end":0.6}]
            assert summary["audio_to_media"] == [{"audio_start":0,"media_start":1.5}]
            assert summary["http_calls"] == {"asr":1}
        finally:
            server.shutdown()
            thread.join(timeout=5)
