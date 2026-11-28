"""Own at most one private local HTTP server for a managed speech job."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time

from .providers import HTTPProvider


@contextmanager
def open_providers(route, models):
    roles = {name:route[name] for name in ("asr","aligner") if name in route}
    managed = {name:spec for name,spec in roles.items() if spec["type"]=="managed_http"}
    clients = {name:HTTPProvider(spec) for name,spec in roles.items() if spec["type"]=="http"}
    if not managed:
        yield clients
        return
    with tempfile.TemporaryDirectory(prefix="ekb-speech-") as tmp:
        ready = Path(tmp)/"port"
        token = secrets.token_urlsafe(32)
        proc = subprocess.Popen([sys.executable,"-I","-X","utf8",Path(__file__).with_name("local_api.py"),"--ready",ready],
                                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
                                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),
                                env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
        try:
            request = dict(roles=managed,folder=str(models),token=token)
            proc.stdin.write(json.dumps(request).encode()+b"\n")
            proc.stdin.close()
            deadline = time.monotonic()+20
            while not ready.exists():
                if proc.poll() is not None or time.monotonic()>=deadline:
                    raise RuntimeError("Local speech API failed to start")
                time.sleep(.05)
            port=int(ready.read_text("ascii"))
            for name,spec in managed.items():
                path = "transcribe" if name=="asr" else "align"
                endpoint = dict(model=spec["model"],revision=spec["manifest_sha256"],granularity="word_or_character",
                                endpoint=f"http://127.0.0.1:{port}/{path}")
                clients[name] = HTTPProvider(endpoint,token=token,timeout=86400,lossless=True)
            yield clients
        finally:
            ready.unlink(missing_ok=True)
            proc.stdin.close()
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
