"""One owned model process per stage, with atomic per-request receipts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import re


SESSION_SCHEMA = "bemarkdown-model-session-v2"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def record_progress(value):
    home, job = os.environ.get("EKB_RUN_MANAGER_HOME"), os.environ.get("EKB_MANAGED_JOB_ID", "")
    if home and re.fullmatch(r"[a-f0-9]{32}", job):
        folder = Path(home) / "jobs" / job
        if folder.is_dir():
            write_json(folder / "model_api_progress.json", {**value, "job_id": job, "updated_epoch": time.time()})


def owned_processes(child):
    import psutil
    try:
        parent = psutil.Process(child.pid)
        return [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def wait_processes_exit(processes, timeout=15):
    import psutil
    _, alive = psutil.wait_procs(processes, timeout=timeout)
    if alive:
        raise RuntimeError("MODEL_SESSION_TREE_EXIT_UNCONFIRMED")


def cleanup_temporary(temporary):
    # Windows can briefly retain closed file handles after process termination.
    for attempt in range(6):
        try:
            temporary.cleanup()
            return
        except PermissionError as error:
            if os.name != "nt" or error.winerror not in {5, 32} or attempt == 5:
                raise
            time.sleep(.05 * 2**attempt)


def stop_process_tree(child, timeout=15):
    descendants = owned_processes(child)
    if child.poll() is not None:
        wait_processes_exit(descendants, timeout)
        return
    if os.name == "nt":
        stopped = subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                 capture_output=True, timeout=timeout)
        if stopped.returncode and child.poll() is None:
            raise RuntimeError("MODEL_SESSION_TREE_EXIT_UNCONFIRMED")
    else:
        import signal
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=timeout)
    wait_processes_exit(descendants, timeout)


class SessionProcess:
    """Serial owner of a reusable worker; all children inherit the manager Job."""

    def __init__(self, config_path, role, *, timeout=540, idle_seconds=120, journal_root=None, session_id=None):
        if not os.environ.get("EKB_MANAGED_JOB_ID"):
            raise RuntimeError("MODEL_SESSION_MANAGED_JOB_REQUIRED")
        self.session_id = session_id or uuid.uuid4().hex
        self.role = role
        self.timeout = timeout
        self.sequence = 0
        self.closed = False
        self.close_receipt = None
        self.config_path = Path(config_path).resolve()
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.temporary = tempfile.TemporaryDirectory(prefix="bemarkdown-session-")
        self.folder = Path(self.temporary.name)
        self.journal = Path(journal_root) / self.session_id if journal_root else self.folder / "receipts"
        self.journal.mkdir(parents=True, exist_ok=True)
        self.log = (self.folder / "worker.log").open("wb")
        command = [sys.executable, "-m", "bemarkdown.model_session", "--config", str(self.config_path),
                   "--role", role, "--folder", str(self.folder), "--session-id", self.session_id,
                   "--idle-seconds", str(idle_seconds)]
        try:
            self.child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=os.name != "nt")
            write_json(self.journal / "OPENED.json", {"session_id": self.session_id, "role": role,
                       "worker_pid": self.child.pid, "config_sha256": self.config_sha256})
        except BaseException:
            self.log.close()
            self.temporary.cleanup()
            raise

    def status(self, request_id=None):
        result = {"session_id": self.session_id, "model_unloaded": self.closed or self.child.poll() is not None,
                  "state": "CLOSED" if self.closed or self.child.poll() is not None else "OPEN"}
        if request_id is not None:
            key = hashlib.sha256(request_id.encode()).hexdigest()
            receipt = self.journal / (key + ".json")
            result["request_state"] = "COMPLETE" if receipt.exists() else "UNKNOWN"
            if receipt.exists():
                result["receipt"] = json.loads(receipt.read_text("utf-8"))
        return result

    def request(self, payload):
        if self.closed:
            raise RuntimeError("MODEL_SESSION_CLOSED")
        if hashlib.sha256(self.config_path.read_bytes()).hexdigest() != self.config_sha256:
            self.close(force=True)
            raise RuntimeError("MODEL_SESSION_CONFIG_CHANGED")
        if payload["role"] != self.role:
            raise ValueError("MODEL_SESSION_ROLE_MISMATCH")
        key = hashlib.sha256(payload["request_id"].encode()).hexdigest()
        receipt = self.journal / (key + ".json")
        fingerprint = payload_digest(payload)
        if receipt.exists():
            saved = json.loads(receipt.read_text("utf-8"))
            if saved["payload_sha256"] != fingerprint:
                raise ValueError("MODEL_SESSION_REQUEST_ID_REUSED_WITH_DIFFERENT_INPUT")
            return saved["result"]
        # Preserve completed receipts; fail before inference when the per-job budget is full.
        journal_bytes = sum(p.stat().st_size for p in self.journal.parent.glob('*/*.json'))
        if journal_bytes + 64 * 1024**2 > 512 * 1024**2:
            raise RuntimeError("MODEL_SESSION_JOURNAL_BUDGET_EXCEEDED")
        sequence = self.sequence
        self.sequence += 1
        request = self.folder / f"request-{sequence:08d}.json"
        response = self.folder / f"response-{sequence:08d}.json"
        write_json(request, payload)
        deadline = time.monotonic() + self.timeout
        try:
            while not response.exists():
                if self.child.poll() is not None:
                    raise RuntimeError(f"MODEL_SESSION_WORKER_EXITED:{self.child.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("MODEL_SESSION_REQUEST_TIMEOUT")
                time.sleep(.02)
            value = json.loads(response.read_text("utf-8"))
            if value.get("error"):
                raise RuntimeError(value["error"])
            if value.get("request_id") != payload["request_id"] or value.get("session_id") != self.session_id:
                raise ValueError("MODEL_SESSION_RESPONSE_BINDING_INVALID")
            write_json(receipt, {"payload_sha256": fingerprint, "result": value})
            request.unlink(missing_ok=True)
            response.unlink()
            return value
        except BaseException:
            self.close(force=True)
            raise

    def close(self, *, force=False):
        if self.closed:
            return dict(self.close_receipt)
        descendants = owned_processes(self.child)
        try:
            if force:
                stop_process_tree(self.child)
            else:
                (self.folder / "STOP").touch()
                try:
                    self.child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    stop_process_tree(self.child)
            if self.child.poll() is None:
                raise RuntimeError("MODEL_SESSION_EXIT_UNCONFIRMED")
            wait_processes_exit(descendants)
            self.closed = True
            result = {"session_id": self.session_id, "model_unloaded": True,
                      "worker_exit_code": self.child.returncode}
            self.close_receipt = result
            write_json(self.journal / "CLOSED.json", result)
            return result
        finally:
            if self.closed:
                self.log.close()
                with (self.folder / "worker.log").open("rb") as stream:
                    stream.seek(max(0, (self.folder / "worker.log").stat().st_size - 8192))
                    (self.journal / "worker-tail.log").write_bytes(stream.read())
                cleanup_temporary(self.temporary)


def worker(config, role, folder, session_id, idle_seconds, *, backend_factory=None):
    from .local_model_server import infer, make_backend
    backend = (backend_factory or make_backend)(role, config["roles"][role])
    sequence = 0
    last_activity = time.monotonic()
    try:
        while not (folder / "STOP").exists():
            request = folder / f"request-{sequence:08d}.json"
            if not request.exists():
                if time.monotonic() - last_activity > idle_seconds:
                    return
                time.sleep(.02)
                continue
            response = folder / f"response-{sequence:08d}.json"
            payload = json.loads(request.read_text("utf-8"))
            try:
                if payload["role"] != role:
                    raise ValueError("MODEL_SESSION_ROLE_MISMATCH")
                before_count = getattr(backend, "metrics", {}).get("model_load_count")
                before_load = getattr(backend, "metrics", {}).get("total_load_seconds")
                result = infer(payload, config, backend_factory=lambda *_: backend, release_backend=False)
                result.update(schema=SESSION_SCHEMA, session_id=session_id, request_complete=True,
                              model_unloaded=False, released=False)
                count = backend.metrics.get("model_load_count")
                total = backend.metrics.get("total_load_seconds")
                result["metrics"]["load_count"] = count - (before_count or 0) if count is not None else None
                result["metrics"]["load_seconds"] = total - (before_load or 0) if total is not None else None
                write_json(response, result)
            except Exception as exc:
                write_json(response, {"error": f"{type(exc).__name__}:{exc}"})
                return
            sequence += 1
            last_activity = time.monotonic()
    finally:
        backend.close()


def main():
    import argparse
    from .local_model_server import load_server_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("layout", "formula", "table", "text"))
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--idle-seconds", required=True, type=float)
    args = parser.parse_args()
    if not os.environ.get("EKB_MANAGED_JOB_ID"):
        raise RuntimeError("MODEL_SESSION_MANAGED_JOB_REQUIRED")
    if args.idle_seconds <= 0:
        raise ValueError("MODEL_SESSION_IDLE_TIMEOUT_INVALID")
    worker(load_server_config(args.config), args.role, args.folder, args.session_id, args.idle_seconds)


if __name__ == "__main__":
    main()
