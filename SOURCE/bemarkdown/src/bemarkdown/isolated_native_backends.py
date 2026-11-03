"""Installable, owned native-model actors with serialized stage boundaries.

The process-wide RLock covers switching, initialization and the complete RPC.
An automatic switch unloads an actor but keeps its proxy reusable. Explicit
close/unload and failed/cancelled RPCs close the proxy permanently. Production
uses NativeBackendFactory injection; activate is only a legacy adapter.
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import weakref
from multiprocessing.connection import Client, Listener

if __package__:
    from .file_handoff import RetryPolicy, atomic_write_text, read_text
else:
    # The actor is launched as a direct script to avoid eager package imports.
    from file_handoff import RetryPolicy, atomic_write_text, read_text

_stage_lock = threading.RLock()
_active = None
_table_staged_active = False
_native_isolation_active = False


class ResultView:
    def __init__(self, value):
        self.json = value


def normalized_result(value):
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    if isinstance(value, dict):
        return {key: normalized_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized_result(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalized_result(item) for item in value)
    if hasattr(value, "json"):
        data = value.json() if callable(value.json) else value.json
        return {"__isolated_result_json__": json.loads(data) if isinstance(data, str) else normalized_result(data)}
    # NumPy/PIL and BeMarkdown public data classes retain pickle semantics.
    return value


def decoded_result(value):
    if isinstance(value, dict):
        if set(value) == {"__isolated_result_json__"}:
            return ResultView(value["__isolated_result_json__"])
        return {key: decoded_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decoded_result(item) for item in value]
    if isinstance(value, tuple):
        return tuple(decoded_result(item) for item in value)
    return value


class RecProxy:
    def __init__(self, owner):
        self.owner = owner

    def predict(self, *args, **kwargs):
        return self.owner._request("rec_predict", args, kwargs)


def _json_config(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_config(item) for item in value]
    return value


def _bounded_call(function, timeout, label):
    """Bound connection/auth/send/recv too, not merely socket readability."""
    done = threading.Event()
    abandoned = threading.Event()
    ownership = threading.Lock()
    result = []

    def run():
        try:
            value = function()
            # A connection arriving after cancellation must never be leaked.
            with ownership:
                if abandoned.is_set() and hasattr(value, "close"):
                    value.close()
                result.append((True, value))
        except BaseException as error:
            result.append((False, error))
        finally:
            done.set()

    thread = threading.Thread(target=run, name="bemarkdown-actor-transport", daemon=True)
    thread.start()
    try:
        if not done.wait(timeout):
            raise TimeoutError(label)
        ok, value = result[0]
        if not ok:
            raise value
        return value
    except BaseException:
        with ownership:
            abandoned.set()
            if result and result[0][0] and hasattr(result[0][1], "close"):
                result[0][1].close()
        raise


class RemoteBackend:
    def __init__(self, kind, args=(), kwargs=None, *, work_root=None,
                 dependency_paths=(), interpreter=None, startup_timeout=300,
                 request_timeout=7200, stop_timeout=60, backend_loader=None):
        self.kind = kind
        self.arguments = args
        self.options = dict(kwargs or {})
        self.work_root = Path(work_root or Path(tempfile.gettempdir()) / "bemarkdown-isolated-backends")
        self.dependency_paths = tuple(str(Path(path).resolve()) for path in dependency_paths)
        # Windows venv python.exe is a redirector: launch its real interpreter.
        self.interpreter = str(interpreter or getattr(sys, "_base_executable", sys.executable))
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.stop_timeout = stop_timeout
        self.backend_loader = backend_loader
        self._process = None
        self._connection = None
        self._log = None
        self._work = None
        self._calls = 0
        self._starts = 0
        self._cached = {}
        self._closed = False
        self._events = []
        self._last_exit_code = None
        self._ownership_error = None
        self._persistent_diagnostics = None

    def _record(self, event, **details):
        self._events.append({"event": event, "time": time.time(), **details})
        if self._work is not None:
            # Evidence survives process termination and every restart.
            try:
                path = self._work / "lifecycle.json"
                atomic_write_text(path, json.dumps(self._events, ensure_ascii=False, indent=2))
            except OSError:
                # A full disk must not prevent process cleanup.
                pass

    def _launch(self, command, environment):
        # OwnedStageProcess additionally binds descendants to a Windows Job.
        from .owned_stage_process import OwnedStageProcess
        stage = {"layout": "pp-doclayout", "ocr": "pp-ocrv6", "formula": "pp-formulanet"}.get(self.kind, self.kind)
        return OwnedStageProcess(command, env=environment, stdout=self._log, stderr=self._log,
                                 stage=stage, journal_path=self._work / "resources.jsonl")

    def _retain_failure(self, error):
        if self._work is None:
            return
        try:
            if self._persistent_diagnostics is None:
                from .stage_failure import retain_stage_failure
                report = retain_stage_failure(self._work, stage=self.kind,
                                              exit_code=self._last_exit_code, error=error)
                self._persistent_diagnostics = report["persistent_diagnostics"]
                self._record("failure_retained", path=self._persistent_diagnostics)
            error.persistent_diagnostics = self._persistent_diagnostics
            error.add_note(f"Persistent actor diagnostics: {self._persistent_diagnostics}")
        except Exception as persistence_error:
            # Preserve the primary failure and its original diagnostic folder.
            error.add_note(f"Could not retain actor diagnostics from {self._work}: {persistence_error}")

    def _ensure(self):
        global _active
        with _stage_lock:
            if self._closed:
                raise RuntimeError("ISOLATED_BACKEND_CLOSED")
            if self._process is not None:
                try:
                    if self._process.poll() is None and self._connection is not None:
                        return
                except BaseException as error:
                    self._closed = True
                    self._record("supervision_failed", error_type=type(error).__name__, error=str(error))
                    try:
                        self._stop(graceful=False)
                    except BaseException as cleanup_error:
                        error.add_note(f"Actor cleanup also failed: {cleanup_error}")
                    self._retain_failure(error)
                    raise
                self._stop(graceful=False)
            if _active is not None and _active is not self:
                _active._stop()
            self._starts += 1
            self.work_root.mkdir(parents=True, exist_ok=True)
            self._work = self.work_root / (self.kind + "-" + secrets.token_hex(8))
            self._work.mkdir()
            self._events = []
            self._persistent_diagnostics = None
            token = secrets.token_bytes(32)
            nonce = secrets.token_hex(32)
            core_src = Path(__file__).resolve().parents[1]
            environment = os.environ.copy()
            paths = [str(core_src), *self.dependency_paths]
            # Preserve installed venv dependencies when launching the base exe.
            paths.extend(str(Path(path).resolve()) for path in sys.path if path and Path(path).is_dir())
            paths.extend(environment.get("PYTHONPATH", "").split(os.pathsep))
            environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
            config = self._work / "config.json"
            try:
                atomic_write_text(config, json.dumps(_json_config({
                    "kind": self.kind, "args": self.arguments, "kwargs": self.options,
                    "auth": token.hex(), "nonce": nonce, "attributes": self._cached,
                    "backend_loader": self.backend_loader,
                }), ensure_ascii=False))
                self._log = (self._work / "worker.log").open("wb")
                # Direct core script avoids eager package imports before bootstrap.
                command = [self.interpreter, "-X", "utf8", str(Path(__file__).resolve()), "--actor", str(config)]
                self._record("launch", interpreter=self.interpreter, pythonpath=environment["PYTHONPATH"])
                self._process = self._launch(command, environment)
                self._record("started", pid=self._process.pid)
                ready = self._work / "ready.json"
                deadline = time.monotonic() + self.startup_timeout
                while not ready.exists():
                    if self._process.poll() is not None:
                        raise RuntimeError(f"ISOLATED_BACKEND_START_FAILED:{self.kind}:{self._work}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("ISOLATED_BACKEND_START_TIMEOUT")
                    time.sleep(.02)
                data = json.loads(read_text(ready, policy=RetryPolicy(
                    timeout_seconds=min(2.0, max(0.0, deadline - time.monotonic())))))
                address = data.get("address")
                if (data.get("pid") != self._process.pid or data.get("nonce") != nonce
                        or not isinstance(address, list) or len(address) != 2
                        or address[0] != "127.0.0.1" or not isinstance(address[1], int)
                        or not 0 < address[1] < 65536):
                    raise RuntimeError("ISOLATED_BACKEND_IDENTITY_MISMATCH")
                self._connection = _bounded_call(
                    lambda: Client(tuple(address), authkey=token),
                    max(0, deadline - time.monotonic()), "ISOLATED_BACKEND_CONNECT_TIMEOUT")
                _active = self
                self._record("connected", pid=self._process.pid)
            except BaseException as error:
                self._closed = True
                self._record("startup_failed", error_type=type(error).__name__, error=str(error))
                try:
                    self._stop(graceful=False)
                except BaseException as cleanup_error:
                    error.add_note(f"Actor cleanup also failed: {cleanup_error}")
                self._retain_failure(error)
                raise

    def _request(self, method, args=(), kwargs=None):
        with _stage_lock:
            self._ensure()
            self._calls += 1
            connection = self._connection

            def exchange():
                connection.send({"method": method, "args": args, "kwargs": kwargs or {}})
                return connection.recv()

            remote_missing_attribute = False
            try:
                value = _bounded_call(exchange, self.request_timeout, "ISOLATED_BACKEND_REQUEST_TIMEOUT")
                if not isinstance(value, dict) or "ok" not in value:
                    raise RuntimeError("ISOLATED_BACKEND_INVALID_RESPONSE")
                if not value["ok"]:
                    if value.get("error_type") == "AttributeError":
                        # getattr(..., default) and hasattr are legitimate
                        # capability discovery, not failed inference requests.
                        remote_missing_attribute = method == "describe"
                        raise AttributeError(value.get("error"))
                    raise RuntimeError(f"ISOLATED_BACKEND_ERROR:{value.get('error_type')}:{value.get('error')}:{self._work}")
                return decoded_result(value["value"])
            except BaseException as error:
                if remote_missing_attribute:
                    raise
                self._closed = True
                self._record("request_failed", method=method, error_type=type(error).__name__, error=str(error))
                try:
                    self._stop(graceful=False)
                except BaseException as cleanup_error:
                    error.add_note(f"Actor cleanup also failed: {cleanup_error}")
                self._retain_failure(error)
                raise

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "release_workspace" and self._closed:
            # Finally blocks may release workspace after a cancelled/failed
            # RPC has already retired the actor. This cleanup-only exception
            # must preserve the primary error and never perform another RPC.
            return lambda: {"release_status": "PROCESS_ALREADY_EXITED"}
        if name == "_rec_model":
            return RecProxy(self)
        value = self._request("describe", (name,))
        if value["callable"]:
            return lambda *args, **kwargs: self._request(name, args, kwargs)
        return value["value"]

    @property
    def retained_by_owner(self):
        return False

    @property
    def character_positions(self):
        return self._cached.get("character_positions", False)

    @character_positions.setter
    def character_positions(self, value):
        with _stage_lock:
            self._cached["character_positions"] = value
            self._request("setattr", ("character_positions", value))

    def __call__(self, *args, **kwargs):
        return self._request("__call__", args, kwargs)

    def unload(self):
        with _stage_lock:
            try:
                if self._closed or self._process is None:
                    return {"unload_status": "PROCESS_ALREADY_EXITED"}
                return self._request("unload")
            finally:
                self._closed = True
                self._stop()

    def close(self):
        with _stage_lock:
            try:
                if not self._closed and self._process is not None:
                    self._request("close")
            finally:
                self._closed = True
                self._stop()

    def _stop(self, *, graceful=True):
        """Idempotent actor eviction; explicit close alone seals the proxy."""
        global _active
        with _stage_lock:
            process, connection, log = self._process, self._connection, self._log
            cleanup_error = self._ownership_error
            stop_error = None

            def exit_code():
                nonlocal cleanup_error
                try:
                    return process.poll()
                except BaseException as error:
                    # Resource rejection is an error report, not proof the root
                    # is still alive. close() must run before inspecting its code.
                    cleanup_error = cleanup_error or error
                    self._closed = True
                    self._record("supervision_failed", error_type=type(error).__name__, error=str(error))
                    return process.returncode

            try:
                if process is not None:
                    if exit_code() is None and graceful and connection is not None:
                        try:
                            _bounded_call(lambda: connection.send({"method": "STOP", "args": (), "kwargs": {}}),
                                          self.stop_timeout, "ISOLATED_BACKEND_STOP_TIMEOUT")
                            process.wait(timeout=self.stop_timeout)
                        except BaseException as error:
                            stop_error = error
                            self._record("stop_failed", error_type=type(error).__name__, error=str(error))
                    if exit_code() is None:
                        self._record("terminate_owned", pid=process.pid)
                        process.terminate()
                        try:
                            process.wait(timeout=self.stop_timeout)
                        except subprocess.TimeoutExpired:
                            self._record("kill_owned", pid=process.pid)
                            process.kill()
                            process.wait(timeout=self.stop_timeout)
            except BaseException as error:
                cleanup_error = error
                self._closed = True
                self._record("cleanup_failed", error_type=type(error).__name__, error=str(error))
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except BaseException as error:
                        self._record("connection_close_failed", error=str(error))
                if process is not None and hasattr(process, "close"):
                    try:
                        process.close()
                    except BaseException as error:
                        cleanup_error = cleanup_error or error
                        self._ownership_error = error
                        self._closed = True
                        self._record("owned_close_failed", error=str(error))
                if process is not None:
                    self._last_exit_code = process.native_poll() if hasattr(process, "native_poll") else process.returncode
                    if self._last_exit_code is None or (hasattr(process, "exit_confirmed") and not process.exit_confirmed):
                        cleanup_error = cleanup_error or RuntimeError("ISOLATED_BACKEND_EXIT_UNCONFIRMED")
                        self._ownership_error = cleanup_error
                        self._closed = True
                        self._record("exit_unconfirmed", pid=process.pid)
                    else:
                        self._record("exited", pid=process.pid, exit_code=self._last_exit_code)
                if log is not None:
                    try:
                        log.close()
                    except BaseException as error:
                        cleanup_error = cleanup_error or error
                        self._record("log_close_failed", error=str(error))
                # Keep ownership on unconfirmed exit; never allow a new actor.
                if (process is None or self._last_exit_code is not None) and self._ownership_error is None:
                    self._process = None
                    if _active is self:
                        _active = None
                elif self._ownership_error is not None:
                    # Root exit alone cannot certify descendant Job cleanup.
                    # Retain a poisoned owner to reject subsequent stage loads.
                    _active = self
                self._connection = None
                self._log = None
            if cleanup_error is not None:
                self._retain_failure(cleanup_error)
                raise cleanup_error
            if stop_error is not None:
                self._retain_failure(stop_error)


def close_active():
    with _stage_lock:
        if _active is not None:
            _active._stop()


@contextmanager
def stage_boundary():
    """Exclude native actors throughout an externally owned model stage."""
    with _stage_lock:
        close_active()
        yield


class NativeBackendFactory:
    """Inject callable adapters without globally replacing native classes."""
    native_isolation_active = True

    def __init__(self, *, models_root=None, config_path=None, mcp_root=None, **actor_options):
        self.actor_options = actor_options
        self.registry_options = {key: value for key, value in {
            "models_root": models_root, "config_path": config_path, "mcp_root": mcp_root,
        }.items() if value is not None}
        self._instances = weakref.WeakSet()
        self._closed = False

    def create(self, kind, *args, **kwargs):
        with _stage_lock:
            if self._closed:
                raise RuntimeError("ISOLATED_BACKEND_FACTORY_CLOSED")
            backend = RemoteBackend(kind, args, {**self.registry_options, **kwargs}, **self.actor_options)
            self._instances.add(backend)
            return backend

    def layout(self, **kwargs):
        return self.create("layout", **kwargs)

    def ocr(self, **kwargs):
        return self.create("ocr", **kwargs)

    def table(self, *args, **kwargs):
        return self.create("table", *args, **kwargs)

    def formula(self, **kwargs):
        return self.create("formula", **kwargs)

    def close_active(self):
        close_active()

    def stage_boundary(self):
        return stage_boundary()

    def close(self):
        with _stage_lock:
            self._closed = True
            first_error = None
            for backend in list(self._instances):
                try:
                    backend.close()
                except BaseException as error:
                    first_error = first_error or error
            if first_error is not None:
                raise first_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def require_table_staged_activation():
    if not _table_staged_active:
        raise RuntimeError("TABLE_STAGES_REQUIRE_ACTIVE_ISOLATED_BACKENDS")


def require_native_backend_activation():
    if not _native_isolation_active:
        raise RuntimeError("NATIVE_LOCAL_REQUIRES_ACTIVE_ISOLATED_BACKENDS")


def activate(*, table_staged=False, factory=None):
    """Legacy monkeypatch adapter; production injects a factory directly."""
    global _table_staged_active, _native_isolation_active
    from bemarkdown import pdf_layout_runtime, pdf_content_router, formula_ocr
    from bemarkdown.pdf import production_runtime, table_runtime
    factory = factory or NativeBackendFactory()
    pdf_layout_runtime.FormalPaddleLayoutRuntime = factory.layout
    production_runtime.FormalPaddleLayoutRuntime = factory.layout
    pdf_content_router.PaddleXPdfOcrRuntime = factory.ocr
    if not table_staged:
        table_runtime.RegistryBackedTableEvidenceProvider = factory.table
        production_runtime.RegistryBackedTableEvidenceProvider = factory.table
    formula_ocr.create_production_formulanet_runtime = factory.formula
    _table_staged_active = bool(table_staged)
    _native_isolation_active = True
    return factory


def _native_runtime(kind, args, kwargs):
    # Each installed core runtime obtains ModelRegistry from .model_registry;
    # models_root/config_path/mcp_root remain the caller's unchanged kwargs.
    targets = {
        "layout": ("bemarkdown.pdf_layout_runtime", "FormalPaddleLayoutRuntime"),
        "ocr": ("bemarkdown.pdf_content_router", "PaddleXPdfOcrRuntime"),
        "table": ("bemarkdown.pdf.table_runtime", "RegistryBackedTableEvidenceProvider"),
        "formula": ("bemarkdown.formula_ocr", "create_production_formulanet_runtime"),
    }
    module_name, name = targets[kind]
    return getattr(importlib.import_module(module_name), name)(*args, **kwargs)


def actor(config):
    data = json.loads(read_text(Path(config)))
    folder = Path(config).parent
    loader = _native_runtime
    if data.get("backend_loader"):
        module_name, name = data["backend_loader"].split(":", 1)
        loader = getattr(importlib.import_module(module_name), name)
    runtime = loader(data["kind"], data["args"], data["kwargs"])
    for name, value in data.get("attributes", {}).items():
        setattr(runtime, name, value)
    with Listener(("127.0.0.1", 0), authkey=bytes.fromhex(data["auth"])) as listener:
        ready = folder / "ready.json"
        atomic_write_text(ready, json.dumps({"address": listener.address, "pid": os.getpid(), "nonce": data["nonce"]}))
        with listener.accept() as connection:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    break
                method, arguments, options = request["method"], request["args"], request["kwargs"]
                if method == "STOP":
                    break
                try:
                    if method == "describe":
                        value = getattr(runtime, arguments[0])
                        value = {"callable": callable(value), "value": None if callable(value) else value}
                    elif method == "setattr":
                        setattr(runtime, *arguments)
                        value = None
                    elif method == "rec_predict":
                        value = [normalized_result(item) for item in runtime._rec_model.predict(*arguments, **options)]
                    else:
                        value = getattr(runtime, method)(*arguments, **options)
                    connection.send({"ok": True, "value": normalized_result(value)})
                except BaseException as error:
                    traceback.print_exc()
                    connection.send({"ok": False, "error_type": type(error).__name__, "error": type(error).__name__ + ": " + str(error)})


atexit.register(close_active)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True)
    actor(parser.parse_args().actor)
