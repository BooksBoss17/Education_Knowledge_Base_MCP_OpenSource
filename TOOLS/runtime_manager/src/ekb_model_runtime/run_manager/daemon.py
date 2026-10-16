"""User-level durable scheduler. This module starts no inference frameworks."""
from __future__ import annotations
import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import socketserver
import sys
import threading
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ekb_model_runtime.run_manager"

from .common import ACTIVE, RunManagerError, RotatingLog, atomic_json, bounded_number, canonical, code_fingerprint, file_sha, fingerprint
from .ipc import auth_key, mac, receive, secure_root, send
from .process import NamedMutex, OwnedProcess, WindowsJob, in_any_job, mutex_name, process_identity
from .registry import ResourceAdapter, load_registry, requirement
from .storage import Store


class Manager:
    def __init__(self, root, registry):
        self.root, self.registry = Path(root), registry
        self.store = Store(root)
        self.adapter = ResourceAdapter(registry["resource_policy"])
        self.key = auth_key(root)
        self.active = {}
        self.active_lock = threading.RLock()
        self.stop = threading.Event()
        self.events = RotatingLog(self.root / "manager.jsonl")
        self.identity = process_identity(os.getpid())
        self.server = None
        self.endpoint = {"schema": "ekb-run-manager-endpoint-v1", "host": "127.0.0.1", "identity": self.identity,
                         "code_fingerprint": code_fingerprint(), "registry_fingerprint": registry["registry_fingerprint"],
                         "independent_of_client_job": True, "host_job_observed": in_any_job(),
                         "independence_basis": "MANDATORY_CREATE_BREAKAWAY_FROM_JOB; outer host Job may remain"}
        for job in self.store.list(("STARTING", "RUNNING", "CANCEL_REQUESTED"), limit=10000):
            old = job.get("process_identity")
            observed = process_identity(old["pid"]) if old else None
            self.store.transition(job["job_id"], "INTERRUPTED", error={"code": "MANAGER_RESTART_NO_AUTORETRY",
                                  "previous_identity": old, "observed_identity": observed,
                                  "identity_matches": bool(old and old == observed)})

    def dispatch(self, operation, args):
        if operation == "ping":
            return self.endpoint
        if operation == "register_workspace":
            workspace = Path(args["workspace"]).resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            programs = args["program_ids"]
            if not programs or set(programs) - set(self.registry["programs"]):
                raise RunManagerError("UNREGISTERED_PROGRAM")
            roots = [str(workspace)] + [str(Path(path).resolve(strict=True)) for path in args.get("input_roots", [])]
            if any(not Path(path).is_dir() for path in roots):
                raise RunManagerError("INPUT_ROOT_NOT_DIRECTORY")
            return self.store.register_workspace(workspace, sorted(set(programs)), sorted(set(roots)))
        if operation == "submit":
            return self.submit(**args)
        if operation == "status":
            return self.store.status(args["job_id"])
        if operation == "lookup":
            return self.store.lookup(args["program_id"], str(Path(args["workspace"]).resolve()), args["public_id"])
        if operation == "find_scope":
            return self.store.find_scope(args["program_id"], str(Path(args["workspace"]).resolve()), args["scope_key"])
        if operation == "list":
            states = args.get("states")
            if states is not None and (not isinstance(states, list) or set(states) - set(ACTIVE) - {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED", "CAPACITY_INSUFFICIENT", "WAIT_EXPIRED"}):
                raise RunManagerError("INVALID_STATE_FILTER")
            limit = int(bounded_number(args.get("limit", 20), "limit", 20))
            return self.store.list(states, limit=limit)
        if operation == "adopt":
            return self.store.adopt(args["job_id"], args["client_id"])
        if operation == "cancel":
            job = self.store.status(args["job_id"])
            reason = str(args.get("reason", "user"))[:500]
            if job["state"] in ("QUEUED", "WAITING_RESOURCES"):
                return self.store.transition(job["job_id"], "CANCELLED", error={"code": "USER_CANCELLED", "reason": reason}, expected=("QUEUED", "WAITING_RESOURCES"))
            if job["state"] in ("STARTING", "RUNNING", "CANCEL_REQUESTED"):
                return self.store.transition(job["job_id"], "CANCEL_REQUESTED", error={"code": "USER_CANCELLED", "reason": reason}, expected=("STARTING", "RUNNING", "CANCEL_REQUESTED"))
            return job
        if operation == "owns_process":
            with self.active_lock:
                owned = self.active.get(args["job_id"])
                proof = bool(owned and owned.job.owns(args["pid"], args["created"]))
            return {"owned": proof, "job_id": args["job_id"], "pid": args["pid"], "created": args["created"],
                    "proof": "LIVE_WINDOWS_JOB_MEMBERSHIP_AND_CREATION_TIME" if proof else "NOT_PROVEN"}
        if operation == "shutdown":
            if self.store.list(ACTIVE, limit=1):
                raise RunManagerError("DAEMON_HAS_ACTIVE_TASKS")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return {"shutdown_requested": True}
        raise RunManagerError("UNKNOWN_OPERATION")

    def submit(self, program_id, workspace, payload, *, scope_key=None, requirement=None, policy=None,
               wait_timeout=900, retry_of=None, config_fingerprint=None, run_timeout=None):
        workspace = str(Path(workspace).resolve(strict=True))
        registered = self.store.workspace(workspace)
        if program_id not in registered["program_ids"] or program_id not in self.registry["programs"]:
            raise RunManagerError("UNREGISTERED_PROGRAM")
        if not isinstance(payload, dict) or len(canonical(payload).encode("utf-8")) > 65536:
            raise RunManagerError("INVALID_TASK_PAYLOAD")
        forbidden = {"env", "environment", "argv", "shell", "command", "interpreter", "script", "token", "password", "secret", "api_key", "authorization"}
        def check(value):
            if isinstance(value, dict):
                if any(str(key).lower() in forbidden for key in value):
                    raise RunManagerError("UNSAFE_PAYLOAD_FIELD")
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(payload)
        payload = json.loads(canonical(payload))
        extra_sources = payload.get("additional_sources", [])
        if not isinstance(extra_sources, list) or len(extra_sources) > 31:
            raise RunManagerError("INVALID_ADDITIONAL_SOURCES")
        if extra_sources and program_id != 'native-convert':
            raise RunManagerError("INVALID_ADDITIONAL_SOURCES")
        bindings = ([payload] if "source" in payload else []) + extra_sources
        for binding in bindings:
            if not isinstance(binding, dict) or not isinstance(binding.get('source'), str):
                raise RunManagerError("INVALID_ADDITIONAL_SOURCES")
            source = Path(binding["source"]).resolve(strict=True)
            if not source.is_file() or not any(source.is_relative_to(Path(root)) for root in registered["input_roots"]):
                raise RunManagerError("SOURCE_OUTSIDE_REGISTERED_ROOTS")
            actual = file_sha(source)
            if binding.get("source_sha256") and binding["source_sha256"] != actual:
                raise RunManagerError("SOURCE_FINGERPRINT_MISMATCH")
            binding.update(source=str(source), source_sha256=actual)
        program = self.registry["programs"][program_id]
        requested = globals()["requirement"](requirement or {})
        requested = {key: max(value, program["requirement"][key]) for key, value in requested.items()}
        effective_policy = None
        try:
            effective_policy = self.adapter.policy(self.adapter.sample())
        except Exception as error:
            if policy is not None:
                raise RunManagerError("POLICY_CANNOT_BE_VERIFIED", str(error)) from error
        if policy is not None and policy != effective_policy:
            raise RunManagerError("POLICY_SNAPSHOT_MISMATCH")
        wait_timeout = bounded_number(wait_timeout, "wait_timeout", 86400)
        maximum = bounded_number(program.get("run_timeout_seconds", 86400), "registered run timeout", 604800)
        runtime = maximum if run_timeout is None else bounded_number(run_timeout, "run_timeout", maximum)
        if scope_key is not None and (not isinstance(scope_key, str) or len(scope_key) > 512):
            raise RunManagerError("INVALID_SCOPE_KEY")
        data = {"program_id": program_id, "workspace": workspace, "scope_key": scope_key, "payload": payload,
                "input_roots": registered["input_roots"],
                "program_fingerprint": program["program_fingerprint"], "registry_fingerprint": self.registry["registry_fingerprint"],
                "config_fingerprint": config_fingerprint or fingerprint({"payload": payload, "requirement": requested}),
                "policy_fingerprint": fingerprint(self.registry["resource_policy"]), "policy": effective_policy,
                "requirement": requested, "resource_kind": program["resource_kind"], "gpu_key": program["gpu_key"],
                "run_timeout_seconds": runtime}
        return self.store.submit(data, wait_timeout, retry_of)

    def finish(self, job, state, error=None):
        with self.active_lock:
            owned = self.active[job["job_id"]]
        try:
            cleanup = owned.close()
        except BaseException as failure:
            self.store.transition(job["job_id"], "INTERRUPTED", error={"code": "OWNED_CLEANUP_UNCONFIRMED", "message": str(failure)})
            # Keep the ownership object and block the global queue.
            self.events.event({"event": "cleanup_unconfirmed", "job_id": job["job_id"], "error_type": type(failure).__name__})
            return
        with self.active_lock:
            self.active.pop(job["job_id"], None)
        self.store.transition(job["job_id"], state, error=error, exit_code=owned.process.returncode,
                              updates={"finished_epoch": time.time(), "cleanup": cleanup,
                                       "stdout_tail": owned.logs["stdout"].tail.decode("utf-8", errors="replace"),
                                       "stderr_tail": owned.logs["stderr"].tail.decode("utf-8", errors="replace"),
                                       "resource_summary": owned.summary})

    def step(self):
        with self.active_lock:
            running = list(self.active.items())
        for identifier, owned in running:
            job = self.store.status(identifier)
            if job["state"] == "INTERRUPTED":
                continue  # Ownership cleanup must be explicitly investigated.
            if job["state"] == "CANCEL_REQUESTED":
                self.finish(job, "CANCELLED", job.get("error"))
                continue
            code = owned.process.poll()
            if code is not None:
                self.finish(job, "SUCCEEDED" if code == 0 else "FAILED", None if code == 0 else {"code": "WORKER_EXIT_NONZERO", "exit_code": code})
                continue
            if time.time() - job["started_epoch"] > job["run_timeout_seconds"]:
                self.finish(job, "FAILED", {"code": "WORKER_RUN_TIMEOUT"})
                continue
            try:
                observed = self.adapter.sample()
                observed.update(owned.job.sample_memory())
                summary = owned.summary
                summary["samples"] += 1
                for name in ("group_current_bytes", "group_working_bytes", "gpu_used_bytes"):
                    summary["sampled_peak_" + name] = max(summary.get("sampled_peak_" + name, 0), observed.get(name, 0))
                for name in ("physical_free_bytes", "commit_headroom_bytes"):
                    if name in observed:
                        summary["sampled_min_" + name] = min(summary.get("sampled_min_" + name, observed[name]), observed[name])
                verdict = self.adapter.check(job["policy"], observed)
                owned.resources.event({"epoch": time.time(), "observation": observed, "verdict": verdict})
                self.store.transition(identifier, "RUNNING", updates={"latest_observation": observed, "resource_summary": summary}, expected=("RUNNING",))
                if verdict.get("decision") == "STOP" or verdict.get("allowed") is False:
                    self.finish(job, "FAILED", {"code": "RESOURCE_BOUND_EXCEEDED", "details": verdict})
            except Exception as error:
                self.finish(job, "FAILED", {"code": "RESOURCE_OBSERVATION_FAILED", "message": str(error)})
        # Conservative initial scheduler: one controlled heavy task globally,
        # including CPU test workers. No two workspaces can both reserve GPU0.
        with self.active_lock:
            occupied = bool(self.active)
        queued = self.store.list(("QUEUED", "WAITING_RESOURCES"), limit=1000)
        for job in queued:
            if time.time() >= job["wait_deadline_epoch"]:
                self.store.transition(job["job_id"], "WAIT_EXPIRED", error={"code": "RESOURCE_WAIT_TIMEOUT"}, expected=("QUEUED", "WAITING_RESOURCES"))
                continue
            if occupied:
                break
            try:
                observed = self.adapter.sample()
                observed.update(WindowsJob.sample_memory(None, pids=[os.getpid()]))
                resolved = self.adapter.policy(observed)
                if job.get("policy") is not None and resolved != job["policy"]:
                    raise RunManagerError("DEVICE_POLICY_CHANGED_AFTER_REGISTRATION")
                decision = self.adapter.admit(resolved, observed, job["requirement"])
            except Exception as error:
                if getattr(error, "decision", None) == "CAPACITY_INSUFFICIENT" or getattr(error, "code", None) in {
                    "GPU_IDENTITY_CHANGED", "GPU_CAPACITY_CHANGED", "DEVICE_POLICY_CHANGED_AFTER_REGISTRATION"}:
                    self.store.transition(job["job_id"], "CAPACITY_INSUFFICIENT", error={"code": getattr(error, "code", "DEVICE_CAPACITY_INSUFFICIENT"), "message": str(error)}, expected=("QUEUED", "WAITING_RESOURCES"))
                    continue
                self.store.transition(job["job_id"], "WAITING_RESOURCES", error={"code": "RESOURCE_OBSERVATION_UNAVAILABLE", "message": str(error)}, expected=("QUEUED", "WAITING_RESOURCES"))
                break
            if decision["decision"] == "CAPACITY_INSUFFICIENT":
                self.store.transition(job["job_id"], "CAPACITY_INSUFFICIENT", error={"code": "DEVICE_CAPACITY_INSUFFICIENT", "details": decision}, expected=("QUEUED", "WAITING_RESOURCES"))
                continue
            if decision["decision"] != "ADMIT":
                self.store.transition(job["job_id"], "WAITING_RESOURCES", error={"code": "RESOURCE_WAIT", "details": decision}, expected=("QUEUED", "WAITING_RESOURCES"))
                break
            starting = self.store.transition(job["job_id"], "STARTING", updates={"policy": resolved,
                         "execution_fingerprint": fingerprint({"registration": job["dedupe_fingerprint"], "policy": resolved}),
                         "admission": decision}, expected=("QUEUED", "WAITING_RESOURCES"))
            if starting["state"] != "STARTING":
                continue
            self.launch(starting)
            break

    def launch(self, job):
        owned = None
        try:
            program = self.registry["programs"][job["program_id"]]
            if file_sha(program["script"]) != program["script_sha256"] or file_sha(program["python"]) != program["actual_python_sha256"]:
                raise RunManagerError("REGISTERED_PROGRAM_CHANGED")
            if job["payload"].get("source") and file_sha(job["payload"]["source"]) != job["payload"]["source_sha256"]:
                raise RunManagerError("SOURCE_CHANGED_BEFORE_START")
            folder = Path(job["job_folder"])
            folder.mkdir(parents=True, exist_ok=False)
            request = folder / "request.json"
            atomic_json(request, {**job, "public_id": job["job_id"][:12], "manager_root": str(self.root)})
            arguments = [value.format(job_id=job["job_id"], workspace=job["workspace"], request_path=str(request)) for value in program["arguments"]]
            command = [program["python"], "-I", "-X", "utf8", program["script"], *arguments]
            environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
            environment.update(EKB_RUN_MANAGER_HOME=str(self.root), EKB_MANAGED_JOB_ID=job["job_id"],
                               BEMARKDOWN_HOST_MEMORY_POLICY_JSON=canonical(job["policy"]), PYTHONIOENCODING="utf-8")
            # Leave an explicit bounded controller allowance outside the worker
            # Job. Live observations include the worker tree and policy floors.
            cap = job["policy"].get("project_ram_cap_gib")
            memory_limit = max(1024**2, int(cap * 1024**3) - 128 * 1024**2) if cap else None
            owned = OwnedProcess(command, cwd=job["workspace"], environment=environment, folder=folder, memory_limit=memory_limit)
            owned.resources = RotatingLog(folder / "resources.jsonl")
            owned.summary = {"samples": 0, "scope": "MANAGER_PLUS_OWNED_WORKER_TREE; whole-device GPU",
                             "controller_reserve_bytes": 128 * 1024**2,
                             "peak_semantics": "ACTUAL_SAMPLED_CURRENT; Windows Job failed-allocation peaks excluded"}
            with self.active_lock:
                self.active[job["job_id"]] = owned
            current = self.store.transition(job["job_id"], "RUNNING", identity=owned.identity,
                        updates={"started_epoch": time.time(), "command": command, "command_fingerprint": fingerprint(command),
                                 "job_memory_limit_bytes": memory_limit}, expected=("STARTING",))
            if current["state"] != "RUNNING":
                self.finish(current, "CANCELLED", current.get("error"))
                return
            owned.resume()
        except Exception as error:
            if owned is not None:
                try:
                    owned.close()
                except BaseException as cleanup_error:
                    self.store.transition(job["job_id"], "INTERRUPTED", error={"code": "OWNED_CLEANUP_UNCONFIRMED", "message": str(cleanup_error)})
                    return
                with self.active_lock:
                    self.active.pop(job["job_id"], None)
            self.store.transition(job["job_id"], "FAILED", error={"code": getattr(error, "code", "WORKER_START_FAILED"), "message": str(error)})

    def loop(self):
        try:
            while not self.stop.wait(.25):
                self.step()
        except BaseException as error:
            self.events.event({"event": "scheduler_failed", "error_type": type(error).__name__, "message": str(error)})
            self.server.shutdown()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, manager):
        self.manager = manager
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(("127.0.0.1", 0), Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(10)
        manager = self.server.manager
        challenge = secrets.token_hex(32)
        try:
            send(self.wfile, {"challenge": challenge, "schema": "ekb-run-manager-auth-v1"})
            envelope = receive(self.rfile)
            request = envelope["request"]
            if not hmac.compare_digest(envelope.get("mac", ""), mac(manager.key, challenge, request)):
                return
            try:
                value = manager.dispatch(request["operation"], request.get("arguments", {}))
                response = {"ok": True, "value": value}
            except Exception as error:
                response = {"ok": False, "error": {"code": getattr(error, "code", "MANAGER_REQUEST_FAILED"),
                                                   "message": str(error), "details": getattr(error, "details", {})}}
            send(self.wfile, {"response": response, "mac": mac(manager.key, challenge, response)})
        except (OSError, ValueError, KeyError, RunManagerError):
            return  # Unauthenticated/malformed input is never logged with secrets.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    secure_root(root)
    with NamedMutex(mutex_name(root, "instance"), timeout=0):
        manager = Manager(root, load_registry(args.registry))
        server = Server(manager)
        manager.server = server
        manager.endpoint["port"] = server.server_address[1]
        atomic_json(root / "endpoint.json", manager.endpoint)
        scheduler = threading.Thread(target=manager.loop, name="ekb-run-scheduler", daemon=True)
        scheduler.start()
        try:
            server.serve_forever(poll_interval=.2)
        finally:
            manager.stop.set()
            scheduler.join(timeout=15)
            for identifier in list(manager.active):
                manager.finish(manager.store.status(identifier), "INTERRUPTED", {"code": "MANAGER_SHUTDOWN"})
            server.server_close()


if __name__ == "__main__":
    main()
