"""User-session-independent daemon bootstrap and authenticated client API."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import RunManagerError, code_fingerprint, home
from .ipc import auth_key, exchange, secure_root
from .process import NamedMutex, mutex_name, process_identity
from .registry import load_registry


class RunManagerClient:
    def __init__(self, root=None):
        self.root = home(root)

    def _call(self, operation, **arguments):
        try:
            endpoint = json.loads((self.root / "endpoint.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RunManagerError("DAEMON_NOT_AVAILABLE") from error
        identity = process_identity(endpoint["identity"]["pid"])
        if identity != endpoint["identity"]:
            raise RunManagerError("STALE_DAEMON_IDENTITY")
        return exchange(self.root, endpoint, operation, arguments)

    def ping(self):
        return self._call("ping")

    def register_workspace(self, workspace, program_ids, input_roots=None):
        return self._call("register_workspace", workspace=str(Path(workspace).resolve()),
                          program_ids=list(program_ids), input_roots=[str(Path(path).resolve()) for path in (input_roots or [])])

    def submit(self, program_id, workspace, payload, *, scope_key=None, requirement=None,
               policy=None, wait_timeout=900, retry_of=None, config_fingerprint=None, run_timeout=None):
        return self._call("submit", program_id=program_id, workspace=str(Path(workspace).resolve()), payload=payload,
                          scope_key=scope_key, requirement=requirement or {}, policy=policy, wait_timeout=wait_timeout,
                          retry_of=retry_of, config_fingerprint=config_fingerprint, run_timeout=run_timeout)

    def status(self, job_id):
        return self._call("status", job_id=job_id)

    get = status

    def lookup(self, program_id, workspace, public_id):
        return self._call("lookup", program_id=program_id, workspace=str(Path(workspace).resolve()), public_id=public_id)

    def find_scope(self, program_id, workspace, scope_key):
        return self._call("find_scope", program_id=program_id, workspace=str(Path(workspace).resolve()), scope_key=scope_key)

    def list(self, states=None, limit=20):
        return self._call("list", states=states, limit=limit)

    def adopt(self, job_id, client_id):
        return self._call("adopt", job_id=job_id, client_id=client_id)

    def cancel(self, job_id, reason="user", wait_seconds=0):
        value = self._call("cancel", job_id=job_id, reason=reason)
        deadline = time.monotonic() + wait_seconds
        while value["state"] == "CANCEL_REQUESTED" and time.monotonic() < deadline:
            time.sleep(.1)
            value = self.status(job_id)
        return value

    def owns_process(self, job_id, pid, created):
        return self._call("owns_process", job_id=job_id, pid=pid, created=str(created))

    def shutdown(self):
        """Administrative drain-only shutdown; refuses while tasks are active."""
        return self._call("shutdown")


def ensure_daemon(*, registry_path, root=None, python=None, timeout=15):
    """Start once under a per-user mutex; never fall back inside client Jobs."""
    root = home(root)
    secure_root(root)
    registry = load_registry(registry_path)
    controlled_python = Path(python or getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    if controlled_python.name.lower() not in {"python.exe", "python3.exe", "pythonw.exe"}:
        raise RunManagerError("UNCONTROLLED_MANAGER_INTERPRETER")
    with NamedMutex(mutex_name(root, "bootstrap"), timeout=timeout):
        auth_key(root, create=True)
        client = RunManagerClient(root)
        try:
            current = client.ping()
        except (OSError, KeyError, ValueError, RunManagerError):
            current = None
        if current:
            if current["code_fingerprint"] != code_fingerprint():
                raise RunManagerError("DAEMON_UPGRADE_REQUIRES_DRAIN")
            if current["registry_fingerprint"] != registry["registry_fingerprint"]:
                raise RunManagerError("REGISTRY_CHANGE_REQUIRES_DRAIN")
            return client
        # CREATE_BREAKAWAY_FROM_JOB is mandatory. If the launching agent Job
        # forbids breakaway, report it instead of claiming independence.
        command = [str(controlled_python), "-I", "-X", "utf8", str(Path(__file__).with_name("daemon.py")),
                   "--root", str(root), "--registry", str(Path(registry_path).resolve())]
        environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        boot = root / "bootstrap.stderr.log"
        if boot.exists() and boot.stat().st_size > 512 * 1024:
            boot.replace(root / "bootstrap.stderr.log.1")
        with boot.open("ab") as stderr:
            try:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=stderr, env=environment, cwd=root,
                                           creationflags=0x01000000 | 0x8)  # BREAKAWAY | DETACHED_PROCESS
            except OSError as error:
                raise RunManagerError("DAEMON_BREAKAWAY_DENIED", str(error)) from error
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RunManagerError("DAEMON_START_FAILED", f"exit={process.returncode}; diagnostics={boot}")
            try:
                current = client.ping()
            except (OSError, KeyError, ValueError, RunManagerError):
                time.sleep(.05)
                continue
            if current["registry_fingerprint"] == registry["registry_fingerprint"] and current["code_fingerprint"] == code_fingerprint():
                return client
            raise RunManagerError("DAEMON_IDENTITY_CHANGED_DURING_START")
        # Do not kill a process merely because startup is slow. A later caller
        # can discover its authenticated endpoint; the singleton mutex prevents
        # a second active manager.
        raise RunManagerError("DAEMON_START_TIMEOUT", f"diagnostics={boot}")
