"""Owned native stages: suspended Windows launch, a nested Job, and strict budgets.

Only stdlib is imported. A stage never relies on a Developer checkout. The
sampler injection is for small fault tests; production uses Windows and NVML.
Job peak counters are diagnostic only, never proof of budget compliance.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid
import weakref

from .host_memory_policy import current_memory_policy

GiB = 1024 ** 3
MiB = 1024 ** 2
CONTEXT_ENV = "BEMARKDOWN_STAGE_ADMISSION_CONTEXT"
_registry_lock = threading.RLock()
_registry = weakref.WeakSet()
_poisoned = {}  # Strong ownership retained until explicit mechanical revalidation.


def revalidate_owned_stages():
    """Bounded diagnostic recheck; never launches or terminates any process.

    Failed cleanup remains a startup blocker until retained Job/root handles
    establish exit. The original terminal receipt is preserved separately.
    """
    with _registry_lock:
        pending = list(_poisoned.items())
    cleared, unconfirmed = [], []
    for key, stage in pending:
        with stage._close_lock:
            try:
                confirmed = not (stage._thread and stage._thread.is_alive())
                confirmed = confirmed and (stage._process is None or stage.native_poll() is not None)
                confirmed = confirmed and (stage._job is None or stage._job.revalidate_exit())
                if confirmed:
                    if stage._job: stage._job.close()
                    if stage._process and stage._process.returncode is not None:
                        stage._process._handle.Close()
                    stage._exit_confirmed = True
                    with _registry_lock:
                        _poisoned.pop(key, None)
                        _registry.discard(stage)
                    cleared.append(stage._identity)
                else:
                    unconfirmed.append(stage._identity)
                diagnostic = {"epoch": time.time(), "owned_exit_confirmed": confirmed,
                              "identity": stage._identity, "action": "observe_retained_handles_only"}
            except BaseException as exc:
                unconfirmed.append(stage._identity)
                diagnostic = {"epoch": time.time(), "owned_exit_confirmed": False,
                              "identity": stage._identity, "error": str(exc)}
            Path(stage.journal_path).with_suffix(".revalidation.json").write_text(
                json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    return {"cleared": cleared, "unconfirmed": unconfirmed}


def _assert_previous_owned_exits():
    with _registry_lock:
        if _poisoned:
            raise ResourceAdmissionError("OWNED_PREVIOUS_EXIT_UNCONFIRMED")


class ResourceAdmissionError(RuntimeError):
    """Admission or continuous observation failed; the owned Job is stopped."""


class _Journal:
    """Two bounded streaming files plus one terminal receipt, without row history."""
    def __init__(self, path, max_bytes=8 * MiB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, row):
        with self.lock:
            if self.stream.closed:
                return
            if self.stream.tell() >= self.max_bytes:
                self.stream.close()
                previous = self.path.with_suffix(self.path.suffix + ".1")
                self.path.replace(previous)
                self.stream = self.path.open("w", encoding="utf-8", buffering=1)
            self.stream.write(json.dumps({"epoch": time.time(), **row}, ensure_ascii=False) + "\n")

    def finish(self, row):
        with self.lock:
            self.write(row)
            self.path.with_suffix(".result.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.stream.close()


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_uint64) for name in ("total_physical", "free_physical",
         "total_pagefile", "free_pagefile", "total_virtual", "free_virtual", "free_extended")]


class _Performance(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
        ("commit_total", "commit_limit", "commit_peak", "physical_total", "physical_available",
         "system_cache", "kernel_total", "kernel_paged", "kernel_nonpaged", "page_size")] + [
        (name, wintypes.DWORD) for name in ("handles", "processes", "threads")]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ("peak_working", "working", "peak_paged", "paged",
         "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]


class _BasicLimit(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
               ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t),
               ("max_working", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
               ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
               ("scheduling", wintypes.DWORD)]


class _Iocounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops",
                                                       "read_bytes", "write_bytes", "other_bytes")]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("basic", _BasicLimit), ("io", _Iocounters),
               ("process_limit", ctypes.c_size_t), ("job_limit", ctypes.c_size_t),
               ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]


class _JobMemory(ctypes.Structure):
    _fields_ = [("current", ctypes.c_uint64), ("peak", ctypes.c_uint64)]


class _JobAccounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("user_time", "kernel_time", "period_user", "period_kernel")] + [
        (name, wintypes.DWORD) for name in ("page_faults", "total_processes", "active_processes", "terminated_processes")]


class _GpuMemory(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("total", "free", "used")]


class _Windows:
    def __init__(self):
        if os.name != "nt":
            raise ResourceAdmissionError("WINDOWS_RESOURCE_OBSERVATION_REQUIRED")
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.ps = ctypes.WinDLL("psapi", use_last_error=True)
        self.nt = ctypes.WinDLL("ntdll")
        definitions = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
            "GetCurrentProcess": ([], wintypes.HANDLE),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "GetProcessTimes": ([wintypes.HANDLE] + [ctypes.c_void_p] * 4, wintypes.BOOL),
            "GlobalMemoryStatusEx": ([ctypes.c_void_p], wintypes.BOOL),
            "GetExitCodeProcess": ([wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        }
        for name, (args, result) in definitions.items():
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = args, result
        self.ps.GetPerformanceInfo.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        self.ps.GetPerformanceInfo.restype = wintypes.BOOL
        self.ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        self.ps.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.nt.NtResumeProcess.argtypes = [wintypes.HANDLE]
        self.nt.NtResumeProcess.restype = ctypes.c_long

    @staticmethod
    def checked(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def private(self, handle):
        m = _ProcessMemory()
        m.cb = ctypes.sizeof(m)
        self.checked(self.ps.GetProcessMemoryInfo(handle, ctypes.byref(m), m.cb))
        return m.private

    def created(self, handle):
        values = [ctypes.c_uint64() for _ in range(4)]
        self.checked(self.k.GetProcessTimes(handle, *[ctypes.byref(v) for v in values]))
        return values[0].value


class _SystemSampler:
    def __init__(self, win, context):
        from .host_memory_policy import DeviceSampler, current_memory_policy
        self.win, self.controller, self.device = win, None, None
        try:
            if context and int(context['controller_pid']) != os.getpid():
                self.controller = win.checked(win.k.OpenProcess(0x1000 | 0x10, False, int(context['controller_pid'])))
                if win.created(self.controller) != int(context['controller_created_ticks']):
                    raise ResourceAdmissionError('OUTER_CONTROLLER_IDENTITY_CHANGED')
            policy=current_memory_policy()
            self.device=DeviceSampler(policy['gpu_index'],policy['gpu_uuid'])
        except BaseException:
            self.close();raise
    def __call__(self):
        row=self.device.sample()
        row.update(host_private_bytes=self.win.private(self.win.k.GetCurrentProcess()),
            controller_private_bytes=self.win.private(self.controller) if self.controller else 0)
        return row
    def close(self):
        if self.controller:self.win.k.CloseHandle(self.controller);self.controller=None
        if self.device:self.device.close();self.device=None

class _Job:
    def __init__(self, win, cap):
        self.win, self.handle = win, None
        self.stopped_identities = []
        self._pending_handles = []
        self.lock = threading.RLock()
        self.handle = win.checked(win.k.CreateJobObjectW(None, None))
        try:
            limits = _ExtendedLimit()
            limits.basic.flags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
            limits.job_limit = cap
            win.checked(win.k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def memory(self):
        with self.lock:
            if not self.handle:
                return 0
            m = _JobMemory()
            self.win.checked(self.win.k.QueryInformationJobObject(self.handle, 28, ctypes.byref(m), ctypes.sizeof(m), None))
            return m.current

    def terminate(self):
        with self.lock:
            if self.handle:
                pinned = []
                confirmed = False
                try:
                    for identity in self.members():
                        handle = self.win.k.OpenProcess(0x100000 | 0x1000, False, identity["pid"])
                        if not handle:
                            if ctypes.get_last_error() == 87: continue
                            self.win.checked(handle)
                        if self.win.created(handle) != identity["created_ticks"]:
                            self.win.k.CloseHandle(handle)
                            raise ResourceAdmissionError("OWNED_PID_IDENTITY_CHANGED")
                        pinned.append((handle, identity))
                    self.win.checked(self.win.k.TerminateJobObject(self.handle, 1))
                    until = time.monotonic() + 5
                    self.wait_empty(timeout=5)
                    for handle, identity in pinned:
                        remaining = max(0, int((until - time.monotonic()) * 1000))
                        if self.win.k.WaitForSingleObject(handle, remaining) != 0:
                            raise ResourceAdmissionError("OWNED_PROCESS_HANDLE_EXIT_NOT_CONFIRMED")
                    if pinned:
                        self.stopped_identities = [identity for _, identity in pinned]
                    confirmed = True
                finally:
                    if confirmed:
                        for handle, _ in pinned: self.win.k.CloseHandle(handle)
                    else:
                        self._pending_handles.extend(pinned)

    def active(self):
        with self.lock:
            if not self.handle:
                return 0
            value = _JobAccounting()
            self.win.checked(self.win.k.QueryInformationJobObject(self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None))
            return value.active_processes

    def members(self):
        """Current Job membership, with PID and creation time, bounded to 32768."""
        with self.lock:
            if not self.handle:
                return []
            count = 64
            while count <= 32768:
                class Pids(ctypes.Structure):
                    _fields_ = [("assigned", wintypes.DWORD), ("listed", wintypes.DWORD),
                               ("pids", ctypes.c_size_t * count)]
                value = Pids()
                ok = self.win.k.QueryInformationJobObject(self.handle, 3, ctypes.byref(value), ctypes.sizeof(value), None)
                if ok and value.listed <= count and value.assigned <= count:
                    break
                if not ok and ctypes.get_last_error() != 234:
                    self.win.checked(ok)
                count *= 2
            else:
                raise ResourceAdmissionError("OWNED_MEMBERSHIP_EXCEEDS_BOUNDED_OBSERVATION")
            identities = []
            for pid in value.pids[:value.listed]:
                handle = self.win.k.OpenProcess(0x1000, False, pid)
                if not handle:
                    if ctypes.get_last_error() == 87:  # exited between snapshot and open
                        continue
                    self.win.checked(handle)
                try:
                    exit_code = wintypes.DWORD()
                    self.win.checked(self.win.k.GetExitCodeProcess(handle, ctypes.byref(exit_code)))
                    if exit_code.value == 259:
                        identities.append({"pid": pid, "created_ticks": self.win.created(handle)})
                finally:
                    self.win.k.CloseHandle(handle)
            return identities

    def wait_empty(self, timeout=5):
        until = time.monotonic() + timeout
        while self.active():
            if time.monotonic() >= until:
                raise ResourceAdmissionError("OWNED_JOB_EXIT_NOT_CONFIRMED")
            time.sleep(0.01)

    def revalidate_exit(self):
        with self.lock:
            if self.active(): return False
            if any(self.win.k.WaitForSingleObject(handle, 0) != 0 for handle, _ in self._pending_handles):
                return False
            for handle, _ in self._pending_handles: self.win.k.CloseHandle(handle)
            self._pending_handles.clear()
            return True

    def close(self):
        with self.lock:
            if self.handle:
                if self._pending_handles and not self.revalidate_exit():
                    raise ResourceAdmissionError("OWNED_RETAINED_PROCESS_EXIT_UNCONFIRMED")
                self.win.checked(self.win.k.CloseHandle(self.handle))
                self.handle = None


class OwnedStageProcess:
    """Popen-like owned stage. Call close(), or use a with block, even after exit.

    An outer stage boundary supplies model serialization; this wrapper does not
    acquire that lock, avoiding cross-thread actor lock ordering problems.
    ``sampler`` is a test hook returning the byte fields documented above.
    """
    def __init__(self, args, *, env=None, stdout=None, stderr=None, stdin=None,
                 cwd=None, timeout=None, creationflags=0, stage="native",
                 journal_path=None, sampler=None, sample_interval=0.1, **kwargs):
        self.args, self.stage = args, stage
        self._process = self._job = self._sampler = self._thread = None
        self._error = None
        self._closed = False
        self._exit_confirmed = False
        self._stop = threading.Event()
        self._close_lock = threading.RLock()
        self._start = time.monotonic()
        self._timeout = timeout
        self._interval = max(0.02, float(sample_interval))
        self._last = None
        self._forecast = None
        if journal_path is None:
            journal_path = Path(tempfile.gettempdir()) / "bemarkdown-owned-stages" / (uuid.uuid4().hex + ".jsonl")
        self.journal_path = str(journal_path)
        self._journal = _Journal(journal_path)
        self._identity = None
        try:
            _assert_previous_owned_exits()
            effective_env = os.environ if env is None else env
            self._memory_policy = current_memory_policy(effective_env)
            context_raw = effective_env.get(CONTEXT_ENV)
            context = json.loads(context_raw) if context_raw else None
            if context:
                self._outer_admission(context)
            self._win = _Windows()
            self._sampler = sampler if sampler is not None else _SystemSampler(self._win, context)
            # Registry lock covers admission and registration of other owned Jobs.
            with _registry_lock:
                from .stage_forecasts import preflight_stage_forecast, ForecastAdmissionError
                deadline=time.monotonic()+self._memory_policy['stage_wait_timeout_seconds']
                waiting=None
                while True:
                    try:
                        row=self._observe(startup=True)
                        if sampler is None:
                            self._forecast=preflight_stage_forecast(args,stage,row,effective_env)
                            self._journal.write({'event':'forecast_admission',**self._forecast})
                        else:self._forecast={'status':'EXPLICIT_SAMPLER_TEST_HOOK_NO_HEAVY_CERTIFICATION'}
                        break
                    except (ResourceAdmissionError,ForecastAdmissionError) as exc:
                        if sampler is not None or getattr(exc,'decision',None)!='WAIT':raise
                        now=time.monotonic()
                        if now>=deadline:
                            error=ResourceAdmissionError('STAGE_RESOURCE_WAIT_TIMEOUT:'+str(exc));error.decision='WAIT_TIMEOUT'
                            raise error from exc
                        if waiting is None:waiting=now
                        self._journal.write({'event':'stage_wait','reason':str(exc),'deadline_monotonic':deadline,
                            'policy_fingerprint':self._memory_policy['policy_fingerprint']})
                        if self._stop.wait(min(self._memory_policy['stage_wait_poll_seconds'],deadline-now)):
                            raise ResourceAdmissionError('STAGE_WAIT_CANCELLED')
                if waiting is not None:self._journal.write({'event':'stage_wait_completed','seconds':time.monotonic()-waiting})
                reserve = row["host_private_bytes"] + row.get("controller_private_bytes", 0) + row["other_owned_bytes"]
                cap = int(self._memory_policy["project_ram_cap_gib"] * GiB - reserve - 0.03 * GiB) // 4096 * 4096
                if cap <= 0:
                    raise ResourceAdmissionError("HOST_RESERVE_EXHAUSTS_PROJECT_RAM_CAP")
                self._job = _Job(self._win, cap)
                # Do not permit job breakaway: every descendant belongs to this Job.
                if creationflags & 0x01000000:
                    raise ValueError("CREATE_BREAKAWAY_FROM_JOB is incompatible with owned stages")
                self._process = subprocess.Popen(args, env=env, stdout=stdout, stderr=stderr,
                    stdin=stdin, cwd=cwd, creationflags=creationflags | 0x4, **kwargs)
                handle = int(self._process._handle)
                self._identity = {"pid": self.pid, "created_ticks": self._win.created(handle)}
                self._win.checked(self._win.k.AssignProcessToJobObject(self._job.handle, handle))
                _registry.add(self)
                self._observe(startup=True)
                status = self._win.nt.NtResumeProcess(handle)
                if status < 0:
                    raise OSError(f"NtResumeProcess failed: {status:#x}")
                self._journal.write({"event": "resumed", "identity": self._identity, "job_cap_bytes": cap})
            self._start=time.monotonic()  # inference clock excludes resource waits
            self._thread = threading.Thread(target=self._monitor, name=f"owned-stage-{self.pid}", daemon=True)
            self._thread.start()
        except BaseException as exc:
            self._forecast = getattr(exc, "forecast", self._forecast)
            self._error = exc
            self.close()
            raise

    def _outer_admission(self, context):
        import hashlib
        runner = Path(context["runner_path"])
        if hashlib.sha256(runner.read_bytes()).hexdigest() != context["runner_sha256"]:
            raise ResourceAdmissionError("STAGE_ADMISSION_RUNNER_CHANGED")
        result = subprocess.run([context["python"], str(runner), "--caller-pid", str(os.getpid()),
            "--stage", self.stage], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, timeout=self._memory_policy["stage_wait_timeout_seconds"]+30)
        if result.returncode:
            raise ResourceAdmissionError(f"STAGE_ADMISSION_REFUSED:{self.stage}:exit={result.returncode}:journal={context['journal_path']}")

    def _observe(self, *, startup=False):
        if startup: _assert_previous_owned_exits()
        row = dict(self._sampler())
        row.setdefault("controller_private_bytes", 0)
        required = ("host_private_bytes", "physical_free_bytes", "commit_headroom_bytes", "gpu_free_bytes", "gpu_used_bytes")
        for key in (*required, "controller_private_bytes"):
            if key not in row or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ResourceAdmissionError(f"RESOURCE_OBSERVATION_INVALID:{key}")
        row["owned_job_current_bytes"] = self._job.memory() if self._job else 0
        row["owned_current_identities"] = self._job.members() if self._job else []
        with _registry_lock:
            row["other_owned_bytes"] = sum(p._job.memory() for p in list(_registry)
                if p is not self and p._job is not None and not p._closed)
        row["group_current_bytes"] = (row["host_private_bytes"] + row.get("controller_private_bytes", 0)
            + row["owned_job_current_bytes"] + row["other_owned_bytes"])
        self._last = row
        self._journal.write({"event": "observation", "startup": startup, **row})
        from .host_memory_policy import check_current, ResourcePolicyError
        try:
            check_current(self._memory_policy,row)
            if startup and row['gpu_free_bytes']<self._memory_policy['gpu_startup_free_floor_bytes']:
                raise ResourcePolicyError('LEGACY_GPU_STARTUP_FREE_FLOOR')
        except ResourcePolicyError as exc:
            error=ResourceAdmissionError(str(exc));error.decision=exc.decision
            raise error from exc
        return row

    def _monitor(self):
        try:
            while not self._stop.wait(self._interval):
                self._observe()
                if self._timeout is not None and time.monotonic() - self._start > self._timeout:
                    raise subprocess.TimeoutExpired(self.args, self._timeout)
                if self._process.poll() is not None:
                    # A successful parent exit must not leave native grandchildren.
                    self._job.terminate()
                    return
        except BaseException as exc:
            self._error = exc
            self._journal.write({"event": "supervision_failed", "error_type": type(exc).__name__, "error": str(exc), "observation": self._last})
            try:
                self._job.terminate()
            except BaseException as cleanup:
                self._journal.write({"event": "termination_failed", "error": str(cleanup)})

    @property
    def pid(self): return self._process.pid
    @property
    def returncode(self): return self._process.returncode if self._process else None
    @property
    def stdin(self): return self._process.stdin
    @property
    def stdout(self): return self._process.stdout
    @property
    def stderr(self): return self._process.stderr

    def poll(self):
        result = self._process.poll()
        if self._error: raise self._error
        return result

    def native_poll(self):
        """Mechanical root exit observation; supervision errors remain in receipt."""
        return self._process.poll() if self._process else None

    @property
    def exit_confirmed(self):
        """True only after close confirmed root and owned Job member exits."""
        return self._exit_confirmed

    def wait(self, timeout=None):
        result = self._process.wait(timeout=timeout)
        if self._error: raise self._error
        return result

    def communicate(self, input=None, timeout=None):
        result = self._process.communicate(input=input, timeout=timeout)
        if self._error: raise self._error
        return result

    def terminate(self):
        if self._job: self._job.terminate()

    kill = terminate

    def close(self):
        with self._close_lock:
            if self._closed:
                if not self._exit_confirmed:
                    raise ResourceAdmissionError("OWNED_PREVIOUS_EXIT_UNCONFIRMED") from self._error
                return
            self._stop.set()
            if self._thread and self._thread is not threading.current_thread():
                self._thread.join(timeout=5)
            cleanup_errors = []
            if self._thread and self._thread.is_alive():
                cleanup_errors.append("SUPERVISOR_THREAD_EXIT_UNCONFIRMED")
            if self._job:
                try: self._job.terminate()
                except BaseException as exc: cleanup_errors.append(str(exc))
                try: self._job.wait_empty()
                except BaseException as exc: cleanup_errors.append(str(exc))
                if not cleanup_errors:
                    try: self._job.close()
                    except BaseException as exc: cleanup_errors.append(str(exc))
            if self._process:
                try:
                    # Assignment failure still owns the original suspended process
                    # via its pinned process handle, never a recycled PID lookup.
                    if self._process.poll() is None: self._process.kill()
                    self._process.wait(timeout=5)
                except BaseException as exc: cleanup_errors.append(str(exc))
                if self._process.returncode is not None:
                    try: self._process._handle.Close()
                    except BaseException as exc: cleanup_errors.append(str(exc))
                for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                    if stream:
                        try: stream.close()
                        except OSError as exc: cleanup_errors.append(str(exc))
            if hasattr(self._sampler, "close"):
                try: self._sampler.close()
                except BaseException as exc: cleanup_errors.append(str(exc))
            self._closed = True
            self._exit_confirmed = not cleanup_errors and (
                self._process is None or self._process.returncode is not None)
            with _registry_lock:
                if self._exit_confirmed: _registry.discard(self)
                else: _poisoned[id(self)] = self
            self._journal.finish({"event": "closed", "stage": self.stage,
                "identity": self._identity, "native_exit_code": self.returncode,
                "error_type": type(self._error).__name__ if self._error else None,
                "error": str(self._error) if self._error else None,
                "forecast": self._forecast,
                "projected_refusal": getattr(self._error, "projected", None),
                "last_observation": self._last, "cleanup_errors": cleanup_errors,
                "stopped_owned_identities": self._job.stopped_identities if self._job else [],
                "owned_exit_confirmed": self._exit_confirmed})
            if cleanup_errors:
                raise ResourceAdmissionError("OWNED_CLEANUP_FAILED:" + ";".join(cleanup_errors)) from self._error

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if exc is not None and self._error is None: self._error = exc
        self.close()


def owned_run(args, *, input=None, capture_output=False, timeout=None, check=False, **kwargs):
    """subprocess.run equivalent with owned Job cleanup on all exit paths."""
    if input is not None:
        if kwargs.get("stdin") is not None: raise ValueError("stdin and input arguments may not both be used")
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr arguments may not be used with capture_output")
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    with OwnedStageProcess(args, timeout=timeout, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            # Job termination closes descendants' pipe writers as well.
            process._process.communicate(timeout=5)
            raise
        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        if check and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, args, output=stdout, stderr=stderr)
        return result
