"""PID identity and owned Windows Jobs. Never discover targets by image name."""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import threading
import time

from .common import RunManagerError, RotatingLog, fingerprint


def _kernel():
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "GetProcessTimes": ([wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4, wintypes.BOOL),
        "QueryFullProcessImageNameW": ([wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
        "IsProcessInJob": ([wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
        "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "CreateMutexW": ([ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
        "ReleaseMutex": ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result
    return library


def _identity(handle, pid):
    kernel = _kernel()
    created, exited, system, user = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
        raise ctypes.WinError(ctypes.get_last_error())
    size = wintypes.DWORD(32768)
    text = ctypes.create_unicode_buffer(size.value)
    if not kernel.QueryFullProcessImageNameW(handle, 0, text, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"pid": pid, "created": str((created.dwHighDateTime << 32) | created.dwLowDateTime),
            "executable": str(Path(text.value).resolve())}


def process_identity(pid):
    if os.name != "nt":
        raise RunManagerError("WINDOWS_REQUIRED")
    kernel = _kernel()
    handle = kernel.OpenProcess(0x1000 | 0x100000, False, int(pid))
    if not handle:
        code = ctypes.get_last_error()
        return None if code in (87, 1168) else {"pid": int(pid), "identity_status": "UNVERIFIABLE", "os_error": code}
    try:
        if kernel.WaitForSingleObject(handle, 0) == 0:
            return None
        try:
            return _identity(handle, int(pid))
        except OSError as error:
            # The executable name can disappear during TerminateJobObject even
            # before the first zero-time wait observes the final signal.
            if kernel.WaitForSingleObject(handle, 20) == 0:
                return None
            return {"pid": int(pid), "identity_status": "UNVERIFIABLE", "os_error": error.winerror}
    finally:
        kernel.CloseHandle(handle)


def in_any_job():
    member = wintypes.BOOL()
    kernel = _kernel()
    if not kernel.IsProcessInJob(ctypes.c_void_p(-1), None, ctypes.byref(member)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(member.value)


class NamedMutex:
    def __init__(self, name, timeout=10):
        self.kernel = _kernel()
        self.handle = self.kernel.CreateMutexW(None, False, "Local\\EKBRunManager-" + name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        result = self.kernel.WaitForSingleObject(self.handle, int(timeout * 1000))
        if result not in (0, 0x80):
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise RunManagerError("MANAGER_LOCK_TIMEOUT")

    def close(self):
        if self.handle:
            self.kernel.ReleaseMutex(self.handle)
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def mutex_name(root, role):
    return fingerprint({"root": str(Path(root).resolve()).casefold(), "role": role})[:32]


class WindowsJob:
    def __init__(self, memory_limit=None, *, breakaway=False):
        self.kernel = _kernel()
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        class Limits(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_longlong), ("JobTime", ctypes.c_longlong),
                        ("Flags", wintypes.DWORD), ("MinWS", ctypes.c_size_t), ("MaxWS", ctypes.c_size_t),
                        ("ActiveLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                        ("Priority", wintypes.DWORD), ("Scheduling", wintypes.DWORD)]
        class Io(ctypes.Structure):
            _fields_ = [("Values", ctypes.c_ulonglong * 6)]
        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Limits), ("IO", Io), ("ProcessMemory", ctypes.c_size_t),
                        ("JobMemory", ctypes.c_size_t), ("PeakProcess", ctypes.c_size_t), ("PeakJob", ctypes.c_size_t)]
        value = Extended()
        value.Basic.Flags = 0x2000 | (0x200 if memory_limit else 0) | (0x800 if breakaway else 0)
        value.JobMemory = int(memory_limit or 0)
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(value), ctypes.sizeof(value)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise error

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def pids(self):
        capacity = 64
        while capacity <= 65536:
            class Pids(ctypes.Structure):
                _fields_ = [("Assigned", wintypes.DWORD), ("Count", wintypes.DWORD), ("Values", ctypes.c_size_t * capacity)]
            value = Pids()
            if self.kernel.QueryInformationJobObject(self.handle, 3, ctypes.byref(value), ctypes.sizeof(value), None):
                return list(value.Values[:value.Count])
            if ctypes.get_last_error() != 234:
                raise ctypes.WinError(ctypes.get_last_error())
            capacity *= 2
        raise RunManagerError("JOB_POPULATION_TOO_LARGE")

    def owns(self, pid, created):
        handle = self.kernel.OpenProcess(0x1000 | 0x100000, False, int(pid))
        if not handle:
            return False
        try:
            member = wintypes.BOOL()
            return (self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(member)) and bool(member.value)
                    and self.kernel.WaitForSingleObject(handle, 0) != 0
                    and _identity(handle, int(pid))["created"] == str(created))
        finally:
            self.kernel.CloseHandle(handle)

    def sample_memory(self, *, pids=None):
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWS", "WorkingSet", "PeakPaged", "Paged", "PeakNonPaged", "NonPaged", "Pagefile", "PeakPagefile", "Private")]
        library = ctypes.WinDLL("psapi", use_last_error=True)
        library.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        library.GetProcessMemoryInfo.restype = wintypes.BOOL
        total_private = total_working = 0
        kernel = self.kernel if self is not None else _kernel()
        measured = pids if pids is not None else [*self.pids(), os.getpid()]
        for pid in set(measured):
            handle = kernel.OpenProcess(0x1000 | 0x10, False, pid)
            if not handle:
                continue
            try:
                value = Counters()
                value.cb = ctypes.sizeof(value)
                if library.GetProcessMemoryInfo(handle, ctypes.byref(value), value.cb):
                    total_private += value.Private
                    total_working += value.WorkingSet
                elif kernel.WaitForSingleObject(handle, 0) != 0:
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel.CloseHandle(handle)
        return {"group_current_bytes": total_private, "group_working_bytes": total_working}

    def close(self, timeout=10):
        if not self.handle:
            return {"exit_confirmed": True}
        pinned = []
        try:
            for pid in self.pids():
                handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
                if handle:
                    member = wintypes.BOOL()
                    if self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(member)) and member.value:
                        pinned.append(handle)
                    else:
                        self.kernel.CloseHandle(handle)
            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + timeout
            while self.pids():
                if time.monotonic() >= deadline:
                    raise RunManagerError("OWNED_EXIT_UNCONFIRMED")
                time.sleep(.02)
            for handle in pinned:
                if self.kernel.WaitForSingleObject(handle, max(1, int((deadline - time.monotonic()) * 1000))) != 0:
                    raise RunManagerError("OWNED_EXIT_UNCONFIRMED")
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            return {"exit_confirmed": True}
        finally:
            for handle in pinned:
                self.kernel.CloseHandle(handle)


class OwnedProcess:
    """Created suspended: registration and Job assignment precede execution."""
    def __init__(self, command, *, cwd, environment, folder, memory_limit=None):
        self.job = WindowsJob(memory_limit)
        self.process = None
        self.threads = []
        self.logs = {name: RotatingLog(Path(folder) / (name + ".log")) for name in ("stdout", "stderr")}
        try:
            self.process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            creationflags=subprocess.CREATE_NO_WINDOW | 0x4)
            self.job.assign(self.process)
            self.identity = _identity(int(self.process._handle), self.process.pid)
        except BaseException:
            try:
                self.job.close()
            finally:
                if self.process is not None:
                    if self.process.poll() is None:
                        self.process.kill()  # Original pinned suspended Popen handle only.
                    self.process.wait(timeout=10)
            raise

    def resume(self):
        def drain(name):
            stream = getattr(self.process, name)
            try:
                while True:
                    data = os.read(stream.fileno(), 65536)
                    if not data:
                        break
                    self.logs[name].write(data)
            finally:
                stream.close()
        for name in self.logs:
            thread = threading.Thread(target=drain, args=(name,), daemon=True)
            thread.start()
            self.threads.append(thread)
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        result = ntdll.NtResumeProcess(int(self.process._handle))
        if result:
            raise RunManagerError("WORKER_RESUME_FAILED", hex(result & 0xffffffff))

    def close(self):
        result = self.job.close()
        self.process.wait(timeout=10)
        for thread in self.threads:
            thread.join(timeout=2)
        return result
