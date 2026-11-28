"""Windows containment for a single OCR session within the manager's outer Job."""
import ctypes
from ctypes import wintypes
import time


class Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in ("user", "kernel", "period_user", "period_kernel")] + [
        (name, wintypes.DWORD) for name in ("faults", "total", "active", "terminated")]


class SessionJob:
    def __init__(self):
        k = self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            fn = getattr(k, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.check(k.CreateJobObjectW(None, None))

    @staticmethod
    def check(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def attach(self, child):
        self.check(self.kernel.AssignProcessToJobObject(self.handle, int(child._handle)))

    def empty(self):
        info = Accounting()
        self.check(self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return info.active == 0

    def terminate(self):
        self.check(self.kernel.TerminateJobObject(self.handle, 1))

    def wait_empty(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.empty():
            if time.monotonic() >= deadline:
                return False
            time.sleep(.05)
        return True

    def close(self):
        self.check(self.kernel.CloseHandle(self.handle))
        self.handle = None
