"""Bounded, fail-closed Windows Job enumeration during child process churn."""
import ctypes
from ctypes import wintypes
import time


def complete_job_members(handle, kernel, *, attempts=5, initial_capacity=128,
                         maximum_capacity=4096, retry_delay=.002, observe=None):
    if (not 1 <= attempts <= 5 or not 1 <= initial_capacity <= maximum_capacity <= 4096
            or not 0 <= retry_delay <= .002):
        raise ValueError('Invalid bounded Job enumeration limits')
    capacity = initial_capacity
    last = None
    for attempt in range(attempts):
        class PidList(ctypes.Structure):
            _fields_ = [('assigned', wintypes.DWORD), ('listed', wintypes.DWORD),
                        ('pids', ctypes.c_size_t * capacity)]
        value = PidList()
        ctypes.set_last_error(0)
        ok = kernel.QueryInformationJobObject(handle, 3, ctypes.byref(value),
                                             ctypes.sizeof(value), None)
        error = ctypes.get_last_error() or (kernel.GetLastError() if not ok else 0)
        last = {'attempt': attempt + 1, 'capacity': capacity, 'assigned': value.assigned,
                'listed': value.listed, 'ok': bool(ok), 'winerror': error}
        if observe is not None:
            observe(last)
        if not ok and error != 234:  # ERROR_MORE_DATA is the sole retryable API error.
            raise OSError(error, 'QueryInformationJobObject failed')
        if ok and value.assigned == value.listed and value.listed <= capacity:
            pids = list(value.pids[:value.listed])
            if any(pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
                raise RuntimeError('Invalid complete Job PID list')
            return pids
        required = max(value.assigned, value.listed, capacity * 2 if not ok else capacity)
        if required > maximum_capacity:
            raise RuntimeError('Job membership exceeds bounded enumeration capacity')
        capacity = max(capacity, required)
        if attempt + 1 < attempts:
            time.sleep(retry_delay)
    raise RuntimeError(f'Incomplete Job membership after bounded requery: {last}')
