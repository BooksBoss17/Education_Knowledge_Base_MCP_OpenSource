"""Sample physical GPU memory without confusing unavailable counters with zero.

The global counter includes the desktop and other processes. Sampling observes
peaks; it neither caps allocations nor proves an absolute memory ceiling.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time
from typing import Self
from uuid import UUID

_MIB = 1024 ** 2
_UNAVAILABLE = 2 ** 64 - 1


class _Memory(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


class _ProcessV2(ctypes.Structure):
    _fields_ = [('pid', ctypes.c_uint), ('usedGpuMemory', ctypes.c_ulonglong),
                ('gpuInstanceId', ctypes.c_uint), ('computeInstanceId', ctypes.c_uint)]


class NvmlReader:
    """Use the installed NVIDIA driver library; no Python NVML dependency."""

    backend = 'nvml-ctypes'

    def __init__(self, device_selector: str):
        self.device_selector = device_selector
        self.lib = ctypes.CDLL('nvml.dll' if os.name == 'nt' else 'libnvidia-ml.so.1')
        self._check(self.lib.nvmlInit_v2())
        try:
            self.handle = ctypes.c_void_p()
            if device_selector.startswith(('GPU-', 'MIG-')):
                self._check(self.lib.nvmlDeviceGetHandleByUUID(
                    ctypes.c_char_p(device_selector.encode('ascii')), ctypes.byref(self.handle)))
            else:
                index = int(device_selector)
                if index < 0:
                    raise ValueError('GPU_DEVICE_INDEX_MUST_BE_NONNEGATIVE')
                self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(
                    ctypes.c_uint(index), ctypes.byref(self.handle)))
        except (OSError, AttributeError, ValueError):
            self.lib.nvmlShutdown()
            raise

    @staticmethod
    def _check(code: int) -> None:
        if code:
            raise OSError(f'NVML_RETURN_CODE:{code}')

    def read_global(self) -> tuple[float, float]:
        memory = _Memory()
        self._check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)))
        return memory.total / _MIB, memory.used / _MIB

    def read_process(self, pid: int) -> float | None:
        query = getattr(self.lib, 'nvmlDeviceGetComputeRunningProcesses_v2', None)
        if query is None:
            return None
        count = ctypes.c_uint(0)
        code = query(self.handle, ctypes.byref(count), None)
        if code == 3:  # NVML_ERROR_NOT_SUPPORTED, common for process accounting.
            return None
        if code not in (0, 7):  # NVML_ERROR_INSUFFICIENT_SIZE is the size query.
            self._check(code)
        if not count.value:
            return None
        # Leave room for contexts appearing between the two NVML calls.
        count.value += 8
        rows = (_ProcessV2 * count.value)()
        code = query(self.handle, ctypes.byref(count), rows)
        if code == 3:
            return None
        self._check(code)
        values = [row.usedGpuMemory for row in rows[:count.value] if row.pid == pid]
        if not values or _UNAVAILABLE in values:
            return None
        return sum(values) / _MIB

    def close(self) -> None:
        self._check(self.lib.nvmlShutdown())


class NvidiaSmiReader:
    """Slower fallback with independent global and process queries."""

    backend = 'nvidia-smi'

    def __init__(self, device_selector: str):
        self.device_selector = device_selector

    def _query(self, query: str) -> str:
        return subprocess.run(
            ['nvidia-smi', query, '--format=csv,noheader,nounits',
             f'--id={self.device_selector}'],
            capture_output=True, text=True, check=True, timeout=3,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        ).stdout

    def read_global(self) -> tuple[float, float]:
        row = self._query('--query-gpu=memory.total,memory.used').strip().splitlines()[0]
        total, used = row.split(',', 1)
        return float(total.strip()), float(used.strip())

    def read_process(self, pid: int) -> float | None:
        values = []
        for line in self._query('--query-compute-apps=pid,used_memory').splitlines():
            process_id, used = (part.strip() for part in line.split(',', 1))
            if int(process_id) != pid:
                continue
            if used in ('[N/A]', 'N/A', '[Not Supported]', 'Not Supported'):
                return None
            values.append(float(used))
        return sum(values) if values else None

    def close(self) -> None:
        pass


class VramMonitor:
    """Observe one explicitly identified physical GPU across a context lifetime."""

    def __init__(self, *, device_selector: str = '0', interval_seconds: float = 0.05,
                 reader=None, pid: int | None = None) -> None:
        if interval_seconds <= 0:
            raise ValueError('GPU_SAMPLING_INTERVAL_MUST_BE_POSITIVE')
        self.pid = os.getpid() if pid is None else pid
        # Torch exposes an unprefixed UUID; NVML expects the GPU- prefix.
        try:
            device_selector = f'GPU-{UUID(device_selector)}'
        except ValueError:
            pass
        self.device_selector = device_selector
        self.interval_seconds = interval_seconds
        self.reader = reader
        self.backend_fallback_reason = None
        self.peak_mib = None
        self.peak_global_mib = None
        self.baseline_global_mib = None
        self.total_mib = None
        self.global_sample_count = 0
        self.process_sample_count = 0
        self.process_unknown_sample_count = 0
        self.global_sampling_error_count = 0
        self.process_sampling_error_count = 0
        self.close_error = None
        self.first_sampling_error = None
        self.samples = []
        self._started = time.perf_counter()
        self._ended = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    @property
    def sampling_error_count(self) -> int:
        return self.global_sampling_error_count + self.process_sampling_error_count

    def _open(self) -> None:
        if self.reader is None:
            try:
                self.reader = NvmlReader(self.device_selector)
            except (OSError, AttributeError, ValueError) as exc:
                self.backend_fallback_reason = f'{type(exc).__name__}: {exc}'
                self.reader = NvidiaSmiReader(self.device_selector)
                # Do not repeatedly start external processes at the NVML cadence.
                self.interval_seconds = max(0.5, self.interval_seconds)

    def _record_error(self, exc: Exception) -> None:
        if self.first_sampling_error is None:
            self.first_sampling_error = f'{type(exc).__name__}: {exc}'

    def sample_once(self) -> None:
        self._open()
        try:
            self.total_mib, used = self.reader.read_global()
            if self.baseline_global_mib is None:
                self.baseline_global_mib = used
            self.peak_global_mib = max(self.peak_global_mib or 0, used)
            self.global_sample_count += 1
            self.samples.append({'elapsed_seconds': time.perf_counter() - self._started,
                                 'global_used_mib': used})
        except (OSError, subprocess.SubprocessError, ValueError, IndexError, AttributeError) as exc:
            self.global_sampling_error_count += 1
            self._record_error(exc)
        try:
            used = self.reader.read_process(self.pid)
            if used is None:
                self.process_unknown_sample_count += 1
            else:
                self.peak_mib = max(self.peak_mib or 0, used)
                self.process_sample_count += 1
        except (OSError, subprocess.SubprocessError, ValueError, IndexError, AttributeError) as exc:
            self.process_sampling_error_count += 1
            self._record_error(exc)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample_once()

    def __enter__(self) -> Self:
        self._started = time.perf_counter()
        self.sample_once()
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.sample_once()
        self._ended = time.perf_counter()
        try:
            self.reader.close()
        except (OSError, AttributeError) as exc:
            self.close_error = f'{type(exc).__name__}: {exc}'

    def _threshold(self, limit_mib: int) -> str:
        if self.peak_global_mib is None:
            return 'UNKNOWN'
        if self.peak_global_mib > limit_mib:
            return 'EXCEEDED'
        if self.global_sampling_error_count:
            return 'INCOMPLETE'
        return 'WITHIN_SAMPLED_PEAK'

    def summary(self) -> dict:
        gaps = [b['elapsed_seconds'] - a['elapsed_seconds']
                for a, b in zip(self.samples, self.samples[1:])]
        return {
            'schema': 'bemarkdown-gpu-memory-observation-v1',
            'scope': 'physical GPU total usage, including desktop and unrelated processes',
            'enforces_allocation_limit': False,
            'backend': getattr(self.reader, 'backend', None),
            'backend_fallback_reason': self.backend_fallback_reason,
            'device_selector': self.device_selector,
            'process_id': self.pid,
            'interval_seconds': self.interval_seconds,
            'duration_seconds': (self._ended or time.perf_counter()) - self._started,
            'max_successful_sample_gap_seconds': max(gaps, default=None),
            'total_mib': self.total_mib,
            'baseline_global_used_mib': self.baseline_global_mib,
            'peak_global_used_mib': self.peak_global_mib,
            'peak_process_used_mib': self.peak_mib,
            'global_sample_count': self.global_sample_count,
            'process_sample_count': self.process_sample_count,
            'process_unknown_sample_count': self.process_unknown_sample_count,
            'global_sampling_error_count': self.global_sampling_error_count,
            'process_sampling_error_count': self.process_sampling_error_count,
            'first_sampling_error': self.first_sampling_error,
            'close_error': self.close_error,
            'budget_6144_mib': self._threshold(6144),
            'ceiling_8192_mib': self._threshold(8192),
        }
