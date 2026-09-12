"""One explicitly owned FormulaNet model for a serial PDF/visual-DOCX batch."""

from __future__ import annotations

import json
import threading


class FormulaRuntimeOwner:
    """Retain at most one model on its construction thread, never predictions."""

    def __init__(self, *, runtime_factory=None):
        if runtime_factory is None:
            from .formulanet_runtime import PaddleFormulaNetRuntime

            runtime_factory = PaddleFormulaNetRuntime
        self._factory = runtime_factory
        self._thread = threading.get_ident()
        self._runtime = None
        self._key = None
        self._active = 0
        self._retire = False
        self._closed = False
        self._counts = {'model_load_count': 0, 'model_unload_count': 0, 'borrow_count': 0,
                        'release_count': 0, 'forced_release_count': 0, 'prediction_calls': 0,
                        'prediction_inputs': 0}

    def _check(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError('FORMULA_MODEL_OWNER_THREAD_MISMATCH')
        if self._closed:
            raise RuntimeError('FORMULA_MODEL_OWNER_CLOSED')

    def acquire(self, **verified_config):
        self._check()
        key = json.dumps(verified_config, sort_keys=True, default=str)
        if self._runtime is not None and key != self._key:
            if self._active:
                raise RuntimeError('FORMULA_MODEL_ACTIVE_BORROW_CONFIG_CHANGED')
            self._unload()
        created = self._runtime is None
        if created:
            self._runtime = self._factory(**verified_config)
            self._key = key
            self._counts['model_load_count'] += 1
        self._active += 1
        self._counts['borrow_count'] += 1
        return _BorrowedFormulaRuntime(self, created=created)

    def _unload(self):
        runtime = self._runtime
        if runtime is None:
            return
        runtime.close()
        self._runtime = None
        self._key = None
        self._retire = False
        self._counts['model_unload_count'] += 1

    def _release(self):
        self._check()
        self._active -= 1
        self._counts['release_count'] += 1
        if self._retire and not self._active:
            self._unload()

    def close(self):
        if self._closed:
            return
        self._check()
        try:
            self._unload()
        finally:
            self._counts['forced_release_count'] += self._active
            self._active = 0
            self._closed = True

    def release_idle_model(self):
        """Free unused shared weights before a route with independent models."""
        self._check()
        if self._active:
            raise RuntimeError('FORMULA_MODEL_ACTIVE_BORROW_CANNOT_RELEASE')
        self._unload()

    def metrics(self):
        return {'schema': 'bemarkdown-formula-model-owner-v1', **self._counts,
                'closed': self._closed, 'active_borrows': self._active,
                'recognition_result_cache': False, 'capacity': 1}

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_exception):
        self.close()


class _BorrowedFormulaRuntime:
    retained_by_owner = True

    def __init__(self, owner, *, created):
        self._owner = owner
        self._closed = False
        self._unloaded_on_release = 0
        self.model_load_count = int(created)
        self.load_seconds = float(getattr(owner._runtime, 'load_seconds', 0.0)) if created else 0.0

    def _runtime(self):
        self._owner._check()
        if self._closed:
            raise RuntimeError('FORMULA_MODEL_BORROW_CLOSED')
        return self._owner._runtime

    def predict(self, paths, *, batch_size):
        runtime = self._runtime()
        self._owner._counts['prediction_calls'] += 1
        self._owner._counts['prediction_inputs'] += len(paths)
        try:
            return runtime.predict(paths, batch_size=batch_size)
        except BaseException:
            self._owner._retire = True
            raise

    def fingerprint(self):
        return self._runtime().fingerprint()

    def release_workspace(self):
        return self._runtime().release_workspace()

    def peak_gpu_memory_bytes(self):
        return self._runtime().peak_gpu_memory_bytes()

    def lifecycle_metrics(self):
        return {'ownership': 'SERIAL_BATCH', 'model_load_count': self.model_load_count,
                'model_load_seconds': self.load_seconds, 'release_count': int(self._closed),
                'model_unload_count': self._unloaded_on_release,
                'retained_until': 'BATCH_END_OR_INFERENCE_FAILURE'}

    def close(self):
        if self._closed:
            return
        if not self._owner._closed:
            self._owner._check()
            self._closed = True
            before = self._owner._counts['model_unload_count']
            try:
                self._owner._release()
            finally:
                self._unloaded_on_release = self._owner._counts['model_unload_count'] - before
        else:
            self._closed = True
