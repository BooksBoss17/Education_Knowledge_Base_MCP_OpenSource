"""Bound idle Paddle CUDA cache without changing live tensors or predictions."""

from __future__ import annotations

import logging
import sys

IDLE_CACHE_BUDGET_BYTES = 512 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)


def release_idle_paddle_cache() -> bool:
    """Release only unused allocations; never import Paddle into other providers.

    Variable-width recognition can retain gigabytes of unused shape allocations
    over a book. This budget is for idle cache, not model memory or a GPU limit.
    Live allocations and all inference settings remain untouched.
    """
    paddle = sys.modules.get('paddle')
    device = getattr(paddle, 'device', None)
    get_device = getattr(device, 'get_device', None)
    if not callable(get_device) or not get_device().startswith('gpu'):
        return False
    cuda = device.cuda
    unused = cuda.memory_reserved() - cuda.memory_allocated()
    if unused <= IDLE_CACHE_BUDGET_BYTES:
        return False
    device.synchronize()
    cuda.empty_cache()
    _LOGGER.info('PADDLE_IDLE_CACHE_RELEASED unused_before_mib=%.2f', unused / 1024**2)
    return True


def predict_with_bounded_cache(model, inputs, *, batch_size, **kwargs):
    """Keep the predictor's original call, batches, padding, and result order."""
    count = 0
    try:
        for value in model.predict(inputs, batch_size=batch_size, **kwargs):
            yield value
            count += 1
            if count % batch_size == 0:
                release_idle_paddle_cache()
    finally:
        # Also reclaim idle cache on partial batches, cancellation or failure.
        release_idle_paddle_cache()
