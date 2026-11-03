"""Aggregate per-step backend usage without confusing it with last-step totals."""
import math


def add_step_usage(total, usage):
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or value < 0:
                raise ValueError('AGENT_USAGE_INVALID_INCREMENT')
            total[key] = total.get(key, 0) + value

