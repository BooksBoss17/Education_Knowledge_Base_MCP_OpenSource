"""Explicit GPU budgets for WeMM; offload weights without changing precision."""
from __future__ import annotations

POLICY = 'wemm-bf16-weight-offload-v1'
LIMITS = {'6gb': 6144, '10gb': 10240}


def plan_budget(profile, total_mib, used_mib):
    if profile not in LIMITS:
        raise ValueError('WeMM resource profile must be 6gb or 10gb')
    if total_mib <= 0 or used_mib < 0 or used_mib > total_mib:
        raise ValueError('Invalid GPU memory measurement')
    ceiling = min(LIMITS[profile], int(total_mib))
    # Account for the display, other apps and our CUDA context before loading.
    # Reserve 256 MiB beyond the allocator plus 1024 MiB for activations/hooks.
    allocator = int(ceiling-used_mib-256)
    weights = allocator-1024
    if weights < 1024:
        raise RuntimeError('WeMM budget unavailable: other GPU use leaves less than 1GiB for weights')
    return dict(policy=POLICY,profile=profile,whole_device_limit_mib=ceiling,
                observed_total_mib=total_mib,observed_used_before_load_mib=used_mib,
                allocator_limit_mib=allocator,gpu_weight_budget_mib=weights,
                activation_reserve_mib=1024,external_growth_reserve_mib=256,
                allocator_fraction=allocator/total_mib,
                placement='ACCELERATE_GPU_WITH_CPU_WEIGHT_OFFLOAD',precision='bfloat16',
                quantization=False)
