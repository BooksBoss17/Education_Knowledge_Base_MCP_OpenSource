"""Explicit GPU budgets for WeMM; offload weights without changing precision."""
from __future__ import annotations
import json
import os

POLICY = 'wemm-bf16-weight-offload-v1'
LIMITS = {'6gb': 6144, '10gb': 10240, 'unrestricted': None}


def plan_budget(profile, total_mib, used_mib):
    snapshot = json.loads(os.environ.get('BEMARKDOWN_HOST_MEMORY_POLICY_JSON', '{}'))
    if snapshot.get('mode') == 'monitor':
        profile = 'unrestricted'
    if profile not in LIMITS:
        raise ValueError(f'Unknown WeMM resource profile: {profile}')
    if total_mib <= 0 or used_mib < 0 or used_mib > total_mib:
        raise ValueError('Invalid GPU memory measurement')
    if profile == 'unrestricted':
        return dict(policy=POLICY, profile=profile, whole_device_limit_mib=None,
                    observed_total_mib=total_mib, observed_used_before_load_mib=used_mib,
                    allocator_fraction=None, gpu_weight_budget_mib=None, allocator_limit_mib=None,
                    precision='bfloat16', quantization=False, placement='AUTO')
    ceiling = min(LIMITS[profile], int(total_mib))
    if snapshot.get('gpu_whole_card_cap_bytes') is not None:
        ceiling = min(ceiling, snapshot['gpu_whole_card_cap_bytes'] / 2**20)
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
