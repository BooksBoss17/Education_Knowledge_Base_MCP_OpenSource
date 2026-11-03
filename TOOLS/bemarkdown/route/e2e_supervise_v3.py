"""Opt-in strict resource supervision; preserves legacy runners and their snapshots."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import e2e_supervise as legacy


@dataclasses.dataclass(frozen=True)
class ResourcePolicy:
    physical_free_gib: float = 8.0
    commit_headroom_gib: float = 9.0
    gpu_free_mib: float = 1024.0
    gpu_admission_free_mib: float = 7168.0
    gpu_used_mib: float = 6144.0
    version: str = 'bemarkdown-v3-admission-and-continuous-resource-policy-v2'

    def __post_init__(self):
        if (self.physical_free_gib < 8 or self.commit_headroom_gib < 9 or
                self.gpu_admission_free_mib < 7168 or self.gpu_free_mib < 1024 or
                self.gpu_used_mib > 6144 or self.gpu_used_mib <= 0):
            raise ValueError('AUTHORIZED_RESOURCE_LIMITS_CANNOT_BE_WEAKENED')

    def check(self, host, gpu):
        values = (host['PhysicalAvailable_gib'], host['commit_headroom_gib'], gpu['free'], gpu['used'])
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise RuntimeError('RESOURCE_OBSERVATION_INVALID')
        if host['PhysicalAvailable_gib'] < self.physical_free_gib:
            raise MemoryError('V3_PHYSICAL_FREE_FLOOR')
        if host['commit_headroom_gib'] < self.commit_headroom_gib:
            raise MemoryError('V3_COMMIT_HEADROOM_FLOOR')
        if gpu['free'] < self.gpu_free_mib:
            raise MemoryError('V3_GPU_FREE_FLOOR')
        if gpu['used'] > self.gpu_used_mib:
            raise MemoryError('V3_GPU_WHOLE_CARD_CAP')

    def check_projected(self, host, gpu, *, family_private_gib, stage_increment, margin):
        """Reserve a measured additional load before it starts; keep hard caps.

        The caller binds the forecast to the executed stage and supplies fresh
        whole-family observations. Resident allocations must not be counted a
        second time in stage_increment; the six-GiB cap remains per whole job.
        """
        self.check(host, gpu)
        values = [family_private_gib]
        for record in (stage_increment, margin):
            if set(record) != {'private_gib', 'working_gib', 'gpu_mib'}:
                raise ValueError('STAGE_FORECAST_FIELDS_REQUIRED')
            values.extend(record.values())
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('STAGE_FORECAST_INVALID')
        private = stage_increment['private_gib'] + margin['private_gib']
        working = stage_increment['working_gib'] + margin['working_gib']
        vram = stage_increment['gpu_mib'] + margin['gpu_mib']
        projected = {
            'family_private_commit_gib': family_private_gib + private,
            'physical_free_gib': host['PhysicalAvailable_gib'] - working,
            'commit_headroom_gib': host['commit_headroom_gib'] - private,
            'gpu_used_mib': gpu['used'] + vram,
            'gpu_free_mib': gpu['free'] - vram,
        }
        if projected['family_private_commit_gib'] > 6.0:
            raise MemoryError('STAGE_PROJECTED_FAMILY_COMMIT_CAP')
        if projected['physical_free_gib'] < self.physical_free_gib:
            raise MemoryError('STAGE_PROJECTED_PHYSICAL_FREE_FLOOR')
        if projected['commit_headroom_gib'] < self.commit_headroom_gib:
            raise MemoryError('STAGE_PROJECTED_COMMIT_HEADROOM_FLOOR')
        if projected['gpu_used_mib'] > self.gpu_used_mib or projected['gpu_free_mib'] < self.gpu_free_mib:
            raise MemoryError('STAGE_PROJECTED_GPU_CAP_OR_RESERVE')
        return projected


class StrictFamily(legacy.LiveFamily):
    """Checks every real family observation, including the sample before child resume."""
    def __init__(self, *args, gpu_sampler=None, **kwargs):
        self.policy = ResourcePolicy()
        self._gpu_sampler = gpu_sampler or legacy.GPU().sample
        super().__init__(*args, **kwargs)

    def record(self, row):
        gpu = self._gpu_sampler()
        row['gpu'] = gpu
        row['resource_policy'] = self.policy.version
        self.peaks['gpu_whole_card_used_mib'] = max(self.peaks.get('gpu_whole_card_used_mib', 0), gpu['used'])
        self.minima['gpu_free_mib'] = min(self.minima.get('gpu_free_mib', float('inf')), gpu['free'])
        # Preserve the triggering observation before fail-closed stops this Job.
        super().record(row)
        self.policy.check(row['host'], gpu)

    def summary(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_summary < self.summary_interval:
            return
        self.summary_writes += 1
        legacy.write(self.out / 'summary.json', {
            'session_started': self.session_started, 'sample_count': self.count,
            'failed': self.failed, 'closed': self.closed, 'budget_gib': self.budget,
            'global_commit_floor_gib': self.policy.commit_headroom_gib,
            'global_physical_floor_gib': self.policy.physical_free_gib,
            'gpu_free_floor_mib': self.policy.gpu_free_mib,
            'gpu_admission_free_floor_mib': self.policy.gpu_admission_free_mib,
            'gpu_whole_card_cap_mib': self.policy.gpu_used_mib,
            'resource_policy': dataclasses.asdict(self.policy),
            'peaks': self.peaks, 'minimum_headroom': self.minima,
            'current_members': self.current, 'observed_identities': list(self.identities.values()),
            'runner_identity': self.runner, 'recent_samples': list(self.samples),
            'recent_limit': self.samples.maxlen, 'summary_writes': self.summary_writes,
        })
        self.last_summary = now


def main():
    # Process-local dependency injection; the historical module and other processes stay intact.
    legacy.LiveFamily = StrictFamily
    config = {}
    if '--config' in sys.argv:
        config_path = Path(sys.argv[sys.argv.index('--config') + 1]).resolve()
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    admission = config.get('stage_admission')
    if not admission:
        return legacy.main()
    manifest_path = Path(admission['manifest_path']).resolve()
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(manifest_path) != admission['manifest_sha256']:
        raise RuntimeError('STAGE_ADMISSION_MANIFEST_CHANGED')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest['version'] != 'measured-stage-admission-v1' or not manifest['stages']:
        raise RuntimeError('STAGE_ADMISSION_MANIFEST_INVALID')
    for bound in manifest['bindings']:
        if digest(bound['path']) != bound['sha256']:
            raise RuntimeError('STAGE_ADMISSION_BOUND_EVIDENCE_CHANGED')
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    runner = Path(__file__).with_name('stage_admission_check.py').resolve()
    original_launch = legacy.F.suspended
    env_name = 'BEMARKDOWN_STAGE_ADMISSION_CONTEXT'
    previous_env = os.environ.get(env_name)
    def guarded_launch(job, *args, **kwargs):
        controller = legacy.process_info(os.getpid())
        os.environ[env_name] = json.dumps({
            'controller_pid': controller['pid'],
            'controller_created_ticks': controller['created_ticks'],
            'job_handle': int(job.handle),
            'python': sys.executable, 'runner_path': str(runner),
            'runner_sha256': digest(runner), 'policy_sha256': digest(__file__),
            'manifest_path': str(manifest_path), 'manifest_sha256': digest(manifest_path),
            'additional_controller_identities': config.get('additional_controller_identities', []),
            'journal_path': str(out / 'stage-admission.jsonl'),
        })
        return original_launch(job, *args, **kwargs)
    legacy.F.suspended = guarded_launch
    try:
        return legacy.main()
    finally:
        legacy.F.suspended = original_launch
        if previous_env is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous_env


if __name__ == '__main__':
    sys.exit(main())
