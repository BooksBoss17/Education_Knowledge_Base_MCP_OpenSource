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


import runpy as _ram_runpy
from pathlib import Path as _RamPath
_host_ram = _ram_runpy.run_path(str(_RamPath(__file__).resolve().parents[1]/'core/bemarkdown/host_memory_policy.py'))
current_memory_policy = _host_ram['current_memory_policy']

@dataclasses.dataclass(frozen=True)
class ResourcePolicy:
    physical_free_gib: float = dataclasses.field(default_factory=lambda: current_memory_policy()['physical_free_floor_gib'])
    commit_headroom_gib: float = dataclasses.field(default_factory=lambda: current_memory_policy()['commit_headroom_floor_gib'])
    gpu_free_mib: float = dataclasses.field(default_factory=lambda: current_memory_policy()['gpu_reserve_bytes']/2**20)
    gpu_admission_free_mib: float = dataclasses.field(default_factory=lambda: current_memory_policy()['gpu_startup_free_floor_bytes']/2**20)
    gpu_used_mib: float = dataclasses.field(default_factory=lambda: current_memory_policy()['gpu_whole_card_cap_bytes']/2**20)
    version: str = 'bemarkdown-device-resource-policy-v2'
    def __post_init__(self):
        p=current_memory_policy()
        if (self.physical_free_gib<p['physical_free_floor_gib'] or self.commit_headroom_gib<p['commit_headroom_floor_gib'] or
            self.gpu_free_mib<p['gpu_reserve_bytes']/2**20 or self.gpu_admission_free_mib<p['gpu_startup_free_floor_bytes']/2**20 or
            not 0<self.gpu_used_mib<=p['gpu_whole_card_cap_bytes']/2**20):
            raise ValueError('AUTHORIZED_RESOURCE_LIMITS_CANNOT_BE_WEAKENED')
    def check(self,host,gpu):
        p=current_memory_policy()
        _host_ram['check_current'](p,_host_ram['observation_from_legacy'](host,gpu,policy=p))
        if host['PhysicalAvailable_gib']<self.physical_free_gib or host['commit_headroom_gib']<self.commit_headroom_gib:
            raise MemoryError('RESOURCE_REQUESTED_STRICTER_RAM_FLOOR')
        if gpu['free']<self.gpu_free_mib or gpu['used']>self.gpu_used_mib:raise MemoryError('RESOURCE_REQUESTED_STRICTER_GPU_GUARD')
    def check_projected(self,host,gpu,*,family_private_gib,stage_increment,margin):
        self.check(host,gpu)
        for record in (stage_increment,margin):
            if set(record)!={'private_gib','working_gib','gpu_mib'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in record.values()):
                raise ValueError('STAGE_FORECAST_INVALID')
        p=current_memory_policy()
        observation=_host_ram['observation_from_legacy'](host,gpu,group_private_gib=family_private_gib,policy=p)
        bound=dict(extra_private_bytes=(stage_increment['private_gib']+margin['private_gib'])*2**30,
            extra_physical_bytes=(stage_increment['working_gib']+margin['working_gib'])*2**30,
            extra_gpu_mib=stage_increment['gpu_mib']+margin['gpu_mib'],host_reserve_bytes=0)
        decision=_host_ram['evaluate_admission'](p,observation,bound)
        if decision['decision']!='ADMIT':raise MemoryError(decision['decision']+':'+','.join(decision['reasons']))
        r=decision['projected']
        return dict(family_private_commit_gib=r['group_current_bytes']/2**30,physical_free_gib=r['physical_free_bytes']/2**30,
            commit_headroom_gib=r['commit_headroom_bytes']/2**30,gpu_used_mib=r['gpu_used_bytes']/2**20,gpu_free_mib=r['gpu_free_bytes']/2**20)

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
    runner = Path(__file__).with_name('native_stage_admission.py').resolve()
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
