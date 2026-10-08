"""Training/inference supervisor: dynamic Job commit cap, host floors, whole-device GPU guard.

Unlike the legacy deployment monitor, training is not capped at 6 GiB.
"""
import argparse
import collections
import ctypes
import ctypes.wintypes as W
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid
import _winapi

P = pathlib.Path(__file__).resolve().parents[1]
R = P.parents[1]
B = R / 'tmp/retention-v6-20261003'
sys.path.insert(0, str(B / 'v41-family-monitor-bounded'))
sys.path.insert(0, str(B))
import job_family_v41 as F
from memory_guard_v33 import sample
from memory_watchdog_v34 import process_info, save
F.probe.K.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
F.probe.K.TerminateJobObject.restype = W.BOOL

class PIDLIST(ctypes.Structure):
    _fields_ = [('assigned', W.DWORD), ('listed', W.DWORD), ('pids', ctypes.c_size_t * 128)]

def members(job):
    p = PIDLIST()
    F.probe.must(F.probe.K.QueryInformationJobObject(job.handle, 3, ctypes.byref(p), ctypes.sizeof(p), None))
    if p.assigned != p.listed:
        raise RuntimeError('INCOMPLETE_JOB_MEMBERSHIP')
    return list(p.pids[:p.listed])

def gpu():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=5)
    rows = [[int(v.strip()) for v in line.split(',')] for line in raw.strip().splitlines()]
    assert len(rows) == 1, 'SINGLE_DEVICE_PLAN_ONLY'
    return dict(zip(('total_mib', 'used_mib', 'free_mib'), rows[0]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('worker')
    ap.add_argument('--expected-private-gib', type=float, required=True)
    ap.add_argument('--expected-working-gib', type=float, required=True)
    ap.add_argument('--expected-gpu-gib', type=float, required=True)
    ap.add_argument('--timeout', type=int, default=86400)
    ap.add_argument('--deployment-budget', action='store_true',
                    help='Enforce the separate 6 GiB deployment process-family budget and 3 GiB host commit floor.')
    ap.add_argument('--minimum-host-commit-gib', type=float, default=None,
                    help='Raise, never lower, the configured host commit reserve for this stage.')
    cut = sys.argv.index('--') if '--' in sys.argv else len(sys.argv)
    args = ap.parse_args(sys.argv[1:cut])
    args.worker_args = sys.argv[cut + 1:] if cut < len(sys.argv) else []
    worker = pathlib.Path(args.worker).resolve()
    assert worker.is_relative_to(P / 'tools') and worker.is_file()
    bootstrap = json.loads((P / 'reports/native-environment-bootstrap.json').read_text('utf-8'))
    python = bootstrap['venv_python']
    out = R / 'tmp/rt-train/runs' / ('native-model-' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    ctrl = process_info(os.getpid())
    host = sample()
    before_gpu = gpu()
    policy = json.loads((P / 'configs/resource-policy.json').read_text('utf-8'))
    commit_floor = policy['training'].get('emergency_commit_headroom_gib', policy['host_floors']['commit_headroom_gib'])
    if args.minimum_host_commit_gib is not None:
        assert args.minimum_host_commit_gib > 0
        commit_floor = max(commit_floor, args.minimum_host_commit_gib)
    private_cap = host['commit_headroom_gib'] - commit_floor - .25
    working_cap = host['PhysicalAvailable_gib'] - 2 - .25
    if args.deployment_budget:
        commit_floor = max(3., commit_floor)
        private_cap = min(host['commit_headroom_gib'] - commit_floor - .25,
                          6. - ctrl['private_commit_gib'] - .02)
        working_cap = min(working_cap, 6.)
    admitted = private_cap >= args.expected_private_gib and working_cap >= args.expected_working_gib and before_gpu['free_mib'] / 1024 >= args.expected_gpu_gib + 1
    admission = {'admitted': admitted, 'phase': worker.stem, 'host': host, 'gpu_whole_device': before_gpu,
                 'controller': ctrl, 'dynamic_private_cap_gib': private_cap, 'dynamic_working_cap_gib': working_cap,
                 'expected_private_gib': args.expected_private_gib, 'expected_working_gib': args.expected_working_gib,
                 'expected_gpu_gib': args.expected_gpu_gib, 'no_fixed_six_gib_training_cap': not args.deployment_budget,
                 'resource_mode': 'deployment_process_family_6gib' if args.deployment_budget else 'dynamic_training_offline',
                 'commit_gap_gib': max(0, args.expected_private_gib - private_cap),
                 'commit_emergency_floor_gib': commit_floor, 'physical_gpu_primary_admission': policy['training'].get('physical_gpu_primary_admission', False)}
    save(out / 'admission.json', admission)
    if not admitted:
        save(out / 'result.json', {'status': 'RESOURCE_ADMISSION_REFUSED', 'training_started': False, 'admission': admission})
        print(json.dumps({'status': 'RESOURCE_ADMISSION_REFUSED', 'result': str(out / 'result.json')}))
        return 2
    job = F.Job(int(private_cap * 2**30) // 4096 * 4096, int(private_cap * 2**30) // 4096 * 4096)
    F.probe.ROOT = out
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    os.environ.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_DISABLED='true', WANDB_MODE='disabled')
    for key, name in {'HF_HOME': 'hf-cache', 'TORCH_HOME': 'torch-cache',
                      'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TRITON_CACHE_DIR': 'triton',
                      'CUDA_CACHE_PATH': 'cuda-cache', 'XDG_CACHE_HOME': 'cache',
                      'TEMP': 'temp', 'TMP': 'temp'}.items():
        directory = R / 'tmp/rt-train' / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ.update(HF_ENABLE_PARALLEL_LOADING='false', TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='1')
    hp = ht = None
    identities = {}
    helper_handles = {}
    recent = collections.deque(maxlen=64)
    peaks = {'project_working_gib': 0., 'project_private_commit_gib': 0., 'whole_device_used_mib': before_gpu['used_mib']}
    reason = None
    result = {'status': 'STARTING'}
    start = time.monotonic()
    try:
        hp, ht, ident = F.suspended(job, [python, str(worker), '--run-dir', str(out), *args.worker_args], 'worker')
        save(out / 'process.json', {'identity': ident, 'controller': ctrl})
        identities[ident['pid']] = ident
        # Job limits and host admission are active before the first user instruction executes.
        F.probe.must(F.probe.K.ResumeThread(ht) != 0xffffffff)
        _winapi.CloseHandle(ht)
        ht = None
        with (out / 'resources.jsonl').open('w', encoding='utf-8', buffering=1) as stream:
            while True:
                native_code = _winapi.GetExitCodeProcess(hp)
                if native_code != 259:
                    result.update(status='COMPLETED' if native_code == 0 else 'WORKER_FAILED', native_exit_code=native_code)
                    break
                observed = []
                for pid in members(job):
                    if pid in helper_handles and _winapi.GetExitCodeProcess(helper_handles[pid]) != 259:
                        continue
                    info = process_info(pid)
                    if info is None:
                        # Exit is checked by the retained native handle, not a recycled PID.
                        if _winapi.GetExitCodeProcess(hp) != 259 or pid not in members(job):
                            continue
                        time.sleep(.02)
                        info = process_info(pid)
                        if info is None and pid in members(job):
                            raise RuntimeError('OWNED_MEMBER_INSPECTION_FAILED')
                    if info is None:
                        continue
                    old = identities.get(pid)
                    if old and (old['created_ticks'] != info['created_ticks'] or old['exe'] != info['exe']):
                        raise RuntimeError('OWNED_PID_IDENTITY_CHANGED')
                    identities[pid] = info
                    if pid != ident['pid'] and pid not in helper_handles:
                        helper_handles[pid] = _winapi.OpenProcess(0x100000 | 0x1000, False, pid)
                        if _winapi.GetExitCodeProcess(helper_handles[pid]) != 259:
                            continue
                    observed.append(info)
                now_host = sample()
                now_gpu = gpu()
                controller = process_info(os.getpid())
                working = sum(x['working_gib'] for x in observed) + controller['working_gib']
                private = sum(x['private_commit_gib'] for x in observed) + controller['private_commit_gib']
                peaks['project_working_gib'] = max(peaks['project_working_gib'], working)
                peaks['project_private_commit_gib'] = max(peaks['project_private_commit_gib'], private)
                peaks['whole_device_used_mib'] = max(peaks['whole_device_used_mib'], now_gpu['used_mib'])
                row = {'epoch': time.time(), 'host': now_host, 'gpu_whole_device': now_gpu, 'members': observed, 'controller': controller, 'project_working_gib': working, 'project_private_commit_gib': private}
                recent.append(row)
                stream.write(json.dumps(row) + '\n')
                save(out / 'latest-resource.json', row)
                reason = ('HOST_COMMIT_EMERGENCY_FLOOR' if now_host['commit_headroom_gib'] < commit_floor else
                          'HOST_PHYSICAL_FLOOR' if now_host['PhysicalAvailable_gib'] < 2 else
                          'PROJECT_WORKING_BUDGET' if working > working_cap else
                          'PROJECT_PRIVATE_BUDGET' if private > private_cap else
                          'GPU_FREE_RESERVE' if now_gpu['free_mib'] < 1024 else
                          'PHASE_TIMEOUT' if time.monotonic() - start > args.timeout else None)
                if reason:
                    save(out / 'stop.json', {'reason': reason, 'epoch': time.time()})
                    F.probe.K.TerminateJobObject(job.handle, 91)
                    result.update(status='STOPPED_WITH_EVIDENCE', reason=reason)
                    break
                time.sleep(.8)
    except BaseException as exc:
        save(out / 'stop.json', {'reason': type(exc).__name__, 'error': str(exc), 'epoch': time.time()})
        result.update(status='STOPPED_WITH_EVIDENCE', error_type=type(exc).__name__, error=str(exc))
    finally:
        if ht is not None:
            _winapi.CloseHandle(ht)
        if hp is not None:
            if _winapi.GetExitCodeProcess(hp) == 259:
                F.probe.must(F.probe.K.TerminateJobObject(job.handle, 91))
            _winapi.WaitForSingleObject(hp, 5000)
            result['native_exit_code'] = _winapi.GetExitCodeProcess(hp)
        result['job_native_stats'] = job.stats()
        job.close()  # closes its retained process handles only after native exit was captured
        for handle in helper_handles.values():
            _winapi.CloseHandle(handle)
        remaining = []
        for pid, ident in identities.items():
            current = process_info(pid)
            if current and current['created_ticks'] == ident['created_ticks']:
                remaining.append(pid)
        result.update(peaks=peaks, observed_identities=list(identities.values()), elapsed_seconds=time.monotonic() - start, after_host=sample(), after_gpu_whole_device=gpu(), all_owned_exited=not remaining, remaining_owned_pids=remaining)
        save(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'result': str(out / 'result.json')}))
    return 0 if result['status'] == 'COMPLETED' else 2

if __name__ == '__main__':
    sys.exit(main())
