"""Current accepted forecast admission including MCP caller, without double load charging."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path

import runpy as _ram_runpy
from pathlib import Path as _RamPath
_host_ram = _ram_runpy.run_path(str(_RamPath(__file__).resolve().parents[1]/'core/bemarkdown/host_memory_policy.py'))
current_memory_policy = _host_ram['current_memory_policy']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def same(a,b):return a.get('pid')==b.get('pid') and a.get('created_ticks')==b.get('created_ticks')
def delegated_lease(stage,caller,lease,descendant):
    return (lease.get('stage')==stage and not same(caller,lease.get('parent',{}))
            and descendant(caller,lease.get('parent',{})))

def descendant(caller,parent):
    import psutil
    import e2e_supervise as legacy
    current=legacy.process_info(parent.get('pid',0))
    if current is None or not same(current,parent):return False
    pid=caller['pid']
    for _ in range(16):
        try:pid=psutil.Process(pid).ppid()
        except (psutil.Error,OSError):return False
        if pid==parent['pid']:
            current=legacy.process_info(pid)
            return current is not None and same(current,parent)
        if pid<=0:return False
    return False

def _admit_once(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--caller-pid',type=int,required=True)
    args=p.parse_args(argv)
    context=json.loads(os.environ['BEMARKDOWN_STAGE_ADMISSION_CONTEXT'])
    event={'stage':args.stage,'epoch':time.time(),'helper_pid':os.getpid(),'status':'REFUSED'}
    try:
        import e2e_supervise as legacy
        from native_e2e_supervise import ResourcePolicy
        # Reuse the proven Job handle duplication, complete membership and
        # additional-controller identity observations; never run its old main.
        from stage_admission_check import observe_family
        if sha(__file__)!=context['runner_sha256']:raise RuntimeError('NATIVE_ADMISSION_RUNNER_CHANGED')
        policy=Path(__file__).with_name('native_e2e_supervise.py')
        if sha(policy)!=context['policy_sha256']:raise RuntimeError('NATIVE_ADMISSION_POLICY_CHANGED')
        manifest_path=Path(context['manifest_path'])
        if sha(manifest_path)!=context['manifest_sha256']:raise RuntimeError('NATIVE_ADMISSION_MANIFEST_CHANGED')
        manifest=read(manifest_path)
        for binding in manifest['bindings']:
            if sha(binding['path'])!=binding['sha256']:raise RuntimeError('NATIVE_ADMISSION_PAYLOAD_CHANGED')
        spec=manifest['native_forecast'];forecast_path=Path(spec['path'])
        if sha(forecast_path)!=spec['sha256']:raise RuntimeError('NATIVE_ACCEPTED_FORECAST_CHANGED')
        forecasts=read(forecast_path)
        if forecasts.get('schema')!='bemarkdown-owned-stage-forecast-v1' or args.stage not in forecasts['stages']:
            raise RuntimeError('NATIVE_FORECAST_STAGE_UNQUALIFIED:'+args.stage)
        core=Path(manifest['native_core_root'])/'bemarkdown'
        for relative,digest in forecasts['source_bindings'].items():
            allowed=[digest] if isinstance(digest,str) else digest
            if sha(core/relative) not in allowed:raise RuntimeError('NATIVE_FORECAST_SOURCE_CHANGED:'+relative)
        rows,controller=observe_family(context)
        callers=[r for r in rows if r['pid']==args.caller_pid and not r.get('additional_shared_controller')]
        if len(callers)!=1 or args.caller_pid==os.getpid():raise RuntimeError('NATIVE_CALLER_NOT_IN_OWNED_JOB')
        caller=callers[0];host=legacy.sample();gpu=legacy.GPU().sample();policy=ResourcePolicy();policy.check(host,gpu)
        family=sum(r['private_commit_gib'] for r in rows)+controller['private_commit_gib']
        working=sum(r['working_gib'] for r in rows)+controller['working_gib']
        if family>current_memory_policy()['project_ram_cap_gib'] or working>current_memory_policy()['project_ram_cap_gib']:raise MemoryError('NATIVE_CURRENT_COMPLETE_FAMILY_CAP')
        event.update(host=host,gpu=gpu,controller=controller,job_members=rows,caller=caller,
            family_private_commit_gib=family,forecast_manifest_sha256=spec['sha256'])
        lease_path=Path(manifest['lease_path'])
        lease=read(lease_path) if lease_path.exists() else {}
        if delegated_lease(args.stage,caller,lease,descendant):
            # The accepted serial parent's pre-create admission already reserved
            # this owned child stage. Repeating the complete load forecast after
            # its imports would double-count the same load; current checks stay.
            event.update(status='ADMITTED',admission='OWNED_CHILD_CURRENT_RECHECK',parent_lease=lease)
        else:
            if gpu['free']*2**20<current_memory_policy()['gpu_startup_free_floor_bytes']:raise MemoryError('NATIVE_STAGE_STARTUP_GPU_FREE_FLOOR')
            bound=forecasts['stages'][args.stage]
            helpers=[r for r in rows if r['pid']==os.getpid()]
            if len(helpers)!=1:raise RuntimeError('NATIVE_HELPER_IDENTITY_UNOBSERVABLE')
            # This process exits before subprocess.run returns to OwnedStageProcess;
            # its private commit is not concurrent with the future model load.
            baseline=family-helpers[0]['private_commit_gib']
            projected=policy.check_projected(host,gpu,family_private_gib=baseline,
                stage_increment={'private_gib':bound['extra_private_bytes']/2**30,
                    'working_gib':bound['extra_physical_bytes']/2**30,'gpu_mib':bound['extra_gpu_mib']},
                margin={'private_gib':.03,'working_gib':0,'gpu_mib':0})
            lease={'stage':args.stage,'parent':{k:caller[k] for k in ('pid','created_ticks')},
                'epoch':time.time(),'forecast_sha256':spec['sha256']}
            temporary=lease_path.with_suffix('.tmp');temporary.write_text(json.dumps(lease)+'\n',encoding='utf-8')
            _host_ram['replace_with_retry'](temporary,lease_path)
            event.update(status='ADMITTED',admission='PARENT_BEFORE_OWNED_CREATE',projected=projected,
                future_baseline_excludes_exiting_helper=helpers[0],bound=bound,lease=lease)
    except Exception as exc:event.update(error_type=type(exc).__name__,reason=str(exc),decision=getattr(exc,'decision',None))
    journal=Path(context['journal_path']);journal.parent.mkdir(parents=True,exist_ok=True)
    if journal.exists() and journal.stat().st_size>=8*2**20:journal.replace(journal.with_suffix('.jsonl.1'))
    with journal.open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    return event

def main(argv=None):
    policy=current_memory_policy();deadline=time.monotonic()+policy['stage_wait_timeout_seconds']
    while True:
        event=_admit_once(argv)
        if event['status']=='ADMITTED':return 0
        reason=event.get('reason','')
        wait=event.get('decision')=='WAIT' or reason.startswith('WAIT:')
        if not wait:return 3
        now=time.monotonic()
        if now>=deadline:
            context=json.loads(os.environ['BEMARKDOWN_STAGE_ADMISSION_CONTEXT'])
            final={'event':'stage_wait_timeout','stage':event['stage'],'epoch':time.time(),
                'decision':'WAIT_TIMEOUT','reason':'STAGE_RESOURCE_WAIT_TIMEOUT:'+reason,
                'policy_fingerprint':policy['policy_fingerprint']}
            with Path(context['journal_path']).open('a',encoding='utf-8') as stream:stream.write(json.dumps(final)+'\n')
            return 3
        context=json.loads(os.environ['BEMARKDOWN_STAGE_ADMISSION_CONTEXT'])
        progress={'event':'stage_wait','stage':event['stage'],'epoch':time.time(),'reason':reason,
            'remaining_seconds':deadline-now,'policy_fingerprint':policy['policy_fingerprint']}
        with Path(context['journal_path']).open('a',encoding='utf-8') as stream:stream.write(json.dumps(progress)+'\n')
        time.sleep(min(policy['stage_wait_poll_seconds'],deadline-now))

if __name__=='__main__':raise SystemExit(main())
