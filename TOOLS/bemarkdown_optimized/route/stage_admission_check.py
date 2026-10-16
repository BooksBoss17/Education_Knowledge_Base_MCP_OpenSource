"""One fresh, read-only observation of the supervising Windows Job and GPU."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import _winapi
import e2e_supervise as legacy
from e2e_supervise_v3 import ResourcePolicy

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def journal_events(path):
    path=Path(path)
    if not path.exists():return
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def observe_family(context):
    controller=legacy.process_info(context['controller_pid'])
    if controller is None or controller['created_ticks']!=context['controller_created_ticks']:
        raise RuntimeError('STAGE_CONTROLLER_IDENTITY_CHANGED')
    process=_winapi.OpenProcess(0x0040|0x1000,False,context['controller_pid'])
    handle=None
    try:
        handle=_winapi.DuplicateHandle(process,context['job_handle'],_winapi.GetCurrentProcess(),0,False,2)
        rows=[]
        for pid in legacy.complete_job_members(handle,legacy.F.probe.K):
            try:pin=_winapi.OpenProcess(0x1000,False,pid)
            except OSError as exc:
                if exc.winerror==87:continue
                raise
            try:
                if _winapi.GetExitCodeProcess(pin)!=259:continue
                row=legacy.process_info(pid)
                if row is None:
                    if _winapi.GetExitCodeProcess(pin)!=259:continue
                    raise RuntimeError('STAGE_LIVE_JOB_MEMBER_UNOBSERVABLE')
                rows.append(row)
            finally:_winapi.CloseHandle(pin)
        if os.getpid() not in {r['pid'] for r in rows}:
            raise RuntimeError('STAGE_HELPER_NOT_IN_SUPERVISED_JOB')
        if controller['pid'] in {r['pid'] for r in rows}:
            raise RuntimeError('STAGE_CONTROLLER_UNEXPECTEDLY_IN_JOB')
        charged={r['pid'] for r in rows}|{controller['pid']}
        for expected in context.get('additional_controller_identities',[]):
            if expected['pid'] in charged:continue
            additional=legacy.process_info(expected['pid'])
            if (additional is None or additional['created_ticks']!=expected['created_ticks']
                    or Path(additional['exe']).resolve()!=Path(expected['exe']).resolve()):
                raise RuntimeError('STAGE_ADDITIONAL_CONTROLLER_IDENTITY_CHANGED')
            additional['additional_shared_controller']=True
            rows.append(additional);charged.add(additional['pid'])
        return rows,controller
    finally:
        if handle is not None:_winapi.CloseHandle(handle)
        _winapi.CloseHandle(process)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--caller-pid',type=int,required=True);a=p.parse_args()
    context=json.loads(os.environ['BEMARKDOWN_STAGE_ADMISSION_CONTEXT'])
    event={'stage':a.stage,'epoch':time.time(),'helper_pid':os.getpid(),'status':'REFUSED'}
    event['supervision_scope']={k:context[k] for k in ('controller_pid','controller_created_ticks','job_handle','manifest_sha256','journal_path')}
    try:
        if sha(__file__)!=context['runner_sha256']:raise RuntimeError('STAGE_RUNNER_CHANGED')
        if sha(Path(__file__).with_name('e2e_supervise_v3.py'))!=context['policy_sha256']:
            raise RuntimeError('STAGE_POLICY_CHANGED')
        if sha(context['manifest_path'])!=context['manifest_sha256']:raise RuntimeError('STAGE_MANIFEST_CHANGED')
        manifest=json.loads(Path(context['manifest_path']).read_text(encoding='utf-8-sig'))
        if manifest['version']!='measured-stage-admission-v1':raise RuntimeError('STAGE_MANIFEST_VERSION')
        has_groups=bool(manifest.get('process_stage_groups'))
        forecast=None if has_groups else manifest['stages'][a.stage]
        rows,controller=observe_family(context)
        callers=[r for r in rows if r['pid']==a.caller_pid and not r.get('additional_shared_controller')]
        if len(callers)!=1 or a.caller_pid==os.getpid():raise RuntimeError('STAGE_CALLER_NOT_IN_SUPERVISED_JOB')
        family_private=sum(r['private_commit_gib'] for r in rows)+controller['private_commit_gib']
        host=legacy.sample();gpu=legacy.GPU().sample()
        event.update(host=host,gpu=gpu,job_members=rows,controller=controller,caller=callers[0],
            family_private_commit_gib=family_private,
            manifest_sha256=context['manifest_sha256'])
        if has_groups:
            module=Path(__file__).with_name('stage_admission_groups.py')
            if sha(module)!=manifest['process_group_resolver_sha256']:raise RuntimeError('STAGE_GROUP_RESOLVER_CHANGED')
            from stage_admission_groups import resolve_forecast
            forecast,group=resolve_forecast(manifest,a.stage,callers[0],context,journal_events(context['journal_path']))
            if group is not None:event['group_admission']=group
        if not forecast.get('evidence'):raise RuntimeError('STAGE_MEASUREMENT_EVIDENCE_REQUIRED')
        event['forecast']=forecast
        projected=ResourcePolicy().check_projected(host,gpu,family_private_gib=family_private,
            stage_increment=forecast['increment'],margin=forecast['margin'])
        event.update(status='ADMITTED',projected=projected)
    except Exception as exc:
        event.update(error_type=type(exc).__name__,reason=str(exc))
    journal=Path(context['journal_path']);journal.parent.mkdir(parents=True,exist_ok=True)
    with journal.open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    return 0 if event['status']=='ADMITTED' else 3

if __name__=='__main__':raise SystemExit(main())
