"""Bound the entire conversion/agent/classifier process family before resume."""
import argparse,ctypes,json,os,sys,time,_winapi
from pathlib import Path
ROOT=Path(__file__).resolve().parent
import job_family_v41 as F
from job_membership import complete_job_members
from memory_guard_v33 import sample
from memory_watchdog_v34 import process_info
import runpy as _ram_runpy
from pathlib import Path as _RamPath
_host_ram = _ram_runpy.run_path(str(_RamPath(__file__).resolve().parents[1]/'core/bemarkdown/host_memory_policy.py'))
current_memory_policy = _host_ram['current_memory_policy']

class LiveFamily(F.Family):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.pinned={};self.retired=[]
    def members(self):
        result=[]
        for pid in complete_job_members(self.job.handle,F.probe.K):
            try:handle=_winapi.OpenProcess(0x1000,False,pid)
            except OSError as exc:
                if exc.winerror==87:continue
                raise
            try:
                if _winapi.GetExitCodeProcess(handle)!=259:continue
                info=process_info(pid)
                if info is None:
                    if _winapi.GetExitCodeProcess(handle)!=259:continue
                    raise RuntimeError('Live owned process identity unavailable')
                old=self.identities.get(pid)
                if old and not self.same_identity(old,info):
                    previous=self.pinned.get(pid)
                    if previous is None or _winapi.GetExitCodeProcess(previous)==259:
                        raise RuntimeError('Live PID identity changed without confirmed previous exit')
                    self.retired.append(old);self.identities.pop(pid,None)
                    _winapi.CloseHandle(previous);self.pinned.pop(pid,None)
                if pid not in self.pinned:self.pinned[pid]=handle;handle=None
                result.append(pid)
            finally:
                if handle is not None:_winapi.CloseHandle(handle)
        return result
    def close(self):
        try:super().close()
        finally:
            for handle in self.pinned.values():_winapi.CloseHandle(handle)
            self.pinned.clear()

class GPU:
 def __init__(self):self.sampler=_host_ram['DeviceSampler']()
 def sample(self):
  row=self.sampler.sample()
  row.update(total=row['gpu_total_bytes']/2**20,free=row['gpu_free_bytes']/2**20,used=row['gpu_used_bytes']/2**20)
  return row
 def __del__(self):
  if hasattr(self,'sampler'):self.sampler.close()

def write(p,v):p.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--python',required=True);ap.add_argument('--worker',required=True);ap.add_argument('--timeout',type=int,default=7200)
    ap.add_argument('--gpu-used-limit-mib',type=int,default=None)
    a,extra=ap.parse_known_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    ctrl=process_info(os.getpid());host=sample();gpu=GPU();before=gpu.sample();ram=current_memory_policy()
    if a.gpu_used_limit_mib is None:a.gpu_used_limit_mib=ram['gpu_whole_card_cap_bytes']/2**20
    if not 0<a.gpu_used_limit_mib<=ram['gpu_whole_card_cap_bytes']/2**20:raise ValueError('GPU_OVERRIDE_CANNOT_WEAKEN_POLICY')
    try:
        _host_ram['check_current'](ram,_host_ram['observation_from_legacy'](host,before,policy=ram))
        if before['free']*2**20<ram['gpu_startup_free_floor_bytes']:raise MemoryError('LEGACY_GPU_STARTUP_FREE_FLOOR')
    except (MemoryError,ValueError) as exc:
        write(out/'result.json',{'status':'ADMISSION_REFUSED','host':host,'gpu':before,'policy':ram,'reason':str(exc)});return 2
    reserve=float(os.environ.get('BEMARKDOWN_ROUTE_CALLER_PRIVATE_RESERVE_GIB','0'))
    if not 0<=reserve<ram["project_ram_cap_gib"]:raise ValueError('ROUTE_CALLER_RESERVE_INVALID')
    cap=int((ram['project_ram_cap_gib']-ctrl['private_commit_gib']-reserve-.03)*2**30)//4096*4096
    if cap<=0:raise MemoryError('ROUTE_CONTROLLER_BUDGET_EXHAUSTED')
    job=F.Job(cap,cap);F.probe.ROOT=out;hp=ht=None;started=time.monotonic();result={'status':'STARTING','gpu_baseline':before,'host_baseline':host}
    family=LiveFamily(job,out/'memory',interval=.1,recent_limit=32)
    try:
        hp,ht,identity=F.suspended(job,[a.python,'-X','utf8',str(Path(a.worker).resolve()),*extra],'e2e-worker')
        write(out/'process.json',identity);family.start_before_resume();family.resume(ht);_winapi.CloseHandle(ht);ht=None
        peak=before['used'];minimum=before['free']
        with (out/'gpu.jsonl').open('w',encoding='utf-8',buffering=1) as f:
            while _winapi.GetExitCodeProcess(hp)==259:
                family.assert_safe();row=gpu.sample();peak=max(peak,row['used']);minimum=min(minimum,row['free'])
                f.write(json.dumps({'epoch':time.time(),**row})+'\n')
                if row['free']<ram['gpu_reserve_bytes']/2**20:raise MemoryError('GPU_FREE_RESERVE')
                if row['used']>a.gpu_used_limit_mib:raise MemoryError('GPU_WHOLE_CARD_USED_LIMIT')
                if time.monotonic()-started>a.timeout:raise TimeoutError('E2E_TIMEOUT')
                time.sleep(.8)
        code=_winapi.GetExitCodeProcess(hp)
        family.assert_safe();result.update(status='COMPLETED' if code==0 else 'WORKER_FAILED',exit_code=code,gpu_whole_card_peak_mib=peak,gpu_minimum_free_mib=minimum)
    except BaseException as exc:
        result.update(status='STOPPED_WITH_EVIDENCE',error_type=type(exc).__name__,error=str(exc));family.fail_closed(exc)
    finally:
        if ht is not None:_winapi.CloseHandle(ht)
        if hp is not None:
            if job.handle and _winapi.GetExitCodeProcess(hp)==259:family.fail_closed(RuntimeError('OWNED_WORKER_STOP'))
            _winapi.WaitForSingleObject(hp,5000);result['native_exit_code']=_winapi.GetExitCodeProcess(hp)
        if job.handle:result['native_job_stats']=job.stats()
        family.close();job.close()
        result.update(elapsed_seconds=time.monotonic()-started,memory=json.loads((out/'memory/summary.json').read_text(encoding='utf-8')),gpu_after=gpu.sample(),retired_owned_process_identities=family.retired)
        write(out/'result.json',result)
    return 0 if result['status']=='COMPLETED' else 2

if __name__=='__main__':sys.exit(main())
