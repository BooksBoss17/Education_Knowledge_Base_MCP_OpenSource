"""One-shot Windows job-capped load probe. No gold or dataset access."""
import ctypes as C, ctypes.wintypes as W, _winapi, subprocess, sys, pathlib, os, json, time, msvcrt, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
# All dependencies are packaged beside this module.
from memory_guard_v33 import sample, check, LIMITS
from memory_watchdog_v34 import Watchdog, process_info, save
K=C.WinDLL('kernel32',use_last_error=True)
SIZE=C.c_size_t
class BASIC(C.Structure):
    _fields_=[('PerProcessUserTimeLimit',C.c_longlong),('PerJobUserTimeLimit',C.c_longlong),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',SIZE),('MaximumWorkingSetSize',SIZE),('ActiveProcessLimit',W.DWORD),('Affinity',SIZE),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
class IO(C.Structure):
    _fields_=[(n,C.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
class EXT(C.Structure):
    _fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',SIZE),('JobMemoryLimit',SIZE),('PeakProcessMemoryUsed',SIZE),('PeakJobMemoryUsed',SIZE)]
for name,args,ret in [('CreateJobObjectW',[W.LPVOID,W.LPCWSTR],W.HANDLE),('SetInformationJobObject',[W.HANDLE,C.c_int,W.LPVOID,W.DWORD],W.BOOL),('QueryInformationJobObject',[W.HANDLE,C.c_int,W.LPVOID,W.DWORD,W.LPVOID],W.BOOL),('AssignProcessToJobObject',[W.HANDLE,W.HANDLE],W.BOOL),('ResumeThread',[W.HANDLE],W.DWORD),('CloseHandle',[W.HANDLE],W.BOOL),('TerminateProcess',[W.HANDLE,W.UINT],W.BOOL)]:
    f=getattr(K,name);f.argtypes=args;f.restype=ret
def must(ok):
    if not ok:raise C.WinError(C.get_last_error())
class Job:
    def __init__(self,process_cap,job_cap):
        self.handle=K.CreateJobObjectW(None,None);must(self.handle);self.children=[];self.files=[]
        e=EXT();e.BasicLimitInformation.LimitFlags=0x100|0x200|0x2000;e.ProcessMemoryLimit=process_cap;e.JobMemoryLimit=job_cap
        try:
            must(K.SetInformationJobObject(self.handle,9,C.byref(e),C.sizeof(e)))
            q=self.stats();assert q['process_cap']==process_cap and q['job_cap']==job_cap and q['flags']==e.BasicLimitInformation.LimitFlags
        except BaseException:self.close();raise
    def stats(self):
        e=EXT();must(K.QueryInformationJobObject(self.handle,9,C.byref(e),C.sizeof(e),None))
        return dict(process_cap=e.ProcessMemoryLimit,job_cap=e.JobMemoryLimit,flags=e.BasicLimitInformation.LimitFlags,peak_process=e.PeakProcessMemoryUsed,peak_job=e.PeakJobMemoryUsed)
    def launch(self,args,name,env=None):
        out=open(ROOT/(name+'.log'),'wb');inp=open(os.devnull,'rb');self.files.extend([out,inp])
        for f in (out,inp):os.set_handle_inheritable(msvcrt.get_osfhandle(f.fileno()),True)
        si=subprocess.STARTUPINFO();si.dwFlags=subprocess.STARTF_USESTDHANDLES;si.hStdInput=msvcrt.get_osfhandle(inp.fileno());si.hStdOutput=si.hStdError=msvcrt.get_osfhandle(out.fileno())
        hp,ht,pid,tid=_winapi.CreateProcess(str(args[0]),subprocess.list2cmdline(args),None,None,True,0x4|0x08000000,env,str(ROOT),si)
        self.children.append((hp,pid))
        try:
            must(K.AssignProcessToJobObject(self.handle,hp))
            identity=process_info(pid)
            assert identity and pathlib.Path(identity['exe']).resolve()==pathlib.Path(args[0]).resolve()
            save(ROOT/(name+'-identity.json'),identity)
            assert K.ResumeThread(ht)!=0xffffffff
            return hp,identity
        except BaseException:K.TerminateProcess(hp,77);raise
        finally:_winapi.CloseHandle(ht)
    def close(self):
        if self.handle:K.CloseHandle(self.handle);self.handle=None
        for hp,pid in self.children:
            _winapi.WaitForSingleObject(hp,5000);_winapi.CloseHandle(hp)
        self.children=[]
        for f in self.files:f.close()
        self.files=[]
def controls():
    child=ROOT/'cpu_control.py'
    child.write_text("import ctypes as c,time,sys,json\nk=c.WinDLL('kernel32',use_last_error=True)\nk.VirtualAlloc.argtypes=[c.c_void_p,c.c_size_t,c.c_ulong,c.c_ulong];k.VirtualAlloc.restype=c.c_void_p\na=[]\nfor i in range(48):\n p=k.VirtualAlloc(None,1024*1024,0x3000,4)\n if not p: print(json.dumps({'denied_at_mib':i,'error':c.get_last_error()}),flush=True);time.sleep(2);sys.exit(42)\n a.append(p)\nprint('UNEXPECTED_ALLOCATION_SUCCESS',flush=True)\nsys.exit(1)\n",encoding='utf-8')
    results=[]
    for name,pc,jc in [('process-cap',32,64),('job-cap',64,32)]:
        j=Job(pc*2**20,jc*2**20)
        try:
            hp,ident=j.launch([sys.executable,str(child)],name)
            assert _winapi.WaitForSingleObject(hp,10000)==0
            code=_winapi.GetExitCodeProcess(hp);stats=j.stats()
            save(ROOT/(name+'-observed.json'),dict(code=code,stats=stats))
            assert code==42 and stats['peak_job']<=jc*2**20 and stats['peak_process']<=pc*2**20
            results.append(dict(name=name,exit_code=code,stats=stats,identity=ident))
        finally:j.close()
    save(ROOT/'cpu-controls.json',dict(passed=True,results=results))
def probe():
    controls()
    limits={**LIMITS,'measured_model_commit_gib':4.,'estimated_model_working_gib':6.5}
    s=sample();decision=check(s,'startup',limits);save(ROOT/'preflight.json',dict(snapshot=s,decision=decision,limits=limits))
    if not decision['allowed']:
        save(ROOT/'result.json',dict(status='NOT_LOADED_RESOURCE_GUARD',preflight=decision));return
    marker=ROOT/'MODEL_ATTEMPTED'
    with marker.open('x') as f:f.write(str(time.time()))
    runtime=json.loads((ROOT.parent/'v30-runtime/runtime.json').read_text());args=runtime['args']
    for key,value in [('--gpu-layers','8'),('--batch-size','32'),('--ubatch-size','16'),('--port','18236')]:args[args.index(key)+1]=value
    # Preserve mmap and CPU-host tensors; --no-host explicitly disables those.
    args.remove('--no-host')
    env={**os.environ,**runtime['task_environment']};j=Job(4*2**30,4*2**30);watch=None;start=time.time();result=dict(args=args,status='STARTING')
    try:
        # Recheck immediately before suspended creation.
        assert check(sample(),'startup',limits)['allowed']
        hp,identity=j.launch(args,'model',env);result['identity']=identity
        watch=Watchdog(ROOT/'watchdog',owned=[identity],limits=limits).start()
        while time.time()-start<120:
            watch.assert_safe()
            code=_winapi.GetExitCodeProcess(hp)
            if code!=259:result.update(status='NATIVE_EXIT',native_exit_code=code);break
            try:
                with urllib.request.urlopen('http://127.0.0.1:18236/health',timeout=1) as r:
                    if r.status==200:result.update(status='HEALTHY_LOAD_ONLY',health=json.load(r));break
            except Exception:pass
            time.sleep(.5)
        else:result['status']='LOAD_TIMEOUT'
    except BaseException as e:result.update(status='STOPPED_ERROR',error=repr(e))
    finally:
        result['elapsed_seconds']=time.time()-start;result['job_stats']=j.stats();result['host_final_before_unload']=sample()
        if watch:watch.close()
        j.close();result['unloaded']=not process_info(result.get('identity',{}).get('pid',0));save(ROOT/'result.json',result)
if __name__=='__main__':
    try:probe()
    except BaseException as e:
        save(ROOT/'result.json',dict(status='NOT_LOADED_CAP_CONTROL_FAILED',error=repr(e),host=sample(),model_attempted=(ROOT/'MODEL_ATTEMPTED').exists()))
        raise
    print((ROOT/'result.json').read_text())
