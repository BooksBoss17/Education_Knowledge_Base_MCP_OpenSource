"""Track host/process peaks and terminate only identity-bound task processes."""
import ctypes,json,pathlib,threading,time
from ctypes import wintypes
from memory_guard_v33 import sample,check,LIMITS
K=ctypes.windll.kernel32;P=ctypes.windll.psapi
K.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
K.OpenProcess.restype=wintypes.HANDLE
K.CloseHandle.argtypes=[wintypes.HANDLE]
class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
K.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
K.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
K.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
P.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
def process_info(pid):
    handle=K.OpenProcess(0x0410,False,pid)
    if not handle:return None
    try:
        size=wintypes.DWORD(32768);buf=ctypes.create_unicode_buffer(size.value)
        if not K.QueryFullProcessImageNameW(handle,0,buf,ctypes.byref(size)):return None
        creation=wintypes.FILETIME();exit_time=wintypes.FILETIME();kernel=wintypes.FILETIME();user=wintypes.FILETIME()
        if not K.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(kernel),ctypes.byref(user)):return None
        c=Counters();c.cb=ctypes.sizeof(c)
        if not P.GetProcessMemoryInfo(handle,ctypes.byref(c),c.cb):return None
        ticks=(creation.dwHighDateTime<<32)|creation.dwLowDateTime
        return {'pid':pid,'exe':buf.value,'created_ticks':ticks,
                'working_gib':c.WorkingSetSize/2**30,'peak_working_gib':c.PeakWorkingSetSize/2**30,
                'private_commit_gib':c.PrivateUsage/2**30,'peak_pagefile_gib':c.PeakPagefileUsage/2**30}
    finally:K.CloseHandle(handle)
def register_process(pid,expected_exe=None,launched_at=None):
    info=process_info(pid)
    if info is None:raise RuntimeError('Task process is not queryable')
    if expected_exe and pathlib.Path(info['exe']).resolve()!=pathlib.Path(expected_exe).resolve():raise RuntimeError('Task executable identity mismatch')
    epoch=info['created_ticks']/10**7-11644473600
    if launched_at is not None and abs(epoch-launched_at)>3:raise RuntimeError('Task creation time mismatch')
    return {k:info[k] for k in ('pid','exe','created_ticks')}
def terminate_owned(identity):
    info=process_info(identity['pid'])
    if info is None:return 'ALREADY_EXITED'
    if any(info[k]!=identity[k] for k in ('pid','exe','created_ticks')):return 'IDENTITY_CHANGED_NOT_STOPPED'
    handle=K.OpenProcess(0x0001|0x1000,False,identity['pid'])
    if not handle:return 'TERMINATE_ACCESS_DENIED'
    try:
        # Recheck creation time on the termination handle to close PID-reuse race.
        a=wintypes.FILETIME();b=wintypes.FILETIME();c=wintypes.FILETIME();d=wintypes.FILETIME()
        if not K.GetProcessTimes(handle,ctypes.byref(a),ctypes.byref(b),ctypes.byref(c),ctypes.byref(d)):return 'IDENTITY_QUERY_FAILED'
        if ((a.dwHighDateTime<<32)|a.dwLowDateTime)!=identity['created_ticks']:return 'IDENTITY_CHANGED_NOT_STOPPED'
        return 'STOPPED' if K.TerminateProcess(handle,66) else 'TERMINATE_FAILED'
    finally:K.CloseHandle(handle)
def save(path,value):
    path=pathlib.Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')
    for attempt in range(5):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1*(attempt+1))
class Watchdog:
    def __init__(self,output,owned=(),tracked=(),limits=LIMITS,sampler=sample):
        self.output=pathlib.Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.owned=list(owned);self.limits=limits;self.sampler=sampler
        self.tracked=list(tracked) or self.owned
        import os
        current=register_process(os.getpid())
        self.tracked=list({x['pid']:x for x in [*self.tracked,current]}.values())
        self.event=threading.Event();self.breached=threading.Event();self.thread=None
        self.peaks={};self.baseline=None
    def tick(self):
        try:
            s=self.sampler();decision=check(s,'running',self.limits)
        except Exception as error:
            s={'timestamp':time.time()};decision={'allowed':False,'problems':['MEMORY_MONITOR_FAILED'],'error_type':type(error).__name__}
        if self.baseline is None:self.baseline=s
        procs=[process_info(x['pid']) for x in self.tracked];procs=[x for x in procs if x]
        s['project_working_gib']=sum(p['working_gib'] for p in procs)
        s['project_private_commit_gib']=sum(p['private_commit_gib'] for p in procs)
        if 'commit_headroom_gib' in s:decision=check(s,'running',self.limits)
        for key in ('project_working_gib','project_private_commit_gib'):
            self.peaks[key]=max(self.peaks.get(key,0),s[key])
        for key in ('CommitTotal_gib','KernelNonpaged_gib'):
            if key in s:self.peaks[key]=max(self.peaks.get(key,0),s[key])
        if 'PhysicalTotal_gib' in s:
            used=s['PhysicalTotal_gib']-s['PhysicalAvailable_gib'];self.peaks['physical_used_gib']=max(self.peaks.get('physical_used_gib',0),used)
        for p in procs:
            for key in ('working_gib','private_commit_gib','peak_working_gib','peak_pagefile_gib'):
                name=str(p['pid'])+':'+key;self.peaks[name]=max(self.peaks.get(name,0),p[key])
        with (self.output/'samples.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'host':s,'processes':procs,'allowed':decision['allowed']})+'\n')
        save(self.output/'peaks.json',{'baseline':self.baseline,'peaks':self.peaks,'latest':s,'limits':self.limits})
        if not decision['allowed']:
            self.breached.set()
            # Durable signal precedes termination. Runner keeps separate per-item
            # checkpoints; no model response or verified result is removed.
            stop={'status':'STOPPED_FOR_MEMORY_SAFETY','decision':decision,'owned':self.owned,'timestamp':time.time()}
            save(self.output/'STOP_FOR_MEMORY.json',stop)
            stop['termination']=[{'pid':x['pid'],'result':terminate_owned(x)} for x in self.owned]
            save(self.output/'STOP_FOR_MEMORY.json',stop)
            return False
        return True
    def _run(self):
        while not self.event.is_set():
            try:
                if not self.tick():break
            except Exception as error:
                self.breached.set()
                try:save(self.output/'STOP_FOR_MEMORY.json',{'status':'STOPPED_MONITOR_FAILURE','error_type':type(error).__name__})
                finally:
                    for identity in self.owned:terminate_owned(identity)
                break
            self.event.wait(self.limits['poll_seconds'])
    def start(self):
        self.thread=threading.Thread(target=self._run,name='task-memory-watchdog',daemon=True);self.thread.start();return self
    def close(self):
        self.event.set()
        if self.thread:self.thread.join(timeout=5)
    def assert_safe(self):
        if self.breached.is_set():raise MemoryError('Task stopped for RAM/commit safety; preserve checkpoint')
