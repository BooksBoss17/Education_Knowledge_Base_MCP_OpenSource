"""One-shot Windows job-capped load probe. No gold or dataset access."""

import ctypes as C, ctypes.wintypes as W, _winapi, subprocess, sys, pathlib, os, msvcrt

ROOT=pathlib.Path(__file__).resolve().parent

sys.path.insert(0,str(ROOT.parent))

from memory_watchdog_v34 import process_info, save

K=C.WinDLL('kernel32',use_last_error=True)

SIZE=C.c_size_t

class BASIC(C.Structure):
    _fields_=[('PerProcessUserTimeLimit',C.c_longlong),('PerJobUserTimeLimit',C.c_longlong),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',SIZE),('MaximumWorkingSetSize',SIZE),('ActiveProcessLimit',W.DWORD),('Affinity',SIZE),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]

class IO(C.Structure):
    _fields_=[(n,C.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]

class EXT(C.Structure):
    _fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',SIZE),('JobMemoryLimit',SIZE),('PeakProcessMemoryUsed',SIZE),('PeakJobMemoryUsed',SIZE)]

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
