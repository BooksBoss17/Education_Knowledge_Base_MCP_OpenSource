"""Owned Job dynamic process-family monitoring. CPU tests only in this revision."""

import ctypes as C,ctypes.wintypes as W,_winapi,pathlib,os,subprocess,msvcrt,time,threading

import probe

from memory_watchdog_v34 import process_info,save

from memory_guard_v33 import sample

class PIDLIST(C.Structure):
    _fields_=[('assigned',W.DWORD),('listed',W.DWORD),('pids',C.c_size_t*128)]

class Family:
    def __init__(self,job,out,budget_gib=5,interval=.05):
        self.job=job;self.out=pathlib.Path(out);self.budget=budget_gib;self.interval=interval
        self.identities={};self.samples=[];self.stop=threading.Event();self.failed=None;self.thread=None;self.runner=process_info(os.getpid())
        if not self.runner:raise RuntimeError('No runner identity')
    def members(self):
        p=PIDLIST();probe.must(probe.K.QueryInformationJobObject(self.job.handle,3,C.byref(p),C.sizeof(p),None))
        if p.assigned!=p.listed:raise RuntimeError('Incomplete Job membership')
        return list(p.pids[:p.listed])
    @staticmethod
    def same_identity(a,b):
        return a['pid']==b['pid'] and a['created_ticks']==b['created_ticks'] and pathlib.Path(a['exe']).resolve()==pathlib.Path(b['exe']).resolve()
    def tick(self):
        pids=self.members();observed=[]
        for pid in pids:
            info=process_info(pid)
            if info is None:
                if pid not in self.members():continue # exited between enumeration and query
                raise RuntimeError('Cannot inspect owned Job member')
            old=self.identities.get(pid)
            if old and not self.same_identity(old,info):raise RuntimeError('PID identity changed; close owned Job only')
            # A queried PID must still belong to our Job before accepting identity.
            if pid not in self.members():continue
            self.identities[pid]=info;observed.append(info)
        runner=process_info(os.getpid())
        if not runner or not self.same_identity(self.runner,runner):raise RuntimeError('Runner identity mismatch')
        processes=observed+[runner]
        s=sample();s['project_working_gib']=sum(x['working_gib'] for x in processes);s['project_private_commit_gib']=sum(x['private_commit_gib'] for x in processes)
        row={'host':s,'job_members':observed,'runner':runner};self.samples.append(row)
        save(self.out,{'identities':list(self.identities.values()),'samples':self.samples})
        if s['project_working_gib']>self.budget or s['project_private_commit_gib']>self.budget:raise MemoryError('Project memory budget exceeded')
        if s['commit_headroom_gib']<3 or s['PhysicalAvailable_gib']<2:raise MemoryError('Global memory floor crossed')
        return row
    def fail_closed(self,error):
        self.failed=repr(error);self.stop.set()
        # No PID-targeted termination: kernel-owned Job containment cannot kill a reused unrelated PID.
        if self.job.handle:probe.K.CloseHandle(self.job.handle);self.job.handle=None
    def loop(self):
        while not self.stop.wait(self.interval):
            try:self.tick()
            except BaseException as e:self.fail_closed(e);break
    def start_before_resume(self):
        try:self.tick()
        except BaseException as e:self.fail_closed(e);raise
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(5)
        if self.thread and self.thread.is_alive():raise RuntimeError('Family monitor did not stop')

def suspended(job,args,name):
    out=open(probe.ROOT/(name+'.log'),'wb');inp=open(os.devnull,'rb');job.files.extend([out,inp])
    for f in (out,inp):os.set_handle_inheritable(msvcrt.get_osfhandle(f.fileno()),True)
    si=subprocess.STARTUPINFO();si.dwFlags=subprocess.STARTF_USESTDHANDLES;si.hStdInput=msvcrt.get_osfhandle(inp.fileno());si.hStdOutput=si.hStdError=msvcrt.get_osfhandle(out.fileno())
    hp,ht,pid,tid=_winapi.CreateProcess(str(args[0]),subprocess.list2cmdline(args),None,None,True,0x4|0x08000000,None,str(probe.ROOT),si)
    job.children.append((hp,pid))
    try:
        probe.must(probe.K.AssignProcessToJobObject(job.handle,hp));ident=process_info(pid)
        if not ident or pathlib.Path(ident['exe']).resolve()!=pathlib.Path(args[0]).resolve():raise RuntimeError('Suspended identity mismatch')
        return hp,ht,ident
    except BaseException:
        probe.K.TerminateProcess(hp,77);_winapi.CloseHandle(ht);raise
