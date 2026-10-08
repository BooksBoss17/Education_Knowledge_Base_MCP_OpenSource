"""Bounded RAM, append-only samples for an identity-checked owned Windows Job.

No model startup or dataset access. All Job members plus the external controller
are charged to BOTH project WS and private commit. Put evaluation drivers in the
same Job; the one external controller is included separately without double count.
"""

import collections, json, os, pathlib, sys, threading, time

HERE=pathlib.Path(__file__).resolve().parent

sys.path.insert(0,str(HERE.parent/'v36-low-commit'))

import probe

from job_family_v4 import Family as LegacyFamily, suspended

from memory_watchdog_v34 import process_info, save

from memory_guard_v33 import sample

class Family(LegacyFamily):
 def __init__(self,job,out,budget_gib=6,interval=.05,recent_limit=64,summary_interval=1.,sampler=sample,inspector=process_info):
  if not 1<=recent_limit<=1024:raise ValueError('recent_limit must be 1..1024')
  if not 0<interval<=2:raise ValueError('monitor interval must be <=2 seconds')
  if not 0<budget_gib<=6:raise ValueError('project budget cannot exceed authorized 6 GiB')
  super().__init__(job,out,budget_gib,interval)
  self.samples=collections.deque(maxlen=recent_limit)
  self.out.mkdir(parents=True,exist_ok=True)
  self.stream=(self.out/'samples.jsonl').open('a',encoding='utf-8',buffering=1)
  self.sampler=sampler;self.inspector=inspector;self.count=0;self.peaks={};self.minima={}
  self.summary_interval=max(1.,summary_interval);self.last_summary=0.;self.current=[]
  self.summary_writes=0;self.session_started=time.time();self.closed=False
 def summary(self,force=False):
  now=time.monotonic()
  if not force and now-self.last_summary<self.summary_interval:return
  self.summary_writes+=1
  save(self.out/'summary.json',{'session_started':self.session_started,'sample_count':self.count,
   'failed':self.failed,'closed':self.closed,'budget_gib':self.budget,
   'global_commit_floor_gib':3,'global_physical_floor_gib':2,
   'peaks':self.peaks,'minimum_headroom':self.minima,'current_members':self.current,
   'observed_identities':list(self.identities.values()),'runner_identity':self.runner,
   'recent_samples':list(self.samples),'recent_limit':self.samples.maxlen,
   'summary_writes':self.summary_writes})
  self.last_summary=now
 def record(self,row):
  self.count+=1;row['sequence']=self.count;row['session_started']=self.session_started
  # One append per sample, no previous samples reread or rewritten.
  self.stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');self.stream.flush()
  self.samples.append(row)
  for key in ('project_working_gib','project_private_commit_gib','CommitTotal_gib'):
   if key in row['host']:self.peaks[key]=max(self.peaks.get(key,0),row['host'][key])
  for key in ('commit_headroom_gib','PhysicalAvailable_gib'):
   self.minima[key]=min(self.minima.get(key,float('inf')),row['host'][key])
  self.summary()
 def tick(self):
  pids=self.members();observed=[]
  for pid in pids:
   info=self.inspector(pid)
   if info is None:
    if pid not in self.members():continue
    raise RuntimeError('Cannot inspect owned Job member')
   old=self.identities.get(pid)
   if old and not self.same_identity(old,info):raise RuntimeError('PID identity changed; close owned Job only')
   if pid not in self.members():continue
   self.identities[pid]={k:info[k] for k in ('pid','exe','created_ticks')};observed.append(info)
  runner=self.inspector(os.getpid())
  if not runner or not self.same_identity(self.runner,runner):raise RuntimeError('Runner identity mismatch')
  self.current=[x['pid'] for x in observed]
  # Charge every Job member, including conhost. Do not double-count a runner in Job.
  processes=observed+([] if runner['pid'] in self.current else [runner])
  s=self.sampler();s['project_working_gib']=sum(x['working_gib'] for x in processes)
  s['project_private_commit_gib']=sum(x['private_commit_gib'] for x in processes)
  row={'host':s,'job_members':observed,'runner':runner};self.record(row)
  if s['project_working_gib']>self.budget or s['project_private_commit_gib']>self.budget:raise MemoryError('Project memory budget exceeded')
  if s['commit_headroom_gib']<3 or s['PhysicalAvailable_gib']<2:raise MemoryError('Global memory floor crossed')
  return row
 def fail_closed(self,error):
  # Close the kernel-owned Job only. Never terminate any enumerated PID directly.
  super().fail_closed(error)
  try:self.summary(force=True)
  except BaseException:pass
 def assert_safe(self):
  if self.failed:raise RuntimeError(self.failed)
  if self.closed or not self.thread or not self.thread.is_alive():raise RuntimeError('Family monitor is not running')
 def start_before_resume(self):
  super().start_before_resume();return self
 def resume(self,thread_handle):
  self.assert_safe()
  try:probe.must(probe.K.ResumeThread(thread_handle)!=0xffffffff)
  except BaseException as error:self.fail_closed(error);raise
 def close(self):
  if self.closed:return
  try:
   super().close();self.closed=True
   self.summary(force=True)
  except BaseException as error:
   self.fail_closed(error);raise
  finally:self.stream.close()
 def __enter__(self):return self
 def __exit__(self,kind,error,tb):
  if error:self.fail_closed(error)
  try:self.close()
  finally:self.job.close()

Job=probe.Job
