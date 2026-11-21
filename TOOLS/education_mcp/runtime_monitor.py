"""Workspace-scoped MCP telemetry. The monitor owns no model process."""
import ctypes
from functools import lru_cache
from contextlib import closing
import json
import os
from pathlib import Path
import re
import runpy
import sqlite3
import subprocess
import sys
import time
import uuid

import psutil

PROCESS_GPU=runpy.run_path(str(Path(__file__).with_name("process_gpu.py")))


SCHEMA = {
    "name": "runtime_monitor",
    "description": "Start/stop a shared workspace capture. status reports capture health; summary groups tools; calls and samples are paginated. models(job_id) returns BeMarkdown per-request response timings and correlated resources. GPU and host metrics cover the local device, including concurrent work.",
    "inputSchema": {"type": "object", "additionalProperties": False, "required": ["action"], "properties": {
        "action": {"enum": ["start","status","stop","summary","calls","samples","models"]},
        "capture_id": {"type":"string","pattern":"^[a-f0-9]{32}$"},
        "job_id": {"type":"string","pattern":"^[a-f0-9]{32}$"},
        "interval_seconds": {"type":"number","minimum":0.2,"maximum":60},
        "duration_seconds": {"type":"integer","minimum":1,"maximum":604800},
        "offset": {"type":"integer","minimum":0}, "limit":{"type":"integer","minimum":1,"maximum":100}
    }}
}

DDL = """
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,started REAL,deadline REAL,interval REAL,pid INTEGER,birth REAL,state TEXT,ended REAL);
CREATE UNIQUE INDEX IF NOT EXISTS single_collector ON runs(state) WHERE state='running';
CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY,run TEXT,name TEXT,action TEXT,pid INTEGER,start REAL,end REAL,seconds REAL,status TEXT,process_cpu_pct REAL,rss_start INTEGER,rss_end INTEGER,job TEXT);
CREATE INDEX IF NOT EXISTS call_run ON calls(run,id);
CREATE TABLE IF NOT EXISTS endpoints(call_id INTEGER PRIMARY KEY, start_json TEXT, end_json TEXT);
CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY,run TEXT,t REAL,cpu_pct REAL,ram_used INTEGER,gpu_pct REAL,vram_used INTEGER,gpu_error TEXT);
CREATE INDEX IF NOT EXISTS sample_window ON samples(run,t);
CREATE TABLE IF NOT EXISTS models(sample INTEGER,run TEXT,job TEXT,role TEXT,request TEXT,worker_pid INTEGER,cpu_pct REAL,rss INTEGER);
CREATE INDEX IF NOT EXISTS model_request ON models(run,job,request);
CREATE INDEX IF NOT EXISTS model_sample ON models(sample);
CREATE TABLE IF NOT EXISTS process_gpu(sample INTEGER,scope TEXT,pid INTEGER,engine_pct REAL,dedicated_bytes INTEGER,PRIMARY KEY(sample,scope,pid));
"""

def connect(path):
    db=sqlite3.connect(path,timeout=3)
    db.row_factory=sqlite3.Row
    return db

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def latest(db, capture_id=None):
    if capture_id:
        return db.execute("SELECT * FROM runs WHERE id=?", (capture_id,)).fetchone()
    return db.execute("SELECT * FROM runs ORDER BY started DESC LIMIT 1").fetchone()

def alive(row):
    try:
        return row["pid"] and abs(psutil.Process(row["pid"]).create_time()-row["birth"])<0.01
    except psutil.Error:
        return False

def proc_stats():
    proc=psutil.Process()
    cpu=proc.cpu_times()
    return cpu.user+cpu.system,proc.memory_info().rss

@lru_cache(maxsize=1)
def gpu_probe():
    try:
        return Nvml(),None
    except (OSError,AttributeError) as exc:
        return None,str(exc)

def host_snapshot():
    gpu,error=gpu_probe()
    util,vram,error=gpu.sample() if gpu else (None,None,error)
    mem=psutil.virtual_memory()
    return dict(epoch=time.time(),gpu0_util_pct=util,gpu0_used_bytes=vram,
                gpu_error=error,host_ram_used_bytes=mem.total-mem.available)

def observed_call(path, name, args, invoke):
    if not path.is_file():
        return invoke()
    token=None
    job=args.get("job_id")
    job=job if isinstance(job,str) and re.fullmatch("[a-f0-9]{32}",job) else None
    try:
        with closing(connect(path)) as db, db:
            run=latest(db)
            if run and run["state"]=="running" and time.time()<run["deadline"] and alive(run):
                rss=psutil.Process().memory_info().rss
                action=args.get("action")
                action=action if isinstance(action,str) and re.fullmatch(r"[a-z_]{1,48}",action) else None
                token=db.execute("INSERT INTO calls(run,name,action,pid,start,status,rss_start) VALUES(?,?,?,?,?,'running',?)",
                                 (run["id"],name,action,os.getpid(),time.time(),rss)).lastrowid
    except (OSError,sqlite3.Error,psutil.Error):
        pass
    if token is not None:
        try:
            before=host_snapshot()
            cpu,_=proc_stats()
        except (OSError,psutil.Error):
            before={}
            cpu=None
    dispatched=time.time()
    start=time.perf_counter()
    status="error"
    try:
        value=invoke()
        status="error" if isinstance(value,dict) and value.get("isError") else "ok"
        return value
    finally:
        if token is not None:
            elapsed=time.perf_counter()-start
            try:
                end=time.time()
                cpu_end,rss_end=proc_stats()
                after=host_snapshot()
                if status=="ok" and name in ("bemarkdown_convert","bemarkdown_convert_image"):
                    try:
                        candidate=json.loads(value["content"][0]["text"]).get("job_id")
                        if isinstance(candidate,str) and re.fullmatch("[a-f0-9]{32}",candidate):
                            job=candidate
                    except (KeyError,IndexError,TypeError,ValueError,AttributeError):
                        pass
                with closing(connect(path)) as db, db:
                    db.execute("UPDATE calls SET start=?,end=?,seconds=?,status=?,process_cpu_pct=?,rss_end=?,job=? WHERE id=?",
                               (dispatched,end,elapsed,status,(cpu_end-cpu)*100/max(elapsed,1e-9) if cpu is not None else None,rss_end,job,token))
                    db.execute("INSERT INTO endpoints VALUES(?,?,?)",(token,json.dumps(before),json.dumps(after)))
            except (OSError,sqlite3.Error,psutil.Error):
                pass


class Nvml:
    class Memory(ctypes.Structure):
        _fields_=[(name,ctypes.c_ulonglong) for name in ("total","free","used")]
    class Util(ctypes.Structure):
        _fields_=[(name,ctypes.c_uint) for name in ("gpu","memory")]
    def __init__(self):
        self.lib=ctypes.WinDLL("nvml.dll") if os.name=="nt" else ctypes.CDLL("libnvidia-ml.so.1")
        self.device=ctypes.c_void_p()
        if self.lib.nvmlInit_v2() or self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.device)):
            raise OSError("NVML GPU 0 unavailable")

    def sample(self):
        util,mem=self.Util(),self.Memory()
        u=self.lib.nvmlDeviceGetUtilizationRates(self.device,ctypes.byref(util))
        m=self.lib.nvmlDeviceGetMemoryInfo(self.device,ctypes.byref(mem))
        return (None if u else util.gpu,None if m else mem.used,None if not (u or m) else f"NVML util={u}, memory={m}")


class LocalWorkers:
    """Follow workspace job IDs and sample only identified open local model sessions."""
    def __init__(self,workspace):
        self.workspace=workspace
        self.jobs=workspace/"tmp/bemarkdown/.mcp/jobs"
        self.home=Path(os.environ.get("EKB_RUN_MANAGER_HOME") or
                       Path(os.environ.get("LOCALAPPDATA",Path.home()/".local/share"))/"BeMarkdown/run-manager")
        self.stamp=None
        self.pending=set()
        self.active=set()
        self.workers={}
        self.cpu_history={}

    def tree_sample(self,key,proc):
        now=time.monotonic()
        last,old=self.cpu_history.get(key,(now,{}))
        current={}
        rss=0
        for child in [proc,*proc.children(recursive=True)]:
            try:
                ident=(child.pid,child.create_time())
                cpu=child.cpu_times()
                current[ident]=cpu.user+cpu.system
                rss+=child.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                continue
        self.cpu_history[key]=(now,current)
        matched=current.keys() & old.keys()
        delta=sum(max(0,current[k]-old[k]) for k in matched)
        return (100*delta/(now-last) if matched and now>last else None,rss if current else None)

    def rows(self):
        if not self.jobs.is_dir():
            return []
        stamp=self.jobs.stat().st_mtime_ns
        if self.stamp!=stamp:
            self.pending.update(p.name for p in self.jobs.iterdir() if re.fullmatch("[a-f0-9]{32}",p.name))
            self.stamp=stamp
        for job in tuple(self.pending):
            try:
                state=read(self.jobs/job/"state.json")
            except (ValueError,OSError):
                continue
            if state.get("status") in ("RUNNING","QUEUED") and state.get("runtime_management")=="PERSISTENT_MANAGER":
                try:
                    request=read(self.home/"jobs"/job/"request.json")
                except (OSError,ValueError):
                    continue
                if request.get("job_id")==job and Path(request["workspace"]).resolve()==self.workspace:
                    self.active.add(job)
            self.pending.remove(job)
        rows=[]
        live=set()
        for job in tuple(self.active):
            folder=self.home/"jobs"/job
            try:
                progress=read(folder/"model_api_progress.json")
                if progress.get("job_id")!=job:
                    continue
            except (OSError,ValueError):
                progress={}
            matched=False
            for opened in (folder/"model_api_sessions").glob("*/OPENED.json"):
                if opened.with_name("CLOSED.json").exists():
                    continue
                try:
                    receipt=read(opened)
                    pid=receipt['worker_pid']
                    key=(job,opened.parent.name,pid)
                    proc=self.workers.get(key)
                    first=proc is None
                    if first:
                        proc=psutil.Process(pid)
                        if proc.create_time()>opened.stat().st_mtime+0.1:
                            continue
                    if not proc.is_running():
                        continue
                    self.workers[key]=proc
                    live.add(key)
                    role=receipt["role"]
                    request=progress.get("request_id") if progress.get("state")=="RUNNING" and progress.get("role")==role and progress.get("session_id") in (None,key[1]) else None
                    pct,rss=self.tree_sample(key,proc)
                    rows.append((job,role,request,pid,pct,rss))
                    matched |= request is not None
                except (psutil.Error,ValueError,KeyError,OSError):
                    continue
            if not matched and progress.get("state")=="RUNNING" and progress.get("request_id"):
                rows.append((job,progress.get("role"),progress["request_id"],None,None,None))
            try:
                if read(self.jobs/job/"state.json").get("status") in ("SUCCEEDED","FAILED","CANCELLED"):
                    self.active.discard(job)
            except (OSError,ValueError):
                pass
        self.workers={k:v for k,v in self.workers.items() if k in live}
        self.cpu_history={k:v for k,v in self.cpu_history.items() if k in live}
        return rows


def collect(path,workspace,run):
    gpu,error=gpu_probe()
    counters=PROCESS_GPU["ProcessGpu"]()
    workers=LocalWorkers(workspace)
    psutil.cpu_percent()
    with closing(connect(path)) as db:
        # The parent launches us inside the transaction reserving the only slot.
        for _ in range(100):
            if latest(db,run):
                break
            time.sleep(.05)
        first=True
        while True:
            record=latest(db,run)
            if not record or record["state"]!="running":
                return
            if time.time()>=record["deadline"]:
                with db:
                    db.execute("UPDATE runs SET state='expired',ended=? WHERE id=? AND state='running'",(time.time(),run))
                return
            pct=psutil.cpu_percent()
            mem=psutil.virtual_memory()
            gpu_pct,vram,gpu_error=gpu.sample() if gpu else (None,None,error)
            now=time.time()
            model_rows=workers.rows()
            groups=[("worker_tree",key[2],[pid for pid,birth in times])
                    for key,(_,times) in workers.cpu_history.items()]
            for caller in db.execute("SELECT pid,min(start) AS start FROM calls WHERE run=? AND status='running' GROUP BY pid",(run,)):
                try:
                    if psutil.Process(caller["pid"]).create_time()<=caller["start"]:
                        groups.append(("tool_process",caller["pid"],[caller["pid"]]))
                except psutil.Error:
                    pass
            usage=counters.sample(bool(groups))
            with db:
                db.execute("BEGIN IMMEDIATE")
                if latest(db,run)["state"]!="running":
                    return
                sample=db.execute("INSERT INTO samples(run,t,cpu_pct,ram_used,gpu_pct,vram_used,gpu_error) VALUES(?,?,?,?,?,?,?)",
                                  (run,now,None if first else pct,mem.total-mem.available,gpu_pct,vram,gpu_error)).lastrowid
                db.executemany("INSERT INTO models VALUES(?,?,?,?,?,?,?,?)",[(sample,run,*v) for v in model_rows])
                db.executemany("INSERT INTO process_gpu VALUES(?,?,?,?,?)",[
                    (sample,scope,pid,*PROCESS_GPU["attributed"](usage,pids)) for scope,pid,pids in groups])
            first=False
            until=min(time.time()+record["interval"],record["deadline"])
            while time.time()<until:
                time.sleep(min(.5,max(0,until-time.time())))
                if latest(db,run)["state"]!="running":
                    return


def start(path,workspace,interval,duration,proposed_id=None):
    if not 0.2<=interval<=60 or not 1<=duration<=604800:
        raise ValueError("Invalid sampling interval or duration")
    path.parent.mkdir(parents=True,exist_ok=True)
    with closing(connect(path)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(DDL)
        db.execute("BEGIN IMMEDIATE")
        row=latest(db)
        if row and row["state"]=="running":
            if alive(row) and row["deadline"]>time.time():
                return dict(row,created=False)
            state="expired" if row["deadline"]<=time.time() else "interrupted"
            db.execute("UPDATE runs SET state=?,ended=? WHERE id=?",(state,min(row["deadline"],time.time()),row["id"]))
        now=time.time()
        run=proposed_id or uuid.uuid4().hex
        if latest(db,run):
            run=uuid.uuid4().hex
        process=subprocess.Popen([sys.executable,"-I","-X","utf8",str(Path(__file__).resolve()),str(path),str(workspace),run],
                                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        birth=psutil.Process(process.pid).create_time()
        db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,'running',NULL)",(run,now,now+duration,interval,process.pid,birth))
        db.commit()
        return dict(latest(db,run),created=True)


def window(db,run,start,end):
    return dict(db.execute("""SELECT count(*) samples,avg(cpu_pct) host_cpu_pct_avg,max(ram_used) host_ram_used_peak_bytes,
                    avg(gpu_pct) gpu0_util_pct_avg,max(gpu_pct) gpu0_util_pct_peak,max(vram_used) gpu0_used_peak_bytes
                    FROM samples WHERE run=? AND t BETWEEN ? AND ?""",(run,start,end)).fetchone())


def model_requests(service,db,run,job,offset,limit):
    package=service.package(job)
    report=read(package/"conversion_report.json")
    visual=report.get("visual_serial",{})
    items=[]
    total=0
    for role,stage in visual.get("stage_metrics",{}).items():
        for req in stage.get("requests",[]):
            total+=1
            if total<=offset or total>offset+limit:
                continue
            rid=req["request_id"]
            values=dict(db.execute("""SELECT count(*) samples,avg(m.cpu_pct) worker_cpu_pct_avg,max(m.rss) worker_rss_peak_bytes,
                        avg(s.cpu_pct) host_cpu_pct_avg,max(s.ram_used) host_ram_used_peak_bytes,
                        avg(s.gpu_pct) gpu0_util_pct_avg,max(s.gpu_pct) gpu0_util_pct_peak,max(s.vram_used) gpu0_used_peak_bytes
                        ,avg(g.engine_pct) worker_gpu_engine_pct_avg,max(g.engine_pct) worker_gpu_engine_pct_peak,
                        max(g.dedicated_bytes) worker_dedicated_vram_peak_bytes
                        FROM models m JOIN samples s ON s.id=m.sample LEFT JOIN process_gpu g
                        ON g.sample=m.sample AND g.scope='worker_tree' AND g.pid=m.worker_pid
                        WHERE m.run=? AND m.job=? AND m.request=?""",
                                   (run,job,rid)).fetchone())
            server=req.get("server",{})
            items.append(dict(role=role,model_id=visual.get("stage_identities",{}).get(role,{}).get("model_id"),
                              request_id=rid,items=req.get("count"),response_seconds=req.get("wall_seconds"),
                              inference_seconds=server.get("inference_seconds"),allocated_peak_bytes=server.get("peak_allocated_bytes"),
                              reserved_peak_bytes=server.get("peak_reserved_bytes"),resources=values))
    return dict(job_id=job,total=total,offset=offset,requests=items)


def command(service,action,capture_id=None,interval_seconds=1,duration_seconds=86400,job_id=None,offset=0,limit=30):
    path=service.workspace_manager.safe_path("tmp/runtime-monitor") / "captures.sqlite3"
    if not 0<=offset or not 1<=limit<=100:
        raise ValueError("Invalid page window")
    if action=="start":
        return start(path,service.workspace,interval_seconds,duration_seconds)
    if not path.is_file():
        if action=="status":
            return dict(state="idle")
        raise ValueError("No capture in this workspace")
    with closing(connect(path)) as db,db:
        if db.execute("PRAGMA user_version").fetchone()[0]<1:
            db.executescript(DDL)
            db.execute("PRAGMA user_version=1")
        record=latest(db,capture_id)
        if not record:
            raise ValueError("Unknown capture")
        run=record["id"]
        if action=="stop":
            db.execute("UPDATE runs SET state='stopped',ended=? WHERE id=? AND state='running'",(time.time(),run))
            record=latest(db,run)
        if action in ("status","stop","summary"):
            result=dict(record)
            if result["state"]=="running" and not alive(record):
                result["state"]="interrupted"
            result.update(database=str(path),resources=window(db,run,record["started"],record["ended"] or time.time()),
                          last_sample=db.execute("SELECT max(t) FROM samples WHERE run=?",(run,)).fetchone()[0])
            if action=="summary":
                result['tools']=[dict(r) for r in db.execute("""SELECT name,action,count(*) calls,sum(status='error') errors,
                                   sum(status='running') unfinished_count,avg(seconds) response_seconds_avg,max(seconds) response_seconds_max,
                                   max(process_cpu_pct) process_cpu_pct_max,max(max(rss_start,coalesce(rss_end,rss_start))) process_rss_endpoint_peak_bytes
                                   FROM calls WHERE run=? GROUP BY name,action""",(run,))]
                result["local_model_windows"]=[dict(r) for r in db.execute("""SELECT m.job,m.role,count(*) samples,
                                  avg(m.cpu_pct) worker_tree_cpu_pct_avg,max(m.rss) worker_tree_rss_peak_bytes,
                                  avg(s.gpu_pct) gpu0_util_pct_avg,max(s.gpu_pct) gpu0_util_pct_peak,
                                  max(s.vram_used) gpu0_used_peak_bytes,max(s.ram_used) host_ram_used_peak_bytes
                                  FROM models m JOIN samples s ON s.id=m.sample WHERE m.run=? GROUP BY m.job,m.role""",(run,))]
            return result
        if action=="models":
            if not job_id:
                raise ValueError("models requires job_id")
            return model_requests(service,db,run,job_id,offset,limit)
        if action=="calls":
            calls=[dict(r) for r in db.execute("SELECT * FROM calls WHERE run=? ORDER BY id LIMIT ? OFFSET ?",(run,limit,offset))]
            for item in calls:
                item["resources"]=window(db,run,item["start"],item["end"] or record["ended"] or time.time())
                bounds=db.execute("SELECT start_json,end_json FROM endpoints WHERE call_id=?",(item["id"],)).fetchone()
                item["endpoint_resources"]=[json.loads(v) for v in bounds] if bounds else None
                item["resources"].update(db.execute("""SELECT avg(g.engine_pct) process_gpu_engine_pct_avg,
                    max(g.engine_pct) process_gpu_engine_pct_peak,max(g.dedicated_bytes) process_dedicated_vram_peak_bytes
                    FROM samples s JOIN process_gpu g ON s.id=g.sample WHERE s.run=? AND g.scope='tool_process' AND
                    g.pid=? AND s.t BETWEEN ? AND ?""",(run,item["pid"],item["start"],item["end"] or record["ended"] or time.time())).fetchone())
            return dict(capture_id=run,offset=offset,calls=calls)
        if action=="samples":
            samples=[dict(r) for r in db.execute("SELECT * FROM samples WHERE run=? ORDER BY id LIMIT ? OFFSET ?",(run,limit,offset))]
            for item in samples:
                item["model_workers"]=[dict(r) for r in db.execute("SELECT job,role,request,worker_pid,cpu_pct,rss FROM models WHERE sample=? AND run=?",(item["id"],run))]
                item["process_gpu"]=[dict(r) for r in db.execute("SELECT scope,pid,engine_pct,dedicated_bytes FROM process_gpu WHERE sample=?",(item["id"],))]
            return dict(capture_id=run,offset=offset,samples=samples)
        raise ValueError("Unknown monitor action")

if __name__=="__main__":
    collect(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
