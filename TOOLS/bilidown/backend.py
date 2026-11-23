"""Durable per-workspace jobs. Only an active download owns a worker process."""
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from . import account
from .direct_network import child_environment
from .site_api import Failure, check_account, identify, metadata, require, streams

ROOT=Path(__file__).resolve().parent
RUNTIME=next((p for p in [ROOT.parent/"model_runtime/src",ROOT.parent/"runtime_manager/src"] if p.is_dir()),None)
if RUNTIME is not None:
    sys.path.insert(0,str(RUNTIME))
from ekb_model_runtime.run_manager.process import WindowsJob, process_identity

PRESETS=json.loads((ROOT/"defaults.json").read_text("utf8"))
ACTIVE={"STARTING","RUNNING"}
HEIGHTS=(360,480,720,1080,2160)
BIN_ROOT = ROOT.parents[1] / ".local/video-bin"
DDL="""
CREATE TABLE IF NOT EXISTS jobs(
 id TEXT PRIMARY KEY, spec TEXT NOT NULL UNIQUE, state TEXT NOT NULL, phase TEXT NOT NULL,
 owner TEXT, stop INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0,
 updated REAL NOT NULL, error TEXT, result TEXT, attempt INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value INTEGER);
"""

def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)

def digest(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def safe(root,rel):
    p=(root/rel).resolve()
    require(p.is_relative_to(root.resolve()),"INVALID_PATH","路径超出当前工作区。")
    return p

@contextmanager
def connect(ws):
    p=safe(ws,"tmp/bilidown/jobs.sqlite3");p.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(p,timeout=15)) as db:
        db.row_factory=sqlite3.Row;db.executescript(DDL)
        with db: yield db

def quality(value):
    require(type(value) is int and value in HEIGHTS,"INVALID_QUALITY","不支持该画质上限。")
    return value

def options(ws):
    with connect(ws) as db:values=dict(db.execute("SELECT key,value FROM settings"))
    return {"height":values.get("height",480),"transport":values.get("transport","aria2"),"output":str(safe(ws,PRESETS["output"])),"account_saved":(account.profile()/"login.dpapi").is_file()}

def settings(ws,height,transport):
    require(transport in ("https","aria2"),"INVALID_TRANSPORT","请选择支持的传输方式。")
    with connect(ws) as db:
        db.execute("INSERT OR REPLACE INTO settings VALUES('height',?)",(quality(height),))
        db.execute("INSERT OR REPLACE INTO settings VALUES('transport',?)",(transport,))
    return options(ws)

def credential(mode):
    require(mode in ("guest","account"),"INVALID_AUTH","身份必须为 guest 或 account。")
    if mode=="account":
        token=account.cookie();check_account(token);return token
    return None

def spec_for(ws,args):
    bv,part=identify(args.get("source"),args.get("part"))
    auth=args.get("auth","guest")
    require(auth in ("guest","account"),"INVALID_AUTH","身份必须为 guest 或 account。")
    if auth=="account":
        token=account.cookie()
    prefs=options(ws)
    preset=PRESETS[auth]
    height=quality(args.get("height",{"32":480,"80":1080}[preset["videoQn"]]))
    return dict(bvid=bv,part=part,height=height,best_available=auth=="guest" and "height" not in args,
                extras=True,audio_qn=int(preset["audioQn"]),auth=auth,transport=prefs["transport"],
                **({"account_key":hashlib.sha256(token.encode()).hexdigest()} if auth=="account" else {}))

def inspect(ws,args):
    spec=spec_for(ws,args);token=credential(spec["auth"])
    meta=metadata(spec["bvid"],spec["part"],token)
    _,fmt=streams(meta,spec["height"],token,best=spec.get("best_available",False),audio_qn=spec.get("audio_qn",192))
    return {**meta,"auth":spec["auth"],"selection":fmt}

def folder(ws,jid):
    require(isinstance(jid,str) and re.fullmatch(r"[0-9a-f]{32}",jid),"UNKNOWN_JOB","下载任务编号无效。")
    previous=safe(ws,"Downloads/BiliDown/"+jid)
    return previous if previous.is_dir() else safe(ws,PRESETS["output"]+"/"+jid)

def lookup(db,jid):
    row=db.execute("SELECT * FROM jobs WHERE id=?",(jid,)).fetchone()
    require(row,"UNKNOWN_JOB","当前工作区没有此任务。")
    return row

def reconcile(ws,db,row):
    if row["state"] in ACTIVE:
        owner=json.loads(row["owner"] or "{}")
        observed=process_identity(owner["pid"]) if owner else None
        if observed and observed.get("identity_status")=="UNVERIFIABLE":
            return row
        if observed!=owner or not owner:
            dest=folder(ws,row["id"])
            if (dest/"source.json").is_file() and (dest/"video.mp4").is_file():
                receipt=json.loads((dest/"source.json").read_text("utf8"))
                if receipt["job_id"]==row["id"] and digest(dest/"video.mp4")==receipt["sha256"]:
                    db.execute("UPDATE jobs SET state='SUCCEEDED',phase='done',result=?,owner=NULL WHERE id=?",(encoded(receipt),row["id"]))
                    return lookup(db,row["id"])
            db.execute("UPDATE jobs SET state='INTERRUPTED',phase='stopped',owner=NULL,error='WORKER_EXITED' WHERE id=?",(row["id"],))
            return lookup(db,row["id"])
    if row["state"]=="SUCCEEDED":
        data=json.loads(row["result"]);dest=folder(ws,row["id"])
        if not (dest/"source.json").is_file() or not (dest/"video.mp4").is_file() or (dest/"video.mp4").stat().st_size!=data["size"]:
            db.execute("UPDATE jobs SET state='FAILED',error='ARTIFACT_CHANGED' WHERE id=?",(row["id"],))
            return lookup(db,row["id"])
    return row

def public(row):
    spec=json.loads(row["spec"]);spec.pop("account_key",None)
    return dict(job_id=row["id"],state=row["state"],phase=row["phase"],bytes=row["bytes"],attempt=row["attempt"],
                cancel_requested=bool(row["stop"]),request=spec,error=row["error"],result=json.loads(row["result"]) if row["state"]=="SUCCEEDED" else None)

def launch(ws,db,jid,task):
    cmd=[sys.executable,"-I","-X","utf8",str(ROOT/"worker.py"),str(ws),jid,str(lookup(db,jid)["attempt"])]
    if task:cmd.extend(task)
    p=subprocess.Popen(cmd,env=child_environment(),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    owner=process_identity(p.pid)
    require(owner and owner.get("created"),"WORKER_START","下载工作进程启动失败。")
    db.execute("UPDATE jobs SET owner=?,updated=? WHERE id=?",(encoded(owner),time.time(),jid))

def command(ws,action,task=None,**args):
    ws=Path(ws).resolve(strict=True)
    if action=="inspect":return inspect(ws,args)
    if action=="verify":
        jid=args.get("job_id")
        with connect(ws) as db:
            row=reconcile(ws,db,lookup(db,jid))
            require(row["state"]=="SUCCEEDED","NOT_COMPLETED","仅可核验已完成的下载。")
        receipt=json.loads(row["result"]);out=folder(ws,jid)
        try:
            verified=(json.loads(safe(out,"source.json").read_text("utf8"))==receipt
                      and digest(safe(out,"video.mp4"))==receipt["sha256"]
                      and all(digest(safe(out,x["name"]))==x["sha256"] for x in receipt.get("attachments",[])))
        except (OSError,ValueError):
            verified=False
        require(verified,"ARTIFACT_CHANGED","成品或来源清单已更改。")
        return {"verified":True,"job_id":jid,"sha256":receipt["sha256"],"size":receipt["size"]}
    spec=spec_for(ws,args) if action=="start" else None
    with connect(ws) as db:
        db.execute("BEGIN IMMEDIATE")
        if action=="list":
            return [public(reconcile(ws,db,r)) for r in db.execute("SELECT * FROM jobs ORDER BY updated DESC LIMIT 50").fetchall()]
        if action=="start":
            row=db.execute("SELECT * FROM jobs WHERE spec=?",(encoded(spec),)).fetchone()
            if row is None:
                other = "https" if spec["transport"] == "aria2" else "aria2"
                row = db.execute("SELECT * FROM jobs WHERE spec=? AND state IN ('STARTING','RUNNING','SUCCEEDED')",
                                 (encoded({**spec,"transport":other}),)).fetchone()
            if row:return public(reconcile(ws,db,row))
            jid=uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,spec,state,phase,updated) VALUES(?,?,'STARTING','starting',?)",(jid,encoded(spec),time.time()))
            launch(ws,db,jid,task)
        else:
            jid=args.get("job_id");row=reconcile(ws,db,lookup(db,jid))
            if action=="cancel" and row["state"] in ACTIVE:
                db.execute("UPDATE jobs SET stop=1 WHERE id=?",(jid,))
            elif action=="retry":
                require(row["state"] in {"FAILED","CANCELLED","INTERRUPTED"},"NOT_RETRYABLE","该任务目前无需重试。")
                require(row["error"]!="ARTIFACT_CHANGED" or not folder(ws,jid).exists(),"ARTIFACT_CHANGED","已交付文件发生变化，请先核对成品。")
                db.execute("UPDATE jobs SET state='STARTING',phase='starting',error=NULL,stop=0,bytes=0,attempt=attempt+1 WHERE id=?",(jid,))
                launch(ws,db,jid,task)
            else:require(action in ("status","cancel"),"INVALID_ACTION","操作无效。")
        return public(lookup(db,jid))

def update(ws,jid,**values):
    values["updated"]=time.time()
    with connect(ws) as db:
        db.execute("UPDATE jobs SET "+",".join(k+"=?" for k in values)+" WHERE id=?",(*values.values(),jid))

def execute(ws,jid,attempt,task=None):
    import runpy
    from .transfer import transfer
    job=WindowsJob()
    job.assign(SimpleNamespace(_handle=-1)) # Children inherit this kill-on-close Job.
    with connect(ws) as db:
        db.execute("BEGIN IMMEDIATE")
        row=lookup(db,jid)
        require(row["state"]=="STARTING" and row["attempt"]==attempt,"OWNER_CHANGED","作业启动代次已改变。")
        # On Windows a venv launcher and its Python worker have separate PIDs.
        db.execute("UPDATE jobs SET owner=? WHERE id=?",(encoded(process_identity(os.getpid())),jid))
    spec=json.loads(row["spec"])
    client=None;entered=False;last_task_check=0
    try:
        if task:
            root,task_id=task;root=Path(root)
            api=runpy.run_path(str(root/"TOOLS/runtime_manager/bootstrap.py"))
            client=api["existing_client"](root);identity=process_identity(os.getpid())
            client.task_run("enter",ws,task_id,call_id=jid,identity=identity);entered=True
        def check_cancel():
            nonlocal last_task_check
            with connect(ws) as db:stop=lookup(db,jid)["stop"]
            if entered and time.monotonic()-last_task_check>=2:
                last_task_check=time.monotonic()
                if client.task_run("inspect",ws,task_id).get("outcome")=="cancel":stop=True
            require(not stop,"CANCELLED","任务已取消。")
        update(ws,jid,state="RUNNING",phase="queued")
        while True:
            check_cancel()
            with connect(ws) as db:
                db.execute("BEGIN IMMEDIATE")
                active=db.execute("SELECT * FROM jobs WHERE state IN ('STARTING','RUNNING') ORDER BY rowid").fetchall()
                running=[reconcile(ws,db,r) for r in active]
                live=[r for r in running if r["state"] in ACTIVE]
                if live[0]["id"]==jid and not any(r["id"]!=jid and r["phase"] not in ("queued","starting") for r in live):
                    db.execute("UPDATE jobs SET phase='resolve' WHERE id=?",(jid,))
                    break
            time.sleep(.4)
        check_cancel();token=credential(spec["auth"])
        if token:require(hashlib.sha256(token.encode()).hexdigest()==spec["account_key"],"ACCOUNT_CHANGED","登录身份已更新，请重新发起下载。")
        meta=metadata(spec["bvid"],spec["part"],token);tracks,fmt=streams(meta,spec["height"],token,best=spec.get("best_available",False),audio_qn=spec.get("audio_qn",192))
        result=transfer(ws,jid,row["attempt"],spec,meta,tracks,fmt,check_cancel)
        update(ws,jid,state="SUCCEEDED",phase="done",owner=None,result=encoded(result),error=None)
    except Exception as e:
        code=e.code if isinstance(e,Failure) else "WORKER_ERROR_"+type(e).__name__
        update(ws,jid,state="CANCELLED" if code=="CANCELLED" else "FAILED",phase="stopped",error=code,owner=None)
    finally:
        if entered:client.task_run("leave",ws,task_id,call_id=jid,identity=identity)

