"""Durable lifecycle of agent-owned task runs. All mutations use the submission lock."""
import json
from pathlib import Path
import re
import runpy
import time
from types import SimpleNamespace
import uuid

from .common import ACTIVE, RunManagerError
from .process import process_identity

FINAL = {"complete":"COMPLETED", "pause":"PAUSED", "cancel":"CANCELLED"}
DDL = """
CREATE TABLE IF NOT EXISTS task_runs(
 id TEXT PRIMARY KEY, workspace TEXT NOT NULL, state TEXT NOT NULL, outcome TEXT,
 proposed TEXT NOT NULL, capture TEXT, receipt TEXT, started REAL NOT NULL);
CREATE TABLE IF NOT EXISTS task_captures(id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS task_calls(id TEXT PRIMARY KEY, task TEXT NOT NULL, identity TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS pending_task_calls ON task_calls(task);
CREATE TABLE IF NOT EXISTS task_jobs(job TEXT NOT NULL, task TEXT NOT NULL, PRIMARY KEY(job,task));
CREATE INDEX IF NOT EXISTS jobs_for_task ON task_jobs(task);
"""

def link_job(db, job, task, reused):
    # An existing job from an older registry may belong to a legacy caller.
    if not task and not reused:
        return
    if reused and not db.execute("SELECT 1 FROM task_jobs WHERE job=?", (job,)).fetchone():
        db.execute("INSERT INTO task_jobs VALUES(?, '')", (job,))
    db.execute("INSERT OR IGNORE INTO task_jobs VALUES(?,?)",(job,task or ""))


class TaskRuns:
    def __init__(self,manager):
        self.manager=manager
        self.store=manager.store
        self.monitor=None
        with self.store.connect() as db:
            self.pending={row[0] for row in db.execute("SELECT id FROM task_runs WHERE state='CLOSING'")}

    def busy(self):
        with self.store.connect() as db:
            return db.execute("SELECT 1 FROM task_runs WHERE state IN ('OPENING','OPEN','CLOSING') LIMIT 1").fetchone() is not None

    def instrumentation(self,workspace):
        program=self.manager.registry["programs"].get("education-convert",{})
        if not program.get("installed_root"):
            raise RunManagerError("TASK_MONITOR_NOT_INSTALLED")
        folder=Path(program["installed_root"])/"TOOLS/education_mcp"
        if self.monitor is None:
            from .registry import verify_runtime_files
            verify_runtime_files(program)
            self.monitor=runpy.run_path(str(folder/"runtime_monitor.py"))
            self.workspace_type=runpy.run_path(str(folder/"workspace_manager.py"))["WorkspaceManager"]
        wm=self.workspace_type(workspace)
        return SimpleNamespace(workspace=Path(workspace),workspace_manager=wm)

    def record(self,task,workspace):
        with self.store.connect() as db:
            row=db.execute("SELECT * FROM task_runs WHERE id=? AND workspace=?",(task,workspace)).fetchone()
        if not row:
            raise RunManagerError("UNKNOWN_TASK_RUN")
        return dict(row)

    def status(self,task,workspace):
        row=self.record(task,workspace)
        with self.store.connect() as db:
            jobs=db.execute("SELECT j.job_id,j.state FROM jobs j JOIN task_jobs l ON l.job=j.job_id WHERE l.task=? ORDER BY j.created",(task,)).fetchall()
            calls=db.execute("SELECT count(*) FROM task_calls WHERE task=?",(task,)).fetchone()[0]
        return dict(task_run_id=task,state=row["state"],outcome=row["outcome"],capture_id=row["capture"],
                    jobs=[dict(j) for j in jobs],inflight_calls=calls,receipt=json.loads(row["receipt"]) if row["receipt"] else None)

    def begin(self,task,workspace,resume):
        self.store.workspace(workspace)
        service=self.instrumentation(workspace)
        with self.store.connect() as db:
            db.execute("INSERT OR IGNORE INTO task_runs VALUES(?,?,'OPENING',NULL,?,NULL,NULL,?)",
                       (task,workspace,uuid.uuid4().hex,time.time()))
        row=self.record(task,workspace)
        if row["state"]=="PAUSED" and resume:
            with self.store.connect() as db:
                db.execute("UPDATE task_runs SET state='OPENING',outcome=NULL,capture=NULL,proposed=?,receipt=NULL WHERE id=?",
                           (uuid.uuid4().hex,task))
            row=self.record(task,workspace)
        if row["state"] in ("OPENING","OPEN"):
            path=service.workspace_manager.safe_path("tmp/runtime-monitor")/"captures.sqlite3"
            cap=self.monitor["start"](path,service.workspace,1,86400,proposed_id=row["proposed"])
            with self.store.connect() as db:
                if cap["created"] or cap["id"]==row["proposed"]:
                    db.execute("INSERT OR IGNORE INTO task_captures VALUES(?)",(cap["id"],))
                db.execute("UPDATE task_runs SET state='OPEN',capture=? WHERE id=?",(cap["id"],task))
        return self.status(task,workspace)

    def operation(self,action,workspace,task_run_id,**args):
        workspace=str(Path(workspace).resolve())
        if not isinstance(task_run_id,str) or not re.fullmatch('[0-9a-f]{32}',task_run_id):
            raise RunManagerError("INVALID_TASK_RUN_ID")
        task=task_run_id
        if action in ("begin","resume"):
            return self.begin(task,workspace,action=="resume")
        row=self.record(task,workspace)
        if action=="inspect":
            return self.status(task,workspace)
        if action in ("enter","leave"):
            call=args["call_id"]
            identity=args["identity"]
            with self.store.connect() as db:
                if action=="enter":
                    if row["state"]!="OPEN":
                        raise RunManagerError("TASK_RUN_NOT_OPEN")
                    if not identity or not identity.get("created") or process_identity(identity["pid"])!=identity:
                        raise RunManagerError("UNCONFIRMED_CALL_OWNER")
                    db.execute("INSERT INTO task_calls VALUES(?,?,?)",(call,task,json.dumps(identity,sort_keys=True)))
                else:
                    db.execute("DELETE FROM task_calls WHERE id=? AND task=? AND identity=?",
                               (call,task,json.dumps(identity,sort_keys=True)))
            return {"task_run_id":task}
        if action!="end" or args.get("outcome") not in FINAL:
            raise RunManagerError("INVALID_TASK_RUN_ACTION")
        outcome=args["outcome"]
        if row["state"]=="OPENING":
            # If begin lost its reply during capture creation, adopt only its reserved ID.
            service=self.instrumentation(workspace)
            path=service.workspace_manager.safe_path("tmp/runtime-monitor")/"captures.sqlite3"
            if path.is_file():
                from contextlib import closing
                with closing(self.monitor["connect"](path)) as captures:
                    capture=self.monitor["latest"](captures,row["proposed"])
                if capture:
                    with self.store.connect() as db:
                        db.execute("INSERT OR IGNORE INTO task_captures VALUES(?)",(capture["id"],))
                        db.execute("UPDATE task_runs SET capture=? WHERE id=?",(capture["id"],task))
        if row["state"] in ("OPEN","OPENING"):
            with self.store.connect() as db:
                db.execute("UPDATE task_runs SET state='CLOSING',outcome=? WHERE id=?",(outcome,task))
            self.pending.add(task)
        elif row["outcome"]!=outcome:
            raise RunManagerError("TASK_RUN_OUTCOME_CONFLICT")
        if task in self.pending:
            self.try_advance(task)
        return self.status(task,workspace)

    def advance(self,task):
        with self.store.connect() as db:
            row=db.execute("SELECT * FROM task_runs WHERE id=?",(task,)).fetchone()
            for call in db.execute("SELECT * FROM task_calls WHERE task=?",(task,)).fetchall():
                previous=json.loads(call["identity"])
                current=process_identity(previous["pid"])
                if current is None or current.get("created") and current["created"]!=previous["created"]:
                    db.execute("DELETE FROM task_calls WHERE id=?",(call["id"],))
                else:
                    return
            db.commit()
            jobs=db.execute("SELECT j.job_id,j.state,j.data,j.error,j.identity FROM jobs j JOIN task_jobs l ON j.job_id=l.job WHERE l.task=?",(task,)).fetchall()
            waiting=False
            job_receipts=[]
            for job in jobs:
                job_id=job["job_id"]
                shared=db.execute("""SELECT 1 FROM task_jobs l LEFT JOIN task_runs t ON l.task=t.id
                    WHERE l.job=? AND l.task!=? AND (l.task='' OR t.state IN ('OPEN','OPENING','CLOSING')) LIMIT 1""",(job_id,task)).fetchone()
                if not shared:
                    shared=db.execute("SELECT 1 FROM adopters WHERE job_id=?",(job_id,)).fetchone()
                job_receipts.append(dict(job_id=job_id,state=job["state"],shared=bool(shared),cleanup=json.loads(job["data"]).get("cleanup")))
                if job["state"]=="INTERRUPTED":
                    old=json.loads(job["identity"]) if job["identity"] else None
                    current=process_identity(old["pid"]) if old else None
                    if job_id in self.manager.active or current and (not current.get("created") or current.get("created")==old.get("created")):
                        return
                if job["state"] in ACTIVE:
                    if row["outcome"]=="cancel" and shared:
                        continue
                    waiting=True
                    if row["outcome"]=="cancel":
                        self.manager.dispatch("cancel",dict(workspace=row["workspace"],job_id=job_id,reason="Task run cancelled"))
            if waiting:
                return
            capture=row["capture"]
            others=db.execute("SELECT 1 FROM task_runs WHERE capture=? AND id!=? AND state IN ('OPEN','CLOSING')",(capture,task)).fetchone()
            owned=db.execute("SELECT 1 FROM task_captures WHERE id=?",(capture,)).fetchone()
        monitor_receipt=None
        if capture:
            service=self.instrumentation(row["workspace"])
            release=bool(owned and not others)
            monitor_receipt=self.monitor["command"](service,"stop" if release else "status",capture_id=capture)
            if release and self.monitor["alive"](monitor_receipt):
                return
            monitor_receipt["released_here"]=release
        receipt=dict(task_run_id=task,outcome=row["outcome"],closed_epoch=time.time(),jobs=job_receipts,monitor=monitor_receipt)
        with self.store.connect() as db:
            db.execute("UPDATE task_runs SET state=?,receipt=? WHERE id=?",(FINAL[row["outcome"]],json.dumps(receipt),task))
        self.pending.discard(task)

    def try_advance(self,task):
        try:
            self.advance(task)
        except Exception as exc:
            detail=dict(close_error=str(exc),retrying=True)
            with self.store.connect() as db:
                db.execute("UPDATE task_runs SET receipt=? WHERE id=? AND state='CLOSING'",(json.dumps(detail),task))
            self.manager.events.event(dict(event="task_close_retry",task=task,error=str(exc)))

    def tick(self):
        if self.pending:
            with self.manager.submission_lock:
                for task in list(self.pending):
                    self.try_advance(task)
