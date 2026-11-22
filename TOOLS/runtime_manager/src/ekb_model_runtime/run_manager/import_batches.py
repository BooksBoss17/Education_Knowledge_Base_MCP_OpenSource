"""Persistent PDF import manifests. Dispatch and client mutations share the manager submission lock."""
from collections import Counter
import json
import os
import sqlite3
import subprocess
import threading
from contextlib import closing
from pathlib import Path
import re
import time

from .common import ACTIVE, RunManagerError, canonical, file_sha, fingerprint

DONE = {"PUBLISHED", "SKIPPED"}
DDL = """
CREATE TABLE IF NOT EXISTS import_batches(
 id TEXT PRIMARY KEY, workspace TEXT NOT NULL, signature TEXT NOT NULL, roots TEXT NOT NULL,
 config TEXT NOT NULL, registry TEXT NOT NULL, desktop INTEGER NOT NULL, attention TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS import_docs(
 batch TEXT NOT NULL, ordinal INTEGER NOT NULL, source TEXT NOT NULL, sha TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'PENDING', attempt INTEGER NOT NULL DEFAULT 0,
 job TEXT, pages INTEGER, markdown TEXT, receipt_sha TEXT, detail TEXT,
 PRIMARY KEY(batch,ordinal));
CREATE TABLE IF NOT EXISTS import_pages(
 batch TEXT NOT NULL, ordinal INTEGER NOT NULL, page INTEGER NOT NULL, claim TEXT,
 expires REAL, note TEXT, PRIMARY KEY(batch,ordinal,page));
CREATE TABLE IF NOT EXISTS import_events(
 seq INTEGER PRIMARY KEY, batch TEXT NOT NULL, at REAL NOT NULL, event TEXT NOT NULL, detail TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS import_event_cursor ON import_events(batch,seq);
"""


def require(condition, code):
    if not condition:
        raise RunManagerError(code)


def event(db, batch, name, detail):
    db.execute("INSERT INTO import_events(batch,at,event,detail) VALUES(?,?,?,?)",
               (batch, time.time(), name, detail))


def local_file(root, relative):
    target = root / relative
    require(target.resolve() == target and target.is_relative_to(root) and target.is_file(), "INVALID_BATCH_FILE")
    return target


class ImportBatches:
    def __init__(self, manager):
        self.manager = manager
        self.store = manager.store
        with self.store.connect() as db:
            db.executescript(DDL)
        self.next_tick = 0

    def begin(self, batch_id, workspace, sources, config, input_roots, notify_desktop=True):
        require(re.fullmatch("[a-f0-9]{32}", batch_id) and isinstance(sources, list)
                and 0 < len(sources) <= 1000, "INVALID_BATCH_MANIFEST")
        task = self.manager.tasks.record(batch_id, workspace)
        require(task["state"] == "OPEN", "TASK_RUN_NOT_OPEN")
        registered = self.store.workspace(workspace)
        allowed = {Path(x) for x in registered["input_roots"]}
        roots = sorted({str(Path(x).resolve(strict=True)) for x in input_roots})
        require(all(Path(x) in allowed for x in roots) and "education-convert" in registered["program_ids"],
                "INPUT_ROOT_OUTSIDE_REGISTRATION")
        docs = []
        for item in sources:
            source = Path(item["source"]).resolve(strict=True)
            require(source.suffix.lower() == ".pdf" and any(source.is_relative_to(p) for p in roots)
                    and source.is_file(), "INVALID_BATCH_SOURCE")
            require(file_sha(source) == item["sha256"], "SOURCE_FINGERPRINT_MISMATCH")
            adopted = item.get("job_id")
            if adopted:
                require(re.fullmatch("[0-9a-f]{32}",adopted), "INVALID_ADOPTED_JOB")
                observed=self.manager.scoped_status(dict(workspace=workspace, job_id=adopted))
                payload=observed["payload"]
                require(observed["program_id"]=="education-convert" and payload.get("operation")=="convert"
                        and not payload.get("derivation") and Path(payload["source"])==source
                        and payload["source_sha256"]==item["sha256"], "ADOPTED_SOURCE_MISMATCH")
                with self.store.connect() as db:
                    require(db.execute("SELECT 1 FROM task_jobs WHERE task=? AND job=?",(batch_id,adopted)).fetchone(),
                            "ADOPTED_JOB_NOT_IN_TASK")
            docs.append((str(source), item["sha256"], adopted))
        require(len(set(Path(path) for path, _, _ in docs)) == len(docs), "DUPLICATE_BATCH_SOURCE")
        require(type(notify_desktop) is bool, "INVALID_NOTIFICATION_SETTING")
        signature = fingerprint([docs, config, roots, notify_desktop])
        with self.store.connect() as db:
            existing = db.execute("SELECT * FROM import_batches WHERE id=?", (batch_id,)).fetchone()
            if existing:
                require(existing["workspace"] == workspace and existing["signature"] == signature, "BATCH_ID_CONFLICT")
            else:
                db.execute("INSERT INTO import_batches(id,workspace,signature,roots,config,registry,desktop) VALUES(?,?,?,?,?,?,?)",
                           (batch_id, workspace, signature, canonical(roots), config, self.manager.registry["registry_fingerprint"],int(notify_desktop)))
                db.executemany("INSERT INTO import_docs(batch,ordinal,source,sha,job,state) VALUES(?,?,?,?,?,?)",
                               [(batch_id,i,path,sha,job,"CONVERTING" if job else "PENDING")
                                for i,(path,sha,job) in enumerate(docs,1)])
                event(db, batch_id, "REGISTERED", str(len(docs)))
        return self.status(batch_id, workspace)

    def verify_close(self,batch_id,workspace):
        with self.store.connect() as db:
            rows=db.execute("SELECT * FROM import_docs WHERE batch=? ORDER BY ordinal",(batch_id,)).fetchall()
            for row in rows:
                require(row["state"] in DONE, "BATCH_HAS_UNFINISHED_DOCUMENTS")
                if row["state"]=="PUBLISHED":
                    self.verify_product(db,row,Path(workspace),row["receipt_sha"])

    def hold(self, batch_id):
        # A user pause drains running work, but must not wait on unavailable resources.
        with self.store.connect() as db:
            docs = db.execute("""SELECT d.job FROM import_docs d JOIN jobs j ON j.job_id=d.job
                WHERE d.batch=? AND d.state='CONVERTING' AND j.scope_key LIKE 'batch/' || d.batch || '/%'
                AND NOT EXISTS (SELECT 1 FROM task_jobs l WHERE l.job=d.job AND l.task!=d.batch)
                AND NOT EXISTS (SELECT 1 FROM adopters a WHERE a.job_id=d.job)""", (batch_id,)).fetchall()
        for row in docs:
            self.store.transition(row["job"], "CANCELLED", expected=("QUEUED","WAITING_RESOURCES"),
                                  error={"code":"BATCH_PAUSE_BEFORE_START"})

    def status(self, batch_id, workspace):
        task = self.manager.tasks.status(batch_id, workspace)
        with self.store.connect() as db:
            require(db.execute("SELECT 1 FROM import_batches WHERE id=? AND workspace=?", (batch_id, workspace)).fetchone(),
                    "UNKNOWN_IMPORT_BATCH")
            docs = [dict(row) for row in db.execute("""
                SELECT d.ordinal,d.source,d.sha,d.state,d.job,j.state AS job_state,d.attempt,d.pages,d.receipt_sha,d.detail,
                (SELECT count(*) FROM import_pages p WHERE p.batch=d.batch AND p.ordinal=d.ordinal AND note IS NOT NULL) AS reviewed
                FROM import_docs d LEFT JOIN jobs j ON j.job_id=d.job WHERE batch=? ORDER BY ordinal""",(batch_id,))]
        monitor = self.health(workspace, task["capture_id"])
        counts = Counter(d["state"] for d in docs)
        if task["state"]!="OPEN":
            state=task["state"]
        elif any(d["state"] in ("PENDING","CONVERTING") for d in docs):
            state="PROCESSING"
        elif counts["FAILED"]:
            state="NEEDS_ATTENTION"
        elif counts["WAITING_AGENT"]:
            state="WAITING_AGENT"
        elif counts["READY_TO_PUBLISH"]:
            state="READY_TO_PUBLISH"
        else:
            state="READY_TO_CLOSE"
        return dict(batch_id=batch_id, state=state, task_state=task["state"], capture_id=task["capture_id"], documents=docs,
                    counts=dict(counts), monitor=monitor, ready_to_close=all(d["state"] in DONE for d in docs))

    def health(self, workspace, capture):
        if not capture:
            return {"state":"missing"}
        service = self.manager.tasks.instrumentation(workspace)
        path = service.workspace_manager.safe_path("tmp/runtime-monitor") / "captures.sqlite3"
        if not path.is_file():
            return {"state":"missing"}
        with closing(sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)) as db:
            db.row_factory=sqlite3.Row
            row=db.execute("SELECT *, (SELECT max(t) FROM samples WHERE run=runs.id) AS last_sample FROM runs WHERE id=?",
                           (capture,)).fetchone()
        if row is None:
            return {"state":"missing"}
        state=row["state"]
        if state=="running":
            if not self.manager.tasks.monitor["alive"](row):
                state="interrupted"
            elif row["deadline"] <= time.time():
                state="expired"
            elif row["last_sample"] is None or time.time()-row["last_sample"]>max(5,3*row["interval"]):
                state="starting" if time.time()-row["started"]<10 else "stale"
        return {"state":state,"last_sample":row["last_sample"]}

    def operation(self, action, workspace, batch_id=None, **args):
        ws = str(Path(workspace).resolve(strict=True))
        if action == "list":
            with self.store.connect() as db:
                rows=db.execute("""SELECT b.id AS batch_id,t.state AS task_state,t.started,b.attention FROM import_batches b
                                   JOIN task_runs t ON t.id=b.id WHERE b.workspace=? ORDER BY t.started DESC LIMIT 100""", (ws,))
                return {"batches":[dict(r) for r in rows]}
        if action == "begin":
            return self.begin(batch_id, ws, **args)
        status = self.status(batch_id, ws)
        if action == "status":
            return status
        with self.store.connect() as db:
            if action == "events":
                rows = db.execute("SELECT seq,at,event,detail FROM import_events WHERE batch=? AND seq>? ORDER BY seq LIMIT 100",
                                  (batch_id,args.get("after",0)))
                return {"events":[dict(row) for row in rows]}
            if action == "pages":
                rows=db.execute("SELECT page,note,expires FROM import_pages WHERE batch=? AND ordinal=? AND page>? ORDER BY page LIMIT 100",
                                (batch_id,args["ordinal"],args.get("after",0)))
                pages = []
                size = 0
                for row in rows:
                    item = dict(row)
                    size += len(canonical(item).encode("utf8"))
                    if size > 512 * 1024:
                        break
                    pages.append(item)
                return {"pages": pages}
            require(status["task_state"] == "OPEN", "TASK_RUN_NOT_OPEN")
            ordinal = args["ordinal"]
            row = db.execute("SELECT * FROM import_docs WHERE batch=? AND ordinal=?", (batch_id,ordinal)).fetchone()
            require(row is not None, "UNKNOWN_BATCH_DOCUMENT")
            if action == "skip":
                require(row["state"] not in ("CONVERTING","PUBLISHED") and args["reason"].strip() and len(args["reason"])<=500, "INVALID_BATCH_SKIP")
                if row["state"]=="SKIPPED":
                    require(row["detail"]==args["reason"], "SKIP_REASON_CONFLICT")
                    return status
                db.execute("UPDATE import_docs SET state='SKIPPED',detail=? WHERE batch=? AND ordinal=?",
                           (args["reason"], batch_id,ordinal))
                event(db,batch_id,"SKIPPED",f"{ordinal}: {args['reason']}")
            elif action == "published":
                require(row["state"] in ("READY_TO_PUBLISH","PUBLISHED"), "INCOMPLETE_REVIEW")
                target = self.verify_product(db, row, Path(ws), args["manifest_sha256"])
                db.execute("UPDATE import_docs SET state='PUBLISHED',detail=?,receipt_sha=? WHERE batch=? AND ordinal=?",
                           (str(target),args['manifest_sha256'],batch_id,ordinal))
                if row["state"] != "PUBLISHED":
                    event(db,batch_id,"PUBLISHED",str(ordinal))
            else:
                return self.review(db, row, action, args)
        return self.status(batch_id, ws)

    def review(self, db, doc, action, args):
        key = (doc["batch"], doc["ordinal"])
        require(doc["state"] in ("WAITING_AGENT","READY_TO_PUBLISH"), "DOCUMENT_NOT_REVIEWABLE")
        token = args["claim_id"]
        require(isinstance(token,str) and re.fullmatch("[a-f0-9]{32}",token), "INVALID_REVIEW_CLAIM")
        if action == "claim":
            old = db.execute("SELECT batch,ordinal,page,note,expires FROM import_pages WHERE claim=? ORDER BY page",(token,)).fetchall()
            active = db.execute("SELECT count(DISTINCT claim) FROM import_pages WHERE batch=? AND expires>? AND note IS NULL AND claim!=?",
                                (key[0],time.time(),token)).fetchone()[0]
            if old:
                require(all((r["batch"],r["ordinal"])==key for r in old), "CLAIM_SCOPE_MISMATCH")
                require(active < 3 or all(r["note"] is not None for r in old), "REVIEW_CLAIMS_BUSY")
                db.execute("UPDATE import_pages SET expires=? WHERE claim=? AND note IS NULL",(time.time()+900,token))
                return dict(claim_id=token,pages=[r["page"] for r in old if r["note"] is None], job_id=doc["job"])
            require(active < 3, "REVIEW_CLAIMS_BUSY")
            pages = [r[0] for r in db.execute("""SELECT page FROM import_pages WHERE batch=? AND ordinal=?
                        AND note IS NULL AND (claim IS NULL OR expires<=?) ORDER BY page LIMIT 4""",(*key,time.time()))]
            db.executemany("UPDATE import_pages SET claim=?,expires=? WHERE batch=? AND ordinal=? AND page=?",
                           [(token,time.time()+900,*key,page) for page in pages])
            return dict(claim_id=token,pages=pages,job_id=doc["job"])
        require(action=="reviewed","UNKNOWN_BATCH_ACTION")
        notes=args["notes"]
        require(isinstance(notes,dict) and 0<len(notes)<=4,"INVALID_REVIEW_NOTES")
        for page,note in notes.items():
            require(isinstance(note,str) and note.strip() and len(note)<=16000,"INVALID_REVIEW_NOTE")
            match=db.execute("SELECT claim,note,expires FROM import_pages WHERE batch=? AND ordinal=? AND page=?",(*key,page)).fetchone()
            require(match and match["claim"]==token and (match["note"]==note or match["note"] is None and match["expires"]>time.time()),
                    "STALE_REVIEW_CLAIM")
            db.execute("UPDATE import_pages SET note=? WHERE batch=? AND ordinal=? AND page=?",(note,*key,page))
        pending=db.execute("SELECT count(*) FROM import_pages WHERE batch=? AND ordinal=? AND note IS NULL",key).fetchone()[0]
        if not pending and doc["state"]!="READY_TO_PUBLISH":
            db.execute("UPDATE import_docs SET state='READY_TO_PUBLISH' WHERE batch=? AND ordinal=?",key)
            event(db,key[0],"READY_TO_PUBLISH",str(key[1]))
        return dict(remaining=pending,job_id=doc["job"])

    def verify_product(self, db, row, ws, expected):
        directory=ws/"KNOWLEDGE_BASE/CURRICULUM_STANDARDS_AND_RESEARCH_PAPERS"/Path(row["source"]).stem
        path=local_file(ws,directory.relative_to(ws)/"manifest.json")
        require(file_sha(path)==expected,"PUBLICATION_FINGERPRINT_CHANGED")
        product=json.loads(path.read_text("utf8"))
        plan=product["review"]
        require(product["schema"]=="curriculum-research-import-v1" and product["source_sha256"]==row["sha"]
                and plan["source_sha256"]==row["sha"] and plan["document_sha256"]==row["markdown"]
                and not plan["unresolved"] and product["pages"]==row["pages"],"PUBLICATION_SOURCE_MISMATCH")
        reviewed={str(r["page"]):r["note"] for r in db.execute("SELECT page,note FROM import_pages WHERE batch=? AND ordinal=?",
                                                               (row["batch"],row["ordinal"]))}
        require(plan["pages"]==reviewed,"PUBLICATION_REVIEW_MISMATCH")
        files=product["files"]
        require(product["modules"] and "00-索引.md" in files and all(m["file"] in files for m in product["modules"]),
                "PUBLICATION_INCOMPLETE")
        actual={p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and p!=path}
        require(set(files)==actual,"PUBLICATION_INVENTORY_MISMATCH")
        for name,digest in files.items():
            require(file_sha(local_file(directory,name))==digest,"PUBLICATION_FILE_CHANGED")
        backup=Path("Original_Backup/CURRICULUM_STANDARDS_AND_RESEARCH_PAPERS")/row["sha"]/Path(row["source"]).name
        require(file_sha(local_file(ws,backup))==row["sha"],"PUBLICATION_BACKUP_MISMATCH")
        return path

    def tick(self):
        if time.monotonic()<self.next_tick:
            return
        self.next_tick=time.monotonic()+1
        with self.manager.submission_lock:
            with self.store.connect() as db:
                batches=db.execute("""SELECT b.*,t.capture FROM import_batches b JOIN task_runs t ON t.id=b.id
                                      WHERE t.state='OPEN' ORDER BY t.started""").fetchall()
            for batch in batches:
                try:
                    self.advance(batch)
                except Exception as exc:
                    self.attention(batch["id"],"BATCH_NEEDS_ATTENTION",str(exc))

    def attention(self,batch,name,detail):
        message=f"{name}: {detail}"
        with self.store.connect() as db:
            changed=db.execute("UPDATE import_batches SET attention=? WHERE id=? AND attention!=?",
                               (message,batch,message)).rowcount
            if changed:
                event(db,batch,name,detail)
        if changed:
            with self.store.connect() as db:
                desktop=db.execute("SELECT desktop FROM import_batches WHERE id=?",(batch,)).fetchone()[0]
            if desktop and os.name=="nt":
                self.notify_desktop(batch,name)
            self.manager.events.event(dict(event=name,batch_id=batch,detail=detail))

    def notify_desktop(self, batch, name):
        # The durable events feed is authoritative if Windows suppresses desktop banners.
        script = (
            "$ErrorActionPreference='Stop';"
            "$null=[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime];"
            "$xml=[Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.XmlDocument,ContentType=WindowsRuntime]::new();"
            f"$xml.LoadXml('<toast><visual><binding template=\"ToastGeneric\"><text>Education MCP</text>"
            f"<text>Import batch {batch}: {name}. Open import_batch status.</text></binding></visual></toast>');"
            "$toast=[Windows.UI.Notifications.ToastNotification,Windows.UI.Notifications,ContentType=WindowsRuntime]::new($xml);"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Microsoft.Windows.PowerShell').Show($toast)"
        )
        def deliver():
            try:
                subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",script],check=True,
                               stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                               timeout=12,creationflags=subprocess.CREATE_NO_WINDOW)
                outcome="DESKTOP_NOTIFICATION_SUBMITTED"
            except (OSError,subprocess.SubprocessError):
                outcome="DESKTOP_NOTIFICATION_UNAVAILABLE"
            with self.store.connect() as db:
                event(db,batch,outcome,name)
        threading.Thread(target=deliver,daemon=True).start()

    def advance(self,batch):
        bid=batch["id"]
        with self.store.connect() as db:
            row=db.execute("""SELECT * FROM import_docs WHERE batch=? AND state IN ('PENDING','CONVERTING')
                              ORDER BY CASE state WHEN 'CONVERTING' THEN 0 ELSE 1 END,ordinal LIMIT 1""",(bid,)).fetchone()
        health=self.health(batch["workspace"],batch["capture"])
        if health["state"] in ("missing","expired","interrupted","stopped"):
            self.attention(bid,"MONITOR_INTERRUPTED",str(batch["capture"]))
            resumed=self.manager.tasks.begin(bid,batch["workspace"],False)
            with self.store.connect() as db:
                event(db,bid,"CAPTURE_RECOVERED",canonical({"previous":batch["capture"],"new":resumed["capture_id"]}))
            return
        if health["state"] not in ("running","starting"):
            self.attention(bid,"MONITOR_STALE",str(batch["capture"]))
            return
        if health["state"]=="starting":
            return
        if not row:
            with self.store.connect() as db:
                counts = {state: count for state,count in db.execute(
                    "SELECT state,count(*) FROM import_docs WHERE batch=? GROUP BY state",(bid,))}
            self.attention(bid,"AGENT_HANDOFF",canonical(counts))
            return
        key=(bid,row["ordinal"])
        if row["state"]=="PENDING":
            require(batch["registry"]==self.manager.registry["registry_fingerprint"],"BATCH_REGISTRY_CHANGED")
            scope=f"batch/{bid}/{row['ordinal']}/{row['attempt']}"
            old=self.store.find_scope("education-convert",batch["workspace"],scope)
            if old is None:
                try:
                    old=self.manager.submit("education-convert",batch["workspace"],
                         dict(operation="convert",source=row["source"],source_sha256=row["sha"],material_type=None,
                              book_title=None,input_roots=json.loads(batch["roots"])),
                         scope_key=scope,config_fingerprint=batch["config"],task_run_id=bid,wait_timeout=86400)
                except (RunManagerError,OSError) as exc:
                    self.set_state(key,"FAILED",str(exc))
                    self.attention(bid,"DOCUMENT_FAILED",str(row["ordinal"]))
                    return
            with self.store.connect() as db:
                db.execute("UPDATE import_docs SET state='CONVERTING',job=?,detail=NULL WHERE batch=? AND ordinal=?",
                           (old["job_id"],*key))
                event(db,bid,"SUBMITTED",old["job_id"])
            return
        job=self.store.status(row["job"])
        if job["state"] in ACTIVE:
            return
        if job["state"]=="SUCCEEDED":
            try:
                self.ready(row,Path(batch["workspace"]))
                self.attention(bid,"WAITING_AGENT","Review pages are available.")
            except (ValueError,KeyError,OSError,RunManagerError) as exc:
                self.set_state(key,"FAILED",str(exc))
            return
        code=(job.get("error") or {}).get("code")
        unused=job["state"]=="WAIT_EXPIRED" or code=="BATCH_PAUSE_BEFORE_START"
        if unused:
            with self.store.connect() as db:
                db.execute("UPDATE import_docs SET state='PENDING',attempt=attempt+1,detail=? WHERE batch=? AND ordinal=?",
                           (job["state"],*key))
                event(db,bid,"RETRY_PENDING",row["job"])
        else:
            detail = f"{job['state']}: {job.get('error')}"
            stdout = Path(batch["workspace"])/"tmp/bemarkdown/.mcp/jobs"/job["job_id"]/"stdout.json"
            if stdout.is_file() and stdout.stat().st_size<=65536:
                try:
                    error=json.loads(stdout.read_text("utf8")).get("error")
                    if isinstance(error,str):
                        detail += " "+error[:500]
                except (ValueError,OSError):
                    pass
            self.set_state(key,"FAILED",detail)
            self.attention(bid,"DOCUMENT_FAILED",str(row["ordinal"]))

    def ready(self,row,ws):
        job=row["job"]
        state=json.loads(local_file(ws,Path('tmp/bemarkdown/.mcp/jobs')/job/'state.json').read_text('utf8'))
        require(state["job_id"]==job and state["status"]=="SUCCEEDED","CONVERSION_RESULT_MISSING")
        pkg=Path(state["conversion"]["package_path"])
        require(pkg.is_relative_to(ws/"tmp/bemarkdown/r"/job[:12]) and pkg.resolve()==pkg,"INVALID_BATCH_PACKAGE")
        manifest=json.loads(local_file(pkg,'package_manifest.json').read_text('utf8'))
        require(manifest['complete'] and manifest['source']['sha256']==row['sha'],"PACKAGE_SOURCE_MISMATCH")
        for filename,field in [('document.md','document_md'),('assets_manifest.jsonl','assets_manifest'),('conversion_report.json','conversion_report')]:
            require(file_sha(local_file(pkg,filename))==manifest['integrity'][field+'_sha256'],"PACKAGE_CHANGED")
        report=json.loads((pkg/'conversion_report.json').read_text('utf8'))
        count=report['document']['pages']
        require(report['source']['sha256']==row['sha'] and not report.get('errors') and type(count) is int and count>0,"INVALID_PAGE_COUNT")
        with self.store.connect() as db:
            db.executemany("INSERT INTO import_pages(batch,ordinal,page) VALUES(?,?,?)",
                           ((row['batch'],row['ordinal'],page) for page in range(1,count+1)))
            db.execute("UPDATE import_docs SET state='WAITING_AGENT',pages=?,markdown=?,detail=? WHERE batch=? AND ordinal=?",
                       (count,manifest['integrity']['document_md_sha256'],str(pkg),row['batch'],row['ordinal']))
            event(db,row['batch'],"WAITING_AGENT",str(row['ordinal']))

    def set_state(self,key,state,detail):
        with self.store.connect() as db:
            db.execute("UPDATE import_docs SET state=?,detail=? WHERE batch=? AND ordinal=?",(state,detail,*key))
            event(db,key[0],state,f"{key[1]}: {detail}")
