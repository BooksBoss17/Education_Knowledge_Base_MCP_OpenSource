"""SQLite WAL registration with atomic active-task deduplication."""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

from .common import ACTIVE, RunManagerError, canonical, fingerprint


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "tasks.sqlite3"
        with self.connect() as database:
            database.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS workspaces(path TEXT PRIMARY KEY, programs TEXT NOT NULL, input_roots TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(
              job_id TEXT PRIMARY KEY, dedupe_key TEXT NOT NULL, state TEXT NOT NULL,
              program_id TEXT NOT NULL, workspace TEXT NOT NULL, data TEXT NOT NULL,
              created REAL NOT NULL, updated REAL NOT NULL, wait_deadline REAL NOT NULL,
              identity TEXT, exit_code INTEGER, error TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS active_task ON jobs(dedupe_key)
              WHERE state IN ('QUEUED','WAITING_RESOURCES','STARTING','RUNNING','CANCEL_REQUESTED');
            CREATE TABLE IF NOT EXISTS adopters(job_id TEXT, client_id TEXT, adopted REAL,
              PRIMARY KEY(job_id,client_id));
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT,at REAL,event TEXT);
            """)
            columns = {row["name"] for row in database.execute("PRAGMA table_info(jobs)")}
            if "scope_key" not in columns:
                database.execute("ALTER TABLE jobs ADD COLUMN scope_key TEXT")
                for row in database.execute("SELECT job_id,data FROM jobs"):
                    scope = json.loads(row["data"]).get("scope_key")
                    if scope is not None:
                        database.execute("UPDATE jobs SET scope_key=? WHERE job_id=?", (scope, row["job_id"]))
            database.execute("CREATE INDEX IF NOT EXISTS scope_lookup ON jobs(program_id,workspace,scope_key,created DESC)")

    @contextmanager
    def connect(self):
        database = sqlite3.connect(self.path, timeout=10)
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA busy_timeout=10000")
        database.execute("PRAGMA synchronous=FULL")
        try:
            yield database
        except BaseException:
            database.rollback()
            raise
        else:
            database.commit()
        finally:
            database.close()

    def workspace(self, workspace):
        with self.connect() as database:
            row = database.execute("SELECT * FROM workspaces WHERE path=?", (str(workspace),)).fetchone()
        if row is None:
            raise RunManagerError("WORKSPACE_NOT_REGISTERED")
        return {"path": row["path"], "program_ids": json.loads(row["programs"]), "input_roots": json.loads(row["input_roots"])}

    def register_workspace(self, workspace, programs, input_roots):
        with self.connect() as database:
            database.execute("INSERT INTO workspaces VALUES(?,?,?) ON CONFLICT(path) DO UPDATE SET programs=excluded.programs,input_roots=excluded.input_roots",
                             (str(workspace), canonical(programs), canonical(input_roots)))
        return self.workspace(workspace)

    def _snapshot(self, row):
        result = json.loads(row["data"])
        result.update(job_id=row["job_id"], state=row["state"], status=row["state"], created_epoch=row["created"],
                      updated_epoch=row["updated"], wait_deadline_epoch=row["wait_deadline"], exit_code=row["exit_code"])
        result["public_id"] = row["job_id"][:12]
        if row["identity"]:
            result["process_identity"] = json.loads(row["identity"])
        if row["error"]:
            result["error"] = json.loads(row["error"])
        return result

    def status(self, job_id):
        with self.connect() as database:
            row = database.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise RunManagerError("UNKNOWN_JOB")
        return self._snapshot(row)

    def list(self, states=None, limit=100):
        with self.connect() as database:
            if states:
                marks = ",".join("?" for _ in states)
                rows = database.execute(f"SELECT * FROM jobs WHERE state IN ({marks}) ORDER BY created LIMIT ?", (*states, limit)).fetchall()
            else:
                rows = database.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self._snapshot(row) for row in rows]

    def lookup(self, program_id, workspace, public_id):
        if not isinstance(public_id, str) or not re.fullmatch(r"[a-f0-9]{12}", public_id):
            raise RunManagerError("INVALID_PUBLIC_JOB_ID")
        with self.connect() as database:
            rows = database.execute("SELECT * FROM jobs WHERE program_id=? AND workspace=? AND job_id LIKE ? LIMIT 2",
                                    (program_id, workspace, public_id + "%")).fetchall()
        if len(rows) != 1:
            raise RunManagerError("UNKNOWN_JOB" if not rows else "AMBIGUOUS_PUBLIC_JOB_ID")
        return self._snapshot(rows[0])

    def find_scope(self, program_id, workspace, scope_key):
        with self.connect() as database:
            row = database.execute("SELECT * FROM jobs WHERE program_id=? AND workspace=? AND scope_key IS ? ORDER BY created DESC LIMIT 1",
                                   (program_id, workspace, scope_key)).fetchone()
        return self._snapshot(row) if row else None

    def submit(self, data, wait_timeout, retry_of=None):
        dedupe = fingerprint({key: data.get(key) for key in ("program_id", "workspace", "scope_key", "payload", "config_fingerprint", "program_fingerprint", "policy_fingerprint", "requirement", "run_timeout_seconds")})
        with self.connect() as database:
            database.execute("BEGIN IMMEDIATE")
            old = database.execute("SELECT * FROM jobs WHERE dedupe_key=? ORDER BY created DESC LIMIT 1", (dedupe,)).fetchone()
            if old and (old["state"] in ACTIVE or old["state"] == "INTERRUPTED" and retry_of != old["job_id"]):
                result = self._snapshot(old)
                result.update(deduplicated=True, retry_required=old["state"] == "INTERRUPTED")
                return result
            if retry_of is not None and (old is None or retry_of != old["job_id"] or old["state"] not in ("INTERRUPTED", "FAILED", "CANCELLED", "WAIT_EXPIRED", "CAPACITY_INSUFFICIENT")):
                raise RunManagerError("INVALID_RETRY_REFERENCE")
            identifier = uuid.uuid4().hex
            now = time.time()
            record = {**data, "job_id": identifier, "retry_of": retry_of,
                      "dedupe_fingerprint": dedupe, "job_folder": str(self.root / "jobs" / identifier)}
            database.execute("INSERT INTO jobs(job_id,dedupe_key,state,program_id,workspace,data,created,updated,wait_deadline,scope_key) VALUES(?,?,?,?,?,?,?,?,?,?)",
                             (identifier, dedupe, "QUEUED", data["program_id"], data["workspace"], canonical(record), now, now, now + wait_timeout, data.get("scope_key")))
        result = self.status(identifier)
        result["deduplicated"] = False
        return result

    def transition(self, job_id, state, *, error=None, identity=None, exit_code=None, updates=None, expected=None):
        with self.connect() as database:
            database.execute("BEGIN IMMEDIATE")
            row = database.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise RunManagerError("UNKNOWN_JOB")
            if expected and row["state"] not in expected:
                return self._snapshot(row)
            data = json.loads(row["data"])
            data.update(updates or {})
            saved_error = canonical(error) if error is not None else None if state in ("STARTING", "RUNNING", "SUCCEEDED") else row["error"]
            database.execute("UPDATE jobs SET state=?,data=?,updated=?,identity=COALESCE(?,identity),exit_code=COALESCE(?,exit_code),error=? WHERE job_id=?",
                             (state, canonical(data), time.time(), canonical(identity) if identity is not None else None,
                              exit_code, saved_error, job_id))
            if state != row["state"] or error is not None:
                database.execute("INSERT INTO events(job_id,at,event) VALUES(?,?,?)", (job_id, time.time(), canonical({"from": row["state"], "to": state, "error": error})))
                database.execute("DELETE FROM events WHERE job_id=? AND seq NOT IN (SELECT seq FROM events WHERE job_id=? ORDER BY seq DESC LIMIT 200)", (job_id, job_id))
        return self.status(job_id)

    def adopt(self, job_id, client_id):
        self.status(job_id)
        if not isinstance(client_id, str) or not 1 <= len(client_id) <= 128:
            raise RunManagerError("INVALID_CLIENT_ID")
        with self.connect() as database:
            database.execute("INSERT INTO adopters VALUES(?,?,?) ON CONFLICT(job_id,client_id) DO UPDATE SET adopted=excluded.adopted",
                             (job_id, client_id, time.time()))
        result = self.status(job_id)
        result["adopted_by"] = client_id
        return result
