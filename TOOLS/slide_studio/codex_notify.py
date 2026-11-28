"""Optional notification for an explicitly verified live Codex app-server thread."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from deck import check
import session_link as link


def endpoint():
    return link.digest(os.environ.get("CLASSPAGE_CODEX_WS") or os.environ.get("CLASSPAGE_CODEX_SOCKET") or "default-local-codex-proxy")


def invoke(mode, thread, proof=None):
    node = shutil.which("node")
    if not node: return "unavailable"
    command = [node, str(Path(__file__).with_name("codex_wake.cjs"))]
    try:
        res = subprocess.run(command, input=json.dumps({"action":mode,"thread":thread,"proof":proof}),
                             capture_output=True,text=True,encoding="utf8",timeout=9,
                             creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        return json.loads(res.stdout)["status"]
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return "uncertain" if mode=="notify" else "unavailable"


def configure(store, deck, room, link_id, enabled):
    check(type(enabled) is bool, "唤醒开关无效")
    with store.db() as db:
        row = link.require_owner(db,deck,room,link_id)
        check(row["state"]=="bound" and row["host"]=="Codex", "仅支持已绑定的 Codex 聊天")
    if enabled:
        check(re.fullmatch(r"[0-9a-fA-F-]{36}",row["thread"]), "配对时需提供当前 Codex 线程的准确 ID")
        outcome=invoke("verify",row["thread"],row["proof_hash"])
        if outcome!="verified":
            reasons={"no_proof":"目标线程最近用户消息中未找到本次配对提示词","not_loaded":"该线程没有在共享服务中运行"}
            raise ValueError(reasons.get(outcome,"无法验证同一 Codex 共享服务；连接继续使用 MCP 等待"))
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        link.require_owner(db,deck,room,link_id)
        db.execute("UPDATE session_links SET wake_endpoint=?,version=version+1 WHERE id=?",
                   (endpoint() if enabled else "",link_id))
    return {"enabled":enabled}


def dispatch_one(store):
    now=time.time()
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        item=db.execute("""
            SELECT m.id,m.link,l.thread FROM session_messages m JOIN session_links l ON l.id=m.link
             WHERE l.state='bound' AND l.wake_endpoint=? AND m.state='queued'
               AND m.wake IN ('ready','busy','unavailable') AND m.retry_at<=?
               AND NOT EXISTS (SELECT 1 FROM session_messages prior WHERE prior.link=m.link
                     AND prior.rowid<m.rowid AND prior.state IN ('queued','received','offered'))
             ORDER BY m.rowid LIMIT 1
        """,(endpoint(),now)).fetchone()
        if not item:return
        db.execute("UPDATE session_messages SET wake='dispatching' WHERE id=?",(item["id"],))
        link.touch(db,item["link"])
    outcome=invoke("notify",item["thread"])
    if outcome=="not_loaded":outcome="unavailable"
    if outcome not in {"sent","busy","unavailable","uncertain"}:outcome="uncertain"
    with store.db() as db:
        db.execute("UPDATE session_messages SET wake=?,retry_at=? WHERE id=? AND wake='dispatching'",
                   (outcome,now+(12 if outcome=="busy" else 40),item["id"]))
        link.touch(db,item["link"])


def watch(store, stop):
    while not stop.wait(1.5):
        try:
            # Most workspaces never enable wake-up. Keep their shared DB read-only when idle.
            with store.db() as db:
                enabled=db.execute("SELECT 1 FROM session_links WHERE state='bound' AND wake_endpoint=? LIMIT 1",
                                   (endpoint(),)).fetchone()
            if enabled:dispatch_one(store)
        except Exception:
            # A pending message remains readable through MCP even if the notification fails.
            if stop.wait(3):return
