"""Single-use pairing and durable, directed feedback for a Studio conversation dock."""
import hashlib
import json
import math
import re
import secrets
import time

from deck import Conflict, check, uid

SCHEMA = """
CREATE TABLE IF NOT EXISTS session_links(
 id TEXT PRIMARY KEY, deck TEXT NOT NULL, room TEXT NOT NULL, state TEXT NOT NULL,
 code_hash TEXT UNIQUE, expires INTEGER NOT NULL, token_hash TEXT UNIQUE,
 host TEXT NOT NULL DEFAULT '', label TEXT NOT NULL DEFAULT '', thread TEXT NOT NULL DEFAULT '',
 version INTEGER NOT NULL DEFAULT 1, proof_hash TEXT, wake_endpoint TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS one_live_dock ON session_links(deck,room)
 WHERE state IN ('pairing','bound');
CREATE INDEX IF NOT EXISTS link_workspace ON session_links(deck,room);
CREATE TABLE IF NOT EXISTS session_messages(
 id TEXT PRIMARY KEY, link TEXT NOT NULL REFERENCES session_links(id), request TEXT NOT NULL,
 payload_hash TEXT NOT NULL, message TEXT NOT NULL, context TEXT NOT NULL, base INTEGER NOT NULL,
 state TEXT NOT NULL, reply TEXT NOT NULL DEFAULT '', reply_hash TEXT, result TEXT,
 wake TEXT NOT NULL DEFAULT 'ready', retry_at REAL NOT NULL DEFAULT 0, UNIQUE(link,request)
);
CREATE INDEX IF NOT EXISTS message_queue ON session_messages(link,state);
"""
HOSTS = ("Codex", "WorkBuddy", "DSH", "其他 MCP 客户端")
ACTIVE = "state IN ('pairing','bound')"
TTL = 600


def digest(value):
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_room(room):
    check(isinstance(room, str) and re.fullmatch(r"[0-9a-f]{32}", room), "无效的对话框标识")


def expire(db):
    db.execute("UPDATE session_links SET state='expired', code_hash=NULL, version=version+1 "
               "WHERE state='pairing' AND expires<=?", (int(time.time()),))


def current_link(db, deck, room):
    return db.execute(f"SELECT * FROM session_links WHERE deck=? AND room=? AND {ACTIVE}",
                      (deck, room)).fetchone()


def require_owner(db, deck, room, link):
    validate_room(room)
    row = current_link(db, deck, room)
    if not row or row["id"] != link:
        raise Conflict("连接已变更，请刷新对话框")
    return row


def authorize(db, credential):
    check(isinstance(credential, str) and 24 <= len(credential) <= 128, "无效或已撤销的会话凭据")
    row = db.execute("SELECT * FROM session_links WHERE token_hash=? AND state='bound'",
                     (digest(credential),)).fetchone()
    check(row is not None, "无效或已撤销的会话凭据")
    return row


def issue(store, deck, room):
    validate_room(room)
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        check(db.execute("SELECT 1 FROM decks WHERE id=?", (deck,)).fetchone(), "课件不存在")
        expire(db)
        prior = current_link(db, deck, room)
        check(not prior or prior["state"] == "pairing", "请先断开已绑定的会话")
        if prior:
            db.execute("UPDATE session_links SET state='revoked',code_hash=NULL,version=version+1 WHERE id=?", (prior["id"],))
        raw = secrets.token_hex(6).upper()
        code = "CP-" + "-".join(raw[i:i+4] for i in range(0, 12, 4))
        key, deadline = uid(), int(time.time()) + TTL
        db.execute("INSERT INTO session_links(id,deck,room,state,code_hash,expires) VALUES(?,?,?,'pairing',?,?)",
                   (key, deck, room, digest(code), deadline))
        return {"link": key, "code": code, "expires": deadline}


def pair(store, code, host, label="", thread=""):
    check(isinstance(code, str) and len(code) <= 32, "无效的配对码")
    check(host in HOSTS and isinstance(label, str) and len(label) <= 80, "无效的宿主或聊天名称")
    check(isinstance(thread, str) and len(thread) <= 160, "无效的聊天标识")
    token = secrets.token_urlsafe(32)
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT id,deck FROM session_links WHERE code_hash=? AND state='pairing' AND expires>?",
                         (digest(code.strip().upper()), int(time.time()))).fetchone()
        check(row, "配对码已使用、过期、作废或不属于此工作区")
        db.execute("UPDATE session_links SET state='bound',proof_hash=code_hash,code_hash=NULL,token_hash=?,host=?,label=?,thread=?,version=version+1 WHERE id=?",
                   (digest(token), host, label.strip(), thread.strip(), row["id"]))
        return {"link": row["id"], "deck": row["deck"], "credential": token, "delivery": "mcp_wait",
                "next": "Keep the credential in this chat. Call studio_session_receive, then studio_session_ack and studio_session_reply."}


def disconnect(store, deck, room, link):
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        require_owner(db, deck, room, link)
        db.execute("UPDATE session_links SET state='revoked',token_hash=NULL,code_hash=NULL,version=version+1 WHERE id=?", (link,))
        db.execute("UPDATE session_messages SET state='cancelled' WHERE link=? AND state IN ('queued','offered','received')", (link,))
    return {}


def recover(store, deck, room, link):
    validate_room(room)
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        expire(db)
        check(not current_link(db, deck, room), "请先断开本对话框现有连接")
        changed = db.execute("UPDATE session_links SET room=?,version=version+1 WHERE id=? AND deck=? AND state='bound'",
                             (room, link, deck))
        check(changed.rowcount == 1, "此会话已断开或课件不匹配")
    return {}


def status(store, deck, room, since=None, include_recoverable=False):
    validate_room(room)
    with store.db() as db:
        db.execute("BEGIN")
        rows = db.execute("SELECT id,state,expires,host,label,thread,version,wake_endpoint FROM session_links WHERE deck=? AND room=? ORDER BY rowid",
                          (deck, room)).fetchall()
        live = next((r for r in reversed(rows) if r["state"] == "bound" or
                     (r["state"] == "pairing" and r["expires"] > time.time())), None)
        marker = f"{sum(r['version'] for r in rows)}:{live['id'] if live else '-'}"
        result = {"marker": marker, "link": None if not live else {
            k: live[k] for k in ("id","state","expires","host","label","thread")}}
        if live:
            result["link"]["wakeEnabled"] = bool(live["wake_endpoint"])
        if marker != since:
            messages = db.execute("""
                SELECT m.id,m.state,m.message,m.reply,m.context,m.base,m.wake,l.host,l.label FROM session_messages m
                JOIN session_links l ON m.link=l.id WHERE l.deck=? AND l.room=? ORDER BY m.rowid DESC LIMIT 100
            """, (deck,room))
            result["messages"] = [{**dict(m), "context": json.loads(m["context"])} for m in messages]
        if include_recoverable:
            result["recoverable"] = [dict(x) for x in db.execute(
                "SELECT id,host,label FROM session_links WHERE deck=? AND room<>? AND state='bound' ORDER BY rowid DESC",
                (deck, room))]
        return result


def send(store, deck, room, link, request, message, base, context):
    check(isinstance(request,str) and 0<len(request)<=100, "缺少发送请求标识")
    check(isinstance(message,str) and 0<len(message.strip())<=8000, "留言为空或过长")
    check(type(base) is int and base >= 0 and isinstance(context,dict), "无效的发送上下文")
    key_hash = digest(packed([message,base,context]))
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = require_owner(db, deck, room, link)
        check(row["state"] == "bound", "聊天尚未完成绑定")
        old = db.execute("SELECT id,payload_hash FROM session_messages WHERE link=? AND request=?", (link,request)).fetchone()
        if old:
            check(old["payload_hash"] == key_hash, "请求 ID 已用于其他留言")
            return {"id":old["id"]}
        doc = store.version(db,deck,base)
        slide_id = context.get("slide")
        check(isinstance(slide_id,str), "无效的页面标识")
        page = doc["slides"].get(slide_id)
        selected = context.get("selection")
        step = context.get("step",0)
        region = context.get("region")
        check(page is not None and isinstance(selected,list) and len(selected)<=len(page["elements"])
              and all(isinstance(i,str) and i in page["elements"] for i in selected)
              and len(set(selected))==len(selected), "选区已失效")
        check(type(step) is int and step >= 0, "无效的播放步")
        check(region is None or (isinstance(region,list) and len(region)==4 and all(type(n) in (int,float) and math.isfinite(n) for n in region) and min(region[2:])>=0), "无效的框选坐标")
        ctx = {"slide":slide_id,"selection":selected,"step":step,"region":region,"pageNumber":doc["order"].index(slide_id)+1}
        fid = uid()
        db.execute("INSERT INTO session_messages(id,link,request,payload_hash,message,context,base,state,retry_at) VALUES(?,?,?,?,?,?,?,'queued',?)",
                   (fid,link,request,key_hash,message,packed(ctx),base,time.time()+2))
        touch(db,link)
        return {"id":fid}


def touch(db, link):
    db.execute("UPDATE session_links SET version=version+1 WHERE id=?", (link,))


def receive(store, credential, timeout=20, stop=None):
    check(type(timeout) is int and 0<=timeout<=45, "等待时长须为 0–45 秒")
    deadline=time.monotonic()+timeout
    while True:
        with store.db() as db:
            binding = authorize(db,credential)
            m = db.execute("SELECT id FROM session_messages WHERE link=? AND state IN ('queued','offered','received') ORDER BY rowid LIMIT 1",
                           (binding["id"],)).fetchone()
        if m:
            with store.db() as db:
                db.execute("BEGIN IMMEDIATE")
                binding = authorize(db,credential)
                m = db.execute("SELECT * FROM session_messages WHERE link=? AND state IN ('queued','offered','received') ORDER BY rowid LIMIT 1",
                               (binding["id"],)).fetchone()
                if not m:
                    continue
                if m["state"] == "queued":
                    db.execute("UPDATE session_messages SET state='offered' WHERE id=?", (m["id"],))
                    touch(db,binding["id"])
                doc = store.version(db,binding["deck"],m["base"])
                ctx = json.loads(m["context"])
                page = doc["slides"][ctx["slide"]]
                ids = ctx["selection"] or ([] if ctx["region"] is not None else page["order"])
                summary = []
                for i in ids[:64]:
                    obj = page["elements"][i]
                    data = {k:obj[k] for k in ("name","type","x","y","w","h") if k in obj}
                    summary.append({"id":i,**data,"text":obj.get("text","")[:2000]})
                return {"status":"message","feedbackId":m["id"],"link":binding["id"],"deck":binding["deck"],
                        "title":doc["title"],"revision":m["base"],"pageTitle":page["title"],
                        "canvas":[doc["width"],doc["height"]],"context":ctx,"objects":summary,
                        "objectCount":len(ids),"message":m["message"],"redelivery":m["state"]!="queued"}
        remaining = deadline-time.monotonic()
        if remaining<=0: return {"status":"timeout"}
        interval=min(.25,remaining)
        if stop is None: time.sleep(interval)
        elif stop.wait(interval): return {"status":"cancelled"}


def ack(store, credential, feedback_id):
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        b=authorize(db,credential)
        m=db.execute("SELECT state FROM session_messages WHERE link=? AND id=?", (b["id"],feedback_id)).fetchone()
        check(m and m["state"] not in {"queued","cancelled"}, "此会话尚未领取该留言")
        if m["state"]=="offered":
            db.execute("UPDATE session_messages SET state='received' WHERE id=?", (feedback_id,))
            touch(db,b["id"])
        return {"feedbackId":feedback_id,"state":"received" if m["state"]=="offered" else m["state"]}


def reply(store, credential, feedback_id, reply, outcome="resolved", edits=None, base=None):
    check(isinstance(reply,str) and 0<len(reply.strip())<=16000 and outcome in {"resolved","needs_input"}, "回复或结果无效")
    if edits is None: edits=[]
    check(isinstance(edits,list) and (not edits or outcome=="resolved"), "编辑与回复状态不匹配")
    signature=digest(packed([reply,outcome,edits,base]))
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        b=authorize(db,credential)
        m=db.execute("SELECT * FROM session_messages WHERE id=? AND link=?", (feedback_id,b["id"])).fetchone()
        check(m, "此留言不属于绑定会话")
        if m["reply_hash"]:
            if m["reply_hash"]!=signature: raise Conflict("此留言已经以另一回执完成")
            return json.loads(m["result"])
        check(m["state"] in {"offered","received"}, "请先领取该留言")
        slide=json.loads(m["context"])["slide"]
        check(all(isinstance(e,dict) and isinstance(e.get("path"),list) and len(e["path"])>=3
                  and e["path"][:2]==["slides",slide] for e in edits), "编辑超出留言对应的页面")
        revision=db.execute("SELECT revision FROM decks WHERE id=?", (b["deck"],)).fetchone()[0]
        if edits:
            cmd=dict(op="edit",actor=b["label"] or b["host"],base=m["base"] if base is None else base,
                     request="bound-"+feedback_id,edits=edits)
            revision=store.transact(b["deck"],cmd,connection=db)["revision"]
        result={"feedbackId":feedback_id,"state":outcome,"revision":revision}
        db.execute("UPDATE session_messages SET state=?,reply=?,reply_hash=?,result=? WHERE id=?",
                   (outcome,reply,signature,packed(result),feedback_id))
        touch(db,b["id"])
        return result
