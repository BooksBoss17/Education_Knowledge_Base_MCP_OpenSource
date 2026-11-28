"""Transaction store shared by the browser, CLI and agent."""

import base64
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from deck import Conflict, apply_edits, asset_ids, check, uid, validate
import session_link


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS decks(id TEXT PRIMARY KEY, revision INTEGER, undo TEXT, redo TEXT);
            CREATE TABLE IF NOT EXISTS versions(deck TEXT, revision INTEGER, body TEXT, actor TEXT, PRIMARY KEY(deck,revision));
            CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY, mime TEXT, content BLOB);
            CREATE TABLE IF NOT EXISTS requests(deck TEXT, id TEXT, payload TEXT, response TEXT, PRIMARY KEY(deck,id));
            CREATE TABLE IF NOT EXISTS feedback(id TEXT PRIMARY KEY,deck TEXT,context TEXT,message TEXT,state TEXT,reply TEXT);
            CREATE TABLE IF NOT EXISTS workspace_options(name TEXT PRIMARY KEY, enabled INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS opened_decks(seq INTEGER PRIMARY KEY AUTOINCREMENT, deck TEXT NOT NULL UNIQUE, opened_ms INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS opened_sources(fingerprint TEXT PRIMARY KEY, deck TEXT NOT NULL);
            """)
            db.executescript(session_link.SCHEMA)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def workspace_options(self, changes=None):
        options = {"grid": True, "compact": False}
        with self.db() as db:
            if changes is not None:
                check(isinstance(changes, dict) and all(
                    k in options and type(v) is bool for k, v in changes.items()
                ), "无效的工作区显示设置")
                db.executemany("INSERT OR REPLACE INTO workspace_options VALUES (?,?)", changes.items())
            for name, enabled in db.execute("SELECT name, enabled FROM workspace_options"):
                if name in options:
                    options[name] = bool(enabled)
        return options

    def asset(self, mime, content):
        check(
            mime
            in {
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/gif",
                "video/mp4",
                "video/webm",
                "audio/mpeg",
                "audio/wav",
                "audio/ogg",
            },
            "不支持的素材格式",
        )
        check(0 < len(content) <= 256 * 1024**2, "素材大小无效")
        key = hashlib.sha256(content).hexdigest()
        with self.db() as db:
            db.execute(
                "INSERT OR IGNORE INTO assets VALUES(?,?,?)", (key, mime, content)
            )
        return key

    @staticmethod
    def verify_assets(db, deck):
        for key in asset_ids(deck):
            check(
                db.execute("SELECT 1 FROM assets WHERE id=?", (key,)).fetchone(),
                "找不到素材 " + key,
            )

    def source_deck(self, fingerprint):
        with self.db() as db:
            row = db.execute("SELECT deck FROM opened_sources WHERE fingerprint=?", (fingerprint,)).fetchone()
            return row["deck"] if row else None

    def create(self, deck, fingerprint=None):
        validate(deck)
        key = uid()
        with self.db() as db:
            if fingerprint:
                db.execute("BEGIN IMMEDIATE")
                existing = db.execute("SELECT deck FROM opened_sources WHERE fingerprint=?", (fingerprint,)).fetchone()
                if existing:
                    return existing["deck"]
            self.verify_assets(db, deck)
            db.execute("INSERT INTO decks VALUES(?,0,'[]','[]')", (key,))
            db.execute(
                "INSERT INTO versions VALUES(?,0,?,?)", (key, dump(deck), "导入")
            )
            if fingerprint:
                db.execute("INSERT INTO opened_sources VALUES(?,?)", (fingerprint, key))
        return key

    def library(self):
        with self.db() as db:
            return [
                dict(row)
                for row in db.execute("""SELECT d.id,d.revision,json_extract(v.body,'$.title') AS title,
              json_array_length(v.body,'$.order') AS pages FROM decks d JOIN versions v ON v.deck=d.id AND v.revision=d.revision ORDER BY d.rowid DESC""")
            ]

    def home_items(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("""
                SELECT d.id, d.revision, json_extract(v.body,'$.title') AS title,
                       json_array_length(v.body,'$.order') AS pages, h.opened_ms
                  FROM decks d JOIN versions v ON v.deck=d.id AND v.revision=d.revision
                  LEFT JOIN opened_decks h ON h.deck=d.id
                 ORDER BY h.seq DESC, d.rowid DESC
            """)]

    def cover(self, key):
        with self.db() as db:
            row = db.execute("""
                SELECT json_extract(v.body,'$.width') AS width, json_extract(v.body,'$.height') AS height,
                       s.value AS page FROM decks d JOIN versions v ON v.deck=d.id AND v.revision=d.revision
                  JOIN json_each(v.body,'$.slides') s ON s.key=json_extract(v.body,'$.order[0]')
                 WHERE d.id=?
            """, (key,)).fetchone()
        check(row, "课件不存在")
        page = json.loads(row["page"])
        for obj in page["elements"].values():
            for field in ("html", "actions", "parameters"):
                obj.pop(field, None)
            if obj["type"] in {"audio", "video"}:
                poster = obj.get("poster")
                obj.update(type="image" if poster else "text", asset=poster or "", text=obj["name"])
                obj.pop("original", None)
        return {"width": row["width"], "height": row["height"], "page": {
            k: page[k] for k in ("elements", "order", "background", "backdrop") if k in page
        }}

    @staticmethod
    def version(db, key, revision):
        row = db.execute(
            "SELECT body FROM versions WHERE deck=? AND revision=?", (key, revision)
        ).fetchone()
        check(row, "未找到课件版本")
        return json.loads(row[0])

    def state(self, key, *, opened=False):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE" if opened else "BEGIN")
            meta = db.execute("SELECT * FROM decks WHERE id=?", (key,)).fetchone()
            check(meta, "课件不存在")
            if opened:
                db.execute("INSERT OR REPLACE INTO opened_decks(deck,opened_ms) VALUES(?,?)",
                           (key, time.time_ns() // 1_000_000))
            return {
                "id": key,
                "revision": meta["revision"],
                "canUndo": bool(json.loads(meta["undo"])),
                "canRedo": bool(json.loads(meta["redo"])),
                "deck": self.version(db, key, meta["revision"]),
                "feedback": [
                    {**dict(r), "context": json.loads(r["context"])}
                    for r in db.execute(
                        "SELECT id,context,message,state,reply FROM feedback WHERE deck=? ORDER BY rowid",
                        (key,),
                    )
                ],
            }

    def transact(self, key, cmd, *, connection=None):
        op = cmd["op"]
        req = cmd["request"]
        check(isinstance(req, str) and 0 < len(req) <= 100, "缺少请求 ID")
        packed = json.dumps(cmd, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with (self.db() if connection is None else nullcontext(connection)) as db:
            if connection is None:
                db.execute("BEGIN IMMEDIATE")
            seen = db.execute(
                "SELECT payload,response FROM requests WHERE deck=? AND id=?",
                (key, req),
            ).fetchone()
            if seen:
                check(seen[0] == packed, "请求 ID 已用于其他操作")
                return json.loads(seen[1])
            meta = db.execute("SELECT * FROM decks WHERE id=?", (key,)).fetchone()
            check(meta, "课件不存在")
            rev = meta["revision"]
            cur = self.version(db, key, rev)
            base = cmd["base"]
            check(type(base) is int and 0 <= base <= rev, "无效基础版本")
            undo, redo = json.loads(meta["undo"]), json.loads(meta["redo"])
            result = {}
            if op == "edit":
                cur = apply_edits(cur, self.version(db, key, base), cmd["edits"])
                self.verify_assets(db, cur)
                undo.append([rev, rev + 1])
                redo.clear()
            elif op in {"undo", "redo"}:
                if rev != base:
                    raise Conflict("课件已更新，请确认后再撤销或重做")
                src, dst = (undo, redo) if op == "undo" else (redo, undo)
                check(src, "没有可撤销或重做的操作")
                item = src.pop()
                cur = self.version(db, key, item[0 if op == "undo" else 1])
                dst.append(item)
            elif op == "comment":
                if "dockRoom" in cmd:
                    session_link.validate_room(cmd["dockRoom"])
                    session_link.expire(db)
                    if session_link.current_link(db, key, cmd["dockRoom"]):
                        raise Conflict("此对话框的连接已变更，请确认接收会话后再发送")
                ctx = cmd["context"]
                page = self.version(db, key, base)["slides"].get(ctx["slide"])
                check(
                    page
                    and isinstance(ctx["selection"], list)
                    and all(i in page["elements"] for i in ctx["selection"]),
                    "选区已失效",
                )
                check(
                    type(ctx.get("step", 0)) is int and ctx.get("step", 0) >= 0,
                    "无效播放步",
                )
                check(
                    isinstance(cmd["message"], str)
                    and 0 < len(cmd["message"].strip()) <= 8000,
                    "意见为空或过长",
                )
                fid = uid()
                db.execute(
                    "INSERT INTO feedback VALUES(?,?,?,?,?,?)",
                    (
                        fid,
                        key,
                        dump({**ctx, "base": base}),
                        cmd["message"],
                        "pending",
                        "",
                    ),
                )
                result["feedbackId"] = fid
            elif op not in {"reply", "clarify", "retry_note"}:
                raise ValueError("未知命令")
            if "feedbackId" in cmd or op in {"reply", "clarify", "retry_note"}:
                check(
                    op in {"edit", "reply", "clarify", "retry_note"}
                    and len(cmd.get("reply", "").strip()) > 0,
                    "缺少处理说明",
                )
                row = db.execute(
                    "UPDATE feedback SET state=?,reply=? WHERE id=? AND deck=? AND state='pending'",
                    (
                        {"retry_note": "pending", "clarify": "needs_input"}.get(
                            op, "resolved"
                        ),
                        cmd["reply"],
                        cmd["feedbackId"],
                        key,
                    ),
                )
                if row.rowcount != 1:
                    raise Conflict("意见已处理或不存在")
            db.execute(
                "INSERT INTO versions VALUES(?,?,?,?)",
                (key, rev + 1, dump(cur), str(cmd.get("actor", "agent"))[:100]),
            )
            db.execute(
                "UPDATE decks SET revision=?,undo=?,redo=? WHERE id=?",
                (rev + 1, dump(undo), dump(redo), key),
            )
            result["revision"] = rev + 1
            db.execute(
                "INSERT INTO requests VALUES(?,?,?,?)", (key, req, packed, dump(result))
            )
            return result

    def portable(self, key):
        deck = self.state(key)["deck"]
        assets = {}
        with self.db() as db:
            for aid in asset_ids(deck):
                r = db.execute(
                    "SELECT mime,content FROM assets WHERE id=?", (aid,)
                ).fetchone()
                assets[aid] = f"data:{r[0]};base64," + base64.b64encode(r[1]).decode()
        return {"deck": deck, "assets": assets}
