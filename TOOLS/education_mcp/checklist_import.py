"""Review and publish source-bound knowledge checklists in an education workspace."""

from collections import Counter
from contextlib import closing
import json
import re
from pathlib import Path
import shutil
import sqlite3
import tempfile

import checklist_sources as sources
from exercise_import import assets_in, check, document, dump, inside, read, sha, verify_folder
from exercise_import import BANK, CATEGORY as HANDOUTS


CATEGORY = "STUDY_NOTES_AND_KNOWLEDGE_CHECKLISTS"
PAGE = re.compile(r"\s*<!--\s*page\s*:\s*(\d+)\s*-->\s*")
HEADING = re.compile(r"^\s*(?:#{1,6}\s+|【(?:知识|考点|规律|方法|易错))")
WORD_HEADING = re.compile(r"^\s*(?:[一二三四五六七八九十]+[、．.]|\d{1,2}[、．.])\s*([^。？；]{1,60})\s*$")
HASH = re.compile(r"[a-f0-9]{64}")


def fingerprint(value):
    return sha(dump(value).encode("utf-8"))


def archived_handout(ws, digest):
    check(isinstance(digest, str) and HASH.fullmatch(digest), "Invalid handout SHA-256")
    bank = inside(ws, BANK)
    check(bank.is_file(), "Handout has no published exercise document")
    with closing(sqlite3.connect(bank.as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("SELECT name,folder,markdown_sha FROM documents WHERE sha=?", (digest,)).fetchone()
    check(row, "Handout has no published exercise document")
    name, relative, text_sha = row
    folder = inside(ws, relative)
    check(folder.is_relative_to(inside(ws, "KNOWLEDGE_BASE/" + HANDOUTS)), "Unexpected handout location")
    manifest = read(inside(folder, "manifest.json"))
    check(manifest["source"] == name and manifest["sha256"] == digest
          and manifest["files"]["全文.md"] == text_sha, "Handout manifest mismatch")
    verify_folder(folder, manifest)
    backup = inside(ws, f"Original_Backup/{HANDOUTS}/{digest}/{name}")
    check(sources.file_sha(backup) == digest, "Handout original changed")
    return {"sha": digest, "name": name, "source": str(backup), "job": None,
            "text": (folder / "全文.md").read_text("utf-8"),
            "images": {ref: str(inside(folder, ref)) for ref in manifest["files"] if ref.startswith("DIAGRAMS/")},
            "backup": backup.relative_to(ws).as_posix(),
            "full_text": (folder / "全文.md").relative_to(ws).as_posix()}


def original_name(doc):
    origins = doc["origins"]
    if origins and all("member" in origin for origin in origins):
        member = Path(origins[0]["member"].replace("\\", "/")).name
        if not re.search(r'[<>:"/\\|?*\x00-\x1f]', member) and not Path(member).is_reserved():
            return member
    return doc["name"]


def load_document(ws, roots, entry, mode):
    job, archived = entry.get("job_id"), entry.get("handout_sha256")
    check(bool(job) != bool(archived), "Supply a job or a published handout fingerprint")
    if archived:
        check(mode == "handout", "A published handout requires handout mode")
        doc = archived_handout(ws, archived)
    else:
        doc = document(ws, job, "讲义", [*roots, ws])
        doc = {key: doc[key] for key in ("sha", "name", "source", "job", "text", "images")}
    doc["origins"] = []
    for origin in entry.get("origins", []):
        if "member" in origin:
            sources.verify_member(roots, origin, doc["sha"])
        else:
            check(sources.file_sha(sources.authorized(origin["path"], roots)) == doc["sha"], "Origin fingerprint mismatch")
        doc["origins"].append(origin)
    doc["name"] = original_name(doc)
    doc["text_sha256"] = sha(doc["text"].encode("utf-8"))
    check(doc["text"].strip(), "Transcription is empty")
    return doc


def read_draft(db, import_id):
    row = db.execute("SELECT input,review,revision,receipt FROM drafts WHERE id=?", (import_id,)).fetchone()
    check(row, "Unknown checklist import")
    return json.loads(row[0]), json.loads(row[1]), row[2], json.loads(row[3]) if row[3] else None


def prepare(ws, roots, db, mode, inputs):
    check(mode in {"standalone", "handout"} and 1 <= len(inputs) <= 2, "Supply one source or a related edition pair")
    docs = [load_document(ws, roots, entry, mode) for entry in inputs]
    check(len({doc["sha"] for doc in docs}) == len(docs), "Duplicate original in edition pair")
    group = fingerprint([mode, sorted(doc["sha"] for doc in docs)])
    import_id = fingerprint([group, sorted((doc["sha"], doc["text_sha256"]) for doc in docs)])[:32]
    db.execute("INSERT OR IGNORE INTO drafts(id,group_id,input,review,revision) VALUES(?,?,?,?,0)",
               (import_id, group, dump({"mode": mode, "documents": docs}), "{}"))
    return status(db, import_id)


def status(db, import_id):
    draft, reviews, revision, receipt = read_draft(db, import_id)
    docs = [{"doc": index, "name": doc["name"], "sha256": doc["sha"],
             "transcription_sha256": doc["text_sha256"], "job_id": doc["job"],
             "lines": len(doc["text"].splitlines()),
             "review": reviews.get(str(index), {})}
            for index, doc in enumerate(draft["documents"], 1)]
    active = db.execute("SELECT active FROM drafts WHERE id=?", (import_id,)).fetchone()[0]
    return {"import_id": import_id, "mode": draft["mode"], "revision": revision,
            "documents": docs, "receipt": receipt, "active": bool(active)}


def anchor_index(doc):
    page = None
    headings = {}
    for number, line in enumerate(doc["text"].splitlines(), 1):
        match = PAGE.fullmatch(line) if doc["name"].lower().endswith(".pdf") else None
        if match:
            page = int(match[1])
        heading = re.match(r"^\s*(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading[1])
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = heading[2]
        elif doc["name"].lower().endswith(".docx") and line.lstrip().startswith("**"):
            heading = WORD_HEADING.fullmatch(line.replace("**", "").strip())
            if heading:
                headings = {1: heading[1].strip()}
        yield number, line, page, list(headings.values())


def read_lines(doc, start, limit):
    total = len(doc["text"].splitlines())
    check(type(start) is int and 1 <= start <= total and type(limit) is int and 1 <= limit <= 250,
          "Invalid line range")
    return {"total": total, "lines": [
        {"line": number, "text": line, "physical_page": page, "headings": headings,
         "candidate_heading": bool(HEADING.match(line.replace("**", "")) or (
             line.lstrip().startswith("**") and WORD_HEADING.fullmatch(line.replace("**", "").strip())))}
        for number, line, page, headings in anchor_index(doc) if start <= number < start + limit
    ]}


def output_title(title):
    check(isinstance(title, str) and title.strip() and len(title) <= 160
          and not re.search(r"[\x00-\x1f<>]", title), "Invalid knowledge section title")
    return title.strip()


def compile_document(doc, review, complete):
    lines = list(anchor_index(doc))
    end = 0
    blocks = []
    check(type(review.get("complete")) is bool and isinstance(review.get("unresolved"), list)
          and isinstance(review.get("decisions"), list), "Invalid review")
    if complete:
        check(review["complete"] and not review["unresolved"], "Unresolved or incomplete review")
    for decision in review["decisions"]:
        start, stop = decision["span"]
        check(type(start) is int and type(stop) is int and end < start <= stop <= len(lines),
              "Line intervals must be ordered and nonoverlapping")
        check(not complete or start == end + 1, "Review leaves uncovered lines")
        end = stop
        value = decision["decision"]
        check(value in {"include", "exclude"}, "Invalid review decision")
        if value == "exclude":
            reason = decision.get("reason")
            check(isinstance(reason, str) and reason.strip(), "Exclusions require a reason")
            continue
        title = output_title(decision["title"])
        excerpt = lines[start-1:stop]
        origin = "\n".join(line for _, line, _, _ in excerpt if not PAGE.fullmatch(line)).strip()
        text = decision.get("text", origin)
        check(isinstance(text, str) and text.strip(), "Empty knowledge section")
        if text != origin:
            check(isinstance(decision.get("basis"), str) and decision["basis"].strip(), "Corrections require source evidence")
        references = set()
        assets_in(origin, lambda ref: references.add(ref) or ref)
        included = set()
        assets_in(text, lambda ref: included.add(ref) or ref)
        check(included == references and included <= doc["images"].keys(), "Knowledge section changes source image references")
        pages = sorted({page for _, _, page, _ in excerpt if page is not None})
        anchor = {"source_sha256": doc["sha"], "lines": [start, stop], "headings": excerpt[0][3]}
        if pages:
            anchor["physical_pages"] = pages
        if text != origin:
            anchor["correction_basis"] = decision["basis"]
        module = output_title(decision["module"]) if "module" in decision else None
        blocks.append((title, text, anchor, included, module))
    check(not complete or end == len(lines), "Review leaves uncovered lines")
    return blocks


def preview(draft, reviews):
    sections = []
    seen = {}
    for index, doc in enumerate(draft["documents"], 1):
        check(str(index) in reviews, "Missing source review")
        blocks = compile_document(doc, reviews[str(index)], True)
        counts = Counter(block[1] for block in blocks)
        for title, text, anchor, images, module in blocks:
            if index == 2 and counts[text] == 1 and text in seen:
                seen[text]["sources"].append(anchor)
                continue
            section = {"id": fingerprint([doc["sha"], anchor["lines"], text])[:20],
                       "title": title, "text": text, "sources": [anchor], "images": sorted(images), "module": module}
            if index == 1 and counts[text] == 1:
                seen[text] = section
            sections.append(section)
    digest = fingerprint([draft["mode"], [(doc["name"], doc["sha"], doc["text_sha256"], doc["origins"])
                                         for doc in draft["documents"]], reviews, sections])
    return sections, digest


def write_original(ws, doc):
    original = inside(ws, f"Original_Backup/{CATEGORY}/{doc['sha']}/{doc['name']}")
    if not original.exists():
        original.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=original.parent) as directory:
            staged = Path(directory) / "file"
            shutil.copyfile(doc["source"], staged)
            check(sources.file_sha(staged) == doc["sha"], "Original changed during copy")
            staged.replace(original)
    check(sources.file_sha(original) == doc["sha"], "Original backup fingerprint mismatch")
    return original.relative_to(ws).as_posix()


def render(title, sections):
    parts = [f"# {title}"]
    for section in sections:
        parts.extend([f'<a id="{section["id"]}"></a>\n## {section["title"]}', section["text"]])
    return ("\n\n".join(parts) + "\n").encode("utf-8")


def markdown_files(title, sections):
    if not any(section["module"] for section in sections):
        return {"知识清单.md": render(title, sections)}
    modules = {}
    for section in sections:
        modules.setdefault(section["module"] or "其他知识要点", []).append(section)
    files = {}
    index = [f"# {title}", ""]
    for number, (module, items) in enumerate(modules.items(), 1):
        name = f"模块{number:02}.md"
        files[name] = render(module, items)
        label = module.replace("[", "\\[").replace("]", "\\]")
        index.append(f"- [{label}]({name})")
    files["知识清单.md"] = ("\n".join(index) + "\n").encode("utf-8")
    return files


def publish(ws, import_id, draft, sections, digest):
    if not sections:
        return {"import_id": import_id, "preview_sha256": digest, "section_count": 0, "disposition": "no_knowledge"}
    docs = draft["documents"]
    entries = []
    for doc in docs:
        entry = {"name": doc["name"], "sha256": doc["sha"], "transcription_sha256": doc["text_sha256"],
                 "job_id": doc["job"], "origins": doc["origins"]}
        if draft["mode"] == "handout":
            handout = archived_handout(ws, doc["sha"])
            check(sha(handout["text"].encode("utf-8")) == doc["text_sha256"], "Published handout transcription differs")
            entry.update(backup=handout["backup"], full_text=handout["full_text"])
        else:
            entry["backup"] = write_original(ws, doc)
        entries.append(entry)
    name = Path(docs[0]["name"]).stem
    base = inside(ws, "KNOWLEDGE_BASE/" + CATEGORY)
    folder = inside(base, f"{name[:42]}__{import_id}")
    assets = {ref for section in sections for ref in section["images"]}
    image_inputs = {key: Path(path) for doc in docs for key, path in doc["images"].items() if key in assets}
    payload = markdown_files(name, sections)
    files = {**{ref: sha(data) for ref, data in payload.items()}, **{ref: Path(ref).stem for ref in assets}}
    manifest = {"schema": 1, "id": import_id, "preview_sha256": digest, "sources": entries, "files": files,
                "sections": [{key: value for key, value in section.items() if key != "text"} for section in sections]}
    if not folder.exists():
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as parent:
            stage = Path(parent) / "checklist"
            stage.mkdir()
            for filename, data in payload.items():
                (stage / filename).write_bytes(data)
            for ref, path in image_inputs.items():
                asset = inside(stage, ref)
                asset.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, asset)
            (stage / "manifest.json").write_text(dump(manifest), encoding="utf-8")
            verify_folder(stage, manifest)
            stage.rename(folder)
    verify_folder(folder, manifest)
    return {"import_id": import_id, "preview_sha256": digest, "folder": folder.relative_to(ws).as_posix(),
            "section_count": len(sections), "disposition": "published"}


def run(ws, roots, action, **args):
    ws = Path(ws).resolve(strict=True)
    if action == "inventory":
        return sources.inventory(args["paths"], roots, args.get("zip_encoding"))
    if action == "stage":
        check(HASH.fullmatch(args["sha256"]), "Invalid source fingerprint")
        return sources.stage(ws, roots, args["origin"], args["sha256"])
    db_path = inside(ws, "tmp/checklist-import/staging.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path, timeout=30)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS drafts(
                   id TEXT PRIMARY KEY, group_id TEXT NOT NULL, input TEXT NOT NULL, review TEXT NOT NULL,
                   revision INTEGER NOT NULL, receipt TEXT, active INTEGER NOT NULL DEFAULT 0)""")
        if action == "list":
            digest = args.get("sha256")
            rows = db.execute("SELECT id,revision,active,receipt,input FROM drafts ORDER BY id")
            return [{"import_id": key, "revision": rev, "active": bool(active),
                     "receipt": json.loads(receipt) if receipt else None}
                    for key, rev, active, receipt, stored in rows
                    if not digest or any(doc["sha"] == digest for doc in json.loads(stored)["documents"])]
        if action in {"prepare", "review", "publish"}:
            db.execute("BEGIN IMMEDIATE")
        if action == "prepare":
            return prepare(ws, roots, db, args["mode"], args["sources"])
        key = args["import_id"]
        draft, reviews, revision, receipt = read_draft(db, key)
        if action == "status":
            return status(db, key)
        if action in {"lines", "review"}:
            index = args["doc"]
            check(type(index) is int and 1 <= index <= len(draft["documents"]), "Invalid document ordinal")
            doc = draft["documents"][index-1]
            if action == "lines":
                return read_lines(doc, args.get("start", 1), args.get("limit", 80))
            check(receipt is None and type(args["revision"]) is int and args["revision"] == revision,
                  "Published or stale draft")
            review = args["review"]
            compile_document(doc, review, review["complete"])
            reviews[str(index)] = review
            db.execute("UPDATE drafts SET review=?,revision=revision+1 WHERE id=?", (dump(reviews), key))
            return status(db, key)
        check(action in {"preview", "publish"}, "Unknown action")
        sections, digest = preview(draft, reviews)
        if action == "preview":
            offset, limit = args.get("offset", 0), args.get("limit", 20)
            check(type(offset) is int and offset >= 0 and type(limit) is int and 1 <= limit <= 250, "Invalid page")
            return {"import_id": key, "preview_sha256": digest, "section_count": len(sections),
                    "sections": sections[offset:offset+limit]}
        check(digest == args["preview_sha256"], "Preview changed")
        if receipt is not None:
            verified = publish(ws, key, draft, sections, digest)
            check(all(receipt.get(field) == value for field, value in verified.items()), "Published receipt mismatch")
            return receipt
        group, = db.execute("SELECT group_id FROM drafts WHERE id=?", (key,)).fetchone()
        previous = db.execute("SELECT id FROM drafts WHERE group_id=? AND active=1", (group,)).fetchone()
        check(not previous or args.get("supersedes") == previous[0], "Existing revision requires explicit supersedes")
        if not sections:
            for doc in draft["documents"]:
                check(sources.file_sha(Path(doc["source"])) == doc["sha"], "Original changed since review")
        receipt = publish(ws, key, draft, sections, digest)
        if previous:
            receipt["supersedes"] = previous[0]
            db.execute("UPDATE drafts SET active=0 WHERE id=?", previous)
        db.execute("UPDATE drafts SET receipt=?,active=1 WHERE id=?", (dump(receipt), key))
        return receipt


SCHEMA = {
    "name": "checklist_import",
    "description": "Inventory selected knowledge materials, review source-bound line intervals and publish checklists.",
    "inputSchema": {
        "type": "object", "required": ["action"], "additionalProperties": False,
        "properties": {
            "action": {"enum": ["inventory", "stage", "prepare", "list", "status", "lines", "review", "preview", "publish"]},
            "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "origin": {"type": "object"}, "sha256": {"type": "string"}, "zip_encoding": {"type": "string"},
            "mode": {"enum": ["standalone", "handout"]},
            "sources": {"type": "array", "items": {"type": "object"}, "minItems": 1, "maxItems": 2},
            "import_id": {"type": "string"}, "supersedes": {"type": "string"},
            "doc": {"type": "integer", "minimum": 1}, "start": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 250}, "offset": {"type": "integer", "minimum": 0},
            "revision": {"type": "integer", "minimum": 0}, "review": {"type": "object"},
            "preview_sha256": {"type": "string"},
        },
    },
}
