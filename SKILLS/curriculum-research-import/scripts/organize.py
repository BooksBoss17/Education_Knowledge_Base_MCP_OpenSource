"""Render a reviewed BeMarkdown PDF into a source-owned collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from itertools import groupby
import re
import shutil
import sqlite3
import time
from contextlib import closing
import tempfile
from urllib.parse import quote, unquote

CATEGORY = "CURRICULUM_STANDARDS_AND_RESEARCH_PAPERS"
PAGE = re.compile(r"^<!-- page:(\d+) -->\s*$")
IMAGE = re.compile(r"""!\[(?:\\.|[^\]\\])*\]\(\s*(?:<(?P<angle>[^>\n]+)>|(?P<plain>[^\s)]+))(?:\s+(?:"[^"\n]*"|'[^'\n]*'))?\s*\)""")


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def safe_name(name):
    require(isinstance(name, str) and name.strip() == name and name and len(name) <= 150
            and not re.search(r'[<>:"/\\|?*\x00-\x1f]|[ .]$', name)
            and not re.match(r'^(CON|PRN|AUX|NUL|COM\d|LPT\d)(\.|$)', name, re.I), "Invalid filename")
    return name


def inventory(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def below(root, *parts):
    path = root.joinpath(*parts)
    require(path.resolve() == path and path.is_relative_to(root), "Redirected output path")
    return path


def join_pages(chunks, joins, notes):
    """Apply explicit, source-boundary decisions to corrected page text."""
    for page in chunks:
        if page not in joins or page + 1 not in chunks:
            continue
        decision = joins.pop(page)
        tail, head = decision["tail"], decision["head"]
        separator = decision.get("separator", "")
        require(tail.strip() and head.strip() and decision["reason"].strip()
                and separator in ("", " "), "Invalid page join")
        left, right = chunks[page].rstrip(), chunks[page + 1].lstrip()
        require(left.count(tail) == right.count(head) == 1, "Ambiguous page join")
        before, _, footer = left.partition(tail)
        header, _, after = right.partition(head)
        require(footer.strip() == decision["footer"].strip()
                and header.strip() == decision["header"].strip(), "Page margins changed")
        for origin, margin in ((page, footer), (page + 1, header)):
            if margin.strip():
                notes.append((origin, margin.strip()))
        require(not before.strip() or before.endswith("\n\n"), "Tail must start at a paragraph boundary")
        chunks[page] = before.rstrip()
        chunks[page + 1] = f"<!-- joined-pages:{page}-{page+1} -->\n\n" + tail + separator + head + after


def normalize_headings(text):
    """Map the source heading hierarchy under the module H1."""
    levels = []
    output = []
    fence = ""
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" ")
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence:
            if stripped.strip() and set(stripped.strip()) == {fence[0]} and len(stripped.strip()) >= len(fence):
                fence = ""
        elif marker:
            fence = marker[1]
        else:
            heading = re.match(r"^(#{1,6})(?=[ \t]+\S)", line)
            if heading:
                level = len(heading[1])
                while levels and levels[-1] >= level:
                    levels.pop()
                levels.append(level)
                line = "#" * min(6, len(levels) + 1) + line[level:]
        output.append(line)
    return "".join(output)


def build(args, dest):
    source = args.source.resolve(strict=True)
    pkg = args.package.resolve(strict=True)
    plan = read(args.plan)
    manifest = read(pkg / "package_manifest.json")
    report = read(pkg / "conversion_report.json")
    require(manifest["complete"] and source.suffix.lower() == ".pdf", "Complete PDF package required")
    source_sha = sha(source)
    require(source_sha == manifest["source"]["sha256"] == report["source"]["sha256"] == plan["source_sha256"], "Source changed")
    for name, field in [("document.md","document_md"),("assets_manifest.jsonl","assets_manifest"),("conversion_report.json","conversion_report")]:
        require(sha(pkg/name) == manifest["integrity"][field+"_sha256"], name+" changed")
    require(plan["document_sha256"] == manifest["integrity"]["document_md_sha256"], "Review belongs to another transcription")
    require(plan["kind"] in ("curriculum_standard","teaching_research"), "Unknown category")
    title = safe_name(plan["title"])
    if report["review_items"]:
        require(plan.get("report_sha256") == manifest["integrity"]["conversion_report_sha256"],
                "Review decisions belong to another conversion report")
    count = report["document"]["pages"]
    require(not report.get("errors") and not plan["unresolved"], "Unresolved review or conversion errors")
    for field, size in [("pages", count), ("decisions", len(report["review_items"]))]:
        notes = plan[field]
        require(set(notes) == {str(i+1) for i in range(size)} and all(isinstance(v,str) and v.strip() for v in notes.values()), "Incomplete "+field)
    lines = (pkg/"document.md").read_text(encoding="utf8").splitlines(keepends=True)
    page_of = []
    anchors = []
    for line in lines:
        marker = PAGE.fullmatch(line)
        if marker:
            anchors.append(int(marker[1]))
        page_of.append(anchors[-1] if anchors else 1)
    require(anchors == list(range(1,count+1)), "Unexpected PDF page markers")
    sections = plan["parts"]
    starts = [s["start_line"] for s in sections]
    require(starts and starts[0]==1 and starts==sorted(set(starts)) and all(type(i) is int and i<=len(lines) for i in starts), "Invalid section boundaries")
    rows = (json.loads(line) for line in (pkg/"assets_manifest.jsonl").read_text(encoding="utf8").splitlines() if line.strip())
    declared = {row["relative_path"]: row["content_sha256"] for row in rows if row.get("relative_path")}
    page_text = {page: "".join(lines[i] for i in indices)
                 for page, indices in groupby(range(len(lines)), key=page_of.__getitem__)}
    image_sources = {}
    def image(match):
        name = match.lastgroup
        ref = Path(unquote(match[name]))
        require(ref.parts and ref.parts[0] in ("assets","evidence","_bemarkdown","DIAGRAMS") and ".." not in ref.parts, "Invalid asset reference")
        src = below(pkg,ref)
        uid = sha(src)
        require(uid == declared.get(ref.as_posix()), "Asset missing from manifest or changed")
        filename = uid + src.suffix.lower()
        diagram = dest/"DIAGRAMS"/filename
        diagram.parent.mkdir(exist_ok=True)
        if not diagram.exists():
            shutil.copyfile(src,diagram)
        require(sha(diagram)==uid,"Asset changed")
        image_sources[ref.as_posix()] = diagram.relative_to(dest).as_posix()
        left,right=match.span(name)
        return match[0][:left-match.start()]+image_sources[ref.as_posix()]+match[0][right-match.start():]
    notes = []
    joins = {}
    for decision in plan.get("page_joins", []):
        page = decision["page"]
        require(type(page) is int and 1 <= page < count and page not in joins, "Invalid join page")
        joins[page] = decision
    modules=[]
    edits=list(plan["corrections"])
    for n,(part,start,end) in enumerate(zip(sections,starts,starts[1:]+[len(lines)+1]),1):
        header=safe_name(part["title"])
        filename=f"{n:02d}-{header}.md"
        body=[]
        chunks = {}
        for page, positions in groupby(range(start-1,end-1), key=page_of.__getitem__):
            chunk="".join(lines[i] for i in positions if not PAGE.fullmatch(lines[i]))
            for edit in edits[:]:
                if edit["page"] != page or edit["old"] not in chunk: continue
                original_page=page_text[page]
                require(edit["old"] and original_page.count(edit["old"])==1 and chunk.count(edit["old"])==1 and edit["reason"].strip(), "Ambiguous correction")
                chunk=chunk.replace(edit["old"],edit["new"],1)
                edits.remove(edit)
            chunks[page] = chunk
        # Record content coverage before a continuation moves into the preceding page.
        content_pages = [page for page, chunk in chunks.items() if chunk.strip()]
        join_pages(chunks, joins, notes)
        for page, chunk in chunks.items():
            chunk=IMAGE.sub(image,chunk)
            require("![" not in IMAGE.sub("",chunk) and not re.search(r"<img\b",chunk,re.I), "Unresolved image syntax")
            if chunk.strip():
                if not body:
                    first, separator, rest = chunk.strip().partition("\n")
                    heading = re.fullmatch(r"#{1,6}\s+(.+)", first.strip())
                    if heading and heading[1].strip() == header:
                        chunk = rest if separator else ""
                body.append(f"<!-- page:{page} -->\n\n"+chunk.strip())
        require(body,"Empty module")
        (dest/filename).write_text("# "+header+"\n\n"+normalize_headings("\n\n".join(body))+"\n",encoding="utf8",newline="\n")
        modules.append(dict(file=filename,start_line=start,end_line=end-1,first_page=content_pages[0],last_page=content_pages[-1]))
    require(not edits,"Unapplied correction")
    require(not joins, "Page join crosses a module or lacks an adjacent page")
    if notes:
        note_file = f"{len(modules)+1:02d}-出版与页边信息.md"
        note_body = []
        for page, margin in notes:
            require(not re.search(r"<img\b",margin,re.I) and "![" not in IMAGE.sub("",margin),
                    "Unresolved image syntax in publication notes")
            note_body.append(f"<!-- page:{page} -->\n\n" + IMAGE.sub(image, margin))
        (dest/note_file).write_text("# 出版与页边信息\n\n" + "\n\n".join(note_body) + "\n",
                                   encoding="utf8", newline="\n")
        modules.append(dict(file=note_file, kind="publication_notes", pages=[p for p, _ in notes],
                            first_page=min(p for p, _ in notes), last_page=max(p for p, _ in notes)))
    index = [f"# {title}", "", f"原件：{source.name}", ""]
    index += [f"- [{m['file'][:-3]}]({quote(m['file'])})" for m in modules]
    (dest/"00-索引.md").write_text("\n".join(index)+"\n",encoding="utf8",newline="\n")
    receipt=dict(schema="curriculum-research-import-v1",source=source.name,source_sha256=source_sha,
                 kind=plan["kind"],title=title,pages=count,modules=modules,images=image_sources,
                 review=plan,files=inventory(dest))
    (dest/"manifest.json").write_bytes(encode(receipt))
    return sha(dest/"manifest.json")


def check_capture(workspace, capture_id):
    database = below(workspace, "tmp", "runtime-monitor", "captures.sqlite3")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("""
            SELECT state, deadline, interval, (SELECT max(t) FROM samples WHERE run=runs.id)
            FROM runs WHERE id=?
        """, (capture_id,)).fetchone()
    now = time.time()
    require(row and row[0] == "running" and row[1] >= now and row[3] is not None
            and 0 <= now - row[3] <= max(5, 3 * row[2]), "No live samples for this workspace capture")


def main():
    parser=argparse.ArgumentParser()
    for flag in ("source","workspace","package","plan"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--publish")
    parser.add_argument("--capture-id", help="Check this workspace capture during a monitored import")
    args=parser.parse_args()
    root=args.workspace.resolve(strict=True)
    if args.capture_id:
        check_capture(root, args.capture_id)
    scratch=below(root,"tmp")
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        candidate=Path(tmp)/"document"
        candidate.mkdir()
        token=build(args,candidate)
        receipt=read(candidate/"manifest.json")
        category=below(root,"KNOWLEDGE_BASE",CATEGORY)
        target=below(category,safe_name(args.source.stem))
        preview=below(scratch,"curriculum-research-previews",token)
        if not args.publish:
            preview.parent.mkdir(exist_ok=True)
            try: candidate.rename(preview)
            except OSError:
                if not preview.is_dir() or inventory(preview)!=inventory(candidate): raise
        if args.publish:
            if args.capture_id:
                check_capture(root, args.capture_id)
            require(args.publish==token,"Preview changed")
            category.mkdir(parents=True,exist_ok=True)
            if target.exists():
                require(inventory(target)==inventory(candidate),"Existing document differs")
            backup=below(root,"Original_Backup",CATEGORY,receipt["source_sha256"],safe_name(args.source.name))
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():
                pending=Path(tmp)/"original"
                shutil.copyfile(args.source,pending)
                require(sha(pending)==receipt["source_sha256"],"Source changed during backup")
                try: os.link(pending,backup)
                except FileExistsError: pass
            require(sha(backup)==receipt["source_sha256"],"Backup differs")
            if not target.exists():
                try: candidate.rename(target)
                except OSError:
                    if not target.exists() or inventory(target)!=inventory(candidate): raise
        print(json.dumps(dict(preview_sha256=token,modules=receipt["modules"],images=receipt["images"],target=str(target),preview=str(preview),published=bool(args.publish)),ensure_ascii=False))


if __name__=="__main__":
    main()
