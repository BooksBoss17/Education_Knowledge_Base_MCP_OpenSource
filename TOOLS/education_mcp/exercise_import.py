"""Exercise import workflow shared by the MCP adapter and workspace clients."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from urllib.parse import unquote

CATEGORY = "ASSIGNMENTS_EXAMS_AND_HANDOUTS"
BANK = "KNOWLEDGE_BASE/EXERCISE_BANK_SQLite/exercises.sqlite3"
TAXONOMY = Path(__file__).with_name("exercise_labels.md")
TYPES = ["单选", "多选", "填空", "实验", "计算", "判断", "作图", "证明"]
IMAGE = re.compile(r"""!\[(?:\\.|[^\]\\])*\]\(\s*(?:<(?P<angle>[^>\n]+)>|(?P<plain>[^\s)]+))(?:\s+(?:"[^"\n]*"|'[^'\n]*'))?\s*\)""")
NUMBER = re.compile(r"^\s*(?:#{1,6}\s+)?(?:\*\*)?\s*((?:例题?|变式|练习)\s*[0-9一二三四五六七八九十]+|\d{1,3}\s*(?:[．、]|\.(?!\d)|\.(?=\d{4}年)))")
SECTION = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*([一二三四五六七八九十]+)[、．.]\s*(.+)")
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions(id TEXT PRIMARY KEY NOT NULL CHECK(length(id)=11 AND id NOT GLOB '*[^0-9]*'), type TEXT NOT NULL, content_hash TEXT NOT NULL, stem TEXT NOT NULL, options TEXT NOT NULL, answer TEXT NOT NULL, explanation TEXT NOT NULL, answer_status TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS match_content ON questions(content_hash);
CREATE TABLE IF NOT EXISTS sequences(type TEXT PRIMARY KEY, counter INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS documents(sha TEXT PRIMARY KEY, name TEXT NOT NULL, folder TEXT NOT NULL, markdown_sha TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(question_id TEXT REFERENCES questions, doc_sha TEXT REFERENCES documents, anchor TEXT, number TEXT, section TEXT, role TEXT, PRIMARY KEY(question_id,doc_sha,anchor,number,section,role));
CREATE TABLE IF NOT EXISTS labels(path TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS tags(question_id TEXT REFERENCES questions, path TEXT REFERENCES labels, scope TEXT, importance TEXT, PRIMARY KEY(question_id,path,scope,importance));
CREATE INDEX IF NOT EXISTS find_tag ON tags(path,importance);
CREATE TABLE IF NOT EXISTS images(sha TEXT PRIMARY KEY, extension TEXT NOT NULL, bytes BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS question_images(question_id TEXT REFERENCES questions, sha TEXT REFERENCES images, PRIMARY KEY(question_id,sha));
CREATE TABLE IF NOT EXISTS imports(id TEXT PRIMARY KEY, digest TEXT NOT NULL, receipt TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS generated_solution_parts(
    question_id TEXT NOT NULL REFERENCES questions, role TEXT NOT NULL,
    position INTEGER NOT NULL, markdown TEXT NOT NULL, verification TEXT NOT NULL,
    PRIMARY KEY(question_id,role,position));
"""

def check(value, message):
    if not value:
        raise ValueError(message)

def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(path.read_text("utf-8-sig"))

def inside(root, relative):
    base = root.resolve()
    path = root / relative
    check(path.resolve().is_relative_to(base), "Path outside workspace")
    return path

def catalog():
    levels = {}
    result = {}
    for line in TAXONOMY.read_text("utf-8-sig").splitlines():
        match = re.fullmatch(r"(#{1,3}) (.+)", line)
        if match:
            levels[len(match[1])] = match[2].strip()
        elif line.startswith("- "):
            leaf = line[2:].strip()
            check(leaf not in result, "Duplicate leaf in taxonomy")
            result[leaf] = "/".join([levels[i] for i in (1,2,3)] + [leaf])
    return result

def assets_in(text, replace):
    def sub(m):
        key = "angle" if m["angle"] is not None else "plain"
        value = replace(unquote(m[key]))
        return m[0][:m.start(key)-m.start()]+value+m[0][m.end(key)-m.start():]
    check("![" not in IMAGE.sub("",text) and not re.search(r"<img\b",text,re.I), "Unsupported image markup")
    return IMAGE.sub(sub,text)

def candidate_lines(text):
    lines=text.splitlines()
    # Some Word lists enumerate every paragraph, including headings and options.
    # Remove that outer numbering only in a sequential run with both kinds of evidence.
    outer={}
    run=[]
    def finish():
        if any(SECTION.match(rest) for _,_,rest in run) and sum(bool(NUMBER.match(rest)) for _,_,rest in run)>=2:
            outer.update((i,rest) for i,_,rest in run)
        run.clear()
    for i,line in enumerate(lines):
        match=re.match(r"^\s*(\d+)\.\s+(.*)$",line)
        if match:
            index=int(match[1])
            if run and index!=run[-1][1]+1:
                finish()
            run.append((i,index,match[2]))
    finish()
    result=[]
    section=""
    for i,line in enumerate(lines):
        content=outer.get(i,line).replace("**","")
        heading=SECTION.match(content)
        if heading:
            section=heading[1]
        match=NUMBER.match(content)
        if match:
            result.append({"line":i+1,"number":match[1],"section":section})
    return result

def document(ws, job, role, allowed):
    check(re.fullmatch("[a-f0-9]{32}",job) and role in {"题卷","答案","讲义","作业","解析版"}, "Invalid job or role")
    state = read(inside(ws, f"tmp/bemarkdown/.mcp/jobs/{job}/state.json"))
    check(state["status"]=="SUCCEEDED", "Conversion incomplete")
    source = Path(state["source"]).resolve(strict=True)
    check(any(source.is_relative_to(p.resolve()) for p in allowed), "Source outside allowed roots")
    pkg = Path(state["conversion"]["package_path"]).resolve(strict=True)
    check(pkg.is_relative_to(inside(ws,"tmp/bemarkdown").resolve()), "Unexpected package location")
    manifest = read(pkg/"package_manifest.json")
    docsha = sha(source.read_bytes())
    check(manifest["complete"] and manifest["source"]["sha256"]==state["source_sha256"]==docsha, "Source fingerprint mismatch")
    for filename, key in [("document.md","document_md"),("assets_manifest.jsonl","assets_manifest"),("conversion_report.json","conversion_report")]:
        check(sha(inside(pkg,filename).read_bytes())==manifest["integrity"][key+"_sha256"], "Package fingerprint mismatch: "+filename)
    declared = {row["relative_path"]:row["content_sha256"] for row in map(json.loads,(pkg/"assets_manifest.jsonl").read_text("utf8").splitlines()) if row.get("relative_path")}
    images = {}
    def bind(ref):
        p = inside(pkg,ref)
        digest = declared.get(ref)
        check(digest and sha(p.read_bytes())==digest, "Unbound image: "+ref)
        target = "DIAGRAMS/"+digest+p.suffix.lower()
        images[target]=str(p)
        return target
    text = assets_in((pkg/"document.md").read_text("utf8"),bind)
    candidates = candidate_lines(text)
    return dict(sha=docsha,role=role,job=job,name=source.name,source=str(source),text=text,images=images,candidates=candidates)

def prepare(ws, db, sources, allowed):
    check(isinstance(sources,list) and sources, "Supply source jobs")
    docs=[document(ws,s["job_id"],s["role"],allowed) for s in sources]
    check(len({d["sha"] for d in docs})==len(docs), "Duplicate original")
    key=sha(dump(sorted((d["sha"],d["role"]) for d in docs)).encode())[:32]
    payload=dump(docs)
    db.execute("INSERT OR IGNORE INTO sets VALUES(?,?,?,0)",(key,payload,"{}"))
    previous=read_set(db,key)[0]
    check([(d["sha"],sha(d["text"].encode())) for d in previous]==[(d["sha"],sha(d["text"].encode())) for d in docs], "Registered document order or transcription changed")
    return status(ws,db,key)

def read_set(db,key):
    row=db.execute("SELECT docs,review,revision FROM sets WHERE id=?",(key,)).fetchone()
    check(row,"Unknown import")
    return json.loads(row[0]),json.loads(row[1]),row[2]

def status(ws,db,key):
    docs,review,revision=read_set(db,key)
    answer_lines = {}
    for i,d in enumerate(docs,1):
        if d["role"] == "答案":
            for c in d["candidates"]:
                match_key = (c.get("section",""),c["number"].rstrip(".、．").strip())
                answer_lines.setdefault(match_key,[]).append([i,c["line"]])
    documents=[]
    for i,d in enumerate(docs,1):
        candidates=[]
        for c in d["candidates"]:
            match_key=(c.get("section",""),c["number"].rstrip(".、．").strip())
            candidates.append({**c,"answer_candidates":answer_lines.get(match_key,[]) if d["role"]!="答案" else []})
        documents.append(dict(doc=i,name=d["name"],role=d["role"],lines=len(d["text"].splitlines()),candidates=candidates))
    bank=inside(ws,BANK)
    receipt=None
    if bank.is_file():
        with closing(sqlite3.connect(bank.as_uri()+"?mode=ro",uri=True)) as published:
            if published.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='imports'").fetchone():
                row=published.execute("SELECT receipt FROM imports WHERE id=?",(key,)).fetchone()
                if row:
                    receipt=json.loads(row[0])
    return dict(import_id=key,revision=revision,documents=documents,reviewed=sorted(review.get("questions",{})),receipt=receipt,
                excluded=review.get("excluded",{}),unresolved=review.get("unresolved",[]),coverage_confirmed=review.get("coverage_confirmed",False))

def valid_span(docs, span):
    i,a,b=span
    check(all(type(n) is int for n in [i,a,b]) and 1<=i<=len(docs) and 1<=a<=b<=len(docs[i-1]["text"].splitlines()), "Invalid source span")
    return docs[i-1],a,b

def compile_review(docs,review,*,previously_published=False):
    vocabulary=catalog()
    check(review.get("coverage_confirmed") is True and review.get("unresolved")==[], "Review every original and resolve outstanding items")
    used=set()
    images={key:value for d in docs for key,value in d["images"].items()}
    compiled=[]
    seen=set()
    questions=review.get("questions",{})
    check(isinstance(questions,dict) and questions,"No reviewed questions")
    for key,q in questions.items():
        typ=q["type"];check(typ in TYPES and q["number"].strip(),"Missing type or original number")
        refs=[]; texts={}; generated=[]
        for role,segments in q["parts"].items():
            check(role in {"stem","options","answer","explanation"} and isinstance(segments,list),"Invalid question parts")
            chunks=[]
            for position,seg in enumerate(segments):
                if seg.get("origin")=="agent":
                    check(role in {"answer","explanation"} and set(seg)=={"origin","text","verification"},
                          "Agent solutions require text and verification without a source span")
                    check(all(isinstance(seg[k],str) and seg[k].strip() for k in ("text","verification")),
                          "Empty agent solution or verification")
                    chunks.append(seg["text"])
                    generated.append(dict(role=role,position=position,text=seg["text"],verification=seg["verification"]))
                    continue
                check(seg.get("origin","source")=="source","Unknown segment origin")
                doc,a,b=valid_span(docs,seg["span"])
                if role=="stem":
                    check(doc["role"]!="答案","Answer-only source cannot supply the question stem")
                source="\n".join(doc["text"].splitlines()[a-1:b])
                content=seg.get("text",source)
                check(isinstance(content,str) and content.strip(),"Empty reviewed text")
                if content!=source:check(str(seg.get("basis","")).strip(),"Corrected excerpt requires source review basis")
                chunks.append(content); refs.append((doc["sha"],f"L{a}-L{b}",role))
                for item in doc["candidates"]:
                    if a<=item["line"]<=b:used.add((seg["span"][0],item["line"]))
            texts[role]="\n\n".join(chunks)
        check(texts.get("stem","").strip() and (typ not in TYPES[:2] or texts.get("options","").strip()),"Missing stem or choice options")
        has_answer=bool(texts.get("answer","").strip())
        agent_answer=any(s["role"]=="answer" for s in generated)
        expected_status="Agent补写" if agent_answer else "已收录"
        legacy_missing=previously_published and not has_answer and q["answer_status"]=="原件无答案"
        check((has_answer and q["answer_status"]==expected_status) or legacy_missing,"Answer or answer provenance missing")
        check(previously_published or texts.get("explanation","").strip(),"Supply the missing explanation before publishing")
        tags=[]
        for t in q["tags"]:
            check(t["leaf"] in vocabulary and t["importance"] in ("主考","关联") and isinstance(t.get("scope","整题"),str),"Unknown tag or importance")
            tags.append((vocabulary[t["leaf"]],t.get("scope","整题"),t["importance"]))
        check(any(t[2]=="主考" for t in tags),"Missing primary tag")
        links={}
        def asset(ref):
            check(ref in images,"Unknown question image: "+ref)
            src=Path(images[ref]);uid=Path(ref).stem
            check(sha(src.read_bytes())==uid,"Question asset changed")
            links[uid]=str(src)
            return "asset:"+uid
        texts={k:assets_in(t,asset) for k,t in texts.items()}
        loc=(next(docsha for docsha,_,role in refs if role=="stem"),q.get("section",""),q["number"])
        check(loc not in seen,"Duplicate section and number");seen.add(loc)
        compiled.append(dict(key=key,number=q["number"],section=q.get("section",""),type=typ,refs=refs,text=texts,tags=sorted(set(tags)),images=links,answer_status=q["answer_status"]))
        if generated:
            compiled[-1]["generated"]=[{**part,"text":assets_in(part["text"],asset)} for part in generated]
    excluded=review.get("excluded",{})
    candidates={(n,c["line"]) for n,d in enumerate(docs,1) for c in d["candidates"]}
    for k,reason in excluded.items():
        loc=tuple(map(int,k.split(":")))
        check(loc in candidates and loc not in used and isinstance(reason,str) and reason.strip(),"Invalid exclusion: "+k)
        used.add(loc)
    check(not candidates-used,"Unaccounted candidate lines: "+str(sorted(candidates-used)[:25]))
    return compiled


def verify_folder(folder, expected):
    check(folder.is_dir() and read(inside(folder,"manifest.json"))==expected,"Existing full document differs")
    for name,digest in expected["files"].items():
        check(sha(inside(folder,name).read_bytes())==digest,"Changed full document file: "+name)

def store_document(ws,doc):
    name=Path(doc["name"]).stem[:55]+"__"+doc["sha"][:12]
    base=inside(ws,"KNOWLEDGE_BASE/"+CATEGORY);base.mkdir(parents=True,exist_ok=True)
    folder=inside(base,name)
    files={"全文.md":sha(doc["text"].encode()),**{ref:Path(ref).stem for ref in doc["images"]}}
    manifest=dict(source=doc["name"],sha256=doc["sha"],files=files)
    if not folder.exists():
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            staged=Path(tmp)/"doc";staged.mkdir()
            (staged/"全文.md").write_bytes(doc["text"].encode())
            for ref,src in doc["images"].items():
                target=inside(staged,ref);target.parent.mkdir(exist_ok=True)
                shutil.copyfile(src,target)
            (staged/"manifest.json").write_text(dump(manifest),encoding="utf8")
            verify_folder(staged,manifest);staged.rename(folder)
    verify_folder(folder,manifest)
    source=Path(doc["source"])
    backup=inside(ws,"Original_Backup/"+CATEGORY+"/"+doc["sha"]+"/"+doc["name"])
    if not backup.exists():
        backup.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=backup.parent) as tmp:
            dest=Path(tmp)/"original";shutil.copyfile(source,dest)
            check(sha(dest.read_bytes())==doc["sha"],"Original changed");dest.replace(backup)
    check(sha(backup.read_bytes())==doc["sha"],"Original backup mismatch")
    return folder.relative_to(ws).as_posix()

def publish(ws,key,docs,items,digest):
    bank=inside(ws,BANK);bank.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(bank,timeout=30)) as db,db:
        db.execute("PRAGMA foreign_keys=ON");db.executescript(DB_SCHEMA)
        db.execute("BEGIN IMMEDIATE")
        existing=db.execute("SELECT digest,receipt FROM imports WHERE id=?",(key,)).fetchone()
        if existing:check(existing[0]==digest,"Already published another revision")
        for d in docs:
            folder=store_document(ws,d)
            db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?)",(d["sha"],d["name"],folder,sha(d["text"].encode())))
        if existing:return json.loads(existing[1])
        db.executemany("INSERT OR IGNORE INTO labels VALUES(?)",[(v,) for v in catalog().values()])
        result={}
        for q in items:
            texts=q["text"];typ=q["type"]
            identity_parts=[typ,texts,q["answer_status"],q["tags"]]
            if "generated" in q:
                identity_parts.append(q["generated"])
            identity=sha(dump(identity_parts).encode())
            found=db.execute("SELECT id FROM questions WHERE content_hash=?",(identity,)).fetchone()
            if found:qid=found[0]
            else:
                db.execute("INSERT INTO sequences VALUES(?,1) ON CONFLICT(type) DO UPDATE SET counter=counter+1",(typ,))
                seq=db.execute("SELECT counter FROM sequences WHERE type=?",(typ,)).fetchone()[0]
                check(seq<10**10,"Question ID range exhausted")
                qid=f"{TYPES.index(typ)}{seq:010d}"
                db.execute("INSERT INTO questions VALUES(?,?,?,?,?,?,?,?)",(qid,typ,identity,texts["stem"],texts.get("options",""),texts.get("answer",""),texts.get("explanation",""),q["answer_status"]))
                db.executemany("INSERT INTO tags VALUES(?,?,?,?)",[(qid,*v) for v in q["tags"]])
                db.executemany("INSERT INTO generated_solution_parts VALUES(?,?,?,?,?)",
                               [(qid,v["role"],v["position"],v["text"],v["verification"]) for v in q.get("generated",[])])
                for uid,src in q["images"].items():
                    data=Path(src).read_bytes();check(sha(data)==uid,"Image changed during publish")
                    db.execute("INSERT OR IGNORE INTO images VALUES(?,?,?)",(uid,Path(src).suffix,data))
                    db.execute("INSERT INTO question_images VALUES(?,?)",(qid,uid))
            for docsha,anchor,role in q["refs"]:
                db.execute("INSERT OR IGNORE INTO sources VALUES(?,?,?,?,?,?)",(qid,docsha,anchor,q["number"],q["section"],role))
            result[q["key"]]=qid
        receipt=dict(import_id=key,preview_sha256=digest,questions=result,documents=[d["sha"] for d in docs],taxonomy_sha256=sha(TAXONOMY.read_bytes()),published=True)
        db.execute("INSERT INTO imports VALUES(?,?,?)",(key,digest,dump(receipt)))
        return receipt

def run(ws,allowed,action,**args):
    ws=Path(ws).resolve(strict=True)
    if action=="labels":
        needle=args.get("query","");return [p for p in catalog().values() if needle in p]
    path=inside(ws,"tmp/exercise-import/staging.sqlite3");path.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(path,timeout=30)) as db,db:
        db.execute("CREATE TABLE IF NOT EXISTS sets(id TEXT PRIMARY KEY,docs TEXT NOT NULL,review TEXT NOT NULL,revision INTEGER NOT NULL)")
        if action=="list":return [{"import_id":a,"revision":b} for a,b in db.execute("SELECT id,revision FROM sets")]
        if action=="prepare":return prepare(ws,db,args["sources"],allowed)
        key=args["import_id"]
        if action in {"review","publish"}:db.execute("BEGIN IMMEDIATE")
        docs,review,revision=read_set(db,key)
        if action=="status":return status(ws,db,key)
        if action=="draft":
            qkey=args["question_key"]
            check(qkey in review.get("questions",{}),"Unknown draft question")
            return dict(import_id=key,revision=revision,question=review["questions"][qkey])
        if action=="lines":
            number=args["doc"];check(type(number) is int and 1<=number<=len(docs),"Invalid doc")
            lines=docs[number-1]["text"].splitlines()
            start=args.get("start",1);limit=args.get("limit",80)
            check(type(start) is int and 1<=start<=len(lines) and type(limit) is int and 1<=limit<=250,"Invalid line range")
            return dict(doc=number,total=len(lines),lines=[(i+1,lines[i]) for i in range(start-1,min(start-1+limit,len(lines)))])
        if action=="review":
            check(args["revision"]==revision,"Stale review revision")
            bank=inside(ws,BANK)
            if bank.exists():
                with closing(sqlite3.connect(bank)) as conn:
                    check(not conn.execute("SELECT 1 FROM imports WHERE id=?",(key,)).fetchone(),"Published import is frozen")
            patch=args["review"]
            check(isinstance(patch,dict) and not set(patch)-{"questions","excluded","unresolved","coverage_confirmed"},"Invalid review fields")
            review["coverage_confirmed"]=False
            for field in ("questions","excluded"):
                for k,v in patch.get(field,{}).items():
                    coll=review.setdefault(field,{})
                    if v is None:coll.pop(k,None)
                    else:coll[k]=v
            for field in ("unresolved","coverage_confirmed"):
                if field in patch:review[field]=patch[field]
            db.execute("UPDATE sets SET review=?,revision=revision+1 WHERE id=?",(dump(review),key))
            return status(ws,db,key)
        check(action in ("preview","publish"),"Unknown action")
        items=compile_review(docs,review,previously_published=status(ws,db,key)["receipt"] is not None)
        digest=sha(dump([key,[(d["sha"],sha(d["text"].encode())) for d in docs],review,sha(TAXONOMY.read_bytes()),items]).encode())
        if action=="preview":
            if "question_key" in args:
                item=next((q for q in items if q["key"]==args["question_key"]),None)
                check(item,"Unknown draft question")
                return dict(import_id=key,revision=revision,preview_sha256=digest,number=item["number"],type=item["type"],
                            parts=item["text"],tags=item["tags"],sources=item["refs"],images=sorted(item["images"]),
                            generated=item.get("generated",[]))
            return dict(import_id=key,revision=revision,preview_sha256=digest,questions=[dict(key=q["key"],number=q["number"],type=q["type"],tags=q["tags"],image_count=len(q["images"])) for q in items])
        check(digest==args["preview_sha256"],"Preview changed")
        return publish(ws,key,docs,items,digest)

SCHEMA={"name":"exercise_import","description":"Prepare exercise sets from successful conversion jobs, inspect numbered lines, look up physics tags, review, preview and publish the full documents and an SQLite question bank.",
        "inputSchema":{"type":"object","additionalProperties":False,"required":["action"],"properties":{
            "action":{"enum":["list","prepare","status","draft","lines","labels","review","preview","publish"]},
            "sources":{"type":"array","items":{"type":"object","properties":{"job_id":{"type":"string"},"role":{"enum":["答案","题卷","作业","讲义","解析版"]}},"required":["job_id","role"],"additionalProperties":False}},
            "import_id":{"type":"string"},"question_key":{"type":"string"},"doc":{"type":"integer"},"start":{"type":"integer"},"limit":{"type":"integer"},"revision":{"type":"integer"},"query":{"type":"string"},"review":{"type":"object"},"preview_sha256":{"type":"string"}}}}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--workspace",required=True,type=Path)
    parser.add_argument("--input-root",action="append",default=[],type=Path)
    parser.add_argument("--request",required=True,type=Path)
    args=parser.parse_args()
    print(dump(run(args.workspace,[args.workspace,*args.input_root],**read(args.request))))

