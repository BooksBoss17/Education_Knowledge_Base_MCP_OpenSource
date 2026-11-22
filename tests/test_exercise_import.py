import hashlib,json,runpy,sqlite3
from pathlib import Path
import pytest

tool=runpy.run_path(str(Path(__file__).resolve().parents[1]/"TOOLS"/"education_mcp"/"exercise_import.py"))
def make_job(ws):
    source=ws/"物理作业.docx";source.write_bytes(b"source")
    pkg=ws/"tmp/bemarkdown/r/set";pkg.mkdir(parents=True)
    asset=pkg/"assets/fig.png";asset.parent.mkdir();asset.write_bytes(b"image")
    text="1. 如图，物体加速度为？\n![示意](assets/fig.png)\nA. 一 B. 二\n答案 A\n分析：结合题图按牛顿第二定律计算。\n"
    (pkg/"document.md").write_text(text,encoding="utf8")
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (pkg/"assets_manifest.jsonl").write_text(json.dumps(dict(relative_path="assets/fig.png",content_sha256=h(asset))),encoding="utf8")
    (pkg/"conversion_report.json").write_text("{}",encoding="utf8")
    integrity={k+"_sha256":h(pkg/f) for k,f in [("document_md","document.md"),("assets_manifest","assets_manifest.jsonl"),("conversion_report","conversion_report.json")]}
    (pkg/"package_manifest.json").write_text(json.dumps(dict(complete=True,source=dict(sha256=h(source)),integrity=integrity)))
    job="1"*32;folder=ws/"tmp/bemarkdown/.mcp/jobs"/job;folder.mkdir(parents=True)
    (folder/"state.json").write_text(json.dumps(dict(status="SUCCEEDED",source=str(source),source_sha256=h(source),conversion=dict(package_path=str(pkg)))))
    return job

def test_review_publish_replay(tmp_path):
    def call(action,**args):return tool["run"](tmp_path,[tmp_path],action,**args)
    j=make_job(tmp_path)
    result=call("prepare",sources=[dict(job_id=j,role="作业")]);key=result["import_id"]
    def act(a,**kw):return call(a,import_id=key,**kw)
    with pytest.raises(ValueError):act("preview")
    q=dict(number="1",type="单选",parts={k:[dict(span=[1,a,b])] for k,a,b in [("stem",1,2),("options",3,3),("answer",4,4),("explanation",5,5)]},answer_status="已收录",tags=[dict(leaf="牛顿第二定律",importance="主考")])
    act("review",revision=0,review=dict(questions={"q1":q},coverage_confirmed=True,unresolved=[]))
    with pytest.raises(ValueError,match="Stale"):act("review",revision=0,review={})
    p=act("preview")
    with pytest.raises(ValueError,match="Preview changed"):act("publish",preview_sha256="x")
    receipt=act("publish",preview_sha256=p["preview_sha256"])
    assert receipt==act("publish",preview_sha256=p["preview_sha256"])
    with sqlite3.connect(tmp_path/tool["BANK"]) as db:
        assert db.execute("SELECT id FROM questions").fetchone()==("00000000001",)
        assert db.execute("SELECT bytes FROM images").fetchone()==(b"image",)
        assert db.execute("SELECT count(*) FROM sources").fetchone()==(4,)
    full=next(tmp_path.glob("KNOWLEDGE_BASE/*/*/全文.md"))
    assert "DIAGRAMS/" in full.read_text("utf8")



def test_concurrent_suite_ids_and_stale_coverage(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    ws=tmp_path
    def call(action,**args):
        return tool["run"](ws,[ws],action,**args)
    job=make_job(ws)
    old=ws/"tmp/bemarkdown/.mcp/jobs"/job/"state.json"
    state=json.loads(old.read_text("utf8"))
    pkg=Path(state["conversion"]["package_path"])
    original=Path(state["source"])
    new_source=ws/"另一题.docx"
    new_source.write_bytes(b"second source")
    import shutil
    pkg2=pkg.with_name("second")
    shutil.copytree(pkg,pkg2)
    md2=pkg2/"document.md"
    md2.write_text(md2.read_text("utf8").replace("加速度","合力"),encoding="utf8")
    m=json.loads((pkg2/"package_manifest.json").read_text("utf8"))
    m["source"]["sha256"]=tool["sha"](new_source.read_bytes())
    m["integrity"]["document_md_sha256"]=tool["sha"](md2.read_bytes())
    (pkg2/"package_manifest.json").write_text(json.dumps(m))
    second="2"*32
    f=old.parent.parent/second/"state.json"
    f.parent.mkdir()
    f.write_text(json.dumps(dict(status="SUCCEEDED",source=str(new_source),source_sha256=m["source"]["sha256"],conversion=dict(package_path=str(pkg2)))))
    previews=[]
    for j in (job,second):
        key=call("prepare",sources=[dict(job_id=j,role="题卷")])["import_id"]
        q=dict(number="1",type="单选",parts={k:[dict(span=[1,a,b])] for k,a,b in [("stem",1,2),("options",3,3),("answer",4,4),("explanation",5,5)]},answer_status="已收录",tags=[dict(leaf="牛顿第二定律",importance="主考")])
        call("review",import_id=key,revision=0,review=dict(questions={"q":q},unresolved=["核对第1题"],coverage_confirmed=True))
        with pytest.raises(ValueError):call("preview",import_id=key)
        assert call("draft",import_id=key,question_key="q")["question"]==q
        call("review",import_id=key,revision=1,review=dict(unresolved=[],coverage_confirmed=True))
        previews.append((key,call("preview",import_id=key)["preview_sha256"]))
    def finish(pair):
        return call("publish",import_id=pair[0],preview_sha256=pair[1])
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts=list(pool.map(finish,[previews[1],previews[0],previews[1]]))
    assert receipts[0]==receipts[2]
    assert {r["questions"]["q"] for r in receipts}=={"00000000001","00000000002"}
    for r in receipts:assert call("status",import_id=r["import_id"])["receipt"]==r


def test_word_paragraph_numbers_do_not_replace_exam_numbers():
    lines = [
        "1. 注意事项", "2. 一、选择题", "3. 1. 下列说法正确的是",
        "4. A. 物体静止", "5. 二、填空题", "6. 1. 填写速度", "2.填写单位",
    ]
    found = tool["candidate_lines"]("\n".join(lines))
    assert [(c["line"],c["number"],c["section"]) for c in found] == [(3,"1.","一"),(6,"1.","二"),(7,"2.","二")]
    assert [c["number"] for c in tool["candidate_lines"]("1. 一道普通题\n2. 第二题")] == ["1.","2."]



def test_bold_section_and_question_starting_with_year():
    text="**三****、****实验**\n3．2024年测了多少次？\n3.14米\n**四、应用**\n1. 说明"
    entries=tool["candidate_lines"](text)
    assert [(e["line"],e["section"]) for e in entries]==[(2,"三"),(5,"四")]

@pytest.mark.parametrize("missing",[{"answer"},{"explanation"},{"answer","explanation"}])
def test_agent_completes_only_missing_fields(tmp_path,missing):
    j=make_job(tmp_path)
    pkg=tmp_path/"tmp/bemarkdown/r/set"
    manifest=json.loads((pkg/"package_manifest.json").read_text("utf8"))
    lines=["1. 小车从静止以 2 m/s² 匀加速 3 s，末速度多大？"]
    parts={"stem":[{"span":[1,1,1]}]}
    expected={"answer":"6 m/s","explanation":"由 v=v₀+at，代入得 v=0+2×3=6 m/s。"}
    for role,content in expected.items():
        if role in missing:
            parts[role]=[dict(origin="agent",text=content,verification="用速度增量除以时间得到 2 m/s²，与给定加速度相符。")]
        else:
            lines.append(content)
            parts[role]=[dict(span=[1,len(lines),len(lines)])]
    md=pkg/"document.md";md.write_text("\n".join(lines),encoding="utf8")
    manifest["integrity"]["document_md_sha256"]=tool["sha"](md.read_bytes())
    (pkg/"package_manifest.json").write_text(json.dumps(manifest),encoding="utf8")
    def call(action,**args):
        return tool["run"](tmp_path,[tmp_path],action,**args)
    key=call("prepare",sources=[dict(job_id=j,role="作业")])["import_id"]
    question=dict(type="计算",number="1",parts=parts,answer_status="Agent补写" if "answer" in missing else "已收录",
                  tags=[dict(leaf="速度与时间的关系",importance="主考")])
    def review(rev):
        return call("review",import_id=key,revision=rev,review=dict(questions={"1":question},unresolved=[],coverage_confirmed=True))
    selected=next(iter(sorted(missing)));saved=parts.pop(selected)
    review(0)
    with pytest.raises(ValueError):call("preview",import_id=key)
    parts[selected]=saved
    saved[0]["span"]=[1,1,1]
    review(1)
    with pytest.raises(ValueError):call("preview",import_id=key)
    del saved[0]["span"]
    review(2)
    preview=call("preview",import_id=key,question_key="1")
    assert {x["role"] for x in preview["generated"]}==missing
    receipt=call("publish",import_id=key,preview_sha256=preview["preview_sha256"])
    assert receipt==call("publish",import_id=key,preview_sha256=preview["preview_sha256"])
    with sqlite3.connect(tmp_path/tool["BANK"]) as db:
        assert db.execute("select answer,explanation from questions").fetchone()==tuple(expected.values())
        assert {r for r, in db.execute("select role from generated_solution_parts")}==missing
        assert {r for r, in db.execute("select role from sources")}==set(parts)-missing
    full=next(tmp_path.glob("KNOWLEDGE_BASE/*/*/全文.md"))
    assert full.read_text("utf8")==md.read_text("utf8")

