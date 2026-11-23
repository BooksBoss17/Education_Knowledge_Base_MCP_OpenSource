from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

from test_exercise_import import make_job

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TOOLS/education_mcp"))
import checklist_import as tool
import exercise_import as exercise


def add_source(ws, job, source, raw, body):
    path = ws / source
    path.write_bytes(raw)
    folder = ws / "tmp/bemarkdown/r" / job
    folder.mkdir(parents=True)
    (folder / "document.md").write_bytes(body.encode())
    asset = folder / "image.png"
    asset.write_bytes(b"image")
    files = {"document_md": "document.md", "assets_manifest": "assets_manifest.jsonl",
             "conversion_report": "conversion_report.json"}
    (folder / files["assets_manifest"]).write_text(json.dumps(
        {"relative_path": "image.png", "content_sha256": tool.sources.file_sha(asset)}), encoding="utf-8")
    (folder / files["conversion_report"]).write_text("{}", encoding="utf-8")
    sha = tool.sources.file_sha(path)
    (folder / "package_manifest.json").write_text(json.dumps({
        "complete": True, "source": {"sha256": sha},
        "integrity": {k + "_sha256": tool.sources.file_sha(folder / v) for k, v in files.items()}
    }), encoding="utf-8")
    state = ws / "tmp/bemarkdown/.mcp/jobs" / job
    state.mkdir(parents=True)
    (state / "state.json").write_text(json.dumps({"status": "SUCCEEDED", "source": str(path),
        "source_sha256": sha, "conversion": {"package_path": str(folder)}}), encoding="utf-8")
    return sha


def include(a, b, title="公式", **kwargs):
    return dict(span=[a, b], decision="include", title=title, **kwargs)


def exclude(a, b):
    return dict(span=[a, b], decision="exclude", reason="题目或目录")


def review(ws, current, doc, decisions, complete=True, unresolved=()):
    return tool.run(ws, [ws], "review", import_id=current["import_id"], doc=doc,
                    revision=current["revision"], review={"decisions": decisions,
                                                          "complete": complete, "unresolved": list(unresolved)})


def test_coverage_images_formula_and_concurrent_replay(tmp_path):
    ws = tmp_path
    run = lambda action, **args: tool.run(ws, [ws], action, **args)
    job = "a" * 32
    add_source(ws, job, "知识.pdf", b"original", "<!-- page:1 -->\n# 速度\n$a$ 恒定时 $v=v_0+at$。\n![参考图](image.png \"原件\")\n例1\n")
    draft = run("prepare", mode="standalone", sources=[{"job_id": job}])
    act = lambda action, **args: run(action, import_id=draft["import_id"], **args)
    assert act("lines", doc=1)["lines"][2]["physical_page"] == 1
    with pytest.raises(ValueError, match="uncovered"):
        review(ws, draft, 1, [include(2, 4)])
    draft = review(ws, draft, 1, [exclude(1, 2), include(3, 4, module="基础")], complete=False, unresolved=["例题边界"])
    with pytest.raises(ValueError, match="Unresolved"):
        act("preview")
    with pytest.raises(ValueError, match="references"):
        review(ws, draft, 1, [exclude(1, 2), include(3, 4, text="改写", basis="原件"), exclude(5, 5)])
    draft = review(ws, draft, 1, [exclude(1, 2), include(3, 4, module="基础"), exclude(5, 5)])
    proposed = act("preview")
    with pytest.raises(ValueError, match="Preview changed"):
        act("publish", preview_sha256="0"*64)
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: act("publish", preview_sha256=proposed["preview_sha256"]), range(2)))
    assert receipts[0] == receipts[1]
    folder = ws / receipts[0]["folder"]
    body = (folder / "模块01.md").read_text("utf-8")
    assert "$a$ 恒定时 $v=v_0+at$" in body and "例1" not in body
    assert len(list((folder / "DIAGRAMS").iterdir())) == 1
    assert tool.read(folder / "manifest.json")["sections"][0]["sources"][0]["physical_pages"] == [1]


def test_handout_reuses_existing_question_bank_and_groups_editions(tmp_path):
    ws = tmp_path
    student_job, teacher_job = make_job(ws), "b"*32
    teacher_sha = add_source(ws, teacher_job, "老师.docx", b"teacher",
                             "1. 如图，物体加速度为？\n![示意](image.png)\n匀变速运动需要恒定加速度。\n")
    run = lambda action, **args: tool.run(ws, [ws], action, **args)
    entry = {"job_id": teacher_job}
    draft = run("prepare", mode="handout", sources=[entry])
    updated = review(ws, draft, 1, [exclude(1, 2), include(3, 3)])
    preview = run("preview", import_id=updated["import_id"])
    with pytest.raises(ValueError, match="no published"):
        run("publish", import_id=updated["import_id"], preview_sha256=preview["preview_sha256"])
    converted = exercise.document(ws, teacher_job, "讲义", [ws])
    exercise.publish(ws, "test", [converted], [], "test")
    recovered = run("prepare", mode="handout", sources=[{"handout_sha256": teacher_sha}])
    assert recovered["import_id"] == updated["import_id"] and recovered["revision"] == 1
    receipt = run("publish", import_id=updated["import_id"], preview_sha256=preview["preview_sha256"])
    manifest = tool.read(ws / receipt["folder"] / "manifest.json")
    assert (ws / manifest["sources"][0]["full_text"]).is_file()
    assert not (ws / "Original_Backup" / tool.CATEGORY).exists()
    editions = run("prepare", mode="standalone", sources=[entry, {"job_id": student_job}])
    editions = review(ws, editions, 1, [include(1, 2), include(3, 3)])
    editions = review(ws, editions, 2, [include(1, 2), exclude(3, 5)])
    sections = run("preview", import_id=editions["import_id"])["sections"]
    assert len(sections) == 2 and len(sections[0]["sources"]) == 2


def test_inventory_and_no_knowledge(tmp_path):
    ws = tmp_path
    job = "c"*32
    digest = add_source(ws, job, "试卷.pdf", b"paper", "1. 题目\n")
    with ZipFile(ws / "样本.zip", "w") as group:
        group.writestr("../../试卷.pdf", b"paper")
    run = lambda action, **args: tool.run(ws, [ws], action, **args)
    result = run("inventory", paths=[str(ws/"试卷.pdf"), str(ws/"样本.zip")])
    assert result["unique"][0]["sha256"] == digest
    origin = result["unique"][0]["sources"][1]
    staged = Path(run("stage", origin=origin, sha256=digest)["source"])
    assert staged.parent == ws / "tmp/checklist-import/inputs" and staged.read_bytes() == b"paper"
    draft = run("prepare", mode="standalone", sources=[{"job_id": job, "origins": [origin]}])
    draft = review(ws, draft, 1, [exclude(1, 1)])
    preview = run("preview", import_id=draft["import_id"])
    receipt = run("publish", import_id=draft["import_id"], preview_sha256=preview["preview_sha256"])
    assert receipt["disposition"] == "no_knowledge"
    assert not (ws / "KNOWLEDGE_BASE" / tool.CATEGORY).exists()


def test_new_transcription_requires_explicit_supersession(tmp_path):
    ws = tmp_path
    run = lambda action, **args: tool.run(ws, [ws], action, **args)
    first, second = "d"*32, "e"*32
    sha = add_source(ws, first, "清单.docx", b"same source", "初稿\n")
    add_source(ws, second, "清单.docx", b"same source", "经核对的新版\n")

    def prepared(job):
        draft = run("prepare", mode="standalone", sources=[{"job_id": job}])
        review(ws, draft, 1, [include(1, 1)])
        preview = run("preview", import_id=draft["import_id"])
        return {"import_id": draft["import_id"], "preview_sha256": preview["preview_sha256"]}

    older = prepared(first)
    receipt = run("publish", **older)
    newer = prepared(second)
    with pytest.raises(ValueError, match="supersedes"):
        run("publish", **newer)
    latest = run("publish", **newer, supersedes=older["import_id"])
    assert latest == run("publish", **newer)
    assert receipt == run("publish", **older)
    rows = run("list", sha256=sha)
    assert [row["receipt"] for row in rows if row["active"]] == [latest]
