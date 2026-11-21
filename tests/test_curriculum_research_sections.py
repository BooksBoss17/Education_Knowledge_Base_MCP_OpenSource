import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace


def test_module_ending_before_next_heading_ignores_empty_page_marker(tmp_path):
    skill = Path(__file__).resolve().parents[1] / "SKILLS/curriculum-research-import"
    build = runpy.run_path(str(skill / "scripts/organize.py"))["build"]
    source = tmp_path / "example.pdf"
    source.write_bytes(b"source identity fixture")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    package = tmp_path / "package"
    package.mkdir()
    payloads = {
        "document.md": "<!-- page:1 -->\n目录内容\n<!-- page:2 -->\n\n## 课程性质\n正文\n",
        "assets_manifest.jsonl": "",
        "conversion_report.json": json.dumps({
            "source": {"sha256": source_hash}, "document": {"pages": 2}, "review_items": []
        }),
    }
    integrity = {}
    for name, content in payloads.items():
        data = content.encode("utf-8")
        (package / name).write_bytes(data)
        key = {"document.md": "document_md", "assets_manifest.jsonl": "assets_manifest",
               "conversion_report.json": "conversion_report"}[name]
        integrity[key + "_sha256"] = hashlib.sha256(data).hexdigest()
    (package / "package_manifest.json").write_text(json.dumps({
        "source": {"sha256": source_hash}, "complete": True, "integrity": integrity
    }), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "source_sha256": source_hash, "document_sha256": integrity["document_md_sha256"],
        "title": "示例", "kind": "curriculum_standard", "pages": {"1": "目录", "2": "正文"},
        "decisions": {}, "unresolved": [], "corrections": [],
        "parts": [{"title": "目录", "start_line": 1}, {"title": "课程性质", "start_line": 5}]
    }), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    build(SimpleNamespace(source=source, package=package, plan=plan), output)
    receipt = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [(m["first_page"], m["last_page"]) for m in receipt["modules"]] == [(1, 1), (2, 2)]
    text = (output / "02-课程性质.md").read_text(encoding="utf-8")
    assert text.count("课程性质") == 1 and "正文" in text


def test_cross_page_join_requires_review_and_preserves_margins():
    import pytest

    helper = Path(__file__).resolve().parents[1] / "SKILLS/curriculum-research-import/scripts/organize.py"
    join_pages = runpy.run_path(str(helper))["join_pages"]
    original = {1: "前段。\n\n科学构建评价\n\n收稿日期", 2: "页眉\n\n体系。下一句。"}
    decision = dict(tail="科学构建评价", head="体系。", footer="收稿日期", header="页眉", reason="原页相邻两栏")
    chunks = original.copy()
    notes = []
    join_pages(chunks, {1: decision}, notes)
    assert chunks[1] == "前段。"
    assert chunks[2].endswith("科学构建评价体系。下一句。")
    assert notes == [(1, "收稿日期"), (2, "页眉")]
    unchanged = {1: "第一段结束。", 2: "独立的新段。"}
    join_pages(unchanged, {}, [])
    assert unchanged == {1: "第一段结束。", 2: "独立的新段。"}
    with pytest.raises(ValueError, match="Page margins changed"):
        join_pages(original.copy(), {1: {**decision, "footer": "其他文字"}}, [])
    with pytest.raises(ValueError, match="Ambiguous page join"):
        join_pages({**original, 2: "页眉\n\n体系。体系。"}, {1: decision}, [])


def test_heading_hierarchy_ignores_body_numbering_and_fenced_code():
    helper = Path(__file__).resolve().parents[1] / "SKILLS/curriculum-research-import/scripts/organize.py"
    normalize = runpy.run_path(str(helper))["normalize_headings"]
    source = "### 甲\n##### 子节\n### 乙\n（1）条款\n~~~text\n# 原样\n~~~\n"
    assert normalize(source) == "## 甲\n### 子节\n## 乙\n（1）条款\n~~~text\n# 原样\n~~~\n"
