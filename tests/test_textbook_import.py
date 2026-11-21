"""Import transaction tests. The runtime API uses the same handlers and lock."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import threading

import fitz
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]/'TOOLS/education_mcp'
API=runpy.run_path(str(ROOT/'textbook_import.py'))
LOCK=runpy.run_path(str(ROOT/'server.py'))['conversion_lock']


@pytest.fixture
def sample(tmp_path):
    package=tmp_path/'bundle'
    images=package/'assets';images.mkdir(parents=True)
    image=images/'motion.png';Image.new('RGB',(8,8),'red').save(image)
    source=tmp_path/'物理试读.pdf'
    with fitz.open() as pdf:
        for title in ['封面','第一章 运动','后记']:
            page=pdf.new_page();page.insert_text((30,30),title,fontname='china-s')
        pdf.save(source)
    md='<!-- page:1 -->\n封面\n<!-- page:2 -->\n![图](assets/motion.png "运动")\n# 第一章 运动\n速度为 $v$。\n<!-- page:3 -->\n# 后记\n结束。'
    (package/'document.md').write_text(md,'utf8')
    data=dict(source=dict(path=str(source),sha256=API['sha'](source)),review_items=[dict(page=2,reason='核实速度符号')],review_item_count=1)
    (package/'conversion_report.json').write_text(json.dumps(data),'utf8')
    (package/'assets_manifest.jsonl').write_text(json.dumps(dict(relative_path='assets/motion.png',content_sha256=API['sha'](image))),'utf8')
    integrity={k:API['sha'](package/n) for n,k in [('document.md','document_md_sha256'),('assets_manifest.jsonl','assets_manifest_sha256'),('conversion_report.json','conversion_report_sha256')]}
    (package/'package_manifest.json').write_text(json.dumps(dict(complete=True,source=data['source'],integrity=integrity)),'utf8')
    workspace=tmp_path/'kb'
    svc=SimpleNamespace(workspace=workspace,jobs=workspace/'jobs',allowed=[tmp_path],closing=threading.Event(),
                        workspace_manager=SimpleNamespace(safe_path=lambda x:workspace/x))
    call=lambda a,**args:API['call'](svc,LOCK,a,**args)
    prepared=call('prepare',package_path=str(package))
    do=lambda a,**kw:call(a,import_id=prepared['import_id'],**kw)
    return do,call,prepared,image,source,workspace


def settle(do,start):
    rev=start['revision']
    do('evidence',page=2)
    result=do('resolve',page=2,issue_ids=[1],decision='keep',revision=rev,source_evidence='第2页速度符号经核对')
    rev=result['revision'];do('preview',revision=rev)
    do('approve',revision=rev,source_evidence='整书三页及图注已审')
    return rev


def test_queued_edits_and_concurrent_commit(sample):
    do,call,start,image,source,workspace=sample
    assert call('prepare',source=str(source))['import_id']==start['import_id']
    with pytest.raises(ValueError,match='unresolved'):do('publish',revision=start['revision'])
    rev=settle(do,start)
    changed=do('flag',page=2,source_evidence='核查图注',revision=rev)
    with pytest.raises(ValueError,match='Stale'):do('publish',revision=rev)
    with pytest.raises(ValueError,match='unresolved'):do('publish',revision=changed['revision'])
    do('evidence',page=2)
    rev=do('resolve',revision=changed['revision'],issue_ids=[2],page=2,decision='keep',source_evidence='第2页图注一致')['revision']
    do('preview',revision=rev);do('approve',revision=rev,source_evidence='复核变更')
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda _:do('publish',revision=rev),range(2)))
    assert results[0]['output']==results[1]['output']
    book=Path(results[0]['output'])
    chapter=(book/'01-第一章 运动.md').read_text('utf8')
    assert r'\(v\)' in chapter and 'DIAGRAMS/'+API['sha'](image)+'.png' in chapter
    assert '![图]' not in (book/'00-封面与目录.md').read_text('utf8')


def test_changed_asset_blocks_publish(sample):
    do,_,start,image,_,workspace=sample
    rev=settle(do,start);Image.new('RGB',(8,8),'blue').save(image)
    with pytest.raises(ValueError,match='asset changed'):do('publish',revision=rev)
    assert not list((workspace/'KNOWLEDGE_BASE/TEXTBOOKS').glob('*'))


def test_failure_before_commit_keeps_library_empty(sample,monkeypatch):
    do,_,start,image,_,workspace=sample
    rev=settle(do,start)
    original=API['call'].__globals__['atomic']
    def fail(path,state):
        if path.name=='manifest.json':raise OSError('receipt unavailable')
        return original(path,state)
    monkeypatch.setitem(API['call'].__globals__,'atomic',fail)
    with pytest.raises(OSError):do('publish',revision=rev)
    assert not list((workspace/'KNOWLEDGE_BASE/TEXTBOOKS').glob('*'))


def test_damaged_dollar_line_cannot_reverse_following_displays():
    fmt=API['FMT']
    lines=['$$v_{y}$','', '$$ x=3 $$','','$$', 'F=ma', '$$', '价格为 '+chr(96)+'$9'+chr(96)]
    normalized,pending=fmt['normalize_math'](lines)
    assert pending==[1] and normalized[0]==lines[0]
    assert normalized[2]==r'\[ x=3 \]'
    assert normalized[4:7]==[r'\[','F=ma',r'\]'] and normalized[-1]==lines[-1]


def test_parts_follow_reference_and_include_organization_metadata(tmp_path):
    fmt=API['FMT']
    pages=['目录\n第一章 力    1\n学生实验    2','![图](assets/p.png)\n# 第一章 力\n正文','## 学生实验\n器材','## 索引\n术语']
    (tmp_path/'assets').mkdir()
    Image.new('RGB',(8,8),'red').save(tmp_path/'assets/p.png')
    parts,missing=fmt['outline'](pages)
    assert not missing
    assert [x['kind'] for x in parts]==['FRONT_MATTER','CHAPTER','APPENDIX','BACK_MATTER']
    manifest=fmt['render'](pages,parts,'书.pdf','a'*64,[tmp_path],tmp_path/'out',False)
    assert manifest['sections'][-1]['file']=='04-后记.md' and manifest['sections'][-1]['organization_metadata']
    assert 'part_type: "POSTSCRIPT"' in (tmp_path/'out/04-后记.md').read_text('utf8')
