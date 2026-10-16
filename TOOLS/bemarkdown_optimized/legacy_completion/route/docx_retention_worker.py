"""DOCX XML/MD source adapter using frozen B200 assets plus 9B D-review.

The mandatory page_index=0 identifies one unpaginated DOCX context unit, never a
PDF coordinate. PDF-only geometry/component protections are not claimed here.
"""
import argparse,hashlib,json,mimetypes,runpy,sys,time,re,copy,io
from pathlib import Path
from PIL import Image
P=Path(__file__).resolve().parents[1];R=P.parents[1];sys.path.insert(0,str(P/'tools'))
sys.path.insert(0,str(P/'reports'))
from contracts import validate_input,digest
from dev_calibration import policy_fingerprint
import probe_gguf_image_runtime as G
from cold_semantic_review import Reviewer
IMAGE=re.compile(r'!\[(?:\\.|[^\]\\])*\]\((?:<[^>\n]+>|[^\n)])+\)')
DEST=re.compile(r'''\s*(?:<([^>]+)>|(\S+?))(?:\s+(?:"[^"]*"|'[^']*'))?\s*''')
def image_refs(text):
    for match in IMAGE.finditer(text):
        token=match[0];dest=DEST.fullmatch(token[token.index('](')+2:-1])
        if dest:yield match,dest.group(1) or dest.group(2)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def emit(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    package=Path(a.package);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    md=(package/'document.md').read_text(encoding='utf-8');assets=[json.loads(x) for x in (package/'assets_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    items=[];resolver=[];mapping={};contexts={};objects=R/'tmp/rt-train/objects';objects.mkdir(parents=True,exist_ok=True)
    for i,asset in enumerate(assets):
        name=asset.get('relative_path')
        if not name:continue
        matches=[m for m,n in image_refs(md) if n==name]
        if not matches:continue
        source=package/name
        with Image.open(source) as img:
            image=img.convert('RGB');w,h=image.size
            if max(w,h)/min(w,h)>200:
                from extreme_aspect_padding_candidate import pad_extreme
                image,_=pad_extreme(image,32)
            path=objects/f'{i:05d}.png';image.save(path);w,h=image.size
        sid='docx:'+name;uses=[];all_text=bytearray();originals=[]
        for j,m in enumerate(matches):
            start=max(0,m.start()-1000);end=min(len(md),m.end()+1000)
            context='DOCX原生XML/Markdown用途；page_index=0仅为未分页文档单元，没有PDF页坐标。\n'+IMAGE.sub('[其他图片]',md[start:end])
            original={'source_part':asset.get('source_part'),'source_locator':asset.get('source_locator'),'markdown_char_range':[start,end]};originals.append(original)
            value=context.encode('utf-8');s=len(all_text);all_text.extend(value);e=len(all_text);all_text.extend(b'\n\n')
            uses.append({'use_id':f'docx-use-{i}-{j}','page_index':0,'context_status':'verified',
                'spans':[{'span_id':f'ctx-{s}-{e}','start_byte':s,'end_byte':e,'text':context,
                'text_sha256':hashlib.sha256(value).hexdigest(),'source_page_index':0,'relation_to_use':'same_page','binding_evidence_refs':[]}],
                'figure_refs':[],'selector_version':'docx-xml-md-unpaginated-v1','source_window_truncated':start>0 or end<len(md)})
        doc=objects/f'{i:05d}.md';doc.write_bytes(all_text);image_sha=sha(path)
        row={'schema_version':'1.0','sample_id':sid,'book_id':'docx-'+sha(package/'conversion_report.json')[:12],'synthetic':False,
            'image':{'ref':'sha256:'+image_sha,'sha256':image_sha,'mime':'image/png','width':w,'height':h},
            'document':{'ref':str(doc),'sha256':sha(doc)},'scope':'asset_use_bag','uses':uses,
            'coverage':{'expected_use_count':len(uses),'covered_use_ids':[u['use_id'] for u in uses],'enumeration_complete':True}}
        validate_input(row);items.append(row);mapping[sid]=name;contexts[sid]=originals
        resolver.append({'image_ref':row['image']['ref'],'object_path':str(path)})
    inputs=out/'docx.inputs.jsonl';resolve=P/'data/splits/v2/resolver.jsonl';emit(inputs,items);emit(resolve,resolver)
    assert items
    plan=P/'reports/docx-firstpass-plan.json'
    write(plan,{'inputs':str(inputs),'inputs_sha256':sha(inputs),'expected_members':len(items),'threshold':.8,
        'policy_fingerprint':policy_fingerprint(.8),'resolver_path':str(resolve),'resolver_sha256':sha(resolve),
        'freeze_tag':'docx-unpaginated-source-v1','result_pointer':'docx-firstpass-pointer.json'})
    first=out/'firstpass';first.mkdir();previous=sys.argv
    try:
        sys.argv=[str(P/'tools/run_gguf_partition_predictions.py'),'--run-dir',str(first),'--evaluation-plan',str(plan)]
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:sys.argv=previous
    assert read(first/'server-unload.json')['owned_server_exited']
    predictions=[json.loads(x) for x in (first/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    by_sid={x['sample_id']:x for x in items};reviews=out/'semantic';reviews.mkdir()
    reviewer=None;results=[];evidence=[]
    prompt='你负责DOCX剩余图片的删除语义复核。先识别目标裁图本身的可见内容，再以提供的每个XML/MD用途判断删除是否无损。文字未实际转录时不得删除；标题、公式、表格、题目选项、科学示意图、需解释的动作均保留。邻文信息不归给裁图。所有用途均确认删除无损方可D；无法确认给U。来源没有PDF页坐标，不推测页面布局。只输出要求的JSON。'
    schema={'type':'object','properties':{'verdict':{'type':'string','enum':['K','D','U']},'deletion_lossless':{'type':'boolean'},
        'visible_content':{'type':'string'},'source_evidence':{'type':'string'},'reason':{'type':'string'}},
        'required':['verdict','deletion_lossless','visible_content','source_evidence','reason'],'additionalProperties':False}
    try:
        for pred in predictions:
            sid=pred['sample_id'];decision=pred['final_decision'];row=by_sid[sid]
            if decision=='D':
                if reviewer is None:reviewer=Reviewer(reviews,'dependency',read(R/'job-request.json').get('reviewer_profile','P24-small-batch'))
                use_decisions=[]
                path=Path(next(r['object_path'] for r in resolver if r['image_ref']==row['image']['ref']))
                for use in row['uses']:
                    text='\n'.join(s['text'] for s in use['spans']);reviewer.ready()
                    for attempt in range(2):
                        try:
                            response=G.request(reviewer.url+'/v1/chat/completions',{'model':'local-retention',
                                'messages':G.materialize_messages([{'role':'system','content':prompt},{'role':'user','content':[{'type':'image','image_ref':'local'},{'type':'text','text':text}]}],path.read_bytes(),'image/png'),
                                'temperature':0,'seed':20261004,'max_tokens':384,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False},
                                'response_format':{'type':'json_schema','json_schema':{'name':'docx_semantic_use','strict':True,'schema':schema}}},timeout=240)
                            break
                        except Exception as exc:
                            import urllib.error
                            body=exc.read().decode(errors='replace') if isinstance(exc,urllib.error.HTTPError) else str(exc)
                            if attempt or not isinstance(exc,urllib.error.HTTPError) or exc.code!=500 or 'bad allocation' not in body.lower():raise
                            write(reviews/'recovery.json',{'status':'SAME_REQUEST_ONE_RETRY','error':body});reviewer.stop();reviewer.ready()
                    reviewer.calls+=1;reviewer.segment_calls+=1;value=json.loads(response['choices'][0]['message']['content'])
                    v=value['verdict'];v=v if v!='D' or (value['deletion_lossless'] and value['source_evidence'].strip() and value['visible_content'].strip()) else 'U'
                    use_decisions.append(v);evidence.append({'sample_id':sid,'use_id':use['use_id'],'value':value,'actual_verdict':v,'usage':response.get('usage'),
                        'request_sha256':digest({'image':row['image']['sha256'],'prompt':prompt,'schema':schema,'text':text})})
                decision='K' if 'K' in use_decisions else 'U' if 'U' in use_decisions else 'D'
            results.append({'asset_name':mapping[sid],'decision':decision,'primary':pred['final_decision'],'image_sha256':row['image']['sha256']})
    finally:
        if reviewer:reviewer.stop()
    write(out/'results.json',{'status':'COMPLETE','adapter':'DOCX_XML_MD_UNPAGINATED_V1','pdf_geometry_claimed':False,'gold_read':False,
        'results':results,'source_bindings':contexts,'semantic_review':evidence,'b200_firstpass_freeze_sha256':sha(first/'predictions.freeze.json')})

if __name__=='__main__':main()
