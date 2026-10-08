"""Pre-registered selective visual-form guard, no truth or sample identity routing."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import probe_gguf_image_runtime as G
import retention_inference as N
from probe_zero99_evidence_contract import OBS, call
from run_zero99_fact_probe import parse, observation_valid
from zero99_toc_evidence import toc_evidence, no_text_photo

P, R = G.P, G.R
PROMPT = '''只判断当前裁图的视觉形式，不判断在教材中该保留还是删除，不根据题材猜测用途。只输出JSON字段form。
PHOTOGRAPH：确实可见真实物体、人物、材料、器材、自然或生活现象的摄影图像。黑白肖像、实物特写、颗粒粉末、光线很暗的实拍，只要真实物体/形态仍可辨认均属于此类。是否含文字不决定此类。
OTHER：纯色/色带/边框、文字/公式/图表/示意图、图标、卡通、抽象形状、文字艺术字造型、没有可辨真实对象的背景碎片。不能因为有渐变、阴影或一点纹理就称为照片。照片背景上仅以某个文字标签为主体的设计图归OTHER。
UNCERTAIN：无法可靠分辨上述两类。不能将不可见对象补想进图片。'''
SCHEMA = {'type':'object','properties':{'form':{'type':'string','enum':['PHOTOGRAPH','OTHER','UNCERTAIN']}},'required':['form'],'additionalProperties':False}

def route(raw, p_d):
    if raw=='U':
        return False
    assert raw in {'K','D'} and isinstance(p_d,(int,float)) and 0 <= p_d <= 1
    return raw == 'D' and p_d < .9

def decision(raw, p_d, form=None):
    if not route(raw,p_d):
        return raw
    return {'PHOTOGRAPH':'K','OTHER':'D','UNCERTAIN':'U'}.get(form,'U')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--plan',required=True);ap.add_argument('--guard-cache-run');ap.add_argument('--toc-thumbnail-check',action='store_true');a=ap.parse_args()
    out=Path(a.run_dir).resolve();assert out.is_relative_to(R/'tmp/rt-train/runs')
    plan=G.read(a.plan);inputs=Path(plan['inputs']);assert N.file_sha(inputs)==plan['inputs_sha256']
    rows=list(N.A.read_rows(inputs));assert len(rows)==plan['expected_members']
    prior_path=Path(plan['baseline_predictions']);assert N.file_sha(prior_path)==plan['baseline_predictions_sha256']
    prior={x['sample_id']:x for x in N.A.read_rows(prior_path)}
    resolver_path=Path(plan['resolver_path']);assert N.file_sha(resolver_path)==plan['resolver_sha256']
    resolver={x['image_ref']:x['object_path'] for x in N.A.read_rows(resolver_path)}
    export=G.read(P/'reports/gguf-candidate-export.json')
    model=next(Path(x['path']) for x in export['files'] if x['file'].endswith('Q4_K_M.gguf'))
    mmproj=next(Path(x['path']) for x in export['files'] if x['file']=='mmproj-F16.gguf')
    for path in [model,mmproj]:assert N.file_sha(path)==next(x['sha256'] for x in export['files'] if x['path']==str(path))
    for x in export['runtime_files']:assert N.file_sha(R/'tmp/llama-retention'/x['file'])==x['sha256']
    cache={}
    if a.guard_cache_run:
        previous=Path(a.guard_cache_run);frozen=G.read(previous/'prediction-freeze.json');registered=G.read(previous/'preregistration.json')
        assert N.file_sha(previous/'predictions.jsonl')==frozen['predictions_sha256']
        assert N.file_sha(previous/'preregistration.json')==frozen['preregistration_sha256']
        assert registered['prompt']==PROMPT and registered['schema']==SCHEMA
        assert registered['model_sha256']==N.file_sha(model) and registered['mmproj_sha256']==N.file_sha(mmproj)
        assert registered['runtime_files']==export['runtime_files']
        cache={x['sample_id']:x for x in N.A.read_rows(previous/'predictions.jsonl') if x['guard_triggered']}
    G.write(out/'preregistration.json',{'status':'FROZEN_BEFORE_VISUAL_GUARD','plan_sha256':N.file_sha(a.plan),'worker_sha256':N.file_sha(__file__),
        'model_sha256':N.file_sha(model),'mmproj_sha256':N.file_sha(mmproj),'runtime_files':export['runtime_files'],'prompt':PROMPT,'schema':SCHEMA,
        'route':'raw D with p_d <0.9 only; features never gold/ID/book/path','final':'raw first pass, PHOTOGRAPH guard K; OTHER D; uncertain U',
        'baseline_predictions_reused':True,'gold_opened':False,'not_full_cold_inference':True,
        'guard_cache_run':a.guard_cache_run,'guard_cache_freeze_sha256':N.file_sha(Path(a.guard_cache_run)/'prediction-freeze.json') if a.guard_cache_run else None})
    G.write(out/'context-guard-preregistration.json',{'enabled':a.toc_thumbnail_check,'rule_sha256':N.file_sha(P/'tools/zero99_toc_evidence.py'),'observation_prompt':OBS,
        'rule':'only raw D low-confidence PHOTOGRAPH; every full bound use is multi-chapter TOC; independent observation PHOTO with no readable text; otherwise unchanged'})
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}';server=None;results=[];began=time.monotonic();calls=0;cached_calls=0;toc_calls=0
    with (out/'server.log').open('wb') as log:
        try:
            server=subprocess.Popen(G.server_command(model,mmproj,port),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            G.await_ready(server,url);load_seconds=time.monotonic()-began
            with (out/'predictions.jsonl').open('x',encoding='utf-8') as stream:
                for index,row in enumerate(rows):
                    sid=row['sample_id'];first=prior[sid];raw=first['raw_decision'];p_d=first['p_d'];form=None;response_sha=None
                    cached=cache.get(sid) if route(raw,p_d) else None
                    if cached:
                        assert cached['image_sha256']==row['image']['sha256']
                        assert cached['raw_decision']==raw and cached['p_d']==p_d
                        form=cached['form'];response_sha=cached['response_sha256'];cached_calls+=1
                    elif route(raw,p_d):
                        path=Path(resolver[row['image']['ref']]);path=path if path.is_absolute() else R/path;data=path.read_bytes()
                        assert hashlib.sha256(data).hexdigest()==row['image']['sha256']
                        messages=[{'role':'system','content':PROMPT},{'role':'user','content':[{'type':'image','image_ref':'local'},{'type':'text','text':'判断图片自身的视觉形式。'}]}]
                        response=G.request(url+'/v1/chat/completions',{'model':'local-retention','messages':G.materialize_messages(messages,data,row['image']['mime']),
                            'temperature':0,'seed':20261004,'max_tokens':40,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False},
                            'response_format':{'type':'json_schema','json_schema':{'name':'visual_form','strict':True,'schema':SCHEMA}}})
                        file=out/f'guard-{index}.json';G.write(file,response);response_sha=N.file_sha(file);calls+=1
                        try:
                            choice=response['choices'][0];value=json.loads(choice['message']['content'])
                            if choice['finish_reason']=='stop' and set(value)=={'form'}:form=value['form']
                        except (ValueError,TypeError,KeyError):pass
                    verdict=decision(raw,p_d,form);toc_proof=None;toc_observation=None;toc_override=False
                    if a.toc_thumbnail_check and route(raw,p_d) and form=='PHOTOGRAPH':
                        toc_proof=toc_evidence(row)
                        if toc_proof:
                            source=Path(row['document']['ref']);source=source if source.is_absolute() else P/source
                            document=source.read_bytes();assert hashlib.sha256(document).hexdigest()==row['document']['sha256']
                            for use in row['uses']:
                                for span in use['spans']:assert document[span['start_byte']:span['end_byte']].decode('utf-8')==span['text']
                            path=Path(resolver[row['image']['ref']]);path=path if path.is_absolute() else R/path;data=path.read_bytes()
                            assert hashlib.sha256(data).hexdigest()==row['image']['sha256']
                            response=call(url,data,row['image']['mime'],OBS,'请仅观察当前图片。',256);G.write(out/f'toc-observation-{index}.json',response);toc_calls+=1
                            toc_observation=parse(response)
                            if observation_valid(toc_observation) and no_text_photo(toc_observation):verdict='D';toc_override=True
                    item={'sample_id':sid,'raw_decision':raw,'p_d':p_d,'guard_triggered':route(raw,p_d),'form':form,'final_decision':verdict,
                          'toc_evidence':toc_proof,'toc_observation':toc_observation,'toc_override':toc_override,
                          'image_sha256':row['image']['sha256'],'response_sha256':response_sha,'guard_cached':bool(cached)}
                    stream.write(json.dumps(item,ensure_ascii=False)+'\n');stream.flush();results.append(item)
                    G.write(out/'progress.json',{'completed':len(results),'total':len(rows),'guard_calls':calls})
            G.write(out/'prediction-freeze.json',{'status':'COMPLETE_FROZEN_BEFORE_SCORE','count':len(results),'predictions_sha256':N.file_sha(out/'predictions.jsonl'),
                'preregistration_sha256':N.file_sha(out/'preregistration.json'),'inputs_sha256':N.file_sha(inputs),'fresh_guard_calls':calls,'gold_opened':False,
                'load_seconds':load_seconds,'elapsed_seconds':time.monotonic()-began,'cached_guard_calls':cached_calls,'fresh_toc_observation_calls':toc_calls,
                'context_guard_preregistration_sha256':N.file_sha(out/'context-guard-preregistration.json'),'scope':'cached first pass + incremental selective guard; not full cold speed'})
            G.write(P/'reports/zero-delete-99-20261007/photo-guard-pointer.json',{'run':str(out),'freeze_sha256':N.file_sha(out/'prediction-freeze.json')})
        finally:
            if server is not None:
                server.terminate()
                try:server.wait(timeout=20)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
                G.write(out/'server-unload.json',{'owned_server_exited':True,'exit_code':server.returncode})
    print('PHOTO_GUARD_PROBE_COMPLETE')

if __name__=='__main__':main()
