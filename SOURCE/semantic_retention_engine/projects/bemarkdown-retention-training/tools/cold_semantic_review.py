"""Fresh semantic inference with bounded owned-service lifetime and frozen per-use outputs."""
import argparse,json,socket,subprocess,time,urllib.error
from pathlib import Path
import probe_gguf_image_runtime as G
import retention_inference as N
from contracts import digest
import probe_zero99_toc_semantic as TOC
import probe_zero99_body_photo_semantic_v3 as BODY
import probe_zero99_body_photo_dependency as DEP
from repair_zero99_toc_contract import PROMPT_V2
from toc_semantic_contract_v2 import normalize_result,repair_schema
from toc_footer_position import footer_witness

def request_spec(row,use,kind,context,repair=False):
    if kind=='toc':
        prompt=PROMPT_V2 if repair else TOC.PROMPT
        schema=repair_schema(len(use['source_lines'])) if repair else TOC.SCHEMA
        maximum=384 if repair else 160
    else:
        module=BODY if kind=='body' else DEP
        prompt,schema,maximum=module.PROMPT,module.request_schema(len(use['source_lines'])),384
    payload={'target_bbox_pdf_pt':use['target_bbox_pdf_pt'],'page_size_pdf_pt':use['source_page_size_pdf_pt'],
             'source_lines':'\n'.join(f'{i}: {s}' for i,s in enumerate(use['source_lines']))}
    if not repair:payload['source_scope']='one actual source use, not crop text transcription'
    text=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    identity=digest({'image':row['image']['sha256'],'text':text,'prompt':prompt,'schema':schema,
                     'maximum':maximum,'context':context})
    return prompt,schema,maximum,text,identity


class Reviewer:
    def __init__(self,out,kind,profile='P24'):
        self.out,self.kind=out,kind
        self.profile=profile
        self.server=None;self.log=None;self.calls=0;self.segment_calls=0;self.segments=[]
        self.recoveries=[]
        self.pending_retry_fingerprint=None
        assets=G.read(G.P/'reports/zero-delete-99-20261007/reviewer9b-assets.json')
        paths={}
        for x in assets['files']:
            assert N.file_sha(x['path'])==x['sha256'];paths[x['file']]=x['path']
        self.paths=paths;self.assets=assets
        self.runtime=G.read(G.P/'reports/gguf-candidate-export.json')['runtime_files']
        for x in self.runtime:assert N.file_sha(G.R/'tmp/llama-retention'/x['file'])==x['sha256']
        self.context=4096 if kind=='toc' else 5120

    def stop(self):
        if self.server is not None:
            if self.server.poll() is None:
                self.server.terminate()
                try:self.server.wait(timeout=20)
                except subprocess.TimeoutExpired:self.server.kill();self.server.wait(timeout=10)
            self.segments[-1].update(exit_code=self.server.returncode,calls=self.segment_calls)
            self.server=None
            self.log.close();self.log=None
            G.write(self.out/'server-lifecycle.json',self.segments)

    def ready(self):
        if self.segment_calls>=12:self.stop()
        if self.server is not None:return
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        self.url=f'http://127.0.0.1:{port}'
        cmd=G.server_command(self.paths['Qwen3.5-9B-Q3_K_M.gguf'],self.paths['mmproj-F16.gguf'],port)
        batch,ubatch=(64,32) if self.profile=='P24-small-batch' else (128,64)
        for flag,value in [('-ngl',24),('-b',batch),('-ub',ubatch),('-c',self.context)]:cmd[cmd.index(flag)+1]=str(value)
        self.log=(self.out/f'server-{len(self.segments):03d}.log').open('wb')
        start=time.monotonic()
        self.server=subprocess.Popen(cmd,stdout=self.log,stderr=subprocess.STDOUT,cwd=self.out,creationflags=subprocess.CREATE_NO_WINDOW)
        self.segments.append({'pid':self.server.pid,'command':cmd})
        G.await_ready(self.server,self.url)
        self.segments[-1]['load_seconds']=time.monotonic()-start;self.segment_calls=0

    def request(self,row,use,repair=False):
        identity=request_spec(row,use,self.kind,self.context,repair)[-1]
        start_attempt=0
        if getattr(self,'pending_retry_fingerprint',None) is not None:
            assert identity==self.pending_retry_fingerprint,'RESUMED_FAILURE_REQUEST_CHANGED'
            self.pending_retry_fingerprint=None;start_attempt=1
        for attempt in range(start_attempt,2):
            try:
                result,receipt=self._request_once(row,use,repair)
                if attempt:
                    recovery=self.recoveries[-1]
                    assert recovery['request_fingerprint']==receipt['request_fingerprint']
                    recovery.update(status='RECOVERED_IDENTICAL_REQUEST',recovered=True,response_sha256=receipt.get('response_sha256'))
                    receipt['allocation_recovery']=recovery
                    G.write(self.out/'request-recoveries.json',self.recoveries)
                    G.write(self.out/'request-failure.json',{**recovery,'recovered':True})
                return result,receipt
            except urllib.error.HTTPError as exc:
                body=exc.read(16384).decode('utf-8',errors='replace')
                identity=request_spec(row,use,self.kind,self.context,repair)[-1]
                failure={'status':'REQUEST_FAILED','sample_id':row['sample_id'],'use_id':use['use_id'],
                    'request_fingerprint':identity,'http_status':exc.code,'body':body,'attempt':attempt+1,
                    'successful_calls_before_failure':self.calls,'recovered':False}
                G.write(self.out/f'request-error-{self.calls:05d}-{attempt}.json',failure)
                G.write(self.out/'request-failure.json',failure)
                if self.segments:self.segments[-1]['failed_calls']=self.segments[-1].get('failed_calls',0)+1
                if exc.code!=500 or 'bad allocation' not in body.lower() or attempt:raise
                self.stop()
                self.recoveries.append({**failure,'status':'FRESH_SERVICE_RETRY_PENDING','owned_server_exited_before_retry':True})
                G.write(self.out/'request-recoveries.json',self.recoveries)
        raise AssertionError('UNREACHABLE_RETRY_STATE')

    def _request_once(self,row,use,repair=False):
        self.ready()
        prompt,schema,maximum,text,identity=request_spec(row,use,self.kind,self.context,repair)
        tokens=len(G.request(self.url+'/tokenize',{'content':prompt+'\n'+text,'add_special':True})['tokens'])
        if tokens+1024+256+maximum>self.context:
            return None,{'error':'REQUEST_BUDGET_EXCEEDED','text_tokens':tokens,'request_fingerprint':identity}
        path=Path(row['image_path']);assert N.file_sha(path)==row['image']['sha256']
        messages=[{'role':'system','content':prompt},{'role':'user','content':[{'type':'image','image_ref':'local'},{'type':'text','text':text}]}]
        start=time.monotonic()
        try:
            response=G.request(self.url+'/v1/chat/completions',{'model':'local-retention',
                'messages':G.materialize_messages(messages,path.read_bytes(),row['image']['mime']),
                'temperature':0,'seed':20261004,'max_tokens':maximum,'cache_prompt':False,
                'chat_template_kwargs':{'enable_thinking':False},
                'response_format':{'type':'json_schema','json_schema':{'name':'source_toc_use' if self.kind=='toc' else 'source_body_photo_use','strict':True,'schema':schema}}},timeout=240)
        except Exception as exc:
            G.write(self.out/'request-failure.json',{'type':type(exc).__name__,'message':str(exc),'sample_id':row['sample_id'],
                'use_id':use['use_id'],'request_fingerprint':identity,'calls_completed':self.calls})
            raise
        response_path=self.out/f'response-{self.calls:05d}.json';G.write(response_path,response)
        self.calls+=1;self.segment_calls+=1
        try:value=json.loads(response['choices'][0]['message'].get('content') or '')
        except (ValueError,TypeError):value=None
        return value,{'response_sha256':N.file_sha(response_path),'request_fingerprint':identity,
                     'text_tokens':tokens,'elapsed_seconds':time.monotonic()-start,'usage':response.get('usage')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--manifest',required=True)
    ap.add_argument('--kind',choices=['toc','body','dependency'],required=True);ap.add_argument('--source-proof')
    ap.add_argument('--profile',choices=['P24','P24-small-batch'],default='P24')
    ap.add_argument('--resume-prefix')
    a=ap.parse_args();out=Path(a.run_dir).resolve();out.mkdir(exist_ok=True)
    assert out.is_relative_to(G.R/'tmp/rt-train/runs')
    items=G.read(a.manifest);assert len({x['sample_id'] for x in items})==len(items)
    # Source preparation is independent of model judgments. No label reader is imported or called.
    footer={(x['sample_id'],x['use_id']):x['nodes'] for x in G.read(a.source_proof)} if a.source_proof else {}
    reviewer=Reviewer(out,a.kind,a.profile);started=time.monotonic();results=[]
    prefix={};prefix_calls=0
    if a.resume_prefix:
        parent=Path(a.resume_prefix).resolve();prior=G.read(parent/'preregistration.json');partial=G.read(parent/'partial-freeze.json')
        assert N.file_sha(parent/'decisions.jsonl')==partial['decisions_sha256']
        assert prior['manifest_sha256']==N.file_sha(a.manifest) and prior['kind']==a.kind
        assert prior['profile']==a.profile and prior['context_tokens']==reviewer.context
        assert prior['assets']==reviewer.assets and prior['runtime']==reviewer.runtime
        prefix={x['sample_id']:x for x in N.A.read_rows(parent/'decisions.jsonl')}
        assert len(prefix)==partial['count'] and set(prefix)<={x['sample_id'] for x in items}
        for row in items:
            if row['sample_id'] not in prefix:continue
            old=prefix[row['sample_id']];assert old['image_sha256']==row['image']['sha256'] and old['coverage_complete']
            assert N.file_sha(row['image_path'])==row['image']['sha256']
            doc=Path(row['source_document']['ref']);doc=doc if doc.is_absolute() else G.P/doc
            assert N.file_sha(doc)==row['source_document']['sha256']
            uses={x['use_id']:x for x in old['uses']}
            assert set(uses)=={x['use_id'] for x in row['uses']}
            for use in row['uses']:
                assert uses[use['use_id']]['request_fingerprint']==request_spec(row,use,a.kind,reviewer.context)[-1]
                prefix_calls+=int('response_sha256' in uses[use['use_id']])
        pending=G.read(parent/'request-failure.json')
        assert pending['request_fingerprint']==partial['failed_request_fingerprint']
        reviewer.pending_retry_fingerprint=pending['request_fingerprint']
        reviewer.recoveries.append({**pending,'status':'FRESH_SERVICE_RETRY_PENDING','owned_server_exited_before_retry':True,
                                   'prior_cold_run_failure':str(parent/'request-failure.json')})
        G.write(out/'checkpoint-reuse.json',{'same_logical_cold_run':True,'prefix':str(parent),'records':len(prefix),
            'partial_freeze_sha256':N.file_sha(parent/'partial-freeze.json'),'request_specs_all_verified':True,
            'runtime_profile_and_model_assets_unchanged':True})
    G.write(out/'preregistration.json',{'kind':a.kind,'manifest_sha256':N.file_sha(a.manifest),
       'source_proof_sha256':N.file_sha(a.source_proof) if a.source_proof else None,
       'runtime':reviewer.runtime,'assets':reviewer.assets,'source_sha256':N.file_sha(__file__),
       'context_tokens':reviewer.context,'profile':a.profile,'maximum_requests_per_owned_service':12,
       'response_cache_used':False,'gold_opened':False,'toc_invalid_contract_repairs_per_use':1,
       'allocation_failure_fresh_service_retries_per_request':1,'resume_prefix':a.resume_prefix})
    try:
        with (out/'decisions.jsonl').open('x',encoding='utf-8',buffering=1) as f:
            for row in items:
                if row['sample_id'] in prefix:
                    result=prefix[row['sample_id']];results.append(result);f.write(json.dumps(result,ensure_ascii=False)+'\n')
                    G.write(out/'progress.json',{'completed':len(results),'total':len(items),'calls':reviewer.calls,'checkpoint_resumed':True})
                    continue
                uses=[]
                for use in row['uses']:
                    raw,receipt=reviewer.request(row,use)
                    if a.kind=='toc':
                        value,reason=normalize_result(raw,use)
                        if value is None:
                            raw,repair_receipt=reviewer.request(row,use,repair=True)
                            value,reason=normalize_result(raw,use);receipt['contract_repair']=repair_receipt
                        if value is None and reason=='FOOTER_POSITION_NOT_ESTABLISHED' and raw:
                            witness=footer_witness(raw['visible_text'],use['target_bbox_pdf_pt'],use['source_page_size_pdf_pt'],footer.get((row['sample_id'],use['use_id']),[]))
                            if witness:value=raw;reason='FOOTER_GLYPH_POSITION_VERIFIED';receipt['footer_witness']=witness
                    else:
                        module=BODY if a.kind=='body' else DEP
                        value=raw if module.validate_result(raw,len(use['source_lines'])) else None
                        reason='VALID_RESULT' if value else 'INVALID_OUTPUT_CONTRACT'
                    uses.append({'use_id':use['use_id'],'verdict':value['verdict'] if value else 'U',
                                 'valid':value is not None,'result':value,'normalization':reason,**receipt})
                values=[x['verdict'] for x in uses]
                final='K' if 'K' in values else 'U' if not values or 'U' in values else 'D'
                assert {u['use_id'] for u in uses}==set(row['coverage']['covered_use_ids'])
                result={'sample_id':row['sample_id'],'image_sha256':row['image']['sha256'],'uses':uses,
                        'final_decision':final,'coverage_complete':True}
                results.append(result);f.write(json.dumps(result,ensure_ascii=False)+'\n')
                G.write(out/'progress.json',{'completed':len(results),'total':len(items),'calls':reviewer.calls})
    finally:
        reviewer.stop();G.write(out/'server-unload.json',{'owned_server_exited':reviewer.server is None})
    G.write(out/'decision-freeze.json',{'status':'COMPLETE_COLD_SEMANTIC_STAGE_BEFORE_SCORE','count':len(results),
       'decisions_sha256':N.file_sha(out/'decisions.jsonl'),'preregistration_sha256':N.file_sha(out/'preregistration.json'),
       'calls':reviewer.calls,'response_cache_hits':0,'server_load_seconds':sum(x['load_seconds'] for x in reviewer.segments),
       'elapsed_seconds':time.monotonic()-started,'gold_opened':False,'owned_servers_exited':True,
       'checkpoint_records_resumed':len(prefix),'checkpoint_completed_calls':prefix_calls,
       'successful_allocation_recoveries':len(reviewer.recoveries)})
    print(json.dumps({'kind':a.kind,'completed':len(results),'calls':reviewer.calls}))

if __name__=='__main__':main()
