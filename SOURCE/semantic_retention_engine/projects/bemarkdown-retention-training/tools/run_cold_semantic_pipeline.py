"""Complete fresh-response pipeline; sequential models and short-lived source builders."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];R=P.parents[1];T=P/'tools'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--plan',required=True);a=ap.parse_args()
    out=Path(a.run_dir).resolve();plan=read(a.plan);assert out.is_relative_to(R/'tmp/rt-train/runs')
    for path,value in plan['dependencies'].items():assert sha(path)==value,'DEPENDENCY_CHANGED_AFTER_PREREGISTRATION'
    started=time.monotonic();events=[]
    write(out/'preregistration.json',{'plan':plan,'plan_sha256':sha(a.plan),'cold_response_cache':True,
        'source_metadata_cache_disclosed':True,'gold_opened':False,'worker_sha256':sha(__file__)})
    def run(script,arguments,label,system=False):
        command=([plan['system_python'],'-I'] if system else [sys.executable])+[str(T/script),*map(str,arguments)]
        env=os.environ.copy()
        if system:
            for k in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','__PYVENV_LAUNCHER__','PYTHONEXECUTABLE'):env.pop(k,None)
        before=time.monotonic()
        with (out/(label+'.log')).open('wb') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        events.append({'stage':label,'exit_code':result.returncode,'seconds':time.monotonic()-before,'command':command})
        write(out/'pipeline-progress.json',events)
        assert result.returncode==0,'STAGE_FAILED: '+label
    def merge(base,override,freeze,kind,directory):
        dest=directory/'composed.predictions.jsonl'
        run('cold_merge_stage.py',['--inputs',plan['bound_inputs'],'--base',base,'--overrides',override,
             '--evidence-freeze',freeze,'--output',dest,'--kind',kind],kind+'-merge')
        return dest
    prefix_seconds=0
    if plan.get('resume_after_guard'):
        prior=Path(plan['resume_after_guard']);terminal=read(prior/'result.json')
        assert terminal['status']=='WORKER_FAILED' and terminal['all_owned_exited']
        old_plan=read(prior/'preregistration.json')['plan']
        for key in ('firstpass_plan','bound_inputs','resolver','guard_template'):
            assert plan[key]==old_plan[key] and sha(plan[key])==old_plan['dependencies'][plan[key]]
        first,guard=prior/'firstpass',prior/'guard'
        ff,gf=read(first/'predictions.freeze.json'),read(guard/'prediction-freeze.json')
        assert sha(first/'predictions.jsonl')==ff['prediction_file_sha256']
        assert sha(guard/'predictions.jsonl')==gf['predictions_sha256'] and gf['cached_guard_calls']==0
        current=guard/'composed.predictions.jsonl';cf=read(current.with_suffix('.freeze.json'))
        assert sha(current)==cf['prediction_file_sha256'] and cf['expected_inputs_sha256']==sha(plan['bound_inputs'])
        assert read(first/'server-unload.json')['owned_server_exited'] and read(guard/'server-unload.json')['owned_server_exited']
        prior_events=read(prior/'pipeline-progress.json')
        prefix_seconds=sum(e['seconds'] for e in prior_events if e['stage'] in ('firstpass','photo-form-guard','guard-merge'))
        write(out/'prefix-continuation-proof.json',{'same_logical_cold_run':True,'prior_run':str(prior),
            'prior_result_sha256':sha(prior/'result.json'),'firstpass_freeze_sha256':sha(first/'predictions.freeze.json'),
            'guard_freeze_sha256':sha(guard/'prediction-freeze.json'),'prefix_seconds':prefix_seconds,
            'rerun_completed_model_requests':False,'interrupted_stage':'locality native input planning filename contract'})
    else:
        first=out/'firstpass';first.mkdir()
        run('run_gguf_partition_predictions.py',['--run-dir',first,'--evaluation-plan',plan['firstpass_plan']],'firstpass')
        assert read(first/'server-unload.json')['owned_server_exited']
        guard=out/'guard';guard.mkdir();guard_plan=read(plan['guard_template'])
        guard_plan.update(inputs=plan['bound_inputs'],inputs_sha256=sha(plan['bound_inputs']),
            baseline_predictions=str(first/'predictions.jsonl'),baseline_predictions_sha256=sha(first/'predictions.jsonl'),
            expected_members=plan['expected_members'],resolver_path=plan['resolver'],resolver_sha256=sha(plan['resolver']))
        write(guard/'plan.json',guard_plan)
        run('probe_zero99_photo_guard.py',['--run-dir',guard,'--plan',guard/'plan.json','--toc-thumbnail-check'],'photo-form-guard')
        assert read(guard/'server-unload.json')['owned_server_exited']
        assert read(guard/'prediction-freeze.json')['cached_guard_calls']==0
        current=merge(first/'predictions.jsonl',guard/'predictions.jsonl',guard/'prediction-freeze.json','guard',guard)
    config={k:plan[k] for k in ('bound_inputs','features','features_sha256')}
    config.update(firstpass_predictions=str(first/'predictions.jsonl'),guard_predictions=str(guard/'predictions.jsonl'))
    write(out/'source-config.json',config)
    local_inputs=out/'locality-inputs'
    run('cold_semantic_inputs.py',['--config',out/'source-config.json','--stage','locality','--baseline',current,'--out',local_inputs],'locality-input-builder')
    count=read(local_inputs/'input-freeze.json')['count']
    if count:
        local_plan=read(plan['locality_template']);local_plan.update(inputs=str(local_inputs/'locality.inputs.jsonl'),
            inputs_sha256=sha(local_inputs/'locality.inputs.jsonl'),expected_members=count,
            resolver_path=plan['resolver'],resolver_sha256=sha(plan['resolver']),
            result_pointer=str(Path('cold-semantic')/(out.name+'-locality-pointer.json')))
        local_plan_path=P/'reports/cold-semantic'/(out.name+'-locality-plan.json');write(local_plan_path,local_plan)
        locality=out/'locality';locality.mkdir()
        run('run_gguf_partition_predictions.py',['--run-dir',locality,'--evaluation-plan',local_plan_path],'locality-inference')
        assert read(locality/'server-unload.json')['owned_server_exited']
        current=merge(current,locality/'predictions.jsonl',locality/'predictions.freeze.json','locality',locality)
    components=out/'components'
    run('cold_source_components.py',['--run-dir',components,'--baseline',current],'source-components',system=True)
    current=merge(current,components/'evidence.jsonl',components/'freeze.json','component',components)
    for stage in ('toc','body','dependency'):
        directory=out/(stage+'-inputs')
        run('cold_semantic_inputs.py',['--config',out/'source-config.json','--stage',stage,'--baseline',current,'--out',directory],stage+'-input-builder')
        if stage=='body':config['body_manifest']=str(directory/'manifest.json');write(out/'source-config.json',config)
        if not read(directory/'input-freeze.json')['count']:continue
        inference=out/stage;inference.mkdir()
        arguments=['--run-dir',inference,'--manifest',directory/'manifest.json','--kind',stage,
                   '--profile',plan.get('semantic_profiles',{}).get(stage,'P24')]
        if stage=='toc':arguments+=['--source-proof',directory/'binding-proof.json']
        run('cold_semantic_review.py',arguments,stage+'-semantic-inference')
        assert read(inference/'server-unload.json')['owned_server_exited']
        assert read(inference/'decision-freeze.json')['response_cache_hits']==0
        current=merge(current,inference/'decisions.jsonl',inference/'decision-freeze.json',stage,inference)
    choices=out/'source-choice'
    run('cold_source_choice.py',['--run-dir',choices,'--baseline',current],'source-image-options',system=True)
    current=merge(current,choices/'decisions.json',choices/'freeze.json','choice',choices)
    result={'status':'COMPLETE_COLD_SEMANTIC_PIPELINE_FROZEN_NOT_SCORED','final_predictions':str(current),
       'final_predictions_sha256':sha(current),'count':plan['expected_members'],'bound_inputs':plan['bound_inputs'],
       'bound_inputs_sha256':sha(plan['bound_inputs']),'elapsed_seconds':prefix_seconds+time.monotonic()-started,'stages':events,
       'resume_after_guard':plan.get('resume_after_guard'),'current_segment_seconds':time.monotonic()-started,
       'prefix_seconds':prefix_seconds,'uninterrupted_cold_run':not bool(plan.get('resume_after_guard')),
       'no_response_cache':True,'gold_opened':False,'plan_sha256':sha(a.plan),
       'scope':'all classifier branches on prepared original MD, source metadata, and images; excludes upstream PDF/OCR conversion',
       'source_only_feature_cache':{'path':plan['features'],'sha256':plan['features_sha256']}}
    write(out/'pipeline-result.json',result);write(plan['result_pointer'],result)
    print('COMPLETE_COLD_SEMANTIC_PIPELINE_FROZEN_NOT_SCORED')

if __name__=='__main__':main()
