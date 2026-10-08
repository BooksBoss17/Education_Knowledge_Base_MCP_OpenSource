"""Owned source preparation, frozen classification and bounded result export."""
import argparse,hashlib,json,os,runpy,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];R=P.parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json.dumps(value,ensure_ascii=False,indent=2).encode('utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def emit(path,values):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        for value in values:f.write(json.dumps(value,ensure_ascii=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args()
    out=Path(a.run_dir).resolve();request=read(R/'job-request.json');runtime=request['runtime'];started=time.monotonic()
    write(R/'state.json',{'status':'PREPARING_SOURCE','run_dir':str(out)})
    env=os.environ.copy()
    for k in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','__PYVENV_LAUNCHER__','PYTHONEXECUTABLE'):env.pop(k,None)
    command=[runtime['source_python'],'-I',str(P/'tools/source_prepare.py'),'--package',request['package_dir'],
             '--job-root',str(R),'--libraries',str(R)]
    if request.get('source_pdf'):command+=['--source-pdf',request['source_pdf']]
    with (out/'source-preparation.log').open('wb') as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    assert code==0,'SOURCE_PREPARATION_FAILED'
    prepared=read(R/'source-preparation.json');selected=request.get('asset_names')
    inputs=Path(prepared['first_inputs']);bound=Path(prepared['bound_inputs'])
    if selected:
        assets=read(R/'asset-map.json');wanted=set(selected)
        assert wanted<={x['asset_name'] for x in assets},'UNKNOWN_ASSET_SELECTION'
        members={x['sample_id'] for x in assets if x['asset_name'] in wanted}
        short=out/'selection';short.mkdir()
        emit(short/'first.inputs.jsonl',(x for x in rows(inputs) if x['sample_id'] in members))
        emit(short/'bound.inputs.jsonl',(x for x in rows(bound) if x['sample_id'] in members))
        inputs,bound=short/'first.inputs.jsonl',short/'bound.inputs.jsonl'
    count=sum(1 for _ in rows(inputs));assert count>0
    sys.path.insert(0,str(P/'tools'))
    plan_dir=P/'reports/cold-semantic';plan_dir.mkdir(parents=True,exist_ok=True)
    first_plan={'inputs':str(inputs),'inputs_sha256':sha(inputs),'expected_members':count,'threshold':.8,
        'policy_fingerprint':prepared['policy_fingerprints']['0.8'],'resolver_path':prepared['resolver'],'resolver_sha256':sha(prepared['resolver']),
        'freeze_tag':'formal-source-bound-firstpass','result_pointer':'cold-semantic/firstpass-pointer.json'}
    write(plan_dir/'firstpass-plan.json',first_plan)
    write(plan_dir/'guard-template.json',{'inputs':str(bound),'inputs_sha256':sha(bound),'expected_members':count,
          'resolver_path':prepared['resolver'],'resolver_sha256':sha(prepared['resolver'])})
    local_plan={**first_plan,'threshold':.5,'policy_fingerprint':prepared['policy_fingerprints']['0.5']}
    write(plan_dir/'locality-template.json',local_plan)
    dependencies={str(path):sha(path) for path in sorted((P/'tools').glob('*.py'))}
    flow_plan={'expected_members':count,'firstpass_plan':str(plan_dir/'firstpass-plan.json'),'bound_inputs':str(bound),
        'resolver':prepared['resolver'],'guard_template':str(plan_dir/'guard-template.json'),'locality_template':str(plan_dir/'locality-template.json'),
        'features':prepared['features'],'features_sha256':prepared['features_sha256'],'system_python':runtime['source_python'],
        'dependencies':dependencies,'result_pointer':str(R/'classification-pointer.json'),
        'semantic_profiles':{'toc':'P24','body':'P24-small-batch','dependency':'P24-small-batch'}}
    plan=plan_dir/'pipeline-plan.json';write(plan,flow_plan)
    classification=out/'classification';classification.mkdir()
    write(R/'state.json',{'status':'CLASSIFYING','run_dir':str(out),'classification_dir':str(classification),'assets':count})
    previous=sys.argv
    try:
        sys.argv=[str(P/'tools/run_cold_semantic_pipeline.py'),'--run-dir',str(classification),'--plan',str(plan)]
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:sys.argv=previous
    flow=read(classification/'pipeline-result.json');predictions=Path(flow['final_predictions'])
    assert sha(predictions)==flow['final_predictions_sha256']
    mapping={x['sample_id']:x for x in read(R/'asset-map.json')};counts={'K':0,'D':0,'U':0};results=R/'results.jsonl'
    with results.open('x',encoding='utf-8',newline='\n') as stream:
        for row in rows(predictions):
            decision=row['final_decision'];counts[decision]+=1;asset=mapping[row['sample_id']]
            stream.write(json.dumps({**asset,'decision':decision,'action':{'K':'KEEP','D':'EXCLUDE_CANDIDATE','U':'REVIEW_KEEP'}[decision],
                'source_modified':False,'decision_fingerprint':row['input_fingerprint']},ensure_ascii=False)+'\n')
    summary={'status':'COMPLETE','assets':count,'counts':counts,'elapsed_seconds':time.monotonic()-started,
        'results_path':str(results),'results_sha256':sha(results),'predictions_sha256':sha(predictions),
        'scheme_sha256':request['scheme_sha256'],'source_preparation':prepared,'run_dir':str(out),
        'source_files_modified':False,'automatic_deletion_executed':False,'quality_on_new_input_not_known_without_labels':True}
    write(R/'summary.json',summary);write(R/'state.json',summary);print('FORMAL_SOURCE_BOUND_CLASSIFICATION_COMPLETE')
if __name__=='__main__':main()
