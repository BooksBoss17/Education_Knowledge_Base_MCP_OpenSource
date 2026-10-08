"""Freeze a complete current-run prediction composition without opening labels."""
import argparse,json
from pathlib import Path
import probe_gguf_image_runtime as G
import retention_inference as N
from contracts import digest

def main():
    ap=argparse.ArgumentParser()
    for key in ('inputs','base','overrides','evidence-freeze','output'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--kind',choices=['guard','locality','component','toc','body','dependency','choice'],required=True)
    ap.add_argument('--inference-kind',choices=['cold_inference','cache_replay'],default='cold_inference')
    a=ap.parse_args();base=Path(a.base);bf=G.read(base.with_suffix('.freeze.json'))
    assert N.file_sha(base)==bf['prediction_file_sha256']
    evidence=G.read(a.evidence_freeze)
    possible=[evidence.get(k) for k in ('prediction_file_sha256','predictions_sha256','decisions_sha256','evidence_sha256')]
    assert N.file_sha(a.overrides) in possible,'OVERRIDE_FILE_NOT_BOUND_BY_FREEZE'
    data=G.read(a.overrides) if a.kind=='choice' else list(N.A.read_rows(a.overrides))
    changes={x['sample_id']:x for x in data};assert len(changes)==len(data)
    rows=list(N.A.read_rows(base));assert set(changes)<={x['sample_id'] for x in rows}
    if a.kind=='guard':assert set(changes)=={x['sample_id'] for x in rows}
    policy=digest({'parent':bf['policy_fingerprint'],'kind':a.kind,'evidence_freeze':N.file_sha(a.evidence_freeze),'source':N.file_sha(__file__)})
    model=digest({'parent':bf['model_fingerprint'],'stage_evidence':N.file_sha(a.evidence_freeze)})
    output=Path(a.output)
    with output.open('x',encoding='utf-8') as f:
        for old in rows:
            override=changes.get(old['sample_id'])
            final=(override['proposed_final'] if a.kind=='component' else override['final_decision']) if override else old['final_decision']
            if override and a.kind in ('component','choice'):assert old['final_decision']=='D' and final in ('D','K')
            if override and a.kind in ('toc','body'):assert old['final_decision']=='K'
            if override and a.kind=='dependency':assert old['final_decision']=='D'
            row={**old,'final_decision':final,'status':'policy_abstention' if final=='U' else 'model_response',
                 'model_fingerprint':model,'policy_fingerprint':policy,
                 'input_fingerprint':digest({'parent':old['input_fingerprint'],'override':override,'policy':policy}),
                 'inference_kind':a.inference_kind,'reason_code':'COLD_'+a.kind.upper() if override else old['reason_code']}
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    N.freeze_predictions(a.inputs,output,model,policy,'complete-current-cold-run-stage-before-score')
    G.write(output.with_suffix('.lineage.json'),{'stage':a.kind,'override_count':len(changes),'count':len(rows),
        'base':str(base),'base_freeze_sha256':N.file_sha(base.with_suffix('.freeze.json')),
        'stage_freeze':a.evidence_freeze,'stage_freeze_sha256':N.file_sha(a.evidence_freeze),
        'timing_scope':'elapsed_ms retains firstpass row time; total measured by pipeline wallclock, not summed from rows',
        'external_response_cache':a.inference_kind=='cache_replay','gold_opened':False})
    print(json.dumps({'kind':a.kind,'count':len(rows),'overrides':len(changes)}))

if __name__=='__main__':main()
