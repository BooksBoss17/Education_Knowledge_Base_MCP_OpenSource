"""Run classification inside an already-owned conversion job; never enqueue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args();payload=json.loads(args.request.read_text(encoding='utf-8'))
    tool=Path(__file__).resolve().parent
    sys.path.insert(0,str(tool))
    import server as S
    context=S.manager_bridge.runtime(S.MCP).trusted_context(S.MCP)
    if not context:raise RuntimeError('INLINE_SCREENING_REQUIRES_MANAGED_CONVERSION')
    package=Path(payload['package_dir']).resolve(strict=True)
    os.environ['SEMANTIC_RETENTION_ALLOWED_ROOTS']=json.dumps([str(package)])
    request=S.validate_request({k:payload[k] for k in ['package_dir','source_pdf','asset_names']})
    source_fingerprint=S.manager_bridge.source_fingerprint(request)
    digest=hashlib.sha256((context['job_id']+source_fingerprint+S.manager_bridge.config_fingerprint(tool)+'purpose-v1').encode()).hexdigest()[:12]
    job=S.JOBS/digest;job.mkdir(parents=True,exist_ok=True)
    identity=dict(parent_manager_job=context['job_id'],source_fingerprint=source_fingerprint)
    if (job/'inline-binding.json').exists():
        if S.read(job/'inline-binding.json')!=identity:raise RuntimeError('INLINE_JOB_IDENTITY_CHANGED')
        state=S.read(job/'state.json') if (job/'state.json').exists() else {}
        if state.get('status')!='COMPLETE':raise RuntimeError('INLINE_JOB_INCOMPLETE_REQUIRES_EXPLICIT_RECOVERY')
    else:
        S.write(job/'inline-binding.json',identity);S.write(job/'request.json',request)
        os.environ['BEMARKDOWN_ASSESS_IMAGE_PURPOSE']='1'
        S.run_job(digest)
        state=S.read(job/'state.json')
    if state.get('status')!='COMPLETE':raise RuntimeError('INLINE_CLASSIFICATION_FAILED:'+json.dumps(state))
    rows=[json.loads(line) for line in (job/'work/results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    S.write(args.result,dict(status='COMPLETE',job_id=digest,parent_manager_job=context['job_id'],
        source_fingerprint=source_fingerprint,rows=rows,receipt=state))

if __name__=='__main__':main()
