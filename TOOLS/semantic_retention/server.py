"""Standalone stdio MCP adapter. No model imports or source mutations here."""
import argparse,contextlib,json,os,re,runpy,subprocess,sys,time,uuid
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bootstrap import read,write,sha,runtime_config,initialize_job

TOOL=Path(__file__).resolve().parent
MCP=TOOL.parents[1]
HOME=Path(os.environ['LOCALAPPDATA'])/'BSR'
JOBS=HOME/'j'
MAX_MESSAGE=1024*1024

def allowed(path):
    roots=json.loads(os.environ.get('SEMANTIC_RETENTION_ALLOWED_ROOTS','[]'))
    if not isinstance(roots,list) or not roots:raise ValueError('Set SEMANTIC_RETENTION_ALLOWED_ROOTS to a JSON array of allowed absolute directories')
    p=Path(path)
    if not p.is_absolute():raise ValueError('Absolute input path required')
    p=p.resolve(strict=True)
    if not any(Path(r).is_absolute() and p.is_relative_to(Path(r).resolve(strict=True)) for r in roots):
        raise ValueError('INPUT_OUTSIDE_ALLOWED_ROOTS')
    return p

def validate_request(args):
    if set(args)-{'package_dir','source_pdf','asset_names'}:raise ValueError('Unknown arguments')
    package=allowed(args['package_dir'])
    if not package.is_dir():raise ValueError('Package directory required')
    for name in ('document.md','conversion_report.json'):
        p=(package/name).resolve(strict=True)
        if not p.is_relative_to(package) or not p.is_file():raise ValueError('Invalid source package')
    report=read(package/'conversion_report.json')
    pdf=allowed(args.get('source_pdf') or report['source']['path'])
    if not pdf.is_file():raise ValueError('Source PDF required')
    selected=args.get('asset_names')
    if selected is not None and (not isinstance(selected,list) or not selected or
            len(selected)>20000 or len(set(selected))!=len(selected) or not all(isinstance(x,str) for x in selected)):
        raise ValueError('asset_names must contain distinct asset names')
    return {'package_dir':str(package),'source_pdf':str(pdf),'asset_names':selected}

@contextlib.contextmanager
def job_lock():
    import msvcrt
    HOME.mkdir(parents=True,exist_ok=True)
    with (HOME/'active.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('CLASSIFIER_JOB_ALREADY_RUNNING')
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def job_path(job_id):
    if not re.fullmatch(r'[0-9a-f]{12}',job_id):raise ValueError('Invalid job_id')
    p=JOBS/job_id
    if not p.is_dir():raise ValueError('Unknown job_id')
    return p

def doctor(deep=False):
    spec=read(TOOL/'SCHEME.json');runtime=runtime_config(TOOL)
    assert Path(runtime['inference_python']).is_file()
    assert Path(runtime['source_python']).is_file()
    assert sha(TOOL/'dist/engine.zip')==spec['engine_zip_sha256']
    checked=0
    for definition in spec['models'].values():
        root=MCP/'MODELS'/definition['directory'];manifest=read(root/'MODEL_MANIFEST.json')
        assert manifest['model_fingerprint']==definition['model_fingerprint']
        for item in manifest['files']:
            p=(root/item['path']).resolve();assert p.is_relative_to(root.resolve())
            assert p.stat().st_size==item['bytes']
            if deep:assert sha(p)==item['sha256']
            checked+=1
    if deep:
        for item in runtime['lock']['llama_files']:
            assert sha(Path(runtime['llama_home'])/item['file'])==item['sha256']
    return {'status':'READY','release':spec['release_id'],'deep':deep,'model_files_checked':checked,
        'bemarkdown_auto_integrated':False,'automatic_deletion':False,'runtime':runtime['inference_python']}

def start(args):
    request=validate_request(args)
    with job_lock():
        JOBS.mkdir(parents=True,exist_ok=True)
        jid=uuid.uuid4().hex[:12];job=JOBS/jid;job.mkdir()
        write(job/'request.json',request);write(job/'state.json',{'status':'QUEUED','job_id':jid})
        with (job/'driver.log').open('wb') as log:
            p=subprocess.Popen([sys.executable,'-I',str(TOOL/'server.py'),'--run-job',jid],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        write(job/'driver.json',{'pid':p.pid,'created_epoch':time.time()})
    # Driver obtains the cross-server lock after the request is committed.
    return {'job_id':jid,'status':'QUEUED','status_tool':'semantic_retention_status'}

def status(args):
    job=job_path(args['job_id']);state=read(job/'state.json')
    return {**state,'job_id':args['job_id']}

def results(args):
    job=job_path(args['job_id']);state=read(job/'state.json')
    if state['status']!='COMPLETE':return {'status':state['status'],'job_id':args['job_id'],'results_available':False}
    offset=int(args.get('offset',0));limit=int(args.get('limit',100))
    if offset<0 or not 1<=limit<=200:raise ValueError('Invalid result page')
    result=[]
    with (job/'work/results.jsonl').open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i>=offset+limit:break
            if i>=offset:result.append(json.loads(line))
    return {'status':'COMPLETE','rows':result,'offset':offset,'next_offset':offset+len(result),
        'total':state['assets'],'summary':state['counts'],'source_modified':False}

def run_job(jid):
    job=job_path(jid)
    try:
        # Brief bounded retry only for the start handler's handover of the lock.
        with job_lock():
            request=validate_request(read(job/'request.json'))
            write(job/'state.json',{'status':'INITIALIZING','job_id':jid})
            ctx=initialize_job(TOOL,MCP,job/'work',request)
            command=[ctx['python'],'-I',ctx['supervisor'],ctx['worker'],
                '--expected-private-gib','5.9','--expected-working-gib','4.9','--expected-gpu-gib','5.6','--deployment-budget']
            write(job/'state.json',{'status':'RUNNING','job_id':jid,'automatic_deletion':False})
            with (job/'supervisor.log').open('w',encoding='utf-8',buffering=1) as log:
                previous=sys.argv
                try:
                    sys.argv=command[2:]
                    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                        try:runpy.run_path(ctx['supervisor'],run_name='__main__');code=0
                        except SystemExit as exc:code=int(exc.code or 0)
                finally:sys.argv=previous
            evidence=list((job/'work/tmp/rt-train/runs').glob('native-model-*/result.json'))
            receipt=read(evidence[0]) if len(evidence)==1 else {}
            summary=job/'work/summary.json'
            if code!=0 or receipt.get('status')!='COMPLETED' or not summary.is_file():
                raise RuntimeError('SUPERVISED_JOB_FAILED: '+json.dumps({'exit_code':code,'receipt':receipt},ensure_ascii=False))
            value=read(summary)
            value.update(job_id=jid,supervision_receipt=str(evidence[0]),supervisor_all_owned_exited=receipt['all_owned_exited'])
            write(job/'state.json',value)
    except Exception as exc:
        write(job/'state.json',{'status':'FAILED','job_id':jid,'error':str(exc),'source_modified':False})
        raise

def schema(props,required=()):return {'type':'object','properties':props,'required':list(required),'additionalProperties':False}
STRING={'type':'string'}
TOOLS=[
 {'name':'semantic_retention_doctor','description':'Check frozen classifier installation, optionally hash all model/runtime files.',
  'inputSchema':schema({'deep':{'type':'boolean'}})},
 {'name':'semantic_retention_start','description':'Start one read-only source-bound image classification job. Returns a job ID; poll status. Never deletes source images.',
  'inputSchema':schema({'package_dir':STRING,'source_pdf':STRING,'asset_names':{'type':'array','items':STRING}},('package_dir',))},
 {'name':'semantic_retention_status','description':'Read job status without starting or repeating inference.',
  'inputSchema':schema({'job_id':STRING},('job_id',))},
 {'name':'semantic_retention_results','description':'Read paginated KEEP, EXCLUDE_CANDIDATE or REVIEW_KEEP decisions from a completed job.',
  'inputSchema':schema({'job_id':STRING,'offset':{'type':'integer','minimum':0},'limit':{'type':'integer','minimum':1,'maximum':200}},('job_id',))}]

def dispatch(method,params):
    if method=='initialize':return {'protocolVersion':'2024-11-05','capabilities':{'tools':{}},'serverInfo':{'name':'semantic-retention','version':'20261008.1'}}
    if method=='ping':return {}
    if method=='tools/list':return {'tools':TOOLS}
    if method=='tools/call':
        functions={'semantic_retention_doctor':lambda a:doctor(a.get('deep',False)),
                   'semantic_retention_start':start,'semantic_retention_status':status,'semantic_retention_results':results}
        try:
            value=functions[params['name']](params.get('arguments',{}))
            return {'content':[{'type':'text','text':json.dumps(value,ensure_ascii=False)}],'isError':False}
        except Exception as exc:return {'content':[{'type':'text','text':str(exc)}],'isError':True}
    raise ValueError('Unknown method')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-job');ap.add_argument('--doctor',action='store_true');a=ap.parse_args()
    if a.run_job:return run_job(a.run_job)
    if a.doctor:print(json.dumps(doctor(True),ensure_ascii=False));return
    while True:
        line=sys.stdin.buffer.readline(MAX_MESSAGE+1)
        if not line:break
        if len(line)>MAX_MESSAGE:raise ValueError('MCP request exceeds 1 MiB')
        request=None
        try:
            request=json.loads(line)
            if 'id' not in request:continue
            response={'jsonrpc':'2.0','id':request['id'],'result':dispatch(request['method'],request.get('params',{}))}
        except Exception as exc:
            response={'jsonrpc':'2.0','id':request.get('id') if isinstance(request,dict) else None,
                      'error':{'code':-32603,'message':str(exc)}}
        sys.stdout.buffer.write((json.dumps(response,ensure_ascii=False)+'\n').encode('utf-8'));sys.stdout.buffer.flush()

if __name__=='__main__':main()
