"""Default native optimized route. Legacy external completion is never automatic."""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os
from pathlib import Path
import sys

SUFFIXES = {'.pdf', '.docx', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def verified_payload(tool):
    tool=Path(tool).resolve();manifest=read(tool/'ROUTE_MANIFEST.json')
    if manifest.get('engine')!='ACCEPTED_NATIVE_CORE_NO_AUTOMATIC_COMPLETION' or manifest.get('resource_policy')!='ADAPTIVE_RAM_AND_WHOLE_GPU_SINGLE_DEVICE_ZERO':
        raise ValueError('NATIVE_ROUTE_POLICY_CHANGED')
    for relative,digest in manifest['files'].items():
        path=(tool/relative).resolve()
        if not path.is_relative_to(tool) or sha(path)!=digest:
            raise ValueError('NATIVE_ROUTE_PAYLOAD_CHANGED:'+relative)
    return manifest

def parse_args(argv=None):
    p=argparse.ArgumentParser()
    for name in ('source','output-root','mcp-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--batch-source',action='append',default=[])
    p.add_argument('--layout-batch-size',type=int,choices=range(1,5),default=1)
    p.add_argument('--caller-pid',type=int)
    p.add_argument('--retry-job-id')
    p.add_argument('--debug',action='store_true');p.add_argument('--json',action='store_true')
    p.add_argument('--no-image-screening',action='store_true')
    p.add_argument('--screen-existing')
    p.add_argument('--repetition-guard',action='store_true')
    p.add_argument('--legacy-completion',action='store_true')
    return p.parse_args(argv)

def stage_admission_payload(tool,root,manifest):
    forecast=tool/'core/bemarkdown/runtime/RESOURCE_FORECASTS_6GB.json'
    measured=read(forecast)
    stages={name:{'increment':{'private_gib':record['extra_private_bytes']/2**30,
        'working_gib':record['extra_physical_bytes']/2**30,'gpu_mib':record['extra_gpu_mib']},
        'margin':{'private_gib':.03,'working_gib':0,'gpu_mib':0},
        'evidence':[{'path':str(forecast),'sha256':sha(forecast)}]}
        for name,record in measured['stages'].items()}
    return {'version':'measured-stage-admission-v1','native_forecast':{'path':str(forecast),'sha256':sha(forecast)},
        'stages':stages,'native_core_root':str(tool/'core'),'lease_path':str(root/'stage-lease.json'),
        'bindings':[{'path':str(tool/relative),'sha256':digest} for relative,digest in manifest['files'].items()]}

def supervise(tool,cfg,root,manifest):
    sys.path.insert(0,str(tool/'route'))
    import e2e_supervise as legacy
    import native_e2e_supervise as supervision
    controllers=[]
    if cfg.get('caller_pid'):
        caller_pid=cfg['caller_pid'];launcher=None
        if caller_pid!=os.getppid():
            import psutil
            parent=psutil.Process(os.getppid())
            if parent.ppid()!=caller_pid or Path(parent.exe()).resolve()!=Path(sys.executable).resolve():
                raise ValueError('NATIVE_CALLER_IS_NOT_PARENT_OR_VENV_LAUNCHER')
            launcher=legacy.process_info(parent.pid)
            if launcher is None:raise ValueError('NATIVE_LAUNCHER_IDENTITY_UNAVAILABLE')
        caller=legacy.process_info(caller_pid)
        if caller is None:raise ValueError('NATIVE_CALLER_IDENTITY_UNAVAILABLE')
        controllers=[caller]+([launcher] if launcher else [])
    os.environ['BEMARKDOWN_ROUTE_CALLER_PRIVATE_RESERVE_GIB']=str(sum(c['private_commit_gib'] for c in controllers))
    base=supervision.StrictFamily
    class CompleteFamily(base):
        def record(self,row):
            charged={p['pid'] for p in row['job_members']}|{row['runner']['pid']}
            additional=[]
            for expected in controllers:
                current=legacy.process_info(expected['pid'])
                if current is None or not self.same_identity(expected,current):
                    raise RuntimeError('NATIVE_CALLER_IDENTITY_CHANGED')
                if current['pid'] not in charged:
                    row['host']['project_private_commit_gib']+=current['private_commit_gib']
                    row['host']['project_working_gib']+=current['working_gib'];additional.append(current)
            row['additional_shared_controllers']=additional
            return super().record(row)
    supervision.StrictFamily=CompleteFamily
    admission=stage_admission_payload(tool,root,manifest)
    admission_path=root/'stage-admission.json';write(admission_path,admission)
    cfg.update(stage_admission={'manifest_path':str(admission_path),'manifest_sha256':sha(admission_path)},
        additional_controller_identities=controllers)
    config_path=root/'config.json';write(config_path,cfg)
    old_argv=sys.argv
    sys.argv=[str(tool/'route/native_e2e_supervise.py'),'--out',str(root/'monitor'),'--python',sys.executable,
        '--worker',str(tool/'route/native_worker.py'),'--timeout','14400','--config',str(config_path)]
    try:
        with (root/'supervisor.stdout.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
            code=supervision.main()
    finally:sys.argv=old_argv
    return code

def main(argv=None,*,_supervise=None):
    args=parse_args(argv);tool=Path(__file__).resolve().parent
    try:
        if args.legacy_completion:
            raise PermissionError('LEGACY_COMPLETION_NOT_AUTOMATIC: use retained legacy_completion/convert.py only with separate explicit authorization for external completion')
        manifest=verified_payload(tool)
        import runpy
        ram_policy=runpy.run_path(str(tool/'core/bemarkdown/host_memory_policy.py'))['activate_host_policy'](tool)
        source=Path(args.source).resolve(strict=True);output=Path(args.output_root).resolve()
        sources=[source]+[Path(value).resolve(strict=True) for value in args.batch_source]
        if len(sources)>32 or len(set(sources))!=len(sources):raise ValueError('NATIVE_BATCH_INPUT_COUNT_OR_DUPLICATE')
        for item in sources:
         if not item.is_file() or item.suffix.lower() not in SUFFIXES:raise ValueError('NATIVE_ROUTE_UNSUPPORTED_SOURCE')
         if output==item.parent or output in item.parents:raise ValueError('NATIVE_OUTPUT_MUST_BE_SEPARATE_FROM_SOURCE')
        if not source.is_file() or source.suffix.lower() not in SUFFIXES:
            raise ValueError('NATIVE_ROUTE_UNSUPPORTED_SOURCE')
        if output==source.parent or output in source.parents:
            raise ValueError('NATIVE_OUTPUT_MUST_BE_SEPARATE_FROM_SOURCE')
        if args.screen_existing:
         if args.no_image_screening or args.batch_source:raise ValueError('SCREEN_EXISTING_REQUIRES_SINGLE_SCREENING_SOURCE')
         sys.path.insert(0,str(tool/'core'))
         from bemarkdown.image_screening_pipeline import screen_existing_conversion
         result=screen_existing_conversion(args.screen_existing,source,output,mcp_root=args.mcp_root)
         result.update(route_release=manifest['release_id'],host_memory_policy=ram_policy)
         print(json.dumps(result,ensure_ascii=False));return 0
        output.mkdir(parents=True,exist_ok=True);root=output/'.ocr-work';root.mkdir(exist_ok=False)
        cfg={'tool_root':str(tool),'run_root':str(root),'output_root':str(output),'input':str(source),
            'input_sha256':sha(source),'mcp_root':str(Path(args.mcp_root).resolve()),'caller_pid':args.caller_pid,
            'debug':args.debug,'repetition_guard':args.repetition_guard,'route_release':manifest['release_id'],'automatic_external_completion':False,
            'route_manifest_sha256':sha(tool/'ROUTE_MANIFEST.json')}
        execute=supervise if _supervise is None else _supervise
        if len(sources)>1:cfg.update(batch_sources=[{'source':str(item),'sha256':sha(item)} for item in sources],layout_batch_size=args.layout_batch_size)
        code=execute(tool,cfg,root,manifest)
        monitor=read(root/'monitor/result.json')
        if code or monitor.get('status')!='COMPLETED':
            result={'success':False,'error':'NATIVE_ROUTE_SUPERVISION_FAILED','monitor':str(root/'monitor/result.json')}
            print(json.dumps(result));return 1
        result=read(root/'result.json')
        if not args.no_image_screening:
         sys.path.insert(0,str(tool/'core'))
         if len(sources)>1:
          sys.path.insert(0,str(tool/'route'))
          from batch_contract import validate_batch_result
          validate_batch_result(result,cfg['batch_sources'],output)
         else:
          if not result.get('success') or result.get('source_sha256')!=cfg['input_sha256']:raise ValueError('NATIVE_SOURCE_RESULT_CONTRACT_FAILED')
          if not Path(result['package_path']).resolve().is_relative_to(output):raise ValueError('NATIVE_PACKAGE_RESULT_ESCAPED')
         from bemarkdown.image_screening_pipeline import finish_conversion
         result=finish_conversion(result,mcp_root=cfg['mcp_root'],work_root=root/'image-screening')
         write(root/'screened-result.json',result)
        if len(sources)>1:
         sys.path.insert(0,str(tool/'route'))
         from batch_contract import validate_batch_result
         validate_batch_result(result,cfg['batch_sources'],output)
         result.update(monitor=str(root/'monitor/result.json'),route_release=manifest['release_id'],host_memory_policy=ram_policy)
         print(json.dumps(result,ensure_ascii=False));return 0 if result.get('success') else 1
        if not result.get('success') or result.get('source_sha256')!=cfg['input_sha256']:
            raise ValueError('NATIVE_SOURCE_RESULT_CONTRACT_FAILED')
        package=Path(result['package_path']).resolve(strict=True)
        if not package.is_relative_to(output) or not (package/'document.md').is_file():
            raise ValueError('NATIVE_PACKAGE_RESULT_ESCAPED')
        result.update(monitor=str(root/'monitor/result.json'),route_release=manifest['release_id'],host_memory_policy=ram_policy)
        print(json.dumps(result,ensure_ascii=False));return 0
    except Exception as exc:
        print(json.dumps({'success':False,'error_type':type(exc).__name__,'error':str(exc)},ensure_ascii=False));return 1

if __name__=='__main__':
    import runpy
    gateway = Path(__file__).resolve().parents[1] / 'runtime_manager/native_gateway.py'
    raise SystemExit(runpy.run_path(str(gateway))['gateway'](Path(__file__).resolve().parent))
