"""Create an isolated job tree using only the installed tool, models and host runtime."""
import hashlib,json,os,shutil,zipfile
from pathlib import Path
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(json.dumps(value,ensure_ascii=False,indent=2).encode('utf-8'))
def runtime_config(tool):
    spec=read(Path(tool)/'SCHEME.json')
    path=Path(os.environ['LOCALAPPDATA'])/'BeMarkdownSemanticRetention/runtimes'/spec['runtime_id']/'runtime.json'
    pointer=Path(tool).parents[1]/'.local/semantic-retention-runtime.json'
    if pointer.is_file():path=Path(read(pointer)['runtime_path'])
    if os.environ.get('SEMANTIC_RETENTION_RUNTIME_CONFIG'):path=Path(os.environ['SEMANTIC_RETENTION_RUNTIME_CONFIG'])
    value=read(path)
    assert value['status']=='VERIFIED_HOST_RUNTIME' and value['no_developer_runtime_dependency']
    return value
def initialize_job(tool,mcp_root,job,request):
    tool,mcp_root,job=Path(tool).resolve(),Path(mcp_root).resolve(),Path(job).resolve()
    spec=read(tool/'SCHEME.json');runtime=runtime_config(tool)
    job.mkdir(parents=True,exist_ok=False)
    bundle=tool/'dist/engine.zip';manifest=read(tool/'dist/engine-manifest.json')
    assert sha(bundle)==spec['engine_zip_sha256']
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            target=(job/name).resolve();assert target.is_relative_to(job)
            data=archive.read(name);assert hashlib.sha256(data).hexdigest()==manifest['files'][name]
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    project=job/'projects/bemarkdown-retention-training';tools=project/'tools'
    for name in ('source_prepare.py','worker.py'):
        target=tools/('portable_job.py' if name=='worker.py' else name);shutil.copy2(tool/name,target)
    # Tiny venv shim: all third-party packages come from the verified host installation.
    env=job/'tmp/rt-train/venvs/q35-native';scripts=env/'Scripts';site=env/'Lib/site-packages'
    scripts.mkdir(parents=True);site.mkdir(parents=True)
    shutil.copy2(runtime['inference_python'],scripts/'python.exe')
    home=Path(runtime['base_python']).parent
    (env/'pyvenv.cfg').write_bytes(('home = '+str(home)+'\ninclude-system-site-packages = false\nversion = 3.12.10\n').encode('utf-8'))
    host_site=Path(runtime['inference_python']).parents[1]/'Lib/site-packages'
    (site/'verified_host.pth').write_bytes(('import site; site.addsitedir('+repr(str(host_site))+')\n').encode('utf-8'))
    native=job/'tmp/llama-retention';native.mkdir(parents=True)
    for item in runtime['lock']['llama_files']:
        source=Path(runtime['llama_home'])/item['file'];assert sha(source)==item['sha256']
        target=native/item['file']
        try:os.link(source,target)
        except OSError:shutil.copy2(source,target)
    models={}
    for role,definition in spec['models'].items():
        root=(mcp_root/'MODELS'/definition['directory']).resolve();assert root.is_relative_to(mcp_root/'MODELS')
        declared=read(root/'MODEL_MANIFEST.json')
        assert declared['model_fingerprint']==definition['model_fingerprint']
        files={x['path']:x for x in declared['files']}
        for relative in (definition['weights'],definition['projector']):
            path=(root/relative).resolve();assert path.is_relative_to(root) and sha(path)==files[relative]['sha256']
        models[role]=(root,files,definition)
    base,files,base_spec=models['base']
    processor=base/base_spec['processor']
    for relative,item in files.items():
        if relative.startswith(base_spec['processor']+'/'):assert sha(base/relative)==item['sha256']
    reports=project/'reports';reports.mkdir(exist_ok=True)
    write(reports/'native-environment-bootstrap.json',{'venv_python':str(scripts/'python.exe')})
    write(reports/'base-download.json',{'status':'DOWNLOADED_VERIFIED','local_path':str(processor),'revision':spec['base_revision']})
    write(project/'configs/train-plan.json',{'base_model':{'revision':spec['base_revision']}})
    export_files=[{'file':Path(relative).name,'path':str(base/relative),'sha256':files[relative]['sha256']} for relative in (base_spec['weights'],base_spec['projector'])]
    write(reports/'gguf-candidate-export.json',{'status':'PAIRED_GGUF_CANDIDATE_EXPORTED_NOT_RUNTIME_OR_QUALITY_APPROVED',
         'converter_commit':spec['converter_commit'],'files':export_files,'runtime_files':runtime['lock']['llama_files']})
    review,review_files,review_spec=models['reviewer']
    write(reports/'zero-delete-99-20261007/reviewer9b-assets.json',{'status':'FROZEN_SHARED_MODEL_ASSETS_VERIFIED',
         'files':[{'file':Path(relative).name,'path':str(review/relative),'sha256':review_files[relative]['sha256']}
                  for relative in (review_spec['weights'],review_spec['projector'])]})
    write(job/'job-request.json',{**request,'tool_root':str(tool),'mcp_root':str(mcp_root),'runtime':runtime,
         'scheme_sha256':sha(tool/'SCHEME.json'),'source_files_are_read_only':True})
    return {'python':str(scripts/'python.exe'),'supervisor':str(tools/'run_native_model_stage.py'),
            'worker':str(tools/'portable_job.py'),'job_root':str(job)}
