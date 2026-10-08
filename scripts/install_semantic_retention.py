"""Opt-in public classifier installation; no conversion-route activation."""
import argparse,hashlib,json,os,shutil,subprocess,sys,urllib.parse,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from downloads import download,safe_path,valid,sha256
from download_retention_model import install as install_parts

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.pending');temp.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(temp,p)

def install_models(cache,local_cache=None,verify_only=False):
    plan=read(ROOT/'scripts/model-downloads.json')['models']
    weights=read(ROOT/'scripts/retention-model-weights.json')
    by_name={x['path']:x for x in weights['files']}
    for m in plan:
        if m['model_id'] not in ('bemarkdown-retention-b200','qwen35-9b-semantic-reviewer'):continue
        manifest=safe_path(ROOT/'MODELS',m['manifest_path'])
        if sha256(manifest)!=m['manifest_sha256']:raise ValueError('Model manifest drift')
        for item in m['files']:
            target=safe_path(ROOT/'MODELS'/m['directory'],item['path'])
            if valid(target,item['sha256'],item['bytes']):continue
            if verify_only:raise ValueError('Missing or mismatched model file: '+item['path'])
            donor=safe_path(Path(local_cache)/m['directory'],item['path']) if local_cache else None
            if donor and valid(donor,item['sha256'],item['bytes']):
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(donor,target);continue
            if item.get('url'):
                cached=download(item['url'],item['sha256'],item['bytes'],cache,workers=4)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(cached,target)
            elif m['model_id']=='bemarkdown-retention-b200' and item['path'] in by_name:
                row=by_name[item['path']]
                if (row['sha256'],row['bytes'])!=(item['sha256'],item['bytes']):raise ValueError('Release weight identity mismatch')
                install_parts({'files':[row]},ROOT/'MODELS'/m['directory'])
            else:raise ValueError('Required tracked metadata is absent or changed: '+str(target))
    return {'status':'MODELS_VERIFIED','automatic_activation':False}

def extract_native(archive,target):
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():continue
            dest=safe_path(target,info.filename)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,dest.open('wb') as stream:shutil.copyfileobj(source,stream,1024*1024)

def query(python,packages):
    code='import sys,json,importlib.metadata as m; print(json.dumps({"version":sys.version.split()[0],"base_prefix":sys.base_prefix,"prefix":sys.prefix,"versions":{x:m.version(x) for x in '+repr(list(packages))+'}}))'
    return json.loads(subprocess.check_output([str(python),'-I','-X','utf8','-c',code],text=True,encoding='utf-8'))

def register_runtime(inference_python,source_python,llama_dir,cache,accept_external=False):
    lock=read(ROOT/'TOOLS/semantic_retention/PUBLIC_RUNTIME_LOCK.json')
    ip=Path(inference_python).resolve(strict=True);sp=Path(source_python).resolve(strict=True)
    inference=query(ip,lock['inference_versions']);source=query(sp,lock['source_versions'])
    if inference['version']!='3.12.10' or source['version']!='3.10.11':raise ValueError('Use the documented pinned Python versions')
    if inference['versions']!=lock['inference_versions'] or source['versions']!=lock['source_versions']:raise ValueError('Dependency versions differ from the frozen stack')
    if not (Path(inference['prefix'])/'Lib/site-packages').is_dir():raise ValueError('Windows inference environment required')
    if llama_dir:native=Path(llama_dir).resolve(strict=True)
    else:
        if not accept_external:raise ValueError('Read external runtime licenses and pass --accept-external-licenses before download')
        native=Path(os.environ['LOCALAPPDATA'])/'BeMarkdownSemanticRetention/native/b11146-cuda124'
        native.mkdir(parents=True,exist_ok=True)
        for item in lock['native_archives']:
            archive=download(item['url'],item['sha256'],item['bytes'],cache,workers=4)
            extract_native(archive,native)
    for item in lock['llama_files']:
        if sha256(safe_path(native,item['file']))!=item['sha256']:raise ValueError('Native runtime hash mismatch: '+item['file'])
    base=Path(inference['base_prefix'])/'python.exe'
    identity={'python_sha256':sha256(base),'source_python_sha256':sha256(sp),
        'inference_versions':lock['inference_versions'],'source_versions':lock['source_versions'],'llama_files':lock['llama_files']}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    receipt=Path(os.environ['LOCALAPPDATA'])/'BeMarkdownSemanticRetention/runtimes'/key/'runtime.json'
    value={'status':'VERIFIED_HOST_RUNTIME','lock':identity,'inference_python':str(ip),'source_python':str(sp),
        'base_python':str(base),'llama_home':str(native),'no_developer_runtime_dependency':True,
        'module_check':{'inference':inference,'source':source},'distribution_scope':'LOCAL_HOST_NOT_GIT',
        'public_runtime_version_and_hash_checks':True,'fresh_model_quality_evaluation_performed':False}
    # Never replace an existing registered runtime with a different installation.
    if receipt.exists():
        existing=read(receipt)
        if existing['lock']!=identity:raise ValueError('Runtime identity collision')
        for role in ('inference_python','source_python','base_python'):
            if not Path(existing[role]).is_file():raise ValueError('Existing runtime is unavailable; repair it explicitly')
    else:write(receipt,value)
    write(ROOT/'.local/semantic-retention-runtime.json',{'runtime_path':str(receipt)})
    client={'mcpServers':{'semantic-retention':{'command':str(sp),'args':['-I','-X','utf8',str(ROOT/'TOOLS/semantic_retention/server.py')],
        'env':{'SEMANTIC_RETENTION_ALLOWED_ROOTS':'[]'}}}}
    write(ROOT/'.local/semantic-retention-client.json',client)
    return {'status':'RUNTIME_REGISTERED','receipt':str(receipt),'client':str(ROOT/'.local/semantic-retention-client.json'),
            'note':'Set allowed input roots in the generated client config before starting jobs; no existing MCP config was changed.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify-only',action='store_true');ap.add_argument('--model-cache',type=Path)
    ap.add_argument('--download-cache',type=Path);ap.add_argument('--register-runtime',action='store_true')
    ap.add_argument('--runtime-only',action='store_true');ap.add_argument('--inference-python');ap.add_argument('--source-python')
    ap.add_argument('--llama-dir');ap.add_argument('--accept-external-licenses',action='store_true')
    a=ap.parse_args();cache=a.download_cache or Path(os.environ['LOCALAPPDATA'])/'EducationMCP/downloads'
    if not a.runtime_only:print(json.dumps(install_models(cache,a.model_cache,a.verify_only)))
    if a.register_runtime:
        if not a.inference_python or not a.source_python:ap.error('Provide --inference-python and --source-python; existing environments are validated, never modified')
        print(json.dumps(register_runtime(a.inference_python,a.source_python,a.llama_dir,cache,a.accept_external_licenses)))

if __name__=='__main__':main()
