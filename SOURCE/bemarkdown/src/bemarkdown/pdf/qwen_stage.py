"""Registry-backed subprocess boundary; no Torch imports in Paddle host."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from ..model_registry import ModelRegistry

MANAGED_RUNTIME_NAME='qwen-ocr-py312-torch214-tf517-v1'

def qwen_python(explicit=None):
    configured=explicit or os.environ.get('BEMARKDOWN_QWEN_PYTHON')
    if configured:
        result=Path(configured).resolve()
        if not result.is_file():raise RuntimeError('QWEN_RUNTIME_PYTHON_MISSING')
        return result
    root=Path(os.environ.get('LOCALAPPDATA',Path.home()/'AppData/Local'))/'BeMarkdown'/'runtimes'/MANAGED_RUNTIME_NAME
    candidate=root/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not candidate.is_file():raise RuntimeError('QWEN_MANAGED_RUNTIME_NOT_INSTALLED')
    return candidate

class QwenStageRunner:
    @classmethod
    def from_registry(cls,*,python=None,crop_root,output_root,model_id='xiaomi-ocr-0',models_root=None,config_path=None,mcp_root=None,**options):
        resolved=ModelRegistry(models_root=models_root,config_path=config_path,mcp_root=mcp_root).resolve(model_id,deep=True)
        return cls(resolved,crop_root,output_root,python=python,**options)

    def __init__(self,resolved,crop_root,output_root,*,python=None,vram_limit_mib=2816,max_new_tokens=2048,**ignored):
        self.resolved=resolved;self.crop_root=Path(crop_root).resolve();self.output_root=Path(output_root).resolve()
        self.python=qwen_python(python);self.vram_limit_mib=vram_limit_mib;self.max_new_tokens=max_new_tokens
        self.stage_metrics=[];self.calls=0

    def prepare_dependencies(self):
        return None  # Loading happens once per stage, only when requests exist.

    def close(self):
        return None  # All child processes have exited before returning results.

    def __call__(self,provider_id,requests):
        if not requests:return []
        self.calls+=1;folder=self.output_root/self.resolved.model_id/f'stage-{self.calls:04d}'
        folder.mkdir(parents=True,exist_ok=True);rows=[]
        for request in requests:
            path=Path(request.crop_ref).resolve();path.relative_to(self.crop_root)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=request.crop_sha256:raise RuntimeError('QWEN_REQUEST_SOURCE_MISMATCH')
            rows.append({'sample_id':request.region_id,'path':str(path),'crop_sha256':request.crop_sha256})
        manifest=folder/'inputs.json';output=folder/'outputs.json'
        value={'model_root':str(self.resolved.model_root),'model_fingerprint':self.resolved.manifest['model_fingerprint'],
            'crop_root':str(self.crop_root),'rows':rows,'vram_limit_mib':self.vram_limit_mib,'max_new_tokens':self.max_new_tokens}
        manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        worker=Path(__file__).parent/'workers/qwen_ocr.py';env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        started=time.perf_counter()
        with (folder/'stdout.log').open('w',encoding='utf8') as stdout,(folder/'stderr.log').open('w',encoding='utf8') as stderr:
            process=subprocess.run([str(self.python),str(worker),'--manifest',str(manifest),'--output',str(output)],env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if process.returncode or not output.exists():raise RuntimeError(f'QWEN_WORKER_FAILED:{self.resolved.model_id}:exit={process.returncode}:diagnostics={folder}')
        values=json.loads(output.read_text(encoding='utf8'))
        if [r['sample_id'] for r in values]!=[r['sample_id'] for r in rows]:raise RuntimeError('QWEN_WORKER_CARDINALITY_ID_MISMATCH')
        metric=json.loads(output.with_suffix('.runtime.json').read_text(encoding='utf8'))
        metric.update(provider_id=provider_id,process_wall_seconds=time.perf_counter()-started);self.stage_metrics.append(metric)
        return values
