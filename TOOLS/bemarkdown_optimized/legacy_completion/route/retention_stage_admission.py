"""Install task-local load gates into an extracted retention engine only."""
import hashlib,json,os
from pathlib import Path

def install_retention_stage_admission(tools, evidence_dir):
    if not os.environ.get('BEMARKDOWN_STAGE_ADMISSION_CONTEXT'):
        return None
    from bemarkdown import stage_admission
    # Retention's private runtime deliberately lacks document-conversion packages.
    # Load this stdlib-only gate by its frozen file, without bemarkdown.__init__.
    gate_path=Path(stage_admission.__file__).resolve()
    gate_sha=hashlib.sha256(gate_path.read_bytes()).hexdigest()
    tools=Path(tools).resolve();evidence_dir=Path(evidence_dir).resolve()
    evidence_dir.mkdir(parents=True,exist_ok=False)
    entries=[]
    for filename,marker,indent,stage in [
        ('run_gguf_partition_predictions.py','            server = subprocess.Popen(',12,'retention-gguf-classify'),
        ('cold_semantic_review.py','        self.server=subprocess.Popen(',8,'retention-gguf-review'),
    ]:
        path=tools/filename;before=path.read_bytes();source=before.decode('utf-8')
        if source.count(marker)!=1:raise RuntimeError('RETENTION_STAGE_GATE_BOUNDARY_CHANGED:'+filename)
        (evidence_dir/(filename+'.before')).write_bytes(before)
        prefix=' '*indent
        source=source.replace(marker,prefix+'import runpy as _bm_gate_runpy, hashlib as _bm_gate_hashlib\n'+
            prefix+'from pathlib import Path as _bm_gate_Path\n'+
            prefix+'if _bm_gate_hashlib.sha256(_bm_gate_Path('+repr(str(gate_path))+').read_bytes()).hexdigest() != '+repr(gate_sha)+':\n'+
            prefix+'    raise RuntimeError("RETENTION_STAGE_GATE_SOURCE_CHANGED")\n'+
            prefix+'_bm_gate_runpy.run_path('+repr(str(gate_path))+')["require_stage_admission"]('+repr(stage)+')\n'+marker)
        compile(source,str(path),'exec')
        after=source.encode('utf-8');path.write_bytes(after)
        entries.append({'path':str(path),'before_sha256':hashlib.sha256(before).hexdigest(),
            'after_sha256':hashlib.sha256(after).hexdigest(),'stage':stage,
            'before_copy':str(evidence_dir/(filename+'.before'))})
    result={'status':'TASK_PRIVATE_ENGINE_LOAD_GATES_INSTALLED','entries':entries,
        'import_root':str(Path(stage_admission.__file__).resolve().parents[1]),
        'standalone_gate_path':str(gate_path),'standalone_gate_sha256':gate_sha,
        'shared_engine_archive_modified':False}
    (evidence_dir/'installation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
