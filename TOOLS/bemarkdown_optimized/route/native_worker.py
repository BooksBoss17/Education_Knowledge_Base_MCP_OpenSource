"""Accepted native core API worker; never loads Flash, completion or DS adapters."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import sys

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def main(argv=None,*,_convert=None):
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args(argv)
    cfg=read(args.config);tool=Path(cfg['tool_root']).resolve();root=Path(cfg['run_root']).resolve()
    if sha(tool/'ROUTE_MANIFEST.json')!=cfg['route_manifest_sha256']:raise ValueError('NATIVE_ROUTE_MANIFEST_CHANGED')
    manifest=read(tool/'ROUTE_MANIFEST.json')
    for relative,digest in manifest['files'].items():
        path=(tool/relative).resolve()
        if not path.is_relative_to(tool) or sha(path)!=digest:raise ValueError('NATIVE_ROUTE_PAYLOAD_CHANGED:'+relative)
    source=Path(cfg['input']).resolve(strict=True)
    if sha(source)!=cfg['input_sha256']:raise ValueError('NATIVE_SOURCE_CHANGED')
    if cfg.get('automatic_external_completion') is not False:raise ValueError('NATIVE_AUTOMATIC_COMPLETION_FORBIDDEN')
    os.environ.update(BEMARKDOWN_RESOURCE_PROFILE='6gb',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONUTF8='1',PYTHONIOENCODING='utf-8',TOKENIZERS_PARALLELISM='false')
    os.environ['BEMARKDOWN_QWEN_REPEAT_GUARD']='1' if cfg.get('repetition_guard') else '0'
    os.environ['BEMARKDOWN_DIAGNOSTICS_ROOT']=str(root/'persistent-diagnostics')
    # Environment/bootstrap remains the existing managed runtime. Only engine
    # Python code is overlaid; its metadata does not claim the base wheel changed.
    sys.path.insert(0,str(tool/'core'))
    from bemarkdown.production import convert_document
    bindings=cfg.get('batch_sources')
    if bindings:
     if not 2<=len(bindings)<=32 or len({item['source'] for item in bindings})!=len(bindings):raise ValueError('NATIVE_BATCH_BINDINGS_INVALID')
     if bindings[0]!={'source':str(source),'sha256':cfg['input_sha256']}:raise ValueError('NATIVE_BATCH_PRIMARY_MISMATCH')
     for item in bindings:
      if sha(item['source'])!=item['sha256']:raise ValueError('NATIVE_BATCH_SOURCE_CHANGED')
     from bemarkdown.production import build_document_id
     if len({build_document_id(item['source']) for item in bindings})!=len(bindings):raise ValueError('NATIVE_BATCH_OUTPUT_IDENTITY_COLLISION')
     from bemarkdown.batch import convert_documents
     value=convert_documents([item['source'] for item in bindings],Path(cfg['output_root']),
      models_root=Path(cfg['mcp_root'])/'MODELS',mcp_root=cfg['mcp_root'],debug=cfg['debug'],
      layout_batch_size=cfg.get('layout_batch_size',1),continue_on_error=True)
     for item in bindings:
      if sha(item['source'])!=item['sha256']:raise ValueError('NATIVE_BATCH_SOURCE_CHANGED_DURING_CONVERSION')
     value.update(automatic_external_completion=False,native_route=True,route_release=cfg['route_release'],source_bindings=bindings)
    else:
     execute=_convert or convert_document
     result=execute(source,Path(cfg['output_root']),models_root=Path(cfg['mcp_root'])/'MODELS',
      mcp_root=cfg['mcp_root'],debug=cfg['debug'])
     value=result.to_dict()
     if value.get('success') is False:raise ValueError('NATIVE_CONVERSION_REPORTED_FAILURE')
     value.update(success=True,automatic_external_completion=False,
      native_route=True,route_release=cfg['route_release'],source_sha256=cfg['input_sha256'])
     if sha(source)!=cfg['input_sha256']:raise ValueError('NATIVE_SOURCE_CHANGED_DURING_CONVERSION')
    (root/'result.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return 0

if __name__=='__main__':raise SystemExit(main())
