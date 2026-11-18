"""Managed conversion entry; the outer shared job owns all model subprocesses."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import secrets
import sys
import tempfile
import threading


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output-root','mcp-root'):p.add_argument('--'+key,required=True)
    p.add_argument('--batch-source',action='append',default=[])
    p.add_argument('--core-options',default='{}')
    p.add_argument('--caller-pid',type=int)
    p.add_argument('--retry-job-id')
    p.add_argument('--layout-batch-size',type=int,default=1)
    for key in ('debug','json','no-image-screening','legacy-completion','repetition-guard'):
        p.add_argument('--'+key,action='store_true')
    p.add_argument('--screen-existing')
    return p.parse_args(argv)


def parse_core_options(raw):
    options = json.loads(raw)
    allowed = {'formula_ocr', 'mtef_cache', 'mtef_cache_dir', 'replace',
               'continue_on_error', 'reuse_formula_model'}
    if not isinstance(options, dict) or options.keys() - allowed:
        raise ValueError('UNKNOWN_CORE_OPTIONS')
    for name in ('replace', 'continue_on_error', 'reuse_formula_model'):
        if name in options and type(options[name]) is not bool:
            raise ValueError('INVALID_CORE_OPTION:' + name)
    for name, choices in (('formula_ocr', ('auto', 'off')), ('mtef_cache', ('off', 'memory', 'persistent'))):
        if name in options and options[name] not in choices:
            raise ValueError('INVALID_CORE_OPTION:' + name)
    if options.get('mtef_cache_dir') is not None and not isinstance(options['mtef_cache_dir'], str):
        raise ValueError('INVALID_CORE_OPTION:mtef_cache_dir')
    return options


def main(argv=None):
    args=parse_args(argv);tool=Path(__file__).resolve().parent
    try:
        if args.legacy_completion or args.screen_existing or args.repetition_guard:
            raise ValueError('LEGACY_ROUTE_OPTIONS_NOT_SUPPORTED_BY_LOCAL_API')
        root=Path(args.mcp_root).resolve(strict=True)
        boot=runpy.run_path(str(root/'TOOLS/runtime_manager/bootstrap.py'))
        if not boot['trusted_context'](root):raise RuntimeError('SHARED_RESOURCE_MANAGER_REQUIRED')
        engine=runpy.run_path(str(tool/'engine.py'));engine['engine_root'](tool)
        manifest=json.loads((tool/'ROUTE_MANIFEST.json').read_text(encoding='utf-8'))
        for relative,expected in manifest['files'].items():
            target=(tool/relative).resolve()
            if not target.is_relative_to(tool) or boot['sha'](target)!=expected:
                raise ValueError('CONVERSION_PAYLOAD_CHANGED:'+relative)
        options = parse_core_options(args.core_options)
        continue_on_error=options.pop('continue_on_error',False)
        options.pop('reuse_formula_model',None)
        from bemarkdown.next_runtime import convert_document
        from bemarkdown.local_model_server import ProcessBackend,create_server
        from bemarkdown.pdf.local_model_api import ROLES,SCHEMA
        token_env='BEMARKDOWN_MODEL_API_TOKEN'
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        # An independently managed local service may be selected by an explicit
        # client configuration; otherwise start the bundled on-demand listener.
        config=os.environ.get('BEMARKDOWN_MODEL_API_CONFIG')
        if not config and (root/'.local/model-api.json').is_file():
            config=root/'.local/model-api.json'
        server=thread=None
        with tempfile.TemporaryDirectory(prefix='bemarkdown-client-') as temporary:
            if not config:
                os.environ[token_env]=secrets.token_hex(32)
                server=create_server(('127.0.0.1',0),ProcessBackend(tool/'config/local-model-server.json'),token=os.environ[token_env])
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                config=Path(temporary)/'client.json'
                config.write_text(json.dumps({'schema':SCHEMA,'roles':{role:{
                    'url':f'http://127.0.0.1:{server.server_port}/v1/{role}',
                    'token_env':token_env,'timeout_seconds':600} for role in ROLES}}),encoding='utf-8')
            results=[]
            try:
                for source in [args.source,*args.batch_source]:
                    try:
                        value=convert_document(source,args.output_root,api_config=config,debug=args.debug,**options).to_dict()
                    except Exception as exc:
                        if not continue_on_error:raise
                        value={'success':False,'source':source,'error':f'{type(exc).__name__}: {exc}'}
                    results.append(value)
            finally:
                if server:
                    server.shutdown();server.server_close();thread.join(timeout=10)
                    if thread.is_alive():raise RuntimeError('MODEL_LISTENER_EXIT_UNCONFIRMED')
        result=results[0] if len(results)==1 else {'success':all(v['success'] for v in results),'results':results}
        result['runtime_route']='LOCAL_MODEL_API_V1'
        print(json.dumps(result,ensure_ascii=False))
        return 0 if result['success'] else 1
    except Exception as exc:
        print(json.dumps({'success':False,'error_type':type(exc).__name__,'error':str(exc)},ensure_ascii=False))
        return 1


if __name__=='__main__':
    tool=Path(__file__).resolve().parent
    gateway=runpy.run_path(str(tool.parent/'runtime_manager/native_gateway.py'))['gateway']
    raise SystemExit(gateway(tool))
