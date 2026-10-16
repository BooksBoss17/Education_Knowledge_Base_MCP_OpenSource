"""CLI frontend: attach to a durable job; the daemon owns conversion lifetime."""
from pathlib import Path
import json
import runpy
import sys
import time


def gateway(tool, argv=None):
    try:
        return _gateway(tool,argv)
    except Exception as exc:
        print(json.dumps(dict(success=False,error_type=type(exc).__name__,error=str(exc)),ensure_ascii=False))
        return 1


def _gateway(tool, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    module = runpy.run_path(str(Path(tool) / 'convert.py'), run_name='_native_parser')
    args = module['parse_args'](argv)
    root = Path(args.mcp_root).resolve(strict=True)
    helpers = runpy.run_path(str(root / 'TOOLS/runtime_manager/bootstrap.py'))
    if helpers['trusted_context'](root):
        return module['main'](argv)
    source = Path(args.source).resolve(strict=True)
    extra_sources = [Path(value).resolve(strict=True) for value in getattr(args, 'batch_source', [])]
    if len(extra_sources) > 31 or len({source, *extra_sources}) != len(extra_sources) + 1:
        raise ValueError('Native batch requires distinct inputs, at most 32')
    additional = [dict(source=str(path), source_sha256=helpers['sha'](path)) for path in extra_sources]
    existing = Path(args.screen_existing).resolve(strict=True) if getattr(args, 'screen_existing', None) else None
    if existing and extra_sources:raise ValueError('Existing package recovery accepts one source')
    package_hash = runpy.run_path(str(root/'TOOLS/runtime_manager/package_binding.py'))['fingerprint'](existing) if existing else None
    output = Path(args.output_root).resolve()
    workspace = output.parent
    workspace.mkdir(parents=True, exist_ok=True)
    roots = list({path.parent for path in [source, *extra_sources]})
    if existing:roots.append(existing)
    client = helpers['client'](root, sys.executable, workspace, roots)
    result = client.submit('native-convert', workspace,
        dict(source=str(source), source_sha256=helpers['sha'](source), output_root=str(output),
             debug=args.debug, legacy_completion=args.legacy_completion,
             additional_sources=additional, layout_batch_size=getattr(args, 'layout_batch_size', 1),
             repetition_guard=bool(getattr(args, 'repetition_guard', False)),
             image_screening=not bool(getattr(args, 'no_image_screening', False)),
             screen_existing=str(existing) if existing else None,existing_package_fingerprint=package_hash),
        config_fingerprint=helpers['sha'](Path(tool) / 'ROUTE_MANIFEST.json'), run_timeout=24*3600,
        retry_of=args.retry_job_id)
    # Logs go to stderr so --json remains a single result object.
    print('managed_job_id=' + result['job_id'], file=sys.stderr, flush=True)
    while result['state'] in {'QUEUED','WAITING_RESOURCES','STARTING','RUNNING','CANCEL_REQUESTED'}:
        time.sleep(.5)
        result = client.status(result['job_id'])
    response = Path(result['job_folder']) / 'native-result.json'
    try:
        converted = json.loads(response.read_text(encoding='utf-8'))
        if not isinstance(converted, dict):
            raise ValueError('Expected conversion object')
    except (OSError, ValueError):
        converted = dict(success=False, error='INCOMPLETE_MANAGED_OUTPUT')
    if result['state'] != 'SUCCEEDED':
        converted['success'] = False
        converted['runtime_manager'] = result
    converted['managed_job_id'] = result['job_id']
    print(json.dumps(converted, ensure_ascii=False))
    return 0 if result['state'] == 'SUCCEEDED' and converted.get('success') else 1
