"""Fixed registered worker for education conversion and image screening."""
from pathlib import Path
import argparse
import json
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manager-job-id', required=True)
    parser.add_argument('--manager-request', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    helpers = runpy.run_path(str(root / 'TOOLS/runtime_manager/bootstrap.py'))
    context = helpers['trusted_context'](root)
    if not context or context['job_id'] != args.manager_job_id:
        raise RuntimeError('WORKER_NOT_OWNED_BY_MANAGER')
    if Path(args.manager_request).resolve(strict=True) != (Path(context['home'])/'jobs'/args.manager_job_id/'request.json').resolve():
        raise ValueError('Unexpected managed request path')
    request = json.loads(Path(args.manager_request).read_text(encoding='utf-8'))
    if request['job_id'] != args.manager_job_id:
        raise ValueError('Manager request identity mismatch')
    payload = request['payload']
    server = runpy.run_path(str(Path(__file__).with_name('server.py')))
    service = server['Service'](root, request['workspace'], sys.executable, payload.get('input_roots', []))
    try:
        service._managed_job_id = args.manager_job_id
        if payload['operation'] == 'convert':
            if helpers['sha'](payload['source']) != payload['source_sha256']:
                raise ValueError('Source changed after submission')
            service.start(payload['source'], payload.get('material_type'), payload.get('book_title'))
            state = service.state(args.manager_job_id)
            if state['status'] != 'SUCCEEDED':
                raise RuntimeError(state.get('error', 'Conversion failed'))
        elif payload['operation'] == 'screen':
            run_root = Path(payload['run_root']).resolve(strict=True)
            manifest = Path(payload['manifest']).resolve(strict=True)
            if not run_root.is_relative_to(service.jobs.parent / 'as') or manifest.parent != run_root:
                raise ValueError('Screening paths escaped workspace')
            if helpers['sha'](manifest) != payload['manifest_sha256']:
                raise ValueError('Screening manifest changed')
            model = (root / 'MODELS/WeMM-Embedding-4B').resolve(strict=True)
            if model != Path(payload['model_path']).resolve(strict=True):
                raise ValueError('Unexpected screening model')
            server['atomic_json'](run_root/'manager.json',dict(job_id=args.manager_job_id))
            server['SCREENING_SERVICE']['compute'](service, payload['run_id'], run_root,
                manifest, model, server['conversion_lock'], payload['profile'])
            state = json.loads((run_root / 'state.json').read_text(encoding='utf-8'))
            if state['status'] != 'SUCCESS':
                raise RuntimeError(state.get('error', 'Screening failed'))
        else:
            raise ValueError('Unknown managed operation')
    finally:
        service.close()


if __name__ == '__main__':
    main()
