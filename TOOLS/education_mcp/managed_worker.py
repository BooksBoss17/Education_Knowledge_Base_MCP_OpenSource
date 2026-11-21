"""Fixed registered worker for education conversion."""
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
            if 'derivation' in payload:
                server['IMAGE_REVIEW']['verify_derivation'](service, payload)
                service._derivation = payload['derivation']
            service.start(payload['source'], payload.get('material_type'), payload.get('book_title'))
            state = service.state(args.manager_job_id)
            if state['status'] != 'SUCCEEDED':
                raise RuntimeError(state.get('error', 'Conversion failed'))
        else:
            raise ValueError('Unknown managed operation')
    finally:
        service.close()


if __name__ == '__main__':
    main()
