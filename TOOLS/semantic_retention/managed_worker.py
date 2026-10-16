"""Private manager worker; never accepts an unverified process identity."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
TOOL = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL))
import manager_bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manager-job-id', required=True)
    parser.add_argument('--manager-request', required=True)
    args = parser.parse_args()
    mcp_root = TOOL.parents[1]
    context = manager_bridge.runtime(mcp_root).trusted_context(mcp_root)
    if not context or context['job_id'] != args.manager_job_id:
        raise RuntimeError('MANAGED_WORKER_CONTEXT_REQUIRED')
    path = Path(args.manager_request).resolve(strict=True)
    expected = (Path(context['home']) / 'jobs' / args.manager_job_id / 'request.json').resolve()
    if path != expected:
        raise ValueError('MANAGED_REQUEST_PATH_MISMATCH')
    request = json.loads(path.read_text(encoding='utf-8'))
    if request['job_id'] != args.manager_job_id or request['public_id'] != args.manager_job_id[:12]:
        raise ValueError('MANAGED_REQUEST_ID_MISMATCH')
    roots = request['input_roots']
    os.environ['SEMANTIC_RETENTION_ALLOWED_ROOTS'] = json.dumps(roots)
    import server
    source_request = server.validate_request(request['payload']['request'])
    if source_request['package_dir'] != str(Path(request['workspace']).resolve()):
        raise ValueError('MANAGED_REQUEST_WORKSPACE_MISMATCH')
    if manager_bridge.source_fingerprint(source_request) != request['payload']['source_fingerprint']:
        raise RuntimeError('SOURCE_CHANGED_AFTER_SUBMISSION')
    public_id = manager_bridge.materialize(server.JOBS, request, source_request)
    server.run_job(public_id)
    state = server.read(server.JOBS / public_id / 'state.json')
    return 0 if state.get('status') == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
