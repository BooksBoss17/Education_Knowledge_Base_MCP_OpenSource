"""Model-free bridge to the installation's persistent run manager."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.digest()


def write_json(path, value):
    """Publish progress atomically so concurrent MCP polls see complete JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='state-', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(.02)
    finally:
        Path(temporary).unlink(missing_ok=True)


def runtime(mcp_root):
    path = Path(mcp_root) / 'TOOLS/runtime_manager/bootstrap.py'
    spec = importlib.util.spec_from_file_location('ekb_managed_entrypoints', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allowed_roots():
    roots = json.loads(os.environ.get('SEMANTIC_RETENTION_ALLOWED_ROOTS', '[]'))
    if not isinstance(roots, list) or not roots:
        raise ValueError('Set SEMANTIC_RETENTION_ALLOWED_ROOTS JSON allowed absolute directories')
    return [str(Path(path).resolve(strict=True)) for path in roots]


def client(mcp_root, request):
    return runtime(mcp_root).client(mcp_root, sys.executable,
                                   request['package_dir'], input_roots=allowed_roots())


def source_fingerprint(request):
    """Bind deduplication to bytes, including changed images and source PDF."""
    package = Path(request['package_dir'])
    digest = hashlib.sha256()
    paths = sorted(path for path in package.rglob('*') if path.is_file())
    for path in paths:
        if not path.resolve(strict=True).is_relative_to(package):
            raise ValueError('SOURCE_PACKAGE_LINK_ESCAPES_ROOT')
        digest.update(path.relative_to(package).as_posix().encode('utf-8') + b'\0')
        digest.update(file_digest(path))
    digest.update(file_digest(request['source_pdf']))
    return digest.hexdigest()


def config_fingerprint(tool):
    digest = hashlib.sha256()
    for name in ('server.py', 'manager_bridge.py', 'managed_worker.py', 'bootstrap.py',
                 'worker.py', 'source_prepare.py', 'image_purpose.py', 'inline_entry.py', 'SCHEME.json', 'dist/engine.zip'):
        path = Path(tool) / name
        digest.update(name.encode('utf-8') + b'\0')
        digest.update(file_digest(path))
    return digest.hexdigest()


def materialize(jobs, snapshot, request):
    """Idempotent aliases survive frontend disconnects and worker races."""
    manager_id = snapshot['job_id']
    if not re.fullmatch(r'[0-9a-f]{32}', manager_id):
        raise ValueError('INVALID_MANAGER_JOB_ID')
    public_id = manager_id[:12]
    job = Path(jobs) / public_id
    job.mkdir(parents=True, exist_ok=True)
    documents = {
        'request.json': request,
        'driver.json': {'manager_job_id': manager_id, 'public_id': public_id,
                        'workspace': request['package_dir'], 'managed': True},
        'state.json': {'status': 'QUEUED', 'job_id': public_id},
    }
    for name, value in documents.items():
        descriptor, temporary = tempfile.mkstemp(dir=job, prefix='alias-', suffix='.tmp')
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
            # NTFS hard-link creation publishes the complete file exclusively.
            os.link(temporary, job / name)
        except FileExistsError:
            if name == 'driver.json':
                old = json.loads((job / name).read_text(encoding='utf-8'))
                if old.get('manager_job_id') != manager_id:
                    raise RuntimeError('PUBLIC_JOB_ID_COLLISION')
        finally:
            Path(temporary).unlink(missing_ok=True)
    return public_id


def submit(mcp_root, tool, jobs, request, retry_of=None):
    result = client(mcp_root, request).submit(
        'semantic-retention', request['package_dir'],
        {'request': request, 'source_fingerprint': source_fingerprint(request)},
        requirement={'extra_private_bytes': int(5.9 * 1024**3) + 128 * 1024**2,
                     'extra_physical_bytes': int(4.9 * 1024**3) + 128 * 1024**2,
                     'extra_gpu_mib': 5.6 * 1024},
        config_fingerprint=config_fingerprint(tool), wait_timeout=900, retry_of=retry_of)
    public_id = materialize(jobs, result, request)
    return {'job_id': public_id, 'status': result.get('state', result.get('status', 'QUEUED')),
            'status_tool': 'semantic_retention_status', 'deduplicated': result.get('deduplicated', False),
            'manager_job_id': result['job_id']}


def managed_status(mcp_root, job, state):
    alias = job / 'driver.json'
    if not alias.is_file():
        return state
    metadata = json.loads(alias.read_text(encoding='utf-8'))
    if not metadata.get('managed'):
        return state  # Preserve historical jobs and their original receipts.
    snapshot = runtime(mcp_root).existing_client(mcp_root).status(metadata['manager_job_id'])
    manager_state = snapshot.get('state', snapshot.get('status'))
    # The inference summary remains the result contract. Manager state is
    # authoritative for waits, cancellation and failure after client exit.
    if state.get('status') != 'COMPLETE' or manager_state != 'SUCCEEDED':
        if manager_state not in ('RUNNING', 'STARTING') or state.get('status') in ('QUEUED', 'FAILED'):
            state = {**state, 'status': 'FAILED' if manager_state == 'SUCCEEDED' else manager_state}
    return {**state, 'manager': snapshot}


def cancel(mcp_root, job):
    metadata = json.loads((job / 'driver.json').read_text(encoding='utf-8'))
    if not metadata.get('managed'):
        raise ValueError('LEGACY_JOB_NOT_MANAGED')
    result = runtime(mcp_root).existing_client(mcp_root).cancel(metadata['manager_job_id'], reason='semantic_retention_cancel')
    return {'job_id': job.name, 'status': result.get('state', result.get('status')), 'manager': result}
