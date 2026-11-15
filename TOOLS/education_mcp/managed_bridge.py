"""Persistent execution integration; frontend shutdown never owns worker lifetime."""
from pathlib import Path
import json
import runpy
import time


def bootstrap(service):
    return runpy.run_path(str(service.mcp_root / 'TOOLS/runtime_manager/bootstrap.py'))


def manager(service):
    return bootstrap(service)['client'](service.mcp_root, service.python, service.workspace, service.allowed)


def submit_conversion(service, source, material_type, book_title, source_sha256, retry_job_id=None):
    helpers = bootstrap(service)
    result = manager(service).submit('education-convert', service.workspace,
        dict(operation='convert', source=str(source), source_sha256=source_sha256,
             material_type=material_type, book_title=book_title,
             input_roots=[str(p) for p in service.allowed]),
        config_fingerprint=helpers['installation_fingerprint'](service.mcp_root), run_timeout=24*3600,
        retry_of=retry_job_id)
    return status(service, result['job_id'])


def status(service, job_id):
    path = service.state_path(job_id)
    if path.is_file():
        legacy = json.loads(path.read_text(encoding='utf-8'))
        if legacy.get('runtime_management') != 'PERSISTENT_MANAGER':
            legacy['runtime_management'] = 'LEGACY_UNMANAGED'
            if legacy.get('status') in {'QUEUED','RUNNING'}:
                legacy['observation'] = 'Legacy process identity must be checked before retry; no automatic restart.'
            return legacy
    try:
        observed = bootstrap(service)['existing_client'](service.mcp_root).status(job_id)
    except Exception as exc:
        if path.is_file() and 'UNKNOWN_JOB' in str(exc):
            legacy = json.loads(path.read_text(encoding='utf-8'))
            legacy['runtime_management'] = 'LEGACY_UNMANAGED'
            if legacy.get('status') in {'QUEUED', 'RUNNING'}:
                legacy['observation'] = 'Legacy process identity must be checked before retry; no automatic restart.'
            return legacy
        raise
    if Path(observed['workspace']).resolve() != service.workspace:
        raise ValueError('Managed job belongs to a different workspace')
    state = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else dict(
        job_id=job_id, source=observed['payload']['source'],
        source_sha256=observed['payload']['source_sha256'])
    actual = observed['state']
    if actual not in {'SUCCEEDED'} or state.get('status') != 'SUCCEEDED':
        state['status'] = actual
    state['runtime_manager'] = observed
    return state


def call(service, action='list', job_id=None, reason=None):
    client = bootstrap(service)['existing_client'](service.mcp_root)
    if action == 'list':
        return client.list(limit=20)
    if not job_id:
        raise ValueError('job_id required')
    if action == 'status':
        return client.status(job_id)
    if action == 'adopt':
        return client.adopt(job_id, 'education-mcp:' + str(service.workspace))
    if action == 'cancel':
        return client.cancel(job_id, reason or 'Explicit user cancellation', wait_seconds=5)
    raise ValueError('Unknown runtime action')


def submit_screen(service, run_id, run_root, manifest, model_path, profile):
    helpers = bootstrap(service)
    result = manager(service).submit('education-screen', service.workspace,
        dict(operation='screen', run_id=run_id, run_root=str(run_root), manifest=str(manifest),
             manifest_sha256=helpers['sha'](manifest), model_path=str(model_path), profile=profile),
        scope_key=run_id, config_fingerprint=helpers['installation_fingerprint'](service.mcp_root),
        run_timeout=7500)
    runpy.run_path(str(Path(__file__).with_name('atomic_files.py')))['atomic_json'](
        run_root / 'manager.json', dict(job_id=result['job_id']))
    return result


def screen_status(service, run_root, run_id=None):
    link = run_root / 'manager.json'
    if not link.is_file():
        if not run_id:
            return None
        client=bootstrap(service)['existing_client'](service.mcp_root)
        match=client.find_scope('education-screen',service.workspace,run_id)
        if not match:
            return None
        runpy.run_path(str(Path(__file__).with_name('atomic_files.py')))['atomic_json'](
            link,dict(job_id=match['job_id']))
    return bootstrap(service)['existing_client'](service.mcp_root).status(json.loads(link.read_text(encoding='utf-8'))['job_id'])
