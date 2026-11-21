"""Persistent execution integration; frontend shutdown never owns worker lifetime."""
from pathlib import Path
import json
import runpy
import time


def bootstrap(service):
    return runpy.run_path(str(service.mcp_root / 'TOOLS/runtime_manager/bootstrap.py'))


def manager(service):
    return bootstrap(service)['client'](service.mcp_root, service.python, service.workspace, service.allowed)


def task_binding(service):
    slot=getattr(service,'_task_run',None)
    task=slot.get() if slot else None
    return {'task_run_id':task} if task else {}


def submit_conversion(service, source, material_type, book_title, source_sha256, retry_job_id=None):
    helpers = bootstrap(service)
    result = manager(service).submit('education-convert', service.workspace,
        dict(operation='convert', source=str(source), source_sha256=source_sha256,
             material_type=material_type, book_title=book_title,
             input_roots=[str(p) for p in service.allowed]),
        config_fingerprint=helpers['installation_fingerprint'](service.mcp_root), run_timeout=24*3600,
        retry_of=retry_job_id, **task_binding(service))
    return status(service, result['job_id'])


def paragraph_job(service, scope_key):
    return manager(service).find_scope('education-convert', service.workspace, scope_key)


def submit_paragraph(service, source, source_sha256, derivation, scope_key, retry_of):
    helpers = bootstrap(service)
    result = manager(service).submit('education-convert', service.workspace,
        dict(operation='convert', source=str(source), source_sha256=source_sha256,
             derivation=derivation, input_roots=[str(service.workspace)]),
        scope_key=scope_key, config_fingerprint=helpers['installation_fingerprint'](service.mcp_root),
        run_timeout=24*3600, retry_of=retry_of, **task_binding(service))
    return status(service, result['job_id'])


def status(service, job_id):
    path = service.state_path(job_id)
    observed = bootstrap(service)['existing_client'](service.mcp_root).status(job_id)
    if Path(observed['workspace']).resolve() != service.workspace:
        raise ValueError('Managed job belongs to a different workspace')
    state = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else dict(
        job_id=job_id, source=observed['payload']['source'],
        source_sha256=observed['payload']['source_sha256'], runtime_management='PERSISTENT_MANAGER')
    if 'derivation' in observed['payload']:
        state['derivation'] = observed['payload']['derivation']
    actual = observed['state']
    if actual not in {'SUCCEEDED'} or state.get('status') != 'SUCCEEDED':
        state['status'] = actual
    state['runtime_manager'] = observed
    progress_path = Path(observed['job_folder']) / 'model_api_progress.json' if observed.get('job_folder') else None
    if progress_path and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        if progress.get('job_id') != job_id:
            raise ValueError('Model API progress belongs to a different job')
        progress['job_state'] = actual
        progress['live'] = actual == 'RUNNING'
        state['model_api_progress'] = progress
    return state


def call(service, action='list', job_id=None, reason=None, task_run_id=None, outcome=None):
    if action in ('begin','resume','inspect','end'):
        client=manager(service) if action in ('begin','resume') else bootstrap(service)['existing_client'](service.mcp_root)
        return client.task_run(action,service.workspace,task_run_id,**({'outcome':outcome} if action=='end' else {}))
    client = bootstrap(service)['existing_client'](service.mcp_root)
    if action == 'list':
        return client.list(limit=20, workspace=service.workspace)
    if not job_id:
        raise ValueError('job_id required')
    if action == 'status':
        return client.status(job_id, workspace=service.workspace)
    if action == 'adopt':
        return client.adopt(job_id, 'education-mcp', workspace=service.workspace)
    if action == 'cancel':
        if task_run_id:
            raise ValueError('For task-scoped cancellation use end with outcome=cancel')
        return client.cancel(job_id, reason or 'Explicit user cancellation', wait_seconds=5, workspace=service.workspace)
    raise ValueError('Unknown runtime action')
