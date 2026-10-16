"""Installation-local bridge to the per-user persistent execution manager."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import runpy


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime(mcp_root):
    root = Path(mcp_root).resolve(strict=True)
    installed = root / 'TOOLS/runtime_manager'
    if str(installed / 'src') not in sys.path:
        sys.path.insert(0, str(installed / 'src'))
    from ekb_model_runtime.run_manager.client import RunManagerClient, ensure_daemon
    from ekb_model_runtime.run_manager.process import process_identity
    return root, installed, RunManagerClient, ensure_daemon, process_identity


def registry(mcp_root, python):
    root, installed, *_ = runtime(mcp_root)
    policy = root / 'TOOLS/bemarkdown_optimized/core/bemarkdown/host_memory_policy.py'
    configuration = root / 'TOOLS/bemarkdown_optimized/HOST_MEMORY_POLICY.json'
    mode = json.loads(configuration.read_text(encoding='utf-8')).get('mode', 'auto')
    local = Path(os.environ.get('LOCALAPPDATA', Path.home()))
    manifest = json.loads((root/'TOOLS/bemarkdown/TOOL_MANIFEST.json').read_text(encoding='utf-8'))
    conversion_python = runpy.run_path(str(root/'TOOLS/education_mcp/runtime_selection.py'))['resolve_runtime'](root).resolve(strict=True)
    semantic_python = None
    semantic_scheme = root/'TOOLS/semantic_retention/SCHEME.json'
    if semantic_scheme.is_file():
        try:
            semantic=runpy.run_path(str(root/'TOOLS/semantic_retention/bootstrap.py'))['runtime_config'](root/'TOOLS/semantic_retention')
            semantic_python=Path(semantic['source_python']).resolve(strict=True)
        except FileNotFoundError:
            pass  # OCR-only installations may explicitly disable image screening.
    programs = {}
    for name, relative in {
        'education-convert': 'education_mcp/managed_worker.py',
        'education-screen': 'education_mcp/managed_worker.py',
        'native-convert': 'runtime_manager/native_worker.py',
        'semantic-retention': 'semantic_retention/managed_worker.py',
    }.items():
        script = root / 'TOOLS' / relative
        if not script.is_file():
            continue
        if name=='semantic-retention' and semantic_python is None:
            continue
        program_python=semantic_python if name=='semantic-retention' else conversion_python
        programs[name] = dict(python=str(program_python), script=str(script),
                              script_sha256=sha(script), installed_root=str(root),
                              arguments=['--manager-job-id', '{job_id}', '--manager-request', '{request_path}'],
                              resource_kind='gpu', gpu_key='GPU:0',
                              requirement=dict(extra_private_bytes=128 * 1024**2,
                                               extra_physical_bytes=128 * 1024**2, extra_gpu_mib=0))
        if name == 'semantic-retention':
            programs[name]['requirement'] = dict(extra_private_bytes=int(5.9*1024**3)+128*1024**2,
                extra_physical_bytes=int(4.9*1024**3)+128*1024**2, extra_gpu_mib=5.6*1024)
        elif name == 'education-screen':
            # Conservative admission for the fixed BF16 4B worker; the worker
            # retains its existing dynamic GPU/CPU placement within these guards.
            programs[name]['requirement'] = dict(extra_private_bytes=11*1024**3,
                extra_physical_bytes=8*1024**3,extra_gpu_mib=3*1024)
    value = dict(schema='ekb-run-manager-registry-v1', programs=programs,
                 resource_policy=dict(module_path=str(policy), sha256=sha(policy), gpu_index=0, mode=mode))
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2).encode('utf-8')
    # Runtime absolute paths are generated on this machine, never shipped in a release.
    base = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'EKB/run-manager-registries'
    base.mkdir(parents=True, exist_ok=True)
    target = base / (hashlib.sha256(encoded).hexdigest() + '.json')
    if not target.exists():
        fd, temporary = tempfile.mkstemp(dir=base, suffix='.tmp')
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
        for attempt in range(6):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt==5:
                    raise
                time.sleep(.02 * 2**attempt)
    return target, programs


def client(mcp_root, python, workspace, input_roots=()):
    _, _, _, ensure_daemon, _ = runtime(mcp_root)
    path, programs = registry(mcp_root, python)
    result = ensure_daemon(registry_path=path, python=programs['native-convert']['python'])
    result.register_workspace(workspace, program_ids=list(programs), input_roots=input_roots)
    return result


def existing_client(mcp_root):
    _, _, Client, ensure_daemon, _ = runtime(mcp_root)
    connection = Client()
    try:
        connection.ping()
        return connection
    except Exception as exc:
        if not any(code in str(exc) for code in ('DAEMON_NOT_AVAILABLE', 'STALE_DAEMON_IDENTITY', 'Connection', '10061')):
            raise
    path, programs = registry(mcp_root, sys.executable)
    return ensure_daemon(registry_path=path, python=programs['native-convert']['python'])


def trusted_context(mcp_root):
    job = os.environ.get('EKB_MANAGED_JOB_ID')
    home = os.environ.get('EKB_RUN_MANAGER_HOME')
    if not job or not home:
        return None
    _, _, Client, _, identity = runtime(mcp_root)
    current = identity(os.getpid())
    result = Client(home).owns_process(job, current['pid'], current['created'])
    if not result.get('owned'):
        raise RuntimeError('MANAGED_PROCESS_OWNERSHIP_NOT_VERIFIED')
    return dict(job_id=job, home=home, identity=current)


def installation_fingerprint(mcp_root):
    root = Path(mcp_root)
    files = sorted((root / 'TOOLS/education_mcp').glob('*.py'))
    files += [root / 'TOOLS/bemarkdown_optimized/ROUTE_MANIFEST.json']
    return hashlib.sha256(json.dumps([(p.name, sha(p)) for p in files],
                                    separators=(',', ':')).encode()).hexdigest()
