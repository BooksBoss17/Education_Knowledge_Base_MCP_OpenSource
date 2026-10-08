"""Atomic local checkpoint lineage and owned-state restore. No external checkpoints accepted."""

import hashlib

import importlib.metadata

import json

import os

import pathlib

import random

import time

import uuid

P = pathlib.Path(__file__).resolve().parents[1]

R = P.parents[1]

def sha_file(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def fingerprint():
    def load(p):
        return json.loads(pathlib.Path(p).read_text('utf-8'))
    deps = load(P / 'reports/native-dependencies.json')
    assert deps['status'] == 'ISOLATED_DEPENDENCIES_INSTALLED_NOT_TRAINED', 'DEPENDENCIES_NOT_LOCKED_AND_INSTALLED'
    lock = P / 'configs/requirements-native.lock'
    assert sha_file(lock) == deps['lock_sha256']
    expected = {'torch': '2.10.0+cu130', 'transformers': '5.5.0', 'trl': '0.24.0', 'peft': '0.18.0', 'bitsandbytes': '0.49.2'}
    versions = {k: importlib.metadata.version(k) for k in expected}
    assert versions == expected, 'ACTUAL_RUNTIME_VERSION_DRIFT'
    report = load(P / 'reports/base-download.json')
    base = pathlib.Path(report['local_path'])
    for row in report['files']:
        stat = (base / row['file']).stat()
        assert stat.st_size == row['size'] and stat.st_mtime_ns == row['mtime_ns'], 'VERIFIED_BASE_FILE_CHANGED'
    sources = sorted((P / 'tools').glob('*.py')) + sorted((P / 'configs').glob('*.json'))
    sources += [P / 'prompts/model-decision-system.txt', P / 'data/splits/v2/split-lock.json']
    files = {p.relative_to(P).as_posix(): sha_file(p) for p in sources}
    bootstrap = load(P / 'reports/native-environment-bootstrap.json')
    for guard in (pathlib.Path(bootstrap['base_python']).parent / 'Lib/site-packages/sitecustomize.py',
                  pathlib.Path(bootstrap['venv_python']).parent.parent / 'Lib/site-packages/sitecustomize.py'):
        assert sha_file(guard) == bootstrap['private_wmi_fallback_sha256'] == sha_file(P / 'tools/native_sitecustomize.py'), 'PRIVATE_RUNTIME_GUARD_CHANGED'
    return {'versions': versions, 'dependency_lock_sha256': deps['lock_sha256'],
            'base_revision': report['revision'], 'base_files': {r['file']: r['sha256'] for r in report['files']},
            'source_files': files}

def source_hash(lineage, name):
    files = lineage['source_files']
    return files.get(name, files.get(name.replace('/', '\\')))

def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}

def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])

def save_checkpoint(model, optimizer, scheduler, step, sampler, out, lineage):
    import torch
    out = pathlib.Path(out).resolve()
    assert out.is_relative_to(R / 'tmp/rt-train'), 'CHECKPOINT_OUTSIDE_WORKSPACE_CACHE'
    if out.exists():
        raise FileExistsError('IMMUTABLE_CHECKPOINT_ALREADY_EXISTS')
    pending = out.parent / (out.name + '.partial-' + uuid.uuid4().hex[:8])
    pending.mkdir(parents=True)
    model.save_pretrained(pending, safe_serialization=True)
    state = {'step': step, 'sampler': sampler, 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict() if scheduler is not None else None, 'rng': rng_state()}
    torch.save(state, pending / 'training-state.pt')
    manifest = {'schema_version': '1.0', 'status': 'COMPLETE', 'step': step, 'sampler': sampler,
                'lineage': lineage, 'created_epoch': time.time(),
                'files_sha256': {p.name: sha_file(p) for p in pending.iterdir() if p.is_file()}}
    (pending / 'checkpoint-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    os.replace(pending, out)
    return out

def validate_checkpoint(path, lineage):
    path = pathlib.Path(path).resolve()
    assert path.is_relative_to(R / 'tmp/rt-train'), 'EXTERNAL_PICKLE_CHECKPOINT_FORBIDDEN'
    manifest = json.loads((path / 'checkpoint-manifest.json').read_text('utf-8'))
    assert manifest['status'] == 'COMPLETE' and manifest['lineage'] == lineage, 'CHECKPOINT_LINEAGE_MISMATCH'
    for name, expected in manifest['files_sha256'].items():
        assert pathlib.Path(name).name == name, 'CHECKPOINT_MANIFEST_PATH_ESCAPE'
        target = (path / name).resolve()
        assert target.is_relative_to(path), 'CHECKPOINT_MANIFEST_SYMLINK_ESCAPE'
        assert sha_file(target) == expected, 'CHECKPOINT_FILE_SHA_MISMATCH'
    return path, manifest

def restore_checkpoint_state(path, optimizer, scheduler, lineage):
    path, manifest = validate_checkpoint(path, lineage)
    import torch
    # Own checkpoint is authenticated before loading its optimizer/RNG pickle.
    state = torch.load(path / 'training-state.pt', map_location='cpu', weights_only=False)
    assert state['step'] == manifest['step'] and state['sampler'] == manifest['sampler']
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        assert state['scheduler'] is not None
        scheduler.load_state_dict(state['scheduler'])
    else:
        assert state['scheduler'] is None
    restore_rng(state['rng'])
    return state
