"""Versioned tool environments with validated, reusable dependency environments."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import uuid
import venv
import zipfile

GPU_PINS = {'paddlepaddle-gpu': '3.2.2', 'torch': '2.13.0+cu130', 'torchvision': '0.28.0+cu130'}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending-' + uuid.uuid4().hex[:8])
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    os.replace(pending, path)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def python_at(root):
    return Path(root) / 'Scripts/python.exe'

def clean_env():
    result = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
        result.pop(name, None)
    result['PYTHONIOENCODING'] = 'utf-8'
    return result

def dependency_contract(root):
    lock = Path(root) / 'scripts/requirements-pypi.lock'
    pins = dict(GPU_PINS)
    archives = {}
    for raw in lock.read_text(encoding='utf-8').splitlines():
        row = raw.strip()
        if not row or row.startswith('#'):
            continue
        if ' @ ' in row:
            name, url = row.split(' @ ', 1)
            match = re.search(r'#sha256=([a-f0-9]{64})$', url)
            if not match:
                raise ValueError('Archive dependency must have a SHA256 pin: ' + name)
            archives[name.lower().replace('_', '-')] = match[1]
        else:
            name, version = row.split('==', 1)
            pins[name.lower().replace('_', '-')] = version
    data = dict(python=[3, 11], bits=64, platform='win32', pins=pins, archives=archives,
                lock_sha256=sha(lock))
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return dict(data, fingerprint=fingerprint)

PROBE = '''
import importlib.metadata as m,json,sys,struct
names=json.loads(sys.argv[1]); packages={}
for name in names:
 try:
  d=m.distribution(name); raw=d.read_text('direct_url.json')
  packages[name]={'version':d.version,'direct_url':json.loads(raw) if raw else {}}
 except m.PackageNotFoundError: pass
print(json.dumps({'python':list(sys.version_info[:2]),'bits':struct.calcsize('P')*8,'platform':sys.platform,'packages':packages}))
'''

def matches_contract(snapshot, contract):
    if any(snapshot.get(k) != contract[k] for k in ('python', 'bits', 'platform')):
        return False
    packages = snapshot.get('packages', {})
    if any(packages.get(n, {}).get('version') != v for n, v in contract['pins'].items()):
        return False
    for name, digest in contract['archives'].items():
        archive = packages.get(name, {}).get('direct_url', {}).get('archive_info', {})
        if archive.get('hashes', {}).get('sha256') != digest and archive.get('hash') != 'sha256=' + digest:
            return False
    return True

def inspect_dependencies(root, contract):
    python = python_at(root)
    if not python.is_file():
        return None
    try:
        result = subprocess.run([str(python), '-I', '-X', 'utf8', '-c', PROBE,
            json.dumps(sorted(set(contract['pins']) | set(contract['archives'])))],
            capture_output=True, check=True, timeout=60, env=clean_env())
        snapshot = json.loads(result.stdout)
        return snapshot if matches_contract(snapshot, contract) else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

def pip_command(root):
    python = str(python_at(root).resolve())
    if os.name == 'nt' and not python.startswith('\\\\?\\'):
        python = '\\\\?\\' + python
    return [python, '-I', '-X', 'utf8', '-m', 'pip', '--disable-pip-version-check']

def unused_target(path):
    path = Path(path)
    if not path.exists():
        return path
    # Never replace an existing environment, including an incomplete attempt.
    return path.with_name(path.name + '-' + uuid.uuid4().hex[:8])

def reserve_target(path):
    while True:
        candidate = unused_target(path)
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue

def ensure_dependencies(root, home, contract, candidates, explicit, run, logs):
    preferred = Path(home) / 'dependencies' / contract['fingerprint'][:16]
    checked = set()
    for candidate in ([explicit] if explicit else [*candidates, preferred]):
        if not candidate:
            continue
        candidate = Path(candidate).resolve()
        profile = candidate / 'bemarkdown-runtime.json'
        if profile.is_file():
            candidate = Path(read(profile)['dependency_runtime']).resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        if inspect_dependencies(candidate, contract):
            return candidate, True
    if explicit:
        raise ValueError('Requested dependency runtime does not match Python/ABI, pinned versions and archive SHA256')
    destination = reserve_target(preferred)
    venv.EnvBuilder(with_pip=True).create(destination)
    pip = pip_command(destination)
    run([*pip, 'install', '--no-deps', '-r', Path(root) / 'scripts/requirements-pypi.lock'], logs / 'pypi.log')
    run([*pip, 'install', '--no-deps', 'paddlepaddle-gpu==' + GPU_PINS['paddlepaddle-gpu'],
        '--index-url', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'], logs / 'paddle.log')
    # Only a new dependency environment installs Torch. Existing donors are read-only.
    run([*pip, 'install', '--no-deps', 'torch==' + GPU_PINS['torch'], 'torchvision==' + GPU_PINS['torchvision'],
        '--index-url', 'https://download.pytorch.org/whl/cu130'], logs / 'torch.log')
    snapshot = inspect_dependencies(destination, contract)
    if snapshot is None:
        raise RuntimeError('Installed dependencies do not match the frozen contract')
    write(destination / 'dependency-contract.json', dict(contract=contract, snapshot=snapshot))
    return destination, False

def verify_tool(runtime, wheel):
    try:
        result = subprocess.run([str(python_at(runtime)), '-I', '-X', 'utf8', '-c',
            'import json,bemarkdown; print(json.dumps(bemarkdown.__file__))'],
            capture_output=True, check=True, timeout=30, env=clean_env())
        module = Path(json.loads(result.stdout)).resolve()
        site = (Path(runtime) / 'Lib/site-packages').resolve()
        if module.parent != site / 'bemarkdown':
            return False
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.startswith('bemarkdown/') and not name.endswith('/'):
                    target = (site / name).resolve()
                    if not target.is_relative_to(site) or not target.is_file() or target.read_bytes() != archive.read(name):
                        return False
        return True
    except (OSError, subprocess.SubprocessError, ValueError):
        return False

def shared_path(dependencies):
    return 'import site; site.addsitedir(' + repr(str(Path(dependencies) / 'Lib/site-packages')) + ')\n'

def ensure_tool(target, dependencies, wheel, contract, run, logs):
    target = Path(target).resolve()
    profile = dict(wheel_sha256=sha(wheel), dependency_runtime=str(Path(dependencies).resolve()),
                   dependency_fingerprint=contract['fingerprint'])
    marker = target / 'bemarkdown-runtime.json'
    pth = target / 'Lib/site-packages/bemarkdown_dependencies.pth'
    if marker.is_file() and read(marker) == profile and pth.is_file() and pth.read_text(encoding='utf-8') == shared_path(dependencies) and verify_tool(target, wheel):
        return target, True
    target = reserve_target(target)
    venv.EnvBuilder(with_pip=False).create(target)
    (target / 'Lib/site-packages/bemarkdown_dependencies.pth').write_text(shared_path(dependencies), encoding='utf-8', newline='\n')
    run([*pip_command(target), 'install', '--no-deps', '--no-index', '--ignore-installed', wheel], logs / 'bemarkdown.log')
    if not verify_tool(target, wheel):
        raise RuntimeError('Installed tool does not match the published wheel')
    write(target / 'bemarkdown-runtime.json', profile)
    return target, False

def select_workspace(explicit, previous, old_client):
    if explicit:
        return str(Path(explicit).resolve())
    if previous.get('workspace'):
        return previous['workspace']
    args = old_client.get('mcpServers', {}).get('education', {}).get('args', [])
    if '--workspace' in args:
        index = args.index('--workspace')
        if index + 1 < len(args):
            return args[index + 1]
    return None

def choose_tool_target(requested, default, active, digest):
    base = Path(requested or default).resolve()
    if active.get('wheel_sha256') == digest and active.get('python'):
        actual = Path(active['python']).resolve().parent.parent
        original = Path(active.get('requested_runtime_root', actual)).resolve()
        if not requested or base in (actual, original):
            return original, actual
    return base, base

def activation_record(wheel, runtime, dependencies, previous, previous_state, old_client):
    aliases = set(previous.get('legacy_runtime_pythons', []))
    for value in (previous.get('python'), previous_state.get('runtime')):
        if value:
            p = Path(value)
            aliases.add(str((python_at(p) if p.suffix.lower() != '.exe' else p).resolve()))
    entry = old_client.get('mcpServers', {}).get('education', {})
    args = entry.get('args', [])
    if '--runtime-python' in args and args.index('--runtime-python') + 1 < len(args):
        aliases.add(str(Path(args[args.index('--runtime-python') + 1]).resolve()))
    return dict(schema='bemarkdown-active-runtime-v1', status='READY_FULL', wheel_sha256=sha(wheel),
                python=str(python_at(runtime).resolve()), dependency_runtime=str(Path(dependencies).resolve()),
                legacy_runtime_pythons=sorted(aliases))
