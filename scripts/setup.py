"""Install the full current Windows MCP without redistributing external binaries."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import venv

from downloads import download, safe_path, sha256, valid

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    os.replace(temporary, path)

def run(command, log, *, capture=False):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('wb') as stream:
        result = subprocess.run([str(x) for x in command], stdout=subprocess.PIPE if capture else stream,
                                stderr=stream if capture else subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(f'Command failed with exit {result.returncode}; see {log}')
    return result.stdout.decode('utf-8-sig') if capture else None

def models(selection, cache, local_cache=None, workers=8):
    plan = read(ROOT / 'scripts/model-downloads.json')['models']
    for model in plan:
        if selection == 'conversion' and model['purpose'] != 'conversion':
            continue
        if selection == 'none':
            continue
        manifest = safe_path(ROOT / 'MODELS', model['manifest_path'])
        if sha256(manifest) != model['manifest_sha256']:
            raise ValueError('Model authority manifest changed: ' + model['model_id'])
        print('Checking model ' + model['model_id'], flush=True)
        directory = safe_path(ROOT / 'MODELS', model['directory'])
        for file in model['files']:
            target = safe_path(directory, file['path'])
            if valid(target, file['sha256'], file['bytes']):
                continue
            cached = safe_path(Path(local_cache) / model['directory'], file['path']) if local_cache else None
            if cached and valid(cached, file['sha256'], file['bytes']):
                source = cached
            else:
                url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/" + urllib.parse.quote(file['path'], safe='/')
                source = download(url, file['sha256'], file['bytes'], cache, workers=workers)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.installing')
            shutil.copyfile(source, temporary)
            if not valid(temporary, file['sha256'], file['bytes']):
                raise ValueError('Copied model file failed verification')
            os.replace(temporary, target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-external-licenses', action='store_true')
    parser.add_argument('--models', choices=['all', 'conversion', 'none'], default='all')
    parser.add_argument('--model-cache', type=Path, help='Optional existing MODELS directory; imported bytes are checked')
    parser.add_argument('--runtime-dir', type=Path)
    parser.add_argument('--download-cache', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--download-workers', type=int, default=8, choices=range(1, 33))
    parser.add_argument('--models-only', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt' or sys.version_info[:2] != (3, 11):
        parser.error('The validated full runtime requires Windows x64 and CPython 3.11. Run setup.ps1.')
    if not args.models_only and not args.accept_external_licenses:
        parser.error('Full setup downloads NVIDIA proprietary runtime components. Read docs/LICENSE_AUDIT.md and pass --accept-external-licenses after accepting their upstream terms.')
    local = ROOT / '.local'
    cache = args.download_cache or Path(os.environ.get('LOCALAPPDATA', str(local))) / 'EducationMCP/downloads'
    models(args.models, cache, args.model_cache, args.download_workers)
    if args.models_only:
        print('Selected model files verified.')
        return
    office_candidates = [os.environ.get('BEMARKDOWN_SOFFICE'), shutil.which('soffice'), shutil.which('soffice.com')]
    office_candidates += [str(Path(os.environ[key]) / 'LibreOffice/program/soffice.com') for key in ('ProgramFiles', 'ProgramFiles(x86)') if os.environ.get(key)]
    if not any(path and Path(path).is_file() for path in office_candidates):
        raise RuntimeError('LibreOffice is required for screenshot DOCX pagination. Run setup.ps1 to install it.')
    manifest = read(ROOT / 'TOOLS/bemarkdown/TOOL_MANIFEST.json')
    wheel = safe_path(ROOT / 'TOOLS/bemarkdown', manifest['wheel']['path'])
    if sha256(wheel) != manifest['wheel']['sha256']:
        raise ValueError('Tool wheel integrity check failed')
    runtime = (args.runtime_dir or Path(os.environ.get('LOCALAPPDATA', str(local))) / 'BeMarkdown/runtimes' / manifest['wheel']['sha256'][:12]).resolve()
    python = runtime / 'Scripts/python.exe'
    state_path = local / 'setup-state.json'
    state = dict(status='INSTALLING', started_at=time.time(), runtime=str(runtime), models=args.models)
    write(state_path, state)
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(runtime)
    # Preserve Windows extended paths inside sys.prefix during wheel extraction.
    # Torch carries deeply nested license files; none are removed or flattened.
    install_python = str(python)
    if os.name == 'nt' and not install_python.startswith('\\\\?\\'):
        install_python = '\\\\?\\' + install_python
    pip = [install_python, '-m', 'pip', '--disable-pip-version-check']
    logs = local / 'logs' / time.strftime('%Y%m%dT%H%M%S')
    state['logs'] = str(logs)
    write(state_path, state)
    try:
        print('Installing frozen Python dependencies; logs are in .local/logs.', flush=True)
        run([*pip, 'install', '--no-deps', '-r', ROOT / 'scripts/requirements-pypi.lock'], logs / 'pypi.log')
        run([*pip, 'install', '--no-deps', 'paddlepaddle-gpu==3.2.2', '--index-url', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'], logs / 'paddle.log')
        # A failed prior extraction can leave METADATA without a RECORD. Overlay
        # the same pinned wheels to repair that state without an uninstall step.
        run([*pip, 'install', '--no-deps', '--ignore-installed', 'torch==2.13.0+cu130', 'torchvision==0.28.0+cu130', '--index-url', 'https://download.pytorch.org/whl/cu130'], logs / 'torch.log')
        run([*pip, 'install', '--no-deps', '--force-reinstall', wheel], logs / 'bemarkdown.log')
        check = subprocess.run([python, '-m', 'pip', 'check'], capture_output=True)
        output = (check.stdout + check.stderr).decode('utf-8', errors='replace').strip()
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        known_override = check.returncode == 1 and len(lines) == 1 and 'paddlepaddle-gpu' in lines[0] and 'nvidia-cudnn-cu12==9.5.1.17' in lines[0] and '9.9.0.52' in lines[0]
        write(local / 'pip-check.json', dict(returncode=check.returncode, expected_cudnn_override=known_override, output=output))
        if check.returncode and not known_override:
            raise RuntimeError('Unexpected dependency conflict; see .local/pip-check.json')
        print('Running full GPU and model validation...', flush=True)
        doctor_text = run([python, '-I', '-X', 'utf8', '-m', 'bemarkdown', 'doctor', '--deep', '--json', '--tool-root', ROOT / 'TOOLS/bemarkdown', '--mcp-root', ROOT], logs / 'doctor.log', capture=True)
        doctor = json.loads(doctor_text)
        write(local / 'doctor.json', doctor)
        if doctor['readiness'] != 'READY_FULL':
            raise RuntimeError('Full GPU/model validation did not pass; see .local/doctor.json')
        install = [python, ROOT / 'TOOLS/education_mcp/install.py', '--mcp-root', ROOT]
        if args.workspace:
            install += ['--workspace', args.workspace.resolve()]
        workspace = json.loads(run(install, logs / 'workspace.log', capture=True))
        config = workspace['mcp_client_configuration']
        config['mcpServers']['education']['args'] += ['--runtime-python', str(python)]
        write(local / 'mcp-client.json', config)
        state.update(status='READY_FULL', completed_at=time.time(), client_configuration=str(local / 'mcp-client.json'))
        write(state_path, state)
        write(logs / 'setup-state.json', state)
        print('READY_FULL. MCP client configuration: ' + str(local / 'mcp-client.json'))
    except BaseException as exc:
        state.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc))
        write(state_path, state)
        write(logs / 'setup-state.json', state)
        raise

if __name__ == '__main__':
    main()
