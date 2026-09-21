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
from verify_release import verify
from runtime_setup import (activation_record, choose_tool_target, clean_env, dependency_contract,
    ensure_dependencies, ensure_tool, python_at, select_workspace, verify_tool)

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending-' + os.urandom(4).hex())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    os.replace(temporary, path)

def run(command, log, *, capture=False):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('wb') as stream:
        result = subprocess.run([str(x) for x in command], stdout=subprocess.PIPE if capture else stream,
                                stderr=stream if capture else subprocess.STDOUT,
                                env=clean_env(), creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
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
    parser.add_argument('--models', choices=['all', 'conversion', 'none'])
    parser.add_argument('--model-cache', type=Path, help='Optional existing MODELS root; only hash-matching files are reused')
    parser.add_argument('--runtime-dir', type=Path)
    parser.add_argument('--dependency-runtime', type=Path, help='Reuse a compatible environment without changing it')
    parser.add_argument('--download-cache', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--download-workers', type=int, default=8, choices=range(1, 33))
    parser.add_argument('--models-only', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt' or sys.version_info[:2] != (3, 11):
        parser.error('The complete installer requires Windows and CPython 3.11. Use setup.ps1.')
    if not args.models_only and not args.accept_external_licenses:
        parser.error('Read docs/LICENSE_AUDIT.md and accept external terms before full installation.')
    verify(ROOT, installed=True)
    local = ROOT / '.local'
    cache = args.download_cache or Path(os.environ.get('LOCALAPPDATA', str(local))) / 'EducationMCP/downloads'
    previous_state = read(local / 'setup-state.json') if (local / 'setup-state.json').is_file() else {}
    previous = read(local / 'runtime-selection.json') if (local / 'runtime-selection.json').is_file() else {}
    old_client = read(local / 'mcp-client.json') if (local / 'mcp-client.json').is_file() else {}
    selection = args.models or previous_state.get('models', 'all')
    workspace_path = select_workspace(args.workspace, previous_state, old_client)
    models(selection, cache, args.model_cache, args.download_workers)
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
    home = Path(os.environ.get('LOCALAPPDATA', str(local))) / 'BeMarkdown/runtimes'
    requested_target, target = choose_tool_target(args.runtime_dir, home / manifest['wheel']['sha256'][:12],
        previous, manifest['wheel']['sha256'])
    state = dict(status='INSTALLING', started_at=time.time(), models=selection, requested_runtime_root=str(requested_target))
    logs = local / 'logs' / (time.strftime('%Y%m%dT%H%M%S') + '-' + os.urandom(4).hex())
    state['logs'] = str(logs)
    write(local / 'last-setup-attempt.json', state)
    write(logs / 'previous-activation.json', previous)
    try:
        contract = dependency_contract(ROOT)
        donors = [previous.get('dependency_runtime'), previous_state.get('dependency_runtime'), previous_state.get('runtime')]
        dependencies, dependencies_reused = ensure_dependencies(ROOT, home, contract, donors, args.dependency_runtime, run, logs)
        runtime, tool_reused = ensure_tool(target, dependencies, wheel, contract, run, logs)
        python = python_at(runtime)
        state.update(runtime=str(runtime), dependency_runtime=str(dependencies), dependency_fingerprint=contract['fingerprint'],
                     dependencies_reused=dependencies_reused, tool_reused=tool_reused)
        write(local / 'last-setup-attempt.json', state)
        check = subprocess.run([python, '-I', '-X', 'utf8', '-m', 'pip', 'check'], capture_output=True, env=clean_env())
        output = (check.stdout + check.stderr).decode('utf-8', errors='replace').strip()
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        accepted = check.returncode == 1 and len(lines) == 1 and 'paddlepaddle-gpu' in lines[0] and 'nvidia-cudnn-cu12==9.5.1.17' in lines[0] and '9.9.0.52' in lines[0]
        write(local / 'pip-check.json', dict(returncode=check.returncode, accepted_cudnn_metadata_override=accepted, output=output))
        if check.returncode and not accepted:
            raise RuntimeError('Unexpected dependency inconsistency; see .local/pip-check.json')
        doctor_text = run([python, '-I', '-X', 'utf8', '-m', 'bemarkdown', 'doctor', '--deep', '--json',
            '--tool-root', ROOT / 'TOOLS/bemarkdown', '--mcp-root', ROOT], logs / 'doctor.log', capture=True)
        doctor = json.loads(doctor_text)
        write(local / 'doctor.json', doctor)
        if doctor['readiness'] != 'READY_FULL':
            raise RuntimeError('Full GPU/model validation did not pass; see .local/doctor.json')
        if not verify_tool(runtime, wheel):
            raise RuntimeError('Tool integrity changed during validation')
        install = [python, '-I', '-X', 'utf8', ROOT / 'TOOLS/education_mcp/install.py', '--mcp-root', ROOT]
        if workspace_path:
            install += ['--workspace', workspace_path]
        workspace = json.loads(run(install, logs / 'workspace.log', capture=True))
        config = workspace['mcp_client_configuration']
        config['mcpServers']['education']['command'] = str(Path(sys._base_executable).resolve())
        write(local / 'mcp-client.json', config)
        client_args = config['mcpServers']['education']['args']
        state.update(status='READY_FULL', completed_at=time.time(), workspace=client_args[client_args.index('--workspace') + 1],
                     client_configuration=str(local / 'mcp-client.json'))
        write(local / 'setup-state.json', state)
        write(logs / 'setup-state.json', state)
        write(local / 'last-setup-attempt.json', state)
        # Activation is the final atomic write; failed preparation never changes the selector.
        active = activation_record(wheel, runtime, dependencies, previous, previous_state, old_client)
        active['requested_runtime_root'] = str(requested_target)
        write(local / 'runtime-selection.json', active)
        print('READY_FULL. MCP client configuration: ' + str(local / 'mcp-client.json'))
    except BaseException as exc:
        state.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc))
        write(local / 'last-setup-attempt.json', state)
        write(logs / 'setup-state.json', state)
        raise

if __name__ == '__main__':
    main()
