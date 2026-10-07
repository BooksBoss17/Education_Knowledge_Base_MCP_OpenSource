from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_streams_and_verifies_parts_then_final_file(tmp_path, monkeypatch):
    module = load('retention_download_test', ROOT / 'scripts/download_retention_model.py')
    data = [b'first bytes', b'second bytes']
    parts = []
    urls = {}
    for number, content in enumerate(data):
        name = f'fixture.part{number}'
        url = 'https://github.com/BooksBoss17/BeMarkdown-Retention-B200/releases/download/test/' + name
        urls[url] = content
        parts.append({'name': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(), 'url': url})
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda request, timeout: io.BytesIO(urls[request.full_url]))
    assembled = b''.join(data)
    manifest = {'files': [{'path': 'fixture.gguf', 'bytes': len(assembled), 'sha256': hashlib.sha256(assembled).hexdigest(), 'parts': parts}]}
    module.install(manifest, tmp_path)
    assert (tmp_path / 'fixture.gguf').read_bytes() == assembled
    module.install(manifest, tmp_path, verify_only=True)
    (tmp_path / 'fixture.gguf').write_bytes(b'wrong content')
    with pytest.raises(ValueError, match='mismatched'):
        module.install(manifest, tmp_path, verify_only=True)
    with pytest.raises(ValueError, match='Unsafe'):
        module.safe(tmp_path, '../outside.gguf')


def test_retention_assets_are_explicit_opt_in(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = load('public_setup_test', ROOT / 'scripts/setup.py')
    model_dir = tmp_path / 'MODELS/BeMarkdown-Retention-B200'
    model_dir.mkdir(parents=True)
    manifest = model_dir / 'MODEL_MANIFEST.json'
    manifest.write_text('{}', encoding='utf-8')
    plan = {'models': [{'model_id': 'bemarkdown-retention-b200', 'purpose': 'standalone-optional', 'distribution': 'github-release', 'directory': 'BeMarkdown-Retention-B200', 'manifest_path': 'BeMarkdown-Retention-B200/MODEL_MANIFEST.json', 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}]}
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'read', lambda path: plan)
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda command, check: calls.append(command))
    module.models('all', tmp_path / 'cache')
    assert calls == []
    module.models('all', tmp_path / 'cache', include_retention=True)
    assert len(calls) == 1 and 'download_retention_model.py' in calls[0][1]
    assert calls[0][-1] == str(model_dir)


def test_public_template_absence_never_loads_a_model_or_excludes(tmp_path):
    module = runpy.run_path(str(ROOT / 'TOOLS/education_mcp/screening_service.py'))
    source = tmp_path / 'synthetic-source'
    source.write_bytes(b'synthetic data')
    markdown = tmp_path / 'document.md'
    markdown.write_text('synthetic markdown', encoding='utf-8')
    state = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    flow = {'current': lambda service, job: (state, tmp_path, markdown, 'synthetic markdown'), 'images': lambda package, text: [{'image_id': 'fixture'}]}
    result = module['request'](SimpleNamespace(), 'fixture-job', flow, None)
    assert result['status'] == 'NOT_CONFIGURED'
    assert result['actual_exclusions'] == 0


def test_public_installer_help_has_no_duplicate_options():
    import subprocess
    import sys
    text = subprocess.check_output([sys.executable, str(ROOT / 'scripts/setup.py'), '--help'], text=True)
    assert '--include-retention-model' in text and '--qwen-python' in text


def test_qwen_installer_reuses_valid_environment_and_rejects_drift(tmp_path, monkeypatch):
    module = load('qwen_installer_test', ROOT / 'scripts/install_qwen_runtime.py')
    target = tmp_path / 'runtimes' / module.NAME / 'Scripts/python.exe'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'fixture')
    expected = module.expected_versions(ROOT / 'TOOLS/bemarkdown')
    monkeypatch.setattr(module, 'query', lambda python: {'python': '3.12.14', 'versions': expected})
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: pytest.fail('No installer is allowed on a verified existing environment'))
    assert module.ensure(ROOT / 'TOOLS/bemarkdown', tmp_path) == target
    wrong = dict(expected, torch='wrong')
    monkeypatch.setattr(module, 'query', lambda python: {'python': '3.12.14', 'versions': wrong})
    with pytest.raises(RuntimeError, match='differs'):
        module.ensure(ROOT / 'TOOLS/bemarkdown', tmp_path)


def test_qwen_cold_pip_calls_only_target_dedicated_environment(tmp_path, monkeypatch):
    module = load('qwen_cold_installer_test', ROOT / 'scripts/install_qwen_runtime.py')
    expected = module.expected_versions(ROOT / 'TOOLS/bemarkdown')
    monkeypatch.setattr(module, 'query', lambda python: {'python': '3.12.14', 'versions': expected})
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: '(3, 12)')
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    source = tmp_path / 'source-python.exe'
    python = module.ensure(ROOT / 'TOOLS/bemarkdown', tmp_path, source)
    assert calls[0] == [str(source), '-m', 'venv', str(python.parent.parent)]
    assert len(calls) == 3
    assert all(command[:4] == [str(python), '-m', 'pip', 'install'] for command in calls[1:])
    assert all('paddlepaddle-gpu' not in ' '.join(command) for command in calls)
