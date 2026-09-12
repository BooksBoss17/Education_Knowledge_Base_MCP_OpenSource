"""Check the public release using Python 3.11 standard library only.

Run before setup. --installed permits downloaded models and local runtime state.
The manifest detects corruption; it is not an authenticated publisher signature.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def safe(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or ':' in name or not path.is_relative_to(root.resolve()):
        raise ValueError('Unsafe manifest path: ' + name)
    return path

def verify(root=ROOT, installed=False):
    manifest = read(root / 'RELEASE_MANIFEST.json')
    expected = {f['path']: f for f in manifest['files']}
    for name, entry in expected.items():
        path = safe(root, name)
        assert path.is_file() and path.stat().st_size == entry['bytes'], name
        assert sha(path) == entry['sha256'], name
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if not installed:
        extra = actual - set(expected) - {'RELEASE_MANIFEST.json'}
        extra = {n for n in extra if not n.startswith('.git/') and '__pycache__' not in Path(n).parts}
        assert not extra, sorted(extra)
    tool = root / 'TOOLS/bemarkdown'
    authority = read(tool / 'TOOL_MANIFEST.json')
    wheel = safe(tool, authority['wheel']['path'])
    assert sha(wheel) == authority['wheel']['sha256']
    for entry in authority['runtime'].values():
        if isinstance(entry, dict) and 'path' in entry and 'sha256' in entry:
            assert sha(safe(tool, entry['path'])) == entry['sha256'], entry['path']
    modules = 0
    with zipfile.ZipFile(wheel) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if name.startswith('bemarkdown/') and not name.endswith('/'):
                source = safe(root / 'SOURCE/bemarkdown/src', name)
                assert source.read_bytes() == archive.read(name), name
                modules += name.endswith('.py')
    plan = read(root / 'scripts/model-downloads.json')['models']
    registry = {m['model_id']: m for m in read(root / 'MODELS/MODEL_REGISTRY.json')['models']}
    assert len(plan) == len(registry) == 13
    for model in plan:
        path = safe(root / 'MODELS', model['manifest_path'])
        assert sha(path) == model['manifest_sha256'] == registry[model['model_id']]['manifest_sha256']
        authority_model = read(path)
        fields = lambda rows: sorted((f['path'], f['sha256'], f['bytes']) for f in rows)
        assert fields(model['files']) == fields(authority_model['files'])
        assert re.fullmatch('[a-f0-9]{40}', model['revision']), model['model_id']
        assert model['license'] == 'Apache-2.0'
        for item in model['files']:
            safe(root / 'MODELS' / model['directory'], item['path'])
            assert '.cache' not in Path(item['path']).parts
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'docs/LICENSE_AUDIT.md', 'docs/VALIDATION.md', 'docs/BEMARKDOWN_TECHNICAL_SCHEME.md'):
        assert (root / name).stat().st_size > 100, name
    forbidden_parts = {'.local', '.venv', '.cache', 'build', 'site-packages', 'node_modules'}
    text_suffixes = {'.py', '.ps1', '.json', '.md', '.txt', '.toml', '.lock', '.cfg', '.ini'}
    for name in expected:
        path = Path(name)
        assert not forbidden_parts.intersection(path.parts), name
        assert not any(part.endswith('.egg-info') for part in path.parts), name
        assert not path.suffix.lower() in {'.safetensors', '.pdiparams', '.pt', '.pth', '.docx', '.pdf'}, name
        assert (root / name).stat().st_size < 90_000_000, name
        if path.suffix.lower() in text_suffixes:
            content = (root / name).read_text(encoding='utf-8-sig')
            assert not re.search(r'[a-z]:[\\/]+Users[\\/]+[^\\/\s<>]+', content, re.I), name
    return dict(status='PASS', files=len(expected), python_modules=modules, models=len(plan), wheel_sha256=sha(wheel), bytes=sum(f['bytes'] for f in expected.values()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(installed=args.installed), indent=2))
