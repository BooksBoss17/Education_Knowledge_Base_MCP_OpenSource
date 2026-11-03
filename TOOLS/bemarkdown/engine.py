"""Resolve and authenticate the installed wheel, never an overlay."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys, zipfile

def selected_python(tool):
    tool=Path(tool).resolve(); root=tool.parents[1]
    manifest=json.loads((tool/'TOOL_MANIFEST.json').read_text(encoding='utf-8'))
    receipt=root/'.local/runtime-selection.json'
    if receipt.is_file():
        value=json.loads(receipt.read_text(encoding='utf-8'))
        if value.get('status')!='READY_FULL' or value.get('wheel_sha256')!=manifest['wheel']['sha256']:
            raise ValueError('UNIFIED_RUNTIME_NOT_ACTIVATED')
        return Path(value['python']).resolve(strict=True)
    return (Path(os.environ['LOCALAPPDATA'])/'BeMarkdown/runtimes'/manifest['wheel']['sha256'][:12]/'Scripts/python.exe').resolve(strict=True)

def engine_root(tool):
    tool=Path(tool).resolve()
    manifest=json.loads((tool/'TOOL_MANIFEST.json').read_text(encoding='utf-8'))
    spec=importlib.util.find_spec('bemarkdown')
    if not spec or not spec.origin:raise ValueError('UNIFIED_ENGINE_NOT_INSTALLED')
    package=Path(spec.origin).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):raise ValueError('UNIFIED_ENGINE_OUTSIDE_RUNTIME')
    wheel=(tool/manifest['wheel']['path']).resolve(strict=True)
    if not wheel.is_relative_to(tool) or hashlib.sha256(wheel.read_bytes()).hexdigest()!=manifest['wheel']['sha256']:
        raise ValueError('UNIFIED_WHEEL_IDENTITY_CHANGED')
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('bemarkdown/') and not name.endswith('/'):
                target=(package.parent/name).resolve()
                if not target.is_relative_to(package) or target.read_bytes()!=archive.read(name):
                    raise ValueError('UNIFIED_INSTALLED_ENGINE_CHANGED:'+name)
    return package
