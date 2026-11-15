"""Resolve the local installation receipt and authenticate its core wheel."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import zipfile


def selected_python(tool):
    tool=Path(tool).resolve();root=tool.parents[1]
    manifest=json.loads((tool/'TOOL_MANIFEST.json').read_text(encoding='utf-8'))
    value=json.loads((root/'.local/runtime-selection.json').read_text(encoding='utf-8'))
    if value.get('status')!='READY_FULL' or value.get('wheel_sha256')!=manifest['wheel']['sha256']:
        raise ValueError('BEMARKDOWN_RUNTIME_NOT_ACTIVATED')
    path=Path(value['python'])
    return (path if path.is_absolute() else root/path).resolve(strict=True)


def engine_root(tool):
    tool=Path(tool).resolve()
    manifest=json.loads((tool/'TOOL_MANIFEST.json').read_text(encoding='utf-8'))
    spec=importlib.util.find_spec('bemarkdown')
    if not spec or not spec.origin:raise ValueError('BEMARKDOWN_CORE_NOT_INSTALLED')
    package=Path(spec.origin).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):raise ValueError('CORE_OUTSIDE_RUNTIME')
    wheel=(tool/manifest['wheel']['path']).resolve(strict=True)
    if not wheel.is_relative_to(tool) or hashlib.sha256(wheel.read_bytes()).hexdigest()!=manifest['wheel']['sha256']:
        raise ValueError('CORE_WHEEL_CHANGED')
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('bemarkdown/') and not name.endswith('/'):
                path=(package.parent/name).resolve()
                if not path.is_relative_to(package) or path.read_bytes()!=archive.read(name):
                    raise ValueError('INSTALLED_CORE_CHANGED:'+name)
    return package
