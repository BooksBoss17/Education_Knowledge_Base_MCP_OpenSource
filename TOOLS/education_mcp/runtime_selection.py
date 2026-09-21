"""Resolve the validated activation record, retaining explicit test overrides."""
from pathlib import Path
import json
import os

def resolve_runtime(root, requested=None):
    root = Path(root).resolve()
    manifest = json.loads((root / 'TOOLS/bemarkdown/TOOL_MANIFEST.json').read_text(encoding='utf-8'))
    digest = manifest['wheel']['sha256']
    path = root / '.local/runtime-selection.json'
    active = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    requested = Path(requested).resolve() if requested else None
    aliases = {Path(p).resolve() for p in active.get('legacy_runtime_pythons', [])} if active else set()
    if requested and requested not in aliases:
        return requested
    if active:
        if active.get('status') != 'READY_FULL' or active.get('wheel_sha256') != digest:
            raise ValueError('Published wheel is not activated. Run setup.ps1 successfully before restarting MCP.')
        return Path(active['python']).resolve()
    home = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'BeMarkdown/runtimes'
    return home / digest[:12] / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
