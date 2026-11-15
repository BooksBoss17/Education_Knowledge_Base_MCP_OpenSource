"""Verify the public inventory, wheel contents and source correspondence."""
from pathlib import Path
import hashlib
import json
import zipfile


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text('utf8'))
    for relative,expected in manifest['files'].items():
        p=(root/relative).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Release file changed: '+relative)
    tool=root/'TOOLS/bemarkdown';m=json.loads((tool/'TOOL_MANIFEST.json').read_text('utf8'))
    with zipfile.ZipFile(tool/m['wheel']['path']) as z:
        for name in z.namelist():
            if name.startswith('bemarkdown/') and not name.endswith('/'):
                if (root/'SOURCE'/name).read_bytes()!=z.read(name):raise ValueError('Wheel/source mismatch: '+name)
    return {'verified':True,'files':len(manifest['files']),'version':m['version']}


if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parents[1])))
