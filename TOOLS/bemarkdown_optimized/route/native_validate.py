"""Validate with the accepted native engine after authenticating its payload."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def verified_core(tool):
    tool=Path(tool).resolve()
    manifest=json.loads((tool/'ROUTE_MANIFEST.json').read_text(encoding='utf-8-sig'))
    if manifest.get('engine')!='ACCEPTED_NATIVE_CORE_NO_AUTOMATIC_COMPLETION':
        raise ValueError('NATIVE_VALIDATOR_ENGINE_POLICY_CHANGED')
    for relative,digest in manifest['files'].items():
        path=(tool/relative).resolve()
        if not path.is_relative_to(tool) or sha(path)!=digest:
            raise ValueError('NATIVE_VALIDATOR_PAYLOAD_CHANGED:'+relative)
    return tool/'core'

def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0]=='validate':arguments.pop(0)
    try:
        core=verified_core(Path(__file__).resolve().parents[1])
        sys.path.insert(0,str(core))
        from bemarkdown.cli import main as core_main
        return core_main(['validate',*arguments])
    except Exception as exc:
        print(json.dumps({'package_path':arguments[0] if arguments else None,'valid':False,
            'errors':[str(exc)],'warnings':[],'error_type':type(exc).__name__},ensure_ascii=False))
        return 2

if __name__=='__main__':raise SystemExit(main())
