"""Stable package fingerprint for explicit classification-only recovery."""
import hashlib
from pathlib import Path

def fingerprint(path):
    root=Path(path).resolve(strict=True)
    for name in ['document.md','conversion_report.json','assets_manifest.jsonl','package_manifest.json']:
        if not (root/name).is_file():raise ValueError('EXISTING_PACKAGE_INCOMPLETE')
    digest=hashlib.sha256()
    for file in sorted(root.rglob('*')):
        if not file.is_file():continue
        if not file.resolve(strict=True).is_relative_to(root):raise ValueError('EXISTING_PACKAGE_LINK_ESCAPES')
        digest.update(file.relative_to(root).as_posix().encode('utf-8')+b'\0')
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()
