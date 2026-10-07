"""Apache-2.0. Stream and verify split GitHub Release model artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import urllib.request


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe artifact path')
    path = (root / str(name)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact escaped output directory')
    return path


def valid(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']


def fetch(part, cache):
    target = safe(cache, part['name'])
    if valid(target, part):
        return target
    if target.exists():
        raise ValueError('Existing cached artifact does not match: ' + str(target))
    if not part['url'].startswith('https://github.com/BooksBoss17/BeMarkdown-Retention-B200/releases/download/'):
        raise ValueError('Unexpected download authority')
    pending = target.with_name(target.name + '.pending')
    request = urllib.request.Request(part['url'], headers={'User-Agent': 'BeMarkdown-Retention-Downloader'})
    digest = hashlib.sha256()
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, pending.open('wb') as stream:
        for block in iter(lambda: response.read(1024 * 1024), b''):
            size += len(block)
            if size > part['bytes']:
                raise ValueError('Download exceeds declared size')
            digest.update(block)
            stream.write(block)
    if size != part['bytes'] or digest.hexdigest() != part['sha256']:
        raise ValueError('Download checksum mismatch: ' + part['name'])
    os.replace(pending, target)
    return target


def install(manifest, output, verify_only=False):
    output = Path(output).resolve()
    for row in manifest['files']:
        target = safe(output, row['path'])
        if valid(target, row):
            print('VERIFIED', row['path'], flush=True)
            continue
        if verify_only or target.exists():
            raise ValueError('Missing or mismatched final model artifact: ' + str(target))
        paths = [fetch(part, output / '.download-parts') for part in row['parts']]
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(target.name + '.assembling')
        digest = hashlib.sha256()
        size = 0
        with pending.open('wb') as destination:
            for path in paths:
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        destination.write(block)
                        digest.update(block)
                        size += len(block)
        if size != row['bytes'] or digest.hexdigest() != row['sha256']:
            raise ValueError('Assembled model checksum mismatch')
        os.replace(pending, target)
        print('INSTALLED', row['path'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'WEIGHTS_MANIFEST.json')
    parser.add_argument('--output', type=Path, default=Path('weights'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    install(json.loads(args.manifest.read_text(encoding='utf-8')), args.output, args.verify_only)
