"""Verified, resumable downloads. Downloaded bytes are never executed here."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.request

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def safe_path(root, relative):
    root = Path(root).resolve()
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts or ':' in relative:
        raise ValueError('Unsafe relative asset path')
    target = (root / part).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Asset path escapes its destination')
    return target

def valid(path, digest, size):
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest

def download(url, digest, size, cache, workers=8):
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / digest
    if valid(target, digest, size):
        return target
    if not url.startswith('https://'):
        raise ValueError('Only HTTPS download sources are supported')
    block = 8 * 1024 * 1024
    if size <= block:
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as response, target.with_suffix('.part').open('wb') as stream:
                    shutil.copyfileobj(response, stream, 1024 * 1024)
                if not valid(target.with_suffix('.part'), digest, size):
                    raise ValueError('Downloaded file does not match its pinned hash')
                os.replace(target.with_suffix('.part'), target)
                return target
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
    chunks = cache / (digest + '.chunks')
    chunks.mkdir(exist_ok=True)
    ranges = [(start, min(start + block, size) - 1) for start in range(0, size, block)]
    def fetch(bounds):
        start, end = bounds
        part = chunks / str(start)
        if part.is_file() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(5):
            try:
                separator = '&' if '?' in url else '?'
                request = urllib.request.Request(url + f'{separator}ekb_range={start}&retry={attempt}', headers={'Range': f'bytes={start}-{end}'})
                began = time.monotonic()
                with urllib.request.urlopen(request, timeout=45) as response, part.with_suffix('.part').open('wb') as stream:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError('Server returned an unexpected byte range')
                    while data := response.read(65536):
                        if time.monotonic() - began > 180:
                            raise TimeoutError('Chunk transfer deadline exceeded')
                        stream.write(data)
                if part.with_suffix('.part').stat().st_size != end - start + 1:
                    raise ValueError('Incomplete byte range')
                os.replace(part.with_suffix('.part'), part)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, bounds) for bounds in ranges]
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append(type(exc).__name__)
            if count % 32 == 0 or count == len(ranges):
                print(f'  {count}/{len(ranges)} chunks checked', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} download chunks failed. Rerun setup to resume.')
    temporary = target.with_suffix('.part')
    with temporary.open('wb') as stream:
        for start, _ in ranges:
            with (chunks / str(start)).open('rb') as source:
                shutil.copyfileobj(source, stream, 1024 * 1024)
    if not valid(temporary, digest, size):
        raise ValueError(f'Assembled download hash mismatch. Remove only the affected cache directory {chunks} and retry.')
    os.replace(temporary, target)
    return target
