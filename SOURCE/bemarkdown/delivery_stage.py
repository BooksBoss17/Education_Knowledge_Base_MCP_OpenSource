"""Artifact completion is deliberately separate from downstream acceptance."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable


def artifact_delivery_stage(root: Path, names: Iterable[str], *, source_sha256: str,
                            scope: str) -> dict:
    """Bind actual delivered bytes without inventing a reviewer receipt.

    Downstream review receipts must remain separate and bind these hashes.
    An audit PASS, a valid package, or an empty review queue is not a receipt.
    """
    if scope not in {'CONVERTER_ARTIFACTS', 'HANDOFF_ARTIFACTS'}:
        raise ValueError('Unsupported delivery completion scope')
    if not isinstance(source_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', source_sha256):
        raise ValueError('Delivery evidence requires a source SHA-256')
    root = root.resolve()
    hashes = {}
    for name in sorted(set(names)):
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f'Missing or unsafe delivery artifact: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return {
        'schema': 'bemarkdown-delivery-stage-v1',
        'complete_scope': scope,
        'source_sha256': source_sha256,
        'artifact_sha256': hashes,
        'agent_review': 'NOT_RECORDED',
        'content_acceptance': 'NOT_RECORDED',
    }
