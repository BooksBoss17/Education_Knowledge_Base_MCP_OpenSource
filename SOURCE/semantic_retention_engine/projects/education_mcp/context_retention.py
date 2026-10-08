"""Positive image retention, label-free prediction and reversible reference selection."""

from __future__ import annotations

import hashlib

import json

from pathlib import Path

import re

import runpy

POLICY = 'context-positive-retention-v2.1'

IMAGE = re.compile(r'!\[(?:\\.|[^\]\\])*\]\((?:<[^>\n]+>|[^\n)])+\)')

DEST = re.compile(r'''\s*(?:<([^>]+)>|(\S+?))(?:\s+(?:"[^"]*"|'[^']*'))?\s*''')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def prepare(package):
    """Only original package data; no standard labels or classified directories."""
    package = Path(package).resolve()
    report = json.loads((package / 'conversion_report.json').read_text('utf-8-sig'))
    text = (package / 'document.md').read_bytes().decode('utf8')
    raw = text.encode('utf8')
    occurrence = runpy.run_path(str(Path(__file__).with_name('image_occurrences.py')))
    mapping = occurrence['resolve'](package, text, report['source']['sha256'], IMAGE, DEST)
    if mapping['status'] != 'VERIFIED':
        raise ValueError('Unavailable occurrence mapping: ' + str(mapping))
    spans = report['markdown_render']['node_spans']
    by_page = {}
    for s in spans:
        by_page.setdefault(s['page_index'], []).append(s)
    assets = {}
    for i, (start, end, _, _, name) in enumerate(occurrence['reference_tokens'](text, IMAGE, DEST)):
        origin = mapping['by_start'][start]
        page = int(origin['source_part'].split(':')[1])
        bbox = origin['bbox_pdf_pt']
        anchor = next(s for s in spans if s['node_id'] == origin['source_document_node_id'])
        # Rank by renderer distance, then restore reading order; never protect a whole page.
        nearby = sorted(by_page[page], key=lambda s: abs(s['byte_start'] - anchor['byte_start']))[:7]
        nearby.sort(key=lambda s: s['byte_start'])
        context = '\n'.join(IMAGE.sub('[其他图片]', raw[s['byte_start']:s['byte_end']].decode('utf8'))
                            for s in nearby if s['node_id'] != anchor['node_id'])[:1200]
        item = assets.setdefault(name, dict(asset_name=name, path=str(package / name),
                                           asset_sha256=sha(package / name), occurrences=[]))
        item['occurrences'].append(dict(id=f'o{i+1:06d}', page=page, bbox=bbox,
                                       source_kind=origin['source_document_kind'],
                                       byte_start=len(text[:start].encode('utf8')),
                                       byte_end=len(text[:end].encode('utf8')), context=context))
    return dict(policy=POLICY, package=str(package), source=report['source'],
                markdown_sha256=hashlib.sha256(raw).hexdigest(), assets=list(assets.values()))
