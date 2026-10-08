"""Label-blind, original-byte-bound MD context variant and fixed diagnostic pilot."""

from __future__ import annotations

import hashlib

import re

IMAGE = re.compile(rb'!\[[^\]\r\n]*\]\([^\)\r\n]*\)')

def context_view(raw, use, proof, max_chars=2400):
    """Build one full available same-page MD text window; never guess page or image."""
    occurrence = proof['occurrence']
    assert occurrence['id'] == use['use_id'] and occurrence['page'] == use['page_index']
    if use['context_status'] != 'verified':
        return None, {'status': 'KEEP_ORIGINAL_EMPTY_OR_MISSING_CONTEXT'}
    a, b = occurrence['byte_start'], occurrence['byte_end']
    match = IMAGE.fullmatch(raw[a:b])
    if match is None:
        return None, {'status': 'NO_EXACT_IMAGE_ANCHOR', 'anchor': [a, b]}
    intervals = [(s['start_byte'], s['end_byte']) for s in use['spans']]
    intervals += [(x['start_byte'], x['end_byte']) for x in proof['omitted_md_ranges']]
    for span in use['spans']:
        actual = raw[span['start_byte']:span['end_byte']].decode('utf-8')
        assert actual == span['text'] and hashlib.sha256(actual.encode()).hexdigest() == span['text_sha256']
    lo = min([a] + [x[0] for x in intervals])
    hi = max([b] + [x[1] for x in intervals])
    window = raw[lo:hi]
    window.decode('utf-8')
    pieces = []
    mappings = []
    cursor = 0
    target_count = 0
    for image in IMAGE.finditer(window):
        pieces.append(window[cursor:image.start()])
        target = lo + image.start() == a and lo + image.end() == b
        target_count += target
        marker = b'[TARGET_IMAGE]' if target else b'[OTHER_IMAGE]'
        pieces.append(marker)
        mappings.append({'source_start': lo + image.start(), 'source_end': lo + image.end(), 'replacement': marker.decode()})
        cursor = image.end()
    pieces.append(window[cursor:])
    text = b''.join(pieces).decode('utf-8')
    assert target_count == 1
    if len(text) > max_chars:
        return None, {'status': 'FULL_PAGE_WINDOW_OVER_BUDGET_KEEP_BASELINE', 'characters': len(text), 'source_window': [lo, hi]}
    return text, {'status': 'FULL_AVAILABLE_PAGE_TEXT_ANCHORED', 'source_window': [lo, hi], 'source_window_sha256': hashlib.sha256(window).hexdigest(),
                  'source_target': [a, b], 'marker_mappings': mappings, 'characters': len(text),
                  'original_text_retained_verbatim': True, 'page_scope_evidence': 'existing verified MD node byte spans and omitted ranges; no PDF page redraw'}
