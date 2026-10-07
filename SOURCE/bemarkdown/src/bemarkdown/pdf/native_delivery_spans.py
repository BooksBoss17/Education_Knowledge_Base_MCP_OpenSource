"""Replay exact reviews over renderer byte spans; edited nodes lose coverage proof."""
from __future__ import annotations
import copy
import hashlib


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def unchanged_delivery_spans(original, current, rendering, journal):
    raw = original.encode('utf-8')
    if rendering.get('offset_unit') != 'UTF8_BYTES' or rendering.get('determinism_sha256') != _sha(raw):
        raise ValueError('RENDERER_MARKDOWN_IDENTITY_MISMATCH')
    spans = copy.deepcopy(rendering.get('node_spans', []))
    seen, end = set(), 0
    for span in spans:
        left, right = span.get('byte_start'), span.get('byte_end')
        if (type(left) is not int or type(right) is not int or not end <= left < right <= len(raw)
                or not span.get('node_id') or span['node_id'] in seen):
            raise ValueError('INVALID_RENDERER_NODE_SPAN')
        if _sha(raw[left:right]) != span.get('sha256'):
            raise ValueError('RENDERER_NODE_BYTES_MISMATCH')
        end = right
        seen.add(span['node_id'])
    if journal.get('original_sha256') != _sha(raw):
        raise ValueError('REVIEW_ORIGINAL_IDENTITY_MISMATCH')
    for revision in journal.get('revisions', []):
        if revision.get('base_sha256') != _sha(raw):
            raise ValueError('REVIEW_BASE_IDENTITY_MISMATCH')
        for replacement in revision.get('replacements', []):
            old, new = replacement['old'].encode('utf-8'), replacement['new'].encode('utf-8')
            if not old or raw.count(old) != 1:
                raise ValueError('REVIEW_REPLACEMENT_NOT_UNIQUE')
            left = raw.index(old)
            right = left + len(old)
            delta = len(new) - len(old)
            following = []
            for span in spans:
                if span['byte_end'] <= left:
                    following.append(span)
                elif span['byte_start'] >= right:
                    span['byte_start'] += delta
                    span['byte_end'] += delta
                    following.append(span)
                # Any intersecting edit loses the original node's coverage claim.
            spans = following
            raw = raw[:left] + new + raw[right:]
        if revision.get('result_sha256') != _sha(raw):
            raise ValueError('REVIEW_RESULT_IDENTITY_MISMATCH')
    if raw != current.encode('utf-8'):
        raise ValueError('CURRENT_MARKDOWN_REPLAY_MISMATCH')
    result = {}
    for span in spans:
        body = raw[span['byte_start']:span['byte_end']]
        if _sha(body) != span['sha256']:
            raise ValueError('CURRENT_NODE_BYTES_MISMATCH')
        result[span['node_id']] = {**span, 'text': body.decode('utf-8'),
                                   'basis': 'RENDERER_BYTES_AND_EXACT_REVIEW_REPLAY',
                                   'current_markdown_sha256': _sha(raw)}
    return result
