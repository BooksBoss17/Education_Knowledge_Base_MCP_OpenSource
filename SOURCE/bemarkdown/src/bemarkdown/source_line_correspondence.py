"""Gold-free proposals from whole source-line sequence to whole parent output.

All optimal global edit paths are considered at every line boundary. There is
no target text input or best-substring search. A unique proposal is still not
source geometry certification; consumers retain that distinction.
"""
from array import array
import hashlib
import json

VERSION = 'global-source-line-correspondence-v1'
MAX_CELLS = 2_000_000


def _compact(text):
    chars = []
    positions = []
    for index, char in enumerate(text):
        if not char.isspace():
            chars.append(char)
            positions.append(index)
    return ''.join(chars), positions


def _matrix(first, second):
    width = len(second) + 1
    rows = [array('H', range(width))]
    for i, a in enumerate(first, 1):
        row = array('H', [i])
        prior = rows[-1]
        for j, b in enumerate(second, 1):
            row.append(min(prior[j] + 1, row[j - 1] + 1, prior[j - 1] + (a != b)))
        rows.append(row)
    return rows


def propose_source_line_correspondence(lines, output, *, source_binding, max_cells=MAX_CELLS):
    # Preserve detector order; caller must provide actual source reading order.
    texts = []
    bounds = [0]
    for row in lines:
        value, _ = _compact(str(row.get('text', '')))
        texts.append(value)
        bounds.append(bounds[-1] + len(value))
    source = ''.join(texts)
    target, positions = _compact(output)
    header = {'version': VERSION, 'source_binding': source_binding,
        'source_sequence_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
        'whole_output_sha256': hashlib.sha256(output.encode('utf-8')).hexdigest(),
        'source_characters': len(source), 'output_characters': len(target),
        'gold_read': False, 'whole_parent_alignment': True,
        'geometry_certified': False, 'automatic_scoring_authorized': False}
    if not source or not target:
        return {**header, 'status': 'NO_SOURCE_OR_OUTPUT_TEXT', 'lines': []}
    if (len(source) + 1) * (len(target) + 1) > max_cells:
        return {**header, 'status': 'BOUNDED_ALIGNMENT_CAP_EXCEEDED', 'lines': []}
    forward = _matrix(source, target)
    reverse = _matrix(source[::-1], target[::-1])
    distance = forward[-1][-1]
    endpoints = {}
    for offset in set(bounds):
        endpoints[offset] = [j for j in range(len(target) + 1)
            if forward[offset][j] + reverse[len(source) - offset][len(target) - j] == distance]
    proposals = []
    for i, row in enumerate(lines):
        start, end = bounds[i], bounds[i + 1]
        starts, ends = endpoints[start], endpoints[end]
        record = {'source_reading_sequence': row.get('reading_sequence', i),
            'source_line_text_sha256': hashlib.sha256(str(row.get('text', '')).encode('utf-8')).hexdigest(),
            'bbox_page_pdf_pt': row.get('bbox_page_pdf_pt'),
            'bbox_local_px': row.get('bbox_local_px'),
            'source_compact_char_range': [start, end],
            'optimal_start_positions': starts, 'optimal_end_positions': ends,
            'confidence': row.get('rec_confidence', row.get('confidence')),
            'geometry_certified': False}
        if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
            record['status'] = 'AMBIGUOUS_GLOBAL_BOUNDARY'
        else:
            a, b = starts[0], ends[0]
            x, y = positions[a], positions[b - 1] + 1
            exact = source[start:end] == target[a:b]
            record.update(status='UNIQUE_GLOBAL_BOUNDARY_PROPOSAL',
                output_compact_char_range=[a, b], output_char_range=[x, y],
                output_utf8_byte_range=[len(output[:x].encode('utf-8')), len(output[:y].encode('utf-8'))],
                output_text_sha256=hashlib.sha256(output[x:y].encode('utf-8')).hexdigest(),
                source_prediction_unchanged=exact,
                contains_math_serialization=any(c in output[x:y] for c in ('$', '\\', '^', '_')),
                independent_scope_review_required=True)
        proposals.append(record)
    return {**header, 'status': 'GLOBAL_BOUNDARY_PROPOSALS_FROZEN_NOT_CERTIFIED',
        'global_edit_distance': distance,
        'global_edit_rate_diagnostic': distance / max(1, len(source), len(target)),
        'lines': proposals}
