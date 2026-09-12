"""Conservative column order supported by repeated native text line extents."""

from __future__ import annotations

import statistics
from itertools import pairwise
from operator import itemgetter


def _row_count(lane, height):
    count, anchor = 0, float('-inf')
    for box in sorted(lane.values(), key=itemgetter(1)):
        if box[1] > anchor + height * 0.5:
            count, anchor = count + 1, box[1]
    return count


def order_native_columns(blocks, page_width, metadata):
    diagnostic = _order_global_columns(blocks, page_width, metadata)
    if diagnostic['reason'] == 'SOURCE_GUTTER_CONFIRMED':
        return diagnostic
    local = _order_local_columns(blocks, page_width, metadata)
    return local or diagnostic


def _order_local_columns(blocks, page_width, metadata):
    """Infer columns within whitespace-bounded sections, using source lines.

    A page can have full-width exposition followed by just three rows of
    two-column exercises. Lines outside that section cannot veto its gutter.
    Conversely, the support window does not extend into a footer below it.
    """
    lines = {}
    for block in blocks:
        atom = metadata.get(block['node_id'], {})
        identity = atom.get('source_identity', {})
        box = identity.get('source_line_bbox_pdf_pt')
        if atom.get('origin_type') == 'NATIVE_LINE' and box:
            lines[str(identity['source_line_id'])] = box
    support = {key: box for key, box in lines.items()
               if 0.18 * page_width <= box[2] - box[0] <= 0.70 * page_width}
    if len(support) < 6:
        return None
    # Gaps are local in both x and y. Endpoints from a different section must
    # not subdivide an otherwise clear gutter into unusably narrow intervals.
    gaps = {(left[2], right[0]) for left in support.values() for right in support.values()
            if right[0] - left[2] >= page_width * 0.025
            and min(left[3], right[3]) - max(left[1], right[1])
            >= 0.5 * min(left[3] - left[1], right[3] - right[1])}
    choices = []
    for lo, hi in sorted(gaps):
        cut = (lo + hi) / 2
        if hi - lo < page_width * 0.025 or not page_width * 0.3 < cut < page_width * 0.7:
            continue
        barriers = sorted((box[1], box[3]) for box in [
            *lines.values(), *(b['bbox_pdf_pt'] for b in blocks)
        ] if box[0] < cut < box[2])
        windows, bottom = [], float('-inf')
        for top, end in barriers:
            if top > bottom:
                windows.append((bottom, top))
            bottom = max(bottom, end)
        windows.append((bottom, float('inf')))
        sections = []
        for top, bottom in windows:
            left = {key: box for key, box in support.items()
                    if box[2] <= cut - page_width * 0.0125 and top <= box[1] and box[3] <= bottom}
            right = {key: box for key, box in support.items()
                     if box[0] >= cut + page_width * 0.0125 and top <= box[1] and box[3] <= bottom}
            if min(len(left), len(right)) < 3:
                continue
            common_top = max(min(b[1] for b in left.values()), min(b[1] for b in right.values()))
            common_bottom = min(max(b[3] for b in left.values()), max(b[3] for b in right.values()))
            left = {key: box for key, box in left.items() if box[3] >= common_top and box[1] <= common_bottom}
            right = {key: box for key, box in right.items() if box[3] >= common_top and box[1] <= common_bottom}
            if min(len(left), len(right)) < 3:
                continue
            height = statistics.median(b[3] - b[1] for b in [*left.values(), *right.values()])
            # Split font runs on one baseline must not count as multiple rows.
            counts = [_row_count(left, height), _row_count(right, height)]
            if min(counts) < 3:
                continue
            overlap = min(max(b[3] for b in left.values()), max(b[3] for b in right.values())) - max(
                min(b[1] for b in left.values()), min(b[1] for b in right.values()))
            if overlap < height * 2:
                continue
            section_top = min(b[1] for b in [*left.values(), *right.values()])
            section_bottom = max(b[3] for b in [*left.values(), *right.values()])
            indices = []
            for index, block in enumerate(blocks):
                box = block['bbox_pdf_pt']
                line_id = block.get('provenance', {}).get('inline_native_line_id')
                if (section_top <= (box[1] + box[3]) / 2 <= section_bottom
                        or line_id in left or line_id in right):
                    indices.append(index)
            if any(blocks[i]['bbox_pdf_pt'][0] < cut < blocks[i]['bbox_pdf_pt'][2] for i in indices):
                continue
            sections.append((indices, counts, [section_top, section_bottom]))
        for section in sections:
            choices.append((sum(section[1]), hi - lo, cut, section))
    if not choices:
        return None
    # Separate vertical sections may have different gutters (a margin note
    # beside exposition above a two-column exercise block, for example).
    sections = []
    consumed = set()
    for _, _, cut, section in sorted(choices, key=lambda item: item[:3], reverse=True):
        if consumed.isdisjoint(section[0]):
            sections.append((cut, *section))
            consumed.update(section[0])
    original_ids = [b['node_id'] for b in blocks]
    original_position = {node: index for index, node in enumerate(original_ids)}
    diagnostics = []
    for cut, indices, counts, extent in sections:
        selected = [blocks[i] for i in indices]
        ordered = [b for side in (0, 1) for b in selected
                   if int((b['bbox_pdf_pt'][0] + b['bbox_pdf_pt'][2]) / 2 > cut) == side]
        for index, block in zip(indices, ordered, strict=True):
            blocks[index] = block
            block.setdefault('order_key', {}).update({
                'before_native_column_order_index': original_position[block['node_id']],
                'native_column_index': int((block['bbox_pdf_pt'][0] + block['bbox_pdf_pt'][2]) / 2 > cut),
                'final_page_order_index': index,
            })
        diagnostics.append({'native_support_row_counts': counts, 'vertical_extent_pt': extent,
                            'gutter_center_pt': cut})
    if sorted(b['node_id'] for b in blocks) != sorted(original_ids):
        raise RuntimeError('NATIVE_COLUMN_ORDER_CONSERVATION_FAILED')
    return {'version': 'source-native-column-order-v2',
            'applied': [b['node_id'] for b in blocks] != original_ids,
            'source_atom_conservation': 'PASS', 'reason': 'LOCAL_SOURCE_GUTTER_CONFIRMED',
            'column_count': 2, 'sections': diagnostics}


def _order_global_columns(blocks, page_width, metadata):
    diagnostic = {'version': 'source-native-column-order-v1', 'applied': False,
                  'source_atom_conservation': 'PASS'}
    lines = {}
    for block in blocks:
        atom = metadata.get(block['node_id'], {})
        identity = atom.get('source_identity', {})
        box = identity.get('source_line_bbox_pdf_pt')
        if atom.get('origin_type') != 'NATIVE_LINE' or not box:
            continue
        if 0.18 * page_width <= box[2] - box[0] <= 0.70 * page_width:
            lines[str(identity['source_line_id'])] = box
    if len(lines) < 8:
        return {**diagnostic, 'reason': 'INSUFFICIENT_NATIVE_LINES'}
    endpoints = sorted({box[side] for box in lines.values() for side in (0, 2)})
    candidates = []
    for lo, hi in pairwise(endpoints):
        cut = (lo + hi) / 2
        if hi - lo < page_width * 0.025 or not page_width * 0.3 < cut < page_width * 0.7:
            continue
        left = [box for box in lines.values() if box[2] <= lo]
        right = [box for box in lines.values() if box[0] >= hi]
        if len(left) + len(right) != len(lines) or min(len(left), len(right)) < 4:
            continue
        typical_height = statistics.median(box[3] - box[1] for box in left + right)
        overlap = min(max(b[3] for b in left), max(b[3] for b in right)) - max(
            min(b[1] for b in left), min(b[1] for b in right))
        if overlap < typical_height * 3:
            continue
        candidates.append((hi - lo, cut, typical_height, len(left), len(right)))
    if not candidates:
        return {**diagnostic, 'reason': 'NO_UNAMBIGUOUS_NATIVE_GUTTER'}
    _, cut, typical_height, left_count, right_count = max(candidates)
    spanning = [b for b in blocks if b['bbox_pdf_pt'][0] < cut < b['bbox_pdf_pt'][2]]
    # Wide content divides the page into sections; it must not silently move to
    # one column. Tall overlapping content makes the proposed order ambiguous.
    groups = []
    for block in sorted(spanning, key=lambda b: (b['bbox_pdf_pt'][1], b['node_id'])):
        box = block['bbox_pdf_pt']
        if not groups or box[1] > groups[-1][1]:
            groups.append([box[1], box[3], {block['node_id']}])
        else:
            groups[-1][1] = max(groups[-1][1], box[3])
            groups[-1][2].add(block['node_id'])
    for top, bottom, members in groups:
        for block in blocks:
            if block['node_id'] in members:
                continue
            box = block['bbox_pdf_pt']
            if box[1] < bottom and box[3] > top:
                if bottom - top > typical_height * 2 or box[3] - box[1] > typical_height * 2:
                    return {**diagnostic, 'reason': 'SPANNING_CONTENT_OVERLAPS_COLUMNS'}
                members.add(block['node_id'])
    original_ids = [b['node_id'] for b in blocks]
    original_position = {node_id: index for index, node_id in enumerate(original_ids)}
    pending = list(blocks)
    ordered = []
    assigned_columns = {}

    def append_columns(section):
        for column in (0, 1):
            lane = [b for b in section if int((b['bbox_pdf_pt'][0] + b['bbox_pdf_pt'][2]) / 2 > cut) == column]
            # Preserve the existing, source-backed inline formula/text order.
            ordered.extend(lane)
            assigned_columns.update((b['node_id'], column) for b in lane)

    for top, _bottom, members in groups:
        before = [b for b in pending if b['bbox_pdf_pt'][3] <= top and b['node_id'] not in members]
        append_columns(before)
        consumed = {b['node_id'] for b in before} | members
        ordered.extend(b for b in pending if b['node_id'] in members)
        assigned_columns.update((node_id, None) for node_id in members)
        pending = [b for b in pending if b['node_id'] not in consumed]
    append_columns(pending)
    if sorted(b['node_id'] for b in ordered) != sorted(original_ids):
        raise RuntimeError('NATIVE_COLUMN_ORDER_CONSERVATION_FAILED')
    changed = [b['node_id'] for b in ordered] != original_ids
    if changed:
        blocks[:] = ordered
        for index, block in enumerate(blocks):
            order = block.setdefault('order_key', {})
            order['before_native_column_order_index'] = original_position[block['node_id']]
            order['native_column_index'] = assigned_columns[block['node_id']]
            order['final_page_order_index'] = index
    return {**diagnostic, 'applied': changed, 'reason': 'SOURCE_GUTTER_CONFIRMED',
            'gutter_center_pt': cut, 'column_count': 2,
            'native_support_line_counts': [left_count, right_count],
            'spanning_sections': len(groups)}
