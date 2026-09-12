"""Keep a side figure's uniquely associated caption beside its image."""
from __future__ import annotations

from collections import defaultdict
from itertools import pairwise


def order_parallel_captioned_figures(blocks):
    """Use an aligned caption row to order adjacent, vertically overlapping figures."""
    by_id = {block['node_id']: block for block in blocks}
    owners = defaultdict(list)
    for block in blocks:
        if block['kind'] == 'CAPTION':
            owners[block.get('relations', {}).get('caption_for')].append(block)
    pairs = []
    for owner_id, captions in owners.items():
        if len(captions) != 1 or owner_id not in by_id:
            continue
        owner, caption = by_id[owner_id], captions[0]
        ib, cb = owner.get('bbox_pdf_pt'), caption.get('bbox_pdf_pt')
        if (owner['kind'] == 'IMAGE' and ib and cb and cb[1] >= ib[3]
                and caption['relations'].get('caption_confidence') == 'HIGH'):
            pairs.append((owner, caption))
    bands = []
    for pair in sorted(pairs, key=lambda pair: pair[1]['bbox_pdf_pt'][1]):
        cb = pair[1]['bbox_pdf_pt']
        if bands:
            anchor = bands[-1][0][1]['bbox_pdf_pt']
            tolerance = min(cb[3] - cb[1], anchor[3] - anchor[1]) * 0.3
        if bands and cb[1] - anchor[1] <= tolerance:
            bands[-1].append(pair)
        else:
            bands.append([pair])
    moves = []
    for band in bands:
        if len(band) < 2:
            continue
        ordered = sorted(band, key=lambda pair: pair[1]['bbox_pdf_pt'][0])
        images = [pair[0] for pair in ordered]
        boxes = [item['bbox_pdf_pt'] for item in images]
        if any(a[2] > b[0] for a, b in pairwise(boxes)):
            continue
        common_height = min(b[3] for b in boxes) - max(b[1] for b in boxes)
        if common_height < min(b[3] - b[1] for b in boxes) * 0.5:
            continue
        positions = sorted(blocks.index(item) for item in images)
        if positions[-1] - positions[0] + 1 != len(positions):
            continue  # Never move an image across an intervening text/other block.
        before = [blocks[index]['node_id'] for index in positions]
        after = [item['node_id'] for item in images]
        if before != after:
            for index, item in zip(positions, images, strict=True):
                blocks[index] = item
            moves.append({'before': before, 'after': after,
                          'basis': 'HIGH_CONFIDENCE_ALIGNED_CAPTIONS_AND_OVERLAPPING_IMAGES'})
    if moves:
        for index, block in enumerate(blocks):
            block.setdefault('order_key', {})['final_page_order_index'] = index
    return {'version': 'parallel-captioned-figure-order-v1',
            'moved_count': sum(sum(a != b for a, b in zip(m['before'], m['after'])) for m in moves),
            'groups': moves}


def attach_side_captions(blocks):
    by_id = {block['node_id']: block for block in blocks}
    captions = defaultdict(list)
    for block in blocks:
        if block['kind'] == 'CAPTION':
            captions[block.get('relations', {}).get('caption_for')].append(block)
    moves = []
    for owner_id, candidates in captions.items():
        if len(candidates) != 1 or owner_id not in by_id:
            continue
        owner, caption = by_id[owner_id], candidates[0]
        if owner['kind'] != 'IMAGE' or caption['relations'].get('caption_confidence') != 'HIGH':
            continue
        image_box, caption_box = owner.get('bbox_pdf_pt'), caption.get('bbox_pdf_pt')
        if not image_box or not caption_box or caption_box[1] < image_box[3]:
            continue
        owner_at, caption_at = blocks.index(owner), blocks.index(caption)
        if caption_at <= owner_at + 1:
            continue
        lane_left, lane_right = min(image_box[0], caption_box[0]), max(image_box[2], caption_box[2])
        between = blocks[owner_at+1:caption_at]
        # Cross only prose in a separate horizontal lane. Same-lane content,
        # another illustration/formula, and ambiguous caption groups are fences.
        if any(block['kind'] != 'TEXT' or not block.get('bbox_pdf_pt') or
               min(lane_right, block['bbox_pdf_pt'][2]) > max(lane_left, block['bbox_pdf_pt'][0])
               for block in between):
            continue
        proof = {'version': 'side-figure-caption-attachment-v1', 'figure_node_id': owner_id,
                 'caption_node_id': caption['node_id'],
                 'crossed_text_node_ids': [block['node_id'] for block in between],
                 'basis': 'HIGH_CONFIDENCE_CAPTION_BELOW_FIGURE_AND_DISJOINT_TEXT_LANE'}
        blocks.pop(caption_at)
        blocks.insert(owner_at+1, caption)
        caption.setdefault('provenance', {})['side_caption_attachment'] = proof
        moves.append(proof)
    if moves:
        for index, block in enumerate(blocks):
            block.setdefault('order_key', {})['final_page_order_index'] = index
    return {'version': 'side-figure-caption-attachment-v1', 'moved_count': len(moves), 'moves': moves}
