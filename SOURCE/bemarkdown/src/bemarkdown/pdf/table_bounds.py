"""Close a detected ruled table over its connected source grid and edge halo."""
from __future__ import annotations
import copy
import math


def _intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def complete_table_grid(image, box, *, blockers=()):
    """Extend only through an observed grid's connected non-white pixels.

    Plain nearby text cannot grow the box unless it is physically connected
    to the grid; independently detected outside content blocks that extension.
    Search-window contact is an uncertainty, never a license to crop farther.
    """
    import cv2
    import numpy as np
    width, height = box[2]-box[0], box[3]-box[1]
    if min(width, height) < 16:
        return None
    margin = max(3, math.ceil(min(width, height)*.1))
    left, top = max(0, math.floor(box[0])-margin), max(0, math.floor(box[1])-margin)
    right, bottom = min(image.width, math.ceil(box[2])+margin), min(image.height, math.ceil(box[3])+margin)
    gray = np.asarray(image.crop((left, top, right, bottom)).convert('L'))
    strong = (gray < 160).astype('uint8')
    _, labels, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    candidates = [(i, s) for i, s in enumerate(stats[1:], 1)
                  if s[2] >= width*.8 and s[3] >= height*.8]
    if not candidates:
        return None
    component_id, component = max(candidates, key=lambda item: item[1][4])
    core = labels == component_id
    def bands(values):
        return int(np.count_nonzero(np.diff(np.r_[False, values, False].astype('int8')) == 1))
    horizontal = bands(core.sum(axis=1) >= width*.45)
    vertical = bands(core.sum(axis=0) >= height*.45)
    if min(horizontal, vertical) < 3:
        return None
    _, halo_labels, halo_stats, _ = cv2.connectedComponentsWithStats((gray < 255).astype('uint8'), connectivity=8)
    halo_id = int(np.bincount(halo_labels[core]).argmax())
    x, y, w, h, area = [int(v) for v in halo_stats[halo_id]]
    extent = [left+x, top+y, left+x+w, top+y+h]
    refined = [min(box[0], extent[0]), min(box[1], extent[1]), max(box[2], extent[2]), max(box[3], extent[3])]
    if refined == list(box):
        return None
    truncated = ((x == 0 and left > 0) or (y == 0 and top > 0)
        or (x+w == right-left and right < image.width) or (y+h == bottom-top and bottom < image.height))
    added = [[refined[0], refined[1], refined[2], box[1]],
             [refined[0], box[3], refined[2], refined[3]],
             [refined[0], box[1], box[0], box[3]],
             [box[2], box[1], refined[2], box[3]]]
    conflicts = [list(b) for b in blockers if any(_intersects(b, strip) for strip in added)]
    return {'schema': 'source-connected-table-grid-bounds-v1',
            'status': 'REVIEW_REQUIRED' if truncated or conflicts else 'SOURCE_SUPPORTED_EXTENSION',
            'original_bbox_render_px': list(box), 'refined_bbox_render_px': refined,
            'grid_bbox_render_px': [left+int(component[0]), top+int(component[1]),
                                   left+int(component[0]+component[2]), top+int(component[1]+component[3])],
            'horizontal_rule_bands': horizontal, 'vertical_rule_bands': vertical,
            'connected_halo_pixel_count': area, 'search_window_truncated': truncated,
            'blocking_regions': conflicts, 'text_recognition_performed': False}


def refine_table_bounds(page, image, transform):
    from ..pdf_region_ir import _region_id
    regions = page.get('canonical_regions', [])
    original = copy.deepcopy(regions)
    evidence, changed = [], {}
    for region in regions:
        if region['semantic_type'] != 'TABLE':
            continue
        box = region['bbox_render_px']
        blockers = [r['bbox_render_px'] for r in original if r['region_id'] != region['region_id']
            and not (box[0] <= r['bbox_render_px'][0] and box[1] <= r['bbox_render_px'][1]
                     and box[2] >= r['bbox_render_px'][2] and box[3] >= r['bbox_render_px'][3])]
        proof = complete_table_grid(image, box, blockers=blockers)
        if proof is None:
            continue
        old_id = region['region_id']
        region['provenance']['table_boundary_refinement'] = proof
        if proof['status'] == 'SOURCE_SUPPORTED_EXTENSION':
            region['bbox_render_px'] = proof['refined_bbox_render_px']
            region['bbox_pdf_pt'] = transform.render_px_to_pdf_pt(region['bbox_render_px'])
            region['bbox_normalized'] = transform.pdf_pt_to_normalized(region['bbox_pdf_pt'])
            region['region_id'] = _region_id(page['document_id'], page['page_index'], region)
            changed[old_id] = region['region_id']
        evidence.append({'region_id': region['region_id'], **proof})
    for region in regions:
        if region.get('parent_region_id') in changed:
            region['parent_region_id'] = changed[region['parent_region_id']]
    return evidence
