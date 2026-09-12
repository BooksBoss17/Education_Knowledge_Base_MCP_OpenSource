"""Source-pixel continuation of parallel-line diagrams before text masking."""
from __future__ import annotations

import copy
import math

import numpy as np


def _runs(image, left, right, top, bottom, minimum):
    groups = []
    for y in range(top, bottom):
        row = image[y, left:right] < 170
        edges = np.flatnonzero(np.diff(np.r_[False, row, False].astype(np.int8)))
        if len(edges) < 2:
            continue
        spans = [(int(a)+left, int(b)+left) for a, b in zip(edges[::2], edges[1::2]) if b-a >= minimum]
        if not spans:
            continue
        a, b = max(spans, key=lambda p: p[1]-p[0])
        if groups and y-groups[-1]['bottom'] <= 2 and abs(a-groups[-1]['left']) < 5 and abs(b-groups[-1]['right']) < 5:
            groups[-1]['bottom'] = y
        else:
            groups.append(dict(top=y, bottom=y, left=a, right=b))
    return groups


def _edge_label_components(gray, box, lines, anchor_left, anchor_right, pad):
    """Locate complete side-label ink near the extreme supported parallel lines."""
    import cv2
    import math
    x0, _, x1, _ = box
    span = x1-x0
    side_width = max(anchor_left-x0, x1-anchor_right, span*.08)
    distance = max(pad, min(span*.2, side_width*.5))
    thickness = float(np.median([line['bottom']-line['top']+1 for line in lines]))
    ink_padding = max(2, math.ceil(pad*.25), math.ceil(thickness*.5))
    selected, truncated = {}, False
    for line in (min(lines, key=lambda row: row['top']), max(lines, key=lambda row: row['bottom'])):
        center = (line['top']+line['bottom'])/2
        left, right = max(0, math.floor(x0)), min(gray.shape[1], math.ceil(x1))
        top, bottom = max(0, math.floor(center-distance*2)), min(gray.shape[0], math.ceil(center+distance*2))
        _, _, stats, _ = cv2.connectedComponentsWithStats((gray[top:bottom,left:right]<170).astype('uint8'), 8)
        for sx, sy, width, height, area in stats[1:]:
            bounds = [int(sx)+left, int(sy)+top, int(sx+width)+left, int(sy+height)+top]
            cx, cy = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
            if (area < 3 or width > side_width*1.5 or height > distance*2
                    or abs(cy-center) > distance or anchor_left <= cx <= anchor_right):
                continue
            if bounds[0] <= left or bounds[2] >= right or bounds[1] <= top or bounds[3] >= bottom:
                truncated = True
            selected[tuple(bounds)] = {'bbox_render_px': bounds, 'ink_area_px': int(area)}
    return list(selected.values()), ink_padding, truncated


def complete_parallel_diagram(image, box, *, pixels_per_point, blockers=()):
    """Extend only when >=3 core lines support the same external stroke span.

    No text is recognized. A separate formula/table/figure blocks expansion.
    The caller retains the original model box and this derived pixel evidence.
    """
    width, height = image.size
    x0, y0, x1, y1 = box
    span = x1-x0
    if span < 30 or y1-y0 < 15:
        return None
    pad = max(2, int(math.ceil(8*pixels_per_point)))
    search = min((y1-y0)*1.5, height*.18)
    left, right = max(0, int(x0)-pad), min(width, int(math.ceil(x1))+pad)
    top, bottom = max(0, int(y0-search)), min(height, int(math.ceil(y1+search)))
    gray = np.asarray(image.convert('L'))
    lines = _runs(gray, left, right, top, bottom, max(24, span*.45))
    core = [line for line in lines if line['top'] >= y0 and line['bottom'] <= y1]
    tolerance = max(3, span*.08)
    clusters = [[other for other in core if abs(other['left']-line['left']) <= tolerance
                 and abs(other['right']-line['right']) <= tolerance] for line in core]
    cluster = max(clusters, key=len, default=[])
    if len(cluster) < 3:
        return None
    anchor_left = float(np.median([line['left'] for line in cluster]))
    anchor_right = float(np.median([line['right'] for line in cluster]))
    outside = [line for line in lines if (line['bottom'] < y0 or line['top'] > y1)
               and abs(line['left']-anchor_left) <= tolerance
               and abs(line['right']-anchor_right) <= tolerance]
    labels, ink_padding, label_truncated = _edge_label_components(
        gray, box, [*cluster, *outside], anchor_left, anchor_right, pad)
    expanded = [x0, max(0, min([y0, *[line['top']-pad for line in outside],
                               *[row['bbox_render_px'][1]-ink_padding for row in labels]])),
                x1, min(height, max([y1, *[line['bottom']+pad for line in outside],
                                    *[row['bbox_render_px'][3]+ink_padding for row in labels]]))]
    if expanded == list(box) and not label_truncated:
        return None
    added = [[x0, expanded[1], x1, y0], [x0, y1, x1, expanded[3]]]
    conflicts = [list(b) for b in blockers if any(
        min(b[2], a[2]) > max(b[0], a[0]) and min(b[3], a[3]) > max(b[1], a[1]) for a in added)]
    return dict(schema='source-parallel-line-diagram-boundary-v1', original_bbox_render_px=list(box),
        refined_bbox_render_px=expanded, core_parallel_lines=cluster, continuation_lines=outside,
        source_label_components=labels, label_ink_padding_px=ink_padding,
        label_search_truncated=label_truncated,
        status='REVIEW_REQUIRED' if conflicts or label_truncated else 'SOURCE_SUPPORTED_EXTENSION',
        blocking_regions=conflicts, padding_px=pad)


def _connected_diagram_edges(image, box, *, pixels_per_point=1, blockers=()):
    """Complete source components already cut by the detected diagram boundary.

    Dark components seed their connected non-white halo. Disconnected nearby
    prose cannot become a seed, and a bounded search never claims completion
    when a seeded component continues beyond its window.
    """
    import cv2
    margin = max(3, math.ceil(min(box[2]-box[0], box[3]-box[1])*.1))
    left, top = max(0, math.floor(box[0])-margin), max(0, math.floor(box[1])-margin)
    right, bottom = min(image.width, math.ceil(box[2])+margin), min(image.height, math.ceil(box[3])+margin)
    gray = np.asarray(image.crop((left, top, right, bottom)).convert('L'))
    strong = (gray < 170).astype('uint8')
    _, labels, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    x0, y0 = max(0, math.floor(box[0])-left), max(0, math.floor(box[1])-top)
    x1, y1 = min(right-left, math.ceil(box[2])-left), min(bottom-top, math.ceil(box[3])-top)
    touched = set(np.unique(labels[y0:y1, x0:x1])) - {0}
    if not touched:
        return None
    # Scanned paper can be uniformly off-white. Its whole background must not
    # become one ink component. Include a bounded antialias fringe around ink.
    halo_radius = max(2, math.ceil(2*pixels_per_point))
    near_ink = cv2.dilate(strong, np.ones((2*halo_radius+1, 2*halo_radius+1), dtype='uint8'))
    _, halos, halo_stats, _ = cv2.connectedComponentsWithStats(
        ((gray < 255) & (near_ink != 0)).astype('uint8'), 8)
    selected = set()
    for label in touched:
        if stats[label][4] < 3:
            continue
        selected.update(set(np.unique(halos[labels == label])) - {0})
    refined, components, truncated = list(box), [], False
    padding = max(1, math.ceil(pixels_per_point))
    for label in sorted(selected):
        x, y, w, h, area = map(int, halo_stats[label])
        extent = [left+x, top+y, left+x+w, top+y+h]
        if (extent[0] >= box[0] and extent[1] >= box[1]
                and extent[2] <= box[2] and extent[3] <= box[3]):
            continue
        truncated |= ((x == 0 and left > 0) or (y == 0 and top > 0)
                      or (x+w == right-left and right < image.width)
                      or (y+h == bottom-top and bottom < image.height))
        refined = [min(refined[0], max(0, extent[0]-padding)),
                   min(refined[1], max(0, extent[1]-padding)),
                   max(refined[2], min(image.width, extent[2]+padding)),
                   max(refined[3], min(image.height, extent[3]+padding))]
        components.append(dict(bbox_render_px=extent, nonwhite_pixel_count=area))
    if not components:
        return None
    strips = [[refined[0], refined[1], refined[2], box[1]],
              [refined[0], box[3], refined[2], refined[3]],
              [refined[0], box[1], box[0], box[3]],
              [box[2], box[1], refined[2], box[3]]]
    conflicts = [list(b) for b in blockers if any(
        min(b[2], s[2]) > max(b[0], s[0]) and min(b[3], s[3]) > max(b[1], s[1]) for s in strips)]
    return dict(schema='source-connected-diagram-edge-bounds-v1',
                original_bbox_render_px=list(box), refined_bbox_render_px=refined,
                source_connected_components=components, search_window_truncated=bool(truncated),
                halo_seed_distance_limit_px=halo_radius,
                blocking_regions=conflicts, text_recognition_performed=False,
                status='REVIEW_REQUIRED' if truncated or conflicts else 'SOURCE_SUPPORTED_EXTENSION')


def complete_diagram_bounds(image, box, *, pixels_per_point=1, blockers=()):
    parallel = complete_parallel_diagram(image, box, pixels_per_point=pixels_per_point, blockers=blockers)
    next_box = parallel['refined_bbox_render_px'] if parallel else box
    connected = _connected_diagram_edges(image, next_box, pixels_per_point=pixels_per_point, blockers=blockers)
    steps = [p for p in (parallel, connected) if p]
    if not steps:
        return None
    if len(steps) == 1:
        return steps[0]
    return dict(schema='source-diagram-boundary-completion-v2',
                original_bbox_render_px=list(box), refined_bbox_render_px=steps[-1]['refined_bbox_render_px'],
                status='REVIEW_REQUIRED' if any(p['status']=='REVIEW_REQUIRED' for p in steps)
                else 'SOURCE_SUPPORTED_EXTENSION',
                blocking_regions=[b for p in steps for b in p.get('blocking_regions', [])],
                text_recognition_performed=False, steps=steps)


def refine_diagram_bounds(page, image, transform):
    from ..pdf_region_ir import _region_id

    regions = page.get('canonical_regions', [])
    evidence = []
    changed = {}
    original = copy.deepcopy(regions)
    for region in regions:
        if region.get('semantic_type') != 'IMAGE':
            continue
        blockers = [r['bbox_render_px'] for r in original if r['region_id'] != region['region_id']
                    and r.get('semantic_type') in {'IMAGE', 'FORMULA', 'TABLE'}
                    and not (r['bbox_render_px'][0] >= region['bbox_render_px'][0]
                             and r['bbox_render_px'][1] >= region['bbox_render_px'][1]
                             and r['bbox_render_px'][2] <= region['bbox_render_px'][2]
                             and r['bbox_render_px'][3] <= region['bbox_render_px'][3])]
        proof = complete_diagram_bounds(image, region['bbox_render_px'],
            pixels_per_point=transform.render_width/transform.page_width_pt, blockers=blockers)
        if proof is None:
            continue
        old_id = region['region_id']
        region['provenance']['diagram_boundary_refinement'] = proof
        if proof['status'] == 'SOURCE_SUPPORTED_EXTENSION':
            region['bbox_render_px'] = proof['refined_bbox_render_px']
            region['bbox_pdf_pt'] = transform.render_px_to_pdf_pt(region['bbox_render_px'])
            region['bbox_normalized'] = transform.pdf_pt_to_normalized(region['bbox_pdf_pt'])
            region['region_id'] = _region_id(page['document_id'], page['page_index'], region)
            changed[old_id] = region['region_id']
        evidence.append(dict(region_id=region['region_id'], **proof))
    for region in regions:
        if region.get('parent_region_id') in changed:
            region['parent_region_id'] = changed[region['parent_region_id']]
    return evidence
