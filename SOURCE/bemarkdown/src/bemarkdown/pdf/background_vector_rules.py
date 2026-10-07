"""Match reviewed heading backdrops by source vector commands, not image similarity."""
from __future__ import annotations
import hashlib
import json

SCHEMA = 'source-heading-backdrop-template-v1'


def _contains(outer, inner, tolerance=0.002):
    return (outer[0]-tolerance <= inner[0] and outer[1]-tolerance <= inner[1]
            and outer[2]+tolerance >= inner[2] and outer[3]+tolerance >= inner[3])


def heading_text(coverage):
    if coverage.get('status') != 'COVERED' or coverage.get('scope') != 'OMITTED_NATIVE_CHARACTERS_ONLY':
        return None
    chars = coverage.get('characters', [])
    if not chars or any(c.get('status') != 'COVERED' for c in chars):
        return None
    value = ''.join(c.get('original_text') or '' for c in chars)
    value = ''.join(value.split())
    return value if 1 <= len(value) <= 16 else None


def source_vector_signature(page, bbox, *, excluded_regions=(), raster_contribution_absent=False):
    """Small translated heading fills only; extra strokes/images/paths veto.

    Translation is normalized, scale and shape are retained in PDF points.
    0.001pt rounding absorbs PDF float representation, not pixel shape changes.
    Caller separately proves delivery of every excluded semantic owner.
    """
    import pymupdf as f
    clip = f.Rect(bbox)
    if page.rotation or clip.is_empty or clip.is_infinite:
        return {'status': 'UNSUPPORTED_GEOMETRY'}
    selected = []
    for image in page.get_image_info():
        overlap = f.Rect(image['bbox']) & clip
        if (not raster_contribution_absent and not overlap.is_empty
                and not any(_contains(region, overlap) for region in excluded_regions)):
            return {'status': 'UNOWNED_RASTER_PIXELS'}
    contexts = []
    for drawing in page.get_drawings(extended=True):
        level = drawing.get('level', 0)
        contexts = [ctx for ctx in contexts if ctx.get('level', 0) < level]
        if drawing['type'] in ('clip', 'group'):
            contexts.append(drawing)
            continue
        rect = drawing.get('rect', drawing.get('scissor'))
        if rect is None or (f.Rect(rect) & clip).is_empty:
            continue
        if any(_contains(region, rect) for region in excluded_regions):
            continue
        for ctx in contexts:
            if ctx['type'] == 'group':
                if ctx.get('blendmode') != 'Normal' or ctx.get('opacity') != 1 or ctx.get('knockout'):
                    return {'status': 'GRAPHICS_GROUP_REQUIRES_REVIEW'}
            else:
                items = ctx.get('items', [])
                if (len(items) != 1 or items[0][0] != 're' or not _contains(items[0][1], clip)):
                    return {'status': 'NONTRIVIAL_GRAPHICS_CLIP'}
        if not _contains(clip, rect):
            return {'status': 'VECTOR_CROSSES_CROP'}
        if drawing['type'] != 'f' or drawing.get('color') is not None:
            return {'status': 'NON_FILL_VECTOR_CONTENT'}
        if not drawing.get('fill') or not 0 < drawing.get('fill_opacity', 0) <= 1:
            return {'status': 'UNSUPPORTED_FILL'}
        selected.append(drawing)
    if not 1 <= len(selected) <= 8:
        return {'status': 'NOT_A_SMALL_HEADING_BACKDROP'}
    left = min(d['rect'][0] for d in selected)
    top = min(d['rect'][1] for d in selected)
    def point(p):
        return [round(p[0]-left, 3), round(p[1]-top, 3)]
    paths = []
    for drawing in selected:
        items = []
        for item in drawing['items']:
            if item[0] == 're':
                box = item[1]
                items.append(['re', point((box[0], box[1])), point((box[2], box[3])), item[2]])
            elif item[0] in ('l', 'c'):
                items.append([item[0], *[point(p) for p in item[1:]]])
            else:
                return {'status': 'UNSUPPORTED_VECTOR_COMMAND'}
        paths.append({'items': items, 'fill': list(drawing['fill']),
                      'fill_opacity': drawing['fill_opacity'], 'even_odd': drawing.get('even_odd', False),
                      'close_path': drawing.get('closePath', False)})
    canonical = json.dumps(paths, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'status': 'SOURCE_VECTOR_SIGNATURE', 'schema': SCHEMA,
            'signature_sha256': hashlib.sha256(canonical.encode()).hexdigest(), 'paths': paths}


def match_template(signature, coverage, templates, source_sha256):
    title = heading_text(coverage)
    if signature.get('status') != 'SOURCE_VECTOR_SIGNATURE' or title is None:
        return None
    matches = [t for t in templates if t.get('schema') == SCHEMA
               and t.get('source_sha256') == source_sha256
               and t.get('classification') == 'REVIEWED_DECORATIVE_HEADING_BACKDROP'
               and t.get('heading_text') == title
               and t.get('signature_sha256') == signature['signature_sha256']]
    return matches[0] if len(matches) == 1 else None
