"""Read unambiguous labels in trusted vector figures from visible PDF text."""
from __future__ import annotations

import hashlib
import re
import unicodedata

from .native_math import native_flat_latex


def _overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _inside(outer, inner, tolerance=0.5):
    return (outer[0] - tolerance <= inner[0] <= inner[2] <= outer[2] + tolerance
            and outer[1] - tolerance <= inner[1] <= inner[3] <= outer[3] + tolerance)


def _owns_text(figure, box):
    # PDF font boxes include ascender/side-bearing space beyond visible ink.
    # Require the center and at least 80% of each box dimension in the figure;
    # this does not truncate or invent characters from a partly shared line.
    return (_inside(figure, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                            (box[0] + box[2]) / 2, (box[1] + box[3]) / 2], tolerance=0)
            and min(figure[2], box[2]) - max(figure[0], box[0]) >= (box[2] - box[0]) * 0.8
            and min(figure[3], box[3]) - max(figure[1], box[1]) >= (box[3] - box[1]) * 0.8)


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _text(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value))


def native_figure_labels(parent, state):
    """Return source-bound labels, or None when this entire figure needs OCR.

    Raster overlays, partial text, cross-line scripts and vector mathematics
    remain on the existing OCR path. Graph strokes are never decoded as text.
    """
    plan = state['plan']
    provenance = plan.get('provenance', {})
    if provenance.get('native_text_trust') != 'HIGH' or provenance.get('page_source_profile') not in {
        'NATIVE_TEXT', 'MIXED_NATIVE_VISUAL',
    }:
        return None
    source = state['source_evidence']
    figure = parent['provenance']['bbox_pdf_pt']
    if any(_overlap(figure, image['bbox_pdf_pt']) for image in source.get('native_images', [])):
        return None
    groups = []
    for line in source.get('native_text', []):
        selected = []
        for span in line.get('spans', []):
            box = span.get('ink_bbox_pdf_pt') or span['bbox_pdf_pt']
            if not _overlap(figure, box):
                continue
            if (not _owns_text(figure, box) or '\ufffd' in span.get('text', '')
                    or span.get('unicode_mapping_issue')):
                return None
            selected.append((span, box))
        for span, box in sorted(selected, key=lambda pair: pair[1][0]):
            if (groups and groups[-1]['line_id'] == line['evidence_id']
                    and box[0] - groups[-1]['bbox'][2] <= (box[3] - box[1]) * 0.8):
                group = groups[-1]
                group['spans'].append(span)
                group['bbox'] = _union([group['bbox'], box])
            else:
                groups.append({'line_id': line['evidence_id'], 'spans': [span], 'bbox': list(box)})
    if not groups:
        return None
    # Close vertical fragments can be a fraction or a script, not two labels.
    for index, group in enumerate(groups):
        a = group['bbox']
        for other in groups[index + 1:]:
            b = other['bbox']
            if min(a[2], b[2]) > max(a[0], b[0]):
                gap = max(a[1], b[1]) - min(a[3], b[3])
                if gap < min(a[3] - a[1], b[3] - b[1]) * 0.5:
                    return None
    geometries = [g for p in source.get('native_math_geometry', []) for g in p.get('evidence', [])]
    labels = []
    for group in sorted(groups, key=lambda g: (g['bbox'][1], g['bbox'][0])):
        value = ''.join(span['text'] for span in group['spans']).strip()
        if not value:
            return None
        box = group['bbox']
        glyphs = {}
        for glyph in geometries:
            if not _overlap(box, glyph['bbox']):
                continue
            if glyph['kind'] != 'NATIVE_CHARACTER':
                # A source-drawn bar or symbol may change the expression.
                if _inside(box, glyph['bbox']):
                    return None
                continue
            if _inside(box, glyph['bbox']):
                glyphs[(tuple(glyph['bbox']), glyph['text'])] = glyph
        latex = None
        if len(_text(value)) > 1 and re.search(r'[A-Za-z0-9\u0370-\u03ff]', value):
            ordered = sorted(glyphs.values(), key=lambda g: (g['bbox'][0], g['origin_pdf_pt'][1]))
            if _text(''.join(g['text'] for g in ordered)) != _text(value):
                return None
            expression = native_flat_latex(ordered)
            if expression is None:
                return None
            if any(token in expression for token in ('\\', '_', '^')):
                latex = expression
        identity = f"{plan['document_id']}:{parent['route_id']}:{','.join(s['span_id'] for s in group['spans'])}"
        labels.append({'id': 'native-figure-label-' + hashlib.sha256(identity.encode()).hexdigest()[:24],
            'text': value, 'latex': latex, 'bbox_pdf_pt': box, 'status': 'NATIVE_SOURCE_TEXT',
            'review_reasons': [], 'source_span_ids': [s['span_id'] for s in group['spans']]})
    return {'schema': 'bemarkdown-figure-label-recognition-v1', 'status': 'NATIVE_SOURCE_TEXT',
            'method': 'visible-native-vector-figure-text-v1', 'parent_route_id': str(parent['route_id']),
            'source_bbox_pdf_pt': list(figure), 'labels': labels, 'three_model_text_evidence': None,
            'source_profile': provenance['page_source_profile'], 'native_text_trust': 'HIGH'}
