"""Complete glyphs cut by an inline formula's layout rectangle.

Only connected source ink that straddles the original rectangle can enlarge it.
Detached neighbouring text is never recruited, and components reaching the
bounded context edge are left alone. This does not infer or rewrite LaTeX.
"""
from __future__ import annotations

from typing import Any
import re


def has_terminal_formula_operator(latex: str) -> bool:
    """A missing right operand is a bounded-crop retry signal, not a correction."""
    value = latex.strip()
    return bool(re.search(
        r'(?:[=+<>^_]|(?<!\\)-|\\(?:to|rightarrow|leftarrow|leftrightarrow|'
        r'times|cdot|div|pm|mp|leq?|geq?|neq?|approx))\s*$', value))


def complete_formula_ink_bounds(image, crop: dict[str, Any], original_box, *, minimum_core_ink_fraction=0.0):
    import cv2
    import numpy as np

    original = [float(v) for v in original_box]
    context = [float(v) for v in crop['bbox_pdf_pt']]
    gray = np.asarray(image.convert('L'))
    ink = (gray < 180).astype(np.uint8)
    height, width = ink.shape
    sx, sy = width / (context[2]-context[0]), height / (context[3]-context[1])
    core = [max(0, int(np.floor((original[0]-context[0])*sx))),
            max(0, int(np.floor((original[1]-context[1])*sy))),
            min(width, int(np.ceil((original[2]-context[0])*sx))),
            min(height, int(np.ceil((original[3]-context[1])*sy)))]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    touched = set(np.unique(labels[core[1]:core[3], core[0]:core[2]])) - {0}
    completed = list(original)
    accepted, bounded_out, deferred = [], [], []
    for label in sorted(touched):
        left, top, box_width, box_height, area = map(int, stats[label])
        right, bottom = left + box_width, top + box_height
        if area < 4 or (left >= core[0] and top >= core[1] and right <= core[2] and bottom <= core[3]):
            continue
        component = [left, top, right, bottom]
        core_pixels = int(np.count_nonzero(labels[core[1]:core[3], core[0]:core[2]] == label))
        core_fraction = core_pixels / area
        if core_fraction < minimum_core_ink_fraction:
            deferred.append({'bbox_context_px': component, 'ink_pixels': area,
                             'core_ink_fraction': core_fraction})
            continue
        if left <= 0 or top <= 0 or right >= width or bottom >= height:
            bounded_out.append(component)
            continue
        # One source pixel of air keeps the completed stroke off the next crop's edge.
        box = [context[0]+(left-1)/sx, context[1]+(top-1)/sy,
               context[0]+(right+1)/sx, context[1]+(bottom+1)/sy]
        completed = [min(completed[0], box[0]), min(completed[1], box[1]),
                     max(completed[2], box[2]), max(completed[3], box[3])]
        accepted.append({'bbox_context_px': component, 'ink_pixels': area})
    return [round(v, 6) for v in completed], {
        'version': 'inline-formula-connected-ink-bounds-v1',
        'original_bbox_pdf_pt': original, 'completed_bbox_pdf_pt': [round(v, 6) for v in completed],
        'context_bbox_pdf_pt': context, 'context_sha256': crop.get('content_sha256'),
        'minimum_core_ink_fraction': minimum_core_ink_fraction,
        'deferred_edge_components': deferred,
        'completed_components': accepted, 'context_edge_components_unchanged': bounded_out,
        'semantic_correction': False,
    }
