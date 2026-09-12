"""Source-pixel evidence for isolated fractions mislabelled as figures.

A fraction bar alone is insufficient: diagram labels also contain fractions.
Require glyph-scale components and reject large connected graphics. No text
recognition runs in this classifier.
"""
from __future__ import annotations


def isolated_fraction_evidence(image, ink_bbox):
    if ink_bbox is None:
        return None
    import cv2
    import numpy as np
    from .fraction_structure_repair import fraction_bars

    x0, y0, x1, y1 = ink_bbox
    width, height = x1-x0, y1-y0
    if min(width, height) < 8:
        return None
    mask = (np.asarray(image.convert('L')) < 160).astype('uint8')
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = [list(map(int, row)) for row in stats[1:] if row[4] >= 5]
    if any(w >= width*.45 and h >= height*.45 for _, _, w, h, _ in components):
        return None
    glyphs = [row for row in components
              if height*.15 <= row[3] <= height*.65
              and 2 <= row[2] <= width*.4 and row[2] < row[3]*3]
    if len(glyphs) < 3:
        return None
    glyph_height = float(np.median([row[3] for row in glyphs]))
    bars = [bar for bar in fraction_bars(image)
            if bar['bbox_px'][2]-bar['bbox_px'][0] >= max(width*.2, glyph_height*1.5)]
    if not bars:
        return None
    return {
        'schema': 'bemarkdown-isolated-fraction-pixels-v1',
        'proposed_semantic_type': 'FORMULA',
        'basis': 'FRACTION_BAR_WITH_GLYPH_SCALE_CONTENT_AND_NO_LARGE_CONNECTED_GRAPHIC',
        'ink_bbox_source_px': list(ink_bbox), 'source_image_size_px': list(image.size),
        'fraction_bars': bars, 'glyph_component_count': len(glyphs),
        'median_glyph_height_px': glyph_height,
        'text_recognition_performed': False,
    }
