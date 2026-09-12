"""Recover scientific-notation scripts from matching, visible native PDF glyphs."""
from __future__ import annotations

import re
import unicodedata


def recover_scientific_cell(text, cell_bbox, evidence):
    if evidence.get('native_text_trust') not in {'HIGH', 'MEDIUM'}:
        return text, None
    if evidence.get('source_profile') in {'IMAGE_ONLY', 'IMAGE_WITH_TEXT_LAYER'}:
        return text, None
    proposals = evidence.get('native_math_geometry', [])
    if not proposals:
        return text, None
    from .native_math import native_flat_latex

    def flattened(value):
        value = unicodedata.normalize('NFKC', value).replace('−', '-')
        return re.sub(r'\s+', '', value)

    candidates = []
    for proposal in proposals:
        box = proposal.get('bbox_pdf_pt')
        if not box or not (cell_bbox[0] - 0.5 <= box[0] < box[2] <= cell_bbox[2] + 0.5
                           and cell_bbox[1] - 0.5 <= box[1] < box[3] <= cell_bbox[3] + 0.5):
            continue
        glyphs = proposal.get('evidence', [])
        if flattened(''.join(g.get('text', '') for g in glyphs)) != flattened(text):
            continue
        # OCR and native glyph identities must agree before geometry may restore
        # script placement. Never guess that a baseline subtraction is an exponent.
        latex = native_flat_latex(glyphs)
        if not latex:
            continue
        match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))\\times10\^\{([+-]?\d+)\}',
                             re.sub(r'\s+', '', latex))
        if not match:
            continue
        exponent = match[2].translate(str.maketrans('0123456789+-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻'))
        restored = f'{match[1]}×10{exponent}'
        candidates.append((restored, proposal.get('geometry_id'), latex))
    if len(candidates) != 1:
        return text, None
    restored, geometry_id, latex = candidates[0]
    return restored, {'schema': 'native-scientific-cell-recovery-v1',
                      'geometry_id': geometry_id, 'native_latex': latex,
                      'ocr_text': text, 'restored_text': restored,
                      'basis': 'Matching full-cell glyph identities and native baseline/font-size geometry'}
