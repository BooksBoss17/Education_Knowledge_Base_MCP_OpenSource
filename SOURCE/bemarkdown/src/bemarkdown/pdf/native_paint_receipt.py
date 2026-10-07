"""Identify actual omitted paint glyphs by source font/Unicode/origin, never nearest bbox."""
from __future__ import annotations

from collections import defaultdict
import copy
import math

from .native_character_sources import raw_character_record, raw_character_ref
from .native_font_unicode import normalized_font_name
from .native_glyph_bounds import character_key


class NativePaintReceipt:
    def __init__(self, page, clip):
        self.page_index = page.number
        self.clip = list(clip)
        self.rotation = page.rotation
        self.operations = []
        self.errors = []
        self.index = defaultdict(list)
        self.claims = defaultdict(list)
        try:
            # Rotation needs an explicitly tested mapping; never guess coordinates.
            if self.rotation:
                self.errors.append('ROTATED_PAGE_MAPPING_UNVERIFIED')
                return
            for bi, block in enumerate(page.get_text('rawdict')['blocks']):
                if block.get('type') != 0:
                    continue
                for li, line in enumerate(block.get('lines', [])):
                    for si, span in enumerate(line.get('spans', [])):
                        chars = span.get('chars', [])
                        for ci, char in enumerate(chars):
                            record = raw_character_record(char, span, raw_character_ref(bi, li, si, ci, len(chars)))
                            key = (normalized_font_name(span['font']), *character_key(char['c'], char['origin']))
                            self.index[key].append(record)
        except Exception as exc:  # ledger failure cannot alter established rendering
            self.errors.append(f'RAW_CHARACTER_INDEX_FAILED:{type(exc).__name__}:{exc}')

    def record(self, operation, text, ctm):
        import pymupdf as fitz
        mupdf = fitz.mupdf
        row = {'operation_index': len(self.operations), 'paint_kind': operation,
               'characters': [], 'errors': []}
        self.operations.append(row)
        try:
            span = text.head
            while span:
                wrapped = mupdf.FzTextSpan(span)
                font = normalized_font_name(mupdf.fz_font_name(wrapped.font()))
                for index in range(span.len):
                    item = wrapped.items(index)
                    point = mupdf.fz_transform_point(mupdf.FzPoint(item.x, item.y), mupdf.FzMatrix(ctm))
                    value = chr(item.ucs) if 0 <= item.ucs <= 0x10ffff else None
                    candidates = self.index.get((font, *character_key(value, (point.x, point.y))), [])
                    status = 'UNIQUE' if len(candidates) == 1 else ('AMBIGUOUS' if candidates else 'UNLINKED')
                    char = {'paint_glyph_index': index, 'font': font, 'original_text': value,
                            'origin_pdf_pt': [point.x, point.y], 'glyph_id': item.gid,
                            'identity_status': status}
                    if status == 'UNIQUE':
                        char.update(copy.deepcopy(candidates[0]))
                        char['intersects_crop'] = not (fitz.Rect(char['bbox_pdf_pt']) & fitz.Rect(self.clip)).is_empty
                        ref = char['source_character']
                        claims = self.claims[(ref['raw_span_id'], ref['character_index'])]
                        claims.append(char)
                        # Raw text extraction may collapse overprinted glyphs. Do not
                        # silently assign two paint events to one extracted character.
                        if len(claims) > 1:
                            for claim in claims:
                                claim['identity_status'] = 'AMBIGUOUS'
                                claim['reason'] = 'MULTIPLE_PAINT_GLYPHS_ONE_RAW_CHARACTER'
                    else:
                        char['intersects_crop'] = None
                        char['candidate_count'] = len(candidates)
                    if operation == 'fill_text' and item.gid >= 0 and not self.rotation:
                        # One PDF text operation can cover the header and body at
                        # once. Use the actual font glyph bounds, with a generous
                        # antialias margin, to avoid requiring off-crop header text.
                        # This never assigns a source identity or forgives on-crop
                        # ambiguity. Stroked text remains conservative.
                        transform = wrapped.trm()
                        transform.e, transform.f = item.x, item.y
                        bounds = mupdf.fz_bound_glyph(wrapped.font(), item.gid,
                                                      mupdf.fz_concat(transform, mupdf.FzMatrix(ctm)))
                        box = [bounds.x0, bounds.y0, bounds.x1, bounds.y1]
                        if all(math.isfinite(v) for v in box) and box[0] < box[2] and box[1] < box[3]:
                            char['paint_bbox_pdf_pt'] = box
                            padded = fitz.Rect(box) + (-1, -1, 1, 1)
                            char['intersects_crop'] = not (padded & fitz.Rect(self.clip)).is_empty
                    row['characters'].append(char)
                span = span.next
        except Exception as exc:
            row['errors'].append(f'PAINT_CHARACTER_READ_FAILED:{type(exc).__name__}:{exc}')

    def to_dict(self):
        complete = not self.errors and all(
            not op['errors'] and all(c['identity_status'] == 'UNIQUE' or c.get('intersects_crop') is False
                                    for c in op['characters'])
            for op in self.operations
        )
        return {
            'schema': 'native-text-omission-receipt-v1',
            'origin': 'ACTUAL_MUPDF_PAINT_CALLBACK',
            'page_index': self.page_index, 'rotation': self.rotation,
            'crop_bbox_pdf_pt': self.clip, 'coordinate_space': 'UNROTATED_PDF_PT',
            'identity_status': 'COMPLETE' if complete else 'INCOMPLETE',
            'operations': self.operations, 'errors': self.errors,
            'coverage_status': 'NOT_VERIFIED', 'authorizes_exclusion': False,
        }
