"""Read embedded TrueType glyph extents for source-only formula crop bounds."""

from __future__ import annotations

import math
import struct
from functools import lru_cache


@lru_cache(maxsize=32)
def _truetype_metrics(data):
    """Bounded SFNT reader; CFF, missing and malformed fonts decline safely."""
    try:
        if data[:4] not in (b'\x00\x01\x00\x00', b'true'):
            return None
        count = struct.unpack_from('>H', data, 4)[0]
        if not 0 < count <= 256 or 12 + count * 16 > len(data):
            return None
        tables = {}
        for index in range(count):
            tag, _, offset, length = struct.unpack_from('>4sIII', data, 12 + index * 16)
            if offset + length > len(data):
                return None
            tables[tag] = data[offset:offset + length]
        head, maxp, loca, glyf = (tables[t] for t in (b'head', b'maxp', b'loca', b'glyf'))
        units = struct.unpack_from('>H', head, 18)[0]
        location_format = struct.unpack_from('>h', head, 50)[0]
        glyph_count = struct.unpack_from('>H', maxp, 4)[0]
        metric_count = struct.unpack_from('>H', tables[b'hhea'], 34)[0]
        if not 16 <= units <= 16384 or location_format not in (0, 1):
            return None
        if not 0 < metric_count <= glyph_count or len(tables[b'hmtx']) < metric_count * 4:
            return None
        size, code = (2, 'H') if location_format == 0 else (4, 'I')
        if len(loca) < (glyph_count + 1) * size:
            return None
        offsets = struct.unpack_from(f'>{glyph_count + 1}{code}', loca)
        if location_format == 0:
            offsets = tuple(value * 2 for value in offsets)
        metrics = []
        for glyph in range(glyph_count):
            start, end = offsets[glyph:glyph + 2]
            if not 0 <= start <= end <= len(glyf):
                return None
            advance = struct.unpack_from('>H', tables[b'hmtx'], min(glyph, metric_count - 1) * 4)[0]
            if end - start < 10 or advance == 0:
                metrics.append(None)
                continue
            _, x0, y0, x1, y1 = struct.unpack_from('>hhhhh', glyf, start)
            metrics.append((x0, y0, x1, y1, advance, units) if x0 < x1 and y0 < y1 else None)
        return tuple(metrics)
    except (KeyError, struct.error, IndexError):
        return None


def character_key(text, origin):
    return text, round(origin[0], 4), round(origin[1], 4)


def page_native_ink_map(page):
    """Map original Unicode/origin to embedded glyph bounds, without OCR."""
    from .cff_glyph_bounds import page_cff_ink_map

    candidates = {}
    for xref, _, _, name, *_ in page.get_fonts():
        candidates.setdefault(name.split('+')[-1], set()).add(xref)
    metrics = {}
    output = page_cff_ink_map(page)
    for span in page.get_texttrace():
        if tuple(span['dir']) != (1.0, 0.0) or span['type'] != 0:
            continue
        name = span['font'].split('+')[-1]
        if name not in metrics:
            refs = candidates.get(name, set())
            # Multiple subsets with the same name cannot be resolved safely.
            metrics[name] = (_truetype_metrics(page.parent.extract_font(next(iter(refs)))[3])
                             if len(refs) == 1 else None)
        font = metrics[name]
        if font is None:
            continue
        for codepoint, glyph, origin, box in span['chars']:
            if not 0 <= glyph < len(font) or font[glyph] is None:
                continue
            x0, y0, x1, y1, advance, units = font[glyph]
            scale_x = (box[2] - box[0]) / advance
            scale_y = span['size'] / units
            bounds = [origin[0] + x0 * scale_x, origin[1] - y1 * scale_y,
                      origin[0] + x1 * scale_x, origin[1] - y0 * scale_y]
            if not all(math.isfinite(v) for v in bounds) or scale_x <= 0 or scale_y <= 0:
                continue
            margin = span['size'] * 0.5
            if any(bounds[i] < box[i] - margin for i in (0, 1)) or any(
                    bounds[i] > box[i] + margin for i in (2, 3)):
                continue
            key = character_key(chr(codepoint), origin)
            if key in output and output[key] != bounds:
                output[key] = None
            else:
                output[key] = bounds
    return output


def formula_ink_bbox(evidence, ink_map):
    """Tighten only when every source component has an independent bound."""
    boxes = []
    for item in evidence:
        if item['kind'] == 'NATIVE_CHARACTER':
            if not item.get('origin_pdf_pt'):
                return None
            box = ink_map.get(character_key(item.get('original_text', item['text']),
                                           item['origin_pdf_pt']))
        elif item['kind'] in ('HORIZONTAL_VECTOR_SUPPORT', 'FILLED_VECTOR_PATH'):
            box = item['bbox']
        else:
            return None
        if box is None:
            return None
        boxes.append(box)
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]
