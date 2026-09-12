"""Bounds from embedded CFF outlines, without Unicode font substitution."""

import math
from functools import lru_cache


@lru_cache(maxsize=16)
def _cff_font(data):
    import fitz

    return fitz.Font(fontbuffer=data)


@lru_cache(maxsize=4096)
def _cff_glyph_metrics(data, glyph):
    import fitz

    font = _cff_font(data)
    path = fitz.mupdf.fz_outline_glyph(font.this, glyph, fitz.mupdf.FzMatrix())
    # A null stroke measures the filled outline. The default stroke state expands
    # the box by its miter limit and is unsuitable for source character bounds.
    rect = fitz.mupdf.ll_fz_bound_path(
        path.m_internal, None, fitz.mupdf.FzMatrix().internal()
    )
    advance = fitz.mupdf.fz_advance_glyph(font.this, glyph, 0)
    values = (rect.x0, rect.y0, rect.x1, rect.y1, advance)
    if not all(math.isfinite(v) for v in values):
        return None
    if rect.x0 >= rect.x1 or rect.y0 >= rect.y1 or advance <= 0:
        return None
    return values


def page_cff_ink_map(page):
    """Resolve subset collisions using the PDF Unicode-to-glyph mapping."""
    from .native_glyph_bounds import character_key

    candidates = {}
    for xref, extension, _, name, *_ in page.get_fonts():
        if extension == "cff":
            candidates.setdefault(name.split("+")[-1], set()).add(xref)
    output, mappings, fonts = {}, {}, {}
    if not candidates:
        return output
    for span in page.get_texttrace():
        if tuple(span["dir"]) != (1.0, 0.0) or span["type"] != 0:
            continue
        refs = candidates.get(span["font"].split("+")[-1], set())
        for codepoint, glyph, origin, box in span["chars"]:
            # This bounded path covers Latin diagram labels. Ambiguous non-Latin
            # subsets decline until their source mapping can be proved as well.
            if not refs or not 0 <= codepoint < 256 or glyph <= 0:
                continue
            matching = []
            try:
                for xref in refs:
                    if xref not in mappings:
                        mappings[xref] = page.parent.get_char_widths(xref, limit=256)
                    if mappings[xref][codepoint][0] == glyph:
                        matching.append(xref)
                if len(matching) != 1:
                    continue
                xref = matching[0]
                if xref not in fonts:
                    fonts[xref] = page.parent.extract_font(xref)[3]
                metrics = _cff_glyph_metrics(fonts[xref], glyph)
            except (AttributeError, IndexError, TypeError, RuntimeError, ValueError):
                # Provider/version/outline failures retain the original text.
                continue
            if metrics is None:
                continue
            x0, y0, x1, y1, advance = metrics
            sx, sy = (box[2] - box[0]) / advance, span["size"]
            bounds = [origin[0] + x0 * sx, origin[1] - y1 * sy,
                      origin[0] + x1 * sx, origin[1] - y0 * sy]
            if sx <= 0 or sy <= 0 or not all(math.isfinite(v) for v in bounds):
                continue
            margin = span["size"] * 0.5
            if any(bounds[i] < box[i] - margin for i in (0, 1)) or any(
                bounds[i] > box[i] + margin for i in (2, 3)
            ):
                continue
            key = character_key(chr(codepoint), origin)
            output[key] = None if key in output and output[key] != bounds else bounds
    return output
