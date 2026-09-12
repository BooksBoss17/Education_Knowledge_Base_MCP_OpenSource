"""Conservative native-text visibility from PDF paint order and opaque rectangles."""

from .native_glyph_bounds import character_key


def _single_rectangle(drawing):
    items = drawing.get("items", [])
    if len(items) == 1 and items[0][0] == "re":
        return tuple(float(v) for v in items[0][1])
    return None


def _intersection(left, right):
    box = (max(left[0], right[0]), max(left[1], right[1]),
           min(left[2], right[2]), min(left[3], right[3]))
    return box if box[2] > box[0] and box[3] > box[1] else None


def opaque_rectangles_in_paint_order(page):
    """Nonrectangular clips and transparent/blended groups decline as evidence."""
    try:
        drawings = page.get_drawings(extended=True)
    except (TypeError, RuntimeError, ValueError):
        return []
    result, stack = [], []
    for drawing in drawings:
        level = drawing.get("level", 0)
        while stack and stack[-1][0] >= level:
            stack.pop()
        kind = drawing.get("type")
        if kind == "group":
            safe = drawing.get("opacity") == 1 and drawing.get("blendmode") == "Normal"
            stack.append((level, safe, None))
            continue
        if kind == "clip":
            clip = _single_rectangle(drawing)
            stack.append((level, clip is not None, clip))
            continue
        if kind not in {"f", "fs"} or drawing.get("fill_opacity") != 1:
            continue
        if any(not safe for _, safe, _ in stack):
            continue
        box = _single_rectangle(drawing)
        if box is None or not isinstance(drawing.get("seqno"), int):
            continue
        for clip in [tuple(page.rect), *(clip for _, _, clip in stack if clip is not None)]:
            box = _intersection(box, clip)
            if box is None:
                break
        if box is not None:
            result.append((drawing["seqno"], box))
    return result


def hidden_native_characters(page):
    """Remove a source character only if every matching paint occurrence is hidden.

    Paint sequence numbers are the shared MuPDF sequence used by get_texttrace
    and get_drawings. Containment uses the full raw character box, not a guessed
    glyph shape. No OCR or interpretation of the character's meaning is involved.
    """
    if not hasattr(page, "get_texttrace") or getattr(page, "rotation", 0) != 0:
        return {}
    trace = page.get_texttrace()
    covers = opaque_rectangles_in_paint_order(page)
    if not covers and not any(s.get("type") == 3 or s.get("opacity") == 0 for s in trace):
        return {}
    occurrences = {}
    for span in trace:
        for codepoint, _, origin, _ in span["chars"]:
            if not 0 <= codepoint <= 0x10ffff:
                continue
            key = character_key(chr(codepoint), origin)
            occurrences.setdefault(key, []).append(span)
    boxes = {}
    for block in page.get_text("rawdict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    if char.get("origin") and char.get("bbox") and char.get("c", "").strip():
                        key = character_key(char["c"], char["origin"])
                        boxes.setdefault(key, []).append(char["bbox"])
    hidden = {}
    for key, raw_boxes in boxes.items():
        box = (min(b[0] for b in raw_boxes), min(b[1] for b in raw_boxes),
               max(b[2] for b in raw_boxes), max(b[3] for b in raw_boxes))
        evidence = []
        for span in occurrences.get(key, []):
            if span.get("type") == 3 or span.get("opacity") == 0:
                evidence.append({"basis": "INVISIBLE_TEXT_PAINT", "text_seqno": span["seqno"]})
                continue
            cover = next(((seqno, rect) for seqno, rect in covers
                          if seqno > span["seqno"]
                          and rect[0] + 0.01 < box[0] and rect[1] + 0.01 < box[1]
                          and rect[2] - 0.01 > box[2] and rect[3] - 0.01 > box[3]), None)
            if cover is None:
                evidence = []
                break
            evidence.append({"basis": "COVERED_BY_LATER_OPAQUE_RECTANGLE",
                             "text_seqno": span["seqno"], "cover_seqno": cover[0],
                             "cover_bbox_pdf_pt": list(cover[1])})
        if evidence:
            hidden[key] = {"text": key[0], "origin_pdf_pt": list(key[1:]),
                           "bbox_pdf_pt": list(box), "paint_evidence": evidence}
    return hidden
