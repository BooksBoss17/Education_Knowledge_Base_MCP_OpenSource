"""Recover visible answer rules between native prose without claiming OCR text."""

from __future__ import annotations

from .visible_vectors import visible_vector_boxes


def recover_drawn_answer_blanks(page, rows):
    if not hasattr(page, 'get_drawings') or not hasattr(page, 'get_pixmap'):
        return rows
    spans = [(row, span) for row in rows for span in row['spans']]
    prose = [(row, span) for row, span in spans
             if sum('\u4e00' <= c <= '\u9fff' for c in span['text']) >= 2]
    if len(prose) < 2:
        return rows
    try:
        vectors = list(visible_vector_boxes(page))
    except (RuntimeError, ValueError):
        return rows
    used = set()
    for left_row, left in prose:
        a = left['bbox_pdf_pt']
        height = a[3] - a[1]
        if height <= 0:
            continue
        rights = [span for _, span in prose if span is not left
                  and 0.5 * height <= span['bbox_pdf_pt'][0] - a[2] <= 8 * height
                  and abs(span['bbox_pdf_pt'][3] - a[3]) <= 0.25 * height]
        if not rights:
            continue
        right = min(rights, key=lambda s: s['bbox_pdf_pt'][0])
        b = right['bbox_pdf_pt']
        if any(span is not left and span is not right
               and span['bbox_pdf_pt'][0] < b[0] - 0.5
               and span['bbox_pdf_pt'][2] > a[2] + 0.5
               and span['bbox_pdf_pt'][1] < min(a[3], b[3])
               and span['bbox_pdf_pt'][3] > max(a[1], b[1])
               for _, span in spans):
            continue
        for index, box in enumerate(vectors):
            if index in used:
                continue
            x0, y0, x1, y1 = box
            tolerance = max(1, height * 0.15)
            if not (0 <= y1 - y0 <= min(1.5, height * 0.12)
                    and abs(x0 - a[2]) <= tolerance
                    and abs(x1 - b[0]) <= tolerance
                    and abs((y0 + y1) / 2 - a[3]) <= 0.25 * height):
                continue
            if any(v[2] - v[0] <= 1.5 and v[3] - v[1] >= height
                   and v[1] <= y0 <= v[3]
                   and min(abs(v[0] - x0), abs(v[0] - x1)) <= tolerance
                   for v in vectors):
                continue
            if not _visible_dark_rule(page, box):
                continue
            span_id = f"{left['span_id']}-drawn-blank-{index:04d}"
            # Stroke centerlines have zero height in get_drawings(). Keep their
            # exact vector geometry separately from the bounded content region.
            region = list(box) if y1 > y0 else [x0, y0 - 0.25, x1, y1 + 0.25]
            recovered = {
                'span_id': span_id,
                'bbox_pdf_pt': region,
                'storage_bbox_pdf_pt': list(region),
                'text': '____',
                'punctuation_characters': [],
                'source_kind': 'DRAWN_ANSWER_BLANK',
                'source_vector_index': index,
                'source_rule_bbox_pdf_pt': list(box),
                'recognition': 'VISIBLE_RULE_BETWEEN_NATIVE_PROSE',
            }
            at = left_row['spans'].index(left) + 1
            left_row['spans'].insert(at, recovered)
            left_row['source_line_span_ids'].insert(at, span_id)
            left_row['text'] = ''.join(s['text'] for s in left_row['spans'])
            bbox = left_row['bbox_pdf_pt']
            left_row['bbox_pdf_pt'] = [min(bbox[0], x0), min(bbox[1], y0),
                                       max(bbox[2], x1), max(bbox[3], y1)]
            # native_char_count continues to count only actual source characters.
            used.add(index)
            break
    return rows


def _visible_dark_rule(page, box):
    import fitz
    import numpy as np

    x0, y0, x1, y1 = box
    clip = fitz.Rect(x0 + 1, y0 - 2, x1 - 1, y1 + 2)
    try:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip,
                                 colorspace=fitz.csRGB, alpha=False)
    except (RuntimeError, ValueError):
        return False
    if pixmap.width < 4 or pixmap.height < 3:
        return False
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3).mean(axis=2)
    dark = (pixels < 190).mean(axis=1)
    return bool(dark.max() >= 0.9 and pixels.mean(axis=1).max() - pixels.mean(axis=1).min() >= 20)
