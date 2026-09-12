"""Guard line voting against text that the shared detector did not cover."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image


def complete_region_line_boxes(image: Image.Image, boxes: Sequence) -> list | None:
    """Reframe isolated text rows around all source ink before recognizing them.

    Detected rows supply vertical anchors, but not horizontal clipping bounds.
    Every split crosses a completely blank raster row. Unexpected extra ink rows
    make the proposal ambiguous, preserving the existing whole-region fallback.
    All three recognizers must consume the newly materialized crops afterwards.
    """
    if not boxes:
        return None
    ink = image.convert('RGB').convert('L').point(lambda v: 255 if v < 200 else 0)
    width, height = ink.size
    clusters = []
    for box, score, polygon in sorted(boxes, key=lambda row: (row[0][1], row[0][0])):
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            return None
        if clusters and min(clusters[-1]['bottom'], box[3]) - box[1] > 0.25 * min(
            clusters[-1]['bottom'] - clusters[-1]['top'], box[3] - box[1]
        ):
            group = clusters[-1]
            group['top'] = min(group['top'], box[1])
            group['bottom'] = max(group['bottom'], box[3])
            group['scores'].append(score)
        else:
            clusters.append({'top':box[1], 'bottom':box[3], 'scores':[score]})
    row_ink = [ink.crop((0, y, width, y + 1)).getbbox() is not None for y in range(height)]
    boundaries = [0]
    for left, right in zip(clusters, clusters[1:]):
        candidates = [y for y in range(max(0, math.ceil(left['bottom'])), min(height, math.floor(right['top']) + 1)) if not row_ink[y]]
        if not candidates:
            return None
        target = (left['bottom'] + right['top']) / 2
        boundaries.append(min(candidates, key=lambda y: abs(y - target)))
    boundaries.append(height)
    result = []
    for group, top, bottom in zip(clusters, boundaries[:-1], boundaries[1:], strict=True):
        bounds = ink.crop((0, top, width, bottom)).getbbox()
        if bounds is None:
            return None
        x0, y0, x1, y1 = bounds
        consecutive_blank = maximum_blank = 0
        for occupied in row_ink[top + y0:top + y1]:
            consecutive_blank = 0 if occupied else consecutive_blank + 1
            maximum_blank = max(maximum_blank, consecutive_blank)
        if maximum_blank > max(2, round((group['bottom'] - group['top']) * 0.22)):
            return None
        box = [max(0, x0 - 2), max(top, top + y0 - 2), min(width, x1 + 2), min(bottom, top + y1 + 2)]
        scores = [value for value in group['scores'] if value is not None]
        polygon = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        result.append((box, min(scores) if scores else None, polygon))
    return result


def split_line_boxes_at_exclusions(image: Image.Image, boxes: Sequence, exclusions: Sequence) -> list:
    """Split around owned inline objects only where masked columns have no ink."""
    ink = image.convert('RGB').convert('L').point(lambda value: 255 if value < 200 else 0)
    result = []
    for box, score, polygon in boxes:
        x0, y0, x1, y1 = [int(value) for value in box]
        gaps = []
        for mask in exclusions:
            mx0, my0, mx1, my1 = mask
            left, right = max(x0, int(mx0)), min(x1, int(mx1))
            if left >= right or min(y1, my1) <= max(y0, my0):
                continue
            strip = ink.crop((left, y0, right, y1))
            if strip.getbbox() is None:
                gaps.append((left, right))
            else:
                # An imperfect ownership box can leave an operator's tail.
                # Cut only through the blank part, retaining every ink pixel
                # in the adjoining fragment for actual recognition.
                runs = []
                start = None
                for column in range(strip.width + 1):
                    blank = column < strip.width and strip.crop((column, 0, column + 1, strip.height)).getbbox() is None
                    if blank and start is None:
                        start = column
                    elif not blank and start is not None:
                        runs.append((start, column))
                        start = None
                if runs:
                    begin, end = max(runs, key=lambda span: span[1] - span[0])
                    if end - begin >= max(4, math.ceil(0.75 * strip.width)):
                        gaps.append((left + begin, left + end))
        if not gaps:
            result.append((box, score, polygon))
            continue
        cursor = x0
        for left, right in sorted(gaps) + [(x1, x1)]:
            if cursor < left and ink.crop((cursor, y0, left, y1)).getbbox() is not None:
                fragment = [cursor, y0, left, y1]
                corners = [[cursor, y0], [left, y0], [left, y1], [cursor, y1]]
                result.append((fragment, score, corners))
            cursor = max(cursor, right)
    return result


def assess_region_line_coverage(
    crop: Mapping[str, Any], lines: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Require dark source pixels to be inside the actual detector line crops.

    Input is already subject to the non-text ownership mask. Undetected text is
    kept on the original region recognition path, where GOT can still recover it.
    The four-pixel allowance only accommodates raster boundary rounding.
    """
    result = {'version':'region-line-ink-coverage-v1', 'eligible':False,
              'line_count':len(lines), 'dark_threshold':200, 'maximum_residual_pixels':4}
    if not lines:
        return {**result, 'reason':'DETECTOR_LINES_UNAVAILABLE'}
    with Image.open(crop['path']) as image:
        ink = image.convert('RGB').convert('L').point(lambda value: 255 if value < 200 else 0)
    width, height = ink.size
    if (width, height) != (int(crop['width']), int(crop['height'])):
        return {**result, 'reason':'SOURCE_CROP_DIMENSION_MISMATCH'}
    total = ink.histogram()[255]
    result['source_ink_pixels'] = total
    if not total:
        return {**result, 'reason':'SOURCE_CROP_NO_DARK_PIXELS'}
    for line in lines:
        box = line.get('bbox_local_px')
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
            return {**result, 'reason':'DETECTOR_LINE_GEOMETRY_INVALID'}
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            return {**result, 'reason':'DETECTOR_LINE_GEOMETRY_INVALID'}
        # These are the integer bounds of the actual PNG consumed by provider A.
        # Never enlarge them to make a detector miss pass the coverage check.
        ink.paste(0, tuple(int(v) for v in box))
    residual = ink.histogram()[255]
    return {**result, 'uncovered_ink_pixels':residual, 'eligible':residual <= 4,
            'reason':'DETECTOR_INK_COVERED' if residual <= 4 else 'DETECTOR_UNCOVERED_INK'}
