"""Recover visible dash sequences omitted by table OCR, without inferring values."""
from collections import deque

import numpy as np
from PIL import Image


def _components(mask):
    remaining = mask.copy()
    height, width = remaining.shape
    for y, x in zip(*np.nonzero(mask)):
        if not remaining[y, x]:
            continue
        remaining[y, x] = False
        queue = deque([(int(x), int(y))])
        left = right = int(x)
        top = bottom = int(y)
        while queue:
            cx, cy = queue.popleft()
            left, right = min(left, cx), max(right, cx)
            top, bottom = min(top, cy), max(bottom, cy)
            for nx, ny in ((cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)):
                if 0 <= nx < width and 0 <= ny < height and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    queue.append((nx, ny))
        yield [left, top, right+1, bottom+1]


def _overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def recover_dashed_cell_markers(path, cells, ocr):
    """Return only observed repeated short horizontal strokes inside a cell.

    OCR-covered ink, continuous rules, cell borders and isolated minus signs are
    excluded. The observed component count is retained; no missing time/value is
    invented and no model confidence is assigned to these pixel-derived marks.
    """
    with Image.open(path) as image:
        gray = np.asarray(image.convert('L'))
    height, width = gray.shape
    markers = []
    for index, cell in enumerate(cells):
        box = cell.get('coordinate', cell.get('bbox', []))
        if len(box) != 4:
            continue
        x0, y0 = max(0, int(box[0])+2), max(0, int(box[1])+2)
        x1, y1 = min(width, int(box[2])-1), min(height, int(box[3])-1)
        if x1 <= x0 or y1 <= y0:
            continue
        parts = []
        for local in _components(gray[y0:y1, x0:x1] < 220):
            w, h = local[2]-local[0], local[3]-local[1]
            if not (1 <= h <= 4 and max(2, 2*h) <= w <= min(16, (x1-x0)/6)):
                continue
            absolute = [local[0]+x0, local[1]+y0, local[2]+x0, local[3]+y0]
            if any(len(row.get('bbox', [])) == 4 and _overlaps(absolute, row['bbox']) for row in ocr):
                continue
            parts.append(absolute)
        bands = []
        for part in sorted(parts, key=lambda p: (p[1]+p[3], p[0])):
            center = (part[1]+part[3])/2
            match = next((band for band in bands if abs(center-band[0]) <= 1), None)
            if match is None:
                bands.append([center, [part]])
            else:
                match[1].append(part)
        for _center, band in bands:
            groups = []
            for part in sorted(band):
                if not groups or not (1 <= part[0]-groups[-1][-1][2] <= 2*(part[2]-part[0])):
                    groups.append([])
                groups[-1].append(part)
            for group in groups:
                if len(group) < 4:
                    continue
                widths = [p[2]-p[0] for p in group]
                gaps = [b[0]-a[2] for a, b in zip(group, group[1:])]
                if max(widths) > 2*min(widths) or max(gaps) > 2*min(gaps):
                    continue
                bounds = [min(p[0] for p in group), min(p[1] for p in group),
                          max(p[2] for p in group), max(p[3] for p in group)]
                if not 0.25*(x1-x0) <= bounds[2]-bounds[0] <= 0.85*(x1-x0):
                    continue
                if any(_overlaps(bounds, row['bbox']) for row in ocr if len(row.get('bbox', [])) == 4):
                    continue
                if any(_overlaps(bounds, row['bbox']) for row in markers):
                    continue
                markers.append({'bbox': bounds, 'text': '-'*len(group), 'confidence': None,
                                'source_feature': 'REPEATED_HORIZONTAL_DASH_COMPONENTS',
                                'source_cell_index': index, 'source_component_bboxes': group})
    return sorted(markers, key=lambda row: (row['bbox'][1], row['bbox'][0]))
