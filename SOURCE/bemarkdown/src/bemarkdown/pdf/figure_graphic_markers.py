"""Identify repeated cyan magnetic-field cross strokes without deleting pixels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _cross_geometry(mask):
    y, x = np.nonzero(mask)
    width, height = int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)
    if min(width, height) < 4:
        return None
    nx, ny = (x - x.min()) / (width - 1), (y - y.min()) / (height - 1)
    diagonal_a, diagonal_b = abs(nx - ny) <= 0.22, abs(nx + ny - 1) <= 0.22
    full = (0.65 <= width / height <= 1.5
            and float((diagonal_a | diagonal_b).mean()) >= 0.9
            and all(int((((nx < 0.5) == left) & ((ny < 0.5) == top)).sum()) >= 2
                    for left in (False, True) for top in (False, True)))
    cx, cy = float((x.min() + x.max()) / 2), float((y.min() + y.max()) / 2)
    if full:
        return {'x': cx, 'y': cy, 'width': width, 'height': height, 'full': True}
    # A circuit can cover half a cross. These two diagonal arms are NOT an
    # independent graphic proof: only a matching full-cross lattice can admit it.
    if not 0.35 <= height / width <= 0.8:
        return None
    for center_y in (float(y.min()), float(y.max())):
        diagonal = abs(abs(x - cx) - abs(y - center_y)) <= max(1.5, width * 0.18)
        if float(diagonal.mean()) >= 0.9 and min(int((x < cx).sum()), int((x > cx).sum())) >= 3:
            return {'x': cx, 'y': center_y, 'width': width, 'height': width, 'full': False}
    return None


def _cross_parts(path):
    import cv2

    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.int16)
    low, high = np.min(rgb, axis=2), np.max(rgb, axis=2)
    contrast = 255 - rgb[:, :, 0]
    cyan = (low < 230) & (rgb[:, :, 1] - rgb[:, :, 0] >= np.maximum(5, contrast * 0.3)) & (
        rgb[:, :, 2] - rgb[:, :, 0] >= np.maximum(5, contrast * 0.3))
    if int(cyan.sum()) < 12:
        return []
    other = (low < 210) & (high - low >= 40) & ~cyan
    _, _, stats, _ = cv2.connectedComponentsWithStats(other.astype(np.uint8), connectivity=8)
    for left, top, width, height, area in stats[1:]:
        if area < 3:
            continue
        edge = left <= 1 or top <= 1 or left + width >= rgb.shape[1]-1 or top + height >= rgb.shape[0]-1
        strip = min(width, height) <= max(2, max(width, height) * 0.3)
        if not (edge and strip):
            return []
    # Opposing hues mix into dark pixels where a cyan stroke meets the wire.
    # Only exempt pixels adjacent to BOTH verified color masks, not arbitrary
    # gray strokes. Separate black letters remain protected.
    kernel = np.ones((3, 3), dtype=np.uint8)
    junction = cv2.dilate(cyan.astype(np.uint8), kernel).astype(bool) & cv2.dilate(
        other.astype(np.uint8), kernel).astype(bool)
    neutral = (low < 170) & (high - low < 35) & ~cyan
    if int((neutral & ~junction).sum()) >= 3:
        return []
    _, labels, stats, _ = cv2.connectedComponentsWithStats(cyan.astype(np.uint8), connectivity=8)
    parts, explained = [], 0
    for index, (_, _, width, height, area) in enumerate(stats[1:], 1):
        if area < 5 or min(width, height) < 3:
            continue
        part = _cross_geometry(labels == index)
        if part is None:
            return []
        explained += int(area)
        parts.append({**part, 'image_width': rgb.shape[1], 'image_height': rgb.shape[0]})
    return parts if explained >= int(cyan.sum()) * 0.9 else []


def _cross_shape(path):
    parts = _cross_parts(path)
    return len(parts) == 1 and parts[0]['full']


def _clusters(values, tolerance):
    groups = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def colored_cross_marker_ids(units):
    """Require visible cross shape/color and a repeated two-dimensional grid.

    Single x variables, neutral/black letters, mixed letter crops and arbitrary
    colored strokes remain text candidates. Original figure pixels are kept.
    """
    candidates = {}
    for unit in units:
        primary = str(unit['evidence']['A'].get('normalized_text') or '').strip()
        selected = str(unit['resolver'].get('selected_text') or '').strip()
        if not all(value and set(value) <= set('xX×入 ') for value in (primary, selected)):
            continue
        request = unit['request']
        try:
            parts = _cross_parts(Path(request['crop_ref']))
        except (OSError, ValueError):
            continue
        if parts:
            x0, y0, x1, y1 = request['bbox']
            if x1 <= x0 or y1 <= y0:
                continue
            candidates[str(request['region_id'])] = [dict(p,
                x=x0 + p['x'] * (x1-x0) / p['image_width'],
                y=y0 + p['y'] * (y1-y0) / p['image_height'],
                width=p['width'] * (x1-x0) / p['image_width'],
                height=p['height'] * (y1-y0) / p['image_height']) for p in parts]
    anchors = [p for parts in candidates.values() for p in parts if p['full']]
    if len(anchors) < 4:
        return set()
    widths, heights = [p['width'] for p in anchors], [p['height'] for p in anchors]
    if min(widths + heights) <= 0 or max(widths) > min(widths) * 1.8 or max(heights) > min(heights) * 1.8:
        return set()
    columns = _clusters([p['x'] for p in anchors], max(widths) * 0.65)
    rows = _clusters([p['y'] for p in anchors], max(heights) * 0.65)
    if len(columns) < 2 or len(rows) < 2:
        return set()

    def cell(part):
        return (min(range(len(columns)), key=lambda i: abs(columns[i] - part['x'])),
                min(range(len(rows)), key=lambda i: abs(rows[i] - part['y'])))

    occupied = {cell(p) for p in anchors}
    if len(occupied) < len(columns) * len(rows) * 0.6:
        return set()
    if any(sum(c == i for c, _ in occupied) < 2 for i in range(len(columns))):
        return set()
    if any(sum(r == i for _, r in occupied) < 2 for i in range(len(rows))):
        return set()

    def on_grid(part):
        column, row = cell(part)
        return (abs(columns[column] - part['x']) <= max(widths) * 0.65
                and abs(rows[row] - part['y']) <= max(heights) * 0.65
                and min(widths) / 1.8 <= part['width'] <= max(widths) * 1.8
                and min(heights) / 1.8 <= part['height'] <= max(heights) * 1.8)

    return {key for key, parts in candidates.items() if all(on_grid(p) for p in parts)}
