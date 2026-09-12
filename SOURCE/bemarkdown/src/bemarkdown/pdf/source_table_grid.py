"""Corroborate detector spans with visible rules before correcting topology."""

from __future__ import annotations

import math
from itertools import pairwise

from PIL import Image

from ..pdf_table_engine import assess_structure


def recover_source_table_grid(path, boxes, tokens):
    """Return replacement tokens only for a complete, source-ruled grid.

    Model row/column counts remain authoritative. Every interior atomic boundary
    is tested, including absent rules inside merged cells. Crop clipping can
    establish an exterior edge but never an interior separator. Raw model tokens
    and boxes are kept by the caller.
    """
    with Image.open(path) as image:
        width, height = image.size
        topology = assess_structure(tokens, [], (width, height))
        rows, cols = topology['rows'], topology['cols']
        diagnostic = {'status': 'NOT_RECOVERED', 'reason': None}
        if not (rows > 0 and cols > 0 and boxes):
            return None, {**diagnostic, 'reason': 'NO_MODEL_GRID'}
        coordinates = [box.get('coordinate') for box in boxes]
        if any(not isinstance(box, (list, tuple)) or len(box) != 4
               or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
               or box[0] >= box[2] or box[1] >= box[3] for box in coordinates):
            return None, {**diagnostic, 'reason': 'INVALID_DETECTOR_BOX'}
        axes, indices = [], {}
        for axis, count in ((0, cols), (1, rows)):
            tolerance = max(1.0, min(b[axis + 2] - b[axis] for b in coordinates) * 0.08)
            edges = sorted((b[side], index, side) for index, b in enumerate(coordinates)
                           for side in (axis, axis + 2))
            clusters = []
            for edge in edges:
                if not clusters or edge[0] - clusters[-1][0][0] > tolerance:
                    clusters.append([])
                clusters[-1].append(edge)
            if len(clusters) != count + 1:
                return None, {**diagnostic, 'reason': 'MODEL_GRID_DIMENSIONS_DISAGREE'}
            axes.append([sum(e[0] for e in cluster) / len(cluster) for cluster in clusters])
            for boundary, cluster in enumerate(clusters):
                for _, index, side in cluster:
                    indices[index, side] = boundary
        occupancy = [[None] * cols for _ in range(rows)]
        spans = []
        for index in range(len(boxes)):
            x0, y0, x1, y1 = [indices[index, side] for side in range(4)]
            if x0 >= x1 or y0 >= y1:
                return None, {**diagnostic, 'reason': 'DEGENERATE_CELL'}
            spans.append((y0, x0, y1 - y0, x1 - x0))
            for row in range(y0, y1):
                for col in range(x0, x1):
                    if occupancy[row][col] is not None:
                        return None, {**diagnostic, 'reason': 'OVERLAPPING_CELLS'}
                    occupancy[row][col] = index
        if any(owner is None for row in occupancy for owner in row):
            return None, {**diagnostic, 'reason': 'UNCOVERED_GRID_SLOT'}
        original = {(c['row'], c['col'], c['rowspan'], c['colspan'])
                    for c in topology['cells']}
        import numpy as np

        pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    # Min-channel contrast detects white rules on pale colored backgrounds.
        light = pixels.min(axis=2)
        gray = pixels.mean(axis=2)
        voids = _source_corner_voids(pixels, light, gray, axes, spans, rows, cols)
        for index in voids:
            row, col, rowspan, colspan = spans[index]
            for y in range(row, row + rowspan):
                for x in range(col, col + colspan):
                    occupancy[y][x] = None
        retained = [span for index, span in enumerate(spans) if index not in voids]
        if set(retained) == original:
            return None, {**diagnostic, 'status': 'NOT_NEEDED', 'reason': 'TOPOLOGY_AGREES'}
        checks = []
    for axis, count, cross_count in ((0, cols, rows), (1, rows, cols)):
        boundaries, cross = axes[axis], axes[1 - axis]
        for boundary in range(count + 1):
            for segment in range(cross_count):
                exterior = boundary in (0, count)
                if axis == 0:
                    before = occupancy[segment][boundary - 1] if boundary else None
                    after = occupancy[segment][boundary] if boundary < count else None
                else:
                    before = occupancy[boundary - 1][segment] if boundary else None
                    after = occupancy[boundary][segment] if boundary < count else None
                required = before != after
                position = boundaries[boundary]
                limit = width if axis == 0 else height
                at_frame = exterior and min(abs(position), abs(position - (limit - 1))) <= 5
                if at_frame and required:
                    checks.append({'axis': axis, 'boundary': boundary, 'segment': segment,
                                   'required': True, 'evidence': 'CROP_FRAME'})
                    continue
                radius = min(8, max(3, min(b - a for a, b in pairwise(boundaries)) * 0.08))
                score = _rule_score(light, gray, axis, position, cross[segment],
                                    cross[segment + 1], radius)
                checks.append({'axis': axis, 'boundary': boundary, 'segment': segment,
                               'required': required, 'rule_support': score})
                if (required and score < 0.85) or (not required and score > 0.15):
                    return None, {**diagnostic, 'reason': 'SOURCE_RULE_DISAGREEMENT',
                                  'checks': checks}
        starts = {(row, col): (rowspan, colspan) for row, col, rowspan, colspan in retained}
    recovered = ['<table>']
    for row in range(rows):
        recovered.append('<tr>')
        for col in range(cols):
            if (row, col) not in starts:
                continue
            rowspan, colspan = starts[row, col]
            attrs = (f' rowspan="{rowspan}"' if rowspan > 1 else '')
            attrs += (f' colspan="{colspan}"' if colspan > 1 else '')
            recovered.extend([f'<td{attrs}>', '</td>'])
        recovered.append('</tr>')
    recovered.append('</table>')
    return recovered, {'status': 'SOURCE_CORROBORATED', 'reason': None,
                'rows': rows, 'cols': cols, 'logical_cells': len(retained),
                'excluded_source_void_indices': sorted(voids),
                'checks': checks, 'boundary_coordinates': axes}


def _source_corner_voids(pixels, light, gray, axes, spans, rows, cols):
    """Reject only uniform corner boxes open on both outward-facing sides.

    A blank ruled cell still has enclosing edges. Require the two inward edges
    and the absence of both outward edges; the whole retained grid is checked by
    the caller. Model boxes remain in provenance, including excluded candidates.
    """
    import numpy as np

    voids = set()
    for index, (row, col, rowspan, colspan) in enumerate(spans):
        end_row, end_col = row + rowspan, col + colspan
        # HTML ragged rows can omit trailing cells, but cannot omit leading cells
        # without shifting their logical column indices.
        if (row != 0 and end_row != rows) or end_col != cols:
            continue
        if rowspan == rows or colspan == cols:
            continue
        bounds = (axes[0][col], axes[1][row], axes[0][end_col], axes[1][end_row])
        x0, y0, x1, y1 = bounds
        margin = min(8, max(4, min(x1 - x0, y1 - y0) * 0.08))
        interior = pixels[math.ceil(y0 + margin):math.floor(y1 - margin),
                          math.ceil(x0 + margin):math.floor(x1 - margin)]
        if not interior.size or float(np.ptp(interior, axis=(0, 1)).max()) > 8:
            continue
        outward = (0 if col == 0 else 2, 1 if row == 0 else 3)
        supported = []
        for side, position in enumerate(bounds):
            axis = side % 2
            start, end = (y0, y1) if axis == 0 else (x0, x1)
            radius = min(8, max(3, min(b - a for a, b in pairwise(axes[axis])) * 0.08))
            score = _rule_score(light, gray, axis, position, start, end, radius)
            supported.append(score <= 0.15 if side in outward else score >= 0.85)
        if all(supported):
            voids.add(index)
    return voids


def _rule_score(light, gray, axis, position, start, end, radius):
    """Fraction of a segment supported by a continuous contrasting thin rule."""
    import numpy as np

    if axis == 1:
        light, gray = light.T, gray.T
    cross_limit, limit = light.shape
    # Ignore intersections; they do not prove the intervening segment exists.
    margin = min(8, max(2, (end - start) * 0.08))
    lo = max(0, math.ceil(start + margin))
    hi = min(cross_limit, math.floor(end - margin))
    if hi - lo < 8:
        return 0.0
    best = 0.0
    for coordinate in range(max(0, int(position - radius)),
                            min(limit, math.ceil(position + radius) + 1)):
        left = coordinate - 4 if coordinate >= 4 else min(limit - 1, coordinate + 4)
        right = coordinate + 4 if coordinate + 4 < limit else max(0, coordinate - 4)
        mid = light[lo:hi, coordinate]
        near = np.maximum(light[lo:hi, left], light[lo:hi, right])
        white = (mid >= 235) & (mid - near >= 8)
        center_gray = gray[lo:hi, coordinate]
        surround = np.minimum(gray[lo:hi, left], gray[lo:hi, right])
        dark = (center_gray < 190) & (surround - center_gray >= 20)
        best = max(best, float(np.mean(white | dark)))
    return best
