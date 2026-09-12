"""Source-pixel evidence for periodic waveforms misdetected as short W/M text."""
from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image


def _turns(curve):
    width = len(curve)
    radius = max(3, int(width * 0.12))
    amplitude = float(np.ptp(curve))
    if amplitude < 5:
        return []
    result = []
    for index in range(radius, width - radius):
        value = curve[index]
        left, right = curve[index - radius:index], curve[index + 1:index + radius + 1]
        kind = None
        if value >= left.max() and value >= right.max() and min(value - left.min(), value - right.min()) >= amplitude * 0.2:
            kind = 'max'
        if value <= left.min() and value <= right.min() and min(left.max() - value, right.max() - value) >= amplitude * 0.2:
            kind = 'min'
        if kind:
            item = (index, kind, float(value))
            if result and result[-1][1] == kind and index - result[-1][0] < radius:
                result[-1] = item
            else:
                result.append(item)
    return result


def waveform_text_candidate(text):
    """Only the short glyph confusions produced by a repeated wave are eligible."""
    return re.fullmatch(r'[wWmMnNvVuU~]{1,3}', str(text).strip()) is not None


def waveform_evidence(path):
    """Return reproducible curve evidence, never an inferred textual meaning.

    Require a thin connected stroke spanning the crop, at least four alternating
    interior extrema, and stable spacing/amplitude. Separate substantial text
    components prevent classification of mixed text/curve crops as graphics.
    """
    import cv2

    try:
        with Image.open(Path(path)) as opened:
            rgb = np.asarray(opened.convert('RGB'))
    except (OSError, ValueError):
        return None
    if min(rgb.shape[:2]) < 12:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [('light', binary > 0), ('dark', binary == 0),
             ('color', np.ptp(rgb.astype(np.int16), axis=2) >= 16)]
    inventories = []
    for mode, mask in masks:
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        inventories.append((mode, count, components, stats))
    for mode, count, components, stats in inventories:
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            if (width < rgb.shape[1] * 0.75 or height < rgb.shape[0] * 0.3
                    or area > width * height * 0.35 or width / height < 1.2):
                continue
            selected = components == component
            mixed = False
            for other_mode, _, other_components, other_stats in inventories:
                if other_mode == 'color':
                    continue
                for index, other in enumerate(other_stats[1:], 1):
                    if (int(other[3]) < height * 0.3 or int(other[4]) < area * 0.1
                            or int(other[4]) > gray.size * 0.35
                            or int(other[4]) > int(other[2]) * int(other[3]) * 0.65):
                        continue
                    if int((selected & (other_components == index)).sum()) < int(other[4]) * 0.5:
                        mixed = True
                        break
                if mixed:
                    break
            if mixed:
                continue
            local = components[y:y + height, x:x + width] == component
            positions = np.asarray([float(np.median(np.nonzero(local[:, i])[0])) if local[:, i].any()
                                    else np.nan for i in range(width)])
            valid = np.isfinite(positions)
            if valid.mean() < 0.85:
                continue
            curve = np.interp(np.arange(width), np.arange(width)[valid], positions[valid])
            curve = cv2.GaussianBlur(curve.reshape(1, -1), (0, 0), sigmaX=max(1, width / 120)).ravel()
            turns = _turns(curve)
            if len(turns) < 4 or any(a[1] == b[1] for a, b in pairwise(turns)):
                continue
            gaps = [b[0] - a[0] for a, b in pairwise(turns)]
            swings = [abs(b[2] - a[2]) for a, b in pairwise(turns)]
            if min(gaps) <= 0 or max(gaps) > min(gaps) * 1.6 or max(swings) > min(swings) * 1.8:
                continue
            return {'version': 'source-periodic-waveform-v1', 'mask': mode,
                    'component_bbox_px': [x, y, x + width, y + height],
                    'component_area_fraction': area / (width * height),
                    'column_coverage': float(valid.mean()), 'turns': turns}
    return None


def waveform_marker_evidence(units):
    result = {}
    for unit in units:
        primary = str(unit['evidence']['A'].get('normalized_text') or '').strip()
        selected = str(unit['resolver'].get('selected_text') or '').strip()
        if not waveform_text_candidate(selected) or (primary and not waveform_text_candidate(primary)):
            continue
        proof = waveform_evidence(unit['request']['crop_ref'])
        if proof is not None:
            result[str(unit['request']['region_id'])] = proof
    return result
