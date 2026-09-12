"""Source-bound verification of one missing fraction numerator and bar.

A generative alternative must preserve every other parsed symbol and script,
fit a unique source bar layout, and match independent local numerator OCR.
This module neither runs a model nor treats its verification as ground truth.
"""
import hashlib
from itertools import permutations
from pathlib import Path
import re

from .inline_math_repair import GREEK


def _seq(items):
    result = []
    for item in items:
        result.extend(item[1] if item[0] == 'seq' else [item])
    return ('seq', tuple(result))


def parse_math(value):
    if not isinstance(value, str) or len(value) > 16000:
        return None
    value = value.strip()
    for opening, closing in ((r'\[', r'\]'), (r'\(', r'\)'), ('$$', '$$'), ('$', '$')):
        if value.startswith(opening) and value.endswith(closing):
            value = value[len(opening):-len(closing)]
            break
    tokens = re.findall(r'\\[A-Za-z]+|\\.|[^\s]', value)
    if len(tokens) > 1024:
        return None
    cursor = 0
    spacing = {r'\,', r'\;', r'\!', '\\ ', '~', r'\quad', r'\qquad', r'\left', r'\right'}

    def argument(required, depth):
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError
        if tokens[cursor] == '{':
            cursor += 1
            return sequence(True, depth+1)
        if required:
            raise ValueError
        return _seq([atom(depth+1)])

    def atom(depth):
        nonlocal cursor
        token = tokens[cursor]
        cursor += 1
        if token in (r'\frac', r'\dfrac', r'\tfrac'):
            numerator, denominator = argument(True, depth), argument(True, depth)
            if not numerator[1] or not denominator[1]:
                raise ValueError
            return ('frac', numerator, denominator)
        if token in (r'\mathrm', r'\mathit'):
            return argument(True, depth)
        if token == '{':
            return sequence(True, depth+1)
        if token.startswith('\\'):
            if token[1:] in GREEK:
                token = GREEK[token[1:]]
            elif token in (r'\times', r'\cdot', r'\le', r'\ge'):
                token = {r'\times': '×', r'\cdot': '·', r'\le': '≤', r'\ge': '≥'}[token]
            else:
                raise ValueError
        if not re.fullmatch(r'[A-Za-z0-9\u0370-\u03ff+\-=(),.\[\]|/×·≤≥]', token):
            raise ValueError
        return ('atom', token)

    def sequence(braced=False, depth=0):
        nonlocal cursor
        if depth > 32:
            raise ValueError
        items = []
        while cursor < len(tokens):
            token = tokens[cursor]
            if token == '}':
                if not braced:
                    raise ValueError
                cursor += 1
                return _seq(items)
            if token in spacing:
                cursor += 1
                continue
            item = atom(depth)
            scripts = {}
            while cursor < len(tokens) and tokens[cursor] in ('_', '^'):
                mark = tokens[cursor]
                cursor += 1
                if mark in scripts:
                    raise ValueError
                scripts[mark] = argument(False, depth)
                if not scripts[mark][1]:
                    raise ValueError
            if scripts:
                if item[0] == 'seq' and len(item[1]) == 1:
                    item = item[1][0]
                item = ('script', item, scripts.get('_'), scripts.get('^'))
            items.append(item)
        if braced:
            raise ValueError
        return _seq(items)

    try:
        parsed = sequence()
        return parsed if parsed[1] and cursor == len(tokens) else None
    except (ValueError, IndexError, RecursionError):
        return None


def render_math(tree):
    tag = tree[0]
    if tag == 'seq':
        return ''.join(render_math(n) for n in tree[1])
    if tag == 'atom':
        return next(('\\'+name+' ' for name, glyph in GREEK.items() if tree[1] == glyph), tree[1])
    if tag == 'frac':
        return r'\frac{' + render_math(tree[1]) + '}{' + render_math(tree[2]) + '}'
    if tag == 'script':
        return ('{' + render_math(tree[1]) + '}'
                + ('_{' + render_math(tree[2]) + '}' if tree[2] is not None else '')
                + ('^{' + render_math(tree[3]) + '}' if tree[3] is not None else ''))
    raise ValueError('FRACTION_TREE_INVALID')


def _count(tree):
    if tree is None or tree[0] == 'atom':
        return 0
    if tree[0] == 'seq':
        return sum(_count(n) for n in tree[1])
    return int(tree[0] == 'frac') + sum(_count(n) for n in tree[1:])


def _deletions(tree, path=()):
    tag = tree[0]
    if tag == 'atom':
        return
    if tag == 'frac':
        yield tree[2], path, tree[1]
    if tag == 'seq':
        for i, child in enumerate(tree[1]):
            for changed, where, numerator in _deletions(child, path+('items', i)):
                yield _seq(tree[1][:i]+(changed,)+tree[1][i+1:]), where, numerator
    else:
        for i, child in enumerate(tree[1:], 1):
            if child is None:
                continue
            for changed, where, numerator in _deletions(child, path+(i,)):
                yield tree[:i]+(changed,)+tree[i+1:], where, numerator


def missing_numerator_relation(primary, candidate):
    first, second = parse_math(primary), parse_math(candidate)
    if first is None or second is None or _count(first) < 1 or _count(second) != _count(first)+1:
        return None
    matches = [(path, num) for changed, path, num in _deletions(second) if _seq([changed]) == first]
    if len(matches) != 1:
        return None
    path, numerator = matches[0]
    return {'fraction_path': list(path), 'numerator_latex': render_math(numerator),
            'primary_fraction_count': _count(first), 'candidate_fraction_count': _count(second)}


def _fraction_nodes(tree):
    nodes = []
    def walk(node, path=(), parent=None, branch=None):
        if node[0] == 'frac':
            index = len(nodes)
            nodes.append({'path': list(path), 'parent': parent, 'branch': branch})
            walk(node[1], path+(1,), index, 'numerator')
            walk(node[2], path+(2,), index, 'denominator')
        elif node[0] == 'seq':
            for i, child in enumerate(node[1]):
                walk(child, path+('items', i), parent, branch)
        elif node[0] == 'script' and _count(node):
            raise ValueError('SCRIPT_FRACTION_GEOMETRY_UNSUPPORTED')
    walk(tree)
    return nodes


def fraction_bars(image):
    import cv2
    import numpy as np
    mask = (np.asarray(image.convert('L')) < 160).astype('uint8')
    height, width = mask.shape
    _, _, glyphs, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = [int(h) for _, _, w, h, area in glyphs[1:] if h >= 4 and w < h*4 and area >= 5]
    glyph_height = float(np.median(heights)) if heights else height*0.2
    minimum = max(6, int(glyph_height*0.55))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (minimum, 1)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    bars = []
    for x, y, w, h, area in stats[1:]:
        if w < minimum or h > max(2, height*0.05) or w < h*8 or area < w*h*0.65:
            continue
        reach = max(8, int(glyph_height*1.6))
        top = mask[max(0, y-reach):y, x:x+w]
        bottom = mask[y+h:min(height, y+h+reach), x:x+w]
        near_top = mask[max(0, y-3):y, x:x+w]
        near_bottom = mask[y+h:min(height, y+h+3), x:x+w]
        if top.sum() < 4 or bottom.sum() < 4:
            continue
        if not ((near_top.sum(axis=1) == 0).any() and (near_bottom.sum(axis=1) == 0).any()):
            continue
        bars.append({'bbox_px': [int(x), int(y), int(x+w), int(y+h)], 'ink_pixels': int(area)})
    return bars


def _map_bars(nodes, bars):
    if len(nodes) != len(bars) or not 1 <= len(nodes) <= 8:
        return None
    groups = {}
    for i, node in enumerate(nodes):
        groups.setdefault((node['parent'], node['branch']), []).append(i)
    solutions = []
    for order in permutations(range(len(bars))):
        boxes = [bars[i]['bbox_px'] for i in order]
        valid = True
        for i, node in enumerate(nodes):
            if node['parent'] is None:
                continue
            x0, y0, x1, y1 = boxes[i]
            px0, py0, px1, py1 = boxes[node['parent']]
            if not (px0-2 <= x0 and x1 <= px1+2 and (
                y1 < py0 if node['branch'] == 'numerator' else y0 > py1)):
                valid = False
                break
        if valid:
            for indices in groups.values():
                if any(boxes[a][2] > boxes[b][0] for a, b in zip(indices, indices[1:])):
                    valid = False
                    break
        if valid:
            solutions.append(order)
            if len(solutions) > 1:
                return None
    return list(solutions[0]) if solutions else None


def source_fraction_risk(primary, image):
    """Screen only for one extra source bar; this does not authorize a repair."""
    tree = parse_math(primary)
    if tree is None:
        return {'required': False, 'reason': 'UNSUPPORTED_PRIMARY'}
    try:
        nodes = _fraction_nodes(tree)
    except ValueError:
        return {'required': False, 'reason': 'UNSUPPORTED_PRIMARY_GEOMETRY'}
    bars = fraction_bars(image)
    required = 1 <= len(nodes) < 8 and len(bars) == len(nodes)+1
    return {'required': required, 'primary_fraction_count': len(nodes),
            'source_bars': bars, 'reason': 'EXTRA_SOURCE_BAR' if required else 'NO_SINGLE_EXTRA_BAR'}


def _pixel_sha(image):
    image = image.convert('RGB')
    return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()


def source_fraction_plan(primary, candidate, image, *, candidate_truncated=False):
    result = {'schema': 'bemarkdown-source-fraction-repair-v1', 'primary': primary, 'candidate': candidate,
              'primary_sha256': hashlib.sha256(primary.encode()).hexdigest(), 'source_pixels_sha256': _pixel_sha(image)}
    if candidate_truncated:
        return {**result, 'status': 'CANDIDATE_TRUNCATED'}
    relation = missing_numerator_relation(primary, candidate)
    if relation is None:
        return {**result, 'status': 'NO_UNIQUE_MISSING_NUMERATOR_RELATION'}
    try:
        nodes = _fraction_nodes(parse_math(candidate))
    except ValueError as exc:
        return {**result, 'status': str(exc)}
    bars = fraction_bars(image)
    mapping = _map_bars(nodes, bars)
    result.update(relation=relation, source_bars=bars)
    if mapping is None:
        return {**result, 'status': 'SOURCE_TOPOLOGY_MISMATCH'}
    index = next(i for i, node in enumerate(nodes) if node['path'] == relation['fraction_path'])
    x0, y0, x1, _ = bars[mapping[index]]['bbox_px']
    upper = max((b['bbox_px'][3] for j, b in enumerate(bars) if j != mapping[index]
        and b['bbox_px'][3] < y0 and b['bbox_px'][0] < x1 and b['bbox_px'][2] > x0), default=0)
    import numpy as np
    mask = np.asarray(image.convert('L')) < 160
    # Stop at a verified blank row so antialiased edges of the fraction bar
    # cannot enter the numerator crop as a minus sign or underline.
    gap_start = max(upper, y0-3)
    gaps = np.flatnonzero(mask[gap_start:y0, x0:x1].sum(axis=1) == 0)
    if not len(gaps):
        return {**result, 'status': 'SOURCE_NUMERATOR_UNRESOLVED'}
    crop_end = gap_start+int(gaps[-1])
    ys, xs = np.nonzero(mask[upper:crop_end, x0:x1])
    if len(xs) < 4:
        return {**result, 'status': 'SOURCE_NUMERATOR_UNRESOLVED'}
    bbox = [x0+int(xs.min()), upper+int(ys.min()), x0+int(xs.max())+1, upper+int(ys.max())+1]
    return {**result, 'status': 'READY_FOR_LOCAL_NUMERATOR', 'numerator_bbox': bbox,
            'bar_mapping': mapping, 'numerator_latex': relation['numerator_latex']}


def bind_numerator_crop(plan, image, path):
    from PIL import ImageOps
    if _pixel_sha(image) != plan['source_pixels_sha256']:
        raise ValueError('FRACTION_SOURCE_PIXELS_CHANGED')
    if plan['status'] != 'READY_FOR_LOCAL_NUMERATOR':
        raise ValueError('FRACTION_NUMERATOR_NOT_READY')
    path = Path(path)
    ImageOps.expand(image.crop(plan['numerator_bbox']), border=4, fill='white').save(path)
    plan['numerator_crop_ref'] = str(path.resolve())
    plan['numerator_crop_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()


def verify_fraction_repair(plan, local_output):
    result = {'text': plan['primary'], 'accepted': False, 'status': plan['status']}
    if plan['status'] != 'READY_FOR_LOCAL_NUMERATOR':
        return result
    digest = plan.get('numerator_crop_sha256')
    if (not digest or local_output.get('source_crop_sha256') != digest
            or hashlib.sha256(Path(plan['numerator_crop_ref']).read_bytes()).hexdigest() != digest):
        raise ValueError('FRACTION_LOCAL_CROP_SHA_MISMATCH')
    if parse_math(local_output.get('latex')) != parse_math(plan['numerator_latex']):
        return {**result, 'status': 'LOCAL_NUMERATOR_DISAGREEMENT'}
    text = plan['candidate'].strip()
    for opening, closing in ((r'\[', r'\]'), (r'\(', r'\)'), ('$$', '$$'), ('$', '$')):
        if text.startswith(opening) and text.endswith(closing):
            text = text[len(opening):-len(closing)].strip()
            break
    return {**result, 'text': text, 'accepted': True, 'status': 'SOURCE_BOUND_FRACTION_REPAIRED'}
