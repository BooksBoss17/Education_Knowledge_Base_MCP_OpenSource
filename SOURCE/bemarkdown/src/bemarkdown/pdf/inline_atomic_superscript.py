"""Source geometry and local recognition for one missing prefix superscript digit."""
from dataclasses import asdict
import hashlib
import re

from .inline_math_repair import parse_simple_math, _unwrap_text


def parse_atomic_math(value):
    if not isinstance(value, str):
        return None
    text = _unwrap_text(value)
    if text is None:
        return None
    text = text.strip()
    for left, right in ((r'\(', r'\)'), (r'\[', r'\]'), ('$$', '$$'), ('$', '$')):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
            break
    return parse_simple_math(text)


def prefix_sup_geometry(image):
    """Require one connected capital base and separate, raised prefix glyphs."""
    import cv2
    import numpy as np
    gray = np.asarray(image.convert('L'))
    mask = (gray < 180).astype('uint8')
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = [list(map(int, row)) for row in stats[1:]]
    if not 3 <= len(components) <= 5 or any(row[4] < 3 for row in components):
        return None
    base_index = max(range(len(components)), key=lambda i: components[i][3])
    bx, by, bw, bh, _ = components[base_index]
    scripts = sorted([r for i, r in enumerate(components) if i != base_index])
    if bh < 8 or bw < 4 or any(
        x+w >= bx or h < 4 or h > bh*0.75 or y > by+bh*0.15 or y+h > by+bh*0.65
        for x, y, w, h, area in scripts
    ):
        return None
    if max(r[1] for r in scripts)-min(r[1] for r in scripts) > max(2, bh*0.15):
        return None
    if any(b[0]-(a[0]+a[2]) > bh*0.4 for a, b in zip(scripts, scripts[1:])):
        return None
    if bx-(scripts[-1][0]+scripts[-1][2]) > bh*0.6:
        return None
    return {'schema': 'bemarkdown-prefix-superscript-source-geometry-v1',
            'source_pixels_sha256': hashlib.sha256(str(image.size).encode()+image.convert('RGB').tobytes()).hexdigest(),
            'base_bbox': [bx, by, bx+bw, by+bh], 'prefix_digit_count': len(scripts),
            'prefix_bboxes': [[x, y, x+w, y+h] for x, y, w, h, _ in scripts]}


def source_superscript_candidates(layout, image):
    """Select visible extra raised glyphs, without supplying their digits to GOT."""
    from PIL import ImageOps
    candidates = []
    pattern = r'(?<![A-Za-z0-9])([0-9]{1,3})([A-Z])(?![A-Za-z0-9])'
    for match in re.finditer(pattern, layout.text):
        bbox = layout.crop_box(image, match.start(), match.end(), prefix_sup=True)
        if bbox is None:
            continue
        crop = ImageOps.expand(image.crop(bbox), border=4, fill='white')
        geometry = prefix_sup_geometry(crop)
        if geometry is None or geometry['prefix_digit_count'] != len(match.group(1))+1:
            continue
        candidates.append({'span': [match.start(), match.end()], 'bbox_source_px': list(bbox),
                           'geometry': geometry, 'source_text': match.group(), 'crop': crop})
    return candidates


def verify_atomic_superscript(context_latex, local_latex, output, image, *, crop_sha256):
    """Two actual local outputs and source glyph structure must agree exactly."""
    context, local = parse_simple_math(context_latex), parse_simple_math(local_latex)
    if context is None or local is None or context == local:
        return None
    if (context.base != local.base or not re.fullmatch('[A-Z]', local.base)
            or any((context.pre_sub, context.post_sub, context.post_sup,
                    local.pre_sub, local.post_sub, local.post_sup))):
        return None
    old, new = context.pre_sup or '', local.pre_sup or ''
    if (not old.isascii() or not old.isdigit() or not new.isascii() or not new.isdigit()
            or not 2 <= len(new) <= 4 or len(new) != len(old)+1
            or not any(new[:i]+new[i+1:] == old for i in range(len(new)))):
        return None
    if (output.get('output_contract_status') != 'PASS'
            or output.get('generation_limit_reached') is not False
            or output.get('source_crop_sha256') != crop_sha256
            or not re.fullmatch('[0-9a-f]{64}', str(crop_sha256))):
        return None
    atomic = parse_atomic_math(output.get('normalized_output'))
    if atomic != local:
        return None
    geometry = prefix_sup_geometry(image)
    if geometry is None or geometry['prefix_digit_count'] != len(new):
        return None
    return {'status': 'SOURCE_LOCAL_SUPERSCRIPT_CONFIRMED', 'candidate_latex': local.latex,
            'context_latex': context_latex, 'local_latex': local_latex,
            'source_crop_sha256': crop_sha256, 'geometry': geometry,
            'atomic_prediction': asdict(atomic), 'formatted_evidence': dict(output)}
