"""Opt-in ordinary-Chinese-text route; retain image and never claim source adjudication.

Eligibility is a conservative empirical risk filter, not proof of OCR correctness.
It must be evaluated separately from source-verified agent decisions.
"""
import hashlib
import math
import re
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw

VERSION = 'qualified-local-plain-text-v1'
PUNCTUATION = '，。！？；：、（）《》“”‘’…—,.!?;:()'


def qualify_plain_text(image, evidence):
    def reject(reason): return {'accepted': False, 'reason': reason, 'version': VERSION}
    path = Path(image['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != image['sha256'] or evidence.get('input_sha256') != image['sha256']:
        raise ValueError('LOCAL_PLAIN_ACTUAL_INPUT_CHANGED')
    candidates = evidence.get('candidates', {})
    pp, ovis = candidates.get('PP'), candidates.get('Ovis')
    if not pp or not ovis or pp.get('status') not in {'OK', 'PASS'} or ovis.get('status') not in {'OK', 'PASS'}:
        return reject('TWO_COMPLETE_RECOGNIZERS_REQUIRED')
    text = str(ovis.get('text', '')).strip()
    compact = ''.join(text.split())
    cjk = sum('\u4e00' <= c <= '\u9fff' for c in compact)
    if len(compact) < 20 or len(compact) > 3000 or cjk / len(compact) < 0.8:
        return reject('ORDINARY_PROSE_LENGTH_OR_SCRIPT')
    if any(not ('\u4e00' <= c <= '\u9fff' or c in PUNCTUATION) for c in compact):
        return reject('MATH_NUMBER_IDENTIFIER_OR_FORMATTING_RISK')
    if ''.join(str(pp.get('text', '')).split()) != compact:
        return reject('EXACT_PUNCTUATION_PRESERVING_AGREEMENT_REQUIRED')
    secondary = candidates.get('Xiaomi')
    if secondary and (secondary.get('status') not in {'OK', 'PASS'} or
                      ''.join(str(secondary.get('text', '')).split()) != compact):
        return reject('SECONDARY_UNRESOLVED_OR_DISAGREES')
    lines = evidence.get('pp_lines', [])
    if not lines or ''.join(''.join(str(r.get('text', '')).split()) for r in lines) != compact:
        return reject('COMPLETE_ORDERED_LINE_EVIDENCE_REQUIRED')
    boxes = []
    for row in lines:
        confidence = row.get('rec_confidence', 0)
        box = row.get('bbox_local_px')
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or confidence < 0.995:
            return reject('LOW_LINE_RECOGNITION_CONFIDENCE')
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            return reject('LINE_GEOMETRY_REQUIRED')
        boxes.append(box)
    with Image.open(path) as original:
        if original.width * original.height > 32_000_000:
            return reject('PIXEL_WINDOW_EXCEEDED')
        if original.size != (image['width'], image['height']):
            raise ValueError('LOCAL_PLAIN_INPUT_GEOMETRY_CHANGED')
        gray = original.convert('L')
    width, height = gray.size
    ink = gray.point(lambda p: 255 if p < 200 else 0)
    if ink.getbbox() is None:
        return reject('NO_VISIBLE_INK')
    bounds = ink.getbbox()
    if bounds[0] < 2 or bounds[1] < 2 or bounds[2] > width - 2 or bounds[3] > height - 2:
        return reject('POSSIBLE_EDGE_TRUNCATION')
    mask = Image.new('L', gray.size)
    draw = ImageDraw.Draw(mask)
    prior_bottom = None
    for left, top, right, bottom in boxes:
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            return reject('LINE_OUTSIDE_INPUT')
        if prior_bottom is not None and top < prior_bottom - 2:
            return reject('NON_SIMPLE_LINE_ORDER_OR_OVERLAP')
        prior_bottom = bottom
        draw.rectangle((max(0, math.floor(left)-2), max(0, math.floor(top)-2),
                        min(width-1, math.ceil(right)+2), min(height-1, math.ceil(bottom)+2)), fill=255)
    if ImageChops.subtract(ink, mask).getbbox() is not None:
        return reject('UNACCOUNTED_VISIBLE_INK')
    return {'accepted': True, 'reason': 'EXACT_LOCAL_PROSE_AGREEMENT_WITH_INPUT_COVERAGE',
            'version': VERSION, 'text': text, 'input_sha256': image['sha256'],
            'source_verified': False, 'preserve_image': True,
            'line_count': len(lines), 'geometry_scope': 'ENTIRE_INPUT',
            'not_a_guarantee_of_correctness': True}


def local_plain_result(task, decision):
    if not decision['accepted'] or decision['input_sha256'] != task.input_sha256:
        raise ValueError('LOCAL_PLAIN_DECISION_UNBOUND')
    return {'task_id': task.task_id, 'task_fingerprint': task.fingerprint,
        'status': 'SUCCESS', 'text': decision['text'], 'uncertain': False,
        'patches': [], 'reason': decision['reason'], 'source_verified': False,
        'text_presence': 'PRESENT', 'visual_content_kind': 'IMAGE_WITH_TEXT',
        'evidence_kind': 'QUALIFIED_LOCAL_OCR_AGREEMENT', 'local_acceptance': decision,
        'raw_transcription': decision['text'], 'schema': VERSION}
