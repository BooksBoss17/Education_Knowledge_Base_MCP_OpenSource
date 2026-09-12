"""Split spatially separated diagram labels while preserving every input pixel."""
from __future__ import annotations

import copy
import hashlib
import io
import re
from pathlib import Path

import numpy as np
from PIL import Image


def spaced_label_boxes(image, text):
    if not re.fullmatch(r'[A-Za-z0-9_\s]+', text) or len(re.findall('[A-Za-z]', text)) < 2:
        return []
    if image.width < 2.5 * image.height:
        return []
    ink = np.asarray(image.convert('L')) < 170
    ys, xs = np.where(ink)
    if not len(xs):
        return []
    minimum_gap = max(6, .4 * (int(ys.max()) - int(ys.min()) + 1))
    columns = ink.any(axis=0)
    cuts, gap_start = [], None
    for x in range(int(xs.min()), int(xs.max()) + 1):
        if not columns[x] and gap_start is None:
            gap_start = x
        elif columns[x] and gap_start is not None:
            if x - gap_start >= minimum_gap:
                cuts.append((gap_start + x) // 2)
            gap_start = None
    if not cuts:
        return []
    bounds = [0, *cuts, image.width]
    boxes = [[left, 0, right, image.height] for left, right in zip(bounds, bounds[1:])]
    if any(right - left < 3 or not ink[:, left:right].any() for left, _, right, _ in boxes):
        return []
    return boxes


def split_spaced_figure_labels(crop, lines, rec_model):
    from ..pdf_content_router import _local_px_to_page_pdf, _model_result_dict

    pending = []
    for index, line in enumerate(lines):
        text = str(line.get('text') or '')
        if not re.fullmatch(r'[A-Za-z0-9_\s]+', text) or len(re.findall('[A-Za-z]', text)) < 2:
            continue
        path = Path(line['line_crop_ref'])
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != line['line_crop_sha256']:
            raise RuntimeError('FIGURE_SEGMENT_PARENT_SHA_MISMATCH')
        with Image.open(path) as source:
            image = source.convert('RGB')
            boxes = spaced_label_boxes(image, str(line.get('text') or ''))
            for box in boxes:
                buffer = io.BytesIO()
                image.crop(box).save(buffer, format='PNG')
                data = buffer.getvalue()
                sha = hashlib.sha256(data).hexdigest()
                target = path.parent / f'figure-part-{sha}.png'
                if not target.exists():
                    target.write_bytes(data)
                pending.append((index, box, target, sha))
    if not pending:
        return {'split_parent_count': 0, 'child_count': 0}
    try:
        predictions = list(rec_model.predict([str(p) for _, _, p, _ in pending], batch_size=16))
        if len(predictions) != len(pending):
            raise RuntimeError('FIGURE_SEGMENT_RECOGNITION_CARDINALITY')
        grouped = {}
        for (index, box, path, sha), prediction in zip(pending, predictions, strict=True):
            recognized = _model_result_dict(prediction)
            value = str(recognized.get('rec_text') or '').strip()
            grouped.setdefault(index, []).append((box, path, sha, value, float(recognized.get('rec_score') or 0)))
        replacement = []
        split_count = child_count = 0
        for index, line in enumerate(lines):
            parts = grouped.get(index, [])
            # Separate simple labels only; operators or newly empty pieces retain the parent.
            if not parts or any(not re.fullmatch(r'[A-Za-z0-9_]+', part[3]) for part in parts):
                replacement.append(line)
                continue
            split_count += 1
            child_count += len(parts)
            x0, y0 = (max(0, int(v)) for v in line['bbox_local_px'][:2])
            for box, path, sha, value, confidence in parts:
                child = copy.deepcopy(line)
                local = [x0 + box[0], y0 + box[1], x0 + box[2], y0 + box[3]]
                child.update(text=value, rec_confidence=confidence,
                    confidence=min(float(line.get('det_confidence', 1)), confidence),
                    bbox_local_px=local, bbox_page_pdf_pt=_local_px_to_page_pdf(local, crop),
                    line_crop_ref=str(path), line_crop_sha256=sha,
                    figure_label_segmentation={'version': 'source-whitespace-label-partition-v1',
                        'parent_crop_ref': line['line_crop_ref'], 'parent_crop_sha256': line['line_crop_sha256'],
                        'child_crop_sha256': sha, 'bbox_parent_px': box,
                        'parent_a_text': line['text'], 'all_parent_pixels_partitioned': True})
                replacement.append(child)
        if not split_count:
            return {'split_parent_count': 0, 'child_count': 0}
        for sequence, line in enumerate(replacement):
            line['reading_sequence'] = sequence
        lines[:] = replacement
        return {'split_parent_count': split_count, 'child_count': child_count}
    except Exception as exc:
        return {'split_parent_count': 0, 'child_count': 0, 'error': f'{type(exc).__name__}:{exc}'}
