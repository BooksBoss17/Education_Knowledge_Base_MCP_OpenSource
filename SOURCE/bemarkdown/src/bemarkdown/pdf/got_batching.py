"""Group independent GOT crops without modifying their pixels or sample identities."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

GOT_CROP_ORDERING = 'aspect-ratio-ascending-v1'
GOT_BATCH_RETRY_POLICY = 'cuda-oom-halving-v1'


def got_mode_groups(rows):
    """Keep plain transcription and formatted transcription in distinct batches."""
    groups = {False: [], True: []}
    for row in rows:
        mode = row.get('ocr_format', False)
        if type(mode) is not bool:
            raise ValueError('GOT_FORMAT_MODE_INVALID')
        groups[mode].append(row)
    return [group for group in groups.values() if group]


def order_got_mode_crops(rows, *, crop_root, batch_size):
    groups = got_mode_groups(rows)
    if not any(row.get('ocr_format', False) for row in rows):
        return order_got_crops(rows, crop_root=crop_root, batch_size=batch_size)
    ordered, reports = [], []
    for group in groups:
        values, report = order_got_crops(group, crop_root=crop_root, batch_size=batch_size)
        ordered.extend(values)
        reports.append({'ocr_format': group[0].get('ocr_format', False), **report})
    return ordered, {'policy': 'format-separated-' + GOT_CROP_ORDERING, 'groups': reports,
                     'input_count': len(rows)}


def adaptive_got_batches(rows, *, batch_size, infer, is_oom, release, events):
    """Retry only unfinished work, releasing the failed inference frame first.

    The inference callback must return a whole batch atomically. Reductions are
    sticky for the remaining input; size one and non-CUDA errors still fail.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
        raise ValueError('GOT_BATCH_SIZE_OUT_OF_RANGE')
    offset = 0
    while offset < len(rows):
        batch_rows = rows[offset:offset + batch_size]
        try:
            result = infer(batch_rows)
        except Exception as exc:  # Retry only explicitly classified CUDA OOM.
            if not is_oom(exc):
                raise
            reduced = max(1, len(batch_rows) // 2)
            events.append({'input_offset': offset, 'attempted_size': len(batch_rows),
                           'next_batch_size': reduced if len(batch_rows) > 1 else None})
            if len(batch_rows) == 1:
                raise
            batch_size = reduced
        else:
            yield batch_rows, result
            offset += len(batch_rows)
            continue
        # Outside the except block: its traceback otherwise retains model
        # tensors, making empty_cache ineffective and causing repeated OOMs.
        release()


def order_got_crops(
    rows: Sequence[Mapping[str, Any]], *, crop_root: Path, batch_size: int,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Reduce batch padding between short symbols and long lines; keep ties stable."""
    ordered = list(rows)
    status = 'SINGLE_BATCH_SOURCE_ORDER'
    error = None
    if len(rows) > batch_size:
        try:
            dimensions = []
            for index, row in enumerate(rows):
                with Image.open(crop_root / row['crop_local_relpath']) as image:
                    dimensions.append((image.width / image.height, index, row))
            ordered = [row for _aspect, _index, row in sorted(dimensions, key=lambda item: item[:2])]
            status = 'ORDERED'
        except (OSError, ValueError, ZeroDivisionError) as exc:
            # An optional scheduling step must not bypass the worker's existing
            # input-failure handling or prevent it from preserving evidence.
            status = 'FALLBACK_SOURCE_ORDER'
            error = type(exc).__name__
    ids = [str(row['sample_id']) for row in ordered]
    digest = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    return ordered, {'policy': GOT_CROP_ORDERING, 'status': status, 'input_count': len(rows),
                     'order_sha256': digest, 'error': error}
