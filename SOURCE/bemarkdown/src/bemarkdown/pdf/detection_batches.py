"""Batch only detector inputs whose existing preprocessing gives equal shapes."""

from __future__ import annotations

import time
from collections import defaultdict


def predict_detector_batches(detector, crops, *, batch_size, max_batch_pixels=1024*1024):
    if batch_size < 1:
        raise ValueError('Detector batch size must be positive')
    if max_batch_pixels < 1:
        raise ValueError('Detector batch pixel budget must be positive')
    paths = [str(crop['path']) for crop in crops]
    resize = getattr(detector, 'pre_tfs', {}).get('Resize')
    supported = callable(getattr(resize, 'resize', None)) and all(
        hasattr(detector, name) for name in ('limit_side_len', 'limit_type', 'max_side_limit'))
    metrics = {'strategy': 'SCALAR', 'requested_batch_size': batch_size, 'max_batch_size': 1,
               'shape_group_count': 0, 'detector_batch_count': len(paths), 'planning_seconds': 0.0,
               'max_batch_pixels': max_batch_pixels, 'oversized_single_inputs': 0}
    if batch_size == 1 or not supported or not paths:
        if not supported and batch_size > 1:
            metrics['strategy'] = 'SCALAR_UNSUPPORTED_PREPROCESSOR'
        outputs = list(detector.predict(paths, batch_size=1)) if paths else []
        if len(outputs) != len(paths):
            raise RuntimeError('PP_OCR_BATCH_DETECTION_CARDINALITY')
        return outputs, metrics

    import numpy as np
    from PIL import Image

    started = time.perf_counter()
    dimensions_to_shape = {}
    groups = defaultdict(list)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            dimensions = image.size
        if dimensions not in dimensions_to_shape:
            width, height = dimensions
            # The real existing resize operator determines shape from dimensions.
            # No resized/padded replacement is passed to detection.
            shape_image, _ = resize.resize(
                np.zeros((height, width, 3), dtype=np.uint8), detector.limit_side_len,
                detector.limit_type, detector.max_side_limit)
            dimensions_to_shape[dimensions] = tuple(shape_image.shape[:2])
        groups[dimensions_to_shape[dimensions]].append((index, path))
    metrics.update(strategy='EXACT_PREPROCESSED_SHAPE', shape_group_count=len(groups),
                   detector_batch_count=0, max_batch_size=0,
                   planning_seconds=time.perf_counter() - started)
    outputs = [None] * len(paths)
    for shape, group in groups.items():
        pixels = shape[0] * shape[1]
        size = min(batch_size, len(group), max(1, max_batch_pixels // pixels))
        if pixels > max_batch_pixels:
            # Preserve the existing scalar input instead of altering source pixels.
            metrics['oversized_single_inputs'] += len(group)
        values = list(detector.predict([path for _, path in group], batch_size=size))
        if len(values) != len(group):
            raise RuntimeError('PP_OCR_BATCH_DETECTION_CARDINALITY')
        metrics['detector_batch_count'] += (len(group) + size - 1) // size
        metrics['max_batch_size'] = max(metrics['max_batch_size'], size)
        for (index, _), value in zip(group, values, strict=True):
            outputs[index] = value
    return outputs, metrics
