"""CPU-only, inactive candidate for the native aspect-ratio restriction.

No model weights, prediction, Gold access, dataset change or production injection.
"""

import hashlib

import math

def pixel_digest(image):
    rgba = image.convert('RGBA')
    return hashlib.sha256(rgba.tobytes()).hexdigest()

def pad_extreme(image, factor, max_aspect=200):
    from PIL import Image
    w, h = image.size
    assert w > 0 and h > 0 and factor > 0 and max_aspect > 0
    if max(w, h) / min(w, h) <= max_aspect:
        return image.copy(), {'changed': False, 'original_size': [w, h], 'output_size': [w, h]}
    rgba = image.convert('RGBA')
    needed = max(factor, math.ceil(max(w, h) / max_aspect))
    new_w, new_h = (max(w, needed), h) if w < h else (w, max(h, needed))
    left, top = (new_w - w) // 2, (new_h - h) // 2
    output = Image.new('RGBA', (new_w, new_h), (255, 255, 255, 255))
    output.paste(rgba, (left, top))  # no mask: preserve all original RGBA values
    box = (left, top, left + w, top + h)
    assert pixel_digest(output.crop(box)) == pixel_digest(image)
    assert max(output.size) / min(output.size) <= max_aspect
    return output, {'changed': True, 'original_size': [w, h], 'output_size': [new_w, new_h],
                    'original_region': list(box), 'original_rgba_sha256': pixel_digest(image),
                    'effective_rgba_sha256': pixel_digest(output), 'resized_or_cropped': False}
