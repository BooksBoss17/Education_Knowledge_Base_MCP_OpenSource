"""Deterministic visible pixels for retrieval, leaving source bytes untouched."""
from PIL import Image, ImageOps

PREPROCESSING = 'white-alpha-composite RGB thumbnail384 pad32 v1'


def prepare_image(original):
    # Converting RGBA directly to RGB exposes hidden RGB values. Composite first.
    # No EXIF change here: opaque images must retain the existing template pixels.
    rgba = original.convert('RGBA')
    background = Image.new('RGBA', rgba.size, 'white')
    background.alpha_composite(rgba)
    image = background.convert('RGB')
    image.thumbnail((384, 384))
    return ImageOps.pad(image, (max(32, image.width), max(32, image.height)), color='white')
