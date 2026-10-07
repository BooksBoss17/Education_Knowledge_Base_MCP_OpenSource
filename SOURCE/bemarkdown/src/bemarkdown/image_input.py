"""Single-frame raster input with explicit source identity and PDF geometry."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import time
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_FORMATS = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.webp': 'WEBP',
                 '.bmp': 'BMP', '.tif': 'TIFF', '.tiff': 'TIFF'}
IMAGE_SUFFIXES = frozenset(IMAGE_FORMATS)
MAX_IMAGE_BYTES = 128 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
IMAGE_PDF_DPI = 216


class InvalidRasterImageError(ValueError):
    """Invalid raster; no models should be loaded for it."""


def _decode(source: Path):
    if source.stat().st_size > MAX_IMAGE_BYTES:
        raise InvalidRasterImageError('Image exceeds 128 MiB input limit')
    payload = source.read_bytes()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != IMAGE_FORMATS.get(source.suffix.lower()):
                    raise InvalidRasterImageError('Image format does not match its extension')
                if getattr(image, 'n_frames', 1) != 1:
                    raise InvalidRasterImageError('Only single-frame images are supported; no frames are discarded')
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InvalidRasterImageError('Image exceeds 40 million pixel input limit')
                details = {'format': image.format, 'original_pixel_size': list(image.size),
                           'exif_orientation': image.getexif().get(274, 1),
                           'source_sha256': hashlib.sha256(payload).hexdigest()}
                upright = ImageOps.exif_transpose(image).convert('RGBA')
                rgb = Image.new('RGB', upright.size, 'white')
                rgb.paste(upright, mask=upright.getchannel('A'))
                details.update(normalized_pixel_size=list(rgb.size),
                               normalization='EXIF_TRANSPOSE_RGB_ALPHA_ON_WHITE_NO_RESIZE')
                return rgb, details
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InvalidRasterImageError(f'Cannot decode raster image: {exc}') from exc


def inspect_raster_image(source: Path) -> dict:
    image, details = _decode(source)
    image.close()
    return details


def convert_raster_image(source: Path, output_dir: Path, *, pdf_runtime,
                         expected_source_sha256: str, debug: bool = False):
    """Reuse OCR/math/table/layout without pretending the input was a PDF."""
    import fitz
    from .production import build_document_id

    started = time.perf_counter()
    image, render = _decode(source)
    if render['source_sha256'] != expected_source_sha256:
        image.close()
        raise InvalidRasterImageError('Image changed after input validation')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    width, height = image.size
    image.close()
    render.update(dpi=IMAGE_PDF_DPI, pdf_points_per_pixel=72 / IMAGE_PDF_DPI,
                  normalized_png_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                  page_count=1, page_index=0)
    with tempfile.TemporaryDirectory(prefix='bmdimg-') as name:
        derived = Path(name) / 'source.pdf'
        with fitz.open() as pdf:
            page = pdf.new_page(width=width * 72 / IMAGE_PDF_DPI,
                                height=height * 72 / IMAGE_PDF_DPI)
            page.insert_image(page.rect, stream=buffer.getvalue())
            pdf.save(derived, no_new_id=True)
        render['pdf_sha256'] = hashlib.sha256(derived.read_bytes()).hexdigest()
        render['preparation_seconds'] = time.perf_counter() - started
        result = pdf_runtime.convert(derived, output_dir,
                                     document_id=build_document_id(derived), debug=debug)
        if debug:
            retained = output_dir / 'debug/source-pagination.pdf'
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(derived, retained)
            render['debug_pagination_ref'] = 'debug/source-pagination.pdf'
        report = result.report
        intermediate = dict(report['source'])
        report['source'] = {'type': 'IMAGE', 'path': str(source), 'file_name': source.name,
                            'size_bytes': source.stat().st_size, 'sha256': expected_source_sha256}
        report['input_transform'] = {'route': 'RASTER_IMAGE_PDF', 'renderer': render,
                                     'intermediate_source': intermediate,
                                     'coordinate_space': 'EXIF_NORMALIZED_IMAGE_PDF_POINTS',
                                     'recognition_policy': 'EXISTING_PDF_LAYOUT_TEXT_FORMULA_TABLE_ROUTES'}
        report.setdefault('timing', {})['total_seconds'] = time.perf_counter() - started
        (output_dir / 'conversion_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return result
