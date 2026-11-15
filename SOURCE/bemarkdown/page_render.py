"""Small rendering boundary, independent of model frameworks and PDF analysis."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageRender:
    page_index: int
    image_bytes: bytes
    render_sha256: str

    @property
    def page_number(self):
        return self.page_index + 1


@dataclass(frozen=True, slots=True)
class PayloadConversionResult:
    source: Path
    output_dir: Path
    report: dict[str, Any]


def render_pdf(source: str | Path) -> list[PageRender]:
    """Retain the verified 200 dpi RGB/JPEG recipe, without native text parsing."""
    import pymupdf
    from PIL import Image

    renders = []
    with pymupdf.open(source) as document:
        if document.needs_pass:
            raise ValueError('ENCRYPTED_PDF_REQUIRES_PASSWORD')
        for index, page in enumerate(document):
            pix = page.get_pixmap(matrix=pymupdf.Matrix(200/72, 200/72),
                                  colorspace=pymupdf.csRGB, alpha=False, annots=True)
            with Image.frombytes('RGB', (pix.width, pix.height), pix.samples) as image:
                stream = BytesIO()
                image.save(stream, format='JPEG', quality=92, optimize=True)
            data = stream.getvalue()
            renders.append(PageRender(index, data, hashlib.sha256(data).hexdigest()))
    return renders
