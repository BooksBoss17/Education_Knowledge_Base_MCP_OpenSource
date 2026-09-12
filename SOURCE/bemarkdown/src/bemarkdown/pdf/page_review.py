"""Source geometry and pixel evidence for page-wide review control records."""

import hashlib


def inspect_page_review_scope(route):
    import fitz

    with fitz.open(route["provenance"]["source_path"]) as document:
        page = document[int(route["page_index"])]
        pixels = page.get_pixmap(dpi=144, colorspace=fitz.csRGB, alpha=False)
        samples = pixels.samples
        return {
            "basis": "FULL_SOURCE_PAGE_RENDER",
            "bbox_pdf_pt": list(page.rect),
            "dpi": 144,
            "render_width": pixels.width,
            "render_height": pixels.height,
            "render_rgb_sha256": hashlib.sha256(samples).hexdigest(),
            "all_pixels_white": bool(samples) and samples.count(255) == len(samples),
        }
