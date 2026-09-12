"""Check whether extracted bytes preserve an image's actual PDF appearance."""

from __future__ import annotations

import hashlib


def verify_native_image_appearance(page, document, xref, bbox):
    import fitz

    audit = {"version": "native-image-source-pixel-comparison-v1", "verified": False}
    try:
        rect = fitz.Rect(bbox)
        if not xref or page.rotation or rect != rect & page.rect:
            return {**audit, "reason": "NON_NATIVE_ROTATED_OR_CLIPPED_PLACEMENT"}
        extracted = document.extract_image(xref)
        if not extracted or not extracted.get("image") or extracted.get("smask"):
            return {**audit, "reason": "NATIVE_BYTES_OR_MASK_CONTEXT_UNAVAILABLE"}
        if fitz.Pixmap(extracted["image"]).alpha:
            return {**audit, "reason": "SOURCE_ALPHA_REQUIRES_BACKDROP"}
        scale = min(200 / 72, 2048 / max(rect.width, rect.height))
        matrix = fitz.Matrix(scale, scale)
        source = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
        with fitz.open() as isolated:
            image_page = isolated.new_page(
                width=page.rect.width, height=page.rect.height
            )
            image_page.insert_image(
                rect, stream=extracted["image"], keep_proportion=False
            )
            standalone = image_page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
        source_hash = hashlib.sha256(source.samples).hexdigest()
        standalone_hash = hashlib.sha256(standalone.samples).hexdigest()
        verified = (source.width, source.height, source_hash) == (
            standalone.width,
            standalone.height,
            standalone_hash,
        )
        return {
            **audit,
            "verified": verified,
            "reason": "EXACT_SOURCE_PIXELS"
            if verified
            else "SOURCE_PAINT_CONTEXT_CHANGES_PIXELS",
            "source_rgb_sha256": source_hash,
            "standalone_rgb_sha256": standalone_hash,
            "width": source.width,
            "height": source.height,
            "dpi": scale * 72,
        }
    except (RuntimeError, ValueError, fitz.mupdf.FzErrorBase) as exc:
        return {**audit, "reason": "SOURCE_APPEARANCE_CHECK_FAILED", "error": f"{type(exc).__name__}: {exc}"}
