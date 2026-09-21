"""Render source PDF graphics while omitting independently recovered native prose.

Forward MuPDF paint operations rather than reconstructing images from raw bytes:
inline images may depend on decode arrays, soft masks, blend modes and clipping.
Text used as a mask or clipping path remains part of the source graphics.
"""

from __future__ import annotations


def render_source_graphics(
    page, bbox, dpi, *, preserve_regions=(), exclude_regions=(), omit_native_prose=True
):
    import fitz

    if not omit_native_prose:
        clip, matrix = fitz.Rect(bbox), fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        excluded = _exclude_semantic_images(pixmap, clip, matrix, exclude_regions)
        return pixmap, {
            "method": "MUPDF_SOURCE_GRAPHICS_WITH_SEMANTIC_IMAGE_OWNERSHIP",
            "omitted_text_operations": 0,
            "text_masks_and_clipping_preserved": True,
            "source_image_paint_preserved": True,
            "native_text_preserved_regions_pdf_pt": [],
            "semantic_image_excluded_regions_pdf_pt": excluded,
        }

    mupdf = fitz.mupdf
    names = [
        name.removeprefix("use_virtual_")
        for name in dir(mupdf.FzDevice2)
        if name.startswith("use_virtual_") and name != "use_virtual_drop_device"
    ]

    def forward(name):
        operation = {"begin_tile": "begin_tile_id", "end_mask": "end_mask_tr"}.get(
            name, name
        )

        def call(device, *args):
            try:
                if name == "begin_mask":
                    device.mask_depth += 1
                if (
                    name in {"fill_text", "stroke_text", "ignore_text"}
                    and device.mask_depth == 0
                    and omit_native_prose
                ):
                    device.omitted_text_operations += 1
                    return None
                if name == "fill_text":
                    args = list(args)
                    args[4] = [
                        mupdf.floats_getitem(args[4], index)
                        for index in range(mupdf.ll_fz_colorspace_n(args[3]))
                    ]
                result = getattr(mupdf, "ll_fz_" + operation)(
                    device.target.m_internal, *args[1:]
                )
                if name == "end_mask":
                    device.mask_depth -= 1
                return result
            except Exception as exc:  # MuPDF can swallow Python director exceptions.
                device.errors.append(f"{name}: {type(exc).__name__}: {exc}")
                raise

        return call

    device_class = type(
        "SourceGraphicsDevice",
        (mupdf.FzDevice2,),
        {name: forward(name) for name in names},
    )
    clip, matrix = fitz.Rect(bbox), fitz.Matrix(dpi / 72, dpi / 72)
    filtered_list = mupdf.fz_new_display_list(fitz.JM_rect_from_py(page.rect))
    device = device_class()
    device.target = mupdf.fz_new_list_device(filtered_list)
    device.errors, device.mask_depth, device.omitted_text_operations = [], 0, 0
    for name in names:
        getattr(device, "use_virtual_" + name)()
    display_list = page.get_displaylist()
    mupdf.fz_run_display_list(
        display_list.this,
        device,
        mupdf.FzMatrix(),
        fitz.JM_rect_from_py(clip),
        mupdf.FzCookie(),
    )
    mupdf.fz_close_device(device)
    if device.errors or device.mask_depth:
        raise RuntimeError("SOURCE_GRAPHICS_RENDER_FAILED:" + repr(device.errors))
    pixmap = fitz.DisplayList(filtered_list).get_pixmap(matrix=matrix, clip=clip)
    protected = [fitz.Rect(region) & clip for region in preserve_regions]
    protected = [region for region in protected if not region.is_empty]
    if protected:
        source_pixels = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        for region in protected:
            pixmap.copy(source_pixels, fitz.IRect(region * matrix) & pixmap.irect)
    excluded = _exclude_semantic_images(pixmap, clip, matrix, exclude_regions)
    return pixmap, {
        "method": "MUPDF_SOURCE_GRAPHICS_WITHOUT_NATIVE_PROSE",
        "omitted_text_operations": device.omitted_text_operations,
        "text_masks_and_clipping_preserved": True,
        "source_image_paint_preserved": True,
        "native_text_preserved_regions_pdf_pt": [list(region) for region in protected],
        "semantic_image_excluded_regions_pdf_pt": excluded,
    }


def _exclude_semantic_images(pixmap, clip, matrix, regions):
    import fitz

    excluded = [fitz.Rect(region) & clip for region in regions]
    excluded = [region for region in excluded if not region.is_empty]
    for region in excluded:
        rectangle = fitz.IRect(region * matrix) & pixmap.irect
        # PyMuPDF 1.27.2.3's rectangle clear writes one extra scanline. A native
        # copy from an all-white pixmap is bounded and avoids per-pixel Python.
        blank = fitz.Pixmap(pixmap.colorspace, rectangle, pixmap.alpha)
        blank.clear_with(255)
        pixmap.copy(blank, rectangle)
    return [list(region) for region in excluded]


def protect_semantic_images_in_background_crops(routes):
    """Exclude primary visual/specialist regions from background-only fallbacks.

    Formula and table routes retain their own source crops even when their
    recognizers request review. Their pixels must not also become a diagram
    through the original page's native-image fallback.
    """
    """Give canonical figures exclusive ownership of overlapping fallback pixels.

    Keep unique fallback content, but exclude pixels emitted by the complete
    source figure crop. Each figure keeps its native labels unchanged.
    """
    semantic_boxes = []
    semantic_owners = []
    for route in routes:
        provenance = route.get("provenance", {})
        box = provenance.get("bbox_pdf_pt")
        if (
            route.get("output_kind") in {"IMAGE", "FORMULA", "TABLE"}
            and box
            and provenance.get("candidate_kinds") == ["MODEL_CANONICAL_REGION"]
        ):
            # Includes the bounded image cropper's normal raster padding.
            padding = 2 * 72 / 200 if route.get('output_kind') == 'IMAGE' else 0
            placements = provenance.get("native_image_placements", [])
            if (
                route.get("adapter") == "IMAGE_NATIVE_EXTRACT"
                and len(placements) == 1
                and placements[0].get("native_appearance", {}).get("verified") is True
                and placements[0].get("bbox_pdf_pt")
            ):
                box = placements[0]["bbox_pdf_pt"]
                padding = 0
            semantic_boxes.append(
                [
                    box[0] - padding,
                    box[1] - padding,
                    box[2] + padding,
                    box[3] + padding,
                ]
            )
            semantic_owners.append({
                "route_id": route.get("route_id"),
                "document_id": route.get("document_id"),
                "page_index": route.get("page_index"),
                "source_candidate_ids": list(route.get("input_candidate_ids", [])),
                "source_region_ids": list(route.get("source_region_ids", [])),
                "output_kind": route.get("output_kind"),
                "bbox_pdf_pt": list(semantic_boxes[-1]),
            })
    for route in routes:
        provenance = route.get("provenance", {})
        box = provenance.get("bbox_pdf_pt")
        if (
            route.get("output_kind") not in {"IMAGE", "REVIEW"}
            or not box
            or provenance.get("candidate_kinds") not in (
                ["NATIVE_IMAGE_FALLBACK"], ["VECTOR_VISUAL_FALLBACK"]
            )
        ):
            continue
        overlapping = [
            region
            for region in semantic_boxes
            if (
                max(box[0], region[0]) < min(box[2], region[2])
                and max(box[1], region[1]) < min(box[3], region[3])
            )
        ]
        if overlapping:
            if route.get("output_kind") == "IMAGE":
                route["adapter"] = "IMAGE_RENDER_CROP"
            provenance["source_graphics_exclude_regions"] = overlapping
            provenance["source_graphics_ownership"] = {
                "version": "source-graphics-ownership-v1",
                "residual_route_id": route.get("route_id"),
                "excluded_owners": [owner for owner in semantic_owners if owner["bbox_pdf_pt"] in overlapping],
                "remaining_content": "UNCLASSIFIED_RESIDUAL",
                "automatic_suppression_allowed": False,
            }
        provenance.pop("source_graphics_preserve_regions", None)


def native_prose_in_fallback(candidate, source_evidence, page_record):
    """Fallback images do not own native prose; semantic figures keep their text."""
    if (
        candidate.get("candidate_kind") not in {"NATIVE_IMAGE_FALLBACK", "VECTOR_VISUAL_FALLBACK"}
        or page_record.get("native_text_trust") != "HIGH"
    ):
        return False
    box = candidate.get("bbox_pdf_pt")
    if not box:
        return False
    return any(
        row.get("text", "").strip()
        and row.get("bbox_pdf_pt")
        and max(box[0], row["bbox_pdf_pt"][0]) < min(box[2], row["bbox_pdf_pt"][2])
        and max(box[1], row["bbox_pdf_pt"][1]) < min(box[3], row["bbox_pdf_pt"][3])
        for row in source_evidence.get("native_text", [])
    )
