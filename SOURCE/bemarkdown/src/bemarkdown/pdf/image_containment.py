"""Conserve source image evidence while emitting each contained visual only once."""

from __future__ import annotations


def _route(block):
    return block.get("provenance", {}).get("route_provenance", {})


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1]) if box else 0


def _contains(outer, inner):
    return bool(
        outer
        and inner
        and outer[0] <= inner[0] + 0.01
        and outer[1] <= inner[1] + 0.01
        and outer[2] >= inner[2] - 0.01
        and outer[3] >= inner[3] - 0.01
    )


def _rank(block):
    route = _route(block)
    return (
        -_area(block.get("bbox_pdf_pt")),
        route.get("source_candidate_kinds") != ["MODEL_CANONICAL_REGION"],
        block["node_id"],
    )


def _ownership_basis(owner, child):
    if owner["page_index"] != child["page_index"]:
        return None
    outer, inner = _route(owner), _route(child)
    owner_native, child_native = (
        outer.get("native_extraction", {}),
        inner.get("native_extraction", {}),
    )
    owner_ids, child_ids = (
        owner_native.get("source_placement_evidence_ids"),
        child_native.get("source_placement_evidence_ids"),
    )
    if (
        owner_ids
        and owner_ids == child_ids
        and all(owner_ids)
        and owner["content"].get("asset_uid") == child["content"].get("asset_uid")
    ):
        return "SAME_NATIVE_PLACEMENT_AND_BYTES"
    render = outer.get("render_crop", {})
    target = child_native.get("source_bbox_pdf_pt") or inner.get("render_crop", {}).get(
        "bbox_pdf_pt"
    )
    if not _contains(render.get("bbox_pdf_pt"), target):
        return None
    graphics = render.get("source_graphics_render", {})
    if any(
        max(box[0], target[0]) < min(box[2], target[2])
        and max(box[1], target[1]) < min(box[3], target[3])
        for box in graphics.get("semantic_image_excluded_regions_pdf_pt", [])
    ):
        return None
    if not graphics or graphics.get("method") == "ORIGINAL_SOURCE_CROP_RETAINED":
        return "COMPLETE_SOURCE_CROP_CONTAINS_CHILD_PIXELS"
    if any(
        _contains(box, target)
        for box in graphics.get("native_text_preserved_regions_pdf_pt", [])
    ):
        return "PROTECTED_FIGURE_PIXELS_INCLUDED_IN_BACKGROUND"
    if (
        graphics.get("source_image_paint_preserved")
        and child_native.get("source_appearance", {}).get("verified") is True
    ):
        return "VERIFIED_NATIVE_IMAGE_PAINT_INCLUDED_IN_BACKGROUND"
    return None


def deduplicate_page_images(blocks):
    """Require source-paint containment, never mere visual/semantic similarity."""
    images = sorted(
        (block for block in blocks if block.get("kind") == "IMAGE"), key=_rank
    )
    suppressed, owners = [], []
    for child in images:
        match = next(
            (
                (owner, basis)
                for owner in owners
                if (basis := _ownership_basis(owner, child))
            ),
            None,
        )
        if match is None:
            owners.append(child)
            continue
        owner, basis = match
        child["visibility"] = "SUPPRESSED_DUPLICATE"
        child.setdefault("provenance", {})["source_image_containment"] = {
            "version": "source-image-paint-containment-v1",
            "owner_node_id": owner["node_id"],
            "basis": basis,
            "source_evidence_retained": True,
        }
        suppressed.append(child)
    suppressed_ids = {block["node_id"] for block in suppressed}
    return [
        block for block in blocks if block["node_id"] not in suppressed_ids
    ], suppressed


def close_near_contained_image_extents(routes):
    """Expand a source crop at most two points to preserve a nested view's edge."""
    images = sorted(
        (
            route
            for route in routes
            if route.get("output_kind") == "IMAGE"
            and route.get("provenance", {}).get("bbox_pdf_pt")
        ),
        key=lambda route: (
            -_area(route["provenance"]["bbox_pdf_pt"]),
            route["route_id"],
        ),
    )
    for index, owner in enumerate(images):
        original = list(owner["provenance"]["bbox_pdf_pt"])
        current, supports = list(original), []
        for child in images[index + 1 :]:
            box = child["provenance"]["bbox_pdf_pt"]
            placements = child["provenance"].get("native_image_placements", [])
            if (
                child.get("adapter") == "IMAGE_NATIVE_EXTRACT"
                and len(placements) == 1
                and placements[0].get("native_appearance", {}).get("verified") is True
                and placements[0].get("bbox_pdf_pt")
            ):
                # Direct extraction emits the actual painted placement. Its
                # detector box can differ on either side by a few pixels.
                box = placements[0]["bbox_pdf_pt"]
            overlap = _area(
                [
                    max(original[0], box[0]),
                    max(original[1], box[1]),
                    min(original[2], box[2]),
                    min(original[3], box[3]),
                ]
            )
            primitive_in_figure = owner["provenance"].get("candidate_kinds") == [
                "MODEL_CANONICAL_REGION"
            ] and child["provenance"].get("candidate_kinds") == [
                "NATIVE_IMAGE_FALLBACK"
            ]
            # Two points can exceed 2% of a small source primitive such as a
            # plotted dot. Keep the existing absolute bound on every side.
            minimum_overlap = 0.8 if primitive_in_figure else 0.98
            if (
                overlap < minimum_overlap * _area(box)
                or box[0] < original[0] - 2
                or box[1] < original[1] - 2
                or box[2] > original[2] + 2
                or box[3] > original[3] + 2
            ):
                continue
            expanded = [
                min(current[0], box[0]),
                min(current[1], box[1]),
                max(current[2], box[2]),
                max(current[3], box[3]),
            ]
            if expanded != current:
                current = expanded
                supports.append(child["route_id"])
        if supports:
            owner["provenance"]["bbox_pdf_pt"] = current
            owner["provenance"]["source_image_extent_closure"] = {
                "original_bbox_pdf_pt": original,
                "support_route_ids": supports,
                "maximum_extension_pt": 2,
                "source_pixels_added_before_deduplication": True,
            }
            owner["adapter"] = "IMAGE_RENDER_CROP"
            owner["decision_reason_codes"].append(
                "SOURCE_IMAGE_EXTENT_CLOSED_OVER_NESTED_VIEWS"
            )
