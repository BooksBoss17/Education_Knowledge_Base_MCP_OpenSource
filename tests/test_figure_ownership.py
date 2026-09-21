"""Residual paint must not masquerade as the figure which owns its caption."""
from copy import deepcopy

from bemarkdown.pdf.graphics_render import protect_semantic_images_in_background_crops
from bemarkdown.pdf_document_ir import _attach_caption_relations


def block(node, box, kind="IMAGE", graphics=None):
    return {"node_id": node, "kind": kind, "bbox_pdf_pt": box, "relations": {},
            "provenance": {"route_provenance": {"render_crop": {"source_graphics_render": graphics or {}}}}}


def test_caption_does_not_attach_to_closer_masked_residual():
    figure = block("figure", [20, 20, 80, 70])
    residue = block("residue", [15, 15, 85, 78], graphics={
        "semantic_image_excluded_regions_pdf_pt": [[20, 20, 80, 70]],
        "post_exclusion_all_white": False})
    caption = block("caption", [20, 80, 80, 90], "CAPTION")
    _attach_caption_relations([figure, residue, caption], 200)
    assert caption["relations"]["caption_for"] == "figure"
    assert caption["provenance"]["caption_candidate_filter"]["rejected"] == [
        {"node_id": "residue", "reason": "MASKED_GRAPHICS_RESIDUAL_NOT_A_CONFIRMED_FIGURE"}]


def test_residual_only_caption_abstains_instead_of_guessing():
    residue = block("residue", [15, 15, 85, 78], graphics={
        "semantic_image_excluded_regions_pdf_pt": [[20, 20, 80, 70]]})
    caption = block("caption", [20, 80, 80, 90], "CAPTION")
    _attach_caption_relations([residue, caption], 200)
    assert caption["relations"]["caption_for"] is None
    assert caption["relations"]["caption_confidence"] == "UNKNOWN"


def test_sparse_unmasked_figure_remains_a_caption_target():
    # No pixel-density heuristic: an arbitrarily sparse real drawing qualifies.
    figure = block("sparse", [20, 20, 80, 70])
    caption = block("caption", [20, 80, 80, 90], "CAPTION")
    _attach_caption_relations([figure, caption], 200)
    assert caption["relations"]["caption_for"] == "sparse"


def test_exclusion_records_actual_owner_route_identities_without_changing_boxes():
    owner = {"route_id": "owner", "document_id": "doc", "page_index": 0,
             "input_candidate_ids": ["candidate-1"], "source_region_ids": ["region-1"],
             "output_kind": "IMAGE", "adapter": "IMAGE_RENDER_CROP",
             "provenance": {"candidate_kinds": ["MODEL_CANONICAL_REGION"], "bbox_pdf_pt": [20, 20, 80, 70]}}
    fallback = {"route_id": "fallback", "output_kind": "IMAGE", "adapter": "IMAGE_RENDER_CROP",
                "provenance": {"candidate_kinds": ["NATIVE_IMAGE_FALLBACK"], "bbox_pdf_pt": [15, 15, 85, 78]}}
    before = deepcopy(owner)
    protect_semantic_images_in_background_crops([owner, fallback])
    assert owner == before
    claims = fallback["provenance"]["source_graphics_ownership"]["excluded_owners"]
    assert len(claims) == 1
    assert claims[0]["route_id"] == "owner"
    assert claims[0]["source_candidate_ids"] == ["candidate-1"]
    assert claims[0]["source_region_ids"] == ["region-1"]
    assert claims[0]["bbox_pdf_pt"] == fallback["provenance"]["source_graphics_exclude_regions"][0]
