"""Remove an external prose-only top strip, with exact source-paint guards.

No detector threshold or pixel-density heuristic is used. Unclaimed native
labels and even a single non-white graphics pixel make the crop abstain.
"""
from copy import deepcopy
import time


def intersects(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def annotate_recovered_prose(routes, source_evidence, page_record):
    if page_record.get("native_text_trust") != "HIGH":
        return
    native = {r["evidence_id"]: r for r in source_evidence.get("native_text", [])
              if r.get("evidence_id") and r.get("bbox_pdf_pt") and r.get("text", "").strip()}
    owners = {}
    for route in routes:
        p = route.get("provenance", {})
        ids = p.get("native_text_evidence_ids", [])
        # Only a multiline paragraph independently assigned to a canonical
        # native text route. Captions/titles and isolated labels do not qualify.
        if (route.get("adapter") == "NATIVE_TEXT_BRIDGE"
                and "MODEL_CANONICAL_REGION" in p.get("candidate_kinds", [])
                and set(route.get("semantic_evidence", [])) <= {"TEXT", "TEXT_LIKE", "BODY"}
                and len([i for i in ids if i in native]) >= 2):
            for identity in ids:
                if identity in native:
                    owners[identity] = route["route_id"]
    for route in routes:
        p = route.get("provenance", {})
        box = p.get("bbox_pdf_pt")
        if (route.get("output_kind") != "IMAGE" or route.get("adapter") != "IMAGE_RENDER_CROP"
                or p.get("candidate_kinds") != ["MODEL_CANONICAL_REGION"] or not box):
            continue
        claimed = [r for identity, r in native.items() if identity in owners and intersects(box, r["bbox_pdf_pt"])
                   and (r["bbox_pdf_pt"][0] < box[0] or r["bbox_pdf_pt"][2] > box[2])]
        if claimed:
            p["prose_edge_candidates"] = {
                "version": "source-owned-prose-edge-v1",
                "claimed_lines": [{"evidence_id": r["evidence_id"], "owner_route_id": owners[r["evidence_id"]],
                                   "bbox_pdf_pt": list(r["bbox_pdf_pt"])} for r in claimed],
                "all_native_lines": [{"evidence_id": identity, "bbox_pdf_pt": list(r["bbox_pdf_pt"])}
                                     for identity, r in native.items()],
            }


def refine_prose_edge(route, crop, cropper):
    started = time.perf_counter()
    revised, result = _refine_prose_edge(route, crop, cropper)
    proof = revised.get("provenance", {}).get("prose_edge_refinement")
    if proof is not None:
        proof["wall_seconds"] = time.perf_counter() - started
    return revised, result


def _refine_prose_edge(route, crop, cropper):
    candidates = route.get("provenance", {}).get("prose_edge_candidates")
    if not candidates:
        return route, crop
    import fitz
    from .graphics_render import render_source_graphics

    proof = {"version": "source-owned-prose-edge-v1", "status": "UNCHANGED",
             "claimed_lines": deepcopy(candidates["claimed_lines"])}
    revised = deepcopy(route)
    revised["provenance"]["prose_edge_refinement"] = proof
    original = list(route["provenance"]["bbox_pdf_pt"])
    rendered = list(crop["bbox_pdf_pt"])
    dpi = crop["dpi"]
    cut = max(line["bbox_pdf_pt"][3] for line in candidates["claimed_lines"]) + 2 * 72 / dpi
    if not rendered[1] < cut < original[3]:
        proof["reason"] = "NO_VALID_TOP_STRIP"
        return revised, crop
    band = [rendered[0], rendered[1], rendered[2], cut]
    claimed = {r["evidence_id"] for r in candidates["claimed_lines"]}
    if any(r["evidence_id"] not in claimed and intersects(band, r["bbox_pdf_pt"])
           for r in candidates["all_native_lines"]):
        proof["reason"] = "UNCLAIMED_NATIVE_TEXT_IN_STRIP"
        return revised, crop
    try:
        with fitz.open(route["provenance"]["source_path"]) as document:
            pixmap, guard = render_source_graphics(document[route["page_index"]], band, dpi,
                                                  omit_native_prose=True)
        if not all(value == 255 for value in pixmap.samples):
            proof["reason"] = "SOURCE_GRAPHICS_IN_STRIP"
            return revised, crop
        # Cropper will add its usual padding; make the effective top equal cut.
        refined = [original[0], cut + crop["padding_pt"], original[2], original[3]]
        if refined[1] >= refined[3]:
            proof["reason"] = "NO_CONTENT_AFTER_PADDING"
            return revised, crop
        revised["provenance"]["bbox_pdf_pt"] = refined
        proof.update(status="TRIMMED_TOP_PROSE", original_bbox_pdf_pt=original,
                     refined_bbox_pdf_pt=refined, removed_strip_pdf_pt=band,
                     source_nontext_all_white=True, paint_guard=guard)
        return revised, cropper.render(revised)
    except (OSError, RuntimeError, ValueError) as exc:
        proof.update(status="UNCHANGED", reason="SOURCE_GUARD_UNAVAILABLE", error_type=type(exc).__name__)
        revised["provenance"]["bbox_pdf_pt"] = original
        return revised, crop
