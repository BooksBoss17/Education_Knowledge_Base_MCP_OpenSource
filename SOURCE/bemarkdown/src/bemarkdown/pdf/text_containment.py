"""Conserve overlapping OCR evidence without repeating its visible text."""


def _projection(block):
    return block.get("provenance", {}).get("route_provenance", {}).get("ordering_atom_projection", {})


def _same_line_geometry(left, right):
    width_a, height_a = left[2] - left[0], left[3] - left[1]
    width_b, height_b = right[2] - right[0], right[3] - right[1]
    if min(width_a, height_a, width_b, height_b) <= 0:
        return False
    overlap_x = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    overlap_y = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    center_y_delta = abs((left[1] + left[3] - right[1] - right[3]) / 2)
    return (overlap_x / max(width_a, width_b) >= 0.98
            and overlap_y / min(height_a, height_b) >= 0.90
            and overlap_y / max(height_a, height_b) >= 0.80
            and center_y_delta <= max(1.0, min(height_a, height_b) * 0.10))


def deduplicate_page_text(blocks):
    candidates = [block for block in blocks
                  if block.get("kind") == "TEXT"
                  and block.get("content", {}).get("text", "").strip()
                  and len(block.get("bbox_pdf_pt") or []) == 4
                  and _projection(block).get("origin_type") == "OCR_RESOLVED_LINE"
                  and _projection(block).get("parent_content_id")]
    candidates.sort(key=lambda block: (block["page_index"], block["bbox_pdf_pt"][1],
                                       block["bbox_pdf_pt"][0], block["node_id"]))
    owners = {}
    suppressed = []
    for child in candidates:
        key = (child["page_index"], child["content"]["text"].strip())
        matches = owners.setdefault(key, [])
        owner = next((candidate for candidate in matches
                      if _projection(candidate)["parent_content_id"] != _projection(child)["parent_content_id"]
                      and _same_line_geometry(candidate["bbox_pdf_pt"], child["bbox_pdf_pt"])), None)
        if owner is None:
            matches.append(child)
            continue
        child["visibility"] = "SUPPRESSED_DUPLICATE"
        child.setdefault("provenance", {})["source_text_containment"] = {
            "version": "overlapping-ocr-line-containment-v1",
            "owner_node_id": owner["node_id"],
            "basis": "SAME_TEXT_AND_SAME_PHYSICAL_LINE_FROM_DIFFERENT_OCR_REGIONS",
            "owner_bbox_pdf_pt": list(owner["bbox_pdf_pt"]),
            "duplicate_bbox_pdf_pt": list(child["bbox_pdf_pt"]),
            "source_evidence_retained": True,
        }
        suppressed.append(child)
    suppressed_ids = {block["node_id"] for block in suppressed}
    return [block for block in blocks if block["node_id"] not in suppressed_ids], suppressed
