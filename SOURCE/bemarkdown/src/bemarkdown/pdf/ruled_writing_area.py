"""Source-only recognition of empty ruled notebook decoration."""

from copy import deepcopy
from hashlib import sha256
from itertools import pairwise
from statistics import median


def _overlaps(a, b):
    return (
        a
        and b
        and max(a[0], b[0]) < min(a[2], b[2])
        and max(a[1], b[1]) < min(a[3], b[3])
    )


def _contains(a, b):
    return b and a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _areas(evidence):
    vectors = evidence.get("native_vectors", [])
    groups = []
    for row in vectors:
        box = row.get("bbox_pdf_pt")
        if not (
            box
            and box[2] - box[0] >= 180
            and 0 <= box[3] - box[1] <= 1.5
            and row.get("drawing_count") == 1
        ):
            continue
        group = next(
            (
                g
                for g in groups
                if abs(g[0]["bbox_pdf_pt"][0] - box[0]) <= 1
                and abs(g[0]["bbox_pdf_pt"][2] - box[2]) <= 1
            ),
            None,
        )
        if group is None:
            groups.append([row])
        else:
            group.append(row)
    for group in groups:
        if len(group) < 5:
            continue
        group.sort(key=lambda r: r["bbox_pdf_pt"][1])
        boxes = [r["bbox_pdf_pt"] for r in group]
        gaps = [b[1] - a[1] for a, b in pairwise(boxes)]
        spacing = median(gaps)
        if not (12 <= spacing <= 40 and max(abs(g - spacing) for g in gaps) <= 0.8):
            continue
        left = min(b[0] for b in boxes)
        area = [
            max(0, left - 1.5 * spacing),
            boxes[0][1] - 2,
            max(b[2] for b in boxes) + 2,
            boxes[-1][3] + 2,
        ]
        if any(
            r.get("text", "").strip() and _overlaps(area, r.get("bbox_pdf_pt"))
            for r in evidence.get("native_text", [])
        ):
            continue
        rule_ids = {r["evidence_id"] for r in group}
        bindings = []
        invalid = False
        for row in vectors + evidence.get("native_images", []):
            box = row.get("bbox_pdf_pt")
            if row.get("evidence_id") in rule_ids or not _overlaps(area, box):
                continue
            # Any source ink in the writing panel could be handwriting, a
            # diagram, a formula or a table. Only small left binding marks fit.
            if not (
                _contains(area, box)
                and box[2] <= left + 1
                and box[2] - box[0] <= spacing
                and box[3] - box[1] <= spacing
            ):
                invalid = True
                break
            if row in vectors:
                bindings.append(box)
        if invalid or len(bindings) < len(group) - 2:
            continue
        if max(b[0] for b in bindings) - min(b[0] for b in bindings) > 1:
            continue
        yield area, sorted(rule_ids)


def preserve_empty_ruled_writing_areas(routes, source_evidence, page_record):
    """Preserve proven blank writing panels as graphics, with route evidence.

    No prediction text or model confidence is used. Scanned pages, written
    notes, table grids and nonuniform graphics keep their existing routes.
    """
    if page_record.get("native_text_trust") != "HIGH":
        return routes, {"area_count": 0}
    result, audits = list(routes), []
    for area, rule_ids in _areas(source_evidence):
        selected = [
            r
            for r in result
            if _contains(area, r.get("provenance", {}).get("bbox_pdf_pt"))
        ]
        if not selected or not any(
            r["adapter"] in {"OCR_TEXT_REGION", "FORMULA_RECOGNITION"} for r in selected
        ):
            continue
        merged = deepcopy(selected[0])
        merged["route_id"] = (
            "writing-area-"
            + sha256(
                "|".join(sorted(r["route_id"] for r in selected)).encode()
            ).hexdigest()[:20]
        )
        for key in ["input_candidate_ids", "source_region_ids", "source_unit_ids"]:
            merged[key] = sorted({v for r in selected for v in r.get(key, [])})
        merged.update(
            adapter="IMAGE_RENDER_CROP",
            output_kind="IMAGE",
            input_kind="REGION",
            requires_gpu=False,
            semantic_evidence=["SOURCE_EMPTY_RULED_WRITING_AREA"],
            decision_reason_codes=["SOURCE_EMPTY_RULED_WRITING_AREA_PRESERVED"],
        )
        audit = {
            "bbox_pdf_pt": area,
            "source_rule_evidence_ids": rule_ids,
            "replaced_route_ids": [r["route_id"] for r in selected],
            "native_text_present": False,
            "source_writing_panel_ink_present": False,
            "original_route_evidence": selected,
        }
        merged["provenance"] = {
            "bbox_pdf_pt": area,
            "candidate_kinds": ["SOURCE_RULED_WRITING_AREA"],
            "source_path": selected[0]["provenance"].get("source_path"),
            "native_text_trust": "HIGH",
            "source_profile": selected[0]["provenance"].get("source_profile"),
            "evidence_ids": rule_ids,
            "native_text_evidence_ids": [],
            "native_image_placements": [],
            "ruled_writing_area": audit,
        }
        ids = set(audit["replaced_route_ids"])
        result = [r for r in result if r["route_id"] not in ids] + [merged]
        audits.append(
            {k: v for k, v in audit.items() if k != "original_route_evidence"}
        )
    return result, {"area_count": len(audits), "areas": audits}
