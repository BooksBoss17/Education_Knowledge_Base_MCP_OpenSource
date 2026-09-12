"""Conserve overlapping layout formula evidence without repeated presentation."""

import re


def _tokens(latex):
    latex = re.sub(r"\\(?:mathrm|mathit|mathbf|textrm)\{([^{}]*)\}", r"\1", latex)
    latex = re.sub(
        r"\\(?:left|right|displaystyle)\b|\\(?:[,;! ]|quad\b|qquad\b)", "", latex
    )
    return tuple(re.findall(r"\\[A-Za-z]+|\d+(?:\.\d+)?|[^\s]", latex))


def _contains(outer, inner):
    return bool(
        outer
        and inner
        and outer[0] <= inner[0] + 0.25
        and outer[1] <= inner[1] + 0.25
        and outer[2] >= inner[2] - 0.25
        and outer[3] >= inner[3] - 0.25
    )


def deduplicate_page_formulas(blocks):
    formulas = [
        b
        for b in blocks
        if b.get("kind") == "FORMULA"
        and b.get("content", {}).get("latex")
        and b.get("bbox_pdf_pt")
        and b.get("provenance", {})
        .get("route_provenance", {})
        .get("source_candidate_kinds")
        == ["MODEL_CANONICAL_REGION"]
    ]
    formulas.sort(
        key=lambda b: (
            -(b["bbox_pdf_pt"][2] - b["bbox_pdf_pt"][0])
            * (b["bbox_pdf_pt"][3] - b["bbox_pdf_pt"][1]),
            b["node_id"],
        )
    )
    owners, suppressed = [], []
    for child in formulas:
        small = _tokens(child["content"]["latex"])
        owner = None
        for candidate in owners:
            if candidate["page_index"] != child["page_index"] or not _contains(
                candidate["bbox_pdf_pt"], child["bbox_pdf_pt"]
            ):
                continue
            large = _tokens(candidate["content"]["latex"])
            # A full expression must match. Matching a variable inside a bar,
            # exponent or fraction is not evidence that the formulas agree.
            matches = small == large or (
                "=" in small
                and any(
                    large[i : i + len(small)] == small
                    and (i == 0 or large[i - 1] in {",", ";"})
                    and (
                        i + len(small) == len(large)
                        or large[i + len(small)] in {",", ";"}
                    )
                    for i in range(len(large) - len(small) + 1)
                )
            )
            if matches:
                owner = candidate
                break
        if owner is None:
            owners.append(child)
            continue
        child["visibility"] = "SUPPRESSED_DUPLICATE"
        child["provenance"]["source_formula_containment"] = {
            "version": "source-formula-containment-v1",
            "owner_node_id": owner["node_id"],
            "basis": "CONTAINED_LAYOUT_AND_IDENTICAL_COMPLETE_EXPRESSION",
            "source_evidence_retained": True,
        }
        suppressed.append(child)
    ids = {b["node_id"] for b in suppressed}
    return [b for b in blocks if b["node_id"] not in ids], suppressed
