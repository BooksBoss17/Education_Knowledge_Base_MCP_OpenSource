"""Conserve overlapping layout formula evidence without repeated presentation."""

import math
import re


def _tokens(latex):
    latex = re.sub(r"\\(?:mathrm|mathit|mathbf|textrm)\{([^{}]*)\}", r"\1", latex)
    latex = re.sub(
        r"\\(?:left|right|displaystyle)\b|\\(?:[,;! ]|quad\b|qquad\b)", "", latex
    )
    return tuple(re.findall(r"\\[A-Za-z]+|\d+(?:\.\d+)?|[^\s]", latex))


def _contains(outer, inner):
    if not outer or not inner or len(outer) != 4 or len(inner) != 4:
        return False
    if not all(math.isfinite(v) for v in (*outer, *inner)):
        return False
    if any(box[2] <= box[0] or box[3] <= box[1] for box in (outer, inner)):
        return False
    # Preserve the previously supported strict-containment tolerance.
    if (
        outer[0] <= inner[0] + 0.25
        and outer[1] <= inner[1] + 0.25
        and outer[2] >= inner[2] - 0.25
        and outer[3] >= inner[3] - 0.25
    ):
        return True
    # Layout boxes can differ by a fraction of a PDF point. A fixed 0.25 pt
    # limit missed the same complete equation in real overlapping detections.
    # Bound both absolute drift and relative coverage: tiny adjacent formulas
    # must not be swallowed by a tolerance larger than their own geometry.
    tolerance = 0.5
    if not (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] >= inner[2] - tolerance
        and outer[3] >= inner[3] - tolerance
    ):
        return False
    overlap = max(0, min(outer[2], inner[2]) - max(outer[0], inner[0])) * max(
        0, min(outer[3], inner[3]) - max(outer[1], inner[1])
    )
    return overlap / ((inner[2] - inner[0]) * (inner[3] - inner[1])) >= 0.99


def _complete_structured_rhs(large, small, outer, inner):
    """Match an entire root/fraction RHS at the same source right edge.

    Nested subexpressions, bare symbols and neighboring repeated expressions
    are not sufficient evidence for suppressing a second detector result.
    """
    if not small or small[0] not in {r"\sqrt", r"\frac", r"\dfrac", r"\tfrac"}:
        return False
    if abs(outer[2] - inner[2]) > 0.5:
        return False
    depth, equals = 0, []
    for index, token in enumerate(large):
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                return False
        elif token == "=" and depth == 0:
            equals.append(index)
    return depth == 0 and len(equals) == 1 and equals[0] > 0 and large[equals[0] + 1:] == small


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
            rhs_match = _complete_structured_rhs(
                large, small, candidate["bbox_pdf_pt"], child["bbox_pdf_pt"]
            )
            if matches or rhs_match:
                owner = candidate
                basis = ("CONTAINED_LAYOUT_AND_IDENTICAL_COMPLETE_EXPRESSION" if matches
                         else "CONTAINED_LAYOUT_AND_IDENTICAL_COMPLETE_STRUCTURED_RHS")
                break
        if owner is None:
            owners.append(child)
            continue
        child["visibility"] = "SUPPRESSED_DUPLICATE"
        child["provenance"]["source_formula_containment"] = {
            "version": "source-formula-containment-v3",
            "owner_node_id": owner["node_id"],
            "basis": basis,
            "source_evidence_retained": True,
            "max_edge_drift_pdf_pt": 0.5,
            "minimum_child_area_coverage_for_extended_tolerance": 0.99,
        }
        suppressed.append(child)
    ids = {b["node_id"] for b in suppressed}
    return [b for b in blocks if b["node_id"] not in ids], suppressed
