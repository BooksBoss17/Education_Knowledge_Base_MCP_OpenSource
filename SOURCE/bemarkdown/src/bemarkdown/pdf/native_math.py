"""Native PDF math geometry from font spans and original vector glyph paths.

No learned predictions or reference answers participate in proposal extraction.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
import unicodedata
from itertools import pairwise

import pymupdf

NATIVE_MATH_GEOMETRY_VERSION = "native-math-geometry-v1"


def normalize_native_math_character(text, font):
    """Decode legacy Symbol PUA codes only for the named standard Symbol font."""
    if (
        len(text) == 1
        and 0xF020 <= ord(text) <= 0xF0FF
        and re.search(r"(?:^|\+)(?:symbol|symbolmt)$", font.lower())
    ):
        # Adobe Symbol encoding: alphabetic Greek and the prime glyph used by
        # legacy Office PDF exporters. Unknown private codes remain unknown.
        upper = dict(
            zip("ABGDEZHQIKLMNXOPRSTUFCYW", "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ", strict=True)
        )
        lower = dict(
            zip("abgdezhqiklmnxoprstufcyw", "αβγδεζηθικλμνξοπρστυφχψω", strict=True)
        )
        symbol = {**upper, **lower, chr(0xA2): "′", chr(0xB4): "×", chr(0xBB): "≈"}
        text = symbol.get(chr(ord(text) - 0xF000), text)
    return unicodedata.normalize("NFKC", text).translate(
        str.maketrans({"−": "-", "∼": "~", "′": "'", "’": "'", "∆": "Δ", "∶": ":"})
    )


def propose_native_math(page, *, hidden_characters=None):
    from .native_glyph_bounds import character_key
    from .native_visibility import hidden_native_characters

    if hidden_characters is None:
        hidden_characters = hidden_native_characters(page)
    items, prose = [], []
    for block in page.get_text("rawdict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            line_characters = [char for span in line["spans"] for char in span["chars"]
                               if not char.get("origin")
                               or character_key(char["c"], char["origin"]) not in hidden_characters]
            visible_character_ids = {id(char) for char in line_characters}
            line_text = unicodedata.normalize(
                "NFKC", "".join(char["c"] for char in line_characters)
            )
            label = re.match(
                r"\s*(?:【[^】]+】\s*)?(?:[A-D]{1,4}[.、]|\d{1,3}[.、](?=\s|[A-D\u3400-\u9fff]))\s*",
                line_text,
            )
            label_length = label.end() if label else 0
            if re.fullmatch(r"\s*\d{1,3}\.\s*[A-D]{1,4}\s*", line_text):
                label_length = len(line_characters)
            label_characters = {id(char) for char in line_characters[:label_length]}
            nonspace = [char for char in line_characters if char["c"].strip()]
            quote_pairs = {"“": "”", "‘": "’", '"': '"', "'": "'"}
            quoted_characters = {
                id(middle)
                for left, middle, right in zip(nonspace, nonspace[1:], nonspace[2:])
                if quote_pairs.get(left["c"]) == right["c"]
            }
            for span in line["spans"]:
                font = span["font"].lower()
                math_font = bool(span["flags"] & 2) or any(
                    name in font for name in ("symbol", "math", "cmmi")
                )
                for char in span["chars"]:
                    if id(char) not in visible_character_ids or id(char) in label_characters:
                        continue
                    original_text = char["c"]
                    text = normalize_native_math_character(original_text, span["font"])
                    box = pymupdf.Rect(char["bbox"])
                    # A quoted operator name such as 用“×”表示 is already native
                    # prose. Promoting it creates a second standalone formula.
                    # Keep its geometry as a barrier, while multi-character
                    # expressions inside quotes remain eligible for math.
                    if id(char) in quoted_characters and text in "=+-×÷±*/<>≈≠≤≥":
                        prose.append(box)
                        continue
                    if re.search(r"[\u3400-\u9fff]", text):
                        prose.append(box)
                    elif re.fullmatch(
                        r"[A-Za-z0-9\u0370-\u03ff=+\-*/^_<>±×÷√∑∫Ω°().:\[\]~']", text
                    ) or (
                        len(text) == 1
                        and unicodedata.category(text) in {"Sm", "Mn", "Me", "Co"}
                    ):
                        seed = (
                            math_font
                            or bool(span["flags"] & 1)
                            or any(
                                name in unicodedata.name(original_text, "")
                                for name in ("MATHEMATICAL", "SUPERSCRIPT", "SUBSCRIPT")
                            )
                            or bool(re.search(r"[\u0370-\u03ff=±×÷√∑∫]", text))
                        )
                        # A math-font colon in ordinary time/ratio text is not
                        # sufficient evidence to route the neighboring digits
                        # through formula OCR. Preserve it in a seeded formula.
                        if unicodedata.category(text) == "Co" or text == ":":
                            seed = False
                        items.append(
                            {
                                "bbox": list(box),
                                "seed": seed,
                                "kind": "NATIVE_CHARACTER",
                                "text": text,
                                "original_text": original_text,
                                "origin_pdf_pt": list(char["origin"]),
                                "superscript_flag": bool(span["flags"] & 1),
                                "size": span["size"],
                                "font": span["font"],
                            }
                        )
    for small in items:
        if not small["text"].isdigit() or small["seed"]:
            continue
        for base in items:
            if (
                base["text"].isdigit()
                and small["size"] <= base["size"] * 0.8
                and -base["size"] * 0.1
                <= small["bbox"][0] - base["bbox"][2]
                <= base["size"] * 0.55
                and small["origin_pdf_pt"][1]
                < base["origin_pdf_pt"][1] - base["size"] * 0.18
            ):
                small["seed"] = True
                break
    native = [pymupdf.Rect(row["bbox"]) for row in items]
    for index, drawing in enumerate(page.get_drawings()):
        box = drawing["rect"]
        if box.height <= 1 and 2 < box.width < page.rect.width * 0.5:
            color = drawing.get("color") or drawing.get("fill")
            if box.width >= 80:
                # Longer bars need numerator and denominator evidence; a long
                # page rule or table border alone cannot connect math rows.
                nearby = [item for item in items if item['kind'] == 'NATIVE_CHARACTER'
                          and box.x0 <= (item['bbox'][0] + item['bbox'][2]) / 2 <= box.x1]
                above = any(0 <= box.y0 - item['bbox'][3] <= item['size'] * 0.5 for item in nearby)
                below = any(0 <= item['bbox'][1] - box.y1 <= item['size'] * 0.5 for item in nearby)
                if not (above and below):
                    continue
            if color and sum(color) / len(color) <= 0.65:
                line_box = pymupdf.Rect(box)
                line_box.y0 -= 0.25
                line_box.y1 += 0.25
                items.append(
                    {
                        "bbox": list(line_box),
                        "seed": False,
                        "kind": "HORIZONTAL_VECTOR_SUPPORT",
                        "drawing_index": index,
                        "size": 10.0,
                    }
                )
            continue
        if not (1 < box.width < 30 and 1 < box.height < 30 and drawing["type"] == "f"):
            continue
        if any(
            (box & char).get_area() >= box.get_area() * 0.65 for char in native + prose
        ):
            continue
        fill = drawing.get("fill")
        if fill and sum(fill) / len(fill) > 0.65:
            continue
        items.append(
            {
                "bbox": list(box),
                # Colored outline artwork alone is not a mathematical glyph.
                # It may still join independently supported native math text;
                # existing layout-model formula routes are unaffected.
                "seed": not fill or max(fill) - min(fill) <= 0.12,
                "kind": "FILLED_VECTOR_PATH",
                "fill_color": list(fill) if fill else None,
                "drawing_index": index,
                "size": max(box.width, box.height),
                "path_segments": len(drawing["items"]),
            }
        )
    parent = list(range(len(items)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, first in enumerate(items):
        a = first["bbox"]
        for j, second in enumerate(items[:i]):
            b = second["bbox"]
            size = max(first["size"], second["size"])
            gap_x = max(0, a[0] - b[2], b[0] - a[2])
            gap_y = max(0, a[1] - b[3], b[1] - a[3])
            overlap_x = min(a[2], b[2]) - max(a[0], b[0])
            overlap_y = min(a[3], b[3]) - max(a[1], b[1])
            horizontal = (
                gap_x <= size * 0.45
                and overlap_y >= min(a[3] - a[1], b[3] - b[1]) * 0.25
            )
            vertical = (
                gap_y <= size * 0.25
                and overlap_x >= min(a[2] - a[0], b[2] - b[0]) * 0.4
            )
            if (first["kind"] == second["kind"] == "NATIVE_CHARACTER"
                    and min(first["size"], second["size"]) > size * 0.83):
                # Separate baselines of ordinary-sized text do not establish a
                # stacked formula. A source fraction bar can connect its rows;
                # smaller script glyphs retain their independent geometry path.
                vertical = False
            if (first['kind'] == second['kind'] == 'NATIVE_CHARACTER'
                    and abs(first['origin_pdf_pt'][1] - second['origin_pdf_pt'][1]) > size * 0.8):
                # A script glyph can touch the next text row's font box. It
                # must not bridge two equations through that accidental overlap.
                horizontal = vertical = False
            if not (horizontal or vertical):
                continue
            union = pymupdf.Rect(a) | pymupdf.Rect(b)
            if any((union & box).get_area() > box.get_area() * 0.4 for box in prose):
                continue
            parent[find(i)] = find(j)
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)
    output = []
    for group in groups.values():
        if not any(item["seed"] for item in group):
            continue
        box = pymupdf.Rect(group[0]["bbox"])
        for item in group[1:]:
            box |= pymupdf.Rect(item["bbox"])
        if not all(math.isfinite(value) for value in box) or box.get_area() <= 0:
            continue
        native_characters = [
            item for item in group if item["kind"] == "NATIVE_CHARACTER"
        ]
        # A literal answer blank remains native text even when its font/position
        # marks it as mathematical. Keep blanks joined to an actual expression.
        if (len(native_characters) >= 3
                and all(item['text'] == '_' for item in native_characters)
                and all(item['kind'] in {'NATIVE_CHARACTER', 'HORIZONTAL_VECTOR_SUPPORT'} for item in group)):
            continue
        if not native_characters and not any(
            item.get("path_segments", 0) >= 6 for item in group
        ):
            continue
        identity = hashlib.sha256(
            json.dumps(group, sort_keys=True).encode()
        ).hexdigest()[:24]
        output.append(
            {
                "geometry_id": f"native-math-{identity}",
                "bbox_pdf_pt": list(box),
                "native_text_diagnostic": "".join(
                    item.get("text", "") for item in group
                ),
                "evidence": group,
                "version": NATIVE_MATH_GEOMETRY_VERSION,
            }
        )
    return sorted(
        output, key=lambda row: (row["bbox_pdf_pt"][1], row["bbox_pdf_pt"][0])
    )


def native_flat_latex(evidence):
    """Read simple native math without OCR; refuse vector or stacked constructs.

    Text and script placement are determined entirely by source characters,
    font sizes and baselines. Missing geometry fails closed to FormulaNet.
    """
    if not evidence or any(item["kind"] != "NATIVE_CHARACTER" for item in evidence):
        return None
    if any(not item.get("origin_pdf_pt") for item in evidence):
        return None
    size = max(item["size"] for item in evidence)
    base = [item for item in evidence if item["size"] >= size * 0.93]
    if not base:
        return None
    baseline = statistics.median(item["origin_pdf_pt"][1] for item in base)
    if any(abs(item["origin_pdf_pt"][1] - baseline) > size * 0.3 for item in base):
        return None
    greek = {
        "α": r"\alpha",
        "β": r"\beta",
        "γ": r"\gamma",
        "δ": r"\delta",
        "ε": r"\epsilon",
        "θ": r"\theta",
        "λ": r"\lambda",
        "μ": r"\mu",
        "ν": r"\nu",
        "π": r"\pi",
        "ρ": r"\rho",
        "σ": r"\sigma",
        "τ": r"\tau",
        "φ": r"\phi",
        "Φ": r"\Phi",
        "ω": r"\omega",
        "Ω": r"\Omega",
        "Δ": r"\Delta",
    }
    operators = {"×": r"\times", "÷": r"\div", "±": r"\pm", "≈": r"\approx"}
    output = []
    for item in sorted(
        evidence, key=lambda row: (row["bbox"][0], row["origin_pdf_pt"][1])
    ):
        text = item["text"]
        unicode_name = unicodedata.name(item.get("original_text", text), "")
        if unicode_name.startswith("MATHEMATICAL") and any(
            style in unicode_name
            for style in (
                "SCRIPT",
                "FRAKTUR",
                "DOUBLE-STRUCK",
                "SANS-SERIF",
                "MONOSPACE",
            )
        ):
            return None
        if text in greek or text in operators:
            token = (greek | operators)[text] + " "
        elif re.fullmatch(r"[A-Za-z0-9=+\-*/().:<>%~']", text):
            token = r"\%" if text == "%" else text
        else:
            return None
        offset = item["origin_pdf_pt"][1] - baseline
        script = ""
        if unicode_name.startswith("SUPERSCRIPT"):
            script = "^"
        elif unicode_name.startswith("SUBSCRIPT"):
            script = "_"
        elif item["size"] < size * 0.93:
            if offset < -size * 0.12:
                script = "^"
            elif offset > size * 0.04:
                script = "_"
            elif item["size"] < size * 0.83:
                return None
        if script:
            if not output:
                return None
            if output[-1][0] == script:
                output[-1][1] += token
            else:
                output.append([script, token])
        else:
            output.append(["", token])
    return "".join(
        f"{role}{{{value.strip()}}}" if role else value for role, value in output
    ).strip()


def native_geometry_latex(evidence):
    """Read flat expressions or one unambiguous source-drawn fraction bar."""
    flat = native_flat_latex(evidence)
    if flat is not None:
        return flat
    bars = [item for item in evidence if item["kind"] == "HORIZONTAL_VECTOR_SUPPORT"]
    characters = [item for item in evidence if item["kind"] == "NATIVE_CHARACTER"]
    if len(bars) != 1 or len(characters) + 1 != len(evidence) or not characters:
        return None
    if any(not item.get("origin_pdf_pt") for item in characters):
        return None
    bar = bars[0]["bbox"]
    height = statistics.median(item["size"] for item in characters)
    if not 0 < bar[3] - bar[1] < height * 0.15:
        return None
    middle = (bar[1] + bar[3]) / 2
    numerator, denominator, before, after = [], [], [], []
    for item in characters:
        box, baseline = item["bbox"], item["origin_pdf_pt"][1]
        center = (box[0] + box[2]) / 2
        if center < bar[0]:
            before.append(item)
        elif center > bar[2]:
            after.append(item)
        elif box[0] < bar[0] - height * 0.2 or box[2] > bar[2] + height * 0.2:
            return None
        elif middle - height * 2 < baseline < middle - height * 0.1:
            numerator.append(item)
        elif middle + height * 0.2 < baseline < middle + height * 2:
            denominator.append(item)
        else:
            return None
    if not numerator or not denominator:
        return None
    # Side expressions must occupy the fraction's main baseline, not another row.
    if any(abs(item["origin_pdf_pt"][1] - middle) > height * 0.8
           for item in before + after if item["size"] >= height * 0.93):
        return None
    parts = [native_flat_latex(items) if items else "" for items in (before, numerator, denominator, after)]
    if any(part is None for part in parts):
        return None
    prefix, top, bottom, suffix = parts
    return prefix + r"\frac{" + top + "}{" + bottom + "}" + suffix


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _split_complete_equation_rows(routes, accepted, source_evidence):
    """Split only independently readable equations in distinct source rows."""
    consumed, records = set(), []
    for route in list(routes):
        peers = [item for item in accepted if route in item[2]]
        if len(peers) < 2 or any(len(item[2]) != 1 for item in peers):
            continue
        if any(not proof.get('native_latex') or '=' not in proof['native_latex']
               for _, proof, _, _ in peers):
            continue
        peers.sort(key=lambda item: item[0]['bbox_pdf_pt'][1])
        boxes = [item[0]['bbox_pdf_pt'] for item in peers]
        original = list(route['provenance']['bbox_pdf_pt'])
        union = list(pymupdf.Rect(boxes[0]))
        for box in boxes[1:]:
            union = list(pymupdf.Rect(union) | pymupdf.Rect(box))
        if (_intersection(union, original) < _area(original) * 0.85
                or _crosses_prose(union, source_evidence)
                or any(box[2] - box[0] < (original[2] - original[0]) * 0.5 for box in boxes)):
            continue
        if any(second[1] < first[3] - min(first[3] - first[1], second[3] - second[1]) * 0.1
               for first, second in pairwise(boxes)):
            continue
        template = copy.deepcopy(route)
        row_ids = []
        for index, (proposal, proof, _, _) in enumerate(peers):
            current = route if index == 0 else copy.deepcopy(template)
            if index:
                digest = hashlib.sha256((template['route_id'] + proposal['geometry_id']).encode()).hexdigest()[:24]
                current.update({
                    'route_id': f'native-math-route-{digest}',
                    'route_role': 'SOURCE_NATIVE_MATH', 'primary_candidate_assignment': False,
                    'input_kind': 'NATIVE_MATH_REGION', 'input_candidate_ids': [],
                    'source_region_ids': [], 'source_unit_ids': [],
                })
                routes.append(current)
            current['provenance']['bbox_pdf_pt'] = list(proposal['bbox_pdf_pt'])
            current['provenance']['native_math_geometry'] = {
                **proof, 'original_layout_bbox_pdf_pt': original,
                'independent_equation_row': index, 'source_layout_route_id': template['route_id'],
            }
            current['decision_reason_codes'].append('INDEPENDENT_SOURCE_EQUATION_ROWS')
            consumed.add(proposal['geometry_id'])
            row_ids.append(current['route_id'])
        records.append({'original_route_id': template['route_id'], 'row_route_ids': row_ids,
                        'original_bbox_pdf_pt': original, 'row_bboxes_pdf_pt': boxes})
    return consumed, records


def refine_native_math_routes(routes, source_evidence, page_record):
    """Promote source-supported math while preserving every primary candidate.

    Image/table regions remain owned by their dedicated routes. Native storage
    background images carry no semantic exclusion authority. Refinements retain
    original layout boxes; new source-geometry routes claim no layout candidates.
    """
    diagnostic = {
        "version": NATIVE_MATH_GEOMETRY_VERSION,
        "added": [],
        "refined": [],
        "excluded": [],
    }
    if page_record.get("native_text_trust") not in {
        "HIGH",
        "MEDIUM",
    } or page_record.get("source_profile") in {"IMAGE_ONLY", "IMAGE_WITH_TEXT_LAYER"}:
        return routes, diagnostic
    proposals = source_evidence.get("native_math_geometry", [])
    if not proposals:
        return routes, diagnostic
    routes = copy.deepcopy(routes)
    formulas = [route for route in routes if route["adapter"] == "FORMULA_RECOGNITION"]
    text_routes = [
        route
        for route in routes
        if route["adapter"] in {"NATIVE_TEXT_BRIDGE", "OCR_TEXT_REGION"}
    ]
    owners = [
        route
        for route in routes
        if route["output_kind"] in {"IMAGE", "TABLE"}
        and route["provenance"].get("candidate_kinds") != ["NATIVE_IMAGE_FALLBACK"]
    ]
    accepted = []
    for proposal in proposals:
        box = proposal["bbox_pdf_pt"]
        if _area(box) <= 0:
            continue
        owners_here = [
            route
            for route in owners
            if _intersection(box, route["provenance"]["bbox_pdf_pt"])
            >= _area(box) * 0.6
        ]
        if owners_here:
            diagnostic["excluded"].append(
                {
                    "geometry_id": proposal["geometry_id"],
                    "reason": "SEMANTIC_IMAGE_OR_TABLE_OWNS_REGION",
                    "owners": [route["route_id"] for route in owners_here],
                }
            )
            continue
        matching_formulas = [
            route
            for route in formulas
            if _intersection(box, route["provenance"]["bbox_pdf_pt"])
            >= min(_area(box), _area(route["provenance"]["bbox_pdf_pt"])) * 0.5
        ]
        matching_text = [
            route
            for route in text_routes
            if _intersection(box, route["provenance"]["bbox_pdf_pt"])
            >= _area(box) * 0.35
        ]
        if not matching_formulas and not matching_text:
            diagnostic["excluded"].append(
                {
                    "geometry_id": proposal["geometry_id"],
                    "reason": "BODY_REGION_SUPPORT_MISSING",
                }
            )
            continue
        # Full native text span boxes include ascenders/descenders and leave a
        # small source crop margin. Vector-only glyphs use their actual ink box.
        proof = {
            "version": NATIVE_MATH_GEOMETRY_VERSION,
            "geometry_id": proposal["geometry_id"],
            "source_bbox_pdf_pt": list(box),
            "source_evidence": copy.deepcopy(proposal["evidence"]),
            "native_latex": native_geometry_latex(proposal["evidence"]),
        }
        accepted.append((proposal, proof, matching_formulas, matching_text))
    split_ids, diagnostic['split_equation_rows'] = _split_complete_equation_rows(
        routes, accepted, source_evidence,
    )
    for proposal, proof, matching_formulas, matching_text in accepted:
        if proposal['geometry_id'] in split_ids:
            continue
        box = proposal["bbox_pdf_pt"]
        if matching_formulas:
            if len(matching_formulas) != 1:
                diagnostic["excluded"].append(
                    {
                        "geometry_id": proposal["geometry_id"],
                        "reason": "AMBIGUOUS_LAYOUT_FORMULA_MATCH",
                    }
                )
                continue
            route = matching_formulas[0]
            peers = [item for item in accepted if route in item[2]]
            if len(peers) != 1:
                diagnostic["excluded"].append(
                    {
                        "geometry_id": proposal["geometry_id"],
                        "reason": "MULTIPLE_NATIVE_COMPONENTS_IN_LAYOUT_FORMULA",
                    }
                )
                continue
            original = list(route["provenance"]["bbox_pdf_pt"])
            if _intersection(box, original) < _area(original) * 0.7:
                proof["native_latex"] = None
            union = list(pymupdf.Rect(original) | pymupdf.Rect(box))
            punctuation = [
                char
                for line in source_evidence.get("native_text", [])
                for span in line.get("spans", [])
                for char in span.get("punctuation_characters", [])
                if box[2] - 0.5 <= char["bbox_pdf_pt"][0] < union[2]
                and box[1]
                <= (char["bbox_pdf_pt"][1] + char["bbox_pdf_pt"][3]) / 2
                <= box[3]
            ]
            if punctuation:
                union[2] = min(char["bbox_pdf_pt"][0] for char in punctuation)
                proof["excluded_native_punctuation"] = copy.deepcopy(punctuation)
            # Do not expand through prose; such conflicts need a separate split.
            if _crosses_prose(union, source_evidence):
                diagnostic["excluded"].append(
                    {
                        "geometry_id": proposal["geometry_id"],
                        "reason": "REFINEMENT_WOULD_INCLUDE_PROSE",
                    }
                )
                continue
            proof["original_layout_bbox_pdf_pt"] = original
            route["provenance"]["bbox_pdf_pt"] = union
            route["provenance"]["native_math_geometry"] = proof
            route["decision_reason_codes"].append(
                "NATIVE_MATH_GEOMETRY_CORROBORATES_CROP"
            )
            diagnostic["refined"].append(route["route_id"])
        else:
            if _crosses_prose(box, source_evidence):
                continue
            identity = {
                "document_id": page_record["document_id"],
                "page_index": page_record["page_index"],
                "geometry_id": proposal["geometry_id"],
                "version": NATIVE_MATH_GEOMETRY_VERSION,
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()[:24]
            route = {
                "route_id": f"native-math-route-{digest}",
                **{key: identity[key] for key in ("document_id", "page_index")},
                "route_role": "SOURCE_NATIVE_MATH",
                "primary_candidate_assignment": False,
                "input_kind": "NATIVE_MATH_REGION",
                "input_candidate_ids": [],
                "source_region_ids": [],
                "source_unit_ids": [],
                "semantic_evidence": ["FORMULA"],
                "page_escalation": "NONE",
                "adapter": "FORMULA_RECOGNITION",
                "output_kind": "FORMULA",
                "decision_reason_codes": ["NATIVE_FONT_AND_VECTOR_MATH_GEOMETRY"],
                "decision_version": NATIVE_MATH_GEOMETRY_VERSION,
                "requires_gpu": True,
                "provenance": {
                    "candidate_kinds": ["NATIVE_MATH_GEOMETRY"],
                    "bbox_pdf_pt": list(box),
                    "evidence_ids": [],
                    "native_text_evidence_ids": [],
                    "native_image_placements": [],
                    "source_path": page_record["source_path"],
                    "native_text_trust": page_record["native_text_trust"],
                    "source_profile": page_record["source_profile"],
                    "native_math_geometry": proof,
                },
            }
            routes.append(route)
            diagnostic["added"].append(route["route_id"])
    return routes, diagnostic


def _crosses_prose(box, source_evidence):
    for line in source_evidence.get("native_text", []):
        for span in line.get("spans", []):
            if (
                re.search(r"[\u3400-\u9fff]", str(span.get("text", "")))
                and _intersection(box, span["bbox_pdf_pt"])
                > _area(span["bbox_pdf_pt"]) * 0.4
            ):
                return True
    return False
