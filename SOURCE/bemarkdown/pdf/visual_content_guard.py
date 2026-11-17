"""Conservative output checks; no model imports or semantic correction."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

# One alternation prevents a single-dollar matcher consuming the inside of $$.
MATH_SPAN = re.compile(
    r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<![\\$])\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)",
    re.DOTALL,
)


def math_key(value: str) -> str:
    """Only lexical equivalences; never infer algebraic equivalence."""
    return re.sub(r"\s+", "", value.replace(r"\left", "").replace(r"\right", ""))


def alphabet_array_without_glyphs(value, crop):
    """Flag an OCR alphabet array when the crop lacks a plausible horizontal row of glyphs."""
    if not isinstance(value, str) or r"\end{array}" not in value:
        return False
    body = re.sub(r"\s+", "", value).strip("$")
    letters = re.findall(r"\\text\{([A-Z])\}", body)
    count = len(letters)
    if count < 6 or "".join(letters) != "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:count]:
        return False
    alphabet = r"\quad".join(r"\text{" + letter + "}" for letter in letters)
    if body not in {r"\begin{array}{" + alignment + "}" + alphabet + r"\end{array}"
                    for alignment in ("c", "l", "r")}:
        return False
    import cv2
    import numpy as np
    from PIL import Image
    with Image.open(crop["path"]) as image:
        if count < 26:
            rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
            if (rgb.max(axis=2) - rgb.min(axis=2) > 40).mean() < 0.15:
                return False
        gray = np.asarray(image.convert("L"))
    for level in (80, 140, 205, 240):
        for ink in (gray < level, gray > level):
            stats = cv2.connectedComponentsWithStats(ink.astype("uint8"))[2][1:]
            shapes = [(x, y + h/2, w, h) for x, y, w, h, area in stats
                      if area >= 4 and h >= 3 and 0.1*h <= w <= 2.2*h]
            if len(shapes) > 200:
                return False  # Complex image: keep the transcription.
            shapes.sort()
            for _, center_y, _, height in shapes:
                end = -1
                aligned = 0
                for x, y, w, h in shapes:
                    if x >= end and abs(y-center_y) < 0.4*height and 0.5*height <= h <= 1.6*height:
                        aligned += 1
                        end = x+w-1
                if aligned >= (count+1)//2:
                    return False
    return True


def output_problem(value: Any, *, formula: bool = False) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "EMPTY_OCR"
    body = re.sub(r"^\s*#{1,6}\s*", "", value).strip()
    if not body or re.fullmatch(r"[$`#\s]+", body):
        return "EMPTY_MARKUP_OUTPUT"
    for match in MATH_SPAN.finditer(body):
        if not next((x for x in match.groups() if x is not None), "").strip():
            return "EMPTY_MATH_BODY"
    remainder = MATH_SPAN.sub("MATH", body)
    if "$$" in remainder or r"\[" in remainder or r"\]" in remainder:
        return "UNPAIRED_MATH_DELIMITER"
    if formula:
        depth = 0
        for brace in re.findall(r"(?<!\\)[{}]", body):
            depth += 1 if brace == "{" else -1
            if depth < 0:
                return "UNBALANCED_FORMULA_BRACES"
        if depth:
            return "UNBALANCED_FORMULA_BRACES"
        if re.search(r"\\(?:frac|sqrt)\s*\{\s*\}", body):
            return "EMPTY_FORMULA_OPERAND"
    if not any(c.isalnum() for c in body) and not any(c in "+−-=<>≤≥±×÷∞∑∫" for c in body):
        return "SYMBOL_ONLY_REVIEW"
    return None


def clean_text_output(value: Any, region: Any, formulas: Sequence[Any],
                      results: Mapping[str, str]) -> tuple[Any, list[dict[str, Any]], int]:
    """Preserve math unless an almost identical source box supplies the same text."""
    if not isinstance(value, str):
        return value, [], 0
    notes: list[dict[str, Any]] = []
    deleted = 0
    x0, y0, x1, y1 = region.bbox
    area = max(0, x1-x0) * max(0, y1-y0)
    companions = []
    for item in formulas:
        candidate = results.get(item.region_id, "")
        if item.page_index != region.page_index or output_problem(candidate, formula=True):
            continue
        a, b, c, d = item.bbox
        intersection = max(0, min(x1, c)-max(x0, a)) * max(0, min(y1, d)-max(y0, b))
        if intersection:
            companions.append((item, candidate, intersection / area if area else 0))

    def replace(match: re.Match[str]) -> str:
        nonlocal deleted
        body = next((x for x in match.groups() if x is not None), "")
        matches = [(item, coverage) for item, result, coverage in companions
                   if math_key(body) and math_key(body) == math_key(result)]
        # A matching expression inside a larger paragraph may be a legitimate
        # second occurrence. Only suppress a whole-output duplicate source box.
        if len(matches) == 1 and matches[0][1] >= 0.98 and value.strip() == match.group(0):
            deleted += 1
            notes.append({"reason": "DUPLICATE_FORMULA_SOURCE_CONFIRMED", "region_id": region.region_id,
                          "formula_region_id": matches[0][0].region_id, "raw_fragment": match.group(0)})
            return ""
        if companions:
            notes.append({"reason": "TEXT_FORMULA_OVERLAP_REVIEW", "region_id": region.region_id,
                          "formula_region_ids": [x[0].region_id for x in companions]})
        return match.group(0)

    return MATH_SPAN.sub(replace, value), notes, deleted


def duplicate_text_regions(regions: Sequence[Any], values: Mapping[str, str]) -> list[dict[str, Any]]:
    """Suppress only source-contained text already present in a retained box."""
    def area(row):
        a,b,c,d=row.bbox
        return max(0,c-a)*max(0,d-b)
    retained = []
    decisions = []
    for row in sorted(regions, key=lambda r:(-area(r),r.reading_order,r.region_id)):
        value = values.get(row.region_id, "")
        key = re.sub(r"\s+", "", value)
        if not key or output_problem(value):
            continue
        a,b,c,d = row.bbox
        for other, other_key in retained:
            if other.page_index != row.page_index or key not in other_key:
                continue
            x0,y0,x1,y1=other.bbox
            intersection=max(0,min(c,x1)-max(a,x0))*max(0,min(d,y1)-max(b,y0))
            coverage=intersection/area(row) if area(row) else 0
            # Accommodate small detector edge differences; the complete text
            # must still occur verbatim (apart from whitespace) in the survivor.
            if coverage >= 0.95:
                decisions.append({"reason":"DUPLICATE_TEXT_SOURCE_CONFIRMED", "region_id":row.region_id,
                                  "retained_region_id":other.region_id, "source_coverage":coverage,
                                  "raw_fragment":value})
                break
        else:
            retained.append((row,key))
    return decisions
