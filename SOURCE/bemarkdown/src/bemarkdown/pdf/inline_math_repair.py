"""Local math candidates with exact surrounding-text preservation.

This module does not run models or claim recognition accuracy. A formatted
GOT candidate must align to the existing body and independently match a local
FormulaNet crop before that one span can be replaced. Source acceptance still
requires evaluation. Unsupported mathematics and disagreements remain pending.
"""

from dataclasses import dataclass
import hashlib
import re
import unicodedata


GREEK = dict(zip(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega".split(),
    "αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ",
    strict=True,
))


@dataclass(frozen=True)
class SimpleMath:
    base: str
    pre_sub: str | None = None
    pre_sup: str | None = None
    post_sub: str | None = None
    post_sup: str | None = None

    @property
    def structured(self):
        return self.base in GREEK.values() or any((self.pre_sub, self.pre_sup, self.post_sub, self.post_sup))

    @property
    def latex(self):
        base = next(("\\" + name for name, value in GREEK.items() if value == self.base), self.base)
        prefix = "{}" if self.pre_sub or self.pre_sup else ""
        if self.pre_sub:
            prefix += "_{" + self.pre_sub + "}"
        if self.pre_sup:
            prefix += "^{" + self.pre_sup + "}"
        if self.pre_sub or self.pre_sup:
            base = r"\mathrm{" + base + "}"
        suffix = ("_{" + self.post_sub + "}" if self.post_sub else "") + ("^{" + self.post_sup + "}" if self.post_sup else "")
        return prefix + base + suffix


def parse_simple_math(value):
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", "", value)
    while text.startswith("{}"):
        text = text[2:]
    cursor = 0

    def scripts():
        nonlocal cursor
        values = {}
        while cursor < len(text) and text[cursor] in "_^":
            mark = text[cursor]
            cursor += 1
            if mark in values:
                raise ValueError
            if cursor < len(text) and text[cursor] == "{":
                end = text.find("}", cursor + 1)
                if end < 0:
                    raise ValueError
                content = text[cursor+1:end]
                cursor = end+1
            else:
                content = text[cursor:cursor+1]
                cursor += 1
            if not re.fullmatch(r"[A-Za-z0-9]{1,16}", content):
                raise ValueError
            values[mark] = content
        return values

    try:
        before = scripts()
        match = re.match(r"\\(?:mathrm|mathbf|mathit|mathtt)\{([A-Za-z]{1,2})\}|\\([A-Za-z]+)|([A-Za-z]{1,2}|[0-9]+|[\u0370-\u03ff])", text[cursor:])
        if not match:
            return None
        cursor += match.end()
        if match.group(2):
            base = GREEK.get(match.group(2))
            if base is None:
                return None
        else:
            base = match.group(1) or match.group(3)
        after = scripts()
        if cursor != len(text):
            return None
        return SimpleMath(base, before.get("_"), before.get("^"), after.get("_"), after.get("^"))
    except ValueError:
        return None


def _alignment_text(value):
    text, offsets = [], []
    for index, original in enumerate(value):
        for char in unicodedata.normalize("NFKC", original):
            if char.isspace() or unicodedata.category(char).startswith("P"):
                continue
            text.append(char)
            offsets.append(index)
    return "".join(text), offsets


def _unwrap_text(value):
    value = value.strip()
    if not value.startswith(r"\text{"):
        return value
    depth = 1
    for index, char in enumerate(value[6:], 6):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[6:index] if index == len(value)-1 else None
    return None


def align_formatted_fields(original, formatted):
    """Match all surrounding prose; return only raw offsets, never rewritten prose."""
    formatted = _unwrap_text(formatted)
    if formatted is None or r"\begin" in formatted or r"\end" in formatted:
        return None
    pattern = r"\\\((.*?)\\\)|(?<!\\)\$([^$]+)\$"
    pieces, atoms, cursor = [], [], 0
    for match in re.finditer(pattern, formatted, re.S):
        literal = formatted[cursor:match.start()]
        if "\\" in literal:
            return None
        normalized_literal = _alignment_text(literal)[0]
        if atoms and not normalized_literal:
            return None
        pieces.append(re.escape(normalized_literal))
        atom = parse_simple_math(next(value for value in match.groups() if value is not None))
        if atom is None:
            return None
        pieces.append(f"(?P<m{len(atoms)}>[A-Za-z0-9\\u0370-\\u03ff]{{1,24}}?)")
        atoms.append(atom)
        cursor = match.end()
    tail = formatted[cursor:]
    if "\\" in tail:
        return None
    pieces.append(re.escape(_alignment_text(tail)[0]))
    normalized, offsets = _alignment_text(original)
    matched = re.fullmatch("".join(pieces), normalized)
    if matched is None:
        return None
    fields = []
    for index, atom in enumerate(atoms):
        start, end = matched.span(f"m{index}")
        raw_start, raw_end = offsets[start], offsets[end-1]+1
        # A subscript may have been recognized as a Chinese full stop. Include
        # that glyph only for local image verification, not on text similarity.
        if atom.post_sub and matched.group(f"m{index}") == atom.base and original[raw_end:raw_end+1] == "。":
            raw_end += 1
        fields.append({"field_index": index, "span": [raw_start, raw_end], "candidate_latex": atom.latex, "structured": atom.structured})
    if any(a["span"][1] > b["span"][0] for a, b in zip(fields, fields[1:])):
        return None
    return fields


def build_source_proposals(layout, image, formatted):
    fields = align_formatted_fields(layout.text, formatted)
    if fields is None:
        return []
    proposals = []
    for field in fields:
        if not field["structured"]:
            continue
        atom = parse_simple_math(field['candidate_latex'])
        box = layout.crop_box(image, *field["span"],
                              prefix_sup=bool(atom.pre_sup), prefix_sub=bool(atom.pre_sub))
        proposals.append({**field, "bbox_source_px": box, "status": "READY_FOR_LOCAL_FORMULA" if box else "SOURCE_BOUNDARY_UNRESOLVED"})
    return proposals


def apply_verified_math(original, formatted, proposals, local_predictions, *, alternative_verifier=None):
    result = {"schema": "bemarkdown-inline-math-local-selection-v1", "text": original,
              "original_text_sha256": hashlib.sha256(original.encode()).hexdigest(), "accepted": [], "rejected": []}
    fields = align_formatted_fields(original, formatted)
    if fields is None:
        return {**result, "status": "BODY_ALIGNMENT_FAILED"}
    by_index = {}
    for prediction in local_predictions:
        index = prediction["field_index"]
        if index in by_index:
            raise ValueError("DUPLICATE_LOCAL_FORMULA")
        by_index[index] = prediction
    field_by_index = {f["field_index"]: f for f in fields}
    seen = set()
    for proposal in proposals:
        index = proposal["field_index"]
        if index in seen:
            raise ValueError("DUPLICATE_INLINE_PROPOSAL")
        seen.add(index)
        field = field_by_index.get(index)
        local = by_index.get(index)
        reason = None
        if field is None or not field["structured"]:
            reason = "NO_SUPPORTED_MATHEMATICAL_FIELD"
        elif local is None:
            reason = "LOCAL_FORMULA_PENDING"
        else:
            if local.get("crop_sha256") != proposal.get("crop_sha256") or not re.fullmatch(r"[0-9a-f]{64}", str(proposal.get("crop_sha256", ""))):
                raise ValueError("LOCAL_FORMULA_CROP_SHA_MISMATCH")
            candidate = parse_simple_math(field["candidate_latex"])
            if candidate != parse_simple_math(proposal["candidate_latex"]):
                reason = "CANDIDATE_CHANGED_AFTER_CROP"
            elif candidate != parse_simple_math(local.get("latex")):
                reason = "LOCAL_FORMULA_DISAGREEMENT"
                alternative = alternative_verifier(field, proposal, local) if alternative_verifier else None
                if alternative is not None:
                    if (alternative.get('status') != 'SOURCE_LOCAL_SUPERSCRIPT_CONFIRMED'
                            or alternative.get('source_crop_sha256') != proposal['crop_sha256']
                            or parse_simple_math(alternative.get('context_latex')) != candidate
                            or parse_simple_math(alternative.get('candidate_latex')) != parse_simple_math(local.get('latex'))):
                        raise ValueError('INLINE_ALTERNATIVE_EVIDENCE_MISMATCH')
                    field = {**field, 'context_candidate_latex': field['candidate_latex'],
                             'candidate_latex': alternative['candidate_latex'],
                             'atomic_superscript_evidence': alternative}
                    reason = None
        if reason:
            result["rejected"].append({"field_index": index, "reason": reason})
        else:
            result["accepted"].append({**field, "crop_sha256": proposal["crop_sha256"],
                                       "previous_text": original[slice(*field["span"])], "replacement": "$" + field["candidate_latex"] + "$"})
    for accepted in sorted(result["accepted"], key=lambda f: f["span"][0], reverse=True):
        start, end = accepted["span"]
        result["text"] = result["text"][:start] + accepted["replacement"] + result["text"][end:]
    result["status"] = "LOCAL_MATH_APPLIED" if result["accepted"] else "NO_CONFIRMED_LOCAL_MATH"
    return result
