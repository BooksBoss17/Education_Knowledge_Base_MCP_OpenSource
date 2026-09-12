"""Production-owned ordinary-text recognition contracts.

This module is the authority for the frozen OCRTextNormalization v1 behavior.
Benchmarks and validation import this interface; production never imports them.
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "ocr-text-normalization-v1"

_SPACE_AROUND_OPERATOR = re.compile(r"\s*([=+\-×÷/%:;])\s*")
_NUMBER = re.compile(r"[+\-]?\d+(?:\.\d+)?")
_UNIT = re.compile(
    r"(?<![A-Za-z])(?:m/s(?:²|2)?|m·s[⁻-]?\d*|kg|mol|Hz|Pa|J|W|V|A|Ω|N|C|T|s|m|g|K|%)(?![A-Za-z])"
)
_GREEK = re.compile(r"[\u0370-\u03ff]")
_SUBSCRIPT = re.compile(r"[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ]")
_SUPERSCRIPT = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿ]")
_OPTION = re.compile(r"(?<![A-Za-z])[A-D](?:[.、)])?(?![A-Za-z])")
_INLINE_EQUATION = re.compile(r"[^\s，。；;]{0,16}[=+\-×÷][^\s，。；;]{0,16}")
_VECTOR = re.compile(r"(?:[⃗→]|\\vec\s*\{?\w+\}?)")


def normalize_text(value: str | None) -> str:
    """Apply only Unicode, line-ending, whitespace, and safe punctuation folding."""

    if value is None:
        return ""
    value = (
        unicodedata.normalize("NFC", str(value))
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    folded = []
    for char in value:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            folded.append(chr(code - 0xFEE0))
        elif char in {"\u00a0", "\u2007", "\u202f", "\u3000"}:
            folded.append(" ")
        else:
            folded.append(char)
    lines = []
    for line in "".join(folded).split("\n"):
        line = re.sub(r"[\t \f\v]+", " ", line).strip()
        line = _SPACE_AROUND_OPERATOR.sub(r"\1", line)
        line = re.sub(r"\b([A-D][.、)])\s+", r"\1", line)
        line = re.sub(r"(?<=[\u3400-\u9fff]) (?=\S)", "", line)
        line = re.sub(r"(?<=\S) (?=[\u3400-\u9fff])", "", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_sensitive_tokens(value: str) -> dict[str, list[str]]:
    """Extract frozen material-risk tokens without interpreting or correcting them."""

    value = normalize_text(value)
    numbers = _NUMBER.findall(value)
    return {
        "numbers": numbers,
        "decimal_points": [token for token in numbers if "." in token],
        "signs": re.findall(r"(?<!\w)[+\-](?=\d|\w)", value),
        "units": _UNIT.findall(value),
        "Greek_letters": _GREEK.findall(value),
        "subscripts": _SUBSCRIPT.findall(value),
        "superscripts": _SUPERSCRIPT.findall(value),
        "option_labels": _OPTION.findall(value),
        "equation_like_inline_tokens": _INLINE_EQUATION.findall(value),
        "vector_notation": _VECTOR.findall(value),
    }
