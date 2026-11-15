from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from bs4 import BeautifulSoup
from lxml import etree

from ._vendor.omml2latex_parser import convert_omml as _vendored_convert_omml
from .eq import EqParseError, convert_eq


@dataclass
class FormulaConversion:
    latex: str | None
    status: str
    component: str
    warnings: list[str]
    error: str | None = None
    intermediate: str | None = None
    semantic_state: str = "present"


_OMML_CONVERTER = None
_OMML_COMPONENT = None


def _load_omml_converter():
    """Load the audited, minimally patched omml2latex 0.1.1 vendor copy."""
    global _OMML_CONVERTER, _OMML_COMPONENT
    if _OMML_CONVERTER is not None:
        return _OMML_CONVERTER, _OMML_COMPONENT
    _OMML_CONVERTER = _vendored_convert_omml
    _OMML_COMPONENT = "omml2latex 0.1.1 (vendored Python 3.11 fix)"
    return _OMML_CONVERTER, _OMML_COMPONENT


def convert_omml(node: etree._Element) -> FormulaConversion:
    converter, component = _load_omml_converter()
    try:
        xml_node = ET.fromstring(etree.tostring(node))
        latex = converter(xml_node).strip()
        # The vendored converter returns Markdown math delimiters. Formula IR
        # stores only the expression; the serializer owns inline/block wrapping.
        if latex.startswith('$$') and latex.endswith('$$'):
            latex = latex[2:-2].strip()
        elif latex.startswith('$') and latex.endswith('$'):
            latex = latex[1:-1].strip()
        if not latex:
            raise ValueError("converter returned empty LaTeX")
        status = "SUCCESS_NORMALIZED"
        warnings = []
        if component.startswith("bemarkdown"):
            warnings.append(
                "Third-party OMML converter unavailable; structural fallback used"
            )
        return FormulaConversion(latex, status, component, warnings)
    except Exception as exc:  # noqa: BLE001 - third-party adapter failure is data
        return FormulaConversion(None, "FAILED_PRESERVED", component, [], str(exc))


def convert_eq_field(source: str) -> FormulaConversion:
    try:
        result = convert_eq(source)
        status = "SUCCESS_APPROXIMATE" if result.approximate else "SUCCESS_EXACT"
        return FormulaConversion(
            result.latex, status, "bemarkdown EQ recursive parser", result.warnings
        )
    except (EqParseError, ValueError) as exc:
        return FormulaConversion(
            None, "FAILED_PRESERVED", "bemarkdown EQ recursive parser", [], str(exc)
        )


def convert_mtef_ole(ole_data: bytes) -> FormulaConversion:
    try:
        from mathtypejx.mtef import mtef_to_mathml

        mathml = mtef_to_mathml(ole_data)
        if not mathml:
            state = _ole_payload_state(ole_data)
            return FormulaConversion(
                None,
                "FAILED_PRESERVED",
                "mathtypejx 0.1.0 + mathml2latex 0.2.12",
                [],
                "mathtypejx returned no MathML",
                semantic_state=state,
            )
        return convert_mathml(
            mathml,
            component="mathtypejx 0.1.0 + mathml2latex 0.2.12",
        )
    except Exception as exc:  # noqa: BLE001 - third-party adapter failure is data
        return FormulaConversion(
            None,
            "FAILED_PRESERVED",
            "mathtypejx 0.1.0 + mathml2latex 0.2.12",
            [],
            str(exc),
            semantic_state="unknown",
        )


def is_mathml_semantically_empty(mathml: str) -> bool:
    """Return true only when a parsed MathML tree has no semantic content."""

    try:
        root = ET.fromstring(mathml)
    except ET.ParseError:
        return False
    content_elements = {
        "mi",
        "mn",
        "mo",
        "mtext",
        "ms",
        "mglyph",
        "mspace",
        "annotation",
        "annotation-xml",
    }
    for element in root.iter():
        name = element.tag.rsplit("}", 1)[-1].split(":")[-1]
        if name in content_elements and (
            name in {"mglyph", "mspace"} or (element.text or "").strip()
        ):
            return False
    return True


def convert_mathml(mathml: str, *, component: str) -> FormulaConversion:
    if is_mathml_semantically_empty(mathml):
        return FormulaConversion(
            None,
            "EMPTY_SEMANTIC_PAYLOAD",
            component,
            [],
            intermediate=mathml,
            semantic_state="empty",
        )
    try:
        from mathml2latex.mathml import process_mathml

        soup = BeautifulSoup(mathml, "xml")
        latex = (process_mathml(soup) or "").strip()
        if not latex:
            raise ValueError("mathml2latex returned empty LaTeX")
        return FormulaConversion(
            latex,
            "SUCCESS_NORMALIZED",
            component,
            [],
            intermediate=mathml,
            semantic_state="present",
        )
    except Exception as exc:  # noqa: BLE001
        return FormulaConversion(
            None,
            "FAILED_PRESERVED",
            component,
            [],
            str(exc),
            intermediate=mathml,
            semantic_state="corrupted",
        )


def convert_mtef_payload(payload: bytes) -> FormulaConversion:
    """Reuse mathtypejx's parser for raw MTEF recovered from a WMF comment."""

    component = "WMF MFCOMMENT + mathtypejx 0.1.0 + mathml2latex 0.2.12"
    try:
        from mathtypejx.mtef.direct_mathml import build_mathml
        from mathtypejx.mtef.mathml import _skip_stream_header
        from mathtypejx.mtef.records3 import parse_equation_v3
        from mathtypejx.mtef.records5 import parse_equation
        from mathtypejx.mtef.stream import ByteStream

        if not payload or payload[0] not in {3, 5}:
            raise ValueError("embedded payload is not MTEF v3/v5")
        version = payload[0]
        stream = ByteStream(payload)
        _skip_stream_header(stream, version)
        record_tree = (
            parse_equation_v3(stream) if version == 3 else parse_equation(stream)
        )
        mathml = build_mathml(record_tree)
        if not mathml:
            raise ValueError("mathtypejx returned no MathML for embedded MTEF")
        return convert_mathml(mathml, component=component)
    except Exception as exc:  # noqa: BLE001
        return FormulaConversion(
            None,
            "FAILED_PRESERVED",
            component,
            [],
            str(exc),
            semantic_state="corrupted",
        )


def _ole_payload_state(ole_data: bytes) -> str:
    try:
        import olefile

        ole = olefile.OleFileIO(io.BytesIO(ole_data))
        try:
            return (
                "corrupted"
                if any(
                    ole.exists(name)
                    for name in ("Equation Native", "EquationNative", "Equation")
                )
                else "unknown"
            )
        finally:
            ole.close()
    except Exception:  # noqa: BLE001
        return "unknown"


_CERTAIN_COMMAND = re.compile(
    r"\\(?:d?frac|tfrac|sqrt|sum|prod|int|oint|begin|left|right|alpha|beta|gamma|delta|epsilon|lambda|mu|nu|pi|rho|sigma|theta|phi|omega|vec|overline|underline|mathrm|text)\b"
)
_MATH_SCRIPT = re.compile(r"(?:[A-Za-z0-9}\]])(?:\^|_)(?:\{|[A-Za-z0-9])")
_PLAIN_EQUATION_OPERATOR = re.compile(r"(?:=|≈|≠|≤|≥|<|>)")


def detect_latex_alt(value: str | None) -> tuple[str, str | None]:
    if not value:
        return "not_latex", None
    text = value.strip()
    wrappers = [("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)")]
    for left, right in wrappers:
        if (
            text.startswith(left)
            and text.endswith(right)
            and len(text) > len(left) + len(right)
        ):
            return "certain_latex", text[len(left) : -len(right)].strip()
    if _CERTAIN_COMMAND.search(text):
        return "certain_latex", text
    if _MATH_SCRIPT.search(text) and not re.search(r"[\u4e00-\u9fff]{2,}", text):
        return "likely_latex", text
    if (
        len(text) <= 256
        and _PLAIN_EQUATION_OPERATOR.search(text)
        and not re.search(r"[\u4e00-\u9fff]", text)
        and not re.search(r"[A-Za-z]{4,}", text)
        and len(re.findall(r"[A-Za-z0-9Α-Ωα-ω]", text)) >= 2
    ):
        return "likely_latex", text
    return "not_latex", None


def _fallback_omml(node: ET.Element) -> str:
    """Small deterministic fallback for common OOXML math structures."""

    def tag(element):
        return element.tag.rsplit("}", 1)[-1]

    def children(element, name):
        return [child for child in element if tag(child) == name]

    def first(element, name):
        return next((child for child in element if tag(child) == name), None)

    def render(element):
        name = tag(element)
        if name in {
            "oMath",
            "oMathPara",
            "e",
            "num",
            "den",
            "deg",
            "sub",
            "sup",
            "lim",
            "fName",
            "mr",
        }:
            return "".join(render(child) for child in element)
        if name == "t":
            return element.text or ""
        if name == "r":
            return "".join(render(child) for child in element if tag(child) != "rPr")
        if name == "f":
            return rf"\frac{{{render(first(element, 'num'))}}}{{{render(first(element, 'den'))}}}"
        if name == "rad":
            degree, body = render(first(element, "deg")), render(first(element, "e"))
            return rf"\sqrt[{degree}]{{{body}}}" if degree else rf"\sqrt{{{body}}}"
        if name == "sSup":
            return rf"{render(first(element, 'e'))}^{{{render(first(element, 'sup'))}}}"
        if name == "sSub":
            return rf"{render(first(element, 'e'))}_{{{render(first(element, 'sub'))}}}"
        if name == "sSubSup":
            return rf"{render(first(element, 'e'))}_{{{render(first(element, 'sub'))}}}^{{{render(first(element, 'sup'))}}}"
        if name == "nary":
            prop = first(element, "naryPr")
            char_node = first(prop, "chr") if prop is not None else None
            char = (
                next(iter(char_node.attrib.values()), "∫")
                if char_node is not None
                else "∫"
            )
            operators = {"∫": "\\int", "∑": "\\sum", "∏": "\\prod", "∮": "\\oint"}
            return (
                operators.get(char, char)
                + rf"_{{{render(first(element, 'sub'))}}}^{{{render(first(element, 'sup'))}}}"
                + render(first(element, "e"))
            )
        if name == "d":
            return rf"\left({render(first(element, 'e'))}\right)"
        if name == "m":
            rows = [
                " & ".join(render(cell) for cell in children(row, "e"))
                for row in children(element, "mr")
            ]
            return "\\begin{matrix}" + " \\\\ ".join(rows) + "\\end{matrix}"
        if name == "eqArr":
            return (
                "\\begin{aligned}"
                + " \\\\ ".join(render(item) for item in children(element, "e"))
                + "\\end{aligned}"
            )
        return "".join(render(child) for child in element)

    return render(node)
