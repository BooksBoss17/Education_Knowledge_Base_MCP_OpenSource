from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .formula import FormulaConversion


class FormulaVerdict(StrEnum):
    VALID = "VALID"
    SUSPICIOUS = "SUSPICIOUS"
    INVALID = "INVALID"
    NON_FORMULA_CONTENT = "NON_FORMULA_CONTENT"


@dataclass(frozen=True)
class FormulaIssue:
    code: str
    message: str
    layer: str
    location: str | None = None


@dataclass(frozen=True)
class FormulaValidation:
    verdict: FormulaVerdict
    issues: tuple[FormulaIssue, ...]
    evidence: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return self.verdict == FormulaVerdict.VALID

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "issues": [asdict(issue) for issue in self.issues],
            "evidence": self.evidence,
        }


_TOKEN_TAGS = {"mi", "mn", "mo", "mtext", "ms", "mglyph"}
_STRUCTURAL_TAGS = {
    "mfrac",
    "msqrt",
    "mroot",
    "msub",
    "msup",
    "msubsup",
    "munder",
    "mover",
    "munderover",
    "mtable",
    "mtr",
    "mtd",
    "mfenced",
}
_PUNCTUATION_FRAGMENT = re.compile(r"^[\s.,;:!?。，；：！？、·…]+$")
_BEGIN_ENV = re.compile(r"\\begin\s*\{([^{}]+)\}")
_END_ENV = re.compile(r"\\end\s*\{([^{}]+)\}")


class FormulaStructuralValidator:
    """Conservative gate for converted formula structure.

    The validator never changes the conversion. It only records structural
    defects and decides whether the result may be accepted as final LaTeX.
    """

    def validate(
        self,
        conversion: FormulaConversion,
        *,
        source_metadata: dict[str, Any] | None = None,
    ) -> FormulaValidation:
        metadata = dict(source_metadata or {})
        issues: list[FormulaIssue] = []
        mathml = getattr(conversion, "intermediate", None)
        mathml_evidence: dict[str, Any] = {
            "present": bool(mathml),
            "parsed": None,
            "structural_nodes": [],
            "token_text": "",
        }
        if mathml:
            self._validate_mathml(mathml, issues, mathml_evidence, metadata)

        latex = conversion.latex
        latex_evidence = {
            "present": latex is not None,
            "balanced_delimiters": None,
            "environments_balanced": None,
        }
        if latex is not None:
            self._validate_latex(latex, issues, latex_evidence, metadata)
        elif not mathml:
            self._add(
                issues,
                "NO_STRUCTURAL_RESULT",
                "Conversion produced neither MathML nor LaTeX.",
                "conversion",
            )

        codes = {issue.code for issue in issues}
        if codes & {
            "MALFORMED_MATHML",
            "EMPTY_MATHML_CONTAINER",
            "UNMATCHED_DELIMITER",
            "UNMATCHED_ENVIRONMENT",
            "NO_STRUCTURAL_RESULT",
        }:
            verdict = FormulaVerdict.INVALID
        elif codes & {"PUNCTUATION_FRAGMENT", "NON_FORMULA_TEXT_FRAGMENT"}:
            verdict = FormulaVerdict.NON_FORMULA_CONTENT
        elif issues:
            verdict = FormulaVerdict.SUSPICIOUS
        else:
            verdict = FormulaVerdict.VALID

        return FormulaValidation(
            verdict,
            tuple(issues),
            {
                "latex": latex,
                "mathml": mathml_evidence,
                "latex_checks": latex_evidence,
                "source_metadata": metadata,
            },
        )

    def _validate_mathml(
        self,
        mathml: str,
        issues: list[FormulaIssue],
        evidence: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        try:
            root = ET.fromstring(mathml)
        except ET.ParseError as exc:
            evidence["parsed"] = False
            evidence["parse_error"] = str(exc)
            self._add(
                issues,
                "MALFORMED_MATHML",
                "MathML cannot be parsed as an XML tree.",
                "mathml",
            )
            return

        evidence["parsed"] = True
        nodes = list(root.iter())
        structural = [self._tag(node) for node in nodes if self._tag(node) in _STRUCTURAL_TAGS]
        token_text = "".join(
            "".join(node.itertext()).strip()
            for node in nodes
            if self._tag(node) in _TOKEN_TAGS
        ).strip()
        evidence["structural_nodes"] = structural
        evidence["token_text"] = token_text
        if not self._meaningful(root) and not structural:
            self._add(
                issues,
                "EMPTY_MATHML_CONTAINER",
                "MathML contains no meaningful mathematical token or glyph.",
                "mathml",
                "/math",
            )
            return

        for index, node in enumerate(nodes):
            tag = self._tag(node)
            children = list(node)
            location = f"/{tag}[{index}]"
            if tag == "mfrac":
                self._required(children, 0, "EMPTY_NUMERATOR", "fraction numerator", issues, location)
                self._required(children, 1, "EMPTY_DENOMINATOR", "fraction denominator", issues, location)
                if len(children) != 2:
                    self._incomplete(issues, "A fraction must contain exactly two operands.", location)
            elif tag == "msqrt":
                if not children or not any(self._meaningful(child) for child in children):
                    self._empty_operand(issues, "EMPTY_RADICAND", "square-root radicand", location)
            elif tag == "mroot":
                self._required(children, 0, "EMPTY_RADICAND", "root radicand", issues, location)
                self._required(children, 1, "EMPTY_ROOT_INDEX", "root index", issues, location)
                if len(children) != 2:
                    self._incomplete(issues, "A root must contain a radicand and index.", location)
            elif tag in {"msub", "msup", "munder", "mover"}:
                self._required(children, 0, "EMPTY_BASE", "script base", issues, location)
                self._required(children, 1, "EMPTY_SCRIPT_OPERAND", "script operand", issues, location)
                if len(children) != 2:
                    self._incomplete(issues, f"{tag} must contain exactly two operands.", location)
            elif tag in {"msubsup", "munderover"}:
                self._required(children, 0, "EMPTY_BASE", "script base", issues, location)
                self._required(children, 1, "EMPTY_SCRIPT_OPERAND", "lower script operand", issues, location)
                self._required(children, 2, "EMPTY_SCRIPT_OPERAND", "upper script operand", issues, location)
                if len(children) != 3:
                    self._incomplete(issues, f"{tag} must contain exactly three operands.", location)
            elif tag in {"mtr", "mtd"} and not any(
                self._meaningful(child) for child in children
            ):
                self._incomplete(issues, f"{tag} has no meaningful cell content.", location)

        if not structural and token_text:
            stripped = token_text.strip()
            if _PUNCTUATION_FRAGMENT.fullmatch(stripped):
                self._add(
                    issues,
                    "PUNCTUATION_FRAGMENT",
                    "MathML contains only punctuation rather than a formula.",
                    "mathml",
                )
            elif self._single_cjk(stripped):
                code = (
                    "NON_FORMULA_TEXT_FRAGMENT"
                    if metadata.get("referenced") is False
                    else "TEXT_ONLY_FRAGMENT"
                )
                self._add(
                    issues,
                    code,
                    "Text-only MathML has no mathematical structure.",
                    "mathml",
                )

    def _validate_latex(
        self,
        latex: str,
        issues: list[FormulaIssue],
        evidence: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        text = self._strip_math_wrappers(latex.strip())
        if not text:
            self._add(
                issues,
                "EMPTY_LATEX",
                "Converted LaTeX is empty after removing display wrappers.",
                "latex",
            )
            return
        if _PUNCTUATION_FRAGMENT.fullmatch(text):
            self._add(
                issues,
                "PUNCTUATION_FRAGMENT",
                "Converted output contains punctuation only.",
                "latex",
            )

        standalone_omml_delimiter = (
            metadata.get("source_type") == "omml" and text in {"(", ")", "[", "]"}
        )
        balanced = standalone_omml_delimiter or self._balanced(text)
        evidence["balanced_delimiters"] = balanced
        evidence["standalone_omml_delimiter"] = standalone_omml_delimiter
        if not balanced:
            self._add(
                issues,
                "UNMATCHED_DELIMITER",
                "LaTeX has unmatched braces, brackets, or parentheses.",
                "latex",
            )

        begin = _BEGIN_ENV.findall(text)
        end = _END_ENV.findall(text)
        environments_balanced = begin == end
        evidence["environments_balanced"] = environments_balanced
        if not environments_balanced:
            self._add(
                issues,
                "UNMATCHED_ENVIRONMENT",
                "LaTeX begin/end environments do not match.",
                "latex",
            )

        for command, codes in (
            ("frac", ("EMPTY_NUMERATOR", "EMPTY_DENOMINATOR")),
            ("dfrac", ("EMPTY_NUMERATOR", "EMPTY_DENOMINATOR")),
            ("tfrac", ("EMPTY_NUMERATOR", "EMPTY_DENOMINATOR")),
        ):
            for groups in self._command_groups(text, command, 2):
                for group, code in zip(groups, codes, strict=True):
                    if not group.strip():
                        self._empty_operand(issues, code, code.lower().replace("_", " "), "latex")

        for groups in self._command_groups(text, "sqrt", 1, optional=True):
            if not groups[-1].strip():
                self._empty_operand(issues, "EMPTY_RADICAND", "square-root radicand", "latex")
        if re.search(r"(?:_|\^)\s*\{\s*\}", text):
            self._empty_operand(issues, "EMPTY_SCRIPT_OPERAND", "script operand", "latex")

    def _required(
        self,
        children: list[ET.Element],
        index: int,
        code: str,
        label: str,
        issues: list[FormulaIssue],
        location: str,
    ) -> None:
        if index >= len(children) or not self._meaningful(children[index]):
            self._empty_operand(issues, code, label, location)

    def _empty_operand(
        self,
        issues: list[FormulaIssue],
        code: str,
        label: str,
        location: str,
    ) -> None:
        self._add(issues, code, f"Required {label} is empty.", "structure", location)
        self._add(
            issues,
            "EMPTY_REQUIRED_OPERAND",
            "A required formula operand is empty.",
            "structure",
            location,
        )
        self._incomplete(issues, "Formula structure is incomplete.", location)

    def _incomplete(
        self, issues: list[FormulaIssue], message: str, location: str
    ) -> None:
        self._add(issues, "INCOMPLETE_STRUCTURE", message, "structure", location)

    @staticmethod
    def _meaningful(node: ET.Element) -> bool:
        tag = FormulaStructuralValidator._tag(node)
        if tag == "mglyph":
            return True
        if tag in _TOKEN_TAGS and "".join(node.itertext()).strip():
            return True
        return any(FormulaStructuralValidator._meaningful(child) for child in node)

    @staticmethod
    def _tag(node: ET.Element) -> str:
        return node.tag.rsplit("}", 1)[-1].split(":")[-1]

    @staticmethod
    def _single_cjk(text: str) -> bool:
        return len(text) == 1 and "CJK" in unicodedata.name(text, "")

    @staticmethod
    def _strip_math_wrappers(text: str) -> str:
        for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                return text[len(left) : len(text) - len(right)].strip()
        return text

    @staticmethod
    def _balanced(text: str) -> bool:
        pairs = {"}": "{", "]": "[", ")": "("}
        stack: list[str] = []
        escaped = False
        for char in text:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char in "{[(":
                stack.append(char)
            elif char in "}])" and (
                not stack or stack.pop() != pairs[char]
            ):
                return False
        return not stack

    @staticmethod
    def _command_groups(
        text: str, command: str, count: int, *, optional: bool = False
    ) -> list[tuple[str, ...]]:
        results: list[tuple[str, ...]] = []
        pattern = re.compile(rf"\\{re.escape(command)}(?![A-Za-z])")
        for match in pattern.finditer(text):
            position = match.end()
            while position < len(text) and text[position].isspace():
                position += 1
            if optional and position < len(text) and text[position] == "[":
                _, position = FormulaStructuralValidator._read_group(text, position, "[", "]")
                while position < len(text) and text[position].isspace():
                    position += 1
            groups: list[str] = []
            for _ in range(count):
                if position >= len(text) or text[position] != "{":
                    break
                group, position = FormulaStructuralValidator._read_group(text, position, "{", "}")
                groups.append(group)
                while position < len(text) and text[position].isspace():
                    position += 1
            if len(groups) == count:
                results.append(tuple(groups))
        return results

    @staticmethod
    def _read_group(text: str, start: int, left: str, right: str) -> tuple[str, int]:
        depth = 0
        escaped = False
        for position in range(start, len(text)):
            char = text[position]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == left:
                depth += 1
            elif char == right:
                depth -= 1
                if depth == 0:
                    return text[start + 1 : position], position + 1
        return text[start + 1 :], len(text)

    @staticmethod
    def _add(
        issues: list[FormulaIssue],
        code: str,
        message: str,
        layer: str,
        location: str | None = None,
    ) -> None:
        issue = FormulaIssue(code, message, layer, location)
        if issue not in issues:
            issues.append(issue)
