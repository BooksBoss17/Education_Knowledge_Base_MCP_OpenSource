"""Model-independent formula output validation and runtime protocol."""
from __future__ import annotations
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

class OcrVerdict(str, Enum):
    VALID = "OCR_VALID"
    VALID_WITH_WARNING = "OCR_VALID_WITH_WARNING"
    SUSPICIOUS = "OCR_SUSPICIOUS"
    INVALID = "OCR_INVALID"
    EMPTY = "OCR_EMPTY"
    INFERENCE_FAILED = "OCR_INFERENCE_FAILED"


@dataclass(frozen=True)
class OcrValidation:
    verdict: OcrVerdict
    issues: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict.value, "issues": list(self.issues)}


class FormulaOcrOutputValidator:
    """Validate raw OCR LaTeX without repairing or normalizing model output."""

    _empty_fraction = re.compile(
        r"\\(?:d?frac|tfrac)\s*\{\s*\}|"
        r"\\(?:d?frac|tfrac)\s*\{[^{}]*\}\s*\{\s*\}"
    )
    _empty_root = re.compile(r"\\sqrt(?:\s*\[[^\]]*\])?\s*\{\s*\}")
    _empty_script = re.compile(r"[_^]\s*\{\s*\}")
    _unicode_script = re.compile(
        r"[_^]\s*\{(?!\s*\\(?:text|mathrm)\b)[^{}]*[\u3400-\u9fff][^{}]*\}"
    )
    _begin = re.compile(r"\\begin\s*\{([^{}]+)\}")
    _end = re.compile(r"\\end\s*\{([^{}]+)\}")
    _sized_delimiter = re.compile(
        r"\\(left|right)\s*(?:\\[A-Za-z]+|\\[{}]|\.|[()\[\]{}|])"
    )

    def validate(self, raw_latex: str | None) -> OcrValidation:
        if raw_latex is None or not raw_latex.strip():
            return OcrValidation(
                OcrVerdict.EMPTY,
                (self._issue("EMPTY_OUTPUT", "Formula recognizer returned no LaTeX."),),
            )
        issues: list[dict[str, Any]] = []
        if any(ord(char) < 32 and char not in "\t\r\n" for char in raw_latex):
            issues.append(
                self._issue(
                    "CONTROL_CHARACTER",
                    "Output contains a non-whitespace control character.",
                )
            )
        delimiter_error = self._delimiter_error(raw_latex)
        if delimiter_error:
            issues.append(self._issue("UNMATCHED_DELIMITER", delimiter_error))
        begins = self._begin.findall(raw_latex)
        ends = self._end.findall(raw_latex)
        if begins != ends:
            issues.append(
                self._issue(
                    "UNMATCHED_ENVIRONMENT",
                    "LaTeX begin/end environments are not balanced and ordered.",
                )
            )
        if issues:
            return OcrValidation(OcrVerdict.INVALID, tuple(issues))

        suspicious: list[dict[str, Any]] = []
        if self._empty_fraction.search(raw_latex):
            suspicious.append(
                self._issue(
                    "EMPTY_FRACTION_OPERAND",
                    "A fraction numerator or denominator is empty.",
                )
            )
        if self._empty_root.search(raw_latex):
            suspicious.append(
                self._issue("EMPTY_RADICAND", "A square-root radicand is empty.")
            )
        if self._empty_script.search(raw_latex):
            suspicious.append(
                self._issue(
                    "EMPTY_SCRIPT_OPERAND", "A subscript or superscript is empty."
                )
            )
        if suspicious:
            return OcrValidation(OcrVerdict.SUSPICIOUS, tuple(suspicious))

        warnings: list[dict[str, Any]] = []
        if self._unicode_script.search(raw_latex):
            warnings.append(
                self._issue(
                    "UNICODE_TEXT_IN_SCRIPT",
                    "CJK text in a script is not wrapped in a text-style command.",
                    severity="warning",
                )
            )
        if warnings:
            return OcrValidation(OcrVerdict.VALID_WITH_WARNING, tuple(warnings))
        return OcrValidation(OcrVerdict.VALID, ())

    @staticmethod
    def _issue(code: str, message: str, *, severity: str = "error") -> dict[str, str]:
        return {"code": code, "message": message, "severity": severity}

    @classmethod
    def _delimiter_error(cls, value: str) -> str | None:
        value = cls._sized_delimiter.sub("", value)
        # A crop can end inside surrounding prose parentheses. Preserve the
        # transcription without requiring printable delimiters to pair up;
        # LaTeX grouping braces must still be structurally complete.
        pairs = {"}": "{"}
        stack: list[str] = []
        escaped = False
        for char in value:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "{":
                stack.append(char)
            elif char == "}" and (not stack or stack.pop() != pairs[char]):
                return f"Unexpected closing delimiter {char!r}."
        if stack:
            return f"Unclosed delimiter {stack[-1]!r}."
        return None


class FormulaNetRuntime(Protocol):
    load_seconds: float

    def fingerprint(self) -> dict[str, Any]: ...

    def predict(self, paths: Sequence[Path], *, batch_size: int) -> list[str]: ...

    def peak_gpu_memory_bytes(self) -> int | None: ...


def benchmark_cache_key(
    png_sha256: str, runtime_fingerprint: dict[str, Any], inference_config: dict[str, Any]
) -> str:
    payload = {
        "png_content_sha256": png_sha256,
        "runtime_fingerprint": runtime_fingerprint,
        "inference_config": inference_config,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


