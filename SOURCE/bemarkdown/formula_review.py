"""Conservative, model-independent review of native formula predictions."""
from __future__ import annotations
import time
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Sequence
from .formula_contract import FormulaOcrOutputValidator, OcrVerdict, OcrValidation
from .ir import DocumentIR, FormulaNode, HyperlinkNode, TableNode
FORMULA_OCR_CONTRACT_VERSION = "bemarkdown-formula-ocr-v1"

class SafetyGateVerdict(str, Enum):
    ACCEPT = "ACCEPT"
    ACCEPT_WITH_WARNING = "ACCEPT_WITH_WARNING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECT_PRESERVE_IMAGE = "REJECT_PRESERVE_IMAGE"



class FormulaOcrState(str, Enum):
    ACCEPTED = "OCR_ACCEPTED"
    ACCEPTED_WITH_WARNING = "OCR_ACCEPTED_WITH_WARNING"
    REVIEW_REQUIRED = "OCR_REVIEW_REQUIRED"
    REJECTED_PRESERVE_IMAGE = "OCR_REJECTED_PRESERVE_IMAGE"
    INFERENCE_FAILED_PRESERVE_IMAGE = "OCR_INFERENCE_FAILED_PRESERVE_IMAGE"



@dataclass(frozen=True)
class SafetyGateDecision:
    verdict: SafetyGateVerdict
    reasons: tuple[dict[str, str], ...]
    risk_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "risk_flags": list(self.risk_flags),
        }



@dataclass(frozen=True)
class FormulaOcrResult:
    formula_id: str
    state: FormulaOcrState
    raw_latex: str | None
    png_sha256: str | None
    width: int | None
    height: int | None
    cache_key: str | None
    cache_hit: bool
    inference_error: str | None
    validator: dict[str, Any]
    safety_gate: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value



class FormulaOcrSafetyGate:
    """Fail closed on anomalies without attempting OCR semantic correction."""

    _punctuation = re.compile(
        r"^(?:"
        r"[\s()\[\]{}.,;:!?。，；？！·•●○]+|"
        r"\^?\\circ|"
        r"\\(?:bullet|cdot|vdots|ldots|dots)"
        r"(?:\\quad\\(?:bullet|cdot|vdots|ldots|dots))*"
        r")$"
    )
    _repeated_subsequence = re.compile(r"(.{8,64})(?:\1){4,}", re.DOTALL)

    def evaluate(
        self,
        *,
        width: int,
        height: int,
        raw_latex: str,
        validation: OcrValidation,
    ) -> SafetyGateDecision:
        reasons: list[dict[str, str]] = []
        risks: list[str] = []
        tiny = width * height <= 10_000 or width <= 120 or height <= 48
        narrow = width <= 200 or width / max(height, 1) <= 0.75
        if tiny:
            risks.append("TINY_SOURCE")
        if narrow:
            risks.append("NARROW_SOURCE")

        length_limit = max(512, width * 8, height * 16)
        if len(raw_latex) > length_limit:
            reasons.append(
                self._reason(
                    "PREDICTION_EXPLOSION",
                    (
                        f"Raw output has {len(raw_latex)} characters for a "
                        f"{width}x{height} source; conservative limit is {length_limit}."
                    ),
                )
            )
        command_count = len(re.findall(r"\\[A-Za-z]+", raw_latex))
        fraction_count = len(re.findall(r"\\(?:d?frac|tfrac)\b", raw_latex))
        if fraction_count >= 12 and fraction_count * 2 >= max(command_count, 1):
            reasons.append(
                self._reason(
                    "REPEATED_LATEX_COMMAND",
                    (
                        f"Fraction command repeats {fraction_count} times in an "
                        "implausibly repetitive prediction."
                    ),
                )
            )
        if len(raw_latex) >= 128 and self._repeated_subsequence.search(raw_latex):
            reasons.append(
                self._reason(
                    "REPEATED_SUBSEQUENCE",
                    "A non-trivial LaTeX subsequence repeats at least five times.",
                )
            )

        if validation.verdict in {
            OcrVerdict.INVALID,
            OcrVerdict.EMPTY,
            OcrVerdict.SUSPICIOUS,
        }:
            reasons.append(
                self._reason(
                    "OCR_VALIDATOR_REJECTED",
                    f"OCR validator verdict is {validation.verdict.value}.",
                )
            )
        if reasons:
            return SafetyGateDecision(
                SafetyGateVerdict.REJECT_PRESERVE_IMAGE,
                tuple(reasons),
                tuple(risks),
            )

        if self._punctuation.fullmatch(raw_latex.strip()):
            return SafetyGateDecision(
                SafetyGateVerdict.REVIEW_REQUIRED,
                (
                    self._reason(
                        "LOW_CONTENT_PUNCTUATION_OR_LIST_MARK",
                        (
                            "Prediction contains only delimiter, punctuation, "
                            "or list-mark-like content."
                        ),
                    ),
                ),
                tuple(risks),
            )
        if validation.verdict is OcrVerdict.VALID_WITH_WARNING:
            return SafetyGateDecision(
                SafetyGateVerdict.ACCEPT_WITH_WARNING,
                (
                    self._reason(
                        "PROJECT_STYLE_WARNING",
                        "Raw LaTeX is structurally usable but has project-style warnings.",
                    ),
                ),
                tuple(risks),
            )
        return SafetyGateDecision(
            SafetyGateVerdict.ACCEPT,
            (),
            tuple(risks),
        )

    @staticmethod
    def _reason(code: str, message: str) -> dict[str, str]:
        return {"code": code, "message": message}



class FormulaReview:
    def __init__(self):
        self.validator = FormulaOcrOutputValidator()
        self.safety_gate = FormulaOcrSafetyGate()
        self.stats = {'validation_seconds': 0.0, 'safety_gate_seconds': 0.0}
        self._runtime_fingerprint = None
        self.batch_size = 1

    def _evaluate_result(
        self,
        node: FormulaNode,
        report: dict[str, Any],
        *,
        raw_latex: str,
        png_sha256: str,
        width: int,
        height: int,
        cache_key: str,
        cache_hit: bool,
    ) -> FormulaOcrResult:
        started = time.perf_counter()
        validation = self.validator.validate(raw_latex)
        self.stats["validation_seconds"] = float(
            self.stats["validation_seconds"]
        ) + (time.perf_counter() - started)
        started = time.perf_counter()
        decision = self.safety_gate.evaluate(
            width=width,
            height=height,
            raw_latex=raw_latex,
            validation=validation,
        )
        self.stats["safety_gate_seconds"] = float(
            self.stats["safety_gate_seconds"]
        ) + (time.perf_counter() - started)
        state = {
            SafetyGateVerdict.ACCEPT: FormulaOcrState.ACCEPTED,
            SafetyGateVerdict.ACCEPT_WITH_WARNING: (
                FormulaOcrState.ACCEPTED_WITH_WARNING
            ),
            SafetyGateVerdict.REVIEW_REQUIRED: FormulaOcrState.REVIEW_REQUIRED,
            SafetyGateVerdict.REJECT_PRESERVE_IMAGE: (
                FormulaOcrState.REJECTED_PRESERVE_IMAGE
            ),
        }[decision.verdict]
        provenance = self._provenance(
            node,
            report,
            png_sha256=png_sha256,
            width=width,
            height=height,
        )
        result = FormulaOcrResult(
            formula_id=node.formula_id,
            state=state,
            raw_latex=raw_latex,
            png_sha256=png_sha256,
            width=width,
            height=height,
            cache_key=cache_key,
            cache_hit=cache_hit,
            inference_error=None,
            validator=validation.to_dict(),
            safety_gate=decision.to_dict(),
            provenance=provenance,
        )
        self._apply_result(node, report, result)
        return result


    def _failed_result(
        self,
        node: FormulaNode,
        report: dict[str, Any],
        error: str,
    ) -> FormulaOcrResult:
        validation = OcrValidation(
            OcrVerdict.INFERENCE_FAILED,
            (
                {
                    "code": "INFERENCE_FAILED",
                    "message": error,
                    "severity": "error",
                },
            ),
        )
        decision = SafetyGateDecision(
            SafetyGateVerdict.REJECT_PRESERVE_IMAGE,
            (
                {
                    "code": "INFERENCE_FAILED",
                    "message": "FormulaNet inference failed; preserve the source PNG.",
                },
            ),
            (),
        )
        result = FormulaOcrResult(
            formula_id=node.formula_id,
            state=FormulaOcrState.INFERENCE_FAILED_PRESERVE_IMAGE,
            raw_latex=None,
            png_sha256=None,
            width=None,
            height=None,
            cache_key=None,
            cache_hit=False,
            inference_error=error,
            validator=validation.to_dict(),
            safety_gate=decision.to_dict(),
            provenance=self._provenance(node, report),
        )
        self._apply_result(node, report, result)
        return result


    def _provenance(
        self,
        node: FormulaNode,
        report: dict[str, Any],
        *,
        png_sha256: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        candidate = next(
            (
                row
                for row in report["formulas"].get("candidate_records", [])
                if row.get("candidate_id") == node.formula_id
            ),
            {},
        )
        fingerprint = self._runtime_fingerprint or {}
        return {
            "source_docx": report.get("source", {}).get("path"),
            "source_part": node.source_part,
            "source_locator": node.source_locator,
            "original_ref": node.original_ref,
            "preview_ref": node.preview_ref,
            "rendered_ref": node.rendered_ref,
            "ole_relationship_id": candidate.get("ole_relationship_id"),
            "preview_relationship_id": candidate.get("preview_relationship_id"),
            "ole_sha256": candidate.get("ole_sha256"),
            "wmf_sha256": candidate.get("wmf_sha256"),
            "png_sha256": png_sha256,
            "width": width,
            "height": height,
            "model_identifier": fingerprint.get("model_identifier"),
            "model_revision": fingerprint.get("model_revision"),
            "model_fingerprint": fingerprint.get("model_sha256"),
            "runtime_fingerprint": fingerprint,
            "batch_size": self.batch_size,
            "precision": "fp32",
            "ocr_contract_version": FORMULA_OCR_CONTRACT_VERSION,
        }


    @staticmethod
    def _apply_result(
        node: FormulaNode,
        report: dict[str, Any],
        result: FormulaOcrResult,
    ) -> None:
        accepted = result.state in {
            FormulaOcrState.ACCEPTED,
            FormulaOcrState.ACCEPTED_WITH_WARNING,
        }
        node.latex = result.raw_latex if accepted else None
        node.status = result.state.value
        node.ocr_provenance = result.to_dict()
        if result.state is FormulaOcrState.ACCEPTED_WITH_WARNING:
            node.warnings.extend(
                issue["code"] for issue in result.validator.get("issues", [])
            )
        if not accepted:
            node.warnings.extend(
                reason["code"] for reason in result.safety_gate.get("reasons", [])
            )
        payload = result.to_dict()
        for collection in ("candidate_records", "records"):
            for record in report["formulas"].get(collection, []):
                key = "candidate_id" if collection == "candidate_records" else "formula_id"
                if record.get(key) == node.formula_id:
                    record.setdefault("structural_status", record.get("status"))
                    record["status"] = result.state.value
                    record["ocr"] = payload
                    if collection == "records":
                        record["latex"] = node.latex


    @staticmethod
    def _finalize_document_report(
        report: dict[str, Any],
        results: Sequence[FormulaOcrResult],
    ) -> None:
        formula_report = report["formula_ocr"]
        counter = defaultdict(int)
        validator_issues = defaultdict(int)
        gate_issues = defaultdict(int)
        for result in results:
            counter[result.state.value] += 1
            for issue in result.validator.get("issues", []):
                validator_issues[issue["code"]] += 1
            for reason in result.safety_gate.get("reasons", []):
                gate_issues[reason["code"]] += 1
        state_fields = {
            FormulaOcrState.ACCEPTED: "accepted",
            FormulaOcrState.ACCEPTED_WITH_WARNING: "accepted_with_warning",
            FormulaOcrState.REVIEW_REQUIRED: "review_required",
            FormulaOcrState.REJECTED_PRESERVE_IMAGE: "rejected_preserved",
            FormulaOcrState.INFERENCE_FAILED_PRESERVE_IMAGE: (
                "inference_failed_preserved"
            ),
        }
        for state, field in state_fields.items():
            formula_report[field] = counter[state.value]
        formula_report["validator_issues"] = dict(sorted(validator_issues.items()))
        formula_report["safety_gate_issues"] = dict(sorted(gate_issues.items()))
        formula_report["records"] = [result.to_dict() for result in results]
        terminal = sum(formula_report[field] for field in state_fields.values())
        formula_report["conservation_ok"] = terminal == formula_report["candidates"]



def _iter_formula_nodes(document: DocumentIR):
    for block in document.blocks:
        if isinstance(block, TableNode):
            for row in block.rows:
                for cell in row:
                    for nested in cell.blocks:
                        yield from _iter_inline_nodes(nested.children)
        else:
            yield from _iter_inline_nodes(block.children)



def _iter_inline_nodes(nodes):
    for node in nodes:
        if isinstance(node, FormulaNode):
            yield node
        elif isinstance(node, HyperlinkNode):
            yield from _iter_inline_nodes(node.children)



def _initialize_report(report):
    report.setdefault("timing", {})
    for key in (
        "formula_ocr_model_load_seconds",
        "formula_ocr_inference_seconds",
        "formula_ocr_validation_seconds",
        "formula_ocr_safety_gate_seconds",
    ):
        report["timing"].setdefault(key, 0.0)
    report.setdefault("runtime", {}).setdefault("vision_or_ocr_used", False)
    return report.setdefault(
        "formula_ocr",
        {
            "enabled": True,
            "candidates": 0,
            "unique_png": 0,
            "inference_calls": 0,
            "batch_count": 0,
            "inferred_unique_png": 0,
            "deduplicated_occurrences": 0,
            "cache_hits": 0,
            "accepted": 0,
            "accepted_with_warning": 0,
            "review_required": 0,
            "rejected_preserved": 0,
            "inference_failed_preserved": 0,
            "validator_issues": {},
            "safety_gate_issues": {},
            "records": [],
            "conservation_ok": True,
        },
    )

