from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image

from .formulanet_runtime import (
    FormulaNetRuntime,
    FormulaOcrOutputValidator,
    OcrValidation,
    OcrVerdict,
    PaddleFormulaNetRuntime,
    benchmark_cache_key,
)
from .ir import (
    DocumentIR,
    FormulaNode,
    HyperlinkNode,
    TableNode,
)

FORMULA_OCR_CONTRACT_VERSION = "bemarkdown-formula-ocr-v1"
FORMULANET_MODEL_IDENTIFIER = "PP-FormulaNet_plus-L"
FORMULANET_MODEL_REVISION = "0809597a77f735bfb35354edb632f2e6dff606f3"


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


class FormulaOcrAdapter:
    """Lazy, content-addressed production boundary for FormulaNet fallback."""

    def __init__(
        self,
        *,
        runtime_factory: Callable[[], FormulaNetRuntime] | None = None,
        batch_size: int = 2,
        validator: FormulaOcrOutputValidator | None = None,
        safety_gate: FormulaOcrSafetyGate | None = None,
    ):
        if batch_size not in {1, 2}:
            raise ValueError("Production FormulaNet batch_size must be 1 or 2")
        self.batch_size = batch_size
        self.runtime_factory = runtime_factory or _default_runtime_factory
        self.validator = validator or FormulaOcrOutputValidator()
        self.safety_gate = safety_gate or FormulaOcrSafetyGate()
        self._runtime: FormulaNetRuntime | None = None
        self._runtime_fingerprint: dict[str, Any] | None = None
        self._prediction_cache: dict[str, str] = {}
        self.stats: dict[str, int | float | None] = {
            "model_load_count": 0,
            "model_load_seconds": 0.0,
            "inference_calls": 0,
            "batch_count": 0,
            "inferred_unique_png": 0,
            "cache_hits": 0,
            "inference_total_seconds": 0.0,
            "validation_seconds": 0.0,
            "safety_gate_seconds": 0.0,
            "peak_gpu_memory_bytes": None,
        }

    @property
    def runtime_fingerprint(self) -> dict[str, Any] | None:
        return self._runtime_fingerprint

    def process(
        self,
        document: DocumentIR,
        output_dir: Path,
        report: dict[str, Any],
    ) -> list[FormulaOcrResult]:
        nodes = [
            node
            for node in _iter_formula_nodes(document)
            if node.status == "RENDERED_FALLBACK" and node.rendered_ref
        ]
        formula_report = _initialize_report(report)
        formula_report["candidates"] = len(nodes)
        if not nodes:
            return []

        model_loads_before = int(self.stats["model_load_count"])
        inference_before = int(self.stats["inference_calls"])
        batch_before = int(self.stats["batch_count"])
        inferred_before = int(self.stats["inferred_unique_png"])
        cache_before = int(self.stats["cache_hits"])
        inference_seconds_before = float(self.stats["inference_total_seconds"])
        validation_before = float(self.stats["validation_seconds"])
        safety_before = float(self.stats["safety_gate_seconds"])

        image_groups: dict[str, list[tuple[FormulaNode, Path, int, int]]] = defaultdict(
            list
        )
        missing_nodes: list[FormulaNode] = []
        for node in nodes:
            path = (output_dir / node.rendered_ref).resolve()
            if not path.is_file():
                missing_nodes.append(node)
                continue
            data = path.read_bytes()
            png_sha = hashlib.sha256(data).hexdigest()
            with Image.open(path) as image:
                width, height = image.size
            image_groups[png_sha].append((node, path, width, height))
        formula_report["unique_png"] = len(image_groups)
        formula_report["deduplicated_occurrences"] = sum(
            len(group) - 1 for group in image_groups.values()
        )

        results: list[FormulaOcrResult] = []
        for node in missing_nodes:
            results.append(
                self._failed_result(
                    node,
                    report,
                    "Rendered formula PNG is missing before FormulaNet inference.",
                )
            )

        if image_groups:
            runtime = None
            inference_error = None
            try:
                runtime = self._ensure_runtime()
            except Exception as exc:  # noqa: BLE001 - preserve all source PNGs
                inference_error = f"{type(exc).__name__}: {exc}"

            fingerprint = (
                self._runtime_fingerprint or runtime.fingerprint()
                if runtime is not None
                else {}
            )
            config = {
                "precision": "fp32",
                "batch_size": self.batch_size,
                "device": fingerprint.get("device"),
                "ocr_contract_version": FORMULA_OCR_CONTRACT_VERSION,
            }
            pending: list[tuple[str, Path, str]] = []
            predictions: dict[str, tuple[str, str, bool]] = {}
            for png_sha, group in image_groups.items():
                key = benchmark_cache_key(png_sha, fingerprint, config)
                if key in self._prediction_cache:
                    predictions[png_sha] = (
                        self._prediction_cache[key],
                        key,
                        True,
                    )
                    self.stats["cache_hits"] = int(self.stats["cache_hits"]) + 1
                else:
                    pending.append((png_sha, group[0][1], key))

            if pending and runtime is not None:
                started = time.perf_counter()
                try:
                    raw_outputs = runtime.predict(
                        [item[1] for item in pending],
                        batch_size=self.batch_size,
                    )
                    if len(raw_outputs) != len(pending):
                        raise RuntimeError(
                            "FormulaNet returned a different number of outputs than inputs"
                        )
                    for (png_sha, _path, key), raw in zip(
                        pending, raw_outputs, strict=True
                    ):
                        self._prediction_cache[key] = raw
                        predictions[png_sha] = (raw, key, False)
                    peak = runtime.peak_gpu_memory_bytes()
                    if peak is not None:
                        previous_peak = self.stats["peak_gpu_memory_bytes"]
                        self.stats["peak_gpu_memory_bytes"] = max(
                            int(previous_peak or 0), peak
                        )
                except Exception as exc:  # noqa: BLE001 - preserve all source PNGs
                    inference_error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - started
                self.stats["inference_calls"] = int(self.stats["inference_calls"]) + 1
                self.stats["batch_count"] = int(self.stats["batch_count"]) + math.ceil(
                    len(pending) / self.batch_size
                )
                self.stats["inferred_unique_png"] = int(
                    self.stats["inferred_unique_png"]
                ) + len(pending)
                self.stats["inference_total_seconds"] = (
                    float(self.stats["inference_total_seconds"]) + elapsed
                )
            if inference_error:
                for group in image_groups.values():
                    for node, _path, _width, _height in group:
                        results.append(
                            self._failed_result(node, report, inference_error)
                        )
            else:
                for png_sha, group in image_groups.items():
                    raw, cache_key, cache_hit = predictions[png_sha]
                    for node, _path, width, height in group:
                        results.append(
                            self._evaluate_result(
                                node,
                                report,
                                raw_latex=raw,
                                png_sha256=png_sha,
                                width=width,
                                height=height,
                                cache_key=cache_key,
                                cache_hit=cache_hit,
                            )
                        )

        formula_report["model_load_count"] = (
            int(self.stats["model_load_count"]) - model_loads_before
        )
        formula_report["inference_calls"] = (
            int(self.stats["inference_calls"]) - inference_before
        )
        formula_report["batch_count"] = int(self.stats["batch_count"]) - batch_before
        formula_report["inferred_unique_png"] = (
            int(self.stats["inferred_unique_png"]) - inferred_before
        )
        formula_report["cache_hits"] = int(self.stats["cache_hits"]) - cache_before
        formula_report["batch_size"] = self.batch_size
        formula_report["precision"] = "fp32"
        formula_report["contract_version"] = FORMULA_OCR_CONTRACT_VERSION
        formula_report["runtime_fingerprint"] = self._runtime_fingerprint
        formula_report["model_load_seconds"] = (
            float(self.stats["model_load_seconds"])
            if formula_report["model_load_count"]
            else 0.0
        )
        report["timing"]["formula_ocr_inference_seconds"] = (
            float(self.stats["inference_total_seconds"]) - inference_seconds_before
        )
        report["timing"]["formula_ocr_validation_seconds"] = (
            float(self.stats["validation_seconds"]) - validation_before
        )
        report["timing"]["formula_ocr_safety_gate_seconds"] = (
            float(self.stats["safety_gate_seconds"]) - safety_before
        )
        report["timing"]["formula_ocr_model_load_seconds"] = formula_report[
            "model_load_seconds"
        ]
        report["runtime"]["vision_or_ocr_used"] = True
        self._finalize_document_report(report, results)
        return results

    @property
    def loaded_runtime(self) -> FormulaNetRuntime | None:
        """Borrow an already loaded engine without triggering model startup."""
        return self._runtime

    def _ensure_runtime(self) -> FormulaNetRuntime:
        if self._runtime is None:
            self._runtime = self.runtime_factory()
            self._runtime_fingerprint = self._runtime.fingerprint()
            self.stats["model_load_count"] = int(self.stats["model_load_count"]) + 1
            self.stats["model_load_seconds"] = float(
                getattr(self._runtime, "load_seconds", 0.0)
            )
            self.stats["peak_gpu_memory_bytes"] = (
                self._runtime.peak_gpu_memory_bytes()
            )
        return self._runtime

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


def select_production_batch(
    baseline: dict[str, str],
    repeated_runs: Sequence[dict[str, str]],
) -> int:
    if len(repeated_runs) != 3:
        raise ValueError("Batch 2 stability gate requires exactly three runs")
    return 2 if all(run == baseline for run in repeated_runs) else 1


def _default_runtime_factory() -> FormulaNetRuntime:
    return create_production_formulanet_runtime()


def create_production_formulanet_runtime(
    *,
    models_root: str | Path | None = None,
    config_path: str | Path | None = None,
    mcp_root: str | Path | None = None,
    deep_model_validation: bool = True,
    runtime_owner=None,
) -> FormulaNetRuntime:
    """Create the GPU runtime only from a validated local Model Registry asset."""

    from .model_registry import MODEL_ID, ModelRegistry

    resolution = ModelRegistry(
        models_root=models_root,
        config_path=config_path,
        mcp_root=mcp_root,
    ).resolve(MODEL_ID, deep=deep_model_validation)
    constructor = PaddleFormulaNetRuntime if runtime_owner is None else runtime_owner.acquire
    return constructor(
        model_identifier=FORMULANET_MODEL_IDENTIFIER,
        model_dir=resolution.model_root,
        model_source="local-model-registry",
        model_revision=FORMULANET_MODEL_REVISION,
        verified_model_inventory=resolution.verified_inventory,
        device="gpu:0",
    )


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


def adapter_stats_snapshot(adapter: FormulaOcrAdapter) -> dict[str, Any]:
    value = dict(adapter.stats)
    value["batch_size"] = adapter.batch_size
    value["contract_version"] = FORMULA_OCR_CONTRACT_VERSION
    value["runtime_fingerprint"] = adapter.runtime_fingerprint
    value["prediction_cache_entries"] = len(adapter._prediction_cache)
    return json.loads(json.dumps(value))
