"""Provider-neutral contracts for ordinary-text OCR evidence.

Providers only adapt model runtimes into a stable evidence record. Comparison,
routing, and selection deliberately live outside this module so a provider can
never become a truth-aware winner selector.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from bemarkdown.text_recognition_contract import normalize_text

TEXT_EVIDENCE_SCHEMA = "bemarkdown-text-recognition-evidence-v1"
TEXT_RECOGNITION_REQUEST_SCHEMA = "bemarkdown-text-recognition-request-v1"
TEXT_PROVIDER_WORKER_REQUEST_SCHEMA = "bemarkdown-text-provider-worker-request-v1"
TEXT_PROVIDER_WORKER_RESPONSE_SCHEMA = "bemarkdown-text-provider-worker-response-v1"
CONSENSUS_COMPARATOR_STATUS = "IMPLEMENTED_V1"
NATIVE_CONFIDENCE_UNAVAILABLE = "NATIVE_CONFIDENCE_UNAVAILABLE"


def _bbox(value: Sequence[float]) -> tuple[float, float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 4 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("TEXT_RECOGNITION_BBOX_MUST_HAVE_POSITIVE_AREA")
    return result  # type: ignore[return-value]


def _fingerprint(value: Any, *, fallback: str) -> str:
    if isinstance(value, Mapping):
        for key in ("weight_sha256", "revision", "model_revision", "model_id"):
            if value.get(key):
                return str(value[key])
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return str(value or fallback)


@dataclass(frozen=True, slots=True)
class TextRecognitionRequest:
    document_id: str
    page_id: str
    region_id: str
    source_candidate_id: str
    bbox: tuple[float, float, float, float]
    crop_ref: str
    crop_sha256: str
    route_kind: str
    text_track: str
    risk_hints: tuple[str, ...] = ()
    source_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        identity = (
            self.document_id,
            self.page_id,
            self.region_id,
            self.source_candidate_id,
            self.crop_ref,
            self.route_kind,
            self.text_track,
        )
        if not all(str(value).strip() for value in identity):
            raise ValueError("TEXT_RECOGNITION_REQUEST_IDENTITY_REQUIRED")
        if len(self.crop_sha256) != 64:
            raise ValueError("TEXT_RECOGNITION_CROP_SHA256_REQUIRED")
        if self.text_track not in {"TEXT_REGION_CROP", "PAGE_TEXT_RECOVERY"}:
            raise ValueError(f"UNSUPPORTED_TEXT_TRACK:{self.text_track}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = TEXT_RECOGNITION_REQUEST_SCHEMA
        value["bbox"] = list(self.bbox)
        value["risk_hints"] = list(self.risk_hints)
        return value


@dataclass(frozen=True, slots=True)
class TextProviderWorkerRequest:
    request_id: str
    provider_id: str
    crop_ref: str
    crop_sha256: str
    generation_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.provider_id.strip():
            raise ValueError("TEXT_PROVIDER_WORKER_IDENTITY_REQUIRED")
        if len(self.crop_sha256) != 64:
            raise ValueError("TEXT_PROVIDER_WORKER_CROP_SHA256_REQUIRED")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": TEXT_PROVIDER_WORKER_REQUEST_SCHEMA, **asdict(self)}


@dataclass(frozen=True, slots=True)
class TextProviderWorkerResponse:
    request_id: str
    provider_id: str
    text: str
    latency_ms: float
    runtime_fingerprint: str
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.provider_id.strip():
            raise ValueError("TEXT_PROVIDER_WORKER_IDENTITY_REQUIRED")
        if self.latency_ms < 0:
            raise ValueError("TEXT_PROVIDER_WORKER_LATENCY_CANNOT_BE_NEGATIVE")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": TEXT_PROVIDER_WORKER_RESPONSE_SCHEMA, **asdict(self)}


@dataclass(frozen=True, slots=True)
class TextRecognitionEvidence:
    # The first seven fields preserve the original TextEvidence constructor.
    provider_id: str
    model_fingerprint: str
    bbox: tuple[float, float, float, float]
    text: str
    confidence: float | None
    source_crop_ref: str
    latency_seconds: float
    model_id: str = ""
    runtime_fingerprint: str = ""
    normalized_text: str = ""
    source_crop_sha256: str = ""
    confidence_semantics: str = NATIVE_CONFIDENCE_UNAVAILABLE
    generation_tokens: int | None = None
    output_contract_status: str = "PASS"
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("TextEvidence provider_id must be present")
        if not self.model_fingerprint.strip():
            raise ValueError("TextEvidence model_fingerprint must be present")
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("TextEvidence confidence must be in [0, 1]")
        if self.latency_seconds < 0:
            raise ValueError("TextEvidence latency cannot be negative")
        if self.generation_tokens is not None and self.generation_tokens < 0:
            raise ValueError("TextEvidence generation_tokens cannot be negative")
        if not self.model_id:
            object.__setattr__(self, "model_id", self.provider_id)
        if not self.normalized_text and self.text:
            object.__setattr__(self, "normalized_text", normalize_text(self.text))

    @property
    def raw_text(self) -> str:
        return self.text

    @property
    def latency_ms(self) -> float:
        return self.latency_seconds * 1000.0

    @property
    def contract_valid(self) -> bool:
        return self.output_contract_status == "PASS" and bool(self.normalized_text)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = TEXT_EVIDENCE_SCHEMA
        value["bbox"] = list(self.bbox)
        value["raw_text"] = self.raw_text
        value["latency_ms"] = self.latency_ms
        value["warnings"] = list(self.warnings)
        return value


# Backward-compatible public name used by the modular PDF pipeline.
TextEvidence = TextRecognitionEvidence


@runtime_checkable
class TextEvidenceProvider(Protocol):
    """One OCR provider; comparison and selection are forbidden here."""

    provider_id: str

    def recognize(self, request: TextRecognitionRequest) -> TextRecognitionEvidence: ...

    def recognize_batch(
        self, requests: Sequence[TextRecognitionRequest]
    ) -> list[TextRecognitionEvidence]: ...


@runtime_checkable
class ConsensusComparator(Protocol):
    def compare(
        self,
        request: TextRecognitionRequest,
        evidence_a: TextRecognitionEvidence,
        evidence_b: TextRecognitionEvidence,
    ) -> Any: ...


class _BatchProviderMixin:
    def recognize_batch(
        self, requests: Sequence[TextRecognitionRequest]
    ) -> list[TextRecognitionEvidence]:
        return [self.recognize(request) for request in requests]  # type: ignore[attr-defined]


class PaddleOcrV6TextEvidenceProvider(_BatchProviderMixin):
    """Adapt the existing PP-OCRv6 det/rec runtime without changing it."""

    provider_id = "pp-ocrv6"

    def __init__(self, runtime: Any):
        self.runtime = runtime

    def recognize(
        self, request: TextRecognitionRequest | dict[str, Any]
    ) -> TextRecognitionEvidence:
        started = time.perf_counter()
        if isinstance(request, TextRecognitionRequest):
            preserved_crop = request.source_provenance.get("crop")
            runtime_request: Any = (
                dict(preserved_crop)
                if isinstance(preserved_crop, Mapping)
                else {
                    "path": request.crop_ref,
                    "bbox_pdf_pt": list(request.bbox),
                    "content_sha256": request.crop_sha256,
                }
            )
            bbox = request.bbox
            crop_ref = request.crop_ref
            crop_sha = request.crop_sha256
        else:
            runtime_request = request
            bbox = _bbox(request["bbox_pdf_pt"])
            crop_ref = str(request.get("crop_ref") or "source-crop-unfingerprinted")
            crop_sha = str(request.get("content_sha256") or "")
        lines = self.runtime.recognize(runtime_request)
        fingerprint = self.runtime.fingerprint()
        text = "\n".join(str(row.get("text") or "") for row in lines if row.get("text"))
        confidences = [
            float(row["confidence"])
            for row in lines
            if row.get("confidence") is not None
        ]
        model_fingerprint = "+".join(
            str(fingerprint.get(name) or "")
            for name in ("det_model_fingerprint", "rec_model_fingerprint")
        ).strip("+")
        return TextRecognitionEvidence(
            provider_id=self.provider_id,
            model_id="PP-OCRv6",
            model_fingerprint=model_fingerprint
            or _fingerprint(fingerprint, fallback="pp-ocrv6-runtime"),
            runtime_fingerprint=_fingerprint(fingerprint, fallback="pp-ocrv6-runtime"),
            bbox=bbox,
            text=text,
            normalized_text=normalize_text(text),
            confidence=(sum(confidences) / len(confidences) if confidences else None),
            confidence_semantics=(
                "NATIVE_DET_REC_MEAN" if confidences else NATIVE_CONFIDENCE_UNAVAILABLE
            ),
            source_crop_ref=(f"sha256:{crop_sha}" if crop_sha else crop_ref),
            source_crop_sha256=crop_sha,
            latency_seconds=time.perf_counter() - started,
        )


class _GenerativeTextEvidenceProvider(_BatchProviderMixin):
    provider_id = ""
    model_id = ""

    def __init__(self, runtime: Any):
        self.runtime = runtime

    def recognize(self, request: TextRecognitionRequest) -> TextRecognitionEvidence:
        started = time.perf_counter()
        result = self.runtime.recognize(request)
        elapsed = time.perf_counter() - started
        return self._evidence(request, result, elapsed)

    def recognize_batch(
        self, requests: Sequence[TextRecognitionRequest]
    ) -> list[TextRecognitionEvidence]:
        recognize_batch = getattr(self.runtime, "recognize_batch", None)
        if not callable(recognize_batch):
            return super().recognize_batch(requests)
        started = time.perf_counter()
        results = list(recognize_batch(requests))
        elapsed = time.perf_counter() - started
        if len(results) != len(requests):
            raise RuntimeError("TEXT_PROVIDER_BATCH_CARDINALITY_MISMATCH")
        per_item = elapsed / len(requests) if requests else 0.0
        return [
            self._evidence(request, result, per_item)
            for request, result in zip(requests, results, strict=True)
        ]

    def _evidence(
        self,
        request: TextRecognitionRequest,
        result: Mapping[str, Any],
        elapsed: float,
    ) -> TextRecognitionEvidence:
        raw = str(
            result.get("raw_text")
            or result.get("extracted_text")
            or result.get("text")
            or ""
        )
        latency_ms = float(result.get("latency_ms") or elapsed * 1000.0)
        return TextRecognitionEvidence(
            provider_id=self.provider_id,
            model_id=self.model_id,
            model_fingerprint=_fingerprint(
                result.get("model_fingerprint"), fallback=self.model_id
            ),
            runtime_fingerprint=_fingerprint(
                result.get("runtime_fingerprint"), fallback="runtime-unfingerprinted"
            ),
            bbox=request.bbox,
            text=raw,
            normalized_text=normalize_text(raw),
            confidence=None,
            confidence_semantics=NATIVE_CONFIDENCE_UNAVAILABLE,
            source_crop_ref=request.crop_ref,
            source_crop_sha256=request.crop_sha256,
            latency_seconds=latency_ms / 1000.0,
            generation_tokens=(
                int(result["generation_tokens"])
                if result.get("generation_tokens") is not None
                else None
            ),
            output_contract_status=str(
                result.get("output_contract_status") or "UNKNOWN"
            ),
            warnings=tuple(str(item) for item in result.get("warnings", ())),
            provenance={"runtime_result_contract": "TEXT_PROVIDER_WORKER_RESPONSE"},
        )


class GOTOCR2TextEvidenceProvider(_GenerativeTextEvidenceProvider):
    provider_id = "got-ocr2"
    model_id = "GOT-OCR2.0"


class HunyuanOCRTextEvidenceProvider(_GenerativeTextEvidenceProvider):
    provider_id = "hunyuan-ocr-1.5"
    model_id = "HunyuanOCR-1.5"


class FrozenOutputTextEvidenceProvider(_BatchProviderMixin):
    """Replay sealed outputs. It never receives truth rows."""

    def __init__(self, provider_id: str, rows: Mapping[str, Mapping[str, Any]]):
        self.provider_id = provider_id
        self.rows = {str(key): dict(value) for key, value in rows.items()}

    def recognize(self, request: TextRecognitionRequest) -> TextRecognitionEvidence:
        try:
            row = self.rows[request.region_id]
        except KeyError as exc:
            raise KeyError(f"FROZEN_OUTPUT_MISSING:{request.region_id}") from exc
        raw = str(
            row.get("extracted_text")
            or row.get("raw_output")
            or row.get("raw_text")
            or ""
        )
        normalized = str(row.get("normalized_output") or normalize_text(raw))
        confidence = row.get("confidence")
        latency = float(row.get("latency_seconds") or row.get("latency") or 0.0)
        contract = str(
            row.get("output_contract_status")
            or ("WARN" if row.get("output_format_violation") else "PASS")
        )
        tokens = row.get("generated_token_count")
        if tokens is None:
            tokens = row.get("generated_tokens")
        return TextRecognitionEvidence(
            provider_id=self.provider_id,
            model_id=str(row.get("model") or self.provider_id),
            model_fingerprint=_fingerprint(
                row.get("model_fingerprint") or row.get("model_revision"),
                fallback=f"frozen:{self.provider_id}",
            ),
            runtime_fingerprint=_fingerprint(
                row.get("runtime_fingerprint"), fallback="frozen-output-replay"
            ),
            bbox=request.bbox,
            text=raw,
            normalized_text=normalized,
            confidence=(float(confidence) if confidence is not None else None),
            confidence_semantics=(
                "NATIVE_CONFIDENCE"
                if confidence is not None
                else NATIVE_CONFIDENCE_UNAVAILABLE
            ),
            source_crop_ref=request.crop_ref,
            source_crop_sha256=str(
                row.get("source_crop_sha256") or request.crop_sha256
            ),
            latency_seconds=latency,
            generation_tokens=(int(tokens) if tokens is not None else None),
            output_contract_status=contract,
            warnings=tuple(
                str(item) for item in row.get("output_format_violation_reasons", ())
            ),
            provenance={"frozen_output_replay": True},
        )


# Contract spelling requested by the architecture document.
PPOCRv6TextEvidenceProvider = PaddleOcrV6TextEvidenceProvider
