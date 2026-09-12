"""Truth-blind conditional two-of-three architecture for ordinary-text OCR.

This module is intentionally a policy deep module: providers produce evidence,
the comparator describes evidence, the router decides whether C is needed, and
the resolver accepts two valid normalized outputs that agree.  This is a
provider-consensus policy, not a confidence or semantic-correction heuristic.
"""

from __future__ import annotations

import asyncio
import copy
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from bemarkdown.text_recognition_contract import (
    extract_sensitive_tokens,
    normalize_text,
)

from .text_evidence import (
    TextEvidenceProvider,
    TextRecognitionEvidence,
    TextRecognitionRequest,
)

THREE_MODEL_OCR_SCHEMA = "bemarkdown-three-model-text-evidence-architecture-v2"
RESOLUTION_STATUSES_V2 = (
    "CONSENSUS_AB",
    "CONSENSUS_AC",
    "CONSENSUS_BC",
    "AGENT_REQUIRED_ALL_DIFFER",
    "AGENT_REQUIRED_RUNTIME_FAILURE",
    "FAILED_PRESERVE_INPUT",
)


@dataclass(frozen=True, slots=True)
class TextConsensusPolicy:
    version: str = "Conditional2Of3ConsensusPolicy-v2"
    ppocr_low_confidence_threshold: float = 0.75
    valid_contract_statuses: tuple[str, ...] = ("PASS",)
    high_risk_tracks: tuple[str, ...] = ("PAGE_TEXT_RECOVERY",)


@dataclass(frozen=True, slots=True)
class ThirdModelRoutingPolicy:
    version: str = "ThirdModelRoutingPolicy-v2"
    trigger_statuses: tuple[str, ...] = (
        "AB_MATERIAL_CONFLICT",
        "AB_OUTPUT_CONTRACT_WARNING",
        "AB_EMPTY_OR_PARTIAL_OUTPUT",
    )
    always_trigger_tracks: tuple[str, ...] = ()


class TextSensitiveTokenDetector:
    """Extract and compare semantic-risk tokens without modifying text."""

    _percent = re.compile(r"[%％]")
    _sign = re.compile(r"(?<!\w)[+\-−](?=\d|\w)")

    def detect(self, value: str) -> dict[str, list[str]]:
        normalized = normalize_text(value)
        result = extract_sensitive_tokens(normalized)
        result["signs"] = self._sign.findall(normalized)
        result["percent_signs"] = self._percent.findall(normalized)
        return result

    def compare(self, left: str, right: str) -> dict[str, Any]:
        left_tokens = self.detect(left)
        right_tokens = self.detect(right)
        kinds = tuple(
            key
            for key in left_tokens
            if left_tokens.get(key, []) != right_tokens.get(key, [])
        )
        return {
            "schema": "bemarkdown-sensitive-token-comparison-v1",
            "left": left_tokens,
            "right": right_tokens,
            "conflict": bool(kinds),
            "conflict_kinds": list(kinds),
        }


@dataclass(frozen=True, slots=True)
class ConsensusAssessment:
    schema: str
    statuses: tuple[str, ...]
    strong_ab: bool
    requires_third_model: bool
    normalized_a: str
    normalized_b: str
    sensitive_conflict_kinds: tuple[str, ...]
    validators_pass: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["statuses"] = list(self.statuses)
        value["sensitive_conflict_kinds"] = list(self.sensitive_conflict_kinds)
        return value


class ConsensusComparatorV1:
    """Describe A/B consensus. This class never selects a winner or calls SDKs."""

    def __init__(
        self,
        policy: TextConsensusPolicy | None = None,
        detector: TextSensitiveTokenDetector | None = None,
    ) -> None:
        self.policy = policy or TextConsensusPolicy()
        self.detector = detector or TextSensitiveTokenDetector()

    def compare(
        self,
        request: TextRecognitionRequest,
        evidence_a: TextRecognitionEvidence,
        evidence_b: TextRecognitionEvidence,
    ) -> ConsensusAssessment:
        a = normalize_text(evidence_a.normalized_text or evidence_a.raw_text)
        b = normalize_text(evidence_b.normalized_text or evidence_b.raw_text)
        statuses: list[str] = []
        if evidence_a.raw_text == evidence_b.raw_text:
            statuses.append("AB_EXACT_AGREEMENT")
        if a == b:
            statuses.append("AB_NORMALIZED_AGREEMENT")
        elif a and b:
            statuses.append("AB_MATERIAL_CONFLICT")
        else:
            statuses.append("AB_EMPTY_OR_PARTIAL_OUTPUT")

        sensitive = self.detector.compare(a, b)
        if sensitive["conflict"]:
            statuses.append("AB_SENSITIVE_TOKEN_CONFLICT")
        if (
            evidence_a.confidence is None
            or evidence_a.confidence < self.policy.ppocr_low_confidence_threshold
        ):
            statuses.append("AB_LOW_CONFIDENCE")
        contracts_valid = all(
            evidence.output_contract_status in self.policy.valid_contract_statuses
            for evidence in (evidence_a, evidence_b)
        )
        if not contracts_valid:
            statuses.append("AB_OUTPUT_CONTRACT_WARNING")
        if request.text_track in self.policy.high_risk_tracks:
            statuses.append("AB_PAGE_RECOVERY_RISK")
        if request.risk_hints:
            statuses.append("OCR_VALIDATOR_WARNING")

        strong_ab = (
            "AB_NORMALIZED_AGREEMENT" in statuses
            and contracts_valid
            and bool(a)
        )
        if strong_ab:
            statuses.append("CONSENSUS_STRONG_AB")
        return ConsensusAssessment(
            schema="bemarkdown-consensus-comparator-v2",
            statuses=tuple(dict.fromkeys(statuses)),
            strong_ab=strong_ab,
            requires_third_model=not strong_ab,
            normalized_a=a,
            normalized_b=b,
            sensitive_conflict_kinds=tuple(sensitive["conflict_kinds"]),
            validators_pass=not request.risk_hints,
        )


@dataclass(frozen=True, slots=True)
class ThirdModelRoutingDecision:
    schema: str
    triggered: bool
    reasons: tuple[str, ...]
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


class ThirdModelRouterV1:
    def __init__(self, policy: ThirdModelRoutingPolicy | None = None) -> None:
        self.policy = policy or ThirdModelRoutingPolicy()

    def route(
        self,
        request: TextRecognitionRequest,
        assessment: ConsensusAssessment,
    ) -> ThirdModelRoutingDecision:
        reasons = [
            status
            for status in assessment.statuses
            if status in self.policy.trigger_statuses
        ]
        if request.text_track in self.policy.always_trigger_tracks:
            reasons.append(request.text_track)
        triggered = assessment.requires_third_model
        return ThirdModelRoutingDecision(
            schema="bemarkdown-third-model-routing-v2",
            triggered=triggered,
            reasons=tuple(dict.fromkeys(reasons)),
            policy_version=self.policy.version,
        )


@dataclass(frozen=True, slots=True)
class EvidenceResolutionV1:
    schema: str
    selected_text: str
    resolution_status: str
    selected_provider_basis: str
    evidence_refs: tuple[str, ...]
    warnings: tuple[str, ...]
    review_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("evidence_refs", "warnings", "review_reasons"):
            value[key] = list(value[key])
        return value


class EvidenceResolverV1:
    """Conditional 2-of-3 Policy v2; diagnostics never create Agent work."""

    def __init__(self, policy: TextConsensusPolicy | None = None) -> None:
        self.policy = policy or TextConsensusPolicy()

    def resolve(
        self,
        request: TextRecognitionRequest,
        evidence_a: TextRecognitionEvidence,
        evidence_b: TextRecognitionEvidence,
        assessment: ConsensusAssessment,
        routing: ThirdModelRoutingDecision,
        evidence_c: TextRecognitionEvidence | None = None,
    ) -> EvidenceResolutionV1:
        refs = [evidence_a.provider_id, evidence_b.provider_id]
        if evidence_c is not None:
            refs.append(evidence_c.provider_id)
        a, b = assessment.normalized_a, assessment.normalized_b
        c = (
            normalize_text(evidence_c.normalized_text or evidence_c.raw_text)
            if evidence_c is not None
            else None
        )
        a_valid = evidence_a.contract_valid
        b_valid = evidence_b.contract_valid
        c_valid = evidence_c.contract_valid if evidence_c is not None else False
        if a_valid and b_valid and a == b:
            return self._result(
                a,
                "CONSENSUS_AB",
                "A_B_VALID_NORMALIZED_AGREEMENT",
                refs,
            )
        if a_valid and c_valid and a == c:
            return self._result(
                a,
                "CONSENSUS_AC",
                "A_C_VALID_NORMALIZED_AGREEMENT",
                refs,
            )
        if b_valid and c_valid and b == c:
            return self._result(
                b,
                "CONSENSUS_BC",
                "B_C_VALID_NORMALIZED_AGREEMENT",
                refs,
            )
        valid_count = sum((a_valid, b_valid, c_valid))
        all_different = valid_count == 3 and len({a, b, c}) == 3
        status = (
            "AGENT_REQUIRED_ALL_DIFFER"
            if all_different
            else "AGENT_REQUIRED_RUNTIME_FAILURE"
        )
        reasons = ["ALL_DIFFER"] if all_different else ["HARD_RUNTIME_FAILURE"]
        if routing.triggered and evidence_c is None:
            reasons.append("MISSING_REQUIRED_EVIDENCE")
        for label, evidence, valid in (
            ("A", evidence_a, a_valid),
            ("B", evidence_b, b_valid),
            ("C", evidence_c, c_valid),
        ):
            if evidence is not None and not valid:
                reasons.append(f"{label}_{evidence.output_contract_status}")
        preserved = next((text for text, valid in ((a, a_valid), (b, b_valid), (c or "", c_valid)) if valid), "")
        return self._result(
            preserved,
            status,
            "VALID_INPUT_PRESERVED_PENDING_AGENT",
            refs,
            warnings=tuple(
                warning
                for evidence in (evidence_a, evidence_b, evidence_c)
                if evidence is not None
                for warning in evidence.warnings
            ),
            review_reasons=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _result(
        text: str,
        status: str,
        basis: str,
        refs: Sequence[str],
        *,
        warnings: tuple[str, ...] = (),
        review_reasons: tuple[str, ...] = (),
    ) -> EvidenceResolutionV1:
        return EvidenceResolutionV1(
            schema="bemarkdown-evidence-resolver-v2",
            selected_text=text,
            resolution_status=status,
            selected_provider_basis=basis,
            evidence_refs=tuple(refs),
            warnings=warnings,
            review_reasons=review_reasons,
        )


@dataclass(slots=True)
class _LifecycleEntry:
    factory: Callable[[], Any]
    resident_policy: str
    instance: Any = None
    load_count: int = 0
    unload_count: int = 0
    total_load_seconds: float = 0.0


class ModelLifecycleManager:
    """Thread-safe lazy residency manager with explicit OCR-stage cleanup."""

    def __init__(self) -> None:
        self._providers: dict[str, _LifecycleEntry] = {}
        self._lock = threading.RLock()

    def register(
        self,
        provider_id: str,
        factory: Callable[[], Any],
        *,
        resident_policy: str = "OCR_STAGE",
    ) -> None:
        with self._lock:
            if provider_id in self._providers:
                raise ValueError(f"PROVIDER_ALREADY_REGISTERED:{provider_id}")
            self._providers[provider_id] = _LifecycleEntry(factory, resident_policy)

    def acquire(self, provider_id: str) -> Any:
        with self._lock:
            entry = self._providers[provider_id]
            if entry.instance is None:
                started = time.perf_counter()
                entry.instance = entry.factory()
                entry.total_load_seconds += time.perf_counter() - started
                entry.load_count += 1
            return entry.instance

    def unload(self, provider_id: str) -> None:
        with self._lock:
            entry = self._providers[provider_id]
            if entry.instance is None:
                return
            close = getattr(entry.instance, "close", None) or getattr(
                entry.instance, "unload", None
            )
            if callable(close):
                close()
            entry.instance = None
            entry.unload_count += 1

    def end_stage(self) -> None:
        with self._lock:
            targets = [
                name
                for name, entry in self._providers.items()
                if entry.resident_policy != "PROCESS"
            ]
        for name in targets:
            self.unload(name)

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "bemarkdown-model-lifecycle-metrics-v1",
                "providers": {
                    name: {
                        "resident_policy": entry.resident_policy,
                        "loaded": entry.instance is not None,
                        "load_count": entry.load_count,
                        "unload_count": entry.unload_count,
                        "total_load_seconds": entry.total_load_seconds,
                    }
                    for name, entry in sorted(self._providers.items())
                },
            }


@dataclass(frozen=True, slots=True)
class TextOCRRuntimeProfile:
    profile_id: str
    target_vram_mib: int
    headroom_mib: int
    ppocr_residency: str
    got_residency: str
    hunyuan_residency: str
    got_batch_size: int
    hunyuan_batch_size: int
    got_generation_budget: int
    hunyuan_generation_budget: int
    backend: str = "transformers"
    async_mode: str = "BOUNDED_PREPROCESS_OVERLAP"
    queue_depth: int = 2

    def __post_init__(self) -> None:
        if self.target_vram_mib <= self.headroom_mib:
            raise ValueError("OCR_RUNTIME_PROFILE_HEADROOM_INVALID")
        if min(self.got_batch_size, self.hunyuan_batch_size, self.queue_depth) < 1:
            raise ValueError("OCR_RUNTIME_PROFILE_BATCH_OR_QUEUE_INVALID")

    @property
    def max_workload_vram_mib(self) -> int:
        return self.target_vram_mib - self.headroom_mib

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "bemarkdown-text-ocr-runtime-profile-v1",
            **asdict(self),
            "max_workload_vram_mib": self.max_workload_vram_mib,
        }


LOW_VRAM_4GB = TextOCRRuntimeProfile(
    "LOW_VRAM_4GB", 4096, 512, "UNLOAD_BEFORE_GOT", "OCR_STAGE", "LAZY_AFTER_GOT_UNLOAD", 1, 1, 256, 224
)
LOW_VRAM_6GB = TextOCRRuntimeProfile(
    "LOW_VRAM_6GB", 6144, 512, "UNLOAD_BEFORE_GOT", "OCR_STAGE", "LAZY_AFTER_GOT_UNLOAD", 1, 1, 256, 224
)
DEV_12GB_REFERENCE = TextOCRRuntimeProfile(
    "DEV_12GB_REFERENCE", 12288, 512, "PROCESS", "OCR_STAGE", "LAZY_OCR_STAGE", 4, 4, 256, 224
)


@dataclass(frozen=True, slots=True)
class AgentReviewCandidate:
    request: TextRecognitionRequest
    reason: str
    evidence_a: TextRecognitionEvidence
    evidence_b: TextRecognitionEvidence
    evidence_c: TextRecognitionEvidence | None

    def to_dict(self) -> dict[str, Any]:
        if _contains_truth(self.request.to_dict()):
            raise ValueError("TRUTH_LEAK")
        return {
            "schema": "bemarkdown-ocr-agent-review-candidate-v1",
            "document_id": self.request.document_id,
            "page_id": self.request.page_id,
            "region_id": self.request.region_id,
            "source_candidate_id": self.request.source_candidate_id,
            "bbox": list(self.request.bbox),
            "crop_ref": self.request.crop_ref,
            "crop_sha256": self.request.crop_sha256,
            "evidence": {
                "A": self.evidence_a.to_dict(),
                "B": self.evidence_b.to_dict(),
                "C": self.evidence_c.to_dict() if self.evidence_c else None,
            },
            "reason": self.reason,
            "source_provenance": dict(self.request.source_provenance),
        }


def _contains_truth(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "truth" in str(key).casefold() or _contains_truth(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_truth(item) for item in value)
    return False


def _runtime_failure_evidence(
    provider: TextEvidenceProvider,
    request: TextRecognitionRequest,
    exc: BaseException,
) -> TextRecognitionEvidence:
    status = "PROVIDER_TIMEOUT" if isinstance(exc, TimeoutError) else "PROVIDER_CRASH"
    provider_id = str(getattr(provider, "provider_id", type(provider).__name__))
    return TextRecognitionEvidence(
        provider_id=provider_id,
        model_id=provider_id,
        model_fingerprint=f"runtime-failure:{provider_id}",
        runtime_fingerprint="runtime-failure",
        bbox=request.bbox,
        text="",
        normalized_text="",
        confidence=None,
        source_crop_ref=request.crop_ref,
        source_crop_sha256=request.crop_sha256,
        latency_seconds=0.0,
        output_contract_status=status,
        warnings=(f"{type(exc).__name__}:{exc}",),
    )


def _safe_recognize(
    provider: TextEvidenceProvider, request: TextRecognitionRequest
) -> TextRecognitionEvidence:
    try:
        return provider.recognize(request)
    except Exception as exc:  # noqa: BLE001 - provider boundary becomes evidence
        return _runtime_failure_evidence(provider, request, exc)


def _safe_recognize_batch(
    provider: TextEvidenceProvider, requests: Sequence[TextRecognitionRequest]
) -> list[TextRecognitionEvidence]:
    if not requests:
        return []
    try:
        values = list(provider.recognize_batch(requests))
        if len(values) != len(requests):
            raise RuntimeError("TEXT_PROVIDER_BATCH_CARDINALITY_MISMATCH")
        return values
    except Exception as exc:  # noqa: BLE001 - isolate a failed batch per request
        if getattr(provider, 'batch_only', False):
            # Scalar dispatch is intentionally unsupported by staged providers;
            # attempting it would hide the real batch failure for every crop.
            return [_runtime_failure_evidence(provider, request, exc) for request in requests]
        return [_safe_recognize(provider, request) for request in requests]


@dataclass(frozen=True, slots=True)
class ThreeModelReplayResult:
    schema: str
    final_outputs: list[dict[str, Any]]
    metrics: dict[str, Any]
    agent_review_candidates: list[dict[str, Any]]


class ThreeModelOCRReplayHarness:
    """Exercise architecture against frozen output providers, without truth."""

    def __init__(
        self,
        provider_a: TextEvidenceProvider,
        provider_b: TextEvidenceProvider,
        provider_c: TextEvidenceProvider,
        *,
        comparator: ConsensusComparatorV1 | None = None,
        router: ThirdModelRouterV1 | None = None,
        resolver: EvidenceResolverV1 | None = None,
    ) -> None:
        self.provider_a = provider_a
        self.provider_b = provider_b
        self.provider_c = provider_c
        self.comparator = comparator or ConsensusComparatorV1()
        self.router = router or ThirdModelRouterV1()
        self.resolver = resolver or EvidenceResolverV1()

    def run(self, requests: Iterable[TextRecognitionRequest]) -> ThreeModelReplayResult:
        request_rows = list(requests)
        final_outputs: list[dict[str, Any]] = []
        agent_candidates: list[dict[str, Any]] = []
        counts = {
            "population": 0,
            "a_calls": 0,
            "b_calls": 0,
            "c_calls": 0,
            "ab_consensus_count": 0,
            "ac_consensus_count": 0,
            "bc_consensus_count": 0,
            "all_differ_count": 0,
            "runtime_failure_agent_count": 0,
            "c_trigger_count": 0,
            "total_agent_required_count": 0,
        }
        for request in request_rows:
            if _contains_truth(request.to_dict()):
                raise ValueError("TRUTH_LEAK")
        evidence_a = _safe_recognize_batch(self.provider_a, request_rows)
        evidence_b = _safe_recognize_batch(self.provider_b, request_rows)
        if len(evidence_a) != len(request_rows) or len(evidence_b) != len(request_rows):
            raise RuntimeError("AB_BATCH_CARDINALITY_MISMATCH")
        counts["population"] = len(request_rows)
        counts["a_calls"] = len(request_rows)
        counts["b_calls"] = len(request_rows)
        pending: list[tuple[int, TextRecognitionRequest, TextRecognitionEvidence, TextRecognitionEvidence, ConsensusAssessment, ThirdModelRoutingDecision]] = []
        resolved: dict[int, tuple[TextRecognitionEvidence, TextRecognitionEvidence, ConsensusAssessment, ThirdModelRoutingDecision, TextRecognitionEvidence | None, EvidenceResolutionV1]] = {}
        for index, (request, a, b) in enumerate(zip(request_rows, evidence_a, evidence_b, strict=True)):
            _validate_evidence_window(request, a)
            _validate_evidence_window(request, b)
            assessment = self.comparator.compare(request, a, b)
            routing = self.router.route(request, assessment)
            if routing.triggered:
                pending.append((index, request, a, b, assessment, routing))
            else:
                resolution = self.resolver.resolve(request, a, b, assessment, routing, None)
                resolved[index] = (a, b, assessment, routing, None, resolution)
        if pending:
            c_rows = _safe_recognize_batch(self.provider_c, [row[1] for row in pending])
            if len(c_rows) != len(pending):
                raise RuntimeError("C_BATCH_CARDINALITY_MISMATCH")
            counts["c_calls"] = len(c_rows)
            counts["c_trigger_count"] = len(c_rows)
            for pending_row, c in zip(pending, c_rows, strict=True):
                index, request, a, b, assessment, routing = pending_row
                _validate_evidence_window(request, c)
                resolution = self.resolver.resolve(request, a, b, assessment, routing, c)
                resolved[index] = (a, b, assessment, routing, c, resolution)
        for index, request in enumerate(request_rows):
            a, b, assessment, routing, c, resolution = resolved[index]
            status_to_count = {
                "CONSENSUS_AB": "ab_consensus_count",
                "CONSENSUS_AC": "ac_consensus_count",
                "CONSENSUS_BC": "bc_consensus_count",
                "AGENT_REQUIRED_ALL_DIFFER": "all_differ_count",
                "AGENT_REQUIRED_RUNTIME_FAILURE": "runtime_failure_agent_count",
            }
            if resolution.resolution_status in status_to_count:
                counts[status_to_count[resolution.resolution_status]] += 1
            if resolution.resolution_status.startswith("AGENT_REQUIRED"):
                counts["total_agent_required_count"] += 1
                agent_candidates.append(
                    AgentReviewCandidate(
                        request,
                        "ALL_DIFFER" if resolution.resolution_status.endswith("ALL_DIFFER") else "RUNTIME_FAILURE",
                        a,
                        b,
                        c,
                    ).to_dict()
                )
            final_outputs.append(
                {
                    "schema": "bemarkdown-three-model-replay-output-v2",
                    "sample_id": request.region_id,
                    "selected_text": resolution.selected_text,
                    "normalized_output": normalize_text(resolution.selected_text),
                    "resolution_status": resolution.resolution_status,
                    "selected_provider_basis": resolution.selected_provider_basis,
                    "third_model_triggered": routing.triggered,
                    "third_model_trigger_reasons": list(routing.reasons),
                    "consensus_statuses": list(assessment.statuses),
                    "review_reasons": list(resolution.review_reasons),
                    "comparator": assessment.to_dict(),
                    "third_model_router": routing.to_dict(),
                    "resolver": resolution.to_dict(),
                    "evidence": {
                        "A": a.to_dict(),
                        "B": b.to_dict(),
                        "C": c.to_dict() if c is not None else None,
                    },
                }
            )
        population = counts["population"]
        metrics: dict[str, Any] = dict(counts)
        metrics["c_trigger_rate"] = (
            counts["c_trigger_count"] / population if population else 0.0
        )
        metrics["truth_blind"] = True
        for key in ("ab_consensus", "ac_consensus", "bc_consensus", "all_differ", "total_agent_required"):
            metrics[f"{key}_rate"] = counts[f"{key}_count"] / population if population else 0.0
        # Read-only compatibility aliases for the v1 smoke/report tooling.
        metrics["ab_accepted_without_c"] = counts["ab_consensus_count"]
        metrics["review_required_count"] = counts["total_agent_required_count"]
        return ThreeModelReplayResult(
            schema="bemarkdown-three-model-replay-v2",
            final_outputs=final_outputs,
            metrics=metrics,
            agent_review_candidates=agent_candidates,
        )


def _validate_evidence_window(
    request: TextRecognitionRequest, evidence: TextRecognitionEvidence
) -> None:
    """Reject evidence that is not bound to the request's canonical crop."""

    if (
        evidence.source_crop_sha256 != request.crop_sha256
        or evidence.source_crop_ref != request.crop_ref
        or tuple(evidence.bbox) != tuple(request.bbox)
    ):
        raise RuntimeError(
            "TEXT_EVIDENCE_SPATIAL_WINDOW_MISMATCH:"
            f"{request.region_id}:{evidence.provider_id}"
        )


class ThreeModelTextRuntimeAdapter:
    """Duck-typed modular-PDF OCR runtime backed by the A/B/C architecture.

    ``PdfContentAdapterExecutor`` calls ``recognize_route`` when available, so
    the adapter can build a complete request and later expose provenance for the
    RegionContentIR row. Reliable native-text routes never enter this runtime.
    """

    def __init__(
        self,
        provider_a: TextEvidenceProvider,
        provider_b: TextEvidenceProvider,
        provider_c: TextEvidenceProvider,
        *,
        comparator: ConsensusComparatorV1 | None = None,
        router: ThirdModelRouterV1 | None = None,
        resolver: EvidenceResolverV1 | None = None,
    ) -> None:
        self.provider_a = provider_a
        self.provider_b = provider_b
        self.provider_c = provider_c
        self.comparator = comparator or ConsensusComparatorV1()
        self.router = router or ThirdModelRouterV1()
        self.resolver = resolver or EvidenceResolverV1()
        self._active_route: Mapping[str, Any] | None = None
        self._last_provenance: dict[str, Any] | None = None
        self._counts = {"A": 0, "B": 0, "C": 0}

    def recognize_route(
        self, route: Mapping[str, Any], crop: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        self._active_route = route
        request = self._request(route, crop)
        a = _safe_recognize(self.provider_a, request)
        b = _safe_recognize(self.provider_b, request)
        self._counts["A"] += 1
        self._counts["B"] += 1
        assessment = self.comparator.compare(request, a, b)
        routing = self.router.route(request, assessment)
        c = None
        if routing.triggered:
            c = _safe_recognize(self.provider_c, request)
            self._counts["C"] += 1
        resolution = self.resolver.resolve(request, a, b, assessment, routing, c)
        self._last_provenance = {
            "schema": THREE_MODEL_OCR_SCHEMA,
            "request": request.to_dict(),
            "evidence": {
                "A": a.to_dict(),
                "B": b.to_dict(),
                "C": c.to_dict() if c is not None else None,
            },
            "comparator": assessment.to_dict(),
            "third_model_router": routing.to_dict(),
            "resolver": resolution.to_dict(),
        }
        if resolution.resolution_status.startswith("AGENT_REQUIRED"):
            self._last_provenance["agent_review_candidate"] = AgentReviewCandidate(
                request,
                "ALL_DIFFER"
                if resolution.resolution_status == "AGENT_REQUIRED_ALL_DIFFER"
                else "RUNTIME_FAILURE",
                a,
                b,
                c,
            ).to_dict()
        if not resolution.selected_text:
            return []
        selected_confidence = (
            a.confidence
            if resolution.selected_text == assessment.normalized_a
            else None
        )
        return [
            {
                "text": resolution.selected_text,
                "confidence": selected_confidence,
                "reading_sequence": 0,
                "resolution_status": resolution.resolution_status,
                "review_reasons": list(resolution.review_reasons),
                "third_model_triggered": routing.triggered,
            }
        ]

    def recognize(self, crop: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self._active_route is None:
            raise RuntimeError("THREE_MODEL_ROUTE_CONTEXT_REQUIRED")
        return self.recognize_route(self._active_route, crop)

    def recognize_direct(self, crop: Mapping[str, Any]) -> list[dict[str, Any]]:
        return self.recognize(crop)

    def consume_last_provenance(self) -> dict[str, Any] | None:
        value = self._last_provenance
        self._last_provenance = None
        return value

    def fingerprint(self) -> dict[str, Any]:
        return {
            "det_model_id": "PP-OCRv6",
            "rec_model_id": "GOT-OCR2.0+conditional-HunyuanOCR-1.5",
            "det_model_fingerprint": "provider-neutral-A",
            "rec_model_fingerprint": "provider-neutral-B+C",
            "architecture_schema": THREE_MODEL_OCR_SCHEMA,
        }

    def metrics(self) -> dict[str, Any]:
        total = self._counts["A"]
        return {
            "schema": "bemarkdown-three-model-runtime-metrics-v1",
            "a_calls": self._counts["A"],
            "b_calls": self._counts["B"],
            "c_calls": self._counts["C"],
            "c_trigger_rate": self._counts["C"] / total if total else 0.0,
        }

    def close(self) -> None:
        seen: set[int] = set()
        for provider in (self.provider_a, self.provider_b, self.provider_c):
            runtime = getattr(provider, "runtime", None)
            if runtime is None or id(runtime) in seen:
                continue
            seen.add(id(runtime))
            close = getattr(runtime, "close", None) or getattr(runtime, "unload", None)
            if callable(close):
                close()

    @staticmethod
    def _request(
        route: Mapping[str, Any], crop: Mapping[str, Any]
    ) -> TextRecognitionRequest:
        return build_text_recognition_request(route, crop)


def build_text_recognition_request(
    route: Mapping[str, Any], crop: Mapping[str, Any]
) -> TextRecognitionRequest:
    """Build the frozen provider-neutral request for one current-run OCR crop."""

    provenance = route.get("provenance")
    route_provenance = provenance if isinstance(provenance, Mapping) else {}
    candidate_ids = list(route.get("input_candidate_ids") or ())
    return TextRecognitionRequest(
        document_id=str(route["document_id"]),
        page_id=f"{route['document_id']}:page:{route['page_index']}",
        region_id=str(route["route_id"]),
        source_candidate_id=(
            str(candidate_ids[0]) if candidate_ids else str(route["route_id"])
        ),
        bbox=tuple(float(value) for value in crop["bbox_pdf_pt"]),  # type: ignore[arg-type]
        crop_ref=str(crop["path"]),
        crop_sha256=str(crop["content_sha256"]),
        route_kind=str(route["adapter"]),
        text_track=(
            "PAGE_TEXT_RECOVERY"
            if route["adapter"] == "PAGE_VISUAL_TEXT_RECOVERY"
            else "TEXT_REGION_CROP"
        ),
        risk_hints=tuple(
            str(value)
            for value in route_provenance.get("ocr_validator_warnings", ())
        ),
        source_provenance={
            "crop": dict(crop),
            "route_reason_codes": list(route.get("decision_reason_codes") or ()),
            "source_profile": route_provenance.get("source_profile"),
            **(
                {
                    "page_recovery_bounded_unit": copy.deepcopy(
                        route_provenance["page_recovery_bounded_unit"]
                    )
                }
                if isinstance(
                    route_provenance.get("page_recovery_bounded_unit"), Mapping
                )
                else {}
            ),
            **(
                {
                    "page_recovery_chunk": copy.deepcopy(
                        route_provenance["page_recovery_chunk"]
                    )
                }
                if isinstance(route_provenance.get("page_recovery_chunk"), Mapping)
                else {}
            ),
        },
    )


class ThreeModelTextOrchestrator:
    """Runtime seam with batch, async queue, timing, and generation config hooks."""

    def __init__(
        self,
        harness: ThreeModelOCRReplayHarness,
        *,
        timing_hook: Callable[[Mapping[str, Any]], None] | None = None,
        generation_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.harness = harness
        self.timing_hook = timing_hook
        self.generation_config = dict(generation_config or {})

    def run_batch(
        self, requests: Sequence[TextRecognitionRequest]
    ) -> ThreeModelReplayResult:
        started = time.perf_counter()
        result = self.harness.run(requests)
        if self.timing_hook is not None:
            self.timing_hook(
                {
                    "event": "three_model_batch_complete",
                    "request_count": len(requests),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        return result

    async def submit(
        self, requests: Sequence[TextRecognitionRequest]
    ) -> ThreeModelReplayResult:
        return await asyncio.to_thread(self.run_batch, requests)


def attach_text_resolution_provenance(
    region_content_ir: dict[str, Any],
    *,
    assessment: ConsensusAssessment,
    routing: ThirdModelRoutingDecision,
    resolution: EvidenceResolutionV1,
) -> dict[str, Any]:
    """Add optional provenance without changing existing RegionContentIR fields."""

    region_content_ir["text_evidence_refs"] = list(resolution.evidence_refs)
    region_content_ir["consensus_status"] = list(assessment.statuses)
    region_content_ir["third_model_triggered"] = routing.triggered
    region_content_ir["third_model_trigger_reasons"] = list(routing.reasons)
    region_content_ir["resolution_status"] = resolution.resolution_status
    region_content_ir["text_resolution"] = resolution.to_dict()
    return region_content_ir
