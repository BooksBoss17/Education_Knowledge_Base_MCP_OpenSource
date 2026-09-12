"""Intrinsic-risk driven specialist planning and execution state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .typed_authority import ContentType
from .visual_router import RegionSourceDecision


class EvidenceExecutionState(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    TRIGGERED = "TRIGGERED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    ALIGNMENT_FAILED = "ALIGNMENT_FAILED"


@dataclass(frozen=True, slots=True)
class SpecialistPlan:
    schema: str
    content_type: str
    models: tuple[str, ...]
    evidence_expected: bool
    evidence_triggered: bool
    evidence_executed: bool
    evidence_available: bool
    alignment_state: str
    source_coverage_state: str
    execution_state: EvidenceExecutionState
    trigger_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["models"] = list(self.models)
        result["execution_state"] = self.execution_state.value
        result["trigger_reasons"] = list(self.trigger_reasons)
        return result


class SpecialistRouter:
    """Plan costly specialist evidence while keeping plus-L cheap/always-on."""

    def route(
        self,
        *,
        content_type: ContentType,
        region_decision: RegionSourceDecision,
        intrinsic_reasons: Iterable[str],
        has_sensitive_content: bool,
        source_coverage_state: str = "SOURCE_COVERED",
    ) -> SpecialistPlan:
        reasons = list(intrinsic_reasons)
        models = ["PP-DocLayout_plus-L"]
        required = False
        if content_type is ContentType.FORMULA:
            required = True
            models.append("PP-FormulaNet_plus-L")
            reasons.append("FORMULA_TARGETED_SPECIALIST")
        elif content_type is ContentType.TABLE:
            required = True
            models.append("TableEngine")
            reasons.append("TABLE_HIGH_COVERAGE_SPECIALIST")
        elif content_type is ContentType.TEXT and (
            region_decision
            in {
                RegionSourceDecision.VISUAL_TEXT_REQUIRED,
                RegionSourceDecision.NATIVE_TEXT_PARTIAL,
            }
            or bool(reasons)
            or has_sensitive_content
        ):
            required = True
            models.append("PP-OCRv6")
            reasons.append("TEXT_RISK_SPECIALIST")
        elif content_type in {ContentType.HEADING, ContentType.READING_ORDER}:
            required = bool(reasons)
        state = (
            EvidenceExecutionState.TRIGGERED
            if required
            else EvidenceExecutionState.NOT_REQUIRED
        )
        return SpecialistPlan(
            schema="bemarkdown-specialist-plan-v1",
            content_type=content_type.value,
            models=tuple(models),
            evidence_expected=required,
            evidence_triggered=required,
            evidence_executed=False,
            evidence_available=False,
            alignment_state="PENDING" if required else "NOT_REQUIRED",
            source_coverage_state=source_coverage_state,
            execution_state=state,
            trigger_reasons=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def with_execution(
        plan: SpecialistPlan,
        *,
        state: EvidenceExecutionState,
        available: bool,
        alignment_state: str,
    ) -> SpecialistPlan:
        return SpecialistPlan(
            **{
                **plan.to_dict(),
                "models": plan.models,
                "trigger_reasons": plan.trigger_reasons,
                "execution_state": state,
                "evidence_executed": state
                in {
                    EvidenceExecutionState.EXECUTED,
                    EvidenceExecutionState.ALIGNMENT_FAILED,
                },
                "evidence_available": available,
                "alignment_state": alignment_state,
            }
        )
