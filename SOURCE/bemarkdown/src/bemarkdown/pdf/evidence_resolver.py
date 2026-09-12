"""Policy-driven primary/secondary evidence resolution."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .specialist_router import EvidenceExecutionState


class ResolverState(str, Enum):
    CONVERGED = "CONVERGED"
    SOURCE_NATIVE_ACCEPTED = "SOURCE_NATIVE_ACCEPTED"
    PRIMARY_PRESERVED_WITH_SUPPORT = "PRIMARY_PRESERVED_WITH_SUPPORT"
    RESIDUAL_CONFLICT = "RESIDUAL_CONFLICT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


@dataclass(frozen=True, slots=True)
class EvidenceResolution:
    schema: str
    state: ResolverState
    accepted_evidence: Any
    primary_preserved: bool
    agreement: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        result["reasons"] = list(self.reasons)
        return result


def _content(value: Any) -> Any:
    if isinstance(value, Mapping) and "content" in value:
        return value["content"]
    return value


def _canonical(value: Any) -> str:
    value = _content(value)
    if isinstance(value, str):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class EvidenceResolver:
    def resolve(
        self,
        *,
        primary_evidence: Any,
        secondary_evidence: Any,
        execution_state: EvidenceExecutionState,
    ) -> EvidenceResolution:
        primary_source = (
            str(primary_evidence.get("source", ""))
            if isinstance(primary_evidence, Mapping)
            else ""
        )
        if execution_state is EvidenceExecutionState.NOT_REQUIRED:
            state = (
                ResolverState.SOURCE_NATIVE_ACCEPTED
                if primary_source.startswith("NATIVE")
                else ResolverState.PRIMARY_PRESERVED_WITH_SUPPORT
            )
            return EvidenceResolution(
                "bemarkdown-evidence-resolution-v1",
                state,
                primary_evidence,
                True,
                False,
                ("SECONDARY_EVIDENCE_NOT_REQUIRED",),
            )
        if execution_state in {
            EvidenceExecutionState.FAILED,
            EvidenceExecutionState.UNAVAILABLE,
            EvidenceExecutionState.ALIGNMENT_FAILED,
        }:
            return EvidenceResolution(
                "bemarkdown-evidence-resolution-v1",
                ResolverState.TECHNICAL_FAILURE,
                primary_evidence,
                True,
                False,
                (execution_state.value,),
            )
        if secondary_evidence is None:
            return EvidenceResolution(
                "bemarkdown-evidence-resolution-v1",
                ResolverState.INSUFFICIENT_EVIDENCE,
                primary_evidence,
                True,
                False,
                ("EXPECTED_SECONDARY_EVIDENCE_MISSING",),
            )
        if _canonical(primary_evidence) == _canonical(secondary_evidence):
            return EvidenceResolution(
                "bemarkdown-evidence-resolution-v1",
                ResolverState.CONVERGED,
                primary_evidence,
                True,
                True,
                ("PRIMARY_SECONDARY_AGREE",),
            )
        return EvidenceResolution(
            "bemarkdown-evidence-resolution-v1",
            ResolverState.RESIDUAL_CONFLICT,
            primary_evidence,
            True,
            False,
            ("PRIMARY_SECONDARY_DISAGREE_PRESERVED",),
        )
