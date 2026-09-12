"""Intrinsic, specialist, resolver, and residual risk separation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .evidence_resolver import EvidenceResolution, ResolverState

RISK_MANIFEST_V2_SCHEMA = "bemarkdown-risk-manifest-v2"


def _material_tokens(value: Any) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return tuple(
        re.findall(
            r"\d+(?:\.\d+)?|m/s(?:\^?[23])?|[=<>≤≥±+\-×÷^_²³₀-₉]|[α-ωΑ-Ω]",
            text,
        )
    )


def sensitive_conflict(first: Any, second: Any) -> dict[str, Any]:
    left = _material_tokens(first)
    right = _material_tokens(second)
    conflict = left != right
    return {
        "schema": "bemarkdown-sensitive-content-risk-v1",
        "left_tokens": list(left),
        "right_tokens": list(right),
        "conflict": conflict,
        "severity": "HIGH"
        if conflict and (left or right)
        else "WATCH"
        if str(first) != str(second)
        else "SAFE",
    }


@dataclass(frozen=True, slots=True)
class RiskManifestV2Node:
    schema: str
    node_id: str
    intrinsic_risk: str
    intrinsic_reasons: tuple[str, ...]
    specialist_policy: dict[str, Any]
    specialist_execution_state: str
    primary_evidence: Any
    secondary_evidence: Any
    resolver_state: str
    source_coverage_state: str
    residual_risk: str
    residual_reasons: tuple[str, ...]
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["intrinsic_reasons"] = list(self.intrinsic_reasons)
        result["residual_reasons"] = list(self.residual_reasons)
        return result


class RiskManifestV2Builder:
    def build_node(
        self,
        *,
        node_id: str,
        intrinsic_risk: str,
        intrinsic_reasons: Iterable[str],
        specialist_policy: Mapping[str, Any],
        primary_evidence: Any,
        secondary_evidence: Any,
        resolver: EvidenceResolution,
        source_coverage_state: str,
    ) -> RiskManifestV2Node:
        reasons = list(resolver.reasons)
        execution = str(specialist_policy.get("execution_state", "NOT_REQUIRED"))
        if source_coverage_state in {
            "SOURCE_REGION_UNCOVERED",
            "SOURCE_CONTENT_MISSING",
        }:
            residual = "CRITICAL"
            reasons.append(source_coverage_state)
        elif resolver.state in {
            ResolverState.RESIDUAL_CONFLICT,
            ResolverState.TECHNICAL_FAILURE,
            ResolverState.INSUFFICIENT_EVIDENCE,
        }:
            residual = "HIGH"
        elif intrinsic_risk == "WATCH":
            residual = "WATCH"
        else:
            residual = "SAFE"
        action = (
            "FUTURE_AGENT_REQUIRED"
            if residual in {"HIGH", "CRITICAL"}
            else "LOCAL_WATCH"
            if residual == "WATCH"
            else "LOCAL_ACCEPT"
        )
        return RiskManifestV2Node(
            schema=RISK_MANIFEST_V2_SCHEMA,
            node_id=node_id,
            intrinsic_risk=intrinsic_risk,
            intrinsic_reasons=tuple(intrinsic_reasons),
            specialist_policy=dict(specialist_policy),
            specialist_execution_state=execution,
            primary_evidence=primary_evidence,
            secondary_evidence=secondary_evidence,
            resolver_state=resolver.state.value,
            source_coverage_state=source_coverage_state,
            residual_risk=residual,
            residual_reasons=tuple(dict.fromkeys(reasons)),
            recommended_action=action,
        )


def build_risk_manifest_v2(
    *, document_id: str, page_id: str, nodes: Iterable[RiskManifestV2Node]
) -> dict[str, Any]:
    rows = [node.to_dict() for node in nodes]
    return {
        "schema": RISK_MANIFEST_V2_SCHEMA,
        "document_id": document_id,
        "page_id": page_id,
        "risks": rows,
        "summary": {
            severity: sum(row["residual_risk"] == severity for row in rows)
            for severity in ("SAFE", "WATCH", "HIGH", "CRITICAL")
        },
    }
