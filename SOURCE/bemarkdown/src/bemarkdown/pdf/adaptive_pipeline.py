"""Native-first adaptive orchestration contracts and Pareto profiles."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

ADAPTIVE_HANDOFF_SCHEMA = "bemarkdown-adaptive-risk-handoff-v1"


class AdaptiveProfile(str, Enum):
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    FAST = "FAST"


@dataclass(frozen=True, slots=True)
class AdaptiveProfileConfig:
    profile: AdaptiveProfile
    max_candidates_per_page_type: int
    max_candidates_per_page: int | None
    native_reliable_text_specialist_on_sensitive_only: bool
    table_high_coverage: bool
    layout_always_on: bool = True

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["profile"] = self.profile.value
        return result


PROFILE_CONFIGS = {
    AdaptiveProfile.CONSERVATIVE: AdaptiveProfileConfig(
        AdaptiveProfile.CONSERVATIVE, 2, None, True, True
    ),
    AdaptiveProfile.BALANCED: AdaptiveProfileConfig(
        AdaptiveProfile.BALANCED, 1, None, True, True
    ),
    AdaptiveProfile.FAST: AdaptiveProfileConfig(AdaptiveProfile.FAST, 1, 2, True, True),
}


class AdaptivePipeline:
    """Run the deterministic native front-end behind one narrow interface."""

    def inspect_source(
        self,
        source_pdf: Any,
        *,
        document_id: str,
        layout_regions_by_page: Mapping[int, Iterable[Mapping[str, Any]]],
    ) -> dict[str, Any]:
        from .native_evidence import NativeEvidenceExtractor
        from .object_profiler import PDFObjectProfiler
        from .visual_router import route_regions

        profiles = PDFObjectProfiler().profile(source_pdf, document_id=document_id)
        native = NativeEvidenceExtractor().extract(source_pdf, profiles=profiles)
        by_page: dict[int, list[Any]] = defaultdict(list)
        for row in native:
            by_page[row.page_index].append(row)
        routing = [
            {
                "page_id": profile.page_id,
                "profile_state": profile.profile_state.value,
                "regions": [
                    row.to_dict()
                    for row in route_regions(
                        page_profile=profile,
                        native_evidence=by_page[profile.page_index],
                        layout_regions=layout_regions_by_page.get(
                            profile.page_index, []
                        ),
                    )
                ],
            }
            for profile in profiles
        ]
        return {
            "schema": "bemarkdown-native-first-adaptive-inspection-v1",
            "document_id": document_id,
            "profiles": [row.to_dict() for row in profiles],
            "native_evidence": [row.to_dict() for row in native],
            "routing": routing,
            "orientation_correction": False,
            "document_unwarping": False,
        }


def _rank(node: Mapping[str, Any]) -> tuple[int, int, str]:
    severity = {"SAFE": 0, "WATCH": 1, "HIGH": 2, "CRITICAL": 3}.get(
        str(node.get("intrinsic_risk", node.get("severity", "SAFE"))), 0
    )
    reasons = set(node.get("intrinsic_reasons", node.get("all_conflict_types", [])))
    material = int(
        bool(
            reasons
            & {
                "TEXT_NUMBER_CONFLICT",
                "TEXT_UNIT_CONFLICT",
                "TEXT_SYMBOL_CONFLICT",
                "FORMULA_SUBSCRIPT_CONFLICT",
                "FORMULA_SUPERSCRIPT_CONFLICT",
                "FORMULA_GLYPH_CONFLICT",
                "FORMULA_TOKEN_CONFLICT",
                "TABLE_CELL_POSITION_CONFLICT",
                "TABLE_STRUCTURE_CONFLICT",
                "SOURCE_CONTENT_MISSING",
                "READING_ORDER_CONFLICT",
            }
        )
    )
    return (-severity, -material, str(node.get("node_id", node.get("risk_id", ""))))


def select_residual_candidates(
    nodes: Iterable[Mapping[str, Any]], profile: AdaptiveProfile | str
) -> set[str]:
    config = PROFILE_CONFIGS[AdaptiveProfile(profile)]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for node in nodes:
        if str(node.get("intrinsic_risk", node.get("severity"))) not in {
            "HIGH",
            "CRITICAL",
        }:
            continue
        page_id = str(node.get("page_id"))
        categories = node.get("risk_categories") or [
            node.get("content_type", node.get("risk_type", "LAYOUT"))
        ]
        for category in categories:
            groups[(page_id, str(category))].append(node)
    selected: set[str] = set()
    per_page: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for (page_id, _content_type), rows in sorted(groups.items()):
        chosen = sorted(rows, key=_rank)[: config.max_candidates_per_page_type]
        per_page[page_id].extend(chosen)
    for page_id, rows in sorted(per_page.items()):
        if config.max_candidates_per_page is not None:
            rows = sorted(rows, key=_rank)[: config.max_candidates_per_page]
        selected.update(str(row.get("node_id", row.get("risk_id"))) for row in rows)
    return selected


def build_adaptive_handoff(
    *, document_id: str, risks: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    candidates = []
    required = {
        "node_id",
        "page_id",
        "source_bbox",
        "source_crop",
        "primary_evidence",
        "secondary_evidence",
        "resolver_state",
        "residual_risk",
        "residual_reasons",
        "context_refs",
        "model_fingerprints",
    }
    for risk in risks:
        if risk.get("residual_risk") not in {"HIGH", "CRITICAL"}:
            continue
        missing = required.difference(risk)
        if missing:
            raise ValueError(f"ADAPTIVE_HANDOFF_FIELDS_MISSING:{sorted(missing)}")
        candidates.append({key: risk[key] for key in sorted(required)})
    return {
        "schema": ADAPTIVE_HANDOFF_SCHEMA,
        "document_id": document_id,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "vision_agent_invoked": False,
        "phase": "PHASE_7B_RESIDUAL_RISK_ONLY",
    }


@dataclass(frozen=True, slots=True)
class AdaptiveVsOracleMetrics:
    schema: str
    pages: int
    specialist_calls_avoided: dict[str, int]
    vl_pages_avoided: int
    vl_regions_avoided: int
    evidence_rows_avoided: int
    oracle_high_critical_retained: int
    oracle_conflict_types_retained: tuple[str, ...]
    truth_material_errors_detected: int
    truth_material_errors_missed: int
    runtime_reduction_ratio: float
    peak_vram_mib: int
    peak_ram_bytes: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["oracle_conflict_types_retained"] = list(
            self.oracle_conflict_types_retained
        )
        return result
