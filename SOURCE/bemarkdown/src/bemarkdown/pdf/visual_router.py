"""Region-level native/visual source routing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

REGION_SOURCE_DECISION_SCHEMA = "bemarkdown-region-source-decision-v1"


class RegionSourceDecision(str, Enum):
    NATIVE_TEXT_RELIABLE = "NATIVE_TEXT_RELIABLE"
    NATIVE_TEXT_PARTIAL = "NATIVE_TEXT_PARTIAL"
    VISUAL_TEXT_REQUIRED = "VISUAL_TEXT_REQUIRED"
    NATIVE_IMAGE_AVAILABLE = "NATIVE_IMAGE_AVAILABLE"
    VISUAL_IMAGE_CROP_REQUIRED = "VISUAL_IMAGE_CROP_REQUIRED"
    FORMULA_VISUAL_REQUIRED = "FORMULA_VISUAL_REQUIRED"
    TABLE_VISUAL_REQUIRED = "TABLE_VISUAL_REQUIRED"


@dataclass(frozen=True, slots=True)
class RegionRoutingResult:
    schema: str
    page_id: str
    region_id: str
    label: str
    bbox_pdf_pt: tuple[float, float, float, float]
    decision: RegionSourceDecision
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bbox_pdf_pt"] = list(self.bbox_pdf_pt)
        result["decision"] = self.decision.value
        result["reasons"] = list(self.reasons)
        return result


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _bbox(row: Any) -> tuple[float, float, float, float]:
    values = _value(row, "bbox_pdf_pt")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _overlap_smaller(first: Sequence[float], second: Sequence[float]) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller = min(first_area, second_area)
    return overlap / smaller if smaller else 0.0


def route_regions(
    *,
    page_profile: Any,
    native_evidence: Iterable[Any],
    layout_regions: Iterable[Mapping[str, Any]],
) -> list[RegionRoutingResult]:
    native = list(native_evidence)
    profile_state = str(_value(page_profile, "profile_state", "VISUAL_REQUIRED"))
    if profile_state.startswith("ProfileState."):
        profile_state = profile_state.split(".")[-1]
    page_id = str(_value(page_profile, "page_id"))
    results: list[RegionRoutingResult] = []
    for index, region in enumerate(layout_regions):
        label = str(region.get("label", "unknown")).lower()
        bbox = tuple(float(value) for value in region["bbox_pdf_pt"])
        overlaps = [row for row in native if _overlap_smaller(bbox, _bbox(row)) >= 0.5]
        native_text = any(_value(row, "evidence_kind") == "TEXT" for row in overlaps)
        native_image = any(_value(row, "evidence_kind") == "IMAGE" for row in overlaps)
        if "formula" in label or label in {"equation"}:
            decision = RegionSourceDecision.FORMULA_VISUAL_REQUIRED
            reasons = ("FORMULA_REQUIRES_VISUAL_RECOGNITION",)
        elif label == "table":
            decision = RegionSourceDecision.TABLE_VISUAL_REQUIRED
            reasons = ("TABLE_REQUIRES_VISUAL_STRUCTURE",)
        elif label in {"image", "chart", "seal", "header_image", "footer_image"}:
            if native_image:
                decision = RegionSourceDecision.NATIVE_IMAGE_AVAILABLE
                reasons = ("OVERLAPPING_NATIVE_IMAGE_OBJECT",)
            else:
                decision = RegionSourceDecision.VISUAL_IMAGE_CROP_REQUIRED
                reasons = ("NO_MATCHING_NATIVE_IMAGE_OBJECT",)
        elif native_text and profile_state == "NATIVE_RELIABLE":
            decision = RegionSourceDecision.NATIVE_TEXT_RELIABLE
            reasons = ("PAGE_AND_REGION_NATIVE_TEXT_RELIABLE",)
        elif native_text:
            decision = RegionSourceDecision.NATIVE_TEXT_PARTIAL
            reasons = ("NATIVE_TEXT_PRESENT_BUT_PAGE_NOT_RELIABLE",)
        else:
            decision = RegionSourceDecision.VISUAL_TEXT_REQUIRED
            reasons = ("NO_OVERLAPPING_NATIVE_TEXT",)
        results.append(
            RegionRoutingResult(
                schema=REGION_SOURCE_DECISION_SCHEMA,
                page_id=page_id,
                region_id=str(region.get("region_id", f"region-{index}")),
                label=label,
                bbox_pdf_pt=bbox,  # type: ignore[arg-type]
                decision=decision,
                reasons=reasons,
            )
        )
    return results
