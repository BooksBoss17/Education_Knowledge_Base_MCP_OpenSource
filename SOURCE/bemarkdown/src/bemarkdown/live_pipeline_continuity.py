from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pdf_content_router import plan_page_content, validate_content_route_plan
from .pdf_layout_fusion import build_page_fusion_ir, validate_page_fusion_ir

LAYOUT_CONTINUITY_SCHEMA = "bemarkdown-layout-continuity-diagnostic-v0"
REPLAY_AUDIT_SCHEMA = "bemarkdown-frozen-replay-leakage-audit-v0"
ASSET_IDENTITY_AUDIT_SCHEMA = "bemarkdown-document-ir-asset-identity-audit-v0"

REPLAY_FLAGS = (
    "frozen_fusion_artifact_used",
    "frozen_router_artifact_used",
    "frozen_content_ir_used",
    "benchmark_replay_path_used",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def pixel_semantic_fingerprint(path: str | Path) -> dict[str, Any]:
    """Fingerprint decoded RGB pixels, independent from PNG/JPEG serialization."""

    path = Path(path)
    with Image.open(path) as source:
        image = source.convert("RGB")
        payload = image.tobytes()
        return {
            "schema": "bemarkdown-render-pixel-semantic-fingerprint-v0",
            "width": image.width,
            "height": image.height,
            "mode": "RGB",
            "pixel_sha256": hashlib.sha256(payload).hexdigest(),
        }


def _without_runtime_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in {"raw_detection_id", "page_render_identity"}
    }


def _numeric_equal(
    historical: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    bbox_tolerance_px: float,
    score_tolerance: float,
) -> bool:
    if (
        historical.get("raw_class_id") != live.get("raw_class_id")
        or historical.get("raw_label") != live.get("raw_label")
        or historical.get("model_identity") != live.get("model_identity")
    ):
        return False
    historical_bbox = historical.get("raw_bbox_render_px") or []
    live_bbox = live.get("raw_bbox_render_px") or []
    if len(historical_bbox) != len(live_bbox):
        return False
    if any(
        abs(float(first) - float(second)) > bbox_tolerance_px
        for first, second in zip(historical_bbox, live_bbox, strict=True)
    ):
        return False
    return (
        abs(float(historical.get("raw_score", 0)) - float(live.get("raw_score", 0)))
        <= score_tolerance
    )


def _detection_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    bbox = row.get("raw_bbox_render_px") or [0, 0, 0, 0]
    return (
        str(row.get("raw_label")),
        int(row.get("raw_class_id", -1)),
        *(round(float(value), 3) for value in bbox),
        round(float(row.get("raw_score", 0)), 4),
    )


def classify_layout_continuity(
    historical_detections: Sequence[Mapping[str, Any]],
    live_detections: Sequence[Mapping[str, Any]],
    historical_regions: Sequence[Mapping[str, Any]],
    live_regions: Sequence[Mapping[str, Any]],
    *,
    encoded_render_equal: bool | None = None,
    pixel_semantic_equal: bool | None = None,
    bbox_tolerance_px: float = 0.5,
    score_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Classify render, raw, canonical, and material Layout continuity separately."""

    historical_detections = [dict(row) for row in historical_detections]
    live_detections = [dict(row) for row in live_detections]
    historical_regions = [dict(row) for row in historical_regions]
    live_regions = [dict(row) for row in live_regions]
    classifications: list[str] = []
    count_changed = len(historical_detections) != len(live_detections)
    historical_labels = [str(row.get("raw_label")) for row in historical_detections]
    live_labels = [str(row.get("raw_label")) for row in live_detections]
    label_changed = Counter(historical_labels) != Counter(live_labels)
    order_changed = historical_labels != live_labels and not label_changed
    canonical_count_changed = len(historical_regions) != len(live_regions)
    historical_types = [str(row.get("semantic_type")) for row in historical_regions]
    live_types = [str(row.get("semantic_type")) for row in live_regions]
    semantic_order_changed = historical_types != live_types

    if count_changed:
        classifications.append("COUNT_CHANGED")
    if label_changed:
        classifications.append("LABEL_CHANGED")
    if canonical_count_changed:
        classifications.append("CANONICAL_REGION_CHANGED")
    if semantic_order_changed:
        classifications.append("SEMANTIC_ORDER_CHANGED")

    raw_exact = historical_detections == live_detections
    canonical_exact = historical_regions == live_regions
    without_identity_equal = (
        [_without_runtime_identity(row) for row in historical_detections]
        == [_without_runtime_identity(row) for row in live_detections]
    )
    numeric_equal = False
    if not count_changed and not label_changed:
        numeric_equal = all(
            _numeric_equal(
                historical,
                live,
                bbox_tolerance_px=bbox_tolerance_px,
                score_tolerance=score_tolerance,
            )
            for historical, live in zip(
                historical_detections, live_detections, strict=True
            )
        )
    unordered_equal = (
        not count_changed
        and not label_changed
        and sorted(historical_detections, key=_detection_sort_key)
        == sorted(live_detections, key=_detection_sort_key)
    )

    material = canonical_count_changed or semantic_order_changed
    if material:
        classifications.append("MATERIAL_CONTINUITY_CHANGE")
    elif order_changed or (not raw_exact and unordered_equal):
        classifications.append("ORDER_ONLY")
    elif (
        not raw_exact
        and without_identity_equal
        and encoded_render_equal is False
        and pixel_semantic_equal is True
    ):
        classifications.append("RENDER_IDENTITY_ONLY")
    elif not raw_exact and numeric_equal:
        classifications.append("FLOAT_TOLERANCE_ONLY")
    elif (
        not raw_exact
        and semantic_sha256(historical_detections)
        == semantic_sha256(live_detections)
    ) or (raw_exact and not canonical_exact):
        classifications.append("RAW_SERIALIZATION_ONLY")
    elif not raw_exact and without_identity_equal:
        classifications.append("RENDER_IDENTITY_ONLY")
    elif raw_exact and canonical_exact:
        classifications.append("IDENTICAL")

    # Dict insertion order never changes canonical JSON. If ordinary equality says
    # equal while object representation differs, call out serialization explicitly.
    if (
        raw_exact
        and _canonical_json(historical_detections)
        == _canonical_json(live_detections)
        and any(
            list(historical) != list(live)
            for historical, live in zip(
                historical_detections, live_detections, strict=True
            )
        )
    ):
        classifications = ["RAW_SERIALIZATION_ONLY"]

    if not classifications:
        classifications.append("RAW_VALUE_CHANGED_NON_MATERIAL")
    primary = classifications[-1]
    if material:
        primary = "MATERIAL_CONTINUITY_CHANGE"
    return {
        "schema": LAYOUT_CONTINUITY_SCHEMA,
        "primary_classification": primary,
        "classifications": sorted(set(classifications)),
        "material": material,
        "raw_exact": raw_exact,
        "canonical_exact": canonical_exact,
        "encoded_render_equal": encoded_render_equal,
        "pixel_semantic_equal": pixel_semantic_equal,
        "raw_detection_count": {
            "historical": len(historical_detections),
            "live": len(live_detections),
        },
        "canonical_region_count": {
            "historical": len(historical_regions),
            "live": len(live_regions),
        },
        "ordered_semantic_types": {
            "historical": historical_types,
            "live": live_types,
        },
        "tolerance": {
            "bbox_render_px": bbox_tolerance_px,
            "raw_score": score_tolerance,
        },
    }


class FrozenReplayLeakageGuard:
    """Fail-closed accounting for forbidden frozen downstream inputs."""

    def __init__(self) -> None:
        self._events: dict[str, list[str]] = {flag: [] for flag in REPLAY_FLAGS}

    def record(self, flag: str, detail: str) -> None:
        if flag not in self._events:
            raise ValueError(f"UNKNOWN_REPLAY_LEAKAGE_FLAG:{flag}")
        self._events[flag].append(str(detail))

    def audit(self) -> dict[str, Any]:
        flags = {key: bool(value) for key, value in self._events.items()}
        count = sum(flags.values())
        return {
            "schema": REPLAY_AUDIT_SCHEMA,
            **flags,
            "events": copy.deepcopy(self._events),
            "leakage_count": count,
            "gate": "PASS" if count == 0 else "FAIL",
        }

    def assert_zero(self) -> None:
        audit = self.audit()
        if audit["gate"] != "PASS":
            raise RuntimeError(f"FROZEN_REPLAY_LEAKAGE:{audit['events']}")


class LiveFusionExecutor:
    """Build FusionIR only from current-run Layout and source evidence."""

    def __init__(self, *, normal_strategy: Mapping[str, Any], escalation_strategy: Mapping[str, Any]):
        self.normal_strategy = copy.deepcopy(dict(normal_strategy))
        self.escalation_strategy = copy.deepcopy(dict(escalation_strategy))

    def execute(
        self,
        *,
        page_irs: Mapping[tuple[str, int], dict[str, Any]],
        raw_pages: Mapping[tuple[str, int], dict[str, Any]],
        page_records: Mapping[tuple[str, int], dict[str, Any]],
        source_evidence: Mapping[tuple[str, int], dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        keys = set(page_irs)
        if not (keys == set(raw_pages) == set(page_records) == set(source_evidence)):
            raise ValueError("LIVE_FUSION_INPUT_PAGE_IDENTITIES_DIFFER")
        pages = []
        issue_count = 0
        for key in sorted(keys):
            control = build_page_fusion_ir(
                page_ir=page_irs[key],
                raw_page=raw_pages[key],
                page_record=page_records[key],
                source_evidence=source_evidence[key],
                strategy=self.escalation_strategy,
            )
            strategy = (
                self.normal_strategy
                if control["page_escalation"]["status"] == "NONE"
                else self.escalation_strategy
            )
            page = build_page_fusion_ir(
                page_ir=page_irs[key],
                raw_page=raw_pages[key],
                page_record=page_records[key],
                source_evidence=source_evidence[key],
                strategy=strategy,
            )
            repeated = build_page_fusion_ir(
                page_ir=page_irs[key],
                raw_page=raw_pages[key],
                page_record=page_records[key],
                source_evidence=source_evidence[key],
                strategy=strategy,
            )
            if page != repeated:
                raise RuntimeError(f"LIVE_FUSION_NONDETERMINISTIC:{key}")
            issue_count += len(validate_page_fusion_ir(page))
            pages.append(page)
        metrics = {
            "schema": "bemarkdown-live-fusion-metrics-v0",
            "page_count": len(pages),
            "candidate_count": sum(len(page["fusion_candidates"]) for page in pages),
            "validation_issue_count": issue_count,
            "semantic_sha256": semantic_sha256(pages),
            "deterministic": True,
            "frozen_fusion_artifact_used": False,
            "gate": "PASS" if pages and issue_count == 0 else "FAIL",
        }
        if metrics["gate"] != "PASS":
            raise RuntimeError(f"LIVE_FUSION_GATE_FAILED:{metrics}")
        return pages, metrics


class LiveRouterExecutor:
    """Route current-run FusionIR and enforce exactly-one primary ownership."""

    def execute(
        self,
        fusion_pages: Sequence[dict[str, Any]],
        *,
        page_records: Mapping[tuple[str, int], dict[str, Any]],
        source_evidence: Mapping[tuple[str, int], dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        plans = []
        for page in fusion_pages:
            key = (str(page["document_id"]), int(page["page_index"]))
            plan = plan_page_content(page, page_records[key], source_evidence[key])
            validate_content_route_plan(plan)
            plans.append(plan)
        repeated = [
            plan_page_content(
                page,
                page_records[(str(page["document_id"]), int(page["page_index"]))],
                source_evidence[(str(page["document_id"]), int(page["page_index"]))],
            )
            for page in fusion_pages
        ]
        if plans != repeated:
            raise RuntimeError("LIVE_ROUTER_NONDETERMINISTIC")
        routes = [route for plan in plans for route in plan["routes"]]
        duplicate_native = 0
        for plan in plans:
            claims = Counter(
                (int(plan["page_index"]), str(unit))
                for route in plan["routes"]
                if route["adapter"] == "NATIVE_TEXT_BRIDGE"
                for unit in route.get("source_unit_ids", [])
            )
            duplicate_native += sum(count - 1 for count in claims.values() if count > 1)
        metrics = {
            "schema": "bemarkdown-live-router-metrics-v0",
            "page_count": len(plans),
            "route_count": len(routes),
            "input_candidate_count": sum(
                int(plan["coverage"]["input_candidates_total"]) for plan in plans
            ),
            "unrouted": sum(int(plan["coverage"]["unrouted"]) for plan in plans),
            "duplicate_primary": sum(
                len(plan["coverage"]["duplicate_primary_candidate_ids"])
                for plan in plans
            ),
            "duplicate_native_primary": duplicate_native,
            "route_population": dict(sorted(Counter(route["adapter"] for route in routes).items())),
            "semantic_sha256": semantic_sha256(plans),
            "deterministic": True,
            "frozen_router_artifact_used": False,
        }
        metrics["gate"] = (
            "PASS"
            if plans
            and metrics["unrouted"] == 0
            and metrics["duplicate_primary"] == 0
            and duplicate_native == 0
            else "FAIL"
        )
        if metrics["gate"] != "PASS":
            raise RuntimeError(f"LIVE_ROUTER_GATE_FAILED:{metrics}")
        return plans, metrics


class DocumentIRAssetIdentityValidator:
    """Reject presentation-local identities from authoritative DocumentIR fields."""

    forbidden_tokens = ("draft-asset-", "profile-local", "\\tmp\\", "/tmp/")

    def validate(self, document: Mapping[str, Any]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        assets = list(document.get("assets", []))
        for index, asset in enumerate(assets):
            uid = str(asset.get("asset_uid") or "")
            if re.fullmatch(r"asset-sha256-[0-9a-f]{64}", uid) is None:
                violations.append(
                    {"location": f"assets[{index}].asset_uid", "reason": "UNSTABLE_ASSET_UID", "value": uid}
                )
            for field in ("relative_path", "source_ref", "source_pdf"):
                if field in asset:
                    violations.append(
                        {"location": f"assets[{index}].{field}", "reason": "PRESENTATION_OR_LOCAL_PATH_IN_AUTHORITY", "value": asset.get(field)}
                    )
        for collection in ("blocks", "suppressed_blocks"):
            for index, block in enumerate(document.get(collection, [])):
                content = block.get("content", {})
                if content.get("asset_ref") is not None:
                    violations.append(
                        {"location": f"{collection}[{index}].content.asset_ref", "reason": "PRESENTATION_PATH_IN_SEMANTIC_CONTENT", "value": content.get("asset_ref")}
                    )
                uid = str(content.get("asset_uid") or "")
                if uid and re.fullmatch(r"asset-sha256-[0-9a-f]{64}", uid) is None:
                    violations.append(
                        {"location": f"{collection}[{index}].content.asset_uid", "reason": "UNSTABLE_ASSET_UID", "value": uid}
                    )
        return {
            "schema": ASSET_IDENTITY_AUDIT_SCHEMA,
            "document_id": document.get("document_id"),
            "asset_count": len(assets),
            "violation_count": len(violations),
            "violations": violations,
            "gate": "PASS" if not violations else "FAIL",
        }
