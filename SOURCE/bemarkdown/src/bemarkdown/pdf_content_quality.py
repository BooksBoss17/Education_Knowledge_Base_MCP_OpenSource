from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

CONTENT_QUALITY_POLICY_VERSION = "core-content-quality-policy-v0"
CONTENT_CONFLICT_GRAPH_SCHEMA = "bemarkdown-content-conflict-graph-v0"
FORMULA_RISK_CANDIDATE_SCHEMA = "bemarkdown-formula-risk-candidate-v0"
FORMULA_DISCOVERY_VERSION = "formula-discovery-v0"
REVIEW_TAXONOMY_VERSION = "content-review-taxonomy-v0"

OCR_ADAPTERS = {"OCR_TEXT_REGION", "PAGE_VISUAL_TEXT_RECOVERY"}
IMAGE_ADAPTERS = {"IMAGE_NATIVE_EXTRACT", "IMAGE_RENDER_CROP"}


def normalize_ocr_text(value: str | None) -> str:
    """Apply evaluation-only Unicode/line-ending normalization without correction."""

    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def evaluate_ocr_pair(reference: str | None, prediction: str | None) -> dict[str, Any]:
    normalized_reference = normalize_ocr_text(reference)
    normalized_prediction = normalize_ocr_text(prediction)
    distance = _levenshtein(normalized_reference, normalized_prediction)
    denominator = max(1, len(normalized_reference))
    cer = distance / denominator
    return {
        "normalization": "unicode-nfkc-line-endings-v0",
        "reference": normalized_reference,
        "prediction": normalized_prediction,
        "reference_characters": len(normalized_reference),
        "prediction_characters": len(normalized_prediction),
        "edit_distance": distance,
        "cer": cer,
        "character_accuracy": max(0.0, (denominator - distance) / denominator),
        "normalized_exact_match": normalized_reference == normalized_prediction,
    }


def classify_ocr_output(text: str | None, *, confidence: float | None) -> dict[str, Any]:
    """Classify OCR without treating confidence as truth or suppressing text."""

    if not normalize_ocr_text(text):
        return {
            "policy_version": CONTENT_QUALITY_POLICY_VERSION,
            "quality_status": "OCR_CONTENT_REVIEW",
            "retain_text": False,
            "review_reasons": ["OCR_EMPTY"],
            "confidence_role": "DIAGNOSTIC_ONLY",
        }
    if confidence is not None and confidence < 0.3:
        return {
            "policy_version": CONTENT_QUALITY_POLICY_VERSION,
            "quality_status": "OCR_CONTENT_WARNING",
            "retain_text": True,
            "review_reasons": ["OCR_LOW_CONFIDENCE_DIAGNOSTIC_ONLY"],
            "confidence_role": "DIAGNOSTIC_ONLY",
        }
    return {
        "policy_version": CONTENT_QUALITY_POLICY_VERSION,
        "quality_status": "OCR_CONTENT_OK",
        "retain_text": True,
        "review_reasons": [],
        "confidence_role": "DIAGNOSTIC_ONLY",
    }


def build_formula_risk_candidates(
    raw_page: dict[str, Any],
    fusion_page: dict[str, Any],
    *,
    strategy: str = "F2",
) -> list[dict[str, Any]]:
    """Build secondary formula-risk evidence without changing primary candidates."""

    if strategy not in {"F1", "F2"}:
        raise ValueError(f"Unsupported formula discovery strategy: {strategy}")
    document_id = str(fusion_page["document_id"])
    page_index = int(fusion_page["page_index"])
    transform = fusion_page.get("render_transform", {})
    scale_x = float(transform.get("scale_x") or 0.0)
    scale_y = float(transform.get("scale_y") or 0.0)
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Formula discovery requires a valid render transform")
    page_geometry = fusion_page.get("page_geometry", {})
    page_area = float(page_geometry.get("width_pt") or 0.0) * float(
        page_geometry.get("height_pt") or 0.0
    )
    existing = list(fusion_page.get("fusion_candidates", []))
    canonical_formula = [row for row in existing if _semantic(row) == "FORMULA"]
    output = []
    for detection in sorted(
        raw_page.get("raw_detections", []),
        key=lambda row: str(row.get("raw_detection_id", "")),
    ):
        if str(detection.get("raw_label", "")).lower() != "formula":
            continue
        score = float(detection.get("raw_score") or 0.0)
        if not 0.3 <= score < 0.5:
            continue
        bbox_px = detection.get("raw_bbox_render_px")
        if not _valid_bbox(bbox_px):
            continue
        bbox = [
            round(float(bbox_px[0]) / scale_x, 6),
            round(float(bbox_px[1]) / scale_y, 6),
            round(float(bbox_px[2]) / scale_x, 6),
            round(float(bbox_px[3]) / scale_y, 6),
        ]
        if any(_overlap_strength(bbox, row.get("bbox_pdf_pt")) >= 0.5 for row in canonical_formula):
            continue
        overlaps = sorted(
            str(row["candidate_id"])
            for row in existing
            if _overlap_strength(bbox, row.get("bbox_pdf_pt")) >= 0.1
        )
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area_ratio = (width * height / page_area) if page_area > 0 else 1.0
        aspect_ratio = width / height
        evidence_types = ["LOW_SCORE_RAW_FORMULA"]
        reasons = ["RAW_FORMULA_SCORE_BETWEEN_CAPTURE_AND_CANONICAL_THRESHOLD"]
        if overlaps:
            evidence_types.append("EXISTING_ROUTE_GEOMETRY_OVERLAP")
            reasons.append("OVERLAPS_EXISTING_GENERIC_CONTENT_ROUTE")
        if strategy == "F2":
            geometry_ok = 0.05 <= aspect_ratio <= 40.0 and area_ratio <= 0.15
            if score < 0.35 or not geometry_ok or not overlaps:
                continue
            evidence_types.append("FORMULA_SHAPED_GEOMETRY")
            reasons.append("F2_CONTEXT_AND_GEOMETRY_GATE_PASSED")
        identity = {
            "schema": FORMULA_RISK_CANDIDATE_SCHEMA,
            "document_id": document_id,
            "page_index": page_index,
            "raw_detection_id": str(detection["raw_detection_id"]),
            "strategy": strategy,
            "bbox_pdf_pt": bbox,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20]
        output.append(
            {
                "schema": FORMULA_RISK_CANDIDATE_SCHEMA,
                "risk_id": f"formula-risk-{digest}",
                "document_id": document_id,
                "page_index": page_index,
                "bbox_pdf_pt": bbox,
                "evidence_types": sorted(evidence_types),
                "raw_layout_detection_ids": [str(detection["raw_detection_id"])],
                "score": score,
                "overlap_with_existing_routes": overlaps,
                "risk_reason": sorted(reasons),
                "strategy": strategy,
                "discovery_version": FORMULA_DISCOVERY_VERSION,
                "primary_route": False,
                "formula_net_required": True,
                "provenance": {
                    "raw_bbox_render_px": [float(value) for value in bbox_px],
                    "render_scale_x": scale_x,
                    "render_scale_y": scale_y,
                    "raw_layout_evidence_mutated": False,
                },
            }
        )
    return sorted(output, key=lambda row: row["risk_id"])


def formula_risk_route(
    risk: dict[str, Any], *, source_path: str, page_escalation: str = "NONE"
) -> dict[str, Any]:
    """Create a secondary FormulaNet route that cannot consume a primary candidate."""

    identity = {
        "risk_id": risk["risk_id"],
        "document_id": risk["document_id"],
        "page_index": int(risk["page_index"]),
        "discovery_version": risk["discovery_version"],
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20]
    return {
        "route_id": f"formula-risk-route-{digest}",
        "route_role": "SECONDARY_FORMULA_RISK",
        "primary_candidate_assignment": False,
        "document_id": str(risk["document_id"]),
        "page_index": int(risk["page_index"]),
        "input_kind": "FORMULA_RISK_REGION",
        "input_candidate_ids": [],
        "source_region_ids": [],
        "source_unit_ids": [],
        "semantic_evidence": ["FORMULA_RISK"],
        "page_escalation": page_escalation,
        "adapter": "FORMULA_RECOGNITION",
        "output_kind": "FORMULA",
        "decision_reason_codes": ["SECONDARY_FORMULA_DISCOVERY_RISK"],
        "decision_version": CONTENT_QUALITY_POLICY_VERSION,
        "requires_gpu": True,
        "provenance": {
            "candidate_kinds": ["FORMULA_RISK_CANDIDATE"],
            "bbox_pdf_pt": list(risk["bbox_pdf_pt"]),
            "evidence_ids": list(risk.get("raw_layout_detection_ids", [])),
            "native_text_evidence_ids": [],
            "native_image_placements": [],
            "source_path": source_path,
            "native_text_trust": "NOT_APPLICABLE",
            "source_profile": "FORMULA_DISCOVERY_SECONDARY",
            "formula_discovery": copy.deepcopy(risk),
        },
    }


def build_content_conflict_graph(region_content: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build and resolve page-local deterministic conflicts without deleting evidence."""

    rows = sorted((copy.deepcopy(row) for row in region_content), key=lambda row: row["content_id"])
    nodes = [_content_node(row) for row in rows]
    for row in rows:
        if _adapter(row) not in OCR_ADAPTERS:
            continue
        for sequence, line in enumerate(row.get("provenance", {}).get("ocr_lines", [])):
            nodes.append(
                {
                    "node_id": f"{row['content_id']}:ocr-line:{int(line.get('reading_sequence', sequence)):04d}",
                    "node_kind": "OCR_LINE",
                    "parent_content_id": row["content_id"],
                    "bbox_pdf_pt": line.get("bbox_page_pdf_pt"),
                    "status": "EVIDENCE",
                    "recoverable": True,
                }
            )
    edges = []
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            evidence = _edge_evidence(first, second)
            if not evidence:
                continue
            conflict_type = _conflict_type(first, second)
            if conflict_type is None:
                continue
            resolution = _resolve_conflict(first, second, conflict_type, evidence)
            identity = {
                "document_id": first["document_id"],
                "page_index": int(first["page_index"]),
                "nodes": sorted([first["content_id"], second["content_id"]]),
                "conflict_type": conflict_type,
                "evidence": evidence,
            }
            digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20]
            edges.append(
                {
                    "conflict_id": f"content-conflict-{digest}",
                    "node_ids": identity["nodes"],
                    "conflict_type": conflict_type,
                    "evidence": evidence,
                    **resolution,
                    "evidence_deleted": False,
                }
            )
    edges.sort(key=lambda edge: edge["conflict_id"])
    graph_identity = {"nodes": nodes, "edges": edges, "policy": CONTENT_QUALITY_POLICY_VERSION}
    return {
        "schema": CONTENT_CONFLICT_GRAPH_SCHEMA,
        "policy_version": CONTENT_QUALITY_POLICY_VERSION,
        "nodes": nodes,
        "edges": edges,
        "semantic_sha256": hashlib.sha256(
            _canonical_json(graph_identity).encode("utf-8")
        ).hexdigest(),
        "invariants": {
            "node_count_before_resolution": len(nodes),
            "node_count_after_resolution": len(nodes),
            "resolved_conflict_deleted_nodes": 0,
            "all_nodes_recoverable": all(node["recoverable"] for node in nodes),
        },
    }


def apply_conflict_metadata(
    region_content: Iterable[dict[str, Any]], graph: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach conflict decisions while retaining every RegionContentIR node."""

    rows = [copy.deepcopy(row) for row in region_content]
    by_id = {row["content_id"]: row for row in rows}
    for row in rows:
        row.setdefault("conflict_ids", [])
        row.setdefault("conflict_resolution", "NO_CONFLICT")
        row.setdefault("secondary_content_refs", [])
        row.setdefault("duplicate_risk", False)
    rank = {
        "NO_CONFLICT": 0,
        "KEEP_BOTH": 1,
        "PRIMARY_SPECIALIZED": 2,
        "SECONDARY_EVIDENCE_ONLY": 3,
        "SUPPRESS_GENERIC_DUPLICATE": 4,
        "REVIEW_REQUIRED": 5,
    }
    for edge in graph.get("edges", []):
        for node_id in edge["node_ids"]:
            row = by_id.get(node_id)
            if row is None:
                continue
            row["conflict_ids"].append(edge["conflict_id"])
            target = edge["resolution"]
            if node_id in edge.get("secondary_node_ids", []):
                target = "SECONDARY_EVIDENCE_ONLY"
            if rank[target] > rank[row["conflict_resolution"]]:
                row["conflict_resolution"] = target
            row["duplicate_risk"] = row["duplicate_risk"] or edge["resolution"] in {
                "KEEP_BOTH",
                "REVIEW_REQUIRED",
            }
        primary = edge.get("primary_node_id")
        if primary in by_id:
            by_id[primary]["secondary_content_refs"].extend(edge.get("secondary_node_ids", []))
    for row in rows:
        row["conflict_ids"] = sorted(set(row["conflict_ids"]))
        row["secondary_content_refs"] = sorted(set(row["secondary_content_refs"]))
    if len(rows) != len(graph.get("nodes", [])) - sum(
        node.get("node_kind") == "OCR_LINE" for node in graph.get("nodes", [])
    ):
        raise ValueError("Conflict metadata application cannot add or delete ContentIR nodes")
    return rows


def build_review_taxonomy(plans: Iterable[dict[str, Any]]) -> dict[str, Any]:
    reason_routes: dict[str, set[str]] = defaultdict(set)
    reason_pages: dict[str, set[tuple[str, int]]] = defaultdict(set)
    reason_documents: dict[str, set[str]] = defaultdict(set)
    reason_candidates: Counter[str] = Counter()
    all_routes: set[str] = set()
    all_pages: set[tuple[str, int]] = set()
    total_candidates = 0
    for plan in plans:
        document_id = str(plan["document_id"])
        page_index = int(plan["page_index"])
        for route in plan.get("routes", []):
            if route.get("adapter") != "CONTENT_REVIEW_REQUIRED":
                continue
            route_id = str(route["route_id"])
            candidate_count = max(1, len(route.get("input_candidate_ids", [])))
            all_routes.add(route_id)
            all_pages.add((document_id, page_index))
            total_candidates += candidate_count
            taxonomy_reasons = set(
                classify_review_reasons(
                    route.get("decision_reason_codes", []),
                    semantic_evidence=route.get("semantic_evidence", []),
                )
            )
            for reason in taxonomy_reasons:
                reason_routes[reason].add(route_id)
                reason_pages[reason].add((document_id, page_index))
                reason_documents[reason].add(document_id)
                reason_candidates[reason] += candidate_count
    reasons = [
        {
            "taxonomy_reason": reason,
            "route_count": len(reason_routes[reason]),
            "candidate_count": reason_candidates[reason],
            "page_count": len(reason_pages[reason]),
            "document_count": len(reason_documents[reason]),
        }
        for reason in sorted(reason_routes)
    ]
    pareto = sorted(reasons, key=lambda row: (-row["route_count"], row["taxonomy_reason"]))
    running = 0
    for row in pareto:
        running += row["route_count"]
        row["cumulative_route_fraction"] = running / max(1, len(all_routes))
    return {
        "schema": "bemarkdown-review-route-taxonomy-v0",
        "taxonomy_version": REVIEW_TAXONOMY_VERSION,
        "totals": {
            "route_count": len(all_routes),
            "candidate_count": total_candidates,
            "page_count": len(all_pages),
        },
        "reasons": reasons,
        "pareto": pareto,
        "top_three_route_fraction": sum(row["route_count"] for row in pareto[:3])
        / max(1, len(all_routes)),
    }


def classify_review_reasons(
    reason_codes: Iterable[str], *, semantic_evidence: Iterable[str] = ()
) -> list[str]:
    route = {"semantic_evidence": list(semantic_evidence)}
    reasons = {_review_taxonomy_reason(str(code), route) for code in reason_codes}
    return sorted(reasons or {"OTHER"})


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _content_node(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": row["content_id"],
        "node_kind": row.get("content_kind", "UNKNOWN"),
        "adapter": _adapter(row),
        "bbox_pdf_pt": row.get("bbox_pdf_pt"),
        "status": row.get("status"),
        "source_candidate_ids": sorted(row.get("source_candidate_ids", [])),
        "source_unit_ids": sorted(row.get("source_unit_ids", [])),
        "recoverable": True,
    }


def _edge_evidence(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any] | None:
    bbox_strength = _overlap_strength(first.get("bbox_pdf_pt"), second.get("bbox_pdf_pt"))
    shared_candidates = sorted(
        set(first.get("source_candidate_ids", [])) & set(second.get("source_candidate_ids", []))
    )
    shared_units = sorted(
        set(first.get("source_unit_ids", [])) & set(second.get("source_unit_ids", []))
    )
    if bbox_strength <= 0 and not shared_candidates and not shared_units:
        return None
    return {
        "bbox_overlap_strength": round(bbox_strength, 8),
        "shared_source_candidate_ids": shared_candidates,
        "shared_source_unit_ids": shared_units,
        "same_source_candidate": bool(shared_candidates),
        "same_source_unit": bool(shared_units),
    }


def _conflict_type(first: dict[str, Any], second: dict[str, Any]) -> str | None:
    left, right = _adapter(first), _adapter(second)
    adapters = {left, right}
    if "NATIVE_TEXT_BRIDGE" in adapters and adapters & OCR_ADAPTERS:
        return "NATIVE_TEXT_VS_OCR"
    if "FORMULA_RECOGNITION" in adapters and adapters & OCR_ADAPTERS:
        return "OCR_VS_FORMULA"
    if "FORMULA_RECOGNITION" in adapters and "NATIVE_TEXT_BRIDGE" in adapters:
        return "NATIVE_TEXT_VS_FORMULA"
    if adapters & OCR_ADAPTERS and adapters & IMAGE_ADAPTERS:
        return "OCR_VS_IMAGE"
    if "TABLE_DEFERRED_PRESERVE" in adapters and adapters & OCR_ADAPTERS:
        return "OCR_VS_TABLE"
    if "FORMULA_RECOGNITION" in adapters and adapters & IMAGE_ADAPTERS:
        return "FORMULA_VS_IMAGE"
    if adapters & IMAGE_ADAPTERS and ({left, right} & {"NATIVE_TEXT_BRIDGE"}):
        return "TEXT_VS_IMAGE"
    if left in OCR_ADAPTERS | {"NATIVE_TEXT_BRIDGE"} and right in OCR_ADAPTERS | {
        "NATIVE_TEXT_BRIDGE"
    }:
        return "OVERLAPPING_TEXT_ROUTES"
    return None


def _resolve_conflict(
    first: dict[str, Any],
    second: dict[str, Any],
    conflict_type: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if conflict_type == "OCR_VS_FORMULA":
        formula = first if _adapter(first) == "FORMULA_RECOGNITION" else second
        ocr = second if formula is first else first
        consensus_status = formula.get("quality_metrics", {}).get(
            "formula_consensus_status"
        )
        if (
            formula.get("status") == "SUCCESS"
            and consensus_status == "CONSENSUS_PASS"
            and evidence["bbox_overlap_strength"] >= 0.5
        ):
            return _resolution("PRIMARY_SPECIALIZED", formula, [ocr])
        return _resolution("REVIEW_REQUIRED")
    if conflict_type == "NATIVE_TEXT_VS_OCR":
        native = first if _adapter(first) == "NATIVE_TEXT_BRIDGE" else second
        ocr = second if native is first else first
        trust = native.get("provenance", {}).get("native_text_trust")
        same_source = evidence["same_source_candidate"] or evidence["same_source_unit"]
        if trust == "HIGH" and same_source:
            return _resolution("PRIMARY_SPECIALIZED", native, [ocr])
        return _resolution("KEEP_BOTH")
    if conflict_type in {"OCR_VS_IMAGE", "OCR_VS_TABLE", "TEXT_VS_IMAGE", "FORMULA_VS_IMAGE"}:
        return _resolution("KEEP_BOTH")
    if conflict_type in {"NATIVE_TEXT_VS_FORMULA", "OVERLAPPING_TEXT_ROUTES"}:
        return _resolution("REVIEW_REQUIRED")
    return _resolution("NO_CONFLICT")


def _resolution(
    resolution: str,
    primary: dict[str, Any] | None = None,
    secondary: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "resolution": resolution,
        "primary_node_id": primary["content_id"] if primary else None,
        "secondary_node_ids": sorted(row["content_id"] for row in (secondary or [])),
        "suppression_scope": (
            "FUTURE_PRIMARY_TEXTUAL_RECONSTRUCTION_ONLY"
            if secondary
            else None
        ),
    }


def _review_taxonomy_reason(code: str, route: dict[str, Any]) -> str:
    if code.startswith("PAGE_ESCALATION"):
        return "PAGE_ESCALATION_REVIEW"
    if "INVALID" in code or "DEGENERATE" in code or "ZERO_AREA" in code:
        return "INVALID_OR_DEGENERATE_GEOMETRY"
    if "SEMANTIC" in code or "CONFLICT" in code:
        return "SEMANTIC_CONFLICT"
    if "NATIVE" in code and "MAPPING" in code:
        return "NATIVE_MAPPING_CONFLICT"
    if "IMAGE" in code and ("MAPPING" in code or "AMBIGU" in code):
        return "IMAGE_MAPPING_AMBIGUITY"
    if "FORMULA" in code and ("SAFETY" in code or "VALIDATOR" in code):
        return "FORMULA_SAFETY_REVIEW"
    if "QUALITY" in code or code in {"OCR_EMPTY", "NATIVE_TEXT_EMPTY"}:
        return "ADAPTER_QUALITY_REVIEW"
    if code == "PRIMARY_ADAPTER_UNRESOLVED" or "UNRESOLVED" in code:
        return "UNRESOLVED_ROUTE_EVIDENCE"
    if "VISUAL_UNKNOWN" in route.get("semantic_evidence", []):
        return "UNRESOLVED_ROUTE_EVIDENCE"
    return "OTHER"


def _semantic(row: dict[str, Any]) -> str:
    return str(row.get("semantic_type") or row.get("semantic_hint") or "UNKNOWN").upper()


def _adapter(row: dict[str, Any]) -> str:
    return str(row.get("provenance", {}).get("route_decision", {}).get("adapter", ""))


def _overlap_strength(first: Any, second: Any) -> float:
    if not _valid_bbox(first) or not _valid_bbox(second):
        return 0.0
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = (float(first[2]) - float(first[0])) * (float(first[3]) - float(first[1]))
    second_area = (float(second[2]) - float(second[0])) * (float(second[3]) - float(second[1]))
    return max(intersection / first_area, intersection / second_area)


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _levenshtein(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for first_index, first_char in enumerate(first, 1):
        current = [first_index]
        for second_index, second_char in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_char != second_char),
                )
            )
        previous = current
    return previous[-1]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
