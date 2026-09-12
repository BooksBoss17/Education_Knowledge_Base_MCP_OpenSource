"""Authoritative Formula/Table/Image ownership for text exclusion."""

from __future__ import annotations

import copy
import hashlib
import io
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

OWNERSHIP_REGION_SCHEMA = "bemarkdown-non-text-ownership-region-v1"
OWNERSHIP_CONTRACT_VERSION = "non-text-first-ownership-mask-v1"
NATIVE_FORMULA_AXIS_MIN_SPAN_COVERAGE = 0.6
NATIVE_FORMULA_AXIS_MIN_HORIZONTAL_COVERAGE = 0.85
NATIVE_FORMULA_CLUSTER_MIN_SPAN_COVERAGE = NATIVE_FORMULA_AXIS_MIN_SPAN_COVERAGE
NATIVE_FORMULA_CLUSTER_MIN_HORIZONTAL_COVERAGE = (
    NATIVE_FORMULA_AXIS_MIN_HORIZONTAL_COVERAGE
)
NATIVE_FORMULA_CLUSTER_OWNER_COVERAGE_THRESHOLD = 0.75
NATIVE_FORMULA_CLUSTER_MIN_SPANS = 2


@dataclass(frozen=True, slots=True)
class NonTextOwnershipRegion:
    document_id: str
    page_index: int
    kind: str
    bbox_pdf_pt: tuple[float, float, float, float]
    owner_route_id: str
    owner_content_id: str
    authority_basis: str
    materialization_status: str
    asset_uid: str | None
    formula_payload_present: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = OWNERSHIP_REGION_SCHEMA
        value["bbox_pdf_pt"] = list(self.bbox_pdf_pt)
        return {"schema": value.pop("schema"), **value}


@dataclass(frozen=True, slots=True)
class NonTextOwnershipMap:
    regions: tuple[NonTextOwnershipRegion, ...]
    unsafe: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class NativeTextExclusionResult:
    rows: tuple[dict[str, Any], ...]
    excluded_spans: tuple[dict[str, Any], ...]
    partial_overlaps: tuple[dict[str, Any], ...]


def build_non_text_ownership(
    rows: Iterable[dict[str, Any]],
) -> NonTextOwnershipMap:
    regions = []
    unsafe = []
    for row in rows:
        kind = str(row.get("content_kind") or "").upper()
        if kind not in {"FORMULA", "TABLE", "IMAGE"}:
            continue
        # An embedded raster is a storage object, not a semantic non-text region.
        # Textbook backgrounds and scanned pages can contain entire paragraphs.
        # Preserve these images, but only localized figure evidence may mask text.
        candidate_kinds = row.get("provenance", {}).get("source_candidate_kinds", [])
        if kind == "IMAGE" and candidate_kinds == ["NATIVE_IMAGE_FALLBACK"]:
            continue
        adapter = str(
            row.get("provenance", {}).get("route_decision", {}).get("adapter") or ""
        )
        if kind == "IMAGE" and adapter == "IMAGE_NATIVE_EXTRACT":
            # Native extraction saves the image object alone. Text painted over
            # it remains separate PDF content and is absent from that asset.
            # Such an asset cannot authorize erasing text from its rectangle.
            continue
        formula_payload_present = bool(str(row.get("latex") or "").strip())
        preserved_asset = _materialized_asset(row)
        authority_basis = None
        if (
            kind == "FORMULA"
            and adapter == "FORMULA_RECOGNITION"
            and formula_payload_present
        ):
            authority_basis = "FORMULA_LATEX_PAYLOAD_SAVED"
        elif kind == "FORMULA" and adapter == "FORMULA_RECOGNITION" and preserved_asset:
            authority_basis = "FORMULA_REVIEW_CROP_PRESERVED"
        elif (
            kind == "TABLE"
            and adapter in {"TABLE_ENGINE", "TABLE_DEFERRED_PRESERVE"}
            and preserved_asset
        ):
            authority_basis = "TABLE_ROUTE_CROP_MATERIALIZED"
        elif (
            kind == "IMAGE"
            and adapter in {"IMAGE_NATIVE_EXTRACT", "IMAGE_RENDER_CROP"}
            and str(row.get("status")) == "SUCCESS"
            and preserved_asset
        ):
            authority_basis = f"{adapter}_MATERIALIZED"
        if authority_basis is None:
            unsafe.append(_unsafe_row(row, kind))
            continue
        bbox = tuple(float(value) for value in row["bbox_pdf_pt"])
        if kind == "IMAGE" and adapter == "IMAGE_RENDER_CROP":
            crop = row.get("provenance", {}).get("render_crop", {})
            rendered = crop.get("bbox_pdf_pt")
            # PdfCropper preserves padding pixels too. Use its recorded source
            # rectangle only when it belongs to the materialized image asset.
            if (isinstance(rendered, (list, tuple)) and len(rendered) == 4
                    and all(math.isfinite(float(v)) for v in rendered)
                    and rendered[0] <= bbox[0] and rendered[1] <= bbox[1]
                    and rendered[2] >= bbox[2] and rendered[3] >= bbox[3]
                    and crop.get("path") and row.get("binary_artifact_ref")
                    and Path(crop["path"]).resolve() == Path(row["binary_artifact_ref"]).resolve()):
                bbox = tuple(float(value) for value in rendered)
        regions.append(
            NonTextOwnershipRegion(
                document_id=str(row["document_id"]),
                page_index=int(row["page_index"]),
                kind=kind,
                bbox_pdf_pt=bbox,  # type: ignore[arg-type]
                owner_route_id=str(row["route_id"]),
                owner_content_id=str(row["content_id"]),
                authority_basis=authority_basis,
                materialization_status="MATERIALIZED",
                asset_uid=None,
                formula_payload_present=formula_payload_present,
            )
        )
    regions.sort(
        key=lambda value: (
            value.document_id,
            value.page_index,
            value.kind,
            value.bbox_pdf_pt,
            value.owner_route_id,
            value.owner_content_id,
        )
    )
    unsafe.sort(
        key=lambda value: (
            value["document_id"],
            value["page_index"],
            value["kind"],
            value["bbox_pdf_pt"],
            value["owner_route_id"],
        )
    )
    return NonTextOwnershipMap(regions=tuple(regions), unsafe=tuple(unsafe))


def _materialized_asset(row: dict[str, Any]) -> str | None:
    provenance = row.get("provenance") or {}
    candidates = [
        row.get("binary_artifact_ref"),
        *(
            provenance.get(key, {}).get("path")
            for key in (
                "crop",
                "preserved_input",
                "preserved_render_crop",
                "render_crop",
                "native_extraction",
            )
        ),
    ]
    for value in candidates:
        if value and (path := Path(str(value))).is_file() and path.stat().st_size > 0:
            return str(path.resolve())
    return None


def _unsafe_row(row: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "document_id": str(row["document_id"]),
        "page_index": int(row["page_index"]),
        "kind": kind,
        "owner_route_id": str(row["route_id"]),
        "owner_content_id": str(row["content_id"]),
        "bbox_pdf_pt": [float(value) for value in row["bbox_pdf_pt"]],
        "reason": "NON_TEXT_OWNERSHIP_UNSAFE",
    }


def derive_text_only_render(
    source_crop: dict[str, Any],
    regions: Iterable[NonTextOwnershipRegion],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Copy one original render and apply the exact page-local exclusion union."""

    source_path = Path(str(source_crop["path"])).resolve(strict=True)
    original_payload = source_path.read_bytes()
    original_sha = hashlib.sha256(original_payload).hexdigest()
    expected_sha = str(source_crop.get("content_sha256") or original_sha)
    if original_sha != expected_sha:
        raise RuntimeError("ORIGINAL_RENDER_SHA256_MISMATCH")
    crop_bbox = tuple(float(value) for value in source_crop["bbox_pdf_pt"])
    with Image.open(io.BytesIO(original_payload)) as opened:
        image = opened.convert("RGB")
    scale_x = float(
        source_crop.get("scale_x") or image.width / (crop_bbox[2] - crop_bbox[0])
    )
    scale_y = float(
        source_crop.get("scale_y") or image.height / (crop_bbox[3] - crop_bbox[1])
    )
    pixel_boxes = []
    mask = Image.new("1", image.size, 0)
    drawer = ImageDraw.Draw(mask)
    for region in regions:
        intersection = _intersection(crop_bbox, region.bbox_pdf_pt)
        if intersection is None:
            continue
        left = max(0, math.floor((intersection[0] - crop_bbox[0]) * scale_x))
        top = max(0, math.floor((intersection[1] - crop_bbox[1]) * scale_y))
        right = min(image.width, math.ceil((intersection[2] - crop_bbox[0]) * scale_x))
        bottom = min(
            image.height, math.ceil((intersection[3] - crop_bbox[1]) * scale_y)
        )
        if right <= left or bottom <= top:
            continue
        pixel_boxes.append([left, top, right, bottom])
        drawer.rectangle((left, top, right - 1, bottom - 1), fill=1)
    image.paste((255, 255, 255), mask=mask)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    payload = stream.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"sha256-{sha}.png"
    if not path.exists():
        path.write_bytes(payload)
    masked_area = mask.histogram()[1]
    return {
        **source_crop,
        "contract": "TEXT_EXCLUSION_MASK",
        "ownership_contract_version": OWNERSHIP_CONTRACT_VERSION,
        "path": str(path),
        "content_sha256": sha,
        "bytes": len(payload),
        "original_render_path": str(source_path),
        "original_render_sha256": original_sha,
        "text_only_render_sha256": sha,
        "ownership_region_count": len(pixel_boxes),
        "ownership_mask_px_bboxes": pixel_boxes,
        "mask_fill_rgb": [255, 255, 255],
        "mask_padding_px": 0,
        "masked_pixel_area": masked_area,
        "masked_page_area_ratio": round(masked_area / (image.width * image.height), 12),
        # Exact pixels, not OCR silence: even one faint unowned pixel prevents
        # this shortcut. White margins around a fully masked formula are valid.
        "post_mask_all_white": bool(pixel_boxes) and all(low == 255 for low, high in image.getextrema()),
        "transient_runtime_artifact": True,
    }


def _intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    result = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return result if result[2] > result[0] and result[3] > result[1] else None


def exclude_native_text(
    rows: Iterable[dict[str, Any]],
    regions: Iterable[NonTextOwnershipRegion],
    *,
    coverage_threshold: float = 0.8,
) -> NativeTextExclusionResult:
    """Exclude source spans fully owned by materialized non-text content."""

    if not 0.0 < coverage_threshold <= 1.0:
        raise ValueError("NATIVE_TEXT_EXCLUSION_THRESHOLD_INVALID")
    ordered_regions = tuple(
        sorted(
            regions,
            key=lambda value: (
                value.kind,
                value.bbox_pdf_pt,
                value.owner_route_id,
                value.owner_content_id,
            ),
        )
    )
    source_rows = tuple(copy.deepcopy(row) for row in rows)
    if not ordered_regions:
        return NativeTextExclusionResult(
            rows=source_rows,
            excluded_spans=(),
            partial_overlaps=(),
        )
    formula_cluster_owners = _formula_span_cluster_owners(
        source_rows,
        ordered_regions,
    )
    output = []
    excluded = []
    partial = []
    for row in source_rows:
        source_spans = list(row.get("spans") or [])
        has_span_geometry = bool(source_spans)
        spans = source_spans or [
            {
                "span_id": str(row.get("evidence_id") or "native-span"),
                "text": str(row.get("text") or ""),
                "bbox_pdf_pt": row.get("bbox_pdf_pt"),
            }
        ]
        retained = []
        for span in spans:
            span_id = str(span.get("span_id") or row.get("evidence_id") or "")
            bbox_value = span.get("bbox_pdf_pt")
            if not (
                isinstance(bbox_value, (list, tuple))
                and len(bbox_value) == 4
                and float(bbox_value[2]) > float(bbox_value[0])
                and float(bbox_value[3]) > float(bbox_value[1])
            ):
                retained.append(span)
                continue
            bbox = tuple(float(value) for value in bbox_value)
            ink = span.get("ink_bbox_pdf_pt")
            if (span.get("ink_bbox_basis") == "EMBEDDED_SOURCE_GLYPH_OUTLINES"
                    and isinstance(ink, (list, tuple)) and len(ink) == 4
                    and all(math.isfinite(float(v)) for v in ink)
                    and ink[0] < ink[2] and ink[1] < ink[3]):
                image_owner = next((region for region in ordered_regions
                    if region.kind == "IMAGE"
                    and _intersection(bbox, region.bbox_pdf_pt) is not None
                    and region.bbox_pdf_pt[0] <= ink[0]
                    and region.bbox_pdf_pt[1] <= ink[1]
                    and region.bbox_pdf_pt[2] >= ink[2]
                    and region.bbox_pdf_pt[3] >= ink[3]), None)
                if image_owner is not None:
                    trace = _native_overlap_trace(
                        row=row, span_id=span_id, span_bbox=bbox,
                        region=image_owner, coverage=1.0,
                        reason="EXCLUDED_BY_NATIVE_IMAGE_GLYPH_GEOMETRY",
                        excluded=True, exclusion_basis="SOURCE_GLYPH_OUTLINES_INSIDE_IMAGE",
                    )
                    trace["ink_bbox_pdf_pt"] = list(ink)
                    excluded.append(trace)
                    continue
            direct_owner = next(
                (
                    region
                    for region in ordered_regions
                    if str(span.get("owner_content_id") or "")
                    == region.owner_content_id
                    or str(span.get("owner_route_id") or "") == region.owner_route_id
                ),
                None,
            )
            if direct_owner is not None:
                excluded.append(
                    _native_overlap_trace(
                        row=row,
                        span_id=span_id,
                        span_bbox=bbox,
                        region=direct_owner,
                        coverage=1.0,
                        reason="EXCLUDED_BY_DIRECT_SOURCE_OWNERSHIP",
                        excluded=True,
                        exclusion_basis="DIRECT_SOURCE_OWNERSHIP",
                    )
                )
                continue
            cluster_owner = formula_cluster_owners.get(
                (str(row.get("evidence_id") or ""), span_id)
            )
            if cluster_owner is not None:
                region, span_coverage, owner_coverage, cluster_span_count = (
                    cluster_owner
                )
                trace = _native_overlap_trace(
                    row=row,
                    span_id=span_id,
                    span_bbox=bbox,
                    region=region,
                    coverage=span_coverage,
                    reason="EXCLUDED_BY_NATIVE_FORMULA_SPAN_CLUSTER_GEOMETRY",
                    excluded=True,
                    exclusion_basis=(
                        "FORMULA_SPAN_CLUSTER_CENTER_INSIDE_AND_OWNER_COVERAGE"
                    ),
                )
                trace["owner_coverage_by_span_cluster"] = round(owner_coverage, 12)
                trace["owner_cluster_span_count"] = cluster_span_count
                excluded.append(trace)
                continue
            axis_owner = _formula_axis_aware_span_owner(bbox, ordered_regions)
            if axis_owner is not None:
                region, span_coverage, horizontal_coverage = axis_owner
                trace = _native_overlap_trace(
                    row=row,
                    span_id=span_id,
                    span_bbox=bbox,
                    region=region,
                    coverage=span_coverage,
                    reason="EXCLUDED_BY_NATIVE_FORMULA_AXIS_GEOMETRY",
                    excluded=True,
                    exclusion_basis=("FORMULA_SPAN_CENTER_INSIDE_AND_AXIS_COVERAGE"),
                )
                trace["horizontal_coverage"] = round(horizontal_coverage, 12)
                excluded.append(trace)
                continue
            candidates = []
            for region in ordered_regions:
                intersection = _intersection(bbox, region.bbox_pdf_pt)
                if intersection is None:
                    continue
                intersection_area = _area(intersection)
                coverage = intersection_area / _area(bbox)
                center_inside = (
                    region.bbox_pdf_pt[0]
                    <= (bbox[0] + bbox[2]) / 2
                    <= region.bbox_pdf_pt[2]
                    and region.bbox_pdf_pt[1]
                    <= (bbox[1] + bbox[3]) / 2
                    <= region.bbox_pdf_pt[3]
                )
                candidates.append((coverage, center_inside, region))
            candidates.sort(
                key=lambda value: (
                    -value[0],
                    value[2].kind,
                    value[2].owner_route_id,
                    value[2].owner_content_id,
                )
            )
            owner = next(
                (
                    (coverage, region)
                    for coverage, center_inside, region in candidates
                    if center_inside and coverage >= coverage_threshold
                ),
                None,
            )
            if owner is not None:
                coverage, region = owner
                excluded.append(
                    _native_overlap_trace(
                        row=row,
                        span_id=span_id,
                        span_bbox=bbox,
                        region=region,
                        coverage=coverage,
                        reason="EXCLUDED_BY_NATIVE_SPAN_GEOMETRY",
                        excluded=True,
                    )
                )
                continue
            if candidates:
                union_regions = tuple(
                    sorted(
                        {candidate[2] for candidate in candidates},
                        key=lambda value: (
                            value.bbox_pdf_pt,
                            value.kind,
                            value.owner_route_id,
                            value.owner_content_id,
                        ),
                    )
                )
                union_coverage = _ownership_union_area(bbox, union_regions) / _area(
                    bbox
                )
                center_inside_union = any(
                    center_inside for _coverage, center_inside, _region in candidates
                )
                if center_inside_union and union_coverage >= coverage_threshold:
                    region = union_regions[0]
                    trace = _native_overlap_trace(
                        row=row,
                        span_id=span_id,
                        span_bbox=bbox,
                        region=region,
                        coverage=union_coverage,
                        reason="EXCLUDED_BY_NATIVE_SPAN_OWNERSHIP_UNION",
                        excluded=True,
                        exclusion_basis=(
                            "OWNERSHIP_UNION_CENTER_INSIDE_AND_COVERAGE_THRESHOLD"
                        ),
                    )
                    trace["owners"] = [
                        {
                            "owner_kind": owner_region.kind,
                            "owner_route_id": owner_region.owner_route_id,
                            "owner_content_id": owner_region.owner_content_id,
                            "owner_bbox": list(owner_region.bbox_pdf_pt),
                        }
                        for owner_region in union_regions
                    ]
                    excluded.append(trace)
                    continue
                coverage, _center_inside, region = candidates[0]
                partial.append(
                    _native_overlap_trace(
                        row=row,
                        span_id=span_id,
                        span_bbox=bbox,
                        region=region,
                        coverage=coverage,
                        reason="PARTIAL_NATIVE_NON_TEXT_OVERLAP",
                        excluded=False,
                    )
                )
            retained.append(span)
        if not retained:
            continue
        if has_span_geometry:
            row["spans"] = retained
            row["text"] = "".join(str(span.get("text") or "") for span in retained)
            row["retained_source_span_ids"] = [
                str(span.get("span_id")) for span in retained if span.get("span_id")
            ]
            row["retained_text_fragments"] = _retained_text_fragments(
                retained, ordered_regions
            )
        output.append(row)
    return NativeTextExclusionResult(
        rows=tuple(output),
        excluded_spans=tuple(excluded),
        partial_overlaps=tuple(partial),
    )


def _retained_text_fragments(spans, regions):
    """Keep formula gaps as ordering fences without changing source line IDs."""
    fragments = []
    visible_geometry = []
    previous_span_box = None
    for span in spans:
        box = span.get("bbox_pdf_pt")
        if not box or len(box) != 4:
            return []
        previous = fragments[-1]["bbox_pdf_pt"] if fragments else None
        separated = previous is not None and any(
            region.bbox_pdf_pt[0] < box[0]
            and (
                region.bbox_pdf_pt[2] > previous[2]
                or (
                    region.kind == "FORMULA" and previous_span_box is not None
                    and (previous_span_box[0] + previous_span_box[2]) / 2
                    < (region.bbox_pdf_pt[0] + region.bbox_pdf_pt[2]) / 2
                    < (box[0] + box[2]) / 2
                )
            )
            and region.bbox_pdf_pt[1] < min(previous[3], box[3])
            and region.bbox_pdf_pt[3] > max(previous[1], box[1])
            for region in regions
        )
        if previous is None or separated:
            fragments.append(
                {"text": "", "bbox_pdf_pt": list(box), "source_span_ids": []}
            )
            visible_geometry.append(False)
        fragment = fragments[-1]
        fragment["text"] += str(span.get("text") or "")
        fragment["source_span_ids"].append(str(span.get("span_id") or ""))
        if str(span.get("text") or "").strip():
            prior = fragment["bbox_pdf_pt"] if visible_geometry[-1] else box
            fragment["bbox_pdf_pt"] = [
                min(prior[0], box[0]), min(prior[1], box[1]),
                max(prior[2], box[2]), max(prior[3], box[3]),
            ]
            visible_geometry[-1] = True
        previous_span_box = box
    return fragments


def _formula_axis_aware_span_owner(
    bbox: tuple[float, float, float, float],
    regions: Iterable[NonTextOwnershipRegion],
) -> tuple[NonTextOwnershipRegion, float, float] | None:
    """Select a Formula owner for an independent glyph without reading its text."""

    candidates = []
    for region in regions:
        if region.kind != "FORMULA":
            continue
        intersection = _intersection(bbox, region.bbox_pdf_pt)
        if intersection is None:
            continue
        center_inside = (
            region.bbox_pdf_pt[0] <= (bbox[0] + bbox[2]) / 2 <= region.bbox_pdf_pt[2]
            and region.bbox_pdf_pt[1]
            <= (bbox[1] + bbox[3]) / 2
            <= region.bbox_pdf_pt[3]
        )
        span_coverage = _area(intersection) / _area(bbox)
        horizontal_coverage = (intersection[2] - intersection[0]) / (bbox[2] - bbox[0])
        if (
            center_inside
            and span_coverage >= NATIVE_FORMULA_AXIS_MIN_SPAN_COVERAGE
            and horizontal_coverage >= NATIVE_FORMULA_AXIS_MIN_HORIZONTAL_COVERAGE
        ):
            candidates.append((span_coverage, horizontal_coverage, region))
    if not candidates:
        return None
    span_coverage, horizontal_coverage, region = min(
        candidates,
        key=lambda value: (
            -value[0],
            -value[1],
            value[2].owner_route_id,
            value[2].owner_content_id,
        ),
    )
    return region, span_coverage, horizontal_coverage


def _formula_span_cluster_owners(
    rows: Iterable[dict[str, Any]],
    regions: Iterable[NonTextOwnershipRegion],
) -> dict[tuple[str, str], tuple[NonTextOwnershipRegion, float, float, int]]:
    """Prove ownership for split formula glyph spans without reading their text."""

    candidates_by_span: dict[
        tuple[str, str],
        list[tuple[float, float, NonTextOwnershipRegion, int]],
    ] = {}
    rows_with_spans = tuple(
        (str(row.get("evidence_id") or ""), tuple(row.get("spans") or ()))
        for row in rows
        if row.get("spans")
    )
    for region in regions:
        if region.kind != "FORMULA":
            continue
        candidates: list[
            tuple[tuple[str, str], tuple[float, float, float, float], float]
        ] = []
        for line_id, spans in rows_with_spans:
            for span in spans:
                span_id = str(span.get("span_id") or "")
                bbox_value = span.get("bbox_pdf_pt")
                if not (
                    span_id
                    and isinstance(bbox_value, (list, tuple))
                    and len(bbox_value) == 4
                    and float(bbox_value[2]) > float(bbox_value[0])
                    and float(bbox_value[3]) > float(bbox_value[1])
                ):
                    continue
                bbox = tuple(float(value) for value in bbox_value)
                intersection = _intersection(bbox, region.bbox_pdf_pt)
                if intersection is None:
                    continue
                center_inside = (
                    region.bbox_pdf_pt[0]
                    <= (bbox[0] + bbox[2]) / 2
                    <= region.bbox_pdf_pt[2]
                    and region.bbox_pdf_pt[1]
                    <= (bbox[1] + bbox[3]) / 2
                    <= region.bbox_pdf_pt[3]
                )
                span_coverage = _area(intersection) / _area(bbox)
                horizontal_coverage = (intersection[2] - intersection[0]) / (
                    bbox[2] - bbox[0]
                )
                if (
                    center_inside
                    and span_coverage >= NATIVE_FORMULA_CLUSTER_MIN_SPAN_COVERAGE
                    and horizontal_coverage
                    >= NATIVE_FORMULA_CLUSTER_MIN_HORIZONTAL_COVERAGE
                ):
                    candidates.append(((line_id, span_id), bbox, span_coverage))
        if len(candidates) < NATIVE_FORMULA_CLUSTER_MIN_SPANS:
            continue
        owner_coverage = _bbox_union_area(
            region.bbox_pdf_pt,
            (bbox for _key, bbox, _coverage in candidates),
        ) / _area(region.bbox_pdf_pt)
        if owner_coverage < NATIVE_FORMULA_CLUSTER_OWNER_COVERAGE_THRESHOLD:
            continue
        for key, _bbox, span_coverage in candidates:
            candidates_by_span.setdefault(key, []).append(
                (owner_coverage, span_coverage, region, len(candidates))
            )
    resolved = {}
    for key, candidates in candidates_by_span.items():
        owner_coverage, span_coverage, region, span_count = min(
            candidates,
            key=lambda value: (
                -value[0],
                -value[1],
                value[2].owner_route_id,
                value[2].owner_content_id,
            ),
        )
        resolved[key] = (region, span_coverage, owner_coverage, span_count)
    return resolved


def _area(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _native_overlap_trace(
    *,
    row: dict[str, Any],
    span_id: str,
    span_bbox: tuple[float, float, float, float],
    region: NonTextOwnershipRegion,
    coverage: float,
    reason: str,
    excluded: bool,
    exclusion_basis: str = "SPAN_CENTER_INSIDE_AND_COVERAGE_THRESHOLD",
) -> dict[str, Any]:
    return {
        "source_line_id": str(row.get("evidence_id") or ""),
        "source_span_id": span_id,
        "source_span_bbox_pdf_pt": list(span_bbox),
        "excluded_by_non_text_owner": excluded,
        "owner_kind": region.kind,
        "owner_route_id": region.owner_route_id,
        "owner_content_id": region.owner_content_id,
        "owner_bbox": list(region.bbox_pdf_pt),
        "exclusion_basis": exclusion_basis,
        "intersection_over_span_area": round(coverage, 12),
        "reason": reason,
    }


def bbox_fully_owned(
    bbox_pdf_pt: Iterable[float],
    regions: Iterable[NonTextOwnershipRegion],
) -> bool:
    """Return whether the exact ownership union covers the requested bbox."""

    target = tuple(float(value) for value in bbox_pdf_pt)
    if len(target) != 4 or _area(target) <= 0:
        raise ValueError("TEXT_ROUTE_BBOX_INVALID")
    return _ownership_union_area(target, regions) >= _area(target) - 1e-9


def _ownership_union_area(
    target: tuple[float, float, float, float],
    regions: Iterable[NonTextOwnershipRegion],
) -> float:
    return _bbox_union_area(target, (region.bbox_pdf_pt for region in regions))


def _bbox_union_area(
    target: tuple[float, float, float, float],
    bboxes: Iterable[tuple[float, float, float, float]],
) -> float:
    clipped = [
        value for bbox in bboxes if (value := _intersection(target, bbox)) is not None
    ]
    if not clipped:
        return 0.0
    xs = sorted(
        {
            target[0],
            target[2],
            *(value for bbox in clipped for value in (bbox[0], bbox[2])),
        }
    )
    union_area = 0.0
    for left, right in pairwise(xs):
        if right <= left:
            continue
        intervals = sorted(
            (bbox[1], bbox[3]) for bbox in clipped if bbox[0] < right and bbox[2] > left
        )
        covered_y = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered_y += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered_y += end - start
        union_area += (right - left) * covered_y
    return union_area
