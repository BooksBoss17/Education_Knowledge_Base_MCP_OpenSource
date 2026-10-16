"""Canonical bounded crops and historical chunks for page visual text recovery."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class PageTextRecoveryChunkingConfig:
    max_lines: int = 8
    max_estimated_chars: int = 160
    max_vertical_gap_px: int = 96
    padding_px: int = 4
    wide_line_ratio: float = 0.65

    def __post_init__(self) -> None:
        if (
            self.max_lines < 1
            or self.max_estimated_chars < 1
            or self.max_vertical_gap_px < 0
            or self.padding_px < 0
            or not 0.5 <= self.wide_line_ratio <= 1.0
        ):
            raise ValueError("PAGE_TEXT_RECOVERY_CHUNK_CONFIG_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "bemarkdown-page-text-recovery-chunking-config-v1",
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class PageTextRecoveryChunk:
    chunk_id: str
    document_id: str
    page_index: int
    source_page_ref: str
    source_page_sha256: str
    bbox_pdf_pt: tuple[float, float, float, float]
    bbox_pixel: tuple[int, int, int, int]
    member_line_ids: tuple[str, ...]
    member_line_count: int
    reading_order_index: int
    crop_ref: str
    crop_sha256: str
    provider_a_text: str
    provider_evidence_refs: tuple[str, ...]
    assembly_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "bbox_pdf_pt",
            "bbox_pixel",
            "member_line_ids",
            "provider_evidence_refs",
        ):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True, slots=True)
class PageTextRecoveryPlan:
    source_route_id: str
    source_page_ref: str
    source_page_sha256: str
    source_pixel_size: tuple[int, int]
    config: PageTextRecoveryChunkingConfig
    source_line_ids: tuple[str, ...]
    source_reading_sequence: tuple[int, ...]
    chunks: tuple[PageTextRecoveryChunk, ...]

    def semantic_contract(self) -> dict[str, Any]:
        return {
            "schema": "bemarkdown-page-text-recovery-plan-v1",
            "source_route_id": self.source_route_id,
            "source_page_sha256": self.source_page_sha256,
            "source_pixel_size": list(self.source_pixel_size),
            "config": self.config.to_dict(),
            "source_line_ids": list(self.source_line_ids),
            "source_reading_sequence": list(self.source_reading_sequence),
            "chunks": [
                {
                    key: value
                    for key, value in chunk.to_dict().items()
                    if key != "crop_ref"
                }
                for chunk in self.chunks
            ],
        }

    def spatial_validation(self) -> dict[str, Any]:
        width, height = self.source_pixel_size
        outside = 0
        member_outside = 0
        empty = 0
        for chunk in self.chunks:
            left, top, right, bottom = chunk.bbox_pixel
            if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
                outside += 1
            if right <= left or bottom <= top:
                empty += 1
            for member in chunk.assembly_provenance["member_bbox_pixel"]:
                m_left, m_top, m_right, m_bottom = member
                if not (
                    left <= m_left <= m_right <= right
                    and top <= m_top <= m_bottom <= bottom
                ):
                    member_outside += 1
        gate = outside == member_outside == empty == 0
        return {
            "chunk_count": len(self.chunks),
            "bbox_outside_source_count": outside,
            "member_outside_chunk_count": member_outside,
            "empty_bbox_count": empty,
            "gate": "PASS" if gate else "FAIL",
        }

    def reading_order_validation(self) -> dict[str, Any]:
        source = list(self.source_reading_sequence)
        assembled = [
            int(sequence)
            for chunk in self.chunks
            for sequence in chunk.assembly_provenance["source_reading_sequences"]
        ]
        duplicate_count = len(assembled) - len(set(assembled))
        inversions = sum(
            assembled[left] > assembled[right]
            for left in range(len(assembled))
            for right in range(left + 1, len(assembled))
        )
        gate = source == assembled and duplicate_count == inversions == 0
        return {
            "source_sequence": source,
            "assembled_sequence": assembled,
            "duplicate_sequence_count": duplicate_count,
            "inversion_count": inversions,
            "gate": "PASS" if gate else "FAIL",
        }

    def line_conservation(self) -> dict[str, Any]:
        assigned = [line_id for chunk in self.chunks for line_id in chunk.member_line_ids]
        source = set(self.source_line_ids)
        assigned_set = set(assigned)
        missing = source.difference(assigned_set)
        duplicates = len(assigned) - len(assigned_set)
        return {
            "source_line_count": len(self.source_line_ids),
            "assigned_line_count": len(assigned),
            "missing_source_line_count": len(missing),
            "duplicate_source_line_count": duplicates,
            "unassigned_source_line_count": len(missing),
            "gate": "PASS" if not missing and duplicates == 0 else "FAIL",
        }


@dataclass(frozen=True, slots=True)
class PageTextRecoveryAssembly:
    selected_text: str
    chunk_ids: tuple[str, ...]
    resolution_statuses: tuple[str, ...]
    chunk_results: tuple[dict[str, Any], ...]
    validation: dict[str, Any]


def assemble_page_text_recovery(
    plan: PageTextRecoveryPlan, results: Sequence[Mapping[str, Any]]
) -> PageTextRecoveryAssembly:
    expected = [chunk.chunk_id for chunk in plan.chunks]
    supplied = [str(result["chunk_id"]) for result in results]
    duplicates = len(supplied) - len(set(supplied))
    missing = set(expected).difference(supplied)
    unknown = set(supplied).difference(expected)
    by_id = {str(result["chunk_id"]): dict(result) for result in results}
    conservation = plan.line_conservation()
    validation = {
        "expected_chunk_count": len(expected),
        "result_chunk_count": len(results),
        "missing_chunk_count": len(missing),
        "duplicate_chunk_count": duplicates,
        "unknown_chunk_count": len(unknown),
        "line_missing_count": conservation["missing_source_line_count"],
        "line_duplicate_count": conservation["duplicate_source_line_count"],
        "gate": (
            "PASS"
            if not missing
            and not unknown
            and duplicates == 0
            and conservation["gate"] == "PASS"
            else "FAIL"
        ),
    }
    if validation["gate"] != "PASS":
        raise RuntimeError(f"PAGE_TEXT_RECOVERY_ASSEMBLY_INVALID:{validation}")
    ordered = tuple(by_id[chunk_id] for chunk_id in expected)
    return PageTextRecoveryAssembly(
        selected_text="\n".join(
            str(result.get("selected_text") or "")
            for result in ordered
            if result.get("selected_text")
        ),
        chunk_ids=tuple(expected),
        resolution_statuses=tuple(
            str(result.get("resolution_status") or "") for result in ordered
        ),
        chunk_results=ordered,
        validation=validation,
    )


@dataclass(frozen=True, slots=True)
class PageBoundedTextUnit:
    """One production voting unit backed by one current-run PP line crop."""

    unit_id: str
    document_id: str
    page_index: int
    source_page_ref: str
    source_page_sha256: str
    bbox_pdf_pt: tuple[float, float, float, float]
    bbox_pixel: tuple[int, int, int, int]
    source_line_id: str
    reading_order_index: int
    source_reading_sequence: int
    crop_ref: str
    crop_sha256: str
    provider_a_text: str
    provider_a_confidence: float | None
    assembly_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("bbox_pdf_pt", "bbox_pixel"):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True, slots=True)
class PageBoundedTextPlan:
    """Production page-recovery plan with exactly one line per voting unit."""

    source_route_id: str
    source_page_ref: str
    source_page_sha256: str
    source_pixel_size: tuple[int, int]
    source_line_ids: tuple[str, ...]
    source_reading_sequence: tuple[int, ...]
    units: tuple[PageBoundedTextUnit, ...]

    def semantic_contract(self) -> dict[str, Any]:
        return {
            "schema": "bemarkdown-page-bounded-text-plan-v1",
            "contract_version": "CANONICAL_BOUNDED_TEXT_CROP_V1",
            "source_route_id": self.source_route_id,
            "source_page_sha256": self.source_page_sha256,
            "source_pixel_size": list(self.source_pixel_size),
            "source_line_ids": list(self.source_line_ids),
            "source_reading_sequence": list(self.source_reading_sequence),
            "units": [
                {
                    key: value
                    for key, value in unit.to_dict().items()
                    if key != "crop_ref"
                }
                for unit in self.units
            ],
        }

    def spatial_validation(self) -> dict[str, Any]:
        width, height = self.source_pixel_size
        outside = 0
        empty = 0
        for unit in self.units:
            left, top, right, bottom = unit.bbox_pixel
            if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
                outside += 1
            if right <= left or bottom <= top:
                empty += 1
        return {
            "bounded_unit_count": len(self.units),
            "bbox_outside_source_count": outside,
            "empty_bbox_count": empty,
            "gate": "PASS" if outside == empty == 0 else "FAIL",
        }

    def reading_order_validation(self) -> dict[str, Any]:
        source = list(self.source_reading_sequence)
        assembled = [unit.source_reading_sequence for unit in self.units]
        duplicates = len(assembled) - len(set(assembled))
        inversions = sum(
            assembled[left] > assembled[right]
            for left in range(len(assembled))
            for right in range(left + 1, len(assembled))
        )
        return {
            "source_sequence": source,
            "assembled_sequence": assembled,
            "duplicate_sequence_count": duplicates,
            "inversion_count": inversions,
            "gate": (
                "PASS"
                if source == assembled and duplicates == inversions == 0
                else "FAIL"
            ),
        }

    def line_conservation(self) -> dict[str, Any]:
        assigned = [unit.source_line_id for unit in self.units]
        source = set(self.source_line_ids)
        assigned_set = set(assigned)
        missing = source.difference(assigned_set)
        duplicates = len(assigned) - len(assigned_set)
        return {
            "source_line_count": len(self.source_line_ids),
            "bounded_voting_unit_count": len(self.units),
            "assigned_line_count": len(assigned),
            "missing_source_line_count": len(missing),
            "duplicate_source_line_count": duplicates,
            "unassigned_source_line_count": len(missing),
            "gate": "PASS" if not missing and duplicates == 0 else "FAIL",
        }


@dataclass(frozen=True, slots=True)
class PageBoundedTextAssembly:
    selected_text: str
    unit_ids: tuple[str, ...]
    resolution_statuses: tuple[str, ...]
    unit_results: tuple[dict[str, Any], ...]
    validation: dict[str, Any]


def assemble_page_bounded_text_recovery(
    plan: PageBoundedTextPlan, results: Sequence[Mapping[str, Any]]
) -> PageBoundedTextAssembly:
    expected = [unit.unit_id for unit in plan.units]
    supplied = [str(result["unit_id"]) for result in results]
    duplicates = len(supplied) - len(set(supplied))
    missing = set(expected).difference(supplied)
    unknown = set(supplied).difference(expected)
    by_id = {str(result["unit_id"]): dict(result) for result in results}
    conservation = plan.line_conservation()
    validation = {
        "expected_bounded_unit_count": len(expected),
        "result_bounded_unit_count": len(results),
        "missing_bounded_unit_count": len(missing),
        "duplicate_bounded_unit_count": duplicates,
        "unknown_bounded_unit_count": len(unknown),
        "line_missing_count": conservation["missing_source_line_count"],
        "line_duplicate_count": conservation["duplicate_source_line_count"],
        "gate": (
            "PASS"
            if not missing
            and not unknown
            and duplicates == 0
            and conservation["gate"] == "PASS"
            else "FAIL"
        ),
    }
    if validation["gate"] != "PASS":
        raise RuntimeError(f"PAGE_BOUNDED_TEXT_ASSEMBLY_INVALID:{validation}")
    ordered = tuple(by_id[unit_id] for unit_id in expected)
    return PageBoundedTextAssembly(
        selected_text="\n".join(
            str(result.get("selected_text") or "")
            for result in ordered
            if result.get("selected_text")
        ),
        unit_ids=tuple(expected),
        resolution_statuses=tuple(
            str(result.get("resolution_status") or "") for result in ordered
        ),
        unit_results=ordered,
        validation=validation,
    )


class PageVisualTextRecoveryBoundedCropper:
    """Materialize the production one-PP-line/one-vote contract."""

    contract_version = "CANONICAL_BOUNDED_TEXT_CROP_V1"

    def prepare(
        self,
        route: dict[str, Any],
        source_crop: dict[str, Any],
        pp_lines: list[dict[str, Any]],
    ) -> PageBoundedTextPlan:
        if route.get("adapter") not in {"PAGE_VISUAL_TEXT_RECOVERY", "OCR_TEXT_REGION"}:
            raise ValueError("PAGE_TEXT_RECOVERY_ROUTE_REQUIRED")
        source_path = Path(str(source_crop["path"])).resolve(strict=True)
        source_sha = _sha256_file(source_path)
        if source_sha != str(source_crop["content_sha256"]):
            raise RuntimeError("PAGE_TEXT_RECOVERY_SOURCE_CROP_MISMATCH")
        source_size = (int(source_crop["width"]), int(source_crop["height"]))
        ordered = sorted(
            pp_lines,
            key=lambda line: (
                int(line["reading_sequence"]),
                tuple(float(value) for value in line["bbox_local_px"]),
                str(line.get("line_crop_sha256") or ""),
            ),
        )
        units: list[PageBoundedTextUnit] = []
        for index, line in enumerate(ordered):
            line_id = _line_id(str(route["route_id"]), line)
            line_crop_ref = str(line.get("line_crop_ref") or "")
            line_crop_sha = str(line.get("line_crop_sha256") or "")
            if not line_crop_ref or len(line_crop_sha) != 64:
                raise RuntimeError(
                    f"PAGE_BOUNDED_TEXT_LINE_CROP_REQUIRED:{line_id}"
                )
            line_path = Path(line_crop_ref).resolve(strict=True)
            if _sha256_file(line_path) != line_crop_sha:
                raise RuntimeError(
                    f"PAGE_BOUNDED_TEXT_LINE_CROP_MISMATCH:{line_id}"
                )
            bbox_pixel = tuple(
                round(float(value)) for value in line["bbox_local_px"]
            )
            bbox_pdf = tuple(float(value) for value in line["bbox_page_pdf_pt"])
            identity = _semantic_sha256(
                {
                    "source_route_id": route["route_id"],
                    "source_line_id": line_id,
                    "reading_sequence": int(line["reading_sequence"]),
                    "crop_sha256": line_crop_sha,
                    "contract_version": self.contract_version,
                }
            )
            units.append(
                PageBoundedTextUnit(
                    unit_id=f"page-text-unit-{identity[:24]}",
                    document_id=str(route["document_id"]),
                    page_index=int(route["page_index"]),
                    source_page_ref=str(source_path),
                    source_page_sha256=source_sha,
                    bbox_pdf_pt=bbox_pdf,  # type: ignore[arg-type]
                    bbox_pixel=bbox_pixel,  # type: ignore[arg-type]
                    source_line_id=line_id,
                    reading_order_index=index,
                    source_reading_sequence=int(line["reading_sequence"]),
                    crop_ref=str(line_path),
                    crop_sha256=line_crop_sha,
                    provider_a_text=str(line.get("text") or ""),
                    provider_a_confidence=(
                        float(line["confidence"])
                        if line.get("confidence") is not None
                        else None
                    ),
                    assembly_provenance={
                        "source_route_id": str(route["route_id"]),
                        **({'figure_label_isolation': dict(line['figure_label_isolation'])}
                           if line.get('figure_label_isolation') else {}),
                        **({'figure_label_segmentation': dict(line['figure_label_segmentation'])}
                           if line.get('figure_label_segmentation') else {}),
                        "contract_version": self.contract_version,
                        "recognition_unit": "ONE_CURRENT_RUN_PP_LINE_CROP",
                        "source_reading_sequence": int(line["reading_sequence"]),
                        "crop_method": "PP_LINE_CROP_REUSE",
                    },
                )
            )
        plan = PageBoundedTextPlan(
            source_route_id=str(route["route_id"]),
            source_page_ref=str(source_path),
            source_page_sha256=source_sha,
            source_pixel_size=source_size,
            source_line_ids=tuple(unit.source_line_id for unit in units),
            source_reading_sequence=tuple(
                unit.source_reading_sequence for unit in units
            ),
            units=tuple(units),
        )
        gates = (
            plan.spatial_validation()["gate"],
            plan.reading_order_validation()["gate"],
            plan.line_conservation()["gate"],
        )
        if set(gates) != {"PASS"}:
            raise RuntimeError(f"PAGE_BOUNDED_TEXT_PLAN_INVALID:{gates}")
        return plan

class PageVisualTextRecoveryChunker:
    """Historical multi-line artifact-replay seam; not used by production."""

    def __init__(self, config: PageTextRecoveryChunkingConfig | None = None) -> None:
        self.config = config or PageTextRecoveryChunkingConfig()

    def prepare(
        self,
        route: dict[str, Any],
        source_crop: dict[str, Any],
        pp_lines: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> PageTextRecoveryPlan:
        if route.get("adapter") != "PAGE_VISUAL_TEXT_RECOVERY":
            raise ValueError("PAGE_TEXT_RECOVERY_ROUTE_REQUIRED")
        source_path = Path(str(source_crop["path"])).resolve(strict=True)
        source_sha = _sha256_file(source_path)
        if source_sha != str(source_crop["content_sha256"]):
            raise RuntimeError("PAGE_TEXT_RECOVERY_SOURCE_CROP_MISMATCH")
        ordered = sorted(
            pp_lines,
            key=lambda line: (
                int(line["reading_sequence"]),
                tuple(float(value) for value in line["bbox_local_px"]),
                str(line.get("line_crop_sha256") or ""),
            ),
        )
        source_width = int(source_crop["width"])
        ordered_lanes = [
            _column_lane(line, source_width, self.config.wide_line_ratio)
            for line in ordered
        ]
        lane_transition_count = sum(
            ordered_lanes[index] != ordered_lanes[index - 1]
            for index in range(1, len(ordered_lanes))
        )
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_chars = 0
        current_lane: str | None = None
        for line in ordered:
            text = str(line.get("text") or "")
            lane = _column_lane(line, source_width, self.config.wide_line_ratio)
            additional = len(text) + (1 if current else 0)
            vertical_gap = (
                max(
                    0.0,
                    float(line["bbox_local_px"][1])
                    - float(current[-1]["bbox_local_px"][3]),
                )
                if current
                else 0.0
            )
            if current and (
                len(current) >= self.config.max_lines
                or current_chars + additional > self.config.max_estimated_chars
                or lane != current_lane
                or vertical_gap > self.config.max_vertical_gap_px
            ):
                groups.append(current)
                current = []
                current_chars = 0
                additional = len(text)
            if not current:
                current_lane = lane
            current.append(line)
            current_chars += additional
        if current:
            groups.append(current)

        target = Path(output_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        chunks = []
        with Image.open(source_path) as opened:
            source_image = opened.convert("RGB")
            for index, group in enumerate(groups):
                member_ids = tuple(
                    _line_id(str(route["route_id"]), line) for line in group
                )
                reuse_line_crop = len(group) == 1 and bool(
                    group[0].get("line_crop_ref")
                )
                bbox_pixel = _pixel_bbox(
                    group,
                    source_image.size,
                    0 if reuse_line_crop else self.config.padding_px,
                )
                bbox_pdf = _union_bbox(group, "bbox_page_pdf_pt")
                if reuse_line_crop:
                    line_crop_path = Path(str(group[0]["line_crop_ref"])).resolve(
                        strict=True
                    )
                    payload = line_crop_path.read_bytes()
                    expected_line_sha = str(group[0].get("line_crop_sha256") or "")
                    if hashlib.sha256(payload).hexdigest() != expected_line_sha:
                        raise RuntimeError("PAGE_TEXT_RECOVERY_LINE_CROP_MISMATCH")
                    crop_method = "PP_LINE_CROP_REUSE"
                else:
                    chunk_image = source_image.crop(bbox_pixel)
                    payload = _png_bytes(chunk_image)
                    crop_method = "SOURCE_PAGE_BBOX_CROP"
                crop_sha = hashlib.sha256(payload).hexdigest()
                crop_path = target / f"sha256-{crop_sha}.png"
                if not crop_path.exists():
                    crop_path.write_bytes(payload)
                identity = _semantic_sha256(
                    {
                        "document_id": route["document_id"],
                        "page_index": int(route["page_index"]),
                        "source_page_sha256": source_sha,
                        "member_line_ids": member_ids,
                        "config": self.config.to_dict(),
                    }
                )
                chunks.append(
                    PageTextRecoveryChunk(
                        chunk_id=f"page-text-chunk-{identity[:24]}",
                        document_id=str(route["document_id"]),
                        page_index=int(route["page_index"]),
                        source_page_ref=str(source_path),
                        source_page_sha256=source_sha,
                        bbox_pdf_pt=bbox_pdf,
                        bbox_pixel=bbox_pixel,
                        member_line_ids=member_ids,
                        member_line_count=len(group),
                        reading_order_index=index,
                        crop_ref=str(crop_path),
                        crop_sha256=crop_sha,
                        provider_a_text="\n".join(
                            str(line.get("text") or "") for line in group
                        ),
                        provider_evidence_refs=tuple(
                            str(line.get("line_crop_sha256") or "") for line in group
                        ),
                        assembly_provenance={
                            "source_route_id": str(route["route_id"]),
                            "grouping_mode": "BOUNDED_MULTI_LINE",
                            "lane_transition_count": lane_transition_count,
                            "crop_method": crop_method,
                            "source_reading_sequences": [
                                int(line["reading_sequence"]) for line in group
                            ],
                            "member_bbox_pixel": [
                                [float(value) for value in line["bbox_local_px"]]
                                for line in group
                            ],
                            "column_lane": _column_lane(
                                group[0], source_image.width, self.config.wide_line_ratio
                            ),
                        },
                    )
                )
        return PageTextRecoveryPlan(
            source_route_id=str(route["route_id"]),
            source_page_ref=str(source_path),
            source_page_sha256=source_sha,
            source_pixel_size=source_image.size,
            config=self.config,
            source_line_ids=tuple(
                _line_id(str(route["route_id"]), line) for line in ordered
            ),
            source_reading_sequence=tuple(
                int(line["reading_sequence"]) for line in ordered
            ),
            chunks=tuple(chunks),
        )


def _line_id(route_id: str, line: dict[str, Any]) -> str:
    identity = _semantic_sha256(
        {
            "route_id": route_id,
            "reading_sequence": int(line["reading_sequence"]),
            "line_crop_sha256": str(line.get("line_crop_sha256") or ""),
            "bbox_local_px": [float(value) for value in line["bbox_local_px"]],
        }
    )
    return f"page-text-line-{identity[:24]}"


def _column_lane(line: dict[str, Any], page_width: int, wide_ratio: float) -> str:
    left, _top, right, _bottom = [
        float(value) for value in line["bbox_local_px"]
    ]
    if (right - left) / page_width >= wide_ratio:
        return "SPAN"
    midpoint = page_width / 2
    if right <= midpoint:
        return "LEFT"
    if left >= midpoint:
        return "RIGHT"
    return "CENTER"


def _pixel_bbox(
    lines: list[dict[str, Any]], size: tuple[int, int], padding: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = _union_bbox(lines, "bbox_local_px")
    return (
        max(0, math.floor(left) - padding),
        max(0, math.floor(top) - padding),
        min(size[0], math.ceil(right) + padding),
        min(size[1], math.ceil(bottom) + padding),
    )


def _union_bbox(
    lines: list[dict[str, Any]], key: str
) -> tuple[float, float, float, float]:
    boxes = [[float(value) for value in line[key]] for line in lines]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False)
    return stream.getvalue()
