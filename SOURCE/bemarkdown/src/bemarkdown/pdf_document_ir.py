from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DOCUMENT_IR_SCHEMA = "bemarkdown-document-ir-v0"
BLOCK_SCHEMA = "bemarkdown-document-block-ir-v0"
READING_ORDER_VERSION = "document-spatial-reading-order-v1"
SPATIAL_ORDER_CONTRACT_VERSION = "page-asc-top-to-bottom-same-band-left-to-right-v1"
DRAFT_PACKAGE_SCHEMA = "bemarkdown-draft-package-v0"
DRAFT_RENDERER_VERSION = "draft-markdown-renderer-v0"
PATCH_SCHEMA = "bemarkdown-output-audit-patch-v0"
NODE_LOCATOR_SCHEMA = "bemarkdown-document-node-locator-v0"
ORDERING_ATOM_CONTRACT_VERSION = "transient-ordering-atom-projection-v1"
POST_ORDER_GROUPING_VERSION = "post-order-text-grouping-v1"

BLOCK_KINDS = {
    "TITLE",
    "TEXT",
    "FORMULA",
    "IMAGE",
    "CAPTION",
    "TABLE",
    "OTHER",
    "HEADER_FOOTER",
    "PAGE_NUMBER",
}


@dataclass(frozen=True, slots=True)
class SpatialReadingOrderConfig:
    """Global deterministic horizontal-band configuration in PDF points."""

    page_height_ratio: float = 0.006
    block_height_ratio: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "schema": "bemarkdown-spatial-reading-order-config-v1",
            "coordinate_space": "CANONICAL_PDF_PT",
        }


@dataclass(frozen=True, slots=True)
class OrderingAtom:
    """Transient source-owned ordering unit; never a persisted IR contract."""

    atom_id: str
    parent_content_id: str
    page_index: int
    kind: str
    bbox_pdf_pt: tuple[float, float, float, float]
    payload: dict[str, Any]
    source_candidate_ids: tuple[str, ...]
    source_region_ids: tuple[str, ...]
    source_unit_ids: tuple[str, ...]
    provenance: dict[str, Any]
    origin_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "parent_content_id": self.parent_content_id,
            "page_index": self.page_index,
            "kind": self.kind,
            "bbox_pdf_pt": list(self.bbox_pdf_pt),
            "payload": copy.deepcopy(self.payload),
            "source_candidate_ids": list(self.source_candidate_ids),
            "source_region_ids": list(self.source_region_ids),
            "source_unit_ids": list(self.source_unit_ids),
            "provenance": copy.deepcopy(self.provenance),
            "origin_type": self.origin_type,
        }


PAGE_FURNITURE_KINDS = {"HEADER_FOOTER", "PAGE_NUMBER"}
PATCH_OPERATIONS = {
    "REPLACE_TEXT",
    "REPLACE_FORMULA",
    "INSERT_BLOCK",
    "DELETE_DUPLICATE",
    "MOVE_BLOCK",
    "SPLIT_BLOCK",
    "MERGE_BLOCK",
    "CHANGE_HEADING_LEVEL",
    "FIX_ASSET_REFERENCE",
    "REPLACE_TABLE",
    "CHANGE_BLOCK_KIND",
    "UPDATE_CAPTION_RELATION",
}
_ANCHOR_RE = re.compile(
    r"<!--\s*bemarkdown:block\s+id=\"(?P<id>[^\"]+)\""
    r"\s+page=\"(?P<page>\d+)\"\s+kind=\"(?P<kind>[A-Z_]+)\"[^>]*-->",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_node_id(
    document_id: str,
    *,
    source_content_ids: Iterable[str],
    source_region_ids: Iterable[str],
    source_unit_ids: Iterable[str],
    kind: str,
    grouping_key: str | None = None,
) -> str:
    """Return a source-owned identity that is independent from presentation order."""

    identity = {
        "document_id": str(document_id),
        "source_content_ids": sorted({str(value) for value in source_content_ids}),
        "source_region_ids": sorted({str(value) for value in source_region_ids}),
        "source_unit_ids": sorted({str(value) for value in source_unit_ids}),
        "kind": str(kind),
        "grouping_key": grouping_key,
        "schema": BLOCK_SCHEMA,
    }
    digest = semantic_sha256(identity)[:24]
    return f"doc-node-{digest}"


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _normalized_bbox(bbox: Any, width: float, height: float) -> list[float] | None:
    if not _valid_bbox(bbox) or width <= 0 or height <= 0:
        return None
    return [
        round(max(0.0, min(1.0, float(bbox[0]) / width)), 8),
        round(max(0.0, min(1.0, float(bbox[1]) / height)), 8),
        round(max(0.0, min(1.0, float(bbox[2]) / width)), 8),
        round(max(0.0, min(1.0, float(bbox[3]) / height)), 8),
    ]


def _kind(row: dict[str, Any]) -> str:
    content_kind = str(row.get("content_kind") or "OTHER").upper()
    semantic = str(row.get("semantic_hint") or "").upper()
    if semantic in BLOCK_KINDS:
        return semantic
    if content_kind in BLOCK_KINDS:
        return content_kind
    if content_kind == "REVIEW":
        if semantic in {"TEXT_LIKE", "VISUAL_UNKNOWN", "UNKNOWN"}:
            return "OTHER"
        return semantic if semantic in BLOCK_KINDS else "OTHER"
    return "TEXT" if content_kind == "TEXT_LIKE" else "OTHER"


def _review_state(row: dict[str, Any], kind: str) -> str:
    status = str(row.get("status") or "REVIEW_REQUIRED")
    if kind == "TABLE" or status == "DEFERRED":
        return "DEFERRED"
    if status in {"REVIEW_REQUIRED", "FAILED_PRESERVE_INPUT"}:
        return "REVIEW_REQUIRED"
    if kind == "FORMULA" and not str(row.get("latex") or "").strip():
        return "REVIEW_REQUIRED"
    if row.get("warnings") or row.get("review_reasons"):
        return "WARNING"
    return "NONE"


def _asset_for_row(
    document_id: str, row: dict[str, Any], kind: str
) -> dict[str, Any] | None:
    needs_asset = (
        kind in {"IMAGE", "TABLE"}
        or (kind == "FORMULA" and not str(row.get("latex") or "").strip())
        or (
            row.get("binary_artifact_ref")
            and not str(row.get("text") or "").strip()
            and not str(row.get("latex") or "").strip()
        )
    )
    if not needs_asset:
        return None
    source_ref = row.get("binary_artifact_ref")
    source_path = Path(str(source_ref)) if source_ref else None
    content_sha256 = None
    if source_path is not None and source_path.is_file():
        digest = hashlib.sha256()
        with source_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        content_sha256 = digest.hexdigest()
    stable_source_identity = {
        "document_id": document_id,
        "content_id": row.get("content_id"),
        "kind": kind,
        "bbox_pdf_pt": row.get("bbox_pdf_pt"),
        "source_candidate_ids": sorted(row.get("source_candidate_ids", [])),
        "source_region_ids": sorted(row.get("source_region_ids", [])),
        "source_unit_ids": sorted(row.get("source_unit_ids", [])),
    }
    identity_digest = content_sha256 or semantic_sha256(stable_source_identity)
    asset_uid = f"asset-sha256-{identity_digest}"
    suffix = Path(str(source_ref)).suffix.lower() if source_ref else ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
        suffix = ".png"
    return {
        "asset_uid": asset_uid,
        "role": {
            "IMAGE": "PDF_IMAGE",
            "TABLE": "TABLE_DEFERRED_CROP",
            "FORMULA": "FORMULA_REVIEW_CROP",
        }.get(kind, "CONTENT_REVIEW_CROP"),
        "status": "AVAILABLE" if source_ref else "SOURCE_RENDER_REQUIRED",
        "content_sha256": content_sha256,
        "media_suffix": suffix,
        "bbox_pdf_pt": row.get("bbox_pdf_pt"),
        "page_index": int(row.get("page_index", 0)),
        "provenance": {
            "source_content_id": row.get("content_id"),
            "source_ref": source_ref,
            "source_pdf": row.get("provenance", {}).get("source_path"),
            "identity_basis": "CONTENT_SHA256"
            if content_sha256
            else "STABLE_SOURCE_OWNERSHIP_DERIVATIVE",
            "presentation_numbering_frozen": False,
        },
    }


def document_asset_source_ref(asset: dict[str, Any]) -> str | None:
    """Return materialization provenance without making it semantic authority."""

    return asset.get("provenance", {}).get("source_ref") or asset.get("source_ref")


def document_asset_presentation_ref(asset: dict[str, Any]) -> str:
    """Derive a package-local path from stable identity at render time."""

    legacy = asset.get("relative_path")
    if legacy:
        return str(legacy)
    suffix = str(asset.get("media_suffix") or ".png").lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
        suffix = ".png"
    uid = str(asset["asset_uid"])
    digest = uid.removeprefix("asset-sha256-")
    return f"assets/asset-{digest[:24]}{suffix}"


def _block_from_content(
    document_id: str,
    row: dict[str, Any],
    page_record: dict[str, Any],
    *,
    page_furniture_policy: str = 'preserve',
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    kind = _kind(row)
    source_content_ids = [str(row["content_id"])]
    source_region_ids = sorted(
        {str(value) for value in row.get("source_region_ids", [])}
    )
    source_unit_ids = sorted({str(value) for value in row.get("source_unit_ids", [])})
    bbox = row.get("bbox_pdf_pt")
    geometry = page_record.get("geometry", {})
    width = float(geometry.get("width_pt") or 0.0)
    height = float(geometry.get("height_pt") or 0.0)
    asset = _asset_for_row(document_id, row, kind)
    review_state = _review_state(row, kind)
    content = {
        "text": row.get("text"),
        "latex": row.get("latex"),
        "asset_uid": asset["asset_uid"] if asset else None,
        "source_status": row.get("status"),
    }
    if row.get('figure_labels') is not None:
        content['figure_labels'] = copy.deepcopy(row['figure_labels'])
    ordering_projection = row.get("provenance", {}).get("ordering_atom_projection", {})
    atom_grouping_key = (
        str(ordering_projection.get("atom_id"))
        if ordering_projection.get("origin_type")
        in {"NATIVE_LINE", "OCR_RESOLVED_LINE"}
        else None
    )
    block = {
        "schema": BLOCK_SCHEMA,
        "node_id": stable_node_id(
            document_id,
            source_content_ids=source_content_ids,
            source_region_ids=source_region_ids,
            source_unit_ids=source_unit_ids,
            kind=kind,
            grouping_key=atom_grouping_key,
        ),
        "page_index": int(row["page_index"]),
        "kind": kind,
        "subtype": str(row.get("semantic_hint") or "BODY"),
        "source_content_ids": source_content_ids,
        "source_candidate_ids": sorted(
            {str(value) for value in row.get("source_candidate_ids", [])}
        ),
        "source_region_ids": source_region_ids,
        "source_unit_ids": source_unit_ids,
        "bbox_pdf_pt": [float(value) for value in bbox] if _valid_bbox(bbox) else None,
        "bbox_normalized": _normalized_bbox(bbox, width, height),
        "order_key": {},
        "content": content,
        "review_state": review_state,
        "visibility": (
            "SUPPRESSED_FROM_MAIN_BODY"
            if (kind in PAGE_FURNITURE_KINDS and page_furniture_policy == 'suppress')
            or row.get("quality_status") in {"MASKED_EMPTY_TEXT_ROUTE", "MASKED_EMPTY_GRAPHICS_ROUTE"}
            or (row.get("quality_status") == "VERIFIED_BLANK_SOURCE_PAGE"
                and row.get("provenance", {}).get("page_scope_review", {}).get("all_pixels_white") is True)
            else "VISIBLE"
        ),
        "relations": {},
        "provenance": {
            "source_content_schema": row.get("schema"),
            "route_id": row.get("route_id"),
            "source_status": row.get("status"),
            "quality_status": row.get("quality_status"),
            "review_reasons": list(row.get("review_reasons", [])),
            "warnings": list(row.get("warnings", [])),
            "source_path": page_record.get("source_path"),
            "route_provenance": copy.deepcopy(row.get("provenance", {})),
            "formula_development_evidence_canonical": False,
        },
    }
    return block, asset


def _ordering_atom_id(
    *,
    parent_content_id: str,
    page_index: int,
    kind: str,
    origin_type: str,
    source_identity: Mapping[str, Any],
) -> str:
    identity = {
        "contract_version": ORDERING_ATOM_CONTRACT_VERSION,
        "parent_content_id": parent_content_id,
        "page_index": page_index,
        "kind": kind,
        "origin_type": origin_type,
        "source_identity": copy.deepcopy(dict(source_identity)),
    }
    return f"ordering-atom-{semantic_sha256(identity)[:24]}"


def _clipped_source_bbox(
    value: Any, *, page_width: float, page_height: float
) -> list[float] | None:
    if not _valid_bbox(value):
        return None
    bbox = [float(item) for item in value]
    clipped = [
        round(max(0.0, min(page_width, bbox[0])), 6),
        round(max(0.0, min(page_height, bbox[1])), 6),
        round(max(0.0, min(page_width, bbox[2])), 6),
        round(max(0.0, min(page_height, bbox[3])), 6),
    ]
    return clipped if _valid_bbox(clipped) else None


def _native_source_unit_id(
    line: Mapping[str, Any], *, page_width: float, page_height: float
) -> str | None:
    bbox = _clipped_source_bbox(
        line.get("bbox_pdf_pt"), page_width=page_width, page_height=page_height
    )
    if bbox is None:
        return None
    source_line_id = str(line.get("evidence_id") or "")
    if not source_line_id:
        return None
    source_identity = {
        "source_kind": "NATIVE_TEXT_LINE",
        "bbox_pdf_pt": bbox,
        "source_block_id": str(line.get("source_block_id") or ""),
        "source_line_id": source_line_id,
        "source_span_ids": sorted(
            str(value) for value in line.get("source_line_span_ids", [source_line_id])
        ),
        "native_char_count": max(0, int(line.get("native_char_count") or 0)),
    }
    return f"source-unit-{semantic_sha256(source_identity)[:16]}"


def _atomized_content_row(
    row: Mapping[str, Any],
    *,
    atom: OrderingAtom,
) -> dict[str, Any]:
    projected = copy.deepcopy(dict(row))
    if atom.kind != _kind(projected):
        projected["semantic_hint"] = atom.kind
    projected["bbox_pdf_pt"] = list(atom.bbox_pdf_pt)
    projected["text"] = atom.payload.get("text")
    projected["latex"] = atom.payload.get("latex")
    projected["binary_artifact_ref"] = atom.payload.get("binary_artifact_ref")
    projected["source_candidate_ids"] = list(atom.source_candidate_ids)
    projected["source_region_ids"] = list(atom.source_region_ids)
    projected["source_unit_ids"] = list(atom.source_unit_ids)
    projected.setdefault("provenance", {})["ordering_atom_projection"] = {
        "contract_version": ORDERING_ATOM_CONTRACT_VERSION,
        "atom_id": atom.atom_id,
        "parent_content_id": atom.parent_content_id,
        "origin_type": atom.origin_type,
        **copy.deepcopy(atom.provenance),
    }
    return projected


def _coarse_atom(
    row: Mapping[str, Any],
    *,
    origin_type: str,
    unresolved_reasons: Iterable[str] = (),
) -> tuple[dict[str, Any], OrderingAtom]:
    bbox = row.get("bbox_pdf_pt")
    if not _valid_bbox(bbox):
        raise ValueError(f"ORDERING_ATOM_GEOMETRY_INVALID:{row.get('content_id')}")
    parent_content_id = str(row["content_id"])
    kind = _kind(dict(row))
    source_identity = {
        "parent_content_id": parent_content_id,
        "source_candidate_ids": sorted(
            str(v) for v in row.get("source_candidate_ids", [])
        ),
        "source_region_ids": sorted(str(v) for v in row.get("source_region_ids", [])),
        "source_unit_ids": sorted(str(v) for v in row.get("source_unit_ids", [])),
    }
    atom = OrderingAtom(
        atom_id=_ordering_atom_id(
            parent_content_id=parent_content_id,
            page_index=int(row["page_index"]),
            kind=kind,
            origin_type=origin_type,
            source_identity=source_identity,
        ),
        parent_content_id=parent_content_id,
        page_index=int(row["page_index"]),
        kind=kind,
        bbox_pdf_pt=tuple(float(value) for value in bbox),
        payload={
            "text": row.get("text"),
            "latex": row.get("latex"),
            "binary_artifact_ref": row.get("binary_artifact_ref"),
        },
        source_candidate_ids=tuple(
            sorted(str(v) for v in row.get("source_candidate_ids", []))
        ),
        source_region_ids=tuple(
            sorted(str(v) for v in row.get("source_region_ids", []))
        ),
        source_unit_ids=tuple(sorted(str(v) for v in row.get("source_unit_ids", []))),
        provenance={
            "source_identity": source_identity,
            "atomicity_status": (
                "ATOMICITY_UNRESOLVED" if unresolved_reasons else "ATOMIC"
            ),
            "unresolved_reasons": sorted({str(v) for v in unresolved_reasons}),
        },
        origin_type=origin_type,
    )
    return _atomized_content_row(row, atom=atom), atom


def _native_line_kind(row, text, bbox, *, page_height):
    """A merged marginalia route cannot give its role to unrelated body lines."""
    kind = _kind(dict(row))
    if kind not in PAGE_FURNITURE_KINDS:
        return kind

    def at_edge(box):
        return (page_height > 0 and _valid_bbox(box)
                and (box[3] <= page_height * 0.1 or box[1] >= page_height * 0.9))

    if kind == "PAGE_NUMBER":
        compact = re.sub(r"\s+", "", text)
        is_folio = re.fullmatch(
            r"(?:第)?[0-9０-９IVXLCDMivxlcdm一二三四五六七八九十百零〇]+(?:页)?",
            compact,
        )
        return kind if is_folio and at_edge(bbox) else "TEXT"
    # A tall parent may merge a real header/footer with body prose. Retain its
    # lines, including short trailing body lines that happen to lie at the edge.
    return kind if at_edge(row.get("bbox_pdf_pt")) and at_edge(bbox) else "TEXT"


def _native_line_atoms(
    row: Mapping[str, Any],
    *,
    page_evidence: Mapping[str, Any],
    page_width: float,
    page_height: float,
) -> tuple[list[dict[str, Any]], list[OrderingAtom], list[str]]:
    provenance = row.get("provenance", {})
    source_line_ids = [str(value) for value in provenance.get("source_line_ids", [])]
    stored_lines = [
        str(value).replace("\r\n", "\n").replace("\r", "\n")
        for value in provenance.get("normalized_source_lines", [])
    ]
    if not source_line_ids:
        return [], [], ["NATIVE_SOURCE_LINE_IDS_MISSING"]
    if len(source_line_ids) != len(stored_lines):
        return [], [], ["NATIVE_SOURCE_LINE_PAYLOAD_CARDINALITY_MISMATCH"]
    evidence_by_id = {
        str(line.get("evidence_id")): line
        for line in page_evidence.get("native_text", [])
        if line.get("evidence_id")
    }
    missing = [line_id for line_id in source_line_ids if line_id not in evidence_by_id]
    if missing:
        return [], [], ["NATIVE_SOURCE_LINE_EVIDENCE_MISSING"]
    parent_units = {str(value) for value in row.get("source_unit_ids", [])}
    excluded_units = set()
    excluded_line_ownership = []
    for excluded_line in page_evidence.get("fully_excluded_native_lines", []):
        original = excluded_line["source_line"]
        traces = excluded_line.get("ownership", [])
        span_ids = {str(span["span_id"]) for span in original.get("spans", [])}
        excluded_span_ids = {
            str(trace["source_span_id"])
            for trace in traces
            if trace.get("excluded_by_non_text_owner") and trace.get("owner_content_id")
        }
        if not span_ids or not span_ids.issubset(excluded_span_ids):
            continue
        candidates = {
            _native_source_unit_id(
                original, page_width=page_width, page_height=page_height
            ),
            f"native-evidence:{original['evidence_id']}",
        }
        owned_units = parent_units.intersection(candidates)
        if owned_units:
            excluded_units.update(owned_units)
            excluded_line_ownership.append(excluded_line)
    projected_rows: list[dict[str, Any]] = []
    atoms: list[OrderingAtom] = []
    recovered_payload: list[str] = []
    consumed_parent_units: set[str] = set()
    for source_line_id, stored_text in zip(source_line_ids, stored_lines, strict=True):
        line = evidence_by_id[source_line_id]
        line_text = (
            str(line.get("text") or "").replace("\r\n", "\n").replace("\r", "\n")
        )
        if line_text != stored_text:
            return [], [], ["NATIVE_SOURCE_LINE_PAYLOAD_DRIFT"]
        bbox = _clipped_source_bbox(
            line.get("bbox_pdf_pt"),
            page_width=page_width,
            page_height=page_height,
        )
        if bbox is None:
            return [], [], ["NATIVE_SOURCE_LINE_GEOMETRY_INVALID"]
        source_unit_id = _native_source_unit_id(
            line, page_width=page_width, page_height=page_height
        )
        source_unit_candidates = {
            value
            for value in (
                source_unit_id,
                f"native-evidence:{source_line_id}",
            )
            if value
        }
        atom_source_units = tuple(
            sorted(
                parent_units.intersection(source_unit_candidates)
                | (excluded_units if not atoms else set())
            )
        )
        consumed_parent_units.update(atom_source_units)
        source_identity = {
            "source_line_id": source_line_id,
            "source_line_bbox_pdf_pt": bbox,
            "fully_excluded_native_line_ownership": excluded_line_ownership
            if not atoms
            else [],
            "source_block_id": str(line.get("source_block_id") or ""),
            "source_span_ids": sorted(
                str(value) for value in line.get("source_line_span_ids", [])
            ),
            "source_unit_id": source_unit_id,
            "bbox_pdf_pt": bbox,
        }
        fragments = line.get("retained_text_fragments") or [
            {
                "text": stored_text,
                "bbox_pdf_pt": bbox,
                "source_span_ids": source_identity["source_span_ids"],
            }
        ]
        if "".join(fragment["text"] for fragment in fragments) != stored_text:
            return [], [], ["NATIVE_FRAGMENT_PAYLOAD_CONSERVATION_FAILURE"]
        for fragment_index, fragment in enumerate(fragments):
            fragment_bbox = _clipped_source_bbox(
                fragment["bbox_pdf_pt"], page_width=page_width, page_height=page_height
            )
            if fragment_bbox is None:
                return [], [], ["NATIVE_FRAGMENT_GEOMETRY_INVALID"]
            source_identity = {
                **source_identity,
                "fragment_index": fragment_index,
                "fragment_count": len(fragments),
                "source_span_ids": fragment["source_span_ids"],
                "bbox_pdf_pt": fragment_bbox,
            }
            line_kind = _native_line_kind(
                row, fragment["text"], fragment_bbox, page_height=page_height
            )
            atom = OrderingAtom(
                atom_id=_ordering_atom_id(
                    parent_content_id=str(row["content_id"]),
                    page_index=int(row["page_index"]),
                    kind=line_kind,
                    origin_type="NATIVE_LINE",
                    source_identity=source_identity,
                ),
                parent_content_id=str(row["content_id"]),
                page_index=int(row["page_index"]),
                kind=line_kind,
                bbox_pdf_pt=tuple(fragment_bbox),
                payload={
                    "text": fragment["text"],
                    "latex": None,
                    "binary_artifact_ref": None,
                },
                source_candidate_ids=tuple(
                    sorted(str(v) for v in row.get("source_candidate_ids", []))
                ),
                source_region_ids=tuple(
                    sorted(str(v) for v in row.get("source_region_ids", []))
                ),
                source_unit_ids=atom_source_units if fragment_index == 0 else (),
                provenance={
                    "source_identity": source_identity,
                    "atomicity_status": "RECOVERED",
                },
                origin_type="NATIVE_LINE",
            )
            atoms.append(atom)
            projected_rows.append(_atomized_content_row(row, atom=atom))
        recovered_payload.append(stored_text)
    if "\n".join(recovered_payload) != str(row.get("text") or ""):
        return [], [], ["NATIVE_PARENT_PAYLOAD_CONSERVATION_FAILURE"]
    if consumed_parent_units != parent_units:
        return [], [], ["NATIVE_SOURCE_UNIT_OWNERSHIP_UNACCOUNTED"]
    return projected_rows, atoms, []


def _ocr_line_atoms(
    row: Mapping[str, Any],
    *,
    page_width: float,
    page_height: float,
) -> tuple[list[dict[str, Any]], list[OrderingAtom], list[str]]:
    provenance = row.get("provenance", {})
    architecture = provenance.get("three_model_text_evidence")
    if not isinstance(architecture, Mapping):
        return [], [], ["OCR_RESOLUTION_PROVENANCE_MISSING"]
    adapter = str(provenance.get("route_decision", {}).get("adapter") or "")
    units: list[Mapping[str, Any]]
    if adapter == "PAGE_VISUAL_TEXT_RECOVERY" or (
        adapter == "OCR_TEXT_REGION" and "bounded_units" in architecture
    ):
        bounded = architecture.get("bounded_units")
        if not isinstance(bounded, list) or not bounded:
            return [], [], ["OCR_BOUNDED_UNITS_MISSING"]
        units = [unit for unit in bounded if isinstance(unit, Mapping)]
    elif adapter == "OCR_TEXT_REGION":
        if len(provenance.get("ocr_lines", [])) != 1:
            return [], [], ["OCR_REGION_PER_LINE_GEOMETRY_UNAVAILABLE"]
        units = [architecture]
    else:
        return [], [], ["OCR_ADAPTER_UNSUPPORTED_FOR_PROJECTION"]

    projected_rows: list[dict[str, Any]] = []
    atoms: list[OrderingAtom] = []
    payloads: list[str] = []
    parent_source_units = {str(value) for value in row.get("source_unit_ids", [])}
    resolved_units = []
    empty_units = []
    for unit in units:
        resolver = unit.get("resolver")
        request = unit.get("request")
        if not isinstance(resolver, Mapping) or not isinstance(request, Mapping):
            return [], [], ["OCR_RESOLVED_UNIT_CONTRACT_INVALID"]
        text = str(resolver.get("selected_text") or "")
        bbox = _clipped_source_bbox(
            request.get("bbox"), page_width=page_width, page_height=page_height
        )
        unit_id = str(request.get("region_id") or "")
        if not unit_id or bbox is None:
            return [], [], ["OCR_RESOLVED_UNIT_TEXT_ID_OR_GEOMETRY_MISSING"]
        source_identity = {
            "unit_id": unit_id,
            "source_crop_sha256": request.get("crop_sha256"),
            "resolution_status": resolver.get("resolution_status"),
            "bbox_pdf_pt": bbox,
        }
        if not text:
            empty_units.append(source_identity)
            continue
        resolved_units.append((text, bbox, unit_id, source_identity))
    if not resolved_units:
        return [], [], ["OCR_RESOLVED_UNITS_ALL_EMPTY"]
    empty_source_units = {unit['unit_id'] for unit in empty_units} & parent_source_units
    for text, bbox, unit_id, source_identity in resolved_units:
        if empty_units and not atoms:
            source_identity = {**source_identity, 'empty_output_units': empty_units}
        owned_units = {unit_id} if unit_id in parent_source_units else set()
        if not atoms:
            owned_units |= empty_source_units
        atom = OrderingAtom(
            atom_id=_ordering_atom_id(
                parent_content_id=str(row["content_id"]),
                page_index=int(row["page_index"]),
                kind=_kind(dict(row)),
                origin_type="OCR_RESOLVED_LINE",
                source_identity=source_identity,
            ),
            parent_content_id=str(row["content_id"]),
            page_index=int(row["page_index"]),
            kind=_kind(dict(row)),
            bbox_pdf_pt=tuple(bbox),
            payload={"text": text, "latex": None, "binary_artifact_ref": None},
            source_candidate_ids=tuple(
                sorted(str(v) for v in row.get("source_candidate_ids", []))
            ),
            source_region_ids=tuple(
                sorted(str(v) for v in row.get("source_region_ids", []))
            ),
            source_unit_ids=tuple(sorted(owned_units)),
            provenance={
                "source_identity": source_identity,
                "atomicity_status": "RECOVERED",
            },
            origin_type="OCR_RESOLVED_LINE",
        )
        atoms.append(atom)
        projected_rows.append(_atomized_content_row(row, atom=atom))
        payloads.append(text)
    if "\n".join(payloads) != str(row.get("text") or ""):
        return [], [], ["OCR_PARENT_PAYLOAD_CONSERVATION_FAILURE"]
    return projected_rows, atoms, (["OCR_EMPTY_OUTPUT_UNIT_EVIDENCE_RETAINED"] if empty_units else [])


def project_region_content_to_ordering_atoms(
    *,
    document_id: str,
    page_records: Iterable[dict[str, Any]],
    region_content: Iterable[dict[str, Any]],
    source_evidence: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project ContentIR rows into deterministic, transient spatial ordering atoms."""

    records = {int(row["page_index"]): row for row in page_records}
    evidence = source_evidence or {}
    projected: list[dict[str, Any]] = []
    atoms: list[OrderingAtom] = []
    parents: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    input_source_units: list[tuple[int, str]] = []
    for original in region_content:
        row = copy.deepcopy(original)
        page_index = int(row["page_index"])
        if page_index not in records:
            raise ValueError(f"Content references unknown page {page_index}")
        input_source_units.extend(
            (page_index, str(value)) for value in row.get("source_unit_ids", [])
        )
        kind = _kind(row)
        adapter = str(
            row.get("provenance", {}).get("route_decision", {}).get("adapter") or ""
        )
        current_rows: list[dict[str, Any]] = []
        current_atoms: list[OrderingAtom] = []
        reasons: list[str] = []
        origin_type = {
            "FORMULA": "ATOMIC_FORMULA",
            "TABLE": "ATOMIC_TABLE",
            "IMAGE": "ATOMIC_IMAGE",
        }.get(kind)
        text_like = kind in {"TEXT", "CAPTION", "OTHER"}
        page_evidence = evidence.get((str(document_id), page_index))
        if (text_like or kind in PAGE_FURNITURE_KINDS) and adapter == "NATIVE_TEXT_BRIDGE" and page_evidence is not None:
            geometry = records[page_index].get("geometry", {})
            current_rows, current_atoms, reasons = _native_line_atoms(
                row,
                page_evidence=page_evidence,
                page_width=float(geometry.get("width_pt") or 0.0),
                page_height=float(geometry.get("height_pt") or 0.0),
            )
        elif text_like and adapter in {"OCR_TEXT_REGION", "PAGE_VISUAL_TEXT_RECOVERY"}:
            geometry = records[page_index].get("geometry", {})
            current_rows, current_atoms, reasons = _ocr_line_atoms(
                row,
                page_width=float(geometry.get("width_pt") or 0.0),
                page_height=float(geometry.get("height_pt") or 0.0),
            )
        if not current_rows:
            fallback_origin = origin_type or "COARSE_FALLBACK"
            if text_like and not reasons:
                reasons = [
                    "ORDERING_EVIDENCE_NOT_PROVIDED"
                    if page_evidence is None and adapter == "NATIVE_TEXT_BRIDGE"
                    else "TEXT_LINE_PROJECTION_NOT_RECOVERABLE"
                ]
            coarse_row, coarse = _coarse_atom(
                row,
                origin_type=fallback_origin,
                unresolved_reasons=reasons
                if fallback_origin == "COARSE_FALLBACK"
                else (),
            )
            current_rows = [coarse_row]
            current_atoms = [coarse]
        projected.extend(current_rows)
        atoms.extend(current_atoms)
        parent_summary = {
            "parent_content_id": str(row["content_id"]),
            "page_index": page_index,
            "kind": kind,
            "adapter": adapter,
            "old_bbox_pdf_pt": copy.deepcopy(row.get("bbox_pdf_pt")),
            "source_unit_count": len(row.get("source_unit_ids", [])),
            "atom_count": len(current_atoms),
            "atom_ids": [atom.atom_id for atom in current_atoms],
            "origin_types": sorted({atom.origin_type for atom in current_atoms}),
            "payload_conserved": (
                "\n".join(str(atom.payload.get("text") or "") for atom in current_atoms)
                == str(row.get("text") or "")
                if text_like
                else True
            ),
            "unresolved_reasons": sorted(set(reasons)),
        }
        parents.append(parent_summary)
        if reasons:
            unresolved.append(parent_summary)

    atom_source_units = [
        (atom.page_index, value)
        for atom in atoms
        for value in atom.source_unit_ids
        if value
    ]

    def serialize_source_unit(value: tuple[int, str]) -> str:
        return f"p{value[0]:06d}:{value[1]}"

    before_counts = Counter(input_source_units)
    duplicate_input_ownership = sorted(
        value for value, count in before_counts.items() if count > 1
    )
    if duplicate_input_ownership:
        raise ValueError(
            "ORDERING_ATOM_DUPLICATE_SOURCE_UNIT_OWNERSHIP:"
            + ",".join(
                serialize_source_unit(value) for value in duplicate_input_ownership
            )
        )
    after_counts = Counter(atom_source_units)
    missing = sorted(value for value in before_counts if after_counts[value] == 0)
    duplicate = sorted(
        value
        for value, count in after_counts.items()
        if count > before_counts.get(value, 0)
    )
    unaccounted = sorted(value for value in after_counts if value not in before_counts)
    diagnostics = {
        "contract_version": ORDERING_ATOM_CONTRACT_VERSION,
        "long_term_ir_schema_created": False,
        "parent_count": len(parents),
        "atom_count": len(atoms),
        "expanded_parent_count": sum(row["atom_count"] > 1 for row in parents),
        "coarse_fallback_count": sum(
            "COARSE_FALLBACK" in row["origin_types"] for row in parents
        ),
        "origin_type_counts": dict(
            sorted(Counter(atom.origin_type for atom in atoms).items())
        ),
        "source_unit_conservation": {
            "before": len(input_source_units),
            "consumed_by_atoms": len(atom_source_units),
            "identity_scope": "PAGE_LOCAL",
            "missing_source_unit_ids": [
                serialize_source_unit(value) for value in missing
            ],
            "duplicate_source_unit_ids": [
                serialize_source_unit(value) for value in duplicate
            ],
            "unaccounted_source_unit_ids": [
                serialize_source_unit(value) for value in unaccounted
            ],
            "gate": "PASS"
            if not missing and not duplicate and not unaccounted
            else "FAIL",
        },
        "payload_conservation_gate": (
            "PASS" if all(row["payload_conserved"] for row in parents) else "FAIL"
        ),
        "unresolved": unresolved,
        "parents": parents,
        "atoms": [atom.to_dict() for atom in atoms],
    }
    return projected, diagnostics


def content_id_from_route(route: dict[str, Any]) -> str:
    identity = {
        "route_id": route["route_id"],
        "schema": "bemarkdown-region-content-ir-v0",
    }
    return f"content-{semantic_sha256(identity)[:20]}"


def preserved_region_content_from_route(
    route: dict[str, Any],
    page_record: dict[str, Any],
    *,
    native_text_evidence: list[dict[str, Any]] | None = None,
    binary_artifact_ref: str | None = None,
) -> dict[str, Any]:
    """Create an honest model-free ContentIR fallback for a route never executed upstream."""

    adapter = str(route["adapter"])
    output_kind = str(route.get("output_kind") or "REVIEW")
    text = None
    status = "REVIEW_REQUIRED"
    review_reasons = ["UPSTREAM_CONTENT_ADAPTER_NOT_EXECUTED_PRESERVE_INPUT"]
    if adapter == "NATIVE_TEXT_BRIDGE":
        selected = set(route.get("provenance", {}).get("native_text_evidence_ids", []))
        rows = [
            row
            for row in native_text_evidence or []
            if str(row.get("evidence_id")) in selected
        ]
        text = "\n".join(
            str(row.get("text", "")).replace("\r\n", "\n").replace("\r", "\n")
            for row in rows
        )
        if text:
            status = "SUCCESS"
            review_reasons = []
    elif adapter == "TABLE_DEFERRED_PRESERVE":
        status = "DEFERRED"
        review_reasons = ["TABLE_ENGINE_NOT_CALLED"]
    elif (
        adapter in {"IMAGE_NATIVE_EXTRACT", "IMAGE_RENDER_CROP"} and binary_artifact_ref
    ):
        status = "SUCCESS"
        review_reasons = []
    return {
        "schema": "bemarkdown-region-content-ir-v0",
        "content_id": content_id_from_route(route),
        "route_id": route["route_id"],
        "document_id": route["document_id"],
        "page_index": int(route["page_index"]),
        "status": status,
        "content_kind": output_kind,
        "semantic_hint": (
            output_kind
            if adapter == "PAGE_VISUAL_TEXT_RECOVERY"
            else route.get("semantic_evidence", [output_kind])[0]
            if route.get("semantic_evidence")
            else output_kind
        ),
        "text": text,
        "latex": None,
        "binary_artifact_ref": binary_artifact_ref,
        "bbox_pdf_pt": route.get("provenance", {}).get("bbox_pdf_pt"),
        "bbox_render_px": None,
        "confidence": None,
        "warnings": [],
        "review_reasons": review_reasons,
        "model_id": None,
        "model_fingerprint": None,
        "quality_status": {
            "SUCCESS": "CONTENT_OK",
            "DEFERRED": "CONTENT_DEFERRED",
        }.get(status, "CONTENT_REVIEW"),
        "quality_metrics": {},
        "conflict_ids": [],
        "conflict_resolution": "NO_CONFLICT",
        "secondary_content_refs": [],
        "duplicate_risk": False,
        "source_candidate_ids": list(route.get("input_candidate_ids", [])),
        "source_region_ids": list(route.get("source_region_ids", [])),
        "source_unit_ids": list(route.get("source_unit_ids", [])),
        "provenance": {
            "source_path": page_record.get("source_path"),
            "route_decision": {
                "adapter": adapter,
                "reason_codes": list(route.get("decision_reason_codes", [])),
                "decision_version": route.get("decision_version"),
            },
            "materialization": "MODEL_FREE_PRESERVE_INPUT_FALLBACK",
            "upstream_adapter_executed": False,
        },
    }


def _horizontal_overlap(first: list[float], second: list[float]) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    denominator = min(first[2] - first[0], second[2] - second[0])
    return overlap / denominator if denominator else 0.0


def _attach_caption_relations(blocks: list[dict[str, Any]], page_height: float) -> None:
    targets = [block for block in blocks if block["kind"] in {"IMAGE", "TABLE"}]
    for caption in (block for block in blocks if block["kind"] == "CAPTION"):
        if not _valid_bbox(caption["bbox_pdf_pt"]):
            caption["relations"]["caption_for"] = None
            caption["relations"]["caption_confidence"] = "UNKNOWN"
            continue
        scored = []
        for target in targets:
            if not _valid_bbox(target["bbox_pdf_pt"]):
                continue
            cb, tb = caption["bbox_pdf_pt"], target["bbox_pdf_pt"]
            vertical_gap = min(abs(cb[1] - tb[3]), abs(tb[1] - cb[3]))
            overlap = _horizontal_overlap(cb, tb)
            if vertical_gap <= max(36.0, page_height * 0.08) and overlap >= 0.25:
                scored.append((vertical_gap, -overlap, target["node_id"]))
        if not scored:
            caption["relations"]["caption_for"] = None
            caption["relations"]["caption_confidence"] = "UNKNOWN"
            continue
        best = min(scored)
        caption["relations"]["caption_for"] = best[2]
        caption["relations"]["caption_confidence"] = (
            "HIGH" if best[0] <= max(18.0, page_height * 0.035) else "MEDIUM"
        )


def _spatial_bbox(
    block: dict[str, Any], *, page_width: float, page_height: float
) -> tuple[float, float, float, float]:
    bbox = block.get("bbox_pdf_pt")
    node_id = str(block.get("node_id") or "")
    kind = str(block.get("kind") or "")
    page_geometry_valid = (
        math.isfinite(page_width)
        and math.isfinite(page_height)
        and page_width > 0
        and page_height > 0
    )
    if (
        not page_geometry_valid
        or not node_id
        or kind not in BLOCK_KINDS
        or not _valid_bbox(bbox)
    ):
        raise ValueError(f"SPATIAL_ORDER_GEOMETRY_INVALID:{node_id or '<missing-id>'}")
    values = tuple(float(value) for value in bbox)
    left, top, right, bottom = values
    # The census serializes page dimensions to 4 decimals, whereas content
    # evidence uses 6. Snap only their combined rounding uncertainty at edges.
    rounding_tolerance = 0.5e-4 + 0.5e-6
    if (
        not all(math.isfinite(value) for value in values)
        or left < -rounding_tolerance
        or top < -rounding_tolerance
        or right > page_width + rounding_tolerance
        or bottom > page_height + rounding_tolerance
    ):
        raise ValueError(f"SPATIAL_ORDER_GEOMETRY_INVALID:{node_id}")
    return (
        max(0.0, left),
        max(0.0, top),
        min(page_width, right),
        min(page_height, bottom),
    )


def _align_inline_formula_order_geometry(blocks, geometry):
    """Use source line height to order small inline glyphs at the text baseline.

    Canonical crop geometry is untouched. Native lines and resolved bounded
    OCR lines provide baselines; unresolved visual regions do not.
    """
    native_lines = {}
    for block in blocks:
        metadata = _atom_projection_metadata(block)
        identity = metadata.get("source_identity", {})
        box = identity.get("source_line_bbox_pdf_pt")
        if metadata.get("origin_type") == "NATIVE_LINE" and _valid_bbox(box):
            text_box = geometry[block["node_id"]]
            native_lines[(str(identity["source_line_id"]), block["node_id"])] = (
                box[0],
                text_box[1],
                box[2],
                text_box[3],
            )
        elif metadata.get("origin_type") == "OCR_RESOLVED_LINE":
            box = identity.get("bbox_pdf_pt")
            if _valid_bbox(box):
                native_lines[(str(identity["unit_id"]), block["node_id"])] = tuple(box)
    for block in blocks:
        if block["kind"] != "FORMULA":
            continue
        box = geometry[block["node_id"]]
        cy = (box[1] + box[3]) / 2
        matches = []
        for line_id, line in native_lines.items():
            overlap = min(box[3], line[3]) - max(box[1], line[1])
            if (
                max(line[0] - box[2], box[0] - line[2], 0)
                <= 1.5 * (line[3] - line[1])
                and overlap >= 0.5 * min(box[3] - box[1], line[3] - line[1])
                and box[3] - box[1] <= 3.0 * (line[3] - line[1])
            ):
                # Temporary document and OCR-unit IDs can change between runs.
                # Resolve equally centered baselines from source geometry first.
                matches.append((abs(cy - (line[1] + line[3]) / 2), -overlap,
                                max(line[0] - box[2], box[0] - line[2], 0),
                                tuple(line), line_id))
        if matches:
            _, _, _, line, line_id = min(matches)
            geometry[block["node_id"]] = (box[0], line[1], box[2], line[3])
            source_block = next(item for item in blocks if item['node_id'] == line_id[1])
            key = ('inline_native_line_id' if _atom_projection_metadata(source_block).get('origin_type') == 'NATIVE_LINE'
                   else 'inline_ocr_line_id')
            block.setdefault("provenance", {})[key] = line_id[0]


def order_page_blocks_spatial_v1(
    blocks: list[dict[str, Any]],
    *,
    page_width: float,
    page_height: float,
    config: SpatialReadingOrderConfig | None = None,
) -> dict[str, Any]:
    """Order one page by fixed-anchor horizontal bands, then left-to-right."""

    config = config or SpatialReadingOrderConfig()
    if not 0 < config.page_height_ratio < 1 or not 0 < config.block_height_ratio <= 1:
        raise ValueError("SPATIAL_ORDER_CONFIG_INVALID")
    geometry = {
        str(block.get("node_id") or ""): _spatial_bbox(
            block, page_width=page_width, page_height=page_height
        )
        for block in blocks
    }
    _align_inline_formula_order_geometry(blocks, geometry)
    candidates = sorted(
        blocks,
        key=lambda block: (
            geometry[str(block["node_id"])][1],
            geometry[str(block["node_id"])][0],
            geometry[str(block["node_id"])][3],
            geometry[str(block["node_id"])][2],
            str(block["node_id"]),
        ),
    )
    bands: list[list[dict[str, Any]]] = []
    while candidates:
        anchor = candidates[0]
        anchor_bbox = geometry[str(anchor["node_id"])]
        anchor_top = anchor_bbox[1]
        reference_height = anchor_bbox[3] - anchor_bbox[1]
        band = [anchor]
        remaining = []
        for block in candidates[1:]:
            bbox = geometry[str(block["node_id"])]
            block_height = bbox[3] - bbox[1]
            tolerance = min(
                page_height * config.page_height_ratio,
                min(block_height, reference_height) * config.block_height_ratio,
            )
            if bbox[1] - anchor_top <= tolerance:
                band.append(block)
            else:
                remaining.append(block)
        bands.append(
            sorted(
                band,
                key=lambda block: (
                    geometry[str(block["node_id"])][0],
                    geometry[str(block["node_id"])][1],
                    geometry[str(block["node_id"])][2],
                    geometry[str(block["node_id"])][3],
                    str(block["kind"]),
                    str(block["node_id"]),
                ),
            )
        )
        candidates = remaining

    ordered = []
    for band_index, band in enumerate(bands):
        for order_in_band, block in enumerate(band):
            left, top, right, bottom = geometry[str(block["node_id"])]
            block["order_key"] = {
                "version": READING_ORDER_VERSION,
                "spatial_order_contract_version": SPATIAL_ORDER_CONTRACT_VERSION,
                "block_id": str(block["node_id"]),
                "type": str(block["kind"]),
                "bbox": [left, top, right, bottom],
                "band_index": band_index,
                "order_in_band": order_in_band,
                "final_page_order_index": len(ordered),
            }
            ordered.append(block)
    blocks[:] = ordered
    return {
        "reading_order_version": READING_ORDER_VERSION,
        "spatial_order_contract_version": SPATIAL_ORDER_CONTRACT_VERSION,
        "spatial_order_config": config.to_dict(),
        "band_count": len(bands),
        "reading_order_risk": "LOW",
        "column_count": 1,
        "wide_block_count": 0,
        "fallback_yx_count": 0,
        "geometry_invalid_count": 0,
        "semantic_reorder_applied": False,
        "column_first_reorder_applied": False,
        "reason_codes": [],
    }


def _order_page(
    blocks: list[dict[str, Any]], page_width: float, page_height: float
) -> dict[str, Any]:
    order = order_page_blocks_spatial_v1(
        blocks,
        page_width=page_width,
        page_height=page_height,
        config=SpatialReadingOrderConfig(),
    )
    from .pdf.native_column_order import order_native_columns

    order["native_columns"] = order_native_columns(
        blocks, page_width, {b["node_id"]: _atom_projection_metadata(b) for b in blocks},
    )
    if order["native_columns"]["applied"]:
        order["column_count"] = 2
        order["column_first_reorder_applied"] = True
    order["native_paragraph_continuity"] = _preserve_native_paragraph_continuity(blocks)
    order["native_index_columns"] = _order_native_index_columns(blocks)
    return order


def _order_native_index_columns(blocks):
    """Read a positively identified index by columns and keep term/page pairs."""
    import statistics

    groups = {}
    for position, block in enumerate(blocks):
        provenance = block.get("provenance", {}).get("route_provenance", {})
        reasons = provenance.get("route_decision", {}).get("reason_codes", [])
        atom = provenance.get("ordering_atom_projection", {})
        if (
            "NATIVE_INDEX_HEADING_AND_ENTRIES_OVERRIDE_LAYOUT_TABLE" in reasons
            and atom.get("origin_type") == "NATIVE_LINE"
            and _valid_bbox(block.get("bbox_pdf_pt"))
        ):
            groups.setdefault(atom.get("parent_content_id"), []).append(
                (position, block)
            )
    diagnostics = []
    for parent, entries in groups.items():
        if len(entries) < 12:
            continue
        ordered_x = sorted(
            (block["bbox_pdf_pt"][0], block["bbox_pdf_pt"][2]) for _, block in entries
        )
        right = ordered_x[0][1]
        gaps = []
        for left, end in ordered_x[1:]:
            if left > right:
                gaps.append((left - right, (left + right) / 2))
            right = max(right, end)
        if not gaps:
            continue
        span = max(pair[1] for pair in ordered_x) - min(pair[0] for pair in ordered_x)
        columns = None
        for gap, split in sorted(gaps, reverse=True):
            if gap < max(18, span * 0.07):
                break
            candidate_columns = [
                [
                    block
                    for _, block in entries
                    if (block["bbox_pdf_pt"][0] < split) == left
                ]
                for left in (True, False)
            ]
            if min(map(len, candidate_columns)) < 4:
                continue
            if any(
                sum(
                    bool(str(block.get("content", {}).get("text") or "").strip())
                    and not str(block.get("content", {}).get("text") or "")
                    .strip()
                    .isdigit()
                    for block in column
                )
                < 3
                for column in candidate_columns
            ):
                continue
            columns = candidate_columns
            break
        if columns is None:
            continue
        output = []
        for column in columns:
            heights = [
                block["bbox_pdf_pt"][3] - block["bbox_pdf_pt"][1] for block in column
            ]
            tolerance = statistics.median(heights) * 0.45
            row = []
            row_y = None
            for block in sorted(
                column,
                key=lambda item: (item["bbox_pdf_pt"][1] + item["bbox_pdf_pt"][3]) / 2,
            ):
                y = (block["bbox_pdf_pt"][1] + block["bbox_pdf_pt"][3]) / 2
                if row and y - row_y > tolerance:
                    output.extend(sorted(row, key=lambda item: item["bbox_pdf_pt"][0]))
                    row = []
                if not row:
                    row_y = y
                row.append(block)
            output.extend(sorted(row, key=lambda item: item["bbox_pdf_pt"][0]))
        for (position, _before), after in zip(entries, output, strict=True):
            blocks[position] = after
        diagnostics.append(
            {
                "parent_content_id": parent,
                "columns": 2,
                "split_x_pt": split,
                "native_line_count": len(entries),
                "payload_conserved": True,
            }
        )
    return {"policy": "native-index-columns-v1", "groups": diagnostics}


def _atom_projection_metadata(block: Mapping[str, Any]) -> Mapping[str, Any]:
    value = (
        block.get("provenance", {})
        .get("route_provenance", {})
        .get("ordering_atom_projection", {})
    )
    return value if isinstance(value, Mapping) else {}


def _preserve_native_paragraph_continuity(
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep source paragraph lines together across a disjoint margin column.

    The same resolved text region and native line evidence are required. PDF
    storage blocks may contain just one line. A formula, image, table or other
    text in the same horizontal lane remains a fence; geometry alone never merges text.
    """

    def source_key(block):
        metadata = _atom_projection_metadata(block)
        identity = metadata.get("source_identity", {})
        if (
            metadata.get("origin_type") != "NATIVE_LINE"
            or not identity.get("source_line_id")
            or not metadata.get("parent_content_id")
        ):
            return None
        return metadata["parent_content_id"], block.get("kind"), block.get("subtype")

    def horizontal_overlap(first, second):
        return max(0.0, min(first[2], second[2]) - max(first[0], second[0]))

    original_ids = [block["node_id"] for block in blocks]
    pending = list(blocks)
    ordered = []
    paragraph_count = 0
    while pending:
        first = pending.pop(0)
        key = source_key(first)
        group = [first]
        index = 0
        while key is not None and index < len(pending):
            candidate = pending[index]
            if source_key(candidate) != key:
                index += 1
                continue
            previous_box = group[-1]["bbox_pdf_pt"]
            next_box = candidate["bbox_pdf_pt"]
            minimum_width = min(
                previous_box[2] - previous_box[0], next_box[2] - next_box[0]
            )
            line_height = max(
                previous_box[3] - previous_box[1], next_box[3] - next_box[1]
            )
            if (
                horizontal_overlap(previous_box, next_box) < minimum_width * 0.5
                or next_box[1] - previous_box[3] > line_height * 1.5
            ):
                break
            lane = [
                min(previous_box[0], next_box[0]),
                previous_box[1],
                max(previous_box[2], next_box[2]),
                next_box[3],
            ]
            if any(
                (
                    horizontal_overlap(lane, other["bbox_pdf_pt"]) > 0
                    and other["bbox_pdf_pt"][1] < lane[3]
                    and other["bbox_pdf_pt"][3] > lane[1]
                ) or (
                    other.get("kind") == "FORMULA"
                    and other.get("provenance", {}).get("inline_native_line_id")
                    == _atom_projection_metadata(group[-1]).get("source_identity", {}).get("source_line_id")
                )
                for other in pending[:index]
            ):
                break
            group.append(pending.pop(index))
        ordered.extend(group)
        paragraph_count += int(len(group) > 1)

    ordered_ids = [block["node_id"] for block in ordered]
    if sorted(ordered_ids) != sorted(original_ids):
        raise RuntimeError("NATIVE_PARAGRAPH_CONTINUITY_CONSERVATION_FAILED")
    moved = sum(
        first != second for first, second in zip(original_ids, ordered_ids, strict=True)
    )
    if moved:
        for index, block in enumerate(ordered):
            block["order_key"]["base_spatial_order_index"] = block["order_key"][
                "final_page_order_index"
            ]
            block["order_key"]["final_page_order_index"] = index
            block["order_key"]["source_paragraph_continuity"] = True
    blocks[:] = ordered
    return {
        "contract": "native-text-region-continuity-v1",
        "paragraph_count": paragraph_count,
        "moved_atom_count": moved,
        "source_atom_conservation": "PASS",
    }


def _can_group_text_atoms(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if first.get("kind") not in {"TEXT", "CAPTION", "OTHER"}:
        return False
    if second.get("kind") != first.get("kind") or second.get("subtype") != first.get(
        "subtype"
    ):
        return False
    first_meta = _atom_projection_metadata(first)
    second_meta = _atom_projection_metadata(second)
    recoverable_origins = {"NATIVE_LINE", "OCR_RESOLVED_LINE"}
    return (
        first_meta.get("origin_type") in recoverable_origins
        and second_meta.get("origin_type") in recoverable_origins
        and first_meta.get("parent_content_id") == second_meta.get("parent_content_id")
    )


def _merge_text_atom_blocks(
    document_id: str, group: list[dict[str, Any]]
) -> dict[str, Any]:
    if not group:
        raise ValueError("POST_ORDER_TEXT_GROUP_EMPTY")
    if len(group) == 1:
        result = copy.deepcopy(group[0])
    else:
        result = copy.deepcopy(group[0])
        bboxes = [block["bbox_pdf_pt"] for block in group]
        result["bbox_pdf_pt"] = [
            min(float(bbox[0]) for bbox in bboxes),
            min(float(bbox[1]) for bbox in bboxes),
            max(float(bbox[2]) for bbox in bboxes),
            max(float(bbox[3]) for bbox in bboxes),
        ]
        result["bbox_normalized"] = None
        result["content"]["text"] = "\n".join(
            str(block.get("content", {}).get("text") or "") for block in group
        )
        result["source_content_ids"] = sorted(
            {value for block in group for value in block["source_content_ids"]}
        )
        result["source_candidate_ids"] = sorted(
            {value for block in group for value in block["source_candidate_ids"]}
        )
        result["source_region_ids"] = sorted(
            {value for block in group for value in block["source_region_ids"]}
        )
        result["source_unit_ids"] = sorted(
            {value for block in group for value in block["source_unit_ids"]}
        )
        review_rank = {"NONE": 0, "WARNING": 1, "REVIEW_REQUIRED": 2, "DEFERRED": 3}
        result["review_state"] = max(
            (str(block["review_state"]) for block in group),
            key=lambda value: review_rank.get(value, 2),
        )
    atom_ids = [str(_atom_projection_metadata(block).get("atom_id")) for block in group]
    result["node_id"] = stable_node_id(
        document_id,
        source_content_ids=result["source_content_ids"],
        source_region_ids=result["source_region_ids"],
        source_unit_ids=result["source_unit_ids"],
        kind=str(result["kind"]),
        grouping_key=semantic_sha256(
            {
                "version": POST_ORDER_GROUPING_VERSION,
                "atom_ids": atom_ids,
            }
        ),
    )
    first_order = copy.deepcopy(group[0].get("order_key", {}))
    first_order.update(
        {
            "block_id": result["node_id"],
            "type": result["kind"],
            "bbox": copy.deepcopy(result["bbox_pdf_pt"]),
            "post_order_grouping_version": POST_ORDER_GROUPING_VERSION,
            "ordered_atom_ids": atom_ids,
        }
    )
    result["order_key"] = first_order
    result["relations"] = {}
    result["provenance"]["ordering_atom_grouping"] = {
        "version": POST_ORDER_GROUPING_VERSION,
        "ordered_atom_ids": atom_ids,
        "atom_count": len(group),
        "ordered_source_line_ids": list(dict.fromkeys(
            _atom_projection_metadata(block).get("source_identity", {}).get("source_line_id")
            for block in group
            if _atom_projection_metadata(block).get("source_identity", {}).get("source_line_id")
        )),
        "parent_content_id": _atom_projection_metadata(group[0]).get(
            "parent_content_id"
        ),
        "crossed_non_text_atom": False,
        "semantic_reorder_applied": False,
    }
    return result


def post_order_text_grouping(
    *, document_id: str, ordered_blocks: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group adjacent recovered text atoms without changing their spatial order."""

    # Group selection is read-only. Each emitted group is deep-copied below,
    # so copying every input block here duplicates the full provenance tree.
    blocks = list(ordered_blocks)
    groups: list[list[dict[str, Any]]] = []
    for block in blocks:
        if groups and _can_group_text_atoms(groups[-1][-1], block):
            groups[-1].append(block)
        else:
            groups.append([block])
    recoverable_origins = {"NATIVE_LINE", "OCR_RESOLVED_LINE"}
    result = [
        _merge_text_atom_blocks(document_id, group)
        if group[0].get("kind") in {"TEXT", "CAPTION", "OTHER"}
        and _atom_projection_metadata(group[0]).get("origin_type")
        in recoverable_origins
        else copy.deepcopy(group[0])
        for group in groups
    ]
    for display_index, block in enumerate(result):
        block["display_index"] = display_index
        block["order_key"]["final_page_order_index"] = display_index
    diagnostics = {
        "version": POST_ORDER_GROUPING_VERSION,
        "input_atom_count": len(blocks),
        "output_group_count": len(result),
        "collapsed_atom_count": len(blocks) - len(result),
        "groups": [
            {
                "node_id": block["node_id"],
                "kind": block["kind"],
                "bbox_pdf_pt": block["bbox_pdf_pt"],
                "ordered_atom_ids": block["order_key"].get("ordered_atom_ids", []),
                "source_content_ids": block["source_content_ids"],
            }
            for block in result
        ],
        "crossed_formula_table_image": False,
        "atom_order_changed": False,
        "gate": "PASS",
    }
    return result, diagnostics


def _ordering_trace_block(block: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _atom_projection_metadata(block)
    text = str(block.get("content", {}).get("text") or "")
    return {
        "atom_id": metadata.get("atom_id"),
        "parent_content_id": metadata.get("parent_content_id"),
        "origin_type": metadata.get("origin_type"),
        "node_id": block.get("node_id"),
        "kind": block.get("kind"),
        "bbox_pdf_pt": copy.deepcopy(block.get("bbox_pdf_pt")),
        "source_unit_ids": copy.deepcopy(block.get("source_unit_ids", [])),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text
        else None,
        "text_preview": text[:160] if text else None,
        "order_key": copy.deepcopy(block.get("order_key", {})),
    }


def assemble_document_ir(
    *,
    document_id: str,
    source: dict[str, Any],
    page_records: list[dict[str, Any]],
    region_content: list[dict[str, Any]],
    development_audit: dict[str, Any] | None = None,
    source_evidence: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    ordering_trace: MutableSequence[dict[str, Any]] | None = None,
    performance_metrics: MutableMapping[str, float] | None = None,
    page_furniture_policy: str = 'preserve',
) -> dict[str, Any]:
    """Assemble authoritative PDF reconstruction state through transient atoms."""
    if page_furniture_policy not in {'preserve', 'suppress'}:
        raise ValueError('PAGE_FURNITURE_POLICY_INVALID')

    pages_by_index = {int(row["page_index"]): row for row in page_records}
    if len(pages_by_index) != len(page_records):
        raise ValueError("Page records must have unique page_index values")
    page_blocks: dict[int, list[dict[str, Any]]] = {
        index: [] for index in pages_by_index
    }
    suppressed: list[dict[str, Any]] = []
    assets: dict[str, dict[str, Any]] = {}
    projection_started = time.perf_counter()
    projected_content, projection = project_region_content_to_ordering_atoms(
        document_id=document_id,
        page_records=page_records,
        region_content=region_content,
        source_evidence=source_evidence,
    )
    if performance_metrics is not None:
        performance_metrics["atomic_projection_wall_seconds"] = (
            performance_metrics.get("atomic_projection_wall_seconds", 0.0)
            + time.perf_counter()
            - projection_started
        )
    for row in projected_content:
        page_index = int(row["page_index"])
        if page_index not in pages_by_index:
            raise ValueError(f"Content references unknown page {page_index}")
        block, asset = _block_from_content(document_id, row, pages_by_index[page_index],
                                           page_furniture_policy=page_furniture_policy)
        if asset:
            assets[asset["asset_uid"]] = asset
        if block["visibility"] == "SUPPRESSED_FROM_MAIN_BODY":
            suppressed.append(block)
        else:
            page_blocks[page_index].append(block)

    pages = []
    ordered_blocks: list[dict[str, Any]] = []
    for page_index in sorted(pages_by_index):
        record = pages_by_index[page_index]
        geometry = record.get("geometry", {})
        width = float(geometry.get("width_pt") or 0.0)
        height = float(geometry.get("height_pt") or 0.0)
        atoms = page_blocks[page_index]
        from .pdf.formula_containment import deduplicate_page_formulas
        from .pdf.image_containment import deduplicate_page_images
        from .pdf.text_containment import deduplicate_page_text

        atoms, duplicate_images = deduplicate_page_images(atoms)
        suppressed.extend(duplicate_images)
        atoms, duplicate_formulas = deduplicate_page_formulas(atoms)
        suppressed.extend(duplicate_formulas)
        atoms, duplicate_text = deduplicate_page_text(atoms)
        suppressed.extend(duplicate_text)
        atoms_before = [_ordering_trace_block(block) for block in atoms]
        spatial_started = time.perf_counter()
        order = _order_page(atoms, width, height)
        if performance_metrics is not None:
            performance_metrics["spatial_sort_wall_seconds"] = (
                performance_metrics.get("spatial_sort_wall_seconds", 0.0)
                + time.perf_counter()
                - spatial_started
            )
        ordered_atom_trace = [_ordering_trace_block(block) for block in atoms]
        grouping_started = time.perf_counter()
        current, grouping = post_order_text_grouping(
            document_id=document_id, ordered_blocks=atoms
        )
        if performance_metrics is not None:
            performance_metrics["post_order_grouping_wall_seconds"] = (
                performance_metrics.get("post_order_grouping_wall_seconds", 0.0)
                + time.perf_counter()
                - grouping_started
            )
        from .pdf.native_margin_notes import attach_native_margin_notes
        order['native_margin_attachment'] = attach_native_margin_notes(current, width)
        if order['native_margin_attachment']['moved_count']:
            order['semantic_reorder_applied'] = True
            order['reason_codes'].append('NATIVE_MARGIN_NOTE_ATTACHED_TO_MAIN_PARAGRAPH')
        _attach_caption_relations(current, height)
        from .pdf.caption_order import attach_side_captions, order_parallel_captioned_figures
        order['parallel_figure_order'] = order_parallel_captioned_figures(current)
        if order['parallel_figure_order']['moved_count']:
            order['semantic_reorder_applied'] = True
            order['reason_codes'].append('PARALLEL_FIGURES_FOLLOW_SOURCE_CAPTIONS')
        order['side_caption_attachment'] = attach_side_captions(current)
        if order['side_caption_attachment']['moved_count']:
            order['semantic_reorder_applied'] = True
            order['reason_codes'].append('SIDE_CAPTION_ATTACHED_TO_SOURCE_FIGURE')
        ordered_blocks.extend(current)
        page_suppressed = sorted(
            (block for block in suppressed if block["page_index"] == page_index),
            key=lambda block: block["node_id"],
        )
        if ordering_trace is not None:
            parent_rows = [
                row
                for row in projection["parents"]
                if int(row["page_index"]) == page_index
            ]
            ordering_trace.append(
                {
                    "contract_version": ORDERING_ATOM_CONTRACT_VERSION,
                    "document_id": document_id,
                    "page_index": page_index,
                    "old_coarse_blocks": parent_rows,
                    "recovered_atoms": atoms_before,
                    "final_atom_order": ordered_atom_trace,
                    "spatial_bands": [
                        {
                            "atom_id": row["atom_id"],
                            "band_index": row["order_key"].get("band_index"),
                            "order_in_band": row["order_key"].get("order_in_band"),
                        }
                        for row in ordered_atom_trace
                    ],
                    "post_order_groups": grouping["groups"],
                    "spatial_order": copy.deepcopy(order),
                    "post_order_grouping": {
                        key: value for key, value in grouping.items() if key != "groups"
                    },
                }
            )
        pages.append(
            {
                "page_index": page_index,
                "width_pt": width,
                "height_pt": height,
                "source_path": record.get("source_path"),
                "source_profile": record.get("source_profile"),
                "visible_node_ids": [block["node_id"] for block in current],
                "suppressed_node_ids": [block["node_id"] for block in page_suppressed],
                **order,
                "ordering_atom_count": len(atoms),
                "post_order_text_group_count": len(current),
            }
        )

    source_content_ids = [
        str(row["content_id"]) for row in region_content if row.get("content_id")
    ]
    candidate_ids = [
        (int(row["page_index"]), str(value))
        for row in region_content
        for value in row.get("source_candidate_ids", [])
    ]
    all_output_blocks = [*ordered_blocks, *suppressed]
    consumed_content_ids = {
        str(value)
        for block in all_output_blocks
        for value in block["source_content_ids"]
    }
    review_items = [
        {
            "node_id": block["node_id"],
            "page_index": int(block["page_index"]),
            "kind": block["kind"],
            "review_state": block["review_state"],
            "asset_uid": block["content"].get("asset_uid"),
            "reason_codes": block["provenance"]["review_reasons"],
        }
        for block in all_output_blocks
        if block["review_state"] != "NONE"
    ]
    accounting = {
        "input_content_nodes": len(source_content_ids),
        "visible_output_nodes": len(ordered_blocks),
        "suppressed_nodes": len(suppressed),
        "grouped_child_nodes": projection["atom_count"] - len(all_output_blocks),
        "transient_ordering_atoms": projection["atom_count"],
        "deferred_nodes": sum(
            block["review_state"] == "DEFERRED" for block in ordered_blocks
        ),
        "review_nodes": len(review_items),
        "unaccounted": len(set(source_content_ids) - consumed_content_ids),
        "duplicate_primary_consumption": sum(
            count - 1 for count in Counter(candidate_ids).values() if count > 1
        ),
    }
    document = {
        "schema_version": DOCUMENT_IR_SCHEMA,
        "document_id": str(document_id),
        "source": copy.deepcopy(source),
        "pages": pages,
        "blocks": ordered_blocks,
        "suppressed_blocks": sorted(
            suppressed, key=lambda block: (block["page_index"], block["node_id"])
        ),
        "assets": [assets[key] for key in sorted(assets)],
        "review_items": sorted(
            review_items, key=lambda row: (row["page_index"], row["node_id"])
        ),
        "accounting": accounting,
        "provenance": {
            "reconstruction_authority": True,
            "page_furniture_policy": page_furniture_policy,
            "reading_order_version": READING_ORDER_VERSION,
            "formula_risk_role": "AUDIT_PRIORITY_ONLY",
            "formula_reference_results_canonical": False,
            "cross_page_paragraph_policy": "NO_AUTO_CROSS_PAGE_PARAGRAPH_MERGE",
            "ordering_atom_projection": {
                key: copy.deepcopy(value)
                for key, value in projection.items()
                if key not in {"parents", "atoms"}
            },
            "post_order_text_grouping_version": POST_ORDER_GROUPING_VERSION,
            "development_audit": copy.deepcopy(development_audit or {}),
        },
    }
    document["audit_priority"] = _audit_priority(document)
    validate_document_ir(document)
    return document


def _audit_priority(document: dict[str, Any]) -> dict[str, Any]:
    risk_weight = {"LOW": 0, "MEDIUM": 2, "HIGH": 4}
    pages = []
    for page in document["pages"]:
        page_index = page["page_index"]
        nodes = [
            block for block in document["blocks"] if block["page_index"] == page_index
        ]
        score = risk_weight[page["reading_order_risk"]]
        reasons = []
        if page["reading_order_risk"] != "LOW":
            reasons.append(f"READING_ORDER_{page['reading_order_risk']}")
        for block in nodes:
            if block["kind"] == "FORMULA" and block["review_state"] != "NONE":
                score += 3
                reasons.append("FORMULA_REVIEW")
            elif block["kind"] == "TABLE":
                score += 3
                reasons.append("TABLE_DEFERRED")
            elif block["kind"] == "OTHER":
                score += 2
                reasons.append("OTHER_UNKNOWN")
            elif block["review_state"] != "NONE":
                score += 1
                reasons.append("CONTENT_REVIEW")
        pages.append(
            {
                "page_index": page_index,
                "score": score,
                "reason_codes": sorted(set(reasons)),
            }
        )
    return {
        "role": "AUDIT_PRIORITY_ONLY",
        "pages": sorted(pages, key=lambda row: (-row["score"], row["page_index"])),
    }


def validate_document_ir(document: dict[str, Any]) -> None:
    if document.get("schema_version") != DOCUMENT_IR_SCHEMA:
        raise ValueError("Unsupported DocumentIR schema")
    pages = {int(page["page_index"]): page for page in document.get("pages", [])}
    all_blocks = [*document.get("blocks", []), *document.get("suppressed_blocks", [])]
    node_ids = [block.get("node_id") for block in all_blocks]
    if any(not value for value in node_ids) or len(node_ids) != len(set(node_ids)):
        raise ValueError("DocumentIR node IDs must be present and unique")
    node_set = set(node_ids)
    asset_set = {asset.get("asset_uid") for asset in document.get("assets", [])}
    if None in asset_set or len(asset_set) != len(document.get("assets", [])):
        raise ValueError("DocumentIR asset UIDs must be present and unique")
    for block in all_blocks:
        if block.get("schema") != BLOCK_SCHEMA:
            raise ValueError("Unsupported DocumentBlockIR schema")
        if block.get("kind") not in BLOCK_KINDS:
            raise ValueError("Invalid DocumentBlockIR kind")
        if int(block.get("page_index", -1)) not in pages:
            raise ValueError("DocumentBlockIR references an invalid page")
        if block.get("bbox_pdf_pt") is not None and not _valid_bbox(
            block["bbox_pdf_pt"]
        ):
            raise ValueError("DocumentBlockIR bbox is invalid")
        if not block.get("source_content_ids") and not block.get("provenance", {}).get(
            "patch_inserted"
        ):
            raise ValueError("DocumentBlockIR requires source provenance")
        asset_uid = block.get("content", {}).get("asset_uid")
        if asset_uid and asset_uid not in asset_set:
            raise ValueError("DocumentBlockIR has a missing asset reference")
        for relation, target in block.get("relations", {}).items():
            if relation.endswith("_confidence") or target is None:
                continue
            if isinstance(target, str) and target not in node_set:
                raise ValueError("DocumentBlockIR relation target does not exist")
    _validate_child_cycles(all_blocks)
    accounting = document.get("accounting", {})
    if accounting.get("unaccounted") != 0:
        raise ValueError("Document reconstruction contains unaccounted content")
    if accounting.get("duplicate_primary_consumption") != 0:
        raise ValueError(
            "Document reconstruction contains duplicate primary consumption"
        )


def _validate_child_cycles(blocks: list[dict[str, Any]]) -> None:
    parent = {
        block["node_id"]: block.get("relations", {}).get("child_of") for block in blocks
    }
    for start in parent:
        seen = set()
        current = start
        while current in parent and parent[current] is not None:
            if current in seen:
                raise ValueError("DocumentBlockIR relation cycle detected")
            seen.add(current)
            current = parent[current]


def _anchor(block: dict[str, Any]) -> str:
    extra = ""
    if block["review_state"] != "NONE":
        extra += f' status="{block["review_state"]}"'
    asset_uid = block.get("content", {}).get("asset_uid")
    if asset_uid:
        extra += f' asset_uid="{asset_uid}"'
    return (
        f'<!-- bemarkdown:block id="{block["node_id"]}" '
        f'page="{block["page_index"]}" kind="{block["kind"]}"{extra} -->'
    )


def _render_block(
    block: dict[str, Any], asset_ref_map: dict[str, str] | None = None
) -> str:
    content = block["content"]
    kind = block["kind"]
    text = str(content.get("text") or "").replace("\r\n", "\n").replace("\r", "\n")
    latex = str(content.get("latex") or "").strip()
    asset_uid = str(content.get("asset_uid") or "")
    asset_ref = (asset_ref_map or {}).get(asset_uid) or content.get("asset_ref")
    if kind == "TITLE":
        body = f"# {text}" if text else "<!-- TITLE_REVIEW_REQUIRED -->"
    elif kind in {"TEXT", "CAPTION", "HEADER_FOOTER", "PAGE_NUMBER"}:
        body = text or (
            f"![文本待确认]({asset_ref})"
            if asset_ref
            else "<!-- TEXT_REVIEW_REQUIRED -->"
        )
    elif kind == "FORMULA":
        body = f"$$\n{latex}\n$$" if latex else f"![公式待确认]({asset_ref})"
    elif kind == "IMAGE":
        from .pdf.figure_labels import figure_alt_text
        body = f"![{figure_alt_text(content)}]({asset_ref})"
    elif kind == "TABLE":
        from .pdf_table_engine import render_table_block

        body = render_table_block(block, asset_ref=asset_ref)
    elif text:
        body = text
    elif asset_ref:
        body = f"![内容待确认]({asset_ref})"
    else:
        body = "<!-- CONTENT_REVIEW_REQUIRED -->"
    return f"{_anchor(block)}\n{body}\n<!-- /bemarkdown:block -->"


def render_draft_markdown(document: dict[str, Any]) -> str:
    validate_document_ir(document)
    asset_ref_map = {
        str(asset["asset_uid"]): document_asset_presentation_ref(asset)
        for asset in document.get("assets", [])
    }
    parts = [
        (
            '<!-- bemarkdown:draft-package contract="bemarkdown-draft-package-v0" '
            'status="development-baseline-not-final" -->'
        )
    ]
    current_page = None
    page_map = {page["page_index"]: page for page in document["pages"]}
    for block in document["blocks"]:
        if block["page_index"] != current_page:
            current_page = block["page_index"]
            risk = page_map[current_page]["reading_order_risk"]
            parts.append(
                f'<!-- bemarkdown:page index="{current_page}" reading_order_risk="{risk}" -->'
            )
        parts.append(_render_block(block, asset_ref_map))
    return "\n\n".join(parts).rstrip() + "\n"


class DraftMarkdownAnchorReader:
    def read(self, markdown: str) -> list[dict[str, Any]]:
        return [
            {
                "node_id": match.group("id"),
                "page_index": int(match.group("page")),
                "kind": match.group("kind"),
                "offset": match.start(),
            }
            for match in _ANCHOR_RE.finditer(markdown)
        ]


def audit_markdown_anchors(document: dict[str, Any], markdown: str) -> dict[str, Any]:
    anchors = DraftMarkdownAnchorReader().read(markdown)
    expected = [block["node_id"] for block in document["blocks"]]
    counts = Counter(anchor["node_id"] for anchor in anchors)
    duplicates = sorted(node_id for node_id, count in counts.items() if count > 1)
    unknown = sorted(set(counts) - set(expected))
    missing = sorted(set(expected) - set(counts))
    coverage = (len(expected) - len(missing)) / len(expected) if expected else 1.0
    return {
        "schema": "bemarkdown-draft-markdown-anchor-audit-v0",
        "renderer_version": DRAFT_RENDERER_VERSION,
        "expected_visible_nodes": len(expected),
        "anchors_found": len(anchors),
        "visible_node_anchor_coverage": coverage,
        "missing_node_ids": missing,
        "duplicate_node_ids": duplicates,
        "unknown_node_ids": unknown,
        "passed": not missing and not duplicates and not unknown,
    }


class DocumentNodeLocator:
    schema = NODE_LOCATOR_SCHEMA

    def __init__(self, document: dict[str, Any]):
        self.document = document
        self._nodes = {block["node_id"]: block for block in document["blocks"]}

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)

    def get_page_nodes(self, page_index: int) -> list[dict[str, Any]]:
        return [
            block
            for block in self.document["blocks"]
            if block["page_index"] == page_index
        ]

    def get_neighboring_nodes(self, node_id: str) -> dict[str, dict[str, Any] | None]:
        positions = {
            block["node_id"]: index
            for index, block in enumerate(self.document["blocks"])
        }
        if node_id not in positions:
            raise KeyError(node_id)
        index = positions[node_id]
        return {
            "previous": self.document["blocks"][index - 1] if index else None,
            "next": (
                self.document["blocks"][index + 1]
                if index + 1 < len(self.document["blocks"])
                else None
            ),
        }

    def get_source_evidence_refs(self, node_id: str) -> dict[str, Any]:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(node_id)
        return {
            "source_content_ids": node["source_content_ids"],
            "source_region_ids": node["source_region_ids"],
            "source_unit_ids": node["source_unit_ids"],
            "source_path": node["provenance"].get("source_path"),
            "bbox_pdf_pt": node["bbox_pdf_pt"],
            "asset_uid": node["content"].get("asset_uid"),
        }


class OutputAuditPatchEngine:
    def apply(
        self, document: dict[str, Any], operations: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        patched = copy.deepcopy(document)
        before_sha = semantic_sha256(patched)
        audit_rows = []
        for index, raw in enumerate(operations):
            operation = copy.deepcopy(raw)
            op = str(operation.get("op"))
            if op not in PATCH_OPERATIONS:
                raise ValueError(f"Unsupported patch operation: {op}")
            operation_id = f"patch-op-{semantic_sha256({'index': index, 'operation': operation})[:20]}"
            created = self._apply_one(patched, operation, operation_id)
            audit_rows.append(
                {
                    "operation_id": operation_id,
                    **operation,
                    "created_node_ids": created,
                    "provenance": {
                        "schema": PATCH_SCHEMA,
                        "stable_id_targeting": True,
                    },
                }
            )
        self._refresh_page_membership(patched)
        after_sha = semantic_sha256(patched)
        return patched, {
            "schema": "bemarkdown-output-audit-patch-log-v0",
            "patch_contract": PATCH_SCHEMA,
            "before_sha256": before_sha,
            "after_sha256": after_sha,
            "operations": audit_rows,
        }

    def _node_index(self, document: dict[str, Any], node_id: str) -> int:
        for index, block in enumerate(document["blocks"]):
            if block["node_id"] == node_id:
                return index
        raise KeyError(node_id)

    def _derived_node_id(
        self, document: dict[str, Any], operation_id: str, ordinal: int
    ) -> str:
        return (
            "doc-node-"
            + semantic_sha256(
                {
                    "document_id": document["document_id"],
                    "operation_id": operation_id,
                    "ordinal": ordinal,
                    "schema": BLOCK_SCHEMA,
                }
            )[:24]
        )

    def _apply_one(
        self, document: dict[str, Any], operation: dict[str, Any], operation_id: str
    ) -> list[str]:
        op = operation["op"]
        if op in {
            "REPLACE_TEXT",
            "REPLACE_FORMULA",
            "CHANGE_HEADING_LEVEL",
            "FIX_ASSET_REFERENCE",
            "REPLACE_TABLE",
            "CHANGE_BLOCK_KIND",
            "UPDATE_CAPTION_RELATION",
        }:
            index = self._node_index(document, operation["target_node_id"])
            block = document["blocks"][index]
            if op == "REPLACE_TEXT":
                block["content"]["text"] = str(operation["new_text"])
            elif op == "REPLACE_FORMULA":
                block["content"]["latex"] = str(operation["new_latex"])
                block["review_state"] = "NONE"
            elif op == "CHANGE_HEADING_LEVEL":
                block["content"]["heading_level"] = int(operation["heading_level"])
            elif op == "FIX_ASSET_REFERENCE":
                block["content"]["asset_uid"] = operation["asset_uid"]
                block["content"]["asset_ref"] = operation["asset_ref"]
            elif op == "REPLACE_TABLE":
                block["content"].update(copy.deepcopy(operation["content"]))
                block["review_state"] = "NONE"
            elif op == "CHANGE_BLOCK_KIND":
                new_kind = str(operation["new_kind"])
                if new_kind not in BLOCK_KINDS:
                    raise ValueError("CHANGE_BLOCK_KIND kind is invalid")
                block["kind"] = new_kind
                block["subtype"] = str(
                    operation.get("new_subtype", "AUDIT_KIND_CORRECTED")
                )
                if operation.get("heading_level") is not None:
                    block["content"]["heading_level"] = int(operation["heading_level"])
            else:
                asset_node_id = operation.get("asset_node_id")
                relations = block.setdefault("relations", {})
                if asset_node_id is None:
                    relations.pop("caption_for", None)
                    relations.pop("caption_for_confidence", None)
                else:
                    asset_node_id = str(asset_node_id)
                    self._node_index(document, asset_node_id)
                    relations["caption_for"] = asset_node_id
                    confidence = operation.get("relation_confidence")
                    if confidence is not None:
                        relations["caption_for_confidence"] = confidence
            block["provenance"].setdefault("patch_operations", []).append(operation_id)
            return []
        if op == "MOVE_BLOCK":
            index = self._node_index(document, operation["target_node_id"])
            block = document["blocks"].pop(index)
            if operation.get("before_node_id"):
                destination = self._node_index(document, operation["before_node_id"])
            elif operation.get("after_node_id"):
                destination = self._node_index(document, operation["after_node_id"]) + 1
            else:
                raise ValueError("MOVE_BLOCK requires before_node_id or after_node_id")
            document["blocks"].insert(destination, block)
            block["provenance"].setdefault("patch_operations", []).append(operation_id)
            return []
        if op == "DELETE_DUPLICATE":
            index = self._node_index(document, operation["target_node_id"])
            block = document["blocks"].pop(index)
            block["visibility"] = "SUPPRESSED_DUPLICATE"
            block["provenance"].setdefault("patch_operations", []).append(operation_id)
            document["suppressed_blocks"].append(block)
            return []
        if op == "INSERT_BLOCK":
            if operation.get("after_node_id"):
                index = self._node_index(document, operation["after_node_id"]) + 1
                page_index = document["blocks"][index - 1]["page_index"]
            elif operation.get("before_node_id"):
                index = self._node_index(document, operation["before_node_id"])
                page_index = document["blocks"][index]["page_index"]
            else:
                index = len(document["blocks"])
                page_index = int(operation["page_index"])
            node_id = self._derived_node_id(document, operation_id, 0)
            block = self._patch_block(
                node_id=node_id,
                page_index=page_index,
                kind=str(operation.get("kind", "OTHER")),
                content=copy.deepcopy(operation.get("content", {})),
                operation_id=operation_id,
            )
            if operation.get("source_bbox_pdf_pt") is not None:
                block["bbox_pdf_pt"] = copy.deepcopy(operation["source_bbox_pdf_pt"])
            if operation.get("relations") is not None:
                block["relations"] = copy.deepcopy(operation["relations"])
            document["blocks"].insert(index, block)
            return [node_id]
        if op == "SPLIT_BLOCK":
            index = self._node_index(document, operation["target_node_id"])
            original = document["blocks"].pop(index)
            parts = list(operation.get("parts", []))
            if len(parts) < 2:
                raise ValueError("SPLIT_BLOCK requires at least two parts")
            created = []
            replacements = []
            for ordinal, part in enumerate(parts):
                node_id = self._derived_node_id(document, operation_id, ordinal)
                block = copy.deepcopy(original)
                block["node_id"] = node_id
                if isinstance(part, dict):
                    kind = str(part.get("kind", block["kind"]))
                    if kind not in BLOCK_KINDS:
                        raise ValueError("SPLIT_BLOCK part kind is invalid")
                    block["kind"] = kind
                    content = part.get("content")
                    if isinstance(content, dict):
                        block["content"].update(copy.deepcopy(content))
                    elif content is not None:
                        field = "latex" if kind == "FORMULA" else "text"
                        block["content"][field] = str(content)
                    if part.get("latex") is not None:
                        block["content"]["latex"] = str(part["latex"])
                    if part.get("text") is not None:
                        block["content"]["text"] = str(part["text"])
                    if part.get("relations") is not None:
                        block["relations"] = copy.deepcopy(part["relations"])
                    if part.get("bbox_pdf_pt") is not None:
                        block["bbox_pdf_pt"] = copy.deepcopy(part["bbox_pdf_pt"])
                    if part.get("caption_for") is not None:
                        block.setdefault("relations", {})["caption_for"] = str(
                            part["caption_for"]
                        )
                    if part.get("segment_key") is not None:
                        block["provenance"]["patch_segment_key"] = str(
                            part["segment_key"]
                        )
                elif block["kind"] == "FORMULA":
                    block["content"]["latex"] = str(part)
                else:
                    block["content"]["text"] = str(part)
                block["provenance"].setdefault("patch_operations", []).append(
                    operation_id
                )
                block["provenance"]["split_from"] = original["node_id"]
                replacements.append(block)
                created.append(node_id)
            document["blocks"][index:index] = replacements
            original["visibility"] = "SUPPRESSED_SPLIT_SOURCE"
            document["suppressed_blocks"].append(original)
            return created
        if op == "MERGE_BLOCK":
            target_ids = list(operation.get("target_node_ids", []))
            if len(target_ids) < 2:
                raise ValueError("MERGE_BLOCK requires at least two target_node_ids")
            indices = [self._node_index(document, node_id) for node_id in target_ids]
            originals = [copy.deepcopy(document["blocks"][index]) for index in indices]
            insertion = min(indices)
            for index in sorted(indices, reverse=True):
                document["blocks"].pop(index)
            first = originals[0]
            node_id = self._derived_node_id(document, operation_id, 0)
            merged = copy.deepcopy(first)
            merged["node_id"] = node_id
            separator = str(operation.get("separator", "\n"))
            field = "latex" if first["kind"] == "FORMULA" else "text"
            merged["content"][field] = separator.join(
                str(block["content"].get(field) or "") for block in originals
            )
            merged["source_content_ids"] = sorted(
                {value for block in originals for value in block["source_content_ids"]}
            )
            merged["source_candidate_ids"] = sorted(
                {
                    value
                    for block in originals
                    for value in block["source_candidate_ids"]
                }
            )
            merged["source_region_ids"] = sorted(
                {value for block in originals for value in block["source_region_ids"]}
            )
            merged["source_unit_ids"] = sorted(
                {value for block in originals for value in block["source_unit_ids"]}
            )
            merged["provenance"].setdefault("patch_operations", []).append(operation_id)
            merged["provenance"]["merged_from"] = target_ids
            result_kind = operation.get("result_kind")
            if result_kind is not None:
                result_kind = str(result_kind)
                if result_kind not in BLOCK_KINDS:
                    raise ValueError("MERGE_BLOCK result kind is invalid")
                merged["kind"] = result_kind
            if operation.get("merged_content") is not None:
                merged["content"].update(copy.deepcopy(operation["merged_content"]))
            document["blocks"].insert(insertion, merged)
            for original in originals:
                original["visibility"] = "SUPPRESSED_MERGED_SOURCE"
                document["suppressed_blocks"].append(original)
            return [node_id]
        raise ValueError(f"Unsupported patch operation: {op}")

    def _patch_block(
        self,
        *,
        node_id: str,
        page_index: int,
        kind: str,
        content: dict[str, Any],
        operation_id: str,
    ) -> dict[str, Any]:
        if kind not in BLOCK_KINDS:
            raise ValueError("INSERT_BLOCK kind is invalid")
        return {
            "schema": BLOCK_SCHEMA,
            "node_id": node_id,
            "page_index": page_index,
            "kind": kind,
            "subtype": "AUDIT_INSERTED",
            "source_content_ids": [],
            "source_candidate_ids": [],
            "source_region_ids": [],
            "source_unit_ids": [],
            "bbox_pdf_pt": None,
            "bbox_normalized": None,
            "order_key": {"version": READING_ORDER_VERSION, "patch_inserted": True},
            "content": {
                "text": content.get("text"),
                "latex": content.get("latex"),
                "asset_uid": content.get("asset_uid"),
                "asset_ref": content.get("asset_ref"),
                "source_status": "AUDIT_INSERTED",
            },
            "review_state": "NONE",
            "visibility": "VISIBLE",
            "relations": {},
            "provenance": {
                "patch_inserted": True,
                "patch_operations": [operation_id],
            },
        }

    def _refresh_page_membership(self, document: dict[str, Any]) -> None:
        for display_index, block in enumerate(document["blocks"]):
            block["display_index"] = display_index
        for page in document["pages"]:
            index = page["page_index"]
            page["visible_node_ids"] = [
                block["node_id"]
                for block in document["blocks"]
                if block["page_index"] == index
            ]
            page["suppressed_node_ids"] = [
                block["node_id"]
                for block in document["suppressed_blocks"]
                if block["page_index"] == index
            ]
        document["accounting"]["visible_output_nodes"] = len(document["blocks"])
        document["accounting"]["suppressed_nodes"] = len(document["suppressed_blocks"])
