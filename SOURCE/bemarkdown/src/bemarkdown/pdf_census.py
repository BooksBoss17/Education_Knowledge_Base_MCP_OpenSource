from __future__ import annotations

import hashlib
import html
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from .pdf_source import (
    ROUTING_BASELINE,
    ROUTING_STATUS,
    ROUTING_THRESHOLDS,
    PdfInspectionError,
    PdfSourceInspector,
    apply_routing_baseline,
)


@dataclass(frozen=True)
class PdfCorpusCensusResult:
    output_dir: Path
    manifest_path: Path
    document_records_path: Path
    page_records_path: Path
    parser_errors_path: Path
    review_html_path: Path
    summary: dict[str, Any]


def run_pdf_corpus_census(
    corpus_a_root: str | Path,
    corpus_b_root: str | Path,
    legacy_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    render_review: bool = True,
    review_limit: int = 32,
    inspector: PdfSourceInspector | None = None,
) -> PdfCorpusCensusResult:
    """Inspect two read-only PDF corpora and emit the Phase 6A baseline."""
    started = time.perf_counter()
    corpus_a_root = Path(corpus_a_root).resolve()
    corpus_b_root = Path(corpus_b_root).resolve()
    legacy_manifest_path = Path(legacy_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    inspector = inspector or PdfSourceInspector()
    legacy_manifest_before = _file_identity(legacy_manifest_path)
    legacy_by_sha = _legacy_labels(legacy_manifest_path)
    sources = [
        ("FORMAT_TESTSET", path)
        for path in _discover_pdfs(corpus_a_root)
    ] + [
        ("EDUCATION_EXTENDED", path)
        for path in _discover_pdfs(corpus_b_root)
    ]

    documents: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    source_before: dict[str, dict[str, Any]] = {}
    for source_group, source in sources:
        before = _file_identity(source)
        source_before[str(source)] = before
        try:
            routed = apply_routing_baseline(inspector.inspect(source))
        except PdfInspectionError as exc:
            errors.append(
                {
                    "source_group": source_group,
                    "source_path": str(source),
                    "sha256": before["sha256"],
                    "bytes": before["bytes"],
                    "error_code": exc.code,
                    "error": exc.detail,
                }
            )
            continue
        legacy_label = (
            legacy_by_sha.get(routed["sha256"])
            if source_group == "FORMAT_TESTSET"
            else None
        )
        compact_pages = []
        for page in routed["pages"]:
            compact = _compact_page_record(
                page,
                document_id=routed["document_id"],
                source_group=source_group,
                source_path=str(source),
                source_sha256=routed["sha256"],
            )
            pages.append(compact)
            compact_pages.append(compact)
        documents.append(
            {
                "document_id": routed["document_id"],
                "source_path": str(source),
                "source_group": source_group,
                "sha256": routed["sha256"],
                "bytes": routed["bytes"],
                "page_count": routed["page_count"],
                "legacy_label": legacy_label,
                "document_profile": routed["document_profile"],
                "routing_baseline": ROUTING_BASELINE,
                "routing_status": ROUTING_STATUS,
                "inspection_wall_seconds": routed["inspection_wall_seconds"],
                "page_source_profile_counts": dict(
                    sorted(Counter(p["source_profile"] for p in compact_pages).items())
                ),
                "page_route_counts": dict(
                    sorted(Counter(p["routing_decision"] for p in compact_pages).items())
                ),
                "pages": compact_pages,
            }
        )

    source_after = {str(path): _file_identity(path) for _, path in sources}
    source_integrity = _source_integrity(source_before, source_after)
    representatives = _representative_pages(pages, limit=review_limit)
    if render_review:
        _render_representatives(representatives, preview_dir)
    _write_review_html(output_dir / "review.html", representatives)

    comparison = _legacy_comparison(documents)
    feature_distribution = _feature_distribution(pages)
    elapsed = time.perf_counter() - started
    summary = _routing_summary(
        documents,
        pages,
        errors,
        source_integrity,
        elapsed,
        rendered_review_page_count=sum(
            bool(row.get("preview_path")) for row in representatives
        ),
    )
    regression = {
        "schema": "bemarkdown-phase6a-regression-summary-v1",
        "checks": {
            "source_files_unchanged": source_integrity["all_unchanged"],
            "all_successful_pages_routed": all(
                page.get("source_profile") and page.get("routing_decision")
                for page in pages
            ),
            "legacy_manifest_not_modified": _file_identity(legacy_manifest_path)
            == legacy_manifest_before,
            "routing_status_is_baseline": ROUTING_STATUS == "BASELINE",
            "pdf_to_markdown_not_implemented": True,
            "paddle_or_ocr_not_invoked": True,
        },
        "safe_parser_error_count": len(errors),
    }
    regression["passed"] = all(regression["checks"].values())
    manifest = {
        "schema": "bemarkdown-pdf-corpus-manifest-v1",
        "routing_baseline": ROUTING_BASELINE,
        "routing_status": ROUTING_STATUS,
        "backend": {
            "library": "PyMuPDF",
            "version": fitz.VersionBind,
            "reason": (
                "single deterministic backend for page text geometry, image "
                "placements/xrefs, vector drawings, and review rendering"
            ),
        },
        "routing_thresholds": dict(ROUTING_THRESHOLDS),
        "source_roots": {
            "FORMAT_TESTSET": str(corpus_a_root),
            "EDUCATION_EXTENDED": str(corpus_b_root),
        },
        "documents": [_manifest_document(row) for row in documents],
        "parser_errors": errors,
    }

    paths = {
        "manifest": output_dir / "pdf_corpus_manifest.json",
        "documents": output_dir / "document_records.jsonl",
        "pages": output_dir / "page_records.jsonl",
        "distribution": output_dir / "feature_distribution.json",
        "routing": output_dir / "routing_summary.json",
        "legacy": output_dir / "legacy_label_comparison.json",
        "representatives": output_dir / "representative_pages.jsonl",
        "errors": output_dir / "parser_error_cases.jsonl",
        "regression": output_dir / "regression_summary.json",
    }
    _write_json(paths["manifest"], manifest)
    _write_jsonl(paths["documents"], [_document_record(row) for row in documents])
    _write_jsonl(paths["pages"], pages)
    _write_json(paths["distribution"], feature_distribution)
    _write_json(paths["routing"], summary)
    _write_json(paths["legacy"], comparison)
    _write_jsonl(paths["representatives"], representatives)
    _write_jsonl(paths["errors"], errors)
    _write_json(paths["regression"], regression)
    return PdfCorpusCensusResult(
        output_dir=output_dir,
        manifest_path=paths["manifest"],
        document_records_path=paths["documents"],
        page_records_path=paths["pages"],
        parser_errors_path=paths["errors"],
        review_html_path=output_dir / "review.html",
        summary=summary,
    )


def _discover_pdfs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(root.rglob("*.pdf"), key=lambda path: str(path).casefold())


def _legacy_labels(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("records")
    if not isinstance(rows, list):
        raise TypeError("Legacy dataset manifest has no records array")
    return {
        row["source_sha256"]: row["formula_signature"]
        for row in rows
        if row.get("file_type") == "PDF"
        and row.get("source_sha256")
        and row.get("formula_signature")
    }


def _compact_page_record(
    page: dict[str, Any],
    *,
    document_id: str,
    source_group: str,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    native_text = dict(page["native_text"])
    native_text.pop("blocks", None)
    vectors = dict(page["vectors"])
    vectors.pop("drawings", None)
    return {
        "document_id": document_id,
        "source_group": source_group,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "page_index": page["page_index"],
        "page_number": page["page_number"],
        "geometry": page["geometry"],
        "native_text": native_text,
        "images": page["images"],
        "vectors": vectors,
        "source_profile": page["source_profile"],
        "native_text_trust": page["native_text_trust"],
        "routing_decision": page["routing_decision"],
        "reason_codes": page["reason_codes"],
        "reading_order": page["reading_order"],
        "structural_warnings": page["structural_warnings"],
    }


def _document_record(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "pages"}


def _manifest_document(document: dict[str, Any]) -> dict[str, Any]:
    value = _document_record(document)
    value["pages"] = [
        {
            "page_index": page["page_index"],
            "page_number": page["page_number"],
            "source_profile": page["source_profile"],
            "native_text_trust": page["native_text_trust"],
            "routing_decision": page["routing_decision"],
            "reason_codes": page["reason_codes"],
            "page_record_ref": (
                f"page_records.jsonl#{document['document_id']}:{page['page_index']}"
            ),
        }
        for page in document["pages"]
    ]
    return value


def _feature_distribution(pages: list[dict[str, Any]]) -> dict[str, Any]:
    features = {
        "native_text_non_whitespace_chars": [
            row["native_text"]["non_whitespace_char_count"] for row in pages
        ],
        "native_text_bbox_union_area_ratio": [
            row["native_text"]["text_bbox_union_area_ratio"] for row in pages
        ],
        "largest_image_coverage_ratio": [
            row["images"]["largest_image_coverage_ratio"] for row in pages
        ],
        "image_page_coverage_ratio": [
            row["images"]["page_coverage_ratio"] for row in pages
        ],
        "vector_drawing_count": [row["vectors"]["drawing_count"] for row in pages],
        "vector_drawing_coverage_ratio": [
            row["vectors"]["drawing_coverage_ratio"] for row in pages
        ],
    }
    return {
        "schema": "bemarkdown-phase6a-feature-distribution-v1",
        "page_count": len(pages),
        "quantiles": {name: _quantiles(values) for name, values in features.items()},
        "selected_thresholds": dict(ROUTING_THRESHOLDS),
        "selection_evidence": {
            "text_gap": {
                "pages_with_zero_non_whitespace_chars": sum(
                    value == 0
                    for value in features["native_text_non_whitespace_chars"]
                ),
                "minimum_positive_non_whitespace_chars": min(
                    (
                        value
                        for value in features["native_text_non_whitespace_chars"]
                        if value > 0
                    ),
                    default=None,
                ),
            },
            "scan_like_raster_gap": {
                "pages_in_0_80_to_0_85_open_interval": sum(
                    0.80 < value < 0.85
                    for value in features["largest_image_coverage_ratio"]
                ),
                "threshold": ROUTING_THRESHOLDS[
                    "scan_like_raster_coverage_ratio"
                ],
            },
        },
    }


def _routing_summary(
    documents: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    source_integrity: dict[str, Any],
    wall_seconds: float,
    *,
    rendered_review_page_count: int,
) -> dict[str, Any]:
    by_group = {}
    for group in ("FORMAT_TESTSET", "EDUCATION_EXTENDED"):
        local_docs = [row for row in documents if row["source_group"] == group]
        local_pages = [row for row in pages if row["source_group"] == group]
        by_group[group] = _counts(local_docs, local_pages)
    image_occurrences = [
        placement for page in pages for placement in page["images"]["placements"]
    ]
    unique_images: dict[tuple[str, str], dict[str, Any]] = {}
    for page in pages:
        for placement in page["images"]["placements"]:
            unique_images.setdefault(
                (page["document_id"], placement["object_identity"]), placement
            )
    return {
        "schema": "bemarkdown-phase6a-routing-summary-v1",
        "routing_baseline": ROUTING_BASELINE,
        "routing_status": ROUTING_STATUS,
        "overall": _counts(documents, pages),
        "by_source_group": by_group,
        "parser_error_count": len(errors),
        "source_integrity": source_integrity,
        "native_image_feasibility": {
            "embedded_raster_occurrence_count": len(image_occurrences),
            "directly_extractable_occurrence_count": sum(
                row["directly_extractable"] for row in image_occurrences
            ),
            "unique_image_object_count": len(unique_images),
            "directly_extractable_unique_object_count": sum(
                row["directly_extractable"] for row in unique_images.values()
            ),
            "alpha_or_mask_occurrence_count": sum(
                row["has_alpha_or_mask"] for row in image_occurrences
            ),
            "reused_occurrence_count": sum(
                row["reused_object"] for row in image_occurrences
            ),
            "extractable_occurrence_bytes": sum(
                row["extractable_bytes"]
                for row in image_occurrences
                if row["directly_extractable"]
            ),
        },
        "performance": {
            "structural_census_wall_seconds": round(wall_seconds, 6),
            "documents_per_second": round(
                (len(documents) + len(errors)) / wall_seconds, 6
            )
            if wall_seconds
            else 0.0,
            "pages_per_second": round(len(pages) / wall_seconds, 6)
            if wall_seconds
            else 0.0,
            "rendered_review_page_count": rendered_review_page_count,
        },
    }


def _counts(
    documents: list[dict[str, Any]], pages: list[dict[str, Any]]
) -> dict[str, Any]:
    total_bytes = sum(row["bytes"] for row in documents)
    document_profiles = Counter(row["document_profile"] for row in documents)
    source_profiles = Counter(row["source_profile"] for row in pages)
    routes = Counter(row["routing_decision"] for row in pages)
    trust = Counter(row["native_text_trust"] for row in pages)
    reading_order = Counter(row["reading_order"]["risk"] for row in pages)
    return {
        "document_count": len(documents),
        "document_bytes": total_bytes,
        "corpus_fingerprint": _corpus_fingerprint(documents),
        "page_count": len(pages),
        "document_profiles": _distribution(document_profiles, len(documents)),
        "page_source_profiles": _distribution(source_profiles, len(pages)),
        "routing_decisions": _distribution(routes, len(pages)),
        "native_text_trust": _distribution(trust, len(pages)),
        "reading_order_risk": _distribution(reading_order, len(pages)),
    }


def _distribution(counter: Counter[str], total: int) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "count": count,
            "ratio": round(count / total, 6) if total else 0.0,
        }
        for name, count in sorted(counter.items())
    }


def _corpus_fingerprint(documents: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda row: row["source_path"].casefold()):
        digest.update(document["source_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(document["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _legacy_comparison(documents: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {"PDF_IMAGE_ONLY": "IMAGE_ONLY", "PDF_MIXED": "MIXED"}
    rows = []
    for document in documents:
        legacy = document.get("legacy_label")
        if not legacy:
            continue
        expected_profile = expected.get(legacy)
        match = expected_profile == document["document_profile"]
        rows.append(
            {
                "document_id": document["document_id"],
                "source_path": document["source_path"],
                "sha256": document["sha256"],
                "legacy_label": legacy,
                "expected_document_profile": expected_profile,
                "new_document_profile": document["document_profile"],
                "match": match,
                "difference_reason": None
                if match
                else (
                    "New profile is derived from page routes and was not changed "
                    "to fit the legacy directory label."
                ),
            }
        )
    return {
        "schema": "bemarkdown-phase6a-legacy-label-comparison-v1",
        "matches": sum(row["match"] for row in rows),
        "differences": sum(not row["match"] for row in rows),
        "documents": rows,
    }


def _representative_pages(
    pages: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    strata: defaultdict[tuple[str, int], set[str]] = defaultdict(set)

    def retain(row: dict[str, Any], label: str) -> None:
        key = (row["document_id"], row["page_index"])
        if key not in selected and len(selected) >= max(0, limit):
            return
        selected.setdefault(key, row)
        strata[key].add(label)

    for row in pages:
        if row["routing_decision"] == "REVIEW_REQUIRED":
            retain(row, "ALL_REVIEW_REQUIRED")
    for field, values in (
        (
            "source_profile",
            (
                "NATIVE_TEXT",
                "IMAGE_ONLY",
                "IMAGE_WITH_TEXT_LAYER",
                "MIXED_NATIVE_VISUAL",
                "UNCERTAIN",
            ),
        ),
        (
            "routing_decision",
            (
                "NATIVE_FIRST",
                "VISUAL_REQUIRED",
                "HYBRID_REQUIRED",
                "REVIEW_REQUIRED",
            ),
        ),
    ):
        for value in values:
            candidate = next((row for row in pages if row[field] == value), None)
            if candidate:
                retain(candidate, f"{field.upper()}_{value}")
    boundaries = (
        (
            "SCAN_LIKE_RASTER_BOUNDARY",
            lambda row: row["images"]["largest_image_coverage_ratio"],
            ROUTING_THRESHOLDS["scan_like_raster_coverage_ratio"],
        ),
        (
            "SIGNIFICANT_RASTER_BOUNDARY",
            lambda row: row["images"]["page_coverage_ratio"],
            ROUTING_THRESHOLDS["significant_raster_coverage_ratio"],
        ),
        (
            "SIGNIFICANT_VECTOR_BOUNDARY",
            lambda row: row["vectors"]["drawing_coverage_ratio"],
            ROUTING_THRESHOLDS["significant_vector_coverage_ratio"],
        ),
        (
            "SUBSTANTIAL_TEXT_BOUNDARY",
            lambda row: row["native_text"]["non_whitespace_char_count"],
            ROUTING_THRESHOLDS["substantial_text_non_whitespace_chars"],
        ),
    )
    for label, getter, threshold in boundaries:
        below = [row for row in pages if getter(row) < threshold]
        above = [row for row in pages if getter(row) >= threshold]
        if below:
            retain(max(below, key=getter), f"{label}_BELOW")
        if above:
            retain(min(above, key=getter), f"{label}_ABOVE")
    broken_retained = 0
    for row in pages:
        if (
            "BROKEN_IMAGE_EXTRACTION" in row["structural_warnings"]
            and broken_retained < 2
        ):
            retain(row, "BROKEN_IMAGE_EXTRACTION")
            broken_retained += 1

    result = []
    for index, (key, row) in enumerate(selected.items(), start=1):
        value = dict(row)
        value["audit_strata"] = sorted(strata[key])
        value["representative_index"] = index
        value["preview_path"] = None
        value["visual_review_status"] = "PENDING_AGENT_VISUAL_REVIEW"
        result.append(value)
    return result


def _render_representatives(rows: list[dict[str, Any]], preview_dir: Path) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    open_document: fitz.Document | None = None
    open_path: str | None = None
    try:
        for row in rows:
            if row["source_path"] != open_path:
                if open_document is not None:
                    open_document.close()
                open_path = row["source_path"]
                open_document = fitz.open(open_path)
            assert open_document is not None
            filename = (
                f"{row['representative_index']:03d}_"
                f"{row['source_sha256'][:12]}_p{row['page_number']:04d}.jpg"
            )
            target = preview_dir / filename
            page = open_document[row["page_index"]]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(4 / 3, 4 / 3), alpha=False)
            pixmap.pil_save(target, format="JPEG", quality=76, optimize=True)
            row["preview_path"] = f"previews/{filename}"
    finally:
        if open_document is not None:
            open_document.close()


def _write_review_html(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        image = (
            f'<img src="{html.escape(row["preview_path"])}" alt="page preview">'
            if row.get("preview_path")
            else '<div class="no-preview">Preview disabled</div>'
        )
        metrics = {
            "text_chars": row["native_text"]["non_whitespace_char_count"],
            "text_bbox_area": row["native_text"]["text_bbox_union_area_ratio"],
            "largest_raster": row["images"]["largest_image_coverage_ratio"],
            "raster_union": row["images"]["page_coverage_ratio"],
            "vector_count": row["vectors"]["drawing_count"],
            "vector_coverage": row["vectors"]["drawing_coverage_ratio"],
            "reading_order_risk": row["reading_order"]["risk"],
        }
        cards.append(
            '<article class="card">'
            + image
            + '<div class="body">'
            + f'<h2>{html.escape(row["document_id"])} - page {row["page_number"]}</h2>'
            + f'<p><strong>{row["source_profile"]}</strong> / '
            + f'<strong>{row["routing_decision"]}</strong> / trust '
            + f'{row["native_text_trust"]}</p>'
            + f'<p>Reasons: {html.escape(", ".join(row["reason_codes"]))}</p>'
            + f'<p>Strata: {html.escape(", ".join(row["audit_strata"]))}</p>'
            + '<pre>'
            + html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))
            + '</pre></div></article>'
        )
    payload = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>BeMarkdown Phase 6A PDF routing review</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;background:#eef2f7;margin:0;padding:24px;color:#172033}
h1{max-width:1400px;margin:0 auto 20px}.grid{max-width:1400px;margin:auto;display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px}
.card{background:white;border:1px solid #d7deea;border-radius:12px;overflow:hidden;box-shadow:0 4px 14px #1f293714}.card img{width:100%;height:520px;object-fit:contain;background:#303640}.body{padding:16px}.body h2{font-size:16px;word-break:break-all}.body p{font-size:13px;line-height:1.45}.body pre{background:#f6f8fb;padding:12px;border-radius:8px;overflow:auto;font-size:12px}.no-preview{height:180px;display:grid;place-items:center;background:#303640;color:white}
</style></head><body><h1>Phase 6A representative page audit</h1><main class="grid">"""
    payload += "".join(cards) + "</main></body></html>\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def _quantiles(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {name: None for name in ("min", "p05", "p25", "p50", "p75", "p95", "max")}
    ordered = sorted(values)

    def pick(fraction: float) -> float | int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p05": pick(0.05),
        "p25": pick(0.25),
        "p50": statistics.median(ordered),
        "p75": pick(0.75),
        "p95": pick(0.95),
        "max": ordered[-1],
    }


def _source_integrity(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    for path in sorted(before, key=str.casefold):
        unchanged = before[path] == after.get(path)
        rows.append(
            {
                "source_path": path,
                "before": before[path],
                "after": after.get(path),
                "unchanged": unchanged,
            }
        )
    return {
        "all_unchanged": all(row["unchanged"] for row in rows),
        "checked_file_count": len(rows),
        "files": rows,
    }


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "sha256": digest.hexdigest(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(pending, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(pending, path)
