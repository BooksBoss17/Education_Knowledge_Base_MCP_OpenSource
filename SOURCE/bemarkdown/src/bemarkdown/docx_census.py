from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .package import PackageIndex
from .pipeline import _finalize_formula_report, _new_report
from .scanner import DocumentScanner

FINAL_STATES = (
    "STRUCTURAL_OMML",
    "STRUCTURAL_EQ",
    "STRUCTURAL_LATEX_ALT",
    "STRUCTURAL_OLE_MTEF",
    "STRUCTURAL_WMF_MTEF",
    "STRUCTURAL_WMF_MATHML",
    "EMPTY_PLACEHOLDER_SUPPRESSED",
    "RENDERED_NEEDS_MATH_OCR",
    "FAILED_NO_PREVIEW",
    "FAILED_RENDERER",
    "UNRESOLVED",
)


@dataclass(frozen=True)
class DocxCensusResult:
    output_dir: Path
    summary_path: Path
    docx_records_path: Path
    candidates_path: Path
    ocr_candidates_path: Path
    failures_path: Path
    structural_rejected_path: Path
    rendered_dir: Path
    audit_dir: Path
    summary: dict[str, Any]


def run_docx_formula_census(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    phase25_census_dir: str | Path | None = None,
    audit_limit: int = 80,
) -> DocxCensusResult:
    """Run the production formula router over every manifest DOCX without OCR."""

    started = time.perf_counter()
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = output_dir / "rendered_png"
    audit_dir = output_dir / "audit_renders"
    work_dir = output_dir / ".work"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docx_manifest = [
        record
        for record in manifest.get("records", [])
        if str(record.get("file_type", "")).upper() == "DOCX"
    ]
    docx_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    aggregate_timing: Counter[str] = Counter()
    aggregate_route_counts: Counter[str] = Counter()

    for index, manifest_record in enumerate(docx_manifest, start=1):
        source = manifest_path.parent / Path(manifest_record["target_relative_path"])
        signature = manifest_record.get("formula_signature") or "UNKNOWN"
        expected_sha = manifest_record.get("source_sha256")
        base_record = {
            "docx_index": index,
            "source_file": str(source),
            "source_relative_path": manifest_record.get("target_relative_path"),
            "formula_signature": signature,
            "expected_sha256": expected_sha,
        }
        document_work = work_dir / f"{index:04d}"
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            source_data = source.read_bytes()
            actual_sha = hashlib.sha256(source_data).hexdigest()
            if expected_sha and actual_sha != expected_sha:
                failure = {
                    **base_record,
                    "status": "SOURCE_HASH_MISMATCH",
                    "actual_sha256": actual_sha,
                    "error": "Source SHA-256 does not match manifest",
                }
                failures.append(failure)
                docx_records.append(failure)
                continue

            report = _new_report(source)
            package = PackageIndex(source)
            report["timing"]["package_read_seconds"] = package.read_seconds
            scan_started = time.perf_counter()
            DocumentScanner(
                package, document_work, report, analysis_mode=True
            ).scan()
            report["timing"]["scan_seconds"] = time.perf_counter() - scan_started
            _finalize_formula_report(report)
            local_candidates = _normalize_candidates(
                report,
                docx_sha256=actual_sha,
                source_file=source,
                formula_signature=signature,
                document_work=document_work,
                output_dir=output_dir,
                rendered_dir=rendered_dir,
            )
            candidates.extend(local_candidates)
            for key, value in report["timing"].items():
                if isinstance(value, int | float):
                    aggregate_timing[key] += value
            for key in (
                "wmf_previews_discovered",
                "preview_suppressed_by_ole",
                "wmf_inspect_calls",
                "wmf_embedded_recovery_calls",
                "wmf_render_calls",
            ):
                aggregate_route_counts[key] += report["formulas"][key]
            local_states = Counter(row["final_state"] for row in local_candidates)
            expected_candidates = (
                report["formulas"]["equation_candidates"]
                - report["formulas"]["non_formula_objects"]
            )
            docx_records.append(
                {
                    **base_record,
                    "status": "SUCCESS",
                    "docx_sha256": actual_sha,
                    "size_bytes": len(source_data),
                    "candidate_occurrences": len(local_candidates),
                    "candidate_fingerprints": len(
                        {row["candidate_fingerprint"] for row in local_candidates}
                    ),
                    "final_states": dict(sorted(local_states.items())),
                    "excluded_non_formula_objects": report["formulas"][
                        "non_formula_objects"
                    ],
                    "candidate_conservation_ok": (
                        len(local_candidates) == expected_candidates
                        and sum(local_states.values()) == len(local_candidates)
                    ),
                    "timing": report["timing"],
                    "route_counts": {
                        key: report["formulas"][key]
                        for key in aggregate_route_counts
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate one source document
            failure = {
                **base_record,
                "status": "FAILED_DOCX",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            docx_records.append(failure)
        finally:
            shutil.rmtree(document_work, ignore_errors=True)

    shutil.rmtree(work_dir, ignore_errors=True)
    audit_records = _save_audit_renders(
        candidates, output_dir, audit_dir, limit=audit_limit
    )
    comparison = _load_phase25_comparison(phase25_census_dir)
    summary = _build_summary(
        docx_manifest,
        docx_records,
        candidates,
        failures,
        aggregate_timing,
        aggregate_route_counts,
        comparison,
        audit_records,
        time.perf_counter() - started,
    )

    paths = {
        "summary": output_dir / "census_summary.json",
        "docx": output_dir / "docx_records.jsonl",
        "candidates": output_dir / "formula_candidates.jsonl",
        "ocr": output_dir / "true_math_ocr_candidates.jsonl",
        "failures": output_dir / "failures.jsonl",
        "rejected": output_dir / "structural_rejected.jsonl",
    }
    report_started = time.perf_counter()
    _write_jsonl(paths["docx"], docx_records)
    _write_jsonl(paths["candidates"], candidates)
    _write_jsonl(
        paths["ocr"], [row for row in candidates if row["needs_math_ocr"]]
    )
    _write_jsonl(paths["failures"], failures)
    _write_jsonl(
        paths["rejected"],
        [row for row in candidates if row["structural_attempts"]],
    )
    summary["performance"]["report_seconds"] = round(
        time.perf_counter() - report_started, 6
    )
    summary["performance"]["total_wall_seconds"] = round(
        time.perf_counter() - started, 6
    )
    _write_json(paths["summary"], summary)
    return DocxCensusResult(
        output_dir,
        paths["summary"],
        paths["docx"],
        paths["candidates"],
        paths["ocr"],
        paths["failures"],
        paths["rejected"],
        rendered_dir,
        audit_dir,
        summary,
    )


def _normalize_candidates(
    report,
    *,
    docx_sha256,
    source_file,
    formula_signature,
    document_work,
    output_dir,
    rendered_dir,
):
    formula_records = {
        row["formula_id"]: row for row in report["formulas"]["records"]
    }
    rows = []
    for candidate in report["formulas"]["candidate_records"]:
        if candidate["classification"] == "NON_FORMULA_OBJECT":
            continue
        formula = formula_records.get(candidate["candidate_id"])
        final_state, successful_route = _final_state(candidate, formula)
        validator = candidate.get("validator") or _last_attempt_validator(candidate)
        structural_attempts = _compact_attempts(candidate.get("structural_attempts", []))
        renderer_metadata = (
            (formula or {}).get("renderer_metadata")
            or candidate.get("renderer_metadata")
            or {}
        )
        render_status = renderer_metadata.get("status")
        render_attempted = bool(renderer_metadata)
        preview_visual_state = candidate.get("preview_visual_state")
        needs_math_ocr = bool(
            final_state == "RENDERED_NEEDS_MATH_OCR"
            and preview_visual_state == "visible"
            and render_status == "success"
        )
        fingerprint = _candidate_fingerprint(candidate, formula, final_state)
        row = {
            "candidate_id": f"{docx_sha256[:16]}:{candidate['candidate_id']}",
            "local_candidate_id": candidate["candidate_id"],
            "docx_sha256": docx_sha256,
            "source_file": str(source_file),
            "formula_signature": formula_signature,
            "source_part": candidate.get("source_part") or "word/document.xml",
            "source_locator": candidate.get("source_locator"),
            "semantic_source_ref": candidate.get("original_ref"),
            "preview_source_ref": candidate.get("preview_ref"),
            "candidate_sources": candidate.get("candidate_sources")
            or _sources_from_type(candidate.get("source_type")),
            "ole_prog_id": candidate.get("prog_id"),
            "ole_relationship_id": candidate.get("ole_relationship_id"),
            "preview_relationship_id": candidate.get("preview_relationship_id"),
            "ole_sha256": candidate.get("ole_sha256"),
            "wmf_sha256": candidate.get("wmf_sha256"),
            "attempted_routes": _attempted_routes(candidate, final_state),
            "successful_route": successful_route,
            "final_state": final_state,
            "validator_verdict": validator.get("verdict") if validator else None,
            "validator_issues": validator.get("issues", []) if validator else [],
            "structural_attempts": structural_attempts,
            "preview_visual_state": preview_visual_state,
            "preview_suppressed": bool(candidate.get("preview_suppressed")),
            "preview_suppressed_by_ole": bool(
                candidate.get("preview_suppressed_by_ole")
            ),
            "render_attempted": render_attempted,
            "render_status": render_status,
            "render_metadata": renderer_metadata or None,
            "needs_math_ocr": needs_math_ocr,
            "candidate_fingerprint": fingerprint,
            "rendered_png_path": None,
            "png_sha256": None,
            "png_bytes": None,
            "width": None,
            "height": None,
        }
        if needs_math_ocr and formula and formula.get("rendered_ref"):
            rendered_source = document_work / formula["rendered_ref"]
            _retain_rendered_png(row, rendered_source, output_dir, rendered_dir)
        rows.append(row)
    return rows


def _final_state(candidate, formula):
    if candidate["classification"] == "EMPTY_PLACEHOLDER":
        return "EMPTY_PLACEHOLDER_SUPPRESSED", "EMPTY_CLASSIFIER"
    status = candidate.get("status")
    source_type = candidate.get("source_type")
    if status in {"SUCCESS_EXACT", "SUCCESS_NORMALIZED", "SUCCESS_APPROXIMATE"}:
        if source_type == "omml":
            return "STRUCTURAL_OMML", "OMML"
        if source_type == "eq":
            return "STRUCTURAL_EQ", "EQ"
        if source_type == "latex_alt":
            return "STRUCTURAL_LATEX_ALT", "LATEX_ALT"
        if source_type == "mtef":
            return "STRUCTURAL_OLE_MTEF", "OLE_MTEF"
        if source_type == "wmf_embedded":
            payload_type = candidate.get("source_payload_type")
            if payload_type == "WMF_MATHML":
                return "STRUCTURAL_WMF_MATHML", "WMF_MATHML"
            return "STRUCTURAL_WMF_MTEF", "WMF_MTEF"
    if status == "RENDERED_FALLBACK":
        metadata = (formula or {}).get("renderer_metadata") or {}
        if metadata.get("status") == "success":
            return "RENDERED_NEEDS_MATH_OCR", "WMF_RENDERER"
        return "FAILED_RENDERER", None
    if status == "FAILED_PRESERVED":
        metadata = (formula or {}).get("renderer_metadata") or candidate.get(
            "renderer_metadata"
        )
        if metadata:
            return "FAILED_RENDERER", None
        return "FAILED_NO_PREVIEW", None
    return "UNRESOLVED", None


def _candidate_fingerprint(candidate, formula, final_state):
    payload_sha = candidate.get("source_payload_sha256")
    prefix = {
        "STRUCTURAL_OMML": "omml",
        "STRUCTURAL_EQ": "eq",
        "STRUCTURAL_LATEX_ALT": "latex_alt",
        "STRUCTURAL_OLE_MTEF": "ole_mtef",
        "STRUCTURAL_WMF_MTEF": "wmf_mtef",
        "STRUCTURAL_WMF_MATHML": "wmf_mathml",
    }.get(final_state)
    if prefix and payload_sha:
        return f"{prefix}:{payload_sha}"
    if final_state == "RENDERED_NEEDS_MATH_OCR" and candidate.get("wmf_sha256"):
        return f"wmf:{candidate['wmf_sha256']}"
    fallback = "|".join(
        str(value or "")
        for value in (
            candidate.get("ole_sha256"),
            candidate.get("wmf_sha256"),
            (formula or {}).get("latex"),
            candidate.get("source_locator"),
        )
    )
    return f"{final_state.lower()}:{hashlib.sha256(fallback.encode('utf-8')).hexdigest()}"


def _attempted_routes(candidate, final_state):
    source_type = candidate.get("source_type")
    if source_type in {"omml", "eq", "latex_alt"}:
        return [source_type.upper()]
    routes = ["OLE_MTEF"]
    inspection = candidate.get("wmf_inspection") or {}
    if inspection.get("embedded_mtef_presence"):
        routes.append("WMF_MTEF")
    elif inspection.get("embedded_mathml_presence"):
        routes.append("WMF_MATHML")
    if final_state == "EMPTY_PLACEHOLDER_SUPPRESSED":
        routes.append("EMPTY_CLASSIFIER")
    if candidate.get("renderer_metadata") or final_state in {
        "RENDERED_NEEDS_MATH_OCR",
        "FAILED_RENDERER",
    }:
        routes.append("WMF_RENDERER")
    return routes


def _last_attempt_validator(candidate):
    attempts = candidate.get("structural_attempts") or []
    return attempts[-1].get("validator") if attempts else None


def _compact_attempts(attempts):
    return [
        {
            "route": attempt.get("route"),
            "payload_type": attempt.get("payload_type"),
            "conversion_status": attempt.get("conversion_status"),
            "converted_latex": attempt.get("converted_latex"),
            "validator_verdict": (attempt.get("validator") or {}).get("verdict"),
            "validator_issues": (attempt.get("validator") or {}).get("issues", []),
        }
        for attempt in attempts
    ]


def _sources_from_type(source_type):
    result = {key: False for key in ("omml", "eq", "latex_alt", "ole", "preview")}
    if source_type in result:
        result[source_type] = True
    return result


def _retain_rendered_png(row, source, output_dir, rendered_dir):
    if not source.is_file():
        row["needs_math_ocr"] = False
        row["final_state"] = "FAILED_RENDERER"
        row["render_status"] = "missing_output"
        return
    data = source.read_bytes()
    png_sha = hashlib.sha256(data).hexdigest()
    stem = row.get("wmf_sha256") or png_sha
    name = f"{stem}_{png_sha[:12]}.png"
    target = rendered_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    width, height = (row.get("render_metadata") or {}).get("final_size") or (None, None)
    row.update(
        {
            "rendered_png_path": target.relative_to(output_dir).as_posix(),
            "png_sha256": png_sha,
            "png_bytes": len(data),
            "width": width,
            "height": height,
        }
    )


def _save_audit_renders(candidates, output_dir, audit_dir, *, limit):
    eligible = sorted(
        (row for row in candidates if row["needs_math_ocr"]),
        key=lambda row: (row["formula_signature"], row["candidate_fingerprint"]),
    )
    if len(eligible) <= limit:
        selected = eligible
    else:
        selected = []
        seen_groups: Counter[tuple] = Counter()
        for row in eligible:
            metadata = row.get("render_metadata") or {}
            width, height = metadata.get("final_size") or (0, 0)
            shape = "wide" if width > 4 * height else "tall" if height > 2 * width else "normal"
            rejected = bool(row["structural_attempts"])
            group = (row["formula_signature"], row.get("ole_prog_id"), rejected, shape)
            if seen_groups[group] < 4:
                selected.append(row)
                seen_groups[group] += 1
                if len(selected) == limit:
                    break
        if len(selected) < limit:
            selected_ids = {row["candidate_id"] for row in selected}
            selected.extend(
                row
                for row in eligible
                if row["candidate_id"] not in selected_ids
            )
            selected = selected[:limit]
    audit_records = []
    for index, row in enumerate(selected, start=1):
        source = output_dir / row["rendered_png_path"]
        target = audit_dir / f"{index:03d}_{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        audit_records.append(
            {
                "candidate_id": row["candidate_id"],
                "audit_path": target.relative_to(output_dir).as_posix(),
                "formula_signature": row["formula_signature"],
                "ole_prog_id": row.get("ole_prog_id"),
                "structural_rejected": bool(row["structural_attempts"]),
                "width": row["width"],
                "height": row["height"],
            }
        )
    return audit_records


def _load_phase25_comparison(directory):
    if directory is None:
        return {
            "occurrences": 18461,
            "unique": 15406,
            "referenced_occurrences": 23371,
            "ocr_wmf_sha256": set(),
            "rendered_wmf_sha256": set(),
            "source": "spec_baseline",
        }
    directory = Path(directory).resolve()
    summary_path = directory / "census_summary.json"
    ocr_path = directory / "ocr_candidates.jsonl"
    records_path = directory / "census_records.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ocr_sha = {
        json.loads(line)["wmf_sha256"]
        for line in ocr_path.read_text(encoding="utf-8").splitlines()
        if line
    }
    rendered_sha = {
        row["wmf_sha256"]
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line and (row := json.loads(line)).get("render_attempted")
    }
    return {
        "occurrences": summary["referenced"]["true_ocr_candidates"]["occurrences"],
        "unique": summary["referenced"]["true_ocr_candidates"]["unique"],
        "referenced_occurrences": summary["referenced"]["occurrences"],
        "ocr_wmf_sha256": ocr_sha,
        "rendered_wmf_sha256": rendered_sha,
        "source": str(directory),
    }


def _build_summary(
    docx_manifest,
    docx_records,
    candidates,
    failures,
    aggregate_timing,
    route_counts,
    comparison,
    audit_records,
    wall_seconds,
):
    fingerprints = {row["candidate_fingerprint"] for row in candidates}
    empty = [row for row in candidates if row["final_state"] == "EMPTY_PLACEHOLDER_SUPPRESSED"]
    real = [row for row in candidates if row not in empty]
    ocr = [row for row in candidates if row["needs_math_ocr"]]
    route_summary = {
        state: _state_metric(candidates, state) for state in FINAL_STATES
    }
    signatures = sorted(
        {record.get("formula_signature") or "UNKNOWN" for record in docx_manifest}
    )
    signature_summary = {}
    for signature in signatures:
        docs = [
            row for row in docx_records if row["formula_signature"] == signature
        ]
        rows = [row for row in candidates if row["formula_signature"] == signature]
        real_rows = [
            row
            for row in rows
            if row["final_state"] != "EMPTY_PLACEHOLDER_SUPPRESSED"
        ]
        ocr_rows = [row for row in rows if row["needs_math_ocr"]]
        failure_rows = [
            row
            for row in rows
            if row["final_state"]
            in {"FAILED_NO_PREVIEW", "FAILED_RENDERER", "UNRESOLVED"}
        ]
        signature_summary[signature] = {
            "docx_total": len(docs),
            "docx_success": sum(row["status"] == "SUCCESS" for row in docs),
            "candidate_occurrences": len(rows),
            "candidate_unique": len({row["candidate_fingerprint"] for row in rows}),
            "structural_success": sum(row["final_state"].startswith("STRUCTURAL_") for row in rows),
            "empty": len(rows) - len(real_rows),
            "true_math_ocr": len(ocr_rows),
            "failures": len(failure_rows),
            "ocr_percentage_of_real": round(100 * len(ocr_rows) / len(real_rows), 6) if real_rows else 0.0,
        }

    ole_wmf = [
        row
        for row in candidates
        if row["candidate_sources"].get("ole")
        and row["candidate_sources"].get("preview")
        and row.get("wmf_sha256")
    ]
    ole_rejected = [
        row
        for row in ole_wmf
        if any(attempt["route"] == "OLE_MTEF" for attempt in row["structural_attempts"])
    ]
    suppressed = [row for row in ole_wmf if row["preview_suppressed_by_ole"]]
    avoided_render = sum(
        row["wmf_sha256"] in comparison["rendered_wmf_sha256"] for row in suppressed
    )
    avoided_ocr = sum(
        row["wmf_sha256"] in comparison["ocr_wmf_sha256"] for row in suppressed
    )
    ocr_unique_png = {row["png_sha256"] for row in ocr if row["png_sha256"]}
    stored_png_rows = {
        row["rendered_png_path"]: row for row in ocr if row["rendered_png_path"]
    }
    unique_png_rows = {
        row["png_sha256"]: row for row in ocr if row["png_sha256"]
    }
    widths = [row["width"] for row in unique_png_rows.values() if row["width"]]
    heights = [row["height"] for row in unique_png_rows.values() if row["height"]]
    png_bytes = sum(row["png_bytes"] for row in unique_png_rows.values())
    stored_png_bytes = sum(row["png_bytes"] for row in stored_png_rows.values())
    wmf_only_rate = (
        100 * comparison["occurrences"] / comparison["referenced_occurrences"]
    )
    full_docx_rate = 100 * len(ocr) / len(real) if real else 0.0
    ocr_violations = sum(
        not (
            row["final_state"] == "RENDERED_NEEDS_MATH_OCR"
            and row["preview_visual_state"] == "visible"
            and row["render_status"] == "success"
            and not row["preview_suppressed"]
        )
        for row in ocr
    )
    suppression_violations = sum(
        row["final_state"] == "STRUCTURAL_OLE_MTEF"
        and row.get("wmf_sha256")
        and not (
            row["preview_suppressed_by_ole"]
            and not row["render_attempted"]
            and row["preview_visual_state"] == "not_inspected"
        )
        for row in candidates
    )
    global_ok = bool(
        len(candidates) == sum(value["occurrences"] for value in route_summary.values())
        and all(
            row.get("candidate_conservation_ok", False)
            for row in docx_records
            if row["status"] == "SUCCESS"
        )
    )
    total_wall = max(wall_seconds, 0.0)
    return {
        "schema": "bemarkdown-docx-formula-census-phase26-v1",
        "docx": {
            "total": len(docx_manifest),
            "success": sum(row["status"] == "SUCCESS" for row in docx_records),
            "failed": len(failures),
            "source_hash_mismatch": sum(
                row["status"] == "SOURCE_HASH_MISMATCH" for row in failures
            ),
        },
        "candidates": {
            "occurrences": len(candidates),
            "unique": len(fingerprints),
            "real_formula_occurrences": len(real),
            "real_formula_unique": len(
                {row["candidate_fingerprint"] for row in real}
            ),
            "empty_occurrences": len(empty),
            "true_math_ocr": {
                "occurrences": len(ocr),
                "unique_fingerprints": len(
                    {row["candidate_fingerprint"] for row in ocr}
                ),
                "unique_png": len(ocr_unique_png),
                "occurrence_percentage_of_real": round(
                    100 * len(ocr) / len(real), 6
                )
                if real
                else 0.0,
            },
        },
        "final_states": route_summary,
        "formula_signatures": signature_summary,
        "ole_benefit": {
            "candidates_with_ole_and_wmf_preview": len(ole_wmf),
            "ole_structural_valid": sum(
                row["final_state"] == "STRUCTURAL_OLE_MTEF" for row in ole_wmf
            ),
            "ole_structural_rejected": len(ole_rejected),
            "ole_empty": sum(
                row["final_state"] == "EMPTY_PLACEHOLDER_SUPPRESSED"
                for row in ole_wmf
            ),
            "preview_suppressed_by_ole": len(suppressed),
            "preview_suppressed_unique_wmf": len(
                {row["wmf_sha256"] for row in suppressed}
            ),
            "avoided_wmf_inspect": len(suppressed),
            "avoided_render_logical_occurrences": avoided_render,
            "avoided_render_unique_wmf": len(
                {
                    row["wmf_sha256"]
                    for row in suppressed
                    if row["wmf_sha256"] in comparison["rendered_wmf_sha256"]
                }
            ),
            "avoided_ocr_logical_occurrences": avoided_ocr,
            "avoided_ocr_unique_wmf": len(
                {
                    row["wmf_sha256"]
                    for row in suppressed
                    if row["wmf_sha256"] in comparison["ocr_wmf_sha256"]
                }
            ),
            "unit_note": (
                "Logical-object occurrences can exceed Phase 2.5 WMF asset "
                "occurrences when one package asset is reused by multiple objects."
            ),
        },
        "phase25_comparison": {
            "source": comparison["source"],
            "wmf_only_ocr_occurrences": comparison["occurrences"],
            "wmf_only_ocr_unique": comparison["unique"],
            "wmf_only_rate": round(wmf_only_rate, 6),
            "full_docx_true_ocr_occurrences": len(ocr),
            "full_docx_true_ocr_rate": round(full_docx_rate, 6),
            "absolute_percentage_point_reduction": round(
                wmf_only_rate - full_docx_rate, 6
            ),
            "relative_rate_reduction_percentage": round(
                100 * (wmf_only_rate - full_docx_rate) / wmf_only_rate, 6
            )
            if wmf_only_rate
            else 0.0,
        },
        "formulanet_input": {
            "occurrences": len(ocr),
            "unique_candidate_fingerprints": len(
                {row["candidate_fingerprint"] for row in ocr}
            ),
            "stored_png_files": len(stored_png_rows),
            "unique_png_content_hashes": len(ocr_unique_png),
            "stored_png_bytes": stored_png_bytes,
            "deduplicated_content_bytes": png_bytes,
            "width": _distribution(widths),
            "height": _distribution(heights),
        },
        "performance": {
            **{key: round(value, 6) for key, value in aggregate_timing.items()},
            **dict(route_counts),
            "total_wall_seconds": round(total_wall, 6),
            "per_docx_average_seconds": round(total_wall / len(docx_manifest), 6)
            if docx_manifest
            else 0.0,
            "per_candidate_average_seconds": round(total_wall / len(candidates), 9)
            if candidates
            else 0.0,
        },
        "audit_renders": {
            "saved": len(audit_records),
            "records": audit_records,
        },
        "conservation": {
            "global_ok": global_ok,
            "failed_docx_count_matches": len(failures)
            == sum(row["status"] != "SUCCESS" for row in docx_records),
            "ocr_definition_violations": ocr_violations,
            "preview_suppression_violations": suppression_violations,
            "all_final_states_known": all(
                row["final_state"] in FINAL_STATES for row in candidates
            ),
        },
    }


def _state_metric(candidates, state):
    rows = [row for row in candidates if row["final_state"] == state]
    return {
        "occurrences": len(rows),
        "unique": len({row["candidate_fingerprint"] for row in rows}),
    }


def _distribution(values):
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    return {
        "min": min(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": max(ordered),
    }


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
