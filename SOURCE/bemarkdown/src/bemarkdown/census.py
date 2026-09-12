from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .formula import FormulaConversion, convert_mathml, convert_mtef_payload
from .validator import FormulaStructuralValidator, FormulaVerdict
from .wmf import WmfInspector
from .wmf_renderer import WmfRenderer


@dataclass(frozen=True)
class CensusResult:
    output_dir: Path
    summary_path: Path
    records_path: Path
    ocr_candidates_path: Path
    structural_rejected_path: Path
    renderer_issues_path: Path
    orphan_path: Path
    audit_dir: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class SampleRegressionResult:
    output_dir: Path
    summary_path: Path
    summary: dict[str, Any]


class _AuditSampler:
    def __init__(self, output_dir: Path, limit: int):
        self.output_dir = output_dir
        self.limit = max(0, limit)
        self.saved = 0
        self.counts: Counter[str] = Counter()
        self.quotas = {
            "Equation_DSMT4_MathType": 8,
            "Equation_KSEE3": 8,
            "Equation_3": 8,
            "structural_rejected": 10,
            "very_wide": 6,
            "very_tall": 6,
            "small_symbol": 8,
            "complex_fraction_or_script": 8,
            "renderer_boundary": 8,
            "general_rendered": 20,
        }

    def retain(self, record: dict[str, Any], temporary_png: Path) -> str | None:
        if self.saved >= self.limit or not temporary_png.is_file():
            return None
        strata = self._strata(record)
        eligible = [
            name for name in strata if self.counts[name] < self.quotas.get(name, 0)
        ]
        if not eligible:
            return None
        filename = f"{self.saved + 1:03d}_{record['wmf_sha256'][:16]}.png"
        target = self.output_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temporary_png, target)
        self.saved += 1
        for name in eligible:
            self.counts[name] += 1
        record["audit_strata"] = strata
        return f"audit_renders/{filename}"

    @staticmethod
    def _strata(record: dict[str, Any]) -> list[str]:
        strata = ["general_rendered"]
        strata.extend(record["source_categories"])
        if record.get("validator_verdict") not in {None, "VALID"}:
            strata.append("structural_rejected")
        latex = record.get("raw_latex") or ""
        if any(token in latex for token in (r"\frac", r"\sqrt", "_", "^")):
            strata.append("complex_fraction_or_script")
        metadata = record.get("render_metadata") or {}
        width, height = metadata.get("final_size") or (0, 0)
        if width and height:
            ratio = width / height
            if ratio >= 4:
                strata.append("very_wide")
            if ratio <= 0.5:
                strata.append("very_tall")
            if max(width, height) <= 120:
                strata.append("small_symbol")
            if ratio >= 8 or ratio <= 0.25:
                strata.append("renderer_boundary")
        return sorted(set(strata))


def run_wmf_census(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    audit_limit: int = 64,
    renderer: WmfRenderer | None = None,
    inspector: WmfInspector | None = None,
    validator: FormulaStructuralValidator | None = None,
) -> CensusResult:
    """Run a deterministic, SHA-deduplicated WMF census without OCR."""

    started = time.perf_counter()
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audit_renders"
    temporary_dir = output_dir / ".render_tmp"
    inspector = inspector or WmfInspector()
    renderer = renderer or WmfRenderer()
    validator = validator or FormulaStructuralValidator()
    sampler = _AuditSampler(audit_dir, audit_limit)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    occurrences = manifest.get("wmf_records")
    if not isinstance(occurrences, list):
        raise TypeError("Manifest has no wmf_records array")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verified_bytes: dict[str, bytes] = {}
    verify_started = time.perf_counter()
    for occurrence in occurrences:
        sha = occurrence.get("wmf_sha256")
        relative = occurrence.get("target_relative_path")
        if not sha or not relative:
            raise ValueError("WMF manifest record is missing hash or target path")
        source = manifest_path.parent / Path(relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        data = source.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha:
            raise ValueError(f"WMF SHA mismatch: {source} expected={sha} actual={actual}")
        groups[sha].append(occurrence)
        verified_bytes.setdefault(sha, data)
    manifest_verify_seconds = time.perf_counter() - verify_started

    timings: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for sha in sorted(groups):
        source_rows = groups[sha]
        data = verified_bytes[sha]
        referenced_rows = [row for row in source_rows if not _is_orphan(row)]
        orphan_rows = [row for row in source_rows if _is_orphan(row)]
        categories = sorted({row.get("status_directory") or row.get("wmf_status") for row in source_rows})

        inspect_started = time.perf_counter()
        inspection = inspector.inspect(data)
        timings["inspect_seconds"] += time.perf_counter() - inspect_started
        record: dict[str, Any] = {
            "wmf_sha256": sha,
            "size_bytes": len(data),
            "occurrence_count": len(source_rows),
            "referenced_occurrence_count": len(referenced_rows),
            "orphan_occurrence_count": len(orphan_rows),
            "source_categories": categories,
            "referenced": bool(referenced_rows),
            "orphan": bool(orphan_rows),
            "linked_progids": sorted({row.get("progid") for row in referenced_rows if row.get("progid")}),
            "linked_ole_objects": _unique_ole_objects(source_rows),
            "source_references": [_source_reference(row) for row in source_rows],
            "inspection": inspection.to_dict(),
            "visual_state": inspection.visual_state,
            "embedded_payload_type": None,
            "structural_conversion_status": "not_attempted",
            "raw_semantic_payload": None,
            "raw_mathml": None,
            "raw_latex": None,
            "conversion_error": None,
            "validator_verdict": None,
            "validator_issues": [],
            "validator_evidence": None,
            "render_attempted": False,
            "render_status": None,
            "render_metadata": None,
            "needs_math_ocr": False,
            "final_state": None,
            "audit_render": None,
        }

        conversion: FormulaConversion | None = None
        validation = None
        if inspection.visual_state == "empty":
            record["final_state"] = "empty"
        else:
            recovery_started = time.perf_counter()
            if inspection.embedded_mathml is not None:
                record["embedded_payload_type"] = "MathML"
                record["raw_semantic_payload"] = inspection.embedded_mathml
                conversion = convert_mathml(
                    inspection.embedded_mathml,
                    component="WMF MFCOMMENT MathML + mathml2latex 0.2.12",
                )
            elif inspection.embedded_mtef is not None:
                record["embedded_payload_type"] = "MTEF"
                record["raw_semantic_payload"] = inspection.embedded_mtef.hex()
                conversion = convert_mtef_payload(inspection.embedded_mtef)
            timings["structural_recovery_seconds"] += time.perf_counter() - recovery_started

            if conversion is not None:
                record["structural_conversion_status"] = conversion.status
                record["raw_mathml"] = conversion.intermediate
                record["raw_latex"] = conversion.latex
                record["conversion_error"] = conversion.error
                validation_started = time.perf_counter()
                validation = validator.validate(
                    conversion,
                    source_metadata={
                        "referenced": bool(referenced_rows),
                        "orphan": bool(orphan_rows) and not referenced_rows,
                        "source_categories": categories,
                    },
                )
                timings["validation_seconds"] += time.perf_counter() - validation_started
                record["validator_verdict"] = validation.verdict.value
                record["validator_issues"] = validation.to_dict()["issues"]
                record["validator_evidence"] = validation.evidence

            if validation is not None and validation.verdict == FormulaVerdict.VALID:
                record["final_state"] = "structural_valid"
            else:
                _render_for_census(record, data, temporary_dir, renderer, timings, sampler)
                if record["render_status"] == "success":
                    record["final_state"] = (
                        "structural_rejected_rendered"
                        if conversion is not None
                        else "no_structural_payload_rendered"
                    )
                elif inspection.visual_state == "uncertain":
                    record["final_state"] = "inspection_uncertain"
                elif record["render_status"] == "unsupported_geometry":
                    record["final_state"] = "unsupported_geometry"
                elif record["render_status"] == "empty":
                    record["final_state"] = "empty_render"
                else:
                    record["final_state"] = "renderer_failed"
                record["needs_math_ocr"] = bool(
                    referenced_rows
                    and inspection.visual_state == "visible"
                    and record["render_status"] == "success"
                )
        rows.append(record)

    shutil.rmtree(temporary_dir, ignore_errors=True)
    timings["total_seconds"] = time.perf_counter() - started
    timings["manifest_verify_seconds"] = manifest_verify_seconds
    summary = _build_summary(rows, len(occurrences), len(groups), timings, sampler)
    paths = {
        "summary": output_dir / "census_summary.json",
        "records": output_dir / "census_records.jsonl",
        "ocr": output_dir / "ocr_candidates.jsonl",
        "rejected": output_dir / "structural_rejected.jsonl",
        "renderer": output_dir / "renderer_issues.jsonl",
        "orphan": output_dir / "orphan_analysis.jsonl",
    }
    _write_json(paths["summary"], summary)
    _write_jsonl(paths["records"], rows)
    _write_jsonl(paths["ocr"], [row for row in rows if row["needs_math_ocr"]])
    _write_jsonl(
        paths["rejected"],
        [
            row
            for row in rows
            if row["embedded_payload_type"]
            and row["validator_verdict"] not in {None, FormulaVerdict.VALID.value}
        ],
    )
    _write_jsonl(
        paths["renderer"],
        [
            row
            for row in rows
            if (row["render_attempted"] and row["render_status"] != "success")
            or row["visual_state"] == "uncertain"
        ],
    )
    _write_jsonl(paths["orphan"], [row for row in rows if row["orphan"]])
    return CensusResult(
        output_dir,
        paths["summary"],
        paths["records"],
        paths["ocr"],
        paths["rejected"],
        paths["renderer"],
        paths["orphan"],
        audit_dir,
        summary,
    )


def revalidate_wmf_samples(
    phase2_summary_path: str | Path,
    output_dir: str | Path,
    *,
    renderer: WmfRenderer | None = None,
    inspector: WmfInspector | None = None,
    validator: FormulaStructuralValidator | None = None,
) -> SampleRegressionResult:
    """Re-run Phase 2 representative WMFs and compare old/new routes."""

    started = time.perf_counter()
    phase2_summary_path = Path(phase2_summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    renders = output_dir / "renders"
    renderer = renderer or WmfRenderer()
    inspector = inspector or WmfInspector()
    validator = validator or FormulaStructuralValidator()
    old = json.loads(phase2_summary_path.read_text(encoding="utf-8"))
    old_records = old.get("samples") or old.get("records")
    if not isinstance(old_records, list):
        raise TypeError("Phase 2 sample summary has no samples array")

    records = []
    for index, previous in enumerate(old_records, start=1):
        source = Path(previous["source"])
        data = source.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if previous.get("sha256") and sha != previous["sha256"]:
            raise ValueError(f"Sample SHA mismatch: {source}")
        inspection = inspector.inspect(data)
        conversion = None
        payload_type = None
        if inspection.visual_state != "empty":
            if inspection.embedded_mathml is not None:
                payload_type = "MathML"
                conversion = convert_mathml(
                    inspection.embedded_mathml,
                    component="WMF MFCOMMENT MathML + mathml2latex 0.2.12",
                )
            elif inspection.embedded_mtef is not None:
                payload_type = "MTEF"
                conversion = convert_mtef_payload(inspection.embedded_mtef)
        referenced = previous.get("status_class") != "ORPHAN_UNREFERENCED"
        validation = (
            validator.validate(
                conversion,
                source_metadata={
                    "referenced": referenced,
                    "source_category": previous.get("status_class"),
                },
            )
            if conversion is not None
            else None
        )
        render_metadata = None
        render_path = None
        if inspection.visual_state == "empty":
            new_classification = "empty"
            final_route = "suppress"
        elif validation is not None and validation.accepted:
            new_classification = "structural_valid"
            final_route = "latex"
        else:
            target = renders / f"{index:02d}_{sha[:16]}.png"
            render_metadata = renderer.render_bytes(data, target)
            if render_metadata.status == "success":
                render_path = str(target.relative_to(output_dir)).replace("\\", "/")
                new_classification = (
                    "structural_rejected_rendered"
                    if conversion is not None
                    else "rendered"
                )
                final_route = "renderer"
            else:
                target.unlink(missing_ok=True)
                new_classification = (
                    f"structural_{validation.verdict.value.lower()}_"
                    f"renderer_{render_metadata.status}"
                    if validation is not None
                    else f"renderer_{render_metadata.status}"
                )
                final_route = "renderer_issue"
        records.append(
            {
                "source": str(source),
                "status_class": previous.get("status_class"),
                "sha256": sha,
                "old_classification": previous.get("route"),
                "new_classification": new_classification,
                "embedded_payload_type": payload_type,
                "raw_latex": conversion.latex if conversion else None,
                "raw_mathml": conversion.intermediate if conversion else None,
                "validator_verdict": validation.verdict.value if validation else None,
                "validator_issues": validation.to_dict()["issues"] if validation else [],
                "final_route": final_route,
                "render": render_metadata.to_dict() if render_metadata else None,
                "render_path": render_path,
            }
        )
    counts = Counter(record["new_classification"] for record in records)
    summary = {
        "schema": "bemarkdown-phase25-sample-regression-v1",
        "source_phase2_summary": str(phase2_summary_path),
        "sample_count": len(records),
        "new_classification_counts": dict(sorted(counts.items())),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "records": records,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    return SampleRegressionResult(output_dir, summary_path, summary)


def _render_for_census(record, data, temporary_dir, renderer, timings, sampler):
    record["render_attempted"] = True
    temporary = temporary_dir / f"{record['wmf_sha256']}.png"
    render_started = time.perf_counter()
    metadata = renderer.render_bytes(data, temporary)
    timings["render_seconds"] += time.perf_counter() - render_started
    record["render_status"] = metadata.status
    record["render_metadata"] = metadata.to_dict()
    if metadata.status == "success":
        record["render_png_sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
        record["audit_render"] = sampler.retain(record, temporary)
    temporary.unlink(missing_ok=True)


def _build_summary(rows, occurrence_total, unique_total, timings, sampler):
    def scope(predicate, occurrence_field):
        selected = [row for row in rows if predicate(row)]
        return {
            "occurrences": sum(row[occurrence_field] for row in selected),
            "unique": len(selected),
        }

    def state_scope(predicate, state_predicate, occurrence_field="occurrence_count"):
        selected = [row for row in rows if predicate(row) and state_predicate(row)]
        occurrence_count = sum(row[occurrence_field] for row in selected)
        denominator_occ = sum(row[occurrence_field] for row in rows if predicate(row))
        denominator_unique = sum(1 for row in rows if predicate(row))
        return _metric(occurrence_count, len(selected), denominator_occ, denominator_unique)

    referenced = lambda row: row["referenced"]
    orphan = lambda row: row["orphan"]
    all_rows = lambda row: True
    final_states = sorted({row["final_state"] for row in rows})
    category_names = sorted({category for row in rows for category in row["source_categories"]})
    referenced_totals = scope(referenced, "referenced_occurrence_count")
    orphan_totals = scope(orphan, "orphan_occurrence_count")
    return {
        "schema": "bemarkdown-wmf-census-phase25-v1",
        "total": {"occurrences": occurrence_total, "unique": unique_total},
        "categories": {
            category: _metric(
                sum(
                    sum(1 for ref in row["source_references"] if ref["source_category"] == category)
                    for row in rows
                ),
                sum(1 for row in rows if category in row["source_categories"]),
                occurrence_total,
                unique_total,
            )
            for category in category_names
        },
        "final_states": {
            state: state_scope(all_rows, lambda row, state=state: row["final_state"] == state)
            for state in final_states
        },
        "referenced": {
            **referenced_totals,
            "denominator_note": "Percentages use referenced occurrence and referenced unique-SHA totals.",
            "empty": state_scope(referenced, lambda row: row["final_state"] == "empty", "referenced_occurrence_count"),
            "valid_structural": state_scope(referenced, lambda row: row["final_state"] == "structural_valid", "referenced_occurrence_count"),
            "structural_rejected": state_scope(referenced, lambda row: row["embedded_payload_type"] is not None and row["validator_verdict"] != "VALID", "referenced_occurrence_count"),
            "renderer_attempted": state_scope(referenced, lambda row: row["render_attempted"], "referenced_occurrence_count"),
            "renderer_success": state_scope(referenced, lambda row: row["render_status"] == "success", "referenced_occurrence_count"),
            "renderer_failed": state_scope(referenced, lambda row: row["render_status"] == "failed", "referenced_occurrence_count"),
            "unsupported_geometry": state_scope(referenced, lambda row: row["render_status"] == "unsupported_geometry", "referenced_occurrence_count"),
            "empty_render": state_scope(referenced, lambda row: row["render_status"] == "empty", "referenced_occurrence_count"),
            "inspection_uncertain": state_scope(referenced, lambda row: row["visual_state"] == "uncertain", "referenced_occurrence_count"),
            "true_ocr_candidates": state_scope(referenced, lambda row: row["needs_math_ocr"], "referenced_occurrence_count"),
        },
        "orphan": {
            **orphan_totals,
            "denominator_note": "Percentages use orphan occurrence and orphan unique-SHA totals; OCR is always false.",
            "empty": state_scope(orphan, lambda row: row["final_state"] == "empty", "orphan_occurrence_count"),
            "semantic_fragment": state_scope(orphan, lambda row: row["validator_verdict"] == "NON_FORMULA_CONTENT", "orphan_occurrence_count"),
            "visible": state_scope(orphan, lambda row: row["visual_state"] == "visible", "orphan_occurrence_count"),
            "renderable": state_scope(orphan, lambda row: row["render_status"] == "success", "orphan_occurrence_count"),
            "other": state_scope(orphan, lambda row: row["final_state"] not in {"empty", "structural_valid", "structural_rejected_rendered", "no_structural_payload_rendered"}, "orphan_occurrence_count"),
        },
        "structural": {
            "detected": state_scope(all_rows, lambda row: row["embedded_payload_type"] is not None),
            "valid": state_scope(all_rows, lambda row: row["validator_verdict"] == "VALID"),
            "suspicious": state_scope(all_rows, lambda row: row["validator_verdict"] == "SUSPICIOUS"),
            "invalid": state_scope(all_rows, lambda row: row["validator_verdict"] == "INVALID"),
            "non_formula": state_scope(all_rows, lambda row: row["validator_verdict"] == "NON_FORMULA_CONTENT"),
        },
        "performance": {
            **{key: round(value, 6) for key, value in timings.items()},
            "render_count": sum(1 for row in rows if row["render_attempted"]),
            "sha_cache_hits": occurrence_total - unique_total,
            "unique_wmf_count": unique_total,
        },
        "audit_renders": {
            "saved": sampler.saved,
            "limit": sampler.limit,
            "strata_counts": dict(sorted(sampler.counts.items())),
        },
        "conservation": {
            "occurrence_ok": occurrence_total == sum(
                value["occurrences"] for value in (
                    _metric(
                        sum(row["occurrence_count"] for row in rows if row["final_state"] == state),
                        sum(1 for row in rows if row["final_state"] == state),
                        occurrence_total,
                        unique_total,
                    )
                    for state in final_states
                )
            ),
            "unique_ok": unique_total == sum(1 for row in rows if row["final_state"] in final_states),
        },
    }


def _metric(occurrences, unique, occurrence_denominator, unique_denominator):
    return {
        "occurrences": occurrences,
        "unique": unique,
        "occurrence_percentage": round(100 * occurrences / occurrence_denominator, 6) if occurrence_denominator else 0.0,
        "unique_percentage": round(100 * unique / unique_denominator, 6) if unique_denominator else 0.0,
        "occurrence_denominator": occurrence_denominator,
        "unique_denominator": unique_denominator,
    }


def _is_orphan(row):
    return (row.get("status_directory") == "Orphan_Unreferenced" or row.get("wmf_status") == "ORPHAN_UNREFERENCED")


def _source_reference(row):
    return {
        "source_docx": row.get("source_docx"),
        "source_docx_relative_path": row.get("source_docx_relative_path"),
        "wmf_internal_path": row.get("wmf_internal_path"),
        "target_relative_path": row.get("target_relative_path"),
        "source_category": row.get("status_directory") or row.get("wmf_status"),
        "progid": row.get("progid"),
        "relationship_parts": row.get("relationship_parts") or [],
    }


def _unique_ole_objects(rows):
    objects = {}
    for row in rows:
        for item in row.get("ole_objects") or []:
            key = (item.get("internal_path"), item.get("sha256"))
            objects[key] = item
    return [objects[key] for key in sorted(objects)]


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
