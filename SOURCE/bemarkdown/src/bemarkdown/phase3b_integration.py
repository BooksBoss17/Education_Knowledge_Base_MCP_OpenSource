from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .formula_ocr import (
    FORMULANET_MODEL_REVISION,
    FormulaOcrAdapter,
    FormulaOcrSafetyGate,
    SafetyGateVerdict,
    adapter_stats_snapshot,
    select_production_batch,
)
from .formulanet_benchmark import load_benchmark_dataset
from .formulanet_runtime import (
    FormulaNetRuntime,
    FormulaOcrOutputValidator,
    PaddleFormulaNetRuntime,
)
from .pipeline import convert_docx


@dataclass(frozen=True)
class BatchStabilityResult:
    summary: dict[str, Any]
    summary_path: Path


@dataclass(frozen=True)
class ReferenceGateResult:
    summary: dict[str, Any]
    summary_path: Path
    records_path: Path


@dataclass(frozen=True)
class IntegrationResult:
    summary: dict[str, Any]
    summary_path: Path
    docx_records_path: Path
    ocr_records_path: Path
    performance_path: Path


def run_batch2_stability(
    phase3a_dir: str | Path,
    output_dir: str | Path,
    *,
    runtime_factory: Callable[[], FormulaNetRuntime] | None = None,
) -> BatchStabilityResult:
    phase3a_dir = Path(phase3a_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = phase3a_dir / "formula_predictions.jsonl"
    baseline_rows = _read_jsonl(predictions_path)
    if len(baseline_rows) != 157:
        raise ValueError("Frozen Phase 3A baseline must contain 157 predictions")
    baseline = {
        row["png_content_sha256"]: row["formula_net_raw_latex"]
        for row in baseline_rows
    }
    if len(baseline) != 157:
        raise ValueError("Frozen Phase 3A baseline PNG SHA values are not unique")
    dataset = load_benchmark_dataset(
        phase3a_dir.parent / "phase26_docx_formula_census_v2"
    )
    if dataset.unique_png_count != 157:
        raise ValueError("Frozen Phase 2.6 input must contain 157 unique PNGs")
    runtime_factory = runtime_factory or _runtime_factory_from_environment
    runtime = runtime_factory()
    current_fingerprint = runtime.fingerprint()
    frozen_fingerprint = json.loads(
        (phase3a_dir / "runtime_fingerprint.json").read_text(encoding="utf-8")
    )
    fingerprint_checks = {
        key: current_fingerprint.get(key) == frozen_fingerprint.get(key)
        for key in (
            "model_identifier",
            "model_revision",
            "model_sha256",
            "device",
            "precision",
        )
    }
    if not all(fingerprint_checks.values()):
        raise ValueError(
            "FormulaNet runtime does not match the frozen Phase 3A fingerprint: "
            + json.dumps(fingerprint_checks, sort_keys=True)
        )

    repeated_runs = []
    run_records = []
    for run_index in range(1, 4):
        started = time.perf_counter()
        outputs = runtime.predict(
            [image.path for image in dataset.images],
            batch_size=2,
        )
        seconds = time.perf_counter() - started
        if len(outputs) != 157:
            raise RuntimeError(
                f"Batch 2 stability run {run_index} returned {len(outputs)} outputs"
            )
        current = {
            image.png_content_sha256: raw
            for image, raw in zip(dataset.images, outputs, strict=True)
        }
        repeated_runs.append(current)
        differences = [
            png_sha
            for png_sha, raw in baseline.items()
            if current.get(png_sha) != raw
        ]
        run_records.append(
            {
                "run": run_index,
                "batch_size": 2,
                "images": len(outputs),
                "seconds": seconds,
                "throughput_images_per_second": len(outputs) / seconds,
                "difference_from_batch1_count": len(differences),
                "difference_png_sha256": differences,
                "raw_output_set_sha256": _mapping_sha256(current),
            }
        )
    production_batch = select_production_batch(baseline, repeated_runs)
    summary = {
        "schema": "bemarkdown-phase3b-batch-stability-v1",
        "phase3a_predictions_sha256": _sha256_file(predictions_path),
        "dataset_integrity": dataset.integrity,
        "runtime_fingerprint": current_fingerprint,
        "frozen_fingerprint_checks": fingerprint_checks,
        "runs": run_records,
        "all_157_x_3_match_batch1": production_batch == 2,
        "production_candidate_batch_size": production_batch,
        "fallback_reason": (
            None
            if production_batch == 2
            else "At least one Batch 2 raw output differed from frozen Batch 1."
        ),
    }
    summary_path = output_dir / "batch2_stability.json"
    _write_json(summary_path, summary)
    return BatchStabilityResult(summary, summary_path)


def evaluate_reference_safety_gate(
    reference_path: str | Path,
    output_dir: str | Path,
) -> ReferenceGateResult:
    reference_path = Path(reference_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(reference_path)
    if len(rows) != 157:
        raise ValueError("Phase 3A.5 reference GT must contain 157 records")
    validator = FormulaOcrOutputValidator()
    gate = FormulaOcrSafetyGate()
    records = []
    decision_counts = Counter()
    recognition_by_decision: dict[str, Counter[str]] = {}
    for row in rows:
        validation = validator.validate(row["formula_net_raw_latex"])
        decision = gate.evaluate(
            width=row["width"],
            height=row["height"],
            raw_latex=row["formula_net_raw_latex"],
            validation=validation,
        )
        decision_counts[decision.verdict.value] += 1
        recognition_by_decision.setdefault(
            decision.verdict.value, Counter()
        )[row["recognition_status"]] += 1
        records.append(
            {
                "review_index": row["review_index"],
                "png_content_sha256": row["png_content_sha256"],
                "width": row["width"],
                "height": row["height"],
                "raw_latex": row["formula_net_raw_latex"],
                "source_content_status": row["source_content_status"],
                "reference_recognition_status": row["recognition_status"],
                "ocr_validator": validation.to_dict(),
                "safety_gate": decision.to_dict(),
            }
        )
    automatic = {
        SafetyGateVerdict.ACCEPT.value,
        SafetyGateVerdict.ACCEPT_WITH_WARNING.value,
    }
    accepted_records = [
        row for row in records if row["safety_gate"]["verdict"] in automatic
    ]
    unsafe = [
        row
        for row in accepted_records
        if row["reference_recognition_status"] in {"MAJOR_ERROR", "HALLUCINATION"}
    ]
    minor_accepted = [
        row
        for row in accepted_records
        if row["reference_recognition_status"] == "MINOR_ERROR"
    ]
    visual_success_accepted = [
        row
        for row in accepted_records
        if row["reference_recognition_status"]
        in {"EXACT", "SEMANTICALLY_CORRECT_STYLE_DIFFERENT"}
    ]
    non_formula = [
        row
        for row in records
        if row["source_content_status"] == "NON_FORMULA_VISIBLE_CONTENT"
    ]
    catastrophic = next(row for row in records if row["review_index"] == 85)
    summary = {
        "schema": "bemarkdown-phase3b-reference-gate-v1",
        "reference_gt_sha256": _sha256_file(reference_path),
        "records": len(records),
        "evaluable_formula_or_symbol": sum(
            row["reference_recognition_status"] != "NOT_APPLICABLE"
            for row in records
        ),
        "decisions": {
            verdict.value: decision_counts[verdict.value]
            for verdict in SafetyGateVerdict
        },
        "recognition_status_by_decision": {
            decision: dict(sorted(counter.items()))
            for decision, counter in sorted(recognition_by_decision.items())
        },
        "automatic_accept_path": len(accepted_records),
        "unsafe_major_or_hallucination_accepted": len(unsafe),
        "minor_error_accepted": len(minor_accepted),
        "visual_success_accepted": len(visual_success_accepted),
        "non_formula": {
            "total": len(non_formula),
            "accepted": sum(
                row["safety_gate"]["verdict"] == SafetyGateVerdict.ACCEPT.value
                for row in non_formula
            ),
            "accepted_with_warning": sum(
                row["safety_gate"]["verdict"]
                == SafetyGateVerdict.ACCEPT_WITH_WARNING.value
                for row in non_formula
            ),
            "review_or_rejected": sum(
                row["safety_gate"]["verdict"] not in automatic
                for row in non_formula
            ),
        },
        "catastrophic_085": {
            "decision": catastrophic["safety_gate"]["verdict"],
            "reason_codes": [
                reason["code"]
                for reason in catastrophic["safety_gate"]["reasons"]
            ],
            "blocked_from_automatic_accept": (
                catastrophic["safety_gate"]["verdict"] not in automatic
            ),
        },
        "gate_pass": len(unsafe) == 0
        and catastrophic["safety_gate"]["verdict"] not in automatic,
        "scope_note": (
            "Reference GT is used only by this offline evaluator. "
            "FormulaOcrAdapter and FormulaOcrSafetyGate do not read GT."
        ),
    }
    summary_path = output_dir / "reference_gate_summary.json"
    records_path = output_dir / "reference_gate_records.jsonl"
    _write_json(summary_path, summary)
    _write_jsonl(records_path, records)
    return ReferenceGateResult(summary, summary_path, records_path)


def run_formula_ocr_integration(
    manifest_path: str | Path,
    phase26_dir: str | Path,
    output_dir: str | Path,
    *,
    adapter: FormulaOcrAdapter,
    sample_sources: list[str | Path] | None = None,
) -> IntegrationResult:
    started = time.perf_counter()
    manifest_path = Path(manifest_path).resolve()
    phase26_dir = Path(phase26_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "integration_summary.json",
        "docx_records.jsonl",
        "ocr_records.jsonl",
        "safety_gate_records.jsonl",
        "review_items.jsonl",
        "rejected_items.jsonl",
        "performance.json",
    ):
        if (output_dir / name).exists():
            raise FileExistsError(f"Phase 3B output already exists: {output_dir / name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_records = [
        row
        for row in manifest.get("records", [])
        if str(row.get("file_type", "")).upper() == "DOCX"
    ]
    if len(manifest_records) != 169:
        raise ValueError(f"Expected 169 DOCX records, found {len(manifest_records)}")
    phase26_summary = json.loads(
        (phase26_dir / "census_summary.json").read_text(encoding="utf-8")
    )
    sample_set = {str(Path(path).resolve()) for path in (sample_sources or [])}
    work_dir = (output_dir / ".work").resolve()
    sample_dir = (output_dir / "sample_documents").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    docx_records = []
    ocr_records = []
    failures = []
    stage_timing = Counter()
    route_counts = Counter()
    formula_totals = Counter()
    asset_links = Counter()
    retained_samples = []

    for index, manifest_record in enumerate(manifest_records, start=1):
        source = (
            manifest_path.parent / Path(manifest_record["target_relative_path"])
        ).resolve()
        expected_sha = manifest_record.get("source_sha256")
        is_sample = str(source) in sample_set
        destination = (
            sample_dir / f"{index:04d}_{_safe_name(source.stem)}"
            if is_sample
            else work_dir / f"{index:04d}"
        ).resolve()
        allowed_root = sample_dir if is_sample else work_dir
        destination.relative_to(allowed_root)
        base = {
            "docx_index": index,
            "source_file": str(source),
            "source_relative_path": manifest_record["target_relative_path"],
            "expected_sha256": expected_sha,
            "sample_retained": is_sample,
        }
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            actual_sha = _sha256_file(source)
            if expected_sha and actual_sha != expected_sha:
                raise ValueError("Source SHA-256 does not match manifest")
            result = convert_docx(
                source,
                destination,
                formula_ocr="auto",
                formula_ocr_adapter=adapter,
            )
            report = result.report
            for key, value in report["timing"].items():
                if isinstance(value, int | float):
                    stage_timing[key] += value
            formulas = report["formulas"]
            formula_ocr = report["formula_ocr"]
            formula_totals.update(
                {
                    "equation_candidates": formulas["equation_candidates"]
                    - formulas["non_formula_objects"],
                    "real_formulas": formulas["real_formulas"],
                    "empty_suppressed": formulas["empty_placeholders_suppressed"],
                    "structural_final": formulas["structural_success"],
                    "other_failed_preserved": formulas["failed_preserved"],
                    "ocr_candidates": formula_ocr["candidates"],
                    "ocr_accepted": formula_ocr["accepted"],
                    "ocr_accepted_with_warning": formula_ocr[
                        "accepted_with_warning"
                    ],
                    "ocr_review_required": formula_ocr["review_required"],
                    "ocr_rejected_preserved": formula_ocr["rejected_preserved"],
                    "ocr_inference_failed_preserved": formula_ocr[
                        "inference_failed_preserved"
                    ],
                }
            )
            for candidate in formulas["candidate_records"]:
                state = _candidate_final_state(candidate)
                if state:
                    route_counts[state] += 1
            local_ocr = [
                {
                    **row,
                    "docx_index": index,
                    "source_file": str(source),
                    "source_relative_path": manifest_record[
                        "target_relative_path"
                    ],
                }
                for row in formula_ocr["records"]
            ]
            ocr_records.extend(local_ocr)
            link_result = _validate_markdown_assets(
                result.markdown_path, result.output_dir
            )
            asset_links.update(link_result)
            if is_sample:
                retained_samples.append(
                    {
                        "docx_index": index,
                        "source_file": str(source),
                        "output": destination.relative_to(output_dir).as_posix(),
                        "quality_status": report["quality_status"],
                        "formula_ocr": formula_ocr,
                        "asset_links": link_result,
                    }
                )
            docx_records.append(
                {
                    **base,
                    "status": "SUCCESS",
                    "actual_sha256": actual_sha,
                    "quality_status": report["quality_status"],
                    "formulas": {
                        "equation_candidates": formulas["equation_candidates"],
                        "real_formulas": formulas["real_formulas"],
                        "empty_suppressed": formulas[
                            "empty_placeholders_suppressed"
                        ],
                        "structural_final": formulas["structural_success"],
                        "other_failed_preserved": formulas["failed_preserved"],
                    },
                    "formula_ocr": formula_ocr,
                    "formula_conservation_ok": formulas["conservation_ok"]
                    and formula_ocr["conservation_ok"]
                    and formula_ocr["real_formula_conservation_ok"],
                    "asset_links": link_result,
                    "timing": report["timing"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate a DOCX regression case
            failure = {
                **base,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            docx_records.append(failure)
        finally:
            if not is_sample and destination.is_dir():
                destination.relative_to(work_dir)
                shutil.rmtree(destination)
    work_dir.relative_to(output_dir)
    shutil.rmtree(work_dir, ignore_errors=True)

    state_total = sum(
        formula_totals[key]
        for key in (
            "structural_final",
            "ocr_accepted",
            "ocr_accepted_with_warning",
            "ocr_review_required",
            "ocr_rejected_preserved",
            "ocr_inference_failed_preserved",
            "other_failed_preserved",
        )
    )
    expected_routes = {
        state: phase26_summary["final_states"][state]["occurrences"]
        for state in (
            "STRUCTURAL_OMML",
            "STRUCTURAL_EQ",
            "STRUCTURAL_LATEX_ALT",
            "STRUCTURAL_OLE_MTEF",
            "STRUCTURAL_WMF_MTEF",
            "STRUCTURAL_WMF_MATHML",
        )
    }
    route_checks = {
        state: route_counts[state] == expected for state, expected in expected_routes.items()
    }
    source_hashes_valid = (
        not failures
        and all(row["status"] == "SUCCESS" for row in docx_records)
        and len(docx_records) == 169
    )
    conservation_ok = (
        formula_totals["real_formulas"] == state_total
        and formula_totals["equation_candidates"]
        == formula_totals["real_formulas"] + formula_totals["empty_suppressed"]
        and all(
            row.get("formula_conservation_ok", False)
            for row in docx_records
            if row["status"] == "SUCCESS"
        )
    )
    phase26_checks = {
        "source_docx_169": len(docx_records) == 169,
        "all_source_sha_valid": source_hashes_valid,
        "equation_candidates_35920": (
            formula_totals["equation_candidates"]
            == phase26_summary["candidates"]["occurrences"]
            == 35_920
        ),
        "real_formulas_35892": (
            formula_totals["real_formulas"]
            == phase26_summary["candidates"]["real_formula_occurrences"]
            == 35_892
        ),
        "empty_suppressed_28": (
            formula_totals["empty_suppressed"]
            == phase26_summary["candidates"]["empty_occurrences"]
            == 28
        ),
        "ocr_candidates_245": (
            formula_totals["ocr_candidates"]
            == phase26_summary["candidates"]["true_math_ocr"]["occurrences"]
            == 245
        ),
        "structural_routes_unchanged": all(route_checks.values()),
        "formula_conservation": conservation_ok,
        "sample_asset_links_valid": asset_links["missing"] == 0,
    }
    integration_summary = {
        "schema": "bemarkdown-phase3b-integration-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "phase26_census_sha256": _sha256_file(
            phase26_dir / "census_summary.json"
        ),
        "docx": {
            "total": len(manifest_records),
            "success": sum(row["status"] == "SUCCESS" for row in docx_records),
            "failed": len(failures),
            "source_sha_valid": sum(
                row["status"] == "SUCCESS" for row in docx_records
            ),
        },
        "formulas": dict(formula_totals),
        "structural_route_counts": {
            state: route_counts[state] for state in expected_routes
        },
        "structural_route_expected": expected_routes,
        "structural_route_checks": route_checks,
        "formula_conservation": {
            "real_formula_terminal_sum": state_total,
            "all_ok": conservation_ok,
        },
        "runtime": adapter_stats_snapshot(adapter),
        "sample_documents": retained_samples,
        "asset_links": dict(asset_links),
        "phase26_preservation_checks": phase26_checks,
        "all_ok": not failures and all(phase26_checks.values()),
    }
    total_wall = time.perf_counter() - started
    baseline_wall = phase26_summary["performance"]["total_wall_seconds"]
    performance = {
        "schema": "bemarkdown-phase3b-performance-v1",
        "total_wall_seconds": total_wall,
        "phase26_no_ocr_wall_seconds": baseline_wall,
        "gross_wall_difference_seconds": total_wall - baseline_wall,
        "comparison_caveat": (
            "Phase 2.6 used census analysis mode; Phase 3B also serialized full "
            "Markdown and assets. FormulaNet component timing is the cleaner OCR cost."
        ),
        "aggregate_document_stage_seconds": dict(stage_timing),
        "formula_ocr_runtime": adapter_stats_snapshot(adapter),
    }
    integration_summary["performance"] = performance

    safety_records = [
        {
            "formula_id": row["formula_id"],
            "docx_index": row["docx_index"],
            "source_file": row["source_file"],
            "png_sha256": row["png_sha256"],
            "state": row["state"],
            "safety_gate": row["safety_gate"],
        }
        for row in ocr_records
    ]
    review_items = [
        row for row in ocr_records if row["state"] == "OCR_REVIEW_REQUIRED"
    ]
    rejected_items = [
        row
        for row in ocr_records
        if row["state"]
        in {
            "OCR_REJECTED_PRESERVE_IMAGE",
            "OCR_INFERENCE_FAILED_PRESERVE_IMAGE",
        }
    ]
    paths = {
        "summary": output_dir / "integration_summary.json",
        "docx": output_dir / "docx_records.jsonl",
        "ocr": output_dir / "ocr_records.jsonl",
        "safety": output_dir / "safety_gate_records.jsonl",
        "review": output_dir / "review_items.jsonl",
        "rejected": output_dir / "rejected_items.jsonl",
        "performance": output_dir / "performance.json",
    }
    _write_jsonl(paths["docx"], docx_records)
    _write_jsonl(paths["ocr"], ocr_records)
    _write_jsonl(paths["safety"], safety_records)
    _write_jsonl(paths["review"], review_items)
    _write_jsonl(paths["rejected"], rejected_items)
    _write_json(paths["performance"], performance)
    _write_json(paths["summary"], integration_summary)
    return IntegrationResult(
        integration_summary,
        paths["summary"],
        paths["docx"],
        paths["ocr"],
        paths["performance"],
    )


def _candidate_final_state(candidate):
    classification = candidate.get("classification")
    status = candidate.get("status")
    structural_status = candidate.get("structural_status", status)
    source_type = candidate.get("source_type")
    if classification == "EMPTY_PLACEHOLDER":
        return "EMPTY_PLACEHOLDER_SUPPRESSED"
    if status and status.startswith("OCR_"):
        return status
    if structural_status in {
        "SUCCESS_EXACT",
        "SUCCESS_NORMALIZED",
        "SUCCESS_APPROXIMATE",
    }:
        return {
            "omml": "STRUCTURAL_OMML",
            "eq": "STRUCTURAL_EQ",
            "latex_alt": "STRUCTURAL_LATEX_ALT",
            "mtef": "STRUCTURAL_OLE_MTEF",
            "wmf_embedded": (
                "STRUCTURAL_WMF_MATHML"
                if candidate.get("source_payload_type") == "WMF_MATHML"
                else "STRUCTURAL_WMF_MTEF"
            ),
        }.get(source_type)
    return None


def _validate_markdown_assets(markdown_path, output_dir):
    markdown = markdown_path.read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]]*\]\(([^)\s]+)", markdown)
    present = 0
    missing = 0
    for link in links:
        target = (output_dir / unquote(link)).resolve()
        try:
            target.relative_to(output_dir.resolve())
        except ValueError:
            missing += 1
            continue
        if target.is_file():
            present += 1
        else:
            missing += 1
    return {"checked": len(links), "present": present, "missing": missing}


def _runtime_factory_from_environment():
    import os

    model_dir = os.environ.get("BEMARKDOWN_FORMULANET_MODEL_DIR")
    return PaddleFormulaNetRuntime(
        model_identifier="PP-FormulaNet_plus-L",
        model_dir=Path(model_dir) if model_dir else None,
        model_revision=FORMULANET_MODEL_REVISION,
        device="gpu:0",
    )


def _mapping_sha256(mapping):
    canonical = json.dumps(
        mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:80] or "document"


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


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
