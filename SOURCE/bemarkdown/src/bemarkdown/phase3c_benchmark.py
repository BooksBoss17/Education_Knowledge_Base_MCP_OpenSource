from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .docx_census import FINAL_STATES, _final_state
from .formula_ocr import FormulaOcrAdapter
from .mtef_cache import MtefCacheContext
from .package import PackageIndex
from .pipeline import (
    _finalize_formula_report,
    _finalize_mtef_cache_report,
    _new_report,
    convert_docx,
)
from .scanner import DocumentScanner


@dataclass(frozen=True)
class Phase3cBenchmarkResult:
    output_dir: Path
    profiling_summary_path: Path
    cache_statistics_path: Path
    regression_summary_path: Path
    output_diff_summary_path: Path
    summary: dict[str, Any]


def run_phase3c_benchmark(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    phase26_dir: str | Path,
    phase3b_dir: str | Path,
    persistent_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Phase3cBenchmarkResult:
    """Run the four isolated, single-worker Phase 3C benchmark modes."""

    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    phase26_dir = Path(phase26_dir).resolve()
    phase3b_dir = Path(phase3b_dir).resolve()
    persistent_dir = Path(
        persistent_dir
        if persistent_dir is not None
        else Path.home() / ".cache" / "bemarkdown" / "mtef"
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        row
        for row in manifest.get("records", [])
        if str(row.get("file_type", "")).upper() == "DOCX"
    ]
    if len(records) != 169:
        raise ValueError(f"Phase 3C requires the frozen 169 DOCX manifest, got {len(records)}")

    modes: dict[str, dict[str, Any]] = {}
    baseline_semantics = None
    baseline_ocr_ids = None
    contexts = {
        "baseline_no_cache": MtefCacheContext(mode="off"),
        "l1_cache": MtefCacheContext(mode="memory"),
    }
    cold_context = MtefCacheContext(
        mode="persistent", persistent_dir=persistent_dir
    )
    removed_entries = cold_context.clear_persistent()
    contexts["l2_cold"] = cold_context

    for name in ("baseline_no_cache", "l1_cache", "l2_cold"):
        if progress:
            progress(f"Phase 3C {name}: starting 169 DOCX")
        result = _run_mode(
            manifest_path,
            records,
            output_dir,
            name,
            contexts[name],
            progress=progress,
        )
        semantics = result.pop("_document_semantics")
        ocr_ids = result.pop("_ocr_candidate_ids")
        if baseline_semantics is None:
            baseline_semantics = semantics
            baseline_ocr_ids = ocr_ids
            result["semantic_differences_from_baseline"] = 0
        else:
            result["semantic_differences_from_baseline"] = _difference_count(
                baseline_semantics, semantics
            )
            result["ocr_route_differences_from_baseline"] = len(
                set(baseline_ocr_ids) ^ set(ocr_ids)
            )
        modes[name] = result
        _write_json(output_dir / f"{name}.json", result)
        if progress:
            progress(
                f"Phase 3C {name}: complete in {result['total_wall_seconds']:.3f}s"
            )

    warm_context = MtefCacheContext(
        mode="persistent", persistent_dir=persistent_dir
    )
    if progress:
        progress("Phase 3C l2_warm: starting 169 DOCX")
    warm = _run_mode(
        manifest_path,
        records,
        output_dir,
        "l2_warm",
        warm_context,
        progress=progress,
    )
    warm_semantics = warm.pop("_document_semantics")
    warm_ocr_ids = warm.pop("_ocr_candidate_ids")
    warm["semantic_differences_from_baseline"] = _difference_count(
        baseline_semantics, warm_semantics
    )
    warm["ocr_route_differences_from_baseline"] = len(
        set(baseline_ocr_ids) ^ set(warm_ocr_ids)
    )
    modes["l2_warm"] = warm
    _write_json(output_dir / "l2_warm.json", warm)
    shutil.rmtree(output_dir / ".work", ignore_errors=True)
    if progress:
        progress(f"Phase 3C l2_warm: complete in {warm['total_wall_seconds']:.3f}s")

    phase26_summary = json.loads(
        (phase26_dir / "census_summary.json").read_text(encoding="utf-8")
    )
    phase3b_summary = json.loads(
        (phase3b_dir / "integration_summary.json").read_text(encoding="utf-8")
    )
    phase3b_ocr = _read_jsonl(phase3b_dir / "ocr_records.jsonl")
    frozen_ocr_ids = sorted(
        f"{row['source_relative_path']}|{row['formula_id']}" for row in phase3b_ocr
    )
    ocr_state_counts = Counter(row["state"] for row in phase3b_ocr)
    ocr_states = {
        state: ocr_state_counts[state]
        for state in (
            "OCR_ACCEPTED",
            "OCR_ACCEPTED_WITH_WARNING",
            "OCR_REVIEW_REQUIRED",
            "OCR_REJECTED_PRESERVE_IMAGE",
            "OCR_INFERENCE_FAILED_PRESERVE_IMAGE",
        )
    }
    frozen_route_difference = len(set(baseline_ocr_ids) ^ set(frozen_ocr_ids))
    samples = _compare_phase3b_samples(
        phase3b_dir,
        phase3b_summary,
    )

    diff_summary = {
        "schema": "bemarkdown-phase3c-output-diff-v1",
        "baseline_semantic_fingerprint": modes["baseline_no_cache"][
            "semantic_fingerprint"
        ],
        "mode_differences": {
            name: {
                "semantic_differences": result.get(
                    "semantic_differences_from_baseline", 0
                ),
                "ocr_route_differences": result.get(
                    "ocr_route_differences_from_baseline", 0
                ),
            }
            for name, result in modes.items()
        },
        "phase3b_ocr_route_differences": frozen_route_difference,
        "representative_markdown": samples,
        "all_zero_difference": bool(
            frozen_route_difference == 0
            and all(
                result.get("semantic_differences_from_baseline", 0) == 0
                and result.get("ocr_route_differences_from_baseline", 0) == 0
                for result in modes.values()
            )
            and all(
                row["cache_off_vs_cache_on_markdown_equal"]
                and row["cache_off_vs_cache_on_assets_equal"]
                and row["cache_off_vs_phase3b_markdown_equal"]
                and row["cache_off_vs_phase3b_assets_equal"]
                for row in samples
            )
        ),
    }

    expected_phase26_states = {
        key: value["occurrences"]
        for key, value in phase26_summary["final_states"].items()
    }
    actual_states = modes["baseline_no_cache"]["final_states"]
    regression = {
        "schema": "bemarkdown-phase3c-regression-v1",
        "docx": modes["baseline_no_cache"]["docx"],
        "candidate_occurrences": modes["baseline_no_cache"][
            "candidate_occurrences"
        ],
        "real_formula_occurrences": modes["baseline_no_cache"][
            "real_formula_occurrences"
        ],
        "empty_occurrences": actual_states.get(
            "EMPTY_PLACEHOLDER_SUPPRESSED", 0
        ),
        "structural_final": sum(
            value for key, value in actual_states.items() if key.startswith("STRUCTURAL_")
        ),
        "ocr_candidates": actual_states.get("RENDERED_NEEDS_MATH_OCR", 0),
        "final_states": actual_states,
        "phase26_final_states_match": actual_states == expected_phase26_states,
        "phase3b_expected": {
            "candidate_occurrences": phase3b_summary["formulas"][
                "equation_candidates"
            ],
            "real_formula_occurrences": phase3b_summary["formulas"][
                "real_formulas"
            ],
            "structural_final": phase3b_summary["formulas"]["structural_final"],
            "ocr_candidates": phase3b_summary["formulas"]["ocr_candidates"],
        },
        "phase3b_ocr_states": ocr_states,
        "phase3b_ocr_route_differences": frozen_route_difference,
        "four_modes_zero_difference": diff_summary["all_zero_difference"],
        "l2_entries_removed_before_cold_run": removed_entries,
    }

    baseline_stage = modes["baseline_no_cache"]["timing"]
    ole_total = baseline_stage.get("ole_mtef_seconds", 0.0)
    profile_keys = (
        "ole_relationship_lookup_seconds",
        "ole_bytes_access_seconds",
        "ole_open_seconds",
        "equation_stream_lookup_seconds",
        "equation_stream_read_seconds",
        "mtef_payload_extraction_seconds",
        "mtef_sha_seconds",
        "mtef_parse_seconds",
        "mtef_to_mathml_seconds",
        "mathml_to_latex_seconds",
        "formula_validation_seconds",
        "mtef_cache_lookup_seconds",
        "mtef_l2_read_seconds",
        "mtef_l2_write_seconds",
    )
    profiling = {
        "schema": "bemarkdown-phase3c-profiling-v1",
        "ole_mtef_seconds": ole_total,
        "stages": {
            key: {
                "seconds": baseline_stage.get(key, 0.0),
                "percentage_of_ole_mtef": round(
                    100 * baseline_stage.get(key, 0.0) / ole_total, 6
                )
                if ole_total
                else 0.0,
            }
            for key in profile_keys
        },
        "uncached_conversion": modes["baseline_no_cache"]["cache"][
            "uncached_conversion"
        ],
        "environment": _environment(),
        "note": "Wall timings are affected by OS file cache and concurrent system load.",
    }
    cache_statistics = {
        "schema": "bemarkdown-phase3c-cache-statistics-v1",
        "persistent_directory": str(persistent_dir),
        "modes": {name: result["cache"] for name, result in modes.items()},
    }
    summary = {
        "schema": "bemarkdown-phase3c-summary-v1",
        "modes": modes,
        "profiling": profiling,
        "regression": regression,
        "output_diff": diff_summary,
    }
    paths = {
        "profiling": output_dir / "profiling_summary.json",
        "cache": output_dir / "cache_statistics.json",
        "regression": output_dir / "regression_summary.json",
        "diff": output_dir / "output_diff_summary.json",
    }
    _write_json(paths["profiling"], profiling)
    _write_json(paths["cache"], cache_statistics)
    _write_json(paths["regression"], regression)
    _write_json(paths["diff"], diff_summary)
    return Phase3cBenchmarkResult(
        output_dir,
        paths["profiling"],
        paths["cache"],
        paths["regression"],
        paths["diff"],
        summary,
    )


def _run_mode(
    manifest_path: Path,
    records: list[dict[str, Any]],
    output_dir: Path,
    name: str,
    context: MtefCacheContext,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    timing: Counter[str] = Counter()
    states: Counter[str] = Counter()
    document_semantics: dict[str, str] = {}
    ocr_candidate_ids: list[str] = []
    failures: list[dict[str, str]] = []
    source_hash_valid = 0
    candidate_occurrences = 0
    non_formula = 0
    work_root = output_dir / ".work" / name
    for index, manifest_record in enumerate(records, start=1):
        relative = str(manifest_record["target_relative_path"]).replace("\\", "/")
        source = manifest_path.parent / Path(manifest_record["target_relative_path"])
        work = work_root / f"{index:04d}"
        try:
            data = source.read_bytes()
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != manifest_record.get("source_sha256"):
                raise ValueError("source SHA-256 mismatch")
            source_hash_valid += 1
            report = _new_report(source)
            package = PackageIndex(source)
            report["timing"]["package_read_seconds"] = package.read_seconds
            report["mtef_cache"]["mode"] = context.mode
            report["mtef_cache"]["contract_fingerprint"] = (
                context.contract_fingerprint
            )
            scan_started = time.perf_counter()
            DocumentScanner(
                package,
                work,
                report,
                analysis_mode=True,
                mtef_cache_context=context,
            ).scan()
            report["timing"]["scan_seconds"] = time.perf_counter() - scan_started
            _finalize_formula_report(report)
            _finalize_mtef_cache_report(report)
            rows = _semantic_rows(report, actual_sha, relative)
            document_semantics[relative] = _semantic_sha(rows)
            for row in rows:
                states[row["final_state"]] += 1
                if row["final_state"] == "RENDERED_NEEDS_MATH_OCR":
                    ocr_candidate_ids.append(f"{relative}|{row['local_candidate_id']}")
            candidate_occurrences += len(rows)
            non_formula += report["formulas"]["non_formula_objects"]
            for key, value in report["timing"].items():
                if isinstance(value, int | float):
                    timing[key] += value
        except Exception as exc:  # noqa: BLE001 - isolate one document
            failures.append(
                {"source_relative_path": relative, "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if progress and (index % 10 == 0 or index == len(records)):
            progress(f"Phase 3C {name}: {index}/{len(records)} DOCX")
    shutil.rmtree(work_root, ignore_errors=True)
    semantic_fingerprint = _semantic_sha(
        [
            {"source_relative_path": key, "semantic_sha256": value}
            for key, value in sorted(document_semantics.items())
        ]
    )
    final_states = {state: states[state] for state in FINAL_STATES}
    empty = final_states.get("EMPTY_PLACEHOLDER_SUPPRESSED", 0)
    return {
        "schema": "bemarkdown-phase3c-mode-v1",
        "name": name,
        "worker_count": 1,
        "docx": {
            "total": len(records),
            "success": len(records) - len(failures),
            "failed": len(failures),
            "source_sha_valid": source_hash_valid,
        },
        "candidate_occurrences": candidate_occurrences,
        "real_formula_occurrences": candidate_occurrences - empty,
        "excluded_non_formula_objects": non_formula,
        "final_states": final_states,
        "semantic_fingerprint": semantic_fingerprint,
        "document_semantic_fingerprints": document_semantics,
        "timing": {key: round(value, 9) for key, value in sorted(timing.items())},
        "total_wall_seconds": round(time.perf_counter() - started, 6),
        "cache": context.snapshot(),
        "failures": failures,
        "_document_semantics": document_semantics,
        "_ocr_candidate_ids": sorted(ocr_candidate_ids),
    }


def _semantic_rows(report: dict[str, Any], docx_sha: str, relative: str):
    formula_by_id = {
        row["formula_id"]: row for row in report["formulas"]["records"]
    }
    rows = []
    for candidate in report["formulas"]["candidate_records"]:
        if candidate["classification"] == "NON_FORMULA_OBJECT":
            continue
        formula = formula_by_id.get(candidate["candidate_id"])
        final_state, _ = _final_state(candidate, formula)
        rows.append(
            {
                "docx_sha256": docx_sha,
                "source_relative_path": relative,
                "local_candidate_id": candidate["candidate_id"],
                "classification": candidate["classification"],
                "source_part": candidate["source_part"],
                "source_locator": candidate["source_locator"],
                "source_type": candidate.get("source_type"),
                "status": candidate["status"],
                "original_ref": candidate.get("original_ref"),
                "preview_ref": candidate.get("preview_ref"),
                "ole_sha256": candidate.get("ole_sha256"),
                "wmf_sha256": candidate.get("wmf_sha256"),
                "mtef_payload_sha256": candidate.get("mtef_payload_sha256"),
                "source_payload_sha256": candidate.get("source_payload_sha256"),
                "source_payload_type": candidate.get("source_payload_type"),
                "semantic_state": candidate.get("semantic_state"),
                "preview_visual_state": candidate.get("preview_visual_state"),
                "preview_suppressed": candidate.get("preview_suppressed"),
                "preview_suppressed_by_ole": candidate.get(
                    "preview_suppressed_by_ole"
                ),
                "validator": candidate.get("validator"),
                "structural_attempts": candidate.get("structural_attempts", []),
                "final_state": final_state,
                "latex": formula.get("latex") if formula else None,
                "warnings": formula.get("warnings", []) if formula else [],
                "error": formula.get("error") if formula else None,
                "component": formula.get("component") if formula else None,
                "rendered_ref": formula.get("rendered_ref") if formula else None,
                "renderer_metadata": (
                    _semantic_renderer_metadata(formula.get("renderer_metadata"))
                    if formula
                    else None
                ),
            }
        )
    return rows


def _semantic_renderer_metadata(value: dict[str, Any] | None):
    if value is None:
        return None
    return {
        key: item
        for key, item in value.items()
        if key not in {"wall_seconds", "cache_hit"}
    }


class _ReplayFormulaRuntime:
    def __init__(self, predictions: dict[str, str], fingerprint: dict[str, Any]):
        self.predictions = predictions
        self._fingerprint = fingerprint

    def fingerprint(self):
        return self._fingerprint

    def predict(self, image_paths, *, batch_size):
        del batch_size
        outputs = []
        for path in image_paths:
            sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            outputs.append(self.predictions[sha])
        return outputs

    @staticmethod
    def peak_gpu_memory_bytes():
        return None


def _compare_phase3b_samples(phase3b_dir: Path, phase3b_summary: dict[str, Any]):
    ocr_rows = _read_jsonl(phase3b_dir / "ocr_records.jsonl")
    predictions: dict[str, str] = {}
    fingerprint = None
    for row in ocr_rows:
        raw = row.get("raw_latex")
        if raw is not None:
            previous = predictions.setdefault(row["png_sha256"], raw)
            if previous != raw:
                raise ValueError("Phase 3B replay data has conflicting PNG predictions")
        fingerprint = fingerprint or row["provenance"]["runtime_fingerprint"]
    off_adapter = FormulaOcrAdapter(
        runtime_factory=lambda: _ReplayFormulaRuntime(predictions, fingerprint),
        batch_size=2,
    )
    on_adapter = FormulaOcrAdapter(
        runtime_factory=lambda: _ReplayFormulaRuntime(predictions, fingerprint),
        batch_size=2,
    )
    shared = MtefCacheContext(mode="memory")
    comparisons = []
    with tempfile.TemporaryDirectory(prefix="bemarkdown-phase3c-samples-") as temp:
        temp_root = Path(temp)
        for sample in phase3b_summary["sample_documents"]:
            source = Path(sample["source_file"])
            name = Path(sample["output"]).name
            off = convert_docx(
                source,
                temp_root / f"{name}-off",
                formula_ocr="auto",
                formula_ocr_adapter=off_adapter,
                mtef_cache="off",
            )
            on = convert_docx(
                source,
                temp_root / f"{name}-on",
                formula_ocr="auto",
                formula_ocr_adapter=on_adapter,
                mtef_cache_context=shared,
            )
            frozen_dir = phase3b_dir / sample["output"]
            off_assets = _asset_hashes(off.output_dir / "assets")
            on_assets = _asset_hashes(on.output_dir / "assets")
            frozen_assets = _asset_hashes(frozen_dir / "assets")
            off_markdown = off.markdown_path.read_bytes()
            on_markdown = on.markdown_path.read_bytes()
            frozen_markdown = (frozen_dir / "document.md").read_bytes()
            comparisons.append(
                {
                    "source_file": str(source),
                    "sample": name,
                    "cache_off_vs_cache_on_markdown_equal": off_markdown
                    == on_markdown,
                    "cache_off_vs_cache_on_assets_equal": off_assets == on_assets,
                    "cache_off_vs_phase3b_markdown_equal": off_markdown
                    == frozen_markdown,
                    "cache_off_vs_phase3b_assets_equal": off_assets
                    == frozen_assets,
                    "markdown_sha256": hashlib.sha256(off_markdown).hexdigest(),
                    "asset_count": len(off_assets),
                }
            )
    return comparisons


def _asset_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _semantic_sha(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _difference_count(expected: dict[str, str], actual: dict[str, str]) -> int:
    keys = set(expected) | set(actual)
    return sum(expected.get(key) != actual.get(key) for key in keys)


def _environment() -> dict[str, Any]:
    try:
        import psutil

        ram_bytes = psutil.virtual_memory().total
    except (ImportError, OSError):
        ram_bytes = None
    return {
        "cpu": platform.processor(),
        "ram_bytes": ram_bytes,
        "os": platform.platform(),
        "python": platform.python_version(),
        "worker_count": 1,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
