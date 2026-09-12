from __future__ import annotations

import importlib.metadata
import json
import re
import socket
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from unittest.mock import patch

from .doctor import run_doctor
from .formula_ocr import FormulaOcrAdapter, FormulaOcrSafetyGate
from .formulanet_runtime import FormulaNetRuntime, FormulaOcrOutputValidator
from .model_registry import (
    MODEL_EXPECTED_FINGERPRINT,
    MODEL_ID,
    ModelFingerprintMismatch,
    ModelRegistry,
    build_model_manifest,
)
from .phase4a_validation import run_phase4a_validation
from .production import convert_document, validate_package
from .release import sha256_file, write_json


class RecordingRuntime:
    def __init__(self, runtime: FormulaNetRuntime):
        self.runtime = runtime
        self.predictions: dict[str, str] = {}
        self.load_seconds = getattr(runtime, "load_seconds", 0.0)

    def predict(self, paths: Sequence[Path], *, batch_size: int) -> list[str]:
        outputs = self.runtime.predict(paths, batch_size=batch_size)
        for path, output in zip(paths, outputs, strict=True):
            self.predictions[sha256_file(path)] = output
        return outputs

    def fingerprint(self) -> dict:
        return self.runtime.fingerprint()

    def peak_gpu_memory_bytes(self) -> int | None:
        return self.runtime.peak_gpu_memory_bytes()


def run_phase4b_validation(
    manifest_path: str | Path,
    phase3a_dir: str | Path,
    phase3b_dir: str | Path,
    models_root: str | Path,
    tool_root: str | Path,
    output_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
    overwrite: bool = False,
) -> dict:
    started = time.perf_counter()
    manifest_path = Path(manifest_path).resolve()
    phase3a_dir = Path(phase3a_dir).resolve()
    phase3b_dir = Path(phase3b_dir).resolve()
    models_root = Path(models_root).resolve()
    tool_root = Path(tool_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Phase 4B evidence exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_started = time.perf_counter()
    resolution = ModelRegistry(models_root=models_root).resolve(MODEL_ID, deep=True)
    registry_seconds = time.perf_counter() - registry_started
    if resolution.manifest["model_fingerprint"] != MODEL_EXPECTED_FINGERPRINT:
        raise RuntimeError("RC model fingerprint differs from the frozen model")

    from .formula_ocr import create_production_formulanet_runtime

    runtime = RecordingRuntime(
        create_production_formulanet_runtime(models_root=models_root)
    )
    adapter = FormulaOcrAdapter(runtime_factory=lambda: runtime, batch_size=2)
    with tempfile.TemporaryDirectory(
        prefix="bemarkdown-phase4b-default-"
    ) as default_dir, patch.dict(
        "os.environ", {"BEMARKDOWN_OUTPUT_ROOT": str(Path(default_dir) / "packages")}
    ):
        phase4a = run_phase4a_validation(
            manifest_path,
            phase3b_dir,
            output_dir / "full_runtime_gate",
            expected_default_output_root=Path(default_dir) / "packages",
            progress=progress,
            overwrite=overwrite,
            formula_ocr_adapter=adapter,
        )

    baseline_rows = _read_jsonl(phase3a_dir / "formula_predictions.jsonl")
    baseline = {
        row["png_content_sha256"]: row["formula_net_raw_latex"]
        for row in baseline_rows
    }
    raw_differences = [
        {
            "png_sha256": png_sha,
            "expected": baseline.get(png_sha),
            "actual": runtime.predictions.get(png_sha),
        }
        for png_sha in sorted(set(baseline) | set(runtime.predictions))
        if baseline.get(png_sha) != runtime.predictions.get(png_sha)
    ]
    review_rows = _read_jsonl(phase3a_dir / "formula_review.jsonl")
    smoke = _formula_smoke(baseline_rows, review_rows, runtime.predictions)

    full_doctor = run_doctor(
        models_root=models_root,
        output_root=output_dir / "doctor-output",
        tool_root=tool_root,
        deep=True,
    )
    degraded_doctor = run_doctor(
        models_root=output_dir / "intentionally-missing-models",
        output_root=output_dir / "doctor-degraded-output",
        tool_root=tool_root,
        deep=True,
    )
    degraded = _degraded_and_no_network_gate(review_rows, output_dir)
    corrupt = _corrupt_model_gate(output_dir)

    dependency_inventory = _dependency_inventory()
    sizes = {
        "schema": "bemarkdown-phase4b-runtime-size-v1",
        "tool_immutable_payload_bytes": _tree_bytes(tool_root, exclude={".runtime"}),
        "model_payload_bytes": resolution.manifest["total_bytes"],
        "model_manifest_bytes": (resolution.model_root / "MODEL_MANIFEST.json").stat().st_size,
        "host_generated_runtime_bytes": _tree_bytes(Path(sys.prefix)),
        "wheel_bytes": (tool_root / json.loads((tool_root / "TOOL_MANIFEST.json").read_text(encoding="utf-8"))["wheel"]["path"]).stat().st_size,
    }
    write_json(output_dir / "runtime_size_summary.json", sizes)
    write_json(output_dir / "dependency_inventory.json", dependency_inventory)
    _write_jsonl(
        output_dir / "doctor_results.jsonl",
        [full_doctor.to_dict(), degraded_doctor.to_dict()],
    )
    write_json(output_dir / "formula_runtime_smoke.json", smoke)
    write_json(
        output_dir / "degraded_mode_results.json",
        {"schema": "bemarkdown-phase4b-degraded-v1", **degraded, "corrupt_model": corrupt},
    )

    summary = {
        "schema": "bemarkdown-phase4b-release-regression-v1",
        "model": {
            "fingerprint": resolution.manifest["model_fingerprint"],
            "file_count": resolution.manifest["file_count"],
            "total_bytes": resolution.manifest["total_bytes"],
            "deep_verified": resolution.fingerprint_verified,
            "registry_seconds": registry_seconds,
        },
        "runtime": runtime.fingerprint(),
        "doctor": {
            "full": full_doctor.readiness,
            "degraded": degraded_doctor.readiness,
        },
        "formula_raw_reproducibility": {
            "expected_unique_png": len(baseline),
            "actual_unique_png": len(runtime.predictions),
            "difference_count": len(raw_differences),
            "differences": raw_differences,
        },
        "formula_smoke": smoke,
        "phase4a_regression": phase4a.summary,
        "degraded_mode": degraded,
        "corrupt_model": corrupt,
        "checks": {},
        "wall_seconds": time.perf_counter() - started,
    }
    summary["checks"] = {
        "doctor_ready_full": full_doctor.readiness == "READY_FULL",
        "doctor_ready_degraded": degraded_doctor.readiness == "READY_DEGRADED",
        "model_fingerprint_frozen": resolution.manifest["model_fingerprint"]
        == MODEL_EXPECTED_FINGERPRINT,
        "real_batch2_all_157": len(runtime.predictions) == len(baseline) == 157,
        "raw_differences_zero": not raw_differences,
        "smoke_complete": smoke["all_ok"],
        "phase4a_all_ok": phase4a.summary["all_ok"],
        "no_model_degraded": degraded["all_ok"],
        "corrupt_model_rejected": corrupt["passed"],
    }
    summary["all_ok"] = all(summary["checks"].values())
    write_json(output_dir / "release_regression_summary.json", summary)
    return summary


def _formula_smoke(baseline_rows, review_rows, predictions) -> dict:
    review_by_sha = {row["png_content_sha256"]: row for row in review_rows}
    by_sha = {row["png_content_sha256"]: row for row in baseline_rows}
    selected: dict[str, dict] = {}
    used: set[str] = set()
    catastrophic = next(row for row in review_rows if row["review_index"] == 85)
    catastrophic_row = by_sha[catastrophic["png_content_sha256"]]
    selected["case_085"] = catastrophic_row
    used.add(catastrophic_row["png_content_sha256"])

    def choose(name, predicate, *, reverse=False):
        candidates = [row for row in baseline_rows if row["png_content_sha256"] not in used and predicate(row)]
        candidates.sort(
            key=lambda row: (len(row["formula_net_raw_latex"]), row["png_content_sha256"]),
            reverse=reverse,
        )
        if not candidates:
            raise RuntimeError(f"No FormulaNet smoke candidate for {name}")
        row = candidates[0]
        used.add(row["png_content_sha256"])
        selected[name] = row

    choose("fraction", lambda row: "\\frac" in row["formula_net_raw_latex"])
    choose("root", lambda row: "\\sqrt" in row["formula_net_raw_latex"])
    choose(
        "cjk_script",
        lambda row: bool(re.search(r"[\u3400-\u9fff]", row["formula_net_raw_latex"]))
        and any(token in row["formula_net_raw_latex"] for token in ("_", "^")),
    )
    choose("long_formula", lambda row: True, reverse=True)
    tiny = min(
        (row for row in baseline_rows if row["png_content_sha256"] not in used),
        key=lambda row: (row["width"] * row["height"], row["png_content_sha256"]),
    )
    used.add(tiny["png_content_sha256"])
    selected["tiny_symbol"] = tiny
    choose(
        "ordinary_single_line",
        lambda row: row["ocr_validator_verdict"] == "OCR_VALID"
        and "\\frac" not in row["formula_net_raw_latex"]
        and "\\sqrt" not in row["formula_net_raw_latex"]
        and len(row["formula_net_raw_latex"]) < 80,
    )
    records = {}
    for name, row in selected.items():
        sha = row["png_content_sha256"]
        actual = predictions.get(sha)
        records[name] = {
            "png_sha256": sha,
            "width": row["width"],
            "height": row["height"],
            "expected_raw_latex": row["formula_net_raw_latex"],
            "actual_raw_latex": actual,
            "raw_match": actual == row["formula_net_raw_latex"],
            "source_png": Path(review_by_sha[sha]["source_png"]).name,
        }
    case = records["case_085"]
    validation = FormulaOcrOutputValidator().validate(case["actual_raw_latex"])
    decision = FormulaOcrSafetyGate().evaluate(
        width=case["width"],
        height=case["height"],
        raw_latex=case["actual_raw_latex"],
        validation=validation,
    )
    case["validator"] = validation.to_dict()
    case["safety_gate"] = decision.to_dict()
    case["not_accepted"] = decision.verdict.value != "ACCEPT"
    return {
        "schema": "bemarkdown-phase4b-formulanet-smoke-v1",
        "batch_size": 2,
        "precision": "fp32",
        "records": records,
        "raw_difference_count": sum(not row["raw_match"] for row in records.values()),
        "case_085_not_accepted": case["not_accepted"],
        "all_ok": all(row["raw_match"] for row in records.values()) and case["not_accepted"],
    }


def _degraded_and_no_network_gate(review_rows, output_dir: Path) -> dict:
    case = next(row for row in review_rows if row["review_index"] == 85)
    source = Path(case["sources"][0]["source_docx"])
    attempts: list[str] = []

    def blocked(*args, **kwargs):
        attempts.append("network")
        raise AssertionError("network access is forbidden")

    missing_root = output_dir / "missing-models"

    def missing_runtime():
        return ModelRegistry(models_root=missing_root).resolve(MODEL_ID)

    adapter = FormulaOcrAdapter(runtime_factory=missing_runtime, batch_size=2)
    with tempfile.TemporaryDirectory(prefix="bemarkdown-phase4b-degraded-") as temporary:
        with patch.object(socket, "create_connection", blocked), patch.object(
            urllib.request, "urlopen", blocked
        ):
            result = convert_document(
                source,
                Path(temporary) / "packages",
                formula_ocr="auto",
                formula_ocr_adapter=adapter,
            )
        report = json.loads(
            (result.package_path / "conversion_report.json").read_text(encoding="utf-8")
        )
        assets = _read_jsonl(result.package_path / "assets_manifest.jsonl")
        formula_assets = [row for row in assets if row["semantic_type"] == "FORMULA_FALLBACK"]
        valid = validate_package(result.package_path).valid
        preserved = bool(formula_assets) and all(
            row["relative_path"] and (result.package_path / row["relative_path"]).is_file()
            for row in formula_assets
        )
        outcome = {
            "package_valid": valid,
            "quality_status": result.quality_status,
            "inference_failed_preserved": report["formula_ocr"]["inference_failed_preserved"],
            "formula_assets_preserved": preserved,
            "network_attempts": len(attempts),
        }
    outcome["all_ok"] = (
        outcome["package_valid"]
        and outcome["quality_status"] == "COMPLETED_WITH_REVIEW_ITEMS"
        and outcome["inference_failed_preserved"] > 0
        and outcome["formula_assets_preserved"]
        and outcome["network_attempts"] == 0
    )
    return outcome


def _corrupt_model_gate(output_dir: Path) -> dict:
    root = output_dir / "corrupt-model-fixture" / "MODELS" / "PP-FormulaNet_plus-L"
    root.mkdir(parents=True)
    (root / "README.md").write_text("license: apache-2.0", encoding="utf-8")
    (root / "weights.bin").write_bytes(b"valid")
    manifest = build_model_manifest(root)
    write_json(root / "MODEL_MANIFEST.json", manifest)
    (root / "weights.bin").write_bytes(b"corrupt")
    try:
        ModelRegistry(models_root=root.parent).resolve(MODEL_ID, deep=True)
        outcome = "ACCEPTED_UNEXPECTEDLY"
        passed = False
    except ModelFingerprintMismatch as exc:
        outcome = str(exc)
        passed = "MODEL_FINGERPRINT_MISMATCH" in outcome
    return {"passed": passed, "outcome": outcome}


def _dependency_inventory() -> dict:
    rows = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "version": distribution.version,
                "license": distribution.metadata.get("License") or "LICENSE_STATUS_UNVERIFIED",
                "source": distribution.metadata.get("Home-page") or distribution.metadata.get("Project-URL"),
                "bundled": False,
            }
        )
    return {
        "schema": "bemarkdown-phase4b-dependency-inventory-v1",
        "distributions": sorted(rows, key=lambda row: row["name"].casefold()),
    }


def _tree_bytes(root: Path, *, exclude: set[str] | None = None) -> int:
    if not root.is_dir():
        return 0
    exclude = exclude or set()
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not any(part in exclude for part in path.relative_to(root).parts)
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
