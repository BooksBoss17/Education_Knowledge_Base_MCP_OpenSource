from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import convert_docx


@dataclass(frozen=True)
class RegressionResult:
    output_dir: Path
    summary_path: Path
    summary: dict[str, Any]


def run_docx_regression(
    baseline_dir: str | Path, output_dir: str | Path
) -> RegressionResult:
    """Re-run every Phase baseline DOCX and compare OMML output exactly."""

    started = time.perf_counter()
    baseline_dir = Path(baseline_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_reports = sorted(
        path
        for path in baseline_dir.glob("*/conversion_report.json")
        if path.parent.name != "wmf_samples"
    )
    if not baseline_reports:
        raise FileNotFoundError(f"No baseline conversion reports in {baseline_dir}")

    records = []
    total_counts: dict[str, int] = {}
    all_omml_exact = True
    for baseline_report_path in baseline_reports:
        baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        source = Path(baseline["source"]["path"])
        case = baseline_report_path.parent.name
        converted = convert_docx(source, output_dir / case, debug=True)
        current = converted.report
        old_omml = [
            item["latex"]
            for item in baseline["formulas"]["records"]
            if item["source_type"] == "omml"
        ]
        new_omml = [
            item["latex"]
            for item in current["formulas"]["records"]
            if item["source_type"] == "omml"
        ]
        omml_exact = old_omml == new_omml
        all_omml_exact = all_omml_exact and omml_exact
        formula_counts = {
            key: current["formulas"][key]
            for key in (
                "equation_candidates",
                "real_formulas",
                "empty_placeholders",
                "structural_success",
                "structural_detected",
                "structural_valid",
                "structural_suspicious",
                "structural_invalid",
                "structural_non_formula",
                "rendered_fallback",
                "failed",
                "unresolved_candidates",
            )
        }
        for key, value in formula_counts.items():
            total_counts[key] = total_counts.get(key, 0) + value
        rejected_attempts = [
            {
                "formula_id": item["formula_id"],
                "source_type": item["source_type"],
                "source_locator": item["source_locator"],
                "attempt": attempt,
                "final_status": item["status"],
                "rendered_ref": item["rendered_ref"],
            }
            for item in current["formulas"]["records"]
            for attempt in item.get("structural_audit", [])
        ]
        records.append(
            {
                "case": case,
                "source": str(source),
                "source_sha256": current["source"]["sha256"],
                "formula_counts": formula_counts,
                "candidate_conservation_ok": current["formulas"][
                    "candidate_conservation_ok"
                ],
                "real_formula_conservation_ok": current["formulas"][
                    "real_formula_conservation_ok"
                ],
                "omml_count": len(new_omml),
                "omml_exact_match": omml_exact,
                "rejected_structural_attempts": rejected_attempts,
                "report_path": str(converted.report_path),
            }
        )
    summary = {
        "schema": "bemarkdown-phase25-docx-regression-v1",
        "baseline_dir": str(baseline_dir),
        "case_count": len(records),
        "totals": total_counts,
        "omml": {
            "total_count": sum(record["omml_count"] for record in records),
            "all_exact_match": all_omml_exact,
            "specialty_case_count": next(
                (
                    record["omml_count"]
                    for record in records
                    if record["case"] == "omml"
                ),
                0,
            ),
            "specialty_case_exact_match": next(
                (
                    record["omml_exact_match"]
                    for record in records
                    if record["case"] == "omml"
                ),
                False,
            ),
        },
        "all_conservation_ok": all(
            record["candidate_conservation_ok"]
            and record["real_formula_conservation_ok"]
            for record in records
        ),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "records": records,
    }
    summary_path = output_dir / "regression_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return RegressionResult(output_dir, summary_path, summary)
