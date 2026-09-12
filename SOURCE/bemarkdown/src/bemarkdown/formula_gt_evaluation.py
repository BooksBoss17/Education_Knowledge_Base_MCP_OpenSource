from __future__ import annotations

import hashlib
import html
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_CONTENT_STATUSES = {
    "COMPLETE_FORMULA",
    "FORMULA_FRAGMENT",
    "SINGLE_SYMBOL",
    "PUNCTUATION_OR_DELIMITER_FRAGMENT",
    "SOURCE_MALFORMED",
    "NON_FORMULA_VISIBLE_CONTENT",
    "UNREADABLE_SOURCE",
}
RECOGNITION_STATUSES = {
    "EXACT",
    "SEMANTICALLY_CORRECT_STYLE_DIFFERENT",
    "MINOR_ERROR",
    "MAJOR_ERROR",
    "HALLUCINATION",
    "OMISSION",
    "NOT_APPLICABLE",
    "NEEDS_HUMAN_REVIEW",
}
REVIEW_STATUSES = {"REVIEWED", "NEEDS_HUMAN_REVIEW"}
REVIEW_CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
ERROR_STATUSES = {"MINOR_ERROR", "MAJOR_ERROR", "HALLUCINATION", "OMISSION"}
EXCLUDED_FROM_RECOGNITION = {"NOT_APPLICABLE", "NEEDS_HUMAN_REVIEW"}
VISUAL_SUCCESS_STATUSES = {"EXACT", "SEMANTICALLY_CORRECT_STYLE_DIFFERENT"}

PHASE26_TRUE_MATH_SHA256 = (
    "1f5f7f207c1837036eb4d5c89e069c96d9ba5779f1fb48c019289d9eec1f6787"
)
PHASE26_SUMMARY_SHA256 = (
    "670e15941f1d60e30d14ec8ffaf1f33a9d351fc365c10fce50b984283c15e6b5"
)


@dataclass(frozen=True)
class FormulaReferenceEvaluationResult:
    summary: dict[str, Any]
    summary_path: Path
    stratified_metrics_path: Path
    error_cases_path: Path
    needs_human_review_path: Path
    review_html_path: Path


def normalize_latex_for_evaluation(latex: str) -> str:
    """Apply only explicitly harmless representation normalization."""

    normalized = re.sub(r"[ \t\r\n]+", " ", latex.strip())
    normalized = re.sub(r"([_^])\{\s*_\s*([^{}]+)\}", r"\1{\2}", normalized)
    normalized = re.sub(r"\{\s*=\s*\}", "=", normalized)
    return normalized


def validate_reference_record(record: dict[str, Any]) -> None:
    required = {
        "review_index",
        "png_content_sha256",
        "formula_net_raw_latex",
        "ocr_validator_verdict",
        "source_candidate_count",
        "source_occurrence_count",
        "review_status",
        "source_content_status",
        "ground_truth_latex",
        "ground_truth_normalization",
        "recognition_status",
        "review_confidence",
        "context_used",
        "review_notes",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"Reference record is missing fields: {', '.join(missing)}")
    if record["review_status"] not in REVIEW_STATUSES:
        raise ValueError(f"Invalid review_status: {record['review_status']}")
    if record["source_content_status"] not in SOURCE_CONTENT_STATUSES:
        raise ValueError(
            f"Invalid source_content_status: {record['source_content_status']}"
        )
    if record["recognition_status"] not in RECOGNITION_STATUSES:
        raise ValueError(f"Invalid recognition_status: {record['recognition_status']}")
    if record["review_confidence"] not in REVIEW_CONFIDENCES:
        raise ValueError(f"Invalid review_confidence: {record['review_confidence']}")
    if not isinstance(record["context_used"], bool):
        raise TypeError("context_used must be a boolean")
    if not isinstance(record["review_notes"], str) or not record["review_notes"]:
        raise ValueError("review_notes must be a non-empty string")
    if record["recognition_status"] == "NEEDS_HUMAN_REVIEW":
        if record["review_status"] != "NEEDS_HUMAN_REVIEW":
            raise ValueError("NEEDS_HUMAN_REVIEW must use the matching review_status")
    elif record["review_status"] != "REVIEWED":
        raise ValueError("Reviewed recognition decisions must use REVIEWED")
    if (
        record["recognition_status"] not in EXCLUDED_FROM_RECOGNITION
        and not record["ground_truth_latex"]
    ):
        raise ValueError("ground_truth_latex is required for recognition evaluation")
    if (
        record["recognition_status"] == "NOT_APPLICABLE"
        and record["source_content_status"] != "NON_FORMULA_VISIBLE_CONTENT"
    ):
        raise ValueError("NOT_APPLICABLE requires NON_FORMULA_VISIBLE_CONTENT")


def build_reference_ground_truth(
    phase3a_dir: Path,
    adjudications_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Materialize explicit reference records from completed visual adjudications."""

    phase3a_dir = phase3a_dir.resolve()
    adjudications = json.loads(adjudications_path.read_text(encoding="utf-8"))
    frozen = _verify_frozen_phase3a(phase3a_dir)
    expected_hashes = adjudications.get("expected_phase3a_sha256", {})
    for name, expected in expected_hashes.items():
        actual = frozen["artifact_sha256"].get(name)
        if actual != expected:
            raise ValueError(
                f"Frozen Phase 3A hash mismatch for {name}: {actual} != {expected}"
            )

    raw_rows = _read_jsonl(phase3a_dir / "formula_review.jsonl")
    actual_indices = {row["review_index"] for row in raw_rows}
    reviewed_indices = set(adjudications["reviewed_indices"])
    if reviewed_indices != actual_indices:
        missing = sorted(actual_indices - reviewed_indices)
        extra = sorted(reviewed_indices - actual_indices)
        raise ValueError(
            f"Visual review coverage mismatch; missing={missing}, extra={extra}"
        )

    source_groups = _invert_groups(
        adjudications.get("source_content_status_overrides", {})
    )
    recognition_groups = _invert_groups(
        adjudications.get("recognition_status_overrides", {})
    )
    confidence_groups = _invert_groups(adjudications.get("confidence_overrides", {}))
    context_used = set(adjudications.get("context_used_indices", []))
    overrides = {
        int(index): value
        for index, value in adjudications.get("record_overrides", {}).items()
    }
    defaults = adjudications["defaults"]
    rows = []
    for raw in raw_rows:
        index = raw["review_index"]
        source_status = source_groups.get(index, defaults["source_content_status"])
        recognition = recognition_groups.get(index, defaults["recognition_status"])
        confidence = confidence_groups.get(index, defaults["review_confidence"])
        override = overrides.get(index, {})
        gt = override.get("ground_truth_latex")
        if "ground_truth_latex" not in override:
            gt = (
                None
                if recognition in EXCLUDED_FROM_RECOGNITION
                else raw["formula_net_raw_latex"]
            )
        row = {
            **raw,
            "reference_gt_schema": "bemarkdown-formula-reference-gt-v1",
            "review_status": (
                "NEEDS_HUMAN_REVIEW"
                if recognition == "NEEDS_HUMAN_REVIEW"
                else "REVIEWED"
            ),
            "source_content_status": source_status,
            "ground_truth_latex": gt,
            "ground_truth_normalization": override.get(
                "ground_truth_normalization",
                defaults["ground_truth_normalization"],
            ),
            "recognition_status": recognition,
            "review_confidence": confidence,
            "context_used": index in context_used,
            "review_notes": override.get("review_notes", defaults["review_notes"]),
        }
        row.pop("review_status_options", None)
        validate_reference_record(row)
        rows.append(row)
    _write_jsonl(output_path, rows)
    return rows


def evaluate_formula_reference(
    reference_path: Path,
    output_dir: Path,
    *,
    phase3a_dir: Path | None = None,
) -> FormulaReferenceEvaluationResult:
    rows = _read_jsonl(reference_path)
    if not rows:
        raise ValueError("Reference GT is empty")
    for row in rows:
        validate_reference_record(row)
    _validate_unique_reference_rows(rows)

    frozen = _verify_frozen_phase3a(phase3a_dir.resolve()) if phase3a_dir else None
    if phase3a_dir:
        _verify_reference_against_phase3a(rows, phase3a_dir.resolve())

    enriched = [_enrich(row) for row in rows]
    review_counts = Counter(row["review_status"] for row in rows)
    recognition_counts = Counter(row["recognition_status"] for row in rows)
    source_counts = Counter(row["source_content_status"] for row in rows)
    confidence_counts = Counter(row["review_confidence"] for row in rows)

    summary = {
        "schema": "bemarkdown-formula-reference-evaluation-v1",
        "reference_ground_truth": {
            "kind": "agent-visual-reviewed reference ground truth",
            "human_verified": False,
            "records": len(rows),
            "reference_jsonl_sha256": _sha256_file(reference_path),
        },
        "frozen_phase3a_verification": frozen,
        "review": {
            "total": len(rows),
            "reviewed": review_counts["REVIEWED"],
            "needs_human_review": recognition_counts["NEEDS_HUMAN_REVIEW"],
            "not_applicable": recognition_counts["NOT_APPLICABLE"],
            "confidence": _complete_counts(confidence_counts, REVIEW_CONFIDENCES),
        },
        "source_content_status": _complete_counts(
            source_counts, SOURCE_CONTENT_STATUSES
        ),
        "unique_png_metrics": _weighted_metrics(enriched, lambda _row: 1),
        "fanout_impact": {
            "candidates": _weighted_metrics(
                enriched, lambda row: row["source_candidate_count"]
            ),
            "occurrences": _weighted_metrics(
                enriched, lambda row: row["source_occurrence_count"]
            ),
        },
        "normalization_rules": [
            "strip outer whitespace",
            "collapse consecutive ordinary whitespace",
            "remove a redundant nested underscore in a script operand",
            "remove braces that contain only an equals sign",
            "never alter variables, Greek letters, operators, scripts, or delimiters",
        ],
    }
    stratified = _stratified_metrics(enriched)
    errors = [row for row in enriched if row["recognition_status"] in ERROR_STATUSES]
    needs_review = [
        row for row in enriched if row["recognition_status"] == "NEEDS_HUMAN_REVIEW"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    stratified_path = output_dir / "stratified_metrics.json"
    errors_path = output_dir / "error_cases.jsonl"
    needs_review_path = output_dir / "needs_human_review.jsonl"
    html_path = output_dir / "formula_review_reference_gt.html"
    _write_json(summary_path, summary)
    _write_json(stratified_path, stratified)
    _write_jsonl(errors_path, errors)
    _write_jsonl(needs_review_path, needs_review)
    html_path.write_text(
        _review_html(enriched, reference_path, output_dir),
        encoding="utf-8",
        newline="\n",
    )
    return FormulaReferenceEvaluationResult(
        summary=summary,
        summary_path=summary_path,
        stratified_metrics_path=stratified_path,
        error_cases_path=errors_path,
        needs_human_review_path=needs_review_path,
        review_html_path=html_path,
    )


def _enrich(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["formula_net_raw_latex"] or ""
    gt = row["ground_truth_latex"]
    eligible = row["recognition_status"] not in EXCLUDED_FROM_RECOGNITION
    raw_exact = bool(eligible and gt is not None and raw == gt)
    normalized_exact = bool(
        eligible
        and gt is not None
        and normalize_latex_for_evaluation(raw)
        == normalize_latex_for_evaluation(gt)
    )
    production_ready = bool(
        eligible
        and row["recognition_status"] == "EXACT"
        and row["ocr_validator_verdict"] == "OCR_VALID"
    )
    return {
        **row,
        "evaluation": {
            "eligible": eligible,
            "raw_exact": raw_exact,
            "normalized_exact": normalized_exact,
            "visual_transcription_success": (
                eligible and row["recognition_status"] in VISUAL_SUCCESS_STATUSES
            ),
            "production_ready": production_ready,
        },
        "derived_formula_types": _formula_types(row),
        "derived_size_stratum": _size_stratum(row),
    }


def _weighted_metrics(rows, weight):
    eligible_rows = [row for row in rows if row["evaluation"]["eligible"]]
    total = sum(weight(row) for row in rows)
    eligible = sum(weight(row) for row in eligible_rows)

    def measure(field):
        count = sum(weight(row) for row in eligible_rows if row["evaluation"][field])
        return {"count": count, "rate": count / eligible if eligible else None}

    recognition = Counter()
    for row in eligible_rows:
        recognition[row["recognition_status"]] += weight(row)
    return {
        "total": total,
        "eligible": eligible,
        "raw_exact": measure("raw_exact"),
        "normalized_exact": measure("normalized_exact"),
        "visual_transcription_success": measure("visual_transcription_success"),
        "production_ready": measure("production_ready"),
        "recognition_status": _complete_counts(
            recognition, RECOGNITION_STATUSES - EXCLUDED_FROM_RECOGNITION
        ),
    }


def _stratified_metrics(rows):
    dimensions: dict[str, dict[str, list[dict[str, Any]]]] = {
        "source_content_status": defaultdict(list),
        "prog_id": defaultdict(list),
        "size": defaultdict(list),
        "formula_type": defaultdict(list),
    }
    for row in rows:
        dimensions["source_content_status"][row["source_content_status"]].append(row)
        dimensions["size"][row["derived_size_stratum"]].append(row)
        for prog_id in sorted(
            {source.get("ole_prog_id") or "UNKNOWN" for source in row.get("sources", [])}
        ):
            dimensions["prog_id"][prog_id].append(row)
        for formula_type in row["derived_formula_types"]:
            dimensions["formula_type"][formula_type].append(row)
    return {
        dimension: {
            name: _weighted_metrics(members, lambda _row: 1)
            for name, members in sorted(groups.items())
        }
        for dimension, groups in dimensions.items()
    }


def _formula_types(row):
    latex = row["ground_truth_latex"] or row["formula_net_raw_latex"] or ""
    types = []
    if r"\frac" in latex:
        types.append("fraction")
    if r"\sqrt" in latex:
        types.append("root")
    if "_" in latex:
        types.append("subscript")
    if "^" in latex:
        types.append("superscript")
    if re.search(r"[\u3400-\u9fff]", latex):
        types.append("Chinese_script_or_text")
    if len(latex) >= 120 or row["width"] >= 1800:
        types.append("long_formula")
    if row["source_content_status"] in {
        "SINGLE_SYMBOL",
        "PUNCTUATION_OR_DELIMITER_FRAGMENT",
    } or row["width"] * row["height"] <= 10_000:
        types.append("tiny_symbol_or_fragment")
    return types or ["other"]


def _size_stratum(row):
    width = row["width"]
    height = row["height"]
    if width * height <= 10_000:
        return "very_small"
    if width <= 200 or width / max(height, 1) <= 0.75:
        return "very_narrow"
    if width > 2000:
        return "very_wide"
    if width > 1200:
        return "wide"
    return "normal"


def _verify_frozen_phase3a(phase3a_dir: Path) -> dict[str, Any]:
    artifacts = [
        "formula_predictions.jsonl",
        "formula_review.jsonl",
        "benchmark_summary.json",
        "runtime_fingerprint.json",
    ]
    missing = [name for name in artifacts if not (phase3a_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Phase 3A artifacts: {missing}")
    predictions = _read_jsonl(phase3a_dir / "formula_predictions.jsonl")
    review = _read_jsonl(phase3a_dir / "formula_review.jsonl")
    summary = json.loads(
        (phase3a_dir / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    prediction_by_sha = {row["png_content_sha256"]: row for row in predictions}
    review_by_sha = {row["png_content_sha256"]: row for row in review}
    checks = {
        "prediction_count_157": len(predictions) == 157,
        "review_count_157": len(review) == 157,
        "prediction_png_unique": len(prediction_by_sha) == len(predictions),
        "review_png_unique": len(review_by_sha) == len(review),
        "prediction_review_png_match": prediction_by_sha.keys() == review_by_sha.keys(),
        "raw_predictions_match_review": all(
            prediction_by_sha[sha]["formula_net_raw_latex"]
            == review_by_sha[sha]["formula_net_raw_latex"]
            for sha in prediction_by_sha.keys() & review_by_sha.keys()
        ),
        "summary_counts_157_199_245": summary["dataset"] == {
            **summary["dataset"],
            "unique_png_content": 157,
            "unique_candidates": 199,
            "occurrences": 245,
        },
        "png_sha256_match": all(
            Path(row["source_png"]).is_file()
            and _sha256_file(Path(row["source_png"])) == row["png_content_sha256"]
            for row in review
        ),
    }
    phase26_dir = phase3a_dir.parent / "phase26_docx_formula_census_v2"
    phase26_hashes = {
        "true_math_ocr_candidates.jsonl": _sha256_file(
            phase26_dir / "true_math_ocr_candidates.jsonl"
        ),
        "census_summary.json": _sha256_file(phase26_dir / "census_summary.json"),
    }
    checks["phase26_true_math_sha256"] = (
        phase26_hashes["true_math_ocr_candidates.jsonl"]
        == PHASE26_TRUE_MATH_SHA256
    )
    checks["phase26_summary_sha256"] = (
        phase26_hashes["census_summary.json"] == PHASE26_SUMMARY_SHA256
    )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Frozen Phase 3A verification failed: {failed}")
    return {
        "all_ok": True,
        "checks": checks,
        "artifact_sha256": {
            name: _sha256_file(phase3a_dir / name) for name in artifacts
        },
        "phase26_sha256": phase26_hashes,
    }


def _verify_reference_against_phase3a(rows, phase3a_dir):
    prediction_by_sha = {
        row["png_content_sha256"]: row
        for row in _read_jsonl(phase3a_dir / "formula_predictions.jsonl")
    }
    if set(prediction_by_sha) != {row["png_content_sha256"] for row in rows}:
        raise ValueError("Reference GT PNG set does not match frozen Phase 3A")
    for row in rows:
        prediction = prediction_by_sha[row["png_content_sha256"]]
        if row["formula_net_raw_latex"] != prediction["formula_net_raw_latex"]:
            raise ValueError(
                f"Frozen raw prediction changed for review_index={row['review_index']}"
            )


def _validate_unique_reference_rows(rows):
    indices = [row["review_index"] for row in rows]
    png_shas = [row["png_content_sha256"] for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError("Reference GT contains duplicate review_index values")
    if len(set(png_shas)) != len(png_shas):
        raise ValueError("Reference GT contains duplicate PNG SHA values")


def _review_html(rows, reference_path, output_dir):
    cards = []
    for row in rows:
        source_path = Path(row.get("source_png", ""))
        image_src = (
            Path(os.path.relpath(source_path, output_dir)).as_posix()
            if source_path.is_absolute()
            else row.get("source_png_relative_to_review", source_path.as_posix())
        )
        sources = []
        for source in row.get("sources", []):
            context = source.get("context", {})
            sources.append(
                "<li><code>"
                + html.escape(source.get("source_docx", "UNKNOWN"))
                + "</code><br><strong>Locator:</strong> <code>"
                + html.escape(source.get("source_locator", "UNKNOWN"))
                + "</code><br><strong>Signature:</strong> <code>"
                + html.escape(source.get("formula_signature", "UNKNOWN"))
                + "</code><br><strong>ProgID:</strong> <code>"
                + html.escape(source.get("ole_prog_id") or "UNKNOWN")
                + "</code><br><strong>Context:</strong> "
                + html.escape(context.get("previous_text", ""))
                + " ⟦ "
                + html.escape(context.get("current_text", ""))
                + " ⟧ "
                + html.escape(context.get("next_text", ""))
                + "</li>"
            )
        status = row["recognition_status"]
        cards.append(
            f"""<article id="formula-{row['review_index']}" data-status="{status}" data-source="{row['source_content_status']}">
<h2>#{row['review_index']:03d} · {html.escape(status)}</h2>
<img src="{html.escape(image_src)}" alt="formula source">
<dl><dt>PNG SHA</dt><dd><code>{row['png_content_sha256']}</code></dd>
<dt>Size</dt><dd>{row['width']} × {row['height']} · {row['derived_size_stratum']}</dd>
<dt>Batch 1 raw LaTeX</dt><dd><pre>{html.escape(row['formula_net_raw_latex'] or '')}</pre></dd>
<dt>Reference GT</dt><dd><pre>{html.escape(row['ground_truth_latex'] or 'null')}</pre></dd>
<dt>Source status</dt><dd>{html.escape(row['source_content_status'])}</dd>
<dt>Recognition / confidence</dt><dd>{html.escape(status)} / {row['review_confidence']}</dd>
<dt>Normalization</dt><dd>{html.escape(row['ground_truth_normalization'])}</dd>
<dt>Validator</dt><dd>{html.escape(row['ocr_validator_verdict'])}<pre>{html.escape(json.dumps(row['ocr_validator_issues'], ensure_ascii=False, indent=2))}</pre></dd>
<dt>Context used</dt><dd>{str(row['context_used']).lower()}</dd>
<dt>Notes</dt><dd>{html.escape(row['review_notes'])}</dd></dl>
<details><summary>Sources and context ({len(sources)})</summary><ol>{''.join(sources)}</ol></details>
</article>"""
        )
    reference_name = html.escape(reference_path.name)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 3A.5 Formula Reference GT</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1150px}}article{{border-top:2px solid #bbb;padding:1rem 0 2rem}}article[data-status*="ERROR"],article[data-status="HALLUCINATION"],article[data-status="OMISSION"]{{border-color:#c62828;background:#fff7f7}}img{{max-width:100%;background:white;border:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}dt{{font-weight:700;margin-top:.6rem}}code{{overflow-wrap:anywhere}}nav{{position:sticky;top:0;background:white;padding:.5rem;border-bottom:1px solid #ddd}}</style>
<script>function filterCards(value){{for(const card of document.querySelectorAll('article')){{card.hidden=value!==''&&card.dataset.status!==value&&card.dataset.source!==value}}}}</script></head>
<body><h1>PP-FormulaNet_plus-L Reference Ground Truth Review</h1>
<p>Agent-visual-reviewed reference GT from <code>{reference_name}</code>; it is not user-signed human-verified gold data.</p>
<nav><label>Filter <select onchange="filterCards(this.value)"><option value="">All</option>{_filter_options(rows)}</select></label></nav>
{''.join(cards)}</body></html>"""


def _filter_options(rows):
    statuses = sorted({row["recognition_status"] for row in rows})
    sources = sorted({row["source_content_status"] for row in rows})
    return "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in statuses + sources
    )


def _invert_groups(groups):
    result = {}
    for value, indices in groups.items():
        for index in indices:
            if index in result:
                raise ValueError(f"Review index {index} appears in multiple groups")
            result[index] = value
    return result


def _complete_counts(counter, values):
    return {value: counter.get(value, 0) for value in sorted(values)}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
