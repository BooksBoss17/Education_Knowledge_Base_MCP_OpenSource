from __future__ import annotations

import hashlib
import html
import json
import math
import os
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree
from PIL import Image

from . import formulanet_runtime as _formulanet_runtime
from .formulanet_runtime import (
    FormulaNetRuntime,
    FormulaOcrOutputValidator,
    OcrValidation,
    OcrVerdict,
    PaddleFormulaNetRuntime,
    benchmark_cache_key,
)
from .namespaces import NS

_directory_fingerprint = _formulanet_runtime._directory_fingerprint
_nvidia_smi_metadata = _formulanet_runtime._nvidia_smi_metadata
_package_version = _formulanet_runtime._package_version
_paddle_compatible_model_dir = _formulanet_runtime._paddle_compatible_model_dir


@dataclass(frozen=True)
class BenchmarkImage:
    png_content_sha256: str
    path: Path
    width: int
    height: int
    occurrences: tuple[dict[str, Any], ...]
    candidate_fingerprints: tuple[str, ...]

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrences)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_fingerprints)


@dataclass(frozen=True)
class BenchmarkDataset:
    source_dir: Path
    occurrences: tuple[dict[str, Any], ...]
    images: tuple[BenchmarkImage, ...]
    integrity: dict[str, Any]

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrences)

    @property
    def candidate_count(self) -> int:
        return len({row["candidate_fingerprint"] for row in self.occurrences})

    @property
    def unique_png_count(self) -> int:
        return len(self.images)


@dataclass(frozen=True)
class FormulaNetBenchmarkResult:
    output_dir: Path
    summary_path: Path
    predictions_path: Path
    candidate_predictions_path: Path
    occurrence_predictions_path: Path
    review_path: Path
    review_html_path: Path
    runtime_fingerprint_path: Path
    failures_path: Path
    summary: dict[str, Any]


def load_benchmark_dataset(source_dir: str | Path) -> BenchmarkDataset:
    source_dir = Path(source_dir).resolve()
    source_path = source_dir / "true_math_ocr_candidates.jsonl"
    summary_path = source_dir / "census_summary.json"
    rows = tuple(_read_jsonl(source_path))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = summary["candidates"]["true_math_ocr"]

    path_ok = True
    hashes_ok = True
    dimensions_ok = True
    bytes_ok = True
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    resolved_paths: dict[str, Path] = {}
    for row in rows:
        relative = Path(row["rendered_png_path"])
        path = (source_dir / relative).resolve()
        try:
            path.relative_to(source_dir)
        except ValueError:
            path_ok = False
            continue
        if not path.is_file():
            path_ok = False
            continue
        data = path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        hashes_ok &= actual_sha == row["png_sha256"]
        bytes_ok &= len(data) == row["png_bytes"]
        with Image.open(path) as image:
            dimensions_ok &= image.size == (row["width"], row["height"])
        groups[row["png_sha256"]].append(row)
        resolved_paths.setdefault(row["png_sha256"], path)

    candidate_ids_unique = len({row["candidate_id"] for row in rows}) == len(rows)
    candidate_to_png: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        candidate_to_png[row["candidate_fingerprint"]].add(row["png_sha256"])
    candidate_mapping_ok = all(len(value) == 1 for value in candidate_to_png.values())
    summary_counts_match = (
        len(rows) == expected["occurrences"]
        and len(candidate_to_png) == expected["unique_fingerprints"]
        and len(groups) == expected["unique_png"]
    )
    checks = {
        "source_jsonl_exists": source_path.is_file(),
        "summary_exists": summary_path.is_file(),
        "summary_counts_match": summary_counts_match,
        "candidate_ids_unique": candidate_ids_unique,
        "candidate_to_png_mapping_consistent": candidate_mapping_ok,
        "png_paths_safe_and_present": path_ok and len(groups) > 0,
        "png_sha256_match": hashes_ok,
        "png_bytes_match": bytes_ok,
        "png_dimensions_match": dimensions_ok,
    }
    integrity = {"all_ok": all(checks.values()), "checks": checks}
    integrity["source_fingerprints"] = {
        "true_math_ocr_candidates_jsonl_sha256": _sha256_file(source_path),
        "census_summary_json_sha256": _sha256_file(summary_path),
    }
    if not integrity["all_ok"]:
        raise ValueError(
            "Phase 2.6 FormulaNet input integrity failed: "
            + json.dumps(integrity, ensure_ascii=False, sort_keys=True)
        )

    images = []
    for png_sha in sorted(groups):
        occurrences = tuple(groups[png_sha])
        representative = occurrences[0]
        images.append(
            BenchmarkImage(
                png_content_sha256=png_sha,
                path=resolved_paths[png_sha],
                width=representative["width"],
                height=representative["height"],
                occurrences=occurrences,
                candidate_fingerprints=tuple(
                    sorted({row["candidate_fingerprint"] for row in occurrences})
                ),
            )
        )
    return BenchmarkDataset(source_dir, rows, tuple(images), integrity)


def run_formulanet_benchmark(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    runtime_factory: Callable[[], FormulaNetRuntime] | None = None,
    cache_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    model_revision: str | None = None,
    batch_sizes: Sequence[int] = (),
) -> FormulaNetBenchmarkResult:
    started = time.perf_counter()
    dataset = load_benchmark_dataset(source_dir)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    validator = FormulaOcrOutputValidator()

    factory = runtime_factory or (
        lambda: PaddleFormulaNetRuntime(
            model_dir=model_dir,
            model_revision=model_revision,
        )
    )
    runtime = factory()
    runtime_fingerprint = runtime.fingerprint()
    inference_config = {
        "device": runtime_fingerprint["device"],
        "precision": runtime_fingerprint["precision"],
        "backend": runtime_fingerprint["backend"],
        "batch_size": 1,
    }
    cache_path = Path(cache_path).resolve() if cache_path else output_dir / "prediction_cache.jsonl"
    cache = _load_prediction_cache(cache_path)

    warmup_started = time.perf_counter()
    runtime.predict([dataset.images[0].path], batch_size=1)
    warmup_seconds = time.perf_counter() - warmup_started

    predictions = []
    inference_times: list[float] = []
    cache_hits = 0
    for image in dataset.images:
        key = benchmark_cache_key(
            image.png_content_sha256, runtime_fingerprint, inference_config
        )
        cached = cache.get(key)
        if cached and cached.get("inference_status") == "SUCCESS":
            raw_latex = cached["formula_net_raw_latex"]
            inference_seconds = cached["inference_seconds"]
            cache_hit = True
            cache_hits += 1
            inference_status = "SUCCESS"
            inference_error = None
        else:
            call_started = time.perf_counter()
            try:
                raw_latex = runtime.predict([image.path], batch_size=1)[0]
                inference_seconds = time.perf_counter() - call_started
                inference_times.append(inference_seconds)
                inference_status = "SUCCESS"
                inference_error = None
                cache_hit = False
            except Exception as exc:  # noqa: BLE001 - record per-image model failure
                inference_seconds = time.perf_counter() - call_started
                raw_latex = None
                inference_status = "FAILED"
                inference_error = f"{type(exc).__name__}: {exc}"
                cache_hit = False
                failures.append(
                    {
                        "png_content_sha256": image.png_content_sha256,
                        "representative_png_path": str(image.path),
                        "error": inference_error,
                    }
                )
        if inference_status == "SUCCESS":
            validation = validator.validate(raw_latex)
        else:
            validation = OcrValidation(
                OcrVerdict.INFERENCE_FAILED,
                (
                    {
                        "code": "INFERENCE_FAILED",
                        "message": inference_error,
                        "severity": "error",
                    },
                ),
            )
        source_signatures = sorted(
            {row["formula_signature"] for row in image.occurrences}
        )
        source_prog_ids = sorted(
            {row.get("ole_prog_id") or "UNKNOWN" for row in image.occurrences}
        )
        rejection_reasons = _rejection_reasons(image.occurrences)
        prediction = {
            "png_content_sha256": image.png_content_sha256,
            "representative_png_path": str(image.path),
            "width": image.width,
            "height": image.height,
            "formula_net_raw_latex": raw_latex,
            "inference_status": inference_status,
            "inference_error": inference_error,
            "inference_seconds": inference_seconds,
            "cache_hit": cache_hit,
            "cache_key": key,
            "ocr_validator_verdict": validation.verdict.value,
            "ocr_validator_issues": list(validation.issues),
            "source_candidate_count": image.candidate_count,
            "source_occurrence_count": image.occurrence_count,
            "source_signatures": source_signatures,
            "source_prog_ids": source_prog_ids,
            "structural_rejection_reasons": rejection_reasons,
        }
        predictions.append(prediction)
        if not cache_hit and inference_status == "SUCCESS":
            cache[key] = prediction
            _write_jsonl(cache_path, cache.values())

    prediction_by_sha = {
        row["png_content_sha256"]: row for row in predictions
    }
    candidate_rows = _candidate_fanout(dataset, prediction_by_sha)
    occurrence_rows = _occurrence_fanout(dataset, prediction_by_sha)
    contexts = FormulaContextExtractor()
    review_rows = _review_rows(dataset, prediction_by_sha, output_dir, contexts)
    batch_results = _benchmark_batches(
        runtime,
        dataset,
        prediction_by_sha,
        batch_sizes=tuple(size for size in batch_sizes if size > 1),
    )
    size_buckets = _size_buckets(dataset.images)
    summary = {
        "schema": "bemarkdown-formulanet-benchmark-phase3a-v1",
        "dataset": {
            "occurrences": dataset.occurrence_count,
            "unique_candidates": dataset.candidate_count,
            "unique_png_content": dataset.unique_png_count,
            "integrity": dataset.integrity,
        },
        "runtime_fingerprint": runtime_fingerprint,
        "inference_config": inference_config,
        "automatic_quality": _quality_counts(
            predictions, candidate_rows, occurrence_rows
        ),
        "groups": {
            "formula_signature": _group_metrics(
                occurrence_rows, lambda row: [row["formula_signature"]]
            ),
            "prog_id": _group_metrics(
                occurrence_rows, lambda row: [row.get("ole_prog_id") or "UNKNOWN"]
            ),
            "structural_rejection_reason": _group_metrics(
                occurrence_rows,
                lambda row: row["structural_rejection_reasons"],
            ),
            "width_bucket": _group_metrics(
                occurrence_rows,
                lambda row: [
                    _bucket(row["width"], size_buckets["width"], axis="width")
                ],
            ),
            "height_bucket": _group_metrics(
                occurrence_rows,
                lambda row: [
                    _bucket(row["height"], size_buckets["height"], axis="height")
                ],
            ),
        },
        "size_buckets": size_buckets,
        "performance": {
            "model_load_seconds": runtime.load_seconds,
            "warmup_seconds": warmup_seconds,
            "baseline_batch_size": 1,
            "cache_hits": cache_hits,
            "actual_inference_count": len(inference_times),
            **_timing_distribution(inference_times),
            "gpu_peak_memory_bytes": runtime.peak_gpu_memory_bytes(),
            "gpu_peak_memory_source": "paddle.device.cuda.max_memory_allocated",
            "batch_comparisons": batch_results,
        },
        "ground_truth": {
            "reviewed": 0,
            "unreviewed": len(review_rows),
            "exact_match_accuracy": "NOT_YET_AVAILABLE",
        },
        "failures": len(failures),
        "cache": {
            "path": str(cache_path),
            "key_fields": [
                "png_content_sha256",
                "runtime_fingerprint",
                "inference_config",
            ],
        },
        "outputs": {
            "benchmark_summary": "benchmark_summary.json",
            "formula_predictions": "formula_predictions.jsonl",
            "candidate_fanout": "formula_candidate_predictions.jsonl",
            "occurrence_fanout": "formula_occurrence_predictions.jsonl",
            "formula_review": "formula_review.jsonl",
            "formula_review_html": "formula_review.html",
            "runtime_fingerprint": "runtime_fingerprint.json",
            "failures": "failures.jsonl",
        },
        "wall_seconds": time.perf_counter() - started,
    }

    paths = {
        "summary": output_dir / "benchmark_summary.json",
        "predictions": output_dir / "formula_predictions.jsonl",
        "candidates": output_dir / "formula_candidate_predictions.jsonl",
        "occurrences": output_dir / "formula_occurrence_predictions.jsonl",
        "review": output_dir / "formula_review.jsonl",
        "html": output_dir / "formula_review.html",
        "runtime": output_dir / "runtime_fingerprint.json",
        "failures": output_dir / "failures.jsonl",
    }
    _write_json(paths["summary"], summary)
    _write_json(paths["runtime"], runtime_fingerprint)
    _write_jsonl(paths["predictions"], predictions)
    _write_jsonl(paths["candidates"], candidate_rows)
    _write_jsonl(paths["occurrences"], occurrence_rows)
    _write_jsonl(paths["review"], review_rows)
    _write_jsonl(paths["failures"], failures)
    paths["html"].write_text(
        _review_html(review_rows), encoding="utf-8", newline="\n"
    )
    return FormulaNetBenchmarkResult(
        output_dir,
        paths["summary"],
        paths["predictions"],
        paths["candidates"],
        paths["occurrences"],
        paths["review"],
        paths["html"],
        paths["runtime"],
        paths["failures"],
        summary,
    )


class FormulaContextExtractor:
    def __init__(self):
        self._documents: dict[Path, tuple[etree._Element, list[etree._Element]]] = {}

    def extract(self, occurrence: dict[str, Any]) -> dict[str, Any]:
        source = Path(occurrence["source_file"])
        try:
            root, paragraphs = self._load(source)
            matches = root.xpath(occurrence["source_locator"], namespaces=NS)
            if len(matches) != 1:
                raise ValueError(f"locator matched {len(matches)} nodes")
            target = matches[0]
            paragraph = next(target.iterancestors(f"{{{NS['w']}}}p"), None)
            if paragraph is None:
                raise ValueError("formula object has no paragraph ancestor")
            index = paragraphs.index(paragraph)
            previous = self._plain_text(paragraphs[index - 1]) if index else ""
            current = self._plain_text(paragraph, target=target)
            following = (
                self._plain_text(paragraphs[index + 1])
                if index + 1 < len(paragraphs)
                else ""
            )
            return {
                "status": "OK",
                "previous_text": _truncate(previous),
                "current_text": _truncate(current),
                "next_text": _truncate(following),
            }
        except Exception as exc:  # noqa: BLE001 - context is advisory metadata
            return {
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
                "previous_text": "",
                "current_text": "[FORMULA]",
                "next_text": "",
            }

    def _load(self, source: Path):
        source = source.resolve()
        if source not in self._documents:
            with zipfile.ZipFile(source) as archive:
                root = etree.fromstring(archive.read("word/document.xml"))
            paragraphs = root.xpath("//w:p", namespaces=NS)
            self._documents[source] = (root, paragraphs)
        return self._documents[source]

    def _plain_text(
        self, node: etree._Element, *, target: etree._Element | None = None
    ) -> str:
        pieces: list[str] = []

        def visit(current: etree._Element) -> None:
            if current is target:
                pieces.append("[FORMULA]")
                return
            if current.tag in {f"{{{NS['w']}}}t", f"{{{NS['m']}}}t"}:
                pieces.append(current.text or "")
                return
            if current.tag == f"{{{NS['w']}}}tab":
                pieces.append("\t")
            elif current.tag in {f"{{{NS['w']}}}br", f"{{{NS['w']}}}cr"}:
                pieces.append("\n")
            for child in current:
                visit(child)

        visit(node)
        return " ".join("".join(pieces).split())


def _candidate_fanout(dataset, prediction_by_sha):
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset.occurrences:
        grouped[row["candidate_fingerprint"]].append(row)
    result = []
    for fingerprint in sorted(grouped):
        occurrences = grouped[fingerprint]
        png_sha = occurrences[0]["png_sha256"]
        prediction = prediction_by_sha[png_sha]
        result.append(
            {
                "candidate_fingerprint": fingerprint,
                "png_content_sha256": png_sha,
                "occurrence_count": len(occurrences),
                "candidate_ids": sorted(row["candidate_id"] for row in occurrences),
                "formula_net_raw_latex": prediction["formula_net_raw_latex"],
                "inference_status": prediction["inference_status"],
                "ocr_validator_verdict": prediction["ocr_validator_verdict"],
                "ocr_validator_issues": prediction["ocr_validator_issues"],
            }
        )
    return result


def _occurrence_fanout(dataset, prediction_by_sha):
    result = []
    for row in dataset.occurrences:
        prediction = prediction_by_sha[row["png_sha256"]]
        result.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_fingerprint": row["candidate_fingerprint"],
                "png_content_sha256": row["png_sha256"],
                "source_file": row["source_file"],
                "source_locator": row["source_locator"],
                "formula_signature": row["formula_signature"],
                "ole_prog_id": row.get("ole_prog_id"),
                "width": row["width"],
                "height": row["height"],
                "structural_rejection_reasons": _rejection_reasons((row,)),
                "formula_net_raw_latex": prediction["formula_net_raw_latex"],
                "inference_status": prediction["inference_status"],
                "ocr_validator_verdict": prediction["ocr_validator_verdict"],
                "ocr_validator_issues": prediction["ocr_validator_issues"],
            }
        )
    return result


def _review_rows(dataset, prediction_by_sha, output_dir, contexts):
    rows = []
    for index, image in enumerate(dataset.images, start=1):
        prediction = prediction_by_sha[image.png_content_sha256]
        sources = []
        for occurrence in image.occurrences:
            sources.append(
                {
                    "candidate_id": occurrence["candidate_id"],
                    "candidate_fingerprint": occurrence["candidate_fingerprint"],
                    "source_docx": occurrence["source_file"],
                    "source_locator": occurrence["source_locator"],
                    "formula_signature": occurrence["formula_signature"],
                    "ole_prog_id": occurrence.get("ole_prog_id"),
                    "structural_rejection_reasons": _rejection_reasons((occurrence,)),
                    "context": contexts.extract(occurrence),
                }
            )
        rows.append(
            {
                "review_index": index,
                "png_content_sha256": image.png_content_sha256,
                "source_png": str(image.path),
                "source_png_relative_to_review": Path(
                    os.path.relpath(image.path, output_dir)
                ).as_posix(),
                "width": image.width,
                "height": image.height,
                "formula_net_raw_latex": prediction["formula_net_raw_latex"],
                "inference_status": prediction["inference_status"],
                "ocr_validator_verdict": prediction["ocr_validator_verdict"],
                "ocr_validator_issues": prediction["ocr_validator_issues"],
                "source_candidate_count": image.candidate_count,
                "source_occurrence_count": image.occurrence_count,
                "sources": sources,
                "review_status": "UNREVIEWED",
                "review_status_options": [
                    "EXACT",
                    "SEMANTICALLY_CORRECT_STYLE_DIFFERENT",
                    "MINOR_ERROR",
                    "MAJOR_ERROR",
                    "NON_FORMULA_VISIBLE_CONTENT",
                    "UNREADABLE_SOURCE",
                    "UNREVIEWED",
                ],
                "ground_truth_latex": None,
                "review_notes": "",
            }
        )
    return rows


def _review_html(rows):
    cards = []
    for row in rows:
        sources = []
        for source in row["sources"]:
            context = source["context"]
            sources.append(
                "<li><code>"
                + html.escape(source["source_docx"])
                + "</code><br>"
                + html.escape(source["source_locator"])
                + "<br><strong>Candidate:</strong> <code>"
                + html.escape(source["candidate_id"])
                + "</code>"
                + "<br><strong>Signature:</strong> <code>"
                + html.escape(source["formula_signature"])
                + "</code>"
                + "<br><strong>ProgID:</strong> <code>"
                + html.escape(source["ole_prog_id"] or "UNKNOWN")
                + "</code>"
                + "<br><strong>Structural rejection:</strong> <code>"
                + html.escape(", ".join(source["structural_rejection_reasons"]))
                + "</code>"
                + "<br><strong>Context:</strong> "
                + html.escape(context["previous_text"])
                + " ⟦ "
                + html.escape(context["current_text"])
                + " ⟧ "
                + html.escape(context["next_text"])
                + "</li>"
            )
        cards.append(
            f"""<article id="formula-{row['review_index']}">
<h2>#{row['review_index']} · {html.escape(row['ocr_validator_verdict'])}</h2>
<img src="{html.escape(row['source_png_relative_to_review'])}" alt="formula source">
<dl><dt>PNG SHA</dt><dd><code>{row['png_content_sha256']}</code></dd>
<dt>Size</dt><dd>{row['width']} × {row['height']}</dd>
<dt>Raw LaTeX</dt><dd><pre>{html.escape(row['formula_net_raw_latex'] or '')}</pre></dd>
<dt>Validator issues</dt><dd><pre>{html.escape(json.dumps(row['ocr_validator_issues'], ensure_ascii=False, indent=2))}</pre></dd>
<dt>Review</dt><dd>UNREVIEWED · ground_truth_latex: null · review_notes: empty</dd></dl>
<details><summary>Sources and context ({len(sources)})</summary><ol>{''.join(sources)}</ol></details>
</article>"""
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Phase 3A Formula Review</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1100px}}article{{border-top:2px solid #bbb;padding:1rem 0 2rem}}img{{max-width:100%;background:white;border:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}dt{{font-weight:700;margin-top:.6rem}}code{{overflow-wrap:anywhere}}</style></head>
<body><h1>PP-FormulaNet_plus-L Review Set</h1>
<p>Raw model output only. Exact Match accuracy is not available until a human fills the review fields in formula_review.jsonl.</p>
{''.join(cards)}</body></html>"""


def _quality_counts(predictions, candidates, occurrences):
    verdicts = [verdict.value for verdict in OcrVerdict]

    def verdict_counts(rows):
        counter = Counter(row["ocr_validator_verdict"] for row in rows)
        return {verdict: counter.get(verdict, 0) for verdict in verdicts}

    def issue_counts(rows):
        counter = Counter()
        for row in rows:
            counter.update(
                {issue["code"] for issue in row["ocr_validator_issues"]}
            )
        return dict(sorted(counter.items()))

    return {
        "unique_png_content": verdict_counts(predictions),
        "unique_candidates": verdict_counts(candidates),
        "occurrences": verdict_counts(occurrences),
        "validator_issue_distribution": {
            "unique_png_content": issue_counts(predictions),
            "unique_candidates": issue_counts(candidates),
            "occurrences": issue_counts(occurrences),
        },
        "inference_success": {
            "unique_png_content": sum(
                row["inference_status"] == "SUCCESS" for row in predictions
            ),
            "unique_candidates": sum(
                row["inference_status"] == "SUCCESS" for row in candidates
            ),
            "occurrences": sum(
                row["inference_status"] == "SUCCESS" for row in occurrences
            ),
        },
    }


def _group_metrics(rows, categories):
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for category in sorted(set(categories(row))):
            groups[category].append(row)
    result = {}
    for category in sorted(groups):
        members = groups[category]
        png_members = {
            row["png_content_sha256"]: row for row in members
        }.values()
        candidate_members = {
            row["candidate_fingerprint"]: row for row in members
        }.values()
        result[category] = {
            "unique_png_content": len(
                {row["png_content_sha256"] for row in members}
            ),
            "unique_candidates": len(
                {row["candidate_fingerprint"] for row in members}
            ),
            "occurrences": len(members),
            "unique_png_verdicts": dict(
                sorted(
                    Counter(
                        row["ocr_validator_verdict"] for row in png_members
                    ).items()
                )
            ),
            "unique_candidate_verdicts": dict(
                sorted(
                    Counter(
                        row["ocr_validator_verdict"] for row in candidate_members
                    ).items()
                )
            ),
            "occurrence_verdicts": dict(
                sorted(
                    Counter(
                        row["ocr_validator_verdict"] for row in members
                    ).items()
                )
            ),
        }
    return result


def _rejection_reasons(occurrences):
    reasons = {
        issue["code"]
        for row in occurrences
        for attempt in row.get("structural_attempts", [])
        for issue in attempt.get("validator_issues", [])
    }
    return sorted(reasons or {"no_structural_payload"})


def _size_buckets(images):
    widths = sorted(image.width for image in images)
    heights = sorted(image.height for image in images)
    return {
        "width": {
            "small_max": _percentile(widths, 25),
            "medium_max": _percentile(widths, 50),
            "wide_max": _percentile(widths, 95),
            "max": max(widths),
        },
        "height": {
            "small_max": _percentile(heights, 25),
            "medium_max": _percentile(heights, 50),
            "wide_max": _percentile(heights, 95),
            "max": max(heights),
        },
    }


def _bucket(value, thresholds, *, axis):
    labels = (
        ("small", "medium", "wide", "very_wide")
        if axis == "width"
        else ("short", "medium", "tall", "very_tall")
    )
    if value <= thresholds["small_max"]:
        return labels[0]
    if value <= thresholds["medium_max"]:
        return labels[1]
    if value <= thresholds["wide_max"]:
        return labels[2]
    return labels[3]


def _benchmark_batches(runtime, dataset, baseline, *, batch_sizes):
    results = []
    for batch_size in batch_sizes:
        started = time.perf_counter()
        try:
            outputs = runtime.predict(
                [image.path for image in dataset.images], batch_size=batch_size
            )
            seconds = time.perf_counter() - started
            differences = []
            for image, raw in zip(dataset.images, outputs, strict=True):
                baseline_raw = baseline[image.png_content_sha256][
                    "formula_net_raw_latex"
                ]
                if raw != baseline_raw:
                    differences.append(
                        {
                            "png_content_sha256": image.png_content_sha256,
                            "baseline_raw_latex": baseline_raw,
                            "batch_raw_latex": raw,
                        }
                    )
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "SUCCESS",
                    "total_seconds": seconds,
                    "throughput_images_per_second": len(outputs) / seconds,
                    "raw_latex_difference_count": len(differences),
                    "raw_latex_differences": differences,
                }
            )
        except Exception as exc:  # noqa: BLE001 - batch experiment is optional
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def _timing_distribution(values):
    if not values:
        return {
            "total_inference_seconds": 0.0,
            "per_image_mean_seconds": None,
            "per_image_median_seconds": None,
            "per_image_p95_seconds": None,
            "per_image_max_seconds": None,
            "throughput_images_per_second": None,
        }
    total = sum(values)
    return {
        "total_inference_seconds": total,
        "per_image_mean_seconds": statistics.mean(values),
        "per_image_median_seconds": statistics.median(values),
        "per_image_p95_seconds": _percentile(sorted(values), 95),
        "per_image_max_seconds": max(values),
        "throughput_images_per_second": len(values) / total,
    }


def _percentile(values, percentile):
    index = min(
        len(values) - 1,
        max(0, math.ceil(percentile / 100 * len(values)) - 1),
    )
    return values[index]


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_prediction_cache(path):
    if not path.is_file():
        return {}
    return {
        row["cache_key"]: row
        for row in _read_jsonl(path)
        if row.get("cache_key")
    }


def _truncate(value, limit=500):
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
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
