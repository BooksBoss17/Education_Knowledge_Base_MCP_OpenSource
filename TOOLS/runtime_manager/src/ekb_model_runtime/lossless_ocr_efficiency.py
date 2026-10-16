"""Deterministic analysis primitives for lossless OCR efficiency research.

The module deliberately contains no model loading or filesystem writes.  Benchmark
workers and artifact builders use this small seam so selection, parity, percentile,
and promotion math can be verified without a GPU or experimental dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _percentile(values: Sequence[float | int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def output_length_distribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize generated token counts with distribution-derived length buckets."""

    records = [dict(row) for row in rows]
    lengths = [
        int(row.get("generated_token_count", row.get("generated_tokens", 0)))
        for row in records
    ]
    percentiles = {
        name: _percentile(lengths, quantile)
        for name, quantile in (
            ("p50", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
            ("p95", 0.95),
            ("p99", 0.99),
        )
    }
    thresholds = {
        "short_max_tokens": percentiles["p50"],
        "medium_max_tokens": percentiles["p75"],
    }
    buckets: dict[str, list[int]] = {"SHORT": [], "MEDIUM": [], "LONG": []}
    for length in lengths:
        if length <= thresholds["short_max_tokens"]:
            buckets["SHORT"].append(length)
        elif length <= thresholds["medium_max_tokens"]:
            buckets["MEDIUM"].append(length)
        else:
            buckets["LONG"].append(length)
    return {
        "population": len(lengths),
        "percentiles": percentiles,
        "min": min(lengths, default=0),
        "max": max(lengths, default=0),
        "mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "thresholds": thresholds,
        "buckets": {
            name: {
                "count": len(values),
                "rate": len(values) / len(lengths) if lengths else 0.0,
                "min": min(values, default=0),
                "max": max(values, default=0),
            }
            for name, values in buckets.items()
        },
    }


def _text_features(text: str) -> tuple[str, ...]:
    features = []
    if re.search(r"[\u3400-\u9fff]", text):
        features.append("CHINESE")
    if re.search(r"[A-Za-z]", text) and "CHINESE" in features:
        features.append("MIXED_CHINESE_ENGLISH")
    if re.search(r"\d", text):
        features.append("NUMERIC")
    if re.search(r"(?:m/s|kg|Hz|N|Pa|J|W|V|A|Ω|℃|°|cm|mm|km|mol)", text):
        features.append("UNIT")
    if re.search(r"[α-ωΑ-ΩπμΩ±×÷√∑∫∆Δ]", text):
        features.append("GREEK_OR_SPECIAL")
    return tuple(features or ("GENERAL",))


def _length_bucket(length: int, p50: float, p75: float) -> str:
    if length <= p50:
        return "SHORT"
    if length <= p75:
        return "MEDIUM"
    return "LONG"


def _select_representative(
    rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count < 1 or count > len(rows):
        raise ValueError("MICROBENCHMARK_COUNT_OUT_OF_RANGE")
    lengths = [
        int(outputs.get(str(row["sample_id"]), {}).get("generated_token_count", 0))
        for row in rows
    ]
    p50 = _percentile(lengths, 0.50)
    p75 = _percentile(lengths, 0.75)
    ranked = []
    for row in rows:
        sample_id = str(row["sample_id"])
        output = outputs.get(sample_id, {})
        length = int(output.get("generated_token_count", 0))
        text = str(output.get("normalized_output") or output.get("extracted_text") or "")
        dimensions = (
            _length_bucket(length, p50, p75),
            str(row.get("track") or "UNKNOWN"),
            str(row.get("stratum") or "UNKNOWN"),
            *_text_features(text),
        )
        digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        ranked.append((dimensions, digest, dict(row), length, _text_features(text)))

    # Greedily maximize new dimension coverage, then use the stable sample digest.
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    remaining = ranked[:]
    while remaining and len(selected) < count:
        remaining.sort(
            key=lambda item: (
                -sum(dimension not in covered for dimension in item[0]),
                item[1],
            )
        )
        dimensions, _digest, row, length, features = remaining.pop(0)
        covered.update(dimensions)
        selected.append(
            {
                "sample_id": str(row["sample_id"]),
                "crop_sha256": str(row["crop_sha256"]),
                "crop_local_relpath": str(row["crop_local_relpath"]),
                "track": str(row.get("track") or "UNKNOWN"),
                "stratum": str(row.get("stratum") or "UNKNOWN"),
                "source_relative_path": str(row.get("source_relative_path") or ""),
                "reference_generated_tokens": length,
                "reference_length_bucket": _length_bucket(length, p50, p75),
                "reference_text_features": list(features),
            }
        )
    return sorted(selected, key=lambda row: row["sample_id"])


def freeze_microbenchmark(
    rows: Sequence[Mapping[str, Any]],
    *,
    got_outputs: Mapping[str, Mapping[str, Any]],
    hunyuan_outputs: Mapping[str, Mapping[str, Any]] | None = None,
    hunyuan_conflict_ids: set[str],
    got_count: int = 100,
    hunyuan_count: int = 80,
) -> dict[str, Any]:
    """Freeze deterministic GOT and conflict-only Hunyuan benchmark subsets."""

    unique = {str(row["sample_id"]): dict(row) for row in rows}
    if len(unique) != len(rows):
        raise ValueError("MICROBENCHMARK_DUPLICATE_SAMPLE_ID")
    ordered = [unique[key] for key in sorted(unique)]
    conflict_rows = [row for row in ordered if str(row["sample_id"]) in hunyuan_conflict_ids]
    if len(conflict_rows) != len(hunyuan_conflict_ids):
        raise ValueError("MICROBENCHMARK_CONFLICT_ID_NOT_IN_BENCHMARK")
    got_samples = _select_representative(ordered, got_outputs, got_count)
    hunyuan_samples = _select_representative(
        conflict_rows, hunyuan_outputs or got_outputs, hunyuan_count
    )
    payload = {
        "schema": "ekb-lossless-ocr-efficiency-microbenchmark-lock-v1",
        "selection": "DETERMINISTIC_GREEDY_DIMENSION_COVERAGE",
        "dynamic_reselection_forbidden": True,
        "providers": {
            "GOT": {"population": got_count, "samples": got_samples},
            "HUNYUAN": {
                "population": hunyuan_count,
                "source_population": len(conflict_rows),
                "conflict_only": True,
                "samples": hunyuan_samples,
            },
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["lock_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def candidate_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    parity_matches: int,
    parity_compared: int,
) -> dict[str, Any]:
    """Apply the fail-closed microbenchmark shortlist gate to one candidate."""

    baseline_latency = float(baseline["mean_latency_seconds"])
    candidate_latency = float(candidate["mean_latency_seconds"])
    speedup = baseline_latency / candidate_latency if candidate_latency > 0 else 0.0
    baseline_vram = float(baseline["peak_torch_vram_bytes"])
    candidate_vram = float(candidate["peak_torch_vram_bytes"])
    vram_delta_percent = (
        (candidate_vram - baseline_vram) / baseline_vram * 100.0 if baseline_vram else 0.0
    )
    baseline_setup = float(baseline.get("model_load_seconds") or 0.0) + float(
        baseline.get("preparation_seconds") or 0.0
    )
    candidate_setup = float(candidate.get("model_load_seconds") or 0.0) + float(
        candidate.get("preparation_seconds") or 0.0
    )
    saving = baseline_latency - candidate_latency
    setup_penalty = candidate_setup - baseline_setup
    break_even = (
        max(0, math.ceil(setup_penalty / saving - 1e-12)) if saving > 0 else None
    )
    parity = parity_matches / parity_compared if parity_compared else 0.0
    reasons = []
    if parity != 1.0:
        reasons.append("NORMALIZED_PARITY_NOT_100_PERCENT")
    if int(candidate.get("oom_count") or 0):
        reasons.append("OOM")
    if int(candidate.get("cpu_fallback_count") or 0):
        reasons.append("CPU_FALLBACK")
    if candidate.get("error"):
        reasons.append("RUNTIME_ERROR")
    if speedup < 1.05 and vram_delta_percent > -10.0:
        reasons.append("BENEFIT_BELOW_SHORTLIST_THRESHOLD")
    return {
        "parity_matches": parity_matches,
        "parity_compared": parity_compared,
        "normalized_output_parity": parity,
        "speedup": round(speedup, 6),
        "latency_delta_percent": round((candidate_latency / baseline_latency - 1.0) * 100.0, 6),
        "vram_delta_bytes": int(candidate_vram - baseline_vram),
        "vram_delta_percent": round(vram_delta_percent, 6),
        "break_even_regions": break_even,
        "estimated_total_seconds": {
            "baseline": {
                str(population): round(
                    baseline_setup + baseline_latency * population, 6
                )
                for population in (10, 50, 100, 500)
            },
            "candidate": {
                str(population): round(
                    candidate_setup + candidate_latency * population, 6
                )
                for population in (10, 50, 100, 500)
            },
        },
        "rejection_reasons": reasons,
        "shortlist_eligible": not reasons,
    }
