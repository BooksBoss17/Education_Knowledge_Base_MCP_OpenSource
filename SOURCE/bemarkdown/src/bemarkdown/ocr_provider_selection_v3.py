"""Truth-side contracts for the OCR Provider Selection v3 benchmark.

The module contains deterministic cohort/crop selection and frozen-output
scoring only.  It does not construct OCR runtimes or participate in the
production PDF pipeline.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .ocr_ensemble_benchmark import classify_error, normalize_text, score_model

SELECTION_SEED = hashlib.sha256(
    b"bemarkdown-ocr-provider-selection-v3"
).hexdigest()
GOLD_STATUSES = {"NATIVE_GOLD", "UNRESOLVED", "OUT_OF_SCOPE"}
MODEL_IDS = ("P", "G", "S", "M")
LENGTH_BUCKETS = ("1-5", "6-15", "16-30", ">30")

_CHINESE = re.compile(r"[\u3400-\u9fff]")
_DIGIT = re.compile(r"\d")
_LATIN = re.compile(r"[A-Za-z]")
_PUNCTUATION = re.compile(r"[^\w\u3400-\u9fff\s]", re.UNICODE)
_UNIT = re.compile(
    r"(?<![A-Za-z])(?:m/s(?:²|2)?|m·s[⁻-]?\d*|kg|mol|Hz|Pa|J|W|V|A|Ω|N|C|T|s|m|g|K|%)(?![A-Za-z])"
)


def deterministic_page_cohort(
    documents: Sequence[Mapping[str, Any]],
    *,
    page_target: int = 20,
    seed: str = SELECTION_SEED,
) -> list[dict[str, Any]]:
    """Select pages by source SHA and page index, preferring one page per PDF."""

    if page_target <= 0:
        raise ValueError("PAGE_TARGET_MUST_BE_POSITIVE")
    canonical: dict[tuple[str, str], Mapping[str, Any]] = {}
    seen_sha: set[str] = set()
    for document in documents:
        path = str(document["canonical_path"])
        source_sha = str(document["sha256"])
        pages = int(document["page_count"])
        if len(source_sha) != 64 or pages <= 0:
            raise ValueError(f"INVALID_SOURCE_DOCUMENT:{path}")
        key = (path.casefold(), source_sha)
        if key in canonical:
            continue
        if source_sha in seen_sha:
            continue
        canonical[key] = document
        seen_sha.add(source_sha)

    per_document: list[list[dict[str, Any]]] = []
    for document in canonical.values():
        source_sha = str(document["sha256"])
        path = str(document["canonical_path"])
        page_rows = []
        eligible_indices = document.get("eligible_page_indices")
        page_indices = (
            [int(value) for value in eligible_indices]
            if isinstance(eligible_indices, Sequence)
            and not isinstance(eligible_indices, (str, bytes))
            else list(range(int(document["page_count"])))
        )
        for page_index in page_indices:
            if page_index < 0 or page_index >= int(document["page_count"]):
                raise ValueError(f"INVALID_ELIGIBLE_PAGE_INDEX:{path}:{page_index}")
            selection_key = _selection_key(seed, source_sha, str(page_index))
            page_rows.append(
                {
                    "document_id": f"clean-pdf-{source_sha[:16]}",
                    "canonical_path": path,
                    "source_sha256": source_sha,
                    "page_index": page_index,
                    "selection_key": selection_key,
                    "selection_seed": seed,
                }
            )
        per_document.append(sorted(page_rows, key=_page_sort_key))

    first_pages = sorted((rows[0] for rows in per_document), key=_page_sort_key)
    selected = first_pages[:page_target]
    if len(selected) < page_target:
        later_pages = sorted(
            (row for rows in per_document for row in rows[1:2]), key=_page_sort_key
        )
        selected.extend(later_pages[: page_target - len(selected)])
    if len(selected) != page_target:
        raise ValueError(f"PAGE_COHORT_INSUFFICIENT:{len(selected)}:{page_target}")
    if max(Counter(row["source_sha256"] for row in selected).values()) > 2:
        raise RuntimeError("PAGE_COHORT_DOCUMENT_CAP_EXCEEDED")
    minimum_documents = min(15, page_target)
    if len({row["source_sha256"] for row in selected}) < minimum_documents:
        raise RuntimeError("PAGE_COHORT_DOCUMENT_DIVERSITY_FAILED")
    return sorted(selected, key=_page_sort_key)


def select_benchmark_crops(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target: int = 120,
    review_cap: int = 160,
    seed: str = SELECTION_SEED,
) -> list[dict[str, Any]]:
    """Build a deterministic, length-stratified review queue.

    The queue may contain up to ``review_cap`` rows so unresolved and
    out-of-scope crops can be replaced without changing the order.
    """

    if target <= 0 or review_cap < target:
        raise ValueError("INVALID_BENCHMARK_CROP_TARGETS")
    unique: dict[str, dict[str, Any]] = {}
    seen_sha: set[str] = set()
    for raw in candidates:
        row = dict(raw)
        crop_id = str(row["crop_id"])
        crop_sha = str(row["crop_sha256"])
        if len(crop_sha) != 64:
            raise ValueError(f"INVALID_CROP_SHA:{crop_id}")
        if crop_id in unique:
            raise ValueError(f"DUPLICATE_CROP_ID:{crop_id}")
        if crop_sha in seen_sha:
            continue
        row["length_bucket"] = length_bucket(str(row.get("pp_text") or ""))
        row["selection_key"] = _selection_key(seed, crop_sha, crop_id)
        unique[crop_id] = row
        seen_sha.add(crop_sha)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        buckets[str(row["length_bucket"])].append(row)
    for rows in buckets.values():
        rows.sort(key=_crop_sort_key)
    quota = math.ceil(review_cap / len(LENGTH_BUCKETS))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for name in LENGTH_BUCKETS:
        for row in buckets.get(name, [])[:quota]:
            selected.append(row)
            selected_ids.add(str(row["crop_id"]))
    if len(selected) < review_cap:
        remainder = sorted(
            (row for row in unique.values() if str(row["crop_id"]) not in selected_ids),
            key=_crop_sort_key,
        )
        selected.extend(remainder[: review_cap - len(selected)])
    if len(selected) < target:
        raise ValueError(f"CROP_CANDIDATE_POOL_INSUFFICIENT:{len(selected)}:{target}")
    return sorted(selected[:review_cap], key=_crop_sort_key)


def validate_same_crop_contract(
    manifest: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(outputs) != set(MODEL_IDS):
        raise ValueError("FOUR_MODEL_OUTPUT_SET_REQUIRED")
    expected = _unique_rows(manifest, id_field="crop_id")
    for model in MODEL_IDS:
        actual = _unique_rows(outputs[model], id_field="sample_id")
        if set(actual) != set(expected):
            raise ValueError(f"MODEL_{model}_OUTPUT_SAMPLE_SET_MISMATCH")
        for crop_id, manifest_row in expected.items():
            if str(actual[crop_id].get("crop_sha256") or "") != str(
                manifest_row["crop_sha256"]
            ):
                raise ValueError(f"MODEL_{model}_CROP_SHA_MISMATCH:{crop_id}")
    return {
        "schema": "bemarkdown-ocr-provider-selection-v3-crop-contract-v1",
        "models": list(MODEL_IDS),
        "crop_count": len(expected),
        "gate": "SAME_CROP_SHA_FOR_ALL_MODELS_PASS",
    }


def gold_accounting(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    values = _unique_rows(rows, id_field="crop_id")
    counts = Counter(str(row.get("status") or "") for row in values.values())
    unexpected = set(counts).difference(GOLD_STATUSES)
    if unexpected:
        raise ValueError(f"INVALID_GOLD_STATUS:{sorted(unexpected)}")
    for crop_id, row in values.items():
        if row.get("status") == "NATIVE_GOLD" and not isinstance(row.get("text"), str):
            raise ValueError(f"NATIVE_GOLD_TEXT_REQUIRED:{crop_id}")
    native = counts["NATIVE_GOLD"]
    return {
        "reviewed": len(values),
        "native_gold": native,
        "unresolved": counts["UNRESOLVED"],
        "out_of_scope": counts["OUT_OF_SCOPE"],
        "accuracy_denominator": native,
    }


def score_provider(
    gold_rows: Sequence[Mapping[str, Any]],
    output_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gold = _unique_rows(gold_rows, id_field="crop_id")
    outputs = _unique_rows(output_rows, id_field="sample_id")
    if set(gold) != set(outputs):
        raise ValueError("PROVIDER_OUTPUT_GOLD_SAMPLE_SET_MISMATCH")
    truth_rows = []
    for crop_id, row in gold.items():
        native = row.get("status") == "NATIVE_GOLD"
        truth_rows.append(
            {
                "sample_id": crop_id,
                "track": "TEXT_REGION_CROP",
                "stratum": str(row.get("length_bucket") or length_bucket(row.get("text"))),
                "truth_status": "NATIVE_GOLD" if native else "UNVERIFIABLE",
                "truth_text": row.get("text") if native else None,
            }
        )
    scored = score_model(truth_rows, list(outputs.values()))
    material_errors = 0
    for crop_id, row in gold.items():
        if row.get("status") != "NATIVE_GOLD":
            continue
        truth_text = str(row.get("text") or "")
        prediction = str(outputs[crop_id].get("raw_output") or "")
        if normalize_text(truth_text) == normalize_text(prediction):
            continue
        if classify_error(truth_text, prediction) not in {
            "PUNCTUATION_ONLY",
            "WHITESPACE_ONLY",
        }:
            material_errors += 1
    denominator = int(scored["verified_denominator"])
    scored["accuracy"]["material_error_count"] = material_errors
    scored["accuracy"]["material_error_rate"] = _ratio(material_errors, denominator)
    latencies = [float(row.get("latency_seconds") or 0.0) for row in outputs.values()]
    runtime_failures = sum(bool(row.get("runtime_failure")) for row in outputs.values())
    unsupported = sum(
        int(row.get("unsupported_character_count") or 0) for row in outputs.values()
    )
    total_latency = sum(latencies)
    scored.update(
        {
            "runtime_failure_count": runtime_failures,
            "unsupported_character_count": unsupported,
            "latency_seconds": {
                "mean": statistics.fmean(latencies) if latencies else 0.0,
                "median": statistics.median(latencies) if latencies else 0.0,
                "p90": _percentile(latencies, 0.90),
                "p95": _percentile(latencies, 0.95),
            },
            "throughput_crops_per_second": (
                len(latencies) / total_latency if total_latency else 0.0
            ),
        }
    )
    return scored


def length_bucket(value: str | None) -> str:
    count = len(normalize_text(value))
    if count <= 5:
        return "1-5"
    if count <= 15:
        return "6-15"
    if count <= 30:
        return "16-30"
    return ">30"


def classify_character_types(value: str | None) -> set[str]:
    text = normalize_text(value)
    result: set[str] = set()
    has_chinese = bool(_CHINESE.search(text))
    has_digits = bool(_DIGIT.search(text))
    has_latin = bool(_LATIN.search(text))
    punctuation_count = len(_PUNCTUATION.findall(text))
    if has_chinese and not has_digits and not has_latin and punctuation_count == 0:
        result.add("Chinese-only")
    if has_chinese and has_digits:
        result.add("mixed Chinese+digits")
    if has_latin:
        result.add("mixed Latin")
    if punctuation_count >= 2 or (text and punctuation_count / len(text) >= 0.25):
        result.add("punctuation-heavy")
    if _UNIT.search(text):
        result.add("unit-heavy")
    return result or {"other"}


def pairwise_error_overlap(
    gold_rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    correctness, normalized = _correctness_and_outputs(gold_rows, outputs)
    pairs = {}
    for left_index, left in enumerate(MODEL_IDS):
        for right in MODEL_IDS[left_index + 1 :]:
            counts = Counter()
            sample_ids: dict[str, list[str]] = defaultdict(list)
            for crop_id in correctness[left]:
                left_correct = correctness[left][crop_id]
                right_correct = correctness[right][crop_id]
                if left_correct and right_correct:
                    status = "both_correct"
                elif left_correct:
                    status = "only_first_correct"
                elif right_correct:
                    status = "only_second_correct"
                elif normalized[left][crop_id] == normalized[right][crop_id]:
                    status = "both_wrong_same"
                else:
                    status = "both_wrong_different"
                counts[status] += 1
                sample_ids[status].append(crop_id)
            pairs[f"{left}-{right}"] = {
                name: counts[name]
                for name in (
                    "both_correct",
                    "only_first_correct",
                    "only_second_correct",
                    "both_wrong_same",
                    "both_wrong_different",
                )
            }
            pairs[f"{left}-{right}"]["sample_ids"] = {
                name: sorted(values) for name, values in sample_ids.items()
            }
    return {"schema": "bemarkdown-ocr-provider-pairwise-overlap-v1", "pairs": pairs}


def pp_error_recovery(
    gold_rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    correctness, _normalized = _correctness_and_outputs(gold_rows, outputs)
    p_wrong = {crop_id for crop_id, correct in correctness["P"].items() if not correct}
    providers = {}
    for model in ("G", "S", "M"):
        recovered = sorted(crop_id for crop_id in p_wrong if correctness[model][crop_id])
        providers[model] = {
            "recovery_count": len(recovered),
            "recovery_rate": _ratio(len(recovered), len(p_wrong)),
            "sample_ids": recovered,
        }
    return {
        "schema": "bemarkdown-ocr-provider-pp-error-recovery-v1",
        "P_wrong_count": len(p_wrong),
        "P_wrong_sample_ids": sorted(p_wrong),
        "providers": providers,
    }


def score_three_model_combo(
    gold_rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    combo: tuple[str, str, str],
) -> dict[str, Any]:
    if len(combo) != 3 or len(set(combo)) != 3 or not set(combo).issubset(MODEL_IDS):
        raise ValueError(f"INVALID_THREE_MODEL_COMBO:{combo}")
    _correctness, normalized = _correctness_and_outputs(gold_rows, outputs)
    gold = {
        str(row["crop_id"]): normalize_text(str(row["text"]))
        for row in gold_rows
        if row.get("status") == "NATIVE_GOLD"
    }
    left, middle, right = combo
    counts = Counter()
    sample_ids: dict[str, list[str]] = defaultdict(list)
    for crop_id, truth in gold.items():
        a, b, c = (
            normalized[model][crop_id] for model in (left, middle, right)
        )
        if a == b:
            counts["ab_consensus"] += 1
        if a == c:
            counts["ac_consensus"] += 1
        if b == c:
            counts["bc_consensus"] += 1
        if a == b == c:
            counts["unanimous"] += 1
            consensus = a
            sample_ids["unanimous"].append(crop_id)
        elif a == b:
            consensus = a
            sample_ids["ab_consensus"].append(crop_id)
        elif a == c:
            consensus = a
            sample_ids["ac_consensus"].append(crop_id)
        elif b == c:
            consensus = b
            sample_ids["bc_consensus"].append(crop_id)
        else:
            consensus = None
            counts["all_different"] += 1
            sample_ids["all_different"].append(crop_id)
        if consensus is None:
            counts["agent_required"] += 1
        elif consensus == truth:
            counts["correct_auto_consensus"] += 1
            sample_ids["correct_auto_consensus"].append(crop_id)
        else:
            counts["wrong_auto_consensus"] += 1
            sample_ids["wrong_auto_consensus"].append(crop_id)
            if a == b == c:
                counts["unanimous_wrong_count"] += 1
                sample_ids["unanimous_wrong"].append(crop_id)
    population = len(gold)
    auto = counts["correct_auto_consensus"] + counts["wrong_auto_consensus"]
    return {
        "schema": "bemarkdown-ocr-provider-three-model-combo-v1",
        "combo_id": "".join(combo),
        "models": list(combo),
        "population": population,
        "ab_consensus": counts["ab_consensus"],
        "ac_consensus": counts["ac_consensus"],
        "bc_consensus": counts["bc_consensus"],
        "unanimous": counts["unanimous"],
        "all_different": counts["all_different"],
        "auto_consensus": auto,
        "auto_consensus_coverage": _ratio(auto, population),
        "agent_required": counts["agent_required"],
        "agent_required_rate": _ratio(counts["agent_required"], population),
        "correct_auto_consensus": counts["correct_auto_consensus"],
        "wrong_auto_consensus": counts["wrong_auto_consensus"],
        "consensus_correctness": _ratio(counts["correct_auto_consensus"], auto),
        "p_correct_given_consensus": _ratio(counts["correct_auto_consensus"], auto),
        "unanimous_wrong_count": counts["unanimous_wrong_count"],
        "sample_ids": {name: sorted(values) for name, values in sample_ids.items()},
    }


def combo_ranking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    result.sort(
        key=lambda row: (
            int(row["wrong_auto_consensus"]),
            -float(row["consensus_correctness"]),
            float(row["agent_required_rate"]),
            str(row["combo_id"]),
        )
    )
    for rank, row in enumerate(result, start=1):
        row["rank"] = rank
        row["ranking_policy"] = (
            "WRONG_CONSENSUS_ASC_THEN_CORRECTNESS_DESC_THEN_AGENT_RATE_ASC"
        )
    return result


def _correctness_and_outputs(
    gold_rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, str]]]:
    if set(outputs) != set(MODEL_IDS):
        raise ValueError("FOUR_MODEL_OUTPUT_SET_REQUIRED")
    gold = {
        str(row["crop_id"]): normalize_text(str(row["text"]))
        for row in gold_rows
        if row.get("status") == "NATIVE_GOLD"
    }
    normalized: dict[str, dict[str, str]] = {}
    correctness: dict[str, dict[str, bool]] = {}
    for model in MODEL_IDS:
        model_rows = _unique_rows(outputs[model], id_field="sample_id")
        if not set(gold).issubset(model_rows):
            raise ValueError(f"MODEL_{model}_NATIVE_GOLD_OUTPUT_MISSING")
        normalized[model] = {
            crop_id: normalize_text(
                str(
                    model_rows[crop_id].get("normalized_output")
                    or model_rows[crop_id].get("raw_output")
                    or ""
                )
            )
            for crop_id in gold
        }
        correctness[model] = {
            crop_id: normalized[model][crop_id] == truth
            for crop_id, truth in gold.items()
        }
    return correctness, normalized


def _selection_key(seed: str, *parts: str) -> str:
    payload = "\x1f".join((seed, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _page_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["selection_key"]),
        str(row["source_sha256"]),
        int(row["page_index"]),
    )


def _crop_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["selection_key"]), str(row["crop_id"])


def _unique_rows(
    rows: Sequence[Mapping[str, Any]], *, id_field: str
) -> dict[str, Mapping[str, Any]]:
    result = {str(row[id_field]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"DUPLICATE_{id_field.upper()}")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
