from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bemarkdown.ocr_ensemble_benchmark import classify_error, normalize_text

EXPECTED_POPULATION = 103


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_benchmark(
    manifest_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(manifest_rows) != EXPECTED_POPULATION:
        raise ValueError(f"FROZEN_BENCHMARK_POPULATION_MISMATCH:{len(manifest_rows)}")
    manifest = _unique(manifest_rows, "crop_id")
    native = {
        str(row["crop_id"]): dict(row)
        for row in gold_rows
        if row.get("status") == "NATIVE_GOLD"
    }
    if len(native) != EXPECTED_POPULATION or set(native) != set(manifest):
        raise ValueError("FROZEN_BENCHMARK_GOLD_IDENTITY_MISMATCH")
    for crop_id, row in manifest.items():
        path = Path(str(row["crop_ref"])).resolve()
        if not path.is_file() or sha256_file(path) != str(row["crop_sha256"]):
            raise ValueError(f"FROZEN_CROP_SHA_MISMATCH:{crop_id}")
    identities = [
        {"crop_id": crop_id, "crop_sha256": str(manifest[crop_id]["crop_sha256"])}
        for crop_id in sorted(manifest)
    ]
    return {
        "schema": "bemarkdown-ch-svtrv2-frozen-benchmark-validation-v1",
        "population": len(identities),
        "unique_crop_ids": len(identities),
        "crop_identity_fingerprint": semantic_sha(identities),
        "gold_fingerprint": semantic_sha([native[key] for key in sorted(native)]),
        "gate": "FROZEN_103_CROP_LOCK_PASS",
    }


def validate_output_accounting(
    manifest_rows: Sequence[Mapping[str, Any]],
    output_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = _unique(manifest_rows, "crop_id")
    outputs = _unique(output_rows, "sample_id")
    if set(outputs) != set(manifest):
        raise ValueError("CH_SVTRV2_REC_OUTPUT_SAMPLE_SET_MISMATCH")
    for crop_id, expected in manifest.items():
        if str(outputs[crop_id].get("crop_sha256") or "") != str(
            expected["crop_sha256"]
        ):
            raise ValueError(f"CH_SVTRV2_REC_CROP_SHA_MISMATCH:{crop_id}")
    return {
        "population": len(outputs),
        "same_crop_sha": True,
        "gate": "CH_SVTRV2_REC_OUTPUT_ACCOUNTING_PASS",
    }


def validate_output_provenance(
    output_rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    *,
    model_fingerprint: str,
    model_identity_sha256: str,
    frozen_lock_sha256: str,
) -> dict[str, Any]:
    runtime_fingerprint = runtime.get("runtime_fingerprint")
    if (
        runtime.get("model_fingerprint") != model_fingerprint
        or runtime.get("model_identity_validation_sha256") != model_identity_sha256
        or runtime.get("frozen_benchmark_lock_sha256") != frozen_lock_sha256
        or runtime.get("device") != "gpu:0"
        or runtime.get("gpu_only") is not True
        or runtime.get("cpu_fallback_count") != 0
        or runtime.get("runtime_failure_count") != 0
        or not runtime_fingerprint
    ):
        raise ValueError("CH_SVTRV2_REC_RUNTIME_PROVENANCE_MISMATCH")
    for row in output_rows:
        raw_output = row.get("raw_output")
        normalized_output = row.get("normalized_output")
        if (
            not isinstance(raw_output, str)
            or row.get("raw_text") != raw_output
            or row.get("normalized_text") != normalized_output
            or normalized_output != normalize_text(raw_output)
        ):
            raise ValueError(
                "CH_SVTRV2_REC_OUTPUT_NORMALIZATION_MISMATCH:"
                f"{row.get('sample_id')}"
            )
        if (
            row.get("model_id") != "ch_SVTRv2_rec"
            or row.get("model_fingerprint") != model_fingerprint
            or row.get("model_identity_validation_sha256") != model_identity_sha256
            or row.get("frozen_benchmark_lock_sha256") != frozen_lock_sha256
            or row.get("runtime_fingerprint") != runtime_fingerprint
            or row.get("device") != "gpu:0"
        ):
            raise ValueError(
                f"CH_SVTRV2_REC_OUTPUT_PROVENANCE_MISMATCH:{row.get('sample_id')}"
            )
    return {
        "population": len(output_rows),
        "model_identity_bound": True,
        "frozen_benchmark_bound": True,
        "gpu_only": True,
        "gate": "CH_SVTRV2_REC_OUTPUT_PROVENANCE_PASS",
    }


def pairwise_analysis(
    gold_rows: Sequence[Mapping[str, Any]],
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
) -> dict[str, Any]:
    truth = _truth(gold_rows)
    left_map = _outputs(left_rows)
    right_map = _outputs(right_rows)
    if set(left_map) != set(truth) or set(right_map) != set(truth):
        raise ValueError("PAIRWISE_SAMPLE_SET_MISMATCH")
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    for crop_id, expected in truth.items():
        left_text = left_map[crop_id]
        right_text = right_map[crop_id]
        if left_text == expected and right_text == expected:
            status = "both_correct"
        elif left_text == expected:
            status = "only_first_correct"
        elif right_text == expected:
            status = "only_second_correct"
        elif left_text == right_text:
            status = "both_wrong_same"
        else:
            status = "both_wrong_different"
        counts[status] += 1
        samples[status].append(crop_id)
    names = (
        "both_correct",
        "only_first_correct",
        "only_second_correct",
        "both_wrong_same",
        "both_wrong_different",
    )
    return {
        "schema": "bemarkdown-ch-svtrv2-pairwise-v1",
        "left": left,
        "right": right,
        "population": len(truth),
        "counts": {name: counts[name] for name in names},
        "sample_ids": {name: sorted(samples[name]) for name in names},
        "gate": "PAIRWISE_ANALYSIS_PASS",
    }


def pp_error_recovery(
    gold_rows: Sequence[Mapping[str, Any]],
    pp_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    truth = _truth(gold_rows)
    pp = _outputs(pp_rows)
    candidate = _outputs(candidate_rows)
    wrong = sorted(crop_id for crop_id, value in truth.items() if pp[crop_id] != value)
    if len(wrong) != 13:
        raise ValueError(f"PP_ERROR_COUNT_MISMATCH:{len(wrong)}")
    recovered = sorted(crop_id for crop_id in wrong if candidate[crop_id] == truth[crop_id])
    material = sorted(
        crop_id
        for crop_id in wrong
        if classify_error(
            next(str(row["text"]) for row in gold_rows if str(row["crop_id"]) == crop_id),
            next(
                str(row.get("raw_output") or "")
                for row in pp_rows
                if str(row["sample_id"]) == crop_id
            ),
        )
        not in {"PUNCTUATION_ONLY", "WHITESPACE_ONLY"}
    )
    if len(material) != 8:
        raise ValueError(f"PP_MATERIAL_ERROR_COUNT_MISMATCH:{len(material)}")
    material_recovered = sorted(set(material) & set(recovered))
    return {
        "schema": "bemarkdown-ch-svtrv2-pp-error-recovery-v1",
        "pp_error_count": len(wrong),
        "pp_error_ids": wrong,
        "ch_svtrv2_rec_recovered_count": len(recovered),
        "ch_svtrv2_rec_recovery_rate": len(recovered) / len(wrong),
        "recovered_crop_ids": recovered,
        "material_pp_error_count": len(material),
        "material_pp_error_recovery_count": len(material_recovered),
        "material_pp_error_recovery_ids": material_recovered,
        "gate": "PP_ERROR_RECOVERY_ANALYSIS_PASS",
    }


def replay_three_model_combo(
    gold_rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    combo: tuple[str, str, str] = ("P", "G", "C"),
) -> dict[str, Any]:
    if set(outputs) != set(combo) or len(set(combo)) != 3:
        raise ValueError("PGC_OUTPUT_SET_REQUIRED")
    truth = _truth(gold_rows)
    normalized = {model: _outputs(rows) for model, rows in outputs.items()}
    if any(set(rows) != set(truth) for rows in normalized.values()):
        raise ValueError("PGC_SAMPLE_SET_MISMATCH")
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    left, middle, right = combo
    for crop_id, expected in truth.items():
        a, b, c = (normalized[model][crop_id] for model in combo)
        if a == b:
            counts["ab_consensus"] += 1
        if a == c:
            counts["ac_consensus"] += 1
        if b == c:
            counts["bc_consensus"] += 1
        if a == b == c:
            counts["unanimous"] += 1
            samples["unanimous"].append(crop_id)
            consensus = a
        elif a == b:
            samples["ab_consensus"].append(crop_id)
            consensus = a
        elif a == c:
            samples["ac_consensus"].append(crop_id)
            consensus = a
        elif b == c:
            samples["bc_consensus"].append(crop_id)
            consensus = b
        else:
            counts["all_different"] += 1
            samples["all_different"].append(crop_id)
            consensus = None
        if consensus is None:
            counts["agent_required"] += 1
        elif consensus == expected:
            counts["correct_auto_consensus"] += 1
            samples["correct_auto_consensus"].append(crop_id)
        else:
            counts["wrong_auto_consensus"] += 1
            samples["wrong_auto_consensus"].append(crop_id)
            if a == b == c:
                counts["unanimous_wrong_count"] += 1
                samples["unanimous_wrong"].append(crop_id)
    population = len(truth)
    auto = counts["correct_auto_consensus"] + counts["wrong_auto_consensus"]
    return {
        "schema": "bemarkdown-ocr-provider-three-model-combo-v1",
        "combo_id": "".join(combo),
        "models": [left, middle, right],
        "population": population,
        "ab_consensus": counts["ab_consensus"],
        "ac_consensus": counts["ac_consensus"],
        "bc_consensus": counts["bc_consensus"],
        "unanimous": counts["unanimous"],
        "all_different": counts["all_different"],
        "auto_consensus": auto,
        "auto_consensus_coverage": auto / population,
        "agent_required": counts["agent_required"],
        "agent_required_rate": counts["agent_required"] / population,
        "correct_auto_consensus": counts["correct_auto_consensus"],
        "wrong_auto_consensus": counts["wrong_auto_consensus"],
        "consensus_correctness": counts["correct_auto_consensus"] / auto,
        "unanimous_wrong_count": counts["unanimous_wrong_count"],
        "sample_ids": {name: sorted(values) for name, values in samples.items()},
        "gate": "PGC_COMBO_REPLAY_PASS",
    }


def compare_pgs_pgc(pgs: Mapping[str, Any], pgc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "bemarkdown-pgs-vs-pgc-v1",
        "old_pgs": dict(pgs),
        "new_pgc": dict(pgc),
        "delta": {
            "auto_consensus_coverage": float(pgc["auto_consensus_coverage"])
            - float(pgs["auto_consensus_coverage"]),
            "agent_required_rate": float(pgc["agent_required_rate"])
            - float(pgs["agent_required_rate"]),
            "wrong_auto_consensus": int(pgc["wrong_auto_consensus"])
            - int(pgs["wrong_auto_consensus"]),
            "consensus_correctness": float(pgc["consensus_correctness"])
            - float(pgs["consensus_correctness"]),
        },
        "gate": "PGS_VS_PGC_COMPARISON_PASS",
    }


def _truth(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row["crop_id"]): normalize_text(str(row["text"]))
        for row in rows
        if row.get("status") == "NATIVE_GOLD"
    }


def _outputs(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    values = _unique(rows, "sample_id")
    return {
        crop_id: normalize_text(
            str(row.get("normalized_output") or row.get("raw_output") or "")
        )
        for crop_id, row in values.items()
    }


def _unique(
    rows: Sequence[Mapping[str, Any]], id_field: str
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row[id_field])
        if key in values:
            raise ValueError(f"DUPLICATE_{id_field.upper()}:{key}")
        values[key] = dict(row)
    return values
