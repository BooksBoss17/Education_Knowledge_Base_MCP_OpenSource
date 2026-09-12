"""GPU-only bounded-crop worker for the production ch_SVTRv2_rec stage."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from ...model_registry import (
    CANONICAL_JSON_INVENTORY_FINGERPRINT_CONTRACT,
    MODEL_MANIFEST_SCHEMA,
    model_inventory,
    model_inventory_fingerprint,
)
from ...paddlex_runtime import import_paddlex_for_paddle_provider
from ...text_recognition_contract import normalize_text

MODEL_NAME = "ch_SVTRv2_rec"
MODEL_ID = "ch-svtrv2-rec"
EXPECTED_MODEL_FINGERPRINT = (
    "bb4ea682c215607bf65df33f7038ffdba2cec1015428ca2d73ceb0eb45c30156"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GPU-only bounded-crop worker for official ch_SVTRv2_rec"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    result = run(
        args.manifest.resolve(),
        args.crop_root.resolve(),
        args.model_dir.resolve(),
        args.model_identity.resolve(),
        args.output.resolve(),
        args.warmup,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def recognize_rows(model, rows, *, batch_size=1, short_batch_size=1, batching_metrics=None):
    """Recognize source-bound rows in batches; retry broken batches as scalars."""
    if isinstance(short_batch_size, bool) or not isinstance(short_batch_size, int) or not 1 <= short_batch_size <= 8:
        raise ValueError('CH_SVTR_SHORT_BATCH_SIZE_INVALID')
    if short_batch_size != 1:
        if batch_size != 1:
            raise ValueError('CH_SVTR_SHORT_BATCH_SIZE_INVALID')
        from PIL import Image

        short, scalar = [], []
        unreadable = 0
        for index, pair in enumerate(rows):
            try:
                with Image.open(pair[1]) as opened:
                    width, height = opened.size
                # This SHA-verified model's resize is 3x48xmax(320, 48*w/h).
                # Group only identical 320-wide tensors, with no extra padding.
                eligible = height > 0 and width > 0 and width * 48 <= height * 320
            except (OSError, ValueError):
                eligible = False
                unreadable += 1
            (short if eligible else scalar).append((index, pair))
        if batching_metrics is not None:
            batching_metrics.update(fixed_width_320_count=len(short), scalar_width_count=len(scalar),
                                    unreadable_header_count=unreadable,
                                    requested_short_batch_size=short_batch_size)
        ordered = [None] * len(rows)
        for group, size in ((short, short_batch_size), (scalar, 1)):
            predictions = recognize_rows(model, [pair for _, pair in group], batch_size=size,
                                         batching_metrics=batching_metrics)
            for (index, _), prediction in zip(group, predictions, strict=True):
                ordered[index] = prediction
        yield from ordered
        return
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        started = time.perf_counter()
        try:
            if batching_metrics is not None:
                batching_metrics['predict_attempts'] = batching_metrics.get('predict_attempts', 0) + 1
            predictions = list(model.predict([str(path) for _, path in chunk], batch_size=batch_size))
            if len(predictions) != len(chunk):
                raise RuntimeError('CH_SVTR_BATCH_CARDINALITY')
        except Exception:  # noqa: BLE001 - third-party batch failure retries individual inputs
            if len(chunk) > 1:
                if batching_metrics is not None:
                    batching_metrics['batch_fallback_count'] = batching_metrics.get('batch_fallback_count', 0) + 1
                yield from recognize_rows(model, chunk, batch_size=1, batching_metrics=batching_metrics)
                continue
            try:
                if batching_metrics is not None:
                    batching_metrics['predict_attempts'] = batching_metrics.get('predict_attempts', 0) + 1
                predictions = list(model.predict(str(chunk[0][1]), batch_size=1))
                if len(predictions) != 1:
                    raise RuntimeError('CH_SVTR_SCALAR_CARDINALITY')
            except Exception as exc:  # noqa: BLE001 - preserve each provider failure as data
                row, path = chunk[0]
                yield row, path, '', None, f'{type(exc).__name__}:{exc}', time.perf_counter() - started
                continue
        latency = (time.perf_counter() - started) / len(chunk)
        for (row, path), prediction in zip(chunk, predictions, strict=True):
            try:
                payload = prediction.json['res']
                yield row, path, str(payload.get('rec_text') or ''), payload.get('rec_score'), None, latency
            except Exception as exc:  # noqa: BLE001 - malformed provider results remain explicit
                yield row, path, '', None, f'{type(exc).__name__}:{exc}', latency


def run(
    manifest_path: Path,
    crop_root: Path,
    model_dir: Path,
    model_identity_path: Path,
    output_path: Path,
    warmup_count: int,
    *, batch_size: int = 1, short_batch_size: int = 1, execution_mode: str = "subprocess",
) -> dict[str, Any]:
    os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] = "true"
    model_fingerprint = _validate_model_identity(model_dir, model_identity_path)
    import psutil

    paddle, paddlex, create_model = import_paddlex_for_paddle_provider()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "bemarkdown-bounded-crop-provider-manifest-v1":
        raise RuntimeError("CH_SVTRV2_REC_PRODUCTION_MANIFEST_INVALID")
    resolved: list[tuple[dict[str, Any], Path]] = []
    for row in manifest.get("rows", []):
        path = (crop_root / str(row["crop_local_relpath"])).resolve(strict=True)
        try:
            path.relative_to(crop_root)
        except ValueError as exc:
            raise RuntimeError("CH_SVTRV2_REC_CROP_OUTSIDE_ROOT") from exc
        if _sha256_file(path) != str(row["crop_sha256"]):
            raise RuntimeError(
                f"CH_SVTRV2_REC_CROP_SHA_MISMATCH:{row['sample_id']}"
            )
        resolved.append((row, path))

    load_started = time.perf_counter()
    model = create_model(MODEL_NAME, model_dir=str(model_dir), device="gpu:0")
    load_seconds = time.perf_counter() - load_started
    if paddle.device.get_device() != "gpu:0":
        raise RuntimeError("CH_SVTRV2_REC_GPU_RUNTIME_FAILED")
    warm_started = time.perf_counter()
    for _row, path in resolved[: min(warmup_count, len(resolved))]:
        list(model.predict(str(path), batch_size=1))
    warmup_seconds = time.perf_counter() - warm_started
    try:
        paddle.device.cuda.reset_max_memory_allocated()
    except (AttributeError, RuntimeError, ValueError):
        pass
    process = psutil.Process()
    peak_ram = process.memory_info().rss
    runtime_fingerprint = _semantic_sha(
        {
            "python": platform.python_version(),
            "paddle": paddle.__version__,
            "paddlex": paddlex.__version__,
            "device": paddle.device.get_device(),
            "cuda": paddle.version.cuda(),
            "model_fingerprint": model_fingerprint,
            "batch_size": batch_size,
            "short_batch_size": short_batch_size,
            "short_batch_policy": "identical-3x48x320-inputs-v1" if short_batch_size > 1 else None,
            "execution_mode": execution_mode,
        }
    )
    outputs = []
    batching_metrics = {'predict_attempts': 0, 'batch_fallback_count': 0}
    stage_started = time.perf_counter()
    for row, path, raw_text, confidence, error, latency in recognize_rows(
        model, resolved, batch_size=batch_size, short_batch_size=short_batch_size,
        batching_metrics=batching_metrics,
    ):
        peak_ram = max(peak_ram, process.memory_info().rss)
        outputs.append(
            {
                "schema": "bemarkdown-ch-svtrv2-rec-production-output-v1",
                "sample_id": str(row["sample_id"]),
                "source_crop_sha256": str(row["crop_sha256"]),
                "raw_text": raw_text,
                "normalized_output": normalize_text(raw_text),
                "confidence": confidence,
                "latency_seconds": latency,
                "runtime_status": "PASS" if error is None else "FAIL",
                "runtime_failure": error is not None,
                "runtime_error": error,
                "output_contract_status": (
                    "PROVIDER_CRASH"
                    if error
                    else "PASS"
                    if normalize_text(raw_text)
                    else "EMPTY_OUTPUT"
                ),
                "warnings": [error] if error else [],
                "model_id": MODEL_NAME,
                "model_fingerprint": model_fingerprint,
                "runtime_fingerprint": runtime_fingerprint,
                "device": "gpu:0",
            }
        )
    stage_seconds = time.perf_counter() - stage_started
    latencies = [float(row["latency_seconds"]) for row in outputs]
    try:
        peak_vram = int(paddle.device.cuda.max_memory_allocated())
    except (AttributeError, RuntimeError, ValueError):
        peak_vram = None
    runtime = {
        "schema": "bemarkdown-ch-svtrv2-rec-production-runtime-v1",
        "provider_id": "B_CH_SVTRV2_REC",
        "model_id": MODEL_NAME,
        "model_fingerprint": model_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "population": len(outputs),
        "calls": len(outputs),
        "total_wall_seconds": stage_seconds,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "model_load_seconds": load_seconds,
        "warmup_count": min(warmup_count, len(resolved)),
        "warmup_seconds": warmup_seconds,
        "peak_vram_bytes": peak_vram,
        "peak_ram_bytes": peak_ram,
        "gpu_only": True,
        "cpu_fallback_count": 0,
        "runtime_failure_count": sum(row["runtime_failure"] for row in outputs),
        "empty_output_count": sum(
            row["output_contract_status"] == "EMPTY_OUTPUT" for row in outputs
        ),
        "load_count": 1,
        "unload_count": 1,
        "process_exit_unload": execution_mode == "subprocess",
        "execution_mode": execution_mode,
        "batch_size": batch_size,
        "short_batch_size": short_batch_size,
        "batching": batching_metrics,
        "gate": "CH_SVTRV2_REC_PRODUCTION_GPU_PASS",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in outputs
        ),
        encoding="utf-8",
        newline="\n",
    )
    output_path.with_suffix(".runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    del model
    gc.collect()
    paddle.device.cuda.empty_cache()
    return runtime


def _validate_model_identity(model_dir: Path, model_identity_path: Path) -> str:
    identity = json.loads(model_identity_path.read_text(encoding="utf-8"))
    inventory = model_inventory(model_dir)
    model_fingerprint = model_inventory_fingerprint(
        inventory, CANONICAL_JSON_INVENTORY_FINGERPRINT_CONTRACT
    )
    formal_identity = (
        identity.get("schema") == MODEL_MANIFEST_SCHEMA
        and identity.get("model_id") == MODEL_ID
        and identity.get("fingerprint_contract")
        == CANONICAL_JSON_INVENTORY_FINGERPRINT_CONTRACT
    )
    legacy_identity = identity.get("gate") == "CH_SVTRV2_REC_IDENTITY_PASS"
    if (
        not (formal_identity or legacy_identity)
        or identity.get("model_fingerprint") != EXPECTED_MODEL_FINGERPRINT
        or model_fingerprint != EXPECTED_MODEL_FINGERPRINT
    ):
        raise RuntimeError("CH_SVTRV2_REC_PRODUCTION_IDENTITY_INVALID")
    return model_fingerprint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
