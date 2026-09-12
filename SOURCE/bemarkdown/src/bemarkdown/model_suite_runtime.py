from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import socket
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

from .model_registry import PADDLE_MODEL_CATALOG, ModelRegistry
from .model_runtime import paddle_compatible_model_dir
from .model_suite import validate_model_suite
from .release import write_json


def run_model_suite_smoke(
    models_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    models_root = Path(models_root).resolve()
    validate_model_suite(models_root, manifest_path=manifest_path, deep=True)
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="bemarkdown-suite-smoke-") as temporary:
            return _run_smoke(models_root, Path(temporary), progress=progress)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return _run_smoke(models_root, output, progress=progress)


def _run_smoke(
    models_root: Path,
    output_dir: Path,
    *,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .paddlex_runtime import import_paddlex_for_paddle_provider

    paddle, paddlex, create_model = import_paddlex_for_paddle_provider()

    if not paddle.device.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PADDLE_MODEL_SUITE_REQUIRES_GPU")
    paddle.set_device("gpu:0")
    fixtures = _write_smoke_fixtures(output_dir / "fixtures")
    registry = ModelRegistry(models_root=models_root)
    network_attempts: list[str] = []
    records: list[dict[str, Any]] = []

    def blocked(*args, **kwargs):
        network_attempts.append("network")
        raise AssertionError("Model suite smoke forbids network access")

    for model_id, spec in PADDLE_MODEL_CATALOG.items():
        resolution = registry.resolve(model_id, deep=True)
        execution_root = paddle_compatible_model_dir(resolution.model_root)
        fixture = fixtures[_fixture_kind(model_id)]
        load_started = time.perf_counter()
        model = None
        load_status = "LOAD_FAIL"
        inference_status = "INFERENCE_FAIL"
        try:
            _reset_peak_memory(paddle)
            with patch.object(socket, "create_connection", blocked), patch.object(
                socket.socket, "connect", blocked
            ), patch.object(urllib.request, "urlopen", blocked):
                model = create_model(
                    spec.directory,
                    model_dir=str(execution_root),
                    device="gpu:0",
                )
                load_seconds = time.perf_counter() - load_started
                load_status = "LOAD_PASS"
                inference_started = time.perf_counter()
                results = list(model.predict(str(fixture), batch_size=1))
                inference_seconds = time.perf_counter() - inference_started
            if len(results) != 1:
                raise RuntimeError(
                    f"Model {model_id} returned {len(results)} results for one fixture"
                )
            result_summary = summarize_model_result(model_id, results[0])
            inference_status = "INFERENCE_PASS"
            if not paddle.device.get_device().startswith("gpu"):
                raise RuntimeError("Paddle device changed away from GPU")
            record = {
                "model_id": model_id,
                "display_name": spec.directory,
                "engine": spec.engine,
                "integration_status": spec.integration_status,
                "logical_model_root": str(resolution.model_root),
                "execution_model_root": str(execution_root),
                "ascii_alias_used": execution_root != resolution.model_root,
                "fixture": fixture.name,
                "load_status": load_status,
                "inference_status": inference_status,
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "peak_gpu_allocation_bytes": _peak_memory(paddle),
                "result_schema": result_summary,
                "unload_status": "PENDING",
            }
        except Exception as exc:  # noqa: BLE001 - record each isolated model boundary
            record = {
                "model_id": model_id,
                "display_name": spec.directory,
                "engine": spec.engine,
                "integration_status": spec.integration_status,
                "logical_model_root": str(resolution.model_root),
                "execution_model_root": str(execution_root),
                "ascii_alias_used": execution_root != resolution.model_root,
                "fixture": fixture.name,
                "load_status": load_status,
                "inference_status": inference_status,
                "error": f"{type(exc).__name__}: {exc}",
                "unload_status": "PENDING",
            }
        finally:
            if model is not None:
                del model
            gc.collect()
            try:
                paddle.device.cuda.empty_cache()
                record["unload_status"] = "UNLOAD_PASS"
                record["gpu_allocation_after_unload_bytes"] = int(
                    paddle.device.cuda.memory_allocated()
                )
            except (AttributeError, RuntimeError, ValueError) as exc:
                record["unload_status"] = "UNLOAD_RECORDED_WITHOUT_COUNTER"
                record["unload_note"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if progress is not None:
            progress(
                f"Model suite {len(records):02d}/09 {model_id}: "
                f"{record['load_status']} {record['inference_status']}"
            )

    all_models_pass = all(
        row["load_status"] == "LOAD_PASS"
        and row["inference_status"] == "INFERENCE_PASS"
        and row["unload_status"].startswith("UNLOAD_")
        for row in records
    )
    summary = {
        "schema": "bemarkdown-model-suite-smoke-v1",
        "runtime": {
            "python": os.sys.version.split()[0],
            "paddlepaddle_gpu": paddle.__version__,
            "paddlex": paddlex.__version__,
            "cuda_runtime": paddle.version.cuda(),
            "cudnn_compiled": paddle.version.cudnn(),
            "cudnn_package": _package_version("nvidia-cudnn-cu12"),
            "device": paddle.device.get_device(),
            "precision": "fp32",
            "lifecycle": "staged_one_model_at_a_time",
        },
        "models": records,
        "model_count": len(records),
        "load_pass": sum(row["load_status"] == "LOAD_PASS" for row in records),
        "inference_pass": sum(
            row["inference_status"] == "INFERENCE_PASS" for row in records
        ),
        "network_attempts": len(network_attempts),
        "download_attempts": 0,
        "accuracy_validated": False,
        "pdf_pipeline_validated": False,
        "all_ok": all_models_pass and len(records) == 9 and not network_attempts,
    }
    _write_jsonl(output_dir / "model_smoke_results.jsonl", records)
    write_json(
        output_dir / "model_runtime_matrix.json",
        {
            "schema": "bemarkdown-model-runtime-matrix-v1",
            "runtime": summary["runtime"],
            "models": [
                {
                    "model_id": row["model_id"],
                    "paddle_3_2_2": row["load_status"],
                    "paddlex_3_7_2": row["load_status"],
                    "gpu": row["inference_status"],
                    "fp32": row["inference_status"],
                }
                for row in records
            ],
        },
    )
    write_json(
        output_dir / "no_network_result.json",
        {
            "schema": "bemarkdown-model-suite-no-network-v1",
            "models_loaded": len(records),
            "network_attempts": len(network_attempts),
            "download_attempts": 0,
            "passed": len(records) == 9 and not network_attempts,
        },
    )
    return summary


def summarize_model_result(model_id: str, result: Any) -> dict[str, Any]:
    keys = sorted(str(key) for key in result) if hasattr(result, "keys") else []
    summary: dict[str, Any] = {"result_keys": keys}
    if model_id == "pp-ocrv6-medium-rec":
        text = result.get("rec_text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Text recognition smoke requires a non-empty result")
        summary.update({"rec_text": text, "rec_score": float(result.get("rec_score", 0))})
    elif model_id == "pp-formulanet-plus-l":
        formula = result.get("rec_formula")
        if not isinstance(formula, str) or not formula.strip():
            raise RuntimeError("Formula smoke requires a non-empty result")
        summary.update({"formula_characters": len(formula)})
    elif "boxes" in keys:
        boxes = result.get("boxes")
        summary["boxes"] = len(boxes) if boxes is not None else 0
    elif "dt_polys" in keys:
        polygons = result.get("dt_polys")
        summary["detected_polygons"] = len(polygons) if polygons is not None else 0
    elif "structure" in keys:
        structure = result.get("structure")
        summary["structure_tokens"] = len(structure) if structure is not None else 0
    elif "label_names" in keys:
        summary["labels"] = [str(value) for value in result.get("label_names") or []]
    return summary


def _fixture_kind(model_id: str) -> str:
    if model_id == "pp-ocrv6-medium-rec":
        return "text"
    if model_id == "pp-formulanet-plus-l":
        return "formula"
    if "wireless" in model_id:
        return "wireless"
    return "wired"


def _write_smoke_fixtures(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    wired = root / "wired-table.png"
    wireless = root / "wireless-table.png"
    text = root / "text-crop.png"
    formula = root / "formula.png"
    font = _smoke_font(30)
    small_font = _smoke_font(22)

    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "BeMarkdown OCR 123 ABC", fill="black", font=font)
    draw.rectangle((30, 90, 610, 390), outline="black", width=3)
    for y in (150, 210, 270, 330):
        draw.line((30, y, 610, y), fill="black", width=2)
    for x in (220, 410):
        draw.line((x, 90, x, 390), fill="black", width=2)
    image.save(wired)

    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    for index in range(5):
        draw.text(
            (50, 70 + index * 60),
            f"Row {index + 1}    Value {index + 10}    Text",
            fill="black",
            font=small_font,
        )
    image.save(wireless)

    image = Image.new("RGB", (720, 100), "white")
    ImageDraw.Draw(image).text(
        (20, 25), "BeMarkdown OCR 2026 ABC xyz", fill="black", font=font
    )
    image.save(text)

    image = Image.new("RGB", (480, 100), "white")
    ImageDraw.Draw(image).text(
        (20, 30), "x = (-b + sqrt(b^2-4ac)) / 2a", fill="black", font=small_font
    )
    image.save(formula)
    return {"wired": wired, "wireless": wireless, "text": text, "formula": formula}


def _smoke_font(size: int):
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _reset_peak_memory(paddle) -> None:
    try:
        paddle.device.cuda.reset_max_memory_allocated()
    except (AttributeError, RuntimeError, ValueError):
        pass


def _peak_memory(paddle) -> int | None:
    try:
        return int(paddle.device.cuda.max_memory_allocated())
    except (AttributeError, RuntimeError, ValueError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
