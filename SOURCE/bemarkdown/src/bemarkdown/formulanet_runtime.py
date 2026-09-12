"""Production FormulaNet runtime and output contract.

Benchmark code consumes this module; runtime conversion never imports the
FormulaNet benchmark harness.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class OcrVerdict(str, Enum):
    VALID = "OCR_VALID"
    VALID_WITH_WARNING = "OCR_VALID_WITH_WARNING"
    SUSPICIOUS = "OCR_SUSPICIOUS"
    INVALID = "OCR_INVALID"
    EMPTY = "OCR_EMPTY"
    INFERENCE_FAILED = "OCR_INFERENCE_FAILED"


@dataclass(frozen=True)
class OcrValidation:
    verdict: OcrVerdict
    issues: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict.value, "issues": list(self.issues)}


class FormulaOcrOutputValidator:
    """Validate raw OCR LaTeX without repairing or normalizing model output."""

    _empty_fraction = re.compile(
        r"\\(?:d?frac|tfrac)\s*\{\s*\}|"
        r"\\(?:d?frac|tfrac)\s*\{[^{}]*\}\s*\{\s*\}"
    )
    _empty_root = re.compile(r"\\sqrt(?:\s*\[[^\]]*\])?\s*\{\s*\}")
    _empty_script = re.compile(r"[_^]\s*\{\s*\}")
    _unicode_script = re.compile(
        r"[_^]\s*\{(?!\s*\\(?:text|mathrm)\b)[^{}]*[\u3400-\u9fff][^{}]*\}"
    )
    _begin = re.compile(r"\\begin\s*\{([^{}]+)\}")
    _end = re.compile(r"\\end\s*\{([^{}]+)\}")
    _sized_delimiter = re.compile(
        r"\\(left|right)\s*(?:\\[A-Za-z]+|\\[{}]|\.|[()\[\]{}|])"
    )

    def validate(self, raw_latex: str | None) -> OcrValidation:
        if raw_latex is None or not raw_latex.strip():
            return OcrValidation(
                OcrVerdict.EMPTY,
                (self._issue("EMPTY_OUTPUT", "FormulaNet returned no LaTeX."),),
            )
        issues: list[dict[str, Any]] = []
        if any(ord(char) < 32 and char not in "\t\r\n" for char in raw_latex):
            issues.append(
                self._issue(
                    "CONTROL_CHARACTER",
                    "Output contains a non-whitespace control character.",
                )
            )
        delimiter_error = self._delimiter_error(raw_latex)
        if delimiter_error:
            issues.append(self._issue("UNMATCHED_DELIMITER", delimiter_error))
        begins = self._begin.findall(raw_latex)
        ends = self._end.findall(raw_latex)
        if begins != ends:
            issues.append(
                self._issue(
                    "UNMATCHED_ENVIRONMENT",
                    "LaTeX begin/end environments are not balanced and ordered.",
                )
            )
        if issues:
            return OcrValidation(OcrVerdict.INVALID, tuple(issues))

        suspicious: list[dict[str, Any]] = []
        if self._empty_fraction.search(raw_latex):
            suspicious.append(
                self._issue(
                    "EMPTY_FRACTION_OPERAND",
                    "A fraction numerator or denominator is empty.",
                )
            )
        if self._empty_root.search(raw_latex):
            suspicious.append(
                self._issue("EMPTY_RADICAND", "A square-root radicand is empty.")
            )
        if self._empty_script.search(raw_latex):
            suspicious.append(
                self._issue(
                    "EMPTY_SCRIPT_OPERAND", "A subscript or superscript is empty."
                )
            )
        if suspicious:
            return OcrValidation(OcrVerdict.SUSPICIOUS, tuple(suspicious))

        warnings: list[dict[str, Any]] = []
        if self._unicode_script.search(raw_latex):
            warnings.append(
                self._issue(
                    "UNICODE_TEXT_IN_SCRIPT",
                    "CJK text in a script is not wrapped in a text-style command.",
                    severity="warning",
                )
            )
        if warnings:
            return OcrValidation(OcrVerdict.VALID_WITH_WARNING, tuple(warnings))
        return OcrValidation(OcrVerdict.VALID, ())

    @staticmethod
    def _issue(code: str, message: str, *, severity: str = "error") -> dict[str, str]:
        return {"code": code, "message": message, "severity": severity}

    @classmethod
    def _delimiter_error(cls, value: str) -> str | None:
        sized_depth = 0
        for match in cls._sized_delimiter.finditer(value):
            sized_depth += 1 if match.group(1) == "left" else -1
            if sized_depth < 0:
                return "A LaTeX \\right delimiter has no preceding \\left."
        if sized_depth:
            return "LaTeX \\left and \\right delimiters are not balanced."
        value = cls._sized_delimiter.sub("", value)
        pairs = {"}": "{", "]": "[", ")": "("}
        stack: list[str] = []
        escaped = False
        for char in value:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char in "{[(":
                stack.append(char)
            elif char in "}])" and (not stack or stack.pop() != pairs[char]):
                return f"Unexpected closing delimiter {char!r}."
        if stack:
            return f"Unclosed delimiter {stack[-1]!r}."
        return None


class FormulaNetRuntime(Protocol):
    load_seconds: float

    def fingerprint(self) -> dict[str, Any]: ...

    def predict(self, paths: Sequence[Path], *, batch_size: int) -> list[str]: ...

    def peak_gpu_memory_bytes(self) -> int | None: ...


def benchmark_cache_key(
    png_sha256: str, runtime_fingerprint: dict[str, Any], inference_config: dict[str, Any]
) -> str:
    payload = {
        "png_content_sha256": png_sha256,
        "runtime_fingerprint": runtime_fingerprint,
        "inference_config": inference_config,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class PaddleFormulaNetRuntime:
    """Lazy PaddleX boundary; importing this module never loads Paddle."""

    def __init__(
        self,
        *,
        model_identifier: str = "PP-FormulaNet_plus-L",
        model_dir: str | Path | None = None,
        device: str = "gpu:0",
        model_source: str = "huggingface:PaddlePaddle/PP-FormulaNet_plus-L",
        model_revision: str | None = None,
        verified_model_inventory: dict[str, Any] | None = None,
    ) -> None:
        os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] = "True"
        from .paddlex_runtime import import_paddlex_for_paddle_provider

        paddle, paddlex, create_model = import_paddlex_for_paddle_provider()

        if not device.startswith("gpu"):
            raise RuntimeError("Phase 3A requires an explicit NVIDIA GPU device")
        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("Installed Paddle runtime is not compiled with CUDA")
        if paddle.device.cuda.device_count() < 1:
            raise RuntimeError("No CUDA device is available to Paddle")
        paddle.set_device(device)
        if not paddle.device.get_device().startswith("gpu"):
            raise RuntimeError("Paddle did not retain the requested GPU device")

        self._paddle = paddle
        self._paddlex = paddlex
        self.model_identifier = model_identifier
        self.model_dir = Path(model_dir).resolve() if model_dir else None
        self.device = device
        self.model_source = model_source
        self.model_revision = model_revision
        # The registry has just verified these bytes for this model load.
        self._verified_model_inventory = copy.deepcopy(verified_model_inventory)
        started = time.perf_counter()
        kwargs: dict[str, Any] = {"device": device}
        if self.model_dir is not None:
            self.runtime_model_dir = _paddle_compatible_model_dir(self.model_dir)
            kwargs["model_dir"] = str(self.runtime_model_dir)
        else:
            self.runtime_model_dir = None
        self._model = create_model(model_identifier, **kwargs)
        self.load_seconds = time.perf_counter() - started
        self._fingerprint = self._build_fingerprint()

    def predict(self, paths: Sequence[Path], *, batch_size: int) -> list[str]:
        if not paths:
            return []
        inputs: str | list[str]
        if len(paths) == 1:
            inputs = str(paths[0])
        else:
            inputs = [str(path) for path in paths]
        results = list(self._model.predict(inputs, batch_size=batch_size))
        if len(results) != len(paths):
            raise RuntimeError(
                f"FormulaNet returned {len(results)} results for {len(paths)} images"
            )
        outputs = []
        for result in results:
            raw = result.get("rec_formula")
            if not isinstance(raw, str):
                raise TypeError("FormulaNet result is missing string rec_formula")
            outputs.append(raw)
        if not self._paddle.device.get_device().startswith("gpu"):
            raise RuntimeError("Paddle device changed away from GPU during inference")
        return outputs

    def fingerprint(self) -> dict[str, Any]:
        return self._fingerprint

    def release_workspace(self) -> dict[str, int]:
        """Free unused allocator cache between stages while retaining model weights."""
        cuda = self._paddle.device.cuda
        reserved_before = int(cuda.memory_reserved())
        cuda.empty_cache()
        return {
            "reserved_before_bytes": reserved_before,
            "reserved_after_bytes": int(cuda.memory_reserved()),
            "allocated_after_bytes": int(cuda.memory_allocated()),
        }

    def close(self) -> None:
        """Release the owned predictor and its unused GPU allocator workspace."""
        if self._model is None:
            return
        self._model = None
        gc.collect()
        self.release_workspace()

    def peak_gpu_memory_bytes(self) -> int | None:
        try:
            return int(self._paddle.device.cuda.max_memory_allocated())
        except (AttributeError, RuntimeError, ValueError):
            return None

    def _build_fingerprint(self) -> dict[str, Any]:
        model_files = self._verified_model_inventory
        if model_files is None and self.model_dir:
            model_files = _directory_fingerprint(self.model_dir)
        gpu = _nvidia_smi_metadata()
        return {
            "model_identifier": self.model_identifier,
            "model_source": self.model_source,
            "model_revision": self.model_revision,
            "model_directory": str(self.model_dir) if self.model_dir else None,
            "runtime_model_directory": (
                str(self.runtime_model_dir) if self.runtime_model_dir else None
            ),
            "model_files": model_files,
            "model_sha256": (
                model_files["directory_sha256"] if model_files else "official-auto"
            ),
            "framework": {
                "paddlepaddle_gpu": self._paddle.__version__,
                "paddlex": self._paddlex.__version__,
                "paddleocr": _package_version("paddleocr"),
                "cuda_runtime": self._paddle.version.cuda(),
                "cudnn_compiled": self._paddle.version.cudnn(),
                "cudnn_package": _package_version("nvidia-cudnn-cu12"),
            },
            "device": self.device,
            "gpu": gpu,
            "precision": "fp32",
            "backend": "PaddleX create_model official/default Paddle inference",
            "hpi": False,
            "device_fallback_disabled": True,
        }


def _paddle_compatible_model_dir(model_dir: Path) -> Path:
    from .model_runtime import paddle_compatible_model_dir

    return paddle_compatible_model_dir(model_dir)


def _directory_fingerprint(directory: Path | None) -> dict[str, Any]:
    if directory is None or not directory.is_dir():
        raise FileNotFoundError(f"FormulaNet model directory is missing: {directory}")
    from .model_registry import model_inventory

    return model_inventory(directory)


def _nvidia_smi_metadata() -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        name, driver, memory = (item.strip() for item in output.split(",", 2))
        return {
            "name": name,
            "driver_version": driver,
            "memory_total_mib": int(memory),
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"name": None, "driver_version": None, "memory_total_mib": None}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
