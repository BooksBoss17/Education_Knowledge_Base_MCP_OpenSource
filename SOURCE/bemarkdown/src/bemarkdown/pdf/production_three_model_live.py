"""Production composition for staged three-model ordinary-text OCR.

Model inference happens before the per-region content executor.  The runtime
returned here only serves current-run, SHA-bound resolutions for the frozen
PP -> ch_SVTRv2_rec -> conditional GOT provider order.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent_task_contract import AgentTaskV2
from ..model_registry import ModelRegistry
from ..pdf_content_router import PdfContentAdapterExecutor
from ..text_recognition_contract import normalize_text
from .got_batching import GOT_BATCH_RETRY_POLICY, GOT_CROP_ORDERING, order_got_crops
from .page_visual_text_recovery import (
    PageBoundedTextPlan,
    PageVisualTextRecoveryBoundedCropper,
    assemble_page_bounded_text_recovery,
)
from .text_evidence import (
    TextEvidenceProvider,
    TextRecognitionEvidence,
    TextRecognitionRequest,
)
from .three_model_ocr import (
    RESOLUTION_STATUSES_V2,
    THREE_MODEL_OCR_SCHEMA,
    ConsensusComparatorV1,
    EvidenceResolverV1,
    ThirdModelRouterV1,
    ThreeModelOCRReplayHarness,
    build_text_recognition_request,
)
from .workers import CH_SVTRV2_WORKER_MODULE, GOT_OCR2_WORKER_MODULE

OCR_ADAPTERS = {"OCR_TEXT_REGION", "PAGE_VISUAL_TEXT_RECOVERY"}
# GOT runs beside the live Paddle formula model. This is the Torch allocator
# allowance, not the entire pipeline or physical-card limit.
GOT_DEFAULT_VRAM_LIMIT_MIB = 2816


@dataclass(frozen=True, slots=True)
class ProductionThreeModelStagedResult:
    runtime: Any
    metrics: dict[str, Any]
    agent_candidates: list[dict[str, Any]]
    provenance_by_route: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _ProductionThreeModelExecutionWorkload:
    source_routes: tuple[dict[str, Any], ...]
    execution_routes: tuple[dict[str, Any], ...]
    crops_by_request: dict[str, dict[str, Any]]
    page_plans_by_route: dict[str, PageBoundedTextPlan]
    request_source_route_ids: dict[str, str]
    region_line_coverage: dict[str, dict[str, Any]]


class CurrentRunPaddleTextEvidenceProvider:
    """Expose the just-completed PP stage as provider-A evidence."""

    provider_id = "A_PP_OCRV6"

    def __init__(
        self,
        outputs_by_route: Mapping[str, Sequence[Mapping[str, Any]]],
        summary: Mapping[str, Any],
    ) -> None:
        self.outputs_by_route = {
            str(key): [dict(row) for row in value]
            for key, value in outputs_by_route.items()
        }
        self.summary = copy.deepcopy(dict(summary))

    def recognize(self, request: TextRecognitionRequest) -> TextRecognitionEvidence:
        return self.recognize_batch([request])[0]

    def page_recovery_lines(self, route_id: str) -> list[dict[str, Any]]:
        """Return current-run PP lines for canonical bounded-crop voting."""

        return copy.deepcopy(self.outputs_by_route.get(route_id, []))

    def recognize_batch(
        self, requests: Sequence[TextRecognitionRequest]
    ) -> list[TextRecognitionEvidence]:
        det = self.summary["det"]["model"]
        rec = self.summary["rec"]["model"]
        runtime_fingerprint = _semantic_sha256(
            {"det": det, "rec": rec, "device": "gpu:0", "precision": "fp32",
             'execution':self.summary.get('execution', {})}
        )
        values = []
        for request in requests:
            unit = request.source_provenance.get("page_recovery_bounded_unit")
            lines = (
                []
                if isinstance(unit, Mapping)
                else self.outputs_by_route.get(request.region_id, [])
            )
            recoveries = [
                copy.deepcopy(dict(line["ocr_input_recovery"]))
                for line in lines
                if line.get("ocr_input_recovery")
            ]
            provenance: dict[str, Any] = {"current_run_pp_stage": True}
            if recoveries:
                provenance["tiny_text_input_recovery"] = recoveries[0]
            if isinstance(unit, Mapping):
                text = str(unit.get("provider_a_text") or "")
                provenance["page_recovery_bounded_unit"] = {
                    "unit_id": str(unit.get("unit_id") or ""),
                    "source_line_id": str(unit.get("source_line_id") or ""),
                    "reading_order_index": int(unit.get("reading_order_index") or 0),
                }
            else:
                text = "\n".join(
                    str(line.get("text") or "") for line in lines if line.get("text")
                )
            confidences = (
                [float(unit["provider_a_confidence"])]
                if isinstance(unit, Mapping)
                and unit.get("provider_a_confidence") is not None
                else [
                    float(line["confidence"])
                    for line in lines
                    if line.get("confidence") is not None
                ]
            )
            values.append(
                TextRecognitionEvidence(
                    provider_id=self.provider_id,
                    model_id="PP-OCRv6",
                    model_fingerprint=(
                        f"{det['model_fingerprint']}+{rec['model_fingerprint']}"
                    ),
                    runtime_fingerprint=runtime_fingerprint,
                    bbox=request.bbox,
                    text=text,
                    normalized_text=normalize_text(text),
                    confidence=(
                        sum(confidences) / len(confidences) if confidences else None
                    ),
                    source_crop_ref=request.crop_ref,
                    source_crop_sha256=request.crop_sha256,
                    latency_seconds=0.0,
                    output_contract_status="PASS" if text else "EMPTY_OUTPUT",
                    provenance=provenance,
                )
            )
        return values


class CurrentRunGenerativeStageProvider:
    """Provider B/C backed by one current-run batch stage invocation."""

    batch_only = True

    def __init__(
        self,
        provider_id: str,
        runner: Callable[
            [str, Sequence[TextRecognitionRequest]], Sequence[Mapping[str, Any]]
        ],
    ) -> None:
        self.provider_id = provider_id
        self.runner = runner

    def recognize(self, request: TextRecognitionRequest) -> TextRecognitionEvidence:
        del request
        raise RuntimeError("GENERATIVE_STAGE_BATCH_REQUIRED")

    def recognize_batch(
        self, requests: Sequence[TextRecognitionRequest]
    ) -> list[TextRecognitionEvidence]:
        outputs = list(self.runner(self.provider_id, requests))
        if len(outputs) != len(requests):
            raise RuntimeError("TEXT_PROVIDER_BATCH_CARDINALITY_MISMATCH")
        return [
            _generative_evidence(self.provider_id, request, output)
            for request, output in zip(requests, outputs, strict=True)
        ]


class CurrentRunRecognitionStageProvider(CurrentRunGenerativeStageProvider):
    """Provider-neutral staged recognizer used by ch_SVTRv2_rec and GOT."""


class SubprocessChSVTRv2StageRunner:
    """Run official ch_SVTRv2_rec once over a SHA-bound bounded-crop stage."""

    provider_id = "B_CH_SVTRV2_REC"

    def _launch_worker(self, command, **kwargs):
        return subprocess.run(command, **kwargs)

    @classmethod
    def from_registry(
        cls,
        *,
        python: str | Path,
        crop_root: str | Path,
        output_root: str | Path,
        models_root: str | Path | None = None,
        config_path: str | Path | None = None,
        mcp_root: str | Path | None = None,
    ) -> SubprocessChSVTRv2StageRunner:
        """Resolve the formal B model exclusively through the model registry."""

        resolution = ModelRegistry(
            models_root=models_root,
            config_path=config_path,
            mcp_root=mcp_root,
        ).resolve("ch-svtrv2-rec", deep=True)
        return cls(
            python=python,
            crop_root=crop_root,
            output_root=output_root,
            model_root=resolution.model_root,
            model_identity=resolution.model_root / "MODEL_MANIFEST.json",
        )

    def __init__(
        self,
        *,
        python: str | Path,
        crop_root: str | Path,
        output_root: str | Path,
        model_root: str | Path,
        model_identity: str | Path,
    ) -> None:
        self.python = Path(python).resolve(strict=True)
        self.crop_root = Path(crop_root).resolve(strict=True)
        self.output_root = Path(output_root).resolve()
        self.model_root = Path(model_root).resolve(strict=True)
        self.model_identity = Path(model_identity).resolve(strict=True)
        self.stage_metrics: dict[str, dict[str, Any]] = {}

    def __call__(
        self, provider_id: str, requests: Sequence[TextRecognitionRequest]
    ) -> list[dict[str, Any]]:
        if provider_id != self.provider_id:
            raise ValueError(f"UNSUPPORTED_CH_SVTRV2_PROVIDER:{provider_id}")
        stage_root = self.output_root / "ch_svtrv2_rec"
        stage_root.mkdir(parents=True, exist_ok=False)
        rows = []
        for request in requests:
            crop_path = Path(request.crop_ref).resolve(strict=True)
            try:
                relative = crop_path.relative_to(self.crop_root).as_posix()
            except ValueError as exc:
                raise RuntimeError(
                    f"CH_SVTRV2_LIVE_CROP_OUTSIDE_ROOT:{crop_path}"
                ) from exc
            if _sha256_file(crop_path) != request.crop_sha256:
                raise RuntimeError(
                    f"CH_SVTRV2_LIVE_SOURCE_CROP_MISMATCH:{request.region_id}"
                )
            rows.append(
                {
                    "sample_id": request.region_id,
                    "crop_local_relpath": relative,
                    "crop_sha256": request.crop_sha256,
                }
            )
        manifest = stage_root / "manifest.json"
        _write_json(
            manifest,
            {
                "schema": "bemarkdown-bounded-crop-provider-manifest-v1",
                "rows": rows,
            },
        )
        output = stage_root / "outputs.jsonl"
        env = os.environ.copy()
        env.update(
            {
                "PADDLE_PDX_DISABLE_DEVICE_FALLBACK": "true",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        completed = self._launch_worker(
            [
                str(self.python),
                "-m",
                CH_SVTRV2_WORKER_MODULE,
                "--manifest",
                str(manifest),
                "--crop-root",
                str(self.crop_root),
                "--model-dir",
                str(self.model_root),
                "--model-identity",
                str(self.model_identity),
                "--output",
                str(output),
            ],
            cwd=self.output_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=14400,
        )
        (stage_root / "stdout.txt").write_text(
            completed.stdout, encoding="utf-8", newline="\n"
        )
        (stage_root / "stderr.txt").write_text(
            completed.stderr, encoding="utf-8", newline="\n"
        )
        runtime_path = output.with_suffix(".runtime.json")
        metrics = (
            json.loads(runtime_path.read_text(encoding="utf-8"))
            if runtime_path.is_file()
            else {
                "schema": "bemarkdown-ch-svtrv2-rec-production-runtime-v1",
                "population": len(requests),
                "runtime_failure_count": len(requests),
                "cpu_fallback_count": 0,
                "error": f"worker-exit:{completed.returncode}",
            }
        )
        metrics.update(
            {
                "process_returncode": completed.returncode,
                "request_region_ids": [request.region_id for request in requests],
            }
        )
        self.stage_metrics[provider_id] = metrics
        output_by_id = (
            {str(row["sample_id"]): row for row in _read_jsonl(output)}
            if output.is_file()
            else {}
        )
        values = []
        for request in requests:
            row = dict(output_by_id.get(request.region_id, {}))
            if not row:
                row = {
                    "model_id": "ch_SVTRv2_rec",
                    "model_fingerprint": (
                        "bb4ea682c215607bf65df33f7038ffdba2cec1015428ca2d73ceb0eb45c30156"
                    ),
                    "runtime_fingerprint": "worker-failure",
                    "source_crop_sha256": request.crop_sha256,
                    "output_contract_status": "PROVIDER_CRASH",
                    "warnings": [f"worker-exit:{completed.returncode}"],
                }
            values.append(row)
        return values


class InProcessChSVTRv2StageRunner(SubprocessChSVTRv2StageRunner):
    """Reuse the current Paddle process while retaining the worker's validation."""

    def _launch_worker(self, command, **kwargs):
        from .workers.ch_svtrv2_rec import run

        def option(name):
            return Path(command[command.index(name) + 1])

        try:
            metrics = run(option('--manifest'), self.crop_root, self.model_root,
                          self.model_identity, option('--output'), 1,
                          batch_size=1, short_batch_size=8, execution_mode='in_process')
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(metrics), stderr='')
        except Exception as exc:
            return subprocess.CompletedProcess(command, 1, stdout='', stderr=f'{type(exc).__name__}:{exc}')


class SubprocessGOTStageRunner:
    """Run packaged SHA-bound GOT; legacy provider ids remain replay-compatible."""

    @classmethod
    def from_registry(
        cls,
        *,
        python: str | Path,
        crop_root: str | Path,
        output_root: str | Path,
        models_root: str | Path | None = None,
        config_path: str | Path | None = None,
        mcp_root: str | Path | None = None,
        generation_budget: int = 256,
        vram_limit_mib: int = GOT_DEFAULT_VRAM_LIMIT_MIB,
        batch_size: int = 32,
        vision_batch_size: int = 1,
    ) -> SubprocessGOTStageRunner:
        """Resolve the frozen formal C model and runtime identity by model_id."""

        resolution = ModelRegistry(
            models_root=models_root,
            config_path=config_path,
            mcp_root=mcp_root,
        ).resolve("got-ocr2-0", deep=True)
        runtime_fingerprint = _semantic_sha256(
            {
                "provider": "C_GOT_OCR2",
                "model_id": resolution.model_id,
                "model_fingerprint": resolution.manifest["model_fingerprint"],
                "precision": resolution.manifest["precision_baseline"],
                "device": resolution.manifest["expected_device"],
                "batch_size": batch_size,
                "batch_retry_policy": GOT_BATCH_RETRY_POLICY,
                "vram_allocator_limit_mib": vram_limit_mib,
                "vision_batch_size": vision_batch_size,
                "crop_ordering": GOT_CROP_ORDERING,
                'vision_attention':'torch-sdpa-relative-position-v1',
                "eos_token": "<|im_end|>",
                "do_sample": False,
                "max_new_tokens": generation_budget,
            }
        )
        return cls(
            python=python,
            crop_root=crop_root,
            output_root=output_root,
            model_roots={"C_GOT_OCR2": resolution.model_root},
            model_fingerprints={
                "C_GOT_OCR2": resolution.manifest["model_fingerprint"]
            },
            runtime_fingerprint=runtime_fingerprint,
            generation_budgets={"C_GOT_OCR2": generation_budget},
            vram_limit_mib=vram_limit_mib,
            batch_size=batch_size,
            vision_batch_size=vision_batch_size,
        )

    def __init__(
        self,
        *,
        python: str | Path,
        crop_root: str | Path,
        output_root: str | Path,
        model_roots: Mapping[str, str | Path],
        model_fingerprints: Mapping[str, str],
        runtime_fingerprint: str,
        generation_budgets: Mapping[str, int],
        vram_limit_mib: int = GOT_DEFAULT_VRAM_LIMIT_MIB,
        batch_size: int = 32,
        vision_batch_size: int = 1,
    ) -> None:
        self.python = Path(python).resolve()
        self.crop_root = Path(crop_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.model_roots = {
            key: Path(value).resolve() for key, value in model_roots.items()
        }
        self.model_fingerprints = dict(model_fingerprints)
        self.runtime_fingerprint = runtime_fingerprint
        self.generation_budgets = dict(generation_budgets)
        self.vram_limit_mib = vram_limit_mib
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("GOT_BATCH_SIZE_OUT_OF_RANGE")
        self.batch_size = batch_size
        if type(vision_batch_size) is not int or not 1 <= vision_batch_size <= 4:
            raise ValueError('GOT_VISION_BATCH_SIZE_OUT_OF_RANGE')
        self.vision_batch_size = vision_batch_size
        self.stage_metrics: dict[str, dict[str, Any]] = {}
        self._deferred_process = None

    @staticmethod
    def _worker_environment():
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', PYTHONIOENCODING='utf-8')
        return env

    def prepare_dependencies(self) -> None:
        from .workers.deferred_got import DeferredGOTProcess

        if self._deferred_process is None:
            self._deferred_process = DeferredGOTProcess(self.python, self.output_root, self._worker_environment())
            self._deferred_process.start()

    def close(self) -> None:
        if self._deferred_process is not None:
            self._deferred_process.close()

    def _launch_worker(self, command, **kwargs):
        if self._deferred_process is not None:
            return self._deferred_process.run(command, **kwargs)
        return subprocess.run(command, **kwargs)

    def __call__(
        self, provider_id: str, requests: Sequence[TextRecognitionRequest]
    ) -> list[dict[str, Any]]:
        if provider_id not in {
            "C_GOT_OCR2",
            "B_GOT_OCR2",
            "C_HUNYUAN_OCR_1_5",
        }:
            raise ValueError(f"UNSUPPORTED_GENERATIVE_PROVIDER:{provider_id}")
        provider_arg = (
            "GOT"
            if provider_id in {"C_GOT_OCR2", "B_GOT_OCR2"}
            else "HUNYUAN"
        )
        stage_root = self.output_root / provider_arg.casefold()
        stage_root.mkdir(parents=True, exist_ok=False)
        manifest = stage_root / "manifest.json"
        rows = []
        for request in requests:
            crop_path = Path(request.crop_ref).resolve()
            try:
                relative = crop_path.relative_to(self.crop_root).as_posix()
            except ValueError as exc:
                raise RuntimeError(
                    f"THREE_MODEL_LIVE_CROP_OUTSIDE_ROOT:{crop_path}"
                ) from exc
            if not crop_path.is_file() or _sha256_file(crop_path) != request.crop_sha256:
                raise RuntimeError(
                    f"THREE_MODEL_LIVE_SOURCE_CROP_MISMATCH:{request.region_id}"
                )
            rows.append(
                {
                    "sample_id": request.region_id,
                    "crop_local_relpath": relative,
                    "crop_sha256": request.crop_sha256,
                }
            )
        for row, request in zip(rows, requests, strict=True):
            row['ocr_format'] = request.source_provenance.get('got_format', False)
        crop_ordering = None
        if provider_arg == 'GOT':
            from .got_batching import order_got_mode_crops
            rows, crop_ordering = order_got_mode_crops(rows, crop_root=self.crop_root, batch_size=self.batch_size)
        _write_json(
            manifest,
            {"schema": "bemarkdown-three-model-live-provider-manifest-v1", "rows": rows},
        )
        metrics_path = stage_root / "metrics.json"
        env = self._worker_environment()
        completed = self._launch_worker(
            [
                str(self.python),
                "-m",
                GOT_OCR2_WORKER_MODULE,
                "--provider",
                provider_arg,
                "--manifest",
                str(manifest),
                "--crop-root",
                str(self.crop_root),
                "--model-root",
                str(self.model_roots[provider_id]),
                "--batch-size",
                str(self.batch_size),
                '--vision-batch-size',
                str(self.vision_batch_size),
                "--generation-budget",
                str(self.generation_budgets[provider_id]),
                "--vram-limit-mib",
                str(self.vram_limit_mib),
                "--output",
                str(metrics_path),
            ],
            cwd=self.output_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=14400,
        )
        (stage_root / "stdout.txt").write_text(
            completed.stdout, encoding="utf-8", newline="\n"
        )
        (stage_root / "stderr.txt").write_text(
            completed.stderr, encoding="utf-8", newline="\n"
        )
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else {
                "provider": provider_arg,
                "population": len(requests),
                "error": f"worker-exit:{completed.returncode}",
                "oom_count": 0,
                "cpu_fallback_count": 0,
            }
        )
        metrics.update(
            {
                "process_returncode": completed.returncode,
                "load_count": 1,
                "unload_count": 1,
                "process_exit_unload": True,
                "request_region_ids": [request.region_id for request in requests],
            }
        )
        if self._deferred_process is not None:
            metrics['dependency_preload'] = self._deferred_process.metrics()
        if crop_ordering is not None:
            metrics['crop_ordering'] = crop_ordering
        self.stage_metrics[provider_id] = metrics
        output_path = metrics_path.with_suffix(".jsonl")
        output_by_id = (
            {str(row["sample_id"]): row for row in _read_jsonl(output_path)}
            if output_path.is_file()
            else {}
        )
        values = []
        for request in requests:
            row = dict(output_by_id.get(request.region_id, {}))
            error = metrics.get("error") or (
                f"worker-exit:{completed.returncode}" if completed.returncode else None
            )
            row.update(
                {
                    "model_id": (
                        "GOT-OCR2.0" if provider_arg == "GOT" else "HunyuanOCR-1.5"
                    ),
                    "model_fingerprint": self.model_fingerprints[provider_id],
                    "runtime_fingerprint": self.runtime_fingerprint,
                    "source_crop_sha256": request.crop_sha256,
                    "output_contract_status": (
                        "PASS"
                        if row.get("normalized_output")
                        else "PROVIDER_CRASH"
                        if error or completed.returncode
                        else "EMPTY_OUTPUT"
                    ),
                    "warnings": [str(error)] if error else [],
                }
            )
            values.append(row)
        return values


# Historical callers imported the provider-neutral name. Keep object identity while
# the production interface names its actual frozen C provider explicitly.
SubprocessGenerativeStageRunner = SubprocessGOTStageRunner


class ProductionThreeModelLiveComposition:
    """Stage A/B for every OCR route and stage C only for Policy-v2 triggers."""

    def __init__(
        self,
        provider_a: TextEvidenceProvider,
        provider_b: TextEvidenceProvider,
        provider_c: TextEvidenceProvider,
        *,
        comparator: ConsensusComparatorV1 | None = None,
        router: ThirdModelRouterV1 | None = None,
        resolver: EvidenceResolverV1 | None = None,
        page_bounded_cropper: PageVisualTextRecoveryBoundedCropper | None = None,
    ) -> None:
        provider_order = (
            str(getattr(provider_a, "provider_id", "")),
            str(getattr(provider_b, "provider_id", "")),
            str(getattr(provider_c, "provider_id", "")),
        )
        expected_order = (
            "A_PP_OCRV6",
            "B_CH_SVTRV2_REC",
            "C_GOT_OCR2",
        )
        if provider_order != expected_order:
            raise ValueError(
                f"PRODUCTION_PROVIDER_ORDER_INVALID:{provider_order}:{expected_order}"
            )
        self.provider_a = provider_a
        self.provider_b = provider_b
        self.provider_c = provider_c
        self.comparator = comparator or ConsensusComparatorV1()
        self.router = router or ThirdModelRouterV1()
        self.resolver = resolver or EvidenceResolverV1()
        self.page_bounded_cropper = (
            page_bounded_cropper or PageVisualTextRecoveryBoundedCropper()
        )

    def resolve(
        self,
        *,
        plans: Sequence[Mapping[str, Any]],
        crops_by_route: Mapping[str, Mapping[str, Any]],
    ) -> ProductionThreeModelStagedResult:
        routes = [
            copy.deepcopy(dict(route))
            for plan in plans
            for route in plan["routes"]
            if route["adapter"] in OCR_ADAPTERS
        ]
        route_ids = [str(route["route_id"]) for route in routes]
        if len(route_ids) != len(set(route_ids)):
            raise RuntimeError("THREE_MODEL_LIVE_DUPLICATE_ROUTE_ID")
        missing = [route_id for route_id in route_ids if route_id not in crops_by_route]
        if missing:
            raise RuntimeError(f"THREE_MODEL_LIVE_CROP_MISSING:{missing}")
        for route_id in route_ids:
            crop = crops_by_route[route_id]
            crop_path = Path(str(crop["path"])).resolve()
            expected_sha = str(crop["content_sha256"])
            if not crop_path.is_file() or _sha256_file(crop_path) != expected_sha:
                raise RuntimeError(
                    f"THREE_MODEL_LIVE_SOURCE_CROP_MISMATCH:{route_id}"
                )
        workload = _prepare_execution_workload(
            routes=routes,
            crops_by_route=crops_by_route,
            provider_a=self.provider_a,
            page_bounded_cropper=self.page_bounded_cropper,
        )
        requests = [
            build_text_recognition_request(
                route, workload.crops_by_request[str(route["route_id"])]
            )
            for route in workload.execution_routes
        ]
        replay = ThreeModelOCRReplayHarness(
            self.provider_a,
            self.provider_b,
            self.provider_c,
            comparator=self.comparator,
            router=self.router,
            resolver=self.resolver,
        ).run(requests)
        review_candidate_by_route = {
            str(row["region_id"]): row for row in replay.agent_review_candidates
        }
        agent_tasks: list[dict[str, Any]] = []
        provenance_by_request: dict[str, dict[str, Any]] = {}
        terminal_counts = Counter(
            str(row["resolution_status"]) for row in replay.final_outputs
        )
        for request, output in zip(requests, replay.final_outputs, strict=True):
            provenance = {
                "schema": THREE_MODEL_OCR_SCHEMA,
                "request": request.to_dict(),
                "evidence": copy.deepcopy(output["evidence"]),
                "comparator": copy.deepcopy(output["comparator"]),
                "third_model_router": copy.deepcopy(output["third_model_router"]),
                "resolver": copy.deepcopy(output["resolver"]),
            }
            review_candidate = review_candidate_by_route.get(request.region_id)
            if review_candidate is not None:
                candidate = build_agent_task_v2_candidate(request, review_candidate)
                agent_tasks.append(candidate)
                provenance["agent_review_candidate"] = copy.deepcopy(candidate)
            provenance_by_request[request.region_id] = provenance
        provenance_by_route: dict[str, dict[str, Any]] = {}
        route_terminal_counts: Counter[str] = Counter()
        for route in workload.source_routes:
            route_id = str(route["route_id"])
            page_plan = workload.page_plans_by_route.get(route_id)
            if page_plan is None:
                provenance = provenance_by_request[route_id]
                provenance_by_route[route_id] = provenance
                route_terminal_counts[str(provenance["resolver"]["resolution_status"])] += 1
                continue
            unit_provenance = [
                provenance_by_request[unit.unit_id] for unit in page_plan.units
            ]
            assembly = assemble_page_bounded_text_recovery(
                page_plan,
                [
                    {
                        "unit_id": unit.unit_id,
                        "selected_text": provenance["resolver"].get("selected_text"),
                        "resolution_status": provenance["resolver"][
                            "resolution_status"
                        ],
                    }
                    for unit, provenance in zip(
                        page_plan.units, unit_provenance, strict=True
                    )
                ],
            )
            aggregate_status = _aggregate_resolution_status(
                assembly.resolution_statuses
            )
            route_terminal_counts[aggregate_status] += 1
            source_request = build_text_recognition_request(
                route, crops_by_route[route_id]
            )
            provenance_by_route[route_id] = {
                "schema": ("bemarkdown-three-model-page-recovery-provenance-v2"
                           if route['adapter'] == 'PAGE_VISUAL_TEXT_RECOVERY'
                           else 'bemarkdown-three-model-region-lines-provenance-v1'),
                "request": source_request.to_dict(),
                "bounded_units": copy.deepcopy(unit_provenance),
                "unit_contracts": [
                    unit.to_dict() for unit in page_plan.units
                ],
                "comparator": {
                    "scope": "CANONICAL_BOUNDED_TEXT_CROPS",
                    "bounded_unit_count": len(page_plan.units),
                },
                "third_model_router": {
                    "triggered": any(
                        bool(provenance["third_model_router"]["triggered"])
                        for provenance in unit_provenance
                    )
                },
                "resolver": {
                    "selected_text": assembly.selected_text,
                    "normalized_text": normalize_text(assembly.selected_text),
                    "resolution_status": aggregate_status,
                    "selected_provider_basis": "BOUNDED_UNIT_READING_ORDER_ASSEMBLY",
                    "review_reasons": sorted(
                        {
                            str(reason)
                            for provenance in unit_provenance
                            for reason in provenance["resolver"].get(
                                "review_reasons", ()
                            )
                        }
                    ),
                },
                "assembly": {
                    "unit_ids": list(assembly.unit_ids),
                    "resolution_statuses": list(assembly.resolution_statuses),
                    "validation": copy.deepcopy(assembly.validation),
                },
                "line_conservation": page_plan.line_conservation(),
                "reading_order_validation": page_plan.reading_order_validation(),
                "spatial_validation": page_plan.spatial_validation(),
                'region_line_coverage':workload.region_line_coverage.get(route_id),
            }
        page_route_ids = {str(route['route_id']) for route in workload.source_routes
                          if route['adapter'] == 'PAGE_VISUAL_TEXT_RECOVERY'}
        region_line_ids = set(workload.page_plans_by_route) - page_route_ids
        page_recovery_bounded_unit_count = sum(len(workload.page_plans_by_route[key].units) for key in page_route_ids)
        ordinary_route_count = len(workload.source_routes) - len(page_route_ids)
        source_page_count = len(
            {
                (str(route["document_id"]), int(route["page_index"]))
                for route in workload.source_routes
            }
        )
        metrics = {
            "schema": "bemarkdown-production-bounded-crop-ocr-accounting-v1",
            "ocr_text_routes": len(workload.source_routes),
            "ordinary_ocr_text_route_count": ordinary_route_count,
            "page_recovery_route_count": len(page_route_ids),
            'region_line_route_count':len(region_line_ids),
            'region_line_bounded_unit_count':sum(len(workload.page_plans_by_route[key].units) for key in region_line_ids),
            'region_line_coverage':copy.deepcopy(workload.region_line_coverage),
            "bounded_voting_unit_count": len(requests),
            "page_recovery_bounded_unit_count": page_recovery_bounded_unit_count,
            "three_model_request_count": len(requests),
            "a_attempts": replay.metrics["a_calls"],
            "b_attempts": replay.metrics["b_calls"],
            "c_attempts": replay.metrics["c_calls"],
            "c_trigger_count": replay.metrics["c_trigger_count"],
            "c_trigger_rate": replay.metrics["c_trigger_rate"],
            "source_page_count": source_page_count,
            "bounded_units_per_page": (
                len(requests) / source_page_count if source_page_count else 0.0
            ),
            "got_calls_per_page": (
                replay.metrics["c_calls"] / source_page_count
                if source_page_count
                else 0.0
            ),
            "terminal_counts": {
                status: terminal_counts.get(status, 0)
                for status in RESOLUTION_STATUSES_V2
            },
            "route_terminal_counts": {
                status: route_terminal_counts.get(status, 0)
                for status in RESOLUTION_STATUSES_V2
            },
            "three_model_text_evidence_rows": len(provenance_by_route),
            "three_model_request_evidence_rows": len(provenance_by_request),
            "request_source_route_ids": dict(
                workload.request_source_route_ids
            ),
            "source_route_request_counts": dict(
                Counter(workload.request_source_route_ids.values())
            ),
            "agent_candidate_count": len(agent_tasks),
        }
        _validate_accounting(metrics)
        metrics["gate"] = "PASS"
        runtime = _ResolvedThreeModelTextRuntime(provenance_by_route, metrics)
        return ProductionThreeModelStagedResult(
            runtime,
            metrics,
            agent_tasks,
            provenance_by_route,
        )

    def build_content_executor(
        self,
        output_dir: str | Path,
        *,
        plans: Sequence[Mapping[str, Any]],
        crops_by_route: Mapping[str, Mapping[str, Any]],
        cropper: Any,
        **executor_options: Any,
    ) -> tuple[PdfContentAdapterExecutor, ProductionThreeModelStagedResult]:
        staged = self.resolve(plans=plans, crops_by_route=crops_by_route)
        executor = PdfContentAdapterExecutor(
            output_dir,
            cropper=cropper,
            ocr_runtime_factory=lambda: staged.runtime,
            require_three_model_provenance=True,
            **executor_options,
        )
        return executor, staged


class ThreeModelLiveRoutePreflight:
    """Bounded fail-fast gate before a large live cohort may continue."""

    def validate(
        self,
        region_rows: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        ocr_rows = [
            row
            for row in region_rows
            if row.get("provenance", {}).get("route_decision", {}).get("adapter")
            in OCR_ADAPTERS
        ]
        for row in region_rows:
            figure = row.get('provenance', {}).get('figure_label_recognition', {})
            evidence = figure.get('three_model_text_evidence')
            if isinstance(evidence, Mapping):
                ocr_rows.append({'provenance': {'three_model_text_evidence': evidence}})
        population = len(ocr_rows)
        if population < 1:
            raise RuntimeError("PRODUCTION_THREE_MODEL_PREFLIGHT_OCR_ROUTE_REQUIRED")
        reported_routes = int(metrics.get("ocr_text_routes", population))
        request_population = int(metrics.get("three_model_request_count", population))
        if reported_routes != population:
            raise RuntimeError("THREE_MODEL_LIVE_ACCOUNTING_FAILED")
        if int(metrics.get("b_attempts", -1)) != request_population:
            raise RuntimeError("PRODUCTION_THREE_MODEL_LIVE_ROUTE_STILL_NOT_WIRED")
        if int(metrics.get("a_attempts", -1)) != request_population:
            raise RuntimeError("THREE_MODEL_LIVE_ACCOUNTING_FAILED")
        provenance_rows = sum(
            isinstance(row.get("provenance", {}).get("three_model_text_evidence"), Mapping)
            for row in ocr_rows
        )
        if provenance_rows != population:
            raise RuntimeError("PRODUCTION_THREE_MODEL_PROVENANCE_MISSING")
        terminal_counts = metrics.get("terminal_counts")
        if isinstance(terminal_counts, Mapping) and sum(
            int(terminal_counts.get(status, 0)) for status in RESOLUTION_STATUSES_V2
        ) != request_population:
            raise RuntimeError("THREE_MODEL_LIVE_ACCOUNTING_FAILED")
        route_terminal_counts = metrics.get("route_terminal_counts")
        if isinstance(route_terminal_counts, Mapping) and sum(
            int(route_terminal_counts.get(status, 0))
            for status in RESOLUTION_STATUSES_V2
        ) != population:
            raise RuntimeError("THREE_MODEL_LIVE_ACCOUNTING_FAILED")
        request_sources = metrics.get("request_source_route_ids")
        if isinstance(request_sources, Mapping) and len(request_sources) != request_population:
            raise RuntimeError("THREE_MODEL_LIVE_ACCOUNTING_FAILED")
        return {
            "schema": "bemarkdown-bounded-crop-ocr-live-route-preflight-v1",
            "ocr_text_routes": population,
            "three_model_request_count": request_population,
            "bounded_voting_unit_count": int(
                metrics.get("bounded_voting_unit_count", request_population)
            ),
            "page_recovery_bounded_unit_count": int(
                metrics.get("page_recovery_bounded_unit_count", 0)
            ),
            "a_attempts": int(metrics["a_attempts"]),
            "b_attempts": int(metrics["b_attempts"]),
            "three_model_text_evidence_rows": provenance_rows,
            "coverage": provenance_rows / population,
            "gate": "PASS",
        }


class _ResolvedThreeModelTextRuntime:
    """Serve current-run staged resolutions through the existing OCR seam."""

    def __init__(
        self,
        provenance_by_route: Mapping[str, Mapping[str, Any]],
        metrics: Mapping[str, Any],
    ) -> None:
        self.provenance_by_route = copy.deepcopy(dict(provenance_by_route))
        self._metrics = copy.deepcopy(dict(metrics))
        self._active_route_id: str | None = None
        self._last_provenance: dict[str, Any] | None = None

    def recognize_route(
        self, route: Mapping[str, Any], crop: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        route_id = str(route["route_id"])
        try:
            provenance = self.provenance_by_route[route_id]
        except KeyError as exc:
            raise RuntimeError(f"THREE_MODEL_LIVE_RESOLUTION_MISSING:{route_id}") from exc
        expected_sha = str(provenance["request"]["crop_sha256"])
        if str(crop.get("content_sha256")) != expected_sha:
            raise RuntimeError(f"THREE_MODEL_LIVE_SOURCE_CROP_MISMATCH:{route_id}")
        self._active_route_id = route_id
        self._last_provenance = copy.deepcopy(provenance)
        if isinstance(provenance.get("bounded_units"), Sequence):
            return [
                _resolved_runtime_line(unit_provenance, index)
                for index, unit_provenance in enumerate(provenance["bounded_units"])
            ]
        resolution = provenance["resolver"]
        from .inline_math_runtime import runtime_selected_text
        selected_text = runtime_selected_text(provenance)
        if not selected_text:
            return []
        evidence_a = provenance["evidence"]["A"]
        confidence = (
            evidence_a.get("confidence")
            if selected_text == str(evidence_a.get("normalized_text") or "")
            else None
        )
        return [
            {
                "text": selected_text,
                "confidence": confidence,
                "reading_sequence": 0,
                "resolution_status": resolution["resolution_status"],
                "review_reasons": list(resolution.get("review_reasons") or ()),
                "third_model_triggered": provenance["third_model_router"]["triggered"],
            }
        ]

    def recognize(self, crop: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self._active_route_id is None:
            raise RuntimeError("THREE_MODEL_ROUTE_CONTEXT_REQUIRED")
        route = {"route_id": self._active_route_id}
        return self.recognize_route(route, crop)

    def recognize_direct(self, crop: Mapping[str, Any]) -> list[dict[str, Any]]:
        return self.recognize(crop)

    def consume_last_provenance(self) -> dict[str, Any] | None:
        value = self._last_provenance
        self._last_provenance = None
        return value

    def fingerprint(self) -> dict[str, Any]:
        model_fingerprints: set[str] = set()
        runtime_fingerprints: set[str] = set()
        for provenance in self.provenance_by_route.values():
            request_rows = (
                provenance["bounded_units"]
                if isinstance(provenance.get("bounded_units"), Sequence)
                else (provenance,)
            )
            for request_row in request_rows:
                for evidence in request_row["evidence"].values():
                    if not isinstance(evidence, Mapping):
                        continue
                    model_fingerprints.add(
                        str(evidence.get("model_fingerprint") or "")
                    )
                    runtime_fingerprints.add(
                        str(evidence.get("runtime_fingerprint") or "")
                    )
        return {
            "det_model_id": "PP-OCRv6",
            "rec_model_id": "ch_SVTRv2_rec+conditional-GOT-OCR2.0",
            "det_model_fingerprint": "+".join(sorted(model_fingerprints)),
            "rec_model_fingerprint": "+".join(sorted(model_fingerprints)),
            "runtime_fingerprint": "+".join(sorted(runtime_fingerprints)),
            "architecture_schema": THREE_MODEL_OCR_SCHEMA,
            "cpu_fallback": False,
        }

    def metrics(self) -> dict[str, Any]:
        return copy.deepcopy(self._metrics)

    def close(self) -> None:
        return None


def _validate_accounting(metrics: Mapping[str, Any]) -> None:
    route_population = int(metrics["ocr_text_routes"])
    request_population = int(metrics.get("three_model_request_count", route_population))
    terminals = metrics["terminal_counts"]
    if (
        int(metrics["a_attempts"]) != request_population
        or int(metrics["b_attempts"]) != request_population
        or int(metrics.get("three_model_request_evidence_rows", request_population))
        != request_population
        or int(metrics["three_model_text_evidence_rows"]) != route_population
        or sum(int(terminals.get(status, 0)) for status in RESOLUTION_STATUSES_V2)
        != request_population
    ):
        raise RuntimeError("THREE_MODEL_LIVE_ACCOUNTING_FAILED")
    route_terminals = metrics.get("route_terminal_counts")
    if isinstance(route_terminals, Mapping) and sum(
        int(route_terminals.get(status, 0)) for status in RESOLUTION_STATUSES_V2
    ) != route_population:
        raise RuntimeError("THREE_MODEL_LIVE_ACCOUNTING_FAILED")
    request_sources = metrics.get("request_source_route_ids")
    if isinstance(request_sources, Mapping) and len(request_sources) != request_population:
        raise RuntimeError("THREE_MODEL_LIVE_ACCOUNTING_FAILED")
    if int(metrics["c_attempts"]) != int(metrics["c_trigger_count"]):
        raise RuntimeError("THREE_MODEL_LIVE_ACCOUNTING_FAILED")


def _prepare_execution_workload(
    *,
    routes: Sequence[Mapping[str, Any]],
    crops_by_route: Mapping[str, Mapping[str, Any]],
    provider_a: TextEvidenceProvider,
    page_bounded_cropper: PageVisualTextRecoveryBoundedCropper,
) -> _ProductionThreeModelExecutionWorkload:
    execution_routes: list[dict[str, Any]] = []
    crops_by_request: dict[str, dict[str, Any]] = {}
    page_plans_by_route: dict[str, PageBoundedTextPlan] = {}
    request_source_route_ids: dict[str, str] = {}
    region_line_coverage: dict[str, dict[str, Any]] = {}
    page_line_loader = getattr(provider_a, "page_recovery_lines", None)
    for source_route in routes:
        route = copy.deepcopy(dict(source_route))
        route_id = str(route["route_id"])
        source_crop = copy.deepcopy(dict(crops_by_route[route_id]))
        if route["adapter"] != "PAGE_VISUAL_TEXT_RECOVERY":
            from .region_line_coverage import assess_region_line_coverage
            lines = page_line_loader(route_id) if callable(page_line_loader) else []
            coverage = assess_region_line_coverage(source_crop, lines)
            region_line_coverage[route_id] = coverage
            if not coverage['eligible'] and not route.get('provenance', {}).get('figure_label_parent_route_id'):
                execution_routes.append(route)
                crops_by_request[route_id] = source_crop
                request_source_route_ids[route_id] = route_id
                continue
        if not callable(page_line_loader):
            raise TypeError("PAGE_TEXT_RECOVERY_PP_LINE_PROVIDER_REQUIRED")
        pp_lines = page_line_loader(route_id)
        if not pp_lines:
            raise RuntimeError(f"PAGE_TEXT_RECOVERY_PP_LINES_REQUIRED:{route_id}")
        page_plan = page_bounded_cropper.prepare(route, source_crop, pp_lines)
        if not page_plan.units:
            raise RuntimeError(f"PAGE_BOUNDED_TEXT_UNITS_REQUIRED:{route_id}")
        page_plans_by_route[route_id] = page_plan
        for unit in page_plan.units:
            unit_route = copy.deepcopy(route)
            unit_route["route_id"] = unit.unit_id
            unit_route["provenance"] = {
                **copy.deepcopy(dict(route.get("provenance") or {})),
                "page_recovery_bounded_unit": unit.to_dict(),
            }
            unit_width = unit.bbox_pixel[2] - unit.bbox_pixel[0]
            unit_height = unit.bbox_pixel[3] - unit.bbox_pixel[1]
            unit_crop = {
                "path": unit.crop_ref,
                "content_sha256": unit.crop_sha256,
                "bbox_pdf_pt": list(unit.bbox_pdf_pt),
                "bbox_render_px": [0, 0, unit_width, unit_height],
                "width": unit_width,
                "height": unit_height,
                "source_path": source_crop.get("source_path"),
                "page_index": source_crop.get("page_index"),
                "source_page_sha256": unit.source_page_sha256,
            }
            execution_routes.append(unit_route)
            crops_by_request[unit.unit_id] = unit_crop
            request_source_route_ids[unit.unit_id] = route_id
    execution_ids = [str(route["route_id"]) for route in execution_routes]
    if len(execution_ids) != len(set(execution_ids)):
        raise RuntimeError("THREE_MODEL_LIVE_DUPLICATE_REQUEST_ID")
    return _ProductionThreeModelExecutionWorkload(
        source_routes=tuple(copy.deepcopy(dict(route)) for route in routes),
        execution_routes=tuple(execution_routes),
        crops_by_request=crops_by_request,
        page_plans_by_route=page_plans_by_route,
        request_source_route_ids=request_source_route_ids,
        region_line_coverage=region_line_coverage,
    )


def _aggregate_resolution_status(statuses: Sequence[str]) -> str:
    status_set = set(statuses)
    for status in (
        "AGENT_REQUIRED_RUNTIME_FAILURE",
        "AGENT_REQUIRED_ALL_DIFFER",
        "CONSENSUS_BC",
        "CONSENSUS_AC",
        "CONSENSUS_AB",
    ):
        if status in status_set:
            return status
    raise RuntimeError(f"PAGE_TEXT_RECOVERY_TERMINAL_STATUS_INVALID:{sorted(status_set)}")


def _resolved_runtime_line(
    provenance: Mapping[str, Any], reading_sequence: int
) -> dict[str, Any]:
    resolution = provenance["resolver"]
    from .inline_math_runtime import runtime_selected_text
    selected_text = runtime_selected_text(provenance)
    evidence_a = provenance["evidence"]["A"]
    confidence = (
        evidence_a.get("confidence")
        if selected_text == str(evidence_a.get("normalized_text") or "")
        else None
    )
    return {
        "text": selected_text,
        "confidence": confidence,
        "reading_sequence": reading_sequence,
        "resolution_status": resolution["resolution_status"],
        "review_reasons": list(resolution.get("review_reasons") or ()),
        "third_model_triggered": provenance["third_model_router"]["triggered"],
    }


def _generative_evidence(
    provider_id: str,
    request: TextRecognitionRequest,
    output: Mapping[str, Any],
) -> TextRecognitionEvidence:
    raw_text = str(
        output.get("raw_text")
        or output.get("raw_output")
        or output.get("extracted_text")
        or ""
    )
    crop_sha = str(output.get("source_crop_sha256") or request.crop_sha256)
    warnings = tuple(str(value) for value in output.get("warnings", ()))
    crop_mismatch = crop_sha != request.crop_sha256
    if crop_mismatch:
        warnings = (*warnings, "SOURCE_CROP_SHA_MISMATCH")
    return TextRecognitionEvidence(
        provider_id=provider_id,
        model_id=str(output.get("model_id") or provider_id),
        model_fingerprint=str(output.get("model_fingerprint") or provider_id),
        runtime_fingerprint=str(
            output.get("runtime_fingerprint") or "runtime-unfingerprinted"
        ),
        bbox=request.bbox,
        text=raw_text,
        normalized_text=str(output.get("normalized_output") or normalize_text(raw_text)),
        confidence=(
            float(output["confidence"])
            if output.get("confidence") is not None
            else None
        ),
        confidence_semantics=(
            "NATIVE_REC_CONFIDENCE"
            if output.get("confidence") is not None
            else "NATIVE_CONFIDENCE_UNAVAILABLE"
        ),
        source_crop_ref=request.crop_ref,
        source_crop_sha256=crop_sha,
        latency_seconds=float(output.get("latency_seconds") or 0.0),
        generation_tokens=(
            int(output["generated_token_count"])
            if output.get("generated_token_count") is not None
            else None
        ),
        output_contract_status=(
            "SOURCE_CROP_MISMATCH"
            if crop_mismatch
            else str(output.get("output_contract_status") or "UNKNOWN")
        ),
        warnings=warnings,
        provenance={"current_run_staged_provider": True},
    )


def build_agent_task_v2_candidate(
    request: TextRecognitionRequest, review_candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert the frozen review evidence into the existing AgentTask v2 contract."""

    task_identity = _semantic_sha256(
        {
            "document_id": request.document_id,
            "page_id": request.page_id,
            "region_id": request.region_id,
            "crop_sha256": request.crop_sha256,
            "reason": review_candidate["reason"],
        }
    )
    return AgentTaskV2(
        task_id=f"three-model-ocr-{task_identity[:24]}",
        document_id=request.document_id,
        page_id=request.page_id,
        region_id=request.region_id,
        source_candidate_id=request.source_candidate_id,
        bbox=request.bbox,
        primary_crop_ref=request.crop_ref,
        primary_crop_sha256=request.crop_sha256,
        route_kind=request.route_kind,
        text_track=request.text_track,
        reason=str(review_candidate["reason"]),  # type: ignore[arg-type]
        source_provenance=request.source_provenance,
        evidence=review_candidate["evidence"],
    ).to_dict()


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
