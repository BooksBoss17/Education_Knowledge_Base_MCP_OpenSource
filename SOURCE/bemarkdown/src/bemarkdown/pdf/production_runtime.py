"""Raw-PDF production orchestration and BeMarkdown Package v1 materialization."""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import platform
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from ..pdf_document_ir import (
    document_asset_presentation_ref,
    document_asset_source_ref,
)
from ..pdf_layout_fusion import (
    ADAPTIVE_LOCAL_GROUPING_STRATEGY,
    PAGE_ESCALATION_FUSION_STRATEGY,
)
from ..pdf_layout_runtime import (
    CAPTURE_FLOOR,
    FormalPaddleLayoutRuntime,
    ProductionPdfPageRenderer,
)
from ..pdf_output_audit import CleanHandoffRenderer
from ..pdf_region_ir import build_page_region_ir, validate_page_region_ir
from .modular_pipeline import ModularPdfPipeline, ModularPipelineResult
from .source_evidence import extract_production_pdf_source_evidence

PDF_REPORT_SCHEMA = "bemarkdown-pdf-conversion-report-v1"
PDF_CAPABILITY_STATUS = "PUBLISHED_INTERNAL"
PDF_QUALITY_VALIDATION_STATUS = "PENDING_ZCODE"
CANONICAL_LAYOUT_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class PdfPayloadConversionResult:
    source: Path
    output_dir: Path
    report: dict[str, Any]


class ProductionPdfRuntime:
    """Run the frozen raw-PDF pipeline behind one injected runtime interface."""

    def __init__(
        self,
        *,
        layout_runtime_factory: Callable[[], Any],
        pipeline_factory: Callable[[Path], ModularPdfPipeline] | None = None,
        page_renderer: ProductionPdfPageRenderer | None = None,
        fusion_strategy: Mapping[str, Any] | None = None,
        page_escalation_strategy: Mapping[str, Any] | None = None,
    ) -> None:
        self.layout_runtime_factory = layout_runtime_factory
        self._prepared_layout_runtime = None
        self.pipeline_factory = pipeline_factory or ModularPdfPipeline
        self.page_renderer = page_renderer or ProductionPdfPageRenderer()
        self.fusion_strategy = copy.deepcopy(
            dict(fusion_strategy or ADAPTIVE_LOCAL_GROUPING_STRATEGY)
        )
        self.page_escalation_strategy = copy.deepcopy(
            dict(page_escalation_strategy or PAGE_ESCALATION_FUSION_STRATEGY)
        )

    def prepare_dependencies(self) -> None:
        """Prepare the layout model while Office renders a visual input."""
        if self._prepared_layout_runtime is not None:
            return
        self._prepared_layout_runtime = self.layout_runtime_factory()
        try:
            self._prepared_layout_runtime.load()
        except BaseException:
            self.close_prepared_resources()
            raise

    def close_prepared_resources(self) -> None:
        runtime = self._prepared_layout_runtime
        self._prepared_layout_runtime = None
        unload = getattr(runtime, 'unload', None)
        if callable(unload):
            unload()

    def convert(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        document_id: str,
        debug: bool = False,
    ) -> PdfPayloadConversionResult:
        started = time.perf_counter()
        source = Path(source).resolve()
        output_dir = Path(output_dir).resolve()
        work_root = Path(
            tempfile.mkdtemp(prefix=f"bmdpdf-{document_id[-12:]}-")
        ).resolve()

        pipeline = self.pipeline_factory(work_root)
        inspected = pipeline.inspect_source(source)
        if inspected["document_id"] != document_id:
            raise RuntimeError("PDF_DOCUMENT_ID_MISMATCH")
        page_records = _page_records(inspected)
        renders = self.page_renderer.render(source)
        if len(renders) != len(page_records):
            raise RuntimeError("PDF_LAYOUT_RENDER_PAGE_COUNT_MISMATCH")
        evidence = extract_production_pdf_source_evidence(page_records)

        layout_runtime = self._prepared_layout_runtime
        self._prepared_layout_runtime = None
        if layout_runtime is None:
            layout_runtime = self.layout_runtime_factory()
        layout_identity: dict[str, Any] = {}
        layout_unload: dict[str, Any] = {"unload_status": "NOT_AVAILABLE"}
        try:
            load = getattr(layout_runtime, "load", None)
            if callable(load):
                layout_identity = dict(load())
            raw_pages, page_region_ir = _run_layout(
                layout_runtime,
                renders=renders,
                page_records=page_records,
                evidence=evidence,
                document_id=document_id,
                layout_identity=layout_identity,
            )
        finally:
            unload = getattr(layout_runtime, "unload", None)
            if callable(unload):
                value = unload()
                if isinstance(value, Mapping):
                    layout_unload = dict(value)

        # The layout model is no longer needed once its source-bound regions exist.
        # Release it before the content models compete for the same GPU memory.
        pipeline_result = pipeline.run_from_page_ir(
            page_region_ir=page_region_ir,
            raw_layout_pages=raw_pages,
            page_records=page_records,
            source_evidence=evidence,
            fusion_strategy=self.fusion_strategy,
            page_escalation_strategy=self.page_escalation_strategy,
            source={
                "type": "PDF",
                "name": source.name,
                "sha256": inspected["sha256"],
                "bytes": inspected["bytes"],
            },
        )

        report = _write_pdf_payload(
            source=source,
            output_dir=output_dir,
            inspected=inspected,
            pipeline_result=pipeline_result,
            layout_identity=layout_identity,
            layout_unload=layout_unload,
            elapsed=time.perf_counter() - started,
            debug=debug,
        )
        shutil.rmtree(work_root)
        return PdfPayloadConversionResult(source, output_dir, report)


class RegistryBackedThreeModelTextResolver:
    """Resolve and stage A/B/C only through the formal model registry."""

    def __init__(
        self,
        work_root: str | Path,
        *,
        models_root: str | Path | None = None,
        config_path: str | Path | None = None,
        mcp_root: str | Path | None = None,
        inline_math: bool = True,
        inline_math_candidate_policy: str = 'source-risk-v3',
        fraction_repair: bool = True,
        inline_atomic_superscripts: bool = True,
        independent_source_pages: bool = False,
    ) -> None:
        self.work_root = Path(work_root).resolve()
        self.models_root = models_root
        self.config_path = config_path
        self.mcp_root = mcp_root
        self.inline_math = inline_math
        self.inline_math_candidate_policy = inline_math_candidate_policy
        self.fraction_repair = fraction_repair
        self.inline_atomic_superscripts = inline_atomic_superscripts
        self.independent_source_pages = independent_source_pages

    def _recognize_provider_a(self, runtime, plans, crops_by_route):
        groups = []
        for plan in plans:
            rows = [(str(route['route_id']), {
                **dict(crops_by_route[str(route['route_id'])]),
                'complete_region_lines': route['adapter'] == 'OCR_TEXT_REGION'
                and not crops_by_route[str(route['route_id'])].get('figure_label_parent_route_id'),
            }) for route in plan['routes'] if str(route['route_id']) in crops_by_route]
            if rows:
                groups.append(rows)
        if not self.independent_source_pages:
            groups = [[row for group in groups for row in group]]
        result = {}
        recognize_batch = getattr(runtime, 'recognize_batch', None)
        for inputs in groups:
            if not inputs:
                continue
            values = (recognize_batch([crop for _, crop in inputs], complete_regions=True)
                      if callable(recognize_batch) else [runtime.recognize(crop) for _, crop in inputs])
            if len(values) != len(inputs):
                raise RuntimeError('PP_OCR_BATCH_ROUTE_CARDINALITY')
            result.update((key, value) for (key, _), value in zip(inputs, values, strict=True))
        return result

    def resolve_with_formula_context(self, *, plans, crops_by_route, formula_runtime, formula_rows):
        return self.resolve(plans=plans, crops_by_route=crops_by_route,
                            formula_runtime=formula_runtime, formula_rows=formula_rows)

    def resolve_with_formula_runtime(self, *, plans, crops_by_route, formula_runtime):
        return self.resolve(plans=plans, crops_by_route=crops_by_route, formula_runtime=formula_runtime)

    def resolve(
        self,
        *,
        plans: Sequence[Mapping[str, Any]],
        crops_by_route: Mapping[str, Mapping[str, Any]],
        formula_runtime=None,
        formula_rows=(),
    ) -> Any:
        from .production_three_model_live import SubprocessGOTStageRunner
        from ..resource_profiles import resource_profile

        profile = resource_profile()

        stage_root = self.work_root / 'three_model_stages'
        stage_root.mkdir()
        runner = SubprocessGOTStageRunner.from_registry(
            vram_limit_mib=profile.got_allocator_mib,
            batch_size=profile.got_batch_size,
            vision_batch_size=profile.got_vision_batch_size,
            python=sys.executable, crop_root=self.work_root, output_root=stage_root,
            models_root=self.models_root, config_path=self.config_path, mcp_root=self.mcp_root,
        )
        try:
            # Full-page routes expand into many line requests during A/B work;
            # their input crop count does not represent that recognition work.
            has_page_recovery = any(
                route.get('adapter') == 'PAGE_VISUAL_TEXT_RECOVERY'
                and str(route.get('route_id')) in crops_by_route
                for plan in plans for route in plan.get('routes', ())
            )
            if profile.overlap_got_dependencies or len(crops_by_route) >= 64 or has_page_recovery:
                runner.prepare_dependencies()
            result = self._resolve_with_prepared_got(
                plans=plans, crops_by_route=crops_by_route,
                provider_c_runner=runner, stage_root=stage_root,
                formula_runtime=formula_runtime,
                formula_rows=formula_rows,
            )
            result.metrics['resource_profile'] = profile.to_dict()
            return result
        finally:
            runner.close()

    def _resolve_with_prepared_got(
        self, *, plans, crops_by_route, provider_c_runner, stage_root, formula_runtime=None, formula_rows=(),
    ) -> Any:
        from ..pdf_content_router import PaddleXPdfOcrRuntime
        from .production_three_model_live import (
            CurrentRunGenerativeStageProvider,
            CurrentRunPaddleTextEvidenceProvider,
            CurrentRunRecognitionStageProvider,
            ProductionThreeModelLiveComposition,
            InProcessChSVTRv2StageRunner,
        )

        for crop in crops_by_route.values():
            Path(str(crop["path"])).resolve().relative_to(self.work_root)
        provider_a_runtime = PaddleXPdfOcrRuntime(
            models_root=self.models_root,
            config_path=self.config_path,
            mcp_root=self.mcp_root,
        )
        outputs_by_route: dict[str, list[dict[str, Any]]] = {}
        provider_a_runtime.character_positions = self.inline_math
        try:
            inputs = [(str(route['route_id']), {**dict(crops_by_route[str(route['route_id'])]),
                    'complete_region_lines':route['adapter'] == 'OCR_TEXT_REGION'
                    and not crops_by_route[str(route['route_id'])].get('figure_label_parent_route_id')})
                      for plan in plans for route in plan['routes'] if str(route['route_id']) in crops_by_route]
            outputs_by_route.update(self._recognize_provider_a(provider_a_runtime, plans, crops_by_route))
            from .figure_labels import isolate_figure_label_lines
            from .figure_label_segments import split_spaced_figure_labels
            figure_isolation = {}
            for route_id, crop in inputs:
                if crop.get('figure_label_parent_route_id') and outputs_by_route[route_id]:
                    figure_isolation[route_id] = isolate_figure_label_lines(
                        crop, outputs_by_route[route_id], provider_a_runtime._rec_model)
                    figure_isolation[route_id]['segmentation'] = split_spaced_figure_labels(
                        crop, outputs_by_route[route_id], provider_a_runtime._rec_model)
            fingerprint = provider_a_runtime.fingerprint()
            if figure_isolation:
                fingerprint.setdefault('execution', {})['figure_label_isolation'] = figure_isolation
        finally:
            provider_a_runtime.close()

        provider_a = CurrentRunPaddleTextEvidenceProvider(
            outputs_by_route,
            {
                'execution':fingerprint.get('execution', {}),
                "det": {
                    "model": {
                        "model_id": fingerprint["det_model_id"],
                        "model_fingerprint": fingerprint["det_model_fingerprint"],
                    }
                },
                "rec": {
                    "model": {
                        "model_id": fingerprint["rec_model_id"],
                        "model_fingerprint": fingerprint["rec_model_fingerprint"],
                    }
                },
            },
        )
        provider_b_runner = InProcessChSVTRv2StageRunner.from_registry(
            python=sys.executable,
            crop_root=self.work_root,
            output_root=stage_root,
            models_root=self.models_root,
            config_path=self.config_path,
            mcp_root=self.mcp_root,
        )
        # No detected label is a detection result, not a whole-figure line vote.
        from .figure_labels import build_figure_label_results, is_figure_label_route
        from .text_scope_review import missing_line_scope_reviews
        source_scope_reviews = missing_line_scope_reviews(plans, outputs_by_route, crops_by_route)
        resolution_plans = []
        for plan in plans:
            current = copy.deepcopy(plan)
            current['routes'] = [route for route in plan['routes'] if
                                 not is_figure_label_route(route) or outputs_by_route.get(str(route['route_id']))]
            current['routes'] = [r for r in current['routes'] if str(r['route_id']) not in source_scope_reviews]
            resolution_plans.append(current)
        from ..formula_ocr import create_production_formulanet_runtime
        from concurrent.futures import ThreadPoolExecutor
        from .figure_labels import FigureMathPrefetch, figure_math_prefetch_requests
        from .figure_symbols import select_figure_math_requests
        from .production_three_model_live import _generative_evidence
        formula_factory = (lambda: formula_runtime) if formula_runtime is not None else (
            lambda: create_production_formulanet_runtime(
                models_root=self.models_root, config_path=self.config_path, mcp_root=self.mcp_root))
        from .inline_math_runtime import InlineMathSession
        inline = InlineMathSession(self.work_root, outputs_by_route,
                                   candidate_policy=self.inline_math_candidate_policy,
                                   atomic_superscripts=self.inline_atomic_superscripts) if self.inline_math else None
        from .fraction_repair_runtime import FractionRepairSession
        fractions = FractionRepairSession(self.work_root, formula_rows) if self.fraction_repair else None
        fraction_runner = fractions.wrap(provider_c_runner) if fractions is not None else provider_c_runner
        text_runner = inline.wrap(fraction_runner) if inline is not None else fraction_runner
        requests = figure_math_prefetch_requests(resolution_plans, crops_by_route, outputs_by_route)
        evidence_b = {}
        candidate_ids = {r['id'] for r in requests if r.get('kind') == 'symbol'}

        def run_b(provider_id, requests):
            outputs = list(provider_b_runner(provider_id, requests))
            for request, output in zip(requests, outputs, strict=True):
                if request.region_id in candidate_ids:
                    evidence_b[request.region_id] = _generative_evidence(provider_id, request, output)
            return outputs

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='figure-formula') as pool:
            pending = FigureMathPrefetch(requests, formula_factory, pool,
                select_requests=lambda values: select_figure_math_requests(values, evidence_b),
                caller_thread_prediction=formula_runtime is not None)
            staged = ProductionThreeModelLiveComposition(
                provider_a,
                CurrentRunRecognitionStageProvider("B_CH_SVTRV2_REC", run_b),
                CurrentRunGenerativeStageProvider("C_GOT_OCR2", pending.overlap_runner(text_runner)),
            ).resolve(plans=resolution_plans, crops_by_route=crops_by_route)
            prefetched = pending.result()
            if inline is not None:
                inline.flush(fraction_runner)
                inline.apply(staged.provenance_by_route, formula_factory)
                staged.runtime.provenance_by_route = copy.deepcopy(staged.provenance_by_route)
                staged.metrics['inline_math'] = copy.deepcopy(inline.metrics)
            if fractions is not None:
                fractions.flush(provider_c_runner)
                fractions.apply(formula_factory)
                staged.metrics['fraction_repair'] = copy.deepcopy(fractions.metrics)
            build_figure_label_results(staged, crops_by_route,
                                      formula_runtime_factory=formula_factory, prefetched=prefetched)
        staged.metrics['figure_formula_overlap'] = {k: v for k, v in prefetched.items() if k != 'outputs'}
        staged.metrics['figure_formula_overlap']['body_formula_runtime_reused'] = formula_runtime is not None
        staged.metrics["provider_a_runtime"] = copy.deepcopy(fingerprint)
        staged.metrics["provider_b_stages"] = copy.deepcopy(
            provider_b_runner.stage_metrics
        )
        staged.metrics["provider_c_stages"] = copy.deepcopy(
            provider_c_runner.stage_metrics
        )
        if inline is not None:
            staged.runtime._metrics = copy.deepcopy(staged.metrics)
        staged.runtime.source_scope_reviews = source_scope_reviews
        staged.metrics['source_scope_review_count'] = len(source_scope_reviews)
        staged.runtime._metrics['source_scope_review_count'] = len(source_scope_reviews)
        return staged


def create_production_pdf_runtime(
    *,
    models_root: str | Path | None = None,
    config_path: str | Path | None = None,
    mcp_root: str | Path | None = None,
    layout_runtime_factory: Callable[[], Any] | None = None,
    independent_source_pages: bool = False,
    text_runtime_resolver_factory: Callable[[Path], Any] | None = None,
    table_evidence_provider_factory: Callable[[Path], Any] | None = None,
    formula_runtime_factory: Callable[[], Any] | None = None,
) -> ProductionPdfRuntime:
    """Build the formal registry-backed PDF runtime without Developer paths."""

    def make_layout_runtime() -> Any:
        if layout_runtime_factory is not None:
            return layout_runtime_factory()
        return FormalPaddleLayoutRuntime(
            models_root=models_root,
            config_path=config_path,
            mcp_root=mcp_root,
            capture_floor=CAPTURE_FLOOR,
        )

    def make_pipeline(work_root: Path) -> ModularPdfPipeline:
        if text_runtime_resolver_factory is None:
            text_resolver = RegistryBackedThreeModelTextResolver(
                work_root,
                models_root=models_root,
                config_path=config_path,
                mcp_root=mcp_root,
                independent_source_pages=independent_source_pages,
            )
        else:
            text_resolver = text_runtime_resolver_factory(work_root)
        if table_evidence_provider_factory is not None:
            table_provider = table_evidence_provider_factory(work_root)
        else:
            from .table_runtime import RegistryBackedTableEvidenceProvider

            table_provider = RegistryBackedTableEvidenceProvider(
                work_root, models_root=models_root, config_path=config_path, mcp_root=mcp_root,
            )

        def make_formula_runtime() -> Any:
            if formula_runtime_factory is not None:
                return formula_runtime_factory()
            from ..formula_ocr import create_production_formulanet_runtime

            return create_production_formulanet_runtime(
                models_root=models_root,
                config_path=config_path,
                mcp_root=mcp_root,
                deep_model_validation=True,
            )

        return ModularPdfPipeline(
            work_root,
            text_runtime_resolver=text_resolver,
            formula_runtime_factory=make_formula_runtime,
            table_evidence_provider=table_provider,
            require_three_model_provenance=True,
        )

    return ProductionPdfRuntime(
        layout_runtime_factory=make_layout_runtime,
        pipeline_factory=make_pipeline,
    )


def _page_records(inspected: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": inspected["document_id"],
            "source_group": "PRODUCTION_RAW_PDF",
            "source_path": inspected["source_path"],
            "source_sha256": inspected["sha256"],
            "page_index": int(page["page_index"]),
            "page_number": int(page["page_number"]),
            **copy.deepcopy(page),
        }
        for page in inspected["pages"]
    ]


def _run_layout(
    runtime: Any,
    *,
    renders: Sequence[Any],
    page_records: Sequence[dict[str, Any]],
    evidence: Mapping[tuple[str, int], dict[str, Any]],
    document_id: str,
    layout_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from PIL import Image

    raw_pages: list[dict[str, Any]] = []
    page_region_ir: list[dict[str, Any]] = []
    record_by_index = {int(record["page_index"]): record for record in page_records}
    for render in renders:
        record = record_by_index[int(render.page_index)]
        with Image.open(BytesIO(render.image_bytes)) as opened:
            image = opened.convert("RGB")
            value = runtime.predict_image(
                image,
                document_id=document_id,
                page_index=render.page_index,
                page_render_identity=render.render_sha256,
            )
        detections, inference_seconds = _prediction_value(value)
        raw_page = {
            "schema": "bemarkdown-raw-layout-page-v0",
            "document_id": document_id,
            "page_index": int(render.page_index),
            "page_number": int(render.page_number),
            "model_identity": (
                f"{layout_identity.get('model_id', 'pp-doclayout-plus-l')}@"
                f"{layout_identity.get('model_fingerprint', 'unknown')}"
            ),
            "page_render_identity": render.render_sha256,
            "capture_floor": CAPTURE_FLOOR,
            "inference_seconds": inference_seconds,
            "raw_detections": detections,
        }
        key = document_id, int(render.page_index)
        page_ir = build_page_region_ir(
            document_id=document_id,
            page_index=render.page_index,
            page_route=str(record["routing_decision"]),
            transform=render.transform,
            raw_detections=detections,
            score_threshold=CANONICAL_LAYOUT_THRESHOLD,
            structural_geometry={
                "native_text_bboxes_pdf_pt": [
                    row["bbox_pdf_pt"] for row in evidence[key]["native_text"]
                ],
                "native_image_bboxes_pdf_pt": [
                    row["bbox_pdf_pt"] for row in evidence[key]["native_images"]
                ],
                "native_vector_bboxes_pdf_pt": [
                    row["bbox_pdf_pt"] for row in evidence[key]["native_vectors"]
                ],
            },
            assign_reading_order=True,
        )
        refine_embedded = getattr(runtime, 'refine_page_region_ir', None)
        if callable(refine_embedded):
            raw_page['embedded_image_semantic_refinement'] = refine_embedded(page_ir, image, render.transform)
        from .diagram_bounds import refine_diagram_bounds
        raw_page['diagram_boundary_refinement'] = refine_diagram_bounds(page_ir, image, render.transform)
        from .table_bounds import refine_table_bounds
        raw_page['table_boundary_refinement'] = refine_table_bounds(page_ir, image, render.transform)
        from .table_formula_review import retain_table_formula_review_hints
        raw_page['table_formula_review_hints'] = retain_table_formula_review_hints(
            page_ir, detections, render.transform, capture_floor=raw_page['capture_floor'])
        issues = validate_page_region_ir(page_ir)
        if issues:
            raise ValueError(
                f"PDF_PRODUCTION_PAGE_REGION_IR_INVALID:{key}:{issues[:3]}"
            )
        raw_pages.append(raw_page)
        page_region_ir.append(page_ir)
    return raw_pages, page_region_ir


def _prediction_value(value: Any) -> tuple[list[dict[str, Any]], float]:
    if isinstance(value, tuple) and len(value) == 2:
        rows, elapsed = value
        return [dict(row) for row in rows], float(elapsed)
    return [dict(row) for row in value], 0.0


def _write_pdf_payload(
    *,
    source: Path,
    output_dir: Path,
    inspected: Mapping[str, Any],
    pipeline_result: ModularPipelineResult,
    layout_identity: Mapping[str, Any],
    layout_unload: Mapping[str, Any],
    elapsed: float,
    debug: bool,
) -> dict[str, Any]:
    document = copy.deepcopy(pipeline_result.document)
    renderer = CleanHandoffRenderer()
    source_asset_map = {
        asset["asset_uid"]: document_asset_presentation_ref(asset)
        for asset in document.get("assets", [])
    }
    visible_assets = set(renderer.render(document, source_asset_map)["required_asset_uids"])
    asset_rows, _unresolved_refs = _materialize_assets(document, output_dir, required_uids=visible_assets)
    rendered = CleanHandoffRenderer().render(
        document,
        {asset["asset_uid"]: document_asset_presentation_ref(asset) for asset in document.get("assets", [])},
        unresolved_assets={row["asset_uid"]: row["asset_id"] for row in asset_rows if row["status"] == "UNRESOLVED"},
    )
    markdown = rendered["markdown"]
    (output_dir / "document.md").write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "assets_manifest.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in asset_rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    report = _pdf_report(
        source=source,
        inspected=inspected,
        document=document,
        pipeline_result=pipeline_result,
        layout_identity=layout_identity,
        layout_unload=layout_unload,
        asset_rows=asset_rows,
        elapsed=elapsed,
    )
    report["markdown_render"] = {key: value for key, value in rendered.items() if key != "markdown"}
    report["assets"]["non_content_source_assets"] = len(document.get("assets", [])) - len(asset_rows)
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir()
        from .figure_label_debug import archive_figure_label_inputs
        archive_figure_label_inputs(document, debug_dir)
        (debug_dir / "document_ir.json").write_text(
            json.dumps(pipeline_result.portable_document_ir, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
    return report


def _materialize_assets(
    document: dict[str, Any], output_dir: Path, *, required_uids: set[str] | None = None
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    ordered = _ordered_assets(document)
    if required_uids is not None:
        ordered = [asset for asset in ordered if asset["asset_uid"] in required_uids]
    rows: list[dict[str, Any]] = []
    unresolved_refs: list[tuple[str, str]] = []
    for index, asset in enumerate(ordered, 1):
        asset_id = f"image_{index:06d}"
        source_ref = document_asset_source_ref(asset)
        source_path = Path(str(source_ref)).resolve() if source_ref else None
        suffix = str(asset.get("media_suffix") or "").lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}" if suffix else ""
        if source_path is not None and source_path.suffix:
            suffix = source_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
            suffix = ".png"
        relative_path = None
        content_sha256 = None
        status = "UNRESOLVED"
        if source_path is not None and source_path.is_file():
            payload = source_path.read_bytes()
            content_sha256 = hashlib.sha256(payload).hexdigest()
            expected_sha = asset.get("content_sha256")
            if expected_sha and expected_sha != content_sha256:
                raise RuntimeError(f"PDF_ASSET_SOURCE_SHA_MISMATCH:{asset_id}")
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            target = assets_dir / f"{asset_id}{suffix}"
            target.write_bytes(payload)
            relative_path = f"assets/{target.name}"
            asset["relative_path"] = relative_path
            status = "RESOLVED"
        else:
            unresolved_refs.append((document_asset_presentation_ref(asset), asset_id))
        mime_type = (
            "image/svg+xml"
            if suffix == ".svg"
            else mimetypes.guess_type(f"asset{suffix}")[0]
            or "application/octet-stream"
        )
        rows.append(
            {
                "manifest_version": "bemarkdown-asset-contract-v1",
                "asset_id": asset_id,
                "asset_uid": str(asset["asset_uid"]),
                "content_sha256": content_sha256,
                "semantic_type": str(asset.get("role") or "PDF_VISUAL"),
                "status": status,
                "mime_type": mime_type,
                "extension": suffix,
                "relative_path": relative_path,
                "source_type": "PDF",
                "source_part": f"page:{int(asset.get('page_index', 0))}",
                "source_locator": str(asset.get("asset_uid")),
                "relationship_id": None,
                "external_target": None,
                "render_method": "PDF_NATIVE_OR_BOUNDED_CROP",
                "provenance": {
                    "bbox_pdf_pt": copy.deepcopy(asset.get("bbox_pdf_pt")),
                    "identity_basis": asset.get("provenance", {}).get(
                        "identity_basis"
                    ),
                },
            }
        )
    return rows, unresolved_refs


def _ordered_assets(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    assets = {
        str(asset["asset_uid"]): asset
        for asset in document.get("assets", [])
    }
    order: list[str] = []
    for block in document.get("blocks", []):
        asset_uid = str(block.get("content", {}).get("asset_uid") or "")
        if asset_uid in assets and asset_uid not in order:
            order.append(asset_uid)
    order.extend(sorted(set(assets) - set(order)))
    return [assets[asset_uid] for asset_uid in order]


def _pdf_report(
    *,
    source: Path,
    inspected: Mapping[str, Any],
    document: Mapping[str, Any],
    pipeline_result: ModularPipelineResult,
    layout_identity: Mapping[str, Any],
    layout_unload: Mapping[str, Any],
    asset_rows: Sequence[Mapping[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    blocks = list(document.get("blocks", []))
    review_blocks = [
        block
        for block in blocks
        if block.get("review_state") in {"DEFERRED", "REVIEW_REQUIRED"}
    ]
    warning_blocks = [
        block for block in blocks if block.get("review_state") == "WARNING"
    ]
    warnings = sorted(
        {
            str(value)
            for row in pipeline_result.region_content
            for value in row.get("warnings", [])
        }
    )
    unresolved_assets = sum(row["relative_path"] is None for row in asset_rows)
    review_item_count = len(review_blocks) + unresolved_assets
    if review_item_count:
        quality_status = "COMPLETED_WITH_REVIEW_ITEMS"
    elif warnings or warning_blocks:
        quality_status = "COMPLETED_WITH_WARNINGS"
    else:
        quality_status = "CLEAN"
    kind_counts = Counter(str(block.get("kind")) for block in blocks)
    metrics = copy.deepcopy(pipeline_result.metrics)
    if not metrics.get("exactly_one_primary_route") or metrics.get("silent_drop_count"):
        raise RuntimeError("PDF_PRODUCTION_ROUTE_CONSERVATION_FAILED")
    return {
        "schema": PDF_REPORT_SCHEMA,
        "source": {
            "type": "PDF",
            "path": str(source),
            "file_name": source.name,
            "size_bytes": int(inspected["bytes"]),
            "sha256": inspected["sha256"],
        },
        "input_validation": {
            "status": "PASS",
            "page_count": int(inspected["page_count"]),
            "backend": copy.deepcopy(inspected.get("backend", {})),
            "encrypted": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "vision_or_ocr_used": bool(layout_identity) or any(
                metrics.get(key, 0) for key in (
                    "ocr_invocation_count", "formula_invocation_count", "table_engine_invocation_count"
                )
            ),
            "layout": _portable_identity(layout_identity),
            "layout_unload": copy.deepcopy(dict(layout_unload)),
            "cpu_fallback": False,
        },
        "pdf_pipeline": {
            "capability_status": PDF_CAPABILITY_STATUS,
            "mode": metrics.get("mode"),
            "fusion_strategy": ADAPTIVE_LOCAL_GROUPING_STRATEGY["strategy_id"],
            "page_escalation_strategy": PAGE_ESCALATION_FUSION_STRATEGY[
                "strategy_id"
            ],
            "metrics": metrics,
        },
        "document": {
            "pages": len(document.get("pages", [])),
            "blocks": len(blocks),
            "block_kind_counts": dict(sorted(kind_counts.items())),
        },
        "formulas": {
            "total": kind_counts.get("FORMULA", 0),
            "conservation_ok": True,
            "semantic_auto_correction_enabled": False,
        },
        "formula_ocr": {
            "enabled": bool(metrics.get("formula_invocation_count")),
            "invocation_count": int(metrics.get("formula_invocation_count", 0)),
            "conservation_ok": True,
            "real_formula_conservation_ok": True,
        },
        "assets": {
            "manifest_version": "bemarkdown-asset-contract-v1",
            "total_visible": len(asset_rows),
            "resolved": len(asset_rows) - unresolved_assets,
            "unresolved": unresolved_assets,
            "records": [copy.deepcopy(dict(row)) for row in asset_rows],
        },
        "ownership": {
            "non_text_first": True,
            "unsafe_ownership_count": metrics.get("unsafe_ownership_count", 0),
            "native_excluded_span_count": metrics.get(
                "native_excluded_span_count", 0
            ),
        },
        "ordering": {
            "contract": (
                "PAGE_ASC_NATIVE_SOURCE_COLUMN_AND_INLINE_ORDER"
                if any(page.get("column_first_reorder_applied") for page in document.get("pages", []))
                else "PAGE_ASC_TOP_TO_BOTTOM_SAME_BAND_LEFT_TO_RIGHT"
            ),
            "semantic_column_reorder_enabled": False,
            "native_column_reorder_enabled": any(
                page.get("column_first_reorder_applied") for page in document.get("pages", [])
            ),
            "native_text_region_continuity": any(
                page.get("native_paragraph_continuity", {}).get("moved_atom_count", 0)
                for page in document.get("pages", [])
            ),
        },
        "review_item_count": review_item_count,
        "quality_status": quality_status,
        "warnings": warnings,
        "errors": [],
        "timing": {
            "total_seconds": round(float(elapsed), 6),
            "scope": "SOURCE_INSPECTION_RENDER_AND_INFERENCE_EXCLUDES_PACKAGE_PUBLICATION",
        },
        "independent_quality_validation": PDF_QUALITY_VALIDATION_STATUS,
        "known_limitations": [
            "HANDWRITING_UNSUPPORTED",
            "FORMULA_SEMANTIC_AUTO_CORRECTION_DISABLED",
            "PDF_INDEPENDENT_VALIDATION_PENDING_ZCODE",
        ],
    }


def _portable_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"model_root", "manifest_path"}
    }
