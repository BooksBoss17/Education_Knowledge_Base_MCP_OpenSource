"""Production-facing modular PDF seam assembled from the Phase 6B authorities."""

from __future__ import annotations

import copy
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ..pdf_content_router import PdfContentAdapterExecutor, plan_page_content
from ..pdf_document_ir import (
    assemble_document_ir,
    render_draft_markdown,
    validate_document_ir,
)
from ..pdf_layout_fusion import build_page_fusion_ir, validate_page_fusion_ir
from ..pdf_output_audit import LeanHandoffPackager
from ..pdf_source import PdfSourceInspector, apply_routing_baseline
from ..pdf_table_engine import TableEngine, apply_table_results
from ..production_gpu_closeout import materialize_missing_asset_refs
from .pipeline_mode import (
    PDFPipelineMode,
    modular_primary_residency_plan,
    resolve_pdf_pipeline_mode,
)

TableEvidenceProvider = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ModularPipelineResult:
    document: dict[str, Any]
    portable_document_ir: dict[str, Any]
    draft_markdown: str
    fusion_pages: list[dict[str, Any]]
    route_plans: list[dict[str, Any]]
    region_content: list[dict[str, Any]]
    page_content: list[dict[str, Any]]
    metrics: dict[str, Any]


class ModularPdfPipeline:
    """Coordinate specialist routes without importing or loading Phase 7 models."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        mode: str | PDFPipelineMode | None = None,
        cropper: Any | None = None,
        ocr_runtime_factory: Callable[[], Any] | None = None,
        text_runtime_resolver: Any | None = None,
        formula_runtime_factory: Callable[[], Any] | None = None,
        table_evidence_provider: TableEvidenceProvider | None = None,
        table_engine: TableEngine | None = None,
        require_three_model_provenance: bool = True,
    ) -> None:
        selected = resolve_pdf_pipeline_mode(mode)
        if selected is not PDFPipelineMode.MODULAR_PRIMARY:
            raise ValueError(
                f"MODULAR_PIPELINE_REQUIRES_MODULAR_PRIMARY:{selected.value}"
            )
        self.mode = selected
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cropper = cropper
        self.ocr_runtime_factory = ocr_runtime_factory
        self.text_runtime_resolver = text_runtime_resolver
        self.formula_runtime_factory = formula_runtime_factory
        self.table_evidence_provider = table_evidence_provider
        self.table_engine = table_engine or TableEngine()
        self.require_three_model_provenance = require_three_model_provenance

    @staticmethod
    def inspect_source(source: str | Path) -> dict[str, Any]:
        return apply_routing_baseline(PdfSourceInspector().inspect(source))

    def run_from_page_ir(
        self,
        *,
        page_region_ir: Sequence[dict[str, Any]],
        raw_layout_pages: Sequence[dict[str, Any]],
        page_records: Sequence[dict[str, Any]],
        source_evidence: Mapping[tuple[str, int], dict[str, Any]],
        fusion_strategy: dict[str, Any],
        page_escalation_strategy: dict[str, Any] | None = None,
        source: dict[str, Any],
    ) -> ModularPipelineResult:
        raw_by_key = _by_page(raw_layout_pages)
        record_by_key = _by_page(page_records)
        fusion_pages = []
        for page_ir in sorted(page_region_ir, key=_page_sort_key):
            key = _page_key(page_ir)
            strategy = fusion_strategy
            if page_escalation_strategy is not None:
                control = build_page_fusion_ir(
                    page_ir=page_ir,
                    raw_page=raw_by_key[key],
                    page_record=record_by_key[key],
                    source_evidence=source_evidence[key],
                    strategy=page_escalation_strategy,
                )
                if control["page_escalation"]["status"] != "NONE":
                    strategy = page_escalation_strategy
            fusion_page = build_page_fusion_ir(
                page_ir=page_ir,
                raw_page=raw_by_key[key],
                page_record=record_by_key[key],
                source_evidence=source_evidence[key],
                strategy=strategy,
            )
            issues = validate_page_fusion_ir(fusion_page)
            if issues:
                raise ValueError(f"MODULAR_PIPELINE_FUSION_INVALID:{key}:{issues[:3]}")
            fusion_pages.append(fusion_page)
        return self.run_from_fusion(
            fusion_pages=fusion_pages,
            page_records=page_records,
            source_evidence=source_evidence,
            source=source,
        )

    def run_from_fusion(
        self,
        *,
        fusion_pages: Sequence[dict[str, Any]],
        page_records: Sequence[dict[str, Any]],
        source_evidence: Mapping[tuple[str, int], dict[str, Any]],
        source: dict[str, Any],
    ) -> ModularPipelineResult:
        records = _by_page(page_records)
        pages = sorted((copy.deepcopy(row) for row in fusion_pages), key=_page_sort_key)
        if not pages:
            raise ValueError("MODULAR_PIPELINE_REQUIRES_AT_LEAST_ONE_PAGE")
        document_ids = {str(row["document_id"]) for row in pages}
        if len(document_ids) != 1:
            raise ValueError("MODULAR_PIPELINE_RUN_IS_ONE_DOCUMENT")

        plans = [
            plan_page_content(
                page,
                records[_page_key(page)],
                source_evidence[_page_key(page)],
                table_adapter=(
                    "TABLE_ENGINE"
                    if self.table_evidence_provider is not None
                    else "TABLE_DEFERRED_PRESERVE"
                ),
            )
            for page in pages
        ]
        executor = PdfContentAdapterExecutor(
            self.output_dir / "content",
            cropper=self.cropper,
            ocr_runtime_factory=self.ocr_runtime_factory,
            formula_runtime_factory=self.formula_runtime_factory,
            require_three_model_provenance=self.require_three_model_provenance,
        )
        region_rows: list[dict[str, Any]] = []
        page_rows: list[dict[str, Any]] = []
        effective_source_evidence = {
            key: copy.deepcopy(dict(value)) for key, value in source_evidence.items()
        }
        ownership_regions: list[dict[str, Any]] = []
        ownership_unsafe: list[dict[str, Any]] = []
        execution_stage_trace: list[dict[str, Any]] = []
        staged_text_runtime: Any | None = None
        text_resolution_seconds = 0.0
        try:
            executor.prepare_formula_batch(plans)
            staged_pages = []
            for plan in plans:
                key = _page_key(plan)
                state = executor.prepare_non_text_page(plan, source_evidence[key])
                prepared = executor.prepare_text_last_inputs(state)
                if self.text_runtime_resolver is not None:
                    from .figure_labels import prepare_figure_label_inputs
                    prepared.update(prepare_figure_label_inputs(executor, state))
                staged_pages.append(
                    (plan, key, state, prepared)
                )
            prepared_by_route = {
                route_id: crop
                for _plan, _key, _state, prepared in staged_pages
                for route_id, crop in prepared.items()
            }
            runtime_prepared_by_route = {
                route_id: crop
                for route_id, crop in prepared_by_route.items()
                if crop.get("text_runtime_resolution_required", True)
            }
            runtime_route_ids = set(runtime_prepared_by_route)
            runtime_plans = []
            for plan, _key, state, _prepared in staged_pages:
                runtime_routes = [
                    copy.deepcopy(route)
                    for route in list(plan["routes"]) + state.get('figure_label_routes', [])
                    if str(route["route_id"]) in runtime_route_ids
                ]
                if runtime_routes:
                    runtime_plan = copy.deepcopy(plan)
                    runtime_plan["routes"] = runtime_routes
                    runtime_plans.append(runtime_plan)
            if self.text_runtime_resolver is not None and runtime_prepared_by_route:
                resolve = getattr(
                    self.text_runtime_resolver,
                    "resolve",
                    self.text_runtime_resolver,
                )
                text_resolution_started = time.perf_counter()
                shared_resolve = getattr(self.text_runtime_resolver, 'resolve_with_formula_runtime', None)
                context_resolve = getattr(self.text_runtime_resolver,
                                          'resolve_with_formula_context', None)
                if callable(context_resolve):
                    formula_rows = [row for _plan, _key, state, _prepared in staged_pages
                                    for row in state['rows_by_route'].values()]
                    staged_text_runtime = context_resolve(plans=runtime_plans,
                        crops_by_route=runtime_prepared_by_route,
                        formula_runtime=executor.prepared_formula_runtime,
                        formula_rows=formula_rows)
                elif callable(shared_resolve):
                    staged_text_runtime = shared_resolve(plans=runtime_plans,
                        crops_by_route=runtime_prepared_by_route,
                        formula_runtime=executor.prepared_formula_runtime)
                else:
                    staged_text_runtime = resolve(
                        plans=runtime_plans,
                        crops_by_route=runtime_prepared_by_route,
                    )
                text_resolution_seconds = time.perf_counter() - text_resolution_started
                resolved_runtime = getattr(
                    staged_text_runtime,
                    "runtime",
                    staged_text_runtime,
                )
                executor.ocr_runtime_factory = lambda: resolved_runtime
                scope_reviews = getattr(resolved_runtime, 'source_scope_reviews', {})
                for route_id, review in scope_reviews.items():
                    prepared_by_route[route_id]['source_scope_review'] = copy.deepcopy(review)
            for plan, key, state, prepared_inputs in staged_pages:
                if state.get('figure_label_routes'):
                    from .figure_labels import attach_figure_label_results
                    attach_figure_label_results(state, getattr(staged_text_runtime, 'runtime', None))
                result = executor.finish_text_page(
                    state,
                    prepared_inputs=prepared_inputs,
                )
                region_rows.extend(result["region_content"])
                page_rows.append(result["page_content"])
                effective_source_evidence[key] = result["effective_source_evidence"]
                ownership_regions.extend(result["non_text_ownership_regions"])
                ownership_unsafe.extend(result["ownership_unsafe_inventory"])
                execution_stage_trace.append(
                    {
                        "document_id": key[0],
                        "page_index": key[1],
                        "events": result["execution_stage_trace"],
                    }
                )
        finally:
            executor.close()

        routes_by_id = {
            str(route["route_id"]): route for plan in plans for route in plan["routes"]
        }
        try:
            region_rows, asset_recovery = materialize_missing_asset_refs(
                region_rows,
                routes_by_id=routes_by_id,
                render_crop=lambda route: executor.cropper.render(route, full_page=False),
            )
        finally:
            # Review-asset recovery can reopen the PDF after model execution closes.
            executor.close()
        document_id = next(iter(document_ids))
        document = assemble_document_ir(
            document_id=document_id,
            source=copy.deepcopy(source),
            page_records=[
                records[key] for key in sorted(records) if key[0] == document_id
            ],
            region_content=region_rows,
            source_evidence=effective_source_evidence,
            development_audit={
                "pipeline_mode": self.mode.value,
                "paddleocr_vl_authority": "NOT_PRODUCTION_AUTHORITY",
                "vision_agent_enabled": False,
            },
        )
        table_invocations = self._apply_tables(
            document, source_evidence=effective_source_evidence, page_records=records
        )
        validate_document_ir(document)
        draft = render_draft_markdown(document)
        route_counts = Counter(
            route["adapter"] for plan in plans for route in plan["routes"]
        )
        coverage = [plan["coverage"] for plan in plans]
        residency = modular_primary_residency_plan()
        metrics = {
            "schema": "bemarkdown-modular-primary-route-metrics-v1",
            "mode": self.mode.value,
            "page_count": len(pages),
            "route_counts": dict(sorted(route_counts.items())),
            "ocr_invocation_count": route_counts["OCR_TEXT_REGION"]
            + route_counts["PAGE_VISUAL_TEXT_RECOVERY"],
            "formula_invocation_count": route_counts["FORMULA_RECOGNITION"],
            "table_engine_invocation_count": table_invocations,
            "paddleocr_vl_load_count": residency["paddleocr_vl_load_count"],
            "pp_doclayout_v3_load_count": residency["pp_doclayout_v3_load_count"],
            "vl_runtime_required": residency["vl_runtime_required"],
            "model_load_count": copy.deepcopy(executor.stats["model_load_count"]),
            "model_unload_count": copy.deepcopy(executor.stats["model_unload_count"]),
            "adapter_seconds": copy.deepcopy(executor.stats["adapter_seconds"]),
            "text_resolution_seconds": round(text_resolution_seconds, 6),
            "formula_batch_execution": copy.deepcopy(executor.stats.get("formula_batch_execution", {})),
            "formula_model_lifecycle": copy.deepcopy(executor.stats.get("formula_model_lifecycle", {})),
            "native_exclusion_seconds": executor.stats.get(
                "native_exclusion_seconds", 0.0
            ),
            "exactly_one_primary_route": all(
                row["unrouted"] == 0 and not row["duplicate_primary_candidate_ids"]
                for row in coverage
            ),
            "silent_drop_count": sum(row["unrouted"] for row in coverage),
            "asset_materialization": asset_recovery,
            "non_text_ownership_region_count": len(ownership_regions),
            "masked_page_count": len(
                {
                    (str(row["document_id"]), int(row["page_index"]))
                    for row in ownership_regions
                }
            ),
            "unsafe_ownership_count": len(ownership_unsafe),
            "native_excluded_span_count": sum(
                int(row.get("native_excluded_span_count") or 0) for row in page_rows
            ),
            "partial_native_overlap_count": sum(
                int(row.get("partial_native_overlap_count") or 0) for row in page_rows
            ),
            "ownership_regions": ownership_regions,
            "ownership_unsafe_inventory": ownership_unsafe,
            "execution_stage_trace": execution_stage_trace,
            "text_runtime": {
                "staged": staged_text_runtime is not None,
                "prepared_route_count": len(prepared_by_route),
                "resolution_route_count": len(runtime_prepared_by_route),
                "masked_empty_route_count": len(prepared_by_route)
                - len(runtime_prepared_by_route),
                "metrics": copy.deepcopy(getattr(staged_text_runtime, "metrics", {})),
                "agent_candidate_count": len(
                    getattr(staged_text_runtime, "agent_candidates", ())
                ),
            },
        }
        return ModularPipelineResult(
            document=document,
            portable_document_ir=portable_document_ir(document),
            draft_markdown=draft,
            fusion_pages=pages,
            route_plans=plans,
            region_content=region_rows,
            page_content=page_rows,
            metrics=metrics,
        )

    def write_handoff(
        self,
        document: dict[str, Any],
        source_package: str | Path,
        target: str | Path,
        *,
        audit_status: str,
    ) -> dict[str, Any]:
        return LeanHandoffPackager().write(
            document,
            source_package,
            target,
            audit_status=audit_status,
        )

    def _apply_tables(self, document: dict[str, Any], *, source_evidence=None, page_records=None) -> int:
        table_blocks = [
            block for block in document["blocks"] if block["kind"] == "TABLE"
        ]
        if not table_blocks:
            return 0
        if self.table_evidence_provider is None:
            return 0
        assets = {
            str(asset["asset_uid"]): asset for asset in document.get("assets", [])
        }
        results = {}
        try:
            for block in table_blocks:
                asset_uid = str(block["content"]["asset_uid"])
                evidence = self.table_evidence_provider(
                    {**block, "document_id": document["document_id"]}, assets[asset_uid]
                )
                key = (document["document_id"], int(block["page_index"]))
                page_source = (source_evidence or {}).get(key, {})
                record = (page_records or {}).get(key, {})
                evidence = {**evidence,
                            "native_math_geometry": page_source.get("native_math_geometry", []),
                            "native_text_trust": record.get("native_text_trust"),
                            "source_profile": record.get("source_profile"),
                            "table_content_ownership": copy.deepcopy(
                                block.get("provenance", {}).get("route_provenance", {}).get(
                                    "table_content_ownership", {}
                                )
                            )}
                results[block["node_id"]] = self.table_engine.build(evidence)
        finally:
            close = getattr(self.table_evidence_provider, "close", None)
            if callable(close):
                close()
        apply_table_results(document, results)
        metrics = getattr(self.table_evidence_provider, "metrics", None)
        if callable(metrics):
            document.setdefault("provenance", {})["table_runtime"] = metrics()
        return len(results)


def portable_document_ir(document: dict[str, Any]) -> dict[str, Any]:
    """Return a path-independent DocumentIR projection for interchange/audit."""

    projected = _strip_local_paths(copy.deepcopy(document))
    projected.setdefault("provenance", {})["local_paths_removed"] = True
    return projected


def _strip_local_paths(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"path", "source_path", "source_ref", "source_pdf"}:
                continue
            if isinstance(item, str) and _is_absolute_path(item):
                continue
            result[key] = _strip_local_paths(item)
        return result
    if isinstance(value, list):
        return [_strip_local_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_local_paths(item) for item in value]
    return value


def _is_absolute_path(value: str) -> bool:
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _page_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["document_id"]), int(row["page_index"])


def _page_sort_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return _page_key(row)


def _by_page(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {_page_key(row): copy.deepcopy(row) for row in rows}
    if len(result) != len(rows):
        raise ValueError("DUPLICATE_DOCUMENT_PAGE_IDENTITY")
    return result
