from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .pdf.non_text_ownership import (
    NonTextOwnershipRegion,
    bbox_fully_owned,
    build_non_text_ownership,
    derive_text_only_render,
    exclude_native_text,
)
from .pdf_content_quality import (
    CONTENT_QUALITY_POLICY_VERSION,
    apply_conflict_metadata,
    build_content_conflict_graph,
    classify_ocr_output,
    classify_review_reasons,
)
from .pdf_formula_consensus import (
    FORMULA_CONSENSUS_VERIFICATION_SCHEMA,
    FORMULA_CONSENSUS_VERIFIER_VERSION,
    apply_formula_consensus_gate,
)
from .pdf_formula_visual import (
    FORMULA_VISUAL_VERIFICATION_SCHEMA,
    FORMULA_VISUAL_VERIFIER_VERSION,
)
from .pdf_ocr_recovery import OCR_REGION_RECOVERY_VERSION, OcrRegionRecoveryPolicy

CONTENT_ROUTE_PLAN_SCHEMA = "bemarkdown-content-route-plan-v0"
REGION_CONTENT_IR_SCHEMA = "bemarkdown-region-content-ir-v0"
PAGE_CONTENT_IR_SCHEMA = "bemarkdown-page-content-ir-v0"
ROUTER_VERSION = "core-content-router-v0.1"
TEXT_SEMANTICS = {
    "TEXT",
    "TITLE",
    "CAPTION",
    "HEADER_FOOTER",
    "PAGE_NUMBER",
    "TEXT_LIKE",
}
ROUTE_PRIORITY = {
    "FORMULA_RECOGNITION": 10,
    "TABLE_ENGINE": 19,
    "TABLE_DEFERRED_PRESERVE": 20,
    "IMAGE_RENDER_CROP": 30,
    "IMAGE_NATIVE_EXTRACT": 31,
    "NATIVE_TEXT_BRIDGE": 40,
    "OCR_TEXT_REGION": 41,
    "PAGE_VISUAL_TEXT_RECOVERY": 50,
    "CONTENT_REVIEW_REQUIRED": 60,
}
NON_TEXT_ADAPTERS = {
    "FORMULA_RECOGNITION",
    "TABLE_ENGINE",
    "TABLE_DEFERRED_PRESERVE",
    "IMAGE_RENDER_CROP",
    "IMAGE_NATIVE_EXTRACT",
}
TEXT_ADAPTERS = {"NATIVE_TEXT_BRIDGE", "OCR_TEXT_REGION", "PAGE_VISUAL_TEXT_RECOVERY"}
NATIVE_NON_TEXT_COVERAGE_THRESHOLD = 0.8


class ProductionThreeModelInvariantError(RuntimeError):
    """A production ordinary-text OCR route escaped the frozen A/B/C contract."""


class PdfContentAdapterExecutor:
    """Execute a validated route plan while preserving every failed input."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        cropper: Any | None = None,
        ocr_runtime_factory: Callable[[], Any] | None = None,
        formula_runtime_factory: Callable[[], Any] | None = None,
        formula_visual_verifier: Any | None = None,
        formula_consensus_verifier: Any | None = None,
        ocr_recovery_policy: OcrRegionRecoveryPolicy | None = None,
        model_pool: Any | None = None,
        require_three_model_provenance: bool = False,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cropper = cropper or PdfCropper(self.output_dir / "temp_content_assets")
        self.ocr_runtime_factory = (
            ocr_runtime_factory or create_production_pdf_ocr_runtime
        )
        self.formula_runtime_factory = (
            formula_runtime_factory or _formula_runtime_factory
        )
        self.formula_visual_verifier = formula_visual_verifier
        self.formula_consensus_verifier = formula_consensus_verifier
        self.ocr_recovery_policy = ocr_recovery_policy or OcrRegionRecoveryPolicy()
        self.model_pool = model_pool
        self.require_three_model_provenance = require_three_model_provenance
        self._ocr_runtime = None
        self._formula_runtime = None
        self._prepared_formula_inputs: dict[str, dict[str, Any]] = {}
        self._formula_predictions: dict[str, str] = {}
        self._active_ownership_regions: tuple[NonTextOwnershipRegion, ...] = ()
        self.stats = {
            "model_load_count": {"ocr": 0, "formula": 0},
            "model_unload_count": {"ocr": 0, "formula": 0},
            "routes_executed": {},
            "adapter_seconds": {},
        }

    def _prepare_formula_input(self, route):
        # Complete mostly owned glyphs before prediction and text masking.
        # Barely intersecting neighbors remain outside; a missing-operand
        # prediction can still request the existing broader, verified retry.
        complete = getattr(self.cropper, 'complete_inline_formula_bounds', None)
        if callable(complete):
            complete(route, require_core_majority=True)
        box = route['provenance']['bbox_pdf_pt']
        return self.cropper.render(route, full_page=False, dpi=400 if box[3] - box[1] < 20 else None)

    def prepare_formula_batch(self, plans: Sequence[dict[str, Any]]) -> None:
        """Recognize independent formula crops in existing safe size buckets."""
        from .production_runtime import FormulaBatchPlanner

        started = time.perf_counter()
        entries = []
        preparation_errors = []
        for plan in plans:
            for route in plan["routes"]:
                if route["adapter"] != "FORMULA_RECOGNITION":
                    continue
                native = route["provenance"].get("native_math_geometry", {})
                if native.get("native_latex", native.get("native_flat_latex")):
                    continue
                try:
                    crop = self._prepare_formula_input(route)
                    self._prepared_formula_inputs[route["route_id"]] = crop
                    entries.append({"id": route["route_id"], "path": crop["path"],
                                    "width": crop["width"], "height": crop["height"]})
                except Exception as exc:
                    preparation_errors.append({"route_id": route["route_id"], "error": type(exc).__name__})
        groups = []
        runtime = None
        if entries:
            try:
                runtime = self._ensure_formula()
            except Exception as exc:
                preparation_errors.append({"route_id": None, "error": type(exc).__name__})
        if runtime is not None:
            for group in FormulaBatchPlanner().plan(entries):
                items = group["items"]
                batch_size = min(group["batch_size"], len(items))
                record = {"bucket": group["bucket"], "batch_size": batch_size, "count": len(items)}
                try:
                    outputs = list(runtime.predict([Path(item["path"]) for item in items], batch_size=batch_size))
                    if len(outputs) != len(items) or any(not isinstance(value, str) for value in outputs):
                        raise ValueError("FORMULA_BATCH_OUTPUT_CARDINALITY_OR_TYPE")
                    self._formula_predictions.update((item["id"], value) for item, value in zip(items, outputs, strict=True))
                    record["status"] = "PASS"
                except Exception as exc:
                    # Normal scalar execution still preserves and reports every input.
                    record.update(status="SCALAR_RETRY", error=type(exc).__name__)
                groups.append(record)
        elapsed = time.perf_counter()-started
        self.stats["adapter_seconds"]["FORMULA_RECOGNITION"] = self.stats["adapter_seconds"].get("FORMULA_RECOGNITION", 0.0) + elapsed
        self.stats["formula_batch_execution"] = {"prepared": len(entries), "prefetched": len(self._formula_predictions),
                                                  "groups": groups, "preparation_errors": preparation_errors}
        release_workspace = getattr(runtime, "release_workspace", None)
        if callable(release_workspace):
            self.stats["formula_batch_execution"]["workspace_release"] = release_workspace()

    def execute(
        self, plan: dict[str, Any], source_evidence: dict[str, Any]
    ) -> dict[str, Any]:
        state = self.prepare_non_text_page(plan, source_evidence)
        return self.finish_text_page(state)

    def prepare_non_text_page(
        self, plan: dict[str, Any], source_evidence: dict[str, Any]
    ) -> dict[str, Any]:
        """Materialize page non-text routes and freeze ownership before text work."""

        validate_content_route_plan(plan)
        routes = list(plan["routes"])
        non_text_routes = sorted(
            (route for route in routes if route["adapter"] in NON_TEXT_ADAPTERS),
            key=lambda route: (ROUTE_PRIORITY[route["adapter"]], route["route_id"]),
        )
        text_routes = sorted(
            (route for route in routes if route["adapter"] in TEXT_ADAPTERS),
            key=lambda route: (ROUTE_PRIORITY[route["adapter"]], route["route_id"]),
        )
        other_routes = [
            route
            for route in routes
            if route["adapter"] not in NON_TEXT_ADAPTERS | TEXT_ADAPTERS
        ]
        rows_by_route: dict[str, dict[str, Any]] = {}
        stage_trace = []
        for route in non_text_routes:
            rows_by_route[str(route["route_id"])] = self._execute_route(
                route, source_evidence
            )
        stage_trace.append(
            {
                "sequence": 0,
                "stage": "NON_TEXT_FIRST",
                "route_ids": [str(route["route_id"]) for route in non_text_routes],
            }
        )
        ownership_started = time.perf_counter()
        ownership = build_non_text_ownership(rows_by_route.values())
        self._active_ownership_regions = ownership.regions
        self.stats["non_text_ownership_build_seconds"] = round(
            self.stats.get("non_text_ownership_build_seconds", 0.0)
            + time.perf_counter()
            - ownership_started,
            6,
        )
        native_exclusion_started = time.perf_counter()
        native_exclusion = exclude_native_text(
            source_evidence.get("native_text", []),
            ownership.regions,
            coverage_threshold=NATIVE_NON_TEXT_COVERAGE_THRESHOLD,
        )
        effective_source_evidence = copy.deepcopy(source_evidence)
        effective_source_evidence["native_text"] = list(native_exclusion.rows)
        retained_line_ids = {row["evidence_id"] for row in native_exclusion.rows}
        effective_source_evidence["fully_excluded_native_lines"] = [
            {
                "source_line": copy.deepcopy(line),
                "ownership": [
                    trace
                    for trace in native_exclusion.excluded_spans
                    if trace["source_line_id"] == line["evidence_id"]
                ],
            }
            for line in source_evidence.get("native_text", [])
            if line["evidence_id"] not in retained_line_ids
        ]
        self.stats["native_exclusion_seconds"] = round(
            self.stats.get("native_exclusion_seconds", 0.0)
            + time.perf_counter()
            - native_exclusion_started,
            6,
        )
        self.stats["native_excluded_span_count"] = len(native_exclusion.excluded_spans)
        self.stats["partial_native_overlap_count"] = len(
            native_exclusion.partial_overlaps
        )
        stage_trace.append(
            {
                "sequence": 1,
                "stage": "OWNERSHIP_MAP_FINALIZED",
                "ownership_region_count": len(ownership.regions),
                "unsafe_ownership_count": len(ownership.unsafe),
            }
        )
        return {
            "plan": plan,
            "routes": routes,
            "non_text_routes": non_text_routes,
            "text_routes": text_routes,
            "other_routes": other_routes,
            "rows_by_route": rows_by_route,
            "ownership": ownership,
            "native_exclusion": native_exclusion,
            "effective_source_evidence": effective_source_evidence,
            "source_evidence": source_evidence,
            "execution_stage_trace": stage_trace,
            "ownership_map_finalized": True,
        }

    def prepare_text_last_inputs(
        self, state: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Render every raster text route from its finalized ownership map."""

        if not state.get("ownership_map_finalized"):
            raise RuntimeError("TEXT_INPUT_BEFORE_OWNERSHIP_MAP_FINALIZED")
        ownership = state["ownership"]
        self._active_ownership_regions = ownership.regions
        prepared: dict[str, dict[str, Any]] = {}
        for route in state["text_routes"]:
            if route["adapter"] not in {"OCR_TEXT_REGION", "PAGE_VISUAL_TEXT_RECOVERY"}:
                continue
            rendered = self._render_text_input(
                route,
                full_page=route["adapter"] == "PAGE_VISUAL_TEXT_RECOVERY",
            )
            fully_owned = bbox_fully_owned(
                route["provenance"].get("bbox_pdf_pt") or rendered["bbox_pdf_pt"], ownership.regions
            )
            prepared[str(route["route_id"])] = {
                **rendered,
                "masked_empty_text_route": fully_owned or bool(rendered.get('post_mask_all_white')),
                "text_runtime_resolution_required": not (fully_owned or rendered.get('post_mask_all_white')),
            }
        return prepared

    def finish_text_page(
        self,
        state: dict[str, Any],
        *,
        prepared_inputs: Mapping[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Execute Text LAST using inputs derived after page ownership finalized."""

        if not state.get("ownership_map_finalized"):
            raise RuntimeError("TEXT_EXECUTION_BEFORE_OWNERSHIP_MAP_FINALIZED")
        plan = state["plan"]
        routes = state["routes"]
        text_routes = state["text_routes"]
        other_routes = state["other_routes"]
        source_evidence = state["source_evidence"]
        rows_by_route = state["rows_by_route"]
        ownership = state["ownership"]
        native_exclusion = state["native_exclusion"]
        effective_source_evidence = state["effective_source_evidence"]
        stage_trace = list(state["execution_stage_trace"])
        self._active_ownership_regions = ownership.regions
        prepared_inputs = prepared_inputs or {}
        for route in text_routes:
            rows_by_route[str(route["route_id"])] = self._execute_route(
                route,
                source_evidence,
                prepared_input=prepared_inputs.get(str(route["route_id"])),
            )
        stage_trace.append(
            {
                "sequence": 2,
                "stage": "TEXT_LAST",
                "route_ids": [str(route["route_id"]) for route in text_routes],
            }
        )
        for route in other_routes:
            rows_by_route[str(route["route_id"])] = self._execute_route(
                route, source_evidence
            )
        rows = [rows_by_route[str(route["route_id"])] for route in routes]
        conflict_graph = build_content_conflict_graph(rows)
        rows = apply_conflict_metadata(rows, conflict_graph)
        page_content = {
            "schema": PAGE_CONTENT_IR_SCHEMA,
            "document_id": plan["document_id"],
            "page_index": plan["page_index"],
            "page_escalation": plan["page_escalation"],
            "route_ids": [route["route_id"] for route in plan["routes"]],
            "content_ids": [row["content_id"] for row in rows],
            "adapter_diagnostics": json.loads(json.dumps(self.stats)),
            "unresolved_inputs": [
                row["route_id"]
                for row in rows
                if row["status"]
                in {"REVIEW_REQUIRED", "FAILED_PRESERVE_INPUT", "DEFERRED"}
            ],
            "review_flags": sorted(
                {reason for row in rows for reason in row.get("review_reasons", [])}
            ),
            "quality_policy_version": CONTENT_QUALITY_POLICY_VERSION,
            "conflict_graph_schema": conflict_graph["schema"],
            "conflict_ids": [edge["conflict_id"] for edge in conflict_graph["edges"]],
            "conflict_graph_sha256": conflict_graph["semantic_sha256"],
            "ownership_map_finalized": True,
            "ownership_region_count": len(ownership.regions),
            "unsafe_ownership_count": len(ownership.unsafe),
            "native_excluded_span_count": len(native_exclusion.excluded_spans),
            "partial_native_overlap_count": len(native_exclusion.partial_overlaps),
        }
        return {
            "region_content": rows,
            "page_content": page_content,
            "content_conflict_graph": conflict_graph,
            "performance": json.loads(json.dumps(self.stats)),
            "non_text_ownership_regions": [
                region.to_dict() for region in ownership.regions
            ],
            "ownership_unsafe_inventory": list(ownership.unsafe),
            "execution_stage_trace": stage_trace,
            "effective_source_evidence": effective_source_evidence,
        }

    def _execute_route(
        self,
        route: dict[str, Any],
        source_evidence: dict[str, Any],
        *,
        prepared_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        adapter = route["adapter"]
        self.stats["routes_executed"][adapter] = (
            self.stats["routes_executed"].get(adapter, 0) + 1
        )
        preserved_input = None
        try:
            if adapter == "NATIVE_TEXT_BRIDGE":
                row = self._native_text(
                    route, source_evidence, self._active_ownership_regions
                )
            elif adapter in {"OCR_TEXT_REGION", "PAGE_VISUAL_TEXT_RECOVERY"}:
                preserved_input = prepared_input or self._render_text_input(
                    route,
                    full_page=adapter == "PAGE_VISUAL_TEXT_RECOVERY",
                )
                if preserved_input.get('source_scope_review'):
                    row = self._source_scope_review(route, preserved_input)
                elif bbox_fully_owned(
                    route["provenance"].get("bbox_pdf_pt") or preserved_input["bbox_pdf_pt"],
                    self._active_ownership_regions,
                ) or preserved_input.get('post_mask_all_white'):
                    row = self._masked_empty_text(route, preserved_input)
                else:
                    row = self._ocr(route, preserved_input)
            elif adapter == "FORMULA_RECOGNITION":
                preserved_input = self._prepared_formula_inputs.get(route["route_id"]) or self._prepare_formula_input(route)
                row = self._formula(route, preserved_input)
            elif adapter == "IMAGE_NATIVE_EXTRACT":
                row = self._native_image(route)
            elif adapter == "IMAGE_RENDER_CROP":
                preserved_input = self.cropper.render(route, full_page=False)
                row = self._image_crop(route, preserved_input)
            elif adapter in {"TABLE_ENGINE", "TABLE_DEFERRED_PRESERVE"}:
                preserved_input = self.cropper.render(route, full_page=False)
                row = self._table_deferred(route, preserved_input)
            elif adapter == "CONTENT_REVIEW_REQUIRED":
                row = self._review(route)
            else:
                raise ValueError(f"Unsupported content adapter: {adapter}")
        except ProductionThreeModelInvariantError:
            raise
        except Exception as exc:  # noqa: BLE001 - the input must survive every adapter failure
            row = self._failed(route, exc, preserved_input)
        elapsed = time.perf_counter() - started
        self.stats["adapter_seconds"][adapter] = round(
            self.stats["adapter_seconds"].get(adapter, 0.0) + elapsed, 6
        )
        return row

    @property
    def prepared_formula_runtime(self):
        """Borrow after non-text execution on this thread; this executor owns cleanup."""
        return self._formula_runtime

    def close(self) -> None:
        try:
            self._close_models()
        finally:
            close = getattr(self.cropper, "close", None)
            if close is not None:
                close()

    def _close_models(self) -> None:
        if self.model_pool is not None:
            self._ocr_runtime = None
            self._formula_runtime = None
            return
        if self._ocr_runtime is not None:
            close = getattr(self._ocr_runtime, "close", None)
            if close is not None:
                close()
            self.stats["model_unload_count"]["ocr"] += 1
            self._ocr_runtime = None
        if self._formula_runtime is not None:
            close = getattr(self._formula_runtime, "close", None)
            if close is not None:
                close()
            if getattr(self._formula_runtime, 'retained_by_owner', False):
                self.stats['formula_model_lifecycle'] = self._formula_runtime.lifecycle_metrics()
                self.stats['model_unload_count']['formula'] += self.stats['formula_model_lifecycle']['model_unload_count']
            else:
                self.stats["model_unload_count"]["formula"] += 1
            self._formula_runtime = None

    def _render_text_input(
        self,
        route: dict[str, Any],
        *,
        full_page: bool,
        dpi: int | None = None,
    ) -> dict[str, Any]:
        original = self.cropper.render(route, full_page=full_page, dpi=dpi)
        regions = tuple(
            region
            for region in self._active_ownership_regions
            if region.document_id == str(route["document_id"])
            and region.page_index == int(route["page_index"])
        )
        if not regions:
            sha = str(original.get("content_sha256") or "")
            return {
                **original,
                "contract": "TEXT_EXCLUSION_MASK",
                "ownership_contract_version": "non-text-first-ownership-mask-v1",
                "original_render_path": original.get("path"),
                "original_render_sha256": sha,
                "text_only_render_sha256": sha,
                "ownership_region_count": 0,
                "ownership_mask_px_bboxes": [],
                "mask_fill_rgb": [255, 255, 255],
                "mask_padding_px": 0,
                "masked_pixel_area": 0,
                "masked_page_area_ratio": 0.0,
                "transient_runtime_artifact": True,
                "text_only_render_reuses_original_bytes": True,
            }
        started = time.perf_counter()
        masked = derive_text_only_render(
            original,
            regions,
            self.output_dir / "transient_text_only",
        )
        self.stats["mask_rasterization_seconds"] = round(
            self.stats.get("mask_rasterization_seconds", 0.0)
            + time.perf_counter()
            - started,
            6,
        )
        return masked

    def _source_scope_review(self, route, crop):
        reviewed_route = copy.deepcopy(route)
        reviewed_route['adapter'] = 'CONTENT_REVIEW_REQUIRED'
        reviewed_route['output_kind'] = 'REVIEW'
        reviewed_route['provenance']['bbox_pdf_pt'] = copy.deepcopy(crop['bbox_pdf_pt'])
        return self._content(reviewed_route, status='REVIEW_REQUIRED',
            binary_artifact_ref=crop['path'], quality_status='RASTER_TEXT_SCOPE_REVIEW_REQUIRED',
            review_reasons=['RASTER_TEXT_SCOPE_UNRESOLVED'],
            provenance={'source_scope_review': copy.deepcopy(crop['source_scope_review']),
                        'original_adapter': route['adapter'], 'render_crop': crop})

    def _masked_empty_text(
        self,
        route: dict[str, Any],
        crop: dict[str, Any] | None,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.stats["masked_empty_text_route_count"] = (
            self.stats.get("masked_empty_text_route_count", 0) + 1
        )
        result = self._content(
            route,
            status="SUCCESS",
            text="",
            quality_status="MASKED_EMPTY_TEXT_ROUTE",
            quality_metrics={"text_retained": False, "ownership_transfer": True},
            provenance={
                **({"crop": crop} if crop is not None else {}),
                "excluded_by_non_text_owner": True,
                "ownership_regions": [
                    region.to_dict() for region in self._active_ownership_regions
                ],
                **(
                    {
                        "native_math_geometry": copy.deepcopy(
                            route["provenance"]["native_math_geometry"]
                        )
                    }
                    if "native_math_geometry" in route["provenance"]
                    else {}
                ),
                **(provenance or {}),
            },
        )

        if result['bbox_pdf_pt'] is None and crop is not None:
            result['bbox_pdf_pt'] = copy.deepcopy(crop['bbox_pdf_pt'])
        return result

    def _native_text(
        self,
        route: dict[str, Any],
        source_evidence: dict[str, Any],
        ownership_regions: tuple[NonTextOwnershipRegion, ...] = (),
    ) -> dict[str, Any]:
        selected = set(route["provenance"].get("native_text_evidence_ids", []))
        selected.update(route["provenance"].get("evidence_ids", []))
        rows = [
            row
            for row in source_evidence.get("native_text", [])
            if str(row.get("evidence_id")) in selected
        ]
        if not rows:
            raise ValueError("Native Text Bridge has no mapped source lines")
        raw_before_exclusion = [str(row.get("text", "")) for row in rows]
        exclusion = exclude_native_text(
            rows,
            ownership_regions,
            coverage_threshold=NATIVE_NON_TEXT_COVERAGE_THRESHOLD,
        )
        rows = list(exclusion.rows)
        if not rows:
            return self._masked_empty_text(
                route,
                None,
                provenance={
                    "native_excluded_spans": list(exclusion.excluded_spans),
                    "partial_native_non_text_overlaps": list(
                        exclusion.partial_overlaps
                    ),
                    "native_exclusion_threshold": NATIVE_NON_TEXT_COVERAGE_THRESHOLD,
                },
            )
        raw = [str(row.get("text", "")) for row in rows]
        normalized = []
        normalizations = []
        for value in raw:
            if "\r\n" in value:
                normalizations.append("CRLF_TO_LF")
            elif "\r" in value:
                normalizations.append("CR_TO_LF")
            normalized.append(value.replace("\r\n", "\n").replace("\r", "\n"))
        normalizations.append("JOIN_SOURCE_LINES_WITH_LF")
        text = "\n".join(normalized)
        return self._content(
            route,
            status="SUCCESS" if text else "REVIEW_REQUIRED",
            text=text,
            confidence=None,
            provenance={
                "raw_source_lines": raw,
                "raw_source_lines_before_exclusion": raw_before_exclusion,
                "normalized_source_lines": normalized,
                "normalization_types": sorted(set(normalizations)),
                "char_count_before": sum(len(value) for value in raw_before_exclusion),
                "char_count_after": len(text),
                "source_line_ids": [str(row.get("evidence_id")) for row in rows],
                "source_block_ids": sorted(
                    {
                        str(row["source_block_id"])
                        for row in rows
                        if row.get("source_block_id")
                    }
                ),
                "source_span_ids": sorted(
                    {
                        str(value)
                        for row in rows
                        for value in row.get(
                            "retained_source_span_ids",
                            row.get("source_line_span_ids", []),
                        )
                    }
                ),
                "native_excluded_spans": list(exclusion.excluded_spans),
                "partial_native_non_text_overlaps": list(exclusion.partial_overlaps),
                "native_exclusion_threshold": NATIVE_NON_TEXT_COVERAGE_THRESHOLD,
            },
            review_reasons=[] if text else ["NATIVE_TEXT_EMPTY"],
        )

    def _ocr(self, route: dict[str, Any], crop: dict[str, Any]) -> dict[str, Any]:
        runtime = self._ensure_ocr()
        recognize_route = getattr(runtime, "recognize_route", None)
        baseline_lines = (
            recognize_route(route, crop)
            if callable(recognize_route)
            else runtime.recognize(crop)
        )
        lines = baseline_lines
        recovery = {
            "policy_version": OCR_REGION_RECOVERY_VERSION,
            "stage": "BASELINE_DET_REC" if baseline_lines else "UNRECOVERED_REVIEW",
            "reason_codes": [],
            "initial_det_count": len(baseline_lines),
            "retry_dpi": None,
            "retry_det_count": None,
            "retry_quality": None,
            "retry_crop": None,
            "direct_rec_used": False,
            "direct_rec_confidence": None,
            "direct_rec_eligibility": None,
            "failures": [],
            "baseline_output_replaced": False,
        }
        direct_review = False
        if route["adapter"] == "OCR_TEXT_REGION" and not baseline_lines:
            retry_crop = None
            retry_lines = []
            try:
                retry_crop = self._render_text_input(route, full_page=False, dpi=300)
                recovery["retry_dpi"] = 300
                recovery["retry_crop"] = retry_crop
                retry_lines = runtime.recognize(retry_crop)
                recovery["retry_det_count"] = len(retry_lines)
            except Exception as exc:  # noqa: BLE001 - retain baseline crop and review
                recovery["failures"].append(
                    {"stage": "RETRY_300_DPI", "error": f"{type(exc).__name__}: {exc}"}
                )
            if retry_lines:
                retry_quality = self.ocr_recovery_policy.classify_retry_det(
                    route, retry_lines
                )
                lines = retry_lines
                recovery["stage"] = "RETRY_300_DPI"
                recovery["retry_quality"] = retry_quality
                recovery["reason_codes"].extend(retry_quality["reason_codes"])
                direct_review = not retry_quality["safe_for_success"]
            else:
                direct_crop = retry_crop or crop
                eligibility = self.ocr_recovery_policy.direct_rec_eligibility(
                    route, direct_crop
                )
                recovery["direct_rec_eligibility"] = eligibility
                if eligibility["eligible"]:
                    try:
                        direct_lines = runtime.recognize_direct(direct_crop)
                        direct_quality = self.ocr_recovery_policy.classify_direct_rec(
                            direct_lines, route
                        )
                        recovery["direct_rec_used"] = True
                        recovery["direct_rec_confidence"] = direct_quality[
                            "mean_confidence"
                        ]
                        recovery["reason_codes"].extend(direct_quality["reason_codes"])
                        if direct_quality["has_text"]:
                            lines = direct_lines
                            recovery["stage"] = (
                                "RETRY_THEN_DIRECT_REC"
                                if retry_crop is not None
                                else "DIRECT_REC_FALLBACK"
                            )
                            direct_review = not direct_quality["safe_for_success"]
                    except Exception as exc:  # noqa: BLE001 - retain crop evidence
                        recovery["failures"].append(
                            {
                                "stage": "DIRECT_REC_FALLBACK",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                if not lines:
                    recovery["stage"] = "UNRECOVERED_REVIEW"
                    recovery["reason_codes"].append("OCR_RECOVERY_EXHAUSTED")
        recovery["reason_codes"] = sorted(set(recovery["reason_codes"]))
        text = "\n".join(
            str(line.get("text", "")) for line in lines if line.get("text")
        )
        confidences = [
            float(line["confidence"])
            for line in lines
            if line.get("confidence") is not None
        ]
        fingerprint = runtime.fingerprint()
        confidence = sum(confidences) / len(confidences) if confidences else None
        quality = classify_ocr_output(text, confidence=confidence)
        status = {
            "OCR_CONTENT_OK": "SUCCESS",
            "OCR_CONTENT_WARNING": "SUCCESS_WITH_WARNING",
            "OCR_CONTENT_REVIEW": "REVIEW_REQUIRED",
        }[quality["quality_status"]]
        if direct_review:
            status = "REVIEW_REQUIRED"
            quality["quality_status"] = "OCR_CONTENT_REVIEW"
            quality["review_reasons"] = sorted(
                set(quality["review_reasons"]) | set(recovery["reason_codes"])
            )
        architecture_review_reasons = sorted(
            {
                str(reason)
                for line in lines
                if str(line.get("resolution_status") or "").startswith("AGENT_REQUIRED")
                for reason in line.get("review_reasons", ())
            }
        )
        agent_required = any(
            str(line.get("resolution_status") or "").startswith("AGENT_REQUIRED")
            for line in lines
        )
        if agent_required:
            status = "REVIEW_REQUIRED"
            quality["quality_status"] = "OCR_CONTENT_REVIEW"
            quality["review_reasons"] = sorted(
                set(quality["review_reasons"])
                | set(architecture_review_reasons)
                | {"THREE_MODEL_AGENT_REQUIRED"}
            )
        consume_provenance = getattr(runtime, "consume_last_provenance", None)
        three_model_provenance = (
            consume_provenance() if callable(consume_provenance) else None
        )
        if self.require_three_model_provenance and not isinstance(
            three_model_provenance, Mapping
        ):
            raise ProductionThreeModelInvariantError(
                "PRODUCTION_THREE_MODEL_PROVENANCE_MISSING"
            )
        return self._content(
            route,
            status=status,
            text=text,
            confidence=confidence,
            model_id=(
                f"{fingerprint.get('det_model_id')}+{fingerprint.get('rec_model_id')}"
            ),
            model_fingerprint=(
                f"{fingerprint.get('det_model_fingerprint')}+"
                f"{fingerprint.get('rec_model_fingerprint')}"
            ),
            provenance={
                "crop": crop,
                "ocr_lines": lines,
                "runtime_fingerprint": fingerprint,
                "duplicate_policy": "LOCAL_SPECIALIZED_OVERLAP_ONLY",
                "ocr_recovery": recovery,
                **(
                    {"three_model_text_evidence": three_model_provenance}
                    if three_model_provenance is not None
                    else {}
                ),
            },
            warnings=(
                quality["review_reasons"]
                if quality["quality_status"] == "OCR_CONTENT_WARNING"
                else []
            ),
            review_reasons=(
                quality["review_reasons"]
                if quality["quality_status"] == "OCR_CONTENT_REVIEW"
                else []
            ),
            quality_status=quality["quality_status"],
            quality_metrics={
                "confidence_role": quality["confidence_role"],
                "mean_confidence": confidence,
                "line_count": len(lines),
                "text_retained": quality["retain_text"],
                "ocr_recovery_stage": recovery["stage"],
            },
            ocr_recovery=recovery,
            quality_gate_version=OCR_REGION_RECOVERY_VERSION,
        )

    def _retry_cut_formula_operand(self, route, crop, latex, runtime):
        from .pdf.formula_ink_bounds import has_terminal_formula_operator
        from .formula_ocr import FormulaOcrSafetyGate
        from .formulanet_runtime import FormulaOcrOutputValidator

        complete = getattr(self.cropper, 'complete_inline_formula_bounds', None)
        if not callable(complete) or not has_terminal_formula_operator(latex):
            return latex, crop
        candidate_route = copy.deepcopy(route)
        try:
            complete(candidate_route)
            proof = candidate_route['provenance'].get('formula_ink_completion')
            if not proof or not proof['completed_components']:
                return latex, crop
            candidate_crop = self.cropper.render(candidate_route, dpi=400)
            if candidate_crop['content_sha256'] == crop['content_sha256']:
                return latex, crop
            prediction = list(runtime.predict([Path(candidate_crop['path'])], batch_size=1))
            if len(prediction) != 1 or not isinstance(prediction[0], str):
                raise ValueError('FORMULA_INK_RETRY_OUTPUT_CARDINALITY_OR_TYPE')
            candidate = prediction[0]
            validation = FormulaOcrOutputValidator().validate(candidate)
            gate = FormulaOcrSafetyGate().evaluate(
                width=candidate_crop['width'], height=candidate_crop['height'],
                raw_latex=candidate, validation=validation)
            accepted = (gate.verdict.value in {'ACCEPT', 'ACCEPT_WITH_WARNING'}
                        and not has_terminal_formula_operator(candidate)
                        and '\ufffd' not in candidate)
            attempt = {'status': 'ACCEPTED' if accepted else 'ORIGINAL_RETAINED',
                       'original_latex': latex, 'candidate_latex': candidate,
                       'candidate_safety_gate': gate.to_dict(),
                       'original_crop_sha256': crop.get('content_sha256'),
                       'candidate_crop_sha256': candidate_crop.get('content_sha256'),
                       'geometry': proof}
            if accepted:
                route['provenance'] = candidate_route['provenance']
                candidate_crop['formula_operand_recovery'] = attempt
                self._formula_predictions[route['route_id']] = candidate
                self._prepared_formula_inputs[route['route_id']] = candidate_crop
                return candidate, candidate_crop
            crop['formula_operand_recovery'] = attempt
        except Exception as exc:
            # The first prediction and its source crop remain available.
            crop['formula_operand_recovery'] = {
                'status': 'ORIGINAL_RETAINED', 'error': f'{type(exc).__name__}: {exc}'}
        return latex, crop

    def _formula(self, route: dict[str, Any], crop: dict[str, Any]) -> dict[str, Any]:
        from .formula_ocr import FormulaOcrSafetyGate
        from .formulanet_runtime import FormulaOcrOutputValidator

        native_geometry = route["provenance"].get("native_math_geometry", {})
        native_latex = native_geometry.get("native_latex", native_geometry.get("native_flat_latex"))
        runtime = None if native_latex else self._ensure_formula()
        raw_latex = native_latex or self._formula_predictions.get(route["route_id"])
        if raw_latex is None:
            raw_latex = runtime.predict([Path(crop["path"])], batch_size=1)[0]
        if runtime is not None:
            raw_latex, crop = self._retry_cut_formula_operand(route, crop, raw_latex, runtime)
        validation = FormulaOcrOutputValidator().validate(raw_latex)
        decision = FormulaOcrSafetyGate().evaluate(
            width=int(crop["width"]),
            height=int(crop["height"]),
            raw_latex=raw_latex,
            validation=validation,
        )
        verdict = decision.verdict.value
        status = {
            "ACCEPT": "SUCCESS",
            "ACCEPT_WITH_WARNING": "REVIEW_REQUIRED",
            "REVIEW_REQUIRED": "REVIEW_REQUIRED",
            "REJECT_PRESERVE_IMAGE": "FAILED_PRESERVE_INPUT",
        }[verdict]
        quality_status = {
            "ACCEPT": "FORMULA_CONTENT_OK",
            "ACCEPT_WITH_WARNING": "FORMULA_CONTENT_REVIEW",
            "REVIEW_REQUIRED": "FORMULA_CONTENT_REVIEW",
            "REJECT_PRESERVE_IMAGE": "FORMULA_CONTENT_REJECT_PRESERVE_INPUT",
        }[verdict]
        formula_review_reasons = [reason["code"] for reason in decision.reasons]
        if verdict == "ACCEPT_WITH_WARNING":
            formula_review_reasons.append("FORMULA_WARNING_REQUIRES_VISUAL_REVIEW")
        content_id = _content_id(route)
        if self.formula_visual_verifier is None:
            verification = {
                "schema": FORMULA_VISUAL_VERIFICATION_SCHEMA,
                "verification_id": f"formula-visual-unavailable-{content_id}",
                "content_id": content_id,
                "source_crop_ref": crop["path"],
                "rendered_formula_ref": None,
                "status": "VERIFICATION_UNAVAILABLE",
                "score": None,
                "features": {},
                "reason_codes": ["FORMULA_RENDERER_UNAVAILABLE"],
                "source_bbox_pdf_pt": list(crop["bbox_pdf_pt"]),
                "source_foreground_bbox": None,
                "render_foreground_bbox": None,
                "policy_version": FORMULA_VISUAL_VERIFIER_VERSION,
                "provenance": {"raw_latex_rewritten": False, "fail_closed": True},
            }
        else:
            try:
                verification = self.formula_visual_verifier.verify(
                    content_id=content_id,
                    raw_latex=raw_latex,
                    source_crop_ref=crop["path"],
                    source_bbox_pdf_pt=list(crop["bbox_pdf_pt"]),
                )
            except Exception as exc:  # noqa: BLE001 - verifier must fail closed
                verification = {
                    "schema": FORMULA_VISUAL_VERIFICATION_SCHEMA,
                    "verification_id": f"formula-visual-failed-{content_id}",
                    "content_id": content_id,
                    "source_crop_ref": crop["path"],
                    "rendered_formula_ref": None,
                    "status": "VERIFICATION_UNAVAILABLE",
                    "score": None,
                    "features": {},
                    "reason_codes": ["FORMULA_VISUAL_VERIFIER_EXCEPTION"],
                    "source_bbox_pdf_pt": list(crop["bbox_pdf_pt"]),
                    "source_foreground_bbox": None,
                    "render_foreground_bbox": None,
                    "policy_version": FORMULA_VISUAL_VERIFIER_VERSION,
                    "provenance": {
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_latex_rewritten": False,
                        "fail_closed": True,
                    },
                }
        if self.formula_consensus_verifier is None:
            consensus = {
                "schema": FORMULA_CONSENSUS_VERIFICATION_SCHEMA,
                "verification_id": f"formula-consensus-unavailable-{content_id}",
                "content_id": content_id,
                "existing_gate": decision.to_dict(),
                "multiview": {},
                "ocr_crosscheck": {},
                "visual_evidence": verification,
                "consensus_status": "CONSENSUS_UNAVAILABLE",
                "risk_score": 1.0,
                "reason_codes": ["FORMULA_CONSENSUS_VERIFIER_UNAVAILABLE"],
                "raw_latex_sha256": hashlib.sha256(
                    raw_latex.encode("utf-8")
                ).hexdigest(),
                "policy_version": FORMULA_CONSENSUS_VERIFIER_VERSION,
                "provenance": {
                    "raw_latex_rewritten": False,
                    "fail_closed": True,
                },
            }
        else:
            try:
                consensus = self.formula_consensus_verifier.verify(
                    content_id=content_id,
                    raw_latex=raw_latex,
                    source_crop_ref=crop["path"],
                    source_bbox_pdf_pt=list(crop["bbox_pdf_pt"]),
                    existing_gate=decision.to_dict(),
                    existing_status=status,
                    visual_evidence=verification,
                    formula_predictor=lambda paths: (
                        runtime or self._ensure_formula()
                    ).predict(paths, batch_size=min(2, len(paths))),
                    ocr_runtime_factory=self._ensure_ocr,
                )
            except Exception as exc:  # noqa: BLE001 - consensus must fail closed
                consensus = {
                    "schema": FORMULA_CONSENSUS_VERIFICATION_SCHEMA,
                    "verification_id": f"formula-consensus-failed-{content_id}",
                    "content_id": content_id,
                    "existing_gate": decision.to_dict(),
                    "multiview": {},
                    "ocr_crosscheck": {},
                    "visual_evidence": verification,
                    "consensus_status": "CONSENSUS_UNAVAILABLE",
                    "risk_score": 1.0,
                    "reason_codes": ["FORMULA_CONSENSUS_VERIFIER_EXCEPTION"],
                    "raw_latex_sha256": hashlib.sha256(
                        raw_latex.encode("utf-8")
                    ).hexdigest(),
                    "policy_version": FORMULA_CONSENSUS_VERIFIER_VERSION,
                    "provenance": {
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_latex_rewritten": False,
                        "fail_closed": True,
                    },
                }
        gated = apply_formula_consensus_gate(
            existing_status=status,
            existing_quality_status=quality_status,
            raw_latex=raw_latex,
            verification=consensus,
        )
        status = gated["status"]
        quality_status = gated["quality_status"]
        formula_review_reasons.extend(gated["review_reasons"])
        fingerprint = (
            runtime.fingerprint()
            if runtime is not None
            else {"source_decoder": "native-pdf-character-baseline-v1"}
        )
        return self._content(
            route,
            status=status,
            # Rejected guesses remain immutable evidence below, while the
            # consumer receives a preserved source image for later review.
            latex=None if status == 'FAILED_PRESERVE_INPUT' else raw_latex,
            model_id=None
            if native_latex
            else str(
                fingerprint.get("model_id")
                or fingerprint.get("model_identifier")
                or "pp-formulanet-plus-l"
            ),
            model_fingerprint=(
                fingerprint.get("model_fingerprint")
                or fingerprint.get("model_sha256")
                or fingerprint.get("directory_sha256")
            ),
            warnings=sorted(
                {issue["code"] for issue in validation.issues} | set(gated["warnings"])
            ),
            review_reasons=sorted(set(formula_review_reasons))
            if status in {"REVIEW_REQUIRED", "FAILED_PRESERVE_INPUT"}
            else [],
            quality_status=quality_status,
            quality_metrics={
                "validator_verdict": validation.verdict.value,
                "safety_gate_verdict": verdict,
                "raw_latex_immutable": True,
                "visual_review_required": verification["status"] != "VISUAL_PASS",
                "formula_visual_verification_status": verification["status"],
                "formula_consensus_status": consensus["consensus_status"],
            },
            provenance={
                "crop": crop,
                "formula_raw_latex": raw_latex,
                "validator": validation.to_dict(),
                "safety_gate": decision.to_dict(),
                "runtime_fingerprint": fingerprint,
                "recognition_backend": "NATIVE_CHARACTER_GEOMETRY"
                if native_latex
                else "FORMULANET",
                "semantic_auto_correction_used": False,
            },
            formula_visual_verification=verification,
            formula_consensus_verification=consensus,
            multiview_outputs=consensus.get("multiview", {}).get("outputs"),
            ocr_formula_crosscheck=consensus.get("ocr_crosscheck"),
            visual_evidence_summary=consensus.get("visual_evidence"),
            consensus_reason_codes=consensus.get("reason_codes"),
            quality_gate_version=FORMULA_CONSENSUS_VERIFIER_VERSION,
        )

    def _native_image(self, route: dict[str, Any]) -> dict[str, Any]:
        import fitz

        placements = route["provenance"].get("native_image_placements", [])
        if len(placements) != 1:
            raise ValueError("Native image extraction requires exactly one placement")
        xref = int(placements[0]["xref"])
        source = Path(route["provenance"]["source_path"])
        with fitz.open(source) as document:
            payload = document.extract_image(xref)
        data = payload.get("image")
        if not data:
            raise ValueError("Native PDF image extraction returned no bytes")
        sha = hashlib.sha256(data).hexdigest()
        extension = str(payload.get("ext") or "bin").lower()
        path = self.output_dir / "temp_content_assets" / f"sha256-{sha}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        artifact = {
            "path": str(path),
            "content_sha256": sha,
            "bytes": len(data),
            "extension": extension,
            "mime": f"image/{extension}",
            "pixel_width": payload.get("width"),
            "pixel_height": payload.get("height"),
            "xref": xref,
            "object_identity": placements[0].get("object_identity"),
            "source_placement_evidence_ids": [placements[0].get("evidence_id")],
            "source_bbox_pdf_pt": placements[0].get("bbox_pdf_pt"),
            "source_appearance": placements[0].get("native_appearance", {}),
        }
        return self._content(
            route,
            status="SUCCESS",
            binary_artifact_ref=str(path),
            provenance={"native_extraction": artifact},
        )

    def _source_graphics_crop(
        self, route: dict[str, Any], crop: dict[str, Any]
    ) -> dict[str, Any]:
        if (route["provenance"].get("source_graphics_without_native_prose")
                or route["provenance"].get("source_graphics_exclude_regions")):
            import fitz

            from .pdf.graphics_render import render_source_graphics

            try:
                with fitz.open(route["provenance"]["source_path"]) as document:
                    pixmap, audit = render_source_graphics(
                        document[int(route["page_index"])], crop["bbox_pdf_pt"], crop["dpi"],
                        preserve_regions=route["provenance"].get("source_graphics_preserve_regions", []),
                        exclude_regions=route["provenance"].get("source_graphics_exclude_regions", []),
                        omit_native_prose=bool(route["provenance"].get("source_graphics_without_native_prose")),
                    )
                data = pixmap.tobytes("png")
                if audit.get('semantic_image_excluded_regions_pdf_pt'):
                    from PIL import Image
                    remaining = Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
                    audit['post_exclusion_all_white'] = all(low == 255 for low, high in remaining.getextrema())
                sha = hashlib.sha256(data).hexdigest()
                path = self.output_dir / "temp_content_assets" / f"sha256-{sha}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                crop = {**crop, "path": str(path), "content_sha256": sha,
                        "bytes": len(data), "width": pixmap.width, "height": pixmap.height,
                        "source_graphics_render": audit}
            except (RuntimeError, ValueError, AttributeError) as exc:
                crop = {**crop, "source_graphics_render": {
                    "method": "ORIGINAL_SOURCE_CROP_RETAINED", "error": str(exc)}}
        return crop

    def _image_crop(
        self, route: dict[str, Any], crop: dict[str, Any]
    ) -> dict[str, Any]:
        from .pdf.prose_edge_crop import refine_prose_edge

        route, crop = refine_prose_edge(route, crop, self.cropper)
        crop = self._source_graphics_crop(route, crop)
        graphics = crop.get("source_graphics_render", {})
        empty = graphics.get("post_exclusion_all_white") is True
        residual = bool(graphics.get("semantic_image_excluded_regions_pdf_pt")) and not empty
        return self._content(
            route, status="REVIEW_REQUIRED" if residual else "SUCCESS", binary_artifact_ref=crop["path"],
            provenance={"render_crop": crop},
            quality_status=("MASKED_EMPTY_GRAPHICS_ROUTE" if empty else
                            "GRAPHICS_RESIDUAL_REVIEW_REQUIRED" if residual else None),
            review_reasons=["GRAPHICS_RESIDUAL_REVIEW_REQUIRED"] if residual else None,
        )

    def _table_deferred(
        self, route: dict[str, Any], crop: dict[str, Any]
    ) -> dict[str, Any]:
        return self._content(
            route,
            status="DEFERRED",
            review_reasons=["TABLE_BRANCH_DEFERRED_PENDING_EXTERNAL_CORPUS"],
            provenance={
                "preserved_bbox_pdf_pt": route["provenance"].get("bbox_pdf_pt"),
                "preserved_render_crop": crop,
                "table_engine_called": False,
            },
        )

    def _review(self, route: dict[str, Any]) -> dict[str, Any]:
        if (not route["provenance"].get("bbox_pdf_pt")
                and "PAGE_ESCALATION_REQUIRES_CONTENT_REVIEW" in route["decision_reason_codes"]):
            from .pdf.page_review import inspect_page_review_scope

            audit = inspect_page_review_scope(route)
            scoped_route = copy.deepcopy(route)
            scoped_route["provenance"]["bbox_pdf_pt"] = audit["bbox_pdf_pt"]
            if audit["all_pixels_white"]:
                result = self._content(
                    scoped_route, status="SUCCESS", text="", review_reasons=[],
                    provenance={"page_scope_review": audit},
                )
                result["quality_status"] = "VERIFIED_BLANK_SOURCE_PAGE"
                return result
            crop = self.cropper.render(scoped_route, full_page=True)
            return self._content(
                scoped_route, status="REVIEW_REQUIRED", binary_artifact_ref=crop["path"],
                review_reasons=list(route["decision_reason_codes"]),
                provenance={"page_scope_review": audit, "preserved_render_crop": crop},
            )
        if (route["provenance"].get("source_graphics_without_native_prose")
                or route["provenance"].get("source_graphics_exclude_regions")):
            crop = self._source_graphics_crop(route, self.cropper.render(route, full_page=False))
            return self._content(
                route, status="REVIEW_REQUIRED", binary_artifact_ref=crop["path"],
                review_reasons=list(route["decision_reason_codes"]),
                provenance={"preserved_route_provenance": route["provenance"],
                            "preserved_render_crop": crop},
            )
        return self._content(
            route,
            status="REVIEW_REQUIRED",
            review_reasons=list(route["decision_reason_codes"]),
            provenance={"preserved_route_provenance": route["provenance"]},
        )

    def _failed(
        self,
        route: dict[str, Any],
        exc: Exception,
        preserved_input: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self._content(
            route,
            status="FAILED_PRESERVE_INPUT",
            review_reasons=[f"{type(exc).__name__}: {exc}"],
            provenance={
                "preserved_input": preserved_input
                or {
                    "source_path": route["provenance"].get("source_path"),
                    "bbox_pdf_pt": route["provenance"].get("bbox_pdf_pt"),
                    "evidence_ids": route["provenance"].get("evidence_ids", []),
                },
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            },
        )

    def _content(
        self,
        route: dict[str, Any],
        *,
        status: str,
        text: str | None = None,
        latex: str | None = None,
        binary_artifact_ref: str | None = None,
        confidence: float | None = None,
        warnings: list[str] | None = None,
        review_reasons: list[str] | None = None,
        model_id: str | None = None,
        model_fingerprint: str | None = None,
        provenance: dict[str, Any] | None = None,
        quality_status: str | None = None,
        quality_metrics: dict[str, Any] | None = None,
        formula_visual_verification: dict[str, Any] | None = None,
        formula_consensus_verification: dict[str, Any] | None = None,
        multiview_outputs: list[dict[str, Any]] | None = None,
        ocr_formula_crosscheck: dict[str, Any] | None = None,
        visual_evidence_summary: dict[str, Any] | None = None,
        consensus_reason_codes: list[str] | None = None,
        ocr_recovery: dict[str, Any] | None = None,
        quality_gate_version: str | None = None,
    ) -> dict[str, Any]:
        boundary_reviews = route['provenance'].get('boundary_reviews', [])
        if boundary_reviews:
            review_reasons = sorted(set(review_reasons or []) | {
                review['kind'] + '_REVIEW_REQUIRED' for review in boundary_reviews})
            if status == 'SUCCESS':
                status = 'REVIEW_REQUIRED'
        content_id = _content_id(route)
        return {
            "schema": REGION_CONTENT_IR_SCHEMA,
            "content_id": content_id,
            "route_id": route["route_id"],
            "document_id": route["document_id"],
            "page_index": route["page_index"],
            "status": status,
            "content_kind": route["output_kind"],
            # Aggregated routes retain child semantic evidence. Its first
            # alphabetical label cannot override the owning page/table route.
            "semantic_hint": route["output_kind"]
                if route["adapter"] in {"PAGE_VISUAL_TEXT_RECOVERY", "TABLE_ENGINE", "TABLE_DEFERRED_PRESERVE"}
            else route["semantic_evidence"][0]
            if route["semantic_evidence"]
            else route["output_kind"],
            "text": text,
            "latex": latex,
            "binary_artifact_ref": binary_artifact_ref,
            "bbox_pdf_pt": route["provenance"].get("bbox_pdf_pt"),
            "bbox_render_px": None,
            "confidence": confidence,
            "warnings": warnings or [],
            "review_reasons": review_reasons or [],
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "quality_status": quality_status or _default_quality_status(status),
            "quality_metrics": quality_metrics or {},
            "conflict_ids": [],
            "conflict_resolution": "NO_CONFLICT",
            "secondary_content_refs": [],
            "duplicate_risk": False,
            "formula_discovery_provenance": route["provenance"].get(
                "formula_discovery"
            ),
            "review_taxonomy": classify_review_reasons(
                review_reasons or [], semantic_evidence=route["semantic_evidence"]
            )
            if review_reasons
            else [],
            "formula_visual_verification": formula_visual_verification,
            "formula_consensus_verification": formula_consensus_verification,
            "multiview_outputs": multiview_outputs,
            "ocr_formula_crosscheck": ocr_formula_crosscheck,
            "visual_evidence_summary": visual_evidence_summary,
            "consensus_reason_codes": consensus_reason_codes,
            "ocr_recovery": ocr_recovery,
            "quality_gate_version": quality_gate_version,
            "source_candidate_ids": list(route["input_candidate_ids"]),
            "source_region_ids": list(route["source_region_ids"]),
            "source_unit_ids": list(route["source_unit_ids"]),
            "provenance": {
                "source_candidate_kinds": list(
                    route["provenance"].get("candidate_kinds", [])
                ),
                "route_decision": {
                    "adapter": route["adapter"],
                    "reason_codes": route["decision_reason_codes"],
                    "decision_version": route["decision_version"],
                },
                "native_text_trust": route["provenance"].get("native_text_trust"),
                "source_profile": route["provenance"].get("source_profile"),
                **({"prose_edge_refinement": copy.deepcopy(route["provenance"]["prose_edge_refinement"])}
                   if "prose_edge_refinement" in route["provenance"] else {}),
                **({"source_graphics_ownership": copy.deepcopy(route["provenance"]["source_graphics_ownership"])}
                   if "source_graphics_ownership" in route["provenance"] else {}),
                **({"table_content_ownership": copy.deepcopy(
                    route["provenance"]["table_content_ownership"]
                )} if "table_content_ownership" in route["provenance"] else {}),
                **({'boundary_reviews': copy.deepcopy(boundary_reviews)} if boundary_reviews else {}),
                **(provenance or {}),
            },
        }

    def _ensure_ocr(self):
        if self._ocr_runtime is None:
            if self.model_pool is None:
                self._ocr_runtime = self.ocr_runtime_factory()
                self.stats["model_load_count"]["ocr"] += 1
            else:
                before = self.model_pool.metrics()["load_count"]
                self._ocr_runtime = self.model_pool.get("ocr")
                self.stats["model_load_count"]["ocr"] += (
                    self.model_pool.metrics()["load_count"] - before
                )
        return self._ocr_runtime

    def _ensure_formula(self):
        if self._formula_runtime is None:
            if self.model_pool is None:
                self._formula_runtime = self.formula_runtime_factory()
                self.stats["model_load_count"]["formula"] += getattr(self._formula_runtime, 'model_load_count', 1)
                if getattr(self._formula_runtime, 'retained_by_owner', False):
                    self.stats['formula_model_lifecycle'] = self._formula_runtime.lifecycle_metrics()
            else:
                before = self.model_pool.metrics()["load_count"]
                self._formula_runtime = self.model_pool.get("formula")
                self.stats["model_load_count"]["formula"] += (
                    self.model_pool.metrics()["load_count"] - before
                )
        return self._formula_runtime


class PdfCropper:
    def __init__(self, output_dir: str | Path, *, dpi: int = 200, padding_px: int = 2,
                 raster_cache_bytes: int = 64 * 1024 * 1024):
        self.output_dir = Path(output_dir).resolve()
        self.dpi = dpi
        self.padding_px = padding_px
        self._native_ink_maps = {}
        self._document = None
        self._document_key = None
        self._pages = {}
        if raster_cache_bytes <= 0:
            raise ValueError("Raster cache budget must be positive")
        self._raster_cache_limit = int(raster_cache_bytes)
        self._rasters = {}
        self._raster_bytes = 0

    def close(self) -> None:
        document = self._document
        self._document = None
        self._document_key = None
        self._pages.clear()
        self._rasters.clear()
        self._raster_bytes = 0
        self._native_ink_maps.clear()
        if document is not None:
            document.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @contextmanager
    def _document_scope(self, source: Path):
        import fitz

        source = source.resolve()
        stat = source.stat()
        key = (str(source), stat.st_mtime_ns, stat.st_size)
        if key != self._document_key:
            self.close()
            self._document = fitz.open(source)
            self._document_key = key
        try:
            yield self._document
        except BaseException:
            self.close()
            raise

    def _cached_page(self, document, page_index: int):
        cached = self._pages.pop(page_index, None)
        if cached is None:
            page = document[page_index]
            cached = (page, page.get_displaylist(annots=True))
        self._pages[page_index] = cached
        # Keep memory bounded for textbooks while reusing expensive image decoding.
        if len(self._pages) > 8:
            self._pages.pop(next(iter(self._pages)))
        return cached

    def _render_raster_crop(self, source, page, display_list, bbox, dpi):
        """Crop stable page pixels, with a bounded LRU for decoded rasters."""
        import fitz

        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        page_pixels = (page.rect * matrix).irect
        clip_pixels = (fitz.Rect(bbox) * matrix).irect & page_pixels
        raster_bytes = page_pixels.width * page_pixels.height * 3
        if raster_bytes > self._raster_cache_limit:
            # A small region on a poster must not allocate a whole poster.
            # A fresh document isolates MuPDF's first-render image cache.
            with fitz.open(source) as isolated:
                return isolated[page.number].get_pixmap(
                    matrix=matrix, clip=fitz.Rect(bbox),
                    colorspace=fitz.csRGB, alpha=False,
                )

        key = (page.number, dpi)
        full_page = self._rasters.pop(key, None)
        if full_page is None:
            while self._rasters and self._raster_bytes + raster_bytes > self._raster_cache_limit:
                oldest = self._rasters.pop(next(iter(self._rasters)))
                self._raster_bytes -= oldest.stride * oldest.height
                del oldest
            # Clipped rendering can resample embedded images differently after
            # another crop. Render the full page before copying integer pixels.
            full_page = display_list.get_pixmap(
                matrix=matrix, colorspace=fitz.csRGB, alpha=False,
            )
            self._raster_bytes += full_page.stride * full_page.height
        self._rasters[key] = full_page
        cropped = fitz.Pixmap(fitz.csRGB, clip_pixels, False)
        cropped.copy(full_page, clip_pixels)
        return cropped

    def complete_inline_formula_bounds(self, route, *, require_core_majority=False):
        from PIL import Image
        from .pdf.formula_ink_bounds import complete_formula_ink_bounds

        provenance = route['provenance']
        box = provenance.get('bbox_pdf_pt')
        minimum_core_fraction = 0.5 if require_core_majority else 0.0
        previous = provenance.get('formula_ink_completion')
        if previous and previous.get('minimum_core_ink_fraction', 0.0) <= minimum_core_fraction:
            return
        if (not _valid_bbox(box)
                or provenance.get('native_math_geometry') or box[3]-box[1] >= 20):
            return
        margin = min(12.0, max(2.0, (box[3]-box[1]) * 0.75))
        context_route = {**route, 'adapter': 'FORMULA_INK_CONTEXT', 'provenance': {
            **provenance, 'bbox_pdf_pt': [box[0]-margin, box[1]-margin,
                                        box[2]+margin, box[3]+margin]}}
        context = self.render(context_route, dpi=400)
        with Image.open(context['path']) as image:
            completed, proof = complete_formula_ink_bounds(
                image, context, box, minimum_core_ink_fraction=minimum_core_fraction)
        provenance['formula_ink_completion'] = proof
        provenance['bbox_pdf_pt'] = completed

    def render(
        self,
        route: dict[str, Any],
        *,
        full_page: bool = False,
        dpi: int | None = None,
    ) -> dict[str, Any]:
        import fitz

        effective_dpi = self.dpi if dpi is None else dpi
        source = Path(route["provenance"]["source_path"])
        with self._document_scope(source) as document:
            page, display_list = self._cached_page(document, int(route["page_index"]))
            page_rect = page.rect
            requested = [0.0, 0.0, page_rect.width, page_rect.height]
            if not full_page:
                requested = route["provenance"].get("bbox_pdf_pt")
            ink_proof = None
            native = route["provenance"].get("native_math_geometry", {})
            if (not full_page and route.get("adapter") == "FORMULA_RECOGNITION"
                    and native.get("source_evidence")
                    and not native.get("native_latex", native.get("native_flat_latex"))):
                original = native.get("original_layout_bbox_pdf_pt")
                source_box = native.get("source_bbox_pdf_pt")
                complete = not original or (
                    _valid_bbox(source_box) and _intersection_area(original, source_box) / _area(original) >= 0.7
                )
                if complete:
                    from .pdf.native_glyph_bounds import (
                        formula_ink_bbox,
                        page_native_ink_map,
                    )

                    stat = source.stat()
                    key = (str(source.resolve()), stat.st_mtime_ns, stat.st_size, page.number)
                    if key not in self._native_ink_maps:
                        if len(self._native_ink_maps) >= 4:
                            self._native_ink_maps.pop(next(iter(self._native_ink_maps)))
                        self._native_ink_maps[key] = page_native_ink_map(page)
                    tight = formula_ink_bbox(native["source_evidence"], self._native_ink_maps[key])
                    if tight and _valid_bbox(tight):
                        clipped = list(fitz.Rect(tight) & fitz.Rect(requested))
                        if _valid_bbox(clipped):
                            ink_proof = {"backend": "EMBEDDED_TRUETYPE_GLYPH_BOUNDS",
                                         "ownership_bbox_pdf_pt": list(requested),
                                         "ink_bbox_pdf_pt": tight}
                            requested = clipped
            contract = build_ocr_crop_contract(
                bbox_pdf_pt=requested,
                page_width_pt=float(page_rect.width),
                page_height_pt=float(page_rect.height),
                dpi=effective_dpi,
                padding_px=self.padding_px,
            )
            if ink_proof:
                contract["native_ink_bounds"] = ink_proof
            if route['provenance'].get('formula_ink_completion'):
                contract['formula_ink_completion'] = copy.deepcopy(
                    route['provenance']['formula_ink_completion'])
            bbox = contract["bbox_pdf_pt"]
            pixmap = self._render_raster_crop(
                source, page, display_list, bbox, effective_dpi,
            )
            data = pixmap.tobytes("png")
        sha = hashlib.sha256(data).hexdigest()
        path = self.output_dir / f"sha256-{sha}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return {
            **contract,
            "path": str(path),
            "bbox_render_px": [0, 0, pixmap.width, pixmap.height],
            "width": pixmap.width,
            "height": pixmap.height,
            "scale_x": pixmap.width / (bbox[2] - bbox[0]),
            "scale_y": pixmap.height / (bbox[3] - bbox[1]),
            "content_sha256": sha,
            "bytes": len(data),
            "mime": "image/png",
            "extension": "png",
            "source_path": str(source),
            "page_index": route["page_index"],
        }


def build_ocr_crop_contract(
    *,
    bbox_pdf_pt: list[float],
    page_width_pt: float,
    page_height_pt: float,
    dpi: int = 200,
    padding_px: int = 2,
) -> dict[str, Any]:
    if not _valid_bbox(bbox_pdf_pt):
        raise ValueError("OCR crop bbox has zero area or is invalid")
    if page_width_pt <= 0 or page_height_pt <= 0 or dpi <= 0 or padding_px < 0:
        raise ValueError("OCR crop geometry, DPI, and padding must be valid")
    padding_pt = padding_px * 72.0 / dpi
    bbox = [
        round(max(0.0, float(bbox_pdf_pt[0]) - padding_pt), 6),
        round(max(0.0, float(bbox_pdf_pt[1]) - padding_pt), 6),
        round(min(page_width_pt, float(bbox_pdf_pt[2]) + padding_pt), 6),
        round(min(page_height_pt, float(bbox_pdf_pt[3]) + padding_pt), 6),
    ]
    if not _valid_bbox(bbox):
        raise ValueError("OCR crop bbox has zero area after clipping")
    return {
        "contract": "pdf-region-ocr-crop-v0",
        "input_coordinate_system": "pdf-points-top-left-rotation-normalized",
        "bbox_pdf_pt": bbox,
        "page_width_pt": float(page_width_pt),
        "page_height_pt": float(page_height_pt),
        "dpi": int(dpi),
        "padding_px": int(padding_px),
        "padding_pt": round(padding_pt, 6),
        "clipped_to_page": bbox != [round(float(v), 6) for v in bbox_pdf_pt],
        "rotation_normalized": True,
        "color_space": "RGB",
        "alpha": False,
        "crop_to_page_transform": {
            "origin_pdf_pt": bbox[:2],
            "pdf_points_per_render_pixel": 72.0 / dpi,
        },
    }


class PaddleXPdfOcrRuntime:
    """Lazy local-only PP-OCR det -> crop -> rec runtime fixed to GPU FP32."""

    def __init__(
        self,
        *,
        models_root: str | Path | None = None,
        config_path: str | Path | None = None,
        mcp_root: str | Path | None = None,
        device: str = "gpu:0",
    ):
        if device != "gpu:0":
            raise ValueError(
                "PDF OCR production runtime requires gpu:0; CPU fallback is forbidden"
            )
        self.models_root = models_root
        self.config_path = config_path
        self.mcp_root = mcp_root
        self.device = device
        self._det_model = None
        self._rec_model = None
        self._fingerprint = None
        self.load_count = 0
        self.unload_count = 0
        self.regions_processed = 0

    def fingerprint(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {**self._fingerprint, 'execution':getattr(self, '_recognition_execution', {'mode':'scalar'})}

    def recognize(self, crop: dict[str, Any]) -> list[dict[str, Any]]:
        from PIL import Image

        self._ensure_loaded()
        path = Path(crop["path"])
        det_results = list(self._det_model.predict(str(path), batch_size=1))
        if len(det_results) != 1:
            raise RuntimeError(
                "PP-OCR detector must return exactly one result per crop"
            )
        det = _model_result_dict(det_results[0])
        raw_polygons = det.get("dt_polys")
        raw_scores = det.get("dt_scores")
        polygons = [
            _plain_list(value)
            for value in ([] if raw_polygons is None else raw_polygons)
        ]
        scores = [float(value) for value in ([] if raw_scores is None else raw_scores)]
        boxes = []
        for index, polygon in enumerate(polygons):
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            if not xs or not ys:
                continue
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            if not _valid_bbox(bbox):
                continue
            boxes.append(
                (bbox, scores[index] if index < len(scores) else None, polygon)
            )
        boxes.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
        lines = []
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            for sequence, (bbox, det_score, polygon) in enumerate(boxes):
                clipped = [
                    max(0, int(bbox[0])),
                    max(0, int(bbox[1])),
                    min(rgb.width, int(bbox[2] + 0.999999)),
                    min(rgb.height, int(bbox[3] + 0.999999)),
                ]
                if not _valid_bbox(clipped):
                    continue
                line_image = rgb.crop(tuple(clipped))
                line_bytes = _image_png_bytes(line_image)
                line_sha = hashlib.sha256(line_bytes).hexdigest()
                line_path = path.parent / f"ocr-line-sha256-{line_sha}.png"
                if not line_path.exists():
                    line_path.write_bytes(line_bytes)
                rec_results = list(
                    self._rec_model.predict(str(line_path), batch_size=1)
                )
                if len(rec_results) != 1:
                    raise RuntimeError(
                        "PP-OCR recognizer must return exactly one result per line"
                    )
                rec = _model_result_dict(rec_results[0])
                text = str(rec.get("rec_text") or "")
                rec_score = float(rec.get("rec_score") or 0.0)
                confidence = (
                    rec_score if det_score is None else min(rec_score, det_score)
                )
                lines.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "det_confidence": det_score,
                        "rec_confidence": rec_score,
                        "bbox_local_px": [float(value) for value in clipped],
                        "bbox_page_pdf_pt": _local_px_to_page_pdf(clipped, crop),
                        "polygon_local_px": polygon,
                        "reading_sequence": sequence,
                        "line_crop_ref": str(line_path.resolve()),
                        "line_crop_sha256": line_sha,
                    }
                )
        self.regions_processed += 1
        return lines

    def recognize_batch(
        self, crops: list[dict[str, Any]], *, det_batch_size: int = 32, rec_batch_size: int = 8,
        complete_regions: bool = False
    ) -> list[list[dict[str, Any]]]:
        """Batch the same detector and line recognizer, retaining source order."""
        from PIL import Image

        if not crops:
            return []
        self._ensure_loaded()
        self._recognition_execution = {'mode':'batch', 'det_batch_size':det_batch_size,
                                       'rec_batch_size':rec_batch_size, 'complete_region_lines':complete_regions,
                                       'split_masked_line_fragments':complete_regions}
        from .pdf.detection_batches import predict_detector_batches

        detections, detector_metrics = predict_detector_batches(self._det_model, crops, batch_size=det_batch_size)
        self._recognition_execution['detector_grouping'] = detector_metrics
        if len(detections) != len(crops):
            raise RuntimeError("PP_OCR_BATCH_DETECTION_CARDINALITY")
        result: list[list[dict[str, Any]]] = [[] for _ in crops]
        pending = []
        for crop_index, (crop, detection) in enumerate(zip(crops, detections, strict=True)):
            det = _model_result_dict(detection)
            raw_polygons, raw_scores = det.get("dt_polys"), det.get("dt_scores")
            polygons = [_plain_list(value) for value in ([] if raw_polygons is None else raw_polygons)]
            scores = [float(value) for value in ([] if raw_scores is None else raw_scores)]
            boxes = []
            for index, polygon in enumerate(polygons):
                xs, ys = [float(point[0]) for point in polygon], [float(point[1]) for point in polygon]
                if not xs or not ys:
                    continue
                box = [min(xs), min(ys), max(xs), max(ys)]
                if _valid_bbox(box):
                    boxes.append((box, scores[index] if index < len(scores) else None, polygon))
            boxes.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
            with Image.open(crop["path"]) as image:
                rgb = image.convert("RGB")
                geometry_policy = 'DETECTOR_BOUNDS'
                if complete_regions and crop.get('complete_region_lines', True):
                    from .pdf.region_line_coverage import complete_region_line_boxes, split_line_boxes_at_exclusions
                    completed = complete_region_line_boxes(rgb, boxes)
                    if completed is not None:
                        boxes = completed
                        geometry_policy = 'SOURCE_INK_COMPLETE_LINE_STRIPS'
                        if crop.get('ownership_mask_px_bboxes'):
                            boxes = split_line_boxes_at_exclusions(rgb, boxes, crop['ownership_mask_px_bboxes'])
                            geometry_policy = 'SOURCE_INK_COMPLETE_MASKED_LINE_FRAGMENTS'
                for sequence, (box, det_score, polygon) in enumerate(boxes):
                    clipped = [max(0, int(box[0])), max(0, int(box[1])),
                               min(rgb.width, int(box[2] + 0.999999)), min(rgb.height, int(box[3] + 0.999999))]
                    if not _valid_bbox(clipped):
                        continue
                    data = _image_png_bytes(rgb.crop(tuple(clipped)))
                    sha = hashlib.sha256(data).hexdigest()
                    path = Path(crop["path"]).parent / f"ocr-line-sha256-{sha}.png"
                    if not path.exists():
                        path.write_bytes(data)
                    line = {"bbox_local_px": [float(v) for v in clipped],
                            "bbox_page_pdf_pt": _local_px_to_page_pdf(clipped, crop),
                            "polygon_local_px": polygon, "reading_sequence": sequence,
                            "line_crop_ref": str(path.resolve()), "line_crop_sha256": sha,
                            "det_confidence": det_score}
                    result[crop_index].append(line)
                    if complete_regions:
                        line['crop_geometry_policy'] = geometry_policy
                    pending.append(line)
        # Similar aspect ratios limit padding without changing output ordering.
        ordered = sorted(pending, key=lambda row: (
            (row["bbox_local_px"][2] - row["bbox_local_px"][0]) /
            (row["bbox_local_px"][3] - row["bbox_local_px"][1])
        ))
        if ordered:
            positioned = bool(getattr(self, 'character_positions', False))
            if positioned:
                from .pdf.ocr_character_geometry import predict_positioned_batch
                recognized = predict_positioned_batch(self._rec_model,
                    [Path(line['line_crop_ref']) for line in ordered], batch_size=rec_batch_size)
            else:
                recognized = list(self._rec_model.predict(
                    [line["line_crop_ref"] for line in ordered], batch_size=rec_batch_size
                ))
            if len(recognized) != len(ordered):
                raise RuntimeError("PP_OCR_BATCH_RECOGNITION_CARDINALITY")
            for line, prediction in zip(ordered, recognized, strict=True):
                rec = _model_result_dict(prediction)
                score = float(rec.get("rec_score") or 0.0)
                confidence = score if line["det_confidence"] is None else min(score, line["det_confidence"])
                value = rec.get('rec_text') or ''
                text = value[0] if positioned and isinstance(value, (tuple, list)) else value
                line.update(text=str(text), confidence=confidence, rec_confidence=score)
                if positioned:
                    from .pdf.ocr_character_geometry import CharacterLayout
                    from dataclasses import asdict
                    try:
                        source = Path(line['line_crop_ref'])
                        if hashlib.sha256(source.read_bytes()).hexdigest() != line['line_crop_sha256']:
                            raise ValueError('CHARACTER_SOURCE_SHA_MISMATCH')
                        with Image.open(source) as image:
                            layout = CharacterLayout.from_recognition(value, image.size)
                        line['character_layout'] = {**asdict(layout), 'source_crop_sha256': line['line_crop_sha256']}
                    except (ValueError, TypeError) as exc:
                        line['character_layout_error'] = str(exc)
        self.regions_processed += len(crops)
        return result

    def recognize_direct(self, crop: dict[str, Any]) -> list[dict[str, Any]]:
        """Run rec on an eligible whole crop without invoking the detector."""

        self._ensure_loaded()
        path = Path(crop["path"])
        rec_results = list(self._rec_model.predict(str(path), batch_size=1))
        if len(rec_results) != 1:
            raise RuntimeError(
                "PP-OCR recognizer must return exactly one direct-rec result"
            )
        rec = _model_result_dict(rec_results[0])
        text = str(rec.get("rec_text") or "")
        confidence = float(rec.get("rec_score") or 0.0)
        self.regions_processed += 1
        if not text:
            return []
        bbox = [0.0, 0.0, float(crop["width"]), float(crop["height"])]
        return [
            {
                "text": text,
                "confidence": confidence,
                "det_confidence": None,
                "rec_confidence": confidence,
                "bbox_local_px": bbox,
                "bbox_page_pdf_pt": _local_px_to_page_pdf(bbox, crop),
                "polygon_local_px": None,
                "reading_sequence": 0,
                "line_crop_ref": str(path.resolve()),
                "line_crop_sha256": crop.get("content_sha256"),
                "direct_rec": True,
            }
        ]

    def close(self) -> None:
        if self._det_model is None and self._rec_model is None:
            return
        import gc

        import paddle

        self._det_model = None
        self._rec_model = None
        gc.collect()
        paddle.device.cuda.empty_cache()
        self.unload_count += 1

    def _ensure_loaded(self) -> None:
        if self._det_model is not None:
            return
        os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] = "True"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from .paddlex_runtime import import_paddlex_for_paddle_provider

        paddle, paddlex, create_model = import_paddlex_for_paddle_provider()

        from .model_registry import MODEL_CATALOG, ModelRegistry
        from .model_runtime import paddle_compatible_model_dir

        if (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.cuda.device_count() < 1
        ):
            raise RuntimeError("PDF_OCR_REQUIRES_GPU")
        paddle.set_device(self.device)
        registry = ModelRegistry(
            models_root=self.models_root,
            config_path=self.config_path,
            mcp_root=self.mcp_root,
        )
        det = registry.resolve("pp-ocrv6-medium-det", deep=True)
        rec = registry.resolve("pp-ocrv6-medium-rec", deep=True)
        self._det_model = create_model(
            MODEL_CATALOG[det.model_id].directory,
            model_dir=str(paddle_compatible_model_dir(det.model_root)),
            device=self.device,
        )
        self._rec_model = create_model(
            MODEL_CATALOG[rec.model_id].directory,
            model_dir=str(paddle_compatible_model_dir(rec.model_root)),
            device=self.device,
        )
        if not paddle.device.get_device().startswith("gpu"):
            self.close()
            raise RuntimeError("Paddle OCR runtime changed away from GPU")
        self._fingerprint = {
            "device": paddle.device.get_device(),
            "precision": "fp32",
            "paddlepaddle_gpu": paddle.__version__,
            "paddlex": paddlex.__version__,
            "det_model_id": det.model_id,
            "rec_model_id": rec.model_id,
            "det_model_fingerprint": det.manifest["model_fingerprint"],
            "rec_model_fingerprint": rec.manifest["model_fingerprint"],
            "det_manifest_verified": det.fingerprint_verified,
            "rec_manifest_verified": rec.fingerprint_verified,
            "cpu_fallback": False,
        }
        self.load_count += 1


def create_paddle_pdf_ocr_runtime(**kwargs) -> PaddleXPdfOcrRuntime:
    return PaddleXPdfOcrRuntime(**kwargs)


def create_production_pdf_ocr_runtime(**kwargs) -> PaddleXPdfOcrRuntime:
    del kwargs
    raise ProductionThreeModelInvariantError("PRODUCTION_THREE_MODEL_STAGE_REQUIRED")


def _formula_runtime_factory():
    from .formula_ocr import create_production_formulanet_runtime

    return create_production_formulanet_runtime(deep_model_validation=True)


def _model_result_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "keys") and hasattr(result, "get"):
        return {str(key): result.get(key) for key in result}
    raise TypeError(
        f"PaddleX model returned unsupported result type: {type(result).__name__}"
    )


def _plain_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        return value.tolist()
    return [item.tolist() if hasattr(item, "tolist") else list(item) for item in value]


def _image_png_bytes(image: Any) -> bytes:
    import io

    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _local_px_to_page_pdf(bbox: list[float], crop: dict[str, Any]) -> list[float]:
    origin = crop["bbox_pdf_pt"][:2]
    scale_x = float(crop["scale_x"])
    scale_y = float(crop["scale_y"])
    return [
        round(float(origin[0]) + float(bbox[0]) / scale_x, 6),
        round(float(origin[1]) + float(bbox[1]) / scale_y, 6),
        round(float(origin[0]) + float(bbox[2]) / scale_x, 6),
        round(float(origin[1]) + float(bbox[3]) / scale_y, 6),
    ]


def plan_page_content(
    fusion_page: dict[str, Any],
    page_record: dict[str, Any],
    source_evidence: dict[str, Any],
    *,
    table_adapter: str = "TABLE_DEFERRED_PRESERVE",
) -> dict[str, Any]:
    """Assign every Fusion input to one deterministic primary content route."""

    candidates = list(fusion_page.get("fusion_candidates", []))
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Fusion page contains duplicate candidate_id values")
    escalation = str(fusion_page.get("page_escalation", {}).get("status", "NONE"))
    routes: list[dict[str, Any]] = []
    if escalation == "VISUAL_PAGE_REVIEW":
        routes.append(
            _route_item(
                fusion_page=fusion_page,
                page_record=page_record,
                candidates=candidates,
                adapter="CONTENT_REVIEW_REQUIRED",
                output_kind="REVIEW",
                reason_codes=["PAGE_ESCALATION_REQUIRES_CONTENT_REVIEW"],
                input_kind="PAGE",
            )
        )
    elif escalation == "VISUAL_PAGE_REQUIRED":
        generic = []
        for candidate in candidates:
            semantic = _semantic(candidate)
            if semantic in {"FORMULA", "TABLE", "IMAGE", "IMAGE_LIKE"}:
                routes.append(
                    _candidate_route(
                        fusion_page,
                        page_record,
                        source_evidence,
                        candidate,
                        table_adapter=table_adapter,
                    )
                )
            else:
                generic.append(candidate)
        routes.append(
            _route_item(
                fusion_page=fusion_page,
                page_record=page_record,
                candidates=generic,
                adapter="PAGE_VISUAL_TEXT_RECOVERY",
                output_kind="TEXT",
                reason_codes=["VISUAL_PAGE_REQUIRED_TEXT_COMPLETENESS"],
                input_kind="PAGE",
            )
        )
    else:
        routes.extend(
            _candidate_route(
                fusion_page,
                page_record,
                source_evidence,
                candidate,
                table_adapter=table_adapter,
            )
            for candidate in candidates
        )
    routes = _merge_native_text_routes(
        routes,
        candidates=candidates,
        fusion_page=fusion_page,
        page_record=page_record,
        source_evidence=source_evidence,
    )
    routes = _merge_table_content_routes(routes, candidates, fusion_page, page_record)
    from .pdf.native_margin_titles import refine_native_margin_titles
    from .pdf.native_math import refine_native_math_routes

    routes, native_title_diagnostic = refine_native_margin_titles(
        routes, source_evidence, page_record,
    )

    routes, native_math_diagnostic = refine_native_math_routes(
        routes, source_evidence, page_record
    )
    from .pdf.graphics_render import protect_semantic_images_in_background_crops
    from .pdf.prose_edge_crop import annotate_recovered_prose
    from .pdf.image_containment import close_near_contained_image_extents
    from .pdf.ruled_writing_area import preserve_empty_ruled_writing_areas

    routes, ruled_area_diagnostic = preserve_empty_ruled_writing_areas(
        routes, source_evidence, page_record
    )
    close_near_contained_image_extents(routes)
    protect_semantic_images_in_background_crops(routes)
    annotate_recovered_prose(routes, source_evidence, page_record)
    routes.sort(
        key=lambda route: (
            ROUTE_PRIORITY[route["adapter"]],
            route["input_candidate_ids"],
            route["route_id"],
        )
    )
    assigned = [value for route in routes for value in route["input_candidate_ids"]]
    unrouted = sorted(set(candidate_ids) - set(assigned))
    duplicate_ids = sorted(
        candidate_id
        for candidate_id in set(assigned)
        if assigned.count(candidate_id) > 1
    )
    plan = {
        "schema": CONTENT_ROUTE_PLAN_SCHEMA,
        "document_id": str(fusion_page["document_id"]),
        "page_index": int(fusion_page["page_index"]),
        "page_escalation": escalation,
        "decision_version": ROUTER_VERSION,
        "routes": routes,
        "coverage": {
            "input_candidates_total": len(candidate_ids),
            "primary_routed": len(set(assigned)),
            "deferred": sum(
                len(route["input_candidate_ids"])
                for route in routes
                if route["adapter"] == "TABLE_DEFERRED_PRESERVE"
            ),
            "review": sum(
                max(1, len(route["input_candidate_ids"]))
                for route in routes
                if route["adapter"] == "CONTENT_REVIEW_REQUIRED"
            ),
            "unrouted": len(unrouted),
            "unrouted_candidate_ids": unrouted,
            "duplicate_primary_candidate_ids": duplicate_ids,
        },
        "provenance": {
            "fusion_schema": fusion_page.get("schema"),
            "fusion_strategy_id": fusion_page.get("strategy_id"),
            "page_source_profile": page_record.get("source_profile"),
            "native_text_trust": page_record.get("native_text_trust"),
            "source_evidence_supplied": bool(source_evidence),
            "native_math_geometry": native_math_diagnostic,
            "native_margin_titles": native_title_diagnostic,
            "empty_ruled_writing_areas": ruled_area_diagnostic,
        },
    }
    validate_content_route_plan(plan)
    return plan


def validate_content_route_plan(plan: dict[str, Any]) -> None:
    """Enforce the v0 exactly-one-primary-route coverage invariant."""

    if plan.get("schema") != CONTENT_ROUTE_PLAN_SCHEMA:
        raise ValueError("Unsupported ContentRoutePlan schema")
    coverage = plan.get("coverage", {})
    if coverage.get("unrouted") != 0:
        raise ValueError("ContentRoutePlan contains unrouted candidates")
    if coverage.get("duplicate_primary_candidate_ids"):
        raise ValueError("ContentRoutePlan contains duplicate primary assignments")
    route_ids = [route.get("route_id") for route in plan.get("routes", [])]
    if len(route_ids) != len(set(route_ids)) or any(not value for value in route_ids):
        raise ValueError("ContentRoutePlan route IDs must be present and unique")


def _candidate_route(
    fusion_page: dict[str, Any],
    page_record: dict[str, Any],
    source_evidence: dict[str, Any],
    candidate: dict[str, Any],
    *,
    table_adapter: str = "TABLE_DEFERRED_PRESERVE",
) -> dict[str, Any]:
    bbox = candidate.get("bbox_pdf_pt")
    if not _valid_bbox(bbox):
        return _route_item(
            fusion_page=fusion_page,
            page_record=page_record,
            candidates=[candidate],
            adapter="CONTENT_REVIEW_REQUIRED",
            output_kind="REVIEW",
            reason_codes=["INVALID_OR_ZERO_AREA_BBOX"],
        )
    semantic = _semantic(candidate)
    if semantic == "FORMULA":
        return _route_item(
            fusion_page=fusion_page,
            page_record=page_record,
            candidates=[candidate],
            adapter="FORMULA_RECOGNITION",
            output_kind="FORMULA",
            reason_codes=["SPECIALIZED_FORMULA_SEMANTIC_PRIORITY"],
        )
    if semantic == "TABLE":
        native_rows = _native_text_matches(candidate, source_evidence)
        if _native_index_evidence(
            page_record,
            source_evidence,
            native_rows,
            page_height=float(
                fusion_page.get("page_geometry", {}).get("height_pt") or 0
            ),
        ):
            return _route_item(
                fusion_page=fusion_page,
                page_record=page_record,
                candidates=[candidate],
                adapter="NATIVE_TEXT_BRIDGE",
                output_kind="TEXT",
                reason_codes=["NATIVE_INDEX_HEADING_AND_ENTRIES_OVERRIDE_LAYOUT_TABLE"],
                native_text_matches=native_rows,
            )
        return _route_item(
            fusion_page=fusion_page,
            page_record=page_record,
            candidates=[candidate],
            adapter=table_adapter,
            output_kind="TABLE",
            reason_codes=(
                ["SPECIALIZED_TABLE_ENGINE_PRIMARY"]
                if table_adapter == "TABLE_ENGINE"
                else ["TABLE_BRANCH_DEFERRED_PENDING_EXTERNAL_CORPUS"]
            ),
        )
    if semantic in {"IMAGE", "IMAGE_LIKE"}:
        from .pdf.graphics_render import native_prose_in_fallback

        matches = _native_image_matches(candidate, source_evidence)
        adapter = "IMAGE_NATIVE_EXTRACT" if len(matches) == 1 else "IMAGE_RENDER_CROP"
        reasons = (
            ["SINGLE_EXTRACTABLE_NATIVE_IMAGE_PLACEMENT"]
            if matches
            else [
                "NATIVE_IMAGE_MAPPING_UNAVAILABLE_OR_AMBIGUOUS",
                "RENDER_CROP_PRESERVES_VISUAL_EXTENT",
            ]
        )
        route = _route_item(
            fusion_page=fusion_page,
            page_record=page_record,
            candidates=[candidate],
            adapter=adapter,
            output_kind="IMAGE",
            reason_codes=reasons,
            native_image_matches=matches,
        )
        if adapter == "IMAGE_RENDER_CROP" and native_prose_in_fallback(candidate, source_evidence, page_record):
            route["provenance"]["source_graphics_without_native_prose"] = True
        return route
    if semantic in TEXT_SEMANTICS:
        native_rows = _native_text_matches(candidate, source_evidence)
        trust = str(page_record.get("native_text_trust", "NONE"))
        profile = str(page_record.get("source_profile", "UNCERTAIN"))
        safe_native = (
            trust in {"HIGH", "MEDIUM"}
            and profile not in {"IMAGE_ONLY", "IMAGE_WITH_TEXT_LAYER"}
            and bool(candidate.get("source_unit_ids") or native_rows)
        )
        return _route_item(
            fusion_page=fusion_page,
            page_record=page_record,
            candidates=[candidate],
            adapter="NATIVE_TEXT_BRIDGE" if safe_native else "OCR_TEXT_REGION",
            output_kind="TEXT",
            reason_codes=(
                ["TRUSTED_NATIVE_TEXT_WITH_COMPLETE_MAPPING"]
                if safe_native
                else ["NATIVE_TEXT_UNTRUSTED_OR_UNMAPPED", "LOCAL_VISUAL_TEXT_RECOVERY"]
            ),
            native_text_matches=native_rows,
        )
    route = _route_item(
        fusion_page=fusion_page,
        page_record=page_record,
        candidates=[candidate],
        adapter="CONTENT_REVIEW_REQUIRED",
        output_kind="REVIEW",
        reason_codes=["PRIMARY_ADAPTER_UNRESOLVED"],
    )
    from .pdf.graphics_render import native_prose_in_fallback

    if native_prose_in_fallback(candidate, source_evidence, page_record):
        route["provenance"]["source_graphics_without_native_prose"] = True
    return route


def _merge_native_text_routes(
    routes: list[dict[str, Any]],
    *,
    candidates: list[dict[str, Any]],
    fusion_page: dict[str, Any],
    page_record: dict[str, Any],
    source_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge Native Bridge routes that claim any shared source line/unit."""

    native = [route for route in routes if route["adapter"] == "NATIVE_TEXT_BRIDGE"]
    if len(native) < 2:
        return routes
    parents = list(range(len(native)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parents[max(left, right)] = min(left, right)

    owner: dict[str, int] = {}
    for index, route in enumerate(native):
        claims = {f"unit:{value}" for value in route["source_unit_ids"]}
        claims.update(
            f"evidence:{value}"
            for value in route["provenance"].get("native_text_evidence_ids", [])
        )
        for claim in claims:
            if claim in owner:
                union(index, owner[claim])
            else:
                owner[claim] = index
    groups: dict[int, list[dict[str, Any]]] = {}
    for index, route in enumerate(native):
        groups.setdefault(find(index), []).append(route)
    if all(len(group) == 1 for group in groups.values()):
        return routes
    candidate_by_id = {
        str(candidate["candidate_id"]): candidate for candidate in candidates
    }
    evidence_by_id = {
        str(row["evidence_id"]): row
        for row in source_evidence.get("native_text", [])
        if row.get("evidence_id")
    }
    merged = [route for route in routes if route["adapter"] != "NATIVE_TEXT_BRIDGE"]
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        candidate_ids = sorted(
            {value for route in group for value in route["input_candidate_ids"]}
        )
        evidence_ids = sorted(
            {
                value
                for route in group
                for value in route["provenance"].get("native_text_evidence_ids", [])
            }
        )
        reasons = {
            reason for route in group for reason in route["decision_reason_codes"]
        }
        reasons.add("SHARED_NATIVE_SOURCE_MERGED")
        merged.append(
            _route_item(
                fusion_page=fusion_page,
                page_record=page_record,
                candidates=[candidate_by_id[value] for value in candidate_ids],
                adapter="NATIVE_TEXT_BRIDGE",
                output_kind="TEXT",
                reason_codes=sorted(reasons),
                native_text_matches=[
                    evidence_by_id[value]
                    for value in evidence_ids
                    if value in evidence_by_id
                ],
            )
        )
    return merged


def _merge_table_content_routes(routes, candidates, fusion_page, page_record):
    """Keep contained cell formulas in their table's recognition context.

    Independent formula routes can mask cell pixels before table OCR and emit
    headers outside the table. Preserve their source evidence under the table
    owner, also when table recognition is deferred. Competing whole-region
    formulas and partial overlaps must retain independent recognition.
    """
    by_id = {str(c['candidate_id']): c for c in candidates}
    tables = [r for r in routes if r['adapter'] in {'TABLE_ENGINE', 'TABLE_DEFERRED_PRESERVE'}]
    owned = {}
    for child in routes:
        if child['adapter'] != 'FORMULA_RECOGNITION':
            continue
        inner = child['provenance'].get('bbox_pdf_pt')
        if not _valid_bbox(inner):
            continue
        owners = []
        for table in tables:
            outer = table['provenance'].get('bbox_pdf_pt')
            if not _valid_bbox(outer):
                continue
            inner_area = (inner[2]-inner[0])*(inner[3]-inner[1])
            outer_area = (outer[2]-outer[0])*(outer[3]-outer[1])
            if (inner_area < outer_area * .5
                    and outer[0] <= inner[0] and outer[1] <= inner[1]
                    and outer[2] >= inner[2] and outer[3] >= inner[3]):
                owners.append((outer_area, table['route_id']))
        if owners:
            owned.setdefault(min(owners)[1], []).append(child)
    child_ids = {c['route_id'] for children in owned.values() for c in children}
    result = []
    for route in routes:
        if route['route_id'] in child_ids:
            continue
        children = owned.get(route['route_id'], [])
        if not children:
            result.append(route)
            continue
        ids = route['input_candidate_ids'] + [cid for child in children for cid in child['input_candidate_ids']]
        merged = _route_item(
            fusion_page=fusion_page, page_record=page_record,
            candidates=[by_id[cid] for cid in ids], adapter=route['adapter'],
            output_kind='TABLE', reason_codes=route['decision_reason_codes'] + ['TABLE_OWNS_CONTAINED_CELL_FORMULAS'],
        )
        merged['provenance']['table_content_ownership'] = {
            'schema': 'bemarkdown-table-content-ownership-v1',
            'basis': 'STRICTLY_CONTAINED_SMALLER_FORMULA_REGION',
            'children': [
                {'source_candidate_ids': child['input_candidate_ids'],
                 'source_region_ids': child['source_region_ids'],
                 'bbox_pdf_pt': child['provenance']['bbox_pdf_pt'],
                 'original_route_id': child['route_id']}
                for child in children
            ],
            'recognition_owner': 'TABLE',
        }
        result.append(merged)
    for route in result:
        if route['adapter'] not in {'TABLE_ENGINE', 'TABLE_DEFERRED_PRESERVE'}:
            continue
        hints = [hint for cid in route['input_candidate_ids']
                 for hint in by_id[cid].get('canonical_region', {}).get('provenance', {}).get(
                     'table_formula_review_hints', [])]
        if hints:
            ownership = route['provenance'].setdefault('table_content_ownership', {
                'schema': 'bemarkdown-table-content-ownership-v1',
                'basis': 'TABLE_SCOPED_FORMULA_REVIEW_EVIDENCE',
                'children': [], 'recognition_owner': 'TABLE'})
            ownership['children'].extend(copy.deepcopy(hints))
    return result


def _boundary_review_evidence(candidates, page_record):
    reviews = []
    for candidate in candidates:
        region = candidate.get('canonical_region', {})
        source = region.get('bbox_pdf_pt')
        pixels = region.get('bbox_render_px')
        if not _valid_bbox(source) or not _valid_bbox(pixels):
            continue
        for key, kind in [('diagram_boundary_refinement', 'DIAGRAM_BOUNDARY'),
                          ('table_boundary_refinement', 'TABLE_BOUNDARY')]:
            proof = region.get('provenance', {}).get(key) or {}
            if proof.get('status') != 'REVIEW_REQUIRED':
                continue
            proposed = proof.get('refined_bbox_render_px')
            projected = list(source)
            if _valid_bbox(proposed):
                scales = [(source[2]-source[0])/(pixels[2]-pixels[0]),
                          (source[3]-source[1])/(pixels[3]-pixels[1])]
                projected = [source[i % 2] + (proposed[i]-pixels[i % 2])*scales[i % 2] for i in range(4)]
            scope = [min(source[0],projected[0]), min(source[1],projected[1]),
                     max(source[2],projected[2]), max(source[3],projected[3])]
            identity = [page_record.get('document_id'), page_record['page_index'], region.get('region_id'), key, proof]
            reviews.append(dict(
                review_id='region-review-' + hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24],
                kind=kind, status='REVIEW_REQUIRED', page_index=page_record['page_index'],
                source_region_id=region.get('region_id'), source_candidate_id=candidate.get('candidate_id'),
                source_bbox_pdf_pt=scope, original_bbox_pdf_pt=copy.deepcopy(source),
                proposed_bbox_pdf_pt=projected, boundary_evidence=copy.deepcopy(proof),
                model_score=region.get('score'), recognition_success=False, text_recognition_requested=False))
    return reviews


def _route_item(
    *,
    fusion_page: dict[str, Any],
    page_record: dict[str, Any],
    candidates: list[dict[str, Any]],
    adapter: str,
    output_kind: str,
    reason_codes: list[str],
    input_kind: str = "REGION",
    native_text_matches: list[dict[str, Any]] | None = None,
    native_image_matches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_ids = sorted(str(candidate["candidate_id"]) for candidate in candidates)
    identity = {
        "document_id": str(fusion_page["document_id"]),
        "page_index": int(fusion_page["page_index"]),
        "input_kind": input_kind,
        "input_candidate_ids": candidate_ids,
        "adapter": adapter,
        "decision_reason_codes": sorted(set(reason_codes)),
        "decision_version": ROUTER_VERSION,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20]
    source_region_ids = sorted(
        {
            str(value)
            for candidate in candidates
            for value in (candidate.get("source_region_id"), candidate.get("region_id"))
            if value
        }
    )
    source_unit_ids = sorted(
        {
            str(value)
            for candidate in candidates
            for value in candidate.get("source_unit_ids", [])
        }
    )
    native_text_matches = native_text_matches or []
    if not source_unit_ids:
        source_unit_ids = [
            f"native-evidence:{row['evidence_id']}"
            for row in native_text_matches
            if row.get("evidence_id")
        ]
    route = {
        "route_id": f"content-route-{digest}",
        "document_id": identity["document_id"],
        "page_index": identity["page_index"],
        "input_kind": input_kind,
        "input_candidate_ids": candidate_ids,
        "source_region_ids": source_region_ids,
        "source_unit_ids": source_unit_ids,
        "semantic_evidence": sorted(
            {
                str(value)
                for candidate in candidates
                for value in (candidate.get("semantic_hint"), _semantic(candidate))
                if value
            }
        ),
        "page_escalation": str(
            fusion_page.get("page_escalation", {}).get("status", "NONE")
        ),
        "adapter": adapter,
        "output_kind": output_kind,
        "decision_reason_codes": identity["decision_reason_codes"],
        "decision_version": ROUTER_VERSION,
        "requires_gpu": adapter
        in {
            "OCR_TEXT_REGION",
            "PAGE_VISUAL_TEXT_RECOVERY",
            "FORMULA_RECOGNITION",
            "TABLE_ENGINE",
        },
        "provenance": {
            "candidate_kinds": sorted(
                str(candidate.get("candidate_kind")) for candidate in candidates
            ),
            "bbox_pdf_pt": _union_bbox(
                [
                    candidate.get("bbox_pdf_pt")
                    for candidate in candidates
                    if _valid_bbox(candidate.get("bbox_pdf_pt"))
                ]
            ),
            "evidence_ids": sorted(
                {
                    str(value)
                    for candidate in candidates
                    for value in candidate.get("evidence_ids", [])
                }
            ),
            "native_text_evidence_ids": sorted(
                str(row["evidence_id"])
                for row in native_text_matches
                if row.get("evidence_id")
            ),
            "native_image_placements": [
                {
                    "evidence_id": row.get("evidence_id"),
                    "xref": row.get("xref"),
                    "object_identity": row.get("object_identity"),
                "bbox_pdf_pt": row.get("bbox_pdf_pt"),
                "native_appearance": row.get("native_appearance", {}),
                }
                for row in native_image_matches or []
            ],
            "source_path": page_record.get("source_path"),
            "native_text_trust": str(page_record.get("native_text_trust", "NONE")),
            "source_profile": str(page_record.get("source_profile", "UNCERTAIN")),
        },
    }

    boundary_reviews = _boundary_review_evidence(candidates, page_record)
    if boundary_reviews:
        route['provenance']['boundary_reviews'] = boundary_reviews
    if "NATIVE_INDEX_HEADING_AND_ENTRIES_OVERRIDE_LAYOUT_TABLE" in reason_codes:
        route["provenance"]["layout_semantic_evidence"] = list(
            route["semantic_evidence"]
        )
        route["semantic_evidence"] = ["TEXT"]
    return route


def _semantic(candidate: dict[str, Any]) -> str:
    canonical = candidate.get("canonical_region") or {}
    return str(
        candidate.get("semantic_type")
        or canonical.get("semantic_type")
        or candidate.get("semantic_hint")
        or "UNKNOWN"
    )


def _native_index_evidence(page_record, source_evidence, native_rows, *, page_height):
    """A titled subject index is textual navigation, even if layout says table."""
    if str(page_record.get("native_text_trust")) not in {"HIGH", "MEDIUM"}:
        return False
    if page_record.get("source_profile") in {"IMAGE_ONLY", "IMAGE_WITH_TEXT_LAYER"}:
        return False
    rows = source_evidence.get("native_text", [])
    headings = {"索引", "名词索引", "关键词索引", "index", "subjectindex"}
    title_found = any(
        re.sub(r"\s+", "", str(row.get("text", ""))).casefold() in headings
        and _valid_bbox(row.get("bbox_pdf_pt"))
        and row["bbox_pdf_pt"][1] < page_height * 0.2
        for row in rows
    )
    if not title_found or len(native_rows) < 12:
        return False
    text = " ".join(str(row.get("text", "")) for row in rows)
    page_key = "页码" in text or bool(
        re.search(r"\bpage\s+numbers?\b", text, re.IGNORECASE)
    )
    entries = [re.sub(r"\s+", "", str(row.get("text", ""))) for row in native_rows]
    return (
        page_key
        and sum(bool(re.fullmatch(r"[A-Z]", value)) for value in entries) >= 3
        and sum(value.isdigit() for value in entries) >= 3
        and sum(len(value) >= 2 and not value.isdigit() for value in entries) >= 3
    )


def _native_text_matches(
    candidate: dict[str, Any], source_evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    evidence_ids = {
        str(value)
        for key in ("evidence_ids", "source_line_ids")
        for value in candidate.get(key, [])
    }
    bbox = candidate.get("bbox_pdf_pt")
    rows = []
    for row in source_evidence.get("native_text", []):
        if (
            str(row.get("evidence_id")) in evidence_ids
            or _overlap_fraction(bbox, row.get("bbox_pdf_pt")) >= 0.5
        ):
            rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("evidence_id", "")))


def _native_image_matches(
    candidate: dict[str, Any], source_evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    matches = []
    for row in source_evidence.get("native_images", []):
        same_extent = (
            _bbox_iou(candidate.get("bbox_pdf_pt"), row.get("bbox_pdf_pt")) >= 0.85
        )
        if (
            same_extent
            and row.get("xref") is not None
            and row.get("object_identity")
            and row.get("native_appearance", {}).get("verified") is True
        ):
            matches.append(row)
    return sorted(matches, key=lambda row: str(row.get("evidence_id", "")))


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _bbox_iou(first: Any, second: Any) -> float:
    if not _valid_bbox(first) or not _valid_bbox(second):
        return 0.0
    intersection = _intersection_area(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union else 0.0


def _overlap_fraction(first: Any, second: Any) -> float:
    if not _valid_bbox(first) or not _valid_bbox(second):
        return 0.0
    return _intersection_area(first, second) / min(_area(first), _area(second))


def _intersection_area(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _area(bbox: list[float]) -> float:
    return (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))


def _union_bbox(boxes: list[Any]) -> list[float] | None:
    valid = [box for box in boxes if _valid_bbox(box)]
    if not valid:
        return None
    return [
        min(float(box[0]) for box in valid),
        min(float(box[1]) for box in valid),
        max(float(box[2]) for box in valid),
        max(float(box[3]) for box in valid),
    ]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_id(route: dict[str, Any]) -> str:
    identity = {"route_id": route["route_id"], "schema": REGION_CONTENT_IR_SCHEMA}
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20]
    return f"content-{digest}"


def _default_quality_status(status: str) -> str:
    return {
        "SUCCESS": "CONTENT_OK",
        "SUCCESS_WITH_WARNING": "CONTENT_WARNING",
        "REVIEW_REQUIRED": "CONTENT_REVIEW",
        "DEFERRED": "CONTENT_DEFERRED",
        "FAILED_PRESERVE_INPUT": "CONTENT_FAILED_PRESERVE_INPUT",
    }.get(status, "CONTENT_REVIEW")


def extract_pdf_source_evidence(
    page_records: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Read immutable PDFs and return line/image evidence used by content adapters."""

    from collections import defaultdict

    import fitz

    from .pdf.native_font_unicode import native_font_unicode_issues, normalized_font_name

    by_source: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for record in page_records:
        by_source[Path(record["source_path"]).resolve()].append(record)
    result = {}
    for source_path, records in sorted(
        by_source.items(), key=lambda item: str(item[0])
    ):
        with fitz.open(source_path) as document:
            font_unicode_cache = {}
            for record in sorted(records, key=lambda row: int(row["page_index"])):
                page = document[int(record["page_index"])]
                font_unicode_issues = native_font_unicode_issues(page, font_unicode_cache)
                native_text = []
                for block_index, block in enumerate(
                    page.get_text("dict").get("blocks", [])
                ):
                    if block.get("type") != 0 or not _valid_bbox(block.get("bbox")):
                        continue
                    for line_index, line in enumerate(block.get("lines", [])):
                        spans = [
                            span
                            for span in line.get("spans", [])
                            if _valid_bbox(span.get("bbox"))
                        ]
                        if not spans:
                            continue
                        bbox = line.get("bbox")
                        if not _valid_bbox(bbox):
                            bbox = _union_bbox([span["bbox"] for span in spans])
                        line_id = f"text-block-{block_index:04d}-line-{line_index:04d}"
                        native_text.append(
                            {
                                "evidence_id": line_id,
                                "source_block_id": f"text-block-{block_index:04d}",
                                "source_line_span_ids": [
                                    f"{line_id}-span-{span_index:04d}"
                                    for span_index, _span in enumerate(spans)
                                ],
                                "spans": [
                                    {
                                        "span_id": f"{line_id}-span-{span_index:04d}",
                                        "bbox_pdf_pt": [
                                            float(value) for value in span["bbox"]
                                        ],
                                        "text": str(span.get("text", "")),
                                        **({"unicode_mapping_issue": dict(font_unicode_issues[
                                            normalized_font_name(str(span.get("font", "")))
                                        ])} if normalized_font_name(str(span.get("font", "")))
                                           in font_unicode_issues else {}),
                                    }
                                    for span_index, span in enumerate(spans)
                                ],
                                "bbox_pdf_pt": [float(value) for value in bbox],
                                "native_char_count": sum(
                                    len(str(span.get("text", ""))) for span in spans
                                ),
                                "text": "".join(
                                    str(span.get("text", "")) for span in spans
                                ),
                                "source_unit": "LINE",
                            }
                        )
                native_images = [
                    {
                        "evidence_id": f"image-{index:04d}-xref-{placement['xref']}",
                        "bbox_pdf_pt": [float(value) for value in placement["bbox"]],
                        "xref": placement["xref"],
                        "object_identity": placement["object_identity"],
                        "directly_extractable": placement.get("directly_extractable"),
                "extension": placement.get("extension"),
                "native_appearance": placement.get("native_appearance", {}),
                    }
                    for index, placement in enumerate(
                        record.get("images", {}).get("placements", [])
                    )
                    if _valid_bbox(placement.get("bbox"))
                ]
                result[(str(record["document_id"]), int(record["page_index"]))] = {
                    "native_text": native_text,
                    "native_images": native_images,
                    "page_visual_bbox": [
                        0.0,
                        0.0,
                        float(record["geometry"]["width_pt"]),
                        float(record["geometry"]["height_pt"]),
                    ],
                }
    return result
