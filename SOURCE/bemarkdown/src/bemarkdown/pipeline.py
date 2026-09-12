from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

from .asset_contract import AssetFinalizer
from .mtef_cache import MtefCacheContext
from .package import DocxResourceLimits, PackageIndex
from .scanner import DocumentScanner
from .serializer import write_markdown


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    output_dir: Path
    markdown_path: Path
    report_path: Path
    debug_ir_path: Path | None
    report: dict


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _new_report(source: Path) -> dict:
    return {
        "schema": "bemarkdown-asset-contract-v1",
        "source": {
            "path": str(source.resolve()),
            "file_name": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "runtime": {
            "python": platform.python_version(),
            "dependencies": {
                "lxml": _version("lxml"),
                "olefile": _version("olefile"),
                "omml2latex": "0.1.1 vendored (Python 3.11 fix)",
                "mathtypejx": _version("mathtypejx"),
                "mathml2latex": _version("mathml2latex"),
            },
            "vision_or_ocr_used": False,
        },
        "input_validation": {
            "status": "PENDING",
            "resource_limits": {},
            "resource_profile": {},
            "external_relationship_policy": "PROVENANCE_ONLY_NO_FETCH",
            "xml_policy": "NO_DTD_NO_ENTITIES_NO_NETWORK",
        },
        "timing": {
            "package_read_seconds": 0.0,
            "scan_seconds": 0.0,
            "formula_conversion_seconds": 0.0,
            "omml_seconds": 0.0,
            "eq_seconds": 0.0,
            "latex_alt_seconds": 0.0,
            "ole_mtef_seconds": 0.0,
            "ole_relationship_lookup_seconds": 0.0,
            "ole_bytes_access_seconds": 0.0,
            "ole_open_seconds": 0.0,
            "equation_stream_lookup_seconds": 0.0,
            "equation_stream_read_seconds": 0.0,
            "mtef_payload_extraction_seconds": 0.0,
            "mtef_sha_seconds": 0.0,
            "mtef_parse_seconds": 0.0,
            "mtef_to_mathml_seconds": 0.0,
            "mathml_to_latex_seconds": 0.0,
            "mtef_cache_lookup_seconds": 0.0,
            "mtef_l2_read_seconds": 0.0,
            "mtef_l2_write_seconds": 0.0,
            "wmf_inspect_seconds": 0.0,
            "wmf_semantic_recovery_seconds": 0.0,
            "formula_validation_seconds": 0.0,
            "wmf_render_seconds": 0.0,
            "asset_export_seconds": 0.0,
            "markdown_serialization_seconds": 0.0,
            "report_serialization_seconds": 0.0,
            "formula_ocr_model_load_seconds": 0.0,
            "formula_ocr_inference_seconds": 0.0,
            "formula_ocr_validation_seconds": 0.0,
            "formula_ocr_safety_gate_seconds": 0.0,
            "total_seconds": 0.0,
        },
        "document": {
            "paragraphs": 0,
            "headings": 0,
            "list_items": 0,
            "tables": 0,
            "alternate_content_selected": {"Choice": 0, "Fallback": 0},
        },
        "formulas": {
            "total": 0,
            "omml": 0,
            "eq": 0,
            "mtef": 0,
            "wmf_embedded": 0,
            "latex_alt": 0,
            "success_exact": 0,
            "success_normalized": 0,
            "success_approximate": 0,
            "rendered_fallback": 0,
            "failed_preserved": 0,
            "equation_candidates": 0,
            "real_formulas": 0,
            "empty_placeholders": 0,
            "empty_placeholders_suppressed": 0,
            "non_formula_objects": 0,
            "unresolved_candidates": 0,
            "structural_success": 0,
            "structural_detected": 0,
            "structural_valid": 0,
            "structural_suspicious": 0,
            "structural_invalid": 0,
            "structural_non_formula": 0,
            "failed": 0,
            "candidate_conservation_ok": False,
            "real_formula_conservation_ok": False,
            "candidate_records": [],
            "duplicate_preview_suppressed": 0,
            "wmf_previews_discovered": 0,
            "preview_suppressed_by_ole": 0,
            "wmf_inspect_calls": 0,
            "wmf_embedded_recovery_calls": 0,
            "wmf_render_calls": 0,
            "conservation_ok": False,
            "records": [],
        },
        "assets": {
            "source_image_objects": 0,
            "exported_assets": 0,
            "formula_previews_suppressed": 0,
            "failed_assets": 0,
            "records": [],
        },
        "drawingml": {
            "groups_total": 0,
            "visual_groups": 0,
            "pure_text_groups": 0,
            "decorative_or_empty_groups": 0,
            "unsupported_groups": 0,
            "visible_labels_total": 0,
            "visible_labels_accounted": 0,
            "records": [],
        },
        "headers": {
            "policy": "include_once",
            "title_candidates": 0,
            "titles_inserted": 0,
            "titles_deduplicated": 0,
            "watermark_or_visual_excluded": 0,
            "records": [],
        },
        "tables": {
            "tables": 0,
            "simple_markdown_tables": 0,
            "html_tables": 0,
            "failed_tables": 0,
        },
        "formula_ocr": {
            "enabled": False,
            "candidates": 0,
            "unique_png": 0,
            "inference_calls": 0,
            "batch_count": 0,
            "inferred_unique_png": 0,
            "deduplicated_occurrences": 0,
            "cache_hits": 0,
            "accepted": 0,
            "accepted_with_warning": 0,
            "review_required": 0,
            "rejected_preserved": 0,
            "inference_failed_preserved": 0,
            "validator_issues": {},
            "safety_gate_issues": {},
            "records": [],
            "conservation_ok": True,
        },
        "mtef_cache": {
            "mode": None,
            "worker_count": 1,
            "contract_fingerprint": None,
            "occurrences": 0,
            "unique_mtef_sha256": 0,
            "duplicates": 0,
            "misses": 0,
            "full_conversion_calls": 0,
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l2_writes": 0,
            "l2_corruptions": 0,
            "l2_write_failures": 0,
            "non_cacheable": 0,
            "_payload_counts": {},
        },
        "quality_status": "CLEAN",
        "unsupported": [],
        "warnings": [],
        "errors": [],
    }


def convert_docx(
    source: str | Path,
    output: str | Path,
    debug: bool = False,
    *,
    formula_ocr: str = "off",
    formula_ocr_adapter=None,
    figure_text_adapter=None,
    mtef_cache: str = "memory",
    mtef_cache_context: MtefCacheContext | None = None,
    mtef_cache_dir: str | Path | None = None,
    resource_limits: DocxResourceLimits | None = None,
) -> ConversionResult:
    started = time.perf_counter()
    source_path = Path(source)
    output_dir = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _new_report(source_path)
    if formula_ocr not in {"off", "auto"}:
        raise ValueError("formula_ocr must be 'off' or 'auto'")
    if mtef_cache not in {"off", "memory", "persistent"}:
        raise ValueError("mtef_cache must be 'off', 'memory', or 'persistent'")
    if formula_ocr == "auto":
        report["formula_ocr"]["enabled"] = True

    package = PackageIndex(source_path, limits=resource_limits)
    report["timing"]["package_read_seconds"] = package.read_seconds
    report["input_validation"] = {
        "status": "PASS",
        "resource_limits": package.limits.to_dict(),
        "resource_profile": package.resource_profile.to_dict(),
        "external_relationship_policy": "PROVENANCE_ONLY_NO_FETCH",
        "xml_policy": "NO_DTD_NO_ENTITIES_NO_NETWORK",
    }
    report["warnings"].extend(package.warnings)

    scan_started = time.perf_counter()
    cache_context = mtef_cache_context or MtefCacheContext(
        mode=mtef_cache, persistent_dir=mtef_cache_dir
    )
    report["mtef_cache"]["mode"] = cache_context.mode
    report["mtef_cache"]["contract_fingerprint"] = cache_context.contract_fingerprint
    document = DocumentScanner(
        package, output_dir, report, mtef_cache_context=cache_context
    ).scan()
    report["timing"]["scan_seconds"] = time.perf_counter() - scan_started

    if formula_ocr == "auto":
        if formula_ocr_adapter is None:
            from .formula_ocr import FormulaOcrAdapter

            formula_ocr_adapter = FormulaOcrAdapter()
        formula_ocr_adapter.process(document, output_dir, report)

    if figure_text_adapter is not None:
        figure_text_adapter.process(
            document, output_dir, report,
            formula_runtime=getattr(formula_ocr_adapter, 'loaded_runtime', None))

    AssetFinalizer(output_dir, report["source"]["sha256"], report).finalize(document)

    markdown_path = output_dir / "document.md"
    serialize_started = time.perf_counter()
    write_markdown(document, markdown_path)
    report["timing"]["markdown_serialization_seconds"] = (
        time.perf_counter() - serialize_started
    )

    debug_path = None
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "document_ir.json"
        debug_path.write_text(
            json.dumps(document.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )

    _finalize_formula_report(report)
    _finalize_mtef_cache_report(report)

    report["timing"]["total_seconds"] = time.perf_counter() - started
    report_path = output_dir / "conversion_report.json"
    report_started = time.perf_counter()
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report["timing"]["report_serialization_seconds"] = (
        time.perf_counter() - report_started
    )
    report["timing"]["total_seconds"] = time.perf_counter() - started
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return ConversionResult(
        source_path, output_dir, markdown_path, report_path, debug_path, report
    )


def _finalize_mtef_cache_report(report: dict) -> None:
    cache = report["mtef_cache"]
    payload_counts = cache.pop("_payload_counts", {})
    cache["unique_mtef_sha256"] = len(payload_counts)
    cache["duplicates"] = max(0, cache["occurrences"] - len(payload_counts))
    cache["top_repeated_payloads"] = [
        {"sha256_prefix": sha[:16], "occurrences": count}
        for sha, count in sorted(
            payload_counts.items(), key=lambda item: (-item[1], item[0])
        )[:20]
        if count > 1
    ]


def _finalize_formula_report(report: dict) -> None:
    formulas = report["formulas"]
    # ``structural_valid`` is a validator verdict count, not the number of
    # structural routes that reached a terminal success state.  A recovered
    # formula can remain a structural final with a conservative validator
    # warning/suspicion (notably some frozen LaTeX-alt records).  Conservation
    # therefore has to follow the mutually exclusive conversion statuses.
    formulas["structural_success"] = sum(
        formulas[key]
        for key in (
            "success_exact",
            "success_normalized",
            "success_approximate",
        )
    )
    formulas["failed"] = formulas["failed_preserved"]
    classified_candidates = (
        formulas["real_formulas"]
        + formulas["empty_placeholders"]
        + formulas["non_formula_objects"]
        + formulas["unresolved_candidates"]
    )
    formulas["candidate_conservation_ok"] = (
        formulas["equation_candidates"] == classified_candidates
    )
    terminal_count = (
        formulas["structural_success"]
        + formulas["rendered_fallback"]
        + formulas["failed"]
    )
    formulas["real_formula_conservation_ok"] = (
        formulas["real_formulas"] == terminal_count
    )
    formulas["total"] = formulas["real_formulas"]
    formulas["conservation_ok"] = (
        formulas["candidate_conservation_ok"]
        and formulas["real_formula_conservation_ok"]
    )
    if not formulas["conservation_ok"]:
        report["errors"].append(
            "Formula conservation failed: "
            f"candidates={formulas['equation_candidates']} "
            f"classified={classified_candidates} real={formulas['real_formulas']} "
            f"terminal={terminal_count}"
        )
    formula_ocr = report.get("formula_ocr") or {}
    if formula_ocr.get("enabled"):
        ocr_terminal = sum(
            formula_ocr.get(key, 0)
            for key in (
                "accepted",
                "accepted_with_warning",
                "review_required",
                "rejected_preserved",
                "inference_failed_preserved",
            )
        )
        formula_ocr["conservation_ok"] = ocr_terminal == formula_ocr.get(
            "candidates", 0
        )
        formula_ocr["real_formula_conservation_ok"] = (
            formulas["real_formulas"]
            == formulas["structural_success"]
            + ocr_terminal
            + formulas["failed_preserved"]
        )
        if not (
            formula_ocr["conservation_ok"]
            and formula_ocr["real_formula_conservation_ok"]
        ):
            report["errors"].append(
                "Formula OCR conservation failed: "
                f"candidates={formula_ocr.get('candidates', 0)} "
                f"terminal={ocr_terminal} real={formulas['real_formulas']}"
            )
    if report["errors"]:
        report["quality_status"] = "FAILED"
    elif (
        report.get("assets", {}).get("unresolved", 0)
        or report.get("drawingml", {}).get("unsupported_groups", 0)
        or report.get("figure_text", {}).get("review_required", 0)
        or report.get("image_content", {}).get("review_required", 0)
        or formula_ocr.get("enabled")
        and any(
            formula_ocr.get(key, 0)
            for key in (
                "review_required",
                "rejected_preserved",
                "inference_failed_preserved",
            )
        )
    ):
        report["quality_status"] = "COMPLETED_WITH_REVIEW_ITEMS"
    elif formula_ocr.get("enabled") and formula_ocr.get("accepted_with_warning", 0) or report.get("warnings"):
        report["quality_status"] = "COMPLETED_WITH_WARNINGS"
    else:
        report["quality_status"] = "CLEAN"
