"""Developer-only census, benchmark, and validation commands.

The installed ``bemarkdown`` console entry uses :mod:`bemarkdown.cli`, whose
interface is intentionally limited to the four production commands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .census import revalidate_wmf_samples, run_wmf_census
from .config import OutputRootConfigurationError
from .doctor import run_doctor
from .docx_census import run_docx_formula_census
from .docx_edge_census import run_docx_edge_census
from .formula_gt_evaluation import (
    build_reference_ground_truth,
    evaluate_formula_reference,
)
from .formula_ocr import FormulaOcrAdapter
from .formulanet_benchmark import run_formulanet_benchmark
from .mtef_cache import MtefCacheContext
from .package import InvalidDocxError
from .phase3b_integration import (
    evaluate_reference_safety_gate,
    run_batch2_stability,
    run_formula_ocr_integration,
)
from .phase3c_benchmark import run_phase3c_benchmark
from .phase3d2_validation import run_phase3d2_validation
from .phase4a_validation import run_phase4a_validation
from .phase4b_validation import run_phase4b_validation
from .production import (
    ExistingPackageError,
    PackageValidationError,
    cleanup_staging,
    convert_document,
    validate_package,
)
from .regression import run_docx_regression

EXIT_SUCCESS = 0
EXIT_INVALID_INPUT = 2
EXIT_ENVIRONMENT = 3
EXIT_CONVERSION_FAILURE = 4
EXIT_PACKAGE_FAILURE = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bemarkdown")
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser(
        "convert", help="Produce one atomic BeMarkdown Package v1"
    )
    convert.add_argument("input", type=Path)
    convert.add_argument(
        "--output-root",
        "--output",
        "-o",
        dest="output_root",
        type=Path,
        help="Package root; defaults through BeMarkdown workspace configuration",
    )
    convert.add_argument("--debug", action="store_true")
    convert.add_argument(
        "--json",
        action="store_true",
        help="Print only the machine-readable production result to stdout",
    )
    convert.add_argument(
        "--formula-ocr",
        choices=("off", "auto"),
        default="auto",
        help="FormulaNet last-resort fallback; production default is auto",
    )
    convert.add_argument(
        "--mtef-cache",
        choices=("off", "memory", "persistent"),
        default="persistent",
        help="OLE/MTEF cache mode; production default is persistent",
    )
    convert.add_argument("--mtef-cache-dir", type=Path)
    convert.add_argument("--models-root", type=Path)
    convert.add_argument("--mcp-root", type=Path)
    convert.add_argument("--config", type=Path)
    convert.add_argument(
        "--no-replace",
        action="store_true",
        help="Fail if the deterministic document_id package already exists",
    )
    validate = subparsers.add_parser(
        "validate", help="Validate a complete BeMarkdown Package v1"
    )
    validate.add_argument("package", type=Path)
    validate.add_argument("--json", action="store_true")
    cleanup = subparsers.add_parser(
        "cleanup-staging", help="Remove only stale BeMarkdown staging jobs"
    )
    cleanup.add_argument("--output-root", type=Path)
    cleanup.add_argument("--older-than-hours", type=float, default=24.0)
    cleanup.add_argument("--json", action="store_true")
    doctor = subparsers.add_parser(
        "doctor", help="Report full, degraded, or not-ready Tool capability"
    )
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--deep", action="store_true")
    doctor.add_argument("--models-root", type=Path)
    doctor.add_argument("--mcp-root", type=Path)
    doctor.add_argument("--output-root", type=Path)
    doctor.add_argument("--tool-root", type=Path)
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--skip-formula-runtime", action="store_true")
    doctor.add_argument("--skip-model-suite-runtime", action="store_true")
    census = subparsers.add_parser(
        "wmf-census", help="Run a SHA-deduplicated deterministic WMF census"
    )
    census.add_argument("manifest", type=Path)
    census.add_argument("--output", "-o", type=Path, required=True)
    census.add_argument("--audit-limit", type=int, default=64)
    samples = subparsers.add_parser(
        "revalidate-wmf-samples", help="Re-run a Phase 2 WMF sample summary"
    )
    samples.add_argument("summary", type=Path)
    samples.add_argument("--output", "-o", type=Path, required=True)
    regression = subparsers.add_parser(
        "regress-docx", help="Re-run baseline DOCX cases and compare OMML exactly"
    )
    regression.add_argument("baseline", type=Path)
    regression.add_argument("--output", "-o", type=Path, required=True)
    docx_census = subparsers.add_parser(
        "docx-formula-census",
        help="Run production formula routing over every manifest DOCX",
    )
    docx_census.add_argument("manifest", type=Path)
    docx_census.add_argument("--output", "-o", type=Path, required=True)
    docx_census.add_argument("--phase25-census", type=Path)
    docx_census.add_argument("--audit-limit", type=int, default=80)
    edge_census = subparsers.add_parser(
        "docx-edge-census",
        help="Audit edge OOXML structures without changing production conversion",
    )
    edge_census.add_argument("manifest", type=Path)
    edge_census.add_argument("--output", "-o", type=Path, required=True)
    formulanet = subparsers.add_parser(
        "formulanet-benchmark",
        help="Benchmark PP-FormulaNet_plus-L on Phase 2.6 OCR candidates",
    )
    formulanet.add_argument("source", type=Path)
    formulanet.add_argument("--output", "-o", type=Path, required=True)
    formulanet.add_argument("--model-dir", type=Path)
    formulanet.add_argument("--model-revision")
    formulanet.add_argument("--cache", type=Path)
    formulanet.add_argument(
        "--batch-size",
        dest="batch_sizes",
        action="append",
        type=int,
        default=[],
        help="Optional comparison batch size; repeat for multiple values",
    )
    reference_build = subparsers.add_parser(
        "formula-reference-build",
        help="Materialize reviewed FormulaNet reference GT without changing raw output",
    )
    reference_build.add_argument("phase3a", type=Path)
    reference_build.add_argument("adjudications", type=Path)
    reference_build.add_argument("--output", "-o", type=Path, required=True)
    reference_evaluate = subparsers.add_parser(
        "formula-reference-evaluate",
        help="Evaluate frozen FormulaNet predictions against reviewed reference GT",
    )
    reference_evaluate.add_argument("reference", type=Path)
    reference_evaluate.add_argument("--phase3a", type=Path)
    reference_evaluate.add_argument("--output", "-o", type=Path, required=True)
    stability = subparsers.add_parser(
        "formula-ocr-stability",
        help="Run the frozen 157-image Batch 2 stability gate three times",
    )
    stability.add_argument("phase3a", type=Path)
    stability.add_argument("--output", "-o", type=Path, required=True)
    reference_gate = subparsers.add_parser(
        "formula-ocr-reference-gate",
        help="Evaluate the deterministic safety gate against Phase 3A.5 reference GT",
    )
    reference_gate.add_argument("reference", type=Path)
    reference_gate.add_argument("--output", "-o", type=Path, required=True)
    integration = subparsers.add_parser(
        "formula-ocr-integration",
        help="Run the 169-DOCX OCR-enabled production-candidate regression",
    )
    integration.add_argument("manifest", type=Path)
    integration.add_argument("--phase26", type=Path, required=True)
    integration.add_argument("--output", "-o", type=Path, required=True)
    integration.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    integration.add_argument(
        "--sample-source",
        action="append",
        type=Path,
        default=[],
        help="Retain full Markdown/assets for this source DOCX; repeat as needed",
    )
    phase3c = subparsers.add_parser(
        "mtef-performance-benchmark",
        help="Run the four single-worker Phase 3C OLE/MTEF cache modes",
    )
    phase3c.add_argument("manifest", type=Path)
    phase3c.add_argument("--phase26", type=Path, required=True)
    phase3c.add_argument("--phase3b", type=Path, required=True)
    phase3c.add_argument("--output", "-o", type=Path, required=True)
    phase3c.add_argument("--mtef-cache-dir", type=Path)
    phase3d2 = subparsers.add_parser(
        "phase3d2-validate",
        help="Run the 169-DOCX visual preservation and Asset Contract v1 gate",
    )
    phase3d2.add_argument("manifest", type=Path)
    phase3d2.add_argument("--phase3d1", type=Path, required=True)
    phase3d2.add_argument("--phase3b", type=Path, required=True)
    phase3d2.add_argument("--output", "-o", type=Path, required=True)
    phase3d2.add_argument("--overwrite", action="store_true")
    phase4a = subparsers.add_parser(
        "phase4a-validate",
        help="Run the 169-DOCX production Package Contract v1 gate",
    )
    phase4a.add_argument("manifest", type=Path)
    phase4a.add_argument("--phase3b", type=Path, required=True)
    phase4a.add_argument("--output", "-o", type=Path, required=True)
    phase4a.add_argument("--expected-default-output-root", type=Path)
    phase4a.add_argument("--overwrite", action="store_true")
    phase4b = subparsers.add_parser(
        "phase4b-validate",
        help="Run the real-runtime DOCX Tool release-candidate gate",
    )
    phase4b.add_argument("manifest", type=Path)
    phase4b.add_argument("--phase3a", type=Path, required=True)
    phase4b.add_argument("--phase3b", type=Path, required=True)
    phase4b.add_argument("--models-root", type=Path, required=True)
    phase4b.add_argument("--tool-root", type=Path, required=True)
    phase4b.add_argument("--output", "-o", type=Path, required=True)
    phase4b.add_argument("--overwrite", action="store_true")
    cache = subparsers.add_parser("cache", help="Inspect or clear BeMarkdown caches")
    cache_subparsers = cache.add_subparsers(dest="cache_command", required=True)
    cache_stats = cache_subparsers.add_parser("stats", help="Show MTEF cache stats")
    cache_stats.add_argument("--mtef-cache-dir", type=Path)
    cache_clear = cache_subparsers.add_parser("clear", help="Clear MTEF cache entries")
    cache_clear.add_argument("--mtef", action="store_true", required=True)
    cache_clear.add_argument("--mtef-cache-dir", type=Path)
    return parser


def _print_cli_error(exc: Exception, exit_code: int, json_mode: bool) -> int:
    payload = {
        "success": False,
        "exit_code": exit_code,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if json_mode:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"ERROR: {exc}", file=sys.stderr)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
        try:
            result = convert_document(
                args.input,
                args.output_root,
                debug=args.debug,
                formula_ocr=args.formula_ocr,
                mtef_cache=args.mtef_cache,
                mtef_cache_dir=args.mtef_cache_dir,
                replace=not args.no_replace,
                models_root=args.models_root,
                config_path=args.config,
                mcp_root=args.mcp_root,
            )
        except (InvalidDocxError, FileNotFoundError) as exc:
            return _print_cli_error(exc, EXIT_INVALID_INPUT, args.json)
        except OutputRootConfigurationError as exc:
            return _print_cli_error(exc, EXIT_ENVIRONMENT, args.json)
        except (ExistingPackageError, PackageValidationError) as exc:
            return _print_cli_error(exc, EXIT_PACKAGE_FAILURE, args.json)
        except Exception as exc:  # noqa: BLE001 - stable production CLI boundary
            return _print_cli_error(exc, EXIT_CONVERSION_FAILURE, args.json)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        else:
            print(result.package_path.resolve())
            print(result.quality_status)
        return EXIT_SUCCESS
    if args.command == "validate":
        result = validate_package(args.package)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        elif result.valid:
            print(f"VALID {result.package_path.resolve()}")
        else:
            print("\n".join(result.errors), file=sys.stderr)
        return EXIT_SUCCESS if result.valid else EXIT_PACKAGE_FAILURE
    if args.command == "cleanup-staging":
        try:
            result = cleanup_staging(
                args.output_root,
                older_than_seconds=args.older_than_hours * 3600,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return _print_cli_error(exc, EXIT_ENVIRONMENT, args.json)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        else:
            print(f"removed={len(result.removed)} retained={len(result.retained)}")
        return EXIT_SUCCESS
    if args.command == "doctor":
        result = run_doctor(
            models_root=args.models_root,
            output_root=args.output_root,
            tool_root=args.tool_root,
            config_path=args.config,
            mcp_root=args.mcp_root,
            deep=args.deep,
            probe_formula_runtime=not args.skip_formula_runtime,
            probe_model_suite_runtime=not args.skip_model_suite_runtime,
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        else:
            print(result.readiness)
            for name, value in result.capabilities.items():
                print(f"{name}: {value['status']} - {value['reason']}")
            for name, value in result.capability_readiness.items():
                print(f"{name}: {value}")
        return EXIT_SUCCESS if result.readiness != "NOT_READY" else EXIT_ENVIRONMENT
    if args.command == "wmf-census":
        result = run_wmf_census(
            args.manifest, args.output, audit_limit=args.audit_limit
        )
        print(result.summary_path.resolve())
        return 0 if all(result.summary["conservation"].values()) else 2
    if args.command == "revalidate-wmf-samples":
        result = revalidate_wmf_samples(args.summary, args.output)
        print(result.summary_path.resolve())
        return 0
    if args.command == "regress-docx":
        result = run_docx_regression(args.baseline, args.output)
        print(result.summary_path.resolve())
        return (
            0
            if result.summary["all_conservation_ok"]
            and result.summary["omml"]["all_exact_match"]
            else 2
        )
    if args.command == "docx-formula-census":
        result = run_docx_formula_census(
            args.manifest,
            args.output,
            phase25_census_dir=args.phase25_census,
            audit_limit=args.audit_limit,
        )
        print(result.summary_path.resolve())
        checks = result.summary["conservation"]
        complete = (
            result.summary["docx"]["failed"] == 0
            and checks["global_ok"]
            and checks["failed_docx_count_matches"]
            and checks["all_final_states_known"]
            and not checks["ocr_definition_violations"]
            and not checks["preview_suppression_violations"]
        )
        return 0 if complete else 2
    if args.command == "docx-edge-census":
        result = run_docx_edge_census(args.manifest, args.output)
        print(result.summary_path.resolve())
        return 0 if result.summary["all_ok"] else 2
    if args.command == "formulanet-benchmark":
        result = run_formulanet_benchmark(
            args.source,
            args.output,
            cache_path=args.cache,
            model_dir=args.model_dir,
            model_revision=args.model_revision,
            batch_sizes=args.batch_sizes,
        )
        print(result.summary_path.resolve())
        complete = (
            result.summary["dataset"]["integrity"]["all_ok"]
            and result.summary["failures"] == 0
            and result.summary["runtime_fingerprint"]["device"].startswith("gpu")
        )
        return 0 if complete else 2
    if args.command == "formula-reference-build":
        rows = build_reference_ground_truth(
            args.phase3a,
            args.adjudications,
            args.output,
        )
        print(args.output.resolve())
        return 0 if rows else 2
    if args.command == "formula-reference-evaluate":
        result = evaluate_formula_reference(
            args.reference,
            args.output,
            phase3a_dir=args.phase3a,
        )
        print(result.summary_path.resolve())
        return 0 if result.summary["review"]["reviewed"] else 2
    if args.command == "formula-ocr-stability":
        result = run_batch2_stability(args.phase3a, args.output)
        print(result.summary_path.resolve())
        return 0 if result.summary["all_157_x_3_match_batch1"] else 2
    if args.command == "formula-ocr-reference-gate":
        result = evaluate_reference_safety_gate(args.reference, args.output)
        print(result.summary_path.resolve())
        return 0 if result.summary["gate_pass"] else 2
    if args.command == "formula-ocr-integration":
        adapter = FormulaOcrAdapter(batch_size=args.batch_size)
        result = run_formula_ocr_integration(
            args.manifest,
            args.phase26,
            args.output,
            adapter=adapter,
            sample_sources=args.sample_source,
        )
        print(result.summary_path.resolve())
        return 0 if result.summary["all_ok"] else 2
    if args.command == "mtef-performance-benchmark":
        result = run_phase3c_benchmark(
            args.manifest,
            args.output,
            phase26_dir=args.phase26,
            phase3b_dir=args.phase3b,
            persistent_dir=args.mtef_cache_dir,
            progress=lambda message: print(message, flush=True),
        )
        print(result.output_diff_summary_path.resolve())
        return 0 if result.summary["output_diff"]["all_zero_difference"] else 2
    if args.command == "phase3d2-validate":
        result = run_phase3d2_validation(
            args.manifest,
            args.phase3d1,
            args.phase3b,
            args.output,
            progress=lambda message: print(message, flush=True),
            overwrite=args.overwrite,
        )
        print(result.summary_path.resolve())
        return 0 if result.summary["all_ok"] else 2
    if args.command == "phase4a-validate":
        result = run_phase4a_validation(
            args.manifest,
            args.phase3b,
            args.output,
            expected_default_output_root=args.expected_default_output_root,
            progress=lambda message: print(message, flush=True),
            overwrite=args.overwrite,
        )
        print(result.summary_path.resolve())
        return EXIT_SUCCESS if result.summary["all_ok"] else EXIT_PACKAGE_FAILURE
    if args.command == "phase4b-validate":
        summary = run_phase4b_validation(
            args.manifest,
            args.phase3a,
            args.phase3b,
            args.models_root,
            args.tool_root,
            args.output,
            progress=lambda message: print(message, flush=True),
            overwrite=args.overwrite,
        )
        print((args.output / "release_regression_summary.json").resolve())
        return EXIT_SUCCESS if summary["all_ok"] else EXIT_PACKAGE_FAILURE
    if args.command == "cache":
        context = MtefCacheContext(
            mode="persistent", persistent_dir=args.mtef_cache_dir
        )
        if args.cache_command == "stats":
            print(json.dumps(context.snapshot(), ensure_ascii=False, indent=2))
            return 0
        removed = context.clear_persistent()
        print(json.dumps({"mtef_entries_removed": removed}, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
