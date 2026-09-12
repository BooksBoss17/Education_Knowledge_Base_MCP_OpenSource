"""Production BeMarkdown command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from .config import OutputRootConfigurationError
from .doctor import run_doctor
from .package import InvalidDocxError
from .pdf_source import PdfInspectionError
from .production import (
    ExistingPackageError,
    PackageValidationError,
    UnsupportedDocumentError,
    cleanup_staging,
    convert_document,
    validate_package,
)

EXIT_SUCCESS = 0
EXIT_INVALID_INPUT = 2
EXIT_ENVIRONMENT = 3
EXIT_CONVERSION_FAILURE = 4
EXIT_PACKAGE_FAILURE = 5


@contextmanager
def _conversion_diagnostics_to_stderr():
    """Keep provider diagnostics out of CLI results, including cached streams."""
    original_stdout = sys.stdout
    original_stdout.flush()
    try:
        stdout_fd = original_stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        stdout_fd = None
    saved_fd = None
    try:
        if stdout_fd is not None:
            saved_fd = os.dup(stdout_fd)
            os.dup2(stderr_fd, stdout_fd)
        with redirect_stdout(sys.stderr):
            yield
    finally:
        if saved_fd is not None:
            try:
                original_stdout.flush()
            finally:
                os.dup2(saved_fd, stdout_fd)
                os.close(saved_fd)


def build_parser() -> argparse.ArgumentParser:
    """Build the installed production CLI parser."""

    parser = argparse.ArgumentParser(prog="bemarkdown")
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = argparse.ArgumentParser(add_help=False)
    convert.add_argument(
        "--output-root",
        "--output",
        "-o",
        dest="output_root",
        type=Path,
        help="Package root; defaults through BeMarkdown workspace configuration",
    )
    convert.add_argument("--debug", action="store_true")
    convert.add_argument('--resource-profile', choices=('6gb', '10gb'),
                         help='Inference scheduling budget; preserves model, precision and recognition scope')
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

    single = subparsers.add_parser(
        "convert", parents=[convert], help="Produce one atomic BeMarkdown Package v1"
    )
    single.add_argument("input", type=Path)
    batch = subparsers.add_parser(
        "convert-batch", parents=[convert], help="Convert files serially in one Python host"
    )
    batch.add_argument("input", type=Path, nargs="+")
    batch.add_argument("--continue-on-error", action="store_true",
                       help="Attempt remaining files after a failure; exit remains nonzero")
    batch.add_argument('--no-reuse-formula-model', action='store_true',
                       help='Rebuild PDF/visual-DOCX FormulaNet for each file instead of retaining batch ownership')

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
    """Run one production CLI command."""

    args = build_parser().parse_args(argv)
    if args.command in {"convert", "convert-batch"}:
        if args.resource_profile is not None:
            import os
            os.environ['BEMARKDOWN_RESOURCE_PROFILE'] = args.resource_profile
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
        options = {
            'debug': args.debug, 'formula_ocr': args.formula_ocr, 'mtef_cache': args.mtef_cache,
            'mtef_cache_dir': args.mtef_cache_dir, 'replace': not args.no_replace,
            'models_root': args.models_root, 'config_path': args.config, 'mcp_root': args.mcp_root,
        }
        if args.command == "convert-batch":
            from .batch import convert_documents

            try:
                with _conversion_diagnostics_to_stderr():
                    batch_result = convert_documents(
                        args.input, args.output_root, continue_on_error=args.continue_on_error,
                        reuse_formula_model=not args.no_reuse_formula_model, **options)
            except Exception as exc:  # noqa: BLE001 - includes batch resource cleanup failures
                return _print_cli_error(exc, EXIT_CONVERSION_FAILURE, args.json)
            if args.json:
                print(json.dumps(batch_result, ensure_ascii=False))
            else:
                for row in batch_result['documents']:
                    destination = row.get('result', {}).get('package_path', row.get('error', ''))
                    print(f"{row['status']} {row['source']}: {destination}")
            failure = next((row for row in batch_result['documents'] if row['status'] == 'FAILED'), None)
            if failure is None:
                return EXIT_SUCCESS
            return {'INVALID_INPUT': EXIT_INVALID_INPUT, 'ENVIRONMENT': EXIT_ENVIRONMENT,
                    'PACKAGE_FAILURE': EXIT_PACKAGE_FAILURE,
                    'CONVERSION_FAILURE': EXIT_CONVERSION_FAILURE}[failure['failure_kind']]
        try:
            with _conversion_diagnostics_to_stderr():
                result = convert_document(
                    args.input,
                    args.output_root,
                    **options,
                )
        except (
            InvalidDocxError,
            PdfInspectionError,
            UnsupportedDocumentError,
            FileNotFoundError,
        ) as exc:
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
