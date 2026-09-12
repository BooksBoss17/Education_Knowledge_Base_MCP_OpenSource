"""Serial complete-document conversion in one reusable Python host."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

from .config import OutputRootConfigurationError
from .formula_runtime_owner import FormulaRuntimeOwner
from .package import InvalidDocxError
from .pdf_source import PdfInspectionError
from .production import (
    ExistingPackageError,
    PackageValidationError,
    UnsupportedDocumentError,
    convert_document,
)

LOGGER = logging.getLogger(__name__)


def _failure_kind(exc: Exception) -> str:
    if isinstance(exc, (InvalidDocxError, PdfInspectionError, UnsupportedDocumentError, FileNotFoundError)):
        return 'INVALID_INPUT'
    if isinstance(exc, OutputRootConfigurationError):
        return 'ENVIRONMENT'
    if isinstance(exc, (ExistingPackageError, PackageValidationError)):
        return 'PACKAGE_FAILURE'
    return 'CONVERSION_FAILURE'


def convert_documents(
    sources: Iterable[str | Path], output_root: str | Path | None = None,
    *, continue_on_error: bool = False, reuse_formula_model: bool = True, **conversion_options,
) -> dict:
    """Convert in input order, retaining ordinary per-file atomic publication.

    PDF/visual-DOCX FormulaNet weights may be retained by this batch's owner;
    every file still validates the registry and recomputes recognition results.
    Timings include work inside this API, not interpreter/import startup.
    """
    started = time.perf_counter()
    if isinstance(sources, (str, Path)):
        sources = [sources]
    paths = [Path(source).resolve() for source in sources]
    if not paths:
        raise ValueError('At least one source document is required')
    documents = []
    stopped = False
    with FormulaRuntimeOwner() as owner:
        if (reuse_formula_model and conversion_options.get('pdf_formula_runtime_factory') is None
                and conversion_options.get('pdf_runtime_factory') is None):
            conversion_options['pdf_formula_runtime_owner'] = owner
        for index, source in enumerate(paths):
            row = {'index': index, 'source': str(source), 'status': 'NOT_RUN', 'wall_seconds': 0.0}
            documents.append(row)
            if stopped:
                continue
            LOGGER.info('Batch document %s/%s: %s', index + 1, len(paths), source)
            file_started = time.perf_counter()
            try:
                result = convert_document(source, output_root, **conversion_options)
                row.update(status='PUBLISHED', result=result.to_dict())
            except Exception as exc:  # noqa: BLE001 - preserve per-file failures at the batch boundary
                row.update(status='FAILED', failure_kind=_failure_kind(exc),
                           error_type=type(exc).__name__, error=str(exc))
                stopped = not continue_on_error
            row['wall_seconds'] = time.perf_counter() - file_started
            LOGGER.info('Batch document %s/%s: %s in %.3fs',
                        index + 1, len(paths), row['status'], row['wall_seconds'])
    published = sum(row['status'] == 'PUBLISHED' for row in documents)
    failed = sum(row['status'] == 'FAILED' for row in documents)
    return {
        'schema': 'bemarkdown-batch-conversion-v1', 'success': published == len(paths),
        'host_pid': os.getpid(), 'requested_count': len(paths), 'published_count': published,
        'failed_count': failed, 'not_run_count': len(paths) - published - failed,
        'wall_seconds': time.perf_counter() - started,
        'timing_scope': 'SERIAL_BATCH_API_INCLUDING_ALL_ATTEMPTED_FILES_EXCLUDING_INTERPRETER_STARTUP',
        'recognition_result_cache': False, 'documents': documents,
        'formula_model_lifecycle': owner.metrics(),
        'note': 'Published packages may contain review items; publication is not recognition accuracy.',
    }
