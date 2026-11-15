"""Unified native Word / local visual API conversion and atomic package delivery."""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from io import BytesIO
from pathlib import Path

from . import package_io as packages
from .config import resolve_output_root
from .image_input import IMAGE_SUFFIXES, _decode
from .page_render import PageRender, render_pdf
from .pdf.local_model_api import create_api_runtime_factory


class VisualApiRuntime:
    def __init__(self, config_path):
        # Reuse role clients across all embedded images to reject model drift.
        self.orchestrator = create_api_runtime_factory(config_path)()

    def convert(self, source, output_dir, *, document_id, debug=False):
        return self.orchestrator.convert_visual(source=source, output_dir=output_dir,
            document_id=document_id, renders=render_pdf(source), debug=debug)

    def convert_image(self, source, output_dir, *, document_id, debug=False):
        source = Path(source)
        image, normalization = _decode(source)
        try:
            stream = BytesIO()
            image.save(stream, format='PNG')
        finally:
            image.close()
        data = stream.getvalue()
        result = self.orchestrator.convert_visual(source=source, output_dir=output_dir,
            document_id=document_id,
            renders=[PageRender(0, data, hashlib.sha256(data).hexdigest())], debug=debug)
        result.report['source']['type'] = 'IMAGE'
        result.report['input_transform'] = {'route': 'ORIGINAL_IMAGE_LOCAL_API', **normalization}
        return result


def convert_document(source, output_root=None, *, api_config=None, formula_ocr='auto',
                     debug=False, replace=True, mtef_cache='persistent',
                     mtef_cache_context=None, mtef_cache_dir=None, resource_limits=None):
    started = time.perf_counter()
    source = Path(source).resolve()
    suffix = source.suffix.lower()
    if suffix not in {'.docx', '.pdf', *IMAGE_SUFFIXES}:
        raise packages.UnsupportedDocumentError(f'Unsupported input: {suffix}')
    if formula_ocr not in {'auto', 'off'}:
        raise ValueError('formula_ocr must be auto or off')
    source_sha = packages._sha256_file(source)
    document_id = packages.build_document_id(source, source_sha)
    source_type = 'DOCX' if suffix == '.docx' else ('PDF' if suffix == '.pdf' else 'IMAGE')
    api_config = api_config or os.environ.get('BEMARKDOWN_MODEL_API_CONFIG')
    if source_type != 'DOCX' or formula_ocr == 'auto':
        if not api_config:
            raise ValueError('BEMARKDOWN_MODEL_API_CONFIG_REQUIRED')
        visual = VisualApiRuntime(api_config)
    root = resolve_output_root(output_root)
    final = root/document_id
    if final.exists() and not replace:
        raise packages.ExistingPackageError(str(final))
    job = root/'.staging'/uuid.uuid4().hex
    staging = job/document_id
    staging.mkdir(parents=True)
    try:
        if source_type == 'DOCX':
            from .pipeline import convert_docx
            from .native_api import NativeFormulaApi, NativeImageApi
            result = convert_docx(source, staging, debug=debug, formula_ocr=formula_ocr,
                formula_ocr_adapter=NativeFormulaApi(api_config, visual.orchestrator.executor.formula_runtime) if formula_ocr == 'auto' else None,
                figure_text_adapter=NativeImageApi(visual) if formula_ocr == 'auto' else None,
                mtef_cache=mtef_cache, mtef_cache_context=mtef_cache_context,
                mtef_cache_dir=mtef_cache_dir, resource_limits=resource_limits)
            result.report['input_transform'] = {'route': 'NATIVE_WORD_WITH_LOCAL_IMAGE_API',
                                               'image_source_policy': 'ORIGINAL_EMBEDDED_MEDIA'}
        else:
            method = visual.convert if source_type == 'PDF' else visual.convert_image
            result = method(source, staging, document_id=document_id, debug=debug)
        if packages._sha256_file(source) != source_sha:
            raise ValueError('SOURCE_CHANGED_DURING_CONVERSION')
        report = result.report
        if source_type == 'DOCX' and (report.get('image_content', {}).get('converted_images')
                                     or report.get('formula_ocr', {}).get('candidates')):
            report['warnings'].append('AUTOMATIC_TRANSCRIPTION_NOT_FULL_BOOK_QUALITY_CERTIFICATION')
            if report['quality_status'] == 'CLEAN':
                report['quality_status'] = 'COMPLETED_WITH_WARNINGS'
        report['review_item_count'] = packages._review_item_count(report)
        report.setdefault('runtime', {}).update(route='bemarkdown-next-local-api-v1',
                                                image_screening_called=False)
        report.setdefault('timing', {})['total_seconds'] = time.perf_counter()-started
        packages._write_json(staging/'conversion_report.json', report)
        validator = packages.PackageValidator()
        validator.require_payload_valid(staging)
        packages._write_json(staging/'package_manifest.json', packages._package_manifest(
            document_id, source, source_sha, report, staging, source_type=source_type))
        validator.require_valid(staging)
        packages._publish_staged(staging, final, replace=replace)
        job.rmdir()
    except BaseException as exc:
        packages._record_failure(job, document_id, source, source_sha, exc)
        raise
    return packages.PackageConversionResult(document_id=document_id, package_path=final,
        source_sha256=source_sha, source_type=source_type, quality_status=report['quality_status'],
        success=True, warnings=tuple(report.get('warnings', [])),
        review_item_count=packages._review_item_count(report))


def convert_documents(sources, output_root=None, *, continue_on_error=False, **options):
    """Serial batch with independent atomic delivery and explicit unrun items."""
    if isinstance(sources, (str, Path)):
        sources = [sources]
    sources = [Path(p).resolve() for p in sources]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError('Provide one or more distinct source files')
    rows, stopped = [], False
    started = time.perf_counter()
    for index, source in enumerate(sources):
        row = {'index':index, 'source':str(source), 'status':'NOT_RUN'}
        rows.append(row)
        if stopped:
            continue
        before = time.perf_counter()
        try:
            row.update(status='PUBLISHED',result=convert_document(source,output_root,**options).to_dict())
        except Exception as exc:
            row.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
            stopped = not continue_on_error
        row['wall_seconds'] = time.perf_counter()-before
    return {'success':all(r['status']=='PUBLISHED' for r in rows),'documents':rows,
            'wall_seconds':time.perf_counter()-started}
