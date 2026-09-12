"""Recognize screenshot-based DOCX through the existing PDF model pipeline.

Native equations stay on the lossless DOCX route: office pagination is not a
reliable renderer for EQ/OLE/OMML. This route requires sparse native body text,
substantial displayed image area, and no structured equation objects.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .package import DocxResourceLimits, PackageIndex

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
NS = {'w': W, 'm': M, 'wp': WP}


def inspect_visual_docx(source: Path, *, limits: DocxResourceLimits | None = None) -> dict[str, Any]:
    package = PackageIndex(source, limits=limits)
    body = package.xml('word/document.xml').find('w:body', NS)
    # Count non-whitespace characters, not indentation or answer-line spacing.
    native_characters = sum(not c.isspace() for c in ''.join(body.xpath('.//w:t/text()', namespaces=NS)))
    equations = len(body.findall('.//m:oMath', NS))
    equations += len(body.xpath('.//*[local-name()="OLEObject"]'))
    instructions = body.xpath('.//w:instrText/text() | .//w:fldSimple/@w:instr', namespaces=NS)
    equations += sum(value.lstrip().upper().startswith('EQ') for value in instructions)
    image_area = 0.0
    images = 0
    for extent in body.findall('.//wp:extent', NS):
        drawing = extent.getparent()
        if not drawing.xpath('.//*[local-name()="blip"]'):
            continue
        try:
            area = int(extent.get('cx', '0')) * int(extent.get('cy', '0')) / 914400**2
        except ValueError:
            continue
        if area > 0:
            image_area += area
            images += 1
    external_content = any(r.external and not r.relationship_type.endswith('/hyperlink') for r in package.relationships.values())
    eligible = native_characters <= 200 and image_area >= 30 and equations == 0 and not external_content
    return {'eligible': eligible, 'native_body_characters': native_characters,
            'displayed_image_count': images, 'displayed_image_area_square_inches': round(image_area, 4),
            'structured_equation_objects': equations, 'external_content_relationships': external_content,
            'route': 'DOCX_SCREENSHOT_PDF' if eligible else 'DOCX_NATIVE'}


def find_soffice() -> Path:
    configured = os.environ.get('BEMARKDOWN_SOFFICE')
    candidates = [Path(configured)] if configured else []
    if not configured:
        for name in ('soffice.com', 'soffice', 'libreoffice'):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))
        for key in ('ProgramFiles', 'ProgramFiles(x86)'):
            if os.environ.get(key):
                candidates.append(Path(os.environ[key])/'LibreOffice/program/soffice.com')
                candidates.append(Path(os.environ[key])/'LibreOffice/program/soffice.exe')
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError('DOCX_SCREENSHOT_RENDERER_UNAVAILABLE: install LibreOffice or set BEMARKDOWN_SOFFICE to its executable')


def render_visual_docx(source: Path, work: Path) -> tuple[Path, dict[str, Any]]:
    executable = find_soffice()
    input_path = work/'source.docx'
    shutil.copyfile(source, input_path)
    profile = work/'office-profile'
    started = time.perf_counter()
    run = subprocess.run(
        [str(executable), f'-env:UserInstallation={profile.as_uri()}', '--headless', '--norestore',
         '--convert-to', 'pdf:writer_pdf_Export', '--outdir', str(work), str(input_path)],
        cwd=work, capture_output=True, timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
    )
    target = work/'source.pdf'
    if run.returncode or not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f'DOCX_SCREENSHOT_RENDER_FAILED: exit={run.returncode}; {run.stderr.decode("utf-8", errors="replace")[-500:]}')
    return target, {'backend': 'LibreOffice Writer PDF export', 'executable': str(executable),
                    'seconds': time.perf_counter()-started,
                    'pdf_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}


def _close_prepared_runtime(runtime):
    close = getattr(runtime, 'close_prepared_resources', None)
    if callable(close):
        close()


def convert_visual_docx(source: Path, output_dir: Path, *, profile: dict[str, Any],
                        pdf_runtime: Any, debug: bool = False):
    if not profile['eligible']:
        raise ValueError('DOCX_SCREENSHOT_ROUTE_INELIGIBLE')
    started = time.perf_counter()
    from .production import build_document_id
    with tempfile.TemporaryDirectory(prefix='bemarkdown-docx-image-') as name, ExitStack() as cleanup:
        cleanup.callback(_close_prepared_runtime, pdf_runtime)
        work = Path(name).resolve()
        prepare = getattr(pdf_runtime, 'prepare_dependencies', None)
        if callable(prepare):
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='bemarkdown-docx-render') as pool:
                rendering = pool.submit(render_visual_docx, source, work)
                preparation_started = time.perf_counter()
                prepare()
                preparation_seconds = time.perf_counter() - preparation_started
                pdf, render = rendering.result()
            render['dependency_preparation_overlapped'] = True
            render['dependency_preparation_seconds'] = preparation_seconds
        else:
            pdf, render = render_visual_docx(source, work)
        result = pdf_runtime.convert(pdf, output_dir, document_id=build_document_id(pdf), debug=debug)
        if debug:
            pagination_copy = output_dir / 'debug' / 'source-pagination.pdf'
            pagination_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pdf, pagination_copy)
            render['debug_pagination_ref'] = 'debug/source-pagination.pdf'
        report = result.report
        intermediate = dict(report['source'])
        report['source'] = {'type': 'DOCX', 'path': str(source), 'file_name': source.name,
                            'size_bytes': source.stat().st_size,
                            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        report['input_transform'] = {'route': 'DOCX_SCREENSHOT_PDF', 'profile': profile,
                                     'renderer': render, 'intermediate_pdf': intermediate,
                                     'source_formula_policy': 'No structured EQ, OLE or OMML objects are rasterized'}
        report['runtime']['vision_or_ocr_used'] = True
        report.setdefault('timing', {})['docx_render_seconds'] = render['seconds']
        report['timing']['docx_total_seconds'] = time.perf_counter()-started
        (output_dir/'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result
