"""Recognize native DOCX raster-image labels with the existing A/B/C pipeline."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import time
from pathlib import Path

from PIL import Image

from .asset_contract import _iter_visible_occurrences
from .ir import ImageNode


class DocxFigureLabelAdapter:
    """Keep original assets while attaching source-bound label evidence."""

    def __init__(self, *, models_root=None, config_path=None, mcp_root=None,
                 resolver_factory=None):
        self.models_root = models_root
        self.config_path = config_path
        self.mcp_root = mcp_root
        self.resolver_factory = resolver_factory

    def _resolver(self, work):
        if self.resolver_factory is not None:
            return self.resolver_factory(work)
        from .pdf.production_runtime import RegistryBackedThreeModelTextResolver

        return RegistryBackedThreeModelTextResolver(
            work, models_root=self.models_root, config_path=self.config_path,
            mcp_root=self.mcp_root)

    def process(self, document, output_dir, report, *, formula_runtime=None):
        started = time.perf_counter()
        output_dir = Path(output_dir).resolve()
        nodes = [occurrence.node for occurrence in _iter_visible_occurrences(document)
                 if isinstance(occurrence.node, ImageNode)]
        summary = {
            'schema': 'bemarkdown-docx-figure-labels-v1', 'enabled': True,
            'image_count': len(nodes), 'raster_candidates': 0,
            'input_preparation_failures': 0,
            'emitted_label_count': 0, 'native_label_count': 0, 'review_required': 0,
            'runtime_metrics': {},
        }
        report['figure_text'] = summary
        if not nodes:
            summary['wall_seconds'] = time.perf_counter() - started
            return summary
        # Model libraries may still enforce MAX_PATH even when the package
        # writer can handle long paths. Keep transient inference files outside
        # the deeply nested publication staging directory.
        with tempfile.TemporaryDirectory(prefix='bmd-docx-labels-') as temporary:
            work = Path(temporary).resolve()
            plans, crops, owners = [], {}, {}
            for frame_index, node in enumerate(nodes):
                proof = {
                    'schema': 'bemarkdown-docx-image-label-evidence-v1',
                    'coordinate_system': 'DOCX_ASSET_PIXEL', 'frame_index': frame_index,
                    'source_part': node.source_part, 'source_locator': node.source_locator,
                    'status': 'PENDING', 'labels': [],
                }
                node.provenance['figure_label_recognition'] = proof
                if node.semantic_type == 'DRAWINGML_GROUP' and 'visible_labels' in node.provenance:
                    # The scanner has already saved these source labels in the
                    # asset manifest. Do not OCR or duplicate that structured text.
                    proof.update(status='SOURCE_STRUCTURED_LABELS',
                                 native_labels=list(node.provenance['visible_labels']))
                    summary['native_label_count'] += len(proof['native_labels'])
                    continue
                try:
                    if not node.asset_path:
                        raise ValueError('DOCX_IMAGE_ASSET_UNAVAILABLE')
                    source = (output_dir / node.asset_path).resolve()
                    source.relative_to(output_dir)
                    proof['source_image_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
                    with Image.open(source) as original:
                        if getattr(original, 'n_frames', 1) != 1:
                            raise ValueError('MULTIFRAME_IMAGE_REQUIRES_REVIEW')
                        rgba = original.convert('RGBA')
                    extrema = rgba.getextrema()
                    if all(low == high for low, high in extrema) or extrema[3][1] == 0:
                        proof['status'] = 'UNIFORM_IMAGE'
                        continue
                    image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
                    parent = 'docx-image-' + hashlib.sha256(
                        f"{report['source']['sha256']}:{node.source_part}:{node.source_locator}:{frame_index}"
                        .encode()).hexdigest()[:24]
                    target = work / f'{parent}.png'
                    image.save(target, format='PNG')
                    digest = hashlib.sha256(target.read_bytes()).hexdigest()
                    width, height = image.size
                    box = [0.0, 0.0, float(width), float(height)]
                    proof.update(image_size_px=[width, height], recognition_input_sha256=digest,
                                 input_transform='SAME_SIZE_RGBA_COMPOSITED_ON_WHITE_RGB')
                    route_id = parent + '-figure-labels'
                    document_id = 'docx-' + report['source']['sha256'][:24]
                    crop = {
                        'path': str(target), 'content_sha256': digest,
                        'bbox_pdf_pt': box, 'bbox_render_px': [0, 0, width, height],
                        'width': width, 'height': height, 'scale_x': 1.0, 'scale_y': 1.0,
                        'source_path': str(source), 'page_index': frame_index,
                        'figure_label_parent_route_id': parent, 'complete_region_lines': False,
                        'coordinate_system': 'DOCX_ASSET_PIXEL_NOT_DOCUMENT_PAGE',
                    }
                    # The shared resolver's internal page/point slots carry
                    # image-frame coordinates here. Published labels explicitly
                    # use asset pixels; this never supplies a DOCX page count.
                    route = {
                        'route_id': route_id, 'document_id': document_id,
                        'page_index': frame_index, 'adapter': 'OCR_TEXT_REGION', 'output_kind': 'TEXT',
                        'input_candidate_ids': [parent], 'decision_reason_codes': ['DOCX_NATIVE_IMAGE_LABELS'],
                        'provenance': {'bbox_pdf_pt': box, 'figure_label_parent_route_id': parent,
                                       'source_profile': 'DOCX_ASSET_RASTER'},
                    }
                    plans.append({'document_id': document_id, 'page_index': frame_index, 'routes': [route]})
                    crops[route_id] = crop
                    owners[route_id] = (node, proof)
                except (OSError, ValueError) as exc:
                    proof.update(status='INPUT_PREPARATION_FAILED', error=f'{type(exc).__name__}:{exc}')
                    summary['input_preparation_failures'] += 1
                    summary['review_required'] += 1
            summary['raster_candidates'] = len(crops)
            if crops:
                try:
                    release = getattr(formula_runtime, 'release_workspace', None)
                    if callable(release):
                        release()
                    staged = self._resolver(work).resolve(
                        plans=plans, crops_by_route=crops, formula_runtime=formula_runtime)
                    summary['runtime_metrics'] = copy.deepcopy(staged.metrics)
                    summary['runtime_metrics']['coordinate_scope'] = 'DOCX_ASSET_FRAMES_NOT_DOCUMENT_PAGES'
                    summary['runtime_metrics']['source_frame_count'] = summary['runtime_metrics'].pop('source_page_count', None)
                    results = staged.runtime.figure_label_results
                    for route_id, (node, proof) in owners.items():
                        result = results.get(route_id)
                        if result is None:
                            proof['status'] = 'RUNTIME_RESULT_UNAVAILABLE'
                            summary['review_required'] += 1
                            continue
                        proof['status'] = result['status']
                        proof['runtime_evidence'] = copy.deepcopy(result)
                        for label in result['labels']:
                            if label.get('role') == 'GRAPHIC_MARKER':
                                continue
                            public = {key: copy.deepcopy(value) for key, value in label.items()
                                      if key not in {'bbox_pdf_pt', 'formula_evidence', 'formula_error'}}
                            public['bbox_asset_px'] = list(label['bbox_pdf_pt'])
                            proof['labels'].append(public)
                        if proof['labels']:
                            from .pdf.figure_labels import figure_alt_text
                            caption = figure_alt_text({'figure_labels': proof['labels']}, escape_brackets=False)
                            node.alt = (node.alt + '；' if node.alt else '') + caption
                            summary['emitted_label_count'] += sum(
                                bool(label.get('latex') or label.get('text'))
                                and 'REQUIRED' not in str(label.get('status', ''))
                                and not label.get('review_reasons')
                                for label in proof['labels'])
                        if result['status'] != 'TEXT_DETECTED' or any(
                            'REQUIRED' in label['status'] or label.get('review_reasons')
                            for label in proof['labels']
                        ):
                            summary['review_required'] += 1
                    report['runtime']['vision_or_ocr_used'] = True
                except Exception as exc:  # noqa: BLE001 - preserve assets and record provider failure
                    summary['error'] = f'{type(exc).__name__}:{exc}'
                    for _, proof in owners.values():
                        if proof['status'] == 'PENDING':
                            proof.update(status='RUNTIME_FAILED', error=summary['error'])
                            summary['review_required'] += 1
        summary['wall_seconds'] = time.perf_counter() - started
        return summary
