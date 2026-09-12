"""Classify embedded Word images before deciding whether to recognize content.

Diagrams retain their original assets. Mixed images reuse the PDF region pipeline
without rasterizing native Word equations or the rest of the Word document.
"""
from __future__ import annotations

import copy
import hashlib
import re
import time
from pathlib import Path

from PIL import Image

from .asset_contract import _iter_visible_occurrences
from .ir import HyperlinkNode, ImageContentNode, ImageNode, TableNode


def _area(box):
    return max(0, box[2]-box[0])*max(0, box[3]-box[1])


def _coverage(inner, outer):
    intersection = [max(inner[0], outer[0]), max(inner[1], outer[1]),
                    min(inner[2], outer[2]), min(inner[3], outer[3])]
    return _area(intersection)/max(_area(inner), 1)


def classify_image_regions(rows, ink_bbox, *, pixel_evidence=None):
    """Use the existing layout hypotheses, not a whole-image OCR heuristic."""
    if ink_bbox is None:
        return 'DECORATIVE_OR_EMPTY'
    if pixel_evidence:
        return 'REGION_CONVERSION_REQUIRED'
    rows = [r for r in rows if r['raw_score'] >= .3]
    # The layout detector can return alternative classes for the exact same box.
    # Do not treat a lower-scoring title alternative as an independent text region.
    best = {}
    for row in rows:
        key = tuple(row['raw_bbox_render_px'])
        if key not in best or row['raw_score'] > best[key]['raw_score']:
            best[key] = row
    regions = list(best.values())
    figures = [r for r in regions if r['raw_label'] in {'image', 'chart'}]
    for figure in figures:
        box = figure['raw_bbox_render_px']
        if _coverage(ink_bbox, box) < .90:
            continue
        outside = [r for r in regions if r not in figures and
                   _coverage(r['raw_bbox_render_px'], box) < .95]
        # A whole formula/table must not be exempted by a competing image box.
        specialists = [r for r in rows if r['raw_label'] in {'formula', 'table'} and
                       _coverage(ink_bbox, r['raw_bbox_render_px']) >= .5]
        if not outside and not specialists:
            return 'DIAGRAM'
    return 'REGION_CONVERSION_REQUIRED'


def _candidate_evidence(value):
    if isinstance(value, dict):
        if isinstance(value.get('agent_review_candidate'), dict):
            yield {'agent_review_candidate': copy.deepcopy(value['agent_review_candidate']),
                   'resolver': copy.deepcopy(value.get('resolver', {}))}
            return
        for child in value.values():
            yield from _candidate_evidence(child)
    elif isinstance(value, list):
        for child in value:
            yield from _candidate_evidence(child)


class _PreparedImageLayout:
    """Reuse source-image layout boxes after lossless placement on a PDF page."""

    def __init__(self, rows, size, identity, pixel_evidence=None):
        self.rows, self.size, self.identity = rows, size, identity
        self.pixel_evidence = pixel_evidence

    def refine_page_region_ir(self, page, image, transform):
        """Retain model predictions and trace a source-pixel semantic override."""
        if not self.pixel_evidence:
            return None
        from .pdf_region_ir import _region_id, apply_reading_order_to_page
        regions = page.get('canonical_regions', [])
        ink = self.pixel_evidence['ink_bbox_source_px']
        sx, sy = image.width/self.size[0], image.height/self.size[1]
        scaled_ink = [v*(sx if i % 2 == 0 else sy) for i, v in enumerate(ink)]
        enclosing = [r for r in regions if _coverage(scaled_ink, r['bbox_render_px']) >= .90]
        if not enclosing or len(regions) != len(enclosing):
            return {'status': 'NOT_APPLIED_NON_ISOLATED_LAYOUT', 'evidence': self.pixel_evidence}
        if any(r['semantic_type'] not in {'IMAGE', 'TITLE', 'TEXT'} for r in enclosing):
            return None
        region = max(enclosing, key=lambda r: (r['score'], r['region_id']))
        previous = copy.deepcopy(region)
        alternatives = [copy.deepcopy(r) for r in enclosing if r is not region]
        region.update(semantic_type='FORMULA', semantic_subtype=None,
                      routing_intent='FORMULA_CONTENT', normalization_reason='SOURCE_PIXEL_ISOLATED_FRACTION')
        region['provenance']['source_pixel_semantic_refinement'] = {
            'evidence': copy.deepcopy(self.pixel_evidence), 'original_region': previous,
            'coextensive_model_alternatives': alternatives,
            'model_score_unchanged': True, 'model_label_unchanged': True,
        }
        region['parent_region_id'] = None
        region['region_id'] = _region_id(page['document_id'], page['page_index'], region)
        page['canonical_regions'] = [region]
        page['suppressed_detections'].extend({
            'raw_detection_id': r['raw_detection_id'],
            'retained_raw_detection_id': region['raw_detection_id'],
            'normalization_reason': 'SOURCE_PIXEL_ISOLATED_FRACTION_ALTERNATIVE',
        } for r in alternatives)
        apply_reading_order_to_page(page)
        return {'status': 'SOURCE_SUPPORTED_FORMULA_ROUTE', 'region_id': region['region_id'],
                'original_region_id': previous['region_id'], 'evidence': self.pixel_evidence}

    def load(self):
        return {**self.identity, 'embedded_image_layout_reused': True}

    def predict_image(self, image, *, document_id, page_index, page_render_identity):
        sx, sy = image.width/self.size[0], image.height/self.size[1]
        if abs(sx-sy) > max(sx, sy)*.01:
            raise ValueError('Embedded-image layout reuse changed aspect ratio')
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row['raw_bbox_render_px'] = [v*(sx if i % 2 == 0 else sy)
                                         for i, v in enumerate(row['raw_bbox_render_px'])]
            row['page_render_identity'] = page_render_identity
            row['raw_detection_id'] = hashlib.sha256(
                f"{document_id}:{page_index}:{row['raw_detection_id']}".encode()).hexdigest()
        return rows, 0.0

    def unload(self):
        return {'unload_status': 'PRECOMPUTED_LAYOUT_NO_MODEL_LOADED'}


class _PreparedImagesLayout:
    """Reuse each original image's layout while batching its independent source page."""
    def __init__(self, jobs, identity):
        self.pages = [_PreparedImageLayout(rows, image.size, identity,
            node.provenance.get('image_region_classification', {}).get('pixel_content_evidence'))
            for _, node, image, rows in jobs]

    def load(self):
        return {**self.pages[0].load(), 'embedded_image_batch_count': len(self.pages)}

    def predict_image(self, image, *, document_id, page_index, page_render_identity):
        return self.pages[page_index].predict_image(image, document_id=document_id,
            page_index=page_index, page_render_identity=page_render_identity)

    def refine_page_region_ir(self, page, image, transform):
        return self.pages[page['page_index']].refine_page_region_ir(page, image, transform)

    def unload(self):
        return {'unload_status': 'PRECOMPUTED_LAYOUT_NO_MODEL_LOADED'}


def _table_review_blocks(blocks):
    """Retain pending table evidence after the nested PDF debug tree goes away."""
    result = []
    cell_fields = ('cell_id', 'row', 'col', 'rowspan', 'colspan', 'text',
                   'bbox_pdf_pt', 'review_state', 'content_type', 'formula_region_evidence')
    for block in blocks:
        if block.get('kind') != 'TABLE':
            continue
        if block.get('visibility') in {'SUPPRESSED_DUPLICATE', 'SUPPRESSED_FROM_MAIN_BODY'}:
            continue
        content = block.get('content', {})
        table = content.get('table_ir') or {}
        pending = [c for c in table.get('cells', []) if c.get('review_state', 'NONE') != 'NONE']
        unassigned = table.get('unassigned_formula_regions', [])
        if not pending and not unassigned and (block.get('review_state') == 'NONE' or table.get('quality', {}).get('status') == 'STRUCTURED_TABLE'):
            continue
        result.append({
            'node_id': block.get('node_id'), 'kind': 'TABLE',
            'page_index': block.get('page_index', 0),
            'bbox_pdf_pt': copy.deepcopy(block.get('bbox_pdf_pt', [])),
            'review_state': block.get('review_state'),
            'content': {
                'source_status': content.get('source_status'),
                'table_serialization': {'content_body': content.get('table_serialization', {}).get('content_body', '')},
                'table_ir': {'quality': copy.deepcopy(table.get('quality', {})),
                             'cells': [{k: copy.deepcopy(c[k]) for k in cell_fields if k in c} for c in pending],
                             'unassigned_formula_regions': copy.deepcopy(unassigned)}},
            'provenance': {'review_reasons': copy.deepcopy(block.get('provenance', {}).get('review_reasons', []))},
        })
    return result


class DocxImageContentAdapter:
    def __init__(self, *, models_root=None, config_path=None, mcp_root=None,
                 layout_factory=None, pdf_runtime_factory=None):
        self.models_root, self.config_path, self.mcp_root = models_root, config_path, mcp_root
        self.layout_factory = layout_factory
        self.pdf_runtime_factory = pdf_runtime_factory

    def _layout(self):
        if self.layout_factory is not None:
            return self.layout_factory()
        from .pdf_layout_runtime import FormalPaddleLayoutRuntime
        return FormalPaddleLayoutRuntime(models_root=self.models_root,
                                        config_path=self.config_path, mcp_root=self.mcp_root)

    def process(self, document, output_dir, report, *, formula_runtime=None):
        started = time.perf_counter()
        output_dir = Path(output_dir).resolve()
        nodes = [o.node for o in _iter_visible_occurrences(document) if isinstance(o.node, ImageNode)]
        summary = {'schema': 'bemarkdown-docx-image-content-v1', 'enabled': True,
                   'image_count': len(nodes), 'diagrams_preserved': 0, 'decorative_preserved': 0,
                   'converted_images': 0, 'review_required': 0, 'sources': [], 'region_reviews': []}
        report['image_content'] = summary
        # The old blanket figure-label adapter is intentionally not part of production.
        report['figure_text'] = {'enabled': False, 'review_required': 0,
                                 'reason': 'DIAGRAMS_PRESERVED_WITHOUT_LABEL_TRANSCRIPTION'}
        release = getattr(formula_runtime, 'release_workspace', None)
        if callable(release):
            release()
        work = output_dir/'.bmd-staging'/'image-content'
        layout = None
        identity = {}
        jobs = []
        replacements = {}
        try:
            for index, node in enumerate(nodes):
                if node.semantic_type == 'DRAWINGML_GROUP' and node.asset_path:
                    self._preserve(node, summary, 'DIAGRAM')
                    continue
                if not node.asset_path:
                    self._review(node, summary, 'SOURCE_IMAGE_UNAVAILABLE')
                    continue
                path = (output_dir/node.asset_path).resolve()
                if not path.is_relative_to(output_dir):
                    raise ValueError('Embedded source image escapes package')
                try:
                    with Image.open(path) as opened:
                        if getattr(opened, 'n_frames', 1) != 1:
                            self._review(node, summary, 'MULTIFRAME_CLASSIFICATION_REQUIRED')
                            continue
                        rgba = opened.convert('RGBA')
                        image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
                    ink = image.convert('L').point(lambda v: 255 if v < 240 else 0).getbbox()
                    if ink is None or min(image.size) <= 3:
                        self._preserve(node, summary, 'DECORATIVE_OR_EMPTY')
                        continue
                    if layout is None:
                        layout = self._layout()
                        identity = layout.load()
                    pixel_sha = hashlib.sha256(
                        f'{image.width}x{image.height}:RGB:'.encode()+image.tobytes()).hexdigest()
                    rows, _seconds = layout.predict_image(
                        image, document_id='docx-'+report['source']['sha256'][:24],
                        page_index=index, page_render_identity=pixel_sha)
                    from .pdf.isolated_image_content import isolated_fraction_evidence
                    pixel_evidence = isolated_fraction_evidence(image, ink)
                    decision = classify_image_regions(rows, ink, pixel_evidence=pixel_evidence)
                    node.provenance['image_region_classification'] = {
                        'pixel_content_evidence': pixel_evidence,
                        'decision': decision, 'image_size_px': list(image.size),
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'input_transform': 'SAME_SIZE_RGBA_COMPOSITED_ON_WHITE_RGB',
                        'input_pixel_sha256': pixel_sha,
                        'layout_model': identity, 'layout_regions': rows,
                    }
                    if decision != 'REGION_CONVERSION_REQUIRED':
                        self._preserve(node, summary, decision)
                    else:
                        jobs.append((index, node, image, rows))
                except (OSError, ValueError) as exc:
                    self._review(node, summary, f'IMAGE_CLASSIFICATION_FAILED:{type(exc).__name__}:{exc}')
        finally:
            if layout is not None:
                layout.unload()
        from .resource_profiles import resource_profile
        if resource_profile().name == '10gb' and len(jobs) > 1:
            replacements.update(self._convert_region_batch(jobs, identity, work, output_dir, summary))
        else:
            for index, node, image, rows in jobs:
                replacement = self._convert_regions(node, image, rows, identity, index, work, output_dir, summary)
                replacements[id(node)] = replacement

        def replace_inlines(children):
            for i, child in enumerate(children):
                if id(child) in replacements:
                    children[i] = replacements[id(child)]
                elif isinstance(child, HyperlinkNode):
                    replace_inlines(child.children)

        for block in document.blocks:
            if isinstance(block, TableNode):
                for row in block.rows:
                    for cell in row:
                        for paragraph in cell.blocks:
                            replace_inlines(paragraph.children)
            else:
                replace_inlines(block.children)
        summary['wall_seconds'] = time.perf_counter()-started
        return summary

    @staticmethod
    def _preserve(node, summary, kind):
        node.alt = '图' if kind == 'DIAGRAM' else (node.alt or '图')
        node.provenance['figure_preservation'] = {
            'status': 'PRESERVED_AS_IMAGE', 'classification': kind, 'text_recognition': 'NOT_REQUESTED'}
        summary['diagrams_preserved' if kind == 'DIAGRAM' else 'decorative_preserved'] += 1

    @staticmethod
    def _review(node, summary, reason):
        node.provenance['image_region_review'] = {'status': 'REQUIRED', 'reason': reason}
        summary['review_required'] += 1
        summary['region_reviews'].append({'source_part': node.source_part,
            'source_locator': node.source_locator, 'relationship_id': node.relationship_id,
            'media_part': node.media_part, 'reason': reason, 'kind': 'IMAGE_REGION_CLASSIFICATION'})

    def _convert_regions(self, node, image, rows, identity, index, work, output_dir, summary):
        import json
        import pymupdf
        from .pdf.production_runtime import create_production_pdf_runtime
        from .production import build_document_id

        item_work = work/f'item-{index:04d}'
        item_work.mkdir(parents=True, exist_ok=False)
        png = item_work/'original.png'
        image.save(png)
        source = item_work/'source.pdf'
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=image.width, height=image.height)
            page.insert_image(page.rect, filename=str(png))
            pdf.save(source)
        prepared = _PreparedImageLayout(rows, image.size, identity,
            node.provenance.get('image_region_classification', {}).get('pixel_content_evidence'))
        factory = self.pdf_runtime_factory or create_production_pdf_runtime
        runtime = factory(models_root=self.models_root, config_path=self.config_path,
                          mcp_root=self.mcp_root, layout_runtime_factory=lambda: prepared)
        package = item_work/'recognized'
        package.mkdir()
        result = runtime.convert(source, package, document_id=build_document_id(source), debug=True)
        ir = json.loads((package/'debug/document_ir.json').read_text(encoding='utf-8'))
        return self._bind_regions(node, image, index, output_dir, summary, package, result.report, ir)

    def _convert_region_batch(self, jobs, identity, work, output_dir, summary):
        import json
        import pymupdf
        from .pdf.production_runtime import create_production_pdf_runtime
        from .production import build_document_id
        from .pdf_output_audit import CleanHandoffRenderer
        item_work = work/'batch'
        item_work.mkdir(parents=True, exist_ok=False)
        source = item_work/'source.pdf'
        with pymupdf.open() as pdf:
            for index, node, image, rows in jobs:
                png = item_work/f'original-{index}.png'
                image.save(png)
                page = pdf.new_page(width=image.width, height=image.height)
                page.insert_image(page.rect, filename=str(png))
            pdf.save(source)
        prepared = _PreparedImagesLayout(jobs, identity)
        factory = self.pdf_runtime_factory or create_production_pdf_runtime
        runtime = factory(models_root=self.models_root, config_path=self.config_path, mcp_root=self.mcp_root,
                          layout_runtime_factory=lambda: prepared, independent_source_pages=True)
        package = item_work/'recognized'; package.mkdir()
        result = runtime.convert(source, package, document_id=build_document_id(source), debug=True)
        ir = json.loads((package/'debug/document_ir.json').read_text(encoding='utf-8'))
        assets = [json.loads(line) for line in (package/'assets_manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        refs = {r['asset_uid']: r['relative_path'] for r in assets if r.get('relative_path')}
        replacements = {}
        for page_index, (index, node, image, rows) in enumerate(jobs):
            page_ir = {**ir, 'blocks': [b for b in ir['blocks'] if b['page_index'] == page_index]}
            uids = {b.get('content', {}).get('asset_uid') for b in page_ir['blocks']}
            unresolved = {r['asset_uid']: r['asset_id'] for r in assets if r['asset_uid'] in uids and r.get('status') == 'UNRESOLVED'}
            markdown = CleanHandoffRenderer().render(page_ir, refs, unresolved_assets=unresolved)['markdown']
            review_count = sum(b.get('review_state') in {'DEFERRED', 'REVIEW_REQUIRED'}
                               for b in page_ir['blocks'] if b.get('visibility') != 'SUPPRESSED_DUPLICATE') + len(unresolved)
            report = {**result.report, 'review_item_count': review_count}
            replacements[id(node)] = self._bind_regions(node, image, index, output_dir, summary,
                                                        package, report, page_ir, markdown=markdown)
        summary['batch_execution'] = {'profile': '10gb', 'independent_image_pages': len(jobs),
                                     'pdf_pipeline_invocations': 1, 'source_pixels_changed': False}
        return replacements

    def _bind_regions(self, node, image, index, output_dir, summary, package, nested_report, ir, *, markdown=None):
        visible = [b for b in ir.get('blocks', []) if b.get('visibility') != 'SUPPRESSED_DUPLICATE']
        source_proof = {'relationship_id': node.relationship_id, 'source_part': node.source_part,
                        'source_locator': node.source_locator, 'media_part': node.media_part,
                        'image_size_px': list(image.size),
                        'markdown_locators': [
                            {'bbox_pixels': b.get('bbox_pdf_pt', []),
                             'text': b.get('content', {}).get('text') or b.get('content', {}).get('latex')}
                            for b in visible if b.get('kind') in {'TEXT', 'TITLE', 'CAPTION', 'FORMULA'}
                            and (b.get('content', {}).get('text') or b.get('content', {}).get('latex'))],
                        'provenance': {'text_region_evidence': list(_candidate_evidence(visible))}}
        source_proof['table_review_blocks'] = _table_review_blocks(visible)
        summary['sources'].append(source_proof)
        for block in visible:
            route_provenance = block.get('provenance', {}).get('route_provenance', {})
            scopes = list(route_provenance.get('boundary_reviews', []))
            if route_provenance.get('source_scope_review'):
                scopes.append(route_provenance['source_scope_review'])
            for scope in scopes:
                summary['region_reviews'].append({**copy.deepcopy(scope),
                    'source_part': node.source_part, 'source_locator': node.source_locator,
                    'relationship_id': node.relationship_id, 'media_part': node.media_part,
                    'bbox_pixels': scope['source_bbox_pdf_pt']})
        summary['converted_images'] += 1
        summary['review_required'] += nested_report.get('review_item_count', 0)
        if not any(b.get('kind') in {'TEXT', 'FORMULA', 'TABLE', 'TITLE', 'CAPTION'} for b in visible):
            self._review(node, summary, 'LAYOUT_DID_NOT_RESOLVE_IMAGE_CONTENT_TYPE')
        if markdown is None:
            markdown = (package/'document.md').read_text(encoding='utf-8')
        assets = {}
        pattern = re.compile(r'!\[(?P<alt>(?:\\.|[^\]\\])*)\]\((?P<path>assets/[^\s)]+)(?:\s+"[^"\n]*")?\)')

        def bind(match):
            source_asset = (package/match['path']).resolve()
            if not source_asset.is_relative_to((package/'assets').resolve()) or not source_asset.is_file():
                raise ValueError('Nested image content asset is not materialized')
            token = f'@@BEMARKDOWN_EMBEDDED_ASSET_{index}_{len(assets)}@@'
            staged_ref = source_asset.relative_to(output_dir).as_posix()
            assets[token] = ImageNode(staged_ref, node.source_part, node.relationship_id,
                node.media_part, alt=re.sub(r'\\([\\\[\]])', r'\1', match['alt']),
                source_locator=f'{node.source_locator}/region[{len(assets)}]',
                provenance={'embedded_image_source': {
                    'relationship_id': node.relationship_id, 'media_part': node.media_part,
                    'image_size_px': list(image.size)}, 'nested_source_asset': match['path']})
            return token

        markdown = pattern.sub(bind, markdown)
        if re.search(r'!\[.*?\]\(assets/', markdown):
            raise ValueError('Nested image content contains an unbound asset reference')
        return ImageContentNode(markdown, assets, node.source_part, node.source_locator,
                                provenance={'embedded_image_source': source_proof})
