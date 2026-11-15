"""Local API adapters for images and unresolved formula previews in native Word."""
from __future__ import annotations

import hashlib
import re
import time
import zipfile
from pathlib import Path

from .formula_review import FormulaReview, _initialize_report, _iter_formula_nodes
from .ir import HyperlinkNode, ImageContentNode, ImageNode, TableNode
from .image_input import IMAGE_SUFFIXES, InvalidRasterImageError
from .pdf.local_model_api import OcrApiRuntime, read_config


class NativeFormulaApi(FormulaReview):
    """Reuse conservative formula validation; cache only inside one document."""

    def __init__(self, config_path, runtime=None):
        super().__init__()
        self.api = runtime or OcrApiRuntime('formula', read_config(config_path)['roles']['formula'])

    def process(self, document, output_dir, report):
        from PIL import Image
        started = time.perf_counter()
        nodes = [n for n in _iter_formula_nodes(document)
                 if n.status == 'RENDERED_FALLBACK' and n.rendered_ref]
        summary = _initialize_report(report)
        summary['candidates'] = len(nodes)
        groups, results = {}, []
        for node in nodes:
            path = (output_dir/node.rendered_ref).resolve()
            if not path.is_relative_to(output_dir.resolve()) or not path.is_file():
                results.append(self._failed_result(node, report, 'Formula preview missing or outside package'))
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, {'path': path, 'nodes': []})['nodes'].append(node)
        summary['unique_png'] = len(groups)
        summary['deduplicated_occurrences'] = sum(len(g['nodes'])-1 for g in groups.values())
        if groups:
            # Protocol/network errors fail the job; a successful empty prediction
            # is handled by the safety gate and preserves the original preview.
            outputs = self.api.recognize_batch([{'path': str(g['path']), 'sha256': sha}
                                               for sha, g in groups.items()])
            identity = self.api.identity
            self._runtime_fingerprint = {
                **identity, 'model_identifier': identity['model_id'],
                'model_sha256': identity['model_fingerprint']}
            for (sha, group), raw in zip(groups.items(), outputs, strict=True):
                with Image.open(group['path']) as image:
                    width, height = image.size
                # Native serializer owns the math delimiters.
                raw = raw.strip()
                for left, right in (('$$','$$'), ('\\[','\\]'), ('\\(','\\)'), ('$','$')):
                    if raw.startswith(left) and raw.endswith(right) and len(raw) >= len(left)+len(right):
                        raw = raw[len(left):-len(right)].strip()
                        break
                key = hashlib.sha256((sha+identity['model_fingerprint']).encode()).hexdigest()
                for index, node in enumerate(group['nodes']):
                    results.append(self._evaluate_result(node, report, raw_latex=raw,
                        png_sha256=sha, width=width, height=height, cache_key=key, cache_hit=index>0))
        summary.update(model_id=self.api.identity.get('model_id', 'api:formula'),
                       runtime_fingerprint=self._runtime_fingerprint,
                       inference_calls=len(self.api.metrics['requests']))
        report['timing']['formula_ocr_inference_seconds'] = time.perf_counter()-started
        report['runtime']['vision_or_ocr_used'] = bool(groups)
        self._finalize_document_report(report, results)
        return results


def _inline_lists(document):
    def nested(children):
        yield children
        for child in children:
            if isinstance(child, HyperlinkNode):
                yield from nested(child.children)
    for block in document.blocks:
        if isinstance(block, TableNode):
            for row in block.rows:
                for cell in row:
                    for paragraph in cell.blocks:
                        yield from nested(paragraph.children)
        else:
            yield from nested(block.children)


class NativeImageApi:
    def __init__(self, visual_runtime):
        self.visual_runtime = visual_runtime

    def process(self, document, output_dir, report, **_):
        summary = report['image_content'] = {'route': 'LOCAL_MODEL_API', 'converted_images': 0,
            'review_required': 0, 'sources': [], 'region_reviews': []}
        pattern = re.compile(r'!\[(?P<alt>(?:\\.|[^\]\\])*)\]\((?P<path>assets/[^\s)]+)(?:\s+"[^"\n]*")?\)')
        # The native scanner deliberately exports original media. A cropped,
        # rotated or flipped Word occurrence must not silently OCR hidden pixels.
        appearances = {}
        from lxml import etree
        from .namespaces import NS
        with zipfile.ZipFile(report['source']['path']) as archive:
            for part in {n.source_part for children in _inline_lists(document)
                         for n in children if isinstance(n, ImageNode)}:
                if part in archive.namelist():
                    appearances[part] = etree.fromstring(archive.read(part),
                        parser=etree.XMLParser(resolve_entities=False,no_network=True))
        image_index = 0
        for children in list(_inline_lists(document)):
            for position, node in enumerate(children):
                if not isinstance(node, ImageNode) or not node.asset_path or node.external_target:
                    continue
                path = (output_dir/node.asset_path).resolve()
                if not path.is_relative_to(output_dir.resolve()) or not path.is_file():
                    raise ValueError('DOCX_IMAGE_SOURCE_MISSING_OR_OUTSIDE_PACKAGE')
                tree=appearances.get(node.source_part)
                owners=tree.xpath(node.source_locator,namespaces=NS) if tree is not None and node.source_locator else []
                appearance=[]
                for owner in owners:
                    appearance.extend(owner.xpath('.//a:srcRect | .//a:xfrm',namespaces=NS))
                transformed=any(any(value not in ('0','false','') for key,value in item.attrib.items()
                    if key in {'l','t','r','b','rot','flipH','flipV'}) for item in appearance)
                if transformed:
                    summary['review_required'] += 1
                    summary['region_reviews'].append({'source_part':node.source_part,
                        'source_locator':node.source_locator,'media_part':node.media_part,
                        'reason':'WORD_IMAGE_APPEARANCE_REQUIRES_REVIEW_ORIGINAL_PRESERVED'})
                    continue
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    summary['review_required'] += 1
                    summary['region_reviews'].append({'source_part':node.source_part,
                        'source_locator':node.source_locator,'media_part':node.media_part,
                        'reason':'NON_RASTER_SOURCE_PRESERVED_WITHOUT_OCR'})
                    continue
                index = image_index
                image_index += 1
                package = output_dir/'_bemarkdown'/'embedded'/f'{index:06d}'
                package.mkdir(parents=True)
                try:
                    result = self.visual_runtime.convert_image(path, package, document_id=f'embedded-{index:06d}')
                except InvalidRasterImageError as exc:
                    summary['review_required'] += 1
                    summary['region_reviews'].append({'source_part':node.source_part,
                        'source_locator':node.source_locator,'media_part':node.media_part,
                        'reason':'INVALID_RASTER_SOURCE_PRESERVED','detail':str(exc)})
                    continue
                stage_identities = result.report['visual_serial']['stage_identities']
                if any(identity.get('invoked') for identity in stage_identities.values()):
                    report['runtime']['vision_or_ocr_used'] = True
                markdown = (package/'document.md').read_text(encoding='utf-8')
                markdown = re.sub(r'<!-- page:\d+ -->', '', markdown).strip()
                assets = {}
                def bind(match):
                    asset = (package/match['path']).resolve()
                    if not asset.is_relative_to((package/'assets').resolve()) or not asset.is_file():
                        raise ValueError('NESTED_ASSET_MISSING_OR_OUTSIDE_PACKAGE')
                    token = f'@@BEMARKDOWN_EMBEDDED_ASSET_{index}_{len(assets)}@@'
                    assets[token] = ImageNode(asset.relative_to(output_dir).as_posix(), node.source_part,
                        node.relationship_id, node.media_part, alt=match['alt'],
                        source_locator=f'{node.source_locator}/region[{len(assets)}]',
                        provenance={'embedded_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                    return token
                markdown = pattern.sub(bind, markdown)
                if not markdown.strip():
                    # An empty layout/text result cannot erase an embedded image.
                    summary['review_required'] += 1
                    continue
                children[position] = ImageContentNode(markdown, assets, node.source_part,
                    node.source_locator, provenance={'image_source_policy': 'ORIGINAL_EMBEDDED_MEDIA'})
                summary['converted_images'] += 1
                summary['review_required'] += result.report.get('review_item_count', 0)
                summary['sources'].append({'source_part': node.source_part,
                    'source_locator': node.source_locator, 'media_part': node.media_part,
                    'stage_identities': stage_identities})
        return summary
