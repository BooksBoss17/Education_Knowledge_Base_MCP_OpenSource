"""Occurrence-specific reviewed heading templates plus executable delivery proof.

Templates are scoped to one immutable source, never global permission to remove
similar images. Candidate source drawings, current Markdown and retained owners
are verified afresh. This module returns plan suggestions and never edits files.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

POLICY = 'source-heading-backdrop-rules-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _crops(value):
    if isinstance(value, dict):
        found = [value] if isinstance(value.get('source_graphics_render'), dict) else []
        return found + [r for child in value.values() for r in _crops(child)]
    if isinstance(value, list):
        return [r for child in value for r in _crops(child)]
    return []


def _residual_node(node):
    if node.get('kind') not in {'IMAGE', 'OTHER'}:
        return False
    prov = node.get('provenance', {})
    if set(prov.get('review_reasons', [])) - {'PRIMARY_ADAPTER_UNRESOLVED'} or prov.get('warnings'):
        return False
    route = prov.get('route_provenance', {})
    kinds = route.get('source_candidate_kinds')
    if kinds != ['VECTOR_VISUAL_FALLBACK']:
        return False
    return route.get('route_decision', {}).get('adapter') in {'CONTENT_REVIEW_REQUIRED', 'IMAGE_RENDER_CROP'}


def _retained_owners(node, document, references):
    route = node.get('provenance', {}).get('route_provenance', {})
    ownership = route.get('source_graphics_ownership')
    if not ownership:
        return []
    visible = {b['node_id']: b for b in document['blocks']}
    all_nodes = [*document['blocks'], *document.get('suppressed_blocks', [])]
    required = []
    for owner in ownership.get('excluded_owners', []):
        candidates = [b for b in all_nodes if b['page_index'] == node['page_index']
                      and b.get('provenance', {}).get('route_id') == owner.get('route_id')]
        if len(candidates) != 1 or owner.get('document_id') != document['document_id']:
            return None
        target = candidates[0]
        if target['node_id'] not in visible:
            redirect = target.get('provenance', {}).get('source_image_containment', {})
            if (redirect.get('basis') != 'COMPLETE_SOURCE_CROP_CONTAINS_CHILD_PIXELS'
                    or redirect.get('source_evidence_retained') is not True):
                return None
            target = visible.get(redirect.get('owner_node_id'), {})
        # Prove retention, not quality acceptance. A retained image may still
        # require boundary review; that obligation must remain unchanged.
        if (target.get('kind') != 'IMAGE' or target.get('content', {}).get('source_status')
                not in {'SUCCESS', 'REVIEW_REQUIRED'}):
            return None
        asset = next((a for a in document['assets'] if a['asset_uid'] == target['content'].get('asset_uid')), None)
        if not asset:
            return None
        expected = asset.get('sha256', asset.get('content_sha256'))
        refs = [r for r in references if r.get('source_document_node_id') == target['node_id']
                and r.get('sha256') == expected and r.get('source_mapping') == 'RENDERER_SPAN_AND_REVIEW_REPLAY']
        if not refs:
            return None
        required.extend(r['image_id'] for r in refs)
    return sorted(set(required))


def evaluate(package, references, *, source_path, source_sha256, markdown, ir_path, template_path):
    started = time.perf_counter()
    package, ir_path, template_path = Path(package), Path(ir_path), Path(template_path)
    result = {'policy_version': POLICY, 'status': 'UNAVAILABLE', 'actions': [], 'records': [], 'actual_exclusions': 0}
    if not template_path.is_file() or not ir_path.is_file():
        return {**result, 'reason': 'NO_SOURCE_SCOPED_TEMPLATES_OR_IR'}
    tracked = [ir_path, template_path, package/'document.md', package/'conversion_report.json']
    tracked += [p for p in (package/'agent_review.json', package/'document.reviewed.md') if p.exists()]
    input_hashes = {str(p):sha(p) for p in tracked}
    import pymupdf
    from PIL import Image
    from bemarkdown.pdf.background_vector_rules import SCHEMA, source_vector_signature, match_template, heading_text
    from bemarkdown.pdf.graphics_render import render_source_graphics
    from bemarkdown.pdf.native_delivery_coverage import attach_replayed_native_text_receipts, audit_omitted_text_delivery
    from bemarkdown.pdf.native_delivery_spans import unchanged_delivery_spans
    registry = json.loads(template_path.read_text('utf-8-sig'))
    if registry.get('schema') != SCHEMA or registry.get('source_sha256') != source_sha256:
        raise ValueError('BACKGROUND_TEMPLATE_SOURCE_MISMATCH')
    templates = registry.get('templates', [])
    if not templates:
        return {**result, 'reason': 'NO_APPROVED_HEADING_TEMPLATES'}
    if sha(source_path) != source_sha256:
        raise ValueError('BACKGROUND_SOURCE_SHA_MISMATCH')
    document = json.loads(ir_path.read_text('utf-8-sig'))
    if document.get('source', {}).get('sha256') != source_sha256:
        raise ValueError('BACKGROUND_IR_SOURCE_MISMATCH')
    original = (package/'document.md').read_text('utf-8')
    report = json.loads((package/'conversion_report.json').read_text('utf-8'))
    journal_path = package/'agent_review.json'
    journal = json.loads(journal_path.read_text('utf-8')) if journal_path.exists() else {
        'original_sha256': hashlib.sha256(original.encode()).hexdigest(), 'revisions': []}
    spans = unchanged_delivery_spans(original, markdown, report['markdown_render'], journal)
    nodes = {b['node_id']: b for b in document['blocks']}
    grouped = defaultdict(list)
    for ref in references:
        node = nodes.get(ref.get('source_document_node_id'))
        if not node or not _residual_node(node) or ref.get('source_mapping') != 'RENDERER_SPAN_AND_REVIEW_REPLAY':
            continue
        crops = _crops(node.get('provenance', {}))
        if len(crops) != 1:
            continue
        crop = crops[0]
        bbox = crop.get('bbox_pdf_pt', [])
        # This policy covers short heading backdrops only; paragraphs/panels are
        # deliberately a different classifier, not silently generalized here.
        if (len(bbox) != 4 or bbox[2]-bbox[0] > 120 or bbox[3]-bbox[1] > 45
                or crop['source_graphics_render'].get('method') != 'MUPDF_SOURCE_GRAPHICS_WITHOUT_NATIVE_PROSE'):
            continue
        grouped[node['page_index']].append((ref, node, crop))
    markdown_sha = hashlib.sha256(markdown.encode()).hexdigest()
    registry_sha = sha(template_path)
    with pymupdf.open(source_path) as source:
        for page_index, items in sorted(grouped.items()):
            page = source[page_index]
            attach_replayed_native_text_receipts(document, page, page_index)
            for ref, node, crop in items:
                record = {'image_id': ref['image_id'], 'source_node_id': node['node_id'],
                          'page_index': page_index, 'status': 'KEEP_PENDING_REVIEW'}
                result['records'].append(record)
                audit = crop['source_graphics_render']
                kwargs = {'preserve_regions': audit.get('native_text_preserved_regions_pdf_pt', []),
                          'exclude_regions': audit.get('semantic_image_excluded_regions_pdf_pt', [])}
                if kwargs['preserve_regions']:
                    record['reason'] = 'PRESERVED_VISUAL_REGION_IN_RESIDUAL'; continue
                raster, proof = render_source_graphics(page, crop['bbox_pdf_pt'], crop['dpi'], **kwargs)
                path = (package/ref['asset_name']).resolve(strict=True)
                if not path.is_relative_to(package.resolve()) or sha(path) != ref['sha256']:
                    raise ValueError('BACKGROUND_ASSET_IDENTITY_MISMATCH')
                with Image.open(path) as image:
                    rgba = image.convert('RGBA')
                    pixels_match = (rgba.getextrema()[3] == (255, 255)
                                    and rgba.size == (raster.width, raster.height)
                                    and rgba.convert('RGB').tobytes() == raster.samples)
                if not pixels_match:
                    record['reason'] = 'CURRENT_ASSET_NOT_REPRODUCED'; continue
                coverage = audit_omitted_text_delivery(proof['native_text_omission_receipt'], document,
                                                       markdown, delivery_spans=spans)
                record['text_coverage'] = coverage
                if heading_text(coverage) is None:
                    record['reason'] = 'HEADING_DELIVERY_NOT_PROVEN'; continue
                retained = _retained_owners(node, document, references)
                if retained is None:
                    record['reason'] = 'EXCLUDED_SEMANTIC_OWNER_NOT_RETAINED'; continue
                owner_refs = [r for r in references if r['image_id'] in retained]
                for owner in owner_refs:
                    owner_path = (package/owner['asset_name']).resolve(strict=True)
                    if not owner_path.is_relative_to(package.resolve()) or sha(owner_path) != owner['sha256']:
                        raise ValueError('BACKGROUND_RETAINED_OWNER_SHA_MISMATCH')
                vector_only, _ = render_source_graphics(page, crop['bbox_pdf_pt'], crop['dpi'], **kwargs,
                                                        omit_raster_paint=True, collect_text_receipt=False)
                no_raster_contribution = raster.samples == vector_only.samples
                signature = source_vector_signature(page, crop['bbox_pdf_pt'], excluded_regions=kwargs['exclude_regions'],
                                                     raster_contribution_absent=no_raster_contribution)
                record['vector_status'] = signature['status']
                template = match_template(signature, coverage, templates, source_sha256)
                if not template:
                    record['reason'] = 'NO_REVIEWED_HEADING_TEMPLATE_MATCH'; continue
                record.update(status='ELIGIBLE_HEADING_BACKGROUND', template_id=template['template_id'],
                              required_retained_image_ids=retained, asset_sha256=ref['sha256'],
                              source_sha256=source_sha256, base_sha256=markdown_sha,
                              templates_sha256=registry_sha, source_vector_signature=signature['signature_sha256'],
                              coverage_status='SOURCE_CONTENT_DELIVERED', raster_contribution_absent=no_raster_contribution)
                record['retained_owner_review_states'] = {r['image_id']:r.get('source_node_review_state') for r in owner_refs}
                evidence = {k:v for k,v in record.items() if k != 'text_coverage'}
                result['actions'].append({'image_id':ref['image_id'], 'action':'remove_decoration', 'text':'',
                                          'source_evidence':json.dumps({'policy':POLICY, **evidence},ensure_ascii=False)})
    if input_hashes != {str(p):sha(p) for p in tracked} or sha(source_path) != source_sha256:
        raise ValueError('BACKGROUND_INPUT_CHANGED_DURING_SCREENING')
    result.update(status='SUCCESS', eligible_occurrences=len(result['actions']), input_hashes=input_hashes,
                  examined_occurrences=len(result['records']), templates_sha256=registry_sha,
                  base_sha256=markdown_sha, elapsed_seconds=time.perf_counter()-started)
    return result


def applicable_actions(result, references, explicit_actions=(), *, blocked_image_ids=()):
    """Never authorize another same-SHA occurrence or remove a required owner."""
    current = {r['image_id']: r for r in references}
    explicit = {r['image_id']: r for r in explicit_actions}
    blocked = set(blocked_image_ids)
    records = {r['image_id']:r for r in result.get('records', [])}
    allowed = []
    for action in result.get('actions', []):
        key = action['image_id']; record = records[key]; ref = current.get(key, {})
        if (key in explicit or key in blocked or ref.get('sha256') != record.get('asset_sha256')
                or ref.get('source_document_node_id') != record.get('source_node_id')
                or ref.get('source_part') != f"page:{record['page_index']}"):
            continue
        if any(owner not in current or explicit.get(owner, {}).get('action', 'preserve') != 'preserve'
               for owner in record.get('required_retained_image_ids', [])):
            continue
        allowed.append(action)
    return allowed
