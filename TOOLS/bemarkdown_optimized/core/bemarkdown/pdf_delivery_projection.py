"""Source/content/render identities over the existing clean PDF renderer.

This preserves its Markdown bytes. A node span is deliberately not advertised
as a character-level geometry mapping for a smaller annotation region.
"""
import hashlib
import json
import math

from .pdf.clean_markdown import validate_clean_render

VERSION = 'pdf-delivery-occurrences-v1'


def _uid(prefix, value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')
    return prefix + hashlib.sha256(data).hexdigest()[:32]


def build_pdf_delivery_projection(document, rendered):
    from .residual_source_delivery import bind_residual_render_coverage
    markdown = rendered['markdown']
    blocks = document['blocks']
    ids = [b['node_id'] for b in blocks]
    errors = validate_clean_render(markdown, rendered, expected_node_ids=ids)
    if errors:
        raise ValueError('PDF_DELIVERY_INVALID_CLEAN_RENDER:' + ';'.join(errors))
    by_id = {b['node_id']: b for b in blocks}
    if len(by_id) != len(blocks):
        raise ValueError('PDF_DELIVERY_DUPLICATE_SOURCE_NODE')
    data = markdown.encode('utf-8')
    source = document.get('source', {})
    source_sha = source.get('sha256') or source.get('source_sha256')
    origin = document.get('provenance', {}).get('source_input_identity', {})
    if origin and origin.get('processing_input_sha256') != source_sha:
        raise ValueError('PDF_DELIVERY_ORIGINAL_SOURCE_MAPPING_INPUT_MISMATCH')
    original_pages = origin.get('original_page_indices_0')
    document_id = document['document_id']
    md_sha = hashlib.sha256(data).hexdigest()
    records = []
    groups = {}
    for span in rendered['node_spans']:
        node_id = span['node_id']
        block = by_id[node_id]
        prov = block.get('provenance', {})
        extra = prov.get('source_image_text', {})
        parent = extra.get('parent_node_id')
        if extra and not parent:
            raise ValueError('PDF_DELIVERY_AUXILIARY_PARENT_NODE_REQUIRED')
        if parent and parent not in by_id:
            raise ValueError('PDF_DELIVERY_AUXILIARY_PARENT_MISSING')
        original = by_id[parent] if parent else block
        if parent:
            actual_uid = original.get('content', {}).get('asset_uid')
            if actual_uid != extra.get('parent_asset_uid'):
                raise ValueError('PDF_DELIVERY_AUXILIARY_ASSET_PARENT_MISMATCH')
        origin_prov = original.get('provenance', {})
        delivery = origin_prov.get('asset_text_delivery', {})
        original_node = original['node_id']
        source_uid = _uid('source-occurrence:', {
            'source_document_sha256': source_sha, 'document_id': document_id,
            'node_id': original_node, 'page_index': original['page_index'],
            'source_region_ids': original.get('source_region_ids', []),
            'source_unit_ids': original.get('source_unit_ids', [])})
        content_uid = delivery.get('content_uid') or _uid('content:', {
            'source_document_sha256': source_sha, 'document_id': document_id,
            'source_content': original.get('content', {}),
            'source_unit_ids': original.get('source_unit_ids', [])})
        a, b = span['byte_start'], span['byte_end']
        payload = data[a:b]
        if hashlib.sha256(payload).hexdigest() != span['sha256']:
            raise ValueError('PDF_DELIVERY_BYTE_RANGE_CHANGED')
        route = origin_prov.get('route_provenance', {})
        source_lines = route.get('ordering_atom_projection', {}).get('source_identity', {})
        record = {'node_id': node_id, 'original_source_node_id': original_node,
            'page_index': block['page_index'], 'kind': block['kind'],
            'original_page_index_0': (original_pages[block['page_index']]
                if original_pages is not None else None),
            'content_uid': content_uid, 'source_occurrence_uid': source_uid,
            'render_occurrence_uid': _uid('render-occurrence:', {
                'source_occurrence_uid': source_uid, 'node_id': node_id,
                'markdown_sha256': md_sha, 'byte_range': [a, b]}),
            'output_markdown_byte_range': [a, b], 'output_text_sha256': span['sha256'],
            'bbox_pdf_pt': original.get('bbox_pdf_pt'),
            'source_region_ids': original.get('source_region_ids', []),
            'source_unit_ids': original.get('source_unit_ids', []),
            'source_line_identity': source_lines,
            'asset_uid': delivery.get('asset_uid') or original.get('content', {}).get('asset_uid'),
            'source_image_sha256': delivery.get('original_image_sha256') or route.get('render_crop', {}).get('content_sha256'),
            'actual_input_sha256': delivery.get('actual_input_sha256') or
                route.get('render_crop', {}).get('content_sha256'),
            'delivery_role': 'AUXILIARY_VISIBLE_TRANSCRIPT' if parent else 'PRIMARY_SOURCE_NODE',
            'geometry_scope': 'SOURCE_NODE_ONLY_NOT_SMALLER_CHARACTER_REGION',
            'character_geometry_certified': False}
        records.append(record)
        groups.setdefault(source_uid, []).append(record['render_occurrence_uid'])
    line_proposals = []
    for span in rendered['node_spans']:
        block = by_id[span['node_id']]
        evidence = block.get('provenance', {}).get('source_line_evidence')
        if not evidence:
            continue
        visible = evidence.get('visible_input')
        transform = evidence.get('crop_transform')
        local = evidence['local_evidence']
        if (not visible or not transform or evidence.get('scope') != 'ENTIRE_VISUAL_INPUT_NO_CONTEXT'
                or local['input_sha256'] != visible['sha256']
                or visible.get('recipe',{}).get('original_sha256') != evidence.get('source_image_sha256')
                or visible.get('recipe',{}).get('no_pixels_cropped') is not True):
            line_proposals.append({'node_id': block['node_id'], 'status': 'SOURCE_LINE_INPUT_OR_TRANSFORM_UNPROVEN'})
            continue
        from .source_line_correspondence import propose_source_line_correspondence
        factor = visible['recipe']['nearest_neighbor_factor']
        crop_origin = transform['origin_pdf_pt']
        scale = transform['pdf_points_per_render_pixel'] / factor
        lines = []
        for row in local.get('pp_lines', []):
            bbox = row.get('bbox_local_px')
            if not bbox:
                continue
            lines.append({**row, 'bbox_page_pdf_pt': [crop_origin[0] + bbox[0]*scale,
                crop_origin[1] + bbox[1]*scale, crop_origin[0] + bbox[2]*scale, crop_origin[1] + bbox[3]*scale]})
        a, b = span['byte_start'], span['byte_end']
        body = data[a:b].decode('utf-8')
        prefix = evidence.get('output_prefix', '')
        if prefix and not body.startswith(prefix):
            line_proposals.append({'node_id': block['node_id'], 'status': 'DECLARED_VISIBLE_TRANSCRIPT_PREFIX_MISMATCH'})
            continue
        prefix_bytes = len(prefix.encode('utf-8'))
        proposed = propose_source_line_correspondence(lines, body[len(prefix):], source_binding={
            'document_id': document_id, 'source_node_id': block['node_id'],
            'source_image_sha256': evidence['source_image_sha256'],
            'actual_local_input_sha256': local['input_sha256'], 'crop_transform': transform,
            'markdown_sha256': md_sha})
        for row in proposed['lines']:
            if 'output_utf8_byte_range' in row:
                x, y = row['output_utf8_byte_range']
                row['final_markdown_utf8_byte_range'] = [a + prefix_bytes + x, a + prefix_bytes + y]
                if hashlib.sha256(data[a + prefix_bytes + x:a + prefix_bytes + y]).hexdigest() != row['output_text_sha256']:
                    raise ValueError('PDF_SOURCE_LINE_FINAL_BYTE_BINDING_CHANGED')
        line_proposals.append({'node_id': block['node_id'], 'proposal': proposed})
    return {'version': VERSION, 'document_id': document_id,
        'source_document_sha256': source_sha, 'markdown_sha256': md_sha,
        'processing_input_sha256': source_sha,
        'original_source_document_sha256': origin.get('original_source_sha256'),
        'source_logical_copies': len(groups), 'render_nodes': len(records),
        'occurrences': records, 'logical_copy_render_members': groups,
        'source_line_correspondence_proposals': line_proposals,
        'auxiliary_asset_spans': rendered.get('auxiliary_asset_spans', []),
        'residual_asset_existing_text_coverage': bind_residual_render_coverage(document, rendered),
        'read_only_projection': True, 'gold_read': False,
        'small_region_character_mapping_claimed': False}


def locate_pdf_region(projection, *, page_index, bbox_pdf_pt, tolerance_pt=0.5, scope_reviews=None, markdown=None):
    """Qualify source scope without reading target text or choosing a substring."""
    target = tuple(float(x) for x in bbox_pdf_pt)
    if len(target) != 4 or not all(math.isfinite(x) for x in target) or target[2] <= target[0] or target[3] <= target[1]:
        raise ValueError('PDF_DELIVERY_REGION_RECTANGLE_INVALID')
    candidates = []
    if scope_reviews is not None:
        if markdown is None:raise ValueError('PDF_SCOPE_CURRENT_MARKDOWN_REQUIRED')
        reviewed=scope_reviews.locate(projection,markdown,page_index=page_index,bbox_pdf_pt=target)
        if reviewed is not None:return reviewed
    exact = []
    for row in projection['occurrences']:
        bbox = row.get('bbox_pdf_pt')
        if row['page_index'] != page_index or not bbox:
            continue
        if min(target[2], bbox[2]) <= max(target[0], bbox[0]) or min(target[3], bbox[3]) <= max(target[1], bbox[1]):
            continue
        candidates.append(row['render_occurrence_uid'])
        if all(abs(a - b) <= tolerance_pt for a, b in zip(target, bbox, strict=True)):
            exact.append(row['render_occurrence_uid'])
    if not candidates:
        return {'status': 'UNMEASURED_NO_INTERSECTING_SOURCE_NODE', 'render_occurrence_uids': []}
    if len(exact) == 1 and len(candidates) == 1:
        return {'status': 'FULL_SOURCE_NODE_SCOPE_MATCH', 'render_occurrence_uids': exact,
                'not_character_level_alignment': True}
    if len(exact) > 1 and len(exact) == len(candidates):
        members = [r for r in projection['occurrences'] if r['render_occurrence_uid'] in exact]
        sources = {r.get('source_occurrence_uid') for r in members}
        if None not in sources and len(sources) == 1:
            return {'status': 'FULL_SOURCE_LOGICAL_COPY_SCOPE_MATCH', 'render_occurrence_uids': exact,
                'source_logical_copies': 1, 'not_character_level_alignment': True,
                'transcription_occurrence_uids': [r['render_occurrence_uid'] for r in members
                    if r.get('delivery_role') == 'AUXILIARY_VISIBLE_TRANSCRIPT'
                    or r.get('kind') in {'TEXT','CAPTION','TITLE','FORMULA','TABLE'}],
                'accuracy_certified': False}
    return {'status': 'UNMEASURED_SMALL_REGION_CHARACTER_BINDING_UNPROVEN',
            'render_occurrence_uids': candidates, 'no_best_substring_selected': True}
