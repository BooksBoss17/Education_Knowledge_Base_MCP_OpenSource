"""Compact consumer Markdown with byte ranges for every source content node."""

from __future__ import annotations

import hashlib
import re


def _native_line_ids(block):
    provenance = block.get('provenance', {})
    grouping = provenance.get('ordering_atom_grouping', {})
    ids = set(grouping.get('ordered_source_line_ids', []))
    atom = provenance.get('route_provenance', {}).get('ordering_atom_projection', {})
    identity = atom.get('source_identity', {})
    if identity.get('source_line_id'):
        ids.add(identity['source_line_id'])
    if block.get('kind') == 'FORMULA' and provenance.get('inline_native_line_id'):
        ids.add(provenance['inline_native_line_id'])
    return ids


def _joinable(first, second):
    if first is None or second is None or first.get('page_index') != second.get('page_index'):
        return False
    if first.get('kind') not in ('TEXT', 'FORMULA') or second.get('kind') not in ('TEXT', 'FORMULA'):
        return False
    # The source line identity, not mere neighboring placement, establishes an
    # inline relationship. Scanned/display math stays in separate blocks.
    return bool(_native_line_ids(first) & _native_line_ids(second))


def render_clean_document(document, asset_ref_map, render_block, *, renderer_version,
                          unresolved_assets=None):
    blocks = document.get('blocks', [])
    unresolved_assets = dict(unresolved_assets or {})
    chunks, mappings, required_assets, auxiliary = [], [], [], []
    position = 0

    def append(text):
        nonlocal position
        chunks.append(text)
        position += len(text.encode('utf-8'))

    previous = None
    previous_body = ''
    seen_unresolved = set()
    for index, block in enumerate(blocks):
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        inline = block.get('kind') == 'FORMULA' and (
            _joinable(previous, block) or _joinable(block, following))
        body, used_asset = render_block(block, asset_ref_map, inline=inline)
        body = body.strip() or '[Unresolved source content]'
        uid = block.get('content', {}).get('asset_uid')
        if uid in unresolved_assets:
            marker = f'[Unresolved image: {unresolved_assets[uid]}]'
            reference = asset_ref_map.get(uid) or block.get('content', {}).get('asset_ref')
            if body.startswith('[Unresolved '):
                body = marker
            elif reference and re.search(r'!\[[^\]]*\]\(' + re.escape(reference) + r'\)', body):
                body = re.sub(r'!\[[^\]]*\]\(' + re.escape(reference) + r'\)', marker, body)
            else:
                body += '\n' + marker
            seen_unresolved.add(uid)
            used_asset = None
        if previous is None or previous['page_index'] != block['page_index']:
            append('\n\n' if previous else '')
        elif _joinable(previous, block):
            punctuation = r'^[，。；：、！？）)\]〉》]'
            append('' if re.search(punctuation, body) or previous_body.endswith(('、', '（', '(', '[')) else ' ')
        else:
            append('\n\n')
        start = position
        append(body)
        mapping = {'node_id': block['node_id'], 'page_index': block['page_index'],
                   'kind': block['kind'], 'byte_start': start, 'byte_end': position,
                   'sha256': hashlib.sha256(body.encode('utf-8')).hexdigest()}
        if block.get('bbox_pdf_pt'):
            mapping['bbox_pdf_pt'] = list(block['bbox_pdf_pt'])
        if block.get('review_state') not in (None, 'NONE'):
            mapping['review_state'] = block['review_state']
        mappings.append(mapping)
        if used_asset:
            required_assets.append(used_asset)
        previous, previous_body = block, body
    # Preserve explicit unresolved asset accounting even for an unreferenced
    # source asset, without attributing it to an unrelated content node.
    for uid in sorted(set(unresolved_assets) - seen_unresolved):
        append('\n\n')
        marker = f'[Unresolved image: {unresolved_assets[uid]}]'
        start = position
        append(marker)
        auxiliary.append({'asset_uid': uid, 'byte_start': start, 'byte_end': position})
    append('\n')
    markdown = ''.join(chunks)
    result = {'schema': 'bemarkdown-clean-handoff-render-v1', 'renderer_version': renderer_version,
              'markdown': markdown, 'represented_node_ids': [b['node_id'] for b in blocks],
              'required_asset_uids': required_assets, 'node_spans': mappings,
              'offset_unit': 'UTF8_BYTES', 'auxiliary_asset_spans': auxiliary,
              'determinism_sha256': hashlib.sha256(markdown.encode('utf-8')).hexdigest()}
    errors = validate_clean_render(markdown, result, expected_node_ids=[b['node_id'] for b in blocks])
    if errors:
        raise RuntimeError('CLEAN_MARKDOWN_SOURCE_MAPPING_FAILED:' + ';'.join(errors))
    return result


def validate_clean_render(markdown, rendering, *, expected_node_ids=None, expected_node_count=None):
    """Check published byte ranges and source-node conservation, without OCR."""
    data = markdown.encode('utf-8')
    errors = []
    if not isinstance(rendering, dict):
        return ['Invalid compact Markdown source mapping']
    if rendering.get('schema') != 'bemarkdown-clean-handoff-render-v1' or rendering.get('offset_unit') != 'UTF8_BYTES':
        return ['Unsupported compact Markdown source mapping']
    if hashlib.sha256(data).hexdigest() != rendering.get('determinism_sha256'):
        errors.append('Compact Markdown hash mismatch')
    spans = rendering.get('node_spans', [])
    if not isinstance(spans, list) or any(not isinstance(span, dict) for span in spans):
        return errors + ['Invalid compact Markdown source spans']
    ids = [span.get('node_id') for span in spans]
    if any(not isinstance(node_id, str) or not node_id for node_id in ids) or len(ids) != len(set(ids)):
        errors.append('Missing or duplicate compact Markdown node identity')
    if ids != rendering.get('represented_node_ids'):
        errors.append('Compact Markdown node order mismatch')
    if expected_node_ids is not None and ids != expected_node_ids:
        errors.append('Compact Markdown source-node conservation failed')
    if expected_node_count is not None and len(ids) != expected_node_count:
        errors.append('Compact Markdown source-node count mismatch')
    previous_end = 0
    for span in spans:
        start, end = span.get('byte_start'), span.get('byte_end')
        if not isinstance(start, int) or not isinstance(end, int) or not previous_end <= start < end <= len(data):
            errors.append('Invalid compact Markdown byte range')
            continue
        payload = data[start:end]
        try:
            payload.decode('utf-8')
        except UnicodeDecodeError:
            errors.append('Compact Markdown byte range splits a Unicode character')
        if hashlib.sha256(payload).hexdigest() != span.get('sha256'):
            errors.append('Compact Markdown node content hash mismatch')
        previous_end = end
    return errors
