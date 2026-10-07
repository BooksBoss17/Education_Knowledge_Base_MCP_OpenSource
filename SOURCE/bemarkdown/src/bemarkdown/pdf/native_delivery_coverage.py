"""Prove omitted native text is still delivered; this is not image classification."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from .native_character_sources import character_identity_key


def attach_replayed_native_text_receipts(document, page, page_index):
    """Enrich an in-memory historical IR copy by explicit source line identity.

    Caller verifies immutable PDF/IR identity first. Existing conversion receipts
    are not replaced, and historical traces are never claimed for this replay.
    """
    from .source_evidence import _native_text_lines
    from .native_character_sources import native_text_source_receipt
    lines = {line['evidence_id']: line for line in _native_text_lines(page)}
    for block in document.get('blocks', []):
        if block['page_index'] != page_index:
            continue
        prov = block.get('provenance', {}).get('route_provenance', {})
        if (prov.get('route_decision', {}).get('adapter') != 'NATIVE_TEXT_BRIDGE'
                or 'native_text_source_receipt' in prov):
            continue
        grouping = block.get('provenance', {}).get('ordering_atom_grouping', {})
        ids = grouping.get('ordered_source_line_ids')
        if not ids:
            identity = prov.get('ordering_atom_projection', {}).get('source_identity', {})
            ids = [identity['source_line_id']] if identity.get('source_line_id') else prov.get('source_line_ids', [])
        if not ids or any(key not in lines for key in ids):
            continue
        rows = [lines[key] for key in ids]
        value = '\n'.join(row['text'] for row in rows).replace('\r\n', '\n').replace('\r', '\n')
        if value != block['content'].get('text'):
            continue
        receipt = native_text_source_receipt({'document_id': document['document_id'], 'page_index': page_index}, rows, value)
        receipt['evidence_origin'] = 'READ_ONLY_SOURCE_REPLAY_NOT_HISTORICAL_TRACE'
        prov['native_text_source_receipt'] = receipt


def _delivered_characters(block, document_id):
    provenance = block.get('provenance', {}).get('route_provenance', {})
    content = block.get('content', {})
    if content.get('source_status') != 'SUCCESS' or block.get('review_state') != 'NONE':
        return []
    receipt = provenance.get('native_text_source_receipt')
    if receipt:
        value = content.get('text')
        if (receipt.get('document_id') != document_id or receipt.get('page_index') != block['page_index']
                or not isinstance(value, str) or value != receipt.get('decoded_text')
                or hashlib.sha256(value.encode()).hexdigest() != receipt.get('decoded_text_sha256')):
            return []
        lines = receipt.get('lines', [])
        if not lines or any(line.get('identity_status') != 'COMPLETE' for line in lines):
            return []
        raw_lines = [''.join(char['original_text'] for char in line['characters']) for line in lines]
        if raw_lines != [line.get('text') for line in lines]:
            return []
        if '\n'.join(raw_lines).replace('\r\n', '\n').replace('\r', '\n') != value:
            return []
        return [char for line in lines for char in line['characters']]
    receipt = provenance.get('native_formula_source_receipt')
    if receipt:
        value = content.get('latex')
        if (receipt.get('document_id') == document_id and receipt.get('page_index') == block['page_index']
                and receipt.get('source_identity_status') == 'COMPLETE'
                and isinstance(value, str) and value == receipt.get('decoded_latex')
                and hashlib.sha256(value.encode()).hexdigest() == receipt.get('decoded_latex_sha256')):
            return receipt.get('characters', [])
    return []


def audit_omitted_text_delivery(omission, document, markdown, *, delivery_spans=None):
    """Bind actual paint receipts to unchanged, uniquely anchored current output.

    Exact node rendering is intentional: edited/merged/removed Markdown requires
    fresh receipts; substring/fuzzy text matches cannot silently clear coverage.
    """
    from ..pdf_document_ir import DraftMarkdownAnchorReader, _render_block, document_asset_presentation_ref
    from ..pdf_output_audit import CleanHandoffRenderer
    anchors = Counter(row['node_id'] for row in DraftMarkdownAnchorReader().read(markdown))
    assets = {str(row['asset_uid']): document_asset_presentation_ref(row) for row in document.get('assets', [])}
    owners = defaultdict(list)
    public_renderer = CleanHandoffRenderer()
    markdown_sha = hashlib.sha256(markdown.encode()).hexdigest()
    for block in document.get('blocks', []):
        if block['page_index'] != omission.get('page_index'):
            continue
        unchanged = anchors[block['node_id']] == 1 and markdown.count(_render_block(block, assets)) == 1
        if delivery_spans is not None:
            span = delivery_spans.get(block['node_id'], {})
            # Public Markdown has no inline node comments. Bind by original
            # renderer UTF-8 spans replayed through the entire review journal.
            unchanged = (span.get('basis') == 'RENDERER_BYTES_AND_EXACT_REVIEW_REPLAY'
                         and span.get('current_markdown_sha256') == markdown_sha
                         and span.get('page_index') == block['page_index']
                         and span.get('kind') == block['kind']
                         and span.get('text') in {
                             public_renderer._render_block(block, assets, inline=inline)[0]
                             for inline in (False, True)})
        if not unchanged:
            continue
        for char in _delivered_characters(block, document['document_id']):
            if not char.get('source_character') or char.get('synthetic'):
                continue
            owners[character_identity_key(char['source_character'])].append((block['node_id'], char))
    rows = []
    for operation in omission.get('operations', []):
        for char in operation.get('characters', []):
            if char.get('intersects_crop') is False:
                continue
            matches = []
            if char.get('identity_status') == 'UNIQUE' and char.get('source_character'):
                for node, candidate in owners.get(character_identity_key(char['source_character']), []):
                    if (candidate['original_text'] == char['original_text']
                            and candidate.get('origin_pdf_pt') == char.get('origin_pdf_pt')
                            and candidate.get('font') == char.get('font')):
                        matches.append(node)
            rows.append({'source_character': char.get('source_character'),
                         'original_text': char.get('original_text'),
                         'owner_node_ids': sorted(set(matches)),
                         'status': 'COVERED' if len(matches) == 1 else 'NOT_PROVEN'})
    complete = (omission.get('schema') == 'native-text-omission-receipt-v1'
                and omission.get('origin') == 'ACTUAL_MUPDF_PAINT_CALLBACK'
                and omission.get('identity_status') == 'COMPLETE'
                and not omission.get('errors')
                and all(row['status'] == 'COVERED' for row in rows))
    return {'schema': 'native-text-delivery-coverage-v1',
            'status': 'COVERED' if complete else 'NOT_PROVEN',
            'scope': 'OMITTED_NATIVE_CHARACTERS_ONLY',
            'characters_required': len(rows), 'characters_covered': sum(row['status'] == 'COVERED' for row in rows),
            'characters': rows, 'markdown_sha256': hashlib.sha256(markdown.encode()).hexdigest(),
            'authorizes_exclusion': False,
            'remaining_gates': ['SEMANTIC_IMAGE_AND_VECTOR_OWNERSHIP', 'RESIDUAL_IMAGE_CLASSIFICATION']}
