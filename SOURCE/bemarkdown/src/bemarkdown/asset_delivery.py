"""Project source-bound asset results into every actual Markdown occurrence.

No scoring references, XML occurrence counts, or best-substring searches enter
this projection. An image and its auxiliary transcript share one logical copy.
"""
import hashlib
import json

from .markdown_asset_refs import image_references, renamed_image_reference

VERSION = 'asset-delivery-ranges-v1'


def _uid(prefix, value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
    return prefix + hashlib.sha256(payload).hexdigest()[:32]


def project_asset_transcriptions(source, assets, answers, *, source_sha256,
                                 renamed_paths=None, excluded_uids=()):
    renamed_paths = renamed_paths or {}
    excluded_uids = set(excluded_uids)
    by_path = {r['relative_path']: r for r in assets if r.get('relative_path')}
    if len(by_path) != sum(bool(r.get('relative_path')) for r in assets):
        raise ValueError('ASSET_DELIVERY_DUPLICATE_PATH')
    chunks = []
    records = []
    ordinal = {}
    cursor = 0
    char_offset = 0
    byte_offset = 0
    for match in image_references(source):
        row = by_path.get(match.destination)
        if row is None:
            raise ValueError('ASSET_DELIVERY_UNBOUND_IMAGE_REFERENCE')
        prefix = source[cursor:match.begin]
        chunks.append(prefix)
        char_offset += len(prefix)
        byte_offset += len(prefix.encode('utf-8'))
        asset_uid = row['asset_uid']
        ordinal[asset_uid] = ordinal.get(asset_uid, 0) + 1
        answer = answers.get(asset_uid)
        original = match[0]
        value = (renamed_image_reference(match, renamed_paths[match.destination])
                 if match.destination in renamed_paths else original)
        mode = 'IMAGE_PRESERVED'
        transcript = ''
        if answer and answer.get('status') == 'SUCCESS' and answer.get('markdown', '').strip():
            transcript = answer['markdown']
            if answer.get('preserve_image', False):
                value += '\n\n图中文字：\n\n' + transcript
                mode = 'IMAGE_AND_TRANSCRIPT_ONE_SOURCE_COPY'
            else:
                value = transcript
                mode = 'TEXT_REPLACED_IMAGE'
        elif asset_uid in excluded_uids:
            if answer and (answer.get('status') != 'SUCCESS'
                           or answer.get('text_presence') not in {None, 'NONE'}):
                raise ValueError('ASSET_DELIVERY_CANNOT_EXCLUDE_UNRESOLVED_TEXT')
            value = ''
            mode = 'EXCLUDED_BY_BOUND_RETENTION'
        elif answer and answer.get('status') != 'SUCCESS':
            mode = 'SOURCE_REVIEW_IMAGE_PRESERVED'
        source_uid = _uid('source-occurrence:', {
            'document': source_sha256, 'asset_uid': asset_uid,
            'source_part': row.get('source_part'), 'source_locator': row.get('source_locator'),
            'rendered_reference_ordinal': ordinal[asset_uid]})
        end_chars = char_offset + len(value)
        end_bytes = byte_offset + len(value.encode('utf-8'))
        records.append({'asset_uid': asset_uid,
            'content_uid': answer.get('content_uid') if answer else None,
            'source_occurrence_uid': source_uid,
            'source_part': row.get('source_part'), 'source_locator': row.get('source_locator'),
            'relationship_id': row.get('relationship_id'),
            'source_image_sha256': row.get('content_sha256'),
            'input_markdown_char_range': [match.begin, match.finish],
            'output_markdown_char_range': [char_offset, end_chars],
            'output_markdown_byte_range': [byte_offset, end_bytes],
            'source_reference': original, 'output_text_sha256': hashlib.sha256(value.encode('utf-8')).hexdigest(),
            'delivery_mode': mode, 'transcription': transcript,
            'transcription_evidence_kind': (answer.get('repair_result', {}).get('evidence_kind')
                                          if answer else None),
            'rendered_reference_ordinal': ordinal[asset_uid]})
        chunks.append(value)
        char_offset = end_chars
        byte_offset = end_bytes
        cursor = match.finish
    chunks.append(source[cursor:])
    output = ''.join(chunks)
    output_bytes = output.encode('utf-8')
    output_sha = hashlib.sha256(output_bytes).hexdigest()
    for record in records:
        a, b = record['output_markdown_byte_range']
        if hashlib.sha256(output_bytes[a:b]).hexdigest() != record['output_text_sha256']:
            raise ValueError('ASSET_DELIVERY_BYTE_RANGE_MISMATCH')
        record['render_occurrence_uid'] = _uid('render-occurrence:', {
            'source_occurrence_uid': record['source_occurrence_uid'],
            'markdown_sha256': output_sha, 'byte_range': [a, b]})
    if len({r['source_occurrence_uid'] for r in records}) != len(records):
        raise ValueError('ASSET_DELIVERY_OCCURRENCE_IDENTITY_COLLISION')
    receipt = {'version': VERSION, 'source_document_sha256': source_sha256,
        'input_markdown_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
        'output_markdown_sha256': output_sha, 'occurrences': records,
        'actual_markdown_occurrences': len(records),
        'word_pagination_certified': False, 'xml_count_used_as_render_count': False,
        'gold_read': False}
    return output, receipt
