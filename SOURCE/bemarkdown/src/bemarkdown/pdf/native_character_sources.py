"""Stable raw-extraction identities shared by native text, math and paint receipts."""
from __future__ import annotations
import copy
import hashlib


def raw_character_ref(block_index, line_index, span_index, character_index, count):
    return {
        'schema': 'pymupdf-raw-character-v1',
        'raw_span_id': (f'raw-text-block-{block_index:04d}'
                        f'-line-{line_index:04d}-span-{span_index:04d}'),
        'block_index': block_index, 'line_index': line_index, 'span_index': span_index,
        'character_index': character_index, 'span_character_count': count,
    }


def raw_character_record(char, span, identity):
    return {
        'source_character': identity,
        'original_text': char['c'],
        'origin_pdf_pt': list(char['origin']) if char.get('origin') is not None else None,
        'bbox_pdf_pt': list(char['bbox']) if char.get('bbox') is not None else None,
        'font': span.get('font'),
        'synthetic': bool(char.get('synthetic', False)),
    }


def character_identity_key(ref):
    """Page-local identity; caller must additionally bind document and page."""
    return ref['raw_span_id'], ref['character_index']


def native_text_source_receipt(route, rows, text):
    lines = []
    for row in rows:
        spans = row.get('spans', [])
        characters = [copy.deepcopy(char) for span in spans for char in span.get('source_characters', [])]
        raw_text = ''.join(c['original_text'] for c in characters)
        complete = bool(characters) and raw_text == row.get('text') and all(
            char.get('origin_pdf_pt') and char.get('font') and not char.get('synthetic')
            for char in characters
        ) and all(
            not span.get('unicode_mapping_issue') and bool(span.get('source_characters'))
            for span in spans
        )
        lines.append({'source_line_id': row.get('evidence_id'), 'text': row.get('text'),
                      'characters': characters,
                      'identity_status': 'COMPLETE' if complete else 'INCOMPLETE'})
    return {'schema': 'native-text-source-receipt-v1', 'document_id': route['document_id'],
            'page_index': route['page_index'], 'lines': lines,
            'decoded_text': text, 'decoded_text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'coverage_status': 'NOT_VERIFIED', 'authorizes_exclusion': False}
