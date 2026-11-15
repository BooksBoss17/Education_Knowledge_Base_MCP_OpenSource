"""Source-bound review handoff for a single-recognizer API route."""
import hashlib
import json


def build_handoff(source, markdown, report):
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if report.get('source', {}).get('sha256') != source_sha:
        raise ValueError('NEXT_HANDOFF_SOURCE_CHANGED')
    return dict(schema='education-mcp-local-api-handoff-v1', source_sha256=source_sha,
        original_markdown_sha256=hashlib.sha256(markdown).hexdigest(),
        candidate_count=0, items=[], evidence_available=True,
        candidate_basis='Single-recognizer transcription has no three-model disagreement candidates. Zero candidates is not zero OCR errors.',
        scope='Source views, generated assets and exact review edits remain available. No automatic image screening or full-text quality certification.',
        image_region_reviews=report.get('image_content', {}).get('region_reviews', []),
        content_reviews=report.get('review_items', []),
        content_review_count=report.get('review_item_count', 0),
        image_screening_summary={'status':'NOT_RUN_BY_ROUTE'}, image_screening_tasks=[],
        model_roles=report.get('visual_serial', {}).get('stage_identities', {}),
        native_image_sources=report.get('image_content', {}).get('sources', []),
        formula_fallback=report.get('formula_ocr', {}))
