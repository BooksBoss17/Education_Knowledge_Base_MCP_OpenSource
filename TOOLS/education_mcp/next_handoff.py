"""Source-bound review handoff for a single-recognizer API route."""
import copy
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


def summarize_handoff(handoff):
    """Keep every review item and source request; fold pixel-level boundary logs."""
    result = copy.deepcopy(handoff)
    result['schema'] = 'education-mcp-residual-ocr-handoff-summary-v1'
    result['full_evidence_view'] = 'handoff'
    result['summary_scope'] = 'All current review items and source references are retained. Only pixel-level boundary diagnostics are folded; use handoff for those details.'
    for row in result.get('image_region_reviews', []):
        details = row.pop('boundary_evidence', None)
        if details is None:
            continue
        methods = []
        def visit(value):
            if isinstance(value, dict):
                if value.get('schema'):
                    summary = {key:child for key,child in value.items()
                               if key in {'schema','status'} or isinstance(child, bool)}
                    if 'blocking_regions' in value:
                        summary['blocking_regions'] = value['blocking_regions']
                    if summary not in methods:
                        methods.append(summary)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(details)
        row['boundary_summary'] = dict(methods=methods, pixel_diagnostics_view='handoff')
    return result
