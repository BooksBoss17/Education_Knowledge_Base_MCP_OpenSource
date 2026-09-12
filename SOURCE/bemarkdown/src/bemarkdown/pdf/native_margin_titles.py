"""Keep prominent native section titles incorrectly labeled as running headers."""

from __future__ import annotations

import copy
import re
import statistics


def refine_native_margin_titles(routes, source_evidence, page_record):
    diagnostic = {'version': 'source-native-margin-title-v1', 'promoted_routes': []}
    height = page_record.get('geometry', {}).get('height_pt', 0)
    if not height or page_record.get('native_text_trust') not in ('HIGH', 'MEDIUM'):
        return routes, diagnostic
    lines = source_evidence.get('native_text', [])
    body_heights = []
    for line in lines:
        for span in line.get('spans', []):
            box = span.get('bbox_pdf_pt')
            if (box and 0.25 * height < (box[1] + box[3]) / 2 < 0.8 * height
                    and len(re.findall(r'[\u3400-\u9fff]', span.get('text', ''))) >= 6):
                body_heights.append(box[3] - box[1])
    if len(body_heights) < 4:
        return routes, diagnostic
    body_height = statistics.median(body_heights)
    revised = copy.deepcopy(routes)
    for route in revised:
        if route['adapter'] != 'NATIVE_TEXT_BRIDGE' or 'HEADER_FOOTER' not in route['semantic_evidence']:
            continue
        provenance = route['provenance']
        ids = set(provenance.get('native_text_evidence_ids', [])) | set(provenance.get('evidence_ids', []))
        selected = [line for line in lines if line.get('evidence_id') in ids]
        if len(selected) != 1:
            continue
        line = selected[0]
        text = line.get('text', '').strip()
        box = line['bbox_pdf_pt']
        if not (2 <= len(text) <= 40 and re.search(r'[\u3400-\u9fff]', text)
                and 0.065 * height < (box[1] + box[3]) / 2 < 0.20 * height):
            continue
        visible = [span['bbox_pdf_pt'][3] - span['bbox_pdf_pt'][1]
                   for span in line.get('spans', []) if span.get('text', '').strip()
                   and re.search(r'[\u3400-\u9fff]', span.get('text', ''))]
        if not visible or min(visible) < body_height * 1.20:
            continue
        proof = {'source_line_id': line['evidence_id'], 'source_text': text,
                 'body_median_span_height_pt': body_height,
                 'title_minimum_span_height_pt': min(visible),
                 'original_semantics': list(route['semantic_evidence'])}
        route['semantic_evidence'] = ['TITLE']
        route['decision_reason_codes'].append('PROMINENT_NATIVE_SECTION_TITLE')
        provenance['native_section_title'] = proof
        diagnostic['promoted_routes'].append({'route_id': route['route_id'], **proof})
    return revised, diagnostic
