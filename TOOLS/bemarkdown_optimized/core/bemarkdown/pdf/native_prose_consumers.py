"""Source-backed candidates for conflicting native paragraph/image consumers.

This preserves native paragraph candidates for review; it does not certify their
semantic role, remove image pixels, or promote captions and short labels.
"""
import hashlib
import re

POLICY = 'canonical-native-prose-consumer-v1'


def _native_line_runs(rows):
    """PDF storage blocks may split every line; use source line geometry."""
    runs = []
    for row in sorted(rows, key=lambda item: (item['bbox_pdf_pt'][1], item['bbox_pdf_pt'][0], item['evidence_id'])):
        box = row['bbox_pdf_pt']
        continuous = False
        if runs:
            previous = runs[-1][-1]['bbox_pdf_pt']
            height = max(box[3]-box[1], previous[3]-previous[1])
            continuous = (height > 0
                          and box[1]-previous[1] >= .5*min(box[3]-box[1], previous[3]-previous[1])
                          and box[1]-previous[3] <= 2*height
                          and abs(box[0]-previous[0]) <= 2.5*height)
        if continuous:
            runs[-1].append(row)
        else:
            runs.append([row])
    return runs


def native_prose_candidates(plan, source_evidence):
    if plan.get('provenance', {}).get('native_text_trust') != 'HIGH':
        return {}
    native = {row['evidence_id']: row for row in source_evidence.get('native_text', [])
              if row.get('evidence_id')}
    claims = {}
    for route in plan.get('routes', []):
        provenance = route.get('provenance', {})
        semantics = set(route.get('semantic_evidence', []))
        box = provenance.get('bbox_pdf_pt')
        if (route.get('adapter') != 'NATIVE_TEXT_BRIDGE'
                or 'MODEL_CANONICAL_REGION' not in provenance.get('candidate_kinds', [])
                or not semantics or not semantics <= {'TEXT', 'TEXT_LIKE'}
                or not isinstance(box, (list, tuple)) or len(box) != 4):
            continue
        candidates = []
        for identity in set(provenance.get('native_text_evidence_ids', [])):
            row = native.get(identity)
            if not row or not row.get('source_block_id') or not row.get('text', '').strip():
                continue
            bounds = row.get('bbox_pdf_pt')
            if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4
                    or not box[0] <= (bounds[0]+bounds[2])/2 <= box[2]
                    or not box[1] <= (bounds[1]+bounds[3])/2 <= box[3]
                    or any(span.get('unicode_mapping_issue') for span in row.get('spans', []))):
                continue
            candidates.append(row)
        for rows in _native_line_runs(candidates):
            rows.sort(key=lambda row: (row['bbox_pdf_pt'][1], row['bbox_pdf_pt'][0], row['evidence_id']))
            # Strong paragraph evidence is required in addition to a canonical
            # body route. Multiple font runs on one baseline do not qualify.
            if len({round(row['bbox_pdf_pt'][1], 2) for row in rows}) < 3:
                continue
            text = '\n'.join(row['text'] for row in rows)
            sentences = sum(text.count(c) for c in '。！？') + len(re.findall(r'(?<!\d)[.!?](?:\s|$)', text))
            if sentences < 2:
                continue
            identities = [row['evidence_id'] for row in rows]
            for row in rows:
                claims[row['evidence_id']] = {
                    'policy': POLICY, 'native_route_id': route['route_id'],
                    'source_block_id': row['source_block_id'], 'source_line_id': row['evidence_id'],
                    'paragraph_basis': 'CANONICAL_ROUTE_AND_CONTIGUOUS_NATIVE_LINES',
                    'supporting_source_block_ids': sorted({item['source_block_id'] for item in rows}),
                    'source_text_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
                    'source_bbox_pdf_pt': list(row['bbox_pdf_pt']),
                    'supporting_source_line_ids': identities, 'sentence_boundary_count': sentences,
                    'status': 'SOURCE_BACKED_CONSUMER_CANDIDATE_REQUIRES_REVIEW',
                }
    return claims
