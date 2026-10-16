"""Conservative ordering of numbered exercises from one resolved OCR region."""
import re


def order_ocr_exercise_columns(blocks, page_width):
    groups = {}
    for position, block in enumerate(blocks):
        atom = block.get('provenance', {}).get('route_provenance', {}).get('ordering_atom_projection', {})
        if block.get('kind') == 'TEXT' and atom.get('origin_type') == 'OCR_RESOLVED_LINE':
            groups.setdefault(atom['parent_content_id'], []).append((position, block, atom))
    applied = []
    for parent, entries in groups.items():
        # Never move text across a formula, image, table or another parent.
        positions = [entry[0] for entry in entries]
        if len(entries) < 6 or positions != list(range(positions[0], positions[-1] + 1)):
            continue
        members = [entry[1] for entry in entries]
        intervals = sorted((b['bbox_pdf_pt'][0], b['bbox_pdf_pt'][2]) for b in members)
        right = intervals[0][1]
        gutters = []
        for left, end in intervals[1:]:
            if left - right >= max(18, page_width * .03):
                gutters.append((left + right) / 2)
            right = max(right, end)
        if len(gutters) != 1:
            continue
        split = gutters[0]
        columns = [sorted([b for b in members if (b['bbox_pdf_pt'][0] < split) == left],
                          key=lambda b: (b['bbox_pdf_pt'][1], b['bbox_pdf_pt'][0]))
                   for left in (True, False)]
        if any(len(column) < 3 for column in columns):
            continue
        spans = [(min(b['bbox_pdf_pt'][1] for b in col), max(b['bbox_pdf_pt'][3] for b in col)) for col in columns]
        if min(s[1] for s in spans) - max(s[0] for s in spans) < .5 * min(s[1] - s[0] for s in spans):
            continue
        numbers = []
        qualified = True
        for col in columns:
            anchors = []
            for b in col:
                match = re.match(r'^\s*(\d{1,3})[.．、](?!\d)\s*\S', str(b.get('content', {}).get('text') or ''))
                if match:
                    anchors.append(int(match.group(1)))
            if len(anchors) < 3 or anchors != list(range(anchors[0], anchors[0] + len(anchors))):
                qualified = False
                break
            numbers.extend(anchors)
        if not qualified or numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            continue
        ordered = [b for col in columns for b in col]
        if [b['node_id'] for b in ordered] == [b['node_id'] for b in members]:
            continue
        receipt = {'policy': 'source-bounded-numbered-ocr-columns-v1', 'parent_content_id': parent,
                   'split_x': split, 'exercise_numbers': numbers, 'needs_review': True,
                   'prior_node_ids': [b['node_id'] for b in members],
                   'ordered_node_ids': [b['node_id'] for b in ordered]}
        for position, block in zip(positions, ordered):
            blocks[position] = block
            block['order_key']['final_page_order_index'] = position
            if block.get('review_state') in (None, 'NONE', 'WARNING'):
                block['review_state'] = 'REVIEW_REQUIRED'
            reasons = block['provenance'].setdefault('review_reasons', [])
            if 'OCR_EXERCISE_COLUMN_ORDER_REQUIRES_REVIEW' not in reasons:
                reasons.append('OCR_EXERCISE_COLUMN_ORDER_REQUIRES_REVIEW')
            block['provenance']['route_provenance']['ordering_atom_projection']['column_order'] = {
                key: receipt[key] for key in ('policy', 'parent_content_id', 'split_x', 'needs_review')
            }
        applied.append(receipt)
    return {'applied': bool(applied), 'groups': applied}
