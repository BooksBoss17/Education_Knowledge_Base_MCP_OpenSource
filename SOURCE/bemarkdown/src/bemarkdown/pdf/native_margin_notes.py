"""Attach repeated narrow native margin paragraphs to their aligned main text."""
from collections import defaultdict


def _paragraph(block):
    provenance = block.get('provenance', {}).get('route_provenance', {})
    return (block.get('kind') == 'TEXT' and block.get('bbox_pdf_pt')
            and provenance.get('native_text_trust') == 'HIGH'
            and provenance.get('ordering_atom_projection', {}).get('origin_type') == 'NATIVE_LINE'
            and len(str(block.get('content', {}).get('text') or '').splitlines()) >= 3)


def attach_native_margin_notes(blocks, page_width):
    """Require two distinct paragraph pairs sharing one narrow margin lane.

    Equal-width columns, isolated sidebars, OCR text and partial vertical matches
    retain their existing order. Text, source geometry and image positions relative
    to main paragraphs remain intact; only the associated note block is moved.
    """
    paragraphs = [b for b in blocks if _paragraph(b)]
    pairs = defaultdict(list)
    for note in paragraphs:
        nb = note['bbox_pdf_pt']
        nw, nh = nb[2] - nb[0], nb[3] - nb[1]
        if not 0 < nw <= page_width * 0.27 or nh <= 0:
            continue
        matches = []
        for main in paragraphs:
            if main is note:
                continue
            mb = main['bbox_pdf_pt']
            mw = mb[2] - mb[0]
            if mw < page_width * 0.45 or nw > mw * 0.55:
                continue
            side = 'LEFT' if nb[2] <= mb[0] else 'RIGHT' if mb[2] <= nb[0] else None
            if side is None:
                continue
            gap = mb[0] - nb[2] if side == 'LEFT' else nb[0] - mb[2]
            if not 4 <= gap <= page_width * 0.1:
                continue
            overlap = min(nb[3], mb[3]) - max(nb[1], mb[1])
            line_height = nh / len(note['content']['text'].splitlines())
            if overlap < nh * 0.9 or abs(nb[3] - mb[3]) > max(8, line_height * 0.6):
                continue
            matches.append((main, side))
        if len(matches) == 1:
            main, side = matches[0]
            pairs[side].append((note, main))
    original = [b['node_id'] for b in blocks]
    attachments = []
    for side, candidates in pairs.items():
        main_count = len({main['node_id'] for _, main in candidates})
        if main_count < 2 or main_count != len(candidates):
            continue
        lefts = [note['bbox_pdf_pt'][0] for note, _ in candidates]
        rights = [note['bbox_pdf_pt'][2] for note, _ in candidates]
        if max(lefts) - min(lefts) > page_width * 0.03 or max(rights) - min(rights) > page_width * 0.03:
            continue
        for note, main in candidates:
            before = blocks.index(note)
            blocks.remove(note)
            blocks.insert(blocks.index(main) + 1, note)
            after = blocks.index(note)
            if after != before:
                proof = {'main_node_id': main['node_id'], 'note_node_id': note['node_id'],
                         'margin_side': side, 'before_index': before, 'after_index': after,
                         'basis': 'REPEATED_NARROW_NATIVE_PARAGRAPHS_WITH_ALIGNED_BOTTOMS'}
                note.setdefault('provenance', {})['native_margin_attachment'] = proof
                attachments.append(proof)
    if sorted(original) != sorted(b['node_id'] for b in blocks):
        raise RuntimeError('NATIVE_MARGIN_NOTE_CONSERVATION_FAILED')
    if attachments:
        for index, block in enumerate(blocks):
            block.setdefault('order_key', {})['final_page_order_index'] = index
    return {'version': 'source-native-margin-note-order-v1',
            'moved_count': len(attachments), 'attachments': attachments}
