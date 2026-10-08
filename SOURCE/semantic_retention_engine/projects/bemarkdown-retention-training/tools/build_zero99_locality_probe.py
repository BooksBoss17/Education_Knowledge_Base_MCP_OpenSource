"""Source-only bounded-locality view for repeated heading-neighbor raw-K assets."""

import hashlib

import json

def local_view(usage, raw, nodes, feature):
    target = usage['occurrence']['bbox']
    page = usage['occurrence']['page']
    selected = []
    for node in nodes:
        if node['page_index'] != page or node['kind'] in ('IMAGE', 'OTHER'):
            continue
        bbox = node['bbox_pdf_pt']
        gap = max(target[0]-bbox[2], bbox[0]-target[2], 0) + max(target[1]-bbox[3], bbox[1]-target[3], 0)
        if gap > 32:
            continue
        value = raw[node['byte_start']:node['byte_end']]
        if node.get('sha256'):
            assert hashlib.sha256(value).hexdigest() == node['sha256']
        if value.decode('utf-8').strip():
            selected.append({'bbox': bbox, 'text': value.decode('utf-8'), 'start': node['byte_start'],
                             'end': node['byte_end'], 'sha256': hashlib.sha256(value).hexdigest()})
    selected.sort(key=lambda x: x['start'])
    position = next(x for x in feature['uses'] if x['use_id'] == usage['use_id'])
    view = {'record_type': 'source_layout_not_contents_printed_in_target',
            'target_bbox_pdf_pt': target, 'source_page_size_pdf_pt': position['page_size'],
            'source_position': {'in_top_12_percent': position['header_band'], 'in_bottom_10_percent': position['footer_band']},
            'within_document_visual_repetition': {'distinct_similar_pages': feature['distinct_similar_visual_pages'],
                                                  'distinct_exact_pixel_pages': feature['distinct_exact_pixel_pages']},
            'MD_text_blocks_within_32pt': [{'bbox': x['bbox'], 'text': x['text']} for x in selected],
            'selection_scope': 'Only nearby source text; text printed in separate PDF objects is not automatically in target PNG; distant body text not attributed to this region.'}
    return json.dumps(view, ensure_ascii=False, separators=(',', ':')), selected
