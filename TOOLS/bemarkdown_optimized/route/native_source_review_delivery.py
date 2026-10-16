"""Deliver approved visible-input transcripts after completion, retaining graphics.

The private index is supplied by the caller. No benchmark identifiers or source
paths are embedded here; upstream model requests and their decisions stay intact.
"""
import copy
import hashlib
import json
from pathlib import Path

from bemarkdown.asset_local_evidence import apply_bound_source_review


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_native_source_reviews(document, original_document, index_path):
    index = _read(index_path)
    if index.get('schema') != 'approved-native-visible-review-index-v1':
        raise ValueError('NATIVE_REVIEW_INDEX_SCHEMA')
    originals = {b['node_id']: b for b in original_document['blocks']}
    source_sha = document['source']['sha256']
    if source_sha != original_document['source']['sha256']:
        raise ValueError('NATIVE_REVIEW_DOCUMENT_CHANGED')
    changes = []
    for block in document['blocks']:
        original = originals.get(block['node_id'])
        if original is None or block != original or block.get('kind') != 'TEXT':
            continue
        provenance = block.get('provenance', {}).get('route_provenance', {}).get('three_model_text_evidence')
        if not provenance or not provenance['resolver'].get('review_required'):
            continue
        if provenance['resolver'].get('source_verified'):
            continue
        request = provenance['request']
        input_sha = request['crop_sha256']
        entries = [e for e in index['entries']
                   if e['source_sha256'] == source_sha and e['actual_input_sha256'] == input_sha]
        if not entries:
            continue
        if len(entries) != 1:
            raise ValueError('NATIVE_REVIEW_AMBIGUOUS_INDEX')
        entry = entries[0]
        for key in ('review', 'receipt'):
            if _sha(entry[key + '_path']) != entry[key + '_sha256']:
                raise ValueError('NATIVE_REVIEW_ARTIFACT_CHANGED:' + key)
        receipt = _read(entry['receipt_path'])
        packet_ref = receipt.get('packet', {})
        packet_path = packet_ref.get('path', receipt.get('packet_path'))
        packet = _read(packet_path)
        crop = request['source_provenance']['crop']
        if (packet['source_sha256'] != source_sha
                or packet['page_index'] != block['page_index']
                or packet['image']['sha256'] != input_sha
                or packet['target_bbox_pixels'] != [0, 0, crop['width'], crop['height']]
                or packet['source_scope'] != 'ENTIRE_ACTUAL_NATIVE_VISIBLE_INPUT_EXCLUDING_WHITE_OWNERSHIP_MASKS'
                or packet.get('prediction_or_reference_text_supplied') is not False
                or any(crop.get(k) != v for k, v in packet['source_geometry'].items())
                or provenance['local_evidence']['input_sha256'] != input_sha):
            raise ValueError('NATIVE_REVIEW_VISIBLE_SCOPE_MISMATCH')
        # Both fields refer to this actual masked input, not the unused full-page
        # request. The shared validator verifies receipt, packet and image bytes.
        answer = {'original_image_sha256': input_sha, 'actual_input_sha256': input_sha,
                  'repair_task': {'source_sha256': source_sha}, 'repair_result': {},
                  'prior_native_answer': copy.deepcopy(provenance['agent_answer'])}
        reviewed = apply_bound_source_review(answer, entry['review_path'], entry['receipt_path'])
        transcript = reviewed['markdown'].strip()
        before_sha = hashlib.sha256(json.dumps(block, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        block['content']['text'] = transcript
        # Do not claim diagram classification or close unrelated review items.
        block['provenance']['native_visible_source_review_delivery'] = {
            'stage': 'POST_COMPLETION_UNCHANGED_NATIVE_SOURCE_REVIEW',
            'original_native_block_sha256': before_sha,
            'source_document_sha256': source_sha, 'actual_input_sha256': input_sha,
            'source_scope': packet['source_scope'], 'source_verified': True,
            'content_kind_verified': False, 'preserve_source_image': True,
            'transcription_status': 'SUCCESS', 'human_verified': False,
            'source_review': reviewed['source_review'],
            'upstream_agent_answer_modified': False}
        changes.append({'node_id': block['node_id'], 'actual_input_sha256': input_sha,
                        'before_block_sha256': before_sha,
                        'review_sha256': entry['review_sha256'],
                        'receipt_sha256': entry['receipt_sha256']})
    return {'version': 'native-visible-source-review-delivery-v1',
            'index_sha256': _sha(index_path), 'changed_nodes': changes,
            'new_model_calls': 0, 'upstream_request_inputs_modified': False,
            'preserve_source_image': True}
