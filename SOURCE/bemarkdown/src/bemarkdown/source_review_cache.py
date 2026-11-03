"""Read-only, content-bound reuse of approved independent source decisions."""
import hashlib
import json
import copy
from pathlib import Path

from .asset_local_evidence import apply_bound_source_review

VERSION = 'approved-source-review-cache-v1'
REUSE_POLICY = 'approved-full-visible-input-alias-v2'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ApprovedSourceReviewCache:
    def __init__(self, index_path):
        self.path = Path(index_path).resolve()
        raw = self.path.read_bytes()
        self.index = json.loads(raw.decode('utf-8'))
        self.index_sha256 = hashlib.sha256(raw).hexdigest()
        if self.index.get('version') != VERSION:
            raise ValueError('SOURCE_REVIEW_CACHE_VERSION')
        self.hits = []

    def lookup(self, *, source_sha256, original_image_sha256, actual_input_sha256, task, original_image_path=None, proof_work=None):
        if _sha(self.path) != self.index_sha256:
            raise ValueError('SOURCE_REVIEW_CACHE_INDEX_CHANGED')
        matched = [r for r in self.index['entries']
            if r['source_sha256'] == source_sha256 and
               r['actual_input_sha256'] == actual_input_sha256 and
               r['scope'] == 'ONLY_VISIBLE_INPUT_NO_CONTEXT_INFERENCE' and
               r['input_contract_version'] == task.candidate_version]
        if not matched:
            return None
        if len(matched) != 1:
            raise ValueError('SOURCE_REVIEW_CACHE_DUPLICATE_APPROVAL')
        entry = matched[0]
        for field in ('review', 'receipt'):
            if _sha(entry[field]['path']) != entry[field]['sha256']:
                raise ValueError('SOURCE_REVIEW_CACHE_APPROVAL_CHANGED')
        if (task.task_type != 'VISUAL_TEXT_EXTRACT' or task.context or task.target_bbox_pixels is not None
                or task.source_sha256 != source_sha256 or task.input_sha256 != actual_input_sha256
                or task.bbox_coordinate_space != 'pixel'):
            return None
        review = json.loads(Path(entry['review']['path']).read_text(encoding='utf-8'))
        viewed = next((r for r in review['source_hashes'] if r['sha256'] == actual_input_sha256), None)
        if viewed is None or _sha(viewed['path']) != actual_input_sha256:
            raise ValueError('SOURCE_REVIEW_ACTUAL_VIEW_CHANGED')
        from PIL import Image
        with Image.open(viewed['path']) as image:
            if tuple(task.bbox) != (0, 0, image.width, image.height):
                return None
        alias = None
        if entry['original_image_sha256'] != original_image_sha256:
            if original_image_path is None or proof_work is None:
                return None
            if _sha(original_image_path) != original_image_sha256:
                raise ValueError('SOURCE_ALIAS_ORIGINAL_CHANGED')
            inspected_original = next((r for r in review['source_hashes']
                if r['sha256'] == entry['original_image_sha256']), None)
            if inspected_original is None or _sha(inspected_original['path']) != inspected_original['sha256']:
                raise ValueError('SOURCE_ALIAS_APPROVED_ORIGINAL_CHANGED')
            from .asset_text_content import prepare_visible_input
            proof_work = Path(proof_work)
            approved = prepare_visible_input(inspected_original['path'], proof_work / 'approved')
            current = prepare_visible_input(original_image_path, proof_work / 'current')
            for visible in (approved, current):
                if (visible['sha256'] != actual_input_sha256 or not visible['recipe']['no_pixels_cropped']
                        or visible['recipe']['version'] != entry['input_contract_version']):
                    return None
            proof = {'policy': REUSE_POLICY, 'source_sha256': source_sha256,
                'original_container_sha256': original_image_sha256,
                'visually_inspected_original_sha256': entry['original_image_sha256'],
                'actual_input_sha256': actual_input_sha256, 'approved_visible_input': approved,
                'current_visible_input': current, 'whole_target_only': True,
                'new_visual_reviews': 0, 'proof_source_sha256': _sha(__file__),
                'input_recipe_source_sha256': _sha(Path(prepare_visible_input.__code__.co_filename))}
            proof_text = json.dumps(proof, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
            proof_path = proof_work / (hashlib.sha256(proof_text.encode('utf-8')).hexdigest() + '.json')
            proof_path.write_text(proof_text, encoding='utf-8')
            alias = {'path': str(proof_path), 'sha256': _sha(proof_path), 'policy': REUSE_POLICY,
                     'new_visual_reviews': 0}
        pending = {'status': 'REVIEW', 'uncertain': True, 'markdown': '',
            'content_uid': task.node_id, 'kind': 'DIAGRAM', 'preserve_image': True,
            'repair_task': task.to_dict(), 'repair_result': {
                'task_id': task.task_id, 'task_fingerprint': task.fingerprint,
                'status': 'REVIEW', 'text': '', 'uncertain': True, 'patches': [],
                'reason': 'Pending source evidence before any local or cloud inference.',
                'text_presence': 'UNCERTAIN', 'visual_content_kind': 'UNCERTAIN',
                'evidence_kind': 'SOURCE_IMAGE_VERIFIED', 'source_verified': False},
            'original_image_sha256': entry['original_image_sha256'],
            'actual_input_sha256': actual_input_sha256}
        accepted = apply_bound_source_review(pending, entry['review']['path'], entry['receipt']['path'])
        accepted['original_image_sha256'] = original_image_sha256
        if alias:
            accepted['source_review']['visible_input_alias'] = alias
            accepted['repair_result']['independent_source_review'] = copy.deepcopy(accepted['source_review'])
        accepted['source_review_cache'] = {'index_path': str(self.path), 'index_sha256': _sha(self.path),
                                         'version': VERSION}
        accepted['source_review_cache']['reuse_policy'] = REUSE_POLICY
        accepted['source_review_cache']['visible_input_alias'] = alias
        self.hits.append({'source_sha256': source_sha256, 'content_uid': task.node_id,
                          'actual_input_sha256': actual_input_sha256})
        return accepted
