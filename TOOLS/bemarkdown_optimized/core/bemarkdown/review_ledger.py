"""Explicit reviewer attestations, bound to the exact task and evidence bytes.

This journal verifies bindings, not the identity of a remote model. Callers must
import reports only from their trusted review workflow, never from source PDFs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

POLICY = 'selective-review-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ReviewLedger:
    """Append-only receipt files; reuse requires the original report to exist."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self._index = None

    def accept(self, task, review_file):
        if not task.get('allow_reuse', True):
            raise ValueError('Mechanical integrity failures must be repaired, not waived')
        path = Path(review_file).resolve()
        raw = path.read_bytes()
        review = json.loads(raw.decode('utf-8-sig'))
        expected = {
            'schema': 'bemarkdown-scoped-review-v1', 'task_id': task['task_id'],
            'task_fingerprint': task['fingerprint'], 'scope': task['scope'],
            'node_ids': task['node_ids'], 'verdict': 'ACCEPTED',
            'reviewer_model': 'gpt-6.1-sol',
        }
        if any(review.get(k) != v for k, v in expected.items()):
            raise ValueError('Review report does not approve this exact scope and input')
        if review.get('source_inspected') is not True or review.get('unresolved') != [] or not str(review.get('conclusion', '')).strip():
            raise ValueError('Incomplete source review cannot be reused')
        if review.get('dependency_hashes') != task['dependency_hashes']:
            raise ValueError('Review dependency bindings differ')
        receipt = {'schema': 'bemarkdown-review-receipt-v1', 'policy': POLICY,
                   'task_id': task['task_id'], 'fingerprint': task['fingerprint'],
                   'scope': task['scope'], 'node_ids': task['node_ids'],
                   'dependency_hashes': task['dependency_hashes'],
                   'report_path': str(path), 'report_sha256': hashlib.sha256(raw).hexdigest(),
                   'trust_basis': 'EXPLICIT_TRUSTED_WORKFLOW_IMPORT_NOT_MODEL_AUTHENTICATION'}
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / (digest(receipt) + '.json')
        if not destination.exists():
            # Exclusive creation, never replace a prior receipt.
            with destination.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write('\n')
        self._index = None
        return receipt

    def _entries(self):
        if self._index is None:
            self._index = {}
            for path in sorted(self.directory.glob('*.json')):
                try:
                    raw = path.read_bytes()
                    row = json.loads(raw.decode('utf-8'))
                    if isinstance(row, dict) and path.stem == digest(row):
                        self._index.setdefault(row.get('task_id'), []).append((path, row, hashlib.sha256(raw).hexdigest()))
                except (OSError, ValueError, TypeError):
                    continue
        return self._index

    def lookup(self, task):
        stale = False
        if not self.directory.is_dir():
            return None, 'NO_RECEIPT'
        for path, receipt, receipt_sha in self._entries().get(task['task_id'], []):
            try:
                if file_sha(path) != receipt_sha:
                    continue
                stale = True
                if any(receipt.get(k) != task.get(k) for k in ('scope', 'node_ids', 'dependency_hashes')):
                    continue
                if receipt.get('policy') != POLICY or receipt.get('fingerprint') != task['fingerprint']:
                    continue
                report_path = Path(receipt['report_path'])
                if file_sha(report_path) != receipt['report_sha256']:
                    continue
                review = json.loads(report_path.read_text(encoding='utf-8-sig'))
                if (review.get('verdict') != 'ACCEPTED' or review.get('reviewer_model') != 'gpt-6.1-sol'
                        or review.get('task_fingerprint') != task['fingerprint']
                        or review.get('dependency_hashes') != task['dependency_hashes']
                        or review.get('scope') != task['scope'] or review.get('node_ids') != task['node_ids']
                        or review.get('unresolved') != [] or review.get('source_inspected') is not True):
                    continue
                return {'receipt_path': str(path), 'receipt_sha256': file_sha(path),
                        'report_path': str(report_path), 'report_sha256': receipt['report_sha256']}, 'MATCHED'
            except (OSError, ValueError, TypeError, KeyError):
                continue
        return None, 'STALE_OR_INVALID_RECEIPT' if stale else 'NO_RECEIPT'
