"""Checked replacement of an existing textbook with durable rollback evidence.

The two published directories cannot be replaced atomically as a pair on Windows.
An exclusive revision lock and write-ahead move journal preserve recovery evidence;
a handled failure restores the original pair and import record without deleting data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import uuid

FILES = runpy.run_path(str(Path(__file__).with_name('atomic_files.py')))
atomic_json = FILES['atomic_json']
replace_path = FILES['replace_path']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)


def snapshot(folder):
    folder = Path(folder)
    if not folder.is_dir() or linked(folder):
        raise ValueError('Revision requires an ordinary existing directory')
    result = {}
    for item in folder.iterdir():
        if not item.is_file() or linked(item):
            raise ValueError('Unexpected nested directory or link in textbook')
        result[item.name] = digest(item)
    return result


def validate_contents(book, images, manifest):
    actual_book = snapshot(book)
    expected_book = {row['file']: row['sha256'] for row in manifest['sections']}
    if actual_book != expected_book:
        raise ValueError('Published or staged textbook content differs from its manifest')
    actual_images = snapshot(images)
    expected_images = {row['sha256'] for row in manifest['images'] if row['action'] == 'preserve'}
    if set(actual_images.values()) != expected_images:
        raise ValueError('Published or staged images differ from their manifest')
    if any(not name.startswith(value[:24]+'.') for name, value in actual_images.items()):
        raise ValueError('Unexpected image filename in textbook')
    return {'book': actual_book, 'images': actual_images}


def revise(staged_book, staged_images, target, diagrams, record, manifest,
           expected_import_sha256, recovery_root):
    """Caller holds the job review lock and has passed normal publication gates."""
    if not isinstance(expected_import_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_import_sha256):
        raise ValueError('Revision requires expected_import_sha256')
    staged_book, staged_images, target, diagrams, record, recovery_root = map(
        Path, (staged_book, staged_images, target, diagrams, record, recovery_root))
    if not record.is_file() or linked(record):
        raise ValueError('Revision requires an existing ordinary import record')
    recovery_root.mkdir(parents=True, exist_ok=True)
    lock = recovery_root/'revision.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError('Revision lock exists; inspect recovery journal before retrying') from None
    os.close(descriptor)
    recovery = None
    moves = []
    journal = {}
    committed = False
    release_lock = False
    try:
        if digest(record) != expected_import_sha256:
            raise ValueError('Existing import record changed; inspect before revising')
        old_bytes = record.read_bytes()
        old = json.loads(old_bytes.decode('utf-8-sig'))
        for key in ('schema', 'source_sha256', 'job_id', 'output', 'diagrams'):
            if old.get(key) != manifest.get(key):
                raise ValueError('Revision source, job or destination identity differs')
        if Path(old['output']).resolve() != target.resolve() or Path(old['diagrams']).resolve() != diagrams.resolve():
            raise ValueError('Revision destination does not match import record')
        before = validate_contents(target, diagrams, old)
        after = validate_contents(staged_book, staged_images, manifest)
        recovery = recovery_root/uuid.uuid4().hex[:12]
        recovery.mkdir()
        (recovery/'previous-import.json').write_bytes(old_bytes)
        journal = dict(status='PREPARED', previous_import_sha256=expected_import_sha256,
                       new_plan_id=manifest['plan_id'], before=before, after=after, moves=[])
        atomic_json(recovery/'journal.json', journal)
        sequence = [(target, recovery/'previous-book'), (diagrams, recovery/'previous-images'),
                    (staged_images, diagrams), (staged_book, target)]
        # Recheck after staging/journal work and before the first published move.
        if digest(record) != expected_import_sha256 or validate_contents(target, diagrams, old) != before:
            raise ValueError('Published textbook changed before revision')
        for source, destination in sequence:
            journal['pending_move'] = {'source': str(source), 'destination': str(destination)}
            atomic_json(recovery/'journal.json', journal)
            replace_path(source, destination)
            moves.append((source, destination))
            journal['moves'].append(journal.pop('pending_move'))
            atomic_json(recovery/'journal.json', journal)
        validate_contents(target, diagrams, manifest)
        if digest(record) != expected_import_sha256:
            raise ValueError('Import record changed during revision')
        atomic_json(record, manifest)
        committed = True
        journal['status'] = 'COMMITTED'
        journal['new_import_sha256'] = digest(record)
        atomic_json(recovery/'journal.json', journal)
        release_lock = True
        return dict(recovery=str(recovery), previous_import_sha256=expected_import_sha256,
                    import_sha256=digest(record))
    except BaseException as error:
        if committed:
            raise RuntimeError(f'Revision content committed; recovery journal requires inspection: {recovery}') from error
        if recovery is not None and not committed:
            journal['error'] = repr(error)
            # Undo only this transaction's completed moves. Nothing is deleted.
            try:
                pending = journal.get('pending_move')
                if pending:
                    pair = (Path(pending['source']), Path(pending['destination']))
                    if pair not in moves and not pair[0].exists() and pair[1].exists():
                        moves.append(pair)
                for source, destination in reversed(moves):
                    if source.exists():
                        raise RuntimeError('Rollback source unexpectedly occupied; retain recovery lock')
                    replace_path(destination, source)
                journal['status'] = 'ROLLED_BACK'
                atomic_json(recovery/'journal.json', journal)
                release_lock = True
            except BaseException as rollback_error:
                journal.update(status='RECOVERY_REQUIRED', rollback_error=repr(rollback_error))
                atomic_json(recovery/'journal.json', journal)
                raise RuntimeError(f'Revision recovery required: {recovery}') from rollback_error
        raise
    finally:
        # Keep the lock after incomplete rollback or interruption of journal commit.
        if recovery is None or release_lock:
            lock.unlink(missing_ok=True)
