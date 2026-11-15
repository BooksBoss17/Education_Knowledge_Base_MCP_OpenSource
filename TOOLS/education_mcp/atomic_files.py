"""Atomic local writes with bounded retry for Windows sharing/lock failures."""
import json
import os
from pathlib import Path
import time
import uuid


def replace_path(source, destination):
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in {5, 32, 33} or attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('t-'+uuid.uuid4().hex[:12]+'.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    # Keep an unsuccessful write for diagnosis; never truncate the prior record.
    replace_path(temporary, path)
