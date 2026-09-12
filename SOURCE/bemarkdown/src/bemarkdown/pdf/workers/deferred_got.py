"""Import GOT dependencies in its isolated process before crops are ready.

No model or CUDA context is loaded until the parent submits the worker's
ordinary argument list. EOF cancels an unused process.
"""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKER_MODULE = 'bemarkdown.pdf.workers.got_ocr2'


class DeferredGOTProcess:
    def __init__(self, python: Path, output_root: Path, env: Mapping[str, str]):
        self.python = python
        self.output_root = output_root
        # Redirected file handles carry bytes; their parent-side encoding does
        # not configure Python's stdout/stderr codec in the child on Windows.
        self.env = {**env, 'PYTHONIOENCODING': 'utf-8'}
        self.process: subprocess.Popen | None = None
        self.used = False
        self.launch_error: str | None = None
        self.stdout_path = output_root / 'dependency-preload.stdout.txt'
        self.stderr_path = output_root / 'dependency-preload.stderr.txt'

    def start(self) -> None:
        if self.process is not None or self.launch_error is not None:
            return
        self.output_root.mkdir(parents=True, exist_ok=True)
        try:
            with self.stdout_path.open('w', encoding='utf-8') as stdout, self.stderr_path.open('w', encoding='utf-8') as stderr:
                self.process = subprocess.Popen(
                    [str(self.python), '-m', __name__], cwd=self.output_root,
                    env=self.env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                    text=True, encoding='utf-8', errors='replace',
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                )
        except OSError as exc:
            self.launch_error = f'{type(exc).__name__}:{exc}'

    def run(self, command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
        process = self.process
        if process is None or process.poll() is not None or list(command[1:3]) != ['-m', WORKER_MODULE]:
            return subprocess.run(command, **kwargs)
        self.used = True
        try:
            process.communicate(json.dumps(list(command[3:])) + '\n', timeout=kwargs.get('timeout'))
        except BaseException:
            process.kill()
            process.communicate()
            raise
        return subprocess.CompletedProcess(
            command, process.returncode,
            stdout=self.stdout_path.read_text(encoding='utf-8'),
            stderr=self.stderr_path.read_text(encoding='utf-8'),
        )

    def close(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.communicate('', timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()

    def metrics(self) -> dict[str, Any]:
        return {'schema': 'bemarkdown-deferred-got-process-v1',
                'launched': self.process is not None, 'used': self.used,
                'launch_error': self.launch_error,
                'process_returncode': self.process.returncode if self.process else None}


def read_command(stream) -> list[str] | None:
    message = stream.readline()
    if not message:
        return None
    arguments = json.loads(message)
    if not isinstance(arguments, list) or not arguments or any(not isinstance(arg, str) for arg in arguments):
        raise ValueError('DEFERRED_GOT_ARGUMENT_LIST_INVALID')
    return arguments


def prepare_imports() -> None:
    import torch  # noqa: F401 - load only in this isolated child process
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401


def main() -> int:
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    started = time.perf_counter()
    prepare_imports()
    print(json.dumps({'schema':'bemarkdown-deferred-import-ready-v1', 'seconds':time.perf_counter()-started}), flush=True)
    arguments = read_command(sys.stdin)
    if arguments is None:
        return 0
    sys.argv = [WORKER_MODULE, *arguments]
    runpy.run_module(WORKER_MODULE, run_name='__main__', alter_sys=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
