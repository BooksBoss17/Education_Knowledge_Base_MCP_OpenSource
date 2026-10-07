"""Provision pinned OCR dependencies separately from the shared Paddle host."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

NAME = 'qwen-ocr-py312-torch214-tf517-v1'


def query(python):
    code = "import importlib.metadata as m,json,sys; print(json.dumps({'python':'.'.join(map(str,sys.version_info[:3])),'versions':{p:m.version(p) for p in ['torch','transformers','accelerate','pillow','huggingface-hub']}}))"
    return json.loads(subprocess.check_output([str(python), '-I', '-c', code], text=True))


def expected_versions(tool_root):
    lock = Path(tool_root) / 'runtime/requirements-qwen-ocr.lock'
    rows = [line.strip() for line in lock.read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]
    return dict(line.split('==', 1) for line in rows)


def verify(python, expected):
    identity = query(python)
    if not identity['python'].startswith('3.12.') or identity['versions'] != expected:
        raise RuntimeError('Isolated OCR environment differs from pinned dependencies; no files were replaced')
    return identity


def ensure(tool_root, home, source_python=None):
    expected = expected_versions(tool_root)
    home = Path(home).resolve()
    target = home / 'runtimes' / NAME
    python = target / 'Scripts/python.exe'
    if python.is_file():
        verify(python, expected)
        return python
    if target.exists():
        raise RuntimeError('Partial OCR environment exists; inspect it before retrying: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_python:
        version = subprocess.check_output([str(source_python), '-I', '-c', 'import sys;print(sys.version_info[:2])'], text=True).strip()
        if version != '(3, 12)':
            raise RuntimeError('--qwen-python must identify CPython 3.12')
        subprocess.run([str(source_python), '-m', 'venv', str(target)], check=True)
    else:
        uv = shutil.which('uv')
        if not uv:
            raise RuntimeError('Install the MIT-licensed uv CLI or provide --qwen-python <CPython-3.12-path>; Paddle was not changed')
        environment = os.environ.copy()
        environment.update(UV_PYTHON_INSTALL_DIR=str(home / 'python'), UV_CACHE_DIR=str(home / 'installer-cache'))
        subprocess.run([uv, 'venv', '--managed-python', '--python', '3.12.14', '--seed', str(target)], env=environment, check=True)
    pip = [str(python), '-m', 'pip', 'install']
    subprocess.run([*pip, 'torch==' + expected['torch'], '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
    subprocess.run([*pip, *[name + '==' + version for name, version in expected.items() if name != 'torch']], check=True)
    verify(python, expected)
    return python


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool-root', type=Path, default=Path(__file__).resolve().parents[1] / 'TOOLS/bemarkdown')
    parser.add_argument('--home', type=Path, default=Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'BeMarkdown')
    parser.add_argument('--qwen-python', type=Path)
    args = parser.parse_args()
    print(ensure(args.tool_root, args.home, args.qwen_python))
