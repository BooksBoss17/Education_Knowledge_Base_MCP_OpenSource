"""One real input image through the matched owned GGUF/mmproj server."""

import argparse

import base64

import hashlib

import json

import pathlib

import socket

import subprocess

import time

import urllib.error

import urllib.request

from contracts import visible_input, validate_input, digest

import training_state as S

P = pathlib.Path(__file__).resolve().parents[1]

R = P.parents[1]

LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def write(path, value):
    pathlib.Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def request(url, payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    headers = {} if body is None else {'Content-Type': 'application/json'}
    with LOCAL_OPENER.open(urllib.request.Request(url, data=body, headers=headers), timeout=timeout) as response:
        return json.load(response)

def server_command(model, projector, port):
    return [str(R / 'tmp/llama-retention/llama-server.exe'), '-m', str(model), '--mmproj', str(projector),
            '--host', '127.0.0.1', '--port', str(port), '--alias', 'local-retention',
            '-ngl', '999', '-c', '4096', '-np', '1', '-b', '256', '-ub', '128',
            '-fa', 'on', '--image-min-tokens', '64', '--image-max-tokens', '1024',
            '--no-cache-prompt', '--cache-ram', '0', '--ctx-checkpoints', '0',
            '--jinja', '--chat-template-kwargs', '{"enable_thinking":false}']

def await_ready(server, url):
    deadline = time.monotonic() + 180
    while True:
        assert server.poll() is None, 'GGUF_SERVER_EXITED_BEFORE_READY'
        try:
            if request(url + '/health', timeout=2).get('status') == 'ok':
                return
        except (urllib.error.URLError, OSError):
            pass
        assert time.monotonic() < deadline, 'GGUF_SERVER_READINESS_TIMEOUT'
        time.sleep(.2)

def materialize_messages(messages, data, mime):
    encoded = base64.b64encode(data).decode('ascii')
    converted = []
    for message in messages:
        content = message['content']
        if isinstance(content, list):
            content = [({'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + encoded}}
                        if part['type'] == 'image' else part) for part in content]
        converted.append({'role': message['role'], 'content': content})
    return converted
