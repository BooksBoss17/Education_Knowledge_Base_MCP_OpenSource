"""Connection-local visual readiness check; no caller model identity is inferred."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import secrets
import time


class VisionGate:
    challenge_seconds = 300
    idle_seconds = 1800

    def __init__(self):
        self.pending = None
        self.verified_until = 0.0

    def status(self):
        return dict(visual_model_required=True, verified=time.monotonic() < self.verified_until,
                    scope='Current MCP connection; repeat after model changes or 30 minutes idle.',
                    note='Pixel-reading checks verify image access, not a trusted model identity.')

    def call(self, action='challenge', challenge_id=None, answer=None):
        if action == 'status':
            return self._text(self.status())
        if action == 'challenge':
            from PIL import Image, ImageDraw, ImageFont
            self.verified_until = 0.0
            code = ''.join(secrets.choice('23456789ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(6))
            identifier = secrets.token_hex(16)
            self.pending = (identifier, code, time.monotonic() + self.challenge_seconds)
            canvas = Image.new('RGB', (540, 120), 'white')
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype(str(Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/arial.ttf'), 60)
            except OSError:
                font = ImageFont.load_default(size=60)
            for index, character in enumerate(code):
                x = 12 + 88 * index
                draw.rounded_rectangle((x, 10, x+76, 109), radius=6, outline='#747474', width=2)
                draw.text((x+38, 59), character, font=font, anchor='mm', fill='#14213d')
            buffer = io.BytesIO()
            canvas.save(buffer, format='PNG')
            result = dict(challenge_id=identifier, expires_in_seconds=self.challenge_seconds,
                          instruction='Visually read the six symbols from left to right. Call bemarkdown_vision action=verify with this challenge_id and the symbols as answer. If the image is unavailable, use an image-capable model; do not guess or read server internals.')
            return dict(content=[dict(type='text', text=json.dumps(result)),
                                 dict(type='image', mimeType='image/png', data=base64.b64encode(buffer.getvalue()).decode('ascii'))])
        if action != 'verify':
            raise ValueError('Use challenge, verify, or status')
        pending = self.pending
        self.pending = None
        self.verified_until = 0.0
        if pending is None or challenge_id != pending[0] or time.monotonic() >= pending[2]:
            raise ValueError('Visual challenge missing or expired; request a fresh challenge')
        if not isinstance(answer, str) or ''.join(answer.split()).upper() != pending[1]:
            raise ValueError('Visual check failed; use an image-capable model and request a fresh challenge')
        self.verified_until = time.monotonic() + self.idle_seconds
        return self._text(self.status())

    def require(self):
        if not self.status()['verified']:
            raise RuntimeError('VISION_MODEL_REQUIRED: only image-capable models are supported. First call bemarkdown_vision action=challenge, view its image, then action=verify. Existing conversion jobs remain intact; do not resubmit them.')
        self.verified_until = time.monotonic() + self.idle_seconds

    @staticmethod
    def _text(value):
        return dict(content=[dict(type='text', text=json.dumps(value))])
