"""Opt-in, bounded token-period guard. A stop is an anomaly, never an OCR result.

Only prolonged exact prose cycles are eligible. Numeric and mathematical cycles
are deliberately left to the original token budget and downstream review.
"""
from __future__ import annotations

import hashlib
import json
import re

VERSION = 'qwen-prose-period-guard-v1'
DEFAULTS = dict(min_generated_tokens=128, min_repeats=12, min_period=6,
                max_period=96, check_stride=8, min_unique_tokens=6)


def guard_config(payload, environ):
    requested = payload.get('repetition_guard', {})
    if not isinstance(requested, dict) or set(requested) - {'enabled', *DEFAULTS}:
        raise ValueError('REPETITION_GUARD_CONFIG_INVALID')
    enabled = requested.get('enabled', environ.get('BEMARKDOWN_QWEN_REPEAT_GUARD') == '1')
    if type(enabled) is not bool:
        raise ValueError('REPETITION_GUARD_ENABLED_INVALID')
    config = dict(DEFAULTS, **{k: v for k, v in requested.items() if k != 'enabled'})
    if any(type(value) is not int or value < 1 for value in config.values()):
        raise ValueError('REPETITION_GUARD_LIMIT_INVALID')
    # Keep this an independently switched conservative detector with bounded work.
    if (config['min_generated_tokens'] < 128 or not 12 <= config['min_repeats'] <= 64 or
            config['min_period'] < 6 or config['max_period'] > 96 or
            config['min_period'] > config['max_period'] or
            config['check_stride'] < 8 or config['min_unique_tokens'] < 6):
        raise ValueError('REPETITION_GUARD_UNSAFE_LIMIT')
    return dict(version=VERSION, enabled=enabled, **config)


def _is_prose(text):
    # Exclude common math syntax and digits even when variable names are words.
    if re.search(r'[0-9\\$^_{}=+*/<>|]', text):
        return False
    letters = [c.lower() for c in text if c.isalpha()]
    words = re.findall(r'[^\W\d_]+', text, flags=re.UNICODE)
    cjk = sum('\u3400' <= c <= '\u9fff' for c in text)
    return (len(letters) >= 24 and len(set(letters)) >= 10 and
            (len(words) >= 6 or cjk >= 24))


class PeriodicRepetitionGuard:
    """Transformers-compatible single-sequence stopping callable; no torch import."""

    def __init__(self, prompt_tokens, tokenizer, config):
        self.prompt_tokens, self.tokenizer, self.config = prompt_tokens, tokenizer, dict(config)
        self.event = None
        self._last_checked = 0

    def observe(self, tokens, *, generated_token_count=None):
        """CPU replay entry point, taking generated tokens only."""
        if self.event is not None:
            return True
        cfg = self.config
        count = len(tokens) if generated_token_count is None else generated_token_count
        if not cfg['enabled'] or count < cfg['min_generated_tokens']:
            return False
        for period in range(cfg['min_period'], min(cfg['max_period'], len(tokens) // cfg['min_repeats']) + 1):
            span = period * cfg['min_repeats']
            cycle = tokens[-period:]
            if len(set(cycle)) < cfg['min_unique_tokens'] or tokens[-span:] != cycle * cfg['min_repeats']:
                continue
            text = self.tokenizer.decode(cycle, skip_special_tokens=True)
            if not _is_prose(text):
                continue
            self.event = dict(status='UNRESOLVED_REPETITION_ANOMALY', version=VERSION,
                              generated_token_count=count, period_tokens=period,
                              observed_repeats=cfg['min_repeats'], period_text=text,
                              period_token_ids=list(cycle),
                              repeated_suffix_sha256=hashlib.sha256(json.dumps(tokens[-span:]).encode()).hexdigest(),
                              completed_transcription=False)
            return True
        return False

    def __call__(self, input_ids, scores=None, **kwargs):
        if self.event is not None:
            return True
        count = input_ids.shape[-1] - self.prompt_tokens
        cfg = self.config
        if (not cfg['enabled'] or count < cfg['min_generated_tokens'] or
                count - self._last_checked < cfg['check_stride']):
            return False
        self._last_checked = count
        window = min(count, cfg['max_period'] * cfg['min_repeats'])
        return self.observe(input_ids[0, -window:].tolist(), generated_token_count=count)


def anomaly_result(guard, tokens):
    """Keep the complete generated prefix as evidence, exclude it from output."""
    if guard.event is None:
        return {}
    return dict(normalized_output='', output_contract_status='REPETITION_ANOMALY',
                runtime_failure=True, error='UNRESOLVED_REPETITION_ANOMALY',
                repetition_anomaly=guard.event, raw_token_ids=list(tokens),
                completed_transcription=False)
