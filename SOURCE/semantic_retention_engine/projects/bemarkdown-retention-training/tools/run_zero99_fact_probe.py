"""Source-bound observation and typed-fact probe. Never opens gold/diagnostic labels."""

import json

import re

def parse(response):
    try:
        choice = response['choices'][0]
        if choice.get('finish_reason') != 'stop':
            return None
        content = choice['message']['content'].strip()
        fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', content, flags=re.DOTALL)
        if fenced:
            content = fenced.group(1)
        value = json.loads(content)
        return value if isinstance(value, dict) else None
    except (KeyError, ValueError, TypeError, IndexError):
        return None

def observation_valid(value):
    return (isinstance(value, dict) and set(value) == {'kind', 'subject', 'visible_text', 'visual_information', 'legibility'}
            and all(isinstance(v, str) for v in value.values())
            and value['kind'] in {'TEXT','TABLE','MATH','GRAPH','DIAGRAM','PHOTO','MARK','PATTERN','UNKNOWN'}
            and value['legibility'] in {'clear','partial','unreadable'})
