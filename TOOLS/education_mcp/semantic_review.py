"""Deterministic paragraph evidence for agent-authored semantic OCR repairs.

This module selects evidence and validates provenance; it does not infer words.
"""
import hashlib
import json
import re


def paragraphs(content):
    result = []
    for match in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)', content):
        text = match.group().rstrip()
        # Headings, images, tables and metadata are not neighbouring prose.
        if text.startswith(('#', '![', '<', '|', '```', '$$', '---')):
            continue
        result.append(dict(start=match.start(), end=match.start() + len(text), text=text))
    return result


def character_count(text):
    """Count non-whitespace Unicode characters, including punctuation."""
    return sum(not char.isspace() for char in text)


def context(content, handoff, task_id, original=None):
    units = paragraphs(content)
    candidates = handoff.get('text_candidates', handoff.get('items', []))
    # Accept only the handoff's actual review candidates, never arbitrary text.
    matches = [item for item in candidates if item.get('task_id') == task_id]
    if len(matches) != 1:
        raise ValueError('Unknown or ambiguous text candidate task_id')

    def locate(item):
        indices = set()
        for span in item.get('markdown_contexts', []):
            old = span.get('original_text', '')
            if not old and original is not None:
                offset = span.get('character_offset')
                length = span.get('character_length')
                if type(offset) is int and type(length) is int and offset >= 0 and length > 0:
                    old = original[offset:offset + length]
            old = old.strip()
            if not old or content.count(old) != 1:
                continue
            start = content.index(old)
            end = start + len(old.rstrip())
            indices.update(i for i, p in enumerate(units)
                           if p['start'] <= start and end <= p['end'])
        return indices

    target_indices = locate(matches[0])
    if len(target_indices) != 1:
        raise ValueError('Candidate cannot map uniquely to one current prose paragraph; use source review')
    target = next(iter(target_indices))
    flagged = set().union(*(locate(item) for item in candidates))
    # Normal case: 3 preceding + 1 following. At the beginning: 0+4, 1+3, 2+2.
    before = min(3, target)
    selected = list(range(target - before, target))
    selected += list(range(target + 1, min(len(units), target + 1 + 4 - before)))
    # At the end, use more preceding paragraphs when available.
    left = target - before - 1
    while len(selected) < min(4, len(units) - 1) and left >= 0:
        selected.insert(0, left)
        left -= 1
    # Each flagged neighbour requires one extra clean paragraph. Flagged extras
    # remain visible but do not satisfy the clean-context requirement.
    needed = sum(i in flagged for i in selected)
    right = max([target, *selected]) + 1
    left = min([target, *selected]) - 1
    while needed and (right < len(units) or left >= 0):
        if right < len(units):
            index = right
            right += 1
        else:
            index = left
            left -= 1
        selected.append(index)
        if index not in flagged:
            needed -= 1
    payload = dict(task_id=task_id,
                   base_sha256=hashlib.sha256(content.encode('utf-8')).hexdigest(),
                   target=dict(paragraph_index=target, **units[target]),
                   neighbours=[dict(paragraph_index=i, flagged=i in flagged, **units[i])
                               for i in sorted(selected)],
                   clean_context_shortfall=needed,
                   count_rule='Non-whitespace Unicode characters, including punctuation',
                   max_character_delta=1,
                   evidence_kind='context_semantic_inference_not_source_verification')
    payload['context_sha256'] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    return payload


def validate_replacement(content, row, handoff, original=None):
    required = {'old', 'new', 'source_evidence', 'method', 'task_id', 'context_sha256'}
    if set(row) != required or not all(isinstance(v, str) for v in row.values()):
        raise ValueError('Invalid semantic replacement record')
    evidence = context(content, handoff, row['task_id'], original)
    if evidence['context_sha256'] != row['context_sha256']:
        raise ValueError('Semantic context changed; request fresh context')
    if row['old'] != evidence['target']['text']:
        raise ValueError('Semantic replacement must cover exactly the target prose paragraph')
    if abs(character_count(row['new']) - character_count(row['old'])) > 1:
        raise ValueError('Semantic repair exceeds plus/minus one character')
    if not row['new'].strip() or not row['source_evidence'].strip():
        raise ValueError('Semantic repair requires text and a contextual rationale')
    if len(paragraphs(row['new'])) != 1 or '\n\n' in row['new']:
        raise ValueError('Semantic repair must remain one prose paragraph')
    return evidence
