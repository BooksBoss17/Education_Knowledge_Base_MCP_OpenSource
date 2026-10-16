"""Bound source adjudication; repeated text and local voting are not evidence of correctness."""
from __future__ import annotations
import copy
import hashlib
import json
import re

SOURCE_REVIEW_VERSION = 'asset-source-review-v2'
RECEIPT_SCHEMA = 'bound-source-review-receipt-v1'


def observation_hash(observation):
    return hashlib.sha256(json.dumps(observation, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def validate_observation(task, observation):
    if task.task_type != 'VISUAL_TEXT_EXTRACT':
        raise ValueError('SOURCE_REVIEW_TASK_KIND')
    if not isinstance(observation, dict) or set(observation) != {
        'input_sha256', 'readability', 'coverage', 'visible_evidence', 'ambiguities'}:
        raise ValueError('SOURCE_OBSERVATION_FIELDS')
    if observation['input_sha256'] != task.input_sha256:
        raise ValueError('SOURCE_OBSERVATION_INPUT_CHANGED')
    if observation['readability'] not in {'CLEAR', 'NO_VISIBLE_TEXT', 'AMBIGUOUS', 'TRUNCATED'}:
        raise ValueError('SOURCE_OBSERVATION_READABILITY')
    if observation['coverage'] not in {'FULL_TARGET', 'PARTIAL_TARGET'}:
        raise ValueError('SOURCE_OBSERVATION_COVERAGE')
    evidence = observation['visible_evidence']
    ambiguities = observation['ambiguities']
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 2400:
        raise ValueError('SOURCE_OBSERVATION_EVIDENCE')
    if not isinstance(ambiguities, list) or len(ambiguities) > 16 or any(
        not isinstance(item, str) or not item.strip() or len(item) > 400 for item in ambiguities):
        raise ValueError('SOURCE_OBSERVATION_AMBIGUITIES')
    return copy.deepcopy(observation)


def reading_hash(value):
    fields = ('task_id', 'task_fingerprint', 'status', 'text', 'evidence_kind', 'uncertain',
              'text_presence', 'visual_content_kind', 'source_observation')
    return observation_hash({key: value.get(key) for key in fields})


def bind_source_results(tasks, results, call):
    """Called only after actual image-read completion and raw response validation."""
    for task in tasks:
        if task.candidate_version != SOURCE_REVIEW_VERSION:
            continue
        value = results[task.task_id]
        observation = validate_observation(task, value['source_observation'])
        actual_reading_hash = reading_hash(value)
        hashes = call.setdefault('source_result_hashes', {})
        if task.task_id in hashes and hashes[task.task_id] != actual_reading_hash:
            raise ValueError('SOURCE_REVIEW_CACHED_READING_CHANGED')
        hashes[task.task_id] = actual_reading_hash
        value['source_review_receipt'] = {
            'schema': RECEIPT_SCHEMA, 'request_key': call['request_key'],
            'task_id': task.task_id, 'task_fingerprint': task.fingerprint,
            'input_sha256': task.input_sha256, 'source_sha256': task.source_sha256,
            'observation_sha256': observation_hash(observation),
            'reading_sha256': actual_reading_hash,
            'sessions': list(call.get('sessions', [])), 'fresh_session': call.get('fresh_session') is True,
            'image_reads': list(call.get('image_reads', [])),
            'prompt_sha256': call['prompt_sha256'], 'result_sha256': call['result_sha256'],
        }
    return results


_VISIBLE_TEX_OPERATORS = frozenset(('sin cos tan cot sec csc sinh cosh tanh coth '
    'arcsin arccos arctan arccot arcsec arccsc log ln lg exp lim limsup liminf '
    'max min sup inf det dim gcd ker arg Pr mod bmod pmod').split())
_LITERAL_OR_MATH_SPANS = re.compile(r'```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\r\n]*`|\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]*\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]')
_HTML_TAGS = re.compile(r'''<(?P<closing>/)?(?P<name>table|thead|tbody|tfoot|tr|th|td|caption|colgroup|col|p|div|span|br|b|strong|em|i|u|sup|sub|style|script)(?=[\s/>])(?:"[^"]*"|'[^']*'|[^'">])*?>''', re.I)


def _without_paired_html_markup(text):
    """Remove proven markup without turning an unpaired x<p>y into x y.

    Work on the whole string so a math span inside a cell cannot detach an
    opening tag from its closing tag. Math/code contents are opaque to tag
    matching; math/code occurring inside an actual attribute stays non-visible.
    """
    protected = list(_LITERAL_OR_MATH_SPANS.finditer(text))
    stacks, containers, remove, voids = {}, [], [], []
    for tag in _HTML_TAGS.finditer(text):
        if any(p.start() <= tag.start() and tag.end() <= p.end() for p in protected):
            continue
        name = tag.group('name').lower()
        if tag.group('closing'):
            if not stacks.get(name): continue
            opening = stacks[name].pop()
            containers.append((opening.start(),tag.end()))
            if name in {'style','script'}:
                remove.append((opening.start(),tag.end()))
            else:
                remove.extend([opening.span(),tag.span()])
        elif name in {'br','col'} or tag.group().rstrip().endswith('/>'):
            voids.append(tag.span())
        else:
            stacks.setdefault(name,[]).append(tag)
    for a,b in voids:
        if any(x <= a and b <= y for x,y in containers): remove.append((a,b))
    chars = list(text)
    for a,b in remove:
        chars[a:b] = ' '*(b-a)
    return ''.join(chars)


def _visible_tokens(text, *, strip_markup=True):
    # Only an omission signal: this does not certify mathematical equivalence.
    # Serialization attributes and non-rendering commands are not source glyphs.
    from html import unescape
    text = str(text)
    if strip_markup: text = _without_paired_html_markup(text)
    tokens = set()
    def collect(part, literal=False, mathematical=False):
        if not literal and not mathematical:
            part = unescape(part)
        if not literal or not strip_markup:
            # Keep vertices scoped to their angle: an O inside POM must not
            # supply an independently printed O elsewhere in the expression.
            def angle_name(match):
                name = ''.join(next(g for g in match.groups() if g).split())
                tokens.add('@angle-name:' + name)
                tokens.update('@angle-vertex:' + letter for letter in name)
                return ' '
            scoped_part = re.sub(
                r'(?:\\angle\b|∠)\s*(?:\{\s*([A-Z]{2,4})\s*\}|([A-Z]{2,4})(?![A-Za-z])|([A-Z](?:\s+[A-Z]){1,3})(?![A-Za-z]|\s+[A-Z]))',
                angle_name, part)
            # Observation prose may quote an angle in backticks. Add its
            # scoped evidence while retaining literal tokens, so descriptions
            # of actual printed code still participate in omission checks.
            if not literal:
                part = scoped_part
        if not literal:
            part = re.sub(r'\\([A-Za-z]+)',
                lambda m: ' '+m.group(1)+' ' if m.group(1) in _VISIBLE_TEX_OPERATORS else ' ', part)
        tokens.update(re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?', part))
    cursor = 0
    for match in _LITERAL_OR_MATH_SPANS.finditer(str(text)):
        collect(str(text)[cursor:match.start()])
        code = match.group()
        if code.startswith(('$','\\(','\\[')):
            collect(code,mathematical=True)
            cursor=match.end()
            continue
        if code.startswith(('```','~~~')):
            code = code[3:-3]
            if '\n' in code: code = code.split('\n',1)[1]
        else: code = code[1:-1]
        collect(code,literal=True)
        cursor = match.end()
    collect(str(text)[cursor:])
    return tokens


def missing_observed_tokens(value, *, primary_text='', local_evidence=None):
    """Find tokens the reviewer itself saw and prior predictions also contain.

    This only requests independent adjudication; it never inserts model-voted text.
    A novel source correction can pass when its observation supports the new reading.
    """
    prior = _visible_tokens(primary_text)
    for candidate in (local_evidence or {}).get('candidates', {}).values():
        if not candidate or candidate.get('status') not in {'OK', 'PASS'}:
            continue
        text = str(candidate.get('text', ''))
        if re.search(r'<\s*img\b|!\[[^\]\n]*\]\s*\(|\[Unresolved image:', text, re.I):
            continue
        prior.update(_visible_tokens(text))
    seen = _visible_tokens(value.get('source_observation', {}).get('visible_evidence', ''), strip_markup=False)
    expected = seen & prior
    # The observation may describe just one angle vertex ("O is closed").
    # Carry that requirement in angle scope; retain any independent O already
    # present in expected so a missing standalone variable still blocks.
    expected.update('@angle-vertex:' + t for t in seen
                    if re.fullmatch(r'[A-Z]', t) and '@angle-vertex:' + t in prior)
    expected.update('@angle-name:' + t for t in seen
                    if re.fullmatch(r'[A-Z]{2,4}', t) and '@angle-name:' + t in prior)
    output = _visible_tokens(value.get('text', ''))
    # Adjacent vertex labels can serialize as OP instead of O P. Only split a
    # short uppercase run made entirely of separately corroborated labels;
    # arbitrary words, lowercase variables, numbers and case changes stay strict.
    labels = {t for t in expected if re.fullmatch(r'[A-Z]', t)}
    for token in tuple(output):
        if re.fullmatch(r'[A-Z]{2,4}', token) and set(token) <= labels:
            output.update(token)
    missing = expected - output
    return sorted({t.removeprefix('@angle-vertex:') if t.startswith('@angle-vertex:')
                   else '∠' + t.removeprefix('@angle-name:') if t.startswith('@angle-name:')
                   else t for t in missing})


def explanatory_label_expansion(value, *, primary_text='', local_evidence=None):
    """Flag prose added around diagram labels, without deleting possible text.

    This is a request for independent review, not a rule that Chinese text or
    parentheses cannot occur in a diagram. A source reading may still introduce
    missing words; this narrow case requires multiple newly annotated labels
    when the primary reading and all usable local readings were symbol-only.
    """
    if value.get('visual_content_kind') != 'IMAGE_WITH_TEXT':
        return False
    cjk = r'[\u4e00-\u9fff]'
    if not primary_text.strip() or re.search(cjk, primary_text):
        return False
    candidates = [str(c.get('text', '')).strip()
                  for c in (local_evidence or {}).get('candidates', {}).values()
                  if c and c.get('status') in {'OK', 'PASS'} and str(c.get('text', '')).strip()]
    if not candidates or any(re.search(cjk, text) for text in candidates):
        return False
    # Require the structural pattern "math label (Chinese annotation)" more
    # than once. Ordinary prose corrections and a single printed parenthesis
    # do not meet this guard; no individual Chinese word is blacklisted.
    annotations = re.findall(
        r'(?:\$[^$\n]+\$|\\\([^\n]*?\\\))\s*[（(]([^()（）\n]*[\u4e00-\u9fff][^()（）\n]*)[）)]',
        str(value.get('text', '')))
    return len(annotations) >= 2 and sum(len(re.findall(cjk, text)) for text in annotations) >= 6


def source_decision(task, value, *, primary_calls, source_calls, primary_text='', local_evidence=None):
    """A bound independent reading may confirm the proposal or correct every local candidate."""
    if value.get('status') != 'SUCCESS' or value.get('uncertain') or not value.get('source_verified'):
        return False, 'SOURCE_REVIEW_UNRESOLVED'
    try:
        observation = validate_observation(task, value.get('source_observation'))
    except ValueError as exc:
        return False, str(exc)
    if observation['coverage'] != 'FULL_TARGET' or observation['ambiguities'] or observation['readability'] not in {'CLEAR', 'NO_VISIBLE_TEXT'}:
        return False, 'SOURCE_REVIEW_VISIBILITY_OR_SCOPE_UNRESOLVED'
    receipt = value.get('source_review_receipt', {})
    expected = {'schema': RECEIPT_SCHEMA, 'task_id': task.task_id, 'task_fingerprint': task.fingerprint,
                'input_sha256': task.input_sha256, 'source_sha256': task.source_sha256,
                'observation_sha256': observation_hash(observation), 'reading_sha256': reading_hash(value)}
    if any(receipt.get(key) != expected_value for key, expected_value in expected.items()):
        return False, 'SOURCE_REVIEW_RECEIPT_UNBOUND'
    calls = [c for c in source_calls if c.get('status') == 'COMPLETE'
             and c.get('request_key') == receipt.get('request_key')]
    if len(calls) != 1:
        return False, 'SOURCE_REVIEW_CALL_UNBOUND'
    call = calls[0]
    if call.get('source_result_hashes', {}).get(task.task_id) != reading_hash(value):
        return False, 'SOURCE_REVIEW_READING_CHANGED'
    if call.get('status') != 'COMPLETE' or not call.get('fresh_session'):
        return False, 'SOURCE_REVIEW_SESSION_UNVERIFIED'
    for key in ('sessions', 'image_reads', 'prompt_sha256', 'result_sha256'):
        if receipt.get(key) != call.get(key):
            return False, 'SOURCE_REVIEW_EXECUTION_CHANGED'
    sessions = set(receipt['sessions'])
    primary_sessions = {s for c in primary_calls for s in c.get('sessions', [])}
    if len(sessions) != 1 or not primary_sessions or sessions & primary_sessions:
        return False, 'SOURCE_REVIEW_CONTEXT_NOT_INDEPENDENT'
    if task.task_id not in call.get('task_ids', []) or not any(
        name.startswith('input_' + task.input_sha256[:20]) for name in receipt['image_reads']):
        return False, 'SOURCE_REVIEW_TARGET_NOT_READ'
    # Same-backend source reading does not resolve an isolated disputed character reliably.
    # Those already have an approved independent-image review path, preserving C/g protection.
    from .pdf.adaptive_ocr import canonical
    if value.get('text_presence') == 'PRESENT' and len(re.findall(r'\w', canonical(value['text']))) <= 2:
        return False, 'SHORT_GLYPH_INDEPENDENT_REVIEW_REQUIRED'
    if explanatory_label_expansion(value, primary_text=primary_text, local_evidence=local_evidence):
        return False, 'SOURCE_REVIEW_EXPLANATORY_LABEL_EXPANSION_REQUIRES_INDEPENDENT_REVIEW'
    missing = missing_observed_tokens(value, primary_text=primary_text, local_evidence=local_evidence)
    if missing:
        return False, 'SOURCE_REVIEW_OBSERVED_TOKEN_OMISSION:' + ','.join(missing)
    return True, 'BOUND_INDEPENDENT_SOURCE_OBSERVATION_ACCEPTED'
