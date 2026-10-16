"""Source-bound transcription results and bounded prose edits.

No inference happens here. A transport must independently verify image-read
receipts before accepting a SOURCE_IMAGE_VERIFIED claim from its backend.
The same result contract is consumed by visual and semantic adapters.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

SCHEMA = 'bemarkdown-transcription-repair-v1'
TASK_TYPES = {'VISUAL_TRANSCRIBE', 'VISUAL_TEXT_EXTRACT', 'PROSE_SEMANTIC', 'BOUNDARY_REVIEW'}
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_IMAGE_REFERENCE = re.compile(r'<\s*img\b|!\[(?:\\.|[^\]])*\]\s*\(', re.I)
_MATH = re.compile(r'\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]*\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]')
_FACT = re.compile(r'[A-Za-z]+(?:[₀-₉⁰-⁹]+)?|[0-9]+(?:\.[0-9]+)?|[₀-₉⁰-⁹]+|[α-ωΑ-Ωμ∆Δ×÷±≤≥≠=<>∫∑√→←↑↓]')
_CONDITION = re.compile(r'至少|至多|不超过|不小于|不大于|大于|小于|相等|相反|同向|反向|向左|向右|向上|向下|增大|减小|增加|减少|上升|下降|只有|仅有|先后|分别|不|未|无|非|仅|只')
_DOMAIN = re.compile(r'千克|公斤|厘米|毫米|分钟|小时|牛顿|焦耳|瓦特|安培|伏特|欧姆|帕斯卡|摄氏|开尔文|米|秒|小车|滑块|导体棒|电流|电压|电阻|磁场|电场|加速度|速度|位移|质量|温度|压强|动量|能量|功率|频率|匀速|匀加速|匀减速|正比|反比|平行|垂直|顺时针|逆时针|静止|运动|恒定|水平|竖直|重力|摩擦|弹簧|电荷|电势|周期|振幅|定律|定则|必须|可能|一定|前|后|左|右|正|负')
_CHINESE_NUMBER = re.compile(r'第[一二三四五六七八九十百]+|[一二三四五六七八九十百千万壹贰叁肆伍陆柒捌玖拾佰仟]+(?=次|个|倍|米|秒|元|千克|牛顿)')
# Text-only repair has no new pixels to justify changing quantities or logic.
# Cover Chinese decimals/fractions, Unicode numerals/exponents and unfenced
# arithmetic rather than relying only on ASCII digits and a short unit list.
_CN_NUM = r'[零〇一二三四五六七八九十百千万亿两半俩仨壹贰叁肆伍陆柒捌玖拾佰仟萬億]'
_CHINESE_QUANTITY = re.compile(
    rf'第{_CN_NUM}+|{_CN_NUM}+(?:点{_CN_NUM}+)?(?:分之{_CN_NUM}+)?'
    r'(?=次|个|倍|米|秒|元|千克|公斤|牛顿|焦耳|瓦特|安培|伏特|欧姆|帕斯卡|度|分|小时|天|台|辆|组|只|条|枚|件|人|段|步|圈|层|种|项|份|块|根|对|套|支|滴|升|克|吨|岁|年|月|日|点钟)'
    rf'|{_CN_NUM}+分之[零〇一二三四五六七八九十百千万亿两半0-9.]+')
_LOGIC_SCOPE = re.compile(r'当且仅当|只要|除非|倘若|假如|如果|如若|若|则|那么|因此|所以|因为|虽然|尽管|但是|然而|而且|并且|或者|或|且|和|与|没有|没|禁止|不能|不得|是否|能否|可否|任意|所有|每个|存在|总是|必然|均|皆|各')
_UNFENCED_MATH = re.compile(r'[+\-−－–—*/\\=^_%‰°℃℉:：()（）\[\]{}]')
_ROLE_LABEL = re.compile(r'[甲乙丙丁]')


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _json_sha(value: Any) -> str:
    return text_sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


@dataclass(frozen=True)
class RepairTask:
    document_id: str
    page_id: str
    node_id: str
    asset_uid: str | None
    source_sha256: str
    input_sha256: str
    candidate: str
    task_type: str
    bbox: tuple[float, float, float, float]
    context: str = ''
    candidate_version: str = 'raw-v1'
    target_bbox_pixels: tuple[float, float, float, float] | None = None
    bbox_coordinate_space: str = 'source'

    def __post_init__(self):
        if self.task_type not in TASK_TYPES:
            raise ValueError('UNKNOWN_REPAIR_TASK_TYPE')
        if self.bbox_coordinate_space not in {'source','pixel','text_offset'}:
            raise ValueError('REPAIR_COORDINATE_SPACE_INVALID')
        if self.bbox_coordinate_space=='text_offset' and self.task_type!='PROSE_SEMANTIC':
            raise ValueError('VISUAL_SOURCE_COORDINATES_REQUIRED')
        if not all(isinstance(x, str) and x for x in (self.document_id, self.page_id, self.node_id, self.candidate_version)):
            raise ValueError('REPAIR_IDENTITY_REQUIRED')
        if not _HASH.fullmatch(self.source_sha256) or not _HASH.fullmatch(self.input_sha256):
            raise ValueError('REPAIR_SOURCE_HASH_REQUIRED')
        if not isinstance(self.candidate, str) or not isinstance(self.context, str):
            raise ValueError('REPAIR_TEXT_TYPE')
        if len(self.candidate) > 6000 or len(self.context) > 12000:
            raise ValueError('REPAIR_INPUT_WINDOW_EXCEEDED')
        if len(self.bbox) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in self.bbox):
            raise ValueError('REPAIR_BBOX_INVALID')
        x0, y0, x1, y1 = self.bbox
        if x0 >= x1 or y0 >= y1:
            raise ValueError('REPAIR_BBOX_EMPTY')
        object.__setattr__(self, 'bbox', tuple(float(x) for x in self.bbox))
        if self.target_bbox_pixels is not None:
            box=self.target_bbox_pixels
            if len(box)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in box) or box[0]<0 or box[1]<0 or box[0]>=box[2] or box[1]>=box[3]:
                raise ValueError('REPAIR_TARGET_PIXEL_BBOX_INVALID')
            object.__setattr__(self,'target_bbox_pixels',tuple(float(x) for x in box))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['bbox'] = list(self.bbox)
        if self.target_bbox_pixels is not None:value['target_bbox_pixels']=list(self.target_bbox_pixels)
        return value

    @property
    def task_id(self) -> str:
        return 'repair:' + _json_sha([self.document_id, self.page_id, self.node_id, self.asset_uid, self.task_type])[:24]

    @property
    def fingerprint(self) -> str:
        return _json_sha({'schema': SCHEMA, **self.to_dict()})


def _protected_spans(text):
    spans = [(m.start(), m.end(), m.group()) for pattern in
             (_MATH, _FACT, _CONDITION, _DOMAIN, _CHINESE_NUMBER, _CHINESE_QUANTITY, _LOGIC_SCOPE, _UNFENCED_MATH, _ROLE_LABEL)
             for m in pattern.finditer(text)]
    spans.extend((i, i + 1, char) for i, char in enumerate(text)
                 if unicodedata.category(char) in {'Nd', 'Nl', 'No', 'Sm', 'Lm', 'Sc'})
    return sorted(spans)


def _edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def apply_semantic_patches(task: RepairTask, patches: Sequence[Mapping[str, Any]]) -> str:
    if task.task_type != 'PROSE_SEMANTIC':
        raise ValueError('SEMANTIC_TASK_REQUIRED')
    if len(patches) > 2:
        raise ValueError('SEMANTIC_EDIT_BUDGET')
    candidate = task.candidate
    if '\n\n' in candidate or candidate.lstrip().startswith(('#', '|', '<', '```')):
        raise ValueError('SEMANTIC_PROSE_REQUIRED')
    ordered = sorted(patches, key=lambda p: p.get('start', -1))
    cursor = 0
    protected = _protected_spans(candidate)
    pieces = []
    edits = 0
    changed_characters = 0
    for patch in ordered:
        if set(patch) != {'start', 'end', 'old', 'new'}:
            raise ValueError('SEMANTIC_PATCH_FIELDS')
        start, end, old, new = (patch[x] for x in ('start', 'end', 'old', 'new'))
        if type(start) is not int or type(end) is not int or not (0 <= start <= end <= len(candidate)):
            raise ValueError('SEMANTIC_PATCH_RANGE')
        if start < cursor:
            raise ValueError('SEMANTIC_PATCH_OVERLAP')
        if not isinstance(old, str) or not isinstance(new, str) or candidate[start:end] != old:
            raise ValueError('SEMANTIC_PATCH_STALE')
        if max(len(old), len(new)) > 8 or '\n' in old + new:
            raise ValueError('SEMANTIC_SPAN_BUDGET')
        if _IMAGE_REFERENCE.search(new):
            raise ValueError('SEMANTIC_IMAGE_REFERENCE')
        if old == new:
            raise ValueError('SEMANTIC_NOOP_PATCH')
        for left, right, _ in protected:
            if (start < right and end > left) or (start == end and left <= start < right):
                raise ValueError('SEMANTIC_PROTECTED_SOURCE_SPAN')
        if _protected_spans(new):
            raise ValueError('SEMANTIC_PROTECTED_INSERTION')
        pieces.extend((candidate[cursor:start], new))
        cursor = end
        edits += _edit_distance(old, new)
        changed_characters += max(len(old), len(new))
    pieces.append(candidate[cursor:])
    output = ''.join(pieces)
    budget = min(12, max(2, int(len(candidate) * .15)))
    if edits > budget or changed_characters > budget:
        raise ValueError('SEMANTIC_EDIT_BUDGET')
    # This also catches protected strings formed by joining adjacent fragments.
    if [s for _, _, s in _protected_spans(output)] != [s for _, _, s in protected]:
        raise ValueError('SEMANTIC_PROTECTED_CONTENT_CHANGED')
    return output


def validate_result(task: RepairTask, result: Mapping[str, Any]) -> dict[str, Any]:
    from .source_review_evidence import SOURCE_REVIEW_VERSION, validate_observation
    fields = {'task_id', 'task_fingerprint', 'status', 'text', 'evidence_kind', 'patches', 'reason', 'uncertain'}
    if task.task_type == 'VISUAL_TEXT_EXTRACT':
        fields |= {'text_presence', 'visual_content_kind'}
    source_adjudication = task.candidate_version == SOURCE_REVIEW_VERSION
    if source_adjudication:
        fields.add('source_observation')
    if set(result) != fields:
        raise ValueError('REPAIR_RESULT_FIELDS')
    if result['task_id'] != task.task_id or result['task_fingerprint'] != task.fingerprint:
        raise ValueError('REPAIR_FINGERPRINT_CHANGED')
    if result['status'] not in {'SUCCESS', 'REVIEW'} or type(result['uncertain']) is not bool:
        raise ValueError('REPAIR_RESULT_STATUS')
    if not isinstance(result['text'], str) or not isinstance(result['reason'], str) or not result['reason'].strip():
        raise ValueError('REPAIR_RESULT_TEXT_OR_REASON')
    if not isinstance(result['patches'], list):
        raise ValueError('REPAIR_RESULT_PATCHES')
    expected = 'CONTEXT_SEMANTIC_INFERENCE' if task.task_type == 'PROSE_SEMANTIC' else 'SOURCE_IMAGE_VERIFIED'
    if result['evidence_kind'] != expected:
        raise ValueError('REPAIR_EVIDENCE_TYPE')
    value = copy.deepcopy(dict(result))
    value['source_verified'] = expected == 'SOURCE_IMAGE_VERIFIED' and result['status'] == 'SUCCESS' and not result['uncertain']
    value['schema'] = SCHEMA
    value['raw_transcription'] = task.candidate
    if source_adjudication:
        observation = validate_observation(task, result['source_observation'])
        value['source_observation'] = observation
        if (observation['coverage'] != 'FULL_TARGET' or observation['ambiguities'] or
                observation['readability'] not in {'CLEAR', 'NO_VISIBLE_TEXT'}):
            value.update(status='REVIEW', text=task.candidate, patches=[], uncertain=True,
                         source_verified=False, text_presence='UNCERTAIN', visual_content_kind='UNCERTAIN')
            return value
        if result['status'] == 'SUCCESS' and not result['uncertain']:
            absence = observation['readability'] == 'NO_VISIBLE_TEXT'
            if absence != (result.get('text_presence') == 'NONE'):
                raise ValueError('SOURCE_OBSERVATION_TEXT_PRESENCE_CONTRADICTION')
    if result['status'] == 'REVIEW' or result['uncertain']:
        value['status'] = 'REVIEW'
        value['text'] = task.candidate
        value['patches'] = []
        if task.task_type == 'VISUAL_TEXT_EXTRACT':
            value.update(text_presence='UNCERTAIN', visual_content_kind='UNCERTAIN')
        return value
    if task.task_type == 'VISUAL_TEXT_EXTRACT':
        if result['text_presence'] not in {'PRESENT', 'NONE'}:
            raise ValueError('VISUAL_TEXT_PRESENCE_INVALID')
        if result['visual_content_kind'] not in {'PURE_TEXT', 'PURE_FORMULA', 'IMAGE_WITH_TEXT', 'NO_TEXT'}:
            raise ValueError('VISUAL_CONTENT_KIND_INVALID')
        if result['text_presence'] == 'NONE':
            if result['text'].strip() or result['visual_content_kind'] != 'NO_TEXT' or result['patches']:
                raise ValueError('VISUAL_NO_TEXT_CONTRADICTION')
            return value  # No text is not a retention/exclusion decision.
        if result['visual_content_kind'] == 'NO_TEXT':
            raise ValueError('VISUAL_PRESENT_TEXT_CONTRADICTION')
    if not result['text'].strip() or len(result['text']) > 12000:
        raise ValueError('REPAIR_OUTPUT_EMPTY_OR_OVERLONG')
    if _IMAGE_REFERENCE.search(result['text']):
        raise ValueError('REPAIR_IMAGE_REFERENCE_UNBOUND')
    if task.task_type == 'PROSE_SEMANTIC':
        if apply_semantic_patches(task, result['patches']) != result['text']:
            raise ValueError('SEMANTIC_PATCH_OUTPUT_MISMATCH')
    elif result['patches']:
        raise ValueError('VISUAL_RESULT_PATCHES_NOT_ALLOWED')
    return value


def apply_node_result(document: Mapping[str, Any], task: RepairTask, result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get('schema') != SCHEMA:
        raise ValueError('REPAIR_RESULT_NOT_VALIDATED')
    source_application = copy.deepcopy(result.get('source_review_application'))
    if source_application:
        source_task=RepairTask(**source_application['source_task'])
        source_result=source_application['source_result']
        if (source_application.get('delivery_task_fingerprint') != task.fingerprint
                or source_task.source_sha256 != task.source_sha256
                or source_task.input_sha256 != task.input_sha256
                or source_task.node_id != task.node_id
                or source_result.get('task_fingerprint') != source_task.fingerprint
                or source_result.get('text') != result.get('text')):
            raise ValueError('SOURCE_REVIEW_APPLICATION_BINDING_CHANGED')
    fields = ('task_id', 'task_fingerprint', 'status', 'text', 'evidence_kind', 'patches', 'reason', 'uncertain')
    result = validate_result(task, {key: result[key] for key in fields})
    output = copy.deepcopy(dict(document))
    nodes = [node for node in output.get('blocks', []) if node.get('node_id') == task.node_id]
    if len(nodes) != 1:
        raise ValueError('REPAIR_NODE_IDENTITY_AMBIGUOUS')
    node = nodes[0]
    content = node.setdefault('content', {})
    kind = node.get('kind')
    field = 'latex' if kind == 'FORMULA' else 'text'
    if kind == 'TABLE':
        raw = content.get('table_serialization', {}).get('body', '')
    else:
        raw = content.get(field, '')
    if raw != task.candidate or result.get('task_fingerprint') != task.fingerprint:
        raise ValueError('REPAIR_NODE_STALE')
    if result.get('schema') != SCHEMA:
        raise ValueError('REPAIR_RESULT_NOT_VALIDATED')
    if result['status'] != 'SUCCESS':
        return output
    if kind == 'TABLE':
        content['table_serialization'] = {'format': 'MARKDOWN', 'body': result['text']}
    else:
        content[field] = result['text'].strip('$').strip() if kind == 'FORMULA' else result['text']
    node['review_state'] = 'NONE'
    provenance=node.setdefault('provenance', {})
    prior=copy.deepcopy(provenance.get('transcription_repair'))
    provenance['transcription_repair'] = {
        'schema': SCHEMA, 'task_id': task.task_id, 'task_fingerprint': task.fingerprint,
        'source_sha256': task.source_sha256, 'input_sha256': task.input_sha256,
        'raw_transcription': prior.get('raw_transcription',task.candidate) if prior else task.candidate,
        'input_candidate':task.candidate,'repaired_text': result['text'],
        'patches': copy.deepcopy(result['patches']), 'evidence_kind': result['evidence_kind'],
        'source_verified': result['source_verified'], 'reason': result['reason'],
    }
    if prior:provenance['transcription_repair']['previous_repair']=prior
    if source_application:
        provenance['transcription_repair']['source_review_application']=source_application
    return output


def apply_markdown_result(content: str, task: RepairTask, result: Mapping[str, Any]) -> str:
    """Use an exact occurrence offset, never a whole-document replacement."""
    if task.task_type!='PROSE_SEMANTIC' or task.bbox_coordinate_space!='text_offset':
        raise ValueError('MARKDOWN_TEXT_OFFSET_TASK_REQUIRED')
    start,end=int(task.bbox[0]),int(task.bbox[2])
    if start!=task.bbox[0] or end!=task.bbox[2] or content[start:end]!=task.candidate:
        raise ValueError('MARKDOWN_REPAIR_STALE')
    fields=('task_id','task_fingerprint','status','text','evidence_kind','patches','reason','uncertain')
    verified=validate_result(task,{k:result[k] for k in fields})
    if verified['status']!='SUCCESS':return content
    return content[:start]+verified['text']+content[end:]
