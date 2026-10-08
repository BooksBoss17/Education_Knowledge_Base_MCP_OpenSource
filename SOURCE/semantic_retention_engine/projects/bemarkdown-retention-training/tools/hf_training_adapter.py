"""Pinned HF/PEFT native adapter. Runtime invariants must pass before formal training."""

import hashlib

import io

import json

import pathlib

import sys

import inspect

P = pathlib.Path(__file__).resolve().parents[1]

R = P.parents[1]

sys.path.insert(0, str(P / 'tools'))

from contracts import visible_input, validate_input, admit_training

def load_json(p):
    return json.loads(pathlib.Path(p).read_text('utf-8-sig'))

def write_json(p, value):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')

def read_rows(p):
    with pathlib.Path(p).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)

def base_config():
    report = load_json(P / 'reports/base-download.json')
    plan = load_json(P / 'configs/train-plan.json')
    assert report['status'] == 'DOWNLOADED_VERIFIED'
    assert report['revision'] == plan['base_model']['revision']
    return pathlib.Path(report['local_path']), plan

class BudgetExceeded(ValueError):
    pass

class LazyProcessor:
    def __init__(self, resolver_path=None):
        from transformers import AutoProcessor
        base, plan = base_config()
        self.plan = plan
        self.processor = AutoProcessor.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
        factor = self.processor.image_processor.patch_size * self.processor.image_processor.merge_size
        self.processor.image_processor.size = {'shortest_edge': 64 * factor**2, 'longest_edge': 1024 * factor**2}
        self.policy = (P / 'prompts/model-decision-system.txt').read_text('utf-8')
        self.resolver = {}
        resolver_path = pathlib.Path(resolver_path).resolve() if resolver_path else P / 'data/splits/v2/resolver.jsonl'
        assert (resolver_path.is_relative_to(P / 'data/splits/v2') or
                resolver_path.is_relative_to(R / 'tmp/rt-train/runs')), 'EXTERNAL_RESOLVER_FORBIDDEN'
        for r in read_rows(resolver_path):
            self.resolver[r['image_ref']] = R / r['object_path']
        self.verified = {}

    def image(self, x):
        from PIL import Image
        path = self.resolver[x['image']['ref']].resolve()
        assert path.is_relative_to(R / 'tmp/rt-train/objects')
        before = path.stat()
        identity = (before.st_size, before.st_mtime_ns, before.st_ino)
        data = path.read_bytes()
        after = path.stat()
        assert identity == (after.st_size, after.st_mtime_ns, after.st_ino), 'TRAINING_COPY_CHANGED_DURING_READ'
        if self.verified.get(x['image']['sha256']) != identity:
            assert hashlib.sha256(data).hexdigest() == x['image']['sha256'], 'TRAINING_COPY_SHA_MISMATCH'
            self.verified[x['image']['sha256']] = identity
        image = Image.open(io.BytesIO(data))
        image.load()
        assert image.size == (x['image']['width'], x['image']['height']), 'DECLARED_IMAGE_SHAPE_MISMATCH'
        return image

    def encode(self, x, y=None, allow_long=False):
        import torch
        validate_input(x)
        if y is not None:
            admit_training(x, y, 'train')
        if max(x['image']['width'], x['image']['height']) / min(x['image']['width'], x['image']['height']) > 200:
            raise BudgetExceeded('NATIVE_PROCESSOR_UNSUPPORTED_ASPECT_RATIO_GT200')
        messages = visible_input(x, self.policy)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        image = self.image(x)
        try:
            prefix = self.processor(text=[prompt], images=[image], return_tensors='pt', truncation=False, padding=False)
            if y is None:
                result = dict(prefix)
                n = result['input_ids'].shape[1]
                if not allow_long and n + 32 > 3072:
                    raise BudgetExceeded('UNTRUNCATED_PROMPT_EXCEEDS_BUDGET')
                return result
            answer = json.dumps({'verdict': y['decision_target']}, separators=(',', ':'))
            full = self.processor.apply_chat_template(messages + [{'role': 'assistant', 'content': answer}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False)
            assert full.startswith(prompt), 'TEMPLATE_COMPLETION_PREFIX_MISMATCH'
            encoded = dict(self.processor(text=[full], images=[image], return_tensors='pt', truncation=False, padding=False))
            ids = encoded['input_ids']
            boundary = prefix['input_ids'].shape[1]
            assert torch.equal(ids[:, :boundary], prefix['input_ids']), 'TOKEN_COMPLETION_PREFIX_MISMATCH'
            n = ids.shape[1]
            if not allow_long and n > 3072:
                raise BudgetExceeded('UNTRUNCATED_TRAINING_SEQUENCE_EXCEEDS_BUDGET')
            labels = torch.full_like(ids, -100)
            eos = self.processor.tokenizer.eos_token_id
            ends = (ids[0, boundary:] == eos).nonzero(as_tuple=False).flatten()
            assert ends.numel(), 'ASSISTANT_EOS_MISSING'
            end = boundary + int(ends[0]) + 1
            labels[:, boundary:end] = ids[:, boundary:end]
            payload = self.processor.tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
            payload_ids = torch.tensor(payload['input_ids'], dtype=ids.dtype)
            assert torch.equal(ids[0, boundary:boundary + payload_ids.numel()], payload_ids), 'DECISION_PAYLOAD_TOKEN_BOUNDARY_MISMATCH'
            start_char = answer.index('"' + y['decision_target'] + '"') + 1
            decision_tokens = [i for i, (start, stop) in enumerate(payload['offset_mapping']) if start == start_char and stop == start_char + 1]
            assert len(decision_tokens) == 1, 'DECISION_COST_TOKEN_NOT_SEPARATE_FROM_JSON'
            assert (labels[:, :boundary] == -100).all()
            image_id = self.processor.image_token_id
            assert (labels[ids == image_id] == -100).all()
            assert labels[0, end - 1].item() == eos
            encoded['labels'] = labels
            encoded['_decision_loss_positions'] = decision_tokens
            encoded['_mask_audit'] = {'tokens': n, 'prompt_tokens': boundary,
                'supervised_tokens': end - boundary, 'eos_position': end - 1,
                'image_token_count': int((ids == image_id).sum()), 'prompt_image_padding_supervised': False}
            return encoded
        finally:
            image.close()

def on_device(encoded, device='cuda:0'):
    import torch
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items() if not k.startswith('_')}

def input_construction_fingerprint():
    return hashlib.sha256((inspect.getsource(LazyProcessor) + inspect.getsource(visible_input) +
        (P / 'prompts/model-decision-system.txt').read_text('utf-8')).encode()).hexdigest()
