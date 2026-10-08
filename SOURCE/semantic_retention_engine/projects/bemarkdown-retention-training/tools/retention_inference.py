"""Label-blind native HF inference, grouped coverage and frozen complete predictions."""

import hashlib

import importlib.metadata

import json

import pathlib

import re

import time

from datetime import datetime, timezone

import hf_training_adapter as A

import inference_plan as I

from contracts import visible_input, digest, validate_schema, source_complete

P, R = A.P, A.R

def file_sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def runtime_fingerprint(adapter_path=None):
    # Inference lineage does not open targets, Gold, split-label hashes or training error lists.
    base = A.load_json(P / 'reports/base-download.json')
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'transformers', 'peft', 'bitsandbytes', 'pillow', 'tokenizers')}
    adapter = {}
    if adapter_path:
        path = pathlib.Path(adapter_path).resolve()
        assert path.is_relative_to(R / 'tmp/rt-train')
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            adapter[name] = file_sha(path / name)
    return digest({'base_revision': base['revision'], 'base_files': {r['file']: r['sha256'] for r in base['files']},
                   'adapter': adapter, 'versions': versions,
                   'processor_visual_tokens': [64, 1024], 'protocol': 'decision_v1', 'thinking': False,
                   'quantization': 'language_nf4_double_quant_visual_original_dtype',
                   'source': {p.name: file_sha(p) for p in (P / 'tools/hf_training_adapter.py', P / 'tools/readonly_safetensors.py', P / 'tools/retention_inference.py', P / 'tools/inference_plan.py', P / 'tools/contracts.py')},
                   'decode': {'sample': False, 'max_new_tokens': 32}})

def parse_verdict(text):
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError):
        return 'U', False
    if not isinstance(value, dict) or set(value) != {'verdict'} or value['verdict'] not in ('K', 'D', 'U'):
        return 'U', False
    return value['verdict'], True

def decision_probability(tokenizer, generated, scores, verdict):
    import torch
    token_ids = {label: tokenizer.encode(label, add_special_tokens=False) for label in ('K', 'D', 'U')}
    if any(len(ids) != 1 for ids in token_ids.values()):
        return None, 'LABEL_NOT_SINGLE_TOKEN'
    ids = generated.tolist()
    for index, token in enumerate(ids):
        prefix = tokenizer.decode(ids[:index], skip_special_tokens=True)
        if re.fullmatch(r'\s*\{\s*"verdict"\s*:\s*"', prefix):
            if token != token_ids[verdict][0] or index >= len(scores):
                return None, 'GENERATED_LABEL_BOUNDARY_MISMATCH'
            logits = scores[index][0].float()
            labels = torch.tensor([token_ids[x][0] for x in ('K', 'D', 'U')], device=logits.device)
            selected = torch.softmax(logits[labels], dim=0)
            full_mass = torch.softmax(logits, dim=0)[labels].sum()
            return {'p_k': float(selected[0]), 'p_d': float(selected[1]), 'p_u': float(selected[2]),
                    'full_vocab_label_mass': float(full_mass),
                    'scope': 'conditional_next_label_token_KDU_given_generated_JSON_prefix'}, None
    return None, 'GENERATED_JSON_LABEL_PREFIX_NOT_FOUND'

class NativeInferencer:
    def __init__(self, model, processor, model_fp, cache_root):
        self.model = model
        self.processor = processor
        self.model_fp = model_fp
        self.cache_root = pathlib.Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0

    def input_fp(self, row):
        return digest({'visible': visible_input(row, self.processor.policy), 'source_image_sha256': row['image']['sha256'],
                       'model': self.model_fp, 'processor_tokens': [64, 1024]})

    def measured_length(self, row):
        encoded = self.processor.encode(row, allow_long=True)
        n = encoded['input_ids'].shape[1] + 32
        del encoded
        return n

    def predict_group(self, row):
        import torch
        key = self.input_fp(row)
        cache = self.cache_root / (key + '.json')
        if cache.exists():
            saved = A.load_json(cache)
            assert saved['input_fp'] == key and saved['model_fp'] == self.model_fp
            self.cache_hits += 1
            return {**saved, 'inference_kind': 'cache_reuse', 'elapsed_ms': 0.}
        start = time.perf_counter()
        encoded = self.processor.encode(row)
        batch = A.on_device(encoded)
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(**batch, do_sample=False, max_new_tokens=32, use_cache=True,
                                         return_dict_in_generate=True, output_scores=True)
        generated = output.sequences[0, batch['input_ids'].shape[1]:]
        text = self.processor.processor.tokenizer.decode(generated, skip_special_tokens=True)
        verdict, valid = parse_verdict(text)
        probability, probability_error = decision_probability(self.processor.processor.tokenizer, generated, output.scores, verdict) if valid else (None, 'INVALID_JSON_PROTOCOL')
        torch.cuda.synchronize()
        response = {'input_fp': key, 'model_fp': self.model_fp, 'raw': verdict, 'valid_protocol': valid,
                    'raw_response': text, 'response_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'probability': probability, 'probability_error': probability_error,
                    'inference_kind': 'model_inference', 'elapsed_ms': (time.perf_counter() - start) * 1000}
        A.write_json(cache, response)
        self.calls += 1
        del encoded, batch, output, generated
        return response

    def predict_asset(self, row, threshold, source_inputs_sha256):
        started = time.perf_counter()
        policy_fp = digest({'d_threshold': threshold, 'aggregation': 'K>U>D', 'missing_source_D_to_U': True,
                            'unknown_D_confidence_to_U': True, 'all_reference_action_after_asset_aggregation': True})
        groups = []
        try:
            plan, requests = I.plan_groups(row, self.measured_length, self.processor.policy, source_inputs_sha256)
            for spec, request in zip(plan['groups'], requests):
                x = request['input']
                if request['over_budget']:
                    response = {'raw': 'U', 'probability': None, 'raw_response': '', 'response_sha256': hashlib.sha256(b'').hexdigest(),
                                'inference_kind': 'not_run', 'elapsed_ms': 0., 'valid_protocol': False,
                                'probability_error': 'SINGLE_USE_OVER_BUDGET'}
                else:
                    response = self.predict_group(x)
                probability = response.get('probability')
                confidence = probability['p_d'] if probability else None
                final = I.final_group(response['raw'], confidence, x, threshold)
                groups.append({**response, 'group_id': spec['group_id'], 'use_ids': spec['use_ids'],
                               'source_complete': source_complete(x), 'final': final})
            result = I.aggregate_asset(plan, groups, threshold)
            status = ('protocol_failure' if any(not g.get('valid_protocol') and g.get('inference_kind') != 'not_run' for g in groups)
                      else 'input_failure' if any(g.get('inference_kind') == 'not_run' for g in groups)
                      else 'model_response' if result['final'] != 'U' else 'policy_abstention')
            reason = 'ALL_GROUPS_EVALUATED_AND_ASSET_ACTION_AGGREGATED'
        except (ValueError, OSError) as exc:
            # Keep a member with explicit U; never silently drop it from the evaluation denominator.
            plan = None
            result = {'raw': 'U', 'final': 'U', 'all_references_action': 'preserve', 'coverage_complete': False}
            status = 'input_failure'
            reason = type(exc).__name__ + ':' + str(exc)
            # CUDA OOM is a phase failure, not thousands of synthetic U results.
            if 'out of memory' in str(exc).lower() or 'cuda error' in str(exc).lower():
                raise
        responses_sha = digest([g['response_sha256'] for g in groups])
        p_ds = [g['probability']['p_d'] for g in groups if g.get('probability')]
        prediction = {'schema_version': '1.0', 'sample_id': row['sample_id'],
                      'input_fingerprint': self.input_fp(row), 'model_fingerprint': self.model_fp,
                      'policy_fingerprint': policy_fp, 'raw_decision': result['raw'], 'final_decision': result['final'],
                      'status': status, 'inference_kind': ('not_run' if not groups or all(g['inference_kind'] == 'not_run' for g in groups)
                                                        else 'cache_replay' if all(g['inference_kind'] == 'cache_reuse' for g in groups) else 'cold_inference'),
                      'elapsed_ms': (time.perf_counter() - started) * 1000, 'raw_response_sha256': responses_sha,
                      'p_d': min(p_ds) if p_ds else None, 'reason_code': reason}
        validate_schema('prediction', prediction)
        return prediction, {'plan': plan, 'groups': groups, 'asset_action': result['all_references_action'], 'coverage_complete': result['coverage_complete']}

def freeze_predictions(inputs, predictions, model_fp, policy_fp, evaluation_id):
    """No labels: exact input-member set, then immutable output SHA."""
    inputs, predictions = pathlib.Path(inputs), pathlib.Path(predictions)
    expected = [r['sample_id'] for r in A.read_rows(inputs)]
    observed = []
    for row in A.read_rows(predictions):
        validate_schema('prediction', row)
        assert row['model_fingerprint'] == model_fp and row['policy_fingerprint'] == policy_fp
        observed.append(row['sample_id'])
    assert len(expected) == len(set(expected)) and len(observed) == len(set(observed))
    assert set(expected) == set(observed), 'INCOMPLETE_PREDICTIONS_CANNOT_FREEZE'
    manifest = {'schema_version': '1.0', 'evaluation_id': evaluation_id, 'expected_inputs_sha256': file_sha(inputs),
                'expected_members_sha256': digest(sorted(expected)), 'prediction_file_sha256': file_sha(predictions),
                'model_fingerprint': model_fp, 'policy_fingerprint': policy_fp, 'count': len(expected),
                'complete': True, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'score_after_freeze': True}
    validate_schema('prediction-freeze', manifest)
    output = predictions.with_suffix('.freeze.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest
