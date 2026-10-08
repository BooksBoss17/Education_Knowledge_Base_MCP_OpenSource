"""Input-only paired GGUF regression with sequential native planning and serving."""
import argparse
import hashlib
import itertools
import json
import math
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
from types import SimpleNamespace

import retention_inference as N
import dev_calibration as D
from contracts import visible_input, digest
import probe_gguf_image_runtime as G

P, R = G.P, G.R


def decision_probability(choice, verdict):
    prefix = ''
    for item in (choice.get('logprobs') or {}).get('content') or []:
        token = item['token']
        if token == verdict and re.fullmatch(r'\s*\{\s*"verdict"\s*:\s*"', prefix):
            values = {p['token']: p['logprob'] for p in item.get('top_logprobs', [])
                      if p['token'] in ('K', 'D')}
            if set(values) != {'K', 'D'} or not all(math.isfinite(v) for v in values.values()):
                return None, 'MISSING_LABEL_IN_TOP_LOGPROBS'
            shift = max(values.values())
            pk, pd = (math.exp(values[k] - shift) for k in ('K', 'D'))
            return {'p_k': pk / (pk + pd), 'p_d': pd / (pk + pd),
                    'source': 'LLAMA_PRE_SAMPLING_TOP_LOGPROBS_NORMALIZED_K_D'}, None
        prefix += token
    return None, 'GENERATED_LABEL_BOUNDARY_MISMATCH'


class GGUFInferencer(N.NativeInferencer):
    def __init__(self, url, server, model_fp, policy, resolver, out):
        super().__init__(None, SimpleNamespace(policy=policy), model_fp, out / 'unused-response-cache')
        self.url, self.server, self.resolver = url, server, resolver
        self.out = out
        self.lengths, self.planner_error = {}, None

    def measured_length(self, row):
        if self.planner_error:
            raise ValueError(self.planner_error)
        return self.lengths[digest(visible_input(row, self.processor.policy))]

    def predict_group(self, row):
        assert self.server.poll() is None, 'OWNED_GGUF_SERVER_EXITED'
        start = time.perf_counter()
        path = pathlib.Path(self.resolver[row['image']['ref']])
        path = path if path.is_absolute() else R / path
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['image']['sha256'], 'ACTUAL_IMAGE_CHANGED'
        messages = G.materialize_messages(visible_input(row, self.processor.policy), data, row['image']['mime'])
        payload = {'model': 'local-retention', 'messages': messages, 'temperature': 0,
                   'max_tokens': 32, 'seed': 20261004, 'logprobs': True, 'top_logprobs': 20,
                   'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False}}
        # Transport/server failures are phase failures, never thousands of synthetic U records.
        try:
            response = G.request(self.url + '/v1/chat/completions', payload, timeout=180)
        except OSError as exc:
            message = exc.read(8192).decode('utf-8', errors='replace') if isinstance(exc, urllib.error.HTTPError) else str(exc)
            G.write(self.out / 'transport-error.json', {'input_fingerprint': self.input_fp(row),
                    'error_type': type(exc).__name__, 'error': message, 'server_alive': self.server.poll() is None})
            raise RuntimeError('GGUF_REQUEST_FAILED:' + type(exc).__name__) from exc
        choice = response['choices'][0]
        text = choice['message'].get('content') or ''
        verdict, valid = N.parse_verdict(text)
        probability, error = decision_probability(choice, verdict) if valid else (None, 'INVALID_JSON_PROTOCOL')
        self.calls += 1
        return {'input_fp': self.input_fp(row), 'model_fp': self.model_fp, 'raw': verdict,
                'valid_protocol': valid, 'raw_response': text,
                'response_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'probability': probability, 'probability_error': error,
                'inference_kind': 'model_inference', 'elapsed_ms': (time.perf_counter() - start) * 1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--evaluation-plan')
    args = parser.parse_args()
    out = pathlib.Path(args.run_dir).resolve()
    assert out.is_relative_to(R / 'tmp/rt-train/runs')
    candidate_path = pathlib.Path(args.evaluation_plan).resolve() if args.evaluation_plan else P / 'reports/final-candidate-and-holdout-plan.json'
    assert candidate_path.is_relative_to(P/'reports')
    candidate = G.read(candidate_path)
    inputs = P / candidate['inputs']
    expected_members = candidate['expected_members']
    assert isinstance(expected_members, int) and expected_members > 0
    assert N.file_sha(inputs) == candidate['inputs_sha256']
    threshold = candidate['threshold']
    assert D.policy_fingerprint(threshold) == candidate['policy_fingerprint']
    resolver_path = pathlib.Path(candidate.get('resolver_path', P / 'data/splits/v2/resolver.jsonl')).resolve()
    assert (resolver_path.is_relative_to(P / 'data/splits/v2') or
            resolver_path.is_relative_to(R / 'tmp/rt-train/runs')), 'EXTERNAL_RESOLVER_FORBIDDEN'
    if 'resolver_sha256' in candidate:
        assert N.file_sha(resolver_path) == candidate['resolver_sha256'], 'RESOLVER_CHANGED_AFTER_PLAN'
    export_path = P / 'reports/gguf-candidate-export.json'
    export = G.read(export_path)
    assert export['status'] == 'PAIRED_GGUF_CANDIDATE_EXPORTED_NOT_RUNTIME_OR_QUALITY_APPROVED'
    model = next(pathlib.Path(x['path']) for x in export['files'] if x['file'].endswith('Q4_K_M.gguf'))
    projector = next(pathlib.Path(x['path']) for x in export['files'] if x['file'] == 'mmproj-F16.gguf')
    for path in (model, projector):
        expected = next(x['sha256'] for x in export['files'] if x['path'] == str(path))
        assert N.file_sha(path) == expected, 'GGUF_ARTIFACT_CHANGED'
    for item in export['runtime_files']:
        assert N.file_sha(R / 'tmp/llama-retention' / item['file']) == item['sha256']
    started = time.monotonic()
    planning = out / 'planning'
    planning.mkdir()
    with (out / 'planning.log').open('wb') as log:
        command = [sys.executable, str(P / 'tools/prepare_gguf_input_lengths.py'),
                   '--run-dir', str(planning), '--inputs', str(inputs)]
        command += ['--resolver', str(resolver_path)]
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    assert result.returncode == 0, 'NATIVE_INPUT_PLANNING_FAILED'
    prepared = G.read(planning / 'plan-result.json')
    assert prepared['count'] == expected_members and prepared['inputs_sha256'] == candidate['inputs_sha256']
    assert N.file_sha(planning / 'lengths.jsonl') == prepared['lengths_sha256']
    policy = (P / 'prompts/model-decision-system.txt').read_text(encoding='utf-8')
    model_fp = digest({'gguf_files': [(x['file'], x['sha256']) for x in export['files']],
                       'runtime': export['runtime_files'], 'converter_commit': export['converter_commit'],
                       'planner_sha256': prepared['lengths_sha256'], 'planner_input_construction': prepared['input_construction_fingerprint'],
                       'resolver_sha256': N.file_sha(resolver_path),
                       'input_transform': candidate.get('input_transform'),
                       'normalized_server_arguments': G.server_command('<language>', '<projector>', 0)[1:],
                       'request': {'max_tokens': 32, 'temperature': 0, 'seed': 20261004, 'top_logprobs': 20,
                                   'cache_prompt': False, 'enable_thinking': False},
                       'sources': {name: N.file_sha(P / 'tools' / name) for name in
                                   ('run_gguf_partition_predictions.py', 'probe_gguf_image_runtime.py',
                                    'prepare_gguf_input_lengths.py', 'contracts.py', 'inference_plan.py', 'retention_inference.py')}})
    G.write(out / 'gguf-evaluation-plan.json', {'status': 'FROZEN_BEFORE_GGUF_PREDICTIONS', 'count': expected_members,
             'model_fingerprint': model_fp, 'policy_fingerprint': candidate['policy_fingerprint'],
             'inputs_sha256': candidate['inputs_sha256'], 'threshold': threshold,
             'targets_opened': False, 'response_cache_empty': True, 'prompt_cache_enabled': False,
             'planner_model_loaded': False, 'scope': 'isolated classifier regression; no full OCR/PDF pipeline'})
    resolver = {r['image_ref']: r['object_path'] for r in N.A.read_rows(resolver_path)}
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    url = 'http://127.0.0.1:' + str(port)
    server = None
    count = 0
    predictions = out / 'predictions.jsonl'
    with (out / 'server.log').open('wb') as log:
        try:
            server = subprocess.Popen(G.server_command(model, projector, port), stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
            load_start = time.monotonic()
            G.await_ready(server, url)
            load_seconds = time.monotonic() - load_start
            ids = {label: G.request(url + '/tokenize', {'content': label, 'add_special': False})['tokens']
                   for label in ('K', 'D')}
            assert all(len(values) == 1 for values in ids.values()), 'LABEL_NOT_SINGLE_TOKEN'
            infer = GGUFInferencer(url, server, model_fp, policy, resolver, out)
            with predictions.open('x', encoding='utf-8', buffering=1) as stream, \
                    (out / 'diagnostics.jsonl').open('x', encoding='utf-8', buffering=1) as details:
                paired = itertools.zip_longest(N.A.read_rows(inputs), N.A.read_rows(planning / 'lengths.jsonl'))
                for x, lengths in paired:
                    assert x is not None and lengths is not None and x['sample_id'] == lengths['sample_id']
                    infer.lengths, infer.planner_error = lengths['lengths'], lengths['error']
                    row, diagnostic = infer.predict_asset(x, threshold, candidate['inputs_sha256'])
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    details.write(json.dumps({'sample_id': x['sample_id'], **diagnostic}, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 32 == 0:
                        G.write(out / 'progress.json', {'completed': count, 'expected': expected_members, 'calls': infer.calls,
                                                      'elapsed_seconds': time.monotonic() - started})
            assert count == expected_members
            N.freeze_predictions(inputs, predictions, model_fp, candidate['policy_fingerprint'],
                                 candidate.get('freeze_tag', 'B200-Q4-paired-complete-holdout'))
        finally:
            if server is not None:
                if server.poll() is None:
                    server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
            G.write(out / 'server-unload.json', {'owned_server_exited': server is None or server.poll() is not None})
    result = {'status': 'COMPLETE_GGUF_PREDICTIONS_FROZEN_NOT_SCORED', 'count': count,
              'predictions': str(predictions), 'inputs': str(inputs), 'run_dir': str(out),
              'model_fingerprint': model_fp, 'policy_fingerprint': candidate['policy_fingerprint'],
              'threshold': threshold, 'calls': infer.calls, 'cache_hits': 0,
              'preprocessing_seconds': prepared['elapsed_seconds'], 'load_and_health_seconds': load_seconds,
              'total_worker_seconds': time.monotonic() - started, 'targets_opened': False,
              'response_cache_empty': True, 'prompt_cache_enabled': False,
              'owned_server_exited': True, 'whole_bemarkdown_pipeline_tested': False}
    G.write(out / 'prediction-result.json', result)
    G.write(P / 'reports' / candidate.get('result_pointer','gguf-holdout-predictions.json'), result)
    print(json.dumps({'status': result['status'], 'count': count}))


if __name__ == '__main__':
    main()
