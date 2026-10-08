"""Original native processor planning, completed before the GGUF server loads."""
import argparse
import json
import pathlib
import time

import hf_training_adapter as A
import retention_inference as N
import inference_plan as I
from contracts import visible_input, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--resolver')
    args = parser.parse_args()
    out = pathlib.Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = pathlib.Path(args.inputs).resolve()
    assert (inputs.is_relative_to(A.P / 'data/splits/v2') or inputs.is_relative_to(A.R/'tmp/rt-train/runs')) and inputs.name.endswith('.inputs.jsonl')
    assert out.is_relative_to(A.R / 'tmp/rt-train/runs')
    processor = A.LazyProcessor(args.resolver)
    source_sha = N.file_sha(inputs)
    started = time.monotonic()
    count = 0
    with (out / 'lengths.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
        for x in A.read_rows(inputs):
            lengths = {}
            error = None
            def measured_length(group):
                encoded = processor.encode(group, allow_long=True)
                length = int(encoded['input_ids'].shape[1]) + 32
                del encoded
                lengths[digest(visible_input(group, processor.policy))] = length
                return length
            try:
                plan, _ = I.plan_groups(x, measured_length, processor.policy, source_sha)
                group_count = len(plan['groups'])
            except (ValueError, OSError) as exc:
                error = type(exc).__name__ + ':' + str(exc)
                group_count = None
            stream.write(json.dumps({'sample_id': x['sample_id'], 'lengths': lengths,
                                     'error': error, 'planned_groups': group_count}, ensure_ascii=False) + '\n')
            count += 1
            if count % 32 == 0:
                A.write_json(out / 'progress.json', {'completed': count, 'elapsed_seconds': time.monotonic() - started})
    result = {'status': 'COMPLETE_INPUT_ONLY_NATIVE_LENGTH_PLANS', 'count': count,
              'inputs_sha256': source_sha, 'lengths_sha256': N.file_sha(out / 'lengths.jsonl'),
              'input_construction_fingerprint': A.input_construction_fingerprint(),
              'resolver_sha256': N.file_sha(pathlib.Path(args.resolver)) if args.resolver else N.file_sha(A.P / 'data/splits/v2/resolver.jsonl'),
              'source_sha256': N.file_sha(pathlib.Path(__file__)),
              'elapsed_seconds': time.monotonic() - started, 'targets_opened': False, 'model_loaded': False}
    A.write_json(out / 'plan-result.json', result)
    print(json.dumps({'status': result['status'], 'count': count}))


if __name__ == '__main__':
    main()
