from pathlib import Path
import argparse
import json
import os
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manager-job-id', required=True)
    parser.add_argument('--manager-request', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    helpers = runpy.run_path(str(Path(__file__).with_name('bootstrap.py')))
    context = helpers['trusted_context'](root)
    if not context or context['job_id'] != args.manager_job_id:
        raise RuntimeError('NATIVE_WORKER_NOT_OWNED')
    if Path(args.manager_request).resolve(strict=True) != (Path(context['home'])/'jobs'/args.manager_job_id/'request.json').resolve():
        raise ValueError('Unexpected managed request path')
    request = json.loads(Path(args.manager_request).read_text(encoding='utf-8'))
    if request['job_id'] != args.manager_job_id:
        raise ValueError('Manager request mismatch')
    payload = request['payload']
    if helpers['sha'](payload['source']) != payload['source_sha256']:
        raise ValueError('Native input changed after submission')
    for item in payload.get('additional_sources', []):
        if helpers['sha'](item['source']) != item['source_sha256']:
            raise ValueError('Native batch input changed after submission')
    command = [sys.executable, '-I', '-X', 'utf8', str(root/'TOOLS/bemarkdown_optimized/convert.py'),
               '--source', payload['source'], '--output-root', payload['output_root'],
               '--mcp-root', str(root), '--caller-pid', str(os.getpid()), '--json']
    for item in payload.get('additional_sources', []):
        command.extend(['--batch-source', item['source']])
    if payload.get('additional_sources'):
        command.extend(['--layout-batch-size', str(payload.get('layout_batch_size', 1))])
    if payload.get('repetition_guard'):
        command.append('--repetition-guard')
    if payload.get('image_screening') is False:
        command.append('--no-image-screening')
    if payload.get('screen_existing'):
        digest=runpy.run_path(str(root/'TOOLS/runtime_manager/package_binding.py'))['fingerprint'](payload['screen_existing'])
        if digest!=payload['existing_package_fingerprint']:raise ValueError('Existing OCR package changed after submission')
        command.extend(['--screen-existing',payload['screen_existing']])
    if payload['debug']:
        command.append('--debug')
    if payload.get('legacy_completion'):
        command.append('--legacy-completion')
    with (Path(request['job_folder'])/'native-result.json').open('w', encoding='utf-8') as out:
        result = subprocess.run(command, stdout=out)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
