"""Frozen native/completion worker; receives source paths, never evaluation IDs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--native-only', action='store_true')
    args = parser.parse_args()
    cfg = read(args.config)
    root = Path(cfg['run_root'])
    tool = Path(cfg['tool_root'])
    core = tool / 'core'
    for relative, expected in cfg['payload_files'].items():
        path = (tool / relative).resolve()
        if not path.is_relative_to(tool.resolve()) or sha(path) != expected:
            raise ValueError('OPTIMIZED_PAYLOAD_CHANGED:' + relative)
    source = Path(cfg['input'])
    if sha(source) != cfg['input_sha256'] or sha(cfg['flash_patch']) != cfg['flash_patch_sha256']:
        raise ValueError('OPTIMIZED_SOURCE_OR_PRIVATE_PROFILE_CHANGED')
    for name in ('source_review_index','native_visible_source_review_index'):
        if cfg.get(name) and sha(cfg[name]) != cfg[name + '_sha256']:
            raise ValueError('OPTIMIZED_APPROVED_REVIEW_INDEX_CHANGED:' + name)
    sys.path.insert(0, str(tool / 'route'))
    sys.path.insert(0, str(core))
    os.environ.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', BEMARKDOWN_RESOURCE_PROFILE='6gb',
        BEMARKDOWN_DISABLE_QWEN_ALLOCATOR_WARMUP='1', BEMARKDOWN_QWEN_READONLY_WEIGHT_MAPS='0',
        HF_DEACTIVATE_ASYNC_LOAD='1', BEMARKDOWN_FORMULA_VISUAL_BATCH_LIMIT='1',
        BEMARKDOWN_NATIVE_PENDING_REFERENCE_SALVAGE='0')
    temp = root / 'temp'
    temp.mkdir(exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temp)
    started = time.monotonic()
    if args.native_only:
        from isolated_native_backends import activate
        activate(table_staged=True)
        from portable_native_conversion import convert_native
        convert_native(source, root / 'native', core,
                       mcp_root=cfg['mcp_root'], flash_patch=cfg['flash_patch'])
        return 0
    # The native child and its model actors must exit before completion imports.
    with (root / 'native.stdout.log').open('wb') as stdout, (root / 'native.stderr.log').open('wb') as stderr:
        result = subprocess.run([sys.executable, '-I', '-X', 'utf8', str(Path(__file__).resolve()),
            '--config', str(Path(args.config).resolve()), '--native-only'], stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError('OPTIMIZED_NATIVE_FAILED:exit=' + str(result.returncode))
    native_result = read(root / 'native/conversion-pointer.json')
    native = Path(native_result['package_path']).resolve()
    if not native.is_relative_to((root / 'native').resolve()):
        raise ValueError('OPTIMIZED_NATIVE_PACKAGE_ESCAPED')
    package = Path(cfg['output_root']) / 'final' / native.name
    package.parent.mkdir(exist_ok=True)
    shutil.copytree(native, package)
    from e2e_completion import CompletionAdapter
    from bemarkdown.asset_local_evidence import BoundedAssetLocalEvidence
    from bemarkdown.source_review_cache import ApprovedSourceReviewCache
    from bemarkdown.production import PackageValidator, _package_manifest
    formal = Path(cfg['mcp_root'])
    factory = lambda work: BoundedAssetLocalEvidence(work, models_root=formal / 'MODELS',
        config_path=formal / 'TOOLS/bemarkdown/bemarkdown.toml', mcp_root=formal)
    approved = ApprovedSourceReviewCache(cfg['source_review_index']) if cfg.get('source_review_index') else None
    adapter = CompletionAdapter(root / 'completion', formal / 'TOOLS/semantic_retention',
        formal, cfg['flash_patch'], grouped_transcription=True, semantic_repairs=False,
        selective_transcription=True, local_evidence_factory=factory, source_review=True,
        local_plain_acceptance=True, post_completion_pending_reference_salvage=True,
        native_symbol_corroboration=True,
        source_review_cache=approved,
        native_visible_source_review_index=cfg.get('native_visible_source_review_index'))
    adapter.agent.bounded_visual_protocol = True
    report = read(package / 'conversion_report.json')
    adapter.complete(source, package, report)
    if sha(source) != cfg['input_sha256']:
        raise ValueError('OPTIMIZED_SOURCE_CHANGED_DURING_CONVERSION')
    report = read(package / 'conversion_report.json')
    report['optimized_route'] = {'release_id': cfg['release_id'], 'semantic_repairs': False,
        'native_models_exited_before_completion': True,
        'timing_scope': 'FRESH_NATIVE_AND_COMPLETION_TO_VALIDATED_PACKAGE',
        'elapsed_seconds': time.monotonic() - started, 'input_sha256': cfg['input_sha256'],
        'configuration_sha256': sha(args.config), 'resource_profile': '6GiB_private_6144MiB_whole_GPU',
        'source_review_is_not_human_gold': True}
    write(package / 'conversion_report.json', report)
    previous = read(package / 'package_manifest.json')
    write(package / 'package_manifest.json', _package_manifest(previous['document_id'], source,
        cfg['input_sha256'], report, package, source_type=previous['source']['type']))
    validation = PackageValidator().validate(package).to_dict()
    if not validation['valid']:
        raise RuntimeError('OPTIMIZED_FINAL_PACKAGE_INVALID')
    value = {**native_result, 'success': True, 'package_path': str(package),
        'source_sha256': cfg['input_sha256'], 'optimized_route': report['optimized_route'],
        'validation': validation, 'quality_status': report['quality_status'],
        'review_item_count': report['review_item_count'], 'warnings': report['warnings']}
    write(root / 'result.json', value)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
