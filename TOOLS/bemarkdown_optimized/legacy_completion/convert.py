"""Quiet, resource-supervised PDF/Word entry for an installed route payload."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verified_payload(tool):
    manifest = read(tool / 'ROUTE_MANIFEST.json')
    for relative, expected in manifest['files'].items():
        path = (tool / relative).resolve()
        if not path.is_relative_to(tool.resolve()) or sha(path) != expected:
            raise ValueError('OPTIMIZED_ROUTE_PAYLOAD_CHANGED:' + relative)
    if manifest['semantic_repairs_enabled'] or manifest['resource_policy'] != '6GiB/6144MiB/8GiB/9GiB':
        raise ValueError('OPTIMIZED_ROUTE_POLICY_CHANGED')
    return manifest


def private_profile(args):
    configuration = {}
    if args.flash_patch:
        if not args.flash_patch_sha256:
            raise ValueError('FLASH_PATCH_SHA256_REQUIRED')
        if not Path(args.flash_patch).is_absolute():
            raise ValueError('FLASH_PATCH_ABSOLUTE_PATH_REQUIRED')
        patch = Path(args.flash_patch).resolve(strict=True)
        expected = args.flash_patch_sha256
    else:
        profile = Path(os.environ.get('BEMARKDOWN_OPTIMIZED_PROFILE',
            str(Path(os.environ['LOCALAPPDATA']) / 'BeMarkdown/optimization/route-profile.json')))
        if not profile.is_file():
            raise ValueError('OPTIMIZED_ROUTE_PRIVATE_PROFILE_NOT_CONFIGURED')
        configuration = read(profile)
        if not Path(configuration['flash_patch_path']).is_absolute():
            raise ValueError('FLASH_PATCH_ABSOLUTE_PATH_REQUIRED')
        patch = Path(configuration['flash_patch_path']).resolve(strict=True)
        expected = configuration['flash_patch_sha256']
    if not patch.is_absolute() or sha(patch) != expected:
        raise ValueError('OPTIMIZED_ROUTE_PRIVATE_PROFILE_CHANGED')
    external = {}
    for name in ('source_review_index', 'native_visible_source_review_index'):
        if configuration.get(name):
            if not Path(configuration[name]).is_absolute():
                raise ValueError('APPROVED_REVIEW_ABSOLUTE_PATH_REQUIRED:' + name)
            path = Path(configuration[name]).resolve(strict=True)
            expected_index = configuration[name + '_sha256']
            if sha(path) != expected_index:
                raise ValueError('OPTIMIZED_APPROVED_REVIEW_INDEX_CHANGED:' + name)
            external[name] = str(path)
            external[name + '_sha256'] = expected_index
    return patch, expected, external


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--mcp-root', required=True)
    parser.add_argument('--flash-patch')
    parser.add_argument('--flash-patch-sha256')
    parser.add_argument('--caller-pid', type=int)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    tool = Path(__file__).resolve().parent
    manifest = verified_payload(tool)
    source = Path(args.source).resolve(strict=True)
    if source.suffix.lower() not in {'.pdf', '.docx', '.doc'}:
        raise ValueError('OPTIMIZED_ROUTE_REQUIRES_PDF_OR_WORD')
    output = Path(args.output_root).resolve()
    if output == source.parent or output in source.parents:
        raise ValueError('OPTIMIZED_OUTPUT_MUST_BE_SEPARATE_FROM_SOURCE')
    output.mkdir(parents=True, exist_ok=True)
    root = output / '.ocr-work'
    root.mkdir(exist_ok=False)
    patch, patch_sha, external = private_profile(args)
    sys.path.insert(0, str(tool / 'route'))
    import e2e_supervise as legacy
    import e2e_supervise_v3 as supervision
    caller = None
    controllers = []
    if args.caller_pid:
        launcher = None
        if args.caller_pid != os.getppid():
            # CPython's Windows venv redirector is a separate OS process.
            # Permit exactly this known one-hop launcher, never arbitrary ancestry.
            import psutil
            parent = psutil.Process(os.getppid())
            if parent.ppid() != args.caller_pid or Path(parent.exe()).resolve() != Path(sys.executable).resolve():
                raise ValueError('OPTIMIZED_CALLER_IS_NOT_PARENT_OR_VENV_LAUNCHER')
            launcher = legacy.process_info(parent.pid)
            if launcher is None:
                raise ValueError('OPTIMIZED_VENV_LAUNCHER_IDENTITY_UNAVAILABLE')
        caller = legacy.process_info(args.caller_pid)
        if caller is None:
            raise ValueError('OPTIMIZED_CALLER_IDENTITY_UNAVAILABLE')
        controllers = [caller] + ([launcher] if launcher is not None else [])
        # Reserve and continuously charge the shared MCP server without ever
        # placing it in this task's kill-on-close Job.
        os.environ['BEMARKDOWN_ROUTE_CALLER_PRIVATE_RESERVE_GIB'] = str(sum(c['private_commit_gib'] for c in controllers))
        base = supervision.StrictFamily
        class CompleteFamily(base):
            def record(self, row):
                charged = {p['pid'] for p in row['job_members']} | {row['runner']['pid']}
                additional = []
                for controller in controllers:
                    current = legacy.process_info(controller['pid'])
                    if current is None or not self.same_identity(controller, current):
                        raise RuntimeError('OPTIMIZED_CALLER_IDENTITY_CHANGED')
                    if current['pid'] not in charged:
                        row['host']['project_private_commit_gib'] += current['private_commit_gib']
                        row['host']['project_working_gib'] += current['working_gib']
                    additional.append(current)
                row['additional_shared_controllers'] = additional
                return super().record(row)
        supervision.StrictFamily = CompleteFamily
    else:
        os.environ.pop('BEMARKDOWN_ROUTE_CALLER_PRIVATE_RESERVE_GIB', None)
    resources = read(tool / 'RESOURCE_STAGES.json')
    for value in resources['stages'].values():
        value['evidence']=[{'path':str(tool/'RESOURCE_STAGES.json'), 'sha256':sha(tool/'RESOURCE_STAGES.json')}]
    admission = {'version': 'measured-stage-admission-v1', 'stages': resources['stages'],
        'bindings': [{'path': str(tool / relative), 'sha256': expected}
                     for relative, expected in manifest['files'].items()],
        'timing_scope': 'NEW_NATIVE_AND_COMPLETION_RUN',
        'unknown_stages': 'REFUSE_BEFORE_LOAD', 'continuous_limits_required': True}
    admission_path = root / 'stage-admission.json'
    write(admission_path, admission)
    cfg = {'tool_root': str(tool), 'run_root': str(root), 'output_root': str(output),
        'input': str(source), 'input_sha256': sha(source), 'mcp_root': str(Path(args.mcp_root).resolve()),
        'flash_patch': str(patch), 'flash_patch_sha256': patch_sha,
        'release_id': manifest['release_id'], 'payload_files': manifest['files'],
        'stage_admission': {'manifest_path': str(admission_path), 'manifest_sha256': sha(admission_path)},
        'caller_identity': caller, 'additional_controller_identities':controllers,
        'semantic_repairs_enabled': False, **external}
    configuration = root / 'config.json'
    write(configuration, cfg)
    old_argv = sys.argv
    try:
        sys.argv = [str(tool / 'route/e2e_supervise_v3.py'), '--out', str(root / 'monitor'),
            '--python', sys.executable, '--worker', str(tool / 'route/portable_optimization_worker.py'),
            '--timeout', '14400', '--config', str(configuration)]
        code = supervision.main()
    finally:
        sys.argv = old_argv
    monitor = read(root / 'monitor/result.json')
    if code or monitor['status'] != 'COMPLETED':
        print(json.dumps({'success': False, 'route_release': manifest['release_id'],
            'status': monitor['status'], 'diagnostic_path': str(root / 'monitor/result.json')}, ensure_ascii=False))
        return code or 1
    result = read(root / 'result.json')
    result['resource_measurement_path'] = str(root / 'monitor/result.json')
    result['route_release'] = manifest['release_id']
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
