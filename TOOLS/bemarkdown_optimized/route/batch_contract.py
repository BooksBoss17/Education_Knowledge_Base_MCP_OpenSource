"""Validate complete native batch result identity before declaring success."""
import json
from pathlib import Path


def validate_batch_result(result, bindings, output):
    if result.get('source_bindings') != bindings:
        raise ValueError('NATIVE_BATCH_SOURCE_RESULT_CONTRACT_FAILED')
    rows = result.get('documents')
    if not isinstance(rows, list) or len(rows) != len(bindings):
        raise ValueError('NATIVE_BATCH_CARDINALITY_MISMATCH')
    packages = set()
    counts = {'PUBLISHED': 0, 'FAILED': 0, 'NOT_RUN': 0}
    for index, (row, binding) in enumerate(zip(rows, bindings)):
        if row.get('index') != index or row.get('source') != binding['source']:
            raise ValueError('NATIVE_BATCH_ORDER_OR_SOURCE_MISMATCH')
        status = row.get('status')
        if status not in counts:
            raise ValueError('NATIVE_BATCH_STATUS_INVALID')
        counts[status] += 1
        if status != 'PUBLISHED':
            continue
        value = row.get('result', {})
        if value.get('success') is False:
            raise ValueError('NATIVE_BATCH_PUBLISHED_FAILURE')
        package = Path(value['package_path']).resolve(strict=True)
        if (not package.is_relative_to(output) or package in packages
                or not (package / 'document.md').is_file()):
            raise ValueError('NATIVE_PACKAGE_RESULT_ESCAPED_OR_DUPLICATED')
        manifest = json.loads((package / 'package_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('source', {}).get('sha256') != binding['sha256']:
            raise ValueError('NATIVE_BATCH_PACKAGE_SOURCE_MISMATCH')
        if 'source_sha256' in value and value['source_sha256'] != binding['sha256']:
            raise ValueError('NATIVE_BATCH_RESULT_SOURCE_MISMATCH')
        packages.add(package)
    if (result.get('requested_count') != len(bindings)
            or result.get('published_count') != counts['PUBLISHED']
            or result.get('failed_count') != counts['FAILED']
            or result.get('not_run_count') != counts['NOT_RUN']
            or result.get('success') is not (counts['PUBLISHED'] == len(bindings))):
        raise ValueError('NATIVE_BATCH_SUMMARY_MISMATCH')
    return result
