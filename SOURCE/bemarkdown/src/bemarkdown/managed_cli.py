"""Installed CLI adapter to the one durable, resource-managed conversion route."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys


def installation_root(explicit=None):
    value = explicit or os.environ.get('BEMARKDOWN_MCP_ROOT')
    if not value:
        receipt = Path(sys.prefix) / 'bemarkdown-installation.json'
        if receipt.is_file():
            value = json.loads(receipt.read_text(encoding='utf-8'))['mcp_root']
    if not value:
        raise ValueError('Select the installed MCP with --mcp-root or BEMARKDOWN_MCP_ROOT')
    root = Path(value).resolve(strict=True)
    if not (root / 'TOOLS/bemarkdown/ROUTE_MANIFEST.json').is_file():
        raise ValueError('Unified BeMarkdown installation is required')
    return root


def conversion(args):
    root = installation_root(args.mcp_root)
    from .config import resolve_output_root
    output = resolve_output_root(args.output_root, config_path=args.config)
    sources = args.input if isinstance(args.input, list) else [args.input]
    command = ['--source', str(sources[0]), '--output-root', str(output),
               '--mcp-root', str(root), '--json']
    for source in sources[1:]:
        command += ['--batch-source', str(source)]
    command += ['--layout-batch-size', str(getattr(args, 'layout_batch_size', 1))]
    for attr, flag in [('debug', '--debug'), ('no_image_screening', '--no-image-screening'),
                       ('repetition_guard', '--repetition-guard')]:
        if getattr(args, attr, False):
            command.append(flag)
    for attr, flag in [('retry_job_id', '--retry-job-id'), ('screen_existing', '--screen-existing')]:
        if getattr(args, attr, None):
            command += [flag, str(getattr(args, attr))]
    if args.resource_profile not in (None, '6gb'):
        raise ValueError('The accepted managed route uses the 6gb execution recipe; device admission remains adaptive')
    options = dict(formula_ocr=args.formula_ocr, mtef_cache=args.mtef_cache,
                   replace=not args.no_replace,
                   continue_on_error=getattr(args, 'continue_on_error', False),
                   reuse_formula_model=not getattr(args, 'no_reuse_formula_model', False))
    for name, value in [('mtef_cache_dir', args.mtef_cache_dir), ('config_path', args.config),
                        ('models_root', args.models_root)]:
        if value is not None:
            options[name] = str(Path(value).resolve())
    command += ['--core-options', json.dumps(options)]
    gateway = runpy.run_path(str(root / 'TOOLS/runtime_manager/native_gateway.py'))['gateway']
    return gateway(root / 'TOOLS/bemarkdown', command)
