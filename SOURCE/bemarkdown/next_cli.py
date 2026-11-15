"""Small CLI for conversion, package validation and local API configuration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='bemarkdown')
    sub = parser.add_subparsers(dest='command', required=True)
    convert = sub.add_parser('convert')
    convert.add_argument('source', type=Path)
    convert.add_argument('--output-root', type=Path)
    convert.add_argument('--api-config', type=Path)
    convert.add_argument('--formula-ocr', choices=['auto','off'], default='auto')
    convert.add_argument('--debug', action='store_true')
    convert.add_argument('--no-replace', action='store_true')
    convert.add_argument('--json', action='store_true')
    validate = sub.add_parser('validate')
    validate.add_argument('package', type=Path)
    validate.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            from .package_io import validate_package
            result = validate_package(args.package)
            value, ok = result.to_dict(), result.valid
        else:
            from .next_runtime import convert_document
            result = convert_document(args.source, args.output_root, api_config=args.api_config,
                formula_ocr=args.formula_ocr, debug=args.debug, replace=not args.no_replace)
            value, ok = result.to_dict(), result.success
        print(json.dumps(value, ensure_ascii=False))
        return 0 if ok else 1
    except Exception as exc:
        print(json.dumps({'success':False, 'error_type':type(exc).__name__, 'error':str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
