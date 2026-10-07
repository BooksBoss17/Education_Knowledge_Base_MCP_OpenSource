"""python -m bemarkdown.figure_benchmark --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import (EvidenceError, audit_package, compare_scores, freeze_corpus, read_json, score,
                   seal, validate_corpus, verify_seal, write_json)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Isolated, source-bound figure benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="Freeze actual source identities; create draft truth")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate-corpus")
    validate.add_argument("--corpus", type=Path, required=True)
    audit = sub.add_parser("audit-package", help="Read-only identity and caption field audit, no visual verdict")
    audit.add_argument("--package", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    capture = sub.add_parser("capture", help="Run ONE frozen page using existing production models")
    capture.add_argument("--corpus", type=Path, required=True)
    capture.add_argument("--page-id", required=True)
    capture.add_argument("--mcp-root", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--work-root", type=Path, required=True, help="Short scratch root outside evidence, under developer tmp")
    scoring = sub.add_parser("score", help="Requires complete Root-reviewed truth and judgments")
    for name in ("corpus", "truth", "judgments", "run", "output"):
        scoring.add_argument(f"--{name}", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    compare = sub.add_parser("compare", help="Pair scored pages with identical corpus and truth")
    compare.add_argument("--left", type=Path, nargs="+", required=True)
    compare.add_argument("--right", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            corpus = freeze_corpus(args.spec, args.output)
            result = {"pages": len(corpus["pages"]), "truth": "DRAFT_NOT_SCORABLE"}
        elif args.command == "validate-corpus":
            validate_corpus(read_json(args.corpus))
            result = {"valid": True}
        elif args.command == "audit-package":
            report = audit_package(args.package)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "audit.json", report)
            seal(args.output)
            result = report["summary"]
        elif args.command == "capture":
            from .runner import capture_run
            report = capture_run(args.corpus, args.page_id, args.output, mcp_root=args.mcp_root, work_root=args.work_root)
            result = {"status": report["status"], "quality_status": report["quality_status"],
                      "timing_seconds": report["timing_seconds"]}
        elif args.command == "score":
            report = score(args.corpus, args.truth, args.judgments, args.run)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "score.json", report)
            seal(args.output)
            result = report
        elif args.command == "compare":
            result = compare_scores([read_json(p) for p in args.left], [read_json(p) for p in args.right])
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "comparison.json", result)
            seal(args.output)
        else:
            result = verify_seal(args.root)
            if not result["valid"]:
                print(json.dumps(result, ensure_ascii=False))
                return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (EvidenceError, OSError, KeyError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
