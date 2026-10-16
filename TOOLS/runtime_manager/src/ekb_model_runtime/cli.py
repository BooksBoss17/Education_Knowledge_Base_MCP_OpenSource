from __future__ import annotations

import argparse
import json

from .doctor import run_doctor


def main() -> int:
    parser = argparse.ArgumentParser(prog="ekb-model-runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--models-root", required=True)
    doctor.add_argument("--platform", default="windows-x86_64")
    doctor.add_argument("--device", default="cuda:0")
    doctor.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    result = run_doctor(
        args.models_root, platform=args.platform, device=args.device, deep=args.deep
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["readiness"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
