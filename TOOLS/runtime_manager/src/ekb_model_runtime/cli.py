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
    drain = sub.add_parser("drain-manager", help="Stop an idle run manager and confirm its exit")
    drain.add_argument("--manager-home", help="Optional explicit run-manager directory")
    args = parser.parse_args()
    if args.command == "drain-manager":
        from .run_manager import RunManagerClient, RunManagerError

        try:
            result = RunManagerClient(args.manager_home).shutdown(wait_seconds=15)
        except (RunManagerError, OSError) as error:
            print(json.dumps({"shutdown_requested": False,
                              "error": getattr(error, "code", type(error).__name__)}))
            return 1
        print(json.dumps(result))
        return 0
    result = run_doctor(
        args.models_root, platform=args.platform, device=args.device, deep=args.deep
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["readiness"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
