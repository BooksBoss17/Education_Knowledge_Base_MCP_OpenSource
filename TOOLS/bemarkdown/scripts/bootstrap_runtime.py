from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap an isolated BeMarkdown runtime")
    parser.add_argument("--tool-root", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--profile", choices=("core", "full"), default="full")
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--no-doctor", action="store_true")
    args = parser.parse_args(argv)

    tool_root = (args.tool_root or Path(__file__).resolve().parents[1]).resolve()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Bootstrap requires the selected host CPython 3.11")
    manifest = json.loads((tool_root / "TOOL_MANIFEST.json").read_text(encoding="utf-8"))
    runtime_dir = (
        args.runtime_dir or _default_runtime_dir(manifest["wheel"]["sha256"])
    ).resolve()
    if runtime_dir.exists():
        raise FileExistsError(f"Runtime directory already exists: {runtime_dir}")
    wheel = tool_root / manifest["wheel"]["path"]
    core_lock = tool_root / manifest["runtime"]["core_lock"]["path"]
    formula_lock = tool_root / manifest["runtime"]["formula_ocr_lock"]["path"]
    text_lock_entry = manifest["runtime"].get("text_ensemble_lock")
    if not isinstance(text_lock_entry, dict):
        raise TypeError("Tool manifest is missing the text ensemble lock")
    text_lock = tool_root / text_lock_entry["path"]
    _require_sha(wheel, manifest["wheel"]["sha256"])
    _require_sha(core_lock, manifest["runtime"]["core_lock"]["sha256"])
    _require_sha(formula_lock, manifest["runtime"]["formula_ocr_lock"]["sha256"])
    _require_sha(text_lock, text_lock_entry["sha256"])

    venv.EnvBuilder(with_pip=True, clear=False).create(runtime_dir)
    python = runtime_dir / "Scripts" / "python.exe"
    pip = [str(python), "-m", "pip"]
    offline = args.wheelhouse is not None
    common = (
        ["--no-index", "--find-links", str(args.wheelhouse.resolve())]
        if offline
        else []
    )

    if offline:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", encoding="utf-8", delete=False
        ) as handle:
            materialized_core = Path(handle.name)
            handle.write(
                core_lock.read_text(encoding="utf-8").replace(
                    "mathtypejx @ git+https://github.com/a917470154/mathtypejx.git@"
                    "7d90e7274c85cf56ac28d4d15e593044693d7e70",
                    "mathtypejx==0.1.0",
                )
            )
        try:
            _run([*pip, "install", *common, "-r", str(materialized_core)])
        finally:
            materialized_core.unlink(missing_ok=True)
    else:
        _run([*pip, "install", "-r", str(core_lock)])

    if args.profile == "full":
        if offline:
            _run(
                [
                    *pip,
                    "install",
                    *common,
                    "--no-deps",
                    "-r",
                    str(formula_lock),
                ]
            )
            _run(
                [
                    *pip,
                    "install",
                    *common,
                    "--no-deps",
                    "-r",
                    str(text_lock),
                ]
            )
        else:
            _run(
                [
                    *pip,
                    "install",
                    "paddlepaddle-gpu==3.2.2",
                    "--index-url",
                    "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
                ]
            )
            _run([*pip, "install", "paddlex[ocr]==3.7.2", "numpy==2.3.5"])
            _run([*pip, "install", "--no-deps", "-r", str(formula_lock)])
            _run(
                [
                    *pip,
                    "install",
                    "--no-deps",
                    "torch==2.13.0+cu130",
                    "torchvision==0.28.0+cu130",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu130",
                ]
            )
            _run([*pip, "install", "--no-deps", "-r", str(text_lock)])

    _run([*pip, "install", "--no-deps", str(wheel)])
    check = subprocess.run([*pip, "check"], capture_output=True, text=True, check=False)
    accepted_conflict = (
        args.profile == "full"
        and check.returncode == 1
        and _only_expected_cudnn_conflict(check.stdout + check.stderr)
    )
    if check.returncode and not accepted_conflict:
        raise RuntimeError(f"pip check failed:\n{check.stdout}{check.stderr}")

    doctor = None
    if not args.no_doctor:
        command = [
            str(python),
            "-m",
            "bemarkdown",
            "doctor",
            "--json",
            "--deep",
            "--tool-root",
            str(tool_root),
            "--config",
            str(tool_root / "config" / "bemarkdown.example.toml"),
        ]
        if args.profile == "core":
            command.append("--skip-formula-runtime")
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        doctor = json.loads(completed.stdout)
        expected = "READY_FULL" if args.profile == "full" else "READY_DEGRADED"
        if doctor["readiness"] != expected:
            raise RuntimeError(
                f"Doctor readiness {doctor['readiness']} does not match {expected}"
            )

    result = {
        "schema": "bemarkdown-bootstrap-result-v1",
        "profile": args.profile,
        "offline": offline,
        "runtime_directory": str(runtime_dir),
        "python": str(python),
        "pip_check": {
            "returncode": check.returncode,
            "expected_cudnn_metadata_override": accepted_conflict,
            "output": (check.stdout + check.stderr).strip(),
        },
        "doctor": doctor,
    }
    (runtime_dir / "bootstrap_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _only_expected_cudnn_conflict(output: str) -> bool:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return len(lines) == 1 and "nvidia-cudnn-cu12==9.5.1.17" in lines[0] and "9.9.0.52" in lines[0]


def _default_runtime_dir(wheel_sha256: str) -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "BeMarkdown" / "runtimes" / wheel_sha256[:12]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _require_sha(path: Path, expected: str) -> None:
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"Immutable payload SHA mismatch: {path.name}")


if __name__ == "__main__":
    raise SystemExit(main())
