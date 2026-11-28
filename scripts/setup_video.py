"""Provision optional video dependencies in the ignored local installation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
SPEECH = ROOT / "TOOLS/speech_transcribe"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_verified(source, destination, expected):
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"Existing dependency differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(destination) != expected:
        destination.unlink()
        raise ValueError(f"Content changed during copy: {source}")


def configure_binaries(ffmpeg, aria2):
    if not ffmpeg or not aria2:
        raise ValueError("Provide both --ffmpeg and --aria2 with executable paths")
    folder = LOCAL / "video-bin"
    hashes = {}
    for name, path in (("ffmpeg", ffmpeg), ("aria2c", aria2)):
        source = path.resolve(strict=True)
        if source.suffix.lower() != ".exe" or not source.is_file():
            raise ValueError(f"Expected Windows executable: {source}")
        hashes[name] = digest(source)
        copy_verified(source, folder / f"{name}.exe", hashes[name])
    temporary = folder / "manifest.pending"
    temporary.write_text(json.dumps(hashes, indent=2), "utf-8")
    temporary.replace(folder / "manifest.json")


def configure_browser():
    npm = shutil.which("npm.cmd")
    node = shutil.which("node.exe")
    if not npm or not node:
        raise ValueError("Install Node.js 20+ with npm first")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if int(version.split(".")[0].lstrip("v")) < 20:
        raise ValueError("Node.js 20+ is required")
    source = ROOT / "TOOLS/bilidown/browser-runtime"
    runtime = LOCAL / "runtimes/bili-browser-v1"
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("package.json", "package-lock.json"):
        shutil.copyfile(source / name, runtime / name)
    node_path = Path(node)
    copy_verified(node_path, runtime / "node.exe", digest(node_path))
    subprocess.run([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund", "--prefix", str(runtime)], check=True)


def configure_speech(source):
    source = source.resolve(strict=True)
    models = []
    for manifest in sorted((SPEECH / "model-manifests").glob("*.json")):
        identity = json.loads(manifest.read_text("utf-8"))
        name = manifest.stem
        original = source / name
        destination = ROOT / "MODELS" / name
        for item in identity["files"]:
            path = original / item["path"]
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Download the specified upstream revision: {path}")
            copy_verified(path, destination / item["path"], item["sha256"])
        models.append((manifest, destination / "MODEL_MANIFEST.json"))
    runtime = LOCAL / "runtimes/qwen3-speech-006"
    python = runtime / "Scripts/python.exe"
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(runtime)
    subprocess.run([python, "-m", "pip", "install", "-r", SPEECH / "requirements.lock"], check=True)
    for manifest, target in models:
        copy_verified(manifest, target, digest(manifest))


def configure_speech_api(source):
    source = source.resolve(strict=True)
    sys.path.insert(0, str(ROOT / "TOOLS"))
    from speech_transcribe.routing import resolve
    route = resolve(ROOT, config_path=source)
    if any(spec["type"] != "http" for name, spec in route.items() if name != "timing"):
        raise ValueError("Use --speech-model-root for a route with managed local models")
    runtime = LOCAL / "runtimes/speech-api-v1"
    python = runtime / "Scripts/python.exe"
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(runtime)
    lock = (SPEECH / "requirements.lock").read_text("utf8").splitlines()
    packages = [line for line in lock if line.startswith(("av==", "numpy=="))]
    if len(packages) != 2:
        raise ValueError("Speech decoder dependencies missing from lock")
    subprocess.run([python, "-m", "pip", "install", *packages], check=True)
    target = LOCAL / "speech-api.json"
    temp = target.with_suffix(".pending")
    shutil.copyfile(source, temp)
    temp.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--aria2", type=Path)
    parser.add_argument("--browser", action="store_true")
    speech = parser.add_mutually_exclusive_group()
    speech.add_argument("--speech-model-root", type=Path)
    speech.add_argument("--speech-api-config", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11):
        parser.error("Use Windows CPython 3.11")
    if not any((args.ffmpeg, args.aria2, args.browser, args.speech_model_root, args.speech_api_config)):
        parser.error("Select at least one optional video dependency")
    if args.ffmpeg or args.aria2:
        configure_binaries(args.ffmpeg, args.aria2)
    if args.browser:
        configure_browser()
    if args.speech_model_root:
        configure_speech(args.speech_model_root)

    if args.speech_api_config:
        configure_speech_api(args.speech_api_config)
