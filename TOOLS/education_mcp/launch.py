"""Resolve the published BeMarkdown runtime and launch the standard stdio MCP service."""
import argparse
import json
import os
import runpy
import subprocess
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--input-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    manager = runpy.run_path(str(Path(__file__).with_name('workspace_manager.py')))['WorkspaceManager'](args.workspace)
    manager.bootstrap()
    args.workspace = manager.root
    root = args.mcp_root.resolve(strict=True)
    manifest = json.loads((root / "TOOLS/bemarkdown/TOOL_MANIFEST.json").read_text(encoding="utf-8"))
    runtime_home = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "BeMarkdown/runtimes"
    python = args.runtime_python or runtime_home / manifest["wheel"]["sha256"][:12] / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        parser.error("Published BeMarkdown runtime is unavailable; configure --runtime-python or prepare that Tool's full runtime first")
    command = [str(python.resolve()), "-I", "-X", "utf8", str(Path(__file__).with_name("server.py")),
               "--mcp-root", str(root), "--workspace", str(args.workspace.resolve()), "--bemarkdown-python", str(python.resolve())]
    for input_root in args.input_root:
        command.extend(["--input-root", str(input_root.resolve())])
    # Windows execv starts a replacement process then exits this launcher.
    # MCP clients observe that exit and close the transport before initialize.
    # Keep the launcher alive while the child inherits the protocol streams.
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
