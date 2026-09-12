"""Deployment entry: initialize the workspace and return standard MCP client configuration."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--mcp-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    manager = runpy.run_path(str(Path(__file__).with_name('workspace_manager.py')))['WorkspaceManager'](args.workspace)
    result = manager.bootstrap()
    result['mcp_client_configuration'] = {'mcpServers': {'education': {
        'command': sys.executable,
        'args': [str(Path(__file__).with_name('launch.py')), '--mcp-root', str(args.mcp_root.resolve()),
                 '--workspace', str(manager.root.resolve())]}}}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
