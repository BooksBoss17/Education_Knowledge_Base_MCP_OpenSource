"""Public MCP SDK acceptance smoke for the Education Knowledge Base adapter.

This intentionally exercises the published stdio launcher with the official MCP
Python SDK. It does not run model inference or bypass the visual-model gate.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys
import tempfile

from mcp import Client, StdioServerParameters

EXPECTED_TOOLS = {
    "bemarkdown_vision",
    "knowledge_workspace",
    "bemarkdown_info",
    "bemarkdown_convert",
    "bemarkdown_status",
    "bemarkdown_read",
    "bemarkdown_source",
    "bemarkdown_convert_image",
    "bemarkdown_review_context",
    "bemarkdown_review",
    "textbook_organize",
}
EXPECTED_PROTOCOL = "2025-11-25"
EXPECTED_SERVER = ("education-knowledge-base", "0.4.0")
EXPECTED_WHEEL_SHA256 = "e5f38d72920299d2543349cb04e1bc54e3aaed42bcea197ac65ef38c5edf3fd5"
BLOCKED_CALLS = {
    "bemarkdown_convert": {"source": "not-used-before-vision-gate.pdf"},
    "bemarkdown_status": {"job_id": "0" * 32},
    "bemarkdown_read": {"job_id": "0" * 32, "kind": "original"},
    "bemarkdown_source": {"job_id": "0" * 32, "kind": "text"},
    "bemarkdown_convert_image": {"job_id": "0" * 32, "asset_name": "assets/not-used.png"},
    "bemarkdown_review_context": {"job_id": "0" * 32, "task_id": "not-used"},
    "bemarkdown_review": {
        "job_id": "0" * 32,
        "base_sha256": "0" * 64,
        "replacements": [{"old": "x", "new": "y", "source_evidence": "not-used"}],
    },
    "textbook_organize": {"job_id": "0" * 32, "action": "inspect"},
}


def text_json(result):
    texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
    assert texts, "tool result did not include a text content block"
    return json.loads(texts[0])


async def exercise(root: Path, runtime_python: Path, workspace: Path, mode: str) -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            str(root / "TOOLS/education_mcp/launch.py"),
            "--mcp-root",
            str(root),
            "--workspace",
            str(workspace),
            "--runtime-python",
            str(runtime_python),
        ],
        cwd=str(root),
    )
    async with Client(params, mode=mode) as client:
        assert client.protocol_version == EXPECTED_PROTOCOL, client.protocol_version
        assert client.server_info is not None
        assert (client.server_info.name, client.server_info.version) == EXPECTED_SERVER

        listing = await client.list_tools()
        names = {tool.name for tool in listing.tools}
        assert names == EXPECTED_TOOLS, (sorted(names), sorted(EXPECTED_TOOLS))
        assert len(listing.tools) == 11

        workspace_result = await client.call_tool("knowledge_workspace", {"action": "inspect"})
        assert not workspace_result.is_error
        workspace_info = text_json(workspace_result)
        assert workspace_info["complete"] is True, workspace_info

        info_result = await client.call_tool("bemarkdown_info", {})
        assert not info_result.is_error
        info = text_json(info_result)
        assert info["server_version"] == "0.4.0"
        assert info["installed_runtime"]["verified"] is True
        assert info["installed_runtime"]["wheel_sha256"] == EXPECTED_WHEEL_SHA256
        assert info["visual_model_requirement"]["verified"] is False

        blocked_tools = []
        for tool_name, arguments in BLOCKED_CALLS.items():
            blocked = await client.call_tool(tool_name, arguments)
            assert blocked.is_error, tool_name
            message = "\n".join(
                block.text for block in blocked.content if getattr(block, "type", None) == "text"
            )
            assert "VISION_MODEL_REQUIRED" in message, (tool_name, message)
            blocked_tools.append(tool_name)

        challenge = await client.call_tool("bemarkdown_vision", {"action": "challenge"})
        assert not challenge.is_error
        challenge_meta = text_json(challenge)
        assert len(challenge_meta["challenge_id"]) == 32
        images = [block for block in challenge.content if getattr(block, "type", None) == "image"]
        assert len(images) == 1
        assert images[0].mime_type == "image/png"
        png = base64.b64decode(images[0].data, validate=True)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 1000

        return {
            "mode": mode,
            "protocol_version": client.protocol_version,
            "server": f"{client.server_info.name} {client.server_info.version}",
            "tools": len(listing.tools),
            "runtime_verified": info["installed_runtime"]["verified"],
            "vision_challenge_png_bytes": len(png),
            "gated_tool_calls": len(blocked_tools),
        }


async def main_async(root: Path, runtime_python: Path) -> list[dict]:
    summaries = []
    with tempfile.TemporaryDirectory(prefix="education-mcp-sdk-") as temp:
        base = Path(temp)
        for mode in ("auto", "legacy"):
            summaries.append(await exercise(root, runtime_python, base / mode, mode))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--mcp-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.mcp_root.resolve(strict=True)
    runtime_python = args.runtime_python.resolve(strict=True)
    summaries = asyncio.run(main_async(root, runtime_python))
    print(json.dumps({"status": "PASS", "sessions": summaries}, indent=2))


if __name__ == "__main__":
    main()
