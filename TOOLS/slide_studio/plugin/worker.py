import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deck import check
from mcp_stdio import TOOLS, invoke_tool
from portable import filename
from store import Store
from studio import DEFAULT_HOME


def execute(request):
    name = request["name"].removeprefix("studio_")
    check(name in TOOLS, "未知课页工具")
    args = request["args"]
    check(isinstance(args, dict), "参数必须为对象")
    workspace = Path(request["workspace"])
    roots = [Path(p) for p in request["roots"]]
    store = Store(DEFAULT_HOME / "studio.sqlite")
    def allowed(path, scope):
        resolved = path.resolve()
        check(any(resolved.is_relative_to(p) for p in scope), "课件路径超出配置的目录")
        return resolved
    if name == "import":
        src = allowed(workspace / args["path"], roots)
        if src.suffix.lower() == ".html":
            images = allowed(src.parent / "images", roots)
            for asset in images.glob("[0-9]*.png"):
                allowed(asset, roots)
        args["path"] = str(src)
    if name in {"export", "package"}:
        directory = allowed(workspace / args["directory"], [workspace])
        output = filename(store.state(args["id"])["deck"]["title"])
        if name == "package":
            output = output[:-5] + ".zip"
        allowed(directory / output, [workspace])
        args["directory"] = str(directory)
    return invoke_tool(store, name, args)


if __name__ == "__main__":
    try:
        response = execute(json.load(sys.stdin))
    except Exception as exc:
        response = {"isError": True, "content": [{"type":"text","text":str(exc)}]}
    print(json.dumps(response, ensure_ascii=True))
