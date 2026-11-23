"""Agent-facing, guest-only video discovery."""
import json
from pathlib import Path
import subprocess
import sys

from .direct_network import child_environment

SCHEMA={"name":"bilibili_search","description":"Read-only guest Bilibili webpage search via Microsoft Playwright MCP and Edge, by title, author nickname, or UID. In title search title_exact requires a literal substring. For author posts compare complete, returned and total; never infer completion from success. Continue at next_page with the same page_size. When a nickname is ambiguous select a candidate UID.",
        "inputSchema":{"type":"object","additionalProperties":False,"properties":{
            "title":{"type":"string","minLength":1,"maxLength":120},"author":{"type":"string","minLength":1,"maxLength":64},
            "uid":{"type":"string","pattern":"^[1-9][0-9]{0,18}$"},"page":{"type":"integer","minimum":1,"maximum":10000},
            "page_size":{"type":"integer","minimum":1,"maximum":50},"max_pages":{"type":"integer","minimum":1,"maximum":20},
            "title_exact":{"type":"boolean"}}}}

def call(service,**args):
    import re
    for key,maxlen in (("title",120),("author",64)):
        if key in args and (not isinstance(args[key],str) or not 1<=len(args[key].strip())<=maxlen):
            raise ValueError("Invalid "+key)
    if "uid" in args and (not isinstance(args["uid"],str) or not re.fullmatch(r"[1-9][0-9]{0,18}",args["uid"])):
        raise ValueError("Invalid UID")
    if not any(args.get(k) for k in ("uid","title","author")) or (args.get("author") and args.get("uid")):
        raise ValueError("Supply title, author or UID; select an author by nickname or UID")
    for key,maximum in (("page",10000),("page_size",50),("max_pages",20)):
        if key in args and (type(args[key]) is not int or not 1<=args[key]<=maximum):raise ValueError("Invalid "+key)
    if "title_exact" in args and type(args["title_exact"]) is not bool:raise ValueError("Invalid title_exact")
    runtime=service.mcp_root/".local/runtimes/bili-browser-v1"
    if not (runtime / "node.exe").is_file() or not (runtime / "node_modules/@playwright/mcp/cli.js").is_file():
        raise ValueError("Install Edge and run scripts/setup_video.py --browser")
    worker=Path(__file__).with_name("browser_search.py")
    try:
        p=subprocess.run([sys.executable,"-I","-X","utf8",worker,runtime],input=json.dumps(args).encode(),capture_output=True,
                         env=child_environment(),timeout=155,creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return dict(complete=False,reason={"code":"BROWSER_TIMEOUT"},next_page=args.get("page",1))
    if p.returncode:raise RuntimeError("Search worker failed")
    return json.loads(p.stdout)
