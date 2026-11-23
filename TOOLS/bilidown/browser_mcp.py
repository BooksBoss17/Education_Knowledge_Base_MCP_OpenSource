"""Private stdio client for the official Microsoft Playwright MCP server."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .direct_network import child_environment
from .browser_network import browser_egress

HERE=Path(__file__).resolve().parent

class Browser:
    def __init__(self,proc,events):
        self.proc=proc; self.events=events; self.seq=0

    async def send(self,message):
        self.proc.stdin.write(json.dumps(message,ensure_ascii=False).encode()+b"\n")
        await self.proc.stdin.drain()

    async def rpc(self,method,params):
        self.seq+=1
        await self.send(dict(jsonrpc="2.0",id=self.seq,method=method,params=params))
        async with asyncio.timeout(30):
            while line:=await self.proc.stdout.readline():
                message=json.loads(line)
                if message.get("id")==self.seq:
                    if "error" in message:raise RuntimeError("PLAYWRIGHT_RPC_ERROR")
                    return message["result"]
            raise RuntimeError("BROWSER_EXITED")

    async def tool(self,name,**args):
        v=await self.rpc("tools/call",dict(name=name,arguments=args))
        if v.get("isError"):raise RuntimeError("BROWSER_ACTION_FAILED")
        return "\n".join(x.get("text","") for x in v["content"])

    async def navigate(self,url):
        await self.tool("browser_navigate",url=url)

    async def evaluate(self,expression):
        text=await self.tool("browser_evaluate",function=expression)
        section=text.partition("### Result\n")[2]
        if not section:raise RuntimeError("BROWSER_RESULT_UNAVAILABLE")
        return json.JSONDecoder().raw_decode(section.lstrip())[0]

    def response(self,uid,page):
        if self.events.is_file():
            for line in reversed(self.events.read_text("utf8").splitlines()):
                try:row=json.loads(line)
                except ValueError:continue
                if row["uid"]==uid and row["page"]==page:return row
        return None

@asynccontextmanager
async def session(runtime):
    with browser_egress() as proxy:
        with tempfile.TemporaryDirectory(prefix="ekb-bili-") as directory:
            tmp=Path(directory);events=tmp/"page.jsonl"
            config=dict(browser=dict(browserName="chromium",isolated=True,launchOptions=dict(
                channel="msedge",headless=True,chromiumSandbox=True,proxy=proxy,args=["--disable-quic","--disable-background-networking","--proxy-bypass-list=<-loopback>"]),
                contextOptions=dict(locale="zh-CN",viewport=dict(width=1440,height=1000),serviceWorkers="block"),
                initPage=[str(HERE/"browser_page.cjs")]),snapshot=dict(mode="none"),
                outputDir=str(tmp),timeouts=dict(action=8000,navigation=18000))
            path=tmp/"browser.json";path.write_text(json.dumps(config),"utf8")
            env=child_environment();env["EKB_BILI_PAGE_EVENTS"]=str(events)
            proc=await asyncio.create_subprocess_exec(runtime/"node.exe",runtime/"node_modules/@playwright/mcp/cli.js",
                "--config",path,"--no-webmcp",stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                cwd=tmp,env=env,creationflags=subprocess.CREATE_NO_WINDOW,limit=2*1024**2)
            browser=Browser(proc,events)
            try:
                await browser.rpc("initialize",dict(protocolVersion="2025-03-26",capabilities={},clientInfo=dict(name="bili-read-only",version="1")))
                await browser.send(dict(jsonrpc="2.0",method="notifications/initialized"))
                yield browser
            finally:
                proc.stdin.close()
                try:await asyncio.wait_for(proc.wait(),7)
                except TimeoutError:
                    proc.kill();await proc.wait()
