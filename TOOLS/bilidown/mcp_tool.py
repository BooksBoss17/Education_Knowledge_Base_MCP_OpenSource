"""Development MCP contract for BiliDown."""
from .backend import command
from .site_api import Failure, require
import time

SCHEMA={
    "name":"bilibili_download",
    "description":"Inspect a Bilibili BV/video link or download one part into this workspace. Guest is default. Use start/status/list/cancel/retry/verify. Quality is a ceiling; read actual selection and manifest. The public build uses guest access.",
    "inputSchema":{"type":"object","additionalProperties":False,"required":["action"],"properties":{
        "action":{"enum":["inspect","start","status","verify","list","cancel","retry"]},
        "source":{"type":"string","maxLength":2048},"part":{"type":"integer","minimum":1,"maximum":99999},
        "height":{"enum":[360,480,720,1080,2160]},"auth":{"enum":["guest"]},
        "job_id":{"type":"string","pattern":"^[0-9a-f]{32}$"},"tab":{"enum":["login","download","settings"]}}}}

def call(service,action,**args):
    task=service._task_run.get()
    binding=[str(service.mcp_root),task] if task else None
    try:
        require(args.get("auth","guest")=="guest" and action!="open_ui","GUEST_ONLY","Public download uses guest access.")
        result=command(service.workspace,action,task=binding,**args)
        # Keep the outer task call open until its new worker has registered.
        if binding and action in ("start","retry") and result["state"]=="STARTING":
            deadline=time.monotonic()+15
            while result["state"]=="STARTING":
                require(time.monotonic()<deadline,"START_PENDING","启动仍在进行，请按工作区任务清单核查。")
                time.sleep(.1)
                result=command(service.workspace,"status",job_id=result["job_id"])
        return {"ok":True,"data":result}
    except Failure as exc:
        return {"ok":False,"error":{"code":exc.code,"message":str(exc)}}

