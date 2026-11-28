"""stdio MCP entry for agents using the same transactional document store."""

import base64
import json
import sys
import threading

import session_link

from authoring import ANIMATION_GUIDANCE

STR = {"type": "string"}


def props(fields, required):
    return {
        "type": "object",
        "properties": fields,
        "required": required,
        "additionalProperties": False,
    }


TOOLS = {
    "session_pair": (
        "Redeem the one-use code pasted by the user in this chat. Keep the returned credential within this conversation; do not print it. Host and label are display names, not authentication.",
        props({"code": STR, "host": {"type":"string", "enum":list(session_link.HOSTS)}, "label": STR, "thread": STR}, ["code","host"]),
    ),
    "session_receive": (
        "Wait for the oldest item addressed to this conversation (0–45 seconds, default 20). History and selected objects are frozen at send time. Acknowledge the feedbackId, then reply. An unfinished item may be redelivered; timeout means no message.",
        props({"credential": STR, "timeout": {"type":"integer","minimum":0,"maximum":45}}, ["credential"]),
    ),
    "session_ack": (
        "Confirm this chat actually received a delivered item. Idempotent.",
        props({"credential": STR, "feedback_id": STR}, ["credential","feedback_id"]),
    ),
    "session_reply": (
        "Reply to a received item; optional page-scoped edits are applied atomically with the receipt. Default base is the original revision. If a field changed, inspect the latest page before explicitly rebasing. Retry with identical arguments.",
        props({"credential":STR,"feedback_id":STR,"reply":STR,"outcome":{"type":"string","enum":["resolved","needs_input"]},
               "base":{"type":"integer","minimum":0},"edits":{"type":"array","items":{"type":"object"}}}, ["credential","feedback_id","reply"]),
    ),
    "catalog": ("List animation keys, transitions or interactive templates. Set section to effects, transitions or templates. Supply a template name to get editable HTML and defaults.", props({"section": STR, "name": STR}, [])),
    "library": ("List locally editable presentations.", props({}, [])),
    "inspect": (
        "Inspect page list or a single slide, stable object IDs, ordered steps and revision.",
        props({"id": STR, "slide": STR}, ["id"]),
    ),
    "create": ("Start an empty presentation.", props({"title": STR}, ["title"])),
    "import": (
        "Import a user-provided local WPS PPT/PPTX or exported HTML package into the editor.",
        props({"path": STR}, ["path"]),
    ),
    "feedback": (
        "Get persisted user comments with page, selection, region and playback step. Poll for new pending items.",
        props({"id": STR}, ["id"]),
    ),
    "edit": (
        "Apply atomic commands. Read inspect beforehand. Supply command={op:'edit',actor,base,request,edits:[{path:['slides',slideId,'elements',objectId,'x'],value:42}]}. Use a fresh request ID per distinct command, reuse for retry. To resolve feedback include feedbackId and reply in the same edit. Animation arrays have target, effect (see studio_catalog or legacy effects), trigger (click/with/after), duration and delay in seconds. To undo/redo use op undo/redo and current base. " + ANIMATION_GUIDANCE,
        props({"id": STR, "command": {"type": "object"}}, ["id", "command"]),
    ),
    "export": (
        "Export offline HTML under the deck title to an existing directory.",
        props({"id": STR, "directory": STR}, ["id", "directory"]),
    ),
    "package": (
        "Export a ZIP for the knowledge base with named HTML and per-slide reference images.",
        props({"id": STR, "directory": STR}, ["id", "directory"]),
    ),
    "snapshot": (
        "Render a page in editing view, at a playback step, or return its original WPS reference. Returns a PNG.",
        props(
            {
                "id": STR,
                "slide": STR,
                "step": {"type": "integer", "minimum": 0},
                "reference": {"type": "boolean"},
                "revision": {"type":"integer", "minimum":0},
            },
            ["id", "slide"],
        ),
    ),
}


def invoke_tool(store, name, args, stop=None):
    from studio import agent_call
    from snapshot import snapshot

    if name == "snapshot":
        data = snapshot(store, **args)
        content = [dict(type="image", mimeType="image/png", data=base64.b64encode(data).decode())]
    else:
        if name == "session_receive":
            value = session_link.receive(store, **args, stop=stop)
        elif name in ("session_pair", "session_ack", "session_reply"):
            handler = {"session_pair": session_link.pair, "session_ack": session_link.ack,
                       "session_reply": session_link.reply}[name]
            value = handler(store, **args)
        else:
            if name not in TOOLS:
                raise ValueError("Unknown tool")
            value = agent_call(store, name, args)
        content = [{"type":"text", "text":json.dumps(value, ensure_ascii=False)}]
    return {"content":content}


def serve(store):

    output_lock = threading.Lock()
    waits_lock = threading.Lock()
    waits = {}

    def respond(message):
        with output_lock:
            print(json.dumps(message, ensure_ascii=True), flush=True)

    def run_tool(ident, name, args, stop=None):
        try:
            result = invoke_tool(store, name, args, stop)
        except Exception as exc:
            result = {"isError":True,"content":[{"type":"text","text":str(exc)}]}
        try:
            respond({"jsonrpc":"2.0","id":ident,"result":result})
        finally:
            if stop is not None:
                with waits_lock:
                    waits.pop(ident, None)

    try:
        for line in sys.stdin:
            ident = None
            try:
                msg = json.loads(line)
                ident = msg.get("id")
                method = msg.get("method")
                if method == "notifications/cancelled":
                    with waits_lock:
                        waiting = waits.get(msg.get("params",{}).get("requestId"))
                        if waiting:
                            waiting[0].set()
                    continue
                if ident is None:
                    continue
                if method == "initialize":
                    requested = msg.get("params",{}).get("protocolVersion")
                    versions = ("2024-11-05","2025-03-26","2025-06-18","2025-11-25")
                    result = {"protocolVersion":requested if requested in versions else versions[-1],
                              "capabilities":{"tools":{}},"serverInfo":{"name":"classpage-studio","version":"1.1"}}
                elif method == "ping":
                    result = {}
                elif method == "tools/list":
                    result = {"tools":[{"name":"studio_"+name,"description":desc,"inputSchema":schema}
                                       for name,(desc,schema) in TOOLS.items()]}
                elif method == "tools/call":
                    name = msg["params"]["name"].removeprefix("studio_")
                    args = msg["params"].get("arguments",{})
                    if name == "session_receive" and args.get("timeout",20) != 0:
                        with waits_lock:
                            if len(waits)>=16 or ident in waits:
                                raise ValueError("此传输已有等待请求，请稍后再试")
                            cancel = threading.Event()
                            thread = threading.Thread(target=run_tool,args=(ident,name,args,cancel),daemon=True)
                            waits[ident] = (cancel,thread)
                            thread.start()
                    else:
                        run_tool(ident,name,args)
                    continue
                else:
                    raise ValueError("Unknown method")
                respond({"jsonrpc":"2.0","id":ident,"result":result})
            except Exception as exc:
                respond({"jsonrpc":"2.0","id":ident,"error":{"code":-32600,"message":str(exc)}})
    finally:
        with waits_lock:
            pending=list(waits.values())
            for cancel,_ in pending:
                cancel.set()
        for _,thread in pending:
            thread.join(timeout=2)
