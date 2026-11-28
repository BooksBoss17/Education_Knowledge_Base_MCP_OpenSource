"""Local speech tool. Heavy inference runs in the shared manager queue."""
from . import routing
import hashlib
from itertools import islice
import json
from pathlib import Path
import re
import runpy

PROGRAM = "speech-transcribe"
MEDIA = {".wav",".mp3",".m4a",".flac",".ogg",".opus",".aac",".webm",".mkv",".mp4",".mov",".avi",".wma",".m4v"}
SCHEMA = {
    "name": "speech_transcribe",
    "description": "Transcribe local audio or video using configured speech providers with optional native or forced-alignment timestamps. start returns a managed job_id. Poll status, then read segments. Paths are limited to configured input roots. Times are seconds from the start of the audio track.",
    "inputSchema": {"type":"object","additionalProperties":False,"required":["action"],"properties":{
        "action":{"enum":["start","status","read"]},"source":{"type":"string"},
        "job_id":{"type":"string","pattern":"^[0-9a-f]{32}$"},"language":{"type":"string"},
        "timestamps":{"type":"boolean"},"offset":{"type":"integer","minimum":0},
        "limit":{"type":"integer","minimum":1,"maximum":4}}}}


def call(service, action, source=None, job_id=None, language=None, timestamps=None, offset=0, limit=1):
    root = service.mcp_root
    api = runpy.run_path(str(root/"TOOLS/education_mcp/managed_bridge.py"))
    if action == "start":
        local_runtime = root / ".local/runtimes/qwen3-speech-006/Scripts/python.exe"
        http_runtime = root / ".local/runtimes/speech-api-v1/Scripts/python.exe"
        local_ready = local_runtime.is_file() and all(
            (root/"MODELS"/name/"MODEL_MANIFEST.json").is_file()
            for name in ("Qwen3-ASR-1.7B","Qwen3-ForcedAligner-0.6B"))
        http_ready = http_runtime.is_file() and (root/".local/speech-api.json").is_file()
        if not (local_ready or http_ready):
            raise ValueError("Run scripts/setup_video.py with --speech-model-root or --speech-api-config")
        if not isinstance(source,str) or not source.strip() or (timestamps is not None and type(timestamps) is not bool):
            raise ValueError("Provide a local media file and a boolean timestamps flag")
        path = Path(source)
        path = (service.workspace/path).resolve(strict=True)
        if not any(path.is_relative_to(p) for p in service.allowed) or not path.is_file() or path.suffix.lower() not in MEDIA:
            raise ValueError("Expected audio or video in a configured input root")
        route=routing.resolve(root,timestamps)
        if language is not None and (not isinstance(language,str) or not language.strip()):
            raise ValueError("Provide a non-empty language or omit it")
        if language is not None and route["asr"]["type"]=="managed_http":
            langs=json.loads((root/"MODELS"/route["asr"]["model"]/"config.json").read_text("utf8"))["support_languages"]
            if not isinstance(language,str) or language.casefold() not in {x.casefold() for x in langs}:
                raise ValueError("Use a supported language name, for example Chinese, or omit for detection")
            language=next(x for x in langs if x.casefold()==language.casefold())
        payload = dict(source=str(path),timestamps=route["timing"]!="none",language=language,route_fingerprint=routing.fingerprint(route))
        manifest = root/"TOOLS/speech_transcribe/RELEASE_MANIFEST.json"
        job = api["manager"](service).submit(PROGRAM,service.workspace,payload,
            config_fingerprint=hashlib.sha256(manifest.read_bytes()).hexdigest(),run_timeout=86400,**api["task_binding"](service))
        return dict(job_id=job["job_id"],state=job["state"])
    if action not in ("read","status") or not re.fullmatch(r"[a-f0-9]{32}",job_id or ""):
        raise ValueError("Expected status or read with a valid job_id")
    client=api["bootstrap"](service)["existing_client"](root)
    job=client.status(job_id,workspace=service.workspace)
    if job["program_id"] != PROGRAM:raise ValueError("Not a speech job")
    result=dict(job_id=job_id,state=job["state"])
    if job["state"]!="SUCCEEDED":
        result["error"]=job.get("error")
        return result
    prefix="Output/speech_transcribe/"+job_id+"/"
    directory=service.workspace_manager.safe_path(prefix.rstrip("/"))
    def result_file(name):
        file=directory/name
        if file.resolve(strict=True)!=file or not file.is_file():
            raise ValueError("Invalid speech result file")
        return file
    result.update(json.loads(result_file("summary.json").read_text("utf8")))
    result["output_directory"]=str(directory)
    if action=="read":
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=4:raise ValueError("Invalid result page")
        with result_file("segments.jsonl").open(encoding="utf8") as f:
            result["segments"]=[json.loads(line) for line in islice(f,offset,offset+limit)]
        end=offset+len(result["segments"])
        result["next_offset"]=end if end<result["segment_count"] else None
    return result
