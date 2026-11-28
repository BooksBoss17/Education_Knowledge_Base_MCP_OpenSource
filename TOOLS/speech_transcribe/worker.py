"""Managed speech worker with configurable recognition and optional alignment."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speech_transcribe import providers, routing
from speech_transcribe.api_session import open_providers


def windows(source, timeline):
    """Decode the first audio track, cutting long inputs near a quiet boundary."""
    import av
    import numpy as np
    rate=16000
    target=120*rate
    with source.open("rb") as handle, av.open(handle,options={"protocol_whitelist":"pipe"}) as media:
        if not media.streams.audio:raise ValueError("NO_AUDIO_TRACK")
        stream=media.streams.audio[0]
        resampler=av.AudioResampler(format="fltp",layout="mono",rate=rate)
        def frames():
            for frame in media.decode(stream):
                yield from resampler.resample(frame)
            yield from resampler.resample(None)
        blocks=[];size=0
        consumed=0
        origin=media.start_time/av.time_base if media.start_time is not None else None
        for frame in frames():
            block=frame.to_ndarray().reshape(-1);blocks.append(block);size+=len(block)
            audio_time=consumed/rate
            media_time=float(frame.pts*frame.time_base)-origin if frame.pts is not None and frame.time_base and origin is not None else None
            last=timeline[-1] if timeline else None
            if last is None or (last["media_start"] is None)!=(media_time is None) or (
                media_time is not None and abs(media_time-last["media_start"]-(audio_time-last["audio_start"]))>1/rate
            ):
                timeline.append(dict(audio_start=audio_time,media_start=media_time))
            consumed+=len(block)
            if size>=target:
                joined=np.concatenate(blocks)
                area=joined[target-5*rate:target].reshape(-1,320)
                cut=target-5*rate+(int(np.argmin(np.mean(area**2,axis=1)))+1)*320
                yield joined[:cut]
                blocks=[joined[cut:]];size=len(blocks[0])
        if size:yield np.concatenate(blocks)


def transcribe(models, source, payload, output, route=None):
    import itertools
    timing_requested=payload["timestamps"]
    route=route or routing.resolve(models.parent,timing_requested)
    timeline=[]
    audio=windows(source,timeline)
    first=next(audio,None)
    if first is None:raise ValueError("EMPTY_AUDIO")
    with open_providers(route,models) as clients:
        return process_windows(itertools.chain([first],audio),timeline,clients,route,payload,output)


def process_windows(audio,timeline,clients,route,payload,output):
    timing_requested=payload["timestamps"]
    timing=route["timing"]
    asr=clients["asr"]
    aligner=clients.get("aligner")
    clock=route.get("aligner",route["asr"])
    granularity=clock.get("granularity","word_or_character") if timing!="none" else "none"
    samples=count=spans=unaligned=0
    with (output/"transcript.txt").open("x",encoding="utf8") as text, (output/"segments.jsonl").open("x",encoding="utf8") as rows:
        for block in audio:
            begin=samples/16000;samples+=len(block);end=samples/16000
            clip=(block,16000)
            pred=asr.recognize(clip,payload["language"],timing=="native")
            words=pred["text"];lang=pred.get("language") or payload["language"] or ""
            if not isinstance(words,str) or not isinstance(lang,str):raise ValueError("Invalid speech prediction")
            row=dict(index=count,start=begin,end=end,language=lang,text=words)
            if timing!="none":
                if timing=="native":
                    timed=pred
                elif words.strip():
                    timed=aligner.align(clip,lang,words)
                else:
                    timed={"timestamps":[]}
                marks=providers.intervals(timed.get("timestamps"),len(block)/16000,begin)
                row["timestamp_granularity"]=granularity
                row["timed_segments" if granularity=="segment" else "timestamps"]=marks
                if timed.get("alignment")=="unsupported_language":
                    row["alignment"]="unsupported_language"
                if words.strip() and not marks:
                    if clock["type"]=="http" and "alignment" not in row:
                        raise ValueError("Requested speech timing missing")
                    unaligned+=1
                spans+=len(marks)
            count+=1;text.write(words+"\n");rows.write(json.dumps(row,ensure_ascii=False)+"\n")
    devices={role:client.device for role,client in clients.items()}
    return dict(http_calls={role:client.calls for role,client in clients.items()},segment_count=count,audio_seconds=samples/16000,timestamp_count=spans,unaligned_segments=unaligned,
                source_sha256=payload["source_sha256"],devices=devices,timestamps_requested=timing_requested,
                timing_mode=timing,timestamp_granularity=granularity,time_base="decoded_audio_seconds",
                audio_to_media=timeline,route_fingerprint=routing.fingerprint(route),
                models={role:route[role]["model"] for role in devices})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manager-job-id",required=True);parser.add_argument("--manager-request",required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    helpers=runpy.run_path(str(root/"TOOLS/runtime_manager/bootstrap.py"))
    ctx=helpers["trusted_context"](root)
    req=Path(args.manager_request).resolve(strict=True)
    if not ctx or ctx["job_id"]!=args.manager_job_id or req!=(Path(ctx["home"])/"jobs"/args.manager_job_id/"request.json").resolve():
        raise ValueError("Unexpected managed request")
    job=json.loads(req.read_text("utf8"));payload=job["payload"]
    if job["job_id"]!=args.manager_job_id:raise ValueError("Manager job mismatch")
    manager=runpy.run_path(str(root/"TOOLS/education_mcp/workspace_manager.py"))["WorkspaceManager"](job["workspace"])
    source=Path(payload["source"]).resolve(strict=True)
    if helpers["sha"](source)!=payload["source_sha256"]:raise ValueError("Source changed")
    tmp=manager.safe_path("tmp/speech_transcribe/"+job["job_id"]);tmp.mkdir(parents=True,exist_ok=False)
    output=manager.safe_path("Output/speech_transcribe/"+job["job_id"]);output.parent.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    route=routing.resolve(root,payload["timestamps"])
    if routing.fingerprint(route)!=payload.get("route_fingerprint"):
        raise ValueError("Speech provider configuration changed after submission; submit again")
    summary=transcribe(root/"MODELS",source,payload,tmp,route)
    if helpers["sha"](source)!=payload["source_sha256"]:raise ValueError("Source changed during transcription")
    (tmp/"summary.json").write_text(json.dumps(summary,ensure_ascii=False),encoding="utf8")
    tmp.rename(output)


if __name__=="__main__":main()
