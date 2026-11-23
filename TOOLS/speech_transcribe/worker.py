"""Managed local Qwen3-ASR worker."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def windows(source):
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
        for frame in frames():
            block=frame.to_ndarray().reshape(-1);blocks.append(block);size+=len(block)
            if size>=target:
                joined=np.concatenate(blocks)
                area=joined[target-5*rate:target].reshape(-1,320)
                cut=target-5*rate+(int(np.argmin(np.mean(area**2,axis=1)))+1)*320
                yield joined[:cut]
                blocks=[joined[cut:]];size=len(blocks[0])
        if size:yield np.concatenate(blocks)


def load_models(models, timestamps):
    import torch
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
    torch.set_num_threads(min(8,torch.get_num_threads()))
    def options(folder):
        size=sum(p.stat().st_size for p in folder.glob("*.safetensors"))
        device="cpu"
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>size*1.15+512*1024**2:
            device="cuda:0"
        return dict(device_map=device,dtype=torch.bfloat16 if device!="cpu" else torch.float32,local_files_only=True)
    asr=models/"Qwen3-ASR-1.7B";opts=options(asr)
    recognizer=Qwen3ASRModel.from_pretrained(str(asr),**opts,max_inference_batch_size=1,max_new_tokens=2048)
    aligner=None;devices={"asr":opts["device_map"]}
    if timestamps:
        folder=models/"Qwen3-ForcedAligner-0.6B";opts=options(folder)
        aligner=Qwen3ForcedAligner.from_pretrained(str(folder),**opts)
        devices["aligner"]=opts["device_map"]
    return recognizer,aligner,devices


def transcribe(models, source, payload, output):
    import itertools
    audio=windows(source)
    first=next(audio,None)
    if first is None:raise ValueError("EMPTY_AUDIO")
    asr,aligner,devices=load_models(models,payload["timestamps"])
    supported=set(aligner.get_supported_languages() or []) if aligner else set()
    samples=count=spans=unaligned=0
    with (output/"transcript.txt").open("x",encoding="utf8") as text, (output/"segments.jsonl").open("x",encoding="utf8") as rows:
        for block in itertools.chain([first],audio):
            begin=samples/16000;samples+=len(block);end=samples/16000
            clip=(block,16000)
            prediction=asr.transcribe(audio=clip,language=payload["language"])[0]
            row=dict(index=count,start=begin,end=end,language=prediction.language,text=prediction.text)
            if aligner:
                row["timestamps"]=[]
                if prediction.text.strip() and supported and prediction.language.lower() not in supported:
                    row["alignment"]="unsupported_language";unaligned+=1
                elif prediction.text.strip():
                    aligned=aligner.align(audio=clip,text=prediction.text,language=prediction.language)[0]
                    for token in aligned:
                        row["timestamps"].append(dict(text=token.text,start=round(begin+token.start_time,3),end=round(begin+token.end_time,3)))
                    if not row["timestamps"]:unaligned+=1
                    spans+=len(row["timestamps"])
            count+=1;text.write(prediction.text+"\n");rows.write(json.dumps(row,ensure_ascii=False)+"\n")
    return dict(segment_count=count,audio_seconds=samples/16000,timestamp_count=spans,unaligned_segments=unaligned,
                source_sha256=payload["source_sha256"],devices=devices,timestamps_requested=payload["timestamps"])


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
    summary=transcribe(root/"MODELS",source,payload,tmp)
    if helpers["sha"](source)!=payload["source_sha256"]:raise ValueError("Source changed during transcription")
    (tmp/"summary.json").write_text(json.dumps(summary,ensure_ascii=False),encoding="utf8")
    tmp.rename(output)


if __name__=="__main__":main()
