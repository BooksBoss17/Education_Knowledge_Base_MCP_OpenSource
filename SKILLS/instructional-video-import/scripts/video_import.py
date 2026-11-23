"""Archive speech evidence and publish one reviewed lesson per video."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text("utf-8-sig"))


def stamp(seconds):
    ms = round(seconds * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}"


def clean_title(value):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', value).strip(' .')[:64].rstrip(' .') or '教学视频'


def local(root, relative):
    path = root / relative
    if path.resolve() != path or not path.is_relative_to(root):
        raise ValueError("Unsafe workspace path")
    return path


def write_once(path, data=None, source=None, expected=None):
    expected = expected or hashlib.sha256(data).hexdigest()
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"Existing content differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        if source:
            shutil.copyfile(source, tmp)
        else:
            tmp.write_bytes(data)
        if digest(tmp) != expected:
            raise ValueError("Copied content changed")
        try:
            os.link(tmp, path)
        except FileExistsError:
            if digest(path) != expected:
                raise ValueError("Concurrent content conflict")
    finally:
        tmp.unlink(missing_ok=True)


def prepare(root, args):
    source = args.source.resolve(strict=True)
    if not re.fullmatch(r"[a-f0-9]{32}", args.job_id):
        raise ValueError("Invalid speech job ID")
    out = local(root, f"Output/speech_transcribe/{args.job_id}")
    summary = read(out / "summary.json")
    sha = digest(source)
    if summary["source_sha256"] != sha or not summary["timestamps_requested"]:
        raise ValueError("Speech result requires matching source hash and timestamps")
    title = clean_title(args.title)
    key = sha[:24]
    category = local(root, "Original_Backup/INSTRUCTIONAL_VIDEOS")
    receipt = {}
    if source.is_relative_to(category) and source.name == "video.mp4":
        origin = local(root,(source.parent/"source.json").relative_to(root))
        if origin.is_file():
            candidate = read(origin)
            if candidate.get("schema")=="bilidown-output-v1" and candidate.get("sha256")==sha:
                receipt = candidate
    if receipt:
        video = local(root, source.relative_to(root))
    else:
        video = local(root, category.relative_to(root) / f"{title}--{key}" / ("video" + source.suffix.lower()))
    video.parent.mkdir(parents=True, exist_ok=True)
    if video != source:
        write_once(video, source=source, expected=sha)
    archive = video.parent
    url = receipt.get("source","")
    if not receipt:
        provenance = local(root,(archive/"source.json").relative_to(root))
        if video == source and provenance.is_file():
            original_name = read(provenance)["original_name"]
        else:
            original_name = source.name
        local_source = dict(schema="local-instructional-video-v1", original_name=original_name, sha256=sha)
        write_once(provenance,(json.dumps(local_source,ensure_ascii=False,indent=2)+"\n").encode())
    key = hashlib.sha256((archive.relative_to(root).as_posix()+"\0"+sha).encode()).hexdigest()[:24]
    lines = [f"# {title} · 时间戳稿件", "", f"原视频 [{video.name}](<{quote(video.name)}>)", f"\nSHA-256 `{sha}`"]
    if url:
        lines += ["", f"[B 站原视频]({url})"]
    count = 0
    duration = summary["audio_seconds"]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid audio duration")
    last_end = 0
    with (out / "segments.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["index"] != count:
                raise ValueError("Missing or unordered speech segment")
            if not (last_end <= row["start"] <= row["end"] <= duration + .001):
                raise ValueError("Invalid speech segment times")
            last_end = row["end"]
            count += 1
            lines += ["", f"## {stamp(row['start'])} – {stamp(row['end'])}", "", row["text"] or "（无转写语音）"]
            marks = row.get("timestamps", [])
            if marks:
                lines += ["", "<details><summary>字词对齐时间</summary>", ""]
                for mark in marks:
                    lines.append(f"- {stamp(mark['start'])} – {stamp(mark['end'])}　{mark['text']}")
                lines += ["", "</details>"]
    if count != summary["segment_count"] or not count:
        raise ValueError("Incomplete speech package")
    transcript = local(root,(archive/"transcript.md").relative_to(root))
    write_once(transcript,("\n".join(lines)+"\n").encode())
    draft = local(root, f"tmp/instructional-video-import/{key}")
    draft.mkdir(parents=True, exist_ok=True)
    meta = dict(title=title,video=video.relative_to(root).as_posix(),transcript=transcript.relative_to(root).as_posix(),
                sha256=sha,duration=summary["audio_seconds"],transcript_sha256=digest(transcript))
    write_once(draft/"source.json",(json.dumps(meta,ensure_ascii=False,indent=2)+"\n").encode())
    return dict(package_id=key,archive=str(archive),draft=str(draft),**meta)


def load_draft(root,key):
    if not re.fullmatch(r"[a-f0-9]{24}",key):
        raise ValueError("Invalid package ID")
    draft = local(root,f"tmp/instructional-video-import/{key}")
    meta = read(local(root,(draft/"source.json").relative_to(root)))
    video = local(root,meta["video"])
    if digest(video)!=meta["sha256"] or digest(local(root,meta["transcript"]))!=meta["transcript_sha256"]:
        raise ValueError("Archived evidence has changed")
    return draft,meta,video


def frames(root,args):
    draft,meta,video=load_draft(root,args.id)
    result=[]
    for seconds in args.seconds:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Frame time out of range")
        ms=round(seconds*1000)
        dest=local(root,(draft/f"frame-{ms:012}.jpg").relative_to(root))
        with tempfile.TemporaryDirectory(dir=draft) as tmp:
            image=Path(tmp)/"frame.jpg"
            subprocess.run([args.ffmpeg,"-nostdin","-v","error","-ss",str(ms/1000),"-i",str(video),
                            "-map","0:v:0","-frames:v","1","-q:v","2",str(image)],check=True)
            if not image.is_file() or not image.stat().st_size:
                raise ValueError("No video frame at requested time")
            write_once(dest,source=image,expected=digest(image))
        result.append(dict(file=dest.name,path=str(dest),seconds=ms/1000))
    return result


def publish(root,args):
    draft,meta,video=load_draft(root,args.id)
    base=local(root,"KNOWLEDGE_BASE/INSTRUCTIONAL_VIDEOS")
    dest=local(root,base.relative_to(root)/f"{meta['title']}--{args.id}")
    base.mkdir(parents=True,exist_ok=True)
    def link(path):
        return quote(os.path.relpath(path,dest).replace(os.sep,'/'),safe='/')
    lines=[f"# {meta['title']}","",f"[原视频]({link(video)}) · [时间戳稿件]({link(root/meta['transcript'])})"]
    sections=read(args.plan)["sections"]
    used={};prev=-1
    if not sections:raise ValueError("Missing reviewed sections")
    for s in sections:
        sec=s["start"]
        if not (math.isfinite(sec) and sec>=0 and sec>=prev and s["heading"].strip() and s["body"].strip()):
            raise ValueError("Invalid lesson section")
        prev=sec
        lines+=["",f"## {stamp(sec)}　{s['heading']}","",s["body"].strip()]
        covered = sec < meta["duration"]
        for pic in s["frames"]:
            name=pic["file"];caption=pic["caption"].strip()
            if not re.fullmatch(r"frame-\d{12}\.jpg",name) or not caption:raise ValueError("Invalid frame reference")
            used[name]=local(root,(draft/name).relative_to(root))
            covered = covered or sec <= int(name[6:-4])/1000
            lines+=["",f"![关键帧](frames/{name})",f"\n{stamp(int(name[6:-4])/1000)}　{caption}"]
        if not covered:raise ValueError("Section beyond audio requires a confirming video frame")
    if not used:raise ValueError("At least one reviewed teaching frame required")
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        package=Path(tmp)/"package";(package/"frames").mkdir(parents=True)
        for name,path in used.items():shutil.copyfile(path,package/"frames"/name)
        (package/"lesson.md").write_bytes(("\n".join(lines)+"\n").encode())
        for attempt in range(4):
            try:
                package.rename(dest)
                break
            except OSError as error:
                if dest.is_dir():
                    def contents(folder):
                        return {f.relative_to(folder): digest(f) for f in folder.rglob("*") if f.is_file()}
                    if contents(dest) != contents(package):
                        raise ValueError(f"Published lesson differs: {dest}") from error
                    break
                if getattr(error, "winerror", None) not in (5, 32) or attempt == 3:
                    raise
                time.sleep(.1 * (attempt + 1))
    return dict(lesson=str(dest/"lesson.md"),frames=len(used),archive=str(video.parent))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    for cmd in ("prepare","frames","publish"):
        c=sub.add_parser(cmd);c.add_argument("--workspace",type=Path,required=True)
        if cmd=="prepare":
            c.add_argument("--source",type=Path,required=True);c.add_argument("--job-id",required=True);c.add_argument("--title",required=True)
        else:
            c.add_argument("--id",required=True)
            if cmd=="frames":
                c.add_argument("--seconds",type=float,nargs="+",required=True);c.add_argument("--ffmpeg",default="ffmpeg")
            else:c.add_argument("--plan",type=Path,required=True)
    args=p.parse_args()
    result={"prepare":prepare,"frames":frames,"publish":publish}[args.command](args.workspace.resolve(strict=True),args)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
