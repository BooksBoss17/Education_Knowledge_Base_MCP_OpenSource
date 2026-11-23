"""BiliDown's aria2 transfer and FFmpeg stream-copy pipeline without Electron IPC."""
import json
from pathlib import Path
import re
import subprocess
import time
import tempfile
from contextlib import ExitStack
from urllib.error import URLError
from urllib.request import Request
from .direct_network import child_environment, direct_opener, native_source

from .backend import BIN_ROOT,digest,safe,folder,update,encoded
from .site_api import Redirects,AGENT,Failure,require,host_of,classify

def fetch(url,ref,target,identity,check_cancel,progress):
    offset=target.stat().st_size if identity and target.exists() else 0
    headers={"User-Agent":AGENT,"Referer":ref,"Accept-Encoding":"identity","Range":"bytes=0-"}
    if offset:
        headers.update({"Range":f"bytes={offset}-","If-Range":identity["etag"]})
    try:
        with direct_opener(Redirects({host_of(url)})).open(Request(url,headers=headers),timeout=20) as res:
            if res.status==206:
                match=re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",res.headers.get("Content-Range",""))
                require(match and int(match[1])==offset and int(match[2])+1==int(match[3]),"MEDIA_RANGE","媒体分段起止位置无效。")
                require(not offset or res.headers.get("ETag")==identity["etag"],"MEDIA_CHANGED","远端媒体版本发生变化。")
                expected=int(match[3])
            else:
                require(res.status==200,"MEDIA_RESPONSE","媒体服务器响应异常。")
                offset=0
                expected=int(res.headers.get("Content-Length") or 0)
            next_update=0
            with target.open("ab" if offset else "wb") as f:
                while True:
                    if time.monotonic()>=next_update:
                        check_cancel();progress();next_update=time.monotonic()+.5
                    block=res.read(256*1024)
                    if not block:break
                    f.write(block)
            require(not expected or target.stat().st_size==expected,"INCOMPLETE_MEDIA","媒体传输不完整。")
    except (URLError,TimeoutError) as ex:raise classify(ex) from None

def run_child(cmd,check_cancel,stdin=None,on_tick=None,capture=False):
    with ExitStack() as stack:
        log=stack.enter_context(tempfile.TemporaryFile()) if capture else None
        p=stack.enter_context(subprocess.Popen(cmd,env=child_environment(),stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL,stderr=log if capture else subprocess.DEVNULL,
                          creationflags=subprocess.CREATE_NO_WINDOW))
        try:
            if stdin is not None:
                p.stdin.write(stdin);p.stdin.close()
            while p.poll() is None:
                check_cancel()
                if on_tick:on_tick()
                time.sleep(.35)
            if log:log.seek(0)
            return p.returncode,log.read(65536).decode("utf8","replace") if capture else ""
        finally:
            if p.poll() is None:p.kill()
            p.wait()

def remote_identity(url,ref):
    try:
        req=Request(url,method="HEAD",headers={"Referer":ref,"User-Agent":AGENT})
        with direct_opener(Redirects({host_of(url)})).open(req,timeout=8) as res:
            size=int(res.headers.get("Content-Length","0"))
            etag=res.headers.get("ETag","")
            return {"size":size,"etag":etag} if size and etag.startswith('"') else None
    except Exception:return None

def attachments(staging,video,meta,ff,check_cancel):
    audio=staging/"audio.m4a"
    code,_=run_child([ff,"-nostdin","-loglevel","error","-i",str(video),"-map","0:a:0","-vn","-c:a","copy",str(audio)],check_cancel)
    require(code==0 and audio.stat().st_size>0,"AUDIO_FAILED","独立音轨保存失败。")
    paths=[audio]
    if meta.get("cover"):
        url=re.sub(r"^http:","https:",meta["cover"])
        host=host_of(url)
        require(host.endswith(".hdslb.com"),"UNSUPPORTED_COVER","站点返回了未知封面来源。")
        try:
            with direct_opener(Redirects({host})).open(Request(url,headers={"Referer":meta["source"],"User-Agent":AGENT}),timeout=15) as res:
                data=res.read(16*1024*1024+1)
        except (URLError,TimeoutError) as e:raise classify(e) from None
        ext="jpg" if data.startswith(b"\xff\xd8\xff") else "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "webp" if data[:4]==b"RIFF" and data[8:12]==b"WEBP" else None
        require(ext and len(data)<=16*1024*1024,"BAD_COVER","封面格式或大小异常。")
        cover=staging/("cover."+ext);cover.write_bytes(data);paths.append(cover)
    return [dict(name=p.name,size=p.stat().st_size,sha256=digest(p)) for p in paths]

def transfer(ws,jid,attempt,spec,meta,tracks,fmt,check_cancel):
    manifest = BIN_ROOT / "manifest.json"
    require(manifest.is_file(), "VIDEO_BIN_SETUP_REQUIRED", "Run scripts/setup_video.py --ffmpeg ... --aria2 ...")
    hashes = json.loads(manifest.read_text("utf8"))
    for name in ("aria2c", "ffmpeg"):
        expected = hashes.get(name)
        path=BIN_ROOT/(name+".exe")
        require(path.is_file() and digest(path)==expected,"DEPENDENCY_CHANGED",f"{name} 文件缺失或指纹已变化。")
    temp=safe(ws,"tmp/bilidown/parts/"+jid);temp.mkdir(parents=True,exist_ok=True)
    dest=folder(ws,jid);dest.parent.mkdir(parents=True,exist_ok=True)
    require(not dest.exists(),"DESTINATION_EXISTS","成品目录已存在，请核对文件。")
    completed=0;files={};transports={}
    for role,url in tracks.items():
        check_cancel();update(ws,jid,phase="download_"+role)
        transports[role]="cached"
        identity=remote_identity(url,meta["source"])
        # A stable strong validator is necessary for cross-attempt byte reuse.
        stamp=encoded([meta["selected"]["cid"],role,fmt,identity,0 if identity else attempt])
        import hashlib
        name=role+"-"+hashlib.sha256(stamp.encode()).hexdigest()[:20]+".part"
        target=safe(temp,name);control=safe(temp,name+".aria2")
        total=identity["size"] if identity else None
        if not (total and target.is_file() and target.stat().st_size==total and not control.exists()):
            def progress():
                size=target.stat().st_size if target.exists() else 0
                update(ws,jid,bytes=completed+size)
            cmd=[BIN_ROOT/"aria2c.exe","--no-conf=true","--enable-rpc=false","--quiet=true","--no-netrc=true",
                 "--auto-file-renaming=false","--allow-overwrite=true","--continue=true","--file-allocation=none",
                 "--split=1","--max-connection-per-server=1","--max-tries=2","--retry-wait=2","--connect-timeout=12","--timeout=30",
                 "--check-certificate=true","--referer="+meta["source"],"--user-agent="+AGENT,
                 "--dir="+str(temp),"--out="+name,"--input-file=-"]
            if spec.get("transport","https")=="aria2":
                try:
                    source = native_source(host_of(url))
                except OSError as error:
                    raise Failure("DIRECT_ROUTE_REQUIRED", "无法确认媒体的物理网卡出口。") from error
                cmd.extend(["--all-proxy=", "--http-proxy=", "--https-proxy=", "--disable-ipv6=true",
                            "--interface="+source])
                transports[role]="aria2"
                code,_=run_child(cmd,check_cancel,(url+"\n  out="+name+"\n").encode(),progress)
                if code in {2,5,6,7}:
                    check_cancel()
                    # aria2 may have non-contiguous bytes. Restart this track from zero.
                    control.unlink(missing_ok=True)
                    target.unlink(missing_ok=True)
                    transports[role]="https_after_aria2"
                    fetch(url,meta["source"],target,None,check_cancel,progress)
                else:
                    require(code==0,"MEDIA_TRANSFER_"+str(code),"媒体传输失败。")
            else:
                transports[role]="https"
                fetch(url,meta["source"],target,identity,check_cancel,progress)
        size=target.stat().st_size
        require(size>0 and (not total or size==total),"INCOMPLETE_MEDIA","媒体流长度不匹配。")
        files[role]=target;completed+=size
    update(ws,jid,phase="merge",bytes=completed)
    staging=safe(dest.parent,".pending-"+jid+"-"+str(attempt));staging.mkdir()
    output=staging/"video.mp4"; ff=BIN_ROOT/"ffmpeg.exe"
    cmd=[ff,"-nostdin","-hide_banner","-loglevel","error"]
    for role in (("video","audio") if "video" in files else ("combined",)):
        cmd+=["-protocol_whitelist","file,pipe","-i",str(files[role])]
    audio="1:a:0" if "audio" in files else "0:a:0"
    cmd+=["-map","0:v:0","-map",audio,"-map_metadata","-1","-c","copy","-movflags","+faststart",str(output)]
    try:
        code,_=run_child(cmd,check_cancel);require(code==0,"MERGE_FAILED","音视频合并失败。")
        update(ws,jid,phase="verify")
        cmd=[ff,"-nostdin","-hide_banner","-nostats","-protocol_whitelist","file,pipe","-i",str(output),"-map","0:v:0","-map","0:a:0","-t","0.1","-f","null","-"]
        code,details=run_child(cmd,check_cancel,capture=True)
        duration=re.search(r"Duration: (\d+):(\d+):([\d.]+)",details)
        require(code==0 and duration and "Video:" in details and "Audio:" in details,"VERIFICATION_FAILED","文件音视频轨校验失败。")
        seconds=int(duration[1])*3600+int(duration[2])*60+float(duration[3])
        require(seconds>0 and abs(seconds-meta["selected"]["duration"])<=max(3,.015*seconds),"INCOMPLETE_MEDIA","成品时长与分 P 时长不符。")
        receipt=dict(schema="bilidown-output-v1",job_id=jid,bvid=meta["bvid"],title=meta["title"],uploader=meta["uploader"],
                     source=meta["source"],part=meta["selected"],duration=seconds,selection=fmt,transports=transports,auth=spec["auth"],
                     size=output.stat().st_size,sha256=digest(output),video=str(dest/"video.mp4"),manifest=str(dest/"source.json"))
        if spec.get("extras"):
            receipt["attachments"]=attachments(staging,output,meta,ff,check_cancel)
        (staging/"source.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf8")
        check_cancel();staging.rename(dest)
    finally:
        if staging.exists():
            for name in ["video.mp4","source.json","audio.m4a","cover.jpg","cover.png","cover.webp"]:(staging/name).unlink(missing_ok=True)
            staging.rmdir()
    # A changed CDN validator can leave a superseded partial from an earlier attempt.
    for path in temp.iterdir():
        if re.fullmatch(r"(audio|video|combined)-[a-f0-9]{20}\.part(?:\.aria2)?",path.name):
            safe(temp,path.name).unlink(missing_ok=True)
    return receipt

