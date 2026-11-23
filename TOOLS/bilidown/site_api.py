"""Headless adapter for the view, DASH and QR routes used by BiliDown 1.1.1."""
import http.cookiejar
import json
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPCookieProcessor, Request
from .direct_network import DirectUnavailable, direct_opener

API = "https://api.bilibili.com"
PASSPORT = "https://passport.bilibili.com"
WEB = "https://www.bilibili.com"
AGENT = "BiliDown/1.1.1 (Education MCP)"
BV = re.compile(r"BV[0-9A-Za-z]{10}")
PAGES = {"www.bilibili.com", "bilibili.com", "m.bilibili.com"}
QR_HOSTS = {"passport.bilibili.com", "account.bilibili.com"}
QUALITIES = {6: 240, 16: 360, 32: 480, 64: 720, 74: 720, 80: 1080, 112: 1080, 116: 1080, 120: 2160, 125: 2160, 126: 2160, 127: 4320}

class Failure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

def require(condition, code, message):
    if not condition:
        raise Failure(code, message)

def host_of(url):
    try:
        p = urlsplit(url)
        port=p.port
    except ValueError:
        raise Failure("INVALID_URL","网址格式无效。") from None
    require(p.scheme=="https" and p.username is None and p.password is None and port in (None,443),
            "INVALID_URL","仅接受 HTTPS 官方链接。")
    return (p.hostname or "").lower()

class Redirects(HTTPRedirectHandler):
    def __init__(self, allowed):
        self.allowed = allowed

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(host_of(urljoin(req.full_url,newurl)) in self.allowed,
                "UNEXPECTED_REDIRECT","站点返回了未允许的重定向。")
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def classify(error):
    reason = getattr(error, 'reason', error)
    if isinstance(reason, DirectUnavailable):
        return Failure('DIRECT_ROUTE_REQUIRED', str(reason))
    if isinstance(error,HTTPError):
        code = error.code
        if code in (412,429): return Failure("SITE_LIMITED",f"站点限制了请求（HTTP {code}）。")
        if code in (401,403): return Failure("ACCESS_DENIED",f"站点拒绝访问（HTTP {code}）。")
        if code==404: return Failure("NOT_FOUND","视频或接口不存在。")
        return Failure(f"HTTP_{code}",f"站点返回 HTTP {code}。")
    return Failure("NETWORK_ERROR","连接站点失败或超时。")

def identify(value, part=None):
    require(isinstance(value,str) and len(value)<=2048,"INVALID_INPUT","请输入 BV 号或 B 站视频链接。")
    value=value.strip()
    if not BV.fullmatch(value):
        host=host_of(value)
        if host=="b23.tv":
            require(re.fullmatch(r"/[A-Za-z0-9_-]{2,32}/?",urlsplit(value).path),"INVALID_URL","短链接格式无效。")
            opener=direct_opener(Redirects(PAGES|{"b23.tv"}))
            try:
                with opener.open(Request(value,headers={"User-Agent":AGENT}),timeout=12) as res:
                    value=res.url
            except (URLError,TimeoutError) as ex: raise classify(ex) from None
        p=urlsplit(value)
        require(host_of(value) in PAGES,"INVALID_URL","只支持 B 站视频页面。")
        m=re.fullmatch(r"/video/(BV[0-9A-Za-z]{10})/?",p.path)
        require(m,"INVALID_URL","链接没有有效的 BV 号。")
        parts=parse_qs(p.query,keep_blank_values=True).get("p",["1"])
        require(len(parts)==1 and re.fullmatch(r"[1-9]\d{0,4}",parts[0]),"INVALID_PART","分 P 参数无效。")
        if part is None: part=int(parts[0])
        value=m[1]
    if part is None: part=1
    require(type(part) is int and 1<=part<=99999,"INVALID_PART","分 P 必须是正整数。")
    return value,part

def api(base,path,params=None,cookie=None,jar=None):
    host=host_of(base)
    require(host in {"api.bilibili.com","passport.bilibili.com"},"INVALID_URL","接口域名无效。")
    headers={"User-Agent":AGENT,"Referer":WEB+"/","Accept":"application/json"}
    if cookie and base==API: headers["Cookie"]="SESSDATA="+cookie
    handlers=[Redirects({host})]
    if jar is not None: handlers.append(HTTPCookieProcessor(jar))
    req=Request(base+path+("?"+urlencode(params) if params else ""),headers=headers)
    try:
        with direct_opener(*handlers).open(req,timeout=15) as res:
            raw=res.read(8*1024*1024+1)
        require(len(raw)<=8*1024*1024,"SITE_RESPONSE","接口响应过大。")
        message=json.loads(raw)
    except (URLError,TimeoutError,socket.timeout) as ex:raise classify(ex) from None
    except (ValueError,UnicodeError):raise Failure("SITE_RESPONSE","接口没有返回可识别的数据。") from None
    require(isinstance(message,dict),"SITE_RESPONSE","接口结构异常。")
    code=message.get("code")
    if code!=0:
        label={-101:"LOGIN_REQUIRED",-10403:"ACCESS_DENIED",-403:"ACCESS_DENIED",-404:"NOT_FOUND",-412:"SITE_LIMITED",-352:"SITE_LIMITED",-509:"SITE_LIMITED"}.get(code,"SITE_ERROR")
        raise Failure(label,f"B 站接口返回 {code}。")
    return message.get("data") or {}

def check_account(cookie):
    require(cookie and api(API,"/x/web-interface/nav",cookie=cookie).get("isLogin"),
            "LOGIN_REQUIRED","登录已失效；可打开 BiliDown 登录。")

def metadata(bv,part,cookie=None):
    d=api(API,"/x/web-interface/view",{"bvid":bv},cookie)
    require(d.get("bvid")==bv,"SITE_RESPONSE","视频身份无法核对。")
    pages=[dict(part=int(x["page"]),cid=int(x["cid"]),title=x["part"],duration=x["duration"]) for x in d.get("pages",[])]
    selected=next((x for x in pages if x["part"]==part),None)
    require(selected,"PART_NOT_FOUND","所选分 P 不存在。")
    return dict(bvid=bv,title=d["title"],uploader=d.get("owner",{}).get("name",""),cover=d.get("pic",""),
                source=f"{WEB}/video/{bv}/?p={part}",pages=pages,selected=selected)

def validated_media_url(raw):
    url="https://"+raw[7:] if raw.startswith("http://") else raw
    h=host_of(url)
    require(h.endswith((".bilivideo.com",".bilivideo.cn")) or h=="upos-hz-mirrorakam.akamaized.net",
            "UNSUPPORTED_CDN","播放接口返回了尚未验证的媒体域名。")
    require(not any(c.isspace() for c in url),"SITE_RESPONSE","媒体地址格式无效。")
    return url

def media_url(track):
    """Only use addresses supplied for the exact selected representation."""
    primary = track.get("baseUrl") or track.get("base_url") or track.get("url")
    backups = track.get("backupUrl") or track.get("backup_url") or []
    for candidate in [primary, *(backups if isinstance(backups,list) else [])]:
        if not isinstance(candidate,str) or not candidate:
            continue
        try:
            return validated_media_url(candidate)
        except Failure:
            continue
    raise Failure("NO_ALLOWED_MEDIA_URL","该音视频轨没有符合直连规则的主地址或备用地址。")


def streams(meta,height,cookie=None,*,best=False,audio_qn=192):
    qn={360:16,480:32,720:64,1080:80,2160:120}[height]
    if best:qn=127
    d=api(API,"/x/player/playurl",dict(bvid=meta["bvid"],cid=meta["selected"]["cid"],qn=qn,fnval=4048,fnver=0,fourk=1),cookie)
    dash=d.get("dash") or {};videos=dash.get("video") or []
    if videos:
        available=sorted({QUALITIES.get(v["id"],v.get("height",0)) for v in videos})
        choices=[v for v in videos if 0<v["id"]<=qn]
        require(choices,"QUALITY_UNAVAILABLE","当前返回的画质均超过设定上限。")
        video=max(choices,key=lambda v:(v["id"],v.get("codecid")==7,v.get("bandwidth",0)))
        audio=dash.get("audio") or []
        require(audio,"NO_AUDIO","未取得可合并的音轨。")
        tiers={30216:64,30232:132,30280:192}
        candidates=[v for v in audio if tiers.get(v.get("id"),0)<=audio_qn]
        require(candidates,"NO_AUDIO","没有符合音质设置的普通音轨。")
        sound=max(candidates,key=lambda v:(tiers.get(v.get("id"),0),v.get("codecs","").startswith("mp4a"),v.get("bandwidth",0)))
        return dict(audio=media_url(sound),video=media_url(video)),dict(height=QUALITIES.get(video["id"],video.get("height")),video_qn=video["id"],audio_qn=tiers.get(sound.get("id")),audio_bandwidth=sound.get("bandwidth"),width=video.get("width"),pixel_height=video.get("height"),available_heights=available,video_codec=video.get("codecs"),audio_codec=sound.get("codecs"),format="dash")
    direct=d.get("durl") or []
    require(len(direct)==1,"NO_PLAYABLE_STREAM","未取得支持的音视频流；请检查该视频的访问条件。")
    quality=QUALITIES.get(d.get("quality"))
    require(quality and (best or quality<=height),"QUALITY_UNAVAILABLE","未取得指定上限以内的视频。")
    return dict(combined=media_url(direct[0])),dict(height=quality,available_heights=[quality],format="progressive")

