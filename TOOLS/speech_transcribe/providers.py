"""Speech adapters return window-relative intervals in seconds."""
import base64
import io
import ipaddress
import json
import math
import os
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import wave


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args):
        return None


class HTTPProvider:
    device = "api"

    def __init__(self, spec, token=None, timeout=120, lossless=False):
        self.spec = spec
        self.lossless = lossless
        self.token = token
        self.timeout = timeout
        self.calls = 0
        host = urlsplit(spec["endpoint"]).hostname
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        self.opener = build_opener(NoRedirect(), ProxyHandler({})) if loopback else build_opener(NoRedirect())

    def request(self, role, clip, language, timing, text=None):
        samples, rate = clip
        spec = self.spec
        body = dict(schema="speech-api-v1", role=role, model=spec["model"], revision=spec["revision"],
                    language=language, granularity=spec["granularity"] if timing else "none")
        if self.lossless:
            body["sample_rate"]=rate
            body["audio_f32le_base64"]=base64.b64encode(samples.astype("<f4",copy=False).tobytes()).decode("ascii")
        else:
            buffer = io.BytesIO()
            with wave.open(buffer,"wb") as wav:
                wav.setparams((1,2,rate,0,"NONE","not compressed"))
                wav.writeframes((samples.clip(-1,1)*32767).round().astype("<i2").tobytes())
            body["audio_wav_base64"]=base64.b64encode(buffer.getvalue()).decode("ascii")
        if text is not None:
            body["text"] = text
        headers = {"Content-Type":"application/json"}
        if self.token:
            headers["Authorization"]="Bearer "+self.token
        elif spec.get("token_env"):
            headers["Authorization"]="Bearer "+os.environ[spec["token_env"]]
        req = Request(spec["endpoint"],data=json.dumps(body).encode(),headers=headers)
        try:
            with self.opener.open(req,timeout=self.timeout) as response:
                raw = response.read(8*1024*1024+1)
        except HTTPError as e:
            raise RuntimeError(f"Speech {role} HTTP {e.code}") from None
        if len(raw)>8*1024*1024:
            raise ValueError("Oversized speech API response")
        value=json.loads(raw)
        if not isinstance(value,dict):
            raise ValueError("Invalid speech API response")
        if timing and value.get("granularity",spec["granularity"]) != spec["granularity"]:
            raise ValueError("Speech timestamp granularity differs from configured capability")
        self.calls += 1
        if self.token and value.get("device") in ("cuda:0","cpu"):
            self.device = value["device"]
        return value

    def recognize(self, clip, language, native):
        return self.request("transcribe",clip,language,native)

    def align(self, clip, language, text):
        return self.request("align",clip,language,True,text)


def intervals(value, length, offset):
    if not isinstance(value,list):
        raise ValueError("Expected speech timestamps array")
    result=[]
    previous=0
    for mark in value:
        text=mark["text"]
        start,end=mark["start"],mark["end"]
        if not isinstance(text,str) or not text.strip() or any(type(t) not in (int,float) or not math.isfinite(t) for t in (start,end)):
            raise ValueError("Invalid speech timestamp")
        if not (previous<=start<=end<=length+.001):
            raise ValueError("Speech timestamps overlap or exceed audio window")
        result.append(dict(text=text,start=round(offset+min(start,length),3),end=round(offset+min(end,length),3)))
        previous=end
    return result
