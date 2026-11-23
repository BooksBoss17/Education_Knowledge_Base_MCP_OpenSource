"""Per-Windows-user BiliDown login, separate from the original desktop profile."""
import base64
import ctypes
from ctypes import wintypes
import hashlib
import http.cookiejar
import io
import json
import os
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import re
import sys
import tempfile

from .site_api import PASSPORT, QR_HOSTS, Failure, api, host_of, require, check_account

def profile():
    return Path(os.environ["LOCALAPPDATA"])/"EKB"/"BiliDown"

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        tmp=Path(f.name)
        try:
            f.write(data);f.flush();os.fsync(f.fileno())
        except BaseException:
            f.close();tmp.unlink(missing_ok=True);raise
    try: tmp.replace(path)
    finally: tmp.unlink(missing_ok=True)

class Blob(ctypes.Structure):
    _fields_=[("size",wintypes.DWORD),("data",ctypes.POINTER(ctypes.c_ubyte))]

def protect(data,encrypt):
    raw=(ctypes.c_ubyte*len(data)).from_buffer_copy(data)
    src=Blob(len(data),raw);out=Blob()
    dll=ctypes.windll.crypt32
    func=dll.CryptProtectData if encrypt else dll.CryptUnprotectData
    if not func(ctypes.byref(src),None,None,None,None,1,ctypes.byref(out)):
        raise Failure("LOGIN_STORAGE","无法读取当前 Windows 用户保存的登录信息。")
    try: return ctypes.string_at(out.data,out.size)
    finally: ctypes.windll.kernel32.LocalFree(out.data)

def cookie():
    path=profile()/"login.dpapi"
    require(path.is_file(),"LOGIN_REQUIRED","请打开人工登录窗口并扫码。")
    value=protect(path.read_bytes(),False).decode("ascii")
    require(re.fullmatch(r"[!-~]+",value) and ";" not in value,"LOGIN_REQUIRED","本地登录信息无效。")
    return value

def qr_new():
    from pathlib import Path
    wheel=Path(__file__).resolve().parent/"vendor"/"segno-1.6.6-py3-none-any.whl"
    require(hashlib.sha256(wheel.read_bytes()).hexdigest()=="28c7d081ed0cf935e0411293a465efd4d500704072cdb039778a2ab8736190c7","DEPENDENCY_CHANGED","二维码组件指纹不匹配。")
    sys.path.insert(0,str(wheel))
    import segno
    data=api(PASSPORT,"/x/passport-login/web/qrcode/generate")
    require(host_of(data["url"]) in QR_HOSTS,"SITE_RESPONSE","扫码地址异常。")
    buf=io.BytesIO();segno.make(data["url"],micro=False).save(buf,kind="png",scale=5)
    return data["qrcode_key"],"data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")

def qr_poll(key):
    jar=http.cookiejar.CookieJar()
    data=api(PASSPORT,"/x/passport-login/web/qrcode/poll",{"qrcode_key":key},jar=jar)
    code=data.get("code")
    if code==0:
        value=next((c.value for c in jar if c.name=="SESSDATA" and c.domain.lstrip(".") in {"bilibili.com","passport.bilibili.com"}),None)
        if not value and data.get("url"):
            callback=urlsplit(data["url"])
            if host_of(data["url"]) in (QR_HOSTS|{"www.bilibili.com"}):
                value=parse_qs(callback.query).get("SESSDATA",[None])[0]
        require(value and re.fullmatch(r"[!-~]+",value) and ";" not in value,"LOGIN_INCOMPLETE","站点尚未返回完整登录凭据。")
        check_account(value)
        atomic(profile()/"login.dpapi",protect(value.encode("ascii"),True))
        return "SIGNED_IN"
    require(code in (86101,86090,86038),"LOGIN_FAILED","扫码登录失败。")
    return {86101:"WAIT_SCAN",86090:"WAIT_CONFIRM",86038:"EXPIRED"}[code]


