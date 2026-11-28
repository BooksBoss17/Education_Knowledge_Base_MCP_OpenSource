"""Read WPS native object exports, legacy knowledge packages and Studio HTML."""

import base64
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile

from deck import add_element, blank, check, element, slide, validate
from motion import NATIVE_EFFECT, NATIVE_TRANSITION, player_script

ROOT = Path(__file__).resolve().parent


class PackageReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_data = False
        self.text = []
        self.articles = []
        self.part = ""
        self.article = False

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "classpage-data":
            self.in_data = True
        if tag == "article":
            self.article = True
            self.articles.append({"h2": "", "pre": ""})
        elif self.article and tag in {"h2", "pre"}:
            self.part = tag

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False
        if tag == "article":
            self.article = False
        if tag == self.part:
            self.part = ""

    def handle_data(self, text):
        if self.in_data:
            self.text.append(text)
        if self.article and self.part:
            self.articles[-1][self.part] += text


def source_fingerprint(name, digests):
    identity = [Path(name).name.casefold(), *digests]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def read_html(store, html, name, images=None, *, reopen=False):
    p = PackageReader()
    p.feed(html)
    fingerprint = None
    if reopen:
        digests = [hashlib.sha256(html.encode("utf8")).hexdigest()]
        if not p.text and images:
            for n in range(1, len(p.articles) + 1):
                digests.append(hashlib.sha256(images[f"{n:03}.png"]).hexdigest())
        fingerprint = source_fingerprint(name, digests)
        existing = store.source_deck(fingerprint)
        if existing:
            return existing
    if p.text:
        pack = json.loads("".join(p.text))
        d = pack["deck"]
        validate(d)
        for key, url in pack["assets"].items():
            head, b64 = url.split(",", 1)
            check(head.startswith("data:") and head.endswith(";base64"), "素材编码无效")
            data = base64.b64decode(b64, validate=True)
            check(hashlib.sha256(data).hexdigest() == key, "素材校验失败")
            store.asset(head[5:-7], data)
        return store.create(d, fingerprint)
    check(p.articles and images, "旧版课件需提供包含 images 的完整目录")
    d = blank(Path(name).stem)
    d["slides"] = {}
    d["order"] = []
    for n, art in enumerate(p.articles, 1):
        page = slide(art["h2"])
        page["notes"] = art["pre"]
        png = images[f"{n:03}.png"]
        check(png[:8] == b"\x89PNG\r\n\x1a\n", "旧版页图损坏")
        w, h = (int.from_bytes(png[k : k + 4], "big") for k in (16, 20))
        d["height"] = round(d["width"] * h / w, 3)
        page["reference"] = store.asset("image/png", png)
        add_element(
            page,
            element(
                "image",
                name="原版画面",
                x=0,
                y=0,
                w=d["width"],
                h=d["height"],
                asset=page["reference"],
            ),
        )
        key = f"p{n}"
        d["slides"][key] = page
        d["order"].append(key)
    d["importReport"] = [
        "从已发布的静态页面恢复逐页画面和文字稿。要恢复对象与动画，请导入原始 PPT。"
    ]
    return store.create(d, fingerprint)


def from_layers(store, folder, title, source=None, *, fingerprint=None):
    raw = json.loads((folder / "deck.json").read_text("utf-8-sig"))
    d = blank(title)
    d.update(width=raw["width"], height=raw["height"], slides={}, order=[])
    notices = set()

    def asset(name):
        file = (folder / name).resolve(strict=True)
        check(file.is_relative_to(folder.resolve()), "无效素材路径")
        return store.asset("image/png", file.read_bytes())

    for s in raw["slides"]:
        n = s["page"]
        page = slide(f"第 {n} 页")
        page.update(
            reference=asset(s["reference"]),
            backdrop=asset(s["background"]),
            background="#FFFFFF",
            transition="none",
        )
        if s["transition"]:
            page["transition"] = NATIVE_TRANSITION.get(s["transition"], "fade")
            if page["transition"] == "fade":
                notices.add(f"第 {n} 页未登记的切换 {s['transition']} 按淡入放映。")
            page["transitionDuration"] = min(20, max(0, s.get("transitionDuration", 0.3)))
        notes = []
        for item in sorted(s["elements"], key=lambda o: o["z"]):
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", item["text"]).replace(
                "\r", "\n"
            )
            kind = "text" if text else "image"
            if item["type"] == 19:
                kind = "table"
            elif not text and item["type"] in (1, 9):
                kind = (
                    "line"
                    if item["type"] == 9
                    else {1: "rect", 9: "ellipse", 33: "arrow"}.get(
                        item.get("geometry"), "image"
                    )
                )
            if any(c in item["text"] for c in ("\x1a", "\x1b")):
                kind = "formula"
            if "asset" not in item and not text:
                notices.add(f"第 {n} 页未导出对象 {item['name']}，请对照原图。")
            o = element(
                kind,
                name=item["name"],
                text=text,
                x=item["x"],
                y=item["y"],
                w=item["w"],
                h=item["h"],
                size=item["fontSize"],
            )
            for k in ("font", "color", "bold"):
                if item.get(k):
                    o[k] = item[k]
            if item.get("asset"):
                o["asset"] = asset(item["asset"])
                # WPS exports line heads outside the zero-width geometry of a connector.
                # Restore their aspect ratio around the original line centre.
                if item.get("ink"):
                    o["x"], o["y"], o["w"], o["h"] = item["ink"]
                elif min(o["w"], o["h"]) <= 1:
                    with (folder / item["asset"]).open("rb") as image:
                        header = image.read(24)
                    pw, ph = struct.unpack(">II", header[16:24])
                    if o["w"] <= 1 and o["h"] > 1:
                        width = o["h"] * pw / ph
                        o["x"] -= (width - o["w"]) / 2
                        o["w"] = width
                    elif o["h"] <= 1 and o["w"] > 1:
                        height = o["w"] * ph / pw
                        o["y"] -= (height - o["h"]) / 2
                        o["h"] = height
                o["original"] = {
                    k: o[k]
                    for k in (
                        "text",
                        "font",
                        "size",
                        "color",
                        "fill",
                        "align",
                        "bold",
                        "type",
                    )
                }
            add_element(page, o, item["id"])
            if text:
                notes.append(text)
        page["notes"] = "\n".join(notes)
        page["title"] = next(
            (t.splitlines()[0][:56] for t in notes if t.strip()), page["title"]
        )
        for step in s["sequence"]:
            if step.get("target") not in page["elements"]:
                notices.add(f"第 {n} 页部分动画的目标不可用。")
                continue
            effect = NATIVE_EFFECT.get((step["effect"], bool(step["exit"])))
            if step["effect"] == 153 and page["elements"][step["target"]]["text"]:
                effect = "in-NumberReveal"
            path = step.get("path", "").strip()
            if path and not effect:
                effect = "custom-path"
            if not effect:
                notices.add(f"第 {n} 页效果 {step['effect']} 使用兼容放映。")
                effect = "exit" if step["exit"] else "appear"
            page["steps"].append(
                dict(
                    target=step["target"],
                    effect=effect,
                    **({"path": path} if path and (effect.startswith("path-") or effect == "custom-path") else {}),
                    **({"direction": step["direction"]} if step.get("direction") else {}),
                    **({"unit": {1:"char", 2:"word"}[step["textUnit"]]} if step.get("textUnit") in (1, 2) else {}),
                    trigger={2: "with", 3: "after"}.get(step["trigger"], "click"),
                    duration=max(0, min(20, step["duration"])),
                    delay=max(0, min(120, step["delay"])),
                )
            )
        key = f"p{n}"
        d["slides"][key] = page
        d["order"].append(key)
    if source and source.suffix.lower() == ".pptx":
        from pptx_media import attach

        notices.update(attach(store, d, source))
    notices.add("复杂组合和公式初始使用 WPS 对象外观；旧式 PPT 中的媒体以封面显示。")
    d["importReport"] = sorted(notices)
    return store.create(d, fingerprint)


def _wps_host():
    import winreg

    for view, sub in [
        (winreg.KEY_WOW64_32KEY, "SysWOW64"),
        (winreg.KEY_WOW64_64KEY, "System32"),
    ]:
        try:
            with winreg.OpenKey(
                winreg.HKEY_CLASSES_ROOT,
                r"KWPP.Application\CLSID",
                0,
                winreg.KEY_READ | view,
            ) as key:
                clsid = winreg.QueryValue(key, None)
            with winreg.OpenKey(
                winreg.HKEY_CLASSES_ROOT,
                "CLSID\\" + clsid + "\\LocalServer32",
                0,
                winreg.KEY_READ | view,
            ):
                host = (
                    Path(os.environ["WINDIR"])
                    / sub
                    / "WindowsPowerShell/v1.0/powershell.exe"
                )
                if host.exists():
                    break
        except FileNotFoundError:
            continue
    else:
        raise ValueError("本机未注册 WPS 演示接口")
    return host


def import_ppt(store, source, *, reopen=False):
    check(source.suffix.lower() in {".ppt", ".pptx"}, "请选择 PPT 或 PPTX")
    with tempfile.TemporaryDirectory(prefix="classpage-wps-") as tmp:
        folder = Path(tmp)
        copy = folder / ("source" + source.suffix)
        shutil.copyfile(source, copy)
        fingerprint = None
        if reopen:
            with copy.open("rb") as stream:
                fingerprint = source_fingerprint(source.name, [hashlib.file_digest(stream, "sha256").hexdigest()])
            existing = store.source_deck(fingerprint)
            if existing:
                return existing
        host = _wps_host()
        result = subprocess.run(
            [
                str(host),
                "-Sta",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "& ([ScriptBlock]::Create([IO.File]::ReadAllText($env:ST_SCRIPT))) -Source $env:ST_SRC -Output $env:ST_OUT",
            ],
            env={
                **os.environ,
                "ST_SRC": str(copy),
                "ST_OUT": tmp,
                "ST_SCRIPT": str(ROOT / "wps_layers.ps1"),
            },
            capture_output=True,
            timeout=1200,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        check(
            result.returncode == 0,
            "WPS 导出失败：" + result.stderr.decode(errors="replace")[-600:],
        )
        return from_layers(store, folder, source.stem, copy, fingerprint=fingerprint)


def offline(store, key):
    ui = ROOT / "ui"
    pack = store.portable(key)
    data = (
        json.dumps(pack, ensure_ascii=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>课件</title>"
        + "<style>"
        + (ui / "player.css").read_text("utf8")
        + "</style><main id='stage'></main><nav id='controls'></nav>"
        + "<script type='application/json' id='classpage-data'>"
        + data
        + "</script><script>"
        + player_script(pack["deck"])
        + "</script><script>const p=JSON.parse(document.querySelector('#classpage-data').textContent);"
        + "document.title=p.deck.title;Classpage.play(p.deck,id=>p.assets[id],stage,controls);</script></html>"
    )


def filename(title):
    result = (
        re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip().rstrip(". ")[:140]
        or "课件"
    )
    if result.split(".")[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        result = "_" + result
    return result + ".html"
