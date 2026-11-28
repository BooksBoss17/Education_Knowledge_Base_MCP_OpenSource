"""Resolve embedded, browser-playable media using presentation order and OOXML relationships."""

import posixpath
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
MIME = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
}


def relations(archive, part):
    parent, name = posixpath.split(part)
    try:
        xml = archive.read(posixpath.join(parent, "_rels", name + ".rels"))
    except KeyError:
        return {}
    out = {}
    for rel in ET.fromstring(xml):
        if rel.get("TargetMode") == "External":
            continue
        value = rel.get("Target", "")
        path = posixpath.normpath(
            value.lstrip("/")
            if value.startswith("/")
            else posixpath.join(parent, value)
        )
        if not path.startswith("../") and ":" not in path:
            out[rel.get("Id")] = path
    return out


def attach(store, deck, source):
    warnings = set()
    with ZipFile(source) as z:
        doc = "ppt/presentation.xml"
        links = relations(z, doc)
        root = ET.fromstring(z.read(doc))
        for i, page_ref in enumerate(root.findall("./{*}sldIdLst/{*}sldId")):
            if i >= len(deck["order"]):
                break
            part = links[page_ref.get("{" + REL + "}id")]
            local = relations(z, part)
            slide = deck["slides"][deck["order"][i]]
            shapes = ET.fromstring(z.read(part)).find(".//{*}spTree")
            if shapes is None:
                continue
            for shape in shapes:
                meta = shape.find("./*/{*}cNvPr")
                if meta is None:
                    continue
                media = [
                    e
                    for e in shape.iter()
                    if e.tag.rsplit("}", 1)[-1] in {"videoFile", "audioFile", "media"}
                ]
                if not media:
                    continue
                paths = [
                    local.get(e.get("{" + REL + "}embed") or e.get("{" + REL + "}link"))
                    for e in media
                ]
                file = next(
                    (p for p in paths if p and PurePosixPath(p).suffix.lower() in MIME),
                    None,
                )
                target = slide["elements"].get(f"s{i + 1}-" + meta.get("id", ""))
                if not file or not target or z.getinfo(file).file_size > 256 * 1024**2:
                    warnings.add(f"第 {i + 1} 页的媒体以封面保留，未嵌入播放文件。")
                    continue
                mime = MIME[PurePosixPath(file).suffix.lower()]
                target["poster"] = target.get("asset", "")
                target["asset"] = store.asset(mime, z.read(file))
                target["type"] = mime.split("/")[0]
                target.pop("original", None)
    return warnings
