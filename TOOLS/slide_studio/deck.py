"""Validated, portable slide document and optimistic edit commands."""

from __future__ import annotations

import copy
import math
import re
import uuid
from motion import CATALOG

FORMAT = "classpage/1"
TYPES = {
    "text",
    "rect",
    "ellipse",
    "arrow",
    "line",
    "image",
    "formula",
    "table",
    "chart",
    "audio",
    "video",
    "component",
}
EFFECTS = {"appear", "fade", "rise", "zoom", "pulse", "exit", "flyout", "component", "custom-path", *CATALOG["effects"]}
TRANSITIONS = {"none", "fade", "push", "zoom", *CATALOG["transitions"]}
ASSET = re.compile(r"[a-f0-9]{64}")
MISSING = object()


class Conflict(ValueError):
    pass


def uid():
    return uuid.uuid4().hex


def check(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def string(value, limit=200000):
    return isinstance(value, str) and len(value) <= limit


def valid_path(path):
    if not string(path, 32768):
        return False
    tokens = re.findall(r"[MmLlCcQqZzEe]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", path)
    if not tokens or tokens[0].upper() != "M" or "".join(tokens) != re.sub(r"[\s,]+", "", path):
        return False
    count, arity = 0, 0
    for i, token in enumerate(tokens):
        if token.isalpha():
            if arity and (count == 0 or count % arity):
                return False
            cmd = token.upper()
            if cmd == "E" and i != len(tokens) - 1:
                return False
            arity = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0, "E": 0}[cmd]
            count = 0
        else:
            if not arity or not number(float(token), -10000, 10000):
                return False
            count += 1
    return not arity or (count > 0 and count % arity == 0)


def color(value):
    return (
        isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) is not None
    )


def ordered(container, field):
    items, order = container[field], container["order"]
    check(
        isinstance(items, dict)
        and isinstance(order, list)
        and len(order) == len(items)
        and set(order) == set(items),
        "顺序和对象集合不一致",
    )
    for key in items:
        check(
            isinstance(key, str)
            and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key)
            and key not in {"__proto__", "constructor", "prototype"},
            "无效对象 ID",
        )
    return items


def asset_ids(deck):
    ids = set()
    for slide in deck["slides"].values():
        ids.update(slide[k] for k in ("reference", "backdrop") if slide.get(k))
        ids.update(
            obj[k]
            for obj in slide["elements"].values()
            for k in ("asset", "poster")
            if obj.get(k)
        )
    return ids


def validate(deck):
    check(
        deck["format"] == FORMAT
        and string(deck["title"], 160)
        and deck["title"].strip(),
        "无效课件格式或名称",
    )
    check(all(number(deck[k], 20, 10000) for k in ("width", "height")), "画布尺寸无效")
    check(
        color(deck["theme"]["accent"]) and color(deck["theme"]["paper"]), "主题颜色无效"
    )
    slides = ordered(deck, "slides")
    check(0 < len(slides) <= 2000, "课件需要 1–2000 页")
    for slide in slides.values():
        check(string(slide["title"], 500) and string(slide["notes"]), "无效标题或备注")
        check(
            color(slide["background"])
            and slide["transition"] in TRANSITIONS,
            "无效页面样式",
        )
        check(number(slide.get("transitionDuration", 0.3), 0, 20), "无效切换时长")
        items = ordered(slide, "elements")
        check(len(items) <= 5000, "页面对象过多")
        for obj in items.values():
            if obj.get("type") == "component":
                check(string(obj.get("html"), 2_000_000), "互动组件需要 HTML 源码")
                params, actions = obj.get("parameters", {}), obj.get("actions", [])
                check(isinstance(params, dict) and len(params) <= 128 and all(
                    string(k, 80) and (type(v) is bool or number(v, -1e12, 1e12) or string(v, 8192))
                    for k, v in params.items()), "无效组件参数")
                check(isinstance(actions, list) and len(actions) <= 128 and
                      all(string(a, 80) and a for a in actions) and len(actions) == len(set(actions)), "无效组件动作")
            check(
                obj["type"] in TYPES
                and string(obj["name"], 500)
                and string(obj["text"]),
                "无效对象",
            )
            check(
                all(number(obj[k], -100000, 100000) for k in ("x", "y", "rotation"))
                and all(number(obj[k], 0.01, 100000) for k in ("w", "h", "size")),
                "对象尺寸无效",
            )
            check(obj["fill"] == "transparent" or color(obj["fill"]), "无效填充")
            check(
                color(obj["color"])
                and obj["align"] in {"left", "center", "right"}
                and type(obj["bold"]) is bool,
                "无效文字样式",
            )
            check(string(obj.get("font", ""), 150), "无效字体")
        steps = slide["steps"]
        check(isinstance(steps, list) and len(steps) <= 10000, "动画序列无效")
        for step in steps:
            if "bookmark" in step:
                check(number(step["bookmark"], 0, 86400), "无效媒体起始秒数")
            if step.get("effect") == "component":
                obj = items.get(step.get("target"), {})
                check(obj.get("type") == "component" and step.get("action") in obj.get("actions", []), "组件动作不可用")
            if step.get("effect") == "custom-path" or "path" in step:
                path = step.get("path")
                check(valid_path(path), "请输入有效的归一化运动路径")
            check(
                step["target"] in items
                and step["effect"] in EFFECTS
                and step["trigger"] in {"click", "with", "after"}
                and number(step["duration"], 0, 20)
                and number(step["delay"], 0, 120),
                "无效动画或动画目标",
            )
    check(
        all(isinstance(a, str) and ASSET.fullmatch(a) for a in asset_ids(deck)),
        "资源 ID 无效",
    )


def element(kind="text", **values):
    return dict(
        {
            "type": kind,
            "name": "文本框",
            "text": "",
            "x": 80,
            "y": 90,
            "w": 280,
            "h": 80,
            "rotation": 0,
            "size": 28,
            "color": "#192D31",
            "fill": "transparent",
            "align": "left",
            "bold": False,
            "font": "system-ui",
        },
        **values,
    )


def slide(title):
    return {
        "title": title,
        "notes": "",
        "background": "#FAF8F0",
        "reference": "",
        "backdrop": "",
        "transition": "fade",
        "order": [],
        "elements": {},
        "steps": [],
    }


def blank(title):
    return {
        "format": FORMAT,
        "title": title,
        "width": 960,
        "height": 540,
        "theme": {"accent": "#177866", "paper": "#FAF8F0"},
        "slides": {"p1": slide("第一页")},
        "order": ["p1"],
    }


def add_element(page, obj, key=None):
    key = key or uid()
    page["elements"][key] = obj
    page["order"].append(key)
    return key


def apply_edits(current, base, edits):
    check(isinstance(edits, list) and 0 < len(edits) <= 2000, "需要 1–2000 个编辑操作")
    draft, prior = copy.deepcopy(current), copy.deepcopy(base)
    for op in edits:
        path = op["path"]
        check(
            isinstance(path, list)
            and 0 < len(path) <= 6
            and all(isinstance(k, str) and k for k in path)
            and path[0] in {"title", "width", "height", "theme", "order", "slides"},
            "无效编辑路径",
        )
        a, b = draft, prior
        for key in path[:-1]:
            if (
                not isinstance(a, dict)
                or key not in a
                or not isinstance(b, dict)
                or key not in b
            ):
                raise Conflict("上级对象已更改：" + "/".join(path))
            a, b = a[key], b[key]
        key = path[-1]
        check(isinstance(a, dict) and isinstance(b, dict), "请整体替换有序列表")
        if a.get(key, MISSING) != b.get(key, MISSING):
            raise Conflict("属性已由其他操作者修改：" + "/".join(path))
        if op.get("remove") is True:
            check(key in a, "要移除的属性不存在")
            del a[key], b[key]
        else:
            a[key] = copy.deepcopy(op["value"])
            b[key] = copy.deepcopy(op["value"])
    validate(draft)
    return draft
