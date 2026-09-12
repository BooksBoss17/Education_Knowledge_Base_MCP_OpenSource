from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from lxml import etree

from .namespaces import NS, local_name

VISUAL_DIAGRAM_GROUP = "VISUAL_DIAGRAM_GROUP"
PURE_TEXT_CONTAINER = "PURE_TEXT_CONTAINER"
DECORATIVE_OR_EMPTY = "DECORATIVE_OR_EMPTY"
UNSUPPORTED_VISUAL_GROUP = "UNSUPPORTED_VISUAL_GROUP"


@dataclass(frozen=True)
class DrawingMLResult:
    classification: str
    feature_matrix: dict[str, Any]
    visible_labels: tuple[str, ...]
    svg: bytes | None
    unsupported_features: tuple[str, ...] = ()


def inspect_drawingml(node: etree._Element) -> DrawingMLResult:
    features = drawingml_features(node)
    labels = tuple(features["visible_textbox_labels"])
    unsupported = []
    if features["embedded_image_count"]:
        unsupported.append("embedded_image_in_group")
    if features["graphic_frame_count"]:
        unsupported.append("graphic_frame")
    if features["three_d_count"]:
        unsupported.append("three_d")

    meaningful_geometry = bool(
        features["custom_geometry_count"]
        or features["connector_count"]
        or features["line_count"]
        or features["filled_shape_count"]
        or features["stroked_shape_count"]
    )
    pure_text = (
        len(labels) > 0
        and features["shape_count"] == 1
        and features["group_count"] == 0
        and features["embedded_image_count"] == 0
        and not meaningful_geometry
    )
    visual = bool(labels) and (
        features["group_count"] > 0
        or features["shape_count"] > 1
        or meaningful_geometry
        or features["has_two_dimensional_layout"]
    )
    if visual and unsupported:
        classification = UNSUPPORTED_VISUAL_GROUP
        svg = None
    elif visual:
        classification = VISUAL_DIAGRAM_GROUP
        svg = render_drawingml_svg(node)
    elif pure_text:
        classification = PURE_TEXT_CONTAINER
        svg = None
    else:
        classification = DECORATIVE_OR_EMPTY
        svg = None
    return DrawingMLResult(
        classification,
        features,
        labels,
        svg,
        tuple(unsupported),
    )


def drawingml_features(node: etree._Element) -> dict[str, Any]:
    textboxes = node.xpath(".//w:txbxContent", namespaces=NS)
    labels = [_visible_text(box) for box in textboxes]
    labels = [label for label in labels if label]
    shapes = node.xpath(".//*[local-name()='wsp']")
    groups = node.xpath(".//*[local-name()='grpSp' or local-name()='wgp']")
    geometry_types = sorted(
        value for value in node.xpath(".//a:prstGeom/@prst", namespaces=NS) if value
    )
    positions = []
    rotations = []
    flips = []
    group_transforms = []
    child_transforms = []
    for xfrm in node.xpath(".//a:xfrm", namespaces=NS):
        item = _xfrm_dict(xfrm)
        positions.append(
            {
                key: item[key]
                for key in ("x", "y", "width", "height")
                if item.get(key) is not None
            }
        )
        if item.get("rotation"):
            rotations.append(item["rotation"])
        if item.get("flip_h") or item.get("flip_v"):
            flips.append({"flip_h": item["flip_h"], "flip_v": item["flip_v"]})
        parent = xfrm.getparent()
        if parent is not None and local_name(parent.tag) == "grpSpPr":
            group_transforms.append(item)
        else:
            child_transforms.append(item)
    distinct_centers = {
        (
            round(item.get("x", 0) + item.get("width", 0) / 2),
            round(item.get("y", 0) + item.get("height", 0) / 2),
        )
        for item in child_transforms
        if item.get("x") is not None and item.get("y") is not None
    }
    return {
        "textbox_count": len(textboxes),
        "visible_textbox_count": len(labels),
        "visible_textbox_labels": labels,
        "shape_count": len(shapes),
        "group_count": len(groups),
        "geometry_types": geometry_types,
        "custom_geometry_count": len(node.xpath(".//a:custGeom", namespaces=NS)),
        "connector_count": len(node.xpath(".//*[local-name()='cxnSp']")),
        "line_count": geometry_types.count("line"),
        "arrow_count": len(node.xpath(".//a:headEnd | .//a:tailEnd", namespaces=NS)),
        "filled_shape_count": len(node.xpath(".//a:solidFill", namespaces=NS)),
        "stroked_shape_count": sum(
            not bool(line.xpath("./a:noFill", namespaces=NS))
            for line in node.xpath(".//a:ln", namespaces=NS)
        ),
        "group_transforms": group_transforms,
        "child_transforms": child_transforms,
        "positions": positions,
        "rotations": rotations,
        "flips": flips,
        "embedded_image_count": len(node.xpath(".//a:blip", namespaces=NS)),
        "graphic_frame_count": len(node.xpath(".//*[local-name()='graphicFrame']")),
        "three_d_count": len(
            node.xpath(".//*[local-name()='scene3d' or local-name()='sp3d']")
        ),
        "has_two_dimensional_layout": len(distinct_centers) > 1,
    }


def render_drawingml_svg(node: etree._Element) -> bytes:
    renderer = _SvgRenderer()
    renderer.render(node)
    return renderer.finish()


Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1, 0, 0, 1, 0, 0)


class _SvgRenderer:
    def __init__(self) -> None:
        self.elements: list[str] = []
        self.boxes: list[tuple[float, float, float, float]] = []

    def render(self, drawing: etree._Element) -> None:
        roots = drawing.xpath(
            ".//*[local-name()='graphicData']/*[local-name()='wgp' or local-name()='wsp']"
        )
        for root in roots:
            if local_name(root.tag) == "wgp":
                self._group(root, IDENTITY, is_root=True)
            else:
                self._shape(root, IDENTITY)

    def _group(
        self, group: etree._Element, parent: Matrix, *, is_root: bool = False
    ) -> None:
        xfrm = group.find("./wpg:grpSpPr/a:xfrm", NS)
        matrix = _multiply(parent, _group_matrix(xfrm)) if xfrm is not None else parent
        for child in group:
            name = local_name(child.tag)
            if name == "grpSp":
                self._group(child, matrix)
            elif name == "wsp":
                self._shape(child, matrix)

    def _shape(self, shape: etree._Element, parent: Matrix) -> None:
        sppr = next(iter(shape.xpath("./*[local-name()='spPr']")), None)
        if sppr is None:
            return
        xfrm = sppr.find("a:xfrm", NS)
        if xfrm is None:
            return
        info = _xfrm_dict(xfrm)
        shape_matrix = _multiply(parent, _shape_placement(info, 1, 1))
        box_points = [
            _apply(shape_matrix, 0, 0),
            _apply(shape_matrix, 1, 0),
            _apply(shape_matrix, 1, 1),
            _apply(shape_matrix, 0, 1),
        ]
        bx = [p[0] for p in box_points]
        by = [p[1] for p in box_points]
        bbox = (min(bx), min(by), max(bx), max(by))
        self.boxes.append(bbox)
        fill = _paint(sppr, "fill", default="none")
        stroke = _paint(sppr.find("a:ln", NS), "stroke", default="none")
        parent_scale = math.sqrt(abs(parent[0] * parent[3] - parent[1] * parent[2]))
        stroke_width = max(
            _number(sppr.find("a:ln", NS), "w", 12700) * parent_scale,
            1,
        )
        style = f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.6f}"'
        preset = sppr.find("a:prstGeom", NS)
        custom = sppr.find("a:custGeom", NS)
        if preset is not None:
            kind = preset.get("prst", "rect")
            self._preset(kind, shape_matrix, style)
        elif custom is not None:
            self._custom(custom, parent, info, style)
        self._text(shape, bbox)

    def _preset(self, kind: str, matrix: Matrix, style: str) -> None:
        points = [_apply(matrix, 0, 0), _apply(matrix, 1, 1)]
        x, y = points[0]
        width, height = points[1][0] - x, points[1][1] - y
        if kind == "ellipse":
            self.elements.append(
                f'<ellipse cx="{x + width / 2:.3f}" cy="{y + height / 2:.3f}" rx="{abs(width) / 2:.3f}" ry="{abs(height) / 2:.3f}" {style}/>'
            )
        elif kind == "line":
            self.elements.append(
                f'<line x1="{x:.3f}" y1="{y:.3f}" x2="{x + width:.3f}" y2="{y + height:.3f}" {style}/>'
            )
        else:
            radius = min(abs(width), abs(height)) * 0.12 if kind == "roundRect" else 0
            self.elements.append(
                f'<rect x="{min(x, x + width):.3f}" y="{min(y, y + height):.3f}" width="{abs(width):.3f}" height="{abs(height):.3f}" rx="{radius:.3f}" {style}/>'
            )

    def _custom(self, custom, parent, info, style) -> None:
        width = max(float(info.get("width") or 1), 1)
        height = max(float(info.get("height") or 1), 1)
        for path in custom.xpath("./a:pathLst/a:path", namespaces=NS):
            pw = max(_number(path, "w", width), 1)
            ph = max(_number(path, "h", height), 1)
            matrix = _multiply(parent, _shape_placement(info, pw, ph))
            commands: list[str] = []
            for command in path:
                name = local_name(command.tag)
                points = [
                    _apply(matrix, _number(pt, "x", 0), _number(pt, "y", 0))
                    for pt in command.xpath("./a:pt", namespaces=NS)
                ]
                if name == "moveTo" and points:
                    commands.append(f"M {points[0][0]:.3f} {points[0][1]:.3f}")
                elif name == "lnTo" and points:
                    commands.append(f"L {points[0][0]:.3f} {points[0][1]:.3f}")
                elif name == "cubicBezTo" and len(points) == 3:
                    commands.append(
                        "C " + " ".join(f"{px:.3f} {py:.3f}" for px, py in points)
                    )
                elif name == "quadBezTo" and len(points) == 2:
                    commands.append(
                        "Q " + " ".join(f"{px:.3f} {py:.3f}" for px, py in points)
                    )
                elif name == "close":
                    commands.append("Z")
            if commands:
                self.elements.append(f'<path d="{" ".join(commands)}" {style}/>')

    def _text(
        self, shape: etree._Element, bbox: tuple[float, float, float, float]
    ) -> None:
        boxes = shape.xpath("./*[local-name()='txbx']/w:txbxContent", namespaces=NS)
        if not boxes:
            return
        lines = _visible_lines(boxes[0])
        if not lines:
            return
        x1, y1, x2, y2 = bbox
        height = max(y2 - y1, 1)
        font_size = max(height * 0.18, 1)
        color = _text_color(boxes[0])
        bold = bool(boxes[0].xpath(".//w:b", namespaces=NS))
        italic = bool(boxes[0].xpath(".//w:i", namespaces=NS))
        attrs = [
            f'fill="{color}"',
            f'font-size="{font_size:.3f}"',
            'text-anchor="middle"',
        ]
        if bold:
            attrs.append('font-weight="bold"')
        if italic:
            attrs.append('font-style="italic"')
        start_y = (y1 + y2) / 2 - (len(lines) - 1) * font_size * 0.55
        spans = "".join(
            f'<tspan x="{(x1 + x2) / 2:.3f}" y="{start_y + i * font_size * 1.1:.3f}">{html.escape(line)}</tspan>'
            for i, line in enumerate(lines)
        )
        self.elements.append(f"<text {' '.join(attrs)}>{spans}</text>")

    def finish(self) -> bytes:
        if not self.boxes:
            raise ValueError("DrawingML renderer found no supported shape bounds")
        min_x = min(box[0] for box in self.boxes)
        min_y = min(box[1] for box in self.boxes)
        max_x = max(box[2] for box in self.boxes)
        max_y = max(box[3] for box in self.boxes)
        padding = max(max_x - min_x, max_y - min_y) * 0.02
        view = (
            min_x - padding,
            min_y - padding,
            max_x - min_x + 2 * padding,
            max_y - min_y + 2 * padding,
        )
        payload = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{view[0]:.3f} {view[1]:.3f} {view[2]:.3f} {view[3]:.3f}" '
            'preserveAspectRatio="xMidYMid meet">\n'
            + "\n".join(self.elements)
            + "\n</svg>\n"
        )
        return payload.encode("utf-8")


def _visible_text(node: etree._Element) -> str:
    return " ".join("".join(node.xpath(".//w:t/text()", namespaces=NS)).split())


def _visible_lines(node: etree._Element) -> list[str]:
    lines = []
    for paragraph in node.xpath(".//w:p", namespaces=NS):
        text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))
        if text.strip():
            lines.append(text.strip())
    return lines


def _xfrm_dict(xfrm: etree._Element) -> dict[str, Any]:
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    choff = xfrm.find("a:chOff", NS)
    chext = xfrm.find("a:chExt", NS)
    return {
        "x": _number(off, "x") if off is not None else None,
        "y": _number(off, "y") if off is not None else None,
        "width": _number(ext, "cx") if ext is not None else None,
        "height": _number(ext, "cy") if ext is not None else None,
        "child_x": _number(choff, "x") if choff is not None else None,
        "child_y": _number(choff, "y") if choff is not None else None,
        "child_width": _number(chext, "cx") if chext is not None else None,
        "child_height": _number(chext, "cy") if chext is not None else None,
        "rotation": _number(xfrm, "rot", 0) / 60000,
        "flip_h": xfrm.get("flipH") in {"1", "true"},
        "flip_v": xfrm.get("flipV") in {"1", "true"},
    }


def _group_matrix(xfrm: etree._Element | None) -> Matrix:
    if xfrm is None:
        return IDENTITY
    info = _xfrm_dict(xfrm)
    x, y = float(info.get("x") or 0), float(info.get("y") or 0)
    width, height = float(info.get("width") or 1), float(info.get("height") or 1)
    child_x, child_y = float(info.get("child_x") or 0), float(info.get("child_y") or 0)
    child_w = float(info.get("child_width") or width or 1)
    child_h = float(info.get("child_height") or height or 1)
    matrix: Matrix = (
        width / child_w,
        0,
        0,
        height / child_h,
        x - child_x * width / child_w,
        y - child_y * height / child_h,
    )
    if info.get("flip_h"):
        matrix = _multiply((-1, 0, 0, 1, 2 * x + width, 0), matrix)
    if info.get("flip_v"):
        matrix = _multiply((1, 0, 0, -1, 0, 2 * y + height), matrix)
    angle = math.radians(float(info.get("rotation") or 0))
    if angle:
        cx, cy = x + width / 2, y + height / 2
        rotate: Matrix = (
            math.cos(angle),
            math.sin(angle),
            -math.sin(angle),
            math.cos(angle),
            cx - math.cos(angle) * cx + math.sin(angle) * cy,
            cy - math.sin(angle) * cx - math.cos(angle) * cy,
        )
        matrix = _multiply(rotate, matrix)
    return matrix


def _shape_placement(
    info: dict[str, Any], unit_width: float, unit_height: float
) -> Matrix:
    x, y = float(info.get("x") or 0), float(info.get("y") or 0)
    width = max(float(info.get("width") or 1), 1)
    height = max(float(info.get("height") or 1), 1)
    matrix: Matrix = (width / unit_width, 0, 0, height / unit_height, x, y)
    cx, cy = x + width / 2, y + height / 2
    if info.get("flip_h"):
        matrix = _multiply((-1, 0, 0, 1, 2 * cx, 0), matrix)
    if info.get("flip_v"):
        matrix = _multiply((1, 0, 0, -1, 0, 2 * cy), matrix)
    angle = math.radians(float(info.get("rotation") or 0))
    if angle:
        rotate: Matrix = (
            math.cos(angle),
            math.sin(angle),
            -math.sin(angle),
            math.cos(angle),
            cx - math.cos(angle) * cx + math.sin(angle) * cy,
            cy - math.sin(angle) * cx - math.cos(angle) * cy,
        )
        matrix = _multiply(rotate, matrix)
    return matrix


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    a, b, c, d, e, f = left
    g, h, i, j, k, l = right
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * l + e,
        b * k + d * l + f,
    )


def _apply(matrix: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _number(node: etree._Element | None, name: str, default: float = 0) -> float:
    if node is None:
        return default
    try:
        return float(node.get(name, default))
    except (TypeError, ValueError):
        return default


def _paint(node: etree._Element | None, role: str, default: str) -> str:
    if node is None or node.find("a:noFill", NS) is not None:
        return "none"
    color = node.find("a:solidFill/a:srgbClr", NS)
    if color is None and local_name(node.tag) == "spPr":
        color = node.find("a:solidFill/a:srgbClr", NS)
    if color is not None and re.fullmatch(r"[0-9A-Fa-f]{6}", color.get("val", "")):
        return "#" + color.get("val")
    scheme = node.find("a:solidFill/a:schemeClr", NS)
    if scheme is not None:
        return {"tx1": "#000000", "lt1": "#ffffff"}.get(scheme.get("val"), "#808080")
    return default


def _text_color(node: etree._Element) -> str:
    values = node.xpath(".//w:color/@w:val", namespaces=NS)
    for value in reversed(values):
        if re.fullmatch(r"[0-9A-Fa-f]{6}", value):
            return "#" + value
    return "#000000"


def feature_fingerprint(features: dict[str, Any]) -> str:
    stable = {
        key: value for key, value in features.items() if key != "visible_textbox_labels"
    }
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
