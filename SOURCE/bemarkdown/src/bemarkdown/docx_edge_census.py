from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree

from .mtef_cache import MtefCacheContext
from .namespaces import local_name
from .package import PackageIndex
from .pipeline import _finalize_formula_report, _new_report
from .scanner import DocumentScanner
from .serializer import MarkdownSerializer

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
V_NS = "urn:schemas-microsoft-com:vml"
O_NS = "urn:schemas-microsoft-com:office:office"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
DGM_NS = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
WPS_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
WPG_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"

NS = {
    "w": W_NS,
    "m": M_NS,
    "a": A_NS,
    "r": R_NS,
    "v": V_NS,
    "o": O_NS,
    "wp": WP_NS,
    "pic": PIC_NS,
    "mc": MC_NS,
    "c": C_NS,
    "dgm": DGM_NS,
    "wps": WPS_NS,
    "wpg": WPG_NS,
}

REQUIRED_EDGE_TYPES = (
    "textbox",
    "header",
    "footer",
    "footnote",
    "endnote",
    "comment",
    "comment_marker",
    "hyperlink",
    "bookmark",
    "field",
    "drawing",
    "vml",
    "smartart",
    "chart",
    "ole",
    "non_math_package",
    "image_relationship",
    "table",
    "table_outside_main_body",
    "list",
    "numbering_definition",
    "numbering_override",
    "section",
    "break",
    "alternate_content",
    "sdt",
    "revision",
    "wrapper",
    "formula_outside_main_body",
)


def _qn(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def _visible_text(node: etree._Element) -> str:
    fragments: list[str] = []

    def visit(item: etree._Element) -> None:
        if not isinstance(item.tag, str):
            return
        if item.tag.startswith(f"{{{M_NS}}}"):
            return
        name = local_name(item.tag)
        if name in {"t", "delText"} and item.text:
            fragments.append(item.text)
            return
        if name == "tab":
            fragments.append("\t")
            return
        for child in item:
            visit(child)
        if name in {"p", "tr", "tc"}:
            fragments.append(" ")

    visit(node)
    return re.sub(r"\s+", " ", "".join(fragments)).strip()


def _short_text(value: str, limit: int = 160) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _record(
    package: PackageIndex,
    part_name: str,
    node: etree._Element,
    *,
    element_type: str,
    subtype: str,
    current_handling: str,
    risk_level: str,
    relationship_id: str | None = None,
    relationship_target: str | None = None,
    notes: str = "",
    metadata: dict[str, Any] | None = None,
    visible_text: str | None = None,
) -> dict[str, Any]:
    text = _visible_text(node) if visible_text is None else visible_text
    return {
        "document_id": package.source.stem,
        "source_path": str(package.source.resolve()),
        "part_name": part_name,
        "element_type": element_type,
        "subtype": subtype,
        "locator": node.getroottree().getpath(node),
        "relationship_id": relationship_id,
        "relationship_target": relationship_target,
        "visible_text_length": len(text),
        "visible_text": _short_text(text),
        "contains_text": bool(text),
        "contains_formula": bool(node.xpath(".//m:oMath", namespaces=NS)),
        "contains_image": bool(node.xpath(".//a:blip | .//v:imagedata", namespaces=NS)),
        "contains_table": bool(node.xpath(".//w:tbl", namespaces=NS)),
        "current_handling": current_handling,
        "risk_level": risk_level,
        "notes": notes,
        "metadata": metadata or {},
    }


def _attr(node: etree._Element, name: str) -> str | None:
    return node.get(_qn(W_NS, name)) or node.get(name)


def _header_footer_subtype(node: etree._Element, part_name: str) -> str:
    instructions = " ".join(
        str(value)
        for value in node.xpath(
            ".//w:fldSimple/@w:instr | .//w:instrText/text()", namespaces=NS
        )
    ).upper()
    text = _visible_text(node)
    if not text:
        return "empty_header" if "header" in part_name else "empty_footer"
    if re.search(r"\b(?:PAGE|NUMPAGES)\b", instructions) or re.search(
        r"(?:第\s*\d+\s*页|共\s*\d+\s*页|试卷第|答案第)", text
    ):
        return "page_number_or_document_label"
    if re.search(r"(?:版权|©|公司|收集整理|全品|学练考)", text):
        return "copyright_or_watermark"
    if re.search(r"(?:学校|中学|学年|校本作业|试卷标题)", text):
        return "school_or_exam_title"
    if "header" in part_name:
        return "other_visible_header"
    return "other_visible_footer"


def _header_footer_risk(subtype: str) -> str:
    if subtype in {
        "empty_header",
        "empty_footer",
        "page_number_or_document_label",
        "copyright_or_watermark",
    }:
        return "P3"
    return "P1"


def _is_unselected_fallback(node: etree._Element) -> bool:
    for ancestor in node.iterancestors():
        if isinstance(ancestor.tag, str) and local_name(ancestor.tag) == "Fallback":
            parent = ancestor.getparent()
            return bool(
                parent is not None and parent.xpath("./mc:Choice", namespaces=NS)
            )
    return False


def _field_type(instruction: str) -> str:
    match = re.match(r"\s*([A-Za-z]+)", instruction)
    value = match.group(1).upper() if match else "OTHER"
    return (
        value
        if value
        in {
            "EQ",
            "PAGE",
            "NUMPAGES",
            "DATE",
            "TIME",
            "REF",
            "PAGEREF",
            "HYPERLINK",
            "TOC",
            "SEQ",
            "SYMBOL",
        }
        else "OTHER"
    )


def _field_handling(field_type: str) -> tuple[str, str]:
    if field_type == "EQ":
        return "SUPPORTED", "P3"
    if field_type in {"REF", "PAGEREF", "HYPERLINK", "TOC", "SEQ", "SYMBOL"}:
        return "PARTIALLY_SUPPORTED", "P1"
    return "PARTIALLY_SUPPORTED", "P2"


def _under(node: etree._Element, name: str) -> bool:
    return any(
        isinstance(parent.tag, str) and local_name(parent.tag) == name
        for parent in node.iterancestors()
    )


def _main_body_handling(node: etree._Element) -> str:
    if _under(node, "txbxContent"):
        return "NOT_REACHED_BY_CURRENT_SCANNER"
    if _under(node, "tc") and not _under(node, "p"):
        return "NOT_REACHED_BY_CURRENT_SCANNER"
    return "SUPPORTED"


def _scan_document_semantics(
    package: PackageIndex, root: etree._Element, records: list[dict[str, Any]]
) -> None:
    part_name = "word/document.xml"
    for node in root.xpath(".//w:hyperlink", namespaces=NS):
        rid = node.get(_qn(R_NS, "id"))
        anchor = _attr(node, "anchor")
        rel = package.relationship(part_name, rid) if rid else None
        target = f"#{anchor}" if anchor else (rel.target if rel else None)
        handling = _main_body_handling(node)
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="hyperlink",
                subtype="internal_anchor" if anchor else "external",
                current_handling=handling,
                risk_level="P3" if handling == "SUPPORTED" else "P1",
                relationship_id=rid,
                relationship_target=target,
                notes="Main-body hyperlinks are emitted as Markdown links.",
            )
        )


def _first_relationship_id(node: etree._Element) -> str | None:
    values = node.xpath(".//@r:embed | .//@r:link | .//@r:id", namespaces=NS)
    return str(values[0]) if values else None


def _scan_tables(
    package: PackageIndex, root: etree._Element, records: list[dict[str, Any]]
) -> None:
    part_name = "word/document.xml"
    for node in root.xpath(".//w:tbl", namespaces=NS):
        depth = 1 + sum(
            local_name(parent.tag) == "tbl" for parent in node.iterancestors()
        )
        cells = node.xpath("./w:tr/w:tc", namespaces=NS)
        nested_count = len(node.xpath(".//w:tbl", namespaces=NS))
        text = _visible_text(node)
        not_reached = depth > 1 or _under(node, "txbxContent")
        handling = "NOT_REACHED_BY_CURRENT_SCANNER" if not_reached else "SUPPORTED"
        risk = (
            "P0"
            if not_reached and text
            else ("P3" if handling == "SUPPORTED" else "P2")
        )
        if depth == 1 and nested_count:
            handling, risk = "PARTIALLY_SUPPORTED", "P0"
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="table",
                subtype="nested" if depth > 1 else "top_level",
                current_handling=handling,
                risk_level=risk,
                notes="Top-level rows/cells and direct cell paragraphs are serialized; nested tables are omitted.",
                metadata={
                    "nesting_depth": depth,
                    "rows": len(node.xpath("./w:tr", namespaces=NS)),
                    "cells": len(cells),
                    "grid_span": len(node.xpath(".//w:gridSpan", namespaces=NS)),
                    "vertical_merge": len(node.xpath(".//w:vMerge", namespaces=NS)),
                    "nested_tables": nested_count,
                    "max_paragraphs_per_cell": max(
                        (len(cell.xpath("./w:p", namespaces=NS)) for cell in cells),
                        default=0,
                    ),
                    "formula_in_cell": bool(
                        node.xpath(".//w:tc//m:oMath", namespaces=NS)
                    ),
                    "image_in_cell": bool(
                        node.xpath(
                            ".//w:tc//a:blip | .//w:tc//v:imagedata", namespaces=NS
                        )
                    ),
                    "list_in_cell": bool(node.xpath(".//w:tc//w:numPr", namespaces=NS)),
                    "field_in_cell": bool(
                        node.xpath(
                            ".//w:tc//w:fldSimple | .//w:tc//w:fldChar", namespaces=NS
                        )
                    ),
                },
            )
        )


def _drawing_subtype(node: etree._Element) -> str:
    if node.xpath(".//c:chart", namespaces=NS):
        return "chart"
    if node.xpath(".//dgm:relIds", namespaces=NS):
        return "smartart"
    if node.xpath(".//wps:wsp | .//wps:txbx", namespaces=NS):
        return "shape_text"
    if node.xpath(".//pic:pic", namespaces=NS):
        return "picture"
    return "other"


def _scan_drawings(
    package: PackageIndex, root: etree._Element, records: list[dict[str, Any]]
) -> None:
    part_name = "word/document.xml"
    for node in root.xpath(".//w:drawing", namespaces=NS):
        subtype = _drawing_subtype(node)
        rid = _first_relationship_id(node)
        rel = package.relationship(part_name, rid) if rid else None
        if (
            subtype == "picture"
            and rel
            and not rel.external
            and rel.target_part in package.parts
        ):
            handling, risk = "PRESERVED_AS_ASSET", "P3"
        elif subtype == "picture":
            handling, risk = "IGNORED_UNKNOWN", "P1"
        else:
            handling = "NOT_REACHED_BY_CURRENT_SCANNER"
            risk = "P0" if subtype == "shape_text" else "P1"
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="drawing",
                subtype=subtype,
                current_handling=handling,
                risk_level=risk,
                relationship_id=rid,
                relationship_target=(
                    rel.target
                    if rel and rel.external
                    else rel.target_part
                    if rel
                    else None
                ),
                notes="Picture blips are assets; chart/SmartArt/shape semantics are not parsed.",
                metadata={"linked": bool(rel and rel.external)},
            )
        )

    for node in root.xpath(".//o:OLEObject", namespaces=NS):
        rid = node.get(_qn(R_NS, "id"))
        rel = package.relationship(part_name, rid) if rid else None
        prog_id = node.get("ProgID") or "UNKNOWN"
        parent = node.getparent()
        object_node = parent
        while object_node is not None and local_name(object_node.tag) != "object":
            object_node = object_node.getparent()
        preview_nodes = (
            object_node.xpath(".//v:imagedata", namespaces=NS)
            if object_node is not None
            else []
        )
        preview_rid = preview_nodes[0].get(_qn(R_NS, "id")) if preview_nodes else None
        preview_rel = (
            package.relationship(part_name, preview_rid) if preview_rid else None
        )
        preview_exists = bool(preview_rel and preview_rel.target_part in package.parts)
        is_formula = (
            prog_id.lower().startswith("equation.") or "mathtype" in prog_id.lower()
        )
        handling = (
            "SUPPORTED"
            if is_formula
            else ("PRESERVED_AS_ASSET" if preview_exists else "IGNORED_UNKNOWN")
        )
        risk = "P3" if is_formula or preview_exists else "P1"
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="ole",
                subtype=prog_id,
                current_handling=handling,
                risk_level=risk,
                relationship_id=rid,
                relationship_target=rel.target_part if rel else None,
                notes="Formula OLE is routed structurally; other OLE keeps an available preview only.",
                metadata={
                    "formula_candidate": is_formula,
                    "preview_relationship_id": preview_rid,
                    "preview_target": preview_rel.target_part if preview_rel else None,
                    "preview_exists": preview_exists,
                    "embedded": not bool(rel and rel.external),
                },
            )
        )

    for node in root.xpath(".//mc:AlternateContent", namespaces=NS):
        choices = node.xpath("./mc:Choice", namespaces=NS)
        fallbacks = node.xpath("./mc:Fallback", namespaces=NS)
        choice_text = _visible_text(choices[0]) if choices else ""
        fallback_text = _visible_text(fallbacks[0]) if fallbacks else ""
        differs = bool(choices and fallbacks and choice_text != fallback_text)
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="alternate_content",
                subtype="choice_and_fallback"
                if choices and fallbacks
                else "single_branch",
                current_handling="PARTIALLY_SUPPORTED" if differs else "SUPPORTED",
                risk_level="P1" if differs else "P3",
                notes="Production always selects the first Choice when present, otherwise Fallback.",
                metadata={
                    "choices": len(choices),
                    "fallbacks": len(fallbacks),
                    "choice_requires": choices[0].get("Requires") if choices else None,
                    "branch_text_differs": differs,
                },
            )
        )

    wrapper_policy = {
        "smartTag": ("SUPPORTED", "P3"),
        "customXml": ("SUPPORTED", "P3"),
        "proofErr": ("IGNORED_INTENTIONALLY", "P3"),
        "permStart": ("IGNORED_INTENTIONALLY", "P2"),
        "permEnd": ("IGNORED_INTENTIONALLY", "P2"),
    }
    for subtype, (handling, risk) in wrapper_policy.items():
        for node in root.xpath(f".//w:{subtype}", namespaces=NS):
            records.append(
                _record(
                    package,
                    part_name,
                    node,
                    element_type="wrapper",
                    subtype=subtype,
                    current_handling=handling,
                    risk_level=risk,
                    notes="Wrapper content is recursive where applicable; marker-only elements are omitted.",
                )
            )

    for node in root.xpath(".//w:fldSimple", namespaces=NS):
        instruction = _attr(node, "instr") or ""
        category = _field_type(instruction)
        handling, risk = _field_handling(category)
        if _main_body_handling(node) != "SUPPORTED":
            handling, risk = "NOT_REACHED_BY_CURRENT_SCANNER", "P0"
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="field",
                subtype=category,
                current_handling=handling,
                risk_level=risk,
                notes="Simple field; EQ is parsed, other cached results are preserved.",
                metadata={"form": "simple", "instruction": _short_text(instruction)},
            )
        )

    stack: list[dict[str, Any]] = []
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        name = local_name(node.tag)
        if name == "fldChar":
            kind = _attr(node, "fldCharType") or ""
            if kind == "begin":
                stack.append(
                    {"begin": node, "instruction": [], "result": [], "separated": False}
                )
            elif kind == "separate" and stack:
                stack[-1]["separated"] = True
            elif kind == "end" and stack:
                frame = stack.pop()
                instruction = "".join(frame["instruction"]).strip()
                result = re.sub(r"\s+", " ", "".join(frame["result"])).strip()
                category = _field_type(instruction)
                handling, risk = _field_handling(category)
                if _main_body_handling(frame["begin"]) != "SUPPORTED":
                    handling, risk = "NOT_REACHED_BY_CURRENT_SCANNER", "P0"
                records.append(
                    _record(
                        package,
                        part_name,
                        frame["begin"],
                        element_type="field",
                        subtype=category,
                        current_handling=handling,
                        risk_level=risk,
                        notes="Complex field; visible result is retained when balanced.",
                        metadata={
                            "form": "complex",
                            "instruction": _short_text(instruction),
                        },
                        visible_text=result,
                    )
                )
            continue
        if name == "instrText" and stack and not stack[-1]["separated"]:
            stack[-1]["instruction"].append(node.text or "")
        elif name in {"t", "delText"} and node.text:
            for frame in stack:
                if frame["separated"]:
                    frame["result"].append(node.text)

    for node in root.xpath(".//w:bookmarkStart", namespaces=NS):
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="bookmark",
                subtype="start",
                current_handling="IGNORED_INTENTIONALLY",
                risk_level="P2",
                notes="Bookmark boundary is omitted; surrounding text is preserved.",
                metadata={"id": _attr(node, "id"), "name": _attr(node, "name")},
            )
        )

    for node in root.xpath(".//w:sdt", namespaces=NS):
        handling = _main_body_handling(node)
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="sdt",
                subtype="content_control",
                current_handling=handling,
                risk_level="P3" if handling == "SUPPORTED" else "P0",
                notes="Body/inline SDT content is recursive; block SDT in a table cell is not.",
            )
        )

    revision_policy = {
        "ins": ("SUPPORTED", "P3"),
        "del": ("IGNORED_INTENTIONALLY", "P3"),
        "moveFrom": ("IGNORED_INTENTIONALLY", "P3"),
        "moveTo": ("NOT_REACHED_BY_CURRENT_SCANNER", "P0"),
    }
    for subtype, (handling, risk) in revision_policy.items():
        for node in root.xpath(f".//w:{subtype}", namespaces=NS):
            if _under(node, "txbxContent"):
                handling, risk = "NOT_REACHED_BY_CURRENT_SCANNER", "P0"
            records.append(
                _record(
                    package,
                    part_name,
                    node,
                    element_type="revision",
                    subtype=subtype,
                    current_handling=handling,
                    risk_level=risk,
                    notes="Production final-view policy inferred from recursive scanner cases.",
                )
            )


def _scan_non_body_parts(package: PackageIndex, records: list[dict[str, Any]]) -> None:
    for part_name in sorted(package.parts):
        element_type = None
        if re.fullmatch(r"word/header\d+\.xml", part_name):
            element_type = "header"
        elif re.fullmatch(r"word/footer\d+\.xml", part_name):
            element_type = "footer"
        if element_type:
            root = package.xml(part_name)
            subtype = _header_footer_subtype(root, part_name)
            records.append(
                _record(
                    package,
                    part_name,
                    root,
                    element_type=element_type,
                    subtype=subtype,
                    current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                    risk_level=_header_footer_risk(subtype),
                    notes="Non-body part is inventoried but not serialized by production.",
                )
            )
            _scan_non_body_details(
                package,
                part_name,
                root,
                records,
                container_type=element_type,
                container_subtype=subtype,
            )

    note_specs = (
        ("word/footnotes.xml", "footnote", "footnote"),
        ("word/endnotes.xml", "endnote", "endnote"),
        ("word/comments.xml", "comment", "comment"),
    )
    for part_name, child_name, element_type in note_specs:
        if not package.has_part(part_name):
            continue
        root = package.xml(part_name)
        for node in root.xpath(f"./w:{child_name}", namespaces=NS):
            identifier = _attr(node, "id") or ""
            # Word reserves negative note IDs for separators/continuations.
            note_type = _attr(node, "type")
            if element_type in {"footnote", "endnote"} and (
                identifier.startswith("-")
                or note_type
                in {"separator", "continuationSeparator", "continuationNotice"}
            ):
                continue
            metadata: dict[str, Any] = {"id": identifier}
            if element_type == "comment":
                metadata.update(
                    {
                        "author": _attr(node, "author"),
                        "date": _attr(node, "date"),
                    }
                )
            records.append(
                _record(
                    package,
                    part_name,
                    node,
                    element_type=element_type,
                    subtype=identifier,
                    current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                    risk_level="P3" if element_type == "comment" else "P0",
                    notes=(
                        "Review metadata is not serialized."
                        if element_type == "comment"
                        else "Referenced note content is not serialized."
                    ),
                    metadata=metadata,
                )
            )
        _scan_non_body_details(
            package,
            part_name,
            root,
            records,
            container_type=element_type,
            container_subtype=element_type,
        )


def _complex_fields(root: etree._Element) -> list[tuple[etree._Element, str, str]]:
    result: list[tuple[etree._Element, str, str]] = []
    stack: list[dict[str, Any]] = []
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        name = local_name(node.tag)
        if name == "fldChar":
            kind = _attr(node, "fldCharType") or ""
            if kind == "begin":
                stack.append(
                    {"begin": node, "instruction": [], "result": [], "separated": False}
                )
            elif kind == "separate" and stack:
                stack[-1]["separated"] = True
            elif kind == "end" and stack:
                frame = stack.pop()
                result.append(
                    (
                        frame["begin"],
                        "".join(frame["instruction"]).strip(),
                        re.sub(r"\s+", " ", "".join(frame["result"])).strip(),
                    )
                )
            continue
        if name == "instrText" and stack and not stack[-1]["separated"]:
            stack[-1]["instruction"].append(node.text or "")
        elif name in {"t", "delText"} and node.text:
            for frame in stack:
                if frame["separated"]:
                    frame["result"].append(node.text)
    return result


def _scan_non_body_details(
    package: PackageIndex,
    part_name: str,
    root: etree._Element,
    records: list[dict[str, Any]],
    *,
    container_type: str,
    container_subtype: str,
) -> None:
    container_risk = (
        _header_footer_risk(container_subtype)
        if container_type in {"header", "footer"}
        else ("P3" if container_type == "comment" else "P0")
    )
    for node in root.xpath(".//w:txbxContent", namespaces=NS):
        if _is_unselected_fallback(node):
            continue
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="textbox",
                subtype=f"{container_type}_textbox",
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level=container_risk,
                notes="Textbox is inside a non-body part and is not serialized.",
            )
        )
    for node in root.xpath(".//m:oMath", namespaces=NS):
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="formula_outside_main_body",
                subtype=container_type,
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level="P3" if container_risk == "P3" else "P1",
                notes="Formula exists outside word/document.xml body routing.",
            )
        )
    for node in root.xpath(".//w:tbl", namespaces=NS):
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="table_outside_main_body",
                subtype=container_type,
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level=container_risk,
                notes="Table exists outside word/document.xml body routing.",
            )
        )
    for node in root.xpath(".//w:fldSimple", namespaces=NS):
        instruction = _attr(node, "instr") or ""
        category = _field_type(instruction)
        risk = "P3" if category in {"PAGE", "NUMPAGES"} else container_risk
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="field",
                subtype=category,
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level=risk,
                notes="Field result is in a non-body part and is not serialized.",
                metadata={"form": "simple", "instruction": _short_text(instruction)},
            )
        )
    for begin, instruction, field_result in _complex_fields(root):
        category = _field_type(instruction)
        risk = "P3" if category in {"PAGE", "NUMPAGES"} else container_risk
        records.append(
            _record(
                package,
                part_name,
                begin,
                element_type="field",
                subtype=category,
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level=risk,
                notes="Complex field result is in a non-body part and is not serialized.",
                metadata={"form": "complex", "instruction": _short_text(instruction)},
                visible_text=field_result,
            )
        )
    for node in root.xpath(".//w:drawing", namespaces=NS):
        subtype = _drawing_subtype(node)
        rid = _first_relationship_id(node)
        rel = package.relationship(part_name, rid) if rid else None
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="drawing",
                subtype=f"{container_type}_{subtype}",
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level=container_risk if subtype == "picture" else "P1",
                relationship_id=rid,
                relationship_target=(
                    rel.target
                    if rel and rel.external
                    else rel.target_part
                    if rel
                    else None
                ),
                notes="Drawing exists in a non-body part and does not enter asset routing.",
            )
        )
    for node in root.xpath(".//mc:AlternateContent", namespaces=NS):
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="alternate_content",
                subtype=f"{container_type}_choice_fallback",
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level=container_risk,
                notes="AlternateContent is present in a non-body part.",
            )
        )
    for node in root.xpath(".//w:hyperlink", namespaces=NS):
        rid = node.get(_qn(R_NS, "id"))
        anchor = _attr(node, "anchor")
        rel = package.relationship(part_name, rid) if rid else None
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="hyperlink",
                subtype=f"{container_type}_{'internal_anchor' if anchor else 'external'}",
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level=container_risk,
                relationship_id=rid,
                relationship_target=(
                    f"#{anchor}" if anchor else rel.target if rel else None
                ),
                notes="Hyperlink exists in a non-body part.",
            )
        )
    for node in root.xpath(".//w:sdt", namespaces=NS):
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="sdt",
                subtype=f"{container_type}_content_control",
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level=container_risk,
                notes="Content control exists in a non-body part.",
            )
        )
    for revision in ("ins", "del", "moveFrom", "moveTo"):
        for node in root.xpath(f".//w:{revision}", namespaces=NS):
            records.append(
                _record(
                    package,
                    part_name,
                    node,
                    element_type="revision",
                    subtype=f"{container_type}_{revision}",
                    current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                    risk_level=container_risk,
                    notes="Revision markup exists in a non-body part.",
                )
            )
    for wrapper in ("smartTag", "customXml", "proofErr", "permStart", "permEnd"):
        for node in root.xpath(f".//w:{wrapper}", namespaces=NS):
            records.append(
                _record(
                    package,
                    part_name,
                    node,
                    element_type="wrapper",
                    subtype=f"{container_type}_{wrapper}",
                    current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                    risk_level=container_risk,
                    notes="Wrapper/marker exists in a non-body part.",
                )
            )
    for node in root.xpath(".//o:OLEObject", namespaces=NS):
        rid = node.get(_qn(R_NS, "id"))
        rel = package.relationship(part_name, rid) if rid else None
        prog_id = node.get("ProgID") or "UNKNOWN"
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="ole",
                subtype=f"{container_type}:{prog_id}",
                current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                risk_level=container_risk,
                relationship_id=rid,
                relationship_target=rel.target_part if rel else None,
                notes="OLE object exists outside main body routing.",
                metadata={
                    "formula_candidate": prog_id.lower().startswith("equation.")
                    or "mathtype" in prog_id.lower()
                },
            )
        )


def _numbering_maps(
    package: PackageIndex,
) -> tuple[dict[str, dict[int, str]], dict[str, str], set[str]]:
    abstract_levels: dict[str, dict[int, str]] = {}
    num_to_abstract: dict[str, str] = {}
    overridden_nums: set[str] = set()
    if not package.has_part("word/numbering.xml"):
        return abstract_levels, num_to_abstract, overridden_nums
    root = package.xml("word/numbering.xml")
    for abstract in root.xpath("./w:abstractNum", namespaces=NS):
        abstract_id = _attr(abstract, "abstractNumId") or ""
        levels: dict[int, str] = {}
        for level in abstract.xpath("./w:lvl", namespaces=NS):
            ilvl = int(_attr(level, "ilvl") or 0)
            fmt = level.find("w:numFmt", NS)
            levels[ilvl] = _attr(fmt, "val") if fmt is not None else "decimal"
        abstract_levels[abstract_id] = levels
    for num in root.xpath("./w:num", namespaces=NS):
        num_id = _attr(num, "numId") or ""
        abstract = num.find("w:abstractNumId", NS)
        num_to_abstract[num_id] = _attr(abstract, "val") if abstract is not None else ""
        if num.xpath("./w:lvlOverride", namespaces=NS):
            overridden_nums.add(num_id)
    return abstract_levels, num_to_abstract, overridden_nums


def _scan_layout_and_numbering(
    package: PackageIndex, root: etree._Element, records: list[dict[str, Any]]
) -> None:
    part_name = "word/document.xml"
    abstract_levels, num_to_abstract, overridden_nums = _numbering_maps(package)
    for node in root.xpath(".//w:numPr", namespaces=NS):
        num_id_nodes = node.xpath("./w:numId", namespaces=NS)
        level_nodes = node.xpath("./w:ilvl", namespaces=NS)
        num_id = _attr(num_id_nodes[0], "val") if num_id_nodes else ""
        level = int(_attr(level_nodes[0], "val") or 0) if level_nodes else 0
        abstract_id = num_to_abstract.get(num_id or "", "")
        fmt = abstract_levels.get(abstract_id, {}).get(level, "decimal")
        overridden = (num_id or "") in overridden_nums
        handling = _main_body_handling(node)
        if handling == "SUPPORTED" and overridden:
            handling = "PARTIALLY_SUPPORTED"
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="list",
                subtype="bullet" if fmt == "bullet" else "numbered",
                current_handling=handling,
                risk_level="P1" if handling != "SUPPORTED" else "P3",
                notes="Basic numId/level formatting is serialized; explicit overrides are not interpreted.",
                metadata={
                    "num_id": num_id,
                    "abstract_num_id": abstract_id,
                    "level": level,
                    "format": fmt,
                    "multilevel": len(abstract_levels.get(abstract_id, {})) > 1,
                    "has_level_override": overridden,
                },
            )
        )

    if package.has_part("word/numbering.xml"):
        numbering = package.xml("word/numbering.xml")
        for node in numbering.xpath("./w:abstractNum", namespaces=NS):
            abstract_id = _attr(node, "abstractNumId") or ""
            levels = abstract_levels.get(abstract_id, {})
            records.append(
                _record(
                    package,
                    "word/numbering.xml",
                    node,
                    element_type="numbering_definition",
                    subtype="multilevel" if len(levels) > 1 else "single_level",
                    current_handling="PARTIALLY_SUPPORTED",
                    risk_level="P2",
                    notes="Production reads level formats but not full numbering semantics.",
                    metadata={"abstract_num_id": abstract_id, "levels": levels},
                )
            )
        for node in numbering.xpath("./w:num/w:lvlOverride", namespaces=NS):
            num = node.getparent()
            start_nodes = node.xpath("./w:startOverride", namespaces=NS)
            records.append(
                _record(
                    package,
                    "word/numbering.xml",
                    node,
                    element_type="numbering_override",
                    subtype="start_override" if start_nodes else "level_override",
                    current_handling="IGNORED_UNKNOWN",
                    risk_level="P1",
                    notes="Custom numbering restart/override is not interpreted.",
                    metadata={
                        "num_id": _attr(num, "numId") if num is not None else None,
                        "level": _attr(node, "ilvl"),
                        "start": _attr(start_nodes[0], "val") if start_nodes else None,
                    },
                )
            )

    for node in root.xpath(".//w:sectPr", namespaces=NS):
        type_nodes = node.xpath("./w:type", namespaces=NS)
        cols_nodes = node.xpath("./w:cols", namespaces=NS)
        section_type = (
            _attr(type_nodes[0], "val") if type_nodes else "nextPage_or_default"
        )
        column_count = int(_attr(cols_nodes[0], "num") or 1) if cols_nodes else 1
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="section",
                subtype=section_type or "nextPage_or_default",
                current_handling="IGNORED_INTENTIONALLY",
                risk_level="P2",
                notes="Section layout is not represented in DocumentIR.",
                metadata={
                    "columns": column_count,
                    "multiple_columns": column_count > 1,
                    "header_references": len(
                        node.xpath("./w:headerReference", namespaces=NS)
                    ),
                    "footer_references": len(
                        node.xpath("./w:footerReference", namespaces=NS)
                    ),
                },
            )
        )
    for node in root.xpath(
        ".//w:br[@w:type='page' or @w:type='column']", namespaces=NS
    ):
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="break",
                subtype=_attr(node, "type") or "line",
                current_handling=(
                    "NOT_REACHED_BY_CURRENT_SCANNER"
                    if _under(node, "txbxContent")
                    else "PARTIALLY_SUPPORTED"
                ),
                risk_level="P2",
                notes="Page/column break is flattened to a generic Markdown line break when reached.",
            )
        )
    for node in root.xpath(".//w:pageBreakBefore", namespaces=NS):
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="break",
                subtype="page_before",
                current_handling="IGNORED_INTENTIONALLY",
                risk_level="P2",
                notes="Paragraph page-break-before is layout-only in current Markdown.",
            )
        )

    for node in root.xpath(".//w:pict", namespaces=NS):
        if _is_unselected_fallback(node):
            continue
        subtype = (
            "textbox"
            if node.xpath(".//w:txbxContent", namespaces=NS)
            else ("image" if node.xpath(".//v:imagedata", namespaces=NS) else "shape")
        )
        rid = _first_relationship_id(node)
        rel = package.relationship(part_name, rid) if rid else None
        handling = (
            "PRESERVED_AS_ASSET"
            if subtype == "image" and rel and rel.target_part in package.parts
            else "NOT_REACHED_BY_CURRENT_SCANNER"
        )
        records.append(
            _record(
                package,
                part_name,
                node,
                element_type="vml",
                subtype=subtype,
                current_handling=handling,
                risk_level="P1"
                if node.xpath(".//w:txbxContent", namespaces=NS)
                else "P2",
                relationship_id=rid,
                relationship_target=rel.target_part if rel else None,
                notes="Selected VML picture/shape branch inventory.",
            )
        )

    for marker in ("commentRangeStart", "commentRangeEnd"):
        for node in root.xpath(f".//w:{marker}", namespaces=NS):
            records.append(
                _record(
                    package,
                    part_name,
                    node,
                    element_type="comment_marker",
                    subtype=marker,
                    current_handling="IGNORED_INTENTIONALLY",
                    risk_level="P3",
                    notes="Comment range marker has no standalone visible content.",
                    metadata={"id": _attr(node, "id")},
                )
            )


def _scan_package_relationship_edges(
    package: PackageIndex, records: list[dict[str, Any]]
) -> None:
    relevant = [
        ((owner, rid), rel)
        for (owner, rid), rel in sorted(package.relationships.items())
        if rel.relationship_type.rsplit("/", 1)[-1] in {"image", "package"}
    ]
    roots: dict[str, etree._Element] = {}
    references_by_owner: dict[str, dict[str, list[etree._Element]]] = {}
    for owner_part in {owner for (owner, _), _ in relevant}:
        if owner_part and package.has_part(owner_part) and owner_part.endswith(".xml"):
            try:
                root = package.xml(owner_part)
            except etree.XMLSyntaxError:
                continue
        else:
            root = package.xml("word/document.xml")
        roots[owner_part] = root
        local: dict[str, list[etree._Element]] = defaultdict(list)
        for node in root.iter():
            if not isinstance(node.tag, str):
                continue
            for attr_name in ("id", "embed", "link"):
                value = node.get(_qn(R_NS, attr_name))
                if value:
                    local[value].append(node)
        references_by_owner[owner_part] = local

    for (owner_part, rid), rel in relevant:
        root = roots.get(owner_part)
        if root is None:
            continue
        references = references_by_owner[owner_part].get(rid, [])
        suffix = rel.relationship_type.rsplit("/", 1)[-1]
        if suffix == "package":
            records.append(
                _record(
                    package,
                    owner_part or "_rels/.rels",
                    references[0] if references else root,
                    element_type="non_math_package",
                    subtype="external" if rel.external else "embedded",
                    current_handling="IGNORED_UNKNOWN",
                    risk_level="P1",
                    relationship_id=rid,
                    relationship_target=rel.target if rel.external else rel.target_part,
                    notes="Embedded/linked package relationship is not semantically parsed.",
                    metadata={
                        "referenced": bool(references),
                        "target_exists": rel.target_part in package.parts
                        if rel.target_part
                        else False,
                    },
                    visible_text="",
                )
            )
            continue
        selected_references = [
            node for node in references if not _is_unselected_fallback(node)
        ]
        formula_preview = any(_under(node, "object") for node in selected_references)
        if not references:
            subtype, handling, risk = "unreferenced", "IGNORED_INTENTIONALLY", "P2"
        elif not selected_references:
            subtype, handling, risk = "fallback_only", "IGNORED_INTENTIONALLY", "P2"
        elif rel.external:
            subtype, handling, risk = "linked", "IGNORED_UNKNOWN", "P1"
        elif owner_part == "word/document.xml":
            subtype = "formula_preview" if formula_preview else "main_body_embedded"
            handling, risk = "PRESERVED_AS_ASSET", "P3"
        elif owner_part.startswith(("word/header", "word/footer")):
            subtype, handling = "non_body_embedded", "NOT_REACHED_BY_CURRENT_SCANNER"
            container_subtype = _header_footer_subtype(root, owner_part)
            risk = _header_footer_risk(container_subtype)
        else:
            subtype, handling, risk = (
                "other_part_embedded",
                "NOT_REACHED_BY_CURRENT_SCANNER",
                "P1",
            )
        records.append(
            _record(
                package,
                owner_part or "_rels/.rels",
                selected_references[0]
                if selected_references
                else (references[0] if references else root),
                element_type="image_relationship",
                subtype=subtype,
                current_handling=handling,
                risk_level=risk,
                relationship_id=rid,
                relationship_target=rel.target if rel.external else rel.target_part,
                notes="Relationship-level embedded/linked image reachability audit.",
                metadata={
                    "external": rel.external,
                    "reference_count": len(references),
                    "selected_reference_count": len(selected_references),
                    "target_exists": rel.target_part in package.parts
                    if rel.target_part
                    else False,
                    "formula_preview": formula_preview,
                },
                visible_text="",
            )
        )


def _scan_chart_and_smartart_parts(
    package: PackageIndex, records: list[dict[str, Any]]
) -> None:
    for part_name in sorted(package.parts):
        if re.fullmatch(r"word/charts/chart\d+\.xml", part_name):
            root = package.xml(part_name)
            title = " ".join(
                root.xpath(".//c:title//a:t/text()", namespaces=NS)
            ).strip()
            workbook = next(
                (
                    rel.target_part
                    for (owner, _), rel in package.relationships.items()
                    if owner == part_name and rel.relationship_type.endswith("/package")
                ),
                None,
            )
            records.append(
                _record(
                    package,
                    part_name,
                    root,
                    element_type="chart",
                    subtype="structured_chart",
                    current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                    risk_level="P1",
                    notes="Chart visual/series semantics are not parsed.",
                    metadata={
                        "title": _short_text(title),
                        "has_title": bool(
                            title or root.xpath(".//c:title", namespaces=NS)
                        ),
                        "series_count": len(root.xpath(".//c:ser", namespaces=NS)),
                        "series_label_values": len(
                            root.xpath(".//c:tx//c:v", namespaces=NS)
                        ),
                        "embedded_workbook": workbook,
                    },
                )
            )
        elif part_name.startswith("word/diagrams/") and part_name.endswith(".xml"):
            root = package.xml(part_name)
            records.append(
                _record(
                    package,
                    part_name,
                    root,
                    element_type="smartart",
                    subtype=Path(part_name).stem,
                    current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                    risk_level="P1",
                    notes="SmartArt diagram semantics are not parsed.",
                )
            )


def scan_docx_edges(source: str | Path | PackageIndex) -> list[dict[str, Any]]:
    """Return deterministic edge-structure records for one DOCX package."""

    package = source if isinstance(source, PackageIndex) else PackageIndex(source)
    records: list[dict[str, Any]] = []
    if package.has_part("word/document.xml"):
        root = package.xml("word/document.xml")
        for node in root.xpath(".//w:txbxContent", namespaces=NS):
            if _is_unselected_fallback(node):
                continue
            text = _visible_text(node)
            records.append(
                _record(
                    package,
                    "word/document.xml",
                    node,
                    element_type="textbox",
                    subtype="w:txbxContent",
                    current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                    risk_level="P0" if text else "P2",
                    notes="Production object/image routing does not recurse into textbox text.",
                )
            )
        for node in root.xpath(".//w:txbxContent//m:oMath", namespaces=NS):
            if _is_unselected_fallback(node):
                continue
            records.append(
                _record(
                    package,
                    "word/document.xml",
                    node,
                    element_type="formula_outside_main_body",
                    subtype="textbox",
                    current_handling="NOT_REACHED_BY_CURRENT_SCANNER",
                    risk_level="P0",
                    notes="Formula is inside selected textbox content outside body flow.",
                )
            )
        for node in root.xpath(
            ".//w:headerReference | .//w:footerReference", namespaces=NS
        ):
            rid = node.get(_qn(R_NS, "id"))
            rel = package.relationship("word/document.xml", rid) if rid else None
            records.append(
                _record(
                    package,
                    "word/document.xml",
                    node,
                    element_type="section_reference",
                    subtype=local_name(node.tag),
                    current_handling="IGNORED_INTENTIONALLY",
                    risk_level="P2",
                    relationship_id=rid,
                    relationship_target=rel.target_part if rel else None,
                    notes="Section-scoped header/footer relationship.",
                    metadata={"type": _attr(node, "type")},
                )
            )
        _scan_document_semantics(package, root, records)
        _scan_tables(package, root, records)
        _scan_drawings(package, root, records)
        _scan_layout_and_numbering(package, root, records)
        for reference_name in (
            "footnoteReference",
            "endnoteReference",
            "commentReference",
        ):
            for node in root.xpath(f".//w:{reference_name}", namespaces=NS):
                records.append(
                    _record(
                        package,
                        "word/document.xml",
                        node,
                        element_type=re.sub(
                            r"(?<!^)(?=[A-Z])", "_", reference_name
                        ).lower(),
                        subtype=_attr(node, "id") or "",
                        current_handling="IGNORED_UNKNOWN",
                        risk_level="P0"
                        if reference_name != "commentReference"
                        else "P3",
                        notes="Reference marker is not represented in DocumentIR.",
                    )
                )
    _scan_non_body_parts(package, records)
    _scan_package_relationship_edges(package, records)
    _scan_chart_and_smartart_parts(package, records)
    return records


@dataclass(frozen=True)
class DocxEdgeCensusResult:
    output_dir: Path
    summary_path: Path
    edge_occurrences_path: Path
    edge_by_document_path: Path
    confirmed_content_loss_path: Path
    representative_cases_path: Path
    current_handling_matrix_path: Path
    regression_summary_path: Path
    summary: dict[str, Any]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _normalized_search_text(value: str) -> str:
    value = re.sub(r"\\([\\`*\[\]])", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def _current_markdown(
    package: PackageIndex,
    temp_root: Path,
    cache_context: MtefCacheContext,
) -> tuple[str, str | None]:
    report = _new_report(package.source)
    report["timing"]["package_read_seconds"] = package.read_seconds
    try:
        document = DocumentScanner(
            package,
            temp_root,
            report,
            analysis_mode=True,
            mtef_cache_context=cache_context,
        ).scan()
        _finalize_formula_report(report)
        return MarkdownSerializer().serialize(document), None
    except Exception as exc:  # noqa: BLE001 - diagnostic census must continue
        return "", f"{type(exc).__name__}: {exc}"


def _inventory(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["element_type"]].append(row)
    risk_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    result: dict[str, dict[str, Any]] = {}
    for edge_type in sorted(set(grouped) | set(REQUIRED_EDGE_TYPES)):
        local = grouped[edge_type]
        if not local:
            result[edge_type] = {
                "documents": 0,
                "occurrences": 0,
                "visible_content_occurrences": 0,
                "visible_text_characters": 0,
                "current_handling": {},
                "highest_risk": None,
                "subtypes": {},
            }
            continue
        result[edge_type] = {
            "documents": len({row["document_id"] for row in local}),
            "occurrences": len(local),
            "visible_content_occurrences": sum(
                bool(row["contains_text"]) for row in local
            ),
            "visible_text_characters": sum(row["visible_text_length"] for row in local),
            "current_handling": dict(
                sorted(Counter(row["current_handling"] for row in local).items())
            ),
            "highest_risk": min(
                (row["risk_level"] for row in local), key=risk_order.__getitem__
            ),
            "subtypes": dict(sorted(Counter(row["subtype"] for row in local).items())),
        }
    return result


def _handling_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix: list[dict[str, Any]] = []
    for edge_type, summary in _inventory(rows).items():
        matrix.append(
            {
                "edge_type": edge_type,
                "documents": summary["documents"],
                "occurrences": summary["occurrences"],
                "visible_content_occurrences": summary["visible_content_occurrences"],
                "current_handling": summary["current_handling"],
                "highest_risk": summary["highest_risk"],
            }
        )
    return {"schema": "bemarkdown-phase3d1-handling-matrix-v1", "edges": matrix}


def _reclassify_repeated_nonbody(rows: list[dict[str, Any]]) -> None:
    frequencies = Counter(
        (row["element_type"], row["visible_text"])
        for row in rows
        if row["element_type"] in {"header", "footer"} and row["visible_text"]
    )
    repeated_parts: set[tuple[str, str]] = set()
    for row in rows:
        if (
            row["element_type"] == "header"
            and row["subtype"] == "other_visible_header"
            and frequencies[("header", row["visible_text"])] >= 3
        ):
            row["subtype"] = "repeated_watermark_or_vendor_header"
            row["risk_level"] = "P3"
            row["notes"] = (
                "Repeated corpus-level header/watermark; not knowledge-body content."
            )
            repeated_parts.add((row["source_path"], row["part_name"]))
    for row in rows:
        if (row["source_path"], row["part_name"]) in repeated_parts and row[
            "element_type"
        ] in {"textbox", "drawing", "alternate_content", "image_relationship", "field"}:
            row["risk_level"] = "P3"


def _detail_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def selected(edge_type: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["element_type"] == edge_type]

    tables = selected("table")
    lists = selected("list")
    sections = selected("section")
    images = selected("image_relationship")
    textboxes = selected("textbox")
    return {
        "textbox": {
            "main_body_documents": len(
                {
                    row["document_id"]
                    for row in textboxes
                    if row["part_name"] == "word/document.xml"
                }
            ),
            "main_body_occurrences": sum(
                row["part_name"] == "word/document.xml" for row in textboxes
            ),
            "main_body_visible_occurrences": sum(
                row["part_name"] == "word/document.xml" and row["contains_text"]
                for row in textboxes
            ),
            "visible_occurrences": sum(row["contains_text"] for row in textboxes),
            "contains_formula": sum(row["contains_formula"] for row in textboxes),
            "contains_image": sum(row["contains_image"] for row in textboxes),
            "contains_table": sum(row["contains_table"] for row in textboxes),
        },
        "header_footer_classification": {
            edge: dict(
                sorted(Counter(row["subtype"] for row in selected(edge)).items())
            )
            for edge in ("header", "footer")
        },
        "fields": dict(
            sorted(Counter(row["subtype"] for row in selected("field")).items())
        ),
        "drawing_subtypes": dict(
            sorted(Counter(row["subtype"] for row in selected("drawing")).items())
        ),
        "table_complexity": {
            "tables": len(tables),
            "documents": len({row["document_id"] for row in tables}),
            "grid_span_occurrences": sum(
                row["metadata"].get("grid_span", 0) for row in tables
            ),
            "vertical_merge_occurrences": sum(
                row["metadata"].get("vertical_merge", 0) for row in tables
            ),
            "nested_table_occurrences": sum(
                row["subtype"] == "nested" for row in tables
            ),
            "max_nesting_depth": max(
                (row["metadata"].get("nesting_depth", 0) for row in tables),
                default=0,
            ),
            "formula_in_cell_tables": sum(
                row["metadata"].get("formula_in_cell", False) for row in tables
            ),
            "image_in_cell_tables": sum(
                row["metadata"].get("image_in_cell", False) for row in tables
            ),
            "list_in_cell_tables": sum(
                row["metadata"].get("list_in_cell", False) for row in tables
            ),
            "field_in_cell_tables": sum(
                row["metadata"].get("field_in_cell", False) for row in tables
            ),
            "max_paragraphs_per_cell": max(
                (row["metadata"].get("max_paragraphs_per_cell", 0) for row in tables),
                default=0,
            ),
        },
        "numbering": {
            "list_paragraphs": len(lists),
            "multilevel_paragraphs": sum(
                row["metadata"].get("multilevel", False) for row in lists
            ),
            "level_overrides": len(selected("numbering_override")),
        },
        "sections": {
            "sections": len(sections),
            "continuous": sum(row["subtype"] == "continuous" for row in sections),
            "multiple_columns": sum(
                row["metadata"].get("multiple_columns", False) for row in sections
            ),
        },
        "images": {
            "relationships": len(images),
            "embedded_main_body": sum(
                row["subtype"] in {"main_body_embedded", "formula_preview"}
                for row in images
            ),
            "linked": sum(row["subtype"] == "linked" for row in images),
            "unreferenced": sum(row["subtype"] == "unreferenced" for row in images),
            "fallback_only": sum(row["subtype"] == "fallback_only" for row in images),
            "non_body": sum(row["subtype"] == "non_body_embedded" for row in images),
            "missing_target": sum(
                not row["metadata"].get("target_exists", False)
                and not row["metadata"].get("external", False)
                for row in images
            ),
        },
    }


def _frozen_regression_summary(project_root: Path) -> dict[str, Any]:
    phase3c = (
        project_root
        / "artifacts"
        / "phase3c_mtef_performance"
        / "regression_summary.json"
    )
    seven = (
        project_root
        / "artifacts"
        / "phase3c_post_docx_regression_final"
        / "regression_summary.json"
    )
    expected = {
        "candidate_occurrences": 35920,
        "real_formula_occurrences": 35892,
        "empty_occurrences": 28,
        "structural_final": 35647,
        "ocr_candidates": 245,
        "ocr_states": {
            "OCR_ACCEPTED": 162,
            "OCR_ACCEPTED_WITH_WARNING": 12,
            "OCR_REVIEW_REQUIRED": 20,
            "OCR_REJECTED_PRESERVE_IMAGE": 51,
            "OCR_INFERENCE_FAILED_PRESERVE_IMAGE": 0,
        },
    }
    payload: dict[str, Any] = {
        "schema": "bemarkdown-phase3d1-regression-v1",
        "frozen_expected": expected,
        "phase3c_summary_path": str(phase3c.resolve()),
        "phase3c_summary_sha256": None,
        "phase3c_matches": False,
        "seven_docx_summary_path": str(seven.resolve()),
        "seven_docx_summary_sha256": None,
        "seven_docx_matches": False,
    }
    if phase3c.is_file():
        data = json.loads(phase3c.read_text(encoding="utf-8"))
        payload["phase3c_summary_sha256"] = hashlib.sha256(
            phase3c.read_bytes()
        ).hexdigest()
        payload["phase3c_observed"] = data
        payload["phase3c_matches"] = (
            data.get("docx", {}).get("total") == 169
            and data.get("docx", {}).get("success") == 169
            and data.get("docx", {}).get("source_sha_valid") == 169
            and all(
                data.get(key) == value
                for key, value in expected.items()
                if key != "ocr_states"
            )
            and data.get("phase3b_ocr_states") == expected["ocr_states"]
            and data.get("four_modes_zero_difference") is True
        )
    if seven.is_file():
        data = json.loads(seven.read_text(encoding="utf-8"))
        payload["seven_docx_summary_sha256"] = hashlib.sha256(
            seven.read_bytes()
        ).hexdigest()
        payload["seven_docx_observed"] = data
        payload["seven_docx_matches"] = (
            data.get("case_count") == 7
            and data.get("all_conservation_ok") is True
            and data.get("omml", {}).get("all_exact_match") is True
        )
    payload["all_frozen_regressions_match"] = (
        payload["phase3c_matches"] and payload["seven_docx_matches"]
    )
    return payload


def run_docx_edge_census(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> DocxEdgeCensusResult:
    """Audit every manifest DOCX and write the Phase 3D-1 evidence bundle."""

    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docx_manifest = [
        row
        for row in manifest.get("records", [])
        if str(row.get("file_type", "")).upper() == "DOCX"
    ]
    all_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    packages: dict[str, PackageIndex] = {}
    source_hash_valid = 0
    source_hash_mismatch = 0
    failures = 0

    for index, manifest_row in enumerate(docx_manifest, start=1):
        source = (
            manifest_path.parent / Path(manifest_row["target_relative_path"])
        ).resolve()
        expected_sha = manifest_row.get("source_sha256")
        base = {
            "document_index": index,
            "document_id": source.stem,
            "source_path": str(source),
            "source_relative_path": manifest_row.get("target_relative_path"),
            "expected_sha256": expected_sha,
        }
        try:
            data = source.read_bytes()
            actual_sha = hashlib.sha256(data).hexdigest()
            if expected_sha and actual_sha != expected_sha:
                source_hash_mismatch += 1
                failures += 1
                document_rows.append(
                    {
                        **base,
                        "status": "SOURCE_HASH_MISMATCH",
                        "actual_sha256": actual_sha,
                    }
                )
                continue
            source_hash_valid += 1
            package = PackageIndex(source)
            source_key = str(manifest_row.get("target_relative_path"))
            packages[source_key] = package
            local = scan_docx_edges(package)
            for row in local:
                row.update(
                    {
                        "document_id": f"docx-{index:04d}",
                        "document_index": index,
                        "source_path": source_key,
                        "source_sha256": actual_sha,
                    }
                )
            all_rows.extend(local)
            local_inventory = _inventory(local)
            document_rows.append(
                {
                    **base,
                    "status": "SUCCESS",
                    "actual_sha256": actual_sha,
                    "edge_occurrences": len(local),
                    "risk_counts": dict(
                        sorted(Counter(row["risk_level"] for row in local).items())
                    ),
                    "edge_types": local_inventory,
                }
            )
        except Exception as exc:  # noqa: BLE001 - record failure and continue corpus
            failures += 1
            document_rows.append(
                {**base, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
            )

    _reclassify_repeated_nonbody(all_rows)
    rows_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        rows_by_index[row["document_index"]].append(row)
    for document_row in document_rows:
        if document_row["status"] != "SUCCESS":
            continue
        local = rows_by_index[document_row["document_index"]]
        document_row["risk_counts"] = dict(
            sorted(Counter(row["risk_level"] for row in local).items())
        )
        document_row["edge_types"] = _inventory(local)

    verification_candidates = [
        row
        for row in all_rows
        if (row["contains_text"] or row["subtype"] == "linked")
        and row["risk_level"] in {"P0", "P1"}
        and row["current_handling"]
        in {
            "NOT_REACHED_BY_CURRENT_SCANNER",
            "PARTIALLY_SUPPORTED",
            "IGNORED_UNKNOWN",
        }
        and row["element_type"] != "alternate_content"
    ]
    verification_sources = {row["source_path"] for row in verification_candidates}
    markdown_by_source: dict[str, tuple[str, str | None]] = {}
    with tempfile.TemporaryDirectory(prefix="bemarkdown-phase3d1-") as temp:
        temp_root = Path(temp)
        cache_context = MtefCacheContext(mode="persistent")
        for serial, source_text in enumerate(sorted(verification_sources), start=1):
            markdown_by_source[source_text] = _current_markdown(
                packages[source_text], temp_root / f"{serial:04d}", cache_context
            )

    confirmed: list[dict[str, Any]] = []
    for row in verification_candidates:
        markdown, error = markdown_by_source[row["source_path"]]
        linked_image = (
            row["element_type"] == "image_relationship" and row["subtype"] == "linked"
        )
        needle = _normalized_search_text(
            row["relationship_target"] if linked_image else row["visible_text"]
        )
        haystack = _normalized_search_text(markdown)
        present = bool(needle and needle in haystack)
        evidence = {
            "comparison": (
                "external image relationship target against production DocumentIR Markdown"
                if linked_image
                else "normalized exact visible-text excerpt against production DocumentIR Markdown"
            ),
            "visible_text_present": present,
            "outcome": (
                "PRESENT"
                if present and len(needle) >= 8
                else "AMBIGUOUS_GLOBAL_TEXT_COLLISION"
                if present
                else "ABSENT"
            ),
            "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "markdown_characters": len(markdown),
            "scanner_error": error,
        }
        row["current_output_evidence"] = evidence
        if (
            not present
            and error is None
            and row["current_handling"] != "PARTIALLY_SUPPORTED"
            and (row["element_type"] != "textbox" or len(needle) >= 8)
            and not (
                row["element_type"] == "drawing"
                and row["subtype"].startswith(("header_", "footer_"))
            )
        ):
            confirmed.append(
                {
                    "document": row["source_path"],
                    "source_sha256": row["source_sha256"],
                    "part": row["part_name"],
                    "locator": row["locator"],
                    "edge_type": row["element_type"],
                    "subtype": row["subtype"],
                    "visible_content_summary": (
                        f"Linked image: {row['relationship_target']}"
                        if linked_image
                        else row["visible_text"]
                    ),
                    "current_output_evidence": evidence,
                    "risk": row["risk_level"],
                    "current_handling": row["current_handling"],
                }
            )

    risk_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    representatives: list[dict[str, Any]] = []
    by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_edge[row["element_type"]].append(row)
    for edge_type in sorted(by_edge):
        local = sorted(
            by_edge[edge_type],
            key=lambda row: (
                risk_order[row["risk_level"]],
                not row["contains_text"],
                -row["visible_text_length"],
                row["source_path"],
                row["locator"],
            ),
        )
        seen_subtypes: set[str] = set()
        for row in local:
            if row["subtype"] in seen_subtypes:
                continue
            representatives.append(
                {
                    key: row.get(key)
                    for key in (
                        "document_id",
                        "source_path",
                        "source_sha256",
                        "part_name",
                        "element_type",
                        "subtype",
                        "locator",
                        "visible_text",
                        "visible_text_length",
                        "contains_formula",
                        "contains_image",
                        "contains_table",
                        "relationship_id",
                        "relationship_target",
                        "current_handling",
                        "risk_level",
                        "notes",
                        "metadata",
                        "current_output_evidence",
                    )
                    if key in row
                }
            )
            seen_subtypes.add(row["subtype"])
            if len(seen_subtypes) >= 3:
                break

    inventory = _inventory(all_rows)
    risk_counts = dict(sorted(Counter(row["risk_level"] for row in all_rows).items()))
    confirmed_counts = dict(sorted(Counter(row["risk"] for row in confirmed).items()))
    regression = _frozen_regression_summary(Path(__file__).resolve().parents[2])
    summary = {
        "schema": "bemarkdown-phase3d1-docx-edge-census-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "corpus": {
            "docx_total": len(docx_manifest),
            "docx_success": len(docx_manifest) - failures,
            "docx_failed": failures,
            "source_sha_valid": source_hash_valid,
            "source_sha_mismatch": source_hash_mismatch,
        },
        "edge_occurrences": len(all_rows),
        "inventory": inventory,
        "risk_counts": risk_counts,
        "details": _detail_summaries(all_rows),
        "verification": {
            "candidate_occurrences": len(verification_candidates),
            "documents_compared": len(verification_sources),
            "scanner_errors": sum(
                bool(error) for _, error in markdown_by_source.values()
            ),
        },
        "confirmed_content_loss": {
            risk: confirmed_counts.get(risk, 0) for risk in ("P0", "P1", "P2", "P3")
        },
        "all_ok": failures == 0 and not source_hash_mismatch,
    }

    summary_path = output_dir / "census_summary.json"
    occurrences_path = output_dir / "edge_occurrences.jsonl"
    by_document_path = output_dir / "edge_by_document.jsonl"
    confirmed_path = output_dir / "confirmed_content_loss.jsonl"
    representatives_path = output_dir / "representative_cases.jsonl"
    handling_path = output_dir / "current_handling_matrix.json"
    regression_path = output_dir / "regression_summary.json"
    _write_json(summary_path, summary)
    _write_jsonl(occurrences_path, all_rows)
    _write_jsonl(by_document_path, document_rows)
    _write_jsonl(confirmed_path, confirmed)
    _write_jsonl(representatives_path, representatives)
    _write_json(handling_path, _handling_matrix(all_rows))
    _write_json(regression_path, regression)
    return DocxEdgeCensusResult(
        output_dir=output_dir,
        summary_path=summary_path,
        edge_occurrences_path=occurrences_path,
        edge_by_document_path=by_document_path,
        confirmed_content_loss_path=confirmed_path,
        representative_cases_path=representatives_path,
        current_handling_matrix_path=handling_path,
        regression_summary_path=regression_path,
        summary=summary,
    )
