from __future__ import annotations

import hashlib
import mimetypes
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from lxml import etree

from .drawingml import (
    DECORATIVE_OR_EMPTY,
    PURE_TEXT_CONTAINER,
    UNSUPPORTED_VISUAL_GROUP,
    VISUAL_DIAGRAM_GROUP,
    inspect_drawingml,
)
from .formula import (
    FormulaConversion,
    convert_eq_field,
    convert_mathml,
    convert_mtef_payload,
    convert_omml,
    detect_latex_alt,
)
from .ir import (
    BreakNode,
    DocumentIR,
    FormulaNode,
    HeadingNode,
    HyperlinkNode,
    ImageNode,
    InlineNode,
    ListItemNode,
    ParagraphNode,
    TableCell,
    TableNode,
    TextNode,
)
from .mtef_cache import MtefCacheContext, MtefCacheResolution
from .namespaces import NS, local_name, qn
from .package import InvalidDocxError, PackageIndex
from .validator import FormulaStructuralValidator, FormulaValidation, FormulaVerdict
from .wmf import WmfInspection, WmfInspector
from .wmf_renderer import WmfRenderer


@dataclass
class FieldFrame:
    instruction: list[str] = field(default_factory=list)
    separated: bool = False
    result: list[InlineNode] = field(default_factory=list)


class AssetExporter:
    def __init__(
        self,
        output_dir: Path,
        package: PackageIndex,
        report: dict,
        *,
        analysis_mode: bool = False,
    ):
        self.output_dir = output_dir
        self.package = package
        self.report = report
        self.analysis_mode = analysis_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ordinary = 0
        self._formula = 0
        self._visual = 0
        self._cache: dict[tuple[str, str], str] = {}
        self._generated_cache: dict[
            tuple[str, tuple[int, int] | None], tuple[str, dict]
        ] = {}

    def export(self, part: str, role: str = "image") -> str:
        cache_key = (part, role)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self.analysis_mode:
            reference = f"package://{part}"
            self._cache[cache_key] = reference
            return reference
        started = time.perf_counter()
        suffix = PurePosixPath(part).suffix.lower() or ".bin"
        if role == "image":
            self._ordinary += 1
            name = f"image_{self._ordinary:04d}{suffix}"
        else:
            self._formula += 1
            name = f"formula_{self._formula:04d}{suffix}"
        target = self.output_dir / name
        target.write_bytes(self.package.data(part))
        relative = f".bmd-staging/{name}"
        self._cache[cache_key] = relative
        self.report["timing"]["asset_export_seconds"] += time.perf_counter() - started
        self.report["assets"]["exported_assets"] += 1
        self.report["assets"]["records"].append(
            {
                "asset_path": relative,
                "source_part": part,
                "role": role,
                "bytes": target.stat().st_size,
                "content_type": self.package.content_types.get(part)
                or mimetypes.guess_type(name)[0]
                or "application/octet-stream",
            }
        )
        return relative

    def generated_formula_target(self) -> tuple[Path, str]:
        name = f"formula_{self._formula + 1:04d}.png"
        return self.output_dir / name, f".bmd-staging/{name}"

    def stage_drawingml_svg(self, payload: bytes) -> str:
        self._visual += 1
        staging = self.output_dir
        staging.mkdir(parents=True, exist_ok=True)
        name = f"drawingml_{self._visual:06d}.svg"
        target = staging / name
        target.write_bytes(payload)
        return f".bmd-staging/{name}"

    def cached_generated(
        self, source_part: str, extent_emu: tuple[int, int] | None
    ) -> tuple[str, dict] | None:
        return self._generated_cache.get((source_part, extent_emu))

    def record_generated_formula(
        self,
        source_part: str,
        extent_emu: tuple[int, int] | None,
        target: Path,
        relative: str,
        metadata: dict,
    ) -> None:
        self._formula += 1
        self._generated_cache[(source_part, extent_emu)] = (relative, metadata)
        self.report["assets"]["exported_assets"] += 1
        self.report["assets"]["records"].append(
            {
                "asset_path": relative,
                "source_part": source_part,
                "role": "rendered_formula",
                "bytes": target.stat().st_size,
                "content_type": "image/png",
                "renderer_metadata": metadata,
            }
        )


class DocumentScanner:
    def __init__(
        self,
        package: PackageIndex,
        output_dir: Path,
        report: dict,
        *,
        analysis_mode: bool = False,
        mtef_cache_context: MtefCacheContext | None = None,
    ):
        self.package = package
        self.report = report
        self.owner_part = "word/document.xml"
        self.assets = AssetExporter(
            output_dir / ".bmd-staging",
            package,
            report,
            analysis_mode=analysis_mode,
        )
        self.styles = self._load_styles()
        self.numbering = self._load_numbering()
        self.list_counters: dict[tuple[str, int], int] = {}
        self.formula_counter = 0
        self.locator_counter = 0
        self.field_stack: list[FieldFrame] = []
        self.wmf_inspector = WmfInspector()
        self.wmf_renderer = WmfRenderer()
        self.formula_validator = FormulaStructuralValidator()
        self.mtef_cache_context = mtef_cache_context or MtefCacheContext()

    def scan(self) -> DocumentIR:
        root = self.package.xml(self.owner_part)
        body = root.find("w:body", NS)
        if body is None:
            raise ValueError("word/document.xml has no w:body")
        blocks = self._body_blocks(body)

        if self.field_stack:
            for frame in self.field_stack:
                self.report["warnings"].append(
                    "Unclosed field preserved from its visible result: "
                    + "".join(frame.instruction)
                )
            self.field_stack.clear()

        metadata, title_blocks = self._header_titles(blocks)
        self._inventory_edges(root)
        return DocumentIR(title_blocks + blocks, metadata=metadata)

    def _body_blocks(self, container: etree._Element) -> list:
        blocks = []
        for child in container:
            name = local_name(child.tag)
            if name == "p":
                blocks.append(self._paragraph(child))
            elif name == "tbl":
                blocks.append(self._table(child))
            elif name in {"oMath", "oMathPara"}:
                locator = self._locator(child)
                blocks.append(
                    ParagraphNode(
                        [self._omml_formula(child, locator, name == "oMathPara")]
                    )
                )
                self.report["document"]["paragraphs"] += 1
            elif name == "AlternateContent":
                blocks.extend(self._body_blocks(self._alternate_branch(child)))
            elif name in {"sdt", "sdtContent", "customXml", "ins"}:
                blocks.extend(self._body_blocks(child))
            elif name not in {"sectPr", "bookmarkStart", "bookmarkEnd", "sdtPr"}:
                self._unsupported("body_element", self.owner_part, name)
        return blocks

    def _paragraph(self, node: etree._Element) -> ParagraphNode:
        self.report["document"]["paragraphs"] += 1
        ppr = node.find("w:pPr", NS)
        style_id = None
        if ppr is not None:
            style = ppr.find("w:pStyle", NS)
            style_id = style.get(qn("w", "val")) if style is not None else None
        style_name = self.styles.get(style_id, style_id) if style_id else None
        locator = self._locator(node)
        children = self._scan_inline_children(node, locator)

        heading_level = self._heading_level(style_id, style_name)
        if heading_level:
            self.report["document"]["headings"] += 1
            return HeadingNode(children, style_id, style_name, level=heading_level)

        num_id, level = self._numbering_properties(ppr)
        if num_id is not None:
            ordered = self.numbering.get((num_id, level), "decimal") != "bullet"
            ordinal = None
            if ordered:
                key = (num_id, level)
                self.list_counters[key] = self.list_counters.get(key, 0) + 1
                ordinal = self.list_counters[key]
            self.report["document"]["list_items"] += 1
            return ListItemNode(
                children,
                style_id,
                style_name,
                level=level,
                ordered=ordered,
                ordinal=ordinal,
                numbering_id=num_id,
            )
        return ParagraphNode(children, style_id, style_name)

    def _scan_inline_children(
        self, container: etree._Element, locator: str
    ) -> list[InlineNode]:
        output: list[InlineNode] = []
        for child in container:
            if local_name(child.tag) == "pPr":
                continue
            self._inline_element(child, locator, output, bold=False, italic=False)
        return output

    def _inline_element(
        self,
        node: etree._Element,
        locator: str,
        output: list[InlineNode],
        bold: bool,
        italic: bool,
    ) -> None:
        name = local_name(node.tag)
        if name == "r":
            rpr = node.find("w:rPr", NS)
            run_bold = bold or self._truthy_property(rpr, "b")
            run_italic = italic or self._truthy_property(rpr, "i")
            align = rpr.find("w:vertAlign", NS) if rpr is not None else None
            vertical_align = align.get(qn("w", "val"), "baseline") if align is not None else "baseline"
            if vertical_align not in {"superscript", "subscript"}:
                vertical_align = "baseline"
            for child in node:
                if local_name(child.tag) != "rPr":
                    self._run_child(child, locator, output, run_bold, run_italic, vertical_align)
            return
        if name == "hyperlink":
            rel_id = node.get(qn("r", "id"))
            anchor = node.get(qn("w", "anchor"))
            target = f"#{anchor}" if anchor else ""
            if rel_id:
                rel = self.package.relationship(self.owner_part, rel_id)
                if rel:
                    target = (
                        rel.target if rel.external else (rel.target_part or rel.target)
                    )
            inner: list[InlineNode] = []
            for child in node:
                self._inline_element(child, locator, inner, bold, italic)
            if target:
                self._emit(HyperlinkNode(target, inner), output)
            else:
                for item in inner:
                    self._emit(item, output)
            return
        if name in {"oMath", "oMathPara"}:
            self._emit(self._omml_formula(node, locator, name == "oMathPara"), output)
            return
        if name == "fldSimple":
            instruction = node.get(qn("w", "instr"), "")
            if instruction.lstrip().upper().startswith("EQ"):
                self._emit(self._eq_formula(instruction, locator), output)
            else:
                for child in node:
                    self._inline_element(child, locator, output, bold, italic)
            return
        if name == "AlternateContent":
            branch = self._alternate_branch(node)
            for child in branch:
                self._inline_element(child, locator, output, bold, italic)
            return
        if name in {"smartTag", "sdt", "sdtContent", "customXml", "ins"}:
            for child in node:
                self._inline_element(child, locator, output, bold, italic)
            return
        if name in {"object", "drawing", "pict"}:
            self._object_or_image(node, locator, output)
            return
        if name not in {
            "bookmarkStart",
            "bookmarkEnd",
            "proofErr",
            "commentRangeStart",
            "commentRangeEnd",
        }:
            self._unsupported("inline_element", self.owner_part, name)

    def _run_child(
        self,
        node: etree._Element,
        locator: str,
        output: list[InlineNode],
        bold: bool,
        italic: bool,
        vertical_align: str = "baseline",
    ) -> None:
        name = local_name(node.tag)
        if name == "fldChar":
            field_type = node.get(qn("w", "fldCharType"), "")
            if field_type == "begin":
                self.field_stack.append(FieldFrame())
            elif field_type == "separate" and self.field_stack:
                self.field_stack[-1].separated = True
            elif field_type == "end" and self.field_stack:
                frame = self.field_stack.pop()
                instruction = "".join(frame.instruction).strip()
                if instruction.upper().startswith("EQ"):
                    self._emit(self._eq_formula(instruction, locator), output)
                else:
                    for item in frame.result:
                        self._emit(item, output)
            return
        if name == "instrText":
            if self.field_stack and not self.field_stack[-1].separated:
                fragment = node.text or ""
                # Word can apply run scripts inside an EQ instruction (for
                # example, the A in \f(mAg,cos30)). Preserve scalar fragments
                # with an equivalent native EQ script command. Do not rewrite
                # instruction syntax or non-EQ fields.
                prefix = "".join(self.field_stack[-1].instruction).lstrip().upper()
                if (vertical_align in {"subscript", "superscript"}
                        and prefix.startswith("EQ") and fragment.strip()
                        and not any(char in fragment for char in "\\,(){}")):
                    direction = "do3" if vertical_align == "subscript" else "up3"
                    fragment = f"\\s\\{direction}({fragment})"
                self.field_stack[-1].instruction.append(fragment)
            return
        if name == "t":
            self._emit(TextNode(node.text or "", bold, italic, vertical_align), output)
            return
        if name == "tab":
            self._emit(TextNode("\t", bold, italic, vertical_align), output)
            return
        if name in {"br", "cr"}:
            self._emit(BreakNode(), output)
            return
        if name in {"oMath", "oMathPara"}:
            self._emit(self._omml_formula(node, locator, name == "oMathPara"), output)
            return
        if name in {"object", "drawing", "pict"}:
            self._object_or_image(node, locator, output)
            return
        if name == "AlternateContent":
            branch = self._alternate_branch(node)
            for child in branch:
                self._run_child(child, locator, output, bold, italic, vertical_align)
            return
        if name in {"sym"}:
            char = node.get(qn("w", "char"), "")
            try:
                value = chr(int(char, 16))
            except ValueError:
                value = f"[symbol:{char}]"
            self._emit(TextNode(value, bold, italic, vertical_align), output)
            return
        if name not in {"lastRenderedPageBreak", "softHyphen", "noBreakHyphen"}:
            for child in node:
                self._run_child(child, locator, output, bold, italic, vertical_align)

    def _emit(self, node: InlineNode, output: list[InlineNode]) -> None:
        if self.field_stack:
            frame = self.field_stack[-1]
            if frame.separated:
                frame.result.append(node)
            return
        output.append(node)

    def _omml_formula(
        self, node: etree._Element, locator: str, block: bool
    ) -> FormulaNode:
        locator = self._locator(node)
        started = time.perf_counter()
        conversion = convert_omml(node)
        elapsed = time.perf_counter() - started
        self.report["timing"]["formula_conversion_seconds"] += elapsed
        self.report["timing"]["omml_seconds"] += elapsed
        canonical = etree.tostring(node, method="c14n")
        return self._formula_node(
            "omml",
            locator,
            "block" if block else "inline",
            conversion,
            conversion_path=["OMML", conversion.component, "LaTeX"],
            candidate_evidence={
                "candidate_sources": self._candidate_sources(omml=True),
                "source_payload_sha256": hashlib.sha256(canonical).hexdigest(),
            },
        )

    def _eq_formula(self, source: str, locator: str) -> FormulaNode:
        started = time.perf_counter()
        conversion = convert_eq_field(source)
        elapsed = time.perf_counter() - started
        self.report["timing"]["formula_conversion_seconds"] += elapsed
        self.report["timing"]["eq_seconds"] += elapsed
        return self._formula_node(
            "eq",
            locator,
            "inline",
            conversion,
            original_ref=source,
            conversion_path=["Word EQ Field", "Tokenizer", "Recursive AST", "LaTeX"],
            candidate_evidence={
                "candidate_sources": self._candidate_sources(eq=True),
                "source_payload_sha256": hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest(),
            },
        )

    def _object_or_image(
        self, node: etree._Element, locator: str, output: list[InlineNode]
    ) -> None:
        locator = self._locator(node)
        ole = node.find(".//o:OLEObject", NS)
        if ole is not None:
            self._ole_formula(node, ole, locator, output)
            return
        if node.xpath(".//w:txbxContent", namespaces=NS):
            self._drawingml_object(node, locator, output)
            return
        blips = node.xpath(".//a:blip", namespaces=NS)
        vml_images = node.xpath(".//v:imagedata", namespaces=NS)
        image_node = blips[0] if blips else (vml_images[0] if vml_images else None)
        if image_node is None:
            if node.xpath(".//w:txbxContent", namespaces=NS):
                self._unsupported("textbox", self.owner_part, locator)
            return
        link_rid = image_node.get(qn("r", "link"))
        embedded_rid = image_node.get(qn("r", "embed"))
        rid = link_rid or embedded_rid or image_node.get(qn("r", "id"))
        if not rid:
            self.report["assets"]["failed_assets"] += 1
            self.report["warnings"].append(
                f"Image without relationship id at {locator}"
            )
            return
        rel = self.package.relationship(self.owner_part, rid)
        if rel is not None and rel.external:
            self.report["assets"]["source_image_objects"] += 1
            alt, title = self._image_metadata(node)
            self._emit(
                ImageNode(
                    None,
                    self.owner_part,
                    rid,
                    None,
                    alt=alt,
                    title=title,
                    source_locator=locator,
                    semantic_type="EXTERNAL_LINKED_IMAGE",
                    status="UNRESOLVED_EXTERNAL",
                    external_target=rel.target,
                    render_method="external_reference_only",
                    provenance={
                        "target_mode": "External",
                        "embedded_fallback_relationship_id": embedded_rid,
                    },
                ),
                output,
            )
            self.report["warnings"].append(
                f"External image preserved without access: {rid} at {locator}"
            )
            return
        if rel is None or rel.target_part not in self.package.parts:
            self.report["assets"]["failed_assets"] += 1
            self.report["warnings"].append(
                f"Missing image relationship {rid} at {locator}"
            )
            return
        self.report["assets"]["source_image_objects"] += 1
        alt, title = self._image_metadata(node)
        media_warning = _media_payload_warning(
            rel.target_part, self.package.data(rel.target_part)
        )
        if media_warning:
            self.report["warnings"].append(media_warning)
        alt_started = time.perf_counter()
        confidence, latex = detect_latex_alt(alt)
        self.report["timing"]["latex_alt_seconds"] += time.perf_counter() - alt_started
        if latex is not None:
            conversion = type(
                "AltConversion",
                (),
                {
                    "latex": latex,
                    "status": "SUCCESS_EXACT",
                    "warnings": [f"LaTeX alt detector confidence: {confidence}"],
                    "error": None,
                    "component": "bemarkdown conservative LaTeX alt detector",
                },
            )()
            formula = self._formula_node(
                "latex_alt",
                locator,
                "inline",
                conversion,
                original_ref=rel.target_part,
                preview_ref=rel.target_part,
                conversion_path=[
                    "DrawingML alternative text",
                    "LaTeX detector",
                    "LaTeX",
                ],
                candidate_evidence={
                    "candidate_sources": self._candidate_sources(
                        latex_alt=True, preview=True
                    ),
                    "source_payload_sha256": hashlib.sha256(
                        latex.encode("utf-8")
                    ).hexdigest(),
                    "preview_sha256": hashlib.sha256(
                        self.package.data(rel.target_part)
                    ).hexdigest(),
                    "preview_relationship_id": rid,
                },
            )
            self.report["assets"]["formula_previews_suppressed"] += 1
            self.report["formulas"]["duplicate_preview_suppressed"] += 1
            self._emit(formula, output)
            return
        asset_path = self.assets.export(rel.target_part, "image")
        semantic_type = "VML_IMAGE" if vml_images and not blips else "EMBEDDED_IMAGE"
        self._emit(
            ImageNode(
                asset_path,
                self.owner_part,
                rid,
                rel.target_part,
                alt=alt,
                title=title,
                source_locator=locator,
                semantic_type=semantic_type,
            ),
            output,
        )

    def _drawingml_object(
        self, node: etree._Element, locator: str, output: list[InlineNode]
    ) -> None:
        result = inspect_drawingml(node)
        drawingml = self.report["drawingml"]
        drawingml["groups_total"] += 1
        drawingml["visible_labels_total"] += len(result.visible_labels)
        drawingml["visible_labels_accounted"] += len(result.visible_labels)
        field = {
            VISUAL_DIAGRAM_GROUP: "visual_groups",
            PURE_TEXT_CONTAINER: "pure_text_groups",
            DECORATIVE_OR_EMPTY: "decorative_or_empty_groups",
            UNSUPPORTED_VISUAL_GROUP: "unsupported_groups",
        }[result.classification]
        drawingml[field] += 1
        record = {
            "source_part": self.owner_part,
            "source_locator": locator,
            "classification": result.classification,
            "visible_labels": list(result.visible_labels),
            "visible_label_count": len(result.visible_labels),
            "features": result.feature_matrix,
            "unsupported_features": list(result.unsupported_features),
        }
        drawingml["records"].append(record)
        if result.classification == PURE_TEXT_CONTAINER:
            for index, label in enumerate(result.visible_labels):
                if index:
                    self._emit(BreakNode(), output)
                self._emit(TextNode(label), output)
            return
        if result.classification == DECORATIVE_OR_EMPTY:
            return
        if result.svg is not None:
            reference = self.assets.stage_drawingml_svg(result.svg)
            status = "RESOLVED"
            render_method = "drawingml_svg"
        else:
            reference = None
            status = "UNSUPPORTED_VISUAL_GROUP"
            render_method = "unsupported_explicit_review"
            self.report["warnings"].append(
                f"Unsupported DrawingML visual group at {locator}: "
                + ", ".join(result.unsupported_features)
            )
        self._emit(
            ImageNode(
                reference,
                self.owner_part,
                "",
                None,
                source_locator=locator,
                semantic_type="DRAWINGML_GROUP",
                status=status,
                render_method=render_method,
                provenance={
                    "selected_branch": "Choice",
                    "classification": result.classification,
                    "visible_labels": list(result.visible_labels),
                    "unsupported_features": list(result.unsupported_features),
                },
            ),
            output,
        )

    def _ole_formula(
        self,
        object_node: etree._Element,
        ole: etree._Element,
        locator: str,
        output: list[InlineNode],
    ) -> None:
        prog_id = ole.get("ProgID") or ""
        ole_rid = ole.get(qn("r", "id"))
        preview_nodes = object_node.xpath(".//v:imagedata", namespaces=NS)
        preview_rid = preview_nodes[0].get(qn("r", "id")) if preview_nodes else None
        relationship_started = time.perf_counter()
        ole_rel = (
            self.package.relationship(self.owner_part, ole_rid) if ole_rid else None
        )
        preview_rel = (
            self.package.relationship(self.owner_part, preview_rid)
            if preview_rid
            else None
        )
        self.report["timing"]["ole_relationship_lookup_seconds"] += (
            time.perf_counter() - relationship_started
        )
        self.report["assets"]["source_image_objects"] += int(preview_rel is not None)
        preview_ref = preview_rel.target_part if preview_rel else None
        original_ref = ole_rel.target_part if ole_rel else ole_rid
        bytes_started = time.perf_counter()
        ole_data = (
            self.package.data(ole_rel.target_part)
            if ole_rel and ole_rel.target_part in self.package.parts
            else None
        )
        self.report["timing"]["ole_bytes_access_seconds"] += (
            time.perf_counter() - bytes_started
        )
        preview_data = (
            self.package.data(preview_rel.target_part)
            if preview_rel and preview_rel.target_part in self.package.parts
            else None
        )
        preview_is_wmf = bool(
            preview_rel
            and preview_rel.target_part in self.package.parts
            and PurePosixPath(preview_rel.target_part).suffix.lower() == ".wmf"
        )
        if preview_is_wmf:
            self.report["formulas"]["wmf_previews_discovered"] += 1
        evidence = {
            "prog_id": prog_id,
            "semantic_state": "unknown",
            "preview_visual_state": "not_inspected",
            "wmf_inspection": None,
            "structural_attempts": [],
            "candidate_sources": self._candidate_sources(
                ole=ole_data is not None, preview=preview_data is not None
            ),
            "ole_relationship_id": ole_rid,
            "preview_relationship_id": preview_rid,
            "ole_sha256": hashlib.sha256(ole_data).hexdigest() if ole_data else None,
            "wmf_sha256": (
                hashlib.sha256(preview_data).hexdigest()
                if preview_data is not None and preview_is_wmf
                else None
            ),
            "preview_suppressed": False,
            "preview_suppressed_by_ole": False,
        }

        normalized_prog_id = prog_id.lower()
        if prog_id and not (
            normalized_prog_id.startswith("equation.")
            or "mathtype" in normalized_prog_id
        ):
            self._record_candidate_without_node(
                "non_formula_object",
                locator,
                "NON_FORMULA_OBJECT",
                original_ref=original_ref,
                preview_ref=preview_ref,
                prog_id=prog_id,
                candidate_sources=evidence["candidate_sources"],
                ole_relationship_id=ole_rid,
                preview_relationship_id=preview_rid,
                ole_sha256=evidence["ole_sha256"],
                wmf_sha256=evidence["wmf_sha256"],
            )
            if preview_rel and preview_rel.target_part in self.package.parts:
                asset = self.assets.export(preview_rel.target_part, "image")
                self._emit(
                    ImageNode(
                        asset,
                        self.owner_part,
                        preview_rid or "",
                        preview_rel.target_part,
                        source_locator=locator,
                        semantic_type="OTHER_VISUAL",
                    ),
                    output,
                )
            return

        if ole_data is None:
            conversion = FormulaConversion(
                None,
                "FAILED_PRESERVED",
                "mathtypejx",
                [],
                f"Missing OLE relationship: {ole_rid}",
                semantic_state="unknown",
            )
        else:
            started = time.perf_counter()
            resolution = self.mtef_cache_context.convert_ole(
                ole_data,
                {
                    "referenced": True,
                    "progid": prog_id,
                    "source_type": "mtef",
                },
            )
            conversion = resolution.conversion
            elapsed = time.perf_counter() - started
            self.report["timing"]["formula_conversion_seconds"] += elapsed
            self.report["timing"]["ole_mtef_seconds"] += elapsed
            self._record_mtef_resolution(resolution)

        evidence["semantic_state"] = conversion.semantic_state
        if ole_data is None:
            direct_validation = self._validate_structural(
                conversion,
                source_type="mtef",
                source_metadata={"referenced": True, "progid": prog_id},
            )
        else:
            direct_validation = resolution.validation
            self._record_structural_validation(direct_validation)
            evidence["mtef_payload_sha256"] = resolution.mtef_sha256
            evidence["mtef_cache_level"] = resolution.cache_level
        if conversion.latex and direct_validation and direct_validation.accepted:
            semantic_payload = conversion.intermediate or conversion.latex
            evidence["source_payload_sha256"] = hashlib.sha256(
                semantic_payload.encode("utf-8")
            ).hexdigest()
            evidence["source_payload_type"] = "OLE_MTEF_MATHML"
            if preview_rel:
                self.report["assets"]["formula_previews_suppressed"] += 1
                self.report["formulas"]["duplicate_preview_suppressed"] += 1
                evidence["preview_suppressed"] = True
            if preview_is_wmf:
                self.report["formulas"]["preview_suppressed_by_ole"] += 1
                evidence["preview_suppressed_by_ole"] = True
            formula = self._formula_node(
                "mtef",
                locator,
                "inline",
                conversion,
                original_ref=original_ref,
                preview_ref=preview_ref,
                conversion_path=[
                    "OLE",
                    "Equation Native",
                    "MTEF v3/v5",
                    "MathML",
                    "LaTeX",
                ],
                candidate_evidence=evidence,
                validation=direct_validation,
            )
            self._emit(formula, output)
            return
        if direct_validation and not direct_validation.accepted:
            evidence["structural_attempts"].append(
                self._structural_audit("OLE_MTEF", conversion, direct_validation, None)
            )

        inspection = self._inspect_wmf_preview(preview_rel)
        evidence["preview_visual_state"] = (
            inspection.visual_state if inspection else "missing"
        )
        evidence["wmf_inspection"] = inspection.to_dict() if inspection else None
        embedded = self._recover_wmf_semantic(inspection)
        embedded_validation = self._validate_structural(
            embedded,
            source_type="wmf_embedded",
            source_metadata={"referenced": True, "progid": prog_id},
        )
        if (
            embedded
            and embedded.latex
            and embedded_validation
            and embedded_validation.accepted
        ):
            self.report["assets"]["formula_previews_suppressed"] += 1
            self.report["formulas"]["duplicate_preview_suppressed"] += 1
            evidence["semantic_state"] = embedded.semantic_state
            evidence["preview_suppressed"] = True
            if inspection and inspection.embedded_mathml is not None:
                semantic_payload = inspection.embedded_mathml.encode("utf-8")
                evidence["source_payload_type"] = "WMF_MATHML"
            elif inspection and inspection.embedded_mtef is not None:
                semantic_payload = inspection.embedded_mtef
                evidence["source_payload_type"] = "WMF_MTEF"
            else:
                semantic_payload = embedded.latex.encode("utf-8")
                evidence["source_payload_type"] = "WMF_CONVERTED_LATEX"
            evidence["source_payload_sha256"] = hashlib.sha256(
                semantic_payload
            ).hexdigest()
            formula = self._formula_node(
                "wmf_embedded",
                locator,
                "inline",
                embedded,
                original_ref=original_ref,
                preview_ref=preview_ref,
                conversion_path=["WMF", "MFCOMMENT", "MTEF/MathML", "LaTeX"],
                candidate_evidence=evidence,
                validation=embedded_validation,
            )
            self._emit(formula, output)
            return
        if embedded and embedded_validation and not embedded_validation.accepted:
            evidence["semantic_state"] = embedded.semantic_state
            evidence["structural_attempts"].append(
                self._structural_audit(
                    "WMF_EMBEDDED", embedded, embedded_validation, inspection
                )
            )

        if (
            conversion.semantic_state == "empty"
            and inspection
            and inspection.visual_state == "empty"
            and not inspection.embedded_mtef_presence
            and not inspection.embedded_mathml_presence
        ):
            self._record_candidate_without_node(
                "empty_placeholder",
                locator,
                "EMPTY_PLACEHOLDER_SUPPRESSED",
                original_ref=original_ref,
                preview_ref=preview_ref,
                prog_id=prog_id,
                semantic_state="empty",
                preview_visual_state="empty",
                wmf_inspection=inspection.to_dict(),
                structural_attempts=evidence["structural_attempts"],
                candidate_sources=evidence["candidate_sources"],
                ole_relationship_id=ole_rid,
                preview_relationship_id=preview_rid,
                ole_sha256=evidence["ole_sha256"],
                wmf_sha256=evidence["wmf_sha256"],
                preview_suppressed=False,
                preview_suppressed_by_ole=False,
            )
            return

        rendered_ref, render_metadata = self._render_wmf_preview(
            preview_rel, object_node
        )
        if render_metadata:
            evidence["renderer_metadata"] = render_metadata
        if rendered_ref:
            rejected_by_validator = bool(evidence["structural_attempts"])
            rendered = FormulaConversion(
                None,
                "RENDERED_FALLBACK",
                "Windows Native GDI WMF renderer",
                [
                    (
                        "Structural formula result was rejected by the validator; "
                        "visible WMF preserved as PNG"
                        if rejected_by_validator
                        else "Semantic formula recovery failed; visible WMF preserved as PNG"
                    )
                ],
                conversion.error,
                semantic_state=conversion.semantic_state,
            )
            formula = self._formula_node(
                "mtef",
                locator,
                "inline",
                rendered,
                original_ref=original_ref,
                preview_ref=preview_ref,
                rendered_ref=rendered_ref,
                renderer_metadata=render_metadata,
                conversion_path=[
                    (
                        "Structural recovery rejected"
                        if rejected_by_validator
                        else "OLE/MTEF failed"
                    ),
                    "WMF",
                    "Win32 GDI",
                    "PNG",
                ],
                candidate_evidence=evidence,
            )
            self._emit(formula, output)
            return

        if (
            conversion.semantic_state == "empty"
            and render_metadata
            and render_metadata.get("status") == "empty"
        ):
            self._record_candidate_without_node(
                "empty_placeholder",
                locator,
                "EMPTY_PLACEHOLDER_SUPPRESSED",
                original_ref=original_ref,
                preview_ref=preview_ref,
                prog_id=prog_id,
                semantic_state="empty",
                preview_visual_state="empty",
                wmf_inspection=inspection.to_dict() if inspection else None,
                renderer_metadata=render_metadata,
                structural_attempts=evidence["structural_attempts"],
                candidate_sources=evidence["candidate_sources"],
                ole_relationship_id=ole_rid,
                preview_relationship_id=preview_rid,
                ole_sha256=evidence["ole_sha256"],
                wmf_sha256=evidence["wmf_sha256"],
                preview_suppressed=False,
                preview_suppressed_by_ole=False,
            )
            return

        preserved = self._preserve_formula_asset(preview_rel, ole_rel)
        proven_visible = bool(inspection and inspection.visual_state == "visible")
        if proven_visible:
            failed = FormulaConversion(
                None,
                "FAILED_PRESERVED",
                conversion.component,
                conversion.warnings,
                conversion.error,
                semantic_state=conversion.semantic_state,
            )
            classification = "real_formula"
        else:
            failed = FormulaConversion(
                None,
                "UNRESOLVED",
                conversion.component,
                conversion.warnings,
                conversion.error,
                semantic_state=conversion.semantic_state,
            )
            classification = "unresolved_candidate"
        formula = self._formula_node(
            "mtef",
            locator,
            "inline",
            failed,
            original_ref=original_ref,
            preview_ref=preserved,
            conversion_path=[
                "OLE",
                "Structural recovery rejected or failed",
                "WMF fallback failed",
            ],
            classification=classification,
            candidate_evidence=evidence,
        )
        self._emit(formula, output)

    def _inspect_wmf_preview(self, preview_rel) -> WmfInspection | None:
        if (
            preview_rel is None
            or preview_rel.target_part not in self.package.parts
            or PurePosixPath(preview_rel.target_part).suffix.lower() != ".wmf"
        ):
            return None
        started = time.perf_counter()
        inspection = self.wmf_inspector.inspect(
            self.package.data(preview_rel.target_part)
        )
        self.report["timing"]["wmf_inspect_seconds"] += time.perf_counter() - started
        self.report["formulas"]["wmf_inspect_calls"] += 1
        return inspection

    def _recover_wmf_semantic(
        self, inspection: WmfInspection | None
    ) -> FormulaConversion | None:
        if inspection is None:
            return None
        if not (inspection.embedded_mathml or inspection.embedded_mtef):
            return None
        self.report["formulas"]["wmf_embedded_recovery_calls"] += 1
        started = time.perf_counter()
        conversion = None
        if inspection.embedded_mathml:
            conversion = convert_mathml(
                inspection.embedded_mathml,
                component="WMF MFCOMMENT MathML + mathml2latex 0.2.12",
            )
        elif inspection.embedded_mtef:
            conversion = convert_mtef_payload(inspection.embedded_mtef)
        elapsed = time.perf_counter() - started
        self.report["timing"]["wmf_semantic_recovery_seconds"] += elapsed
        self.report["timing"]["formula_conversion_seconds"] += elapsed
        return conversion

    def _render_wmf_preview(self, preview_rel, object_node):
        if (
            preview_rel is None
            or preview_rel.target_part not in self.package.parts
            or PurePosixPath(preview_rel.target_part).suffix.lower() != ".wmf"
        ):
            return None, None
        extent_emu = self._word_extent_emu(object_node)
        cached = self.assets.cached_generated(preview_rel.target_part, extent_emu)
        if cached:
            relative, cached_metadata = cached
            return relative, {**cached_metadata, "cache_hit": True}
        target, relative = self.assets.generated_formula_target()
        self.report["formulas"]["wmf_render_calls"] += 1
        started = time.perf_counter()
        metadata = self.wmf_renderer.render_bytes(
            self.package.data(preview_rel.target_part),
            target,
            extent_emu=extent_emu,
        )
        elapsed = time.perf_counter() - started
        self.report["timing"]["wmf_render_seconds"] += elapsed
        details = metadata.to_dict()
        if metadata.status == "success":
            self.assets.record_generated_formula(
                preview_rel.target_part, extent_emu, target, relative, details
            )
            return relative, details
        target.unlink(missing_ok=True)
        return None, details

    @staticmethod
    def _word_extent_emu(object_node: etree._Element) -> tuple[int, int] | None:
        extent = object_node.find(".//wp:extent", NS)
        if extent is not None:
            try:
                return int(extent.get("cx")), int(extent.get("cy"))
            except (TypeError, ValueError):
                pass
        shapes = object_node.xpath(".//v:shape", namespaces=NS)
        if shapes:
            style = shapes[0].get("style", "")
            values = {}
            for item in style.split(";"):
                if ":" in item:
                    key, value = item.split(":", 1)
                    values[key.strip().lower()] = value.strip().lower()
            try:
                if values.get("width", "").endswith("pt") and values.get(
                    "height", ""
                ).endswith("pt"):
                    width = float(values["width"][:-2]) * 12700
                    height = float(values["height"][:-2]) * 12700
                    return round(width), round(height)
            except ValueError:
                pass
        return None

    def _preserve_formula_asset(self, preview_rel, ole_rel) -> str | None:
        if preview_rel and preview_rel.target_part in self.package.parts:
            return self.assets.export(preview_rel.target_part, "formula")
        if ole_rel and ole_rel.target_part in self.package.parts:
            return self.assets.export(ole_rel.target_part, "formula")
        return None

    def _formula_node(
        self,
        source_type: str,
        locator: str,
        display_mode: str,
        conversion,
        original_ref: str | None = None,
        preview_ref: str | None = None,
        rendered_ref: str | None = None,
        renderer_metadata: dict | None = None,
        conversion_path: list[str] | None = None,
        classification: str = "real_formula",
        candidate_evidence: dict | None = None,
        validation: FormulaValidation | None = None,
    ) -> FormulaNode:
        if validation is None:
            validation = self._validate_structural(
                conversion,
                source_type=source_type,
                source_metadata={"referenced": True, "source_type": source_type},
            )
        self.formula_counter += 1
        formula_id = f"formula_{self.formula_counter:05d}"
        node = FormulaNode(
            formula_id=formula_id,
            source_type=source_type,
            source_part=self.owner_part,
            source_locator=locator,
            display_mode=display_mode,
            latex=conversion.latex,
            status=conversion.status,
            original_ref=original_ref,
            preview_ref=preview_ref,
            rendered_ref=rendered_ref,
            warnings=list(conversion.warnings),
            error=conversion.error,
            conversion_path=conversion_path or [],
            renderer_metadata=renderer_metadata,
        )
        formulas = self.report["formulas"]
        formulas["equation_candidates"] += 1
        if classification == "real_formula":
            formulas["real_formulas"] += 1
            formulas[source_type if source_type != "latex_alt" else "latex_alt"] += 1
            status_key = conversion.status.lower()
            formulas[status_key] += 1
        elif classification == "unresolved_candidate":
            formulas["unresolved_candidates"] += 1
        else:
            raise ValueError(f"Unknown candidate classification: {classification}")
        evidence = candidate_evidence or {}
        formulas["candidate_records"].append(
            {
                "candidate_id": formula_id,
                "classification": classification.upper(),
                "status": conversion.status,
                "source_type": source_type,
                "source_part": self.owner_part,
                "source_locator": locator,
                "original_ref": original_ref,
                "preview_ref": preview_ref,
                **evidence,
                "validator": validation.to_dict() if validation else None,
            }
        )
        formulas["records"].append(
            {
                "formula_id": formula_id,
                "source_type": source_type,
                "source_part": self.owner_part,
                "source_locator": locator,
                "display_mode": display_mode,
                "latex": conversion.latex,
                "status": conversion.status,
                "original_ref": original_ref,
                "preview_ref": preview_ref,
                "rendered_ref": rendered_ref,
                "warnings": list(conversion.warnings),
                "error": conversion.error,
                "conversion_path": conversion_path or [],
                "component": conversion.component,
                "renderer_metadata": renderer_metadata,
                "validator": validation.to_dict() if validation else None,
                "structural_audit": evidence.get("structural_attempts") or [],
            }
        )
        return node

    def _validate_structural(
        self,
        conversion: FormulaConversion | None,
        *,
        source_type: str,
        source_metadata: dict,
    ) -> FormulaValidation | None:
        if conversion is None or (
            conversion.latex is None and conversion.intermediate is None
        ):
            return None
        started = time.perf_counter()
        validation = self.formula_validator.validate(
            conversion,
            source_metadata={**source_metadata, "source_type": source_type},
        )
        elapsed = time.perf_counter() - started
        self.report["timing"]["formula_validation_seconds"] += elapsed
        self._record_structural_validation(validation)
        return validation

    def _record_structural_validation(
        self, validation: FormulaValidation | None
    ) -> None:
        if validation is None:
            return
        formulas = self.report["formulas"]
        formulas["structural_detected"] += 1
        verdict_key = {
            FormulaVerdict.VALID: "structural_valid",
            FormulaVerdict.SUSPICIOUS: "structural_suspicious",
            FormulaVerdict.INVALID: "structural_invalid",
            FormulaVerdict.NON_FORMULA_CONTENT: "structural_non_formula",
        }[validation.verdict]
        formulas[verdict_key] += 1

    def _record_mtef_resolution(self, resolution: MtefCacheResolution) -> None:
        timing_map = {
            "ole_open_seconds": "ole_open_seconds",
            "equation_stream_lookup_seconds": "equation_stream_lookup_seconds",
            "equation_stream_read_seconds": "equation_stream_read_seconds",
            "mtef_payload_extraction_seconds": "mtef_payload_extraction_seconds",
            "mtef_sha_seconds": "mtef_sha_seconds",
            "mtef_parse_seconds": "mtef_parse_seconds",
            "mtef_to_mathml_seconds": "mtef_to_mathml_seconds",
            "mathml_to_latex_seconds": "mathml_to_latex_seconds",
            "cache_lookup_seconds": "mtef_cache_lookup_seconds",
            "l2_read_seconds": "mtef_l2_read_seconds",
            "l2_write_seconds": "mtef_l2_write_seconds",
            "formula_validation_seconds": "formula_validation_seconds",
        }
        for source, target in timing_map.items():
            self.report["timing"][target] += resolution.timings.get(source, 0.0)
        cache = self.report["mtef_cache"]
        cache["occurrences"] += int(resolution.mtef_sha256 is not None)
        if resolution.mtef_sha256:
            payload_counts = cache["_payload_counts"]
            payload_counts[resolution.mtef_sha256] = (
                payload_counts.get(resolution.mtef_sha256, 0) + 1
            )
        if resolution.cache_level == "l1":
            cache["l1_hits"] += 1
        else:
            if self.mtef_cache_context.mode != "off":
                cache["l1_misses"] += 1
            if resolution.cache_level == "l2":
                cache["l2_hits"] += 1
            else:
                if self.mtef_cache_context.mode == "persistent":
                    cache["l2_misses"] += 1
                cache["misses"] += 1
                cache["full_conversion_calls"] += 1
        cache["l2_writes"] += int(resolution.l2_write)
        cache["l2_write_failures"] += int(resolution.l2_write_failure)
        cache["l2_corruptions"] += int(resolution.l2_corruption)
        cache["non_cacheable"] += int(not resolution.cacheable)
        self.report["warnings"].extend(resolution.warnings)

    @staticmethod
    def _structural_audit(
        route: str,
        conversion: FormulaConversion,
        validation: FormulaValidation,
        inspection: WmfInspection | None,
    ) -> dict:
        payload_type = None
        raw_payload = None
        if route == "WMF_EMBEDDED" and inspection is not None:
            if inspection.embedded_mathml is not None:
                payload_type = "MathML"
                raw_payload = inspection.embedded_mathml
            elif inspection.embedded_mtef is not None:
                payload_type = "MTEF"
                raw_payload = inspection.embedded_mtef.hex()
        return {
            "route": route,
            "payload_type": payload_type,
            "raw_semantic_payload": raw_payload,
            "converted_mathml": conversion.intermediate,
            "converted_latex": conversion.latex,
            "conversion_status": conversion.status,
            "conversion_error": conversion.error,
            "validator": validation.to_dict(),
        }

    @staticmethod
    def _candidate_sources(**present: bool) -> dict[str, bool]:
        sources = {
            "omml": False,
            "eq": False,
            "latex_alt": False,
            "ole": False,
            "preview": False,
        }
        sources.update(present)
        return sources

    def _record_candidate_without_node(
        self,
        classification: str,
        locator: str,
        status: str,
        **evidence,
    ) -> str:
        self.formula_counter += 1
        candidate_id = f"formula_{self.formula_counter:05d}"
        formulas = self.report["formulas"]
        formulas["equation_candidates"] += 1
        key = {
            "empty_placeholder": "empty_placeholders",
            "non_formula_object": "non_formula_objects",
        }[classification]
        formulas[key] += 1
        if classification == "empty_placeholder":
            formulas["empty_placeholders_suppressed"] += 1
        formulas["candidate_records"].append(
            {
                "candidate_id": candidate_id,
                "classification": classification.upper(),
                "status": status,
                "source_part": self.owner_part,
                "source_locator": locator,
                **evidence,
            }
        )
        return candidate_id

    def _table(self, node: etree._Element) -> TableNode:
        self.report["document"]["tables"] += 1
        rows: list[list[TableCell]] = []
        complex_table = False
        for tr in node.findall("w:tr", NS):
            row = []
            column = 0
            for tc in tr.findall("w:tc", NS):
                tcpr = tc.find("w:tcPr", NS)
                span_node = tcpr.find("w:gridSpan", NS) if tcpr is not None else None
                colspan = (
                    int(span_node.get(qn("w", "val"), "1"))
                    if span_node is not None
                    else 1
                )
                vmerge_node = tcpr.find("w:vMerge", NS) if tcpr is not None else None
                vmerge = None
                if vmerge_node is not None:
                    vmerge = vmerge_node.get(qn("w", "val"), "continue")
                complex_table = complex_table or colspan > 1 or vmerge is not None
                blocks = [self._paragraph(p) for p in tc.findall("w:p", NS)]
                row.append(TableCell(blocks, column, colspan, vmerge))
                column += colspan
            rows.append(row)
        table = TableNode(rows, complex_table)
        tables = self.report["tables"]
        tables["tables"] += 1
        if complex_table:
            tables["html_tables"] += 1
        else:
            tables["simple_markdown_tables"] += 1
        return table

    def _load_styles(self) -> dict[str, str]:
        if not self.package.has_part("word/styles.xml"):
            return {}
        try:
            root = self.package.xml("word/styles.xml")
        except InvalidDocxError as exc:
            self.report["warnings"].append(f"Ignoring malformed styles.xml: {exc}")
            return {}
        result = {}
        for style in root.findall("w:style", NS):
            style_id = style.get(qn("w", "styleId"))
            name = style.find("w:name", NS)
            if style_id and name is not None:
                result[style_id] = name.get(qn("w", "val"), style_id)
        return result

    def _load_numbering(self) -> dict[tuple[str, int], str]:
        if not self.package.has_part("word/numbering.xml"):
            return {}
        try:
            root = self.package.xml("word/numbering.xml")
        except InvalidDocxError as exc:
            self.report["warnings"].append(f"Ignoring malformed numbering.xml: {exc}")
            return {}
        abstract_levels: dict[tuple[str, int], str] = {}
        for abstract in root.findall("w:abstractNum", NS):
            abstract_id = abstract.get(qn("w", "abstractNumId"), "")
            for level in abstract.findall("w:lvl", NS):
                ilvl = int(level.get(qn("w", "ilvl"), "0"))
                num_fmt = level.find("w:numFmt", NS)
                abstract_levels[(abstract_id, ilvl)] = (
                    num_fmt.get(qn("w", "val"), "decimal")
                    if num_fmt is not None
                    else "decimal"
                )
        result = {}
        for num in root.findall("w:num", NS):
            num_id = num.get(qn("w", "numId"), "")
            abstract = num.find("w:abstractNumId", NS)
            abstract_id = (
                abstract.get(qn("w", "val"), "") if abstract is not None else ""
            )
            for (candidate, level), fmt in abstract_levels.items():
                if candidate == abstract_id:
                    result[(num_id, level)] = fmt
        return result

    @staticmethod
    def _truthy_property(parent: etree._Element | None, name: str) -> bool:
        if parent is None:
            return False
        node = parent.find(f"w:{name}", NS)
        if node is None:
            return False
        return node.get(qn("w", "val"), "true").lower() not in {"0", "false", "off"}

    @staticmethod
    def _heading_level(style_id: str | None, style_name: str | None) -> int | None:
        for value in (style_id or "", style_name or ""):
            match = re.search(r"(?:heading|标题)\s*([1-9])", value, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _numbering_properties(ppr: etree._Element | None) -> tuple[str | None, int]:
        if ppr is None:
            return None, 0
        numpr = ppr.find("w:numPr", NS)
        if numpr is None:
            return None, 0
        num_id_node = numpr.find("w:numId", NS)
        level_node = numpr.find("w:ilvl", NS)
        num_id = num_id_node.get(qn("w", "val")) if num_id_node is not None else None
        level = (
            int(level_node.get(qn("w", "val"), "0")) if level_node is not None else 0
        )
        return num_id, level

    def _alternate_branch(self, node: etree._Element) -> etree._Element:
        choice = node.find("mc:Choice", NS)
        fallback = node.find("mc:Fallback", NS)
        selected = choice if choice is not None else fallback
        branch = "Choice" if choice is not None else "Fallback"
        self.report["document"]["alternate_content_selected"][branch] += 1
        return selected if selected is not None else node

    @staticmethod
    def _image_metadata(node: etree._Element) -> tuple[str | None, str | None]:
        candidates = node.xpath(".//wp:docPr | .//pic:cNvPr", namespaces=NS)
        alt = next(
            (item.get("descr") for item in candidates if item.get("descr")), None
        )
        title = next(
            (
                item.get("title") or item.get("name")
                for item in candidates
                if item.get("title") or item.get("name")
            ),
            None,
        )
        return alt, title

    def _locator(self, node: etree._Element) -> str:
        self.locator_counter += 1
        try:
            return node.getroottree().getpath(node)
        except (AttributeError, ValueError):
            return f"{self.owner_part}#node-{self.locator_counter}"

    def _unsupported(self, category: str, part: str, detail: str) -> None:
        self.report["unsupported"].append(
            {"category": category, "part": part, "detail": detail}
        )

    def _inventory_edges(self, root: etree._Element) -> None:
        textboxes = len(root.xpath(".//w:txbxContent", namespaces=NS))
        if textboxes:
            self._unsupported("textboxes_detected", self.owner_part, str(textboxes))
        for entry in self.package.non_body_inventory():
            self._unsupported(
                "non_body_part_not_serialized", str(entry["part"]), str(entry)
            )

    def _header_titles(
        self, blocks: list
    ) -> tuple[dict[str, list[dict[str, str]]], list[ParagraphNode]]:
        header_report = self.report["headers"]
        candidates: list[dict[str, str]] = []
        unique_candidates: list[dict[str, str]] = []
        seen_parts: set[str] = set()
        for part in self.package.related_parts(self.owner_part, "header"):
            if part in seen_parts or not self.package.has_part(part):
                continue
            seen_parts.add(part)
            try:
                root = self.package.xml(part)
            except InvalidDocxError as exc:
                self.report["warnings"].append(
                    f"Ignoring malformed optional header part {part}: {exc}"
                )
                continue
            for paragraph in root.findall("w:p", NS):
                plain_text = "".join(
                    paragraph.xpath(
                        ".//w:t[not(ancestor::w:txbxContent)]/text()",
                        namespaces=NS,
                    )
                )
                text = re.sub(r"\s+", " ", plain_text).strip()
                if not text:
                    continue
                instructions = " ".join(
                    paragraph.xpath(
                        ".//w:fldSimple/@w:instr | .//w:instrText/text()",
                        namespaces=NS,
                    )
                ).upper()
                if re.search(r"\b(?:PAGE|SECTIONPAGES|NUMPAGES)\b", instructions):
                    continue
                visual_watermark = bool(
                    paragraph.xpath(
                        ".//a:graphicData[contains(@uri, 'wordprocessingGroup')]"
                        " | .//v:textpath | .//w:txbxContent",
                        namespaces=NS,
                    )
                )
                locator = paragraph.getroottree().getpath(paragraph)
                if visual_watermark:
                    header_report["watermark_or_visual_excluded"] += 1
                    header_report["records"].append(
                        {
                            "source_part": part,
                            "source_locator": locator,
                            "text": text,
                            "status": "EXCLUDED_VISUAL_OR_WATERMARK",
                        }
                    )
                    continue
                normalized = re.sub(r"\s+", "", text).casefold()
                candidate = {
                    "text": text,
                    "normalized": normalized,
                    "source_part": part,
                    "source_locator": locator,
                }
                candidates.append(candidate)
                if any(item["normalized"] == normalized for item in unique_candidates):
                    header_report["titles_deduplicated"] += 1
                    header_report["records"].append(
                        {**candidate, "status": "DEDUPLICATED_SECTION"}
                    )
                    continue
                unique_candidates.append(candidate)

        header_report["title_candidates"] = len(candidates)
        body_prefix = "".join(
            child.text
            for block in blocks[:5]
            if isinstance(block, ParagraphNode)
            for child in block.children
            if isinstance(child, TextNode)
        )
        normalized_body = re.sub(r"\s+", "", body_prefix).casefold()
        inserted: list[ParagraphNode] = []
        for candidate in unique_candidates:
            if candidate["normalized"] and candidate["normalized"] in normalized_body:
                header_report["titles_deduplicated"] += 1
                status = "DEDUPLICATED_BODY"
            else:
                inserted.append(ParagraphNode([TextNode(candidate["text"])]))
                header_report["titles_inserted"] += 1
                status = "INSERTED_ONCE"
            header_report["records"].append({**candidate, "status": status})
        return {"header_title_candidates": candidates}, inserted


def _media_payload_warning(part: str, data: bytes) -> str | None:
    suffix = PurePosixPath(part).suffix.lower()
    signatures = {
        ".png": (b"\x89PNG\r\n\x1a\n",),
        ".jpg": (b"\xff\xd8\xff",),
        ".jpeg": (b"\xff\xd8\xff",),
        ".gif": (b"GIF87a", b"GIF89a"),
        ".bmp": (b"BM",),
        ".tif": (b"II*\x00", b"MM\x00*"),
        ".tiff": (b"II*\x00", b"MM\x00*"),
        ".webp": (b"RIFF",),
    }
    if not data:
        return f"Empty image part preserved for review: {part}"
    expected = signatures.get(suffix)
    if expected and not any(data.startswith(signature) for signature in expected):
        return f"Image signature does not match extension; bytes preserved: {part}"
    if suffix == ".webp" and data[8:12] != b"WEBP":
        return f"Image signature does not match extension; bytes preserved: {part}"
    return None
