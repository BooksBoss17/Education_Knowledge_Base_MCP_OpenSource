from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FormulaStatus = Literal[
    "SUCCESS_EXACT",
    "SUCCESS_NORMALIZED",
    "SUCCESS_APPROXIMATE",
    "RENDERED_FALLBACK",
    "FAILED_PRESERVED",
    "UNRESOLVED",
    "OCR_ACCEPTED",
    "OCR_ACCEPTED_WITH_WARNING",
    "OCR_REVIEW_REQUIRED",
    "OCR_REJECTED_PRESERVE_IMAGE",
    "OCR_INFERENCE_FAILED_PRESERVE_IMAGE",
]


@dataclass
class TextNode:
    text: str
    bold: bool = False
    italic: bool = False
    vertical_align: Literal["baseline", "superscript", "subscript"] = "baseline"
    kind: str = field(init=False, default="text")


@dataclass
class BreakNode:
    kind: str = field(init=False, default="break")


@dataclass
class FormulaNode:
    formula_id: str
    source_type: str
    source_part: str
    source_locator: str
    display_mode: str
    latex: str | None
    status: FormulaStatus
    original_ref: str | None = None
    preview_ref: str | None = None
    rendered_ref: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    conversion_path: list[str] = field(default_factory=list)
    renderer_metadata: dict[str, Any] | None = None
    ocr_provenance: dict[str, Any] | None = None
    kind: str = field(init=False, default="formula")


@dataclass
class ImageNode:
    asset_path: str | None
    source_part: str
    relationship_id: str
    media_part: str | None
    alt: str | None = None
    title: str | None = None
    source_locator: str = ""
    semantic_type: str = "EMBEDDED_IMAGE"
    status: str = "PENDING"
    mime_type: str | None = None
    extension: str | None = None
    external_target: str | None = None
    render_method: str = "package_copy"
    provenance: dict[str, Any] = field(default_factory=dict)
    asset_id: str | None = None
    asset_uid: str | None = None
    content_sha256: str | None = None
    kind: str = field(init=False, default="image")


@dataclass
class ImageContentNode:
    """Structured converter Markdown replacing a mixed-content embedded image.

    Asset tokens are resolved by the native serializer after asset finalization;
    nested conversion paths must never escape into the consumer document.
    """

    markdown: str
    assets: dict[str, ImageNode]
    source_part: str
    source_locator: str
    provenance: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="image_content")


@dataclass
class HyperlinkNode:
    target: str
    children: list[InlineNode]
    kind: str = field(init=False, default="hyperlink")


InlineNode = TextNode | BreakNode | FormulaNode | ImageNode | ImageContentNode | HyperlinkNode


@dataclass
class ParagraphNode:
    children: list[InlineNode]
    style_id: str | None = None
    style_name: str | None = None
    kind: str = field(init=False, default="paragraph")


@dataclass
class HeadingNode(ParagraphNode):
    level: int = 1
    kind: str = field(init=False, default="heading")


@dataclass
class ListItemNode(ParagraphNode):
    level: int = 0
    ordered: bool = False
    ordinal: int | None = None
    numbering_id: str | None = None
    kind: str = field(init=False, default="list_item")


@dataclass
class TableCell:
    blocks: list[ParagraphNode]
    column: int
    colspan: int = 1
    vmerge: str | None = None


@dataclass
class TableNode:
    rows: list[list[TableCell]]
    complex: bool
    kind: str = field(init=False, default="table")


BlockNode = ParagraphNode | HeadingNode | ListItemNode | TableNode


@dataclass
class DocumentIR:
    blocks: list[BlockNode]
    source_part: str = "word/document.xml"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
