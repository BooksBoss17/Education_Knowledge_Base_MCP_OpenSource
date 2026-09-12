from __future__ import annotations

import html
import re
from pathlib import Path

from .ir import (
    BlockNode,
    BreakNode,
    DocumentIR,
    FormulaNode,
    HeadingNode,
    HyperlinkNode,
    ImageNode,
    ImageContentNode,
    InlineNode,
    ListItemNode,
    TableCell,
    TableNode,
    TextNode,
)


class MarkdownSerializer:
    def serialize(self, document: DocumentIR) -> str:
        chunks = [self._block(block) for block in document.blocks]
        text = "\n\n".join(chunk for chunk in chunks if chunk != "")
        return text.rstrip() + "\n"

    def _block(self, block: BlockNode) -> str:
        if isinstance(block, HeadingNode):
            return f"{'#' * block.level} {self._inlines(block.children)}"
        if isinstance(block, ListItemNode):
            marker = f"{block.ordinal or 1}." if block.ordered else "-"
            return f"{'  ' * block.level}{marker} {self._inlines(block.children)}"
        if isinstance(block, TableNode):
            return self._table(block)
        return self._inlines(block.children)

    def _inlines(self, nodes: list[InlineNode]) -> str:
        return "".join(self._inline(node) for node in nodes)

    def _inline(self, node: InlineNode) -> str:
        if isinstance(node, TextNode):
            value = self._escape_text(node.text)
            if node.vertical_align in {"superscript", "subscript"}:
                tag = "sup" if node.vertical_align == "superscript" else "sub"
                value = f"<{tag}>{html.escape(node.text)}</{tag}>"
            if node.bold and value:
                value = f"**{value}**"
            if node.italic and value:
                value = f"*{value}*"
            return value
        if isinstance(node, BreakNode):
            return "  \n"
        if isinstance(node, HyperlinkNode):
            label = self._inlines(node.children) or node.target
            if self._contains_image_content(node.children):
                target = html.escape(self._escape_url(node.target), quote=True)
                # Block content cannot be the label of a Markdown inline link.
                # Blank lines delimit CommonMark HTML blocks so nested Markdown
                # paragraphs, math, tables and images remain parseable.
                return f'\n\n<a href="{target}">\n\n{label.strip()}\n\n</a>\n\n'
            return f"[{label}]({self._escape_url(node.target)})"
        if isinstance(node, ImageContentNode):
            value = node.markdown
            for token, asset in node.assets.items():
                if token not in value:
                    raise ValueError("Embedded image content contains an unreferenced asset")
                value = value.replace(token, self._inline(asset))
            return "\n\n" + value.strip() + "\n\n"
        if isinstance(node, ImageNode):
            if not node.asset_path:
                label = node.asset_id or "unassigned"
                return f"[Unresolved image: {label}]"
            alt = (node.alt or "").replace("]", "\\]")
            title = f' "{node.title.replace(chr(34), chr(39))}"' if node.title else ""
            return f"![{alt}]({self._escape_url(node.asset_path)}{title})"
        if isinstance(node, FormulaNode):
            if node.latex:
                if node.display_mode == "block":
                    return f"\n$$\n{node.latex}\n$$\n"
                return f"${node.latex}$"
            if (
                node.status
                in {
                    "RENDERED_FALLBACK",
                    "FAILED_PRESERVED",
                    "OCR_REVIEW_REQUIRED",
                    "OCR_REJECTED_PRESERVE_IMAGE",
                    "OCR_INFERENCE_FAILED_PRESERVE_IMAGE",
                }
                and node.rendered_ref
            ):
                return f"![]({self._escape_url(node.rendered_ref)})"
            marker = f"<!-- BEMARKDOWN_UNRESOLVED_FORMULA id={node.formula_id} -->"
            if node.rendered_ref:
                return f"{marker}[Unresolved formula asset]({self._escape_url(node.rendered_ref)})"
            if node.preview_ref:
                return f"{marker}[Unresolved formula asset]({self._escape_url(node.preview_ref)})"
            if node.original_ref:
                return f"{marker}`[unresolved formula: {node.original_ref}]`"
            return f"{marker}`[unresolved formula]`"
        raise TypeError(f"Unsupported inline node: {type(node)!r}")

    def _table(self, table: TableNode) -> str:
        if table.complex or any(self._contains_image_content(block.children)
                                for row in table.rows for cell in row for block in cell.blocks):
            return self._html_table(table)
        if not table.rows:
            return ""
        rows = [[self._cell_markdown(cell) for cell in row] for row in table.rows]
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        header = rows[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
        return "\n".join(lines)

    def _html_table(self, table: TableNode) -> str:
        lines = ["<table>"]
        for row_index, row in enumerate(table.rows):
            lines.append("  <tr>")
            for cell in row:
                if cell.vmerge == "continue":
                    continue
                attrs = []
                if cell.colspan > 1:
                    attrs.append(f'colspan="{cell.colspan}"')
                if cell.vmerge == "restart":
                    rowspan = self._rowspan(table, row_index, cell.column)
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                attr_text = " " + " ".join(attrs) if attrs else ""
                if any(self._contains_image_content(block.children) for block in cell.blocks):
                    content = self._cell_plain(cell).strip()
                    lines.extend([f'<td{attr_text}>', '', content, '', '</td>'])
                else:
                    content = html.escape(self._cell_plain(cell), quote=False).replace("\n", "<br>")
                    lines.append(f"    <td{attr_text}>{content}</td>")
            lines.append("  </tr>")
        lines.append("</table>")
        return "\n".join(lines)

    @classmethod
    def _contains_image_content(cls, nodes):
        return any(isinstance(node, ImageContentNode)
                   or (isinstance(node, HyperlinkNode) and cls._contains_image_content(node.children))
                   for node in nodes)

    @staticmethod
    def _rowspan(table: TableNode, row_index: int, column: int) -> int:
        count = 1
        for later in table.rows[row_index + 1 :]:
            match = next((cell for cell in later if cell.column == column), None)
            if match is None or match.vmerge != "continue":
                break
            count += 1
        return count

    def _cell_markdown(self, cell: TableCell) -> str:
        return "<br>".join(
            self._inlines(block.children) for block in cell.blocks
        ).replace("|", "\\|")

    def _cell_plain(self, cell: TableCell) -> str:
        return "\n".join(self._inlines(block.children) for block in cell.blocks)

    @staticmethod
    def _escape_text(value: str) -> str:
        return re.sub(r"([\\`*\[\]])", r"\\\1", value)

    @staticmethod
    def _escape_url(value: str) -> str:
        return value.replace(" ", "%20").replace(")", "%29")


def write_markdown(document: DocumentIR, target: Path) -> None:
    target.write_text(
        MarkdownSerializer().serialize(document), encoding="utf-8", newline="\n"
    )
