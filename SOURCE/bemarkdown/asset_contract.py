from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .ir import DocumentIR, FormulaNode, HyperlinkNode, ImageContentNode, ImageNode, TableNode

ASSET_MANIFEST_VERSION = "bemarkdown-asset-contract-v1"
VISIBLE_FORMULA_STATES = {
    "RENDERED_FALLBACK",
    "FAILED_PRESERVED",
    "OCR_REVIEW_REQUIRED",
    "OCR_REJECTED_PRESERVE_IMAGE",
    "OCR_INFERENCE_FAILED_PRESERVE_IMAGE",
}


@dataclass(frozen=True)
class _Occurrence:
    node: ImageNode | FormulaNode
    semantic_type: str
    status: str
    source_part: str
    source_locator: str
    relationship_id: str | None
    source_ref: str | None
    external_target: str | None
    render_method: str
    provenance: dict[str, Any]


class AssetFinalizer:
    """Assign final visible asset identities after DocumentIR is complete."""

    def __init__(self, output_dir: Path, document_sha256: str, report: dict[str, Any]):
        self.output_dir = output_dir
        self.assets_dir = output_dir / "assets"
        self.document_sha256 = document_sha256
        self.report = report

    def finalize(self, document: DocumentIR) -> list[dict[str, Any]]:
        occurrences = list(_iter_visible_occurrences(document))
        rows: list[dict[str, Any]] = []
        final_paths: set[Path] = set()
        for index, occurrence in enumerate(occurrences, 1):
            asset_id = f"image_{index:06d}"
            content = self._read_staged(occurrence.source_ref)
            extension, mime_type = self._format(occurrence, content)
            relative_path = None
            content_sha = (
                hashlib.sha256(content).hexdigest() if content is not None else None
            )
            if content is not None:
                self.assets_dir.mkdir(parents=True, exist_ok=True)
                target = self.assets_dir / f"{asset_id}{extension}"
                target.write_bytes(content)
                final_paths.add(target.resolve())
                relative_path = f"assets/{target.name}"
            uid_payload = "\x1f".join(
                (
                    self.document_sha256,
                    occurrence.source_part,
                    occurrence.source_locator,
                    occurrence.relationship_id or "",
                    occurrence.provenance.get("selected_branch", ""),
                    occurrence.semantic_type,
                )
            ).encode("utf-8")
            asset_uid = hashlib.sha256(uid_payload).hexdigest()
            provenance = dict(occurrence.provenance)
            if occurrence.source_ref and occurrence.source_ref.startswith(
                ".bmd-staging/"
            ):
                provenance.setdefault(
                    "legacy_relative_path",
                    "assets/" + PurePosixPath(occurrence.source_ref).name,
                )
            row = {
                "manifest_version": ASSET_MANIFEST_VERSION,
                "asset_id": asset_id,
                "asset_uid": asset_uid,
                "content_sha256": content_sha,
                "semantic_type": occurrence.semantic_type,
                "status": occurrence.status,
                "mime_type": mime_type,
                "extension": extension or None,
                "relative_path": relative_path,
                "source_type": "DOCX",
                "source_part": occurrence.source_part,
                "source_locator": occurrence.source_locator,
                "relationship_id": occurrence.relationship_id,
                "external_target": occurrence.external_target,
                "render_method": occurrence.render_method,
                "provenance": provenance,
            }
            rows.append(row)
            self._apply(occurrence.node, row)

        manifest = self.output_dir / "assets_manifest.jsonl"
        manifest.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
            newline="\n",
        )
        self._synchronize_formula_refs(document)
        self._remove_staging(final_paths)
        self._update_report(rows)
        return rows

    def _read_staged(self, source_ref: str | None) -> bytes | None:
        if not source_ref:
            return None
        path = (self.output_dir / source_ref).resolve()
        try:
            path.relative_to(self.output_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Asset staging path escapes output directory: {source_ref}"
            ) from exc
        return path.read_bytes() if path.is_file() else None

    @staticmethod
    def _format(
        occurrence: _Occurrence, content: bytes | None
    ) -> tuple[str, str | None]:
        if content is None:
            return "", None
        suffix = PurePosixPath(occurrence.source_ref or "").suffix.lower()
        if occurrence.semantic_type == "DRAWINGML_GROUP":
            return ".svg", "image/svg+xml"
        suffix = suffix or ".bin"
        return suffix, mimetypes.guess_type(f"x{suffix}")[
            0
        ] or "application/octet-stream"

    @staticmethod
    def _apply(node: ImageNode | FormulaNode, row: dict[str, Any]) -> None:
        if isinstance(node, ImageNode):
            node.asset_id = row["asset_id"]
            node.asset_uid = row["asset_uid"]
            node.content_sha256 = row["content_sha256"]
            node.asset_path = row["relative_path"]
            node.status = row["status"]
            node.mime_type = row["mime_type"]
            node.extension = row["extension"]
        else:
            node.rendered_ref = row["relative_path"]

    def _remove_staging(self, final_paths: set[Path]) -> None:
        if self.assets_dir.is_dir():
            for path in self.assets_dir.iterdir():
                if path.is_file() and path.resolve() not in final_paths:
                    path.unlink()
        staging_dir = self.output_dir / ".bmd-staging"
        if staging_dir.is_dir():
            shutil.rmtree(staging_dir)

    def _update_report(self, rows: list[dict[str, Any]]) -> None:
        assets = self.report["assets"]
        assets["manifest_version"] = ASSET_MANIFEST_VERSION
        assets["total_visible"] = len(rows)
        assets["resolved"] = sum(row["status"] == "RESOLVED" for row in rows)
        assets["unresolved"] = sum(row["status"] != "RESOLVED" for row in rows)
        for semantic in (
            "EMBEDDED_IMAGE",
            "DRAWINGML_GROUP",
            "VML_IMAGE",
            "FORMULA_FALLBACK",
            "EXTERNAL_LINKED_IMAGE",
            "OTHER_VISUAL",
        ):
            assets[semantic.lower()] = sum(
                row["semantic_type"] == semantic for row in rows
            )
        assets["records"] = rows
        assets["exported_assets"] = sum(
            row["relative_path"] is not None for row in rows
        )

    def _synchronize_formula_refs(self, document: DocumentIR) -> None:
        for node in _iter_formula_nodes(document):
            final_ref = (
                node.rendered_ref
                if node.status in VISIBLE_FORMULA_STATES
                or (node.status == "UNRESOLVED" and node.rendered_ref)
                else None
            )
            if final_ref is None:
                node.rendered_ref = None
            if node.ocr_provenance is not None:
                provenance = node.ocr_provenance.setdefault("provenance", {})
                provenance.setdefault(
                    "temporary_render_ref", provenance.get("rendered_ref")
                )
                provenance["final_asset_ref"] = final_ref
            for collection in ("candidate_records", "records"):
                for record in self.report["formulas"].get(collection, []):
                    key = (
                        "candidate_id"
                        if collection == "candidate_records"
                        else "formula_id"
                    )
                    if record.get(key) != node.formula_id:
                        continue
                    record["rendered_ref"] = final_ref
                    if final_ref is None and record.get("status", "").startswith(
                        "OCR_ACCEPTED"
                    ):
                        record["temporary_render_suppressed"] = True


def _iter_visible_occurrences(document: DocumentIR) -> Iterator[_Occurrence]:
    seen: set[int] = set()

    def inlines(nodes):
        for node in nodes:
            if isinstance(node, ImageContentNode):
                yield from inlines(node.assets.values())
            elif isinstance(node, HyperlinkNode):
                yield from inlines(node.children)
            elif isinstance(node, ImageNode):
                if id(node) in seen:
                    continue
                seen.add(id(node))
                yield _Occurrence(
                    node=node,
                    semantic_type=node.semantic_type,
                    status=(
                        node.status
                        if node.status != "PENDING"
                        else ("RESOLVED" if node.asset_path else "UNRESOLVED")
                    ),
                    source_part=node.source_part,
                    source_locator=node.source_locator,
                    relationship_id=node.relationship_id or None,
                    source_ref=node.asset_path,
                    external_target=node.external_target,
                    render_method=node.render_method,
                    provenance=node.provenance,
                )
            elif isinstance(node, FormulaNode) and (
                node.status in VISIBLE_FORMULA_STATES
                or node.status == "UNRESOLVED"
                and bool(node.preview_ref)
            ):
                if id(node) in seen:
                    continue
                seen.add(id(node))
                yield _Occurrence(
                    node=node,
                    semantic_type="FORMULA_FALLBACK",
                    status="RESOLVED" if node.rendered_ref else "UNRESOLVED",
                    source_part=node.source_part,
                    source_locator=node.source_locator,
                    relationship_id=(node.ocr_provenance or {})
                    .get("provenance", {})
                    .get("preview_relationship_id"),
                    source_ref=node.rendered_ref or node.preview_ref,
                    external_target=None,
                    render_method="wmf_portable_png",
                    provenance={
                        "formula_id": node.formula_id,
                        "formula_status": node.status,
                        "ocr_provenance": node.ocr_provenance,
                    },
                )

    for block in document.blocks:
        if isinstance(block, TableNode):
            for row in block.rows:
                for cell in row:
                    for nested in cell.blocks:
                        yield from inlines(nested.children)
        else:
            yield from inlines(block.children)


def _iter_formula_nodes(document: DocumentIR) -> Iterator[FormulaNode]:
    def inlines(nodes):
        for node in nodes:
            if isinstance(node, FormulaNode):
                yield node
            elif isinstance(node, HyperlinkNode):
                yield from inlines(node.children)

    for block in document.blocks:
        if isinstance(block, TableNode):
            for row in block.rows:
                for cell in row:
                    for nested in cell.blocks:
                        yield from inlines(nested.children)
        else:
            yield from inlines(block.children)
