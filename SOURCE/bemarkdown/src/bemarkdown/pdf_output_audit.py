from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .formula import FormulaConversion
from .pdf_document_ir import (
    OutputAuditPatchEngine,
    _render_block,
    document_asset_presentation_ref,
    semantic_sha256,
    validate_document_ir,
)
from .pdf_table_engine import TableAuditEvidenceAccessor, validate_table_ir
from .validator import FormulaStructuralValidator

PAGE_PACKAGE_SCHEMA = "bemarkdown-output-audit-page-package-v0"
RESULT_SCHEMA = "bemarkdown-output-audit-result-v0"
RESULT_STATUSES = {"NO_CHANGE", "PATCHED", "UNRESOLVED", "TECHNICAL_FAILURE"}
PATCH_OPERATIONS = {
    "REPLACE_TEXT",
    "REPLACE_FORMULA",
    "INSERT_BLOCK",
    "DELETE_DUPLICATE",
    "MOVE_BLOCK",
    "SPLIT_BLOCK",
    "MERGE_BLOCK",
    "CHANGE_HEADING_LEVEL",
    "FIX_ASSET_REFERENCE",
    "REPLACE_TABLE",
    "CHANGE_BLOCK_KIND",
    "UPDATE_CAPTION_RELATION",
}
CORRECTION_BASES = {
    "VISUAL_DIRECT",
    "VISUAL_CONTEXT_DISAMBIGUATION",
    "STRUCTURE_VISUAL",
    "SOURCE_CONTENT_ANOMALY_ONLY",
}
ISSUE_TAXONOMY = {
    "TEXT_VISUAL_MISMATCH",
    "TEXT_MISSING",
    "TEXT_DUPLICATE",
    "FORMULA_VISUAL_MISMATCH",
    "FORMULA_MISSING",
    "FORMULA_BOUNDARY_ERROR",
    "IMAGE_MISSING",
    "IMAGE_WRONG_REFERENCE",
    "CAPTION_ASSOCIATION_ERROR",
    "TABLE_STRUCTURE_ERROR",
    "TABLE_TEXT_ERROR",
    "TABLE_MISSING",
    "READING_ORDER_ERROR",
    "HEADING_LEVEL_ERROR",
    "BLOCK_BOUNDARY_ERROR",
    "SOURCE_CONTENT_ANOMALY",
    "UNRESOLVED_VISUAL_AMBIGUITY",
}


def _patch_node_ids(patch: dict[str, Any]) -> set[str]:
    values = {
        str(patch[field])
        for field in (
            "target_node_id",
            "before_node_id",
            "after_node_id",
            "caption_node_id",
            "asset_node_id",
        )
        if patch.get(field)
    }
    values.update(str(value) for value in patch.get("target_node_ids", []))
    return values


def _node_view(document: dict[str, Any], node_ids: set[str]) -> list[dict[str, Any]]:
    blocks = {
        block["node_id"]: block
        for block in [
            *document.get("blocks", []),
            *document.get("suppressed_blocks", []),
        ]
    }
    return [
        {
            "node_id": node_id,
            "page_index": blocks[node_id]["page_index"],
            "kind": blocks[node_id]["kind"],
            "content": copy.deepcopy(blocks[node_id].get("content", {})),
            "relations": copy.deepcopy(blocks[node_id].get("relations", {})),
        }
        for node_id in sorted(node_ids)
        if node_id in blocks
    ]


def _unintended_node_mutations(
    before: dict[str, Any], after: dict[str, Any], allowed_node_ids: set[str]
) -> list[str]:
    fields = (
        "kind",
        "subtype",
        "content",
        "relations",
        "source_content_ids",
        "source_candidate_ids",
        "source_region_ids",
        "source_unit_ids",
        "bbox_pdf_pt",
        "review_state",
        "visibility",
    )
    before_nodes = {
        block["node_id"]: block
        for block in [*before.get("blocks", []), *before.get("suppressed_blocks", [])]
    }
    after_nodes = {
        block["node_id"]: block
        for block in [*after.get("blocks", []), *after.get("suppressed_blocks", [])]
    }
    changed = []
    for node_id in sorted(set(before_nodes) & set(after_nodes) - allowed_node_ids):
        left = {field: before_nodes[node_id].get(field) for field in fields}
        right = {field: after_nodes[node_id].get(field) for field in fields}
        if left != right:
            changed.append(node_id)
    return changed


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PageAuditPackageBuilder:
    schema = PAGE_PACKAGE_SCHEMA

    def __init__(self, *, render_scale: float = 2.0, context_nodes: int = 3):
        self.render_scale = float(render_scale)
        self.context_nodes = int(context_nodes)

    def build(
        self,
        document: dict[str, Any],
        page_index: int,
        output: str | Path,
    ) -> dict[str, Any]:
        validate_document_ir(document)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        source_pdf = Path(str(document["source"]["path"])).resolve()
        if not source_pdf.is_file():
            raise FileNotFoundError(source_pdf)
        page_nodes = [
            block
            for block in document.get("blocks", [])
            if int(block["page_index"]) == int(page_index)
        ]
        if not page_nodes:
            raise ValueError(f"Output Audit page contains no visible nodes: {page_index}")
        audit_page_id = "audit-page-" + semantic_sha256(
            {
                "document_id": document["document_id"],
                "page_index": int(page_index),
                "source_sha256": document["source"]["sha256"],
                "schema": self.schema,
            }
        )[:24]
        clean_path = output / "clean.png"
        self._render_source_page(source_pdf, page_index, clean_path)
        overlay_path = output / "overlay.png"
        self._write_overlay(clean_path, overlay_path, page_nodes)
        markdown = "\n\n".join(_render_block(block) for block in page_nodes).rstrip() + "\n"
        (output / "current_page.md").write_text(markdown, encoding="utf-8", newline="\n")
        portable_nodes = [self._portable_node(block) for block in page_nodes]
        _write_json(output / "page_nodes.json", portable_nodes)

        all_nodes = document.get("blocks", [])
        positions = [index for index, block in enumerate(all_nodes) if block in page_nodes]
        first, last = min(positions), max(positions)
        context = [
            self._portable_node(block)
            for block in [
                *all_nodes[max(0, first - self.context_nodes) : first],
                *all_nodes[last + 1 : last + 1 + self.context_nodes],
            ]
        ]
        _write_json(
            output / "context.json",
            {
                "schema": "bemarkdown-output-audit-bounded-context-v0",
                "max_nodes_each_side": self.context_nodes,
                "nodes": context,
            },
        )
        evidence_refs = []
        evidence_dir = output / "evidence"
        for block in page_nodes:
            if block.get("kind") == "TABLE":
                value = TableAuditEvidenceAccessor().get(block)
            elif block.get("kind") == "FORMULA":
                value = {
                    "schema": "bemarkdown-output-audit-formula-evidence-v0",
                    "stable_node_id": block["node_id"],
                    "current_latex": block.get("content", {}).get("latex"),
                    "source_crop": block.get("content", {}).get("asset_ref"),
                    "review_state": block.get("review_state"),
                }
            else:
                continue
            relative = f"evidence/{block['node_id']}.json"
            _write_json(evidence_dir / f"{block['node_id']}.json", value)
            evidence_refs.append(relative)
        page_record = next(
            page for page in document["pages"] if int(page["page_index"]) == int(page_index)
        )
        package = {
            "schema": self.schema,
            "audit_page_id": audit_page_id,
            "document_id": document["document_id"],
            "page_index": int(page_index),
            "source_page_fingerprint": semantic_sha256(
                {
                    "source_sha256": document["source"]["sha256"],
                    "page_index": int(page_index),
                    "clean_png_sha256": _sha256_file(clean_path),
                }
            ),
            "files": {
                "clean_source": "clean.png",
                "node_overlay": "overlay.png",
                "current_page_markdown": "current_page.md",
                "page_nodes": "page_nodes.json",
                "context": "context.json",
                "evidence": evidence_refs,
            },
            "node_ids": [block["node_id"] for block in page_nodes],
            "context_node_ids": [block["node_id"] for block in context],
            "review_risk_summary": {
                "reading_order_risk": page_record.get("reading_order_risk"),
                "review_node_count": sum(
                    block.get("review_state") not in {None, "NONE"} for block in page_nodes
                ),
                "kind_counts": dict(sorted(Counter(block["kind"] for block in page_nodes).items())),
            },
            "truth_bearing_fields_included": False,
        }
        _write_json(output / "request.json", package)
        return package

    def _render_source_page(self, source_pdf: Path, page_index: int, target: Path) -> None:
        import fitz

        with fitz.open(source_pdf) as pdf:
            if page_index < 0 or page_index >= pdf.page_count:
                raise ValueError(f"Source PDF page index is invalid: {page_index}")
            pixmap = pdf[page_index].get_pixmap(
                matrix=fitz.Matrix(self.render_scale, self.render_scale), alpha=False
            )
            pixmap.save(target)

    def _write_overlay(
        self,
        clean_path: Path,
        overlay_path: Path,
        page_nodes: list[dict[str, Any]],
    ) -> None:
        from PIL import Image, ImageDraw

        image = Image.open(clean_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for ordinal, block in enumerate(page_nodes, 1):
            bbox = block.get("bbox_pdf_pt")
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, (int, float)) for value in bbox)
            ):
                continue
            scaled = tuple(round(float(value) * self.render_scale) for value in bbox)
            color = (220, 32 + (ordinal * 37) % 160, 32 + (ordinal * 71) % 160)
            draw.rectangle(scaled, outline=color, width=2)
            label = f"{ordinal}:{block['node_id'][-8:]}"
            x, y = scaled[0], max(0, scaled[1] - 12)
            draw.rectangle((x, y, x + 7 * len(label), y + 12), fill="white")
            draw.text((x + 1, y), label, fill=color)
        image.save(overlay_path, format="PNG")

    @staticmethod
    def _portable_node(block: dict[str, Any]) -> dict[str, Any]:
        content = block.get("content", {})
        return {
            "node_id": block["node_id"],
            "page_index": block["page_index"],
            "kind": block["kind"],
            "subtype": block.get("subtype"),
            "bbox_pdf_pt": copy.deepcopy(block.get("bbox_pdf_pt")),
            "current_content": {
                "text": content.get("text"),
                "latex": content.get("latex"),
                "asset_ref": content.get("asset_ref"),
                "source_status": content.get("source_status"),
                "table_status": content.get("table_ir", {})
                .get("quality", {})
                .get("status"),
            },
            "relations": copy.deepcopy(block.get("relations", {})),
            "review_state": block.get("review_state"),
        }


def _page_features(document: dict[str, Any], page_index: int) -> set[str]:
    page = next(page for page in document["pages"] if int(page["page_index"]) == page_index)
    blocks = [
        block for block in document.get("blocks", []) if int(block["page_index"]) == page_index
    ]
    kinds = {str(block["kind"]) for block in blocks}
    features = {f"KIND_{kind}" for kind in kinds}
    features.add(f"READING_ORDER_{page.get('reading_order_risk', 'UNKNOWN')}")
    features.add(f"SOURCE_{page.get('source_profile', 'UNKNOWN')}")
    if int(page.get("column_count") or 1) > 1:
        features.add("MULTI_COLUMN")
    if len(kinds) >= 3:
        features.add("MIXED_CONTENT")
    if any(block.get("review_state") not in {None, "NONE"} for block in blocks):
        features.add("REVIEW_STATE")
    if any(
        "OCR" in json.dumps(block.get("provenance", {}), ensure_ascii=False).upper()
        for block in blocks
    ):
        features.add("OCR_ROUTE")
    for block in blocks:
        if block.get("kind") == "TABLE":
            status = (
                block.get("content", {})
                .get("table_ir", {})
                .get("quality", {})
                .get("status", "UNKNOWN")
            )
            features.add(f"TABLE_{status}")
    return features


def _select_pages(document: dict[str, Any], count: int) -> list[int]:
    candidates = [int(page["page_index"]) for page in document.get("pages", [])]
    feature_map = {page: _page_features(document, page) for page in candidates}
    selected: list[int] = []
    covered: set[str] = set()
    weights = {
        "READING_ORDER_HIGH": 8,
        "READING_ORDER_MEDIUM": 4,
        "MULTI_COLUMN": 4,
        "MIXED_CONTENT": 5,
        "OCR_ROUTE": 3,
        "REVIEW_STATE": 2,
        "KIND_FORMULA": 4,
        "KIND_TABLE": 5,
        "KIND_IMAGE": 3,
        "KIND_CAPTION": 3,
        "KIND_TITLE": 2,
        "TABLE_STRUCTURED_TABLE": 3,
        "TABLE_PARTIAL_TABLE": 5,
        "TABLE_TABLE_REVIEW_REQUIRED": 5,
    }
    while candidates and len(selected) < count:
        candidates.sort(
            key=lambda page: (
                -sum(weights.get(feature, 1) for feature in feature_map[page] - covered),
                semantic_sha256({"document_id": document["document_id"], "page": page}),
            )
        )
        chosen = candidates.pop(0)
        selected.append(chosen)
        covered.update(feature_map[chosen])
    return sorted(selected)


def build_output_audit_eval_split(
    documents: dict[str, dict[str, Any]],
    *,
    reference_documents: int = 12,
    validation_documents: int = 6,
    pages_per_document: int = 5,
) -> dict[str, Any]:
    if reference_documents + validation_documents > len(documents):
        raise ValueError("Output Audit split requests more documents than available")
    ranked = []
    for document_id, document in documents.items():
        features = set().union(
            *(
                _page_features(document, int(page["page_index"]))
                for page in document.get("pages", [])
            )
        )
        ranked.append(
            (
                -len(features),
                semantic_sha256({"document_id": document_id, "schema": "output-audit-split-v0"}),
                document_id,
            )
        )
    ranked.sort()
    reference_ids = [row[2] for row in ranked[:reference_documents]]
    validation_ids = [
        row[2]
        for row in ranked[reference_documents : reference_documents + validation_documents]
    ]

    def rows(ids: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "document_id": document_id,
                "page_indices": _select_pages(documents[document_id], pages_per_document),
                "selection_basis": "PRE_AUDIT_METADATA_ONLY",
            }
            for document_id in sorted(ids)
        ]

    split = {
        "schema": "bemarkdown-output-audit-eval-split-v0",
        "selection_features": [
            "document_page_type",
            "reading_order_risk",
            "formula_density",
            "table_status_distribution",
            "ocr_route_density",
            "image_density",
            "native_scanned_profile",
        ],
        "post_audit_features_used": False,
        "reference": rows(reference_ids),
        "validation": rows(validation_ids),
        "document_overlap_count": 0,
        "page_overlap_count": 0,
        "validation_status": "NOT_CONSUMED",
    }
    split["reference_document_count"] = len(split["reference"])
    split["reference_page_count"] = sum(len(row["page_indices"]) for row in split["reference"])
    split["validation_document_count"] = len(split["validation"])
    split["validation_page_count"] = sum(len(row["page_indices"]) for row in split["validation"])
    split["split_sha256"] = semantic_sha256(split)
    return split


def output_audit_result_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": RESULT_SCHEMA,
        "type": "object",
        "required": ["schema", "audit_page_id", "status", "issues", "patches", "unresolved"],
        "properties": {
            "schema": {"const": RESULT_SCHEMA},
            "audit_page_id": {"type": "string"},
            "status": {"enum": sorted(RESULT_STATUSES)},
            "issues": {"type": "array", "items": {"type": "object"}},
            "patches": {"type": "array", "items": {"type": "object"}},
            "unresolved": {"type": "array", "items": {"type": "object"}},
        },
        "additionalProperties": False,
        "complete_row_required_for_every_page": True,
        "issue_taxonomy": sorted(ISSUE_TAXONOMY),
        "patch_operations": sorted(PATCH_OPERATIONS),
        "correction_bases": sorted(CORRECTION_BASES),
        "semantic_guess_only_is_automatic_basis": False,
    }


class ReferenceAuditBundleWriter:
    schema = "bemarkdown-output-audit-reference-bundle-v0"

    def write(
        self,
        documents: dict[str, dict[str, Any]],
        split: dict[str, Any],
        target: str | Path,
        archive: str | Path,
    ) -> dict[str, Any]:
        target = Path(target)
        archive = Path(archive)
        if target.exists() or archive.exists():
            raise FileExistsError("Fresh Output Audit Reference bundle target already exists")
        (target / "pages").mkdir(parents=True)
        manifest_rows = []
        builder = PageAuditPackageBuilder()
        for cohort in split["reference"]:
            document = documents[cohort["document_id"]]
            for page_index in cohort["page_indices"]:
                audit_page_id = "audit-page-" + semantic_sha256(
                    {
                        "document_id": document["document_id"],
                        "page_index": page_index,
                        "source_sha256": document["source"]["sha256"],
                        "schema": PAGE_PACKAGE_SCHEMA,
                    }
                )[:24]
                page_dir = target / "pages" / audit_page_id
                package = builder.build(document, page_index, page_dir)
                manifest_rows.append(
                    {
                        "audit_page_id": package["audit_page_id"],
                        "document_id": document["document_id"],
                        "page_index": page_index,
                        "package": f"pages/{audit_page_id}/request.json",
                        "clean_png_sha256": _sha256_file(page_dir / "clean.png"),
                        "overlay_png_sha256": _sha256_file(page_dir / "overlay.png"),
                        "truth_bearing_fields_included": False,
                    }
                )
        (target / "audit_manifest.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in manifest_rows
            ),
            encoding="utf-8",
            newline="\n",
        )
        _write_json(target / "result_schema.json", output_audit_result_schema())
        (target / "README.md").write_text(
            "# Fresh Output Audit Reference Bundle v0\n\n"
            "Audit every page package under `pages/` in a fresh isolated vision context. "
            "Compare the source page with the current candidate and stable node overlay. "
            "Return one complete JSON object per manifest row; do not rewrite a document. "
            "This bundle contains no Reference truth, expected patch, or Validation data.\n",
            encoding="utf-8",
            newline="\n",
        )
        (target / "OUTPUT_AUDIT_REFERENCE_HANDOFF.md").write_text(
            "# Output Audit Reference Handoff\n\n"
            "Use `result_schema.json`. Allowed corrections must be source-visible conversion "
            "errors and must use stable-ID patches with source evidence. If uncertain, return "
            "`UNRESOLVED`; source-content anomalies are recorded but not rewritten.\n",
            encoding="utf-8",
            newline="\n",
        )
        bundle = {
            "schema": self.schema,
            "status": "OUTPUT_AUDIT_REFERENCE_QUALITY_PENDING_EXTERNAL_VISION",
            "split_sha256": split["split_sha256"],
            "document_count": len({row["document_id"] for row in manifest_rows}),
            "page_count": len(manifest_rows),
            "truth_bearing_fields_included": False,
            "provider_binding": None,
            "validation_included": False,
            "validation_status": "NOT_CONSUMED",
        }
        _write_json(target / "bundle_manifest.json", bundle)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle_zip:
            for path in sorted(target.rglob("*")):
                if path.is_file():
                    bundle_zip.write(path, path.relative_to(target).as_posix())
        bundle["archive_path"] = str(archive.resolve())
        bundle["archive_sha256"] = _sha256_file(archive)
        bundle["archive_bytes"] = archive.stat().st_size
        return bundle


class OutputAuditResultImporter:
    """Fail-closed importer for complete provider-neutral page results."""

    def parse(
        self,
        rows: list[dict[str, Any]],
        page_packages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        expected = {str(row["audit_page_id"]): row for row in page_packages}
        ids = [str(row.get("audit_page_id") or "") for row in rows]
        duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"Output Audit result contains duplicate pages: {duplicates}")
        unknown = sorted(set(ids) - set(expected))
        if unknown:
            raise ValueError(f"Output Audit result contains unknown pages: {unknown}")
        missing = sorted(set(expected) - set(ids))
        if missing:
            raise ValueError(f"Output Audit result is missing pages: {missing}")

        parsed = []
        for row in rows:
            if row.get("schema") != RESULT_SCHEMA:
                raise ValueError("Unsupported Output Audit result schema")
            if row.get("status") not in RESULT_STATUSES:
                raise ValueError("Unsupported Output Audit result status")
            for field in ("issues", "patches", "unresolved"):
                if not isinstance(row.get(field), list):
                    raise TypeError(f"Output Audit result field {field} must be a list")
            if row["status"] == "NO_CHANGE" and row["patches"]:
                raise ValueError("NO_CHANGE Output Audit result cannot contain patches")
            for issue in [*row["issues"], *row["unresolved"]]:
                issue_type = issue.get("issue_type")
                if issue_type not in ISSUE_TAXONOMY:
                    raise ValueError(f"Unsupported Output Audit issue type: {issue_type}")
            package = expected[str(row["audit_page_id"])]
            allowed_nodes = set(package.get("node_ids", []))
            for patch in row["patches"]:
                op = patch.get("op")
                if op not in PATCH_OPERATIONS:
                    raise ValueError(f"Unsupported Output Audit patch operation: {op}")
                referenced = {
                    str(patch[field])
                    for field in (
                        "target_node_id",
                        "before_node_id",
                        "after_node_id",
                        "caption_node_id",
                        "asset_node_id",
                    )
                    if patch.get(field)
                }
                referenced.update(str(value) for value in patch.get("target_node_ids", []))
                unknown_nodes = sorted(referenced - allowed_nodes)
                if unknown_nodes:
                    raise ValueError(
                        f"Output Audit patch references unknown node(s): {unknown_nodes}"
                    )
                if patch.get("issue_type") not in ISSUE_TAXONOMY:
                    raise ValueError("Output Audit patch requires a known issue type")
            parsed.append(row)
        return parsed


class SourceFidelityGuard:
    schema = "bemarkdown-source-fidelity-guard-v0"

    def validate(
        self,
        document: dict[str, Any],
        page_package: dict[str, Any],
        patch: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        basis = patch.get("correction_basis")
        if basis not in CORRECTION_BASES or basis in {
            "SOURCE_CONTENT_ANOMALY_ONLY",
            "SEMANTIC_GUESS_ONLY",
        }:
            reasons.append("AUTOMATIC_BASIS_NOT_VISUAL")
        if not patch.get("source_evidence_refs"):
            reasons.append("SOURCE_EVIDENCE_MISSING")
        for reference in patch.get("source_evidence_refs", []):
            path = str(reference).split("#", 1)[0].replace("\\", "/")
            if path.startswith(("/", "../")) or "/../" in path:
                reasons.append("SOURCE_EVIDENCE_REF_UNSAFE")
        page_index = int(page_package["page_index"])
        allowed = set(page_package.get("node_ids", []))
        referenced = _patch_node_ids(patch)
        if referenced - allowed:
            reasons.append("PATCH_TARGET_OUTSIDE_AUDIT_PAGE")
        by_id = {block["node_id"]: block for block in document.get("blocks", [])}
        if any(
            node_id in by_id and int(by_id[node_id]["page_index"]) != page_index
            for node_id in referenced
        ):
            reasons.append("PATCH_CROSSES_PAGE")
        if patch.get("op") == "INSERT_BLOCK" and int(
            patch.get("page_index", page_index)
        ) != page_index:
            reasons.append("PATCH_CROSSES_PAGE")
        if patch.get("op") in {"REPLACE_TEXT", "REPLACE_FORMULA"}:
            node = by_id.get(str(patch.get("target_node_id")))
            current = ""
            if node:
                field = "new_text" if patch["op"] == "REPLACE_TEXT" else "new_latex"
                source_field = "text" if field == "new_text" else "latex"
                current = str(node.get("content", {}).get(source_field) or "")
                replacement = str(patch.get(field) or "")
                if len(replacement) > max(2000, len(current) * 3 + 100):
                    reasons.append("UNBOUNDED_REWRITE")
        return sorted(set(reasons))


def validate_audited_document(document: dict[str, Any]) -> dict[str, Any]:
    validate_document_ir(document)
    formula_checked = 0
    table_checked = 0
    formula_validator = FormulaStructuralValidator()
    for block in document.get("blocks", []):
        if block.get("kind") == "FORMULA" and block.get("content", {}).get("latex"):
            conversion = FormulaConversion(
                latex=str(block["content"]["latex"]),
                status="AUDIT_PATCH",
                component="output_audit",
                warnings=[],
            )
            verdict = formula_validator.validate(conversion).verdict.value
            if verdict == "INVALID":
                raise ValueError(f"Patched formula is invalid: {block['node_id']}")
            formula_checked += 1
        table = block.get("content", {}).get("table_ir")
        if block.get("kind") == "TABLE" and table:
            validate_table_ir(table)
            table_checked += 1
    return {
        "schema": "bemarkdown-output-audit-machine-validation-v0",
        "document_ir_valid": True,
        "formula_nodes_checked": formula_checked,
        "table_nodes_checked": table_checked,
        "asset_refs_valid": True,
        "graph_relations_valid": True,
        "accounting_valid": True,
    }


class OutputAuditPatchPipeline:
    """Apply a complete page result without allowing silent patch failure."""

    schema = "bemarkdown-output-audit-patch-application-v0"

    def __init__(self, guard: SourceFidelityGuard | None = None):
        self.guard = guard or SourceFidelityGuard()

    def apply(
        self,
        document: dict[str, Any],
        page_package: dict[str, Any],
        result: dict[str, Any],
        *,
        agent: dict[str, Any],
        timestamp: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        OutputAuditResultImporter().parse([result], [page_package])
        current = copy.deepcopy(document)
        result_fingerprint = semantic_sha256(result)
        applied: list[dict[str, Any]] = []
        timestamp = timestamp or datetime.now(UTC).isoformat()
        for ordinal, patch in enumerate(result["patches"]):
            node_ids = _patch_node_ids(patch)
            before = _node_view(current, node_ids)
            patch_id = str(
                patch.get("patch_id")
                or "audit-patch-"
                + semantic_sha256(
                    {
                        "audit_page_id": result["audit_page_id"],
                        "ordinal": ordinal,
                        "patch": patch,
                    }
                )[:20]
            )
            base = {
                "patch_id": patch_id,
                "audit_page_id": result["audit_page_id"],
                "target_node_ids": sorted(node_ids),
                "op": patch["op"],
                "issue_type": patch.get("issue_type"),
                "correction_basis": patch.get("correction_basis"),
                "source_evidence_refs": list(patch.get("source_evidence_refs", [])),
                "before_content": before,
                "before_sha256": semantic_sha256(before),
                "agent": copy.deepcopy(agent),
                "result_fingerprint": result_fingerprint,
                "timestamp": timestamp,
            }
            reasons = self.guard.validate(current, page_package, patch)
            if reasons:
                applied.append(
                    {
                        **base,
                        "state": "REJECTED",
                        "reason": "PATCH_REJECTED_SOURCE_FIDELITY",
                        "reason_codes": reasons,
                        "after_content": before,
                        "after_sha256": semantic_sha256(before),
                    }
                )
                continue
            try:
                candidate, engine_log = OutputAuditPatchEngine().apply(current, [patch])
                unintended = _unintended_node_mutations(current, candidate, node_ids)
                if unintended:
                    raise ValueError(
                        f"Patch modified non-target nodes: {unintended}"
                    )
                validation = validate_audited_document(candidate)
                created = set(engine_log["operations"][0].get("created_node_ids", []))
                after = _node_view(candidate, node_ids | created)
                current = candidate
                applied.append(
                    {
                        **base,
                        "state": "APPLIED",
                        "reason": None,
                        "reason_codes": [],
                        "after_content": after,
                        "after_sha256": semantic_sha256(after),
                        "created_node_ids": sorted(created),
                        "machine_validation": validation,
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                applied.append(
                    {
                        **base,
                        "state": "REJECTED",
                        "reason": "PATCH_REJECTED_MACHINE_VALIDATION",
                        "reason_codes": [type(exc).__name__, str(exc)],
                        "after_content": before,
                        "after_sha256": semantic_sha256(before),
                    }
                )
        unresolved = [
            {
                "state": "UNRESOLVED",
                "audit_page_id": result["audit_page_id"],
                **copy.deepcopy(row),
            }
            for row in result["unresolved"]
        ]
        report = {
            "schema": self.schema,
            "audit_page_id": result["audit_page_id"],
            "result_status": result["status"],
            "result_fingerprint": result_fingerprint,
            "patches": applied,
            "unresolved": unresolved,
            "counts": dict(Counter(row["state"] for row in [*applied, *unresolved])),
            "silent_failure_count": 0,
        }
        return current, report


def classify_asset_lifecycle(document: dict[str, Any]) -> list[dict[str, Any]]:
    references: dict[str, list[dict[str, Any]]] = {}
    for block in document.get("blocks", []):
        asset_uid = block.get("content", {}).get("asset_uid")
        if asset_uid:
            references.setdefault(str(asset_uid), []).append(block)
    rows = []
    for asset in document.get("assets", []):
        asset_uid = str(asset["asset_uid"])
        role = str(asset.get("role") or "").upper()
        blocks = references.get(asset_uid, [])
        if any(token in role for token in ("TRANSIENT", "NORMALIZED", "DEBUG", "TEMPORARY")):
            lifecycle = "TRANSIENT_ASSET"
            reason = "TRANSIENT_ROLE"
        elif any(block.get("kind") == "IMAGE" for block in blocks):
            lifecycle = "CONTENT_ASSET"
            reason = "VISIBLE_IMAGE"
        elif any(
            block.get("kind") == "FORMULA"
            and not str(block.get("content", {}).get("latex") or "").strip()
            for block in blocks
        ):
            lifecycle = "CONTENT_ASSET"
            reason = "UNRESOLVED_FORMULA_FALLBACK"
        elif any(
            block.get("kind") == "TABLE"
            and block.get("content", {}).get("table_ir", {}).get("quality", {}).get("status")
            != "STRUCTURED_TABLE"
            for block in blocks
        ):
            lifecycle = "CONTENT_ASSET"
            reason = "UNRESOLVED_TABLE_FALLBACK"
        elif any(
            block.get("kind") in {"TEXT", "TITLE", "CAPTION", "OTHER"}
            and not str(block.get("content", {}).get("text") or "").strip()
            for block in blocks
        ):
            lifecycle = "CONTENT_ASSET"
            reason = "UNRESOLVED_CONTENT_FALLBACK"
        else:
            lifecycle = "AUDIT_ASSET"
            reason = "SOURCE_OR_REVIEW_EVIDENCE_ONLY"
        rows.append(
            {
                "asset_uid": asset_uid,
                "source_relative_path": document_asset_presentation_ref(asset),
                "role": asset.get("role"),
                "lifecycle": lifecycle,
                "reason": reason,
            }
        )
    return rows


class CleanHandoffRenderer:
    """Render consumer Markdown directly from DocumentIR, never from Draft text."""

    version = "clean-handoff-markdown-renderer-v1"

    def render(
        self,
        document: dict[str, Any],
        asset_ref_map: dict[str, str] | None = None,
        *,
        unresolved_assets: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        from .pdf.clean_markdown import render_clean_document

        return render_clean_document(
            document, dict(asset_ref_map or {}), self._render_block,
            renderer_version=self.version, unresolved_assets=unresolved_assets,
        )

    def _render_block(
        self, block: dict[str, Any], asset_ref_map: dict[str, str], *, inline: bool = False
    ) -> tuple[str, str | None]:
        content = block.get("content", {})
        kind = block.get("kind")
        text = str(content.get("text") or "").replace("\r\n", "\n").replace("\r", "\n")
        latex = str(content.get("latex") or "").strip()
        asset_uid = str(content.get("asset_uid") or "") or None
        asset_ref = asset_ref_map.get(asset_uid or "", content.get("asset_ref"))
        if kind == "TITLE":
            level = min(6, max(1, int(content.get("heading_level") or 1)))
            return (f"{'#' * level} {text}" if text else "[Unresolved title]", None)
        if kind in {"TEXT", "CAPTION"}:
            if text:
                return text, None
            if asset_ref:
                return f"![文本视觉保留]({asset_ref})", asset_uid
            return "[Unresolved text]", None
        if kind == "FORMULA":
            latex = latex or text.strip()
            if latex:
                return (f"${latex}$" if inline else f"$$\n{latex}\n$$"), None
            if asset_ref:
                return f"![公式视觉保留]({asset_ref})", asset_uid
            return "[Unresolved formula]", None
        if kind == "IMAGE":
            if asset_ref:
                from .pdf.figure_labels import figure_alt_text
                return f"![{figure_alt_text(content)}]({asset_ref})", asset_uid
            return "[Unresolved image]", None
        if kind == "TABLE":
            serialized = content.get("table_serialization", {})
            output_format = serialized.get("format")
            if output_format in {"MARKDOWN", "HTML"}:
                from .pdf_table_engine import (
                    TABLE_SERIALIZER_SCHEMA,
                    render_table_block,
                )

                if serialized.get("schema") == TABLE_SERIALIZER_SCHEMA and content.get("table_ir"):
                    # Structured tables retain their source crop too. Re-render
                    # with the current package reference before registering it.
                    return render_table_block(block, asset_ref=asset_ref), asset_uid
                return str(serialized.get("body") or "[Unresolved table]"), None
            if output_format == "PARTIAL_WITH_IMAGE":
                structured = "\n".join(
                    line
                    for line in str(serialized.get("body") or "").splitlines()
                    if not line.strip().startswith("<!--")
                    and not line.strip().startswith("![")
                ).strip()
                fallback = f"![表格视觉保留]({asset_ref})" if asset_ref else "[Unresolved table]"
                return "\n".join(value for value in (structured, fallback) if value), asset_uid
            if asset_ref:
                return f"![表格视觉保留]({asset_ref})", asset_uid
            return "[Unresolved table]", None
        if text:
            return text, None
        if asset_ref:
            return f"![内容视觉保留]({asset_ref})", asset_uid
        return "[Unresolved source content]", None


def project_handoff_document_ir(
    document: dict[str, Any], asset_ref_map: dict[str, str]
) -> list[dict[str, Any]]:
    rows = []
    for block in document.get("blocks", []):
        content = block.get("content", {})
        asset_uid = str(content.get("asset_uid") or "")
        projected_content: dict[str, Any] = {}
        for field in ("text", "latex", "heading_level", "source_status"):
            if content.get(field) is not None:
                projected_content[field] = copy.deepcopy(content[field])
        if asset_uid in asset_ref_map:
            projected_content["asset_uid"] = asset_uid
            projected_content["asset_ref"] = asset_ref_map[asset_uid]
        table = content.get("table_ir")
        if block.get("kind") == "TABLE" and table:
            projected_content["table"] = {
                "schema": table.get("schema"),
                "table_id": table.get("table_id"),
                "rows": table.get("rows"),
                "columns": table.get("columns", table.get("cols")),
                "cells": [
                    {
                        key: copy.deepcopy(cell.get(key))
                        for key in (
                            "cell_id",
                            "row",
                            "column",
                            "rowspan",
                            "colspan",
                            "text",
                            "content_type",
                            "review_state",
                        )
                        if cell.get(key) is not None
                    }
                    for cell in table.get("cells", [])
                ],
                "quality": copy.deepcopy(table.get("quality")),
            }
            serialized = content.get("table_serialization", {})
            projected_content["table_serialization"] = {
                "format": serialized.get("format"),
                "body": "\n".join(
                    line
                    for line in str(serialized.get("body") or "").splitlines()
                    if not line.strip().startswith("<!--")
                ),
            }
        rows.append(
            {
                "schema": "bemarkdown-handoff-document-node-v0",
                "node_id": block["node_id"],
                "page": block["page_index"],
                "kind": block["kind"],
                "subtype": block.get("subtype"),
                "content": projected_content,
                "reading_order": copy.deepcopy(block.get("order_key")),
                "bbox_pdf_pt": copy.deepcopy(block.get("bbox_pdf_pt")),
                "relations": copy.deepcopy(block.get("relations", {})),
                "unresolved_state": (
                    block.get("review_state")
                    if block.get("review_state") not in {None, "NONE"}
                    else None
                ),
                "source_provenance": {
                    "source_content_ids": copy.deepcopy(block.get("source_content_ids", [])),
                    "source_region_ids": copy.deepcopy(block.get("source_region_ids", [])),
                    "source_unit_ids": copy.deepcopy(block.get("source_unit_ids", [])),
                },
            }
        )
    return rows


class LeanHandoffPackager:
    schema = "bemarkdown-handoff-package-v0"

    def write(
        self,
        document: dict[str, Any],
        source_package: str | Path,
        target: str | Path,
        *,
        audit_status: str,
    ) -> dict[str, Any]:
        validate_document_ir(document)
        source_package = Path(source_package).resolve()
        target = Path(target).resolve()
        if target.exists():
            raise FileExistsError(f"Handoff target already exists: {target}")
        target.mkdir(parents=True)
        asset_dir = target / "assets"
        metadata_dir = target / "_bemarkdown"
        asset_dir.mkdir()
        metadata_dir.mkdir()

        lifecycle = classify_asset_lifecycle(document)
        asset_by_uid = {str(asset["asset_uid"]): asset for asset in document.get("assets", [])}
        asset_ref_map: dict[str, str] = {}
        linked_bytes = 0
        for row in lifecycle:
            if row["lifecycle"] != "CONTENT_ASSET":
                continue
            uid = row["asset_uid"]
            asset = asset_by_uid[uid]
            relative = Path(document_asset_presentation_ref(asset))
            source = (source_package / relative).resolve()
            if source_package not in source.parents or not source.is_file():
                raise ValueError(f"Missing or unsafe content asset: {relative.as_posix()}")
            destination = asset_dir / relative.name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            asset_ref_map[uid] = f"assets/{relative.name}"
            linked_bytes += destination.stat().st_size

        renderer = CleanHandoffRenderer()
        rendered = renderer.render(document, asset_ref_map)
        second = renderer.render(copy.deepcopy(document), asset_ref_map)
        if rendered["markdown"] != second["markdown"]:
            raise RuntimeError("Clean Handoff renderer is not deterministic")
        (target / "document.md").write_text(
            rendered["markdown"], encoding="utf-8", newline="\n"
        )
        projection = project_handoff_document_ir(document, asset_ref_map)
        (metadata_dir / "document_ir.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in projection
            ),
            encoding="utf-8",
            newline="\n",
        )
        unresolved_count = sum(
            block.get("review_state") not in {None, "NONE"}
            for block in document.get("blocks", [])
        )
        manifest = {
            "schema_version": self.schema,
            "document_id": document["document_id"],
            "source_name": document["source"]["name"],
            "source_sha256": document["source"]["sha256"],
            "bemarkdown_version": "0.1.0+phase6b2i1",
            "audit_status": audit_status,
            "unresolved_count": unresolved_count,
            "markdown": "../document.md",
            "document_ir": "document_ir.jsonl",
        }
        (metadata_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        markdown = rendered["markdown"]
        refs = re.findall(r"!\[[^\]]*\]\((assets/[^)]+)\)", markdown)
        missing = [ref for ref in refs if not (target / ref).is_file()]
        expected_nodes = {block["node_id"] for block in document.get("blocks", [])}
        projected_nodes = {row["node_id"] for row in projection}
        if expected_nodes != projected_nodes:
            raise RuntimeError("Handoff DocumentIR projection lost visible nodes")
        internal_markers = sum(
            token in markdown
            for token in (
                "<!--",
                "bemarkdown:block",
                "REVIEW_REQUIRED",
                "TABLE_PARTIAL_REVIEW_REQUIRED",
            )
        )
        metrics = {
            "schema": "bemarkdown-handoff-package-metrics-v0",
            "document_id": document["document_id"],
            "visible_nodes": len(expected_nodes),
            "projected_nodes": len(projected_nodes),
            "content_assets_retained": sum(
                row["lifecycle"] == "CONTENT_ASSET" for row in lifecycle
            ),
            "audit_assets_excluded": sum(
                row["lifecycle"] == "AUDIT_ASSET" for row in lifecycle
            ),
            "transient_assets_excluded": sum(
                row["lifecycle"] == "TRANSIENT_ASSET" for row in lifecycle
            ),
            "content_asset_bytes": linked_bytes,
            "handoff_anchor_count": markdown.count("bemarkdown:block"),
            "internal_technical_marker_count": internal_markers,
            "missing_content_asset_refs": len(missing),
            "deterministic": True,
            "passed": not missing and not internal_markers and expected_nodes == projected_nodes,
        }
        if not metrics["passed"]:
            raise RuntimeError(f"Lean Handoff package validation failed: {metrics}")
        return metrics
