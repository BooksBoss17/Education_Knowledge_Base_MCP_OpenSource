"""Production-owned reference patch planning and dependency contracts."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

from .pdf_document_ir import semantic_sha256

PATCH_PLAN_SCHEMA = "bemarkdown-reference-patch-plan-v0"
DEPENDENCY_SCHEMA = "bemarkdown-reference-patch-dependency-report-v0"

_TARGET_FIELDS = (
    "target_node_id",
    "node_id",
)
_DESTRUCTIVE_OPERATIONS = {"DELETE_DUPLICATE", "SPLIT_BLOCK", "MERGE_BLOCK"}


def _first(value: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    return None


def _node_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _raw_target_ids(patch: dict[str, Any]) -> list[str]:
    values = _node_list(patch.get("target_node_ids"))
    if not values:
        values = _node_list(_first(patch, *_TARGET_FIELDS))
    if not values and patch.get("node_ids"):
        values = _node_list(patch["node_ids"])
    return list(dict.fromkeys(values))


def _raw_reference_ids(value: Any) -> set[str]:
    fields = {
        "node_id",
        "node_ids",
        "target_node_id",
        "target_node_ids",
        "before_node_id",
        "after_node_id",
        "anchor_node_id",
        "asset_node_id",
        "caption_for",
        "keep_node_id",
        "canonical_node_id",
        "duplicate_of",
        "duplicate_of_node_id",
        "duplicate_of_node_ids",
        "into_node_id",
        "surviving_node_id",
    }
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in fields:
                result.update(_node_list(item))
            elif key != "new_block_id":
                result.update(_raw_reference_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_raw_reference_ids(item))
    return result


def _evidence_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, str):
        refs.append(value)
    elif isinstance(value, list):
        for item in value:
            refs.extend(_evidence_refs(item))
    elif isinstance(value, dict):
        for key in ("file", "overlay_file", "path", "source_crop"):
            if isinstance(value.get(key), str):
                refs.append(str(value[key]))
    return list(dict.fromkeys(refs))


def _matching_issues(
    result: dict[str, Any], reference_ids: set[str]
) -> list[dict[str, Any]]:
    issues = list(result.get("issues", []))
    matching = [issue for issue in issues if _raw_reference_ids(issue) & reference_ids]
    if matching:
        return matching
    issue_types = {str(issue.get("issue_type")) for issue in issues}
    return issues if len(issue_types) == 1 else []


def _normalized_parts(patch: dict[str, Any]) -> list[Any]:
    parts = _first(patch, "parts", "segments")
    if parts is None and isinstance(patch.get("replacement"), dict):
        parts = patch["replacement"].get("parts")
    if not isinstance(parts, list):
        return []
    normalized = []
    for part in parts:
        if not isinstance(part, dict):
            normalized.append(str(part))
            continue
        item = copy.deepcopy(part)
        if isinstance(item.get("content"), str):
            field = "latex" if item.get("kind") == "FORMULA" else "text"
            item[field] = item.pop("content")
        normalized.append(item)
    return normalized


def _replacement_content(patch: dict[str, Any], field: str) -> Any:
    aliases = {
        "text": ("new_text", "replacement_text", "text"),
        "latex": ("new_latex", "replacement_latex", "latex"),
        "html": ("new_table_html", "replacement_html", "table_html", "html"),
    }
    value = _first(patch, *aliases[field])
    replacement = patch.get("replacement")
    if value is None and isinstance(replacement, dict):
        value = replacement.get(field)
    return value


def _patch_writes(operation: str, targets: list[str]) -> list[str]:
    field = {
        "REPLACE_TEXT": "content.text",
        "REPLACE_FORMULA": "content.latex",
        "REPLACE_TABLE": "content.table",
        "CHANGE_BLOCK_KIND": "kind",
        "UPDATE_CAPTION_RELATION": "relations.caption_for",
        "MOVE_BLOCK": "reading_order",
        "DELETE_DUPLICATE": "visibility",
        "SPLIT_BLOCK": "existence",
        "MERGE_BLOCK": "existence",
    }.get(operation)
    if operation == "INSERT_BLOCK":
        return []
    return [f"{node_id}:{field}" for node_id in targets] if field else []


class ReferencePatchPlanner:
    """Normalize external wire aliases before any DocumentIR mutation."""

    def plan(
        self,
        results: list[dict[str, Any]],
        page_packages: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for result in results:
            audit_page_id = str(result["audit_page_id"])
            package = page_packages[audit_page_id]
            result_fingerprint = semantic_sha256(result)
            for ordinal, raw in enumerate(result["patches"]):
                rows.append(
                    self._normalize(
                        result,
                        package,
                        raw,
                        ordinal,
                        result_fingerprint,
                    )
                )
        return {
            "schema": PATCH_PLAN_SCHEMA,
            "patch_count": len(rows),
            "page_count": len({row["audit_page_id"] for row in rows}),
            "operation_counts": dict(
                sorted(Counter(row["operation"] for row in rows).items())
            ),
            "issue_type_counts": dict(
                sorted(
                    Counter(
                        str(row["issue_type"] or "UNSPECIFIED") for row in rows
                    ).items()
                )
            ),
            "normalization_error_count": sum(
                bool(row["normalization_errors"]) for row in rows
            ),
            "patches": rows,
            "fingerprint": semantic_sha256(rows),
        }

    def _normalize(
        self,
        result: dict[str, Any],
        package: dict[str, Any],
        raw: dict[str, Any],
        ordinal: int,
        result_fingerprint: str,
    ) -> dict[str, Any]:
        operation = str(_first(raw, "op", "operation") or "")
        targets = _raw_target_ids(raw)
        reference_ids = _raw_reference_ids(raw)
        issues = _matching_issues(result, reference_ids)
        issue_types = sorted(
            {str(issue["issue_type"]) for issue in issues if issue.get("issue_type")}
        )
        issue_type = raw.get("issue_type")
        if issue_type is None and len(issue_types) == 1:
            issue_type = issue_types[0]
        refs = _evidence_refs(_first(raw, "source_evidence_refs", "source_evidence"))
        evidence_origin = "PATCH"
        if not refs:
            refs = list(
                dict.fromkeys(
                    ref
                    for issue in issues
                    for ref in _evidence_refs(
                        _first(
                            issue,
                            "source_evidence_refs",
                            "source_evidence",
                            "evidence",
                        )
                    )
                )
            )
            evidence_origin = "ISSUE" if refs else "NONE"
        patch_id = "reference-patch-" + semantic_sha256(
            {
                "audit_page_id": result["audit_page_id"],
                "ordinal": ordinal,
                "patch": raw,
                "result_fingerprint": result_fingerprint,
            }
        )[:24]
        normalized: dict[str, Any] = {
            "op": operation,
            "correction_basis": _first(raw, "correction_basis", "basis"),
            "source_evidence_refs": refs,
            "issue_type": issue_type,
        }
        errors: list[str] = []
        created_aliases: list[str] = []
        required_aliases: list[str] = []
        anchors: list[str] = []

        if operation in {
            "REPLACE_TEXT",
            "REPLACE_FORMULA",
            "REPLACE_TABLE",
            "CHANGE_BLOCK_KIND",
            "UPDATE_CAPTION_RELATION",
            "SPLIT_BLOCK",
        }:
            if len(targets) != 1:
                errors.append("TARGET_CARDINALITY_INVALID")
            elif targets:
                normalized["target_node_id"] = targets[0]

        if operation == "REPLACE_TEXT":
            value = _replacement_content(raw, "text")
            if value is None:
                errors.append("REPLACEMENT_TEXT_MISSING")
            else:
                normalized["new_text"] = str(value)
        elif operation == "REPLACE_FORMULA":
            value = _replacement_content(raw, "latex")
            if value is None:
                errors.append("REPLACEMENT_LATEX_MISSING")
            else:
                normalized["new_latex"] = str(value)
        elif operation == "REPLACE_TABLE":
            value = _replacement_content(raw, "html")
            if value is None:
                errors.append("REPLACEMENT_TABLE_MISSING")
            else:
                normalized["content"] = {
                    "table_serialization": {
                        "schema": "bemarkdown-table-serializer-v0",
                        "format": "HTML",
                        "body": str(value),
                    }
                }
        elif operation == "DELETE_DUPLICATE":
            if not targets:
                errors.append("TARGET_MISSING")
            normalized["target_node_ids"] = targets
        elif operation == "MOVE_BLOCK":
            if not targets:
                errors.append("TARGET_MISSING")
            normalized["target_node_ids"] = targets
            before = raw.get("before_node_id")
            after = raw.get("after_node_id")
            split = raw.get("after_split_segment")
            if (
                isinstance(split, dict)
                and split.get("node_id")
                and split.get("segment_key")
            ):
                alias = f"split:{split['node_id']}:{split['segment_key']}"
                normalized["after_node_alias"] = alias
                required_aliases.append(alias)
            elif before:
                normalized["before_node_id"] = str(before)
                anchors.append(str(before))
            elif after:
                normalized["after_node_id"] = str(after)
                anchors.append(str(after))
            else:
                errors.append("MOVE_ANCHOR_MISSING")
        elif operation == "INSERT_BLOCK":
            anchor = raw.get("anchor_node_id")
            position = str(raw.get("position") or "").lower()
            before = raw.get("before_node_id")
            after = raw.get("after_node_id")
            if anchor and position == "before":
                before = anchor
            elif anchor and position == "after":
                after = anchor
            if after:
                normalized["after_node_id"] = str(after)
                anchors.append(str(after))
            elif before:
                normalized["before_node_id"] = str(before)
                anchors.append(str(before))
            else:
                normalized["page_index"] = int(package["page_index"])
            block = raw.get("new_block") or raw.get("replacement")
            kind = _first(raw, "kind", "block_kind")
            text = raw.get("text")
            content = raw.get("content")
            if isinstance(block, dict):
                kind = block.get("kind", kind)
                text = block.get("text", text)
                content = block.get("content", content)
                if block.get("bbox_pdf_pt") is not None:
                    normalized["source_bbox_pdf_pt"] = block["bbox_pdf_pt"]
            normalized["kind"] = str(kind or "OTHER")
            if isinstance(content, dict):
                normalized["content"] = copy.deepcopy(content)
            else:
                normalized["content"] = {
                    "text": str(text if text is not None else content or "")
                }
            if raw.get("relations") is not None:
                normalized["relations"] = copy.deepcopy(raw["relations"])
            if raw.get("source_bbox_pdf_pt") is not None:
                normalized["source_bbox_pdf_pt"] = copy.deepcopy(
                    raw["source_bbox_pdf_pt"]
                )
            if raw.get("new_block_id"):
                alias = f"insert:{raw['new_block_id']}"
                normalized["created_alias"] = alias
                created_aliases.append(alias)
        elif operation == "SPLIT_BLOCK":
            parts = _normalized_parts(raw)
            if len(parts) < 2:
                errors.append("SPLIT_PARTS_INVALID")
            normalized["parts"] = parts
            if targets:
                for index, part in enumerate(parts):
                    if isinstance(part, dict) and part.get("segment_key"):
                        alias = f"split:{targets[0]}:{part['segment_key']}"
                    else:
                        alias = f"split:{patch_id}:{index}"
                    created_aliases.append(alias)
        elif operation == "MERGE_BLOCK":
            if len(targets) < 2:
                errors.append("MERGE_TARGETS_INVALID")
            normalized["target_node_ids"] = targets
            result_kind = raw.get("result_kind")
            if result_kind is not None:
                normalized["result_kind"] = str(result_kind)
            merged_content: dict[str, Any] = {}
            value = _first(raw, "replacement_text", "text")
            if value is not None:
                merged_content["text"] = str(value)
            if raw.get("latex") is not None:
                merged_content["latex"] = str(raw["latex"])
            if merged_content:
                normalized["merged_content"] = merged_content
        elif operation == "CHANGE_BLOCK_KIND":
            new_kind = _first(raw, "new_kind", "to_kind")
            if new_kind is None:
                errors.append("NEW_KIND_MISSING")
            else:
                normalized["new_kind"] = str(new_kind)
            if raw.get("heading_level") is not None:
                normalized["heading_level"] = int(raw["heading_level"])
        elif operation == "UPDATE_CAPTION_RELATION":
            caption_for = raw.get("caption_for")
            values = _node_list(caption_for)
            if len(values) > 1:
                errors.append("CAPTION_TARGET_CARDINALITY_INVALID")
            normalized["asset_node_id"] = values[0] if values else None
            if values:
                anchors.extend(values)
            if raw.get("caption_confidence") is not None:
                normalized["relation_confidence"] = raw["caption_confidence"]
        elif operation not in {
            "REPLACE_TEXT",
            "REPLACE_FORMULA",
            "REPLACE_TABLE",
            "DELETE_DUPLICATE",
            "MOVE_BLOCK",
            "INSERT_BLOCK",
            "SPLIT_BLOCK",
            "MERGE_BLOCK",
            "CHANGE_BLOCK_KIND",
            "UPDATE_CAPTION_RELATION",
        }:
            errors.append("OPERATION_UNSUPPORTED")

        return {
            "schema": PATCH_PLAN_SCHEMA,
            "patch_id": patch_id,
            "audit_page_id": str(result["audit_page_id"]),
            "document_id": str(package["document_id"]),
            "page_index": int(package["page_index"]),
            "ordinal": ordinal,
            "operation": operation,
            "target_node_ids": targets,
            "anchor_node_ids": list(dict.fromkeys(anchors)),
            "created_aliases": created_aliases,
            "required_aliases": required_aliases,
            "writes": _patch_writes(operation, targets),
            "consumes": targets if operation in _DESTRUCTIVE_OPERATIONS else [],
            "issue_type": issue_type,
            "correction_basis": normalized["correction_basis"],
            "source_evidence_refs": refs,
            "source_evidence_origin": evidence_origin,
            "original_result_fingerprint": result_fingerprint,
            "original_patch_fingerprint": semantic_sha256(raw),
            "normalized_patch": normalized,
            "normalization_errors": sorted(set(errors)),
        }


class ReferencePatchDependencyAnalyzer:
    def analyze(self, plan_rows: list[dict[str, Any]]) -> dict[str, Any]:
        alias_producers: dict[str, str] = {}
        statuses: dict[str, str] = {}
        consumed: dict[tuple[str, str], str] = {}
        writes: dict[tuple[str, str], str] = {}
        rows = []
        for row in plan_rows:
            patch_id = row["patch_id"]
            page = row["audit_page_id"]
            reasons = list(row["normalization_errors"])
            dependencies: list[str] = []
            status = "PATCH_DEPENDENCY_VALID"
            for alias in row["required_aliases"]:
                producer = alias_producers.get(alias)
                if producer is None or statuses.get(producer) != "PATCH_DEPENDENCY_VALID":
                    status = "PATCH_BLOCKED_BY_DEPENDENCY"
                    reasons.append(f"CREATED_ALIAS_UNAVAILABLE:{alias}")
                else:
                    dependencies.append(producer)
            for node_id in [*row["target_node_ids"], *row["anchor_node_ids"]]:
                producer = consumed.get((page, node_id))
                if producer:
                    status = "PATCH_BLOCKED_BY_DEPENDENCY"
                    reasons.append(f"NODE_CONSUMED_BY:{producer}")
                    dependencies.append(producer)
            if reasons and status == "PATCH_DEPENDENCY_VALID":
                status = "PATCH_CONFLICT"
            if status == "PATCH_DEPENDENCY_VALID":
                conflicting = [
                    writes[(page, key)]
                    for key in row["writes"]
                    if (page, key) in writes
                ]
                if conflicting:
                    status = "PATCH_CONFLICT"
                    reasons.extend(
                        f"INCOMPATIBLE_WRITE_WITH:{value}" for value in conflicting
                    )
                    dependencies.extend(conflicting)
            statuses[patch_id] = status
            if status == "PATCH_DEPENDENCY_VALID":
                for alias in row["created_aliases"]:
                    alias_producers[alias] = patch_id
                for node_id in row["consumes"]:
                    consumed[(page, node_id)] = patch_id
                for key in row["writes"]:
                    writes[(page, key)] = patch_id
            rows.append(
                {
                    "patch_id": patch_id,
                    "audit_page_id": page,
                    "status": status,
                    "dependencies": sorted(set(dependencies)),
                    "reason_codes": sorted(set(reasons)),
                }
            )
        return {
            "schema": DEPENDENCY_SCHEMA,
            "patch_count": len(rows),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in rows).items())
            ),
            "rows": rows,
            "fingerprint": semantic_sha256(rows),
        }


__all__ = [
    "DEPENDENCY_SCHEMA",
    "PATCH_PLAN_SCHEMA",
    "ReferencePatchDependencyAnalyzer",
    "ReferencePatchPlanner",
]
