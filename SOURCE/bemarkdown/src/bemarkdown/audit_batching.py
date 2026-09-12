"""Bounded, checkpointable orchestration for page-level audit work.

The primitives in this module are provider- and cohort-neutral.  Reference
sharding is one development caller; a future production caller can use the same
planner, whole-batch validator, state model, and deterministic merger.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from .pdf_output_audit import (
    CORRECTION_BASES,
    ISSUE_TAXONOMY,
    PATCH_OPERATIONS,
    RESULT_SCHEMA,
    RESULT_STATUSES,
)

MAX_PAGES_PER_SHARD = 6
DEFAULT_MAX_PAYLOAD_BYTES = 12 * 1024 * 1024
DEFAULT_MAX_IMAGE_COUNT = 24
DEFAULT_MAX_EVIDENCE_BYTES = 8 * 1024 * 1024

_RESULT_FIELDS = {
    "schema",
    "audit_page_id",
    "status",
    "issues",
    "patches",
    "unresolved",
}
_NODE_ID_FIELDS = {
    "node_id",
    "target_node_id",
    "before_node_id",
    "after_node_id",
    "caption_node_id",
    "asset_node_id",
    "keep_node_id",
    "caption_for",
}
_NODE_IDS_FIELDS = {"node_ids", "target_node_ids"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path: str | Path, value: Any) -> None:
    _atomic_write(
        Path(path),
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    data = b"".join(_json_bytes(row) + b"\n" for row in rows)
    _atomic_write(Path(path), data)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _referenced_node_ids(value: Any) -> set[str]:
    references: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _NODE_ID_FIELDS and item:
                if isinstance(item, list):
                    references.update(str(node_id) for node_id in item if node_id)
                else:
                    references.add(str(item))
            elif key in _NODE_IDS_FIELDS:
                if isinstance(item, list):
                    references.update(str(node_id) for node_id in item if node_id)
                elif item:
                    references.add(str(item))
            else:
                references.update(_referenced_node_ids(item))
    elif isinstance(value, list):
        for item in value:
            references.update(_referenced_node_ids(item))
    return references


def _row_error_detail(
    row: Any, package: dict[str, Any]
) -> tuple[str, str] | None:
    if not isinstance(row, dict) or set(row) != _RESULT_FIELDS:
        return "SCHEMA_INVALID", "TOP_LEVEL_FIELDS_INVALID"
    if row.get("schema") != RESULT_SCHEMA or row.get("status") not in RESULT_STATUSES:
        return "SCHEMA_INVALID", "RESULT_SCHEMA_OR_STATUS_INVALID"
    for field in ("issues", "patches", "unresolved"):
        if not isinstance(row.get(field), list) or not all(
            isinstance(item, dict) for item in row[field]
        ):
            return "SCHEMA_INVALID", f"{field.upper()}_OBJECT_ARRAY_REQUIRED"
    if row["status"] == "NO_CHANGE" and row["patches"]:
        return "SCHEMA_INVALID", "NO_CHANGE_HAS_PATCHES"
    if row["status"] == "PATCHED" and not row["patches"]:
        return "SCHEMA_INVALID", "PATCHED_WITHOUT_PATCHES"
    if any(
        issue.get("issue_type") not in ISSUE_TAXONOMY
        for issue in [*row["issues"], *row["unresolved"]]
    ):
        return "SCHEMA_INVALID", "ISSUE_TYPE_INVALID"
    for patch in row["patches"]:
        operation = patch.get("op", patch.get("operation"))
        if operation not in PATCH_OPERATIONS:
            return "PATCH_OPERATION_INVALID", "PATCH_OPERATION_INVALID"
        if patch.get("op") and patch.get("operation") and patch["op"] != patch["operation"]:
            return "PATCH_OPERATION_INVALID", "PATCH_OPERATION_ALIAS_CONFLICT"
        issue_type = patch.get("issue_type")
        if issue_type is not None and issue_type not in ISSUE_TAXONOMY:
            return "SCHEMA_INVALID", "PATCH_ISSUE_TYPE_INVALID"
        basis = patch.get("correction_basis", patch.get("basis"))
        if basis is not None and basis not in CORRECTION_BASES:
            return "SCHEMA_INVALID", "PATCH_CORRECTION_BASIS_INVALID"
    allowed_nodes = {str(value) for value in package.get("node_ids", [])}
    referenced_nodes = _referenced_node_ids(
        [*row["issues"], *row["patches"], *row["unresolved"]]
    )
    if referenced_nodes - allowed_nodes:
        return "NODE_REFERENCE_INVALID", "NODE_REFERENCE_INVALID"
    return None


def _row_error(row: Any, package: dict[str, Any]) -> str | None:
    detail = _row_error_detail(row, package)
    return detail[0] if detail else None


@dataclass(frozen=True)
class AuditRecoveryResult:
    valid_rows: list[dict[str, Any]]
    report: dict[str, Any]


def recover_audit_rows(
    canonical_manifest: list[dict[str, Any]],
    page_packages: dict[str, dict[str, Any]],
    source_files: Iterable[str | Path],
) -> AuditRecoveryResult:
    """Recover only individually valid rows; duplicated IDs fail closed."""

    canonical_ids = [str(row["audit_page_id"]) for row in canonical_manifest]
    canonical_set = set(canonical_ids)
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    sources = []
    for source_value in sorted({Path(value).resolve() for value in source_files}):
        source = Path(source_value)
        raw = source.read_text(encoding="utf-8")
        parsed_ids = []
        parseable = 0
        nonempty = 0
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            nonempty += 1
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                excluded.append(
                    {
                        "source": str(source),
                        "line": line_number,
                        "audit_page_id": None,
                        "reason": "JSON_PARSE_ERROR",
                        "detail": str(exc),
                    }
                )
                continue
            parseable += 1
            audit_page_id = str(row.get("audit_page_id") or "") if isinstance(row, dict) else ""
            parsed_ids.append(audit_page_id)
            candidates.append(
                {
                    "source": source,
                    "line": line_number,
                    "audit_page_id": audit_page_id,
                    "row": row,
                }
            )
        sources.append(
            {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "nonempty_row_count": nonempty,
                "parseable_row_count": parseable,
                "audit_page_ids": parsed_ids,
                "fingerprint": _fingerprint(parsed_ids),
            }
        )

    counts = Counter(item["audit_page_id"] for item in candidates if item["audit_page_id"])
    duplicate_ids = {audit_page_id for audit_page_id, count in counts.items() if count > 1}
    valid_by_id: dict[str, dict[str, Any]] = {}
    for item in candidates:
        audit_page_id = item["audit_page_id"]
        reason = None
        if audit_page_id in duplicate_ids:
            reason = "DUPLICATE_ID"
        elif audit_page_id not in canonical_set:
            reason = "UNKNOWN_ID"
        elif not isinstance(item["row"], dict):
            reason = "SCHEMA_INVALID"
        else:
            reason = _row_error(item["row"], page_packages[audit_page_id])
        if reason:
            excluded.append(
                {
                    "source": str(item["source"]),
                    "line": item["line"],
                    "audit_page_id": audit_page_id or None,
                    "reason": reason,
                }
            )
        else:
            valid_by_id[audit_page_id] = item["row"]

    valid_rows = [valid_by_id[audit_page_id] for audit_page_id in canonical_ids if audit_page_id in valid_by_id]
    report = {
        "schema": "bemarkdown-audit-recovery-report-v0",
        "canonical_row_count": len(canonical_ids),
        "source_file_count": len(sources),
        "sources": sources,
        "valid_row_count": len(valid_rows),
        "valid_audit_page_ids": [row["audit_page_id"] for row in valid_rows],
        "valid_rows_fingerprint": _fingerprint(valid_rows),
        "excluded_row_count": len(excluded),
        "excluded": sorted(
            excluded,
            key=lambda item: (item["source"], item["line"], item["reason"]),
        ),
    }
    return AuditRecoveryResult(valid_rows=valid_rows, report=report)


@dataclass(frozen=True)
class AuditBatchManifest:
    rows: list[dict[str, Any]]
    fingerprint: str


class AuditBatchPlanner:
    """Make deterministic, consecutive batches from pre-audit payload facts."""

    def __init__(
        self,
        *,
        max_pages_per_batch: int = MAX_PAGES_PER_SHARD,
        max_payload_bytes: int | None = DEFAULT_MAX_PAYLOAD_BYTES,
        max_image_count: int | None = DEFAULT_MAX_IMAGE_COUNT,
        max_evidence_bytes: int | None = DEFAULT_MAX_EVIDENCE_BYTES,
        prefer_document_boundaries: bool = False,
        batch_id_prefix: str = "reference-shard",
    ):
        if max_pages_per_batch < 1:
            raise ValueError("max_pages_per_batch must be positive")
        for value in (max_payload_bytes, max_image_count, max_evidence_bytes):
            if value is not None and value < 1:
                raise ValueError("configured payload limits must be positive")
        self.max_pages_per_batch = int(max_pages_per_batch)
        self.max_payload_bytes = max_payload_bytes
        self.max_image_count = max_image_count
        self.max_evidence_bytes = max_evidence_bytes
        self.prefer_document_boundaries = bool(prefer_document_boundaries)
        self.batch_id_prefix = batch_id_prefix

    def plan(
        self,
        manifest_rows: list[dict[str, Any]],
        page_features: dict[str, dict[str, Any]],
    ) -> AuditBatchManifest:
        ids = [str(row["audit_page_id"]) for row in manifest_rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Audit batch input IDs must be unique")
        missing_features = sorted(set(ids) - set(page_features))
        if missing_features:
            raise ValueError(f"Page features missing for {missing_features}")

        groups: list[list[tuple[int, dict[str, Any], dict[str, Any]]]] = []
        current: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        totals = {"payload_bytes": 0, "image_count": 0, "evidence_bytes": 0}
        for fallback_position, row in enumerate(manifest_rows):
            canonical_position = int(row.get("canonical_order", fallback_position))
            feature = page_features[str(row["audit_page_id"])]
            next_totals = {
                key: totals[key] + int(feature.get(key, 0)) for key in totals
            }
            boundary = bool(
                current
                and self.prefer_document_boundaries
                and str(current[-1][1].get("document_id")) != str(row.get("document_id"))
            )
            exceeded = bool(
                current
                and (
                    len(current) >= self.max_pages_per_batch
                    or (
                        self.max_payload_bytes is not None
                        and next_totals["payload_bytes"] > self.max_payload_bytes
                    )
                    or (
                        self.max_image_count is not None
                        and next_totals["image_count"] > self.max_image_count
                    )
                    or (
                        self.max_evidence_bytes is not None
                        and next_totals["evidence_bytes"] > self.max_evidence_bytes
                    )
                )
            )
            if boundary or exceeded:
                groups.append(current)
                current = []
                totals = {"payload_bytes": 0, "image_count": 0, "evidence_bytes": 0}
            current.append((canonical_position, row, feature))
            totals = {key: totals[key] + int(feature.get(key, 0)) for key in totals}
        if current:
            groups.append(current)

        batches = []
        for ordinal, group in enumerate(groups, start=1):
            features = [item[2] for item in group]
            audit_page_ids = [str(item[1]["audit_page_id"]) for item in group]
            batches.append(
                {
                    "shard_id": f"{self.batch_id_prefix}-{ordinal:03d}",
                    "audit_page_ids": audit_page_ids,
                    "page_count": len(group),
                    "payload_bytes": sum(int(item.get("payload_bytes", 0)) for item in features),
                    "image_count": sum(int(item.get("image_count", 0)) for item in features),
                    "evidence_image_count": sum(
                        int(item.get("evidence_image_count", 0)) for item in features
                    ),
                    "evidence_bytes": sum(int(item.get("evidence_bytes", 0)) for item in features),
                    "canonical_order_start": group[0][0],
                    "canonical_order_end": group[-1][0],
                    "documents": [
                        {
                            "audit_page_id": str(item[1]["audit_page_id"]),
                            "document_id": str(item[1].get("document_id") or ""),
                            "page_index": int(item[1].get("page_index", 0)),
                        }
                        for item in group
                    ],
                }
            )
        return AuditBatchManifest(rows=batches, fingerprint=_fingerprint(batches))


def collect_zip_page_features(
    canonical_bundle: str | Path, canonical_manifest: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(canonical_bundle) as archive:
        infos = archive.infolist()
        for row in canonical_manifest:
            audit_page_id = str(row["audit_page_id"])
            prefix = f"pages/{audit_page_id}/"
            owned = [info for info in infos if info.filename.startswith(prefix) and not info.is_dir()]
            image_infos = [info for info in owned if Path(info.filename).suffix.lower() in _IMAGE_SUFFIXES]
            evidence_infos = [info for info in owned if info.filename.startswith(f"{prefix}evidence/")]
            features[audit_page_id] = {
                "document_id": row.get("document_id"),
                "page_index": row.get("page_index"),
                "payload_bytes": sum(info.file_size for info in owned),
                "image_count": len(image_infos),
                "evidence_image_count": sum(
                    Path(info.filename).suffix.lower() in _IMAGE_SUFFIXES
                    for info in evidence_infos
                ),
                "evidence_bytes": sum(info.file_size for info in evidence_infos),
            }
    return features


def load_bundle_contract(
    canonical_bundle: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    with zipfile.ZipFile(canonical_bundle) as archive:
        manifest = [
            json.loads(line)
            for line in archive.read("audit_manifest.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
        packages = {
            str(row["audit_page_id"]): json.loads(archive.read(str(row["package"])))
            for row in manifest
        }
    return manifest, packages


def _zip_write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


class ReferenceShardBundleWriter:
    """Project canonical page packages into minimal deterministic shard ZIPs."""

    def write(
        self,
        canonical_bundle: str | Path,
        canonical_manifest: list[dict[str, Any]],
        shard: dict[str, Any],
        output: str | Path,
    ) -> dict[str, Any]:
        canonical_bundle = Path(canonical_bundle)
        output = Path(output)
        selected_ids = [str(value) for value in shard["audit_page_ids"]]
        selected_set = set(selected_ids)
        selected_rows = [
            row for row in canonical_manifest if str(row["audit_page_id"]) in selected_set
        ]
        if [str(row["audit_page_id"]) for row in selected_rows] != selected_ids:
            raise ValueError("Shard IDs must be a canonical-order subset")
        if len(selected_ids) != len(selected_set):
            raise ValueError("Shard IDs must be unique")

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with zipfile.ZipFile(canonical_bundle) as source, zipfile.ZipFile(
            temporary, "w"
        ) as target:
            readme = (
                "# Fresh Output Audit Reference Shard\n\n"
                f"shard_id: `{shard['shard_id']}`  \n"
                f"expected result row count: `{len(selected_rows)}`\n\n"
                "Audit every manifest page in this shard in a fresh isolated vision context. "
                "Return one complete JSON object per manifest row. This shard contains no "
                "Reference truth, expected patch, recovered result, or Validation data.\n"
            ).encode()
            _zip_write(target, "README.md", readme)
            for name in ("OUTPUT_AUDIT_REFERENCE_HANDOFF.md", "result_schema.json"):
                _zip_write(target, name, source.read(name))
            manifest_data = b"".join(_json_bytes(row) + b"\n" for row in selected_rows)
            _zip_write(target, "audit_manifest.jsonl", manifest_data)
            prefixes = tuple(f"pages/{audit_page_id}/" for audit_page_id in selected_ids)
            page_entries = sorted(
                (
                    info
                    for info in source.infolist()
                    if not info.is_dir() and info.filename.startswith(prefixes)
                ),
                key=lambda info: info.filename,
            )
            for info in page_entries:
                lowered = info.filename.lower()
                if any(token in lowered for token in ("validation", "truth", "expected_patch")):
                    raise ValueError(f"Forbidden shard entry: {info.filename}")
                _zip_write(target, info.filename, source.read(info.filename))
        os.replace(temporary, output)
        return {
            **shard,
            "source_bundle_sha": sha256_file(canonical_bundle),
            "archive_path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "page_count": len(selected_rows),
            "audit_page_ids": selected_ids,
        }


class AuditBatchResultValidator:
    """Validate complete batch coverage; never salvage a partial batch."""

    _ERROR_KEYS = (
        "json_parse",
        "schema",
        "duplicate",
        "unknown",
        "missing",
        "node_reference",
        "patch_operation",
    )

    def validate(
        self,
        result_path: str | Path,
        expected_manifest: list[dict[str, Any]],
        page_packages: dict[str, dict[str, Any]],
        report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        result_path = Path(result_path)
        errors = {key: 0 for key in self._ERROR_KEYS}
        details = []
        rows = []
        for line_number, line in enumerate(
            result_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors["json_parse"] += 1
                details.append({"line": line_number, "reason": "JSON_PARSE_ERROR", "detail": str(exc)})
                continue
            rows.append((line_number, row))

        expected_ids = [str(row["audit_page_id"]) for row in expected_manifest]
        expected_set = set(expected_ids)
        result_ids = [
            str(row.get("audit_page_id") or "") if isinstance(row, dict) else ""
            for _, row in rows
        ]
        duplicates = {value for value, count in Counter(result_ids).items() if count > 1}
        unknown = set(result_ids) - expected_set
        missing = expected_set - set(result_ids)
        errors["duplicate"] = len(duplicates)
        errors["unknown"] = len(unknown)
        errors["missing"] = len(missing)

        for line_number, row in rows:
            audit_page_id = str(row.get("audit_page_id") or "") if isinstance(row, dict) else ""
            if audit_page_id in duplicates or audit_page_id not in expected_set:
                continue
            error_detail = _row_error_detail(row, page_packages[audit_page_id])
            reason = error_detail[0] if error_detail else None
            if reason == "SCHEMA_INVALID":
                errors["schema"] += 1
            elif reason == "NODE_REFERENCE_INVALID":
                errors["node_reference"] += 1
            elif reason == "PATCH_OPERATION_INVALID":
                errors["patch_operation"] += 1
            if reason:
                details.append(
                    {
                        "line": line_number,
                        "audit_page_id": audit_page_id,
                        "reason": reason,
                        "contract_error": error_detail[1],
                    }
                )

        valid = not any(errors.values()) and len(rows) == len(expected_ids)
        report = {
            "schema": "bemarkdown-audit-batch-result-validation-v0",
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "expected_rows": len(expected_ids),
            "observed_parseable_rows": len(rows),
            "validated_rows": len(rows) if valid else 0,
            "status": "VALIDATED" if valid else "INVALID_RESULT",
            "whole_batch_fail_closed": True,
            "errors": errors,
            "details": details,
        }
        if report_path is not None:
            write_json(report_path, report)
        return report


def validate_reference_shard_result(
    result_path: str | Path,
    expected_manifest: list[dict[str, Any]],
    page_packages: dict[str, dict[str, Any]],
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    return AuditBatchResultValidator().validate(
        result_path, expected_manifest, page_packages, report_path
    )


@dataclass(frozen=True)
class AuditBatchResultInventory:
    assignments: dict[str, Path]
    report: dict[str, Any]


class AuditBatchResultDiscoverer:
    """Assign result files to batches from their page IDs, never filename alone."""

    def discover(
        self,
        results_dir: str | Path,
        batches: list[dict[str, Any]],
    ) -> AuditBatchResultInventory:
        results_dir = Path(results_dir)
        expected = {
            str(batch["shard_id"]): {
                str(audit_page_id) for audit_page_id in batch["audit_page_ids"]
            }
            for batch in batches
        }
        candidates = []
        paths_by_shard: dict[str, list[Path]] = {shard_id: [] for shard_id in expected}
        for path in sorted(results_dir.glob("*.jsonl")):
            parse_errors = []
            audit_page_ids = []
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(
                        {
                            "line": line_number,
                            "error": type(exc).__name__,
                        }
                    )
                    continue
                if isinstance(row, dict) and row.get("audit_page_id"):
                    audit_page_ids.append(str(row["audit_page_id"]))
            observed = set(audit_page_ids)
            matching = sorted(
                shard_id
                for shard_id, expected_ids in expected.items()
                if observed & expected_ids
            )
            detected = matching[0] if len(matching) == 1 else None
            if detected:
                paths_by_shard[detected].append(path.resolve())
            candidates.append(
                {
                    "filename": path.name,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=UTC
                    ).isoformat(),
                    "sha256": sha256_file(path),
                    "parse_error_count": len(parse_errors),
                    "parse_errors": parse_errors,
                    "observed_row_count": len(audit_page_ids),
                    "unique_audit_page_ids": len(observed),
                    "detected_shard_id": detected,
                    "matching_shard_ids": matching,
                }
            )

        missing = sorted(
            shard_id for shard_id, paths in paths_by_shard.items() if not paths
        )
        ambiguous = sorted(
            shard_id for shard_id, paths in paths_by_shard.items() if len(paths) > 1
        )
        unassigned = sorted(
            row["filename"] for row in candidates if row["detected_shard_id"] is None
        )
        assignments = {
            shard_id: paths[0]
            for shard_id, paths in paths_by_shard.items()
            if len(paths) == 1
        }
        ready = not missing and not ambiguous and not unassigned
        if not ready:
            assignments = {}
        report = {
            "schema": "bemarkdown-audit-batch-result-inventory-v0",
            "results_dir": str(results_dir.resolve()),
            "expected_shards": sorted(expected),
            "candidate_file_count": len(candidates),
            "candidates": candidates,
            "missing_shards": missing,
            "ambiguous_shards": ambiguous,
            "unassigned_candidates": unassigned,
            "ready": ready,
            "assignment_basis": "RESULT_CONTENT_AUDIT_PAGE_IDS",
            "filename_only_assignment_allowed": False,
        }
        return AuditBatchResultInventory(assignments=assignments, report=report)


class AuditBatchState:
    STATUSES: ClassVar[set[str]] = {
        "PENDING",
        "RESULT_PRESENT_UNVALIDATED",
        "VALIDATED",
        "INVALID_RESULT",
    }

    @classmethod
    def create(
        cls,
        canonical_total: int,
        recovered_valid: int,
        shards: list[dict[str, Any]],
        results_dir: str | Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        results_dir = Path(results_dir)
        shard_rows = []
        for shard in shards:
            ordinal = str(shard["shard_id"]).rsplit("-", 1)[-1]
            result = results_dir / f"output_audit_reference_shard_{ordinal}_results.jsonl"
            status = "RESULT_PRESENT_UNVALIDATED" if result.is_file() else "PENDING"
            shard_rows.append(
                {
                    "shard_id": shard["shard_id"],
                    "status": status,
                    "result_path": str(result.resolve()),
                    "expected_rows": shard["page_count"],
                }
            )
        status_counts = Counter(row["status"] for row in shard_rows)
        master = {
            "canonical_total": canonical_total,
            "recovered_valid": recovered_valid,
            "remaining": canonical_total - recovered_valid,
            "shard_count": len(shard_rows),
            "pending_shards": status_counts["PENDING"],
            "validated_shards": status_counts["VALIDATED"],
            "final_merged": False,
        }
        return master, {"schema": "bemarkdown-audit-batch-state-v0", "shards": shard_rows}


class AuditBatchMerger:
    """Write a canonical-order final JSONL only when complete coverage is proven."""

    def merge(
        self,
        canonical_manifest: list[dict[str, Any]],
        recovered_rows: list[dict[str, Any]],
        validated_batch_rows: dict[str, list[dict[str, Any]]],
        shard_statuses: dict[str, str],
        final_path: str | Path,
        readiness_path: str | Path,
    ) -> dict[str, Any]:
        canonical_ids = [str(row["audit_page_id"]) for row in canonical_manifest]
        canonical_set = set(canonical_ids)
        all_rows = [*recovered_rows]
        for shard_id in sorted(validated_batch_rows):
            all_rows.extend(validated_batch_rows[shard_id])
        result_ids = [str(row.get("audit_page_id") or "") for row in all_rows]
        duplicates = sorted(value for value, count in Counter(result_ids).items() if count > 1)
        unknown = sorted(set(result_ids) - canonical_set)
        missing = [audit_page_id for audit_page_id in canonical_ids if audit_page_id not in set(result_ids)]
        missing_shards = sorted(
            shard_id for shard_id, status in shard_statuses.items() if status != "VALIDATED"
        )
        invalid_shards = sorted(
            shard_id
            for shard_id, status in shard_statuses.items()
            if status == "INVALID_RESULT"
        )
        pending_shards = sorted(
            shard_id
            for shard_id, status in shard_statuses.items()
            if status in {"PENDING", "RESULT_PRESENT_UNVALIDATED"}
        )
        ready = not duplicates and not unknown and not missing and not missing_shards
        report = {
            "schema": "bemarkdown-audit-final-merge-readiness-v0",
            "ready": ready,
            "canonical_rows": len(canonical_ids),
            "observed_rows": len(all_rows),
            "unique_ids": len(set(result_ids)),
            "coverage": f"{len(canonical_ids) - len(missing)}/{len(canonical_ids)}",
            "missing": missing,
            "unknown": unknown,
            "duplicate": duplicates,
            "missing_shards": missing_shards,
            "invalid_shards": invalid_shards,
            "pending_shards": pending_shards,
        }
        if ready:
            by_id = {str(row["audit_page_id"]): row for row in all_rows}
            ordered = [by_id[audit_page_id] for audit_page_id in canonical_ids]
            write_jsonl(final_path, ordered)
            report["rows"] = len(ordered)
            report["final_path"] = str(Path(final_path).resolve())
            report["final_sha256"] = sha256_file(final_path)
        write_json(readiness_path, report)
        return report


def merge_reference_audit_results(
    canonical_manifest: list[dict[str, Any]],
    recovered_rows: list[dict[str, Any]],
    validated_shard_rows: dict[str, list[dict[str, Any]]],
    shard_statuses: dict[str, str],
    final_path: str | Path,
    readiness_path: str | Path,
) -> dict[str, Any]:
    return AuditBatchMerger().merge(
        canonical_manifest,
        recovered_rows,
        validated_shard_rows,
        shard_statuses,
        final_path,
        readiness_path,
    )
