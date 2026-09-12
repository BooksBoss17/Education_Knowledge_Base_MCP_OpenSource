from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from .package import DocxResourceLimits, PackageIndex
from .wmf import WmfInspector


def audit_docx_resource_corpus(
    manifest_path: str | Path,
    *,
    limits: DocxResourceLimits | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    limits = limits or DocxResourceLimits()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        row
        for row in manifest.get("records", [])
        if str(row.get("file_type", "")).upper() == "DOCX"
    ]
    document_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        source = (manifest_path.parent / record["target_relative_path"]).resolve()
        actual_sha = _sha256_file(source)
        expected_sha = record.get("source_sha256")
        package = PackageIndex(source, limits=limits)
        wmf_records = 0
        wmf_width = 0
        wmf_height = 0
        wmf_inch = 0
        wmf_occurrences = 0
        for name, data in package.parts.items():
            if (
                not name.startswith("word/media/")
                or PurePosixPath(name).suffix.lower() != ".wmf"
            ):
                continue
            wmf_occurrences += 1
            inspection = WmfInspector().inspect(data)
            wmf_records = max(wmf_records, inspection.record_count)
            wmf_inch = max(wmf_inch, inspection.inch or 0)
            if inspection.logical_bounds:
                left, top, right, bottom = inspection.logical_bounds
                wmf_width = max(wmf_width, max(0, right - left))
                wmf_height = max(wmf_height, max(0, bottom - top))
        document_rows.append(
            {
                "document_id": f"docx-{index:04d}",
                "source_path": record["target_relative_path"],
                "source_sha256": actual_sha,
                "source_sha_valid": actual_sha == expected_sha,
                **package.resource_profile.to_dict(),
                "largest_wmf_record_count": wmf_records,
                "largest_wmf_logical_width": wmf_width,
                "largest_wmf_logical_height": wmf_height,
                "largest_wmf_inch": wmf_inch,
                "wmf_occurrences": wmf_occurrences,
                "image_header_warnings": sum(
                    warning.startswith("Could not inspect image dimensions")
                    for warning in package.warnings
                ),
            }
        )
        if progress:
            progress(f"Resource audit {index:03d}/{len(records)}")

    metrics = (
        "archive_bytes",
        "total_uncompressed_bytes",
        "part_count",
        "largest_xml_bytes",
        "largest_media_bytes",
        "largest_ole_bytes",
        "largest_other_bytes",
        "maximum_compression_ratio",
        "largest_image_width",
        "largest_image_height",
        "largest_image_pixels",
        "largest_wmf_record_count",
        "largest_wmf_logical_width",
        "largest_wmf_logical_height",
        "largest_wmf_inch",
    )
    observed = {
        metric: {
            "max": max((row[metric] for row in document_rows), default=0),
            "p95": _percentile([row[metric] for row in document_rows], 0.95),
        }
        for metric in metrics
    }
    defaults = limits.to_dict()
    metric_to_limit = {
        "archive_bytes": "max_archive_bytes",
        "total_uncompressed_bytes": "max_total_uncompressed_bytes",
        "part_count": "max_part_count",
        "largest_xml_bytes": "max_single_xml_bytes",
        "largest_media_bytes": "max_single_media_bytes",
        "largest_ole_bytes": "max_single_ole_bytes",
        "largest_other_bytes": "max_single_other_bytes",
        "maximum_compression_ratio": "max_compression_ratio",
        "largest_image_width": "max_image_width",
        "largest_image_height": "max_image_height",
        "largest_image_pixels": "max_image_pixels",
    }
    headroom = {}
    for metric, limit_name in metric_to_limit.items():
        maximum = observed[metric]["max"]
        limit = defaults[limit_name]
        headroom[metric] = {
            "limit": limit,
            "observed_max": maximum,
            "multiple": round(limit / maximum, 3) if maximum else None,
        }
    return {
        "schema": "bemarkdown-phase4a-resource-census-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "docx_total": len(records),
        "source_sha_valid": sum(row["source_sha_valid"] for row in document_rows),
        "observed": observed,
        "chosen_defaults": defaults,
        "headroom": headroom,
        "all_within_defaults": len(document_rows) == len(records)
        and all(row["source_sha_valid"] for row in document_rows),
        "documents": document_rows,
    }


def _percentile(values: list[int | float], percentile: float) -> int | float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
