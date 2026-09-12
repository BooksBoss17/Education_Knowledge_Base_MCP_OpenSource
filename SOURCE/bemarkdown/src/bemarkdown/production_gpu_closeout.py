from __future__ import annotations

import copy
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .production_runtime import DocumentWindowScheduler, semantic_sha256

FROZEN_FORMAL_PAGE_COUNT = 100
FROZEN_FORMAL_DOCUMENT_COUNT = 16
FULL_EQUIVALENCE_LAYERS = (
    "page_region_ir",
    "fusion",
    "router",
    "content_ir",
    "formula",
    "table_ir",
    "document_ir",
    "lean_handoff",
)
REQUIRED_STAGE_NAMES = (
    "source_extraction_render",
    "layout",
    "fusion_router",
    "ocr_det",
    "ocr_rec",
    "formulanet",
    "table_classifier",
    "slanext",
    "rt_detr",
    "table_ocr",
    "document_ir",
    "draft",
    "asset_finalize",
    "lean_handoff",
)
REQUIRED_PROMOTION_GATES = (
    "portable_full_measured",
    "optimized_full_measured",
    "layout_batching_measured",
    "ocr_batching_measured",
    "formula_batching_measured",
    "vram_envelopes_measured",
    "model_pool_validated",
    "vram_scheduler_validated",
    "selected_profile_no_oom",
    "silent_cpu_fallback_count_zero",
    "live_layout_to_frozen_fusion_continuity",
    "full_output_equivalence_passed",
    "lean_handoff_passed",
    "optimized_materially_faster",
)

REVIEW_ASSET_REQUIREMENT_POLICY_VERSION = "review-asset-requirement-policy-v2"
_VISUAL_ASSET_KINDS = {"FORMULA", "TABLE", "IMAGE"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_asset_requirement(row: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve source-bound visual fallbacks without promoting recognition."""

    content_kind = str(row.get("content_kind") or "OTHER").upper()
    semantic_hint = str(row.get("semantic_hint") or "").upper()
    semantic_kind = (
        semantic_hint
        if semantic_hint in _VISUAL_ASSET_KINDS
        else content_kind
        if content_kind in _VISUAL_ASSET_KINDS
        else "OTHER"
    )
    if semantic_kind == "FORMULA":
        required = not str(row.get("latex") or "").strip()
        reason = "FORMULA_WITHOUT_LATEX_REQUIRES_REVIEW_CROP" if required else "FORMULA_LATEX_PRESENT"
    elif semantic_kind == "TABLE":
        required = True
        reason = "TABLE_REQUIRES_VISUAL_FALLBACK"
    elif semantic_kind == "IMAGE":
        required = True
        reason = "IMAGE_REQUIRES_VISUAL_SOURCE"
    elif (
        content_kind in {"REVIEW", "OTHER"}
        and "VECTOR_VISUAL_FALLBACK"
        in row.get("provenance", {}).get("source_candidate_kinds", [])
        and not str(row.get("text") or "").strip()
        and not str(row.get("latex") or "").strip()
    ):
        # An unknown vector can be a diagram or a decoration. Its source
        # geometry is evidence to preserve, not a claim of semantic success.
        required = True
        reason = "SOURCE_VECTOR_WITHOUT_CONTENT_REQUIRES_REVIEW_CROP"
    else:
        required = False
        reason = "TEXT_OR_GENERIC_REVIEW_HAS_NO_REQUIRED_BINARY_ASSET"
    return {
        "policy_version": REVIEW_ASSET_REQUIREMENT_POLICY_VERSION,
        "content_kind": content_kind,
        "semantic_hint": semantic_hint,
        "semantic_kind": semantic_kind,
        "requires_source_visual_asset": required,
        "reason": reason,
    }


def _valid_source_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _validate_materialized_review_asset(
    *,
    row: Mapping[str, Any],
    path: str | None,
    crop: Mapping[str, Any] | None,
    route: Mapping[str, Any] | None,
) -> dict[str, Any]:
    route_id = str(row.get("route_id") or "")
    if not _valid_source_bbox(row.get("bbox_pdf_pt")):
        raise ValueError(
            f"REVIEW_SOURCE_CROP_MATERIALIZATION_FAILED:SOURCE_BBOX_INVALID:{route_id}"
        )
    if not path:
        raise ValueError(
            f"REVIEW_REQUIRED_ASSET_NOT_MATERIALIZED:BINARY_ARTIFACT_REF_MISSING:{route_id}"
        )
    source = Path(path)
    if not source.is_file():
        raise ValueError(
            f"REVIEW_REQUIRED_ASSET_NOT_MATERIALIZED:SOURCE_FILE_MISSING:{route_id}:{path}"
        )
    byte_count = source.stat().st_size
    if byte_count <= 0:
        raise ValueError(
            f"REVIEW_REQUIRED_ASSET_NOT_MATERIALIZED:SOURCE_FILE_EMPTY:{route_id}:{path}"
        )
    actual_sha256 = sha256_file(source)
    expected_sha256 = str((crop or {}).get("content_sha256") or "")
    if expected_sha256 and expected_sha256 != actual_sha256:
        raise ValueError(
            f"REVIEW_REQUIRED_ASSET_NOT_MATERIALIZED:SOURCE_SHA256_MISMATCH:{route_id}"
        )
    width = (crop or {}).get("width")
    height = (crop or {}).get("height")
    if (width is not None or height is not None) and (
        not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or float(width) <= 0
        or float(height) <= 0
    ):
        raise ValueError(
            f"REVIEW_SOURCE_CROP_MATERIALIZATION_FAILED:CROP_DIMENSIONS_INVALID:{route_id}"
        )
    crop_bbox = (crop or {}).get("bbox_pdf_pt")
    if crop_bbox is not None and not _valid_source_bbox(crop_bbox):
        raise ValueError(
            f"REVIEW_SOURCE_CROP_MATERIALIZATION_FAILED:CROP_BBOX_INVALID:{route_id}"
        )
    route_source = str((route or {}).get("provenance", {}).get("source_path") or "")
    crop_source = str((crop or {}).get("source_path") or "")
    if route_source and crop_source and Path(route_source).resolve() != Path(crop_source).resolve():
        raise ValueError(
            f"REVIEW_SOURCE_CROP_MATERIALIZATION_FAILED:SOURCE_AUTHORITY_MISMATCH:{route_id}"
        )
    return {
        "path": str(source),
        "content_sha256": actual_sha256,
        "bytes": byte_count,
        "width": width,
        "height": height,
        "bbox_pdf_pt": list(crop_bbox) if crop_bbox is not None else None,
        "source_path": crop_source or route_source or None,
        "page_index": (crop or {}).get("page_index", (route or {}).get("page_index")),
    }


def materialize_missing_asset_refs(
    rows: Sequence[Mapping[str, Any]],
    *,
    routes_by_id: Mapping[str, Mapping[str, Any]],
    render_crop: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep failed/deferred visual nodes packageable without changing their status."""

    recovered = []
    required_count = 0
    validated_count = 0
    output = copy.deepcopy(list(rows))
    for row in output:
        requirement = review_asset_requirement(row)
        if not requirement["requires_source_visual_asset"]:
            continue
        required_count += 1
        provenance = row.setdefault("provenance", {})
        candidates = (
            provenance.get("performance_closeout_recovery_crop"),
            provenance.get("preserved_render_crop"),
            provenance.get("preserved_input"),
            provenance.get("crop"),
        )
        preserved_crop = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, Mapping) and candidate.get("path")
            ),
            None,
        )
        path = str(row.get("binary_artifact_ref") or "") or (
            str(preserved_crop["path"]) if preserved_crop is not None else None
        )
        method = "REUSE_EXISTING_BINARY_ARTIFACT"
        crop = preserved_crop
        route_id = str(row.get("route_id"))
        route = routes_by_id.get(route_id)
        if not row.get("binary_artifact_ref") and preserved_crop is not None:
            method = "REUSE_PRESERVED_RENDER_CROP"
        if path is None:
            if route is None:
                raise ValueError(f"ASSET_RECOVERY_ROUTE_MISSING:{route_id}")
            if not _valid_source_bbox(row.get("bbox_pdf_pt")):
                raise ValueError(
                    "REVIEW_SOURCE_CROP_MATERIALIZATION_FAILED:"
                    f"SOURCE_BBOX_INVALID:{route_id}"
                )
            try:
                crop = render_crop(dict(route))
            except Exception as exc:
                raise ValueError(
                    "REVIEW_SOURCE_CROP_MATERIALIZATION_FAILED:"
                    f"RENDER_FAILED:{route_id}:{type(exc).__name__}:{exc}"
                ) from exc
            path = str(crop.get("path") or "") if isinstance(crop, Mapping) else None
            provenance["performance_closeout_recovery_crop"] = crop
            method = "RENDER_FALLBACK_CROP"
        validation = _validate_materialized_review_asset(
            row=row,
            path=path,
            crop=crop,
            route=route,
        )
        row["binary_artifact_ref"] = path
        provenance["performance_closeout_asset_recovery"] = {
            "method": method,
            "semantic_status_unchanged": True,
            "requirement": requirement,
            "validation": validation,
        }
        validated_count += 1
        if method != "REUSE_EXISTING_BINARY_ARTIFACT":
            recovered.append(
                {
                    "route_id": row.get("route_id"),
                    "content_kind": row.get("content_kind"),
                    "semantic_hint": row.get("semantic_hint"),
                    "status": row.get("status"),
                    "method": method,
                }
            )
    return output, {
        "schema": "bemarkdown-phase6b2p1-asset-materialization-recovery-v0",
        "policy_version": REVIEW_ASSET_REQUIREMENT_POLICY_VERSION,
        "required_count": required_count,
        "validated_count": validated_count,
        "recovered_count": len(recovered),
        "rows": recovered,
        "semantic_statuses_changed": 0,
    }


def validate_frozen_formal_workload(
    manifest: Mapping[str, Any],
    *,
    observed_manifest_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate the already-frozen performance workload without reselecting pages."""

    profiles = manifest.get("profiles")
    if not isinstance(profiles, Mapping) or "Formal" not in profiles:
        raise ValueError("FORMAL_PERFORMANCE_PROFILE_MISSING")
    formal = profiles["Formal"]
    pages = list(formal.get("pages") or [])
    identities = [
        (str(row.get("document_id")), int(row.get("page_index", -1))) for row in pages
    ]
    documents = {document_id for document_id, _ in identities}
    checks = {
        "manifest_sha256": observed_manifest_sha256.lower()
        == expected_manifest_sha256.lower(),
        "page_count": len(pages) == FROZEN_FORMAL_PAGE_COUNT,
        "declared_page_count": int(formal.get("page_count", -1))
        == FROZEN_FORMAL_PAGE_COUNT,
        "document_count": len(documents) == FROZEN_FORMAL_DOCUMENT_COUNT,
        "page_identities_unique": len(identities) == len(set(identities)),
        "source_sha256_present": all(
            len(str(row.get("source_sha256") or "")) == 64 for row in pages
        ),
        "page_ids_match": all(
            str(row.get("page_id")) == f"{document_id}:{page_index}"
            for row, (document_id, page_index) in zip(pages, identities, strict=True)
        ),
        "source_integrity_frozen": bool(
            manifest.get("source_integrity", {}).get("all_unchanged")
        ),
        "performance_only_role": manifest.get("role")
        == "PERFORMANCE_ONLY_NOT_SEMANTIC_QUALITY",
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError("FROZEN_FORMAL_WORKLOAD_INVALID:" + ",".join(failed))
    return {
        "schema": "bemarkdown-phase6b2p1-frozen-workload-validation-v0",
        "manifest_sha256": observed_manifest_sha256.lower(),
        "page_count": len(pages),
        "document_count": len(documents),
        "page_identity_sha256": semantic_sha256(identities),
        "source_sha256_count": len({str(row["source_sha256"]) for row in pages}),
        "checks": checks,
        "gate": "PASS",
    }


def validate_stage_breakdown(stage_seconds: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_STAGE_NAMES) - set(stage_seconds))
    invalid = sorted(
        name
        for name in REQUIRED_STAGE_NAMES
        if name in stage_seconds
        and (
            isinstance(stage_seconds[name], bool)
            or not isinstance(stage_seconds[name], (int, float))
            or float(stage_seconds[name]) < 0
        )
    )
    return {
        "required": list(REQUIRED_STAGE_NAMES),
        "missing": missing,
        "invalid": invalid,
        "gate": "PASS" if not missing and not invalid else "FAIL",
    }


def select_equivalent_batch(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_fingerprint: str,
    vram_budget_bytes: int,
) -> dict[str, Any]:
    """Select maximum measured throughput, not simply the largest batch."""

    eligible = [
        dict(row)
        for row in candidates
        if not row.get("oom")
        and row.get("output_fingerprint") == baseline_fingerprint
        and int(row.get("semantic_mismatch_count", 0)) == 0
        and row.get("peak_vram_bytes") is not None
        and int(row["peak_vram_bytes"]) <= vram_budget_bytes
    ]
    if not eligible:
        raise ValueError("NO_EQUIVALENT_BATCH_WITHIN_VRAM_BUDGET")
    return max(
        eligible,
        key=lambda row: (
            float(row.get("throughput_per_second") or 0.0),
            -int(row["batch_size"]),
        ),
    )


def layer_fingerprint(
    rows: Any,
    *,
    id_fields: Sequence[str] = (),
    exclude_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Create a compact semantic snapshot while excluding performance metadata."""

    default_exclusions = {
        "captured_at",
        "created_at",
        "elapsed_seconds",
        "inference_seconds",
        "latency_seconds",
        "load_seconds",
        "performance",
        "process_id",
        "resource_samples",
        "runtime_seconds",
        "timestamp",
        "wall_seconds",
    }
    exclusions = default_exclusions | (exclude_keys or set())
    value = rows if isinstance(rows, list) else [rows]
    ids = []
    for row in value:
        if isinstance(row, Mapping):
            identity = tuple(str(row.get(field)) for field in id_fields)
            if id_fields:
                ids.append(identity)
    return {
        "node_count": len(value),
        "id_count": len(ids),
        "ids_unique": len(ids) == len(set(ids)),
        "ids_sha256": semantic_sha256(sorted(ids)) if id_fields else None,
        "content_semantic_sha256": semantic_sha256(value, exclude_keys=exclusions),
    }


def compare_layer_fingerprints(
    portable: Mapping[str, Mapping[str, Any]],
    optimized: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for layer in FULL_EQUIVALENCE_LAYERS:
        left = portable.get(layer)
        right = optimized.get(layer)
        if left is None and right is None:
            classification = "NOT_APPLICABLE"
        elif left is None or right is None:
            classification = "MISMATCH"
        elif dict(left) == dict(right):
            classification = "IDENTICAL"
        elif (
            left.get("node_count") == right.get("node_count")
            and left.get("ids_sha256") == right.get("ids_sha256")
            and left.get("content_semantic_sha256")
            == right.get("content_semantic_sha256")
        ):
            classification = "SEMANTICALLY_EQUIVALENT"
        else:
            classification = "MISMATCH"
        rows.append(
            {
                "layer": layer,
                "classification": classification,
                "portable": left,
                "optimized": right,
            }
        )
    mismatches = [row["layer"] for row in rows if row["classification"] == "MISMATCH"]
    return {
        "schema": "bemarkdown-phase6b2p1-full-output-equivalence-v0",
        "layers": rows,
        "classification_counts": dict(
            sorted(Counter(row["classification"] for row in rows).items())
        ),
        "material_mismatch_count": len(mismatches),
        "mismatch_layers": mismatches,
        "gate": "PASS" if not mismatches else "FAIL",
    }


def decide_profile_promotion(
    gates: Mapping[str, bool],
    *,
    portable_wall_seconds: float,
    optimized_wall_seconds: float,
    minimum_material_speedup: float = 1.3,
) -> dict[str, Any]:
    observed = {name: bool(gates.get(name, False)) for name in REQUIRED_PROMOTION_GATES}
    if portable_wall_seconds <= 0 or optimized_wall_seconds <= 0:
        raise ValueError("FULL_PIPELINE_WALL_SECONDS_MUST_BE_POSITIVE")
    speedup = portable_wall_seconds / optimized_wall_seconds
    observed["optimized_materially_faster"] = speedup >= minimum_material_speedup
    failed = sorted(name for name, passed in observed.items() if not passed)
    promoted = not failed
    return {
        "schema": "bemarkdown-phase6b2p1-profile-promotion-decision-v0",
        "gates": observed,
        "minimum_material_speedup": minimum_material_speedup,
        "full_local_speedup": round(speedup, 6),
        "failed_gates": failed,
        "promoted": promoted,
        "default_profile": "optimized_fp32" if promoted else "portable_fp32",
        "status": (
            "OPTIMIZED_FP32_PROFILE_READY"
            if promoted
            else "OPTIMIZED_FP32_PROFILE_NOT_PROMOTED"
        ),
        "phase_status": (
            "PHASE_6B2P_COMPLETE"
            if promoted
            else "PHASE_6B2P_CORRECTNESS_PASSED_PERFORMANCE_PARTIAL"
        ),
    }


def vram_envelope_complete(envelope: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required_points = (
        "before_load_bytes",
        "after_load_bytes",
        "warm_steady_state_bytes",
        "peak_inference_bytes",
        "after_unload_bytes",
    )
    incomplete = {
        family: [name for name in required_points if row.get(name) is None]
        for family, row in envelope.items()
        if any(row.get(name) is None for name in required_points)
    }
    return {
        "families": len(envelope),
        "required_points": list(required_points),
        "incomplete": incomplete,
        "gate": "PASS" if envelope and not incomplete else "FAIL",
    }


def assert_cold_namespace_isolation(portable: str | Path, optimized: str | Path) -> None:
    left = Path(portable).resolve()
    right = Path(optimized).resolve()
    if left == right or left in right.parents or right in left.parents:
        raise ValueError("BENCHMARK_CACHE_NAMESPACES_MUST_BE_DISJOINT")


def summarize_closeout_storage(
    *,
    portable_root: str | Path,
    optimized_root: str | Path,
    cache_root: str | Path,
    portable_render_bytes: Sequence[int],
    optimized_render_bytes: Sequence[int],
    max_window_pages: int = 24,
    byte_budget: int = 256 * 1024**2,
) -> dict[str, Any]:
    """Measure logical disk use and the bounded source-window byte envelope."""

    def files_under(root: Path) -> list[Path]:
        return [path for path in root.rglob("*") if path.is_file()]

    def logical_metrics(root: Path) -> dict[str, int]:
        files = files_under(root)
        return {
            "file_count": len(files),
            "logical_bytes": sum(path.stat().st_size for path in files),
        }

    def profile_metrics(root_value: str | Path, render_bytes: Sequence[int]) -> dict[str, Any]:
        root = Path(root_value)
        estimates = [int(value) for value in render_bytes]
        if not estimates or any(value < 0 for value in estimates):
            raise ValueError("SOURCE_RENDER_BYTES_MUST_BE_NON_EMPTY_AND_NON_NEGATIVE")
        effective_budget = max(int(byte_budget), max(estimates))
        windows = DocumentWindowScheduler(
            max_pages=max_window_pages,
            byte_budget=effective_budget,
        ).plan(
            [
                {"page_index": index, "estimated_bytes": value}
                for index, value in enumerate(estimates)
            ]
        )
        handoff_root = root / "handoff"
        handoff_asset_files = [
            path
            for path in files_under(handoff_root)
            if "assets" in path.relative_to(handoff_root).parts
        ]
        return {
            "profile_root": str(root.resolve()),
            "profile_disk": logical_metrics(root),
            "content_asset_store": logical_metrics(root / "content"),
            "handoff_assets": {
                "file_count": len(handoff_asset_files),
                "logical_bytes": sum(path.stat().st_size for path in handoff_asset_files),
            },
            "source_window_queue": {
                "measurement": "DOCUMENT_WINDOW_SCHEDULER_MAX_ESTIMATED_BYTES",
                "page_count": len(estimates),
                "window_count": len(windows),
                "max_pages": max_window_pages,
                "byte_budget": effective_budget,
                "max_bytes_observed": max(
                    int(window["estimated_bytes"]) for window in windows
                ),
            },
        }

    cache_path = Path(cache_root)
    return {
        "schema": "bemarkdown-phase6b2p1-storage-queue-resources-v0",
        "portable": profile_metrics(portable_root, portable_render_bytes),
        "optimized": profile_metrics(optimized_root, optimized_render_bytes),
        "cache_disk": {
            "root": str(cache_path.resolve()),
            **logical_metrics(cache_path),
        },
    }
