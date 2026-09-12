from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CORPUS_MANIFEST_SCHEMA = "bemarkdown-corpus-v3-document-manifest-v1"
QUALITY_SAMPLE_SCHEMA = "bemarkdown-corpus-v3-quality-sample-v1"
EVAL_INSTANCE_SCHEMA = "bemarkdown-corpus-v3-eval-instance-v1"
TRUTH_SCHEMA_VERSION = "bemarkdown-corpus-v3-source-truth-v1"
TRUTH_CAPTURE_PAGE_SCHEMA = "bemarkdown-corpus-v3-truth-capture-page-v1"
SAMPLING_SEED = 63320260825
QUALITY_SAMPLE_SIZE = 50
MAX_TRUTH_SHARD_PAGES = 6

SOURCE_STRATA = (
    "native-text",
    "scan-heavy",
    "math-symbol-heavy-proxy",
    "image-heavy",
    "table/grid-proxy",
    "multi-column-proxy",
    "mixed",
    "plain-text",
    "source-other",
)

TRUTH_CAPTURE_ALLOWED_FIELDS = {
    "schema",
    "shard_id",
    "page_id",
    "document_id",
    "page_index",
    "source_pdf_sha256",
    "source_image",
    "source_image_sha256",
    "source_image_bytes",
}

TRUTH_LEAKAGE_TOKENS = (
    "truth",
    "expected",
    "gold",
    "reference answer",
    "reference_answer",
    "hidden labels",
    "hidden_labels",
)

REGISTERED_HASH_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
SHA256_PATTERN = re.compile(rb"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_registered_hashes(
    root: str | Path, *, exclude_roots: Sequence[str | Path] = ()
) -> set[str]:
    base = Path(root).resolve()
    excluded = [Path(path).resolve() for path in exclude_roots]
    hashes: set[str] = set()
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in REGISTERED_HASH_EXTENSIONS:
            continue
        resolved = path.resolve()
        if any(resolved == item or item in resolved.parents for item in excluded):
            continue
        tail = b""
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                data = tail + chunk
                hashes.update(match.group().decode("ascii").lower() for match in SHA256_PATTERN.finditer(data))
                tail = data[-63:]
    return hashes


def classify_document_role(relative_path: str, page_count: int) -> str:
    name = Path(relative_path).stem.casefold()
    if any(token in name for token in ("双解析", "解析", "讲义", "笔记", "handout", "notes")):
        return "handout/notes-like"
    if any(token in name for token in ("标准", "规范", "specification", "standard")):
        return "standard/specification-like"
    if any(token in name for token in ("作业", "试卷", "练习", "worksheet", "exam")):
        return "exam/worksheet-like"
    if page_count >= 100:
        return "long-form/book-like"
    return "handout/notes-like"


def classify_source_profile(
    *, page_count: int, text_page_count: int, image_page_count: int
) -> str:
    if page_count <= 0:
        raise ValueError("SOURCE_PAGE_COUNT_REQUIRED")
    text_ratio = text_page_count / page_count
    image_ratio = image_page_count / page_count
    if text_ratio <= 0.1 and image_ratio >= 0.5:
        return "SCAN_HEAVY"
    if text_ratio >= 0.8:
        return "NATIVE_TEXT"
    return "MIXED_NATIVE_VISUAL"


def inspect_source_pdf(
    path: str | Path, *, input_root: str | Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import fitz

    source = Path(path).resolve()
    root = Path(input_root).resolve()
    try:
        relative_path = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"SOURCE_OUTSIDE_INPUT_ROOT:{source}") from exc
    source_sha256 = sha256_file(source)
    page_rows: list[dict[str, Any]] = []
    with fitz.open(source) as document:
        if document.needs_pass:
            raise ValueError(f"ENCRYPTED_SOURCE_UNSUPPORTED:{relative_path}")
        page_count = int(document.page_count)
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()
            text_chars = len(text)
            image_count = len(page.get_images(full=True))
            drawing_count = len(page.get_drawings())
            blocks = [
                block
                for block in page.get_text("blocks")
                if len(str(block[4]).strip()) >= 20
            ]
            midpoint = float(page.rect.width) / 2.0
            left_blocks = sum(float(block[2]) <= midpoint * 1.1 for block in blocks)
            right_blocks = sum(float(block[0]) >= midpoint * 0.9 for block in blocks)
            multi_column_proxy = left_blocks >= 2 and right_blocks >= 2
            math_symbol_count = sum(
                character in "=<>±×÷√∑∫∆ΔθλμωΩπ^"
                for character in text
            ) + max(0, sum(character.isdigit() for character in text) - 10) // 5
            strata = _page_strata(
                text_chars=text_chars,
                image_count=image_count,
                drawing_count=drawing_count,
                math_symbol_count=math_symbol_count,
                multi_column_proxy=multi_column_proxy,
            )
            page_rows.append(
                {
                    "page_index": page_index,
                    "text_chars": text_chars,
                    "image_count": image_count,
                    "drawing_count": drawing_count,
                    "math_symbol_count": math_symbol_count,
                    "multi_column_proxy": multi_column_proxy,
                    "width_pt": round(float(page.rect.width), 6),
                    "height_pt": round(float(page.rect.height), 6),
                    "source_strata": strata,
                }
            )
    if page_count == 0:
        raise ValueError(f"EMPTY_SOURCE_PDF:{relative_path}")
    text_page_count = sum(row["text_chars"] > 20 for row in page_rows)
    image_page_count = sum(row["image_count"] > 0 for row in page_rows)
    source_profile = classify_source_profile(
        page_count=page_count,
        text_page_count=text_page_count,
        image_page_count=image_page_count,
    )
    stem = re.sub(r"[^\w.-]+", "_", source.stem, flags=re.UNICODE).strip("_.")
    document_id = f"{stem or 'document'}__{source_sha256[:12]}"
    document_row = {
        "document_id": document_id,
        "relative_path": relative_path,
        "bytes": source.stat().st_size,
        "sha256": source_sha256,
        "page_count": page_count,
        "source_profile": source_profile,
        "document_role": classify_document_role(relative_path, page_count),
        "source_metadata": {
            "text_page_count": text_page_count,
            "image_page_count": image_page_count,
            "drawing_proxy_page_count": sum(
                row["drawing_count"] >= 8 for row in page_rows
            ),
            "math_symbol_proxy_page_count": sum(
                "math-symbol-heavy-proxy" in row["source_strata"]
                for row in page_rows
            ),
            "multi_column_proxy_page_count": sum(
                row["multi_column_proxy"] for row in page_rows
            ),
        },
    }
    pages = [
        {
            **row,
            "page_id": f"{document_id}:{row['page_index']}",
            "document_id": document_id,
            "source_pdf_sha256": source_sha256,
            "source_relative_path": relative_path,
        }
        for row in page_rows
    ]
    return document_row, pages


def _page_strata(
    *,
    text_chars: int,
    image_count: int,
    drawing_count: int,
    math_symbol_count: int,
    multi_column_proxy: bool,
) -> list[str]:
    strata = []
    if text_chars > 80:
        strata.append("native-text")
    if text_chars <= 20 and image_count > 0:
        strata.append("scan-heavy")
    if math_symbol_count >= 8:
        strata.append("math-symbol-heavy-proxy")
    if image_count > 0:
        strata.append("image-heavy")
    if drawing_count >= 8:
        strata.append("table/grid-proxy")
    if multi_column_proxy:
        strata.append("multi-column-proxy")
    if text_chars > 20 and image_count > 0:
        strata.append("mixed")
    if text_chars > 200 and image_count == 0 and drawing_count < 5:
        strata.append("plain-text")
    return [name for name in SOURCE_STRATA if name in strata] or ["source-other"]


def build_corpus_manifest(
    documents: Sequence[Mapping[str, Any]], *, input_root_contract: str
) -> dict[str, Any]:
    normalized = sorted(
        (dict(document) for document in documents),
        key=lambda row: (str(row["relative_path"]), str(row["sha256"])),
    )
    hashes = [str(row["sha256"]) for row in normalized]
    if len(hashes) != len(set(hashes)):
        raise ValueError("DUPLICATE_SOURCE_SHA256")
    core = {
        "schema": CORPUS_MANIFEST_SCHEMA,
        "documents": normalized,
    }
    return {
        **core,
        "input_root_contract": input_root_contract,
        "document_count": len(normalized),
        "page_count": sum(int(row["page_count"]) for row in normalized),
        "corpus_fingerprint": semantic_sha256(core),
        "document_list_immutable": True,
    }


def validate_freshness(
    documents: Sequence[Mapping[str, Any]], known_hashes: Iterable[str]
) -> dict[str, Any]:
    registered = {str(value).lower() for value in known_hashes}
    overlaps = [
        {
            "document_id": str(row["document_id"]),
            "sha256": str(row["sha256"]),
        }
        for row in documents
        if str(row["sha256"]).lower() in registered
    ]
    return {
        "schema": "bemarkdown-corpus-v3-freshness-audit-v1",
        "registered_hash_count": len(registered),
        "candidate_document_count": len(documents),
        "exact_sha_overlap_count": len(overlaps),
        "overlaps": overlaps,
        "gate": "PASS" if not overlaps else "FAIL",
    }


def validate_formal_scope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    documents = list(manifest["documents"])
    profiles = {str(row["source_profile"]) for row in documents}
    roles = {str(row["document_role"]) for row in documents}
    checks = {
        "documents_at_least_8": int(manifest["document_count"]) >= 8,
        "pages_at_least_100": int(manifest["page_count"]) >= 100,
        "native_present": "NATIVE_TEXT" in profiles
        or "MIXED_NATIVE_VISUAL" in profiles,
        "scanned_present": "SCAN_HEAVY" in profiles,
        "at_least_3_document_roles": len(roles) >= 3,
    }
    return {
        "schema": "bemarkdown-corpus-v3-formal-scope-audit-v1",
        "checks": checks,
        "source_profiles": sorted(profiles),
        "document_roles": sorted(roles),
        "scope_limited": len(roles) < 3,
        "gate": "PASS" if all(checks.values()) else "FAIL",
    }


def select_quality_sample(
    page_features: Sequence[Mapping[str, Any]], *, sample_size: int, seed: int
) -> dict[str, Any]:
    pages = sorted((dict(row) for row in page_features), key=lambda row: row["page_id"])
    if len(pages) < sample_size:
        raise ValueError("QUALITY_SAMPLE_POPULATION_TOO_SMALL")
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pages:
        by_document[str(row["document_id"])].append(row)
    if len(by_document) < 8:
        raise ValueError("QUALITY_SAMPLE_REQUIRES_8_DOCUMENTS")
    document_order = sorted(
        by_document,
        key=lambda document_id: _stable_tiebreak(seed, f"document:{document_id}"),
    )
    quotas = Counter({document_id: 0 for document_id in document_order})
    remaining = sample_size
    while remaining:
        progressed = False
        for document_id in document_order:
            if remaining == 0:
                break
            if quotas[document_id] < len(by_document[document_id]):
                quotas[document_id] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("QUALITY_SAMPLE_ALLOCATION_FAILED")
    selected = []
    for document_id in document_order:
        candidates = sorted(
            by_document[document_id],
            key=lambda row: _stable_tiebreak(seed, str(row["page_id"])),
        )
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[str] = set()
        for stratum in SOURCE_STRATA:
            candidate = next(
                (
                    row
                    for row in candidates
                    if stratum in row.get("source_strata", [])
                    and str(row["page_id"]) not in chosen_ids
                ),
                None,
            )
            if candidate is not None and len(chosen) < quotas[document_id]:
                chosen.append(candidate)
                chosen_ids.add(str(candidate["page_id"]))
        for candidate in candidates:
            if len(chosen) >= quotas[document_id]:
                break
            if str(candidate["page_id"]) not in chosen_ids:
                chosen.append(candidate)
                chosen_ids.add(str(candidate["page_id"]))
        selected.extend(chosen)
    selected = sorted(
        selected,
        key=lambda row: (str(row["document_id"]), int(row["page_index"])),
    )
    stratum_counts = Counter(
        stratum for row in selected for stratum in row.get("source_strata", [])
    )
    unavailable = [name for name in SOURCE_STRATA if stratum_counts[name] == 0]
    core = {
        "schema": QUALITY_SAMPLE_SCHEMA,
        "selection_basis": "SOURCE_ONLY",
        "sample_size": sample_size,
        "sampling_seed": seed,
        "sampling_strategy": "BALANCED_DOCUMENT_SPREAD_THEN_SOURCE_STRATA_STABLE_HASH",
        "pages": selected,
        "document_page_counts": dict(
            sorted(Counter(row["document_id"] for row in selected).items())
        ),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "unavailable_strata": unavailable,
    }
    return {**core, "quality_sample_sha256": semantic_sha256(core)}


def _stable_tiebreak(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def build_truth_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": TRUTH_SCHEMA_VERSION,
        "title": "Corpus v3 source-only page truth",
        "type": "object",
        "required": [
            "schema",
            "page_id",
            "source_only_attestation",
            "visible_content_preservation",
            "critical_items",
        ],
        "properties": {
            "schema": {"const": TRUTH_SCHEMA_VERSION},
            "page_id": {"type": "string", "minLength": 1},
            "source_only_attestation": {"const": True},
            "reviewer_id": {"type": "string", "minLength": 1},
            "visible_content_preservation": {
                "type": "array",
                "items": {"type": "string"},
            },
            "text_transcription": {"type": "array", "items": {"type": "object"}},
            "formula_transcription": {"type": "array", "items": {"type": "object"}},
            "tables": {"type": "array", "items": {"type": "object"}},
            "images": {"type": "array", "items": {"type": "object"}},
            "caption_associations": {"type": "array", "items": {"type": "object"}},
            "reading_order": {"type": "array", "items": {"type": "object"}},
            "heading_structure": {"type": "array", "items": {"type": "object"}},
            "missing_content_checks": {"type": "array", "items": {"type": "object"}},
            "critical_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["critical_id", "kind", "source_description"],
                    "properties": {
                        "critical_id": {"type": "string"},
                        "kind": {
                            "enum": [
                                "FORMULA",
                                "TABLE",
                                "IMAGE_OR_DIAGRAM",
                                "HEADING_STRUCTURE",
                                "READING_ORDER",
                                "OTHER_MATERIAL_UNIT",
                            ]
                        },
                        "source_description": {"type": "string"},
                    },
                },
            },
            "reference_uncertain": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "additionalProperties": False,
    }


def canonicalize_source_truth_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_page_ids: set[str]
) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    observed = [str(row.get("page_id") or "") for row in normalized]
    if len(observed) != len(set(observed)):
        raise ValueError("DUPLICATE_SOURCE_TRUTH_PAGE_ID")
    if set(observed) != expected_page_ids:
        raise ValueError("SOURCE_TRUTH_PAGE_COVERAGE_MISMATCH")
    for row in normalized:
        _validate_source_truth_row(row)
    return sorted(normalized, key=lambda row: str(row["page_id"]))


def _validate_source_truth_row(row: Mapping[str, Any]) -> None:
    schema = build_truth_schema()
    allowed = set(schema["properties"])
    required = set(schema["required"])
    additional = set(row).difference(allowed)
    if additional:
        raise ValueError(f"SOURCE_TRUTH_ADDITIONAL_PROPERTY:{sorted(additional)}")
    missing = required.difference(row)
    if missing:
        raise ValueError(f"SOURCE_TRUTH_REQUIRED_FIELD_MISSING:{sorted(missing)}")
    if row.get("schema") != TRUTH_SCHEMA_VERSION:
        raise ValueError("SOURCE_TRUTH_SCHEMA_MISMATCH")
    if not isinstance(row.get("page_id"), str) or not row["page_id"]:
        raise ValueError("SOURCE_TRUTH_PAGE_ID_INVALID")
    if row.get("source_only_attestation") is not True:
        raise ValueError("SOURCE_ONLY_ATTESTATION_REQUIRED")

    array_fields = (
        "visible_content_preservation",
        "text_transcription",
        "formula_transcription",
        "tables",
        "images",
        "caption_associations",
        "reading_order",
        "heading_structure",
        "missing_content_checks",
        "critical_items",
    )
    for name in array_fields:
        if name in row and not isinstance(row[name], list):
            raise ValueError(f"SOURCE_TRUTH_FIELD_TYPE:{name}:array")
    for value in row.get("visible_content_preservation", []):
        if not isinstance(value, str):
            raise TypeError("SOURCE_TRUTH_FIELD_TYPE:visible_content_preservation:string")
    for name in array_fields[1:]:
        for value in row.get(name, []):
            if not isinstance(value, Mapping):
                raise TypeError(f"SOURCE_TRUTH_FIELD_TYPE:{name}:object")

    if "reviewer_id" in row and (
        not isinstance(row["reviewer_id"], str) or not row["reviewer_id"]
    ):
        raise ValueError("SOURCE_TRUTH_FIELD_TYPE:reviewer_id:string")
    if "reference_uncertain" in row and not isinstance(
        row["reference_uncertain"], bool
    ):
        raise ValueError("SOURCE_TRUTH_FIELD_TYPE:reference_uncertain:boolean")
    if "notes" in row and not isinstance(row["notes"], str):
        raise ValueError("SOURCE_TRUTH_FIELD_TYPE:notes:string")

    critical_kinds = set(
        schema["properties"]["critical_items"]["items"]["properties"]["kind"][
            "enum"
        ]
    )
    for item in row.get("critical_items", []):
        for name in ("critical_id", "kind", "source_description"):
            if name not in item:
                raise ValueError(
                    f"SOURCE_TRUTH_CRITICAL_ITEM_REQUIRED_FIELD:{name}"
                )
        if not isinstance(item["critical_id"], str) or not item["critical_id"]:
            raise ValueError("SOURCE_TRUTH_CRITICAL_ID_INVALID")
        if item["kind"] not in critical_kinds:
            raise ValueError("SOURCE_TRUTH_CRITICAL_KIND_INVALID")
        if not isinstance(item["source_description"], str) or not item[
            "source_description"
        ]:
            raise ValueError("SOURCE_TRUTH_CRITICAL_DESCRIPTION_INVALID")


def build_truth_fingerprint(
    rows: Sequence[Mapping[str, Any]],
    *,
    corpus_fingerprint: str,
    eval_instance_sha256: str,
    quality_sample_sha256: str,
    truth_schema_sha256: str,
    truth_frozen_file_sha256: str,
    source_shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered_rows = sorted(
        (dict(row) for row in rows), key=lambda row: str(row["page_id"])
    )
    shard_manifest = sorted(
        (
            {
                "name": str(row["name"]),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for row in source_shards
        ),
        key=lambda row: row["name"],
    )
    page_ids = [str(row["page_id"]) for row in ordered_rows]
    core = {
        "schema": "bemarkdown-corpus-v3-truth-fingerprint-v1",
        "status": "CORPUS_V3_TRUTH_FROZEN",
        "corpus_fingerprint": corpus_fingerprint,
        "eval_instance_sha256": eval_instance_sha256,
        "quality_sample_sha256": quality_sample_sha256,
        "truth_schema_sha256": truth_schema_sha256,
        "truth_row_count": len(ordered_rows),
        "truth_unique_page_count": len(set(page_ids)),
        "truth_page_membership_sha256": semantic_sha256(sorted(page_ids)),
        "truth_semantic_sha256": semantic_sha256(ordered_rows),
        "truth_frozen_file_sha256": truth_frozen_file_sha256,
        "source_shards": shard_manifest,
        "source_only_attestation_complete": all(
            row.get("source_only_attestation") is True for row in ordered_rows
        ),
        "frozen_before_model_output": True,
        "model_output_started": False,
        "truth_content_committed": False,
    }
    return {**core, "truth_fingerprint_sha256": semantic_sha256(core)}


def build_eval_instance(
    *,
    corpus_fingerprint: str,
    quality_sample_sha256: str,
    truth_schema_sha256: str,
    sampling_seed: int,
) -> dict[str, Any]:
    thresholds = {
        "machine_invalid_applied_patch": {"maximum": 0, "hard": True},
        "material_false_correction": {"maximum": 0, "hard": True},
        "silent_material_error": {"maximum": 0, "hard": True},
        "critical_content_preservation": {"minimum": 1.0, "hard": True},
        "end_to_end_handoff_completeness": {"minimum": 1.0, "hard": True},
        "source_document_loss": {"maximum": 0, "hard": True},
        "silent_content_drop": {"maximum": 0, "hard": True},
        "end_to_end_page_acceptable_rate": {"minimum": 0.98, "hard": True},
        "non_material_false_correction_rate": {
            "maximum": 0.02,
            "hard_when_denominator_at_least": 20,
        },
        "true_correction_rate": {
            "minimum": 0.85,
            "hard_when_denominator_at_least": 20,
        },
        "unresolved_page_rate": {"maximum": 0.05, "hard": True},
        "oom": {"maximum": 0, "hard": True},
        "silent_cpu_fallback": {"maximum": 0, "hard": True},
        "crash": {"maximum": 0, "hard": True},
        "checkpoint_corruption": {"maximum": 0, "hard": True},
        "local_throughput_pages_per_second": {"minimum": 0.5, "hard": True},
    }
    core = {
        "schema": EVAL_INSTANCE_SCHEMA,
        "corpus_fingerprint": corpus_fingerprint,
        "quality_sample_sha256": quality_sample_sha256,
        "quality_sample_size": QUALITY_SAMPLE_SIZE,
        "sampling_seed": sampling_seed,
        "sampling_basis": "SOURCE_ONLY",
        "truth_mode": "INDEPENDENT_SOURCE_ONLY_CAPTURE",
        "truth_schema": TRUTH_SCHEMA_VERSION,
        "truth_schema_sha256": truth_schema_sha256,
        "metric_definitions": {
            "acceptable": "No material conversion error; minor non-semantic variation allowed.",
            "material_false_correction": "Faithful before-state changed into a material error.",
            "silent_material_error": "Material final error without unresolved, review, or fallback signal.",
            "reference_uncertain": "Excluded from correctness denominator and always counted.",
        },
        "denominators": {
            "page_quality": "ACCEPTABLE + MATERIAL_ERROR",
            "all_auto_patches": "Every auto-applied patch on the frozen 50-page sample",
            "truth_confirmed_errors": "All truth-confirmed pre-audit conversion errors",
        },
        "thresholds": thresholds,
        "provider_requirements": {
            "production_contract": "Output Audit Production v1",
            "real_provider_required": True,
            "vision_required": True,
            "fake_stub_allowed": False,
            "one_content_agent_call_per_batch": True,
            "content_retry": 0,
            "technical_retry_maximum": 1,
        },
        "audit_batching": {
            "minimum_pages": 4,
            "maximum_pages": 6,
            "hard_page_ceiling": 6,
            "self_check_required": "SOURCE_MATCH_CONFIRMED",
            "must_resolve": False,
        },
        "runtime_contract": {
            "profile": "portable_fp32",
            "layout_authority": "pdf-layout-authority-v1",
            "layout_render": {
                "dpi": 200,
                "color_space": "RGB",
                "encoding": "JPEG",
                "jpeg_quality": 92,
                "jpeg_optimize": True,
                "layout_batch": 1,
            },
            "independent_cold_cache_namespace_required": True,
        },
        "model_output_allowed_before_truth_frozen": False,
        "frozen_before_model_output": True,
    }
    return {**core, "eval_instance_sha256": semantic_sha256(core)}


def partition_truth_shards(
    pages: Sequence[Mapping[str, Any]], *, max_pages: int
) -> list[dict[str, Any]]:
    if max_pages <= 0 or max_pages > MAX_TRUTH_SHARD_PAGES:
        raise ValueError("TRUTH_SHARD_PAGE_LIMIT_INVALID")
    ordered = sorted(
        (dict(row) for row in pages),
        key=lambda row: (str(row["document_id"]), int(row["page_index"])),
    )
    return [
        {
            "schema": "bemarkdown-corpus-v3-truth-capture-shard-v1",
            "shard_id": f"shard-{index // max_pages + 1:03d}",
            "pages": ordered[index : index + max_pages],
        }
        for index in range(0, len(ordered), max_pages)
    ]


def validate_truth_capture_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_page_ids: set[str]
) -> None:
    observed = [str(row.get("page_id") or "") for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("DUPLICATE_TRUTH_CAPTURE_PAGE_ID")
    if set(observed) != expected_page_ids:
        raise ValueError("TRUTH_CAPTURE_PAGE_COVERAGE_MISMATCH")
    for row in rows:
        unexpected = set(row).difference(TRUTH_CAPTURE_ALLOWED_FIELDS)
        if unexpected:
            raise ValueError(
                f"MODEL_OUTPUT_FIELD_IN_TRUTH_CAPTURE:{sorted(unexpected)}"
            )


def scan_audit_bundle_for_truth_leakage(root: str | Path) -> dict[str, Any]:
    bundle = Path(root)
    findings = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle).as_posix()
        relative_lower = relative.casefold()
        for token in TRUTH_LEAKAGE_TOKENS:
            if token in relative_lower:
                findings.append(
                    {"location": relative, "reason": "FORBIDDEN_TOKEN_IN_PATH", "token": token}
                )
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").casefold()
        for token in TRUTH_LEAKAGE_TOKENS:
            if token in text:
                findings.append(
                    {"location": relative, "reason": "FORBIDDEN_TOKEN_IN_CONTENT", "token": token}
                )
    return {
        "schema": "bemarkdown-corpus-v3-truth-leakage-scan-v1",
        "bundle": str(bundle),
        "leakage_count": len(findings),
        "findings": findings,
        "gate": "PASS" if not findings else "FAIL",
    }


def render_source_page_png(
    source_pdf: str | Path,
    *,
    page_index: int,
    target: str | Path,
    dpi: int = 200,
) -> dict[str, Any]:
    import fitz
    from PIL import Image

    source = Path(source_pdf)
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source) as document:
        page = document[page_index]
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        payload = pixmap.tobytes("png")
    destination.write_bytes(payload)
    with Image.open(destination) as image:
        image.verify()
    with Image.open(destination) as image:
        verified_width, verified_height = image.size
    if (verified_width, verified_height) != (pixmap.width, pixmap.height):
        raise RuntimeError(f"SOURCE_PAGE_RENDER_DIMENSION_MISMATCH:{destination}")
    return {
        "path": str(destination),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "width": pixmap.width,
        "height": pixmap.height,
        "dpi": dpi,
        "color_space": "RGB",
        "alpha": False,
    }
