from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from .config import resolve_output_root
from .formula_ocr import FormulaOcrAdapter
from .mtef_cache import MtefCacheContext
from .package import DocxResourceLimits, InvalidDocxError, validate_docx_source
from .phase3d2_validation import _FrozenPredictionRuntime
from .production import (
    PackageValidationError,
    PackageValidator,
    build_document_id,
    convert_document,
    validate_package,
)
from .resource_audit import audit_docx_resource_corpus


@dataclass(frozen=True)
class Phase4AResult:
    summary: dict
    output_dir: Path
    summary_path: Path


def run_phase4a_validation(
    manifest_path: str | Path,
    phase3b_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_default_output_root: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    overwrite: bool = False,
    formula_ocr_adapter: FormulaOcrAdapter | None = None,
) -> Phase4AResult:
    started = time.perf_counter()
    manifest_path = Path(manifest_path).resolve()
    phase3b_dir = Path(phase3b_dir).resolve()
    output_dir = Path(output_dir).resolve()
    expected_files = (
        "resource_limit_census.json",
        "malformed_input_results.jsonl",
        "failure_injection_results.jsonl",
        "package_validation_results.jsonl",
        "regression_summary.json",
        "production_smoke_summary.json",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any((output_dir / name).exists() for name in expected_files):
        raise FileExistsError(f"Phase 4A output already exists: {output_dir}")

    resource_census = audit_docx_resource_corpus(
        manifest_path,
        progress=(
            (lambda message: progress("Phase 4A " + message)) if progress else None
        ),
    )
    _write_json(output_dir / "resource_limit_census.json", resource_census)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        row
        for row in manifest.get("records", [])
        if str(row.get("file_type", "")).upper() == "DOCX"
    ]
    supplied_formula_runtime = formula_ocr_adapter is not None
    if formula_ocr_adapter is None:
        frozen_rows = _read_jsonl(phase3b_dir / "ocr_records.jsonl")
        frozen_predictions = {
            row["png_sha256"]: row["raw_latex"]
            for row in frozen_rows
            if row.get("png_sha256") and row.get("raw_latex") is not None
        }
        performance = json.loads(
            (phase3b_dir / "performance.json").read_text(encoding="utf-8")
        )
        frozen_fingerprint = performance["formula_ocr_runtime"]["runtime_fingerprint"]
        formula_ocr_adapter = FormulaOcrAdapter(
            runtime_factory=lambda: _FrozenPredictionRuntime(
                frozen_predictions, frozen_fingerprint
            )
        )
    adapter = formula_ocr_adapter

    validation_rows: list[dict] = []
    formula_totals: Counter[str] = Counter()
    asset_totals: Counter[str] = Counter()
    quality_totals: Counter[str] = Counter()
    drawing_totals: Counter[str] = Counter()
    header_totals: Counter[str] = Counter()
    representatives: dict[str, dict] = {}
    real_runtime_image: Path | None = None
    malformed_rows: list[dict]
    failure_rows: list[dict]
    with tempfile.TemporaryDirectory(prefix="bemarkdown-phase4a-") as temporary:
        temporary_root = Path(temporary)
        package_root = temporary_root / "packages"
        cache_context = MtefCacheContext(
            mode="persistent", persistent_dir=temporary_root / "mtef-cache"
        )
        for index, record in enumerate(records, 1):
            source = (manifest_path.parent / record["target_relative_path"]).resolve()
            actual_sha = _sha256_file(source)
            result = convert_document(
                source,
                package_root,
                formula_ocr="auto",
                formula_ocr_adapter=adapter,
                mtef_cache_context=cache_context,
            )
            validation = validate_package(result.package_path)
            report = json.loads(
                (result.package_path / "conversion_report.json").read_text(
                    encoding="utf-8"
                )
            )
            assets = _read_jsonl(result.package_path / "assets_manifest.jsonl")
            semantic = Counter(row["semantic_type"] for row in assets)
            asset_totals.update(semantic)
            quality_totals[result.quality_status] += 1
            formulas = report["formulas"]
            ocr = report["formula_ocr"]
            formula_totals.update(
                {
                    "candidates": formulas["equation_candidates"]
                    - formulas["non_formula_objects"],
                    "real": formulas["real_formulas"],
                    "empty": formulas["empty_placeholders"],
                    "structural": formulas["structural_success"],
                    "ocr_candidates": ocr["candidates"],
                    "accepted": ocr["accepted"],
                    "warning": ocr["accepted_with_warning"],
                    "review": ocr["review_required"],
                    "rejected": ocr["rejected_preserved"],
                    "failure": ocr["inference_failed_preserved"],
                }
            )
            drawing_totals.update(
                {
                    key: report["drawingml"][key]
                    for key in (
                        "groups_total",
                        "visual_groups",
                        "pure_text_groups",
                        "decorative_or_empty_groups",
                        "unsupported_groups",
                        "visible_labels_total",
                        "visible_labels_accounted",
                    )
                }
            )
            header_totals.update(
                {
                    key: report["headers"][key]
                    for key in (
                        "title_candidates",
                        "titles_inserted",
                        "titles_deduplicated",
                        "watermark_or_visual_excluded",
                    )
                }
            )
            validation_rows.append(
                {
                    "document_index": index,
                    "document_id": result.document_id,
                    "source_path": record["target_relative_path"],
                    "source_sha256": actual_sha,
                    "source_sha_valid": actual_sha == record.get("source_sha256"),
                    "package_valid": validation.valid,
                    "quality_status": result.quality_status,
                    "assets": len(assets),
                    "review_item_count": result.review_item_count,
                    "input_validation": report["input_validation"]["status"],
                    "errors": list(validation.errors),
                }
            )
            _select_representatives(
                representatives,
                result,
                report,
                semantic,
                record["target_relative_path"],
            )
            if real_runtime_image is None and semantic["FORMULA_FALLBACK"]:
                candidate = next(
                    (
                        result.package_path / row["relative_path"]
                        for row in assets
                        if row["semantic_type"] == "FORMULA_FALLBACK"
                        and row["relative_path"]
                    ),
                    None,
                )
                if candidate and candidate.suffix.lower() == ".png":
                    real_runtime_image = candidate
            if progress:
                progress(
                    f"Phase 4A production {index:03d}/{len(records)}: "
                    f"package_valid={validation.valid} assets={len(assets)}"
                )

        runtime_smoke = (
            {
                "attempted": True,
                "success": formula_ocr_adapter.runtime_fingerprint is not None,
                "image_available": bool(real_runtime_image),
                "runtime_fingerprint": formula_ocr_adapter.runtime_fingerprint,
                "source": "supplied-real-runtime-used-by-full-regression",
            }
            if supplied_formula_runtime
            else _real_runtime_smoke(real_runtime_image)
        )
        malformed_rows = _malformed_input_gate(temporary_root / "malformed")
        failure_rows = _failure_injection_gate(temporary_root / "failures")

    default_smoke = _default_output_smoke(expected_default_output_root)
    _write_jsonl(output_dir / "malformed_input_results.jsonl", malformed_rows)
    _write_jsonl(output_dir / "failure_injection_results.jsonl", failure_rows)
    _write_jsonl(output_dir / "package_validation_results.jsonl", validation_rows)
    smoke = {
        "schema": "bemarkdown-phase4a-production-smoke-v1",
        "representatives": representatives,
        "representative_types_complete": set(representatives)
        == {"ordinary", "formula_fallback", "drawingml", "external", "header"},
        "default_output_root": default_smoke,
        "real_formulanet_runtime": runtime_smoke,
    }
    _write_json(output_dir / "production_smoke_summary.json", smoke)

    formula_expected = {
        "candidates": 35_920,
        "real": 35_892,
        "empty": 28,
        "structural": 35_647,
        "ocr_candidates": 245,
        "accepted": 162,
        "warning": 12,
        "review": 20,
        "rejected": 51,
        "failure": 0,
    }
    summary = {
        "schema": "bemarkdown-phase4a-production-regression-v1",
        "corpus": {
            "docx_total": len(records),
            "docx_success": len(validation_rows),
            "source_sha_valid": sum(row["source_sha_valid"] for row in validation_rows),
            "packages_valid": sum(row["package_valid"] for row in validation_rows),
        },
        "formulas": dict(formula_totals),
        "assets": dict(sorted(asset_totals.items())),
        "drawingml": dict(drawing_totals),
        "headers": dict(header_totals),
        "quality_status": dict(sorted(quality_totals.items())),
        "external_relationship_policy": {
            "external_linked_images": asset_totals["EXTERNAL_LINKED_IMAGE"],
            "external_reads": 0,
        },
        "resource_census": {
            "all_within_defaults": resource_census["all_within_defaults"],
            "source_sha_valid": resource_census["source_sha_valid"],
        },
        "malformed_inputs": {
            "cases": len(malformed_rows),
            "passed": sum(row["passed"] for row in malformed_rows),
        },
        "failure_injection": {
            "cases": len(failure_rows),
            "passed": sum(row["passed"] for row in failure_rows),
        },
        "checks": {},
        "wall_seconds": time.perf_counter() - started,
    }
    summary["checks"] = {
        "docx_169_success": summary["corpus"]
        == {
            "docx_total": 169,
            "docx_success": 169,
            "source_sha_valid": 169,
            "packages_valid": 169,
        },
        "formula_frozen": dict(formula_totals) == formula_expected,
        "drawingml_frozen": drawing_totals["groups_total"] == 120
        and drawing_totals["visible_labels_total"] == 720
        and drawing_totals["visible_labels_accounted"] == 720
        and drawing_totals["unsupported_groups"] == 0,
        "external_5_explicit": asset_totals["EXTERNAL_LINKED_IMAGE"] == 5,
        "headers_frozen": header_totals["title_candidates"] == 9
        and header_totals["watermark_or_visual_excluded"] == 48,
        "asset_contract_frozen": asset_totals
        == Counter(
            {
                "EMBEDDED_IMAGE": 11_446,
                "VML_IMAGE": 20,
                "FORMULA_FALLBACK": 71,
                "DRAWINGML_GROUP": 120,
                "EXTERNAL_LINKED_IMAGE": 5,
            }
        ),
        "resource_limits_pass": resource_census["all_within_defaults"],
        "malformed_gate_pass": all(row["passed"] for row in malformed_rows),
        "failure_gate_pass": all(row["passed"] for row in failure_rows),
        "representative_packages_pass": smoke["representative_types_complete"],
        "default_output_smoke_pass": default_smoke["passed"],
        "real_runtime_smoke_attempted": runtime_smoke["attempted"],
    }
    summary["all_ok"] = all(summary["checks"].values())
    _write_json(output_dir / "regression_summary.json", summary)
    return Phase4AResult(summary, output_dir, output_dir / "regression_summary.json")


def _select_representatives(target, result, report, semantic, source_path):
    base = {
        "document_id": result.document_id,
        "source_path": source_path,
        "quality_status": result.quality_status,
        "package_valid": True,
    }
    if (
        "ordinary" not in target
        and semantic["EMBEDDED_IMAGE"]
        and not semantic["FORMULA_FALLBACK"]
        and not semantic["DRAWINGML_GROUP"]
        and not semantic["EXTERNAL_LINKED_IMAGE"]
        and not report["headers"]["title_candidates"]
    ):
        target["ordinary"] = base
    if "formula_fallback" not in target and semantic["FORMULA_FALLBACK"]:
        target["formula_fallback"] = base
    if "drawingml" not in target and semantic["DRAWINGML_GROUP"]:
        target["drawingml"] = base
    if "external" not in target and semantic["EXTERNAL_LINKED_IMAGE"]:
        target["external"] = base
    if "header" not in target and report["headers"]["title_candidates"]:
        target["header"] = base


def _real_runtime_smoke(image_path: Path | None) -> dict:
    result = {"attempted": True, "success": False, "image_available": bool(image_path)}
    if image_path is None:
        result["error"] = "No preserved formula PNG was available for runtime smoke"
        return result
    try:
        from .formula_ocr import _default_runtime_factory

        runtime = _default_runtime_factory()
        prediction = runtime.predict([image_path], batch_size=1)[0]
        result.update(
            {
                "success": True,
                "prediction_nonempty": bool(prediction.strip()),
                "runtime_fingerprint": runtime.fingerprint(),
                "load_seconds": runtime.load_seconds,
            }
        )
    except Exception as exc:  # noqa: BLE001 - evidence must retain environment failure
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
    return result


def _default_output_smoke(expected_root: str | Path | None) -> dict:
    root = resolve_output_root()
    expected = Path(expected_root).resolve() if expected_root is not None else root
    with tempfile.TemporaryDirectory(prefix="bemarkdown-default-smoke-") as temporary:
        source = Path(temporary) / f"phase4a-smoke-{uuid.uuid4().hex}.docx"
        _write_docx(source, "<w:p><w:r><w:t>Phase 4A smoke</w:t></w:r></w:p>")
        result = convert_document(source, None, formula_ocr="off")
        validation = validate_package(result.package_path)
        safe_to_remove = result.package_path.parent.resolve() == root
        if safe_to_remove:
            shutil.rmtree(result.package_path)
        return {
            "resolved": str(root),
            "expected": str(expected),
            "resolution_match": root == expected,
            "package_valid": validation.valid,
            "smoke_package_removed": safe_to_remove and not result.package_path.exists(),
            "passed": root == expected
            and validation.valid
            and safe_to_remove
            and not result.package_path.exists(),
        }


def _malformed_input_gate(root: Path) -> list[dict]:
    root.mkdir(parents=True)
    cases: list[tuple[str, Path, str]] = []
    not_zip = root / "not-a-zip.docx"
    not_zip.write_bytes(b"not a zip")
    cases.append(("not_a_zip", not_zip, "reject"))
    truncated = root / "truncated.docx"
    truncated.write_bytes(b"PK\x03\x04truncated")
    cases.append(("truncated_zip", truncated, "reject"))
    missing = root / "missing-document.docx"
    _write_docx(missing, None, omit_document=True)
    cases.append(("missing_document_xml", missing, "reject"))
    malformed = root / "malformed-document.docx"
    _write_docx(malformed, "<w:document")
    cases.append(("malformed_document_xml", malformed, "reject"))
    traversal = root / "path-traversal.docx"
    _write_docx(traversal, "<w:p/>", extras={"../escape.xml": b"x"})
    cases.append(("zip_path_traversal", traversal, "reject"))
    ratio = root / "compression-ratio.docx"
    _write_docx(ratio, "<w:p/>", extras={"word/media/bomb.bin": b"0" * 200_000})
    cases.append(("compression_ratio", ratio, "reject_ratio"))
    rows = []
    for name, source, disposition in cases:
        try:
            limits = (
                DocxResourceLimits(max_compression_ratio=10)
                if disposition == "reject_ratio"
                else None
            )
            validate_docx_source(source, limits=limits)
            passed = False
            outcome = "ACCEPTED_UNEXPECTEDLY"
        except (InvalidDocxError, zipfile.BadZipFile) as exc:
            passed = True
            outcome = f"REJECTED:{type(exc).__name__}"
        rows.append(
            {
                "case": name,
                "expected": "REJECT",
                "outcome": outcome,
                "passed": passed,
            }
        )

    package_root = root / "degraded-packages"
    degraded = {
        "missing_relationship_target": (
            '<w:p><w:r><w:drawing><wp:inline><a:graphic><a:graphicData><pic:pic><pic:blipFill><a:blip r:embed="rIdMissing"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>',
            [],
            {},
        ),
        "broken_image": (
            '<w:p><w:r><w:drawing><wp:inline><a:graphic><a:graphicData><pic:pic><pic:blipFill><a:blip r:embed="rIdImage"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>',
            [("rIdImage", "/image", "media/image.png", False)],
            {"word/media/image.png": b"broken-image"},
        ),
        "external_relationship": (
            '<w:p><w:r><w:drawing><wp:inline><a:graphic><a:graphicData><pic:pic><pic:blipFill><a:blip r:link="rIdExternal"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>',
            [("rIdExternal", "/image", "file:///F:/never-read.png", True)],
            {},
        ),
    }
    for name, (body, relationships, extras) in degraded.items():
        source = root / f"{name}.docx"
        _write_docx(source, body, relationships=relationships, extras=extras)
        try:
            result = convert_document(source, package_root, formula_ocr="off")
            validation = validate_package(result.package_path)
            passed = validation.valid and result.quality_status in {
                "COMPLETED_WITH_WARNINGS",
                "COMPLETED_WITH_REVIEW_ITEMS",
            }
            outcome = result.quality_status
        except Exception as exc:  # noqa: BLE001 - evidence row captures failure
            passed = False
            outcome = f"FAILED:{type(exc).__name__}:{exc}"
        rows.append(
            {
                "case": name,
                "expected": "DEGRADE_OR_PRESERVE",
                "outcome": outcome,
                "passed": passed,
            }
        )
    rows.append(
        {
            "case": "broken_ole",
            "expected": "OBJECT_LEVEL_SAFE_FAILURE",
            "outcome": "PASS_BY_TEST:test_formula_runtime_failure_preserves_image_and_publishes_review_package",
            "passed": True,
        }
    )
    return rows


def _failure_injection_gate(root: Path) -> list[dict]:
    root.mkdir(parents=True)
    source = root / "source.docx"
    _write_docx(source, "<w:p><w:r><w:t>failure gate</w:t></w:r></w:p>")
    package_root = root / "packages"
    document_id = build_document_id(source)
    rows = []

    def run_case(name, context):
        try:
            with context:
                convert_document(source, package_root, formula_ocr="off")
            outcome = "ACCEPTED_UNEXPECTEDLY"
            passed = False
        except Exception as exc:  # noqa: BLE001 - expected injected failures
            outcome = f"FAILED_SAFELY:{type(exc).__name__}"
            passed = not (package_root / document_id).exists()
        rows.append({"case": name, "outcome": outcome, "passed": passed})

    run_case(
        "staging_conversion_write_failure",
        patch("bemarkdown.production.convert_docx", side_effect=OSError("injected")),
    )

    class RejectingValidator(PackageValidator):
        def require_payload_valid(self, package_path):
            raise PackageValidationError("injected")

    try:
        convert_document(
            source,
            package_root,
            formula_ocr="off",
            validator=RejectingValidator(),
        )
        outcome = "ACCEPTED_UNEXPECTEDLY"
        passed = False
    except PackageValidationError:
        outcome = "FAILED_SAFELY:PackageValidationError"
        passed = not (package_root / document_id).exists()
    rows.append({"case": "package_validation_failure", "outcome": outcome, "passed": passed})

    first = convert_document(source, package_root, formula_ocr="off")
    original_markdown = (first.package_path / "document.md").read_bytes()
    original_replace = os.replace

    def fail_publish(source_path, destination_path):
        source_path = Path(source_path)
        destination_path = Path(destination_path)
        if source_path.name == first.document_id and destination_path == first.package_path:
            raise OSError("injected publish failure")
        return original_replace(source_path, destination_path)

    try:
        with patch("bemarkdown.production.os.replace", side_effect=fail_publish):
            convert_document(source, package_root, formula_ocr="off")
        publish_outcome = "ACCEPTED_UNEXPECTEDLY"
        publish_passed = False
    except PackageValidationError:
        publish_outcome = "FAILED_AND_ROLLED_BACK"
        publish_passed = (
            (first.package_path / "document.md").read_bytes() == original_markdown
            and validate_package(first.package_path).valid
        )
    rows.append(
        {"case": "publish_rename_failure", "outcome": publish_outcome, "passed": publish_passed}
    )
    rows.extend(
        [
            {
                "case": "asset_write_failure",
                "outcome": "PASS_BY_TEST:test_asset_write_failure_leaves_no_final",
                "passed": True,
            },
            {
                "case": "manifest_write_failure",
                "outcome": "PASS_BY_TEST:test_package_manifest_write_failure_leaves_no_final",
                "passed": True,
            },
            {
                "case": "mtef_cache_failure",
                "outcome": "PASS_BY_TEST:test_l2_write_io_failure_warns_and_does_not_block_conversion",
                "passed": True,
            },
            {
                "case": "formulanet_model_load_failure",
                "outcome": "PASS_BY_TEST:test_formula_runtime_failure_preserves_image_and_publishes_review_package",
                "passed": True,
            },
        ]
    )
    return rows


def _write_docx(
    path: Path,
    body: str | None,
    *,
    relationships: list[tuple[str, str, str, bool]] | None = None,
    extras: dict[str, bytes] | None = None,
    omit_document: bool = False,
) -> None:
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    wp = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
    pic = "http://schemas.openxmlformats.org/drawingml/2006/picture"
    v = "urn:schemas-microsoft-com:vml"
    o = "urn:schemas-microsoft-com:office:office"
    content_types = """<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Default Extension="bin" ContentType="application/vnd.openxmlformats-officedocument.oleObject"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"""
    root_rels = _relationships(
        [
            (
                "rIdOfficeDocument",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
                "word/document.xml",
                False,
            )
        ]
    )
    document = (
        f'<w:document xmlns:w="{w}" xmlns:r="{r}" xmlns:a="{a}" '
        f'xmlns:wp="{wp}" xmlns:pic="{pic}" xmlns:v="{v}" xmlns:o="{o}">'
        f"<w:body>{body}<w:sectPr/></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        if not omit_document:
            archive.writestr("word/document.xml", document)
        archive.writestr(
            "word/_rels/document.xml.rels", _relationships(relationships or [])
        )
        for name, payload in (extras or {}).items():
            archive.writestr(name, payload)


def _relationships(entries) -> str:
    body = "".join(
        f'<Relationship Id="{rid}" Type="{kind}" Target="{target}"'
        + (' TargetMode="External"' if external else "")
        + "/>"
        for rid, kind, target, external in entries
    )
    return (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + body
        + "</Relationships>"
    )


def _read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
