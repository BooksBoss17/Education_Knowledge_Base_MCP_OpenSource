from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .config import resolve_output_root
from .file_handoff import replace_path
from .delivery_stage import artifact_delivery_stage

PACKAGE_CONTRACT = "bemarkdown-package-v1"
QUALITY_STATUSES = {
    "CLEAN",
    "COMPLETED_WITH_WARNINGS",
    "COMPLETED_WITH_REVIEW_ITEMS",
    "FAILED",
}
DOCUMENT_ID_SHA_PREFIX = 12
LOGGER = logging.getLogger("bemarkdown.production")


class ExistingPackageError(FileExistsError):
    pass


class PackageValidationError(RuntimeError):
    pass


class UnsupportedDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class PackageConversionResult:
    document_id: str
    package_path: Path
    source_sha256: str
    source_type: str
    quality_status: str
    success: bool
    warnings: tuple[str, ...]
    review_item_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["package_path"] = str(self.package_path.resolve())
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True)
class PackageValidationResult:
    package_path: Path
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    document_id: str | None = None
    quality_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_path": str(self.package_path.resolve()),
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "document_id": self.document_id,
            "quality_status": self.quality_status,
        }


@dataclass(frozen=True)
class StagingCleanupResult:
    output_root: Path
    removed: tuple[str, ...]
    retained: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root.resolve()),
            "removed": list(self.removed),
            "retained": list(self.retained),
        }


def build_document_id(source: str | Path, source_sha256: str | None = None) -> str:
    source = Path(source)
    source_sha256 = source_sha256 or _sha256_file(source)
    normalized = unicodedata.normalize("NFC", source.stem)
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", normalized)
    safe = re.sub(r"\s+", "_", safe).strip(" ._")
    if not safe:
        safe = "document"
    if safe.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        safe = "document_" + safe
    safe = safe[:80].rstrip(" ._") or "document"
    return f"{safe}__{source_sha256[:DOCUMENT_ID_SHA_PREFIX]}"


class PackageValidator:
    """Validate publish payloads and complete BeMarkdown Package v1 directories."""

    def validate_payload(self, package_path: str | Path) -> PackageValidationResult:
        package_path = Path(package_path)
        errors: list[str] = []
        warnings: list[str] = []
        required = ("document.md", "assets_manifest.jsonl", "conversion_report.json")
        for name in required:
            if not (package_path / name).is_file():
                errors.append(f"Required artifact is missing: {name}")
        if errors:
            return PackageValidationResult(package_path, False, tuple(errors))

        try:
            markdown = (package_path / "document.md").read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"document.md is unreadable UTF-8: {exc}")
            markdown = ""
        try:
            report = json.loads(
                (package_path / "conversion_report.json").read_text(encoding="utf-8")
            )
            if not isinstance(report, dict):
                raise TypeError("root must be an object")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"conversion_report.json is invalid: {exc}")
            report = {}
        rows: list[dict[str, Any]] = []
        try:
            for number, line in enumerate(
                (package_path / "assets_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines(),
                1,
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"line {number} is not an object")
                rows.append(row)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"assets_manifest.jsonl is invalid: {exc}")

        self._validate_assets(package_path, markdown, rows, errors)
        quality = report.get("quality_status")
        if quality not in QUALITY_STATUSES or quality == "FAILED":
            errors.append(f"Invalid publishable quality_status: {quality!r}")
        if report.get("errors"):
            errors.append("conversion_report.json contains conversion errors")
        formulas = report.get("formulas", {})
        if formulas.get("conservation_ok") is not True:
            errors.append("Formula conservation is not true")
        ocr = report.get("formula_ocr", {})
        if ocr.get("enabled") and (
            ocr.get("conservation_ok") is not True
            or ocr.get("real_formula_conservation_ok") is not True
        ):
            errors.append("Formula OCR conservation is not true")
        assets = report.get("assets", {})
        if assets.get("total_visible") != len(rows):
            errors.append("Asset total does not match assets_manifest.jsonl")
        if "markdown_render" in report:
            from .pdf.clean_markdown import validate_clean_render

            errors.extend(validate_clean_render(
                markdown, report["markdown_render"],
                expected_node_count=report.get("document", {}).get("blocks"),
            ))
        if report.get("warnings"):
            warnings.extend(str(value) for value in report["warnings"])
        return PackageValidationResult(
            package_path,
            not errors,
            tuple(errors),
            tuple(warnings),
            quality_status=quality if isinstance(quality, str) else None,
        )

    def require_payload_valid(self, package_path: str | Path) -> None:
        result = self.validate_payload(package_path)
        if not result.valid:
            raise PackageValidationError("; ".join(result.errors))

    def validate(self, package_path: str | Path) -> PackageValidationResult:
        package_path = Path(package_path)
        payload = self.validate_payload(package_path)
        errors = list(payload.errors)
        warnings = list(payload.warnings)
        manifest_path = package_path / "package_manifest.json"
        manifest: dict[str, Any] = {}
        if not manifest_path.is_file():
            errors.append("Required artifact is missing: package_manifest.json")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise TypeError("root must be an object")
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
                errors.append(f"package_manifest.json is invalid: {exc}")
                manifest = {}
        if manifest:
            if manifest.get("package_contract") != PACKAGE_CONTRACT:
                errors.append("Unsupported package_contract")
            if manifest.get("complete") is not True:
                errors.append("Package complete flag is not true")
            document_id = manifest.get("document_id")
            if document_id != package_path.name:
                errors.append("Package directory name does not match document_id")
            quality = manifest.get("quality_status")
            if quality != payload.quality_status:
                errors.append("Package and conversion quality_status disagree")
            expected_artifacts = {
                "markdown": "document.md",
                "assets_manifest": "assets_manifest.jsonl",
                "conversion_report": "conversion_report.json",
            }
            if isinstance(manifest.get('artifacts'), dict) and 'review_plan' in manifest['artifacts']:
                expected_artifacts['review_plan']='review_plan.json'
                if 'review_source_pdf' in manifest['artifacts']:
                    expected_artifacts['review_source_pdf']='_bemarkdown/review_source.pdf'
            if manifest.get("artifacts") != expected_artifacts:
                errors.append("Package artifact map is not canonical")
            integrity = manifest.get("integrity", {})
            integrity_keys = {
                "document_md_sha256": "document.md",
                "assets_manifest_sha256": "assets_manifest.jsonl",
                "conversion_report_sha256": "conversion_report.json",
            }
            if 'review_plan' in expected_artifacts:
                integrity_keys['review_plan_sha256']='review_plan.json'
                if not (package_path/'review_plan.json').is_file():
                    errors.append('Required artifact is missing: review_plan.json')
            if 'review_source_pdf' in expected_artifacts:
                integrity_keys['review_source_pdf_sha256']='_bemarkdown/review_source.pdf'
                if not (package_path/'_bemarkdown/review_source.pdf').is_file():
                    errors.append('Required artifact is missing: review source PDF')
            for key, name in integrity_keys.items():
                path = package_path / name
                if path.is_file() and integrity.get(key) != _sha256_file(path):
                    errors.append(f"{name} SHA-256 does not match package manifest")
            if 'delivery_stage' in manifest:
                try:
                    expected_stage = artifact_delivery_stage(
                        package_path, integrity_keys.values(),
                        source_sha256=(manifest.get('source') or {}).get('sha256'),
                        scope='CONVERTER_ARTIFACTS',
                    )
                    if manifest['delivery_stage'] != expected_stage:
                        errors.append('Delivery stage does not match converter artifact evidence')
                except (OSError, ValueError, AttributeError):
                    errors.append('Delivery stage artifacts unavailable')
        return PackageValidationResult(
            package_path,
            not errors,
            tuple(errors),
            tuple(warnings),
            manifest.get("document_id") if manifest else None,
            manifest.get("quality_status") if manifest else payload.quality_status,
        )

    def require_valid(self, package_path: str | Path) -> None:
        result = self.validate(package_path)
        if not result.valid:
            raise PackageValidationError("; ".join(result.errors))

    @staticmethod
    def _validate_assets(package_path, markdown, rows, errors) -> None:
        ids = [row.get("asset_id") for row in rows]
        expected = [f"image_{index:06d}" for index in range(1, len(rows) + 1)]
        if ids != expected:
            errors.append("Asset IDs are not continuous in reading order")
        uids = [row.get("asset_uid") for row in rows]
        if None in uids or len(uids) != len(set(uids)):
            errors.append("Asset UIDs are missing or duplicated")
        resolved_paths = []
        for row in rows:
            asset_id = row.get("asset_id")
            relative = row.get("relative_path")
            if relative is None:
                if f"[Unresolved image: {asset_id}]" not in markdown:
                    errors.append(f"Unresolved marker is missing for {asset_id}")
                if row.get("content_sha256") is not None:
                    errors.append(f"Unresolved asset {asset_id} has a content SHA")
                continue
            try:
                path = _safe_package_path(package_path, relative)
            except PackageValidationError as exc:
                errors.append(str(exc))
                continue
            resolved_paths.append(relative)
            if not path.is_file():
                errors.append(f"Resolved asset is missing: {relative}")
                continue
            if _sha256_file(path) != row.get("content_sha256"):
                errors.append(f"Resolved asset SHA mismatch: {relative}")
            if path.suffix.lower() != row.get("extension"):
                errors.append(f"Resolved asset extension mismatch: {relative}")
            guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if guessed != row.get("mime_type") and not (
                path.suffix.lower() == ".svg" and row.get("mime_type") == "image/svg+xml"
            ):
                errors.append(f"Resolved asset MIME mismatch: {relative}")
            if relative not in markdown:
                errors.append(f"Resolved asset is not referenced by Markdown: {relative}")
        # Check consumer links as well as manifest rows. Otherwise a serializer
        # can emit an undeclared temporary image that vanishes after publication.
        # The generated inline-image dialect permits escaped brackets in alt text.
        declared = set(resolved_paths)
        image_pattern = (
            r'(?<!\\)!\[(?:\\.|[^\]\\])*\]\(\s*'
            r'(<[^>\r\n]*>|[^\r\n]*?)(?:\s+"[^"\r\n]*")?\s*\)'
        )
        for match in re.finditer(image_pattern, markdown):
            reference = match.group(1).strip()
            if reference.startswith("<") and reference.endswith(">"):
                reference = reference[1:-1]
            if reference not in declared:
                errors.append(f"Markdown image is not a declared package asset: {reference}")
        asset_dir = package_path / "assets"
        actual = (
            sorted(f"assets/{path.name}" for path in asset_dir.iterdir() if path.is_file())
            if asset_dir.is_dir()
            else []
        )
        if Counter(actual) != Counter(resolved_paths):
            errors.append("Assets directory does not exactly match the asset manifest")


def validate_package(package_path: str | Path) -> PackageValidationResult:
    return PackageValidator().validate(package_path)


def cleanup_staging(
    output_root: str | Path | None = None,
    *,
    older_than_seconds: float = 24 * 3600,
    now: float | None = None,
) -> StagingCleanupResult:
    if older_than_seconds < 0:
        raise ValueError("older_than_seconds must be non-negative")
    root = resolve_output_root(output_root)
    staging = root / ".staging"
    if not staging.is_dir():
        return StagingCleanupResult(root, (), ())
    cutoff = (time.time() if now is None else now) - older_than_seconds
    removed: list[str] = []
    retained: list[str] = []
    for path in sorted(staging.iterdir(), key=lambda value: value.name):
        if not path.is_dir() or path.stat().st_mtime > cutoff:
            retained.append(path.name)
            continue
        resolved = path.resolve()
        if resolved.parent != staging.resolve():
            raise RuntimeError(f"Refusing to clean non-child staging path: {resolved}")
        shutil.rmtree(resolved)
        removed.append(path.name)
    return StagingCleanupResult(root, tuple(removed), tuple(retained))


def _package_manifest(
    document_id, source, source_sha, report, staging, *, source_type="DOCX"
):
    manifest = {
        "package_contract": PACKAGE_CONTRACT,
        "document_id": document_id,
        "source": {"type": source_type, "name": source.name, "sha256": source_sha},
        "converter": {"name": "bemarkdown", "version": _converter_version()},
        "quality_status": report["quality_status"],
        "artifacts": {
            "markdown": "document.md",
            "assets_manifest": "assets_manifest.jsonl",
            "conversion_report": "conversion_report.json",
        },
        "integrity": {
            "document_md_sha256": _sha256_file(staging / "document.md"),
            "assets_manifest_sha256": _sha256_file(
                staging / "assets_manifest.jsonl"
            ),
            "conversion_report_sha256": _sha256_file(
                staging / "conversion_report.json"
            ),
        },
        "complete": True,
        "delivery_stage": artifact_delivery_stage(
            staging, ('document.md', 'assets_manifest.jsonl', 'conversion_report.json'),
            source_sha256=source_sha, scope='CONVERTER_ARTIFACTS',
        ),
    }
    if (staging/'review_plan.json').is_file():
        manifest['artifacts']['review_plan']='review_plan.json'
        manifest['integrity']['review_plan_sha256']=_sha256_file(staging/'review_plan.json')
        if (staging/'_bemarkdown/review_source.pdf').is_file():
            manifest['artifacts']['review_source_pdf']='_bemarkdown/review_source.pdf'
            manifest['integrity']['review_source_pdf_sha256']=_sha256_file(staging/'_bemarkdown/review_source.pdf')
        manifest['delivery_stage']=artifact_delivery_stage(
            staging, manifest['artifacts'].values(), source_sha256=source_sha, scope='CONVERTER_ARTIFACTS')
    return manifest


def _publish_staged(staging: Path, final_path: Path, *, replace: bool) -> None:
    backup = staging.parent / f"{staging.name}.backup"
    if final_path.exists():
        if not replace:
            raise ExistingPackageError(f"Package already exists: {final_path}")
        try:
            replace_path(final_path, backup)
        except OSError as exc:
            raise PackageValidationError(
                f"Package publish could not stage the previous package: {exc}"
            ) from exc
        try:
            replace_path(staging, final_path)
        except OSError as exc:
            try:
                replace_path(backup, final_path)
            except OSError as rollback_exc:
                raise PackageValidationError(
                    f"Package publish failed and rollback failed: {exc}; {rollback_exc}"
                ) from rollback_exc
            raise PackageValidationError(
                f"Package publish failed; previous package was restored: {exc}"
            ) from exc
        else:
            try:
                shutil.rmtree(backup)
            except OSError:
                LOGGER.warning(
                    "Published replacement but could not remove backup %s", backup
                )
        return
    try:
        replace_path(staging, final_path)
    except OSError as exc:
        raise PackageValidationError(f"Package publish failed: {exc}") from exc


def _record_failure(staging, document_id, source, source_sha, exc) -> None:
    if not staging.is_dir():
        return
    payload = {
        "schema": "bemarkdown-staging-failure-v1",
        "document_id": document_id,
        "source": {"name": source.name, "sha256": source_sha},
        "failed_at": datetime.now(UTC).isoformat(),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    try:
        _write_json(staging / "failure.json", payload)
    except OSError:
        LOGGER.exception("Could not write staging failure record for %s", staging)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def _safe_package_path(package_path: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise PackageValidationError(f"Unsafe package artifact path: {relative!r}")
    resolved = (package_path / Path(*posix.parts)).resolve()
    try:
        resolved.relative_to(package_path.resolve())
    except ValueError as exc:
        raise PackageValidationError(
            f"Package artifact escapes package root: {relative!r}"
        ) from exc
    return resolved


def _review_item_count(report: dict[str, Any]) -> int:
    explicit = report.get("review_item_count")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    return sum(
        (
            report.get("assets", {}).get("unresolved", 0),
            report.get("drawingml", {}).get("unsupported_groups", 0),
            report.get("image_content", {}).get("review_required", 0),
            report.get("formula_ocr", {}).get("review_required", 0),
            report.get("formula_ocr", {}).get("rejected_preserved", 0),
            report.get("formula_ocr", {}).get("inference_failed_preserved", 0),
        )
    )


def _converter_version() -> str:
    try:
        return importlib.metadata.version("bemarkdown")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
