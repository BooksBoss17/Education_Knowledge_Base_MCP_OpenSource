from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont

FORMULA_PAGE_AUDIT_SCHEMA = "bemarkdown-formula-page-audit-package-v0"
FORMULA_PAGE_AUDIT_VERSION = "formula-page-audit-v0"
VISION_AUDIT_REQUEST_SCHEMA = "bemarkdown-vision-formula-page-audit-request-v0"
MAX_FORMULAS_PER_SHEET = 16

_AUDIT_PRIORITY = {
    "AUTO_ACCEPT": "NORMAL_PRIORITY",
    "REVIEW_REQUIRED": "HIGH_PRIORITY",
    "HARD_FAILURE_REVIEW": "HARD_FAILURE_PRIORITY",
}
_TRUTH_KEY_FRAGMENTS = (
    "truth",
    "human_quality",
    "formulanet_label",
    "reference_latex",
    "expected_correction",
    "ground_truth",
)


class FormulaPageAuditBuilder:
    """Build all page-level Vision audit images behind one deterministic interface."""

    def __init__(
        self,
        asset_root: str | Path,
        *,
        dpi: int = 250,
        max_formulas_per_sheet: int = MAX_FORMULAS_PER_SHEET,
    ):
        if dpi not in {200, 250, 300}:
            raise ValueError("Formula audit DPI must be one of 200, 250, or 300")
        if not 1 <= max_formulas_per_sheet <= MAX_FORMULAS_PER_SHEET:
            raise ValueError("Formula audit sheets may contain at most 16 formulas")
        self.asset_root = Path(asset_root).resolve()
        self.dpi = dpi
        self.max_formulas_per_sheet = max_formulas_per_sheet

    def build(
        self,
        *,
        document_id: str,
        page_index: int,
        source_pdf: str | Path,
        source_pdf_sha256: str,
        known_formulas: list[dict[str, Any]],
        page_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _assert_no_truth_leakage(known_formulas)
        source = Path(source_pdf).resolve()
        if _sha256_file(source) != source_pdf_sha256:
            raise ValueError("Source PDF SHA-256 mismatch")
        formulas = sorted(known_formulas, key=lambda row: str(row["formula_id"]))
        if len({row["formula_id"] for row in formulas}) != len(formulas):
            raise ValueError("Known formula IDs must be unique")

        page_image, page_size = self._render_page(source, page_index)
        clean_asset = self._write_png(page_image, "clean")
        normalized = [
            self._known_formula(row, index, page_size)
            for index, row in enumerate(formulas, start=1)
        ]
        overlay_asset = self._write_png(
            self._overlay(page_image, normalized, page_size), "overlay"
        )
        sheets = self._sheets(page_image, normalized, page_size)
        audit_material = {
            "version": FORMULA_PAGE_AUDIT_VERSION,
            "document_id": document_id,
            "page_index": int(page_index),
            "source_pdf_sha256": source_pdf_sha256,
            "dpi": self.dpi,
            "known_formulas": [
                {
                    "formula_id": row["formula_id"],
                    "bbox_pdf_pt": row["bbox_pdf_pt"],
                    "current_latex": row["current_latex"],
                    "audit_priority": row["audit_priority"],
                }
                for row in normalized
            ],
        }
        audit_id = "formula-audit-" + _sha256_json(audit_material)[:24]
        priority_counts: dict[str, int] = {}
        for row in normalized:
            priority_counts[row["audit_priority"]] = (
                priority_counts.get(row["audit_priority"], 0) + 1
            )
        return {
            "schema": FORMULA_PAGE_AUDIT_SCHEMA,
            "package_version": FORMULA_PAGE_AUDIT_VERSION,
            "audit_id": audit_id,
            "document_id": document_id,
            "page_index": int(page_index),
            "source_pdf_sha256": source_pdf_sha256,
            "render_dpi": self.dpi,
            "known_formula_ids": [row["formula_id"] for row in normalized],
            "images": {
                "clean_page": clean_asset,
                "overlay_page": overlay_asset,
            },
            "formula_sheet_images": sheets,
            "known_formulas": normalized,
            "page_context": page_context or {},
            "risk_priority_summary": dict(sorted(priority_counts.items())),
        }

    def _render_page(self, source: Path, page_index: int) -> tuple[Image.Image, list[float]]:
        with fitz.open(source) as document:
            if not 0 <= page_index < document.page_count:
                raise ValueError("Formula audit page index is out of range")
            page = document.load_page(page_index)
            page_size = [float(page.rect.width), float(page.rect.height)]
            pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return image, page_size

    @staticmethod
    def _known_formula(
        row: dict[str, Any], index: int, page_size: list[float]
    ) -> dict[str, Any]:
        bbox = _valid_bbox(row.get("bbox_pdf_pt"), page_size)
        risk_decision = row.get("risk", {}).get("decision") or row.get("risk_decision")
        if risk_decision not in _AUDIT_PRIORITY:
            raise ValueError("Known formula requires a supported Risk decision")
        return {
            "formula_id": str(row["formula_id"]),
            "overlay_id": f"F{index:03d}",
            "bbox_pdf_pt": bbox,
            "bbox_normalized": [
                round(1000 * bbox[0] / page_size[0], 6),
                round(1000 * bbox[1] / page_size[1], 6),
                round(1000 * bbox[2] / page_size[0], 6),
                round(1000 * bbox[3] / page_size[1], 6),
            ],
            "current_latex": str(row.get("current_latex") or ""),
            "audit_priority": _AUDIT_PRIORITY[risk_decision],
            "crop_ref": row.get("crop_ref"),
        }

    @staticmethod
    def _overlay(
        page_image: Image.Image,
        formulas: list[dict[str, Any]],
        page_size: list[float],
    ) -> Image.Image:
        overlay = page_image.copy()
        draw = ImageDraw.Draw(overlay)
        scale_x = overlay.width / page_size[0]
        scale_y = overlay.height / page_size[1]
        for row in formulas:
            bbox = row["bbox_pdf_pt"]
            pixels = [
                round(bbox[0] * scale_x),
                round(bbox[1] * scale_y),
                round(bbox[2] * scale_x),
                round(bbox[3] * scale_y),
            ]
            draw.rectangle(pixels, outline=(214, 48, 49), width=2)
            label_box = [pixels[0], max(0, pixels[1] - 18), pixels[0] + 44, pixels[1]]
            draw.rectangle(label_box, fill=(255, 249, 196), outline=(214, 48, 49))
            draw.text((label_box[0] + 2, label_box[1] + 2), row["overlay_id"], fill=(90, 20, 20))
        return overlay

    def _sheets(
        self,
        page_image: Image.Image,
        formulas: list[dict[str, Any]],
        page_size: list[float],
    ) -> list[dict[str, Any]]:
        output = []
        scale_x = page_image.width / page_size[0]
        scale_y = page_image.height / page_size[1]
        font = ImageFont.load_default()
        for start in range(0, len(formulas), self.max_formulas_per_sheet):
            group = formulas[start : start + self.max_formulas_per_sheet]
            tiles = []
            for row in group:
                bbox = row["bbox_pdf_pt"]
                crop_box = (
                    max(0, int(bbox[0] * scale_x) - 8),
                    max(0, int(bbox[1] * scale_y) - 8),
                    min(page_image.width, int(bbox[2] * scale_x) + 9),
                    min(page_image.height, int(bbox[3] * scale_y) + 9),
                )
                crop = page_image.crop(crop_box)
                crop.thumbnail((500, 150))
                tiles.append((row, crop))
            sheet = Image.new("RGB", (1200, max(220, len(tiles) * 190)), "white")
            draw = ImageDraw.Draw(sheet)
            for position, (row, crop) in enumerate(tiles):
                top = position * 190 + 12
                sheet.paste(crop, (18, top + 24))
                draw.text(
                    (18, top),
                    f"{row['overlay_id']}  {row['formula_id']}  {row['current_latex']}",
                    fill="black",
                    font=font,
                )
            asset = self._write_png(sheet, "sheet")
            output.append(
                {
                    **asset,
                    "sheet_index": len(output),
                    "formula_ids": [row["formula_id"] for row in group],
                }
            )
        return output

    def _write_png(self, image: Image.Image, kind: str) -> dict[str, Any]:
        stream = io.BytesIO()
        image.save(stream, format="PNG", optimize=True)
        data = stream.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        path = self.asset_root / kind / f"sha256-{sha}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return {
            "path": str(path),
            "sha256": sha,
            "bytes": len(data),
            "width": image.width,
            "height": image.height,
        }


class VisionFormulaPageAuditProvider:
    """Export and validate provider-neutral page-audit requests."""

    def __init__(self, *, render_validator):
        self._render_validator = render_validator

    def build_request(self, package: dict[str, Any]) -> dict[str, Any]:
        if package.get("schema") != FORMULA_PAGE_AUDIT_SCHEMA:
            raise ValueError("Unsupported formula page-audit package schema")
        known_ids = [str(value) for value in package.get("known_formula_ids", [])]
        if len(known_ids) != len(set(known_ids)):
            raise ValueError("Known formula IDs must be unique")
        image_manifest = [
            {"role": role, "sha256": asset["sha256"]}
            for role, asset in sorted(package["images"].items())
        ] + [
            {
                "role": f"formula_sheet_{int(asset.get('sheet_index', index)):03d}",
                "sha256": asset["sha256"],
            }
            for index, asset in enumerate(package.get("formula_sheet_images", []))
        ]
        request = {
            "schema": VISION_AUDIT_REQUEST_SCHEMA,
            "provider": None,
            "instruction_version": "vision-formula-page-audit-v0",
            "audit_id": package["audit_id"],
            "document_id": package["document_id"],
            "page_index": int(package["page_index"]),
            "known_formula_ids": known_ids,
            "images": package["images"],
            "formula_sheet_images": package.get("formula_sheet_images", []),
            "image_manifest": image_manifest,
            "image_manifest_sha256": _sha256_json(image_manifest),
            "known_formulas": package.get("known_formulas", []),
            "page_context": package.get("page_context", {}),
            "instructions": {
                "audit_every_known_formula": True,
                "report_only_changed_known_formulas": True,
                "discover_unboxed_formulas": True,
                "coordinate_space": "normalized-0-1000",
                "objectives": [
                    "Check every numbered known formula against the source image.",
                    "Scan the clean page for clear formulas that have no overlay number.",
                ],
                "constraints": [
                    "Transcribe only what is visible in the supplied images.",
                    "Do not repair meaning from physics or mathematical expectations.",
                    "Do not treat question numbers, ordinary numbers, units, or prose as formulas.",
                    "Return strict JSON matching the expected result schema.",
                ],
            },
        }
        _assert_no_truth_leakage(request)
        return request

    @staticmethod
    def submit_or_export(
        request: dict[str, Any], output_path: str | Path
    ) -> dict[str, Any]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return {
            "status": "VISION_AGENT_EXTERNAL_EVAL_PENDING",
            "audit_id": request["audit_id"],
            "request_path": str(output.resolve()),
            "request_sha256": _sha256_file(output),
        }

    def parse_result(
        self,
        request: dict[str, Any],
        response: dict[str, Any] | str,
        *,
        attempt: int = 1,
    ) -> dict[str, Any]:
        known_ids = list(request["known_formula_ids"])
        try:
            parsed = json.loads(response) if isinstance(response, str) else response
            if not isinstance(parsed, dict):
                raise TypeError("Vision response must be a JSON object")
            if parsed.get("audit_id") != request["audit_id"]:
                raise ValueError("Vision response audit ID mismatch")
            checked = parsed.get("checked_formula_ids")
            if not isinstance(checked, list) or len(checked) != len(set(checked)):
                raise ValueError("Checked formula IDs must be a unique list")
            if set(checked) != set(known_ids) or not parsed.get("audit_complete"):
                return self._failure(
                    request, attempt, ["AUDIT_INCOMPLETE"]
                )
            results = parsed.get("known_formula_results")
            proposals = parsed.get("new_formula_proposals")
            if not isinstance(results, list) or not isinstance(proposals, list):
                raise TypeError("Vision formula results must be lists")
            proposal_signatures = set()
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    raise TypeError("New formula proposal must be an object")
                bbox_1000 = _valid_bbox_1000(proposal.get("bbox_1000"))
                latex = proposal.get("latex")
                if (
                    proposal.get("visibility") != "CLEAR"
                    or not isinstance(latex, str)
                    or not latex.strip()
                    or not self._render_validator(latex)
                ):
                    raise ValueError("New formula proposal is invalid or ambiguous")
                signature = (tuple(bbox_1000), latex)
                if signature in proposal_signatures:
                    raise ValueError("Duplicate new formula proposal")
                proposal_signatures.add(signature)
            by_id: dict[str, dict[str, Any]] = {}
            for row in results:
                if not isinstance(row, dict):
                    raise TypeError("Known formula result must be an object")
                formula_id = str(row.get("formula_id", ""))
                if formula_id not in known_ids or formula_id in by_id:
                    raise ValueError("Unknown or duplicate known formula result")
                decision = row.get("decision")
                if decision not in {"CONFIRMED_CURRENT", "REPLACED", "UNRESOLVED"}:
                    raise ValueError("Unsupported known formula decision")
                if decision == "REPLACED":
                    latex = row.get("latex")
                    if not isinstance(latex, str) or not latex.strip():
                        raise ValueError("Replacement LaTeX is required")
                    if not self._render_validator(latex):
                        raise ValueError("Replacement LaTeX is not renderable")
                by_id[formula_id] = dict(row)
            normalized = [
                by_id.get(
                    formula_id,
                    {"formula_id": formula_id, "decision": "CONFIRMED_CURRENT"},
                )
                for formula_id in known_ids
            ]
            return {
                "status": "VISION_AUDIT_PARSED",
                "audit_id": request["audit_id"],
                "known_formula_results": normalized,
                "new_formula_proposals": proposals,
                "known_formula_ids_for_human": [],
                "error_codes": [],
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return self._failure(request, attempt, ["VISION_RESPONSE_INVALID"])

    @staticmethod
    def _failure(
        request: dict[str, Any], attempt: int, error_codes: list[str]
    ) -> dict[str, Any]:
        terminal = attempt >= 2
        return {
            "status": "HUMAN_REVIEW_PENDING" if terminal else "VISION_AUDIT_RETRY_REQUIRED",
            "audit_id": request["audit_id"],
            "known_formula_results": [],
            "new_formula_proposals": [],
            "known_formula_ids_for_human": list(request["known_formula_ids"]),
            "error_codes": error_codes,
        }


class VisionFormulaAuditResultImporter:
    """Import external JSON/JSONL results with frozen-request provenance checks."""

    def __init__(
        self,
        expected_requests: list[dict[str, Any]],
        *,
        split_sha256: str | None = None,
        request_schema_sha256: str | None = None,
        bundle_image_manifest_sha256: str | None = None,
        cohort: str = "REFERENCE",
        validation: bool = False,
        truth_sidecar: str | Path | None = None,
        consumed_once_marker: str | Path | None = None,
    ):
        self._requests = {str(row["audit_id"]): row for row in expected_requests}
        if len(self._requests) != len(expected_requests):
            raise ValueError("Expected request audit IDs must be unique")
        self._split_sha256 = split_sha256
        self._request_schema_sha256 = request_schema_sha256
        self._bundle_image_manifest_sha256 = bundle_image_manifest_sha256
        self._cohort = str(cohort).upper()
        self._validation = bool(validation)
        self._consumed_once_marker = (
            Path(consumed_once_marker) if consumed_once_marker is not None else None
        )
        self._imported_audit_ids: set[str] = set()
        if self._cohort not in {"REFERENCE", "VALIDATION"}:
            raise ValueError("Importer cohort must be REFERENCE or VALIDATION")
        if self._cohort == "VALIDATION" and not self._validation:
            raise ValueError("Validation import requires explicit validation mode")
        if self._cohort == "VALIDATION" and self._consumed_once_marker is None:
            raise ValueError("Validation import requires a consumed-once marker")
        if self._consumed_once_marker is not None and self._consumed_once_marker.exists():
            raise ValueError("Validation results were already consumed once")
        self._validate_request_provenance()
        if truth_sidecar is not None:
            self._validate_truth_sidecar(Path(truth_sidecar))

    def _validate_request_provenance(self) -> None:
        expected = {
            "vision_split_sha256": self._split_sha256,
            "request_schema_sha256": self._request_schema_sha256,
            "bundle_image_manifest_sha256": self._bundle_image_manifest_sha256,
        }
        for request in self._requests.values():
            for field, value in expected.items():
                if value is not None and request.get(field) != value:
                    raise ValueError(f"Expected request {field} mismatch")

    def _validate_truth_sidecar(self, path: Path) -> None:
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Evaluator truth sidecar is invalid") from exc
        cohorts = {str(row.get("evaluator_cohort", "")).upper() for row in rows}
        if self._cohort == "REFERENCE" and "VALIDATION" in cohorts:
            raise ValueError("Reference importer cannot load Validation truth")
        if not rows or cohorts != {self._cohort}:
            raise ValueError("Evaluator truth sidecar cohort mismatch")

    def import_path(self, source: str | Path) -> dict[str, Any]:
        path = Path(source)
        try:
            if path.suffix.lower() == ".jsonl":
                payloads = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                payloads = value if isinstance(value, list) else [value]
        except (OSError, json.JSONDecodeError):
            return self._rejected("INVALID_JSON")
        return self.import_payloads(payloads)

    def import_payloads(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        imported = []
        for payload in payloads:
            if not isinstance(payload, dict):
                return self._rejected("INVALID_RESULT_OBJECT")
            if payload.get("schema") != "bemarkdown-vision-formula-page-audit-result-v0":
                return self._rejected("RESULT_SCHEMA_MISMATCH")
            audit_id = str(payload.get("audit_id", ""))
            if audit_id in seen:
                return self._rejected("DUPLICATE_RESPONSE")
            seen.add(audit_id)
            request = self._requests.get(audit_id)
            if request is None:
                return self._rejected("UNKNOWN_AUDIT_ID")
            if (
                self._split_sha256 is not None
                and payload.get("vision_split_sha256") != self._split_sha256
            ):
                return self._rejected("SPLIT_SHA_MISMATCH")
            if (
                self._request_schema_sha256 is not None
                and payload.get("request_schema_sha256")
                != self._request_schema_sha256
            ):
                return self._rejected("REQUEST_SCHEMA_SHA_MISMATCH")
            if (
                self._bundle_image_manifest_sha256 is not None
                and payload.get("bundle_image_manifest_sha256")
                != self._bundle_image_manifest_sha256
            ):
                return self._rejected("BUNDLE_IMAGE_SHA_MISMATCH")
            if payload.get("image_manifest_sha256") != request.get(
                "image_manifest_sha256"
            ):
                return self._rejected("IMAGE_SHA_MISMATCH")
            checked = payload.get("checked_formula_ids")
            if (
                not isinstance(checked, list)
                or len(checked) != len(set(checked))
                or set(checked) != set(request["known_formula_ids"])
            ):
                return self._rejected("CHECKED_IDS_MISMATCH")
            if audit_id in self._imported_audit_ids:
                return self._rejected("DUPLICATE_RESPONSE")
            imported.append(
                {
                    **payload,
                    "raw_response_sha256": _sha256_json(payload),
                }
            )
        if self._cohort == "VALIDATION" and imported:
            self._write_consumed_once_marker(imported)
        self._imported_audit_ids.update(seen)
        return {
            "schema": "bemarkdown-vision-formula-result-import-v0",
            "status": "VISION_RESULTS_IMPORTED",
            "response_count": len(imported),
            "responses": imported,
            "error_codes": [],
        }

    def _write_consumed_once_marker(self, imported: list[dict[str, Any]]) -> None:
        assert self._consumed_once_marker is not None
        marker = {
            "schema": "bemarkdown-vision-validation-consumed-once-v0",
            "cohort": "VALIDATION",
            "consumed_once": True,
            "consumed_once_at_utc": datetime.now(UTC).isoformat(),
            "response_count": len(imported),
            "audit_ids": sorted(str(row["audit_id"]) for row in imported),
        }
        self._consumed_once_marker.parent.mkdir(parents=True, exist_ok=True)
        lock = self._consumed_once_marker.with_suffix(
            self._consumed_once_marker.suffix + ".lock"
        )
        try:
            with lock.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write("validation-consume-lock\n")
        except FileExistsError as exc:
            raise ValueError("Validation result consumption is already in progress") from exc
        temporary = self._consumed_once_marker.with_suffix(
            self._consumed_once_marker.suffix + ".tmp"
        )
        try:
            if self._consumed_once_marker.exists():
                raise ValueError("Validation results were already consumed once")
            temporary.write_text(
                json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            temporary.replace(self._consumed_once_marker)
        finally:
            temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)

    @staticmethod
    def _rejected(error_code: str) -> dict[str, Any]:
        return {
            "schema": "bemarkdown-vision-formula-result-import-v0",
            "status": "VISION_RESULT_IMPORT_REJECTED",
            "response_count": 0,
            "responses": [],
            "error_codes": [error_code],
        }


def normalize_discovery_proposals(
    *,
    audit_id: str,
    document_id: str,
    page_index: int,
    proposals: list[dict[str, Any]],
    known_bboxes_pdf_pt: list[list[float]],
    page_size_pdf_pt: list[float],
    render_size_px: list[int],
    render_validator,
) -> dict[str, Any]:
    """Validate Vision discoveries and isolate them from primary Router output."""
    normalized = []
    discovered = []
    for index, proposal in enumerate(proposals):
        bbox_1000 = _valid_bbox_1000(proposal.get("bbox_1000"))
        latex = proposal.get("latex")
        visibility = proposal.get("visibility")
        if visibility != "CLEAR":
            raise ValueError("New formula visibility must be CLEAR")
        if not isinstance(latex, str) or not latex.strip() or not render_validator(latex):
            raise ValueError("New formula LaTeX must be valid and renderable")
        bbox_pdf = [
            round(bbox_1000[0] * page_size_pdf_pt[0] / 1000, 6),
            round(bbox_1000[1] * page_size_pdf_pt[1] / 1000, 6),
            round(bbox_1000[2] * page_size_pdf_pt[0] / 1000, 6),
            round(bbox_1000[3] * page_size_pdf_pt[1] / 1000, 6),
        ]
        bbox_pixels = [
            round(bbox_1000[0] * render_size_px[0] / 1000),
            round(bbox_1000[1] * render_size_px[1] / 1000),
            round(bbox_1000[2] * render_size_px[0] / 1000),
            round(bbox_1000[3] * render_size_px[1] / 1000),
        ]
        overlaps = [_bbox_overlap(bbox_pdf, known) for known in known_bboxes_pdf_pt]
        if any(item[1] >= 0.8 or item[0] >= 0.5 for item in overlaps):
            status = "KNOWN_FORMULA_DUPLICATE"
        elif any(item[2] > 0 for item in overlaps):
            status = "DISCOVERY_REVIEW_CONFLICT"
        else:
            status = "AGENT_VISION_DISCOVERED"
        formula_id = "vision-formula-" + _sha256_json(
            {
                "audit_id": audit_id,
                "bbox_1000": bbox_1000,
                "latex": latex,
                "proposal_index": index,
            }
        )[:24]
        row = {
            "schema": "bemarkdown-vision-discovered-formula-ir-v0",
            "vision_formula_id": formula_id,
            "document_id": document_id,
            "page_index": int(page_index),
            "bbox_1000": bbox_1000,
            "bbox_pdf_pt": bbox_pdf,
            "bbox_render_px": bbox_pixels,
            "latex": latex,
            "visibility": visibility,
            "status": status,
            "audit_id": audit_id,
            "provenance": {"primary_route_modified": False},
        }
        normalized.append(row)
        if status == "AGENT_VISION_DISCOVERED":
            discovered.append(row)
    return {"proposals": normalized, "vision_discovered_formulas": discovered}


def apply_page_audit_writeback(
    formulas: list[dict[str, Any]],
    parsed_result: dict[str, Any],
    *,
    render_validator,
) -> list[dict[str, Any]]:
    """Apply page decisions by stable ID without reopening Agent terminal states."""
    original_ids = [str(row["formula_id"]) for row in formulas]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("Formula IDs must be unique for writeback")
    if parsed_result.get("status") == "VISION_AUDIT_RETRY_REQUIRED":
        return [
            {**row, "resolution_state": "VISION_REVIEW_PENDING"} for row in formulas
        ]
    if parsed_result.get("status") != "VISION_AUDIT_PARSED":
        return [
            {**row, "resolution_state": "HUMAN_REVIEW_PENDING"} for row in formulas
        ]
    decisions = parsed_result.get("known_formula_results", [])
    by_id = {str(row.get("formula_id")): row for row in decisions}
    if len(by_id) != len(decisions) or set(by_id) != set(original_ids):
        raise ValueError("Writeback decisions must exactly cover stable formula IDs")
    output = []
    for formula in formulas:
        formula_id = str(formula["formula_id"])
        decision = by_id[formula_id]
        state = "AGENT_VISION_RESOLVED"
        latex = formula.get("latex", "")
        resolution_provenance = None
        if decision.get("decision") == "REPLACED":
            replacement = decision.get("latex")
            if not isinstance(replacement, str) or not render_validator(replacement):
                raise ValueError("Replacement failed machine validation")
            latex = replacement
        elif decision.get("decision") == "RECOVERED_MISSING":
            if formula.get("latex") not in {None, ""}:
                raise ValueError("RECOVERED_MISSING requires an empty original formula")
            replacement = decision.get("latex")
            if not isinstance(replacement, str) or not render_validator(replacement):
                raise ValueError("Recovered formula failed machine validation")
            latex = replacement
            resolution_provenance = {
                "original_current_latex": None,
                "current_latex_state": "MISSING",
                "recovered_latex": replacement,
                "resolver": "VISION_AGENT",
            }
        elif decision.get("decision") == "UNRESOLVED":
            state = "HUMAN_REVIEW_PENDING"
        elif decision.get("decision") != "CONFIRMED_CURRENT":
            raise ValueError("Unsupported page-audit writeback decision")
        written = {
            **formula,
            "latex": latex,
            "resolution_state": state,
            "audit_id": parsed_result.get("audit_id"),
            "resolver_kind": "VISION_AGENT",
        }
        if resolution_provenance is not None:
            written["resolution_provenance"] = resolution_provenance
        output.append(written)
    return output


def formula_assurance_dispatch(
    formulas: list[dict[str, Any]],
    *,
    vision_input_available: bool,
    strict_no_vision: bool = False,
) -> dict[str, Any]:
    ids = [str(row["formula_id"]) for row in formulas]
    if vision_input_available:
        return {
            "mode": "FULL_AUTO_VISION_ASSURANCE",
            "vision_formula_ids": ids,
            "human_formula_ids": [],
            "preserved_formula_ids": [],
            "lower_assurance": False,
        }
    if strict_no_vision:
        return {
            "mode": "NO_VISION_STRICT_ASSURANCE",
            "vision_formula_ids": [],
            "human_formula_ids": ids,
            "preserved_formula_ids": [],
            "lower_assurance": True,
        }
    human = [
        str(row["formula_id"])
        for row in formulas
        if row.get("audit_priority") in {"HIGH_PRIORITY", "HARD_FAILURE_PRIORITY"}
    ]
    return {
        "mode": "NO_VISION_STANDARD_ASSURANCE",
        "vision_formula_ids": [],
        "human_formula_ids": human,
        "preserved_formula_ids": [formula_id for formula_id in ids if formula_id not in human],
        "lower_assurance": True,
    }


def freeze_discovery_page_split(
    pages: list[dict[str, Any]], *, reference_count: int = 50
) -> dict[str, Any]:
    """Freeze a deterministic document/profile/density-stratified page split."""
    if not 0 < reference_count < len(pages):
        raise ValueError("Discovery split requires non-empty reference and validation sets")
    normalized = []
    for page in pages:
        page_id = f"{page['document_id']}:{int(page['page_index'])}"
        count = len(page.get("formula_occurrences", []))
        density = "ZERO" if count == 0 else "LOW" if count <= 3 else "MEDIUM" if count <= 8 else "HIGH"
        normalized.append(
            {
                **page,
                "page_id": page_id,
                "density_band": density,
                "natural_miss_count": len(page.get("natural_miss_ids", [])),
            }
        )
    if len({row["page_id"] for row in normalized}) != len(normalized):
        raise ValueError("Discovery page IDs must be unique")

    miss_pages = sorted(
        (row for row in normalized if row["natural_miss_count"]),
        key=lambda row: _sha256_json(row["page_id"]),
    )
    reference = [miss_pages[0]] if miss_pages else []
    validation = [miss_pages[1]] if len(miss_pages) > 1 else []
    reserved = {row["page_id"] for row in reference + validation}

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in normalized:
        if row["page_id"] in reserved:
            continue
        key = (
            str(row["document_id"]),
            str(row.get("source_profile", "UNKNOWN")),
            row["density_band"],
        )
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: _sha256_json(row["page_id"]))
    interleaved = []
    group_keys = sorted(groups, key=_sha256_json)
    while any(groups[key] for key in group_keys):
        for key in group_keys:
            if groups[key]:
                interleaved.append(groups[key].pop(0))

    validation_count = len(pages) - reference_count
    for row in interleaved:
        if len(reference) >= reference_count:
            validation.append(row)
        elif len(validation) >= validation_count:
            reference.append(row)
        else:
            assigned = len(reference) + len(validation)
            target_reference_now = reference_count * (assigned + 1) / len(pages)
            (reference if len(reference) < target_reference_now else validation).append(row)
    if len(reference) != reference_count or len(validation) != validation_count:
        raise AssertionError("Discovery split allocation failed")
    manifest_material = {
        "schema": "bemarkdown-vision-discovery-page-split-v0",
        "selection": "deterministic-document-profile-density-stratified-v0",
        "reference_page_ids": [row["page_id"] for row in reference],
        "validation_page_ids": [row["page_id"] for row in validation],
        "reference_natural_miss_count": sum(
            row["natural_miss_count"] for row in reference
        ),
        "validation_natural_miss_count": sum(
            row["natural_miss_count"] for row in validation
        ),
    }
    return {**manifest_material, "split_sha256": _sha256_json(manifest_material)}


def select_controlled_mask_formula_ids(
    page_id: str, formula_ids: list[str]
) -> list[str]:
    """Select one or two formulas per page by hash, independent of difficulty."""
    unique_ids = {str(value) for value in formula_ids}
    if not unique_ids:
        return []
    count = min(len(unique_ids), 1 + int(_sha256_json(page_id)[0], 16) % 2)
    return sorted(
        unique_ids, key=lambda formula_id: _sha256_json([page_id, formula_id])
    )[:count]


def match_discovery_proposals(
    proposals: list[dict[str, Any]], references: list[dict[str, Any]]
) -> dict[str, Any]:
    """Deterministically match discoveries using overlap, coverage, and centers."""
    usable = [row for row in proposals if row.get("status") == "AGENT_VISION_DISCOVERED"]
    candidates = []
    for proposal in usable:
        for reference in references:
            iou, proposal_coverage, intersection = _bbox_overlap(
                proposal["bbox_pdf_pt"], reference["bbox_pdf_pt"]
            )
            reference_coverage = _bbox_overlap(
                reference["bbox_pdf_pt"], proposal["bbox_pdf_pt"]
            )[1]
            center_score = _center_match_score(
                proposal["bbox_pdf_pt"], reference["bbox_pdf_pt"]
            )
            if iou >= 0.1 or reference_coverage >= 0.5 or center_score > 0:
                candidates.append(
                    (
                        -(max(iou, reference_coverage) + center_score),
                        str(proposal["vision_formula_id"]),
                        str(reference["occurrence_id"]),
                        proposal_coverage,
                        intersection,
                    )
                )
    proposal_matches: dict[str, str] = {}
    reference_matches: dict[str, str] = {}
    for _, proposal_id, reference_id, _, _ in sorted(candidates):
        if proposal_id not in proposal_matches and reference_id not in reference_matches:
            proposal_matches[proposal_id] = reference_id
            reference_matches[reference_id] = proposal_id

    assignments = []
    for proposal in usable:
        proposal_id = str(proposal["vision_formula_id"])
        assignments.append(
            {
                "proposal_id": proposal_id,
                "reference_id": proposal_matches.get(proposal_id),
                "status": "MATCH" if proposal_id in proposal_matches else "FALSE_PROPOSAL",
            }
        )
    for reference in references:
        reference_id = str(reference["occurrence_id"])
        if reference_id not in reference_matches:
            assignments.append(
                {
                    "proposal_id": None,
                    "reference_id": reference_id,
                    "status": "UNMATCHED_REFERENCE",
                }
            )

    def metrics(cohort: str) -> dict[str, Any]:
        cohort_ids = {
            str(row["occurrence_id"])
            for row in references
            if row.get("cohort") == cohort
        }
        matched = len(cohort_ids & reference_matches.keys())
        return {
            "reference_count": len(cohort_ids),
            "matched": matched,
            "recall": matched / len(cohort_ids) if cohort_ids else None,
        }

    return {
        "schema": "bemarkdown-vision-discovery-match-result-v0",
        "matcher_version": "vision-discovery-matcher-v0",
        "assignments": assignments,
        "natural_miss_metrics": metrics("NATURAL_MISS"),
        "controlled_mask_metrics": metrics("CONTROLLED_MASK"),
        "false_new_formula_proposals": sum(
            row["status"] == "FALSE_PROPOSAL" for row in assignments
        ),
    }


def _valid_bbox_1000(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Discovery bbox_1000 must have four coordinates")
    bbox = [float(number) for number in value]
    if not (0 <= bbox[0] < bbox[2] <= 1000 and 0 <= bbox[1] < bbox[3] <= 1000):
        raise ValueError("Discovery bbox_1000 is invalid")
    return [int(number) if number.is_integer() else number for number in bbox]


def _bbox_overlap(left: list[float], right: list[float]) -> tuple[float, float, float]:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return (
        intersection / union if union else 0.0,
        intersection / left_area if left_area else 0.0,
        intersection,
    )


def _center_match_score(left: list[float], right: list[float]) -> float:
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    contained = (
        right[0] <= left_center[0] <= right[2]
        and right[1] <= left_center[1] <= right[3]
    ) or (
        left[0] <= right_center[0] <= left[2]
        and left[1] <= right_center[1] <= left[3]
    )
    if contained:
        return 0.25
    scale = max(right[2] - right[0], right[3] - right[1], 1.0)
    distance = ((left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2) ** 0.5
    return 0.1 if distance <= 0.35 * scale else 0.0


def formula_audit_priority(risk_decision: str) -> str:
    try:
        return _AUDIT_PRIORITY[risk_decision]
    except KeyError as exc:
        raise ValueError("Unsupported Formula Risk decision") from exc


def _assert_no_truth_leakage(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in _TRUTH_KEY_FRAGMENTS):
                raise ValueError(f"Truth-bearing field is forbidden in Agent package: {key}")
            _assert_no_truth_leakage(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_truth_leakage(child)


def _valid_bbox(value: Any, page_size: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Formula bbox must have four coordinates")
    bbox = [float(number) for number in value]
    if not (
        0 <= bbox[0] < bbox[2] <= page_size[0]
        and 0 <= bbox[1] < bbox[3] <= page_size[1]
    ):
        raise ValueError("Formula bbox is outside the page")
    return bbox


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
