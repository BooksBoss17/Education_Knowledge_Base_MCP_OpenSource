from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .formula_gt_evaluation import normalize_latex_for_evaluation
from .formulanet_runtime import FormulaOcrOutputValidator, OcrVerdict

BLIND_REQUEST_SCHEMA = "bemarkdown-vision-formula-blind-request-v0"
BLIND_OBSERVATION_SCHEMA = "bemarkdown-vision-formula-blind-observation-v0"
ADJUDICATION_REQUEST_SCHEMA = "bemarkdown-vision-formula-adjudication-request-v0"
ADJUDICATION_RESULT_SCHEMA = "bemarkdown-vision-formula-adjudication-result-v0"
RESEGMENTATION_SCHEMA = "bemarkdown-vision-formula-resegmentation-ir-v0"
RESOLUTION_V2_SCHEMA = "bemarkdown-vision-formula-resolution-v2"
BLIND_PROTOCOL_VERSION = "blind-first-formula-v0"
ADJUDICATION_PROTOCOL_VERSION = "candidate-neutral-adjudication-v0"
MAX_FORMULAS_PER_BLIND_SHEET = 16

_BLIND_FORBIDDEN_FRAGMENTS = (
    "current_latex",
    "current_latex_state",
    "current_formula",
    "candidate_latex",
    "formulanet",
    "risk_score",
    "audit_priority",
    "previous_vision",
    "human_truth",
    "ground_truth",
    "expected_correction",
    "material",
    "acceptable",
)
_OBSERVATIONS = {
    "SINGLE_FORMULA",
    "REGION_NOT_SINGLE_FORMULA",
    "NO_FORMULA_VISIBLE",
    "UNRESOLVED",
}
_ADJUDICATION_DECISIONS = {
    "CANDIDATE_A",
    "CANDIDATE_B",
    "BOTH_EQUIVALENT",
    "NEITHER_SOURCE_LATEX",
    "REGION_NOT_SINGLE_FORMULA",
    "UNRESOLVED",
}
_AGENT_TERMINAL_DECISIONS = {
    "CONFIRMED_CURRENT",
    "REPLACED",
    "RECOVERED_MISSING",
    "RESEGMENTED_FORMULA",
}

BLIND_INPUT_CONTRACT = {
    "schema": BLIND_REQUEST_SCHEMA,
    "protocol_version": BLIND_PROTOCOL_VERSION,
    "agent_visible_fields": [
        "formula_id",
        "overlay_id",
        "bbox_normalized",
        "source_images",
        "instruction",
    ],
    "one_explicit_observation_per_formula": True,
    "candidate_metadata_join_stage": "AFTER_PASS_A_IMPORT",
}

BLIND_OBSERVATION_CONTRACT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BeMarkdown blind formula observation v0",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "audit_id",
        "vision_split_sha256",
        "blind_input_contract_sha256",
        "bundle_image_manifest_sha256",
        "image_manifest_sha256",
        "audit_complete",
        "observations",
        "new_formula_proposals",
    ],
    "properties": {
        "schema": {"const": BLIND_OBSERVATION_SCHEMA},
        "audit_id": {"type": "string", "minLength": 1},
        "vision_split_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "blind_input_contract_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "bundle_image_manifest_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "image_manifest_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "audit_complete": {"const": True},
        "observations": {"type": "array"},
        "new_formula_proposals": {"type": "array"},
    },
}

ADJUDICATION_CONTRACT = {
    "schema": ADJUDICATION_RESULT_SCHEMA,
    "protocol_version": ADJUDICATION_PROTOCOL_VERSION,
    "decisions": sorted(_ADJUDICATION_DECISIONS),
    "candidate_sources_agent_visible": False,
    "candidate_order": "STABLE_SHA256",
    "both_equivalent_preserves_original": True,
}

VISION_FORMULA_RESEGMENTATION_CONTRACT = {
    "schema": RESEGMENTATION_SCHEMA,
    "coordinate_space": "crop-top-left-0-0-bottom-right-1000-1000",
    "visibility": ["CLEAR", "AMBIGUOUS"],
    "minimum_clear_subregions_for_auto_resolution": 1,
    "primary_route_mutation": False,
}

VISION_FORMULA_RESOLUTION_V2_CONTRACT = {
    "schema": RESOLUTION_V2_SCHEMA,
    "final_decisions": [
        "CONFIRMED_CURRENT",
        "REPLACED",
        "RECOVERED_MISSING",
        "RESEGMENTED_FORMULA",
        "NO_FORMULA_VISIBLE",
        "UNRESOLVED",
    ],
    "agent_terminal": sorted(_AGENT_TERMINAL_DECISIONS),
    "human_fallback": ["UNRESOLVED"],
    "route_visual_reject": ["NO_FORMULA_VISIBLE"],
}


def _canonical_sha256(value: Any) -> str:
    material = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _asset_from_bytes(
    root: Path,
    data: bytes,
    *,
    width: int,
    height: int,
    generation: str,
) -> dict[str, Any]:
    sha = _sha256_bytes(data)
    relative = PurePosixPath("images") / f"sha256-{sha}.png"
    path = root.joinpath(*relative.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return {
        "path": relative.as_posix(),
        "sha256": sha,
        "bytes": len(data),
        "width": width,
        "height": height,
        "generation": generation,
    }


def _asset_from_image(
    root: Path, image: Image.Image, *, generation: str
) -> dict[str, Any]:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="PNG", optimize=True)
    return _asset_from_bytes(
        root,
        stream.getvalue(),
        width=image.width,
        height=image.height,
        generation=generation,
    )


def _named_asset_from_image(
    root: Path, name: str, image: Image.Image, *, generation: str
) -> dict[str, Any]:
    """Write one short, case-local PNG while retaining content-addressed evidence."""

    if PurePosixPath(name).name != name or not name.lower().endswith(".png"):
        raise ValueError("Named image asset must be a case-local PNG filename")
    stream = io.BytesIO()
    converted = image.convert("RGB")
    converted.save(stream, format="PNG", optimize=True)
    data = stream.getvalue()
    path = root / name
    path.write_bytes(data)
    return {
        "path": name,
        "sha256": _sha256_bytes(data),
        "bytes": len(data),
        "width": converted.width,
        "height": converted.height,
        "generation": generation,
    }


def _asset_from_existing(
    source_root: Path, target_root: Path, asset: dict[str, Any]
) -> dict[str, Any]:
    relative = PurePosixPath(str(asset["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Blind source asset must use a relative path")
    source = source_root.joinpath(*relative.parts)
    if _sha256_file(source) != str(asset["sha256"]):
        raise ValueError("Blind source clean-page SHA-256 mismatch")
    target_relative = PurePosixPath("images") / f"sha256-{asset['sha256']}.png"
    target = target_root.joinpath(*target_relative.parts)
    if not target.exists():
        _link_or_copy(source, target)
    with Image.open(target) as image:
        width, height = image.size
    return {
        "path": target_relative.as_posix(),
        "sha256": str(asset["sha256"]),
        "bytes": target.stat().st_size,
        "width": width,
        "height": height,
        "generation": "SOURCE_CLEAN_PAGE_COPY",
    }


def _valid_bbox_1000(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
        and 0 <= float(value[0]) < float(value[2]) <= 1000
        and 0 <= float(value[1]) < float(value[3]) <= 1000
    )


def _valid_bbox_pdf(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
        and float(value[0]) < float(value[2])
        and float(value[1]) < float(value[3])
    )


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = (float(left[2]) - float(left[0])) * (
        float(left[3]) - float(left[1])
    )
    right_area = (float(right[2]) - float(right[0])) * (
        float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _latex_valid(
    latex: Any,
    *,
    syntax_validator: FormulaOcrOutputValidator,
    render_validator: Callable[[str], bool],
) -> bool:
    if not isinstance(latex, str) or not latex.strip():
        return False
    syntax = syntax_validator.validate(latex)
    return (
        syntax.verdict in {OcrVerdict.VALID, OcrVerdict.VALID_WITH_WARNING}
        and bool(render_validator(latex))
        and "```" not in latex
        and "formula-review" not in latex
    )


def _render_blind_overlay(
    clean: Image.Image, formulas: list[dict[str, Any]]
) -> Image.Image:
    overlay = clean.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for row in formulas:
        bbox = row["bbox_normalized"]
        pixels = [
            round(float(bbox[0]) * overlay.width / 1000),
            round(float(bbox[1]) * overlay.height / 1000),
            round(float(bbox[2]) * overlay.width / 1000),
            round(float(bbox[3]) * overlay.height / 1000),
        ]
        draw.rectangle(pixels, outline=(214, 48, 49), width=2)
        label_box = [pixels[0], max(0, pixels[1] - 18), pixels[0] + 44, pixels[1]]
        draw.rectangle(label_box, fill=(255, 249, 196), outline=(214, 48, 49))
        draw.text(
            (label_box[0] + 2, label_box[1] + 2),
            row["overlay_id"],
            fill=(90, 20, 20),
            font=font,
        )
    return overlay


def _render_blind_sheets(
    root: Path, clean: Image.Image, formulas: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    font = ImageFont.load_default()
    for start in range(0, len(formulas), MAX_FORMULAS_PER_BLIND_SHEET):
        group = formulas[start : start + MAX_FORMULAS_PER_BLIND_SHEET]
        tiles = []
        for row in group:
            bbox = row["bbox_normalized"]
            crop_box = (
                max(0, round(float(bbox[0]) * clean.width / 1000) - 8),
                max(0, round(float(bbox[1]) * clean.height / 1000) - 8),
                min(clean.width, round(float(bbox[2]) * clean.width / 1000) + 9),
                min(clean.height, round(float(bbox[3]) * clean.height / 1000) + 9),
            )
            crop = clean.crop(crop_box)
            crop.thumbnail((500, 150))
            tiles.append((row, crop))
        sheet = Image.new("RGB", (1200, max(220, len(tiles) * 190)), "white")
        draw = ImageDraw.Draw(sheet)
        for position, (row, crop) in enumerate(tiles):
            top = position * 190 + 12
            draw.text((18, top), row["overlay_id"], fill="black", font=font)
            sheet.paste(crop, (18, top + 24))
        asset = _asset_from_image(
            root, sheet, generation="BLIND_SOURCE_CROP_SHEET_RENDER"
        )
        output.append(
            {
                **asset,
                "sheet_index": len(output),
                "formula_ids": [row["formula_id"] for row in group],
                "label_fields": ["overlay_id"],
            }
        )
    return output


def build_blind_reference_bundle_v4(
    source_bundle: str | Path,
    target_bundle: str | Path,
    *,
    expected_cohort: str = "REFERENCE",
) -> dict[str, Any]:
    """Build a PASS A-only bundle with newly rendered blind images."""

    source = Path(source_bundle).resolve()
    target = Path(target_bundle).resolve()
    cohort = str(expected_cohort).upper()
    if cohort not in {"REFERENCE", "VALIDATION"}:
        raise ValueError("Blind bundle cohort must be Reference or Validation")
    if target.exists():
        raise FileExistsError(f"Blind {cohort.title()} target already exists: {target}")
    source_manifest = json.loads(
        (source / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    if str(source_manifest.get("cohort", "")).upper() != cohort:
        raise ValueError(f"Blind bundle source must be {cohort.title()}")
    request_paths = sorted((source / "requests").glob("*.json"))
    if len(request_paths) != int(source_manifest["request_count"]):
        raise ValueError("Blind source request count mismatch")
    source_requests = [json.loads(path.read_text(encoding="utf-8")) for path in request_paths]
    by_page = {
        f"{row['document_id']}:{int(row['page_index'])}": row for row in source_requests
    }
    if len(by_page) != len(source_requests):
        raise ValueError("Blind source contains duplicate pages")
    try:
        ordered = [by_page[page_id] for page_id in source_manifest["page_ids"]]
    except KeyError as exc:
        raise ValueError("Blind source page membership mismatch") from exc

    target.mkdir(parents=True)
    (target / "requests").mkdir()
    (target / "images").mkdir()
    requests = []
    assets_by_path: dict[str, dict[str, Any]] = {}
    formula_ids_by_audit: dict[str, list[str]] = {}
    input_contract_sha = _canonical_sha256(BLIND_INPUT_CONTRACT)
    for source_request in ordered:
        formulas = []
        for index, source_formula in enumerate(
            source_request["known_formulas"], start=1
        ):
            bbox = source_formula.get("bbox_normalized")
            if not _valid_bbox_1000(bbox):
                raise ValueError("Blind source formula bbox is invalid")
            formulas.append(
                {
                    "formula_id": str(source_formula["formula_id"]),
                    "overlay_id": str(source_formula.get("overlay_id") or f"F{index:03d}"),
                    "bbox_normalized": [float(value) for value in bbox],
                    "sheet_index": (index - 1) // MAX_FORMULAS_PER_BLIND_SHEET,
                }
            )
        formula_ids = [row["formula_id"] for row in formulas]
        if formula_ids != [str(value) for value in source_request["known_formula_ids"]]:
            raise ValueError("Blind source formula order or membership mismatch")
        clean_asset = _asset_from_existing(
            source, target, source_request["images"]["clean_page"]
        )
        with Image.open(target.joinpath(*PurePosixPath(clean_asset["path"]).parts)) as raw:
            clean = raw.convert("RGB")
        overlay_asset = _asset_from_image(
            target,
            _render_blind_overlay(clean, formulas),
            generation="BLIND_ID_ONLY_OVERLAY_RENDER",
        )
        sheets = _render_blind_sheets(target, clean, formulas)
        page_assets = {"clean_page": clean_asset, "overlay_page": overlay_asset}
        for asset in list(page_assets.values()) + sheets:
            assets_by_path[asset["path"]] = {
                key: asset[key]
                for key in ("path", "sha256", "bytes", "width", "height", "generation")
            }
        image_material = [
            {"role": role, "path": asset["path"], "sha256": asset["sha256"]}
            for role, asset in sorted(page_assets.items())
        ] + [
            {
                "role": f"blind_formula_sheet_{asset['sheet_index']:03d}",
                "path": asset["path"],
                "sha256": asset["sha256"],
            }
            for asset in sheets
        ]
        audit_material = {
            "protocol_version": BLIND_PROTOCOL_VERSION,
            "document_id": source_request["document_id"],
            "page_index": int(source_request["page_index"]),
            "vision_split_sha256": source_manifest["vision_split_sha256"],
            "formula_ids": formula_ids,
            "image_manifest": image_material,
        }
        audit_id = "blind-audit-" + _canonical_sha256(audit_material)[:24]
        formula_ids_by_audit[audit_id] = formula_ids
        requests.append(
            {
                "schema": BLIND_REQUEST_SCHEMA,
                "audit_id": audit_id,
                "document_id": source_request["document_id"],
                "page_index": int(source_request["page_index"]),
                "vision_split_sha256": source_manifest["vision_split_sha256"],
                "blind_input_contract_sha256": input_contract_sha,
                "bundle_image_manifest_sha256": None,
                "image_manifest_sha256": _canonical_sha256(image_material),
                "audit_complete_required": True,
                "known_formula_ids": formula_ids,
                "known_formulas": formulas,
                "images": page_assets,
                "blind_formula_sheet_images": sheets,
                "instruction_version": "blind-formula-observation-v0",
                "new_formula_proposals_mode": "DIAGNOSTIC_ONLY",
            }
        )

    bundle_images = [assets_by_path[path] for path in sorted(assets_by_path)]
    bundle_image_sha = _canonical_sha256(
        {"schema": "bemarkdown-blind-image-manifest-v0", "images": bundle_images}
    )
    for request in requests:
        request["bundle_image_manifest_sha256"] = bundle_image_sha
        _write_json(target / "requests" / f"{request['audit_id']}.json", request)
    _write_json(target / "blind_result_schema.json", BLIND_OBSERVATION_CONTRACT)
    _write_text(
        target / "README.md",
        "# BeMarkdown independent visual formula observation\n\n"
        "This package is the independent source-image observation stage. Inspect "
        "only the supplied source images. There is no answer candidate to compare. "
        "Return one explicit observation row for every formula ID.\n",
    )
    _write_text(
        target / f"VISION_BLIND_{cohort}_HANDOFF.md",
        "# Independent visual formula observation handoff\n\n"
        "1. Observe the supplied source image without assuming an answer.\n"
        "2. For every numbered region choose SINGLE_FORMULA, "
        "REGION_NOT_SINGLE_FORMULA, NO_FORMULA_VISIBLE, or UNRESOLVED.\n"
        "3. For SINGLE_FORMULA, transcribe only visible source evidence as LaTeX.\n"
        "4. For a mixed region, report clear formula subregions in crop coordinates.\n"
        "5. Do not repair or guess from subject knowledge.\n"
        "6. Return exactly one row for every formula ID.\n",
    )
    manifest = {
        "schema": f"bemarkdown-vision-blind-{cohort.lower()}-bundle-v4",
        "bundle_name": target.name,
        "cohort": cohort,
        "status": f"READY_FOR_EXTERNAL_BLIND_{cohort}_EVAL",
        "protocol_version": BLIND_PROTOCOL_VERSION,
        "vision_split_sha256": source_manifest["vision_split_sha256"],
        "blind_input_contract_sha256": input_contract_sha,
        "blind_result_contract_sha256": _canonical_sha256(BLIND_OBSERVATION_CONTRACT),
        "bundle_image_manifest_sha256": bundle_image_sha,
        "page_count": len(requests),
        "request_count": len(requests),
        "known_formula_count": sum(len(row["known_formula_ids"]) for row in requests),
        "image_count": len(bundle_images),
        "image_bytes": sum(int(row["bytes"]) for row in bundle_images),
        "external_request_count": len(requests),
        "future_pass_b_worst_case_count": sum(
            len(row["known_formula_ids"]) for row in requests
        ),
        "page_ids": [
            f"{row['document_id']}:{int(row['page_index'])}" for row in requests
        ],
        "formula_ids_by_audit": formula_ids_by_audit,
        "page_membership_unchanged": [
            f"{row['document_id']}:{int(row['page_index'])}" for row in requests
        ]
        == source_manifest["page_ids"],
        "formula_membership_unchanged": sum(
            len(row["known_formula_ids"]) for row in requests
        )
        == int(source_manifest["known_formula_count"]),
        "image_generation": {
            "clean_pages": "SOURCE_PAGE_PIXELS_ONLY",
            "overlays": "NEW_ID_ONLY_RENDER",
            "formula_sheets": "NEW_ID_PLUS_SOURCE_CROP_RENDER",
            "old_overlay_or_sheet_reused": False,
        },
        "provider_binding": None,
        "validation_consumed": False,
    }
    manifest["bundle_sha256"] = _canonical_sha256(
        {
            "manifest": manifest,
            "requests": [_canonical_sha256(row) for row in requests],
            "images": bundle_images,
            "result_contract": BLIND_OBSERVATION_CONTRACT,
        }
    )
    _write_json(target / "bundle_manifest.json", manifest)
    return manifest


def build_blind_validation_bundle_v4(
    source_bundle: str | Path, target_bundle: str | Path
) -> dict[str, Any]:
    """Build the fixed-membership Validation PASS A bundle after protocol freeze."""

    return build_blind_reference_bundle_v4(
        source_bundle, target_bundle, expected_cohort="VALIDATION"
    )


def scan_blind_input_leakage(bundle: str | Path) -> dict[str, Any]:
    """Scan all Agent-visible text, filenames, and PNG metadata fail-closed."""

    root = Path(bundle).resolve()
    findings = []
    text_suffixes = {".json", ".jsonl", ".md", ".txt", ".html", ".svg"}
    scanned_text_files = 0
    scanned_image_metadata = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        lowered_name = relative.lower()
        for fragment in _BLIND_FORBIDDEN_FRAGMENTS:
            if fragment in lowered_name:
                findings.append(
                    {"path": relative, "surface": "FILENAME", "fragment": fragment}
                )
        if path.suffix.lower() in text_suffixes:
            scanned_text_files += 1
            lowered = path.read_text(encoding="utf-8").lower()
            for fragment in _BLIND_FORBIDDEN_FRAGMENTS:
                if fragment in lowered:
                    findings.append(
                        {"path": relative, "surface": "TEXT", "fragment": fragment}
                    )
        elif path.suffix.lower() == ".png":
            scanned_image_metadata += 1
            with Image.open(path) as image:
                metadata = json.dumps(image.info, ensure_ascii=False, sort_keys=True).lower()
            for fragment in _BLIND_FORBIDDEN_FRAGMENTS:
                if fragment in metadata:
                    findings.append(
                        {
                            "path": relative,
                            "surface": "PNG_METADATA",
                            "fragment": fragment,
                        }
                    )
    return {
        "schema": "bemarkdown-blind-input-leakage-scan-v0",
        "status": (
            "BLIND_INPUT_LEAKAGE_ZERO" if not findings else "BLIND_INPUT_LEAKAGE_FOUND"
        ),
        "leakage_count": len(findings),
        "findings": findings,
        "scanned_text_files": scanned_text_files,
        "scanned_image_metadata": scanned_image_metadata,
        "image_pixel_source_policy": "CLEAN_PAGE_PLUS_NEW_BLIND_RENDERERS",
    }


class BlindResultImporter:
    """Import PASS A observations without access to internal candidate metadata."""

    def __init__(
        self,
        expected_requests: list[dict[str, Any]],
        *,
        render_validator: Callable[[str], bool] | None = None,
    ):
        self._requests = {str(row["audit_id"]): row for row in expected_requests}
        if len(self._requests) != len(expected_requests):
            raise ValueError("Blind request audit IDs must be unique")
        self._syntax_validator = FormulaOcrOutputValidator()
        self._render_validator = render_validator or (
            lambda latex: isinstance(latex, str) and bool(latex.strip())
        )
        self._imported: set[str] = set()

    def import_payloads(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        seen = set()
        imported = []
        for payload in payloads:
            if not isinstance(payload, dict):
                return self._rejected("INVALID_BLIND_RESULT_OBJECT")
            if payload.get("schema") != BLIND_OBSERVATION_SCHEMA:
                return self._rejected("BLIND_RESULT_SCHEMA_MISMATCH")
            audit_id = str(payload.get("audit_id", ""))
            if audit_id in seen or audit_id in self._imported:
                return self._rejected("DUPLICATE_BLIND_RESPONSE")
            seen.add(audit_id)
            request = self._requests.get(audit_id)
            if request is None:
                return self._rejected("UNKNOWN_BLIND_AUDIT_ID")
            if payload.get("audit_complete") is not True:
                return self._rejected("BLIND_AUDIT_INCOMPLETE")
            for field, code in (
                ("vision_split_sha256", "BLIND_SPLIT_SHA_MISMATCH"),
                ("blind_input_contract_sha256", "BLIND_CONTRACT_SHA_MISMATCH"),
                ("bundle_image_manifest_sha256", "BLIND_BUNDLE_IMAGE_SHA_MISMATCH"),
                ("image_manifest_sha256", "BLIND_IMAGE_SHA_MISMATCH"),
            ):
                if payload.get(field) != request.get(field):
                    return self._rejected(code)
            observations = payload.get("observations")
            if not isinstance(observations, list):
                return self._rejected("BLIND_AUDIT_INCOMPLETE")
            formula_ids = [str(row.get("formula_id", "")) for row in observations]
            known_ids = [str(value) for value in request["known_formula_ids"]]
            if len(formula_ids) != len(set(formula_ids)):
                return self._rejected("DUPLICATE_BLIND_RESULT_ROW")
            if set(formula_ids) - set(known_ids):
                return self._rejected("UNKNOWN_BLIND_FORMULA_ID")
            if set(formula_ids) != set(known_ids):
                return self._rejected("BLIND_AUDIT_INCOMPLETE")
            for row in observations:
                error = self._validate_observation(row)
                if error:
                    return self._rejected(error)
            if not isinstance(payload.get("new_formula_proposals"), list):
                return self._rejected("INVALID_BLIND_DISCOVERY_DIAGNOSTIC")
            imported.append(
                {**payload, "raw_response_sha256": _canonical_sha256(payload)}
            )
        self._imported.update(seen)
        return {
            "schema": "bemarkdown-blind-result-import-v0",
            "status": "BLIND_RESULTS_IMPORTED",
            "response_count": len(imported),
            "responses": imported,
            "discovery_mode": "DIAGNOSTIC_ONLY",
            "error_codes": [],
        }

    def _validate_observation(self, row: dict[str, Any]) -> str | None:
        observation = row.get("observation")
        if observation not in _OBSERVATIONS:
            return "INVALID_BLIND_OBSERVATION"
        latex = row.get("latex")
        subregions = row.get("formula_subregions")
        if not isinstance(subregions, list):
            return "INVALID_BLIND_SUBREGION"
        if observation == "SINGLE_FORMULA":
            if not _latex_valid(
                latex,
                syntax_validator=self._syntax_validator,
                render_validator=self._render_validator,
            ):
                return "INVALID_BLIND_LATEX"
            if subregions:
                return "INVALID_BLIND_SUBREGION"
        elif observation == "REGION_NOT_SINGLE_FORMULA":
            if latex is not None:
                return "BLIND_LATEX_FORBIDDEN"
            for subregion in subregions:
                if self._validate_subregion(subregion):
                    return "INVALID_BLIND_SUBREGION"
        elif latex is not None:
            return "BLIND_LATEX_FORBIDDEN"
        elif subregions:
            return "INVALID_BLIND_SUBREGION"
        return None

    def _validate_subregion(self, row: dict[str, Any]) -> bool:
        if not isinstance(row, dict) or not _valid_bbox_1000(row.get("bbox_crop_1000")):
            return True
        visibility = row.get("visibility")
        if visibility not in {"CLEAR", "AMBIGUOUS"}:
            return True
        latex = row.get("latex")
        if visibility == "CLEAR":
            return not _latex_valid(
                latex,
                syntax_validator=self._syntax_validator,
                render_validator=self._render_validator,
            )
        if latex is None:
            return False
        return not _latex_valid(
            latex,
            syntax_validator=self._syntax_validator,
            render_validator=self._render_validator,
        )

    @staticmethod
    def _rejected(code: str) -> dict[str, Any]:
        return {
            "schema": "bemarkdown-blind-result-import-v0",
            "status": "BLIND_RESULT_IMPORT_REJECTED",
            "response_count": 0,
            "responses": [],
            "error_codes": [code],
        }


class FormulaCandidateComparator:
    """Join internal candidate metadata only after a validated blind observation."""

    def __init__(self, *, render_validator: Callable[[str], bool] | None = None):
        self._syntax_validator = FormulaOcrOutputValidator()
        self._render_validator = render_validator or (
            lambda latex: isinstance(latex, str) and bool(latex.strip())
        )

    def compare(
        self, internal_formula: dict[str, Any], blind_observation: dict[str, Any]
    ) -> dict[str, Any]:
        formula_id = str(internal_formula["formula_id"])
        if formula_id != str(blind_observation.get("formula_id", "")):
            raise ValueError("Blind observation formula ID mismatch")
        observation = blind_observation.get("observation")
        current = internal_formula.get("current_latex")
        current_state = (
            "PRESENT" if isinstance(current, str) and current.strip() else "MISSING"
        )
        base = {
            "schema": "bemarkdown-blind-candidate-comparison-v0",
            "formula_id": formula_id,
            "current_latex_state": current_state,
            "blind_observation": observation,
        }
        if observation == "REGION_NOT_SINGLE_FORMULA":
            return {
                **base,
                "comparison": "BLIND_REGION_RESEGMENTATION_REQUIRED",
                "formula_subregions": blind_observation.get("formula_subregions", []),
                "next_action": "REGION_HANDLING",
            }
        if observation == "NO_FORMULA_VISIBLE":
            return {
                **base,
                "comparison": "BLIND_NO_FORMULA_VISIBLE",
                "final_decision": "NO_FORMULA_VISIBLE",
                "next_action": "FORMULA_ROUTE_VISUAL_REJECT",
            }
        if observation == "UNRESOLVED":
            return {
                **base,
                "comparison": "BLIND_UNRESOLVED",
                "final_decision": "UNRESOLVED",
                "next_action": "HUMAN_REVIEW_PENDING",
            }
        if observation != "SINGLE_FORMULA":
            raise ValueError("Unsupported blind observation")
        blind_latex = blind_observation.get("latex")
        if not _latex_valid(
            blind_latex,
            syntax_validator=self._syntax_validator,
            render_validator=self._render_validator,
        ):
            raise ValueError("Blind formula failed machine validation")
        if current_state == "MISSING":
            return {
                **base,
                "comparison": "BLIND_MISSING_RECOVERY",
                "final_decision": "RECOVERED_MISSING",
                "final_latex": blind_latex,
                "next_action": "AGENT_VISION_RESOLVED",
            }
        assert isinstance(current, str)
        if current == blind_latex or normalize_latex_for_evaluation(
            current
        ) == normalize_latex_for_evaluation(blind_latex):
            return {
                **base,
                "comparison": "BLIND_INDEPENDENT_MATCH",
                "final_decision": "CONFIRMED_CURRENT",
                "final_latex": current,
                "next_action": "AGENT_VISION_RESOLVED",
            }
        return {
            **base,
            "comparison": "VISION_CANDIDATE_DISAGREEMENT",
            "next_action": "PASS_B_ADJUDICATION",
            "private_candidates": {
                "original_latex": current,
                "blind_observation_latex": blind_latex,
            },
        }


class AdjudicationBundleBuilder:
    """Build source/candidate image cases while keeping source mapping private."""

    def __init__(
        self,
        output_root: str | Path,
        candidate_renderer: Callable[[str], Image.Image],
        *,
        render_contract: dict[str, Any] | None = None,
    ):
        self.output_root = Path(output_root).resolve()
        self._renderer = candidate_renderer
        self.render_contract = render_contract or {
            "schema": "bemarkdown-adjudication-render-contract-v1",
            "renderer": "CALLER_SUPPLIED_IDENTICAL_PIPELINE",
            "same_renderer": True,
            "same_canvas": True,
            "same_font": True,
            "same_padding": True,
            "same_background": True,
            "same_foreground": True,
            "same_normalization": True,
            "same_image_format": True,
        }

    @staticmethod
    def candidate_order(formula_id: str) -> str:
        digest = hashlib.sha256(
            f"{formula_id}:{ADJUDICATION_PROTOCOL_VERSION}".encode()
        ).digest()
        return "CURRENT_AS_A" if digest[0] % 2 == 0 else "CURRENT_AS_B"

    def build_case(
        self,
        *,
        formula_id: str,
        audit_id: str,
        source_image: str | Path,
        original_latex: str,
        blind_latex: str,
        page_context_image: str | Path | None = None,
    ) -> dict[str, Any]:
        source = Path(source_image).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        case_id = "adjudication-" + hashlib.sha256(
            f"{formula_id}:{ADJUDICATION_PROTOCOL_VERSION}".encode()
        ).hexdigest()[:24]
        case_root = self.output_root / case_id
        case_root.mkdir(parents=True, exist_ok=False)
        with Image.open(source) as raw:
            source_asset = _named_asset_from_image(
                case_root,
                "source.png",
                raw.convert("RGB"),
                generation="SOURCE_CROP_COPY",
            )
        order = self.candidate_order(formula_id)
        if order == "CURRENT_AS_A":
            candidate_a, candidate_b = original_latex, blind_latex
        else:
            candidate_a, candidate_b = blind_latex, original_latex
        a_render = self._renderer(candidate_a)
        b_render = self._renderer(candidate_b)
        if a_render.size != b_render.size:
            raise ValueError("Adjudication candidate renders must use the same canvas")
        if a_render.mode != b_render.mode:
            raise ValueError("Adjudication candidate renders must use the same image mode")
        a_asset = _named_asset_from_image(
            case_root,
            "candidate-a.png",
            a_render,
            generation="ANONYMOUS_CANDIDATE_A_RENDER",
        )
        b_asset = _named_asset_from_image(
            case_root,
            "candidate-b.png",
            b_render,
            generation="ANONYMOUS_CANDIDATE_B_RENDER",
        )
        public = {
            "schema": ADJUDICATION_REQUEST_SCHEMA,
            "case_id": case_id,
            "formula_id": formula_id,
            "protocol_version": ADJUDICATION_PROTOCOL_VERSION,
            "source_image": source_asset,
            "candidate_a_image": a_asset,
            "candidate_b_image": b_asset,
            "render_contract_sha256": _canonical_sha256(self.render_contract),
            "allowed_decisions": sorted(_ADJUDICATION_DECISIONS),
        }
        if page_context_image is not None:
            context_path = Path(page_context_image).resolve()
            if not context_path.is_file():
                raise FileNotFoundError(context_path)
            with Image.open(context_path) as raw:
                public["page_context_image"] = _named_asset_from_image(
                    case_root,
                    "context.png",
                    raw.convert("RGB"),
                    generation="OPTIONAL_SOURCE_PAGE_CONTEXT_COPY",
                )
        _write_json(case_root / "request.json", public)
        return {
            "public_request": public,
            "private_mapping": {
                "audit_id": audit_id,
                "CANDIDATE_A": candidate_a,
                "CANDIDATE_B": candidate_b,
            },
            "stable_order": order,
            "render_integrity": {
                "same_renderer": True,
                "same_canvas": a_render.size == b_render.size,
                "same_image_mode": a_render.mode == b_render.mode,
                "canvas": list(a_render.size),
                "image_mode": a_render.mode,
                "render_contract_sha256": _canonical_sha256(self.render_contract),
            },
        }


def build_vision_formula_resegmentation_ir(
    *,
    formula_id: str,
    audit_id: str,
    source_bbox_pdf_pt: list[float],
    formula_subregions: list[dict[str, Any]],
    render_validator: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    if not _valid_bbox_pdf(source_bbox_pdf_pt):
        raise ValueError("Resegmentation source bbox is invalid")
    if not isinstance(formula_subregions, list) or not formula_subregions:
        raise ValueError("Resegmentation requires at least one subregion")
    render = render_validator or (
        lambda latex: isinstance(latex, str) and bool(latex.strip())
    )
    syntax = FormulaOcrOutputValidator()
    for row in formula_subregions:
        if not isinstance(row, dict) or not _valid_bbox_1000(row.get("bbox_crop_1000")):
            raise ValueError("Resegmentation subregion bbox is invalid")
        if row.get("visibility") not in {"CLEAR", "AMBIGUOUS"}:
            raise ValueError("Resegmentation visibility is invalid")
        if not _latex_valid(
            row.get("latex"), syntax_validator=syntax, render_validator=render
        ):
            raise ValueError("Resegmentation LaTeX is invalid")
    for index, left in enumerate(formula_subregions):
        for right in formula_subregions[index + 1 :]:
            if _bbox_iou(left["bbox_crop_1000"], right["bbox_crop_1000"]) >= 0.7:
                raise ValueError("Resegmentation subregions have abnormal overlap")
    clear = [row for row in formula_subregions if row["visibility"] == "CLEAR"]
    if not clear:
        raise ValueError("Resegmentation requires at least one CLEAR subregion")
    x0, y0, x1, y1 = [float(value) for value in source_bbox_pdf_pt]
    width = x1 - x0
    height = y1 - y0
    subregions = []
    for index, row in enumerate(formula_subregions, start=1):
        bbox = [float(value) for value in row["bbox_crop_1000"]]
        pdf_bbox = [
            round(x0 + bbox[0] * width / 1000, 6),
            round(y0 + bbox[1] * height / 1000, 6),
            round(x0 + bbox[2] * width / 1000, 6),
            round(y0 + bbox[3] * height / 1000, 6),
        ]
        subregion_id = "vision-subregion-" + hashlib.sha256(
            f"{formula_id}:{index}:{bbox}:{row['latex']}".encode()
        ).hexdigest()[:20]
        subregions.append(
            {
                "subregion_id": subregion_id,
                "bbox_crop_1000": bbox,
                "bbox_pdf_pt": pdf_bbox,
                "latex": row["latex"],
                "visibility": row["visibility"],
            }
        )
    clear_output = [row for row in subregions if row["visibility"] == "CLEAR"]
    return {
        "schema": RESEGMENTATION_SCHEMA,
        "formula_id": formula_id,
        "audit_id": audit_id,
        "source_bbox_pdf_pt": [float(value) for value in source_bbox_pdf_pt],
        "subregions": subregions,
        "status": "VISION_RESEGMENTED",
        "primary_route_mutated": False,
        "primary_formula_resolution": (
            clear_output[0]["latex"] if len(clear_output) == 1 else None
        ),
        "secondary_formula_subrecords": clear_output,
    }


class AdjudicationResultResolver:
    """Validate PASS B output and map it to final resolution v2."""

    def __init__(self, *, render_validator: Callable[[str], bool] | None = None):
        self._syntax_validator = FormulaOcrOutputValidator()
        self._render_validator = render_validator or (
            lambda latex: isinstance(latex, str) and bool(latex.strip())
        )

    def resolve(
        self,
        result: dict[str, Any],
        *,
        private_mapping: dict[str, str],
        original_latex: str,
        audit_id: str,
        source_bbox_pdf_pt: list[float],
    ) -> dict[str, Any]:
        decision = result.get("decision")
        if decision not in _ADJUDICATION_DECISIONS:
            raise ValueError("Unsupported adjudication decision")
        formula_id = str(result["formula_id"])
        latex = result.get("latex")
        subregions = result.get("formula_subregions")
        if not isinstance(subregions, list):
            raise TypeError("Adjudication subregions must be an array")
        if decision in {"CANDIDATE_A", "CANDIDATE_B"}:
            if latex is not None or subregions:
                raise ValueError("Candidate choice cannot include LaTeX or subregions")
            chosen = private_mapping[decision]
            final = (
                "CONFIRMED_CURRENT"
                if normalize_latex_for_evaluation(chosen)
                == normalize_latex_for_evaluation(original_latex)
                else "REPLACED"
            )
            return self._resolution(formula_id, final, chosen)
        if decision == "BOTH_EQUIVALENT":
            if latex is not None or subregions:
                raise ValueError("Equivalent decision cannot include extra content")
            return self._resolution(
                formula_id, "CONFIRMED_CURRENT", original_latex
            )
        if decision == "NEITHER_SOURCE_LATEX":
            if subregions or not _latex_valid(
                latex,
                syntax_validator=self._syntax_validator,
                render_validator=self._render_validator,
            ):
                raise ValueError("Adjudication source LaTeX is invalid")
            return self._resolution(formula_id, "REPLACED", latex)
        if decision == "REGION_NOT_SINGLE_FORMULA":
            if latex is not None:
                raise ValueError("Region adjudication cannot include top-level LaTeX")
            try:
                ir = build_vision_formula_resegmentation_ir(
                    formula_id=formula_id,
                    audit_id=audit_id,
                    source_bbox_pdf_pt=source_bbox_pdf_pt,
                    formula_subregions=subregions,
                    render_validator=self._render_validator,
                )
            except ValueError as exc:
                return {
                    **self._resolution(formula_id, "UNRESOLVED", None),
                    "reason": f"RESEGMENTATION_REQUIRES_HUMAN: {exc}",
                }
            return {
                **self._resolution(
                    formula_id,
                    "RESEGMENTED_FORMULA",
                    ir["primary_formula_resolution"],
                ),
                "resegmentation_ir": ir,
            }
        if latex is not None or subregions:
            raise ValueError("Unresolved adjudication cannot include extra content")
        return self._resolution(formula_id, "UNRESOLVED", None)

    @staticmethod
    def _resolution(
        formula_id: str, decision: str, final_latex: str | None
    ) -> dict[str, Any]:
        return {
            "schema": RESOLUTION_V2_SCHEMA,
            "formula_id": formula_id,
            "final_decision": decision,
            "final_latex": final_latex,
            "resolution_state": resolution_state_for_decision(decision),
        }


def resolution_state_for_decision(decision: str) -> str:
    if decision in _AGENT_TERMINAL_DECISIONS:
        return "AGENT_VISION_RESOLVED"
    if decision == "UNRESOLVED":
        return "HUMAN_REVIEW_PENDING"
    if decision == "NO_FORMULA_VISIBLE":
        return "FORMULA_ROUTE_VISUAL_REJECT"
    raise ValueError("Unsupported Vision formula resolution decision")


def apply_resolution_v2(
    formula: dict[str, Any],
    resolution: dict[str, Any],
    *,
    render_validator: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Attach a resolution without deleting or reassigning the primary route."""

    if str(formula.get("formula_id")) != str(resolution.get("formula_id")):
        raise ValueError("Resolution formula ID mismatch")
    decision = str(resolution.get("final_decision", ""))
    render = render_validator or (
        lambda latex: isinstance(latex, str) and bool(latex.strip())
    )
    syntax = FormulaOcrOutputValidator()
    final_latex = resolution.get("final_latex")
    terminal_valid = True
    if decision in {"CONFIRMED_CURRENT", "REPLACED", "RECOVERED_MISSING"}:
        terminal_valid = _latex_valid(
            final_latex, syntax_validator=syntax, render_validator=render
        )
    elif decision == "RESEGMENTED_FORMULA":
        ir = resolution.get("resegmentation_ir")
        terminal_valid = bool(
            isinstance(ir, dict)
            and ir.get("schema") == RESEGMENTATION_SCHEMA
            and ir.get("status") == "VISION_RESEGMENTED"
            and ir.get("primary_route_mutated") is False
            and ir.get("secondary_formula_subrecords")
        )
    if decision in _AGENT_TERMINAL_DECISIONS and not terminal_valid:
        resolution = {
            **resolution,
            "final_decision": "UNRESOLVED",
            "final_latex": None,
            "invalid_final_reason": "FINAL_MACHINE_VALIDATION_FAILED",
        }
        decision = "UNRESOLVED"
    state = resolution_state_for_decision(decision)
    output = {
        **formula,
        "resolution_state": state,
        "vision_resolution_v2": resolution,
        "primary_route_mutated": False,
    }
    if decision in {
        "CONFIRMED_CURRENT",
        "REPLACED",
        "RECOVERED_MISSING",
        "RESEGMENTED_FORMULA",
    } and resolution.get("final_latex") is not None:
        output["latex"] = resolution["final_latex"]
    if decision == "RECOVERED_MISSING":
        if formula.get("latex") not in {None, ""}:
            raise ValueError("RECOVERED_MISSING requires an empty original formula")
        output["resolution_provenance"] = {
            "original_current_latex": None,
            "current_latex_state": "MISSING",
            "recovered_latex": resolution["final_latex"],
            "resolver": "VISION_AGENT",
        }
    return output
