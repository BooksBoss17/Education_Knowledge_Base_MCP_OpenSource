from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .formulanet_runtime import FormulaOcrOutputValidator, OcrVerdict
from .vision_evaluation_context import assert_quality_gate_eligible

LEGACY_SPARSE_REFERENCE_RESULT_SHA256 = (
    "eae2adb4ce7391001cf8adbba4d8809b9df584efac451b1dbe08190e595954ab"
)
RESULT_SCHEMA_V0 = "bemarkdown-vision-formula-page-audit-result-v0"
RESULT_SCHEMA_V1 = "bemarkdown-vision-formula-page-audit-result-v1"
REQUEST_SCHEMA_V1 = "bemarkdown-vision-formula-page-audit-request-v1"

REQUEST_CONTRACT_V1 = {
    "schema": REQUEST_SCHEMA_V1,
    "evaluation_protocol": "vision-recognition-page-disjoint-v3-result-v1",
    "required_provenance": [
        "vision_split_sha256",
        "request_schema_sha256",
        "bundle_image_manifest_sha256",
        "image_manifest_sha256",
    ],
    "required_membership": [
        "audit_id",
        "document_id",
        "page_index",
        "known_formula_ids",
    ],
    "known_formula_required_fields": [
        "formula_id",
        "current_latex_state",
        "current_latex",
        "audit_priority",
        "bbox_normalized",
        "crop_ref",
    ],
    "one_explicit_result_row_per_known_formula": True,
}

RESULT_CONTRACT_V1 = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BeMarkdown Vision Formula Page Audit Result v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "audit_id",
        "vision_split_sha256",
        "request_schema_sha256",
        "bundle_image_manifest_sha256",
        "image_manifest_sha256",
        "audit_complete",
        "checked_formula_ids",
        "known_formula_results",
        "new_formula_proposals",
    ],
    "properties": {
        "schema": {"const": RESULT_SCHEMA_V1},
        "audit_id": {"type": "string", "minLength": 1},
        "vision_split_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "request_schema_sha256": {
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
        "checked_formula_ids": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "known_formula_results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "formula_id",
                    "current_latex_state",
                    "decision",
                    "latex",
                ],
                "properties": {
                    "formula_id": {"type": "string", "minLength": 1},
                    "current_latex_state": {"enum": ["PRESENT", "MISSING"]},
                    "decision": {
                        "enum": [
                            "CONFIRMED_CURRENT",
                            "REPLACED",
                            "RECOVERED_MISSING",
                            "UNRESOLVED",
                        ]
                    },
                    "latex": {"type": ["string", "null"]},
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {"current_latex_state": {"const": "PRESENT"}}
                        },
                        "then": {
                            "properties": {
                                "decision": {
                                    "enum": [
                                        "CONFIRMED_CURRENT",
                                        "REPLACED",
                                        "UNRESOLVED",
                                    ]
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"current_latex_state": {"const": "MISSING"}}
                        },
                        "then": {
                            "properties": {
                                "decision": {
                                    "enum": ["RECOVERED_MISSING", "UNRESOLVED"]
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {
                                "decision": {"enum": ["REPLACED", "RECOVERED_MISSING"]}
                            }
                        },
                        "then": {
                            "properties": {"latex": {"type": "string", "minLength": 1}}
                        },
                        "else": {"properties": {"latex": {"const": None}}},
                    },
                ],
            },
        },
        "new_formula_proposals": {"type": "array"},
    },
}


def _request_index(requests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(request["audit_id"]): request for request in requests}
    if len(indexed) != len(requests):
        raise ValueError("Reference requests contain duplicate audit IDs")
    return indexed


def _current_latex_state(formula: dict[str, Any]) -> str:
    current = formula.get("current_latex")
    return "PRESENT" if isinstance(current, str) and current.strip() else "MISSING"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


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


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def build_reference_replay_bundle_v3(
    source_bundle: str | Path, target_bundle: str | Path
) -> dict[str, Any]:
    """Upgrade an immutable Reference v2 bundle to explicit request/result v1."""

    source = Path(source_bundle).resolve()
    target = Path(target_bundle).resolve()
    if target.exists():
        raise FileExistsError(f"Reference replay target already exists: {target}")
    source_manifest = json.loads(
        (source / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    if str(source_manifest.get("cohort", "")).upper() != "REFERENCE":
        raise ValueError("Reference replay source must be a Reference bundle")
    request_paths = sorted((source / "requests").glob("*.json"))
    if len(request_paths) != int(source_manifest["request_count"]):
        raise ValueError("Reference replay source request count mismatch")
    source_requests = [json.loads(path.read_text(encoding="utf-8")) for path in request_paths]
    requests_by_page = {
        f"{request['document_id']}:{int(request['page_index'])}": request
        for request in source_requests
    }
    if len(requests_by_page) != len(source_requests):
        raise ValueError("Reference replay source contains duplicate pages")
    try:
        ordered_source_requests = [
            requests_by_page[page_id] for page_id in source_manifest["page_ids"]
        ]
    except KeyError as exc:
        raise ValueError("Reference replay source page membership mismatch") from exc
    target.mkdir(parents=True)
    (target / "requests").mkdir()
    (target / "images").mkdir()
    requests = []
    used_assets: dict[str, dict[str, Any]] = {}
    for request in ordered_source_requests:
        upgraded = dict(request)
        upgraded["schema"] = REQUEST_SCHEMA_V1
        upgraded["instruction_version"] = "vision-formula-page-audit-v1"
        upgraded["instructions"] = {
            "audit_every_known_formula": True,
            "one_result_row_per_known_formula": True,
            "omit_result_rows": False,
            "discover_unboxed_formulas": True,
            "coordinate_space": "normalized-0-1000",
            "present_decisions": [
                "CONFIRMED_CURRENT",
                "REPLACED",
                "UNRESOLVED",
            ],
            "missing_decisions": ["RECOVERED_MISSING", "UNRESOLVED"],
            "constraints": [
                "Transcribe only visible source evidence from the supplied images.",
                "Do not repair meaning from subject-matter expectations.",
                "Do not treat question numbers, ordinary numbers, units, or prose as formulas.",
                "Return strict JSON matching the expected result schema.",
            ],
        }
        upgraded_formulas = []
        for formula in request["known_formulas"]:
            state = _current_latex_state(formula)
            upgraded_formulas.append(
                {
                    **formula,
                    "current_latex_state": state,
                    "current_latex": (
                        formula.get("current_latex") if state == "PRESENT" else None
                    ),
                }
            )
        upgraded["known_formulas"] = upgraded_formulas
        for asset in list(request.get("images", {}).values()) + list(
            request.get("formula_sheet_images", [])
        ):
            relative = PurePosixPath(str(asset["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    "Reference replay source contains a non-relative asset"
                )
            source_asset = source.joinpath(*relative.parts)
            if _sha256_file(source_asset) != asset["sha256"]:
                raise ValueError("Reference replay source image SHA-256 mismatch")
            target_asset = target.joinpath(*relative.parts)
            if not target_asset.exists():
                _link_or_copy(source_asset, target_asset)
            used_assets[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": str(asset["sha256"]),
                "bytes": source_asset.stat().st_size,
            }
        requests.append(upgraded)

    images = [used_assets[path] for path in sorted(used_assets)]
    image_manifest_material = {
        "schema": "bemarkdown-bundle-image-manifest-v0",
        "images": images,
    }
    bundle_image_sha = _canonical_sha256(image_manifest_material)
    request_schema_sha = _canonical_sha256(REQUEST_CONTRACT_V1)
    for request in requests:
        request["request_schema_sha256"] = request_schema_sha
        request["bundle_image_manifest_sha256"] = bundle_image_sha
        _write_json(target / "requests" / f"{request['audit_id']}.json", request)
    _write_json(target / "result_schema.json", RESULT_CONTRACT_V1)
    _write_json(
        target / "image_manifest.json",
        {**image_manifest_material, "image_manifest_sha256": bundle_image_sha},
    )
    page_ids = [
        f"{request['document_id']}:{int(request['page_index'])}" for request in requests
    ]
    source_formula_ids = {
        str(formula_id)
        for values in source_manifest["expected_checked_formula_ids"].values()
        for formula_id in values
    }
    replay_formula_ids = {
        str(formula_id)
        for request in requests
        for formula_id in request["known_formula_ids"]
    }
    manifest = {
        "schema": "bemarkdown-external-vision-recognition-eval-bundle-v3",
        "bundle_name": target.name,
        "cohort": "REFERENCE",
        "status": "READY_FOR_EXTERNAL_REFERENCE_V1_REPLAY",
        "vision_split_sha256": source_manifest["vision_split_sha256"],
        "request_schema_sha256": request_schema_sha,
        "result_schema_sha256": _canonical_sha256(RESULT_CONTRACT_V1),
        "image_manifest_sha256": bundle_image_sha,
        "request_count": len(requests),
        "page_count": len(page_ids),
        "known_formula_count": sum(
            len(request["known_formula_ids"]) for request in requests
        ),
        "image_count": len(images),
        "page_ids": page_ids,
        "expected_checked_formula_ids": {
            request["audit_id"]: request["known_formula_ids"] for request in requests
        },
        "truth_bearing_fields_included": False,
        "provider_binding": None,
        "source_bundle_schema": source_manifest.get("schema"),
        "source_bundle_sha256": source_manifest.get("bundle_sha256"),
        "page_membership_unchanged": page_ids == source_manifest["page_ids"],
        "formula_membership_unchanged": replay_formula_ids == source_formula_ids,
        "validation_consumed": False,
    }
    manifest["bundle_sha256"] = _canonical_sha256(
        {
            "manifest": manifest,
            "requests": [_canonical_sha256(request) for request in requests],
            "images": images,
            "result_schema": RESULT_CONTRACT_V1,
        }
    )
    _write_json(target / "bundle_manifest.json", manifest)
    (target / "README.md").write_text(
        "# BeMarkdown Vision Recognition Reference Replay Bundle v3\n\n"
        "Process every JSON task under `requests/` and return strict JSON or JSONL "
        "matching `result_schema.json`. Every known formula requires one result row.\n",
        encoding="utf-8",
        newline="\n",
    )
    (target / "VISION_REFERENCE_EVAL_HANDOFF.md").write_text(
        "# Vision Reference Evaluation Handoff v1\n\n"
        "For every request, inspect every known formula ID and return exactly one "
        "result row for every ID; no formula may be omitted.\n\n"
        "For `PRESENT`, use `CONFIRMED_CURRENT` with null LaTeX when current LaTeX "
        "matches the image, `REPLACED` with non-empty source-faithful LaTeX when it "
        "does not, or `UNRESOLVED` with null LaTeX.\n\n"
        "For `MISSING`, use `RECOVERED_MISSING` with non-empty source-faithful LaTeX "
        "when the image can be transcribed, or `UNRESOLVED` with null LaTeX. Never "
        "use `CONFIRMED_CURRENT` for `MISSING`.\n\n"
        "Copy the four provenance SHA fields from each request and preserve the full "
        "`checked_formula_ids` set. Judge visible image evidence only.\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _validate_legacy_v0_structure(
    payloads: list[dict[str, Any]], requests: list[dict[str, Any]]
) -> None:
    indexed = _request_index(requests)
    if len(payloads) != len(indexed):
        raise ValueError(
            "Legacy Reference result must contain every audit exactly once"
        )
    seen_audits: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict) or payload.get("schema") != RESULT_SCHEMA_V0:
            raise ValueError("Legacy Reference result schema mismatch")
        audit_id = str(payload.get("audit_id", ""))
        if audit_id in seen_audits:
            raise ValueError("Legacy Reference result contains duplicate audits")
        seen_audits.add(audit_id)
        request = indexed.get(audit_id)
        if request is None:
            raise ValueError("Legacy Reference result contains an unknown audit")
        for field in (
            "vision_split_sha256",
            "request_schema_sha256",
            "bundle_image_manifest_sha256",
            "image_manifest_sha256",
        ):
            if payload.get(field) != request.get(field):
                raise ValueError(f"Legacy Reference result {field} mismatch")
        if payload.get("audit_complete") is not True:
            raise ValueError("Legacy Reference audit is incomplete")
        checked = payload.get("checked_formula_ids")
        expected = [str(value) for value in request["known_formula_ids"]]
        if (
            not isinstance(checked, list)
            or len(checked) != len(set(checked))
            or set(checked) != set(expected)
        ):
            raise ValueError("Legacy Reference checked formula IDs mismatch")
        explicit = payload.get("known_formula_results")
        if not isinstance(explicit, list):
            raise TypeError("Legacy Reference explicit results must be an array")
        explicit_ids = [str(row.get("formula_id", "")) for row in explicit]
        if len(explicit_ids) != len(set(explicit_ids)):
            raise ValueError(
                "Legacy Reference explicit results contain duplicate formula IDs"
            )
        if not set(explicit_ids) <= set(expected):
            raise ValueError(
                "Legacy Reference explicit results contain an unknown formula ID"
            )
        for row in explicit:
            decision = row.get("decision")
            latex = row.get("latex")
            if decision == "REPLACED":
                if not isinstance(latex, str) or not latex.strip():
                    raise ValueError(
                        "Legacy Reference replacement requires non-empty LaTeX"
                    )
            elif decision in {"CONFIRMED_CURRENT", "UNRESOLVED"}:
                if latex is not None:
                    raise ValueError(
                        "Legacy Reference non-replacement LaTeX must be null"
                    )
            else:
                raise ValueError("Legacy Reference result decision is unsupported")
        if not isinstance(payload.get("new_formula_proposals"), list):
            raise TypeError("Legacy Reference proposals must be an array")
    if seen_audits != set(indexed):
        raise ValueError("Legacy Reference result is missing an audit")


def legacy_sparse_reference_import_v0(
    payloads: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    *,
    external_result_sha256: str,
    cohort: str,
) -> list[dict[str, Any]]:
    """Normalize the one SHA-pinned sparse Reference result to explicit v1 rows."""

    if str(cohort).upper() != "REFERENCE":
        raise ValueError("Legacy sparse migration is Reference-only")
    if external_result_sha256.lower() != LEGACY_SPARSE_REFERENCE_RESULT_SHA256:
        raise ValueError("Legacy sparse Reference result SHA-256 is not approved")
    _validate_legacy_v0_structure(payloads, requests)
    requests_by_audit = _request_index(requests)
    output = []
    for payload in payloads:
        request = requests_by_audit[str(payload["audit_id"])]
        formula_by_id = {
            str(formula["formula_id"]): formula for formula in request["known_formulas"]
        }
        explicit = {
            str(row["formula_id"]): row for row in payload["known_formula_results"]
        }
        normalized_rows = []
        for formula_id in request["known_formula_ids"]:
            formula = formula_by_id[str(formula_id)]
            current_state = _current_latex_state(formula)
            source = explicit.get(str(formula_id))
            if source is None:
                decision = "CONFIRMED_CURRENT"
                latex = None
                provenance = {
                    "migration_reason": "LEGACY_SPARSE_IMPLICIT_CONFIRM",
                    "source_result_schema": RESULT_SCHEMA_V0,
                }
            else:
                decision = str(source["decision"])
                latex = source.get("latex")
                if decision == "REPLACED" and current_state == "MISSING":
                    decision = "RECOVERED_MISSING"
                provenance = {"source_result_schema": RESULT_SCHEMA_V0}
            normalized_rows.append(
                {
                    "formula_id": str(formula_id),
                    "current_latex_state": current_state,
                    "decision": decision,
                    "latex": latex,
                    "provenance": provenance,
                }
            )
        output.append(
            {
                **payload,
                "schema": RESULT_SCHEMA_V1,
                "source_result_schema": RESULT_SCHEMA_V0,
                "migration_reason": "LEGACY_SPARSE_IMPLICIT_CONFIRM",
                "known_formula_results": normalized_rows,
            }
        )
    return output


class VisionFormulaAuditResultImporterV1:
    """Validate the explicit-row v1 result contract behind one fail-closed seam."""

    def __init__(
        self,
        expected_requests: list[dict[str, Any]],
        *,
        cohort: str = "REFERENCE",
        render_validator=None,
        validation: bool = False,
        truth_sidecar: str | Path | None = None,
        consumed_once_marker: str | Path | None = None,
    ):
        self._requests = _request_index(expected_requests)
        self._cohort = str(cohort).upper()
        if self._cohort not in {"REFERENCE", "VALIDATION"}:
            raise ValueError("Importer cohort must be REFERENCE or VALIDATION")
        self._validation = bool(validation)
        self._consumed_once_marker = (
            Path(consumed_once_marker) if consumed_once_marker is not None else None
        )
        if self._cohort == "VALIDATION" and not self._validation:
            raise ValueError("Validation import requires explicit validation mode")
        if self._cohort == "VALIDATION" and self._consumed_once_marker is None:
            raise ValueError("Validation import requires a consumed-once marker")
        if self._consumed_once_marker is not None and self._consumed_once_marker.exists():
            raise ValueError("Validation results were already consumed once")
        if truth_sidecar is not None:
            self._validate_truth_sidecar(Path(truth_sidecar))
        self._imported_audit_ids: set[str] = set()
        self._syntax_validator = FormulaOcrOutputValidator()
        self._render_validator = render_validator or (
            lambda latex: bool(isinstance(latex, str) and latex.strip())
        )

    def import_payloads(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        imported = []
        for payload in payloads:
            if not isinstance(payload, dict):
                return self._rejected("INVALID_RESULT_OBJECT")
            if payload.get("schema") != RESULT_SCHEMA_V1:
                return self._rejected("RESULT_SCHEMA_MISMATCH")
            audit_id = str(payload.get("audit_id", ""))
            if audit_id in seen or audit_id in self._imported_audit_ids:
                return self._rejected("DUPLICATE_RESPONSE")
            seen.add(audit_id)
            request = self._requests.get(audit_id)
            if request is None:
                return self._rejected("UNKNOWN_AUDIT_ID")
            if payload.get("audit_complete") is not True:
                return self._rejected("AUDIT_INCOMPLETE")
            for field, error_code in (
                ("vision_split_sha256", "SPLIT_SHA_MISMATCH"),
                ("request_schema_sha256", "REQUEST_SCHEMA_SHA_MISMATCH"),
                ("bundle_image_manifest_sha256", "BUNDLE_IMAGE_SHA_MISMATCH"),
                ("image_manifest_sha256", "IMAGE_SHA_MISMATCH"),
            ):
                if payload.get(field) != request.get(field):
                    return self._rejected(error_code)
            known_ids = [str(value) for value in request["known_formula_ids"]]
            checked = payload.get("checked_formula_ids")
            if (
                not isinstance(checked, list)
                or len(checked) != len(set(checked))
                or set(checked) != set(known_ids)
            ):
                return self._rejected("AUDIT_INCOMPLETE")
            rows = payload.get("known_formula_results")
            if not isinstance(rows, list):
                return self._rejected("AUDIT_INCOMPLETE")
            row_ids = [str(row.get("formula_id", "")) for row in rows]
            if len(row_ids) != len(set(row_ids)):
                return self._rejected("DUPLICATE_RESULT_ROW")
            if set(row_ids) - set(known_ids):
                return self._rejected("UNKNOWN_FORMULA_ID")
            if set(row_ids) != set(known_ids):
                return self._rejected("AUDIT_INCOMPLETE")
            formulas = {
                str(formula["formula_id"]): formula
                for formula in request["known_formulas"]
            }
            for row in rows:
                error = self._validate_result_row(row, formulas[str(row["formula_id"])])
                if error is not None:
                    return self._rejected(error)
            if not isinstance(payload.get("new_formula_proposals"), list):
                return self._rejected("INVALID_DISCOVERY_PROPOSALS")
            material = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            imported.append(
                {**payload, "raw_response_sha256": hashlib.sha256(material).hexdigest()}
            )
        if self._cohort == "VALIDATION" and imported:
            self._write_consumed_once_marker(imported)
        self._imported_audit_ids.update(seen)
        return {
            "schema": "bemarkdown-vision-formula-result-import-v1",
            "status": "VISION_RESULTS_IMPORTED",
            "response_count": len(imported),
            "responses": imported,
            "error_codes": [],
        }

    def _validate_result_row(
        self,
        row: dict[str, Any], formula: dict[str, Any]
    ) -> str | None:
        expected_state = _current_latex_state(formula)
        if row.get("current_latex_state") != expected_state:
            return "CURRENT_LATEX_STATE_MISMATCH"
        decision = row.get("decision")
        allowed = (
            {"CONFIRMED_CURRENT", "REPLACED", "UNRESOLVED"}
            if expected_state == "PRESENT"
            else {"RECOVERED_MISSING", "UNRESOLVED"}
        )
        if decision not in allowed:
            return "DECISION_STATE_MISMATCH"
        latex = row.get("latex")
        if decision in {"REPLACED", "RECOVERED_MISSING"}:
            if not isinstance(latex, str) or not latex.strip():
                return "LATEX_REQUIRED"
            syntax = self._syntax_validator.validate(latex)
            if (
                syntax.verdict not in {OcrVerdict.VALID, OcrVerdict.VALID_WITH_WARNING}
                or not self._render_validator(latex)
                or "```" in latex
                or "formula-review" in latex
            ):
                return "LATEX_MACHINE_VALIDATION_FAILED"
        elif latex is not None:
            return "LATEX_FORBIDDEN"
        return None

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

    def _write_consumed_once_marker(self, imported: list[dict[str, Any]]) -> None:
        assert self._consumed_once_marker is not None
        marker = {
            "schema": "bemarkdown-vision-validation-consumed-once-v1",
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
            "schema": "bemarkdown-vision-formula-result-import-v1",
            "status": "VISION_RESULT_IMPORT_REJECTED",
            "response_count": 0,
            "responses": [],
            "error_codes": [error_code],
        }


def evaluate_reference_results(
    normalized_payloads: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    reference_truth_rows: list[dict[str, Any]],
    *,
    replacement_quality_reviews: dict[str, dict[str, Any]],
    render_validator,
    evaluation_role: str = "UNKNOWN",
    pass_a_context: dict[str, Any] | None = None,
    pass_b_context: dict[str, Any] | None = None,
    pass_b_required: bool = False,
    truth_loaded_before_final_resolution: bool = False,
) -> dict[str, Any]:
    """Score only frozen certain Reference truth and keep all other rows diagnostic."""

    assert_quality_gate_eligible(
        evaluation_role=evaluation_role,
        pass_a_context=pass_a_context,
        pass_b_context=pass_b_context,
        pass_b_required=pass_b_required,
        truth_loaded_before_final_resolution=truth_loaded_before_final_resolution,
    )

    if not reference_truth_rows or {
        str(row.get("evaluator_cohort", "")).upper() for row in reference_truth_rows
    } != {"REFERENCE"}:
        raise ValueError("Reference scoring accepts Reference truth only")
    truth_by_id = {str(row["content_id"]): row for row in reference_truth_rows}
    if len(truth_by_id) != len(reference_truth_rows):
        raise ValueError("Reference truth contains duplicate formula IDs")
    request_by_audit = _request_index(requests)
    formulas: dict[str, dict[str, Any]] = {}
    for request in requests:
        for formula in request["known_formulas"]:
            formula_id = str(formula["formula_id"])
            if formula_id in formulas:
                raise ValueError("Reference requests contain duplicate formula IDs")
            formulas[formula_id] = formula
    decisions: dict[str, dict[str, Any]] = {}
    for payload in normalized_payloads:
        audit_id = str(payload.get("audit_id", ""))
        if audit_id not in request_by_audit:
            raise ValueError("Normalized Reference result contains an unknown audit")
        for row in payload.get("known_formula_results", []):
            formula_id = str(row["formula_id"])
            if formula_id in decisions:
                raise ValueError(
                    "Normalized Reference result contains duplicate formula IDs"
                )
            decisions[formula_id] = row
    if set(decisions) != set(formulas):
        raise ValueError("Normalized Reference result must cover every known formula")

    acceptable_labels = {"CORRECT", "MINOR_ERROR"}
    material_labels = {"MAJOR_ERROR", "UNUSABLE"}
    allowed_final_labels = acceptable_labels | material_labels
    rows = []
    for formula_id, formula in formulas.items():
        decision = decisions[formula_id]
        truth = truth_by_id.get(formula_id)
        current = formula.get("current_latex")
        action = str(decision["decision"])
        final_latex = (
            current
            if action == "CONFIRMED_CURRENT"
            else decision.get("latex")
            if action in {"REPLACED", "RECOVERED_MISSING"}
            else None
        )
        original_quality = truth.get("formulanet_label") if truth else None
        if truth is None:
            final_quality = "NO_FROZEN_ACCURACY_TRUTH"
        elif original_quality == "REFERENCE_UNCERTAIN":
            final_quality = "REFERENCE_UNCERTAIN"
        elif action == "UNRESOLVED":
            final_quality = "UNRESOLVED"
        elif action == "CONFIRMED_CURRENT":
            final_quality = original_quality
        else:
            review = replacement_quality_reviews.get(formula_id)
            if review is None:
                raise ValueError(
                    f"Changed frozen-truth formula requires a visual quality review: {formula_id}"
                )
            final_quality = review.get("final_quality")
            if final_quality not in allowed_final_labels:
                raise ValueError(
                    "Replacement visual review has an unsupported final quality"
                )
        rows.append(
            {
                "formula_id": formula_id,
                "reference_id": truth.get("reference_id") if truth else None,
                "current_latex_state": _current_latex_state(formula),
                "current_latex": current,
                "vision_decision": action,
                "vision_final_latex": final_latex,
                "truth_label": original_quality,
                "final_quality": final_quality,
                "replacement_quality_review": replacement_quality_reviews.get(
                    formula_id
                ),
            }
        )

    certain = [
        row for row in rows if row["truth_label"] in acceptable_labels | material_labels
    ]
    material = [row for row in certain if row["truth_label"] in material_labels]
    acceptable = [row for row in certain if row["truth_label"] in acceptable_labels]
    uncertain = [row for row in rows if row["truth_label"] == "REFERENCE_UNCERTAIN"]
    no_truth = [row for row in rows if row["truth_label"] is None]
    material_final_acceptable = sum(
        row["final_quality"] in acceptable_labels for row in material
    )
    acceptable_material_damage = sum(
        row["final_quality"] in material_labels for row in acceptable
    )
    acceptable_unresolved = sum(
        row["final_quality"] == "UNRESOLVED" for row in acceptable
    )
    gate_passed = (
        material_final_acceptable == len(material)
        and acceptable_material_damage <= 1
        and acceptable_unresolved <= 1
    )
    accuracy_metrics = {
        "schema": "bemarkdown-vision-reference-accuracy-metrics-v1",
        "truth_cohort_consumed": "REFERENCE_ONLY",
        "accuracy_denominator": len(certain),
        "uncertain_excluded": len(uncertain),
        "no_truth_rows_excluded": len(no_truth),
        "reference_material_total": len(material),
        "reference_material_final_acceptable": material_final_acceptable,
        "reference_material_final_material": sum(
            row["final_quality"] in material_labels for row in material
        ),
        "reference_material_unresolved": sum(
            row["final_quality"] == "UNRESOLVED" for row in material
        ),
        "reference_acceptable_total": len(acceptable),
        "reference_acceptable_final_acceptable": sum(
            row["final_quality"] in acceptable_labels for row in acceptable
        ),
        "reference_acceptable_materially_damaged": acceptable_material_damage,
        "reference_acceptable_unresolved": acceptable_unresolved,
        "unnecessary_human_fallback": acceptable_unresolved,
        "gate_passed": gate_passed,
        "status": (
            "VISION_REFERENCE_QUALITY_SIGNAL_PASSED"
            if gate_passed
            else "VISION_REFERENCE_QUALITY_SIGNAL_FAILED"
        ),
    }

    validator = FormulaOcrOutputValidator()
    missing_rows = [row for row in rows if row["current_latex_state"] == "MISSING"]
    recovered = [
        row for row in missing_rows if row["vision_decision"] == "RECOVERED_MISSING"
    ]
    recovery_validations = []
    for row in recovered:
        latex = row["vision_final_latex"]
        syntax = validator.validate(latex)
        project_format = bool(
            latex and "```" not in latex and "formula-review" not in latex
        )
        renderable = bool(latex and render_validator(latex))
        recovery_validations.append(
            {
                "formula_id": row["formula_id"],
                "syntax": syntax.to_dict(),
                "parseable": syntax.verdict
                in {OcrVerdict.VALID, OcrVerdict.VALID_WITH_WARNING},
                "renderable": renderable,
                "project_format": project_format,
            }
        )
    parseable = sum(row["parseable"] for row in recovery_validations)
    renderable = sum(row["renderable"] for row in recovery_validations)
    missing_metrics = {
        "schema": "bemarkdown-vision-reference-missing-current-metrics-v1",
        "accuracy_claim": False,
        "missing_total": len(missing_rows),
        "recovered_missing": len(recovered),
        "unresolved": sum(
            row["vision_decision"] == "UNRESOLVED" for row in missing_rows
        ),
        "parseable": parseable,
        "parseable_rate": parseable / len(recovered) if recovered else None,
        "renderable": renderable,
        "renderable_rate": renderable / len(recovered) if recovered else None,
        "validations": recovery_validations,
    }
    distribution = {
        "schema": "bemarkdown-vision-reference-result-distribution-v1",
        "known_formula_total": len(rows),
        "current_latex_state_counts": dict(
            sorted(Counter(row["current_latex_state"] for row in rows).items())
        ),
        "decision_counts": dict(
            sorted(Counter(row["vision_decision"] for row in rows).items())
        ),
        "frozen_truth_rows": len(reference_truth_rows),
        "certain_truth_rows": len(certain),
        "rows": rows,
    }
    return {
        "accuracy_metrics": accuracy_metrics,
        "material_cases": [row for row in material],
        "acceptable_damage_cases": [
            row for row in acceptable if row["final_quality"] in material_labels
        ],
        "missing_current_metrics": missing_metrics,
        "distribution": distribution,
    }
