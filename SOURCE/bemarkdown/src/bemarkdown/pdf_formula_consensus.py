from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

FORMULA_CONSENSUS_VERIFICATION_SCHEMA = (
    "bemarkdown-formula-consensus-verification-ir-v0"
)
FORMULA_CONSENSUS_VERIFIER_VERSION = "formula-consensus-verifier-v0"
FORMULA_MULTIVIEW_SCHEMA = "bemarkdown-formula-multiview-ir-v0"
FORMULA_OCR_CROSSCHECK_SCHEMA = "bemarkdown-formula-ocr-crosscheck-ir-v0"

_ACCEPTING_GATE_VERDICTS = {"ACCEPT", "ACCEPT_WITH_WARNING"}
_REVIEW_GATE_VERDICTS = {
    "REVIEW_REQUIRED",
    "REJECT_PRESERVE_IMAGE",
    "INFERENCE_FAILED",
}


def build_deterministic_formula_views(
    source_crop_ref: str | Path,
    output_dir: str | Path,
    *,
    max_views: int = 4,
    include_high_resolution: bool = True,
) -> list[dict[str, Any]]:
    """Build V0 plus at most three deterministic, semantics-neutral image views."""

    if not 1 <= max_views <= 4:
        raise ValueError("Formula multi-view count must be between 1 and 4")
    source_path = Path(source_crop_ref).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")

    candidates: list[tuple[str, Image.Image, str]] = []
    background = _corner_background(source)
    candidates.append(
        (
            "V1",
            ImageOps.expand(source, border=8, fill=background),
            "DETERMINISTIC_PADDING_8PX",
        )
    )
    foreground = _foreground_bbox(source)
    normalized = source.crop(tuple(foreground))
    normalized = ImageOps.expand(normalized, border=8, fill=background)
    candidates.append(("V2", normalized, "FOREGROUND_TRIM_PLUS_PADDING_8PX"))
    if include_high_resolution:
        high_resolution = normalized.resize(
            (max(1, normalized.width * 2), max(1, normalized.height * 2)),
            Image.Resampling.LANCZOS,
        )
        candidates.append(("V3", high_resolution, "DETERMINISTIC_2X_RERENDER_RECROP"))

    rows = [_view_record("V0", source_path, "ORIGINAL_SOURCE_CROP")]
    for view_id, image, transform in candidates[: max_views - 1]:
        path = output / f"{view_id.lower()}.png"
        image.save(path, format="PNG", optimize=False)
        rows.append(_view_record(view_id, path, transform))
    return rows


def analyze_multiview_outputs(
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare raw outputs without semantic-equivalence rewriting."""

    if not outputs or len(outputs) > 4:
        raise ValueError("Formula multi-view output count must be between 1 and 4")
    normalized = [
        _normalize_latex_for_comparison(row.get("raw_latex")) for row in outputs
    ]
    failures = [
        row["view_id"]
        for row, value in zip(outputs, normalized, strict=True)
        if row.get("error") or value is None
    ]
    distances = []
    for left_index, left in enumerate(normalized):
        for right_index in range(left_index + 1, len(normalized)):
            right = normalized[right_index]
            distance = None
            if left is not None and right is not None:
                distance = _normalized_edit_distance(left, right)
            distances.append(
                {
                    "left": outputs[left_index]["view_id"],
                    "right": outputs[right_index]["view_id"],
                    "normalized_edit_distance": distance,
                }
            )
    numeric_distances = [
        row["normalized_edit_distance"]
        for row in distances
        if row["normalized_edit_distance"] is not None
    ]
    available = [value for value in normalized if value is not None]
    counts = Counter(available)
    majority_value, majority_count = counts.most_common(1)[0] if counts else (None, 0)
    exact = bool(available) and len(available) == len(outputs) and len(counts) == 1
    status_values = {
        str(row.get("safety_status"))
        for row in outputs
        if row.get("safety_status") is not None
    }
    status_disagreement = len(status_values) > 1
    max_distance = max(numeric_distances, default=None)
    reasons: list[str] = []
    if failures:
        stability = "MULTIVIEW_STRONG_DISAGREEMENT"
        reasons.append("MULTIVIEW_VIEW_FAILURE")
    elif exact and not status_disagreement:
        stability = "MULTIVIEW_STABLE"
    elif status_disagreement or (max_distance is not None and max_distance >= 0.35):
        stability = "MULTIVIEW_STRONG_DISAGREEMENT"
        reasons.append(
            "MULTIVIEW_STATUS_DISAGREEMENT"
            if status_disagreement
            else "MULTIVIEW_LARGE_EDIT_DISTANCE"
        )
    else:
        stability = "MULTIVIEW_WEAK_DISAGREEMENT"
        reasons.append("MULTIVIEW_NONEXACT_OUTPUT")
    return {
        "schema": FORMULA_MULTIVIEW_SCHEMA,
        "stability_status": stability,
        "view_count": len(outputs),
        "extra_formula_calls": max(0, len(outputs) - 1),
        "exact_consensus": exact,
        "majority_consensus": majority_count > len(outputs) / 2,
        "majority_count": majority_count,
        "majority_normalized_latex": majority_value,
        "edit_distance_to_v0": [
            row for row in distances if row["left"] == outputs[0]["view_id"]
        ],
        "pairwise_edit_distances": distances,
        "max_normalized_edit_distance": max_distance,
        "status_disagreement": status_disagreement,
        "failed_view_ids": failures,
        "evidence_unavailable": bool(failures),
        "reason_codes": sorted(reasons),
        "outputs": outputs,
        "comparison_normalization": "OUTER_WHITESPACE_AND_LINE_ENDINGS_ONLY",
    }


def classify_formula_ocr_crosscheck(
    source_lines: list[dict[str, Any]],
    render_lines: list[dict[str, Any]],
    *,
    pipeline_error: str | None = None,
    low_confidence_threshold: float = 0.45,
    strong_confidence_threshold: float = 0.65,
    contradiction_distance_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare OCR-on-source and OCR-on-render as weak independent evidence."""

    source_raw = "\n".join(str(row.get("text") or "") for row in source_lines)
    render_raw = "\n".join(str(row.get("text") or "") for row in render_lines)
    source_normalized = _normalize_ocr_text(source_raw)
    render_normalized = _normalize_ocr_text(render_raw)
    source_confidence = _mean_confidence(source_lines)
    render_confidence = _mean_confidence(render_lines)
    distance = (
        _normalized_edit_distance(source_normalized, render_normalized)
        if source_normalized and render_normalized
        else None
    )
    reasons: list[str] = []
    if pipeline_error:
        status = "EVIDENCE_UNAVAILABLE"
        reasons.append("FORMULA_OCR_CROSSCHECK_FAILURE")
    elif not source_normalized or not render_normalized:
        status = "EVIDENCE_UNAVAILABLE"
        reasons.append("FORMULA_OCR_EMPTY_EVIDENCE")
    elif min(source_confidence or 0.0, render_confidence or 0.0) < low_confidence_threshold:
        status = "WEAK_EVIDENCE"
        reasons.append("FORMULA_OCR_LOW_CONFIDENCE")
    elif source_normalized == render_normalized:
        status = "SUPPORTING_AGREEMENT"
    elif (
        min(source_confidence or 0.0, render_confidence or 0.0)
        >= strong_confidence_threshold
        and distance is not None
        and distance >= contradiction_distance_threshold
    ):
        status = "STRONG_CONTRADICTION"
        reasons.append("FORMULA_OCR_STRONG_CONTRADICTION")
    else:
        status = "WEAK_EVIDENCE"
        reasons.append("FORMULA_OCR_NONEXACT_WEAK_EVIDENCE")
    return {
        "schema": FORMULA_OCR_CROSSCHECK_SCHEMA,
        "pipeline": "FORMULA_OCR_CROSSCHECK",
        "evidence_status": status,
        "source_ocr_raw": source_raw,
        "source_ocr_confidence": source_confidence,
        "render_ocr_raw": render_raw,
        "render_ocr_confidence": render_confidence,
        "source_normalized": source_normalized,
        "render_normalized": render_normalized,
        "normalized_edit_distance": distance,
        "reason_codes": sorted(reasons),
        "pipeline_error": pipeline_error,
        "pipeline_failure": pipeline_error is not None,
        "latex_repair_used": False,
    }


class FormulaConsensusVerifier:
    """Fuse independent evidence conservatively without changing raw LaTeX."""

    def verify(
        self,
        *,
        content_id: str,
        raw_latex: str,
        existing_gate: dict[str, Any],
        existing_status: str,
        multiview: dict[str, Any],
        ocr_crosscheck: dict[str, Any],
        visual_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        raw_sha = hashlib.sha256(raw_latex.encode("utf-8")).hexdigest()
        identity = {
            "content_id": content_id,
            "raw_latex_sha256": raw_sha,
            "policy_version": FORMULA_CONSENSUS_VERIFIER_VERSION,
        }
        reasons: set[str] = set()
        gate_verdict = str(existing_gate.get("verdict") or "UNKNOWN")
        risk_score = 0.0
        unavailable = bool(multiview.get("evidence_unavailable"))
        unavailable |= bool(ocr_crosscheck.get("pipeline_failure"))
        unavailable |= visual_evidence.get("status") == "VERIFICATION_UNAVAILABLE"

        if unavailable:
            consensus_status = "CONSENSUS_UNAVAILABLE"
            reasons.add("CONSENSUS_PIPELINE_UNAVAILABLE")
            risk_score = 1.0
        elif (
            gate_verdict in _REVIEW_GATE_VERDICTS
            or gate_verdict not in _ACCEPTING_GATE_VERDICTS
            or existing_status not in {"SUCCESS", "SUCCESS_WITH_WARNING"}
        ):
            consensus_status = "CONSENSUS_REVIEW"
            reasons.add("EXISTING_GATE_REVIEW_OR_REJECT")
            risk_score = 1.0
        else:
            multiview_status = multiview.get("stability_status")
            ocr_status = ocr_crosscheck.get("evidence_status")
            visual_status = visual_evidence.get("status")
            visual_features = visual_evidence.get("features") or {}
            strong = False
            if multiview_status == "MULTIVIEW_STRONG_DISAGREEMENT":
                risk_score += 0.7
                reasons.add("MULTIVIEW_STRONG_DISAGREEMENT")
                strong = True
            elif multiview_status == "MULTIVIEW_WEAK_DISAGREEMENT":
                risk_score += 0.16
                reasons.add("MULTIVIEW_WEAK_DISAGREEMENT")
            if ocr_status == "STRONG_CONTRADICTION":
                risk_score += 0.7
                reasons.add("FORMULA_OCR_STRONG_CONTRADICTION")
                strong = True
            elif ocr_status in {"WEAK_EVIDENCE", "EVIDENCE_UNAVAILABLE"}:
                risk_score += 0.05
                reasons.add(f"FORMULA_OCR_{ocr_status}")
            if visual_status == "VISUAL_REVIEW":
                risk_score += 0.24
                reasons.add("AUXILIARY_VISUAL_REVIEW")
            elif visual_status == "VISUAL_WARNING":
                risk_score += 0.10
                reasons.add("AUXILIARY_VISUAL_WARNING")
            local_mismatch = float(
                visual_features.get("local_window_foreground_mismatch") or 0.0
            )
            localized_projection = float(
                visual_features.get("localized_projection_mismatch") or 0.0
            )
            if local_mismatch >= 0.62 or localized_projection >= 0.62:
                risk_score += 0.3
                reasons.add("LOCALIZED_VISUAL_CONTRADICTION")
            if gate_verdict == "ACCEPT_WITH_WARNING":
                risk_score += 0.2
                reasons.add("EXISTING_GATE_WARNING")

            if strong or risk_score >= 0.62:
                consensus_status = "CONSENSUS_REVIEW"
            elif risk_score > 0:
                consensus_status = "CONSENSUS_WARNING"
            else:
                consensus_status = "CONSENSUS_PASS"

        reasons.update(multiview.get("reason_codes") or [])
        reasons.update(ocr_crosscheck.get("reason_codes") or [])
        if consensus_status in {"CONSENSUS_REVIEW", "CONSENSUS_UNAVAILABLE"}:
            reasons.update(visual_evidence.get("reason_codes") or [])
        verification_id = "formula-consensus-" + hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()[:20]
        return {
            "schema": FORMULA_CONSENSUS_VERIFICATION_SCHEMA,
            "verification_id": verification_id,
            "content_id": content_id,
            "existing_gate": existing_gate,
            "multiview": multiview,
            "ocr_crosscheck": ocr_crosscheck,
            "visual_evidence": visual_evidence,
            "consensus_status": consensus_status,
            "risk_score": round(min(1.0, risk_score), 8),
            "reason_codes": sorted(reasons),
            "raw_latex_sha256": raw_sha,
            "policy_version": FORMULA_CONSENSUS_VERIFIER_VERSION,
            "provenance": {
                "raw_latex_rewritten": False,
                "risk_direction": "ESCALATE_ONLY",
                "fail_closed": consensus_status == "CONSENSUS_UNAVAILABLE",
                "semantic_repair_used": False,
            },
        }


class FormulaConsensusPipeline:
    """Bounded runtime orchestration for E1/E2/E3 and conservative fusion."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        renderer: Any,
        max_views: int = 4,
        verifier: FormulaConsensusVerifier | None = None,
    ):
        if not 1 <= max_views <= 4:
            raise ValueError("Formula consensus max_views must be between 1 and 4")
        self.output_dir = Path(output_dir).resolve()
        self.renderer = renderer
        self.max_views = max_views
        self.verifier = verifier or FormulaConsensusVerifier()

    def verify(
        self,
        *,
        content_id: str,
        raw_latex: str,
        source_crop_ref: str | Path,
        source_bbox_pdf_pt: list[float],
        existing_gate: dict[str, Any],
        existing_status: str,
        visual_evidence: dict[str, Any],
        formula_predictor: Any,
        ocr_runtime_factory: Any,
    ) -> dict[str, Any]:
        raw_sha = hashlib.sha256(raw_latex.encode("utf-8")).hexdigest()
        gate_verdict = str(existing_gate.get("verdict") or "UNKNOWN")
        try:
            view_dir = self.output_dir / hashlib.sha256(
                content_id.encode("utf-8")
            ).hexdigest()[:20]
            views = build_deterministic_formula_views(
                source_crop_ref,
                view_dir,
                max_views=self.max_views if gate_verdict in _ACCEPTING_GATE_VERDICTS else 1,
            )
            outputs = [
                {
                    "view_id": "V0",
                    "path": views[0]["path"],
                    "raw_latex": raw_latex,
                    "validator_status": str(
                        existing_gate.get("validator_status") or "EXISTING_VALIDATOR"
                    ),
                    "safety_status": gate_verdict,
                    "renderability": gate_verdict in _ACCEPTING_GATE_VERDICTS,
                    "error": None,
                }
            ]
            if len(views) > 1:
                outputs.extend(
                    self._predict_additional_views(views[1:], formula_predictor)
                )
            multiview = analyze_multiview_outputs(outputs)
            rendered = self.renderer(content_id=content_id, raw_latex=raw_latex)
            render_path = Path(rendered["path"]).resolve()
            ocr_runtime = ocr_runtime_factory()
            source_crop = _ocr_crop_record(
                Path(source_crop_ref).resolve(), source_bbox_pdf_pt
            )
            render_crop = _ocr_crop_record(render_path, source_bbox_pdf_pt)
            source_lines = ocr_runtime.recognize_direct(source_crop)
            render_lines = ocr_runtime.recognize_direct(render_crop)
            ocr_crosscheck = classify_formula_ocr_crosscheck(
                source_lines, render_lines
            )
            result = self.verifier.verify(
                content_id=content_id,
                raw_latex=raw_latex,
                existing_gate=existing_gate,
                existing_status=existing_status,
                multiview=multiview,
                ocr_crosscheck=ocr_crosscheck,
                visual_evidence=visual_evidence,
            )
            result["provenance"].update(
                {
                    "source_crop_retained": True,
                    "view_count": len(views),
                    "extra_formula_calls": max(0, len(views) - 1),
                    "ocr_crosscheck_calls": 2,
                    "rendered_formula_ref": str(render_path),
                    "raw_latex_sha256_before": raw_sha,
                    "raw_latex_sha256_after": raw_sha,
                }
            )
            return result
        except Exception as exc:  # noqa: BLE001 - every verifier failure is Review
            multiview = {
                "schema": FORMULA_MULTIVIEW_SCHEMA,
                "stability_status": "MULTIVIEW_STRONG_DISAGREEMENT",
                "evidence_unavailable": True,
                "outputs": [
                    {
                        "view_id": "V0",
                        "path": str(Path(source_crop_ref).resolve()),
                        "raw_latex": raw_latex,
                        "safety_status": gate_verdict,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ],
                "reason_codes": ["MULTIVIEW_PIPELINE_FAILURE"],
            }
            result = self.verifier.verify(
                content_id=content_id,
                raw_latex=raw_latex,
                existing_gate=existing_gate,
                existing_status=existing_status,
                multiview=multiview,
                ocr_crosscheck=classify_formula_ocr_crosscheck(
                    [], [], pipeline_error=f"{type(exc).__name__}: {exc}"
                ),
                visual_evidence={
                    **visual_evidence,
                    "status": "VERIFICATION_UNAVAILABLE",
                    "reason_codes": sorted(
                        set(visual_evidence.get("reason_codes") or [])
                        | {"CONSENSUS_PIPELINE_FAILURE"}
                    ),
                },
            )
            result["provenance"].update(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_crop_retained": True,
                    "raw_latex_sha256_before": raw_sha,
                    "raw_latex_sha256_after": raw_sha,
                }
            )
            return result

    @staticmethod
    def _predict_additional_views(
        views: list[dict[str, Any]], formula_predictor: Any
    ) -> list[dict[str, Any]]:
        from .formula_ocr import FormulaOcrSafetyGate
        from .formulanet_runtime import FormulaOcrOutputValidator

        paths = [Path(row["path"]) for row in views]
        try:
            predictions = formula_predictor(paths)
            if len(predictions) != len(views):
                raise RuntimeError("Formula predictor output count mismatch")
        except Exception as exc:  # noqa: BLE001 - retain each failed view
            return [
                {
                    "view_id": row["view_id"],
                    "path": row["path"],
                    "raw_latex": None,
                    "validator_status": "INFERENCE_FAILED",
                    "safety_status": "INFERENCE_FAILED",
                    "renderability": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                for row in views
            ]
        validator = FormulaOcrOutputValidator()
        safety_gate = FormulaOcrSafetyGate()
        outputs = []
        for view, prediction in zip(views, predictions, strict=True):
            validation = validator.validate(prediction)
            decision = safety_gate.evaluate(
                width=int(view["width"]),
                height=int(view["height"]),
                raw_latex=prediction,
                validation=validation,
            )
            outputs.append(
                {
                    "view_id": view["view_id"],
                    "path": view["path"],
                    "raw_latex": prediction,
                    "validator_status": validation.verdict.value,
                    "validator_issues": list(validation.issues),
                    "safety_status": decision.verdict.value,
                    "safety_reasons": list(decision.reasons),
                    "renderability": validation.verdict.value
                    not in {"OCR_EMPTY", "OCR_INVALID", "OCR_INFERENCE_FAILED"},
                    "error": None,
                }
            )
        return outputs


def apply_formula_consensus_gate(
    *,
    existing_status: str,
    existing_quality_status: str,
    raw_latex: str,
    verification: dict[str, Any],
) -> dict[str, Any]:
    consensus_status = verification.get("consensus_status")
    if existing_status in {"FAILED_PRESERVE_INPUT", "DEFERRED"}:
        status = existing_status
        quality = existing_quality_status
    elif existing_status == "REVIEW_REQUIRED":
        status = existing_status
        quality = "FORMULA_CONTENT_REVIEW"
    elif consensus_status == "CONSENSUS_PASS":
        status = existing_status
        quality = existing_quality_status
    elif consensus_status == "CONSENSUS_WARNING":
        status = "SUCCESS_WITH_WARNING"
        quality = "FORMULA_CONTENT_WARNING"
    else:
        status = "REVIEW_REQUIRED"
        quality = "FORMULA_CONTENT_REVIEW"
    return {
        "status": status,
        "quality_status": quality,
        "raw_latex": raw_latex,
        "review_reasons": (
            []
            if consensus_status in {"CONSENSUS_PASS", "CONSENSUS_WARNING"}
            else sorted(
                set(verification.get("reason_codes") or [])
                | {"FORMULA_CONSENSUS_VERIFICATION_REVIEW"}
            )
        ),
        "warnings": (
            sorted(verification.get("reason_codes") or [])
            if consensus_status == "CONSENSUS_WARNING"
            else []
        ),
    }


def _view_record(view_id: str, path: Path, transform: str) -> dict[str, Any]:
    resolved = path.resolve()
    data = resolved.read_bytes()
    with Image.open(resolved) as image:
        width, height = image.size
    return {
        "view_id": view_id,
        "path": str(resolved),
        "content_sha256": hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
        "transform": transform,
        "deterministic": True,
    }


def _ocr_crop_record(path: Path, bbox_pdf_pt: list[float]) -> dict[str, Any]:
    data = path.read_bytes()
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "dpi": 200,
        "bbox_pdf_pt": list(bbox_pdf_pt),
        "scale_x": width
        / max(1e-9, float(bbox_pdf_pt[2]) - float(bbox_pdf_pt[0])),
        "scale_y": height
        / max(1e-9, float(bbox_pdf_pt[3]) - float(bbox_pdf_pt[1])),
        "content_sha256": hashlib.sha256(data).hexdigest(),
    }


def _corner_background(image: Image.Image) -> tuple[int, int, int]:
    corners = [
        image.getpixel((0, 0)),
        image.getpixel((image.width - 1, 0)),
        image.getpixel((0, image.height - 1)),
        image.getpixel((image.width - 1, image.height - 1)),
    ]
    return tuple(sorted(int(pixel[index]) for pixel in corners)[2] for index in range(3))


def _foreground_bbox(image: Image.Image) -> list[int]:
    gray = ImageOps.grayscale(image)
    background = sorted(
        [
            gray.getpixel((0, 0)),
            gray.getpixel((gray.width - 1, 0)),
            gray.getpixel((0, gray.height - 1)),
            gray.getpixel((gray.width - 1, gray.height - 1)),
        ]
    )[2]
    threshold = max(32, background - 32)
    coordinates = [
        (x, y)
        for y in range(gray.height)
        for x in range(gray.width)
        if gray.getpixel((x, y)) < threshold
    ]
    if not coordinates:
        raise ValueError("Formula crop has no deterministic foreground")
    return [
        min(x for x, _ in coordinates),
        min(y for _, y in coordinates),
        max(x for x, _ in coordinates) + 1,
        max(y for _, y in coordinates) + 1,
    ]


def _normalize_latex_for_comparison(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_ocr_text(value: str) -> str:
    return "".join(
        unicodedata.normalize("NFKC", value)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split()
    )


def _mean_confidence(lines: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["confidence"])
        for row in lines
        if row.get("confidence") is not None
    ]
    return round(sum(values) / len(values), 8) if values else None


def _normalized_edit_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return round(previous[-1] / max(1, len(left), len(right)), 8)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
