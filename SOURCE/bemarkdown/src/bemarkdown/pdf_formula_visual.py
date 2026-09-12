from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

FORMULA_VISUAL_VERIFICATION_SCHEMA = "bemarkdown-formula-visual-verification-ir-v0"
FORMULA_VISUAL_VERIFIER_VERSION = "formula-visual-verifier-v0"


@dataclass(frozen=True)
class FormulaVisualPolicy:
    review_score_threshold: float = 0.42
    warning_score_threshold: float = 0.28
    aspect_ratio_review_threshold: float = 0.45
    density_review_threshold: float = 0.25
    projection_review_threshold: float = 0.55
    overlap_review_threshold: float = 0.32
    component_log_review_threshold: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "review_score_threshold": self.review_score_threshold,
            "warning_score_threshold": self.warning_score_threshold,
            "aspect_ratio_review_threshold": self.aspect_ratio_review_threshold,
            "density_review_threshold": self.density_review_threshold,
            "projection_review_threshold": self.projection_review_threshold,
            "overlap_review_threshold": self.overlap_review_threshold,
            "component_log_review_threshold": self.component_log_review_threshold,
        }


class FormulaVisualVerifier:
    """Compare source/render structure and only escalate recognition risk."""

    def __init__(
        self,
        policy: FormulaVisualPolicy | None = None,
        *,
        renderer: Callable[..., dict[str, Any]] | None = None,
    ):
        self.policy = policy or FormulaVisualPolicy()
        self.renderer = renderer

    def verify(
        self,
        *,
        content_id: str,
        raw_latex: str,
        source_crop_ref: str | Path,
        rendered_formula_ref: str | Path | None = None,
        source_bbox_pdf_pt: list[float],
        renderer_fingerprint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = Path(source_crop_ref).resolve()
        resolved_renderer_fingerprint = dict(renderer_fingerprint or {})
        render_error = None
        try:
            if rendered_formula_ref is None:
                if self.renderer is None:
                    raise RuntimeError("FORMULA_RENDERER_UNAVAILABLE")
                rendered_result = self.renderer(
                    content_id=content_id, raw_latex=raw_latex
                )
                rendered_formula_ref = rendered_result["path"]
                resolved_renderer_fingerprint = dict(
                    rendered_result.get("renderer_fingerprint") or {}
                )
            rendered = Path(rendered_formula_ref).resolve()
        except Exception as exc:  # noqa: BLE001 - renderer must fail closed
            rendered = None
            render_error = f"{type(exc).__name__}: {exc}"
        identity = {
            "content_id": content_id,
            "source_crop_ref": str(source),
            "rendered_formula_ref": str(rendered) if rendered is not None else None,
            "raw_latex_sha256": hashlib.sha256(raw_latex.encode("utf-8")).hexdigest(),
            "policy_version": FORMULA_VISUAL_VERIFIER_VERSION,
        }
        verification_id = "formula-visual-" + hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()[:20]
        if rendered is None:
            return {
                "schema": FORMULA_VISUAL_VERIFICATION_SCHEMA,
                "verification_id": verification_id,
                "content_id": content_id,
                "source_crop_ref": str(source),
                "rendered_formula_ref": None,
                "status": "VERIFICATION_UNAVAILABLE",
                "score": None,
                "features": {},
                "reason_codes": ["FORMULA_RENDERER_UNAVAILABLE"],
                "source_bbox_pdf_pt": list(source_bbox_pdf_pt),
                "source_foreground_bbox": None,
                "render_foreground_bbox": None,
                "policy_version": FORMULA_VISUAL_VERIFIER_VERSION,
                "provenance": {
                    "error": render_error,
                    "raw_latex_sha256": identity["raw_latex_sha256"],
                    "raw_latex_rewritten": False,
                    "renderer_fingerprint": resolved_renderer_fingerprint,
                    "fail_closed": True,
                },
            }
        try:
            features = extract_formula_visual_features(source, rendered)
            source_normalized = normalize_formula_foreground(source)
            render_normalized = normalize_formula_foreground(rendered)
            status, score, reasons = self._classify(features)
        except Exception as exc:  # noqa: BLE001 - visual safety must fail closed
            return {
                "schema": FORMULA_VISUAL_VERIFICATION_SCHEMA,
                "verification_id": verification_id,
                "content_id": content_id,
                "source_crop_ref": str(source),
                "rendered_formula_ref": str(rendered),
                "status": "VERIFICATION_UNAVAILABLE",
                "score": None,
                "features": {},
                "reason_codes": ["FEATURE_EXTRACTION_FAILED"],
                "source_bbox_pdf_pt": list(source_bbox_pdf_pt),
                "source_foreground_bbox": None,
                "render_foreground_bbox": None,
                "policy_version": FORMULA_VISUAL_VERIFIER_VERSION,
                "provenance": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_latex_sha256": identity["raw_latex_sha256"],
                    "raw_latex_rewritten": False,
                    "renderer_fingerprint": resolved_renderer_fingerprint,
                    "normalization": "grayscale-threshold-trim-rigid-alignment-v0",
                },
            }
        return {
            "schema": FORMULA_VISUAL_VERIFICATION_SCHEMA,
            "verification_id": verification_id,
            "content_id": content_id,
            "source_crop_ref": str(source),
            "rendered_formula_ref": str(rendered),
            "status": status,
            "score": score,
            "features": features,
            "reason_codes": reasons,
            "source_bbox_pdf_pt": list(source_bbox_pdf_pt),
            "source_foreground_bbox": source_normalized["foreground_bbox"],
            "render_foreground_bbox": render_normalized["foreground_bbox"],
            "policy_version": FORMULA_VISUAL_VERIFIER_VERSION,
            "provenance": {
                "raw_latex_sha256": identity["raw_latex_sha256"],
                "raw_latex_rewritten": False,
                "renderer_fingerprint": resolved_renderer_fingerprint,
                "normalization": "grayscale-threshold-trim-rigid-alignment-v0",
                "alignment": "aspect-preserving-centroid-plus-small-translation-search",
                "non_rigid_warping_used": False,
                "policy": self.policy.to_dict(),
            },
        }

    def _classify(self, features: dict[str, float | int]) -> tuple[str, float, list[str]]:
        reasons = []
        if features["aspect_ratio_log_difference"] > self.policy.aspect_ratio_review_threshold:
            reasons.append("ASPECT_RATIO_MISMATCH")
        if features["foreground_density_difference"] > self.policy.density_review_threshold:
            reasons.append("FOREGROUND_DENSITY_MISMATCH")
        if min(
            features["horizontal_projection_similarity"],
            features["vertical_projection_similarity"],
        ) < self.policy.projection_review_threshold:
            reasons.append("PROJECTION_STRUCTURE_MISMATCH")
        if features["aligned_foreground_iou"] < self.policy.overlap_review_threshold:
            reasons.append("ALIGNED_FOREGROUND_MISMATCH")
        if features["component_count_log_difference"] > self.policy.component_log_review_threshold:
            reasons.append("CONNECTED_COMPONENT_MISMATCH")
        score = round(
            0.30 * min(1.0, features["aspect_ratio_log_difference"] / 1.0)
            + 0.15 * min(1.0, features["foreground_density_difference"] / 0.35)
            + 0.075 * (1.0 - features["horizontal_projection_similarity"])
            + 0.075 * (1.0 - features["vertical_projection_similarity"])
            + 0.25 * (1.0 - features["aligned_foreground_iou"])
            + 0.15 * min(1.0, features["component_count_log_difference"] / 1.5),
            8,
        )
        if score >= self.policy.review_score_threshold or len(reasons) >= 2:
            return "VISUAL_REVIEW", score, sorted(reasons or ["COMPOSITE_VISUAL_MISMATCH"])
        if score >= self.policy.warning_score_threshold or reasons:
            return "VISUAL_WARNING", score, sorted(reasons or ["COMPOSITE_VISUAL_WARNING"])
        return "VISUAL_PASS", score, []


def normalize_formula_foreground(path: str | Path) -> dict[str, Any]:
    mask, bbox, original_size = _foreground_mask(Path(path))
    height = len(mask)
    width = len(mask[0]) if mask else 0
    foreground = sum(sum(row) for row in mask)
    return {
        "foreground_bbox": bbox,
        "original_size": list(original_size),
        "trimmed_size": [width, height],
        "foreground_pixels": foreground,
        "foreground_density": round(foreground / max(1, width * height), 8),
        "component_count": _component_count(mask),
    }


def extract_formula_visual_features(
    source_crop_ref: str | Path, rendered_formula_ref: str | Path
) -> dict[str, float | int]:
    source, _source_bbox, _source_size = _foreground_mask(Path(source_crop_ref))
    rendered, _render_bbox, _render_size = _foreground_mask(Path(rendered_formula_ref))
    source_height, source_width = len(source), len(source[0])
    render_height, render_width = len(rendered), len(rendered[0])
    source_ratio = source_width / max(1, source_height)
    render_ratio = render_width / max(1, render_height)
    source_density = sum(sum(row) for row in source) / max(1, source_width * source_height)
    render_density = sum(sum(row) for row in rendered) / max(1, render_width * render_height)
    source_horizontal = _projection(source, axis="horizontal")
    render_horizontal = _projection(rendered, axis="horizontal")
    source_vertical = _projection(source, axis="vertical")
    render_vertical = _projection(rendered, axis="vertical")
    source_components = _component_count(source)
    render_components = _component_count(rendered)
    source_canvas = _fit_canvas(source)
    render_canvas = _fit_canvas(rendered)
    local_mismatch, mismatch_concentration = _localized_mismatch(
        source_canvas, render_canvas
    )
    localized_projection_mismatch = _localized_projection_mismatch(
        source_canvas, render_canvas
    )
    return {
        "aspect_ratio_log_difference": round(abs(math.log(source_ratio / render_ratio)), 8),
        "foreground_density_difference": round(abs(source_density - render_density), 8),
        "horizontal_projection_similarity": round(
            _cosine_similarity(_resample(source_horizontal, 64), _resample(render_horizontal, 64)),
            8,
        ),
        "vertical_projection_similarity": round(
            _cosine_similarity(_resample(source_vertical, 64), _resample(render_vertical, 64)),
            8,
        ),
        "aligned_foreground_iou": round(_aligned_iou(source, rendered), 8),
        "source_component_count": source_components,
        "render_component_count": render_components,
        "component_count_log_difference": round(
            abs(math.log((source_components + 1) / (render_components + 1))), 8
        ),
        "local_window_foreground_mismatch": round(local_mismatch, 8),
        "localized_projection_mismatch": round(localized_projection_mismatch, 8),
        "coarse_component_correspondence": round(
            min(source_components, render_components)
            / max(1, source_components, render_components),
            8,
        ),
        "mismatch_concentration": round(mismatch_concentration, 8),
    }


def apply_formula_visual_gate(
    *,
    existing_status: str,
    existing_quality_status: str,
    raw_latex: str,
    verification: dict[str, Any],
) -> dict[str, Any]:
    if existing_status in {"FAILED_PRESERVE_INPUT", "DEFERRED"}:
        status = existing_status
        quality = existing_quality_status
    elif existing_status == "REVIEW_REQUIRED":
        status = existing_status
        quality = "FORMULA_CONTENT_REVIEW"
    elif verification.get("status") == "VISUAL_PASS":
        status = existing_status
        quality = existing_quality_status
    else:
        status = "REVIEW_REQUIRED"
        quality = "FORMULA_CONTENT_REVIEW"
    return {
        "status": status,
        "quality_status": quality,
        "raw_latex": raw_latex,
        "review_reasons": (
            []
            if verification.get("status") == "VISUAL_PASS"
            else sorted(
                set(verification.get("reason_codes", []))
                | {"FORMULA_VISUAL_VERIFICATION_REVIEW"}
            )
        ),
    }


def _foreground_mask(path: Path) -> tuple[list[list[int]], list[int], tuple[int, int]]:
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image)
        original_size = gray.size
        corners = [
            gray.getpixel((0, 0)),
            gray.getpixel((gray.width - 1, 0)),
            gray.getpixel((0, gray.height - 1)),
            gray.getpixel((gray.width - 1, gray.height - 1)),
        ]
        background = sorted(corners)[len(corners) // 2]
        threshold = max(32, background - 32)
        coordinates = []
        for y in range(gray.height):
            for x in range(gray.width):
                if gray.getpixel((x, y)) < threshold:
                    coordinates.append((x, y))
        if not coordinates:
            raise ValueError(f"No foreground pixels in {path}")
        left = min(x for x, _ in coordinates)
        top = min(y for _, y in coordinates)
        right = max(x for x, _ in coordinates) + 1
        bottom = max(y for _, y in coordinates) + 1
        pixels = gray.load()
        mask = [
            [1 if pixels[x, y] < threshold else 0 for x in range(left, right)]
            for y in range(top, bottom)
        ]
    return mask, [left, top, right, bottom], original_size


def _projection(mask: list[list[int]], *, axis: str) -> list[float]:
    height = len(mask)
    width = len(mask[0])
    if axis == "horizontal":
        return [sum(row) / width for row in mask]
    return [sum(mask[y][x] for y in range(height)) / height for x in range(width)]


def _resample(values: list[float], size: int) -> list[float]:
    if len(values) == 1:
        return values * size
    output = []
    for index in range(size):
        position = index * (len(values) - 1) / max(1, size - 1)
        left = int(position)
        right = min(len(values) - 1, left + 1)
        fraction = position - left
        output.append(values[left] * (1.0 - fraction) + values[right] * fraction)
    return output


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    dot = sum(left * right for left, right in zip(first, second, strict=True))
    left_norm = math.sqrt(sum(value * value for value in first))
    right_norm = math.sqrt(sum(value * value for value in second))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _component_count(mask: list[list[int]]) -> int:
    height = len(mask)
    width = len(mask[0])
    seen = set()
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or (x, y) in seen:
                continue
            count += 1
            queue = deque([(x, y)])
            seen.add((x, y))
            while queue:
                current_x, current_y = queue.popleft()
                for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
                    next_x, next_y = current_x + dx, current_y + dy
                    if (
                        0 <= next_x < width
                        and 0 <= next_y < height
                        and mask[next_y][next_x]
                        and (next_x, next_y) not in seen
                    ):
                        seen.add((next_x, next_y))
                        queue.append((next_x, next_y))
    return count


def _aligned_iou(first: list[list[int]], second: list[list[int]]) -> float:
    first_canvas = _fit_canvas(first)
    second_canvas = _fit_canvas(second)
    best = 0.0
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            intersection = 0
            union = 0
            for y in range(64):
                for x in range(128):
                    left = first_canvas[y][x]
                    source_x, source_y = x - dx, y - dy
                    right = (
                        second_canvas[source_y][source_x]
                        if 0 <= source_x < 128 and 0 <= source_y < 64
                        else 0
                    )
                    intersection += left & right
                    union += left | right
            best = max(best, intersection / max(1, union))
    return best


def _fit_canvas(mask: list[list[int]]) -> list[list[int]]:
    source = Image.new("1", (len(mask[0]), len(mask)), 0)
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                source.putpixel((x, y), 1)
    scale = min(112 / source.width, 52 / source.height)
    width = max(1, round(source.width * scale))
    height = max(1, round(source.height * scale))
    resized = source.resize((width, height), Image.Resampling.NEAREST)
    canvas = Image.new("1", (128, 64), 0)
    canvas.paste(resized, ((128 - width) // 2, (64 - height) // 2))
    return [[1 if canvas.getpixel((x, y)) else 0 for x in range(128)] for y in range(64)]


def _localized_mismatch(
    first: list[list[int]], second: list[list[int]]
) -> tuple[float, float]:
    window_mismatches = []
    mismatch_counts = []
    for top in range(0, 64, 16):
        for left in range(0, 128, 32):
            first_foreground = 0
            second_foreground = 0
            mismatch = 0
            for y in range(top, top + 16):
                for x in range(left, left + 32):
                    first_foreground += first[y][x]
                    second_foreground += second[y][x]
                    mismatch += first[y][x] != second[y][x]
            window_mismatches.append(
                abs(first_foreground - second_foreground) / (16 * 32)
            )
            mismatch_counts.append(mismatch)
    total_mismatch = sum(mismatch_counts)
    concentration = max(mismatch_counts, default=0) / max(1, total_mismatch)
    return max(window_mismatches, default=0.0), concentration


def _localized_projection_mismatch(
    first: list[list[int]], second: list[list[int]]
) -> float:
    first_horizontal = _projection(first, axis="horizontal")
    second_horizontal = _projection(second, axis="horizontal")
    first_vertical = _projection(first, axis="vertical")
    second_vertical = _projection(second, axis="vertical")
    horizontal = [
        abs(sum(first_horizontal[index : index + 8]) - sum(second_horizontal[index : index + 8]))
        / 8
        for index in range(0, 64, 8)
    ]
    vertical = [
        abs(sum(first_vertical[index : index + 16]) - sum(second_vertical[index : index + 16]))
        / 16
        for index in range(0, 128, 16)
    ]
    return max(horizontal + vertical, default=0.0)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
