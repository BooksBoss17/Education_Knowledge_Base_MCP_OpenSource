from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OCR_REGION_RECOVERY_VERSION = "ocr-region-recovery-v0"
TEXT_SEMANTIC_HINTS = {
    "TEXT",
    "TEXT_LIKE",
    "TITLE",
    "CAPTION",
    "HEADER_FOOTER",
    "PAGE_NUMBER",
}
RETRY_SUCCESS_SEMANTIC_HINTS = {
    "TEXT",
    "TEXT_LIKE",
    "TITLE",
    "CAPTION",
    "PAGE_NUMBER",
}


@dataclass(frozen=True)
class OcrRegionRecoveryPolicy:
    direct_rec_min_aspect_ratio: float = 1.5
    direct_rec_max_aspect_ratio: float = 40.0
    direct_rec_max_height_px: int = 160
    direct_rec_max_area_px: int = 250_000
    direct_rec_max_bbox_height_pt: float = 60.0
    direct_rec_min_confidence: float = 0.5
    retry_det_min_confidence: float = 0.6

    def direct_rec_eligibility(
        self, route: dict[str, Any], crop: dict[str, Any]
    ) -> dict[str, Any]:
        semantic = {str(value).upper() for value in route.get("semantic_evidence", [])}
        width = int(crop.get("width") or 0)
        height = int(crop.get("height") or 0)
        bbox = route.get("provenance", {}).get("bbox_pdf_pt") or []
        bbox_height = (
            float(bbox[3]) - float(bbox[1])
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4
            else float("inf")
        )
        ratio = width / max(1, height)
        reasons = []
        if route.get("adapter") != "OCR_TEXT_REGION":
            reasons.append("NOT_REGION_OCR")
        if not semantic or not semantic.issubset(TEXT_SEMANTIC_HINTS):
            reasons.append("NON_TEXT_SEMANTIC")
        if width <= 0 or height <= 0:
            reasons.append("INVALID_CROP_GEOMETRY")
        if not self.direct_rec_min_aspect_ratio <= ratio <= self.direct_rec_max_aspect_ratio:
            reasons.append("NOT_SINGLE_LINE_ASPECT")
        if height > self.direct_rec_max_height_px or width * height > self.direct_rec_max_area_px:
            reasons.append("CROP_TOO_LARGE_FOR_DIRECT_REC")
        if bbox_height > self.direct_rec_max_bbox_height_pt:
            reasons.append("BBOX_TOO_TALL_FOR_DIRECT_REC")
        return {
            "policy_version": OCR_REGION_RECOVERY_VERSION,
            "eligible": not reasons,
            "reason_codes": sorted(reasons or ["SINGLE_LINE_GEOMETRY_ELIGIBLE"]),
            "features": {
                "crop_width_px": width,
                "crop_height_px": height,
                "crop_aspect_ratio": round(ratio, 8),
                "crop_area_px": width * height,
                "bbox_height_pt": round(bbox_height, 8),
                "semantic_evidence": sorted(semantic),
                "ocr_output_used_for_eligibility": False,
            },
        }

    def classify_direct_rec(
        self,
        lines: list[dict[str, Any]],
        route: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        usable = [row for row in lines if str(row.get("text") or "").strip()]
        confidences = [
            float(row["confidence"])
            for row in usable
            if row.get("confidence") is not None
        ]
        confidence = sum(confidences) / len(confidences) if confidences else None
        has_text = bool(usable)
        semantic = (
            {str(value).upper() for value in route.get("semantic_evidence", [])}
            if route is not None
            else set()
        )
        semantic_safe = route is None or (
            bool(semantic) and semantic.issubset(RETRY_SUCCESS_SEMANTIC_HINTS)
        )
        safe = (
            has_text
            and confidence is not None
            and confidence >= self.direct_rec_min_confidence
            and semantic_safe
        )
        reasons = []
        if has_text and not safe:
            reasons.append("DIRECT_REC_LOW_CONFIDENCE_REVIEW")
        if not has_text:
            reasons.append("DIRECT_REC_EMPTY")
        if has_text and not semantic_safe:
            reasons.append("DIRECT_REC_SEMANTIC_REVIEW")
        return {
            "policy_version": OCR_REGION_RECOVERY_VERSION,
            "has_text": has_text,
            "safe_for_success": safe,
            "mean_confidence": confidence,
            "semantic_evidence": sorted(semantic),
            "reason_codes": reasons,
        }

    def classify_retry_det(
        self, route: dict[str, Any], lines: list[dict[str, Any]]
    ) -> dict[str, Any]:
        usable = [row for row in lines if str(row.get("text") or "").strip()]
        confidences = [
            float(row["confidence"])
            for row in usable
            if row.get("confidence") is not None
        ]
        confidence = sum(confidences) / len(confidences) if confidences else None
        semantic = {str(value).upper() for value in route.get("semantic_evidence", [])}
        reasons = []
        if not usable:
            reasons.append("RETRY_DET_EMPTY")
        if not semantic or not semantic.issubset(RETRY_SUCCESS_SEMANTIC_HINTS):
            reasons.append("RETRY_DET_SEMANTIC_REVIEW")
        if confidence is None or confidence < self.retry_det_min_confidence:
            reasons.append("RETRY_DET_LOW_CONFIDENCE_REVIEW")
        return {
            "policy_version": OCR_REGION_RECOVERY_VERSION,
            "has_text": bool(usable),
            "safe_for_success": not reasons,
            "mean_confidence": confidence,
            "semantic_evidence": sorted(semantic),
            "reason_codes": sorted(reasons),
        }
