from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image


def classify_pre_ocr_visual(
    route: Mapping[str, Any], crop_path: Path
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "bemarkdown-pre-ocr-visual-gate-v1",
        "action": "OCR_TEXT",
        "reason_code": "VISUAL_TEXT_GATE_NOT_PROVEN",
        "content_cropped": False,
    }
    semantic = str((route.get("semantic_evidence") or [""])[0])
    if route.get("adapter") != "OCR_TEXT_REGION" or semantic != "HEADER_FOOTER":
        return result
    try:
        with Image.open(crop_path) as source:
            image = source.convert("RGB")
    except OSError as exc:
        return {**result, "inspection_error": f"{type(exc).__name__}: {exc}"}

    pixels = list(image.get_flattened_data())
    nonwhite = [pixel for pixel in pixels if min(pixel) < 245]
    chromatic = [
        pixel for pixel in nonwhite if max(pixel) - min(pixel) >= 30
    ]
    dark = [
        pixel
        for pixel in nonwhite
        if max(pixel) < 128 and max(pixel) - min(pixel) < 30
    ]
    nonwhite_count = len(nonwhite)
    features = {
        "width": image.width,
        "height": image.height,
        "nonwhite_pixel_count": nonwhite_count,
        "nonwhite_ratio": nonwhite_count / max(1, len(pixels)),
        "chromatic_nonwhite_ratio": len(chromatic) / max(1, nonwhite_count),
        "dark_nonwhite_ratio": len(dark) / max(1, nonwhite_count),
    }
    proven = (
        image.width <= 128
        and image.height <= 100
        and nonwhite_count >= 64
        and features["nonwhite_ratio"] >= 0.2
        and features["chromatic_nonwhite_ratio"] >= 0.8
        and features["dark_nonwhite_ratio"] <= 0.07
    )
    return {
        **result,
        "action": "PRESERVE_GRAPHIC" if proven else "OCR_TEXT",
        "reason_code": (
            "SMALL_CHROMATIC_NON_TEXT_GRAPHIC"
            if proven
            else "VISUAL_TEXT_GATE_NOT_PROVEN"
        ),
        "features": features,
    }


def reroute_pre_ocr_visual(
    route: Mapping[str, Any], crop_path: Path
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(route))
    decision = classify_pre_ocr_visual(route, crop_path)
    if decision["action"] != "PRESERVE_GRAPHIC":
        return updated
    updated["adapter"] = "IMAGE_RENDER_CROP"
    updated["output_kind"] = "IMAGE"
    reasons = list(updated.get("decision_reason_codes") or ())
    for reason in (
        "SMALL_CHROMATIC_NON_TEXT_GRAPHIC",
        "RENDER_CROP_PRESERVES_VISUAL_EXTENT",
    ):
        if reason not in reasons:
            reasons.append(reason)
    updated["decision_reason_codes"] = reasons
    provenance = dict(updated.get("provenance") or {})
    provenance["pre_ocr_visual_gate"] = decision
    updated["provenance"] = provenance
    return updated


def reroute_prepared_visuals(
    plans: list[dict[str, Any]], crops_by_route: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    evaluated = 0
    rerouted_ids = []
    for plan in plans:
        routes = list(plan.get("routes") or ())
        for index, route in enumerate(routes):
            if route.get("adapter") != "OCR_TEXT_REGION":
                continue
            evaluated += 1
            route_id = str(route["route_id"])
            crop = crops_by_route.get(route_id) or {}
            crop_path = crop.get("path")
            if not crop_path:
                continue
            updated = reroute_pre_ocr_visual(route, Path(str(crop_path)))
            routes[index] = updated
            if updated["adapter"] == "IMAGE_RENDER_CROP":
                rerouted_ids.append(route_id)
        plan["routes"] = routes
    return {
        "schema": "bemarkdown-pre-ocr-visual-gate-audit-v1",
        "evaluated_ocr_routes": evaluated,
        "rerouted_to_visual_preservation": len(rerouted_ids),
        "rerouted_route_ids": sorted(rerouted_ids),
        "gate": "PASS",
    }
