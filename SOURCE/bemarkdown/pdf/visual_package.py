"""Package writer shared by local API and direct visual adapters."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from .visual_serial_pipeline import VisualSerialPolicy
from ..page_render import PageRender as LayoutPageRender

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package_paths(value, output_dir):
    """Keep evidence links valid after atomic staging publication or migration."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {'path', 'source_path'} and isinstance(item, str):
                path = Path(item)
                if path.is_absolute() and path.is_relative_to(output_dir):
                    item = path.relative_to(output_dir).as_posix()
            result[key] = _package_paths(item, output_dir)
        return result
    if isinstance(value, (list, tuple)):
        return [_package_paths(item, output_dir) for item in value]
    return value

def _write_visual_package(
    *,
    payload: Mapping[str, Any],
    plan: Any,
    stage_result: Mapping[str, Any],
    source: str | Path,
    output_dir: str | Path,
    document_id: str,
    renders: Sequence[LayoutPageRender],
    policy: VisualSerialPolicy | None = None,
    debug: bool = False,
) -> Any:
    """Materialize a minimal BeMarkdown package without native-PDF evidence."""

    from ..page_render import PayloadConversionResult

    output_dir = Path(output_dir).resolve()
    source = Path(source).resolve()
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    formula = stage_result.get("formula", {})
    formula_markdown_ids = set(stage_result.get("formula_markdown_ids", []))
    text = stage_result.get("text", {})
    table = stage_result.get("table", {})
    content: list[str] = []
    asset_rows: list[dict[str, Any]] = []
    review_items = [dict(item) for item in stage_result.get("review_items", [])]
    visual_regions = list(plan.formula_regions) + list(plan.table_regions) + list(plan.text_regions) + list(plan.preserved_regions)
    from .visual_serial_pipeline import _as_region
    groups = stage_result.get('spatial_assembly', [])
    consumed = set(stage_result.get('assembly_consumed_ids', [])) | set(stage_result.get('contained_formula_aliases', {}))
    visual_regions = [r for r in visual_regions if r.region_id not in consumed]
    visual_regions.extend(_as_region(g['parent'], 0) for g in groups)
    text = {**text, **stage_result.get('assembled_text', {})}
    from .spatial_assembly import caption_owners
    captions = caption_owners(visual_regions, text)
    caption_ids = {i for ids in captions.values() for i in ids}
    region_pages = {region.region_id: region.page_index + 1 for region in visual_regions}
    for item in review_items:
        if item.get("region_id") in region_pages:
            item.setdefault("page", region_pages[item["region_id"]])
    review_region_ids = {item.get("region_id") for item in review_items}
    suppressed = set(stage_result.get("suppressed_text_region_ids", []))
    # Preserve explicit page anchors even when V3 finds no region on a blank
    # or watermark-only page.  The renderer's page count is authoritative for
    # visual routing and keeps the assembled document aligned with the source.
    page_indices = (
        list(range(len(renders)))
        if renders
        else sorted({region.page_index for region in visual_regions})
    )
    for page_index in page_indices:
        content.append(f"\n<!-- page:{page_index + 1} -->\n")
        page_regions = sorted(
            (region for region in visual_regions if region.page_index == page_index),
            key=lambda region: (region.reading_order, region.region_id),
        )
        if not page_regions and renders:
            # A detector returning zero boxes is not evidence of a blank page.
            from PIL import Image
            from io import BytesIO
            fallback = output_dir/'_bemarkdown'/f'undetected-page-{page_index}.png'
            fallback.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(BytesIO(renders[page_index].image_bytes)) as image:
                image.save(fallback, format='PNG')
                box = [0, 0, image.width, image.height]
            page_regions = [SimpleNamespace(region_id=f'p{page_index:04d}-undetected',
                label='unresolved_page', crop={'path':str(fallback)}, bbox=box)]
            review_items.append({'page':page_index+1, 'reason':'NO_LAYOUT_REGIONS_SOURCE_PRESERVED'})
        for region in page_regions:
            if region.region_id in suppressed or region.region_id in caption_ids:
                continue
            value = str(formula.get(region.region_id, table.get(region.region_id, text.get(region.region_id, '')))).strip()
            if value:
                if region.region_id in formula and region.region_id not in formula_markdown_ids:
                    content.append(f'${value}$' if region.label == 'inline_formula' else f'$$\n{value}\n$$')
                elif region.label in {'doc_title','paragraph_title'}:
                    content.append(('# ' if region.label == 'doc_title' else '## ')+value.lstrip('# ').strip())
                else:
                    content.append(value)
            metadata = getattr(region, 'metadata', {})
            fallback_source = metadata.get('fallback_formula_region_id')
            evidence_already_preserved = fallback_source and any(row['source_locator'] == fallback_source for row in asset_rows)
            if (not value or region.region_id in review_region_ids or metadata.get('preserve_source')) and not evidence_already_preserved:
                if not value and (region.region_id in formula or region.region_id in text or region.region_id in table):
                    review_items.append({'page':page_index+1, 'region_id':region.region_id,
                                         'reason':'EMPTY_OCR_SOURCE_PRESERVED'})
                elif ('table' in region.label or 'algorithm' in region.label) and region.region_id not in table:
                    review_items.append({'page':page_index+1, 'region_id':region.region_id,
                                         'reason':'STRUCTURED_REGION_PRESERVED_AS_IMAGE'})
                elif region.label in {'image', 'chart'}:
                    review_items.append({'page':page_index+1, 'region_id':region.region_id,
                                         'reason':'VISUAL_REGION_TEXT_CONTENT_UNVERIFIED'})
                source_crop = Path(str(region.crop.get("source_path") or region.crop["path"])).resolve()
                # BeMarkdown's published package contract requires continuous
                # image_000001-style asset IDs in manifest reading order.
                asset_id = f"image_{len(asset_rows) + 1:06d}"
                suffix = source_crop.suffix.lower() or ".png"
                target = assets_dir / f"{asset_id}{suffix}"
                target.write_bytes(source_crop.read_bytes())
                relative = f"assets/{target.name}"
                digest = _sha256(target)
                caption = " ".join(text[i] for i in captions.get(region.region_id, []))
                content.append((caption+"\n\n" if caption else "")+f"![{region.label}]({relative})")
                asset_rows.append(
                    {
                        "manifest_version": "bemarkdown-asset-contract-v1",
                        "asset_id": asset_id,
                        "asset_uid": f"visual:{document_id}:{region.region_id}",
                        "content_sha256": digest,
                        "semantic_type": region.label,
                        "status": "RESOLVED",
                        "mime_type": "image/png",
                        "extension": suffix,
                        "relative_path": relative,
                        "source_type": "VISUAL_RENDER",
                        "source_part": f"page:{page_index + 1}",
                        "source_locator": region.region_id,
                        "relationship_id": None,
                        "external_target": None,
                        "render_method": "BOUNDED_VISUAL_CROP",
                        "provenance": {"bbox_render_px": list(region.bbox), "caption_region_ids": captions.get(region.region_id, []), "source_aliases": metadata.get("source_aliases", []), "original_bbox": metadata.get("original_bbox")},
                    }
                )
    (output_dir / "document.md").write_text("\n\n".join(content).strip() + "\n", encoding="utf-8")
    (output_dir / "assets_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in asset_rows),
        encoding="utf-8",
    )
    source_bytes = source.read_bytes()
    preserved_ids = {row['source_locator'] for row in asset_rows}
    formula_conservation = all(
        str(formula.get(region.region_id, '')).strip() or region.region_id in preserved_ids
        for region in plan.formula_regions
    )
    report = {
        "schema": "bemarkdown-pdf-conversion-report-v1",
        "quality_status": "COMPLETED_WITH_REVIEW_ITEMS" if review_items else "COMPLETED_WITH_WARNINGS",
        "review_item_count": len(review_items),
        "review_items": review_items,
        "source": {
            "type": "PDF",
            "path": str(source),
            "file_name": source.name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "size_bytes": len(source_bytes),
        },
        "document": {"pages": len(renders), "blocks": len(visual_regions)},
        "formulas": {
            "detected": len(plan.formula_regions),
            "recognized": sum(bool(str(value).strip()) for value in formula.values()),
            "failed_preserved": sum(not str(value).strip() for value in formula.values()),
            "conservation_ok": formula_conservation,
        },
        "tables": {
            "enabled": bool(plan.policy.table_model_id),
            "model_id": payload.get("stage_identities", {}).get("table", {}).get("model_id", plan.policy.table_model_id),
            "detected": len(plan.table_regions),
            "recognized": sum(bool(value) for value in table.values()),
            "failed_preserved": sum(not value for value in table.values()),
            "source_preserved": all(r.region_id in preserved_ids for r in plan.table_regions),
            "quality_certified": False,
        },
        "formula_ocr": {
            "enabled": True,
            "model_id": payload.get("stage_identities", {}).get("formula", {}).get("model_id", plan.policy.formula_model_id),
            "conservation_ok": formula_conservation,
            "real_formula_conservation_ok": formula_conservation,
            "all_formula_results_usable": all(bool(str(value).strip()) for value in formula.values()) and len(formula) == len(plan.formula_regions),
        },
        "assets": {"total_visible": len(asset_rows), "resolved": len(asset_rows), "unresolved": 0},
        "warnings": [
            "AUTOMATIC_TRANSCRIPTION_NOT_FULL_BOOK_QUALITY_CERTIFICATION",
            "TABLE_TRANSCRIPTION_REQUIRES_REVIEW_ALGORITHMS_PRESERVED" if plan.policy.table_model_id else "SCANNED_TABLES_AND_ALGORITHMS_PRESERVED_AS_IMAGES",
        ],
        "errors": [],
        "visual_serial": {
            "policy": dict(payload["policy"]),
            "evidence_path_base": "PACKAGE_ROOT",
            "plan": _package_paths(dict(payload["plan"]), output_dir),
            "result": _package_paths(dict(stage_result), output_dir),
            "stage_identities": dict(payload.get("stage_identities", {})),
            "stage_metrics": dict(payload.get("stage_metrics", {})),
            "parse_pdf_native": False,
            "image_screening_called": False,
            "final_review_called": False,
        },
        "timing": {"stage_trace": list(stage_result.get("stage_trace", []))},
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        (debug_dir / "visual_serial_result.json").write_text(
            json.dumps(_package_paths({"payload": payload, "stage_result": stage_result}, output_dir), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return PayloadConversionResult(source, output_dir, report)


