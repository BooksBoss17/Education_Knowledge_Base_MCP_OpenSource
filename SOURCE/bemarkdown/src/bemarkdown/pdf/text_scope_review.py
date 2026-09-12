"""Separate missing text localization from text-recognition disagreement."""
from __future__ import annotations
import hashlib
from PIL import Image, ImageChops


def missing_line_scope_reviews(plans, outputs_by_route, crops_by_route):
    reviews = {}
    for plan in plans:
        for route in plan['routes']:
            route_id = str(route['route_id'])
            if route['adapter'] != 'PAGE_VISUAL_TEXT_RECOVERY' or route_id not in crops_by_route:
                continue
            if outputs_by_route.get(route_id):
                continue
            crop = crops_by_route[route_id]
            with Image.open(crop['path']) as opened:
                image = opened.convert('RGB')
                pixels = ImageChops.invert(image).getbbox()
            bbox = list(crop['bbox_pdf_pt'])
            located = bbox if pixels is None else [
                bbox[0]+pixels[0]*(bbox[2]-bbox[0])/image.width,
                bbox[1]+pixels[1]*(bbox[3]-bbox[1])/image.height,
                bbox[0]+pixels[2]*(bbox[2]-bbox[0])/image.width,
                bbox[1]+pixels[3]*(bbox[3]-bbox[1])/image.height,
            ]
            reviews[route_id] = {
                'schema': 'bemarkdown-raster-text-scope-review-v1',
                'review_id': hashlib.sha256((route_id+crop['content_sha256']).encode()).hexdigest()[:24],
                'kind': 'RASTER_TEXT_SCOPE', 'status': 'REVIEW_REQUIRED',
                'reason': 'NO_DETECTED_TEXT_LINES_AFTER_NON_TEXT_EXCLUSION',
                'document_id': route['document_id'], 'page_index': route['page_index'],
                'source_bbox_pdf_pt': located, 'crop_bbox_pdf_pt': bbox,
                'remaining_bbox_crop_px': list(pixels) if pixels else None,
                'source_crop_sha256': crop['content_sha256'],
                'ownership_region_count': crop.get('ownership_region_count', 0),
                'recognition_success': False, 'text_recognition_requested': False,
                'detected_line_count': 0,
            }
    return reviews
