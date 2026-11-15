"""Trim duplicate partial lines only when another crop owns all removed ink."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

TEXT_LABELS = frozenset({'text', 'content', 'abstract', 'reference_content'})
POLICY = 'TEXT_EDGE_COMPLETE_OWNER_V1'


def _pixel_box(region, width, height):
    box = tuple(int(v) for v in region.bbox)
    if box != region.bbox:
        return None
    left, top, right, bottom = box
    return box if 0 <= left < right <= width and 0 <= top < bottom <= height else None


def _cut_line(ink, upper, lower):
    """Return a blank-row cut and its complete source line, or decline."""
    ux0, uy0, ux1, uy1 = upper
    lx0, ly0, lx1, ly1 = lower
    rows = ink[uy0:ly1, lx0:lx1].any(axis=1)
    offset = ly0 - uy0
    if not rows[offset]:
        return None, 'EDGE_HAS_NO_PARTIAL_LINE'
    start = offset
    while start and rows[start - 1]:
        start -= 1
    end = offset + 1
    while end < len(rows) and rows[end]:
        end += 1
    # A complete line needs blank rows above and below, within the owner.
    if start < 2 or end - start < 4 or uy0 + end + 2 > uy1:
        return None, 'OWNER_DOES_NOT_BOUND_COMPLETE_LINE'
    if rows[start - 2:start].any() or rows[end:end + 2].any():
        return None, 'NO_CLEAR_LINE_GAP'
    gap_end = end
    while gap_end < len(rows) and not rows[gap_end]:
        gap_end += 1
    if gap_end == len(rows):
        return None, 'NO_REMAINING_TEXT_LINE'
    cut = min(uy1, uy0 + (end + gap_end) // 2)
    # Every removed foreground pixel must have an identical owner on the page.
    strip = ink[ly0:cut, lx0:lx1]
    covered = np.zeros_like(strip)
    left, right = max(ux0, lx0) - lx0, min(ux1, lx1) - lx0
    covered[:, left:right] = True
    if np.any(strip & ~covered):
        return None, 'REMOVED_INK_NOT_COVERED'
    return (cut, uy0 + start, uy0 + end, int(strip.sum())), None


def repair_text_overlap(plan, renders, output_dir):
    """Adjust top edges before formula masking, preserving original evidence."""
    if output_dir is None:
        return plan, []
    folder = Path(output_dir) / '_bemarkdown/visual_serial/text_overlap_crops'
    rows = {r.region_id: r for r in plan.all_regions}
    eligible = {r.region_id for r in plan.text_regions} | set(plan.formula_overlap_removed)
    decisions = []
    for page in sorted({r.page_index for r in plan.all_regions}):
        candidates = sorted((r for r in rows.values() if r.page_index == page
                             and r.region_id in eligible and r.normalized_label in TEXT_LABELS),
                            key=lambda r: (r.bbox[1], r.reading_order, r.region_id))
        if len(candidates) < 2:
            continue
        render = next((r for index, r in enumerate(renders) if getattr(r, 'page_index', index) == page), None)
        if render is None:
            continue
        with Image.open(BytesIO(render.image_bytes)) as original:
            image = original.convert('RGB')
        try:
            pixels = np.asarray(image)
            gray = np.asarray(image.convert('L'))
            # A high foreground threshold protects antialiased edges. Colored
            # or noisy pages without clean whitespace are left for review.
            ink = gray < 245
            verified = {}

            def matches_page(region, box):
                if region.region_id not in verified:
                    path = Path(region.crop.get('path', ''))
                    if not path.is_file():
                        verified[region.region_id] = False
                    else:
                        data = path.read_bytes()
                        expected = region.crop.get('sha256')
                        with Image.open(BytesIO(data)) as source:
                            crop_pixels = np.asarray(source.convert('RGB'))
                        x0, y0, x1, y1 = box
                        verified[region.region_id] = bool(expected and sha256(data).hexdigest() == expected
                            and np.array_equal(crop_pixels, pixels[y0:y1, x0:x1]))
                return verified[region.region_id]

            for index, candidate in enumerate(candidates):
                lower = rows[candidate.region_id]
                lower_box = _pixel_box(lower, image.width, image.height)
                if lower_box is None:
                    continue
                lx0, ly0, lx1, ly1 = lower_box
                for previous in reversed(candidates[:index]):
                    upper = rows[previous.region_id]
                    upper_box = _pixel_box(upper, image.width, image.height)
                    if upper_box is None or upper.reading_order >= lower.reading_order:
                        continue
                    ux0, uy0, ux1, uy1 = upper_box
                    shared_width = min(ux1, lx1) - max(ux0, lx0)
                    if not (uy0 < ly0 < uy1 < ly1 and shared_width >= .9 * (lx1 - lx0)):
                        continue
                    note = {'region_id': lower.region_id, 'owner_region_id': upper.region_id,
                            'page': page + 1, 'policy': POLICY, 'original_bbox': list(lower.bbox),
                            'owner_bbox': list(upper.bbox)}
                    cut_info, reason = _cut_line(ink, upper_box, lower_box)
                    if cut_info and not (matches_page(upper, upper_box) and matches_page(lower, lower_box)):
                        cut_info, reason = None, 'CROP_SOURCE_NOT_PIXEL_IDENTICAL'
                    if cut_info is None:
                        decisions.append({**note, 'action': 'PRESERVED', 'reason': reason})
                        continue
                    cut, line_top, line_bottom, count = cut_info
                    blockers = [r for r in plan.preserved_regions + plan.table_regions
                                if r.page_index == page and r.bbox[0] < lx1 and r.bbox[2] > lx0
                                and r.bbox[1] < cut and r.bbox[3] > line_top]
                    if blockers:
                        decisions.append({**note, 'action': 'PRESERVED', 'reason': 'STRUCTURED_REGION_OVERLAP'})
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    name = sha256(lower.region_id.encode('utf-8')).hexdigest()[:20] + '.png'
                    path = folder / name
                    box = (lx0, cut, lx1, ly1)
                    with image.crop(box) as crop:
                        crop.save(path, format='PNG')
                    decision = {**note, 'action': 'TRIMMED', 'reason': 'COMPLETE_LINE_OWNED_BY_UPPER',
                                'bbox': list(box), 'line_y': [line_top, line_bottom],
                                'removed_ink_pixels': count, 'uncovered_ink_pixels': 0,
                                'original_crop': dict(lower.crop), 'owner_crop': dict(upper.crop)}
                    metadata = {**dict(lower.metadata), 'text_overlap_adjustment': decision}
                    crop = {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest(),
                            'source_path': lower.crop['path'], 'source_sha256': lower.crop['sha256']}
                    rows[lower.region_id] = replace(lower, bbox=box, crop=crop, metadata=metadata)
                    decisions.append(decision)
                    break
        finally:
            image.close()
    def updated(regions):
        return tuple(rows[r.region_id] for r in regions)
    return replace(plan, all_regions=updated(plan.all_regions), text_regions=updated(plan.text_regions)), decisions
