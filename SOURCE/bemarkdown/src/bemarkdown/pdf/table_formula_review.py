"""Keep weak in-table formula detections as review evidence, never recognition."""
from __future__ import annotations

import copy
import math


def _valid(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in box)
            and box[0] < box[2] and box[1] < box[3])


def _area(box):
    return (box[2]-box[0]) * (box[3]-box[1])


def retain_table_formula_review_hints(page_ir, detections, transform, *, capture_floor):
    tables = [r for r in page_ir.get('canonical_regions', [])
              if r.get('semantic_type') == 'TABLE' and _valid(r.get('bbox_pdf_pt'))]
    figures = [r['bbox_pdf_pt'] for r in page_ir.get('canonical_regions', [])
               if r.get('semantic_type') == 'IMAGE' and _valid(r.get('bbox_pdf_pt'))]
    threshold = float(page_ir['score_threshold'])
    retained, blocked = [], []
    for raw in detections:
        score = raw.get('raw_score')
        if raw.get('raw_label') != 'formula' or not isinstance(score, (int, float)):
            continue
        if not capture_floor <= score < threshold or not _valid(raw.get('raw_bbox_render_px')):
            continue
        box = transform.render_px_to_pdf_pt(raw['raw_bbox_render_px'])
        if not _valid(box):
            continue
        if any(min(box[2], f[2]) > max(box[0], f[0]) and
               min(box[3], f[3]) > max(box[1], f[1]) for f in figures):
            blocked.append(raw['raw_detection_id'])
            continue
        owners = [r for r in tables if r['bbox_pdf_pt'][0] <= box[0]
                  and r['bbox_pdf_pt'][1] <= box[1] and r['bbox_pdf_pt'][2] >= box[2]
                  and r['bbox_pdf_pt'][3] >= box[3] and _area(box) < _area(r['bbox_pdf_pt']) * .5]
        if not owners:
            continue
        owner = min(owners, key=lambda r: (_area(r['bbox_pdf_pt']), r['region_id']))
        hint = dict(source_raw_detection_ids=[raw['raw_detection_id']],
                    bbox_pdf_pt=list(box), raw_score=score,
                    raw_model_evidence=copy.deepcopy(raw),
                    canonical_score_threshold=threshold,
                    basis='BELOW_THRESHOLD_FORMULA_INSIDE_TABLE',
                    status='REVIEW_REQUIRED', recognition_success=False)
        owner.setdefault('provenance', {}).setdefault('table_formula_review_hints', []).append(hint)
        retained.append(raw['raw_detection_id'])
    return dict(schema='table-formula-review-hints-v1', retained_raw_detection_ids=retained,
                blocked_by_figure_raw_detection_ids=blocked, canonical_threshold_unchanged=True)
