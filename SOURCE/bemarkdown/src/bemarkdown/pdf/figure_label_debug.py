"""Archive actual current-run figure OCR pixels for reproducible debug review."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _io_path(path):
    value = str(Path(path).absolute())
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return Path(value)


def archive_figure_label_inputs(document, debug_dir: Path):
    target = _io_path(debug_dir / 'figure_labels')
    rows = []
    copied = {}
    names = {}

    def preserve(path, expected_sha):
        if not path or not expected_sha:
            raise RuntimeError('FIGURE_DEBUG_SOURCE_REFERENCE_REQUIRED')
        payload = _io_path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise RuntimeError('FIGURE_DEBUG_SOURCE_SHA_MISMATCH')
        if expected_sha not in copied:
            name = f'i{len(copied) + 1:04d}.png'
            target.mkdir(parents=True, exist_ok=True)
            (target / name).write_bytes(payload)
            copied[expected_sha] = len(payload)
            names[expected_sha] = name
        return names[expected_sha]

    for block in [*document.get('blocks', []), *document.get('suppressed_blocks', [])]:
        evidence = block.get('provenance', {}).get('route_provenance', {}).get('figure_label_recognition')
        if not evidence:
            continue
        contracts = (evidence.get('three_model_text_evidence') or {}).get('unit_contracts', [])
        labels = {label['id']: label for label in evidence.get('labels', [])}
        for unit in contracts:
            label = labels[unit['unit_id']]
            proof = unit.get('assembly_provenance', {}).get('figure_label_isolation', {})
            row = {
                'node_id': block['node_id'], 'page_index': block['page_index'],
                'figure_bbox_pdf_pt': block['bbox_pdf_pt'], 'unit_id': unit['unit_id'],
                'bbox_pdf_pt': unit['bbox_pdf_pt'], 'bbox_pixel': unit['bbox_pixel'],
                'source_figure_sha256': unit['source_page_sha256'],
                'source_figure_ref': preserve(unit.get('source_page_ref'), unit['source_page_sha256']),
                'crop_sha256': unit['crop_sha256'],
                'crop_ref': preserve(unit.get('crop_ref'), unit['crop_sha256']),
                'text_a': unit['provider_a_text'], 'a_confidence': unit.get('provider_a_confidence'),
                'selected_text': label['text'], 'latex': label.get('latex'), 'status': label['status'],
            }
            if proof:
                row['original_crop_sha256'] = proof['original_crop_sha256']
                row['original_crop_ref'] = preserve(proof.get('original_crop_ref'), proof['original_crop_sha256'])
                row['isolation'] = {key: value for key, value in proof.items() if not key.endswith('_ref')}
            segmentation = unit.get('assembly_provenance', {}).get('figure_label_segmentation')
            if segmentation:
                row['segmentation_parent_sha256'] = segmentation['parent_crop_sha256']
                row['segmentation_parent_ref'] = preserve(segmentation['parent_crop_ref'], segmentation['parent_crop_sha256'])
                row['segmentation'] = {key: value for key, value in segmentation.items() if not key.endswith('_ref')}
            rows.append(row)
    result = {'schema': 'bemarkdown-actual-figure-inputs-v1', 'unit_count': len(rows),
              'unique_image_count': len(copied), 'image_bytes': sum(copied.values()), 'rows': rows}
    if rows:
        (target / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
