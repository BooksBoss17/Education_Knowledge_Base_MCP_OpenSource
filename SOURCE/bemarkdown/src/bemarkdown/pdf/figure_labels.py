"""Source-bound figure labels sharing the production text and formula models."""
from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image


FIGURE_LABEL_SCHEMA = 'bemarkdown-figure-label-recognition-v1'


def is_figure_label_route(route: dict[str, Any]) -> bool:
    return bool(route.get('provenance', {}).get('figure_label_parent_route_id'))


def prepare_figure_label_inputs(executor, state):
    """Keep materialized diagrams out of the text recognition queue.

    Non-text execution has already preserved each figure and established its
    ownership mask. A second OCR route would bypass that mask and reinterpret
    graphical marks as text. Native labels are also retained in the image.
    Formula and table routes remain independently classified upstream.
    """
    for parent in state['non_text_routes']:
        if parent['adapter'] not in {'IMAGE_RENDER_CROP', 'IMAGE_NATIVE_EXTRACT'}:
            continue
        row = state['rows_by_route'][str(parent['route_id'])]
        kinds = row.get('provenance', {}).get('source_candidate_kinds', [])
        if kinds == ['NATIVE_IMAGE_FALLBACK']:
            # Composite page storage is not a semantic diagram; its body text,
            # formulas, tables and figures retain their own canonical routes.
            continue
        if not row.get('binary_artifact_ref'):
            continue
        row.setdefault('provenance', {})['figure_preservation'] = {
            'schema': 'bemarkdown-figure-preservation-v1',
            'status': 'PRESERVED_AS_IMAGE',
            'text_recognition': 'NOT_REQUESTED',
            'reason': 'DIAGRAM_CONTENT_RETAINED_IN_IMAGE',
        }
    state['figure_label_routes'] = []
    return {}


def isolate_figure_label_lines(crop, lines, rec_model):
    """Re-recognize derived crops with graph ink/fringes removed locally.

    A/B/C all subsequently receive the same derived crop. The unmodified
    parent and original detector crop remain bound in the transformation proof.
    """
    import cv2
    import numpy as np

    path = Path(crop['path'])
    with Image.open(path) as source:
        image = source.convert('RGB')
    pixels = np.asarray(image)
    ink = (np.asarray(image.convert('L')) < 170).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    candidates = []
    for line in lines:
        x0, y0, x1, y1 = line['bbox_local_px']
        box = (max(0, int(x0)), max(0, int(y0)),
               min(image.width, math.ceil(x1)), min(image.height, math.ceil(y1)))
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        local = labels[y0:y1, x0:x1]
        removed, retained = [], []
        for label in set(np.unique(local)) - {0}:
            left, top, width, height, area = map(int, stats[label])
            contained = left >= x0-2 and top >= y0-2 and left+width <= x1+2 and top+height <= y1+2
            distant = width > 3.5*(x1-x0) or height > 3.5*(y1-y0)
            if not contained and distant:
                removed.append(int(label))
            elif contained and area >= 4:
                retained.append(height)
        if not removed or not any(h >= (y1-y0)*.2 for h in retained):
            continue
        original_path = Path(line['line_crop_ref'])
        original_sha = hashlib.sha256(original_path.read_bytes()).hexdigest()
        if original_sha != line['line_crop_sha256']:
            raise RuntimeError('FIGURE_LABEL_ORIGINAL_CROP_SHA_MISMATCH')
        original = pixels[y0:y1, x0:x1].copy()
        with Image.open(original_path) as original_image:
            same_pixels = np.array_equal(original, np.asarray(original_image.convert('RGB')))
        if not same_pixels:
            # Resized/warped detector crops need their own explicit transform.
            continue
        remove = np.isin(local, removed).astype(np.uint8)
        protect = ((local != 0) & (remove == 0)).astype(np.uint8)
        mask = cv2.dilate(remove, np.ones((5, 5), np.uint8)) & (
            1-cv2.dilate(protect, np.ones((3, 3), np.uint8)))
        derived = original.copy()
        derived[mask.astype(bool)] = 255
        if not np.array_equal(derived[protect.astype(bool)], original[protect.astype(bool)]):
            raise RuntimeError('FIGURE_LABEL_RETAINED_INK_CHANGED')
        from io import BytesIO
        stream = BytesIO()
        Image.fromarray(derived).save(stream, format='PNG')
        data = stream.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        target = path.parent / f'figure-label-sha256-{digest}.png'
        if not target.exists():
            target.write_bytes(data)
        proof = {
            'version': 'local-figure-graph-halo-removal-v1',
            'source_figure_sha256': crop['content_sha256'],
            'original_crop_ref': str(original_path), 'original_crop_sha256': original_sha,
            'derived_crop_sha256': digest, 'bbox_figure_px': list(box),
            'removed_component_bboxes_px': [
                [int(stats[k][0]), int(stats[k][1]), int(stats[k][0]+stats[k][2]), int(stats[k][1]+stats[k][3])]
                for k in removed],
            'fringe_radius_pixels': 2, 'glyph_protection_radius_pixels': 1,
            'removed_pixels': int(mask.sum()), 'retained_ink_pixels_identical': True,
            'original_a_text': str(line.get('text') or ''),
            'original_a_rec_confidence': line.get('rec_confidence'),
        }
        candidates.append((line, target, digest, proof))
    if not candidates:
        return {'candidate_count': 0, 'derived_count': 0, 'inference_seconds': 0.0}
    started = time.perf_counter()
    try:
        values = list(rec_model.predict([str(p) for _, p, _, _ in candidates], batch_size=16))
        if len(values) != len(candidates):
            raise RuntimeError('FIGURE_LABEL_RECOGNITION_CARDINALITY')
        from ..pdf_content_router import _model_result_dict
        parsed = [_model_result_dict(value) for value in values]
        texts = [(str(value.get('rec_text') or ''), float(value.get('rec_score') or 0)) for value in parsed]
        for (line, target, digest, proof), (text, confidence) in zip(candidates, texts, strict=True):
            line.update(text=text, rec_confidence=confidence,
                        confidence=min(float(line.get('det_confidence', 1)), confidence),
                        line_crop_ref=str(target), line_crop_sha256=digest,
                        figure_label_isolation=proof)
        return {'candidate_count': len(candidates), 'derived_count': len(candidates),
                'inference_seconds': time.perf_counter()-started}
    except Exception as exc:
        # Original A results and original B/C inputs remain aligned on failure.
        return {'candidate_count': len(candidates), 'derived_count': 0,
                'inference_seconds': time.perf_counter()-started,
                'error': f'{type(exc).__name__}:{exc}'}


def _mathematical_label(text):
    value = text.strip()
    words = re.findall(r'[A-Za-z]+', value)
    if not words or max(map(len, words)) > 3 or re.search(r'[\u3400-\u9fff]', value):
        return False
    return bool(re.search(r'[\d/_]', value) or re.fullmatch(r'[A-Z][A-Za-z]{1,2}', value))


def _predict_figure_math(requests, factory, *, batch_size=8):
    started = time.perf_counter()
    runtime = None
    try:
        for request in requests:
            if hashlib.sha256(request['path'].read_bytes()).hexdigest() != request['crop_sha256']:
                raise RuntimeError('FIGURE_MATH_CROP_SHA_MISMATCH')
        runtime = factory()
        predictions = {}
        unique_inputs = 0
        # Keep formula/symbol groups separate. Reuse only byte-identical inputs
        # within this call; every source occurrence was verified above.
        for kind in ('formula', 'symbol'):
            group = [r for r in requests if r.get('kind', 'formula') == kind]
            if not group:
                continue
            unique, index_by_sha = [], {}
            for request in group:
                digest = request['crop_sha256']
                if digest not in index_by_sha:
                    index_by_sha[digest] = len(unique)
                    unique.append(request)
            values = list(runtime.predict([r['path'] for r in unique], batch_size=batch_size))
            if len(values) != len(unique):
                raise RuntimeError('FIGURE_MATH_OUTPUT_CARDINALITY')
            unique_inputs += len(unique)
            predictions.update((r['id'], values[index_by_sha[r['crop_sha256']]]) for r in group)
        fingerprint = runtime.fingerprint()
        return {'outputs': {key: {'latex': value, 'runtime_fingerprint': fingerprint}
                            for key, value in predictions.items()},
                'exact_duplicate_execution': {
                    'requested_inputs': len(requests), 'unique_inputs': unique_inputs,
                    'reused_inputs': len(requests)-unique_inputs,
                    'scope': 'CURRENT_FIGURE_CALL_SAME_KIND_AND_VERIFIED_CROP_BYTES',
                    'cross_document_cache': False,
                },
                'seconds': time.perf_counter()-started}
    except Exception as exc:
        return {'outputs': {}, 'error': f'{type(exc).__name__}:{exc}',
                'seconds': time.perf_counter()-started}
    finally:
        release_workspace = getattr(runtime, 'release_workspace', None)
        if callable(release_workspace):
            release_workspace()
        del runtime
        import gc
        gc.collect()


def figure_math_prefetch_requests(plans, crops_by_route, outputs_by_route):
    from ..text_recognition_contract import normalize_text
    from .figure_symbols import symbol_family
    from .figure_waveforms import waveform_evidence, waveform_text_candidate
    from .page_visual_text_recovery import PageVisualTextRecoveryBoundedCropper
    requests = []
    for plan in plans:
        for route in plan['routes']:
            if not is_figure_label_route(route):
                continue
            route_id = str(route['route_id'])
            lines = outputs_by_route.get(route_id, [])
            if not lines:
                continue
            units = PageVisualTextRecoveryBoundedCropper().prepare(route, crops_by_route[route_id], lines).units
            for unit in units:
                primary = normalize_text(unit.provider_a_text)
                mathematical = _mathematical_label(primary)
                if mathematical and waveform_text_candidate(primary) and waveform_evidence(unit.crop_ref) is not None:
                    continue
                if mathematical or symbol_family(primary) is not None:
                    requests.append({'id': unit.unit_id, 'path': Path(unit.crop_ref),
                        'crop_sha256': unit.crop_sha256, 'kind': 'formula' if mathematical else 'symbol',
                        'text_a': primary, 'confidence_a': unit.provider_a_confidence})
    return requests


class FigureMathPrefetch:
    """One Paddle model instance overlaps only the separate GOT process."""
    def __init__(self, requests, factory, executor, *, select_requests=None, caller_thread_prediction=False):
        self.requests, self.factory, self.executor = requests, factory, executor
        self.select_requests = select_requests
        self.selected = False
        self.caller_thread_prediction = caller_thread_prediction
        self.caller_result = None
        self.c_submitted_before_formula = False
        self.future = None
        self.started_before_c = False
        self.batch_size = 8

    def overlap_runner(self, runner):
        def call(provider_id, requests):
            self._select()
            if self.future is None and self.caller_result is None:
                # Keep FormulaNet activations within the shared GPU budget while
                # GOT runs. Non-overlapping formula work retains the larger batch.
                self.batch_size = 2
            if self.caller_thread_prediction and self.requests:
                # Paddle predictors remain on their creating/calling thread. Only
                # the separate GOT process and its pipe I/O run on the helper thread.
                running_c = self.executor.submit(runner, provider_id, requests)
                self.c_submitted_before_formula = True
                self.start()
                return running_c.result()
            self.start()
            self.started_before_c = bool(self.requests)
            return runner(provider_id, requests)
        return call

    def _select(self):
        if not self.selected:
            if self.select_requests is not None:
                self.requests = self.select_requests(self.requests)
            self.selected = True

    def start(self):
        self._select()
        if self.caller_thread_prediction:
            if self.requests and self.caller_result is None:
                self.caller_result = _predict_figure_math(self.requests, self.factory, batch_size=self.batch_size)
            return
        if self.requests and self.future is None:
            self.future = self.executor.submit(_predict_figure_math, self.requests, self.factory,
                                               batch_size=self.batch_size)

    def result(self):
        self.start()
        started = time.perf_counter()
        result = (self.caller_result if self.caller_result is not None else
                  self.future.result() if self.future is not None else {'outputs': {}, 'seconds': 0.0})
        return {**result, 'wait_after_c_seconds': time.perf_counter()-started,
                'started_before_c': self.started_before_c, 'requested_count': len(self.requests),
                'batch_size': self.batch_size,
                'c_submitted_before_formula': self.c_submitted_before_formula,
                'formula_thread_mode': 'CALLER' if self.caller_thread_prediction else 'ISOLATED_MODEL_THREAD'}


def build_figure_label_results(staged, crops_by_route, *, formula_runtime_factory, prefetched=None):
    """Attach A/B/C resolutions and mathematical-label evidence to images."""
    from .figure_graphic_markers import colored_cross_marker_ids
    from .figure_symbols import (
        SINGLE_SYMBOL_GATE_VERSION,
        accepts_symbol_candidate,
        needs_symbol_verification,
    )
    from .figure_waveforms import waveform_marker_evidence
    results, mathematical, symbols = {}, [], {}
    math_text_evidence = {}
    for route_id, crop in crops_by_route.items():
        if not crop.get('figure_label_parent_route_id'):
            continue
        evidence = staged.runtime.provenance_by_route.get(route_id)
        result = {
            'schema': FIGURE_LABEL_SCHEMA, 'parent_route_id': crop['figure_label_parent_route_id'],
            'status': 'TEXT_DETECTED' if evidence else 'NO_TEXT_DETECTED',
            'source_crop_sha256': crop['content_sha256'], 'source_bbox_pdf_pt': list(crop['bbox_pdf_pt']),
            'labels': [], 'three_model_text_evidence': copy.deepcopy(evidence),
        }
        results[route_id] = result
        if not evidence:
            continue
        markers = colored_cross_marker_ids(evidence.get('bounded_units', []))
        waves = waveform_marker_evidence(evidence.get('bounded_units', []))
        for unit in evidence.get('bounded_units', []):
            request, resolution = unit['request'], unit['resolver']
            math_text_evidence[str(request['region_id'])] = unit['evidence']
            value = str(resolution.get('selected_text') or '')
            primary = str(unit['evidence']['A'].get('normalized_text') or '')
            label = {'id': str(request['region_id']), 'text': value, 'latex': None,
                     'bbox_pdf_pt': list(request['bbox']), 'crop_sha256': request['crop_sha256'],
                     'status': str(resolution['resolution_status']),
                     'review_reasons': list(resolution.get('review_reasons', []))}
            result['labels'].append(label)
            if label['id'] in markers or label['id'] in waves:
                label.update(role='GRAPHIC_MARKER', status='SOURCE_GRAPHIC_MARKER',
                             graphic_marker_basis='source-periodic-waveform-v1' if label['id'] in waves
                             else 'source-cyan-cross-lattice-v2')
                if label['id'] in waves:
                    label['graphic_marker_evidence'] = waves[label['id']]
                continue
            if _mathematical_label(primary) or _mathematical_label(value):
                mathematical.append((label, Path(request['crop_ref'])))
            elif needs_symbol_verification(primary, unit['evidence']['A'].get('confidence'),
                                           unit['evidence']['B'].get('normalized_text', '')):
                mathematical.append((label, Path(request['crop_ref'])))
                symbols[label['id']] = primary
    graphic_count = sum(label.get('role') == 'GRAPHIC_MARKER' for r in results.values() for label in r['labels'])
    metrics = {'figure_count': len(results), 'label_count': sum(len(r['labels']) for r in results.values()) - graphic_count,
               'graphic_marker_count': graphic_count,
        'math_label_count': len(mathematical), 'symbol_verification_count': len(symbols),
        'formula_failure_count': 0}
    if mathematical:
        from ..formula_ocr import FormulaOcrSafetyGate
        from ..formulanet_runtime import FormulaOcrOutputValidator
        started = time.perf_counter()
        try:
            predictions = dict((prefetched or {}).get('outputs', {}))
            pending = [{'id': label['id'], 'path': path, 'crop_sha256': label['crop_sha256'],
                        'kind': 'symbol' if label['id'] in symbols else 'formula'}
                       for label, path in mathematical if label['id'] not in predictions]
            metrics['prefetched_math_count'] = len(mathematical)-len(pending)
            if pending:
                result = _predict_figure_math(pending, formula_runtime_factory)
                if result.get('error'):
                    raise RuntimeError(result['error'])
                predictions.update(result['outputs'])
            for label, path in mathematical:
                prediction = predictions[label['id']]
                latex = prediction['latex']
                validation = FormulaOcrOutputValidator().validate(latex)
                with Image.open(path) as image:
                    decision = FormulaOcrSafetyGate().evaluate(
                        width=image.width, height=image.height, raw_latex=latex, validation=validation)
                label['formula_evidence'] = {'raw_latex': latex, 'source_crop_sha256': label['crop_sha256'],
                    'runtime_fingerprint': prediction['runtime_fingerprint'], 'gate': decision.to_dict()}
                if label['id'] in symbols:
                    accepted = accepts_symbol_candidate(symbols[label['id']], latex)
                    label['formula_evidence']['single_symbol_gate'] = {
                        'version': SINGLE_SYMBOL_GATE_VERSION,
                        'provider_a_text': symbols[label['id']], 'accepted': accepted}
                    if not accepted:
                        continue
                elif decision.verdict.value != 'REJECT_PRESERVE_IMAGE':
                    from .figure_formula_support import figure_formula_content_support
                    support = figure_formula_content_support(latex, math_text_evidence[label['id']])
                    label['formula_evidence']['text_token_support'] = support
                    if not support['supported']:
                        label['status'] = 'FORMULA_REVIEW_REQUIRED'
                        label['review_reasons'] = sorted(set(label['review_reasons']) | {'FIGURE_FORMULA_TEXT_TOKEN_DISAGREEMENT'})
                        metrics['formula_content_review_count'] = metrics.get('formula_content_review_count', 0) + 1
                        continue
                if decision.verdict.value != 'REJECT_PRESERVE_IMAGE':
                    label['latex'] = latex
                    label['status'] = 'FORMULA_RECOGNIZED' if decision.verdict.value == 'ACCEPT' and not label['review_reasons'] else 'FORMULA_REVIEW_REQUIRED'
                label['review_reasons'] = sorted(set(label['review_reasons']) | {r['code'] for r in decision.reasons})
        except Exception as exc:
            metrics['formula_failure_count'] = len(mathematical)
            for label, _ in mathematical:
                label['status'] = 'FORMULA_REVIEW_REQUIRED'
                label['review_reasons'] = sorted(set(label['review_reasons']) | {'FIGURE_FORMULA_RUNTIME_FAILURE'})
                label['formula_error'] = f'{type(exc).__name__}:{exc}'
        finally:
            metrics['formula_seconds'] = time.perf_counter()-started
    staged.runtime.figure_label_results = results
    staged.metrics['figure_labels'] = metrics


def attach_figure_label_results(state, runtime):
    results = getattr(runtime, 'figure_label_results', {})
    for route in state.get('figure_label_routes', []):
        route_id = str(route['route_id'])
        parent_id = route['provenance']['figure_label_parent_route_id']
        row = state['rows_by_route'][parent_id]
        result = results.get(route_id)
        if result is None:
            result = {'schema': FIGURE_LABEL_SCHEMA, 'status': 'RUNTIME_RESULT_UNAVAILABLE', 'labels': []}
        row['figure_labels'] = [{k: copy.deepcopy(v) for k, v in label.items()
                                 if k not in {'formula_evidence', 'formula_error'}} for label in result['labels']
                                if label.get('role') != 'GRAPHIC_MARKER']
        row.setdefault('provenance', {})['figure_label_recognition'] = copy.deepcopy(result)
        if result['status'] == 'RUNTIME_RESULT_UNAVAILABLE' or any(
            'REQUIRED' in label['status'] or label.get('review_reasons') for label in result['labels']
        ):
            row['review_reasons'] = sorted(set(row.get('review_reasons', [])) | {'FIGURE_LABEL_REVIEW_REQUIRED'})


def figure_alt_text(content, *, escape_brackets=True):
    parts = []
    pending = 0
    for label in content.get('figure_labels', []):
        if label.get('role') == 'GRAPHIC_MARKER':
            continue
        if 'REQUIRED' in str(label.get('status', '')) or label.get('review_reasons'):
            pending += 1
            continue
        value = f"${label['latex']}$" if label.get('latex') else str(label.get('text') or '')
        if value:
            text = re.sub(r'\s+', ' ', value)
            if escape_brackets:
                text = text.replace('[', r'\[').replace(']', r'\]')
            parts.append(text)
    if pending:
        if parts:
            return '图（图内文字：' + '；'.join(parts) + f'；另有{pending}处文字待识别）'
        return f'图（{pending}处文字待识别）'
    return '图（图内文字：' + '；'.join(parts) + '）' if parts else '图'
