"""Optional source-bound math recovery after ordinary text resolution.

Formatted line evidence is auxiliary, never a vote. Only locally corroborated
math spans can change; original resolver and provider evidence remain intact.
"""

import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import re
import time

from .inline_math_repair import align_formatted_fields, apply_verified_math, build_source_proposals
from .ocr_character_geometry import CharacterLayout
from .text_evidence import TextRecognitionRequest


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_selected_text(provenance):
    original = str(provenance['resolver'].get('selected_text') or '')
    repair = provenance.get('inline_math_repair')
    if not repair or not repair.get('accepted'):
        return original
    if hashlib.sha256(original.encode()).hexdigest() != repair['original_text_sha256']:
        raise ValueError('INLINE_MATH_ORIGINAL_TEXT_CHANGED')
    reconstructed = original
    right = len(original)
    for field in sorted(repair['accepted'], key=lambda item: item['span'][0], reverse=True):
        start, end = field['span']
        if not 0 <= start < end <= right or original[start:end] != field['previous_text']:
            raise ValueError('INLINE_MATH_ACCEPTED_SPAN_CHANGED')
        replacement = '$' + field['candidate_latex'] + '$'
        if replacement != field['replacement']:
            raise ValueError('INLINE_MATH_REPLACEMENT_CHANGED')
        reconstructed = reconstructed[:start] + replacement + reconstructed[end:]
        right = start
    if reconstructed != repair['text']:
        raise ValueError('INLINE_MATH_OUTPUT_CHANGED')
    return reconstructed


class InlineMathSession:
    def __init__(self, work_root, outputs_by_route, *, candidate_policy='all', atomic_superscripts=False):
        if candidate_policy not in ('all', 'source-risk-v3'):
            raise ValueError('INLINE_MATH_CANDIDATE_POLICY_INVALID')
        self.root = Path(work_root).resolve()
        self.requests = []
        self.sources = {}
        self.outputs = {}
        self.atomic_requests, self.atomic_sources, self.atomic_outputs = [], {}, {}
        self.executed = False
        self.metrics = {'schema': 'bemarkdown-inline-math-runtime-v1', 'enabled': True,
            'auxiliary_requests': 0, 'auxiliary_successful_outputs': 0, 'source_rejections': 0,
            'formula_requests': 0, 'formula_successful_outputs': 0,
            'accepted_fields': 0, 'rejected_fields': 0, 'matched_units': 0,
            'unmatched_source_lines': 0}
        self.metrics.update(candidate_policy=candidate_policy, candidate_decisions=[])
        self.metrics.update(atomic_superscripts=atomic_superscripts, atomic_auxiliary_requests=0,
                            atomic_auxiliary_successful_outputs=0, atomic_accepted_fields=0)
        for lines in outputs_by_route.values():
            for line in lines:
                text = str(line.get('text') or '')
                if not re.search(r'[\u4e00-\u9fff]', text) or not re.search(r'[A-Za-z\u0370-\u03ff]', text):
                    continue
                geometry = line.get('character_layout')
                if not geometry:
                    continue
                try:
                    path = Path(line['line_crop_ref']).resolve()
                    path.relative_to(self.root)
                    digest = _sha(path)
                    if digest != line['line_crop_sha256'] or digest != geometry['source_crop_sha256']:
                        raise ValueError('INLINE_MATH_SOURCE_SHA_MISMATCH')
                    layout = CharacterLayout(geometry['text'], int(geometry['width']), int(geometry['height']),
                                             tuple(geometry['centers']))
                    if layout.text != text or len(layout.centers) != len(text):
                        raise ValueError('INLINE_MATH_SOURCE_LAYOUT_MISMATCH')
                except (KeyError, TypeError, ValueError, OSError):
                    self.metrics['source_rejections'] += 1
                    continue
                if digest in self.sources:
                    # Identical pixels with inconsistent recognition cannot share source geometry.
                    if self.sources[digest]['layout'] != layout:
                        self.sources[digest]['conflict'] = True
                        self.metrics['source_rejections'] += 1
                    continue
                self.sources[digest] = {'path': path, 'layout': layout}
        for digest, source in self.sources.items():
            if source.get('conflict'):
                continue
            layout = source['layout']
            if candidate_policy == 'source-risk-v3':
                from PIL import Image
                from .inline_math_candidates import formatted_context_risk
                with Image.open(source['path']) as image:
                    risk = formatted_context_risk(layout, image)
                self.metrics['candidate_decisions'].append({'source_crop_sha256': digest,
                    'text': layout.text, 'character_layout': asdict(layout), **risk})
                if not risk['required']:
                    continue
            self.requests.append(TextRecognitionRequest(
                document_id='current-document-inline-math', page_id='source-line',
                region_id='inline-math-' + digest, source_candidate_id=digest,
                bbox=(0, 0, layout.width, layout.height), crop_ref=str(source['path']), crop_sha256=digest,
                route_kind='INLINE_MATH_CONTEXT', text_track='TEXT_REGION_CROP',
                source_provenance={'got_format': True, 'purpose': 'INLINE_MATH_AUXILIARY'}))
        if atomic_superscripts:
            from PIL import Image
            from .inline_atomic_superscript import source_superscript_candidates
            atomic_dir = self.root/'inline_atomic_crops'
            for request in self.requests:
                source = self.sources[request.crop_sha256]
                with Image.open(source['path']) as image:
                    candidates = source_superscript_candidates(source['layout'], image)
                for candidate in candidates:
                    crop = candidate.pop('crop')
                    atomic_dir.mkdir(exist_ok=True)
                    path = atomic_dir/f'{request.crop_sha256}-{candidate["span"][0]}.png'
                    crop.save(path)
                    digest = _sha(path)
                    if digest in self.atomic_sources:
                        continue
                    self.atomic_sources[digest] = {'path': path, 'source_line_sha256': request.crop_sha256, **candidate}
                    self.atomic_requests.append(TextRecognitionRequest(
                        document_id=request.document_id, page_id=request.page_id,
                        region_id='inline-atomic-'+digest, source_candidate_id=request.crop_sha256,
                        bbox=(0, 0, *crop.size), crop_ref=str(path), crop_sha256=digest,
                        route_kind='INLINE_ATOMIC_SUPERSCRIPT', text_track='TEXT_REGION_CROP',
                        source_provenance={'got_format': True, 'purpose': 'INLINE_ATOMIC_SUPERSCRIPT_AUXILIARY'}))
        self.metrics['context_auxiliary_requests'] = len(self.requests)
        self.metrics['atomic_auxiliary_requests'] = len(self.atomic_requests)
        self.metrics['auxiliary_requests'] = len(self.requests)+len(self.atomic_requests)

    def wrap(self, runner):
        def run(provider_id, requests):
            if self.executed:
                raise ValueError('INLINE_MATH_GOT_ALREADY_EXECUTED')
            self.executed = True
            auxiliary = self.requests + self.atomic_requests
            combined = list(requests) + auxiliary
            if len({r.region_id for r in combined}) != len(combined):
                raise ValueError('INLINE_MATH_REQUEST_ID_COLLISION')
            values = list(runner(provider_id, combined))
            if len(values) != len(combined):
                raise ValueError('INLINE_MATH_GOT_CARDINALITY')
            for request, value in zip(auxiliary, values[len(requests):], strict=True):
                if value.get('source_crop_sha256') != request.crop_sha256:
                    raise ValueError('INLINE_MATH_GOT_SOURCE_MISMATCH')
                target = self.atomic_outputs if request.route_kind == 'INLINE_ATOMIC_SUPERSCRIPT' else self.outputs
                target[request.crop_sha256] = copy.deepcopy(value)
                if value.get('output_contract_status') == 'PASS' and value.get('normalized_output'):
                    self.metrics['auxiliary_successful_outputs'] += 1
                    if request.route_kind == 'INLINE_ATOMIC_SUPERSCRIPT':
                        self.metrics['atomic_auxiliary_successful_outputs'] += 1
            return values[:len(requests)]
        return run

    def flush(self, runner):
        if not self.executed and (self.requests or self.atomic_requests):
            self.wrap(runner)('C_GOT_OCR2', [])

    def apply(self, provenance_by_route, formula_factory):
        from PIL import Image, ImageOps

        started = time.perf_counter()
        targets = []
        seen = set()
        crop_dir = self.root / 'inline_math_crops'
        for route in provenance_by_route.values():
            for unit in route.get('bounded_units', [route]):
                request = unit['request']
                digest = request['crop_sha256']
                source = self.sources.get(digest)
                output = self.outputs.get(digest)
                if not source or source.get('conflict') or not output:
                    continue
                if output.get('output_contract_status') != 'PASS':
                    continue
                path = Path(request['crop_ref']).resolve()
                path.relative_to(self.root)
                if _sha(path) != digest or _sha(source['path']) != digest:
                    raise ValueError('INLINE_MATH_RUNTIME_SOURCE_CHANGED')
                seen.add(digest)
                self.metrics['matched_units'] += 1
                formatted = str(output['normalized_output'])
                original = str(unit['resolver'].get('selected_text') or '')
                fields = align_formatted_fields(original, formatted)
                source_fields = align_formatted_fields(source['layout'].text, formatted)
                if fields is None or source_fields is None or len(fields) != len(source_fields):
                    unit['inline_math_repair'] = {
                        'status': 'BODY_ALIGNMENT_FAILED', 'accepted': [], 'rejected': [],
                        'text': original, 'original_text_sha256': hashlib.sha256(original.encode()).hexdigest(),
                        'source_line_sha256': digest, 'formatted_evidence': copy.deepcopy(output),
                        'source_character_layout': asdict(source['layout']),
                        'proposals': [], 'local_predictions': [],
                    }
                    continue
                with Image.open(source['path']) as image:
                    proposals = build_source_proposals(source['layout'], image, formatted)
                    for proposal in proposals:
                        if proposal['status'] != 'READY_FOR_LOCAL_FORMULA':
                            continue
                        crop_dir.mkdir(exist_ok=True)
                        crop = crop_dir / f'{digest}-{proposal["field_index"]}.png'
                        ImageOps.expand(image.crop(proposal['bbox_source_px']), border=4, fill='white').save(crop)
                        proposal['crop_sha256'] = _sha(crop)
                        proposal['crop_ref'] = str(crop)
                targets.append((unit, original, formatted, proposals, digest, output))
        self.metrics['unmatched_source_lines'] = len(self.requests) - len(seen)
        unique = {}
        for _, _, _, proposals, _, _ in targets:
            for proposal in proposals:
                if proposal.get('crop_ref'):
                    unique.setdefault(proposal['crop_sha256'], Path(proposal['crop_ref']))
        predictions = {}
        self.metrics['formula_requests'] = len(unique)
        if unique:
            runtime = formula_factory()
            try:
                for digest, path in unique.items():
                    if _sha(path) != digest:
                        raise ValueError('INLINE_MATH_FORMULA_SOURCE_CHANGED')
                values = list(runtime.predict(list(unique.values()), batch_size=3))
                if len(values) != len(unique):
                    raise ValueError('INLINE_MATH_FORMULA_CARDINALITY')
                predictions = dict(zip(unique, values, strict=True))
                self.metrics['formula_successful_outputs'] = sum(bool(v) for v in values)
                fingerprint = getattr(runtime, 'fingerprint', None)
                if callable(fingerprint):
                    self.metrics['formula_runtime'] = fingerprint()
            finally:
                release = getattr(runtime, 'release_workspace', None)
                if callable(release):
                    release()
        for unit, original, formatted, proposals, source_sha, output in targets:
            local = [{'field_index': p['field_index'], 'crop_sha256': p['crop_sha256'],
                      'latex': predictions[p['crop_sha256']]} for p in proposals if p.get('crop_ref')]
            repair = apply_verified_math(original, formatted, proposals, local,
                                         alternative_verifier=self._verify_atomic_alternative)
            repair.update(source_line_sha256=source_sha, formatted_evidence=copy.deepcopy(output),
                          source_character_layout=asdict(self.sources[source_sha]['layout']),
                          proposals=proposals, local_predictions=local)
            unit['inline_math_repair'] = repair
            self.metrics['accepted_fields'] += len(repair['accepted'])
            self.metrics['atomic_accepted_fields'] += sum('atomic_superscript_evidence' in row for row in repair['accepted'])
            self.metrics['rejected_fields'] += len(repair['rejected'])
        self.metrics['local_repair_seconds'] = time.perf_counter() - started
        if self.atomic_requests:
            self.metrics['atomic_predictions'] = [
                {'crop_sha256': digest, 'source_line_sha256': self.atomic_sources[digest]['source_line_sha256'],
                 'geometry': self.atomic_sources[digest]['geometry'], 'output': value}
                for digest, value in self.atomic_outputs.items()]

    def _verify_atomic_alternative(self, field, proposal, local):
        from PIL import Image
        from .inline_atomic_superscript import verify_atomic_superscript
        digest = proposal.get('crop_sha256')
        source, output = self.atomic_sources.get(digest), self.atomic_outputs.get(digest)
        if not source or not output:
            return None
        if _sha(source['path']) != digest or _sha(proposal['crop_ref']) != digest:
            raise ValueError('INLINE_ATOMIC_SOURCE_CHANGED')
        with Image.open(source['path']) as image:
            return verify_atomic_superscript(field['candidate_latex'], local['latex'], output, image,
                                            crop_sha256=digest)
