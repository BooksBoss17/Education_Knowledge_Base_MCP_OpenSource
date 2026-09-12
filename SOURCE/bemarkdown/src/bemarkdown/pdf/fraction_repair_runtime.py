"""Optional fraction repair sharing the existing GOT and FormulaNet runtimes."""
import copy
import hashlib
from pathlib import Path
import time

from .text_evidence import TextRecognitionRequest
from .fraction_structure_repair import (
    source_fraction_risk, source_fraction_plan, bind_numerator_crop, verify_fraction_repair,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FractionRepairSession:
    def __init__(self, work_root, rows):
        from PIL import Image
        self.root = Path(work_root).resolve()
        self.requests, self.sources, self.outputs = [], {}, {}
        self.executed = False
        self.metrics = {'schema': 'bemarkdown-fraction-repair-runtime-v1', 'enabled': True,
                        'auxiliary_requests': 0, 'auxiliary_successful_outputs': 0,
                        'formula_requests': 0, 'formula_successful_outputs': 0,
                        'accepted_repairs': 0, 'source_rejections': 0, 'candidate_decisions': []}
        for row in rows:
            provenance = row.get('provenance', {})
            if provenance.get('recognition_backend') != 'FORMULANET' or not row.get('latex'):
                continue
            try:
                crop = provenance['crop']
                path = Path(crop['path']).resolve()
                path.relative_to(self.root)
                digest = _sha(path)
                if digest != crop['content_sha256']:
                    raise ValueError('FRACTION_SOURCE_SHA_MISMATCH')
                with Image.open(path) as image:
                    risk = source_fraction_risk(row['latex'], image)
                    size = image.size
            except (KeyError, TypeError, ValueError, OSError):
                self.metrics['source_rejections'] += 1
                continue
            self.metrics['candidate_decisions'].append({'content_id': row['content_id'],
                                                        'source_crop_sha256': digest, **risk})
            if not risk['required']:
                continue
            region_id = 'fraction-repair-' + row['content_id']
            if region_id in self.sources:
                raise ValueError('FRACTION_DUPLICATE_CONTENT_ID')
            self.sources[region_id] = {'row': row, 'path': path, 'digest': digest, 'primary': row['latex']}
            self.requests.append(TextRecognitionRequest(
                document_id=row['document_id'], page_id=str(row['page_index']), region_id=region_id,
                source_candidate_id=digest, bbox=(0, 0, *size), crop_ref=str(path), crop_sha256=digest,
                route_kind='FRACTION_STRUCTURE_CONTEXT', text_track='TEXT_REGION_CROP',
                source_provenance={'got_format': True, 'purpose': 'FRACTION_STRUCTURE_AUXILIARY'}))
        self.metrics['auxiliary_requests'] = len(self.requests)

    def wrap(self, runner):
        def run(provider_id, requests):
            if self.executed:
                raise ValueError('FRACTION_GOT_ALREADY_EXECUTED')
            self.executed = True
            combined = list(requests) + self.requests
            if len({r.region_id for r in combined}) != len(combined):
                raise ValueError('FRACTION_REQUEST_ID_COLLISION')
            values = list(runner(provider_id, combined))
            if len(values) != len(combined):
                raise ValueError('FRACTION_GOT_CARDINALITY')
            for request, value in zip(self.requests, values[len(requests):], strict=True):
                if value.get('source_crop_sha256') != request.crop_sha256:
                    raise ValueError('FRACTION_GOT_SOURCE_MISMATCH')
                self.outputs[request.region_id] = copy.deepcopy(value)
                if value.get('output_contract_status') == 'PASS' and value.get('normalized_output'):
                    self.metrics['auxiliary_successful_outputs'] += 1
            return values[:len(requests)]
        return run

    def flush(self, runner):
        if not self.executed and self.requests:
            self.wrap(runner)('C_GOT_OCR2', [])

    def apply(self, formula_factory):
        from PIL import Image
        started = time.perf_counter()
        ready = []
        crop_dir = self.root/'fraction_numerator_crops'
        for region_id, source in self.sources.items():
            output = self.outputs.get(region_id)
            if output is None:
                continue
            row = source['row']
            if _sha(source['path']) != source['digest'] or row['latex'] != source['primary']:
                raise ValueError('FRACTION_SOURCE_CHANGED_BEFORE_APPLY')
            evidence = {'formatted_evidence': copy.deepcopy(output),
                        'source_crop_sha256': source['digest'], 'primary_latex': source['primary']}
            row['provenance']['fraction_structure_repair'] = evidence
            if output.get('output_contract_status') != 'PASS' or not output.get('normalized_output'):
                evidence['status'] = 'FORMATTED_OUTPUT_UNAVAILABLE'
                continue
            with Image.open(source['path']) as image:
                plan = source_fraction_plan(source['primary'], str(output['normalized_output']), image,
                            candidate_truncated=output.get('generation_limit_reached') is not False)
                evidence['plan'] = plan
                evidence['status'] = plan['status']
                if plan['status'] == 'READY_FOR_LOCAL_NUMERATOR':
                    crop_dir.mkdir(exist_ok=True)
                    bind_numerator_crop(plan, image, crop_dir/f'{region_id}.png')
                    ready.append((row, evidence, plan))
        unique = {plan['numerator_crop_sha256']: Path(plan['numerator_crop_ref']) for _, _, plan in ready}
        self.metrics['formula_requests'] = len(unique)
        if unique:
            for digest, path in unique.items():
                if _sha(path) != digest:
                    raise ValueError('FRACTION_NUMERATOR_CHANGED_BEFORE_INFERENCE')
            runtime = formula_factory()
            predicted = list(runtime.predict(list(unique.values()), batch_size=3))
            if len(predicted) != len(unique):
                raise ValueError('FRACTION_FORMULA_CARDINALITY')
            local = dict(zip(unique, predicted, strict=True))
            self.metrics['formula_successful_outputs'] = sum(bool(v) for v in predicted)
            self.metrics['formula_runtime'] = runtime.fingerprint()
            for row, evidence, plan in ready:
                prediction = {'latex': local[plan['numerator_crop_sha256']],
                              'source_crop_sha256': plan['numerator_crop_sha256']}
                result = verify_fraction_repair(plan, prediction)
                evidence.update(local_output=prediction, verification=result, status=result['status'])
                if result['accepted']:
                    row['latex'] = result['text']
                    row['quality_metrics'].update(fraction_repair_accepted=True,
                        formula_assessments_apply_to='formula_raw_latex',
                        effective_latex_sha256=hashlib.sha256(result['text'].encode()).hexdigest())
                    # Old visual/consensus assessments remain attached to their raw input.
                    # A structural repair does not itself certify semantic correctness.
                    row['status'] = 'REVIEW_REQUIRED'
                    row['quality_status'] = 'FORMULA_CONTENT_REVIEW'
                    row['review_reasons'] = sorted(set(row.get('review_reasons', [])) |
                                                  {'SOURCE_FRACTION_REPAIR_REQUIRES_REVIEW'})
                    from ..pdf_content_quality import classify_review_reasons
                    row['review_taxonomy'] = classify_review_reasons(row['review_reasons'],
                        semantic_evidence=[row.get('semantic_hint', 'formula')])
                    self.metrics['accepted_repairs'] += 1
        self.metrics['local_repair_seconds'] = time.perf_counter()-started
