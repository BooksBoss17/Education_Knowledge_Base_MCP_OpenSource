"""Label-blind, source-qualified whole-corpus positive component evidence only."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import pymupdf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_zero99_primitive_probe import primitive_facts, read, rows, sha, write
from zero99_registered_component import registered_iou, positive_component

P = Path(__file__).resolve().parents[1]
R = P.parents[1]


def source_view(raw, nodes, occurrence):
    box = occurrence['bbox']
    candidates = []
    for node in nodes:
        if node['page_index'] != occurrence['page'] or node['kind'] in ('IMAGE', 'OTHER'):
            continue
        value = raw[node['byte_start']:node['byte_end']]
        if node.get('sha256'):
            assert hashlib.sha256(value).hexdigest() == node['sha256']
        text = value.decode('utf-8')
        if not text.strip():
            continue
        b = node['bbox_pdf_pt']
        gap = max(box[0]-b[2], b[0]-box[2], 0) + max(box[1]-b[3], b[1]-box[3], 0)
        candidates.append({'bbox': b, 'text': text, 'start': node['byte_start'], 'gap': gap})
    candidates.sort(key=lambda x: (x['gap'], x['start']))
    chosen, chars = [], 0
    for item in candidates:
        if len(chosen) >= 20 or (chosen and chars >= 1000):
            break
        chosen.append(item)
        chars += len(item['text'])
    chosen.sort(key=lambda x: x['start'])
    return {'target_bbox': box, 'original_MD_text_blocks': [{'bbox': x['bbox'], 'text': x['text']} for x in chosen]}


def simple_match(drawing, box):
    if max(abs(a-b) for a, b in zip(box, drawing['rect'])) > .5 or drawing.get('fill') is None:
        return False
    kinds = collections.Counter(x[0] for x in drawing['items'])
    return kinds == {'l': 3} or kinds == {'re': 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--baseline', required=True)
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    assert run.is_relative_to(R / 'tmp/rt-train/runs')
    out = run
    out.mkdir(exist_ok=True)
    assert not (out / 'freeze.json').exists(), 'COMPLETE_STAGE_MUST_NOT_RERUN'
    completed = {x['sample_id']: x for x in rows(out / 'evidence.jsonl')} if (out / 'evidence.jsonl').exists() else {}
    baseline_path = Path(args.baseline).resolve()
    cohort = {x['sample_id'] for x in rows(baseline_path)}
    baseline_freeze = read(baseline_path.with_suffix('.freeze.json'))
    assert sha(baseline_path) == baseline_freeze['prediction_file_sha256']
    baseline = {x['sample_id']: x for x in rows(baseline_path)}
    originals = [x for x in rows(P / 'data/prepared/v2/inputs.jsonl') if x['sample_id'] in cohort]
    bindings = {x['sample_id']: x for x in rows(P / 'data/prepared/v2/binding-proof.jsonl')}
    resolver = {x['image_ref']: x['object_path'] for x in rows(P / 'data/splits/v2/resolver.jsonl')}
    books = {x['book_id']: x for x in read(P / 'data/manifests/source-inventory.json')['books']}
    eligible = [x for x in originals if baseline[x['sample_id']]['final_decision'] == 'D'
                and x['image']['width'] * x['image']['height'] <= 262144]
    old_evidence, old_inputs, old_visibility = {}, {}, {}
    registry = {'status': 'PREREGISTERED_ALL_SOURCE_QUALIFIED_KEEP_ONLY_COMPONENTS',
        'baseline_freeze': baseline_freeze, 'max_crop_pixels': 262144,
        'eligibility': 'current final D, native filled rectangle/closed three-line shape matching own source bbox within .5pt',
        'registered_shape_iou_minimum': .8, 'visible_colour_fraction_minimum': .8,
        'symbol_question_and_connected_stroke_rule_unchanged': True,
        'quality_gate': {'FN_strictly_less_than': 10, 'FP_at_most': 184, 'new_K_to_D': 0},
        'source_sha256': sha(__file__), 'rule_sha256': sha(P / 'tools/zero99_registered_component.py'),
        'primitive_source_sha256': sha(P / 'tools/build_zero99_primitive_probe.py'),
        'old_evidence_sha256': None,
        'cached_witness_reuse_requires_same_input_image_and_source_pdf': False, 'prior_witness_cache_disabled': True,
        'initial_size_eligible_count': len(eligible), 'gold_opened': False}
    if (out / 'preregistration.json').exists():
        previous = read(out / 'preregistration.json')
        for key in ('baseline_freeze', 'rule_sha256', 'primitive_source_sha256', 'old_evidence_sha256', 'quality_gate'):
            assert previous[key] == registry[key], 'RESUME_STRATEGY_OR_INPUT_CHANGED'
        assert set(completed) <= {x['sample_id'] for x in eligible}
        write(out / 'resume-amendment.json', {'reason': 'Add required target_bbox to source view; keep rule and completed results unchanged',
            'source_sha256': sha(__file__), 'previous_source_sha256': previous['source_sha256'],
            'completed_records_reused': len(completed), 'complete_records_sha256': sha(out / 'evidence.jsonl'),
            'original_preregistration_preserved': True, 'labels_opened': False})
    else:
        write(out / 'preregistration.json', registry)
    grouped = collections.defaultdict(list)
    for row in eligible:
        if row['sample_id'] not in completed:
            grouped[row['book_id']].append(row)
    counts = collections.Counter()
    counts['processed_assets'] = len(completed)
    counts['resumed_complete_assets'] = len(completed)
    counts['promoted_assets'] = sum(x['proposed_final'] == 'K' for x in completed.values())
    started = time.monotonic()
    with (out / 'evidence.jsonl').open('a', encoding='utf-8', buffering=1) as evidence_stream:
        for book, book_rows in grouped.items():
            prepared = json.loads((R / books[book]['inputs_json']).read_text(encoding='utf-8-sig'))
            conversion_path = Path(prepared['package']) / 'conversion_report.json'
            conversion = read(conversion_path)
            pdf_path = Path(conversion['source']['path'])
            pdf_sha = sha(pdf_path)
            assert pdf_sha == conversion['source']['sha256']
            raw = (P / book_rows[0]['document']['ref']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == book_rows[0]['document']['sha256']
            nodes = conversion['markdown_render']['node_spans']
            page_cache = collections.OrderedDict()
            with pymupdf.open(pdf_path) as pdf:
                for row in book_rows:
                    sid = row['sample_id']
                    witnesses, proofs = [], []
                    cached = sid in old_evidence and sid in old_inputs and old_inputs[sid]['image'] == row['image']
                    cached = cached and all(x['source_pdf_sha256'] == pdf_sha for x in old_visibility[sid]['uses'])
                    if cached:
                        proofs = old_evidence[sid]['uses']
                        witnesses = [dict(x['witness'], use_id=x['use_id']) for x in proofs if x.get('witness')]
                        counts['reused_previous_assets'] += 1
                    else:
                        rgb = None
                        for usage in bindings[sid]['uses']:
                            occurrence = usage['occurrence']
                            page_id = occurrence['page']
                            if page_id not in page_cache:
                                if len(page_cache) >= 4:
                                    page_cache.popitem(last=False)
                                page = pdf[page_id]
                                spans = [s for b in page.get_text('dict', flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)['blocks']
                                         for line in b.get('lines', []) for s in line.get('spans', [])]
                                page_cache[page_id] = (page.get_drawings(), spans, [page.rect.width, page.rect.height])
                            drawings, spans, page_size = page_cache[page_id]
                            box = occurrence['bbox']
                            if not any(simple_match(d, box) for d in drawings):
                                continue
                            counts['geometry_qualified_uses'] += 1
                            if rgb is None:
                                path = Path(resolver[row['image']['ref']])
                                if not path.is_absolute():
                                    path = R / path
                                assert sha(path) == row['image']['sha256']
                                with Image.open(path) as image:
                                    rgba = image.convert('RGBA')
                                    rgb = np.asarray(Image.alpha_composite(Image.new('RGBA', rgba.size, (255,255,255,255)), rgba).convert('RGB'))
                                counts['newly_decoded_geometry_candidates'] += 1
                            summary, primitive_proof = primitive_facts(rgb, box, drawings, spans)
                            if summary is None:
                                continue
                            view = source_view(raw, nodes, occurrence)
                            view['source_primitive_visibility'] = summary
                            registered = []
                            for existing in primitive_proof['matched_paths']:
                                drawing = next(d for d in drawings if d.get('seqno') == existing['paint_order'])
                                iou, frame = registered_iou(rgb, drawing, box, page_size)
                                supported = np.zeros(rgb.shape[:2], dtype=bool)
                                for channel in ('fill', 'color'):
                                    colour = drawing.get(channel)
                                    if colour is None:
                                        continue
                                    vector = 255.-np.asarray(colour)*255.
                                    norm = float(np.dot(vector, vector))
                                    if norm <= 1e-6:
                                        continue
                                    delta = 255.-rgb.astype(float)
                                    alpha = (delta @ vector) / norm
                                    residual = np.linalg.norm(delta-alpha[...,None]*vector,axis=-1)
                                    relative = residual / np.maximum(np.linalg.norm(delta,axis=-1), 1.)
                                    supported |= (alpha >= .02) & (alpha <= 1.15) & (relative <= .15)
                                registered.append({**existing, 'registered_iou': iou, 'registration': frame,
                                                   'pixels_supported_union': int(supported.sum())})
                            witness = positive_component(view, registered, int(np.count_nonzero(np.min(rgb,axis=-1)<245)))
                            proofs.append({'use_id': usage['use_id'], 'target_bbox': box, 'source_pdf_sha256': pdf_sha,
                                           'registered_paths': registered, 'source_view': view, 'witness': witness})
                            if witness:
                                witnesses.append({**witness, 'use_id': usage['use_id']})
                    counts['processed_assets'] += 1
                    counts['promoted_assets'] += bool(witnesses)
                    result = {'sample_id': sid, 'base_final': 'D', 'proposed_final': 'K' if witnesses else 'D',
                              'witnesses': witnesses, 'proofs': proofs, 'cache_reused': cached}
                    evidence_stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    if counts['processed_assets'] % 64 == 0:
                        write(run / 'progress.json', {'counts': dict(counts), 'elapsed_seconds': time.monotonic()-started})
    assert counts['processed_assets'] == len(eligible)
    write(out / 'freeze.json', {'status': 'COMPLETE_SOURCE_COMPONENT_EVIDENCE_FROZEN_NOT_SCORED',
        'counts': dict(counts), 'evidence_sha256': sha(out / 'evidence.jsonl'),
        'preregistration_sha256': sha(out / 'preregistration.json'), 'gold_opened': False,
        'elapsed_seconds': time.monotonic()-started, 'run_dir': str(run)})
    write(P / 'reports/zero-delete-99-20261007/cold-component-pointer.json', {'directory': str(out), 'run_dir': str(run)})
    print(json.dumps(dict(counts)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
