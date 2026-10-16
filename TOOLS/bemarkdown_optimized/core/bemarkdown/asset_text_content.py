"""Content-level visual text requests, independent of retention and placement."""
import copy
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageStat
from .transcription_repair import RepairTask
from .source_review_evidence import SOURCE_REVIEW_VERSION, source_decision

VERSION = 'asset-visible-text-content-v1'
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(65536), b''): h.update(block)
    return h.hexdigest()
def objsha(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
def prepare_visible_input(source, folder):
    source = Path(source).resolve(); original_hash = sha(source)
    with Image.open(source) as original:
        if original.width * original.height > 32000000: raise ValueError('ASSET_PIXEL_WINDOW_EXCEEDED')
        rgba = original.convert('RGBA'); alpha = rgba.getchannel('A')
        # Gray is used only when transparency and bright foreground would hide
        # content on white. Uniform black opaque glyphs are never called blank.
        transparent = alpha.getextrema()[0] < 255
        bright_foreground = alpha.getextrema()[1] > 0 and ImageStat.Stat(rgba.convert('L'), mask=alpha).mean[0] > 230
        background = (127, 127, 127) if transparent and bright_foreground else (255, 255, 255)
        visible = Image.new('RGB', rgba.size, background); visible.paste(rgba, mask=alpha)
        no_text_blank = alpha.getextrema()[1] == 0 or (alpha.getextrema() == (255, 255) and visible.getextrema() == ((255, 255),) * 3)
        factor = 4 if visible.height < 80 and visible.width * visible.height * 16 <= 1048576 else 1
        if factor > 1: visible = visible.resize((visible.width * factor, visible.height * factor), Image.Resampling.NEAREST)
        recipe = {'version': VERSION, 'original_sha256': original_hash, 'original_size': list(original.size),
            'alpha_background_rgb': list(background), 'nearest_neighbor_factor': factor, 'no_pixels_cropped': True}
        path = Path(folder) / (objsha(recipe) + '.png'); path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists(): visible.save(path)
        else:
            with Image.open(path) as existing:
                if existing.size != visible.size or existing.convert('RGB').tobytes() != visible.tobytes(): raise ValueError('BOUND_VISIBLE_INPUT_CHANGED')
        return {'path': str(path), 'sha256': sha(path), 'width': visible.width, 'height': visible.height,
            'recipe': recipe, 'blank_source': bool(no_text_blank)}

def transcribe_assets(tasks, work, backend_factory, *, local_evidence_runner=None, source_review_factory=None, source_review_lookup=None, local_plain_acceptance=False, native_symbol_corroboration=False):
    work = Path(work); grouped = {}; aliases = {}; prepared = {}; blank = {}
    for item in tasks:
        if sha(item['image_path']) != item['image_sha256']: raise ValueError('ASSET_ORIGINAL_SHA_CHANGED')
        image = prepare_visible_input(item['image_path'], work / 'inputs')
        source_sha = item.get('source_sha256', item['image_sha256'])
        key = objsha({'source_sha256': source_sha, 'visible_sha256': image['sha256'], 'contract': VERSION,
                      'scope': 'ENTIRE_VISIBLE_IMAGE_ONLY_NO_CONTEXT_INFERENCE'})
        content_uid = 'content:' + key[:24]
        aliases.setdefault(content_uid, []).append(item); prepared[content_uid] = image
        if image['blank_source']: blank[content_uid] = True; continue
        if content_uid not in grouped:
            grouped[content_uid] = RepairTask(document_id=item.get('document_id', source_sha[:16]),
                page_id='asset-content', node_id=content_uid, asset_uid=content_uid,
                source_sha256=source_sha, input_sha256=image['sha256'], candidate='',
                task_type='VISUAL_TEXT_EXTRACT', bbox=(0, 0, image['width'], image['height']),
                bbox_coordinate_space='pixel', candidate_version=VERSION)
    preverified = {}
    if source_review_lookup is not None:
        for uid, task in list(grouped.items()):
            owners = aliases[uid]
            originals = {o['image_sha256']: o for o in owners}
            approved = {}
            for original_sha, owner in originals.items():
                value = source_review_lookup(source_sha256=task.source_sha256,
                    original_image_sha256=original_sha, actual_input_sha256=prepared[uid]['sha256'],
                    task=task, original_image_path=owner['image_path'], proof_work=work / 'source-alias-proofs')
                if value is None:
                    continue
                if (value.get('status') != 'SUCCESS'
                        or value.get('actual_input_sha256') != prepared[uid]['sha256']
                        or value.get('original_image_sha256') != original_sha):
                    raise ValueError('ASSET_SOURCE_CACHE_BINDING_OR_STATUS')
                approved[original_sha] = value
            if len(approved) == len(originals):
                if len({(a['markdown'], a['preserve_image']) for a in approved.values()}) != 1:
                    raise ValueError('ASSET_SOURCE_CACHE_CONFLICTING_APPROVALS')
                preverified[uid] = approved
                del grouped[uid]
    local = {}
    if local_evidence_runner is not None and grouped:
        local_inputs = [{**prepared[uid], 'content_uid': uid,
                         'source_sha256': task.source_sha256}
                        for uid, task in grouped.items()]
        # Return only after every local model subprocess exits. Construct the
        # cloud backend after this boundary so its waits cannot retain models.
        local = local_evidence_runner(local_inputs)
        if set(local) != set(grouped):
            raise ValueError('ASSET_LOCAL_EVIDENCE_POPULATION_MISMATCH')
        for uid, value in local.items():
            if value['input_sha256'] != prepared[uid]['sha256']:
                raise ValueError('ASSET_LOCAL_EVIDENCE_INPUT_MISMATCH')
    local_results = {}
    if local_plain_acceptance:
        from .local_plain_acceptance import qualify_plain_text, local_plain_result
        decisions = {}
        for uid, task in grouped.items():
            if uid not in local:
                continue
            decision = qualify_plain_text(prepared[uid], local[uid])
            decisions[uid] = decision
            if decision['accepted']:
                local_results[task.task_id] = local_plain_result(task, decision)
        work.mkdir(parents=True, exist_ok=True)
        (work / 'local-acceptance-decisions.json').write_text(
            json.dumps(decisions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    backend = backend_factory(work / 'backend')
    requests = [t for t in grouped.values() if t.task_id not in local_results]
    images = {t.task_id: prepared[t.node_id]['path'] for t in requests}
    validated = backend.run(requests, images) if requests else {}
    validated.update(local_results)
    native_decisions = {}
    native_accepted = {}
    if native_symbol_corroboration:
        from .native_symbol_evidence import qualify_native_symbol
        for uid, task in grouped.items():
            decision = qualify_native_symbol(task, aliases[uid], prepared[uid], validated[task.task_id], local.get(uid))
            native_decisions[uid] = decision
            if decision['accepted']:
                native_accepted[uid] = decision
        work.mkdir(parents=True, exist_ok=True)
        (work / 'native-symbol-decisions.json').write_text(
            json.dumps(native_decisions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    source_reviews = {}
    accepted_source_review_tasks = {}
    source_review_calls = []
    unresolved_source_reviews = {}
    if source_review_factory is not None and local:
        from .asset_local_evidence import candidate_conflicts, independent_review_reasons
        source_tasks = []
        source_images = {}
        source_owners = {}
        for uid, task in grouped.items():
            proposal = validated[task.task_id]
            if uid in native_accepted:
                continue
            conflicts = candidate_conflicts(local[uid], proposal['text']) if proposal['status'] == 'SUCCESS' else []
            risk_reasons = independent_review_reasons(local[uid])
            if 'ISOLATED_SYMBOL_SOURCE_REVIEW' in risk_reasons:
                unresolved_source_reviews[uid] = {'reason': 'INDEPENDENT_SHORT_SYMBOL_SOURCE_REVIEW_REQUIRED',
                    'result': None, 'repeated_same_backend_review_skipped': True}
                continue
            if not conflicts and not risk_reasons:
                continue
            source_task = RepairTask(document_id=task.document_id, page_id=task.page_id,
                node_id=uid, asset_uid=uid, source_sha256=task.source_sha256,
                input_sha256=task.input_sha256, candidate='', task_type='VISUAL_TEXT_EXTRACT',
                bbox=task.bbox, bbox_coordinate_space='pixel', candidate_version=SOURCE_REVIEW_VERSION)
            source_tasks.append(source_task)
            source_images[source_task.task_id] = prepared[uid]['path']
            source_owners[source_task.task_id] = uid
        if source_tasks:
            reviewer = source_review_factory(work / 'source-review')
            reviewed = reviewer.run(source_tasks, source_images)
            source_review_calls = reviewer.calls
            for source_task in source_tasks:
                uid = source_owners[source_task.task_id]
                reviewed_result = reviewed[source_task.task_id]
                proposal = validated[grouped[uid].task_id]
                accepted, decision_reason = source_decision(source_task, reviewed_result,
                    primary_calls=backend.calls, source_calls=source_review_calls,
                    primary_text=proposal['text'], local_evidence=local[uid])
                if accepted:
                    source_reviews[uid] = {'result': reviewed_result,
                        'task': source_task.to_dict(), 'original_proposal': copy.deepcopy(proposal)}
                    accepted_source_review_tasks[uid] = source_task
                else:
                    unresolved_source_reviews[uid] = {'result': reviewed_result,
                        'task': source_task.to_dict(), 'reason': decision_reason}
    answers = {}
    conflict_rows = {}
    for content_uid, owners in aliases.items():
        image = prepared[content_uid]
        if content_uid in preverified:
            for owner in owners:
                answers[owner['id']] = {**copy.deepcopy(preverified[content_uid][owner['image_sha256']]), 'id': owner['id']}
            continue
        if content_uid in blank:
            result = {'status':'SUCCESS','text':'','reason':'Original source has no visible foreground or is uniformly opaque white.',
                      'text_presence':'NONE','visual_content_kind':'NO_TEXT','source_verified':False,
                      'evidence_kind':'PIXEL_VERIFIED_NO_VISIBLE_TEXT'}
            task = None
        else:
            task = grouped[content_uid]; result = validated[task.task_id]
            if content_uid in source_reviews:
                review = source_reviews[content_uid]
                result = review['result']
                task = accepted_source_review_tasks[content_uid]
            if content_uid in local and result['status'] == 'SUCCESS':
                from .asset_local_evidence import candidate_conflicts, independent_review_reasons
                conflicts = candidate_conflicts(local[content_uid], result['text'])
                risk_reasons = independent_review_reasons(local[content_uid])
                if (conflicts or risk_reasons) and content_uid not in source_reviews and content_uid not in native_accepted:
                    conflict_rows[content_uid] = {'models': conflicts, 'risk_reasons': risk_reasons,
                                                  'proposal': copy.deepcopy(result)}
                    result = {**result, 'status': 'REVIEW', 'text': task.candidate,
                              'uncertain': True, 'patches': [],
                              'text_presence': 'UNCERTAIN',
                              'visual_content_kind': 'UNCERTAIN',
                              'reason': 'SOURCE_REVIEW_REQUIRED: '
                                        + ','.join(conflicts + risk_reasons)}
        text = result['text']; kind = result['visual_content_kind']
        if result['status'] == 'SUCCESS' and kind == 'PURE_FORMULA' and text.strip() and '$' not in text:
            text = '$' + text.strip() + '$'
        for owner in owners:
            answers[owner['id']] = {'id': owner['id'], 'kind':'TEXT' if kind in {'PURE_TEXT','PURE_FORMULA'} else 'DIAGRAM',
                'status': result['status'], 'markdown': text, 'uncertain': result['status'] != 'SUCCESS',
                'reason': result['reason'], 'content_uid': content_uid,
                'text_presence': result['text_presence'], 'visual_content_kind': kind,
                'preserve_image': kind not in {'PURE_TEXT','PURE_FORMULA'} or content_uid in source_reviews or content_uid in native_accepted,
                'transcription_status': 'NO_VISIBLE_TEXT' if result['text_presence'] == 'NONE' else result['status'],
                'repair_task': task.to_dict() if task else None, 'repair_result': result,
                'original_image_sha256': owner['image_sha256'], 'actual_input_sha256': image['sha256'],
                'local_evidence': local.get(content_uid),
                'visible_input': image,
                'source_review_required': content_uid in conflict_rows,
                'local_conflicting_models': conflict_rows.get(content_uid, {}).get('models', []),
                'proposed_repair_result': conflict_rows.get(content_uid, {}).get('proposal')}
            if content_uid in native_accepted:
                answers[owner['id']]['native_symbol_corroboration'] = copy.deepcopy(native_accepted[content_uid])
            if content_uid in source_reviews:
                answers[owner['id']]['source_review'] = source_reviews[content_uid]
                answers[owner['id']]['proposed_repair_result'] = source_reviews[content_uid]['original_proposal']
            elif content_uid in unresolved_source_reviews:
                answers[owner['id']]['unresolved_source_review'] = unresolved_source_reviews[content_uid]
    work.mkdir(parents=True, exist_ok=True)
    (work / 'content-bindings.json').write_text(json.dumps({'version':VERSION,'logical_owners':len(tasks),
        'unique_contents':len(aliases),'executed_content_requests':len(requests),
        'local_evidence_contents':len(local),'source_review_contents':len(conflict_rows)+len(source_reviews),
        'source_review_requested_contents':len(source_reviews)+len(unresolved_source_reviews),
        'source_review_resolved_contents':len(source_reviews),
        'source_review_unresolved_contents':len(unresolved_source_reviews),
        'approved_source_review_cache_contents':len(preverified),
        'bindings':[{'content_uid':uid,'owner_ids':[o['id'] for o in items],'visible_input':prepared[uid]} for uid,items in aliases.items()]},
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return answers, list(backend.calls) + list(source_review_calls)
