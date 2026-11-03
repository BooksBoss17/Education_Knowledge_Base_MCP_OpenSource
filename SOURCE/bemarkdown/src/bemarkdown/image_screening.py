"""Source-bound candidate selection and transactional image screening output.

Model decisions never remove OCR evidence or edit the original conversion.
Generic whole-page review obligations are deliberately not classification gates.
"""
from __future__ import annotations
import collections
import copy
import hashlib
import os
import json
from pathlib import Path
import shutil
import uuid

POLICY='integrated-image-screening-v1'
TEXT_KINDS={'TITLE','TEXT','TABLE','FORMULA'}
KINDS=TEXT_KINDS|{'DIAGRAM','PHOTO','DECORATION','OTHER','UNKNOWN'}
PROTECTED_ROLES={'CONTENT_REVIEW_CROP','FORMULA_VISUAL_FALLBACK','TABLE_VISUAL_FALLBACK','TEXT_VISUAL_FALLBACK'}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    from .file_handoff import atomic_write_text
    target=Path(path).resolve()
    if os.name=='nt' and not str(target).startswith('\\\\?\\'):
        raw=str(target)
        target=Path('\\\\?\\UNC\\'+raw[2:] if raw.startswith('\\\\') else '\\\\?\\'+raw)
    atomic_write_text(target,json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def uids(value):
    if isinstance(value,dict):
        if value.get('asset_uid'):yield str(value['asset_uid'])
        for child in value.values():yield from uids(child)
    elif isinstance(value,list):
        for child in value:yield from uids(child)

def load_document(package):
    package=Path(package).resolve(strict=True)
    path=next((package/name for name in ['_bemarkdown/document_ir.json','debug/document_ir.json'] if (package/name).is_file()),None)
    if path is None:raise ValueError('SCREENING_DOCUMENT_IR_UNAVAILABLE')
    document=read(path)
    rows=[json.loads(line) for line in (package/'assets_manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    report=read(package/'conversion_report.json')
    represented=set(report['markdown_render']['represented_node_ids'])
    document['blocks']=[b for b in document['blocks'] if b['node_id'] in represented]
    mapping={r['asset_uid']:r['relative_path'] for r in rows if r['status']=='RESOLVED'}
    from .pdf_output_audit import CleanHandoffRenderer
    rendered=CleanHandoffRenderer().render(document,mapping)
    if rendered['markdown']!=(package/'document.md').read_text(encoding='utf-8'):
        raise ValueError('SCREENING_IR_RENDER_MISMATCH')
    return document,rows,report

def select_assets(package):
    package=Path(package).resolve(strict=True)
    document,assets,report=load_document(package)
    definitions={a['asset_uid']:a for a in document['assets']}
    rows=[]
    for asset in assets:
        uid=asset['asset_uid'];reasons=[];blockers=[]
        definition=definitions.get(uid,{})
        related=[b for b in document['blocks'] if uid in set(uids(b))]
        role=definition.get('role',asset.get('semantic_type'))
        if role in PROTECTED_ROLES:reasons.append('TRANSCRIPTION_EVIDENCE_ROLE')
        if asset['status']!='RESOLVED':reasons.append('ASSET_UNRESOLVED')
        if not related:reasons.append('ASSET_USAGE_UNBOUND')
        for block in related:
            if block['kind']!='IMAGE':reasons.append('NON_IMAGE_CONTENT_EVIDENCE')
            provenance=block.get('provenance',{})
            flags=provenance.get('review_reasons',[])+provenance.get('warnings',[])
            blockers.extend(str(flag) for flag in flags if str(flag).startswith('DIAGRAM_BOUNDARY_'))
            if any(str(flag).startswith(('OCR_PENDING','OCR_BOUNDED','RASTER_TEXT','TEXT_LINE_',
                                         'THREE_MODEL_','FORMULA_','TABLE_','ISOLATED_SYMBOL_')) for flag in flags):
                reasons.append('TRANSCRIPTION_REVIEW_MARKER')
            if 'TEXT' in str(block.get('subtype','')):reasons.append('TEXT_SUBTYPE_EVIDENCE')
        path=(package/(asset.get('relative_path') or '')).resolve()
        if not path.is_relative_to(package):raise ValueError('SCREENING_ASSET_ESCAPES_PACKAGE')
        if asset['status']=='RESOLVED' and sha(path)!=asset['content_sha256']:raise ValueError('SCREENING_ASSET_HASH_MISMATCH')
        rows.append(dict(asset_uid=uid,asset_name=asset.get('relative_path'),sha256=asset.get('content_sha256'),
            node_ids=[b['node_id'] for b in related],protected=bool(reasons),reasons=sorted(set(reasons)),exclusion_blockers=sorted(set(blockers)),
            disposition='PROTECTED_TRANSCRIPTION_EVIDENCE' if reasons else 'CLASSIFY'))
    return dict(schema=POLICY,source=report['source'],package=str(package),markdown_sha256=sha(package/'document.md'),
                assets=rows,candidate_count=sum(not r['protected'] for r in rows),protected_count=sum(r['protected'] for r in rows))

def validate_results(selection,results):
    expected={r['asset_name']:r for r in selection['assets'] if not r['protected']}
    actual={r['asset_name']:r for r in results}
    if len(actual)!=len(results) or actual.keys()!=expected.keys():raise ValueError('SCREENING_RESULT_COVERAGE_MISMATCH')
    decisions=[]
    for name,row in actual.items():
        if row['image_sha256']!=expected[name]['sha256']:raise ValueError('SCREENING_RESULT_HASH_MISMATCH')
        if row['decision'] not in {'K','D','U'}:raise ValueError('SCREENING_DECISION_INVALID')
        purpose=row.get('purpose',{})
        kind=purpose.get('kind','UNKNOWN')
        if kind not in KINDS:raise ValueError('SCREENING_PURPOSE_INVALID')
        # Both independent contracts must permit omission. A readable-text,
        # instructional or uncertain purpose overrides a discard candidate.
        discard_kind=(kind in {'DECORATION','OTHER'} and purpose.get('information_present') is False) or kind=='PHOTO'
        exclude=(row['decision']=='D' and not expected[name]['exclusion_blockers'] and discard_kind
                 and purpose.get('contains_text') is False
                 and purpose.get('uncertain') is False)
        action='EXCLUDE' if exclude else 'TRANSCRIBE' if kind in {'TITLE','TEXT'} else 'KEEP'
        classification_review=(row['decision']=='U' or kind=='UNKNOWN' or purpose.get('uncertain',True)
                               or (row['decision']=='D' and not exclude))
        review=action=='TRANSCRIBE' or bool(expected[name]['exclusion_blockers']) or classification_review
        decisions.append(dict(expected[name],retention_decision=row['decision'],purpose=purpose,
            action=action,needs_review=bool(review),classification_review_required=bool(classification_review),
            needs_transcription=action=='TRANSCRIBE'))
    return decisions

def export_screened(package,destination,selection,results,receipt):
    """Keep the source package intact; publish a separate canonical package."""
    from .pdf_output_audit import CleanHandoffRenderer
    from .selective_review import write_review_plan,bind_original_image
    from .production import PackageValidator,_package_manifest,_publish_staged
    package=Path(package).resolve(strict=True);destination=Path(destination).resolve()
    if destination==package or destination.is_relative_to(package):raise ValueError('SCREENING_OUTPUT_MUST_BE_SEPARATE')
    if destination.name!=package.name:raise ValueError('SCREENING_DOCUMENT_ID_CHANGED')
    if destination.exists():raise FileExistsError(destination)
    assert selection['package']==str(package) and selection['markdown_sha256']==sha(package/'document.md')
    decisions=validate_results(selection,results)
    excluded={r['asset_uid'] for r in decisions if r['action']=='EXCLUDE'}
    protected={r['asset_uid'] for r in selection['assets'] if r['protected']}
    assert not excluded&protected
    document,assets,report=load_document(package)
    removed=[b for b in document['blocks'] if set(uids(b))&excluded]
    if any(b['kind']!='IMAGE' or not set(uids(b))<=excluded for b in removed):
        raise ValueError('SCREENING_EXCLUSION_TOUCHES_OTHER_CONTENT')
    document['blocks']=[b for b in document['blocks'] if b not in removed]
    document.setdefault('suppressed_blocks',[]).extend(dict(b,image_screening_exclusion=True) for b in removed)
    removed_ids={b['node_id'] for b in removed}
    old_reviews=document.get('review_items',[])
    document['review_items']=[r for r in old_reviews if r.get('node_id') not in removed_ids]
    document['screening_suppressed_review_items']=[r for r in old_reviews if r.get('node_id') in removed_ids]
    for index,block in enumerate(document['blocks']):block['display_index']=index
    oldmap={a['asset_uid']:a['relative_path'] for a in assets if a['status']=='RESOLVED'}
    required=CleanHandoffRenderer().render(document,oldmap)['required_asset_uids']
    survivors=[a for a in assets if a['asset_uid'] in set(required)]
    stage=destination.parent/'.screening-staging'/uuid.uuid4().hex/package.name
    stage.parent.mkdir(parents=True)
    shutil.copytree(package,stage,ignore=shutil.ignore_patterns('assets'))
    new_assets=[];mapping={};path_map=[]
    for index,asset in enumerate(survivors,1):
        row=copy.deepcopy(asset);old=asset['relative_path'];identifier=f'image_{index:06d}'
        row['asset_id']=identifier
        if old:
            relative=f"assets/{identifier}{asset['extension']}"
            (stage/'assets').mkdir(exist_ok=True)
            shutil.copyfile(package/old,stage/relative)
            row['relative_path']=relative;mapping[row['asset_uid']]=relative
            path_map.append(dict(asset_uid=row['asset_uid'],old=old,new=relative,sha256=row['content_sha256']))
        new_assets.append(row)
    rendered=CleanHandoffRenderer().render(document,mapping)
    (stage/'document.md').write_bytes(rendered['markdown'].encode('utf-8'))
    (stage/'assets_manifest.jsonl').write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new_assets).encode('utf-8'))
    pdf=stage/'_bemarkdown/review_source.pdf'
    paginated=report['source']['type']=='IMAGE' or report.get('input_transform',{}).get('route')=='DOCX_SCREENSHOT_PDF'
    source_path=pdf if paginated and pdf.is_file() else Path(report['source']['path'])
    plan=write_review_plan(document,rendered,root=stage,asset_ref_map=mapping,source_path=source_path)
    if report['source']['type']=='IMAGE':
        plan=bind_original_image(plan,Path(report['source']['path']),report['input_transform']['renderer'])
    elif paginated:
        reference={'path':'_bemarkdown/review_source.pdf','sha256':sha(pdf)}
        plan=bind_original_image(plan,Path(report['source']['path']),
            {'route':'DOCX_SCREENSHOT_PDF','pdf_sha256':reference['sha256']},source_type='DOCX',page_render_reference=reference)
    # This queue augments the existing review obligations; it never clears
    # content, boundary, reading-order or coverage tasks merely from K/D/U.
    classifications={r['asset_uid']:r for r in decisions}
    for task in plan['tasks']:
        target_uids=set()
        for node_id in task['node_ids']:
            block=next((b for b in document['blocks'] if b['node_id']==node_id),None)
            if block:target_uids.update(uids(block))
        task['image_screening']=[dict(asset_uid=uid,kind=classifications[uid]['purpose'].get('kind'),
            action=classifications[uid]['action']) for uid in sorted(target_uids&classifications.keys())]
        if task['image_screening']:
            digest=hashlib.sha256(json.dumps(task['image_screening'],sort_keys=True).encode()).hexdigest()
            task['dependency_hashes']['image_screening']=digest
            task['fingerprint']=hashlib.sha256((task['fingerprint']+':'+digest).encode()).hexdigest()
    write(stage/'review_plan.json',plan)
    report['markdown_render']={k:v for k,v in rendered.items() if k!='markdown'}
    report['document'].update(blocks=len(document['blocks']),block_kind_counts=dict(collections.Counter(b['kind'] for b in document['blocks'])))
    report['assets'].update(total_visible=len(new_assets),resolved=sum(a['status']=='RESOLVED' for a in new_assets),
                            unresolved=sum(a['status']=='UNRESOLVED' for a in new_assets),records=new_assets,
                            non_content_source_assets=len(document['assets'])-len(new_assets))
    report['selective_review'].update(sha256=sha(stage/'review_plan.json'),summary=plan['summary'])
    report['image_screening']=dict(schema=POLICY,status='COMPLETE',protected_count=selection['protected_count'],
        classified_count=len(decisions),excluded_count=len(excluded),path_map=path_map,
        decisions=decisions,transcription_candidates=[r['asset_uid'] for r in decisions if r['needs_transcription']],
        original_package=str(package),original_markdown_sha256=selection['markdown_sha256'],receipt=receipt)
    report['review_item_count']=sum(t['scope']!='DELIVERY' for t in plan['tasks'])
    write(stage/'conversion_report.json',report)
    write(stage/'_bemarkdown/document_ir.json',document)
    if (stage/'debug/document_ir.json').is_file():write(stage/'debug/document_ir.json',document)
    old_manifest=read(package/'package_manifest.json')
    manifest=_package_manifest(package.name,Path(report['source']['path']),old_manifest['source']['sha256'],
                               report,stage,source_type=old_manifest['source']['type'])
    write(stage/'package_manifest.json',manifest)
    validation=PackageValidator().validate(stage)
    if not validation.valid:raise ValueError('SCREENING_PACKAGE_INVALID:'+str(validation.errors))
    _publish_staged(stage,destination,replace=False)
    return dict(package_path=str(destination),image_screening=report['image_screening'],
                review_item_count=report['review_item_count'],quality_status=report['quality_status'])
