"""Opt-in native crop recognition: one PP stage, local stages, bounded source review.

Reuses only the current PP result for the identical supplied bytes. Local pending
text is retained as a hypothesis, never upgraded to source-verified text.
"""
import copy,hashlib,json,shutil,os
from collections import Counter
from pathlib import Path
from PIL import Image
from bootstrap import sha,write
from e2e_completion import FlashAgent
from bemarkdown.asset_local_evidence import BoundedAssetLocalEvidence
from bemarkdown.pdf.adaptive_ocr import anomalies
from bemarkdown.pdf.production_runtime import RegistryBackedThreeModelTextResolver
from bemarkdown.pdf.production_three_model_live import (
    ProductionThreeModelStagedResult,_ResolvedThreeModelTextRuntime,build_text_recognition_request)
from bemarkdown.pdf_content_router import PaddleXPdfOcrRuntime

VERSION='native-local-source-review-v2'

def original_source_review_required(crop):
    """Known producer exclusions are not missing glyphs to recover from a mask."""
    return (crop.get('contract')=='TEXT_EXCLUSION_MASK'
        and crop.get('text_only_render_sha256')==crop.get('content_sha256')
        and bool(crop.get('original_render_sha256'))
        and crop.get('original_render_sha256')!=crop.get('content_sha256')
        and bool(crop.get('ownership_mask_px_bboxes')))

def select_native_delivery(local,answer,pp_text,*,pending_reference_salvage=False):
    result=answer.get('repair_result',{})
    local_accept=(result.get('evidence_kind')=='QUALIFIED_LOCAL_OCR_AGREEMENT'
                  and result.get('local_acceptance',{}).get('accepted') is True)
    source_accept=(result.get('source_verified') is True and result.get('evidence_kind')=='SOURCE_IMAGE_VERIFIED')
    good=(answer.get('status')=='SUCCESS' and answer.get('uncertain') is False
          and answer.get('kind') in {'TEXT','TITLE'} and bool(answer.get('markdown','').strip())
          and (local_accept or source_accept))
    if good:
        return {'selected_text':answer['markdown'],'selected_model':'QualifiedLocalOCR' if local_accept else 'DeepSeekFlash',
            'resolution_status':'ADAPTIVE_PRIMARY','review_required':False,'review_reasons':[],
            'preserve_source_image':local_accept or answer.get('preserve_image',False),
            'source_verified':source_accept,'acceptance_kind':result['evidence_kind']}
    choice=local.get('decision',{});text=str(choice.get('selected_text',''));model=choice.get('selected_model','')
    salvage=None
    if pending_reference_salvage:
        from bemarkdown.native_reference_salvage import pending_reference_text
        primary=local.get('candidates',{}).get('Ovis') or {}
        salvage=pending_reference_text(str(primary.get('text','')),str(primary.get('status','')),local.get('input_sha256'))
        if salvage is not None:
            text=salvage['text'];model='OvisOCR2_reference_stripped'
    if anomalies(text,'PASS'):
        text=pp_text;model='PP-OCRv6'
    selected={'selected_text':text,'selected_model':model+'_pending',
        'resolution_status':'AGENT_REQUIRED_ALL_DIFFER','review_required':True,
        'review_reasons':list(dict.fromkeys([*choice.get('review_reasons',[]),answer.get('reason','SOURCE_NOT_VERIFIED')])),
        'preserve_source_image':True,'source_verified':False,'acceptance_kind':'UNVERIFIED_LOCAL_HYPOTHESIS'}
    if salvage is not None:
        selected['pending_reference_salvage']=salvage
        selected['uncertain']=True
        selected['review_reasons'].append('UNVERIFIED_REFERENCE_STRIPPED_PRIMARY_REVIEW')
    return selected

class NativeLocalRuntime(_ResolvedThreeModelTextRuntime):
    def fingerprint(self):
        value=super().fingerprint()
        value.update(rec_model_id='PP-OCRv6+OvisOCR2+selective-Xiaomi+DeepSeekFlash',architecture_schema=VERSION)
        value['pending_reference_salvage_enabled']=bool(self._metrics.get('pending_reference_salvage_enabled',False))
        if value['pending_reference_salvage_enabled']:
            from bemarkdown.native_reference_salvage import VERSION as salvage_version
            value['pending_reference_salvage_version']=salvage_version
        return value

def apply_pending_native_reference_salvage(document,original_document):
    """Change only the untouched native fallback after completion has finished.

    This phase is downstream of model requests: their candidate/context inputs
    and their rejection receipts stay immutable. Successful replacements and
    newly emitted asset transcripts can never enter this branch.
    """
    originals={b['node_id']:b for b in original_document['blocks']}
    changes=[];considered=0
    def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    for block in document['blocks']:
        original=originals.get(block['node_id'])
        if original is None or block!=original or block.get('kind')!='TEXT':continue
        provenance=block.get('provenance',{}).get('route_provenance',{}).get('three_model_text_evidence')
        if not provenance or provenance.get('schema')!=VERSION:continue
        old=provenance['resolver']
        if not old.get('review_required') or old.get('source_verified'):continue
        local=provenance['local_evidence'];answer=provenance['agent_answer']
        request=provenance['request'];input_sha=request['crop_sha256']
        if local.get('input_sha256')!=input_sha:raise ValueError('PENDING_NATIVE_REFERENCE_INPUT_CHANGED')
        considered+=1
        resolution=select_native_delivery(local,answer,local['candidates']['PP']['text'],pending_reference_salvage=True)
        if 'pending_reference_salvage' not in resolution:continue
        rid=request['region_id']
        route={'route_id':rid};crop={'content_sha256':input_sha}
        before=NativeLocalRuntime({rid:provenance},{}).recognize_route(route,crop)
        if len(before)!=1 or before[0]['text'].strip()!=block['content']['text']:
            raise ValueError('PENDING_NATIVE_RUNTIME_TO_NODE_BINDING_CHANGED')
        before_sha=identity(block)
        candidate=copy.deepcopy(provenance);resolution['normalized_text']=resolution['selected_text'];candidate['resolver']=resolution
        runtime=NativeLocalRuntime({rid:candidate},{'pending_reference_salvage_enabled':True})
        after=runtime.recognize_route(route,crop)
        assert len(after)==1 and resolution['uncertain'] and not resolution['source_verified']
        block['content']['text']=after[0]['text'].strip()
        block['provenance']['route_provenance']['three_model_text_evidence']=candidate
        block['provenance']['pending_native_reference_delivery']={
            'stage':'POST_COMPLETION_UNCHANGED_NATIVE_FALLBACK','original_native_block_sha256':before_sha,
            'source_input_sha256':input_sha,'runtime_fingerprint':runtime.fingerprint(),
            'source_verified':False,'uncertain':True,'preserve_source_image':True}
        changes.append({'node_id':block['node_id'],'source_input_sha256':input_sha,
            'before_block_sha256':before_sha,'after_block_sha256':identity(block),
            'prior_resolver':copy.deepcopy(old),'new_resolver':copy.deepcopy(resolution)})
    return {'version':'post-completion-pending-native-reference-v1','considered_native_pending_nodes':considered,
        'changed_nodes':changes,'new_model_calls':0,'upstream_request_inputs_modified':False,
        'source_verified':False,'review_closed':False}

class NativeLocalFlashResolver:
    def __init__(self,work_root,*,models_root,config_path,mcp_root,patch,agent_root,
                 local_evidence_factory=None,pending_reference_salvage=None):
        self.work_root=Path(work_root);self.models_root=models_root;self.config_path=config_path;self.mcp_root=mcp_root
        self.agent=FlashAgent(Path(agent_root)/('local-ocr-'+hashlib.sha256(str(work_root).encode()).hexdigest()[:10]),
            patch,source_review=True,local_plain_acceptance=True,bounded_visual_protocol=True)
        self.local_evidence_factory=local_evidence_factory or BoundedAssetLocalEvidence
        value=os.environ.get('BEMARKDOWN_NATIVE_PENDING_REFERENCE_SALVAGE','0')
        if value not in {'0','1'}:raise ValueError('NATIVE_REFERENCE_SALVAGE_SWITCH_INVALID')
        self.pending_reference_salvage=(value=='1') if pending_reference_salvage is None else bool(pending_reference_salvage)

    def resolve(self,*,plans,crops_by_route):
        from isolated_native_backends import require_native_backend_activation
        require_native_backend_activation()
        routes=[r for p in plans for r in p['routes'] if r['adapter'] in ('OCR_TEXT_REGION','PAGE_VISUAL_TEXT_RECOVERY') and str(r['route_id']) in crops_by_route]
        inputs={};owners={};route_by_uid={};requests={}
        folder=self.agent.work/'native-inputs';folder.mkdir(parents=True,exist_ok=True)
        for route in routes:
            rid=str(route['route_id']);crop=crops_by_route[rid];source=Path(crop['path'])
            digest=sha(source)
            if digest!=crop['content_sha256']:raise ValueError('NATIVE_SOURCE_CROP_CHANGED')
            # PP preprocessing scope participates in identity; source geometry remains per owner.
            scope={'sha256':digest,'complete_region_lines':route['adapter']=='OCR_TEXT_REGION' and not crop.get('figure_label_parent_route_id'),
                'ownership_mask_px_bboxes':crop.get('ownership_mask_px_bboxes') or []}
            uid='native:'+hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest()[:24]
            owners[rid]=uid;requests[rid]=build_text_recognition_request(route,crop)
            if uid in inputs:continue
            path=folder/(uid.split(':')[1]+source.suffix.lower());shutil.copyfile(source,path)
            if sha(path)!=digest:raise ValueError('NATIVE_COPIED_INPUT_CHANGED')
            with Image.open(path) as image:width,height=image.size
            inputs[uid]={'content_uid':uid,'path':str(path),'sha256':digest,'width':width,'height':height,'source_sha256':digest}
            route_by_uid[uid]=route
        representative_ids={str(r['route_id']) for r in route_by_uid.values()}
        pp_plans=[{**p,'routes':[r for r in p['routes'] if str(r['route_id']) in representative_ids]} for p in plans]
        pp=PaddleXPdfOcrRuntime(models_root=self.models_root,config_path=self.config_path,mcp_root=self.mcp_root)
        helper=RegistryBackedThreeModelTextResolver(self.work_root,models_root=self.models_root,config_path=self.config_path,mcp_root=self.mcp_root)
        try:
            outputs=helper._recognize_provider_a(pp,pp_plans,crops_by_route);identity=pp.fingerprint()
        finally:pp.close()
        seeds={'runtime_fingerprint':identity,'results':[]}
        for uid,row in inputs.items():
            original_lines=outputs.get(str(route_by_uid[uid]['route_id']),[])
            # Recognition/coverage reuse is local to identical input pixels. Do
            # not attach the representative owner's page coordinates to aliases.
            lines=[{k:copy.deepcopy(v) for k,v in line.items() if k not in {'bbox_page_pdf_pt','page_index'}} for line in original_lines]
            seeds['results'].append({'content_uid':uid,'input_sha256':row['sha256'],
                'lines':lines,'text':'\n'.join(str(x.get('text','')) for x in lines)})
        write(self.agent.work/'current-pp-seed.json',{'version':VERSION,'results':seeds,
            'input_bindings':inputs,'owner_bindings':owners,'current_run':True,'gold_read':False})
        runner=self.local_evidence_factory(self.agent.work/'native-local',models_root=self.models_root,
            config_path=self.config_path,mcp_root=self.mcp_root,pp_frozen=seeds)
        local=runner(list(inputs.values())) if inputs else {}
        if set(local)!=set(inputs):raise ValueError('NATIVE_LOCAL_POPULATION_CHANGED')
        tasks=[];owner_local={}
        for route in routes:
            rid=str(route['route_id']);uid=owners[rid];row=inputs[uid];evidence=local[uid]
            if evidence.get('input_sha256')!=row['sha256']:raise ValueError('NATIVE_LOCAL_INPUT_CHANGED')
            owner_local[rid]=evidence
            previous=str(evidence.get('decision',{}).get('selected_text',''))
            tasks.append({'id':rid,'mode':'OCR_PENDING','image_path':row['path'],'image_sha256':row['sha256'],
                'previous_text':previous,'source_page':route.get('page_index'),
                'figure_label_scope':bool(crops_by_route[rid].get('figure_label_parent_route_id'))})
        deferred={t['id'] for t in tasks if original_source_review_required(crops_by_route[t['id']])}
        cloud_tasks=[t for t in tasks if t['id'] not in deferred]
        answers=self.agent.run(cloud_tasks,native_local_evidence={t['id']:owner_local[t['id']] for t in cloud_tasks}) if cloud_tasks else {}
        for rid in deferred:
            answers[rid]={'status':'REVIEW','uncertain':True,'kind':'TEXT','markdown':'',
                'reason':'PRODUCER_MASK_REQUIRES_UNMASKED_ORIGINAL_SOURCE_REVIEW',
                'deferred_to':'COMPLETION_ORIGINAL_SOURCE_CROP','model_read_on_mask_skipped':True}
        provenance={};candidates=[]
        for route in routes:
            rid=str(route['route_id']);request=requests[rid];value=answers[rid];evidence=owner_local[rid]
            pp_text=evidence['candidates']['PP']['text']
            resolution=select_native_delivery(evidence,value,pp_text,pending_reference_salvage=self.pending_reference_salvage)
            resolution['normalized_text']=resolution['selected_text']
            a={'model_id':'PP-OCRv6','text':pp_text,'normalized_text':pp_text,
                'model_fingerprint':identity['rec_model_fingerprint'],'runtime_fingerprint':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
                'source_crop_sha256':request.crop_sha256,'output_contract_status':'PASS' if pp_text else 'EMPTY_OUTPUT'}
            row={'schema':VERSION,'request':request.to_dict(),'evidence':{'A':a,'B':evidence.get('ovis_evidence'),'C':evidence.get('xiaomi_evidence')},
                'resolver':resolution,'local_evidence':evidence,'agent_answer':value,
                'third_model_router':{'triggered':evidence.get('xiaomi_evidence') is not None,'reasons':evidence.get('secondary_trigger_reasons',[])},
                'comparator':{'scope':'ONE_ACTUAL_SOURCE_CROP','gold_read':False},'source_verified':resolution['source_verified']}
            if resolution['review_required']:
                candidate={'task_id':'native-local-pending:'+rid,'reason':value['reason'],'bbox':list(request.bbox),
                    'page_id':request.page_id,'primary_crop_sha256':request.crop_sha256,
                    'source_provenance':request.source_provenance,'evidence':row['evidence']}
                row['agent_review_candidate']=candidate;candidates.append(candidate)
            provenance[rid]=row
        metrics={'schema':VERSION,'ocr_text_routes':len(routes),'unique_local_inputs':len(inputs),
            'pending_reference_salvage_enabled':self.pending_reference_salvage,
            'pending_reference_salvage_count':sum('pending_reference_salvage' in p['resolver'] for p in provenance.values()),
            'pp_actual_inputs':len(inputs),'pp_reused_for_local_stage':True,
            'masked_crops_deferred_to_original_source':len(deferred),
            'selected_models':dict(Counter(p['resolver']['selected_model'] for p in provenance.values())),
            'review_required':len(candidates),'agent_calls':self.agent.calls,'gold_read':False}
        write(self.work_root/'native-local-text-resolution.json',{'metrics':metrics,'provenance':provenance})
        return ProductionThreeModelStagedResult(NativeLocalRuntime(provenance,metrics),metrics,candidates,copy.deepcopy(provenance))
