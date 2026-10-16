"""PP source-line evidence with direct Flash completion of visual text routes."""
import copy,hashlib,json
from pathlib import Path
from collections import Counter
from e2e_completion import FlashAgent
from bootstrap import sha,write
from bemarkdown.pdf.production_three_model_live import ProductionThreeModelStagedResult,_ResolvedThreeModelTextRuntime,build_text_recognition_request
from bemarkdown.pdf.production_runtime import RegistryBackedThreeModelTextResolver
from bemarkdown.pdf_content_router import PaddleXPdfOcrRuntime

class FlashResolvedRuntime(_ResolvedThreeModelTextRuntime):
    def fingerprint(self):
        result=super().fingerprint();result.update(rec_model_id='DeepSeekFlash+PP-OCRv6-source-evidence',
            architecture_schema='bemarkdown-flash-direct-visual-text-v1',local_ovis_or_xiaomi_loaded=False)
        return result

class FlashTextResolver:
    def __init__(self,work_root,*,models_root,config_path,mcp_root,patch,agent_root):
        self.work_root=Path(work_root);self.models_root=models_root;self.config_path=config_path;self.mcp_root=mcp_root
        self.agent=FlashAgent(Path(agent_root)/('ocr-'+hashlib.sha256(str(work_root).encode()).hexdigest()[:10]),patch)
    def resolve(self,*,plans,crops_by_route):
        routes=[r for p in plans for r in p['routes'] if r['adapter'] in ('OCR_TEXT_REGION','PAGE_VISUAL_TEXT_RECOVERY') and str(r['route_id']) in crops_by_route]
        pp=PaddleXPdfOcrRuntime(models_root=self.models_root,config_path=self.config_path,mcp_root=self.mcp_root)
        helper=RegistryBackedThreeModelTextResolver(self.work_root,models_root=self.models_root,config_path=self.config_path,mcp_root=self.mcp_root)
        try:
            outputs=helper._recognize_provider_a(pp,plans,crops_by_route);identity=pp.fingerprint()
        finally:pp.close()
        tasks=[];requests={};provenance={}
        for route in routes:
            rid=str(route['route_id']);crop=crops_by_route[rid];path=Path(crop['path'])
            assert sha(path)==crop['content_sha256']
            request=build_text_recognition_request(route,crop);requests[rid]=request
            text='\n'.join(str(x.get('text','')) for x in outputs.get(rid,[]))
            tasks.append({'id':rid,'mode':'OCR_PENDING','image_path':str(path),'image_sha256':sha(path),
                'previous_text':text[:1800],'source_page':route.get('page_index'),'crop_binding':'current source crop, not neighboring page',
                'figure_label_scope':bool(crop.get('figure_label_parent_route_id'))})
        answers=self.agent.run(tasks) if tasks else {};terminal=Counter();candidates=[]
        model_fp='harness-account:deepseek-flash';runtime_fp=hashlib.sha256(Path(self.agent.patch).read_bytes()).hexdigest()
        for route in routes:
            rid=str(route['route_id']);request=requests[rid];value=answers[rid]
            pp_text='\n'.join(str(x.get('text','')) for x in outputs.get(rid,[]))
            good=value['status']=='SUCCESS' and not value['uncertain'] and value['kind'] in ('TEXT','TITLE')
            selected=value['markdown'] if good else pp_text
            status='ADAPTIVE_PRIMARY' if good else 'AGENT_REQUIRED_ALL_DIFFER';terminal[status]+=1
            evidence={'model_id':'DeepSeekFlash','provider_id':'DEEPSEEK_ACCOUNT_FLASH','text':value['markdown'],'normalized_text':value['markdown'],
                'confidence':None,'model_fingerprint':model_fp,'runtime_fingerprint':runtime_fp,'source_crop_sha256':request.crop_sha256,
                'output_contract_status':'PASS' if good else 'REVIEW_REQUIRED','actual_agent_model':'deepseek-flash'}
            row={'schema':'bemarkdown-flash-direct-visual-text-v1','request':request.to_dict(),
                'evidence':{'A':{'model_id':'PP-OCRv6','text':pp_text,'normalized_text':pp_text,'confidence':None,
                    'model_fingerprint':identity['rec_model_fingerprint'],'output_contract_status':'PASS' if pp_text else 'EMPTY_OUTPUT'},'B':evidence,'C':None},
                'comparator':{'scope':'ONE_ACTUAL_SOURCE_CROP','gold_read':False,'local_three_model_consensus_used':False},
                'third_model_router':{'triggered':False,'reasons':[]},
                'resolver':{'selected_text':selected,'normalized_text':selected,'selected_model':'DeepSeekFlash' if good else 'PP-OCRv6_pending',
                    'resolution_status':status,'review_required':not good,'review_reasons':[] if good else [value['reason']]}}
            if not good:
                candidate={'task_id':'flash-pending:'+rid,'reason':value['reason'],'bbox':list(request.bbox),'page_id':request.page_id,
                    'primary_crop_sha256':request.crop_sha256,'source_provenance':request.source_provenance,'evidence':row['evidence']}
                row['agent_review_candidate']=candidate;candidates.append(candidate)
            provenance[rid]=row
        metrics={'schema':'bemarkdown-flash-direct-visual-text-v1','ocr_text_routes':len(routes),'a_attempts':len(routes),
            'b_attempts':len(tasks),'c_attempts':0,'selected_models':{'DeepSeekFlash':sum(x['resolver']['selected_model']=='DeepSeekFlash' for x in provenance.values())},
            'terminal_counts':dict(terminal),'review_required':len(candidates),'accounting_status':'PASS','agent_calls':self.agent.calls,
            'policy_change':'Visual text routes go directly to user-specified Flash; PP line evidence only. Ovis/Xiaomi not loaded in this trial.'}
        runtime=FlashResolvedRuntime(provenance,metrics);write(self.work_root/'flash-text-resolution.json',{'metrics':metrics,'provenance':provenance})
        return ProductionThreeModelStagedResult(runtime,metrics,candidates,copy.deepcopy(provenance))

def pdf_factory(*,models_root,config_path=None,mcp_root=None,patch,agent_root,native_local_first=False,**kwargs):
    from bemarkdown.pdf.production_runtime import create_production_pdf_runtime
    resolver=FlashTextResolver
    if native_local_first:
        from native_local_flash_resolver import NativeLocalFlashResolver
        resolver=NativeLocalFlashResolver
    return create_production_pdf_runtime(models_root=models_root,config_path=config_path,mcp_root=mcp_root,
        text_runtime_resolver_factory=lambda work:resolver(work,models_root=models_root,config_path=config_path,mcp_root=mcp_root,patch=patch,agent_root=agent_root),**kwargs)
