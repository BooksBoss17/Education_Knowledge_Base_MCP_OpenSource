"""Production whole-block composition for PP evidence, Ovis, conditional Xiaomi."""
from __future__ import annotations
import copy,hashlib
from collections import Counter
from pathlib import Path
from .adaptive_ocr import RoutingPolicy,POLICY_VERSION
from .production_three_model_live import (ProductionThreeModelStagedResult,
    _ResolvedThreeModelTextRuntime,build_text_recognition_request)

def candidate(evidence):
    return {'text':evidence.text,'status':evidence.output_contract_status,
            'confidence':evidence.confidence}

class AdaptiveResolvedRuntime(_ResolvedThreeModelTextRuntime):
    def recognize_route(self,route,crop):
        rows=super().recognize_route(route,crop)
        # A Paddle confidence is not the selected generative model's confidence.
        for row in rows:row['confidence']=None
        return rows

    def fingerprint(self):
        result=super().fingerprint()
        result.update(rec_model_id='OvisOCR2+conditional-Xiaomi-OCR-0',architecture_schema=POLICY_VERSION)
        return result

class AdaptiveThreeModelComposition:
    def __init__(self,provider_a,provider_b,provider_c,*,policy=None):
        self.provider_a=provider_a;self.provider_b=provider_b;self.provider_c=provider_c
        self.policy=policy or RoutingPolicy()

    def resolve(self,*,plans,crops_by_route):
        routes=[route for plan in plans for route in plan['routes'] if route['adapter'] in {'OCR_TEXT_REGION','PAGE_VISUAL_TEXT_RECOVERY'}]
        ids=[str(r['route_id']) for r in routes]
        if len(ids)!=len(set(ids)):raise ValueError('ADAPTIVE_DUPLICATE_ROUTE_ID')
        requests=[]
        for route in routes:
            crop=crops_by_route[str(route['route_id'])];path=Path(crop['path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=crop['content_sha256']:
                raise RuntimeError('ADAPTIVE_SOURCE_CROP_MISMATCH')
            requests.append(build_text_recognition_request(route,crop))
        a=self.provider_a.recognize_batch(requests)
        b=self.provider_b.recognize_batch(requests)
        if len(a)!=len(requests) or len(b)!=len(requests):raise RuntimeError('ADAPTIVE_PRIMARY_CARDINALITY')
        triggers={}
        for request,ae,be in zip(requests,a,b,strict=True):
            reasons=self.policy.trigger(candidate(ae),candidate(be),crops_by_route[request.region_id])
            if reasons:triggers[request.region_id]=reasons
        pending=[r for r in requests if r.region_id in triggers]
        secondary=self.provider_c.recognize_batch(pending) if pending else []
        if len(secondary)!=len(pending):raise RuntimeError('ADAPTIVE_SECONDARY_CARDINALITY')
        c={r.region_id:e for r,e in zip(pending,secondary,strict=True)}
        provenance={};terminal=Counter();selected=Counter()
        for request,ae,be in zip(requests,a,b,strict=True):
            ce=c.get(request.region_id)
            decision=self.policy.select(candidate(ae),candidate(be),candidate(ce) if ce else None)
            status='AGENT_REQUIRED_ALL_DIFFER' if decision['review_required'] else 'ADAPTIVE_PRIMARY' if decision['selected_model']=='OvisOCR2' else 'ADAPTIVE_SECONDARY'
            terminal[status]+=1;selected[decision['selected_model']]+=1
            provenance[request.region_id]={'schema':POLICY_VERSION,'request':request.to_dict(),
                # Stable slots keep existing provenance consumers compatible;
                # each record carries its actual new provider/model identity.
                'evidence':{'A':ae.to_dict(),'B':be.to_dict(),'C':ce.to_dict() if ce else None},
                'comparator':{'scope':'WHOLE_SOURCE_BLOCK_STRUCTURES_PRESERVED','gold_read':False},
                'third_model_router':{'triggered':ce is not None,'reasons':list(triggers.get(request.region_id,()))},
                'resolver':{**decision,'resolution_status':status,'normalized_text':decision['selected_text'],
                    'selected_provider_basis':decision['selected_model']},
                'policy_version':self.policy.version,'policy_fingerprint':self.policy.fingerprint()}
        metrics={'schema':POLICY_VERSION,'policy_fingerprint':self.policy.fingerprint(),
            'ocr_text_routes':len(routes),'three_model_request_count':len(requests),
            'a_attempts':len(a),'b_attempts':len(b),'c_attempts':len(c),'c_trigger_count':len(c),
            'c_trigger_rate':len(c)/max(1,len(routes)),'terminal_counts':dict(terminal),
            'selected_models':dict(selected),'whole_block_formula_inputs':True,
            'pp_fast_path_enabled':self.policy.pp_fast_path,'accounting_status':'PASS',
            'review_required':sum(r['resolver']['review_required'] for r in provenance.values())}
        runtime=AdaptiveResolvedRuntime(provenance,metrics)
        return ProductionThreeModelStagedResult(runtime,metrics,[],copy.deepcopy(provenance))
