"""CPU-only contracts. Not a trainer or a claim of model quality."""

from __future__ import annotations

import hashlib,json,pathlib

from functools import lru_cache

from jsonschema import Draft202012Validator

P=pathlib.Path(__file__).resolve().parents[1]

LABEL_MAP={'heading_or_footer':'K','transcribable_content':'K','diagram':'K','decoration':'D'}

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

@lru_cache(maxsize=16)
def _schema_validator(name,mtime_ns,size):
 schema=json.loads((P/'schemas'/(name+'.schema.json')).read_text('utf-8'))
 Draft202012Validator.check_schema(schema)
 return Draft202012Validator(schema)

def validate_schema(name,row):
 stat=(P/'schemas'/(name+'.schema.json')).stat()
 _schema_validator(name,stat.st_mtime_ns,stat.st_size).validate(row)

def validate_input(row):
 validate_schema('input',row)
 if row['image']['ref']!='sha256:'+row['image']['sha256']:raise ValueError('IMAGE_REF_SHA_MISMATCH')
 ids=[u['use_id'] for u in row['uses']]
 if len(ids)!=len(set(ids)):raise ValueError('DUPLICATE_USE_ID')
 cov=row['coverage']
 if set(ids)!=set(cov['covered_use_ids']):raise ValueError('COVERAGE_IDS_MISMATCH')
 if len(ids)>cov['expected_use_count']:raise ValueError('MORE_USES_THAN_INVENTORY')
 if cov['enumeration_complete'] and len(ids)!=cov['expected_use_count']:raise ValueError('INCOMPLETE_INVENTORY_FALSELY_COMPLETE')
 for use in row['uses']:
  if use['context_status']=='verified_empty' and use['spans']:raise ValueError('NONEMPTY_VERIFIED_EMPTY')
  if use['context_status']=='verified' and not use['spans']:raise ValueError('VERIFIED_WITHOUT_SPAN')
  for s in use['spans']:
   b=s['text'].encode()
   if hashlib.sha256(b).hexdigest()!=s['text_sha256']:raise ValueError('SPAN_TEXT_SHA_MISMATCH')
   if s['end_byte']-s['start_byte']!=len(b):raise ValueError('UTF8_BYTE_RANGE_MISMATCH')
   if s['source_page_index']!=use['page_index'] and (s['relation_to_use']!='verified_cross_page_reference' or not s['binding_evidence_refs']):raise ValueError('UNPROVEN_CROSS_PAGE_CONTEXT')
   if s['relation_to_use']=='same_page' and s['source_page_index']!=use['page_index']:raise ValueError('FALSE_SAME_PAGE_CLAIM')
 return row

def source_complete(row):
 return row['coverage']['enumeration_complete'] and all(u['context_status']!='missing' and not u['source_window_truncated'] for u in row['uses'])

def require_review(target,scope):
 r=target['review_record']
 if not r or r['reviewer_kind']!='human' or scope not in r['approved_scopes']:raise ValueError('UNVERIFIED_REVIEW_SCOPE')

def admit_training(row,target,role,allow_synthetic=False):
 validate_input(row);validate_schema('target',target)
 if role!='train':raise ValueError('NONTRAIN_ROLE')
 if row['synthetic'] and not allow_synthetic:raise ValueError('SYNTHETIC_NOT_TRAINING_DATA')
 if row['sample_id']!=target['sample_id']:raise ValueError('TARGET_MEMBER_MISMATCH')
 if target['annotation_status'] not in ('gold_mapped','human_verified'):raise ValueError('UNAPPROVED_LABEL')
 if target['target_scope']=='asset':
  if target['truth_scope'] not in ('asset_category_proxy','asset_semantic_all_uses'):raise ValueError('TRUTH_SCOPE_MISMATCH')
  if target['truth_scope']=='asset_category_proxy' and target['gold_class'] is None:raise ValueError('CATEGORY_PROXY_WITHOUT_CLASS')
  if target['truth_scope']=='asset_category_proxy' and LABEL_MAP[target['gold_class']]!=target['asset_gold_label']:raise ValueError('GOLD_CATEGORY_MAPPING_CHANGED')
  if target['truth_scope']=='asset_semantic_all_uses':require_review(target,'asset_semantic_label')
  if row['scope']!='asset_use_bag' or not source_complete(row):raise ValueError('ASSET_LABEL_REQUIRES_COMPLETE_USE_BAG')
  if target['label_provenance']!='current_asset_gold' or target['decision_target']!=target['asset_gold_label']:raise ValueError('ASSET_GOLD_CHANGED')
 elif target['target_scope']=='reviewed_use_set':
  if target['truth_scope']!='use_semantic_review':raise ValueError('TRUTH_SCOPE_MISMATCH')
  if row['scope']!='use_set' or target['annotation_status']!='human_verified' or target['label_provenance']!='human_verified_usage':raise ValueError('ASSET_LABEL_CANNOT_BECOME_USE_LABEL')
  require_review(target,'use_label')
 elif target['target_scope']=='abstention_stress':
  if target['truth_scope']!='abstention_annotation':raise ValueError('TRUTH_SCOPE_MISMATCH')
  if row['scope']!='abstention_stress' or target['annotation_status']!='human_verified' or target['label_provenance']!='human_verified_abstention':raise ValueError('UNVERIFIED_ABSTENTION')
  require_review(target,'abstention')
 if target['decision_target']=='U' and target['target_scope']!='abstention_stress':raise ValueError('OLD_MODEL_U_IS_NOT_GOLD')
 if target['supervision']=='evidence_verified':
  require_review(target,'evidence')
  e=target['evidence']
  if not e or e['approval']!='human_approved':raise ValueError('UNVERIFIED_EVIDENCE')
  ids={s['span_id'] for u in row['uses'] for s in u['spans']}
  if not set(e['source_span_ids']).issubset(ids):raise ValueError('EVIDENCE_REF_NOT_IN_INPUT')
 if target['supervision']=='none':raise ValueError('NO_SUPERVISION')
 return True

def visible_input(row,system_prompt):
 """An allowlist renderer: no Gold class, path, book id, split, errors, or old prediction."""
 validate_input(row)
 uses=[{'view':i+1,'context_status':u['context_status'],'text':'\n'.join(s['text'] for s in u['spans']),
        'figure_refs':u['figure_refs'],'window_complete':not u['source_window_truncated']} for i,u in enumerate(row['uses'])]
 # This is request-local completeness. Global asset coverage belongs to the aggregator.
 body={'task_scope':'provided_use_set','this_use_set_complete':source_complete(row),
       'source_image_size_px':{'width':row['image']['width'],'height':row['image']['height']},'views':uses}
 return [{'role':'system','content':system_prompt},{'role':'user','content':[{'type':'image','image_ref':row['image']['ref']},{'type':'text','text':json.dumps(body,ensure_ascii=False,separators=(',',':'))}]}]

def aggregate(values,coverage_complete=True):
 if not values or any(v not in 'KDU' for v in values):raise ValueError('INVALID_USE_DECISIONS')
 if 'K' in values:return 'K'
 if not coverage_complete or 'U' in values:return 'U'
 return 'D'

def asset_plan_coverage(row):
 validate_schema('asset-plan',row)
 ids=[];groups=[]
 for g in row['groups']:ids+=g['use_ids'];groups.append(g['group_id'])
 if len(groups)!=len(set(groups)) or len(ids)!=len(set(ids)):raise ValueError('DUPLICATE_GROUP_USE')
 if not set(ids).issubset(row['global_use_ids']):raise ValueError('UNKNOWN_GLOBAL_USE')
 return bool(row['enumeration_complete'] and set(ids)==set(row['global_use_ids']))

def assert_no_cross_split_image_leak(rows):
 seen={}
 for r in rows:
  h=r['image_sha256'];role=r['role']
  if h in seen and seen[h]!=role:raise ValueError('CROSS_SPLIT_IMAGE_LEAK')
  seen[h]=role

def visible_label_conflicts(pairs,system_prompt):
 groups={}
 for row,target in pairs:
  key=digest(visible_input(row,system_prompt));groups.setdefault(key,set()).add(target['decision_target'])
 return {k:sorted(v) for k,v in groups.items() if len(v)>1}

def metrics(gold,predictions):
 if not gold or set(gold)!=set(predictions):raise ValueError('INCOMPLETE_PREDICTIONS')
 if any(x not in ('K','D') for x in gold.values()) or any(x not in ('K','D','U') for x in predictions.values()):raise ValueError('INVALID_LABEL')
 n=len(gold);tp=sum(gold[i]=='K' and predictions[i]=='K' for i in gold)
 fn=sum(gold[i]=='K' and predictions[i]=='D' for i in gold)
 correct=sum(gold[i]==predictions[i] for i in gold);out=sum(v!='D' for v in predictions.values())
 kept_needed=sum(gold[i]=='K' and predictions[i]!='D' for i in gold)
 nk=sum(v=='K' for v in gold.values());nd=n-nk
 precision=kept_needed/out if out else None;recall=kept_needed/nk if nk else None
 result={'total':n,'accuracy_u_wrong':correct/n,'output_keep_precision':precision,'output_necessary_recall':recall,
         'false_exclusions':fn,'unresolved':sum(v=='U' for v in predictions.values()),
         'decided_false_retention':sum(gold[i]=='D' and predictions[i]=='K' for i in gold),
         'output_false_retention':sum(gold[i]=='D' and predictions[i]!='D' for i in gold),
         'decoration_removal_recall':sum(gold[i]=='D' and predictions[i]=='D' for i in gold)/nd if nd else None}
 result['gate_passed']=bool(nk and nd and result['accuracy_u_wrong']>.97 and precision is not None and precision>.97 and recall==1 and fn==0)
 return result
