"""Exact-MD text with source geometry, no semantic role guesses or Gold routing."""

import hashlib

import json

def gap(a,b):
    return max(a[0]-b[2],b[0]-a[2],0)+max(a[1]-b[3],b[1]-a[3],0)

def layout_view(raw,use,occurrence,nodes,peers,text_budget=1000,max_nodes=20):
    assert use['page_index']==occurrence['page'] and use['use_id']==occurrence['id']
    target=occurrence['bbox'];candidates=[]
    for node in nodes:
        if node['page_index']!=use['page_index'] or node['kind'] in {'IMAGE','OTHER'}:continue
        start,end=node['byte_start'],node['byte_end'];value=raw[start:end];text=value.decode('utf-8')
        if node.get('sha256'):assert hashlib.sha256(value).hexdigest()==node['sha256']
        if not text.strip():continue
        candidates.append({'bbox':node['bbox_pdf_pt'],'text':text,'start':start,'end':end,'gap':gap(target,node['bbox_pdf_pt'])})
    candidates.sort(key=lambda x:(x['gap'],x['start']))
    chosen=[];length=0
    for c in candidates:
        if len(chosen)>=max_nodes:break
        if length+len(c['text'])>text_budget:continue
        chosen.append(c);length+=len(c['text'])
    chosen.sort(key=lambda c:c['start'])
    nearby=[x for x in peers if x['page']==occurrence['page'] and x['id']!=occurrence['id']]
    nearby.sort(key=lambda x:gap(target,x['bbox']))
    view={'record_type':'source_layout_not_contents_printed_in_target','coordinate_unit':'PDF point, origin upper left',
          'target_bbox':[round(v,1) for v in target],
          'other_image_bboxes':[[round(v,1) for v in x['bbox']] for x in nearby[:6]],
          'original_MD_text_blocks':[{'bbox':[round(v,1) for v in c['bbox']],'text':c['text']} for c in chosen]}
    text=json.dumps(view,ensure_ascii=False,separators=(',',':'))
    proof={'target_bbox_exact':target,'source_occurrence_bytes':[occurrence['byte_start'],occurrence['byte_end']],
           'selected_spans':[{'start':c['start'],'end':c['end'],'sha256':hashlib.sha256(c['text'].encode()).hexdigest(),'bbox':c['bbox']} for c in chosen],
           'text_budget':text_budget,'max_nodes':max_nodes,'all_source_text_is_verbatim':True,'neighbor_text_not_claimed_as_image_content':True,
           'selected_text_chars':length,'source_page':use['page_index'],'source_nodes_on_other_pages_used':False}
    return text,proof
