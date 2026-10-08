"""Build page-bound Markdown evidence and compact model inputs for new packages.

Context validates the package/report source hashes. Native PDF text is used only
for geometry of captions whose actual text is present in the same-page Markdown.
Neither Gold nor previous image classifications enter this construction.
"""

import importlib.util,pathlib,re,unicodedata

B=pathlib.Path(__file__).parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

V20=module('source_v20_builder',B/'source_pairing_v20.py')

V24=module('source_v24_builder',B/'source_pairing_v24.py')

def canonical(text):
    return re.sub(r'[\s#*_`$\\{}]','',unicodedata.normalize('NFKC',text)).replace('－','-').replace('—','-')

def build(context,prepared,asset,occurrence,observation=None,max_chars=2400):
    page=occurrence['page'];box=occurrence['bbox'];facts=context.facts(asset,occurrence)
    raw=context.raw
    nodes=[]
    for n in context.pages.get(page,[]):
        if n['kind'] in ('IMAGE','OTHER'):continue
        start,end=n['byte_start'],n['byte_end']
        actual=raw[start:end].decode('utf-8')
        assert actual==n['text'],'Node does not match actual MD bytes'
        nodes.append({'text':actual,'bbox':n['bbox_pdf_pt'],'kind':n['kind'],
                      'byte_start':start,'byte_end':end,'page':page})
    page_text='\n'.join(n['text'] for n in nodes)
    peer_regions=[{'id':a['asset_name']+':'+str(i),'bbox':o['bbox']}
                  for a in prepared['assets'] for i,o in enumerate(a['occurrences']) if o['page']==page]
    caption_nodes=[{'text':n['text'],'bbox':n['bbox']} for n in nodes]
    normalized_page=canonical(page_text)
    for n in context.native.get(page,[]):
        text=n['text'].strip()
        if V20.FIGURE.match(text) and len(text)<=160 and canonical(text) in normalized_page:
            caption_nodes.append({'text':text,'bbox':n['bbox']})
    bindings=V20.caption_bindings(caption_nodes,peer_regions)
    cards=[]
    for n in nodes:
        refs=[]
        for number in sorted({V20.normalize_number(m.group(1)) for m in V20.REFERENCE.finditer(n['text'])}):
            assigned=V24.attribute(number,box,(observation or {}).get('visible_text',''),bindings.get(number,[]),[])
            refs.append({'figure':number,**assigned})
        cards.append({'field':'actual_same_page_md','text':n['text'],'bbox':n['bbox'],
                      'gap':V20.gap(box,n['bbox']),'figure_refs':refs,
                      'source_provenance':{'pages':[page],'byte_start':n['byte_start'],
                         'byte_end':n['byte_end'],'method':'exact current MD node bytes'}})
    # Bound target captions first, then nearest complete blocks. Never take
    # arbitrary previous/following lines from another page.
    priority={'TARGET_CAPTION_IMAGE':0,'OWN_PRINTED_CAPTION':0,
              'TARGET_REGION_OR_COMPONENT':1,'TARGET_GROUP_CANDIDATE':2}
    cards.sort(key=lambda c:(min((priority.get(r['assignment'],3) for r in c['figure_refs']),default=3),c['gap']))
    chosen=[];budget=0;seen=set()
    for c in cards:
        text=c['text'].strip()
        if not text or text in seen:continue
        if budget+len(text)>max_chars:continue
        if len(chosen)>=5:break
        chosen.append(c);seen.add(text);budget+=len(text)
    usage={'page':page,'page_role':{'document_role':facts['document_role'],
                 'page_number':facts['page_number'],'role_evidence':{
                    'directory_heading_present':bool(re.search(r'目录|contents',page_text,re.I)),
                    'dot_leader_entries':len(re.findall(r'[·.]{5,}\s*\d+',page_text)),
                    'warning':'页角色不能覆盖裁图自身的有意义文字、真实数据或实际科学用途。'}},
           'figure_attribution':{'records':chosen},
           'warning':'所有文本均来自目标实际同页MD；仍须核对归属，不能当成图内内容。'}
    proof={'target_page':page,'target_bbox':box,'document_role':facts['document_role'],
           'page_nodes':nodes,'page_text':page_text,'image_regions':peer_regions,
           'caption_bindings':bindings,'layout':facts.get('layout',[]),
           'source_usage_candidates':usage,'facts':facts}
    return usage,proof
