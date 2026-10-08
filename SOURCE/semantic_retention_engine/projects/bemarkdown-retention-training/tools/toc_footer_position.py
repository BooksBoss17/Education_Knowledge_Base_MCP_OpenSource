"""Resolve a visible singleton footer token by its own source-text position, not canvas size."""

import re

import unicodedata

def token(text):
    value=re.sub(r'[\s()（）\[\]【】\-—–]','',unicodedata.normalize('NFKC',text))
    return value if re.fullmatch(r'\d{1,3}|[IVXLCDMivxlcdm]{1,6}',value) else None

def footer_witness(visible_text,target_bbox,page_size,nodes):
    value=token(visible_text)
    if value is None:return None
    matches=[]
    for node in nodes:
        box=node['bbox']
        inside=(box[0]>=target_bbox[0]-.5 and box[1]>=target_bbox[1]-.5 and
                box[2]<=target_bbox[2]+.5 and box[3]<=target_bbox[3]+.5)
        if inside and token(node['text'])==value:matches.append(node)
    # Duplicate matching numbers inside the source region remain ambiguous.
    if len(matches)!=1:return None
    match=matches[0];box=match['bbox']
    if box[1]<.9*page_size[1]:return None
    peers=[]
    for node in nodes:
        if node is match or token(node['text']) is not None:continue
        other=node['bbox']
        overlap=max(0,min(box[3],other[3])-max(box[1],other[1]))
        gap=max(box[0]-other[2],other[0]-box[2],0)
        if overlap>=.5*min(box[3]-box[1],other[3]-other[1]) and gap<=40 and re.search(r'[\u3400-\u9fffA-Za-z]',node['text']):
            peers.append(node)
    if peers:return None
    return {'visible_token':value,'source_text':match['text'],'source_bbox':box,
            'unique_token_in_target_region':True,'glyph_in_bottom10percent':True,
            'no_same_line_catalogue_text_within_40pt':True,
            'scope':'position validation of an already-observed singleton; not source text claimed as image content'}
