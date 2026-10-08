"""Resolve enclosing layout regions versus directly adjacent caption images."""

import importlib.util, pathlib

spec=importlib.util.spec_from_file_location('v21',pathlib.Path(__file__).parent/'v21-source-attribution/source_pairing_v21.py')

V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)

V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])

def contains(outer,inner):return V.intersection(outer,inner)>=.999*area(inner)

def adjacent_below(image,caption):
    gap=caption[1]-image[3]
    width=min(image[2]-image[0],caption[2]-caption[0])
    overlap=max(0,min(image[2],caption[2])-max(image[0],caption[0]))
    return 0<=gap<=24 and width>0 and overlap/width>=.5

def attribute(number,target,visible_text,records,neighbor_captions):
    base=V.attribute(number,target,visible_text,records,neighbor_captions)
    owners={o['image_id']:o for r in records for o in r['owners']}
    if len(owners)<2:return base
    eligible={}
    for record in records:
        cap=record.get('caption_bbox')
        if not cap:continue
        for ident,owner in owners.items():
            box=owner['bbox']
            enclosing=[o for j,o in owners.items() if j!=ident and contains(o['bbox'],box) and area(box)<.75*area(o['bbox'])]
            if enclosing and adjacent_below(box,cap):eligible[ident]=owner
    # Distinct plausible child images remain ambiguous; enclosing layout regions
    # alone do not compete with an image directly above its caption.
    if len(eligible)!=1:return base
    chosen=next(iter(eligible.values()))
    if any(j!=chosen['image_id'] and not contains(o['bbox'],chosen['bbox']) for j,o in owners.items()):return base
    overlap=V.intersection(target,chosen['bbox'])
    if overlap>=.999*area(target) and contains(target,chosen['bbox']):label='TARGET_CAPTION_IMAGE'
    elif overlap==0:label='OTHER_IMAGE_REGION'
    else:return base
    return {'assignment':label,'caption_owner':chosen,'figure':number,
            'evidence':'single nested image directly above horizontally overlapping caption; enclosing layout regions removed; navigation only'}
