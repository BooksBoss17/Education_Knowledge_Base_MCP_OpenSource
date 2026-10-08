"""Bind figure references to their caption's image, not nearest prose.

Navigation evidence only. Tied assignments remain ambiguous; this module does
not decide retention, infer visible content, or read gold/pixels.
"""

import re

FIGURE=re.compile(r'^\s*图\s*(\d+(?:[.．－—–-]\d+)+)(?:\s|[\u3400-\u9fff]|$)')

REFERENCE=re.compile(r'图\s*(\d+(?:[.．－—–-]\d+)+)')

def normalize_number(value):
    return re.sub('[．－—–-]', '.', value)

def gap(a,b):
    return max(a[0]-b[2],b[0]-a[2],0)+max(a[1]-b[3],b[1]-a[3],0)

def caption_bindings(text_nodes,image_regions):
    bindings={}
    for node in text_nodes:
        text=node['text'].strip()
        match=FIGURE.match(text)
        if not match or len(text)>160 or node['bbox'][3]-node['bbox'][1]>55:
            continue
        candidates=[]
        for image in image_regions:
            b=image['bbox'];caption=node['bbox']
            horizontal=min(b[2],caption[2])-max(b[0],caption[0])
            distance=gap(b,caption)
            if horizontal<=0 or distance>90:
                continue
            candidates.append(dict(image_id=image['id'],bbox=b,distance=distance))
        candidates.sort(key=lambda c:c['distance'])
        if not candidates:
            continue
        best=candidates[0]['distance']
        owners=[c for c in candidates if c['distance']-best<10]
        number=normalize_number(match.group(1))
        bindings.setdefault(number,[]).append(dict(caption=text,caption_bbox=node['bbox'],
                                                  owners=owners,ambiguous=len(owners)!=1))
    return bindings

def attributed_references(text,target_bbox,bindings):
    result=[]
    for number in sorted({normalize_number(m.group(1)) for m in REFERENCE.finditer(text)}):
        records=bindings.get(number,[])
        owners=[o for record in records for o in record['owners']]
        overlaps=[]
        for owner in owners:
            b=owner['bbox'];intersection=max(0,min(b[2],target_bbox[2])-max(b[0],target_bbox[0]))*max(0,min(b[3],target_bbox[3])-max(b[1],target_bbox[1]))
            area=max(.001,(target_bbox[2]-target_bbox[0])*(target_bbox[3]-target_bbox[1]))
            overlaps.append(intersection/area)
        assigned='UNKNOWN'
        if records and not any(r['ambiguous'] for r in records):
            assigned='TARGET_REGION_OR_COMPONENT' if max(overlaps,default=0)>=.8 else 'OTHER_IMAGE_REGION'
        result.append(dict(figure=number,assignment=assigned,caption_records=records))
    return result
