"""Add source-primitive visibility facts to the frozen 128 spatial inputs; no verdict rules."""

import collections, hashlib, json, math, unicodedata

from pathlib import Path

import numpy as np

import pymupdf

from PIL import Image,ImageDraw

from probe_zero99_vector_visibility import color_support,near_point_to_box

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def simple_shape_iou(rgb,drawing,box):
    if drawing.get('fill') is None or drawing.get('fill_opacity',0)<=0:return None
    items=drawing['items'];points=[]
    if len(items)==1 and items[0][0]=='re':
        rect=items[0][1];points=[rect.tl,rect.tr,rect.br,rect.bl]
    elif 3<=len(items)<=5 and all(i[0]=='l' for i in items):
        points=[i[1] for i in items]+[items[-1][2]]
        if math.hypot(points[0].x-points[-1].x,points[0].y-points[-1].y)>.1:return None
    else:return None
    image=Image.fromarray(rgb);image.thumbnail((256,256));w,h=image.size
    mask=Image.new('L',(w*4,h*4),0)
    coords=[((p.x-box[0])/(box[2]-box[0])*w*4,(p.y-box[1])/(box[3]-box[1])*h*4) for p in points]
    ImageDraw.Draw(mask).polygon(coords,fill=255);expected=np.asarray(mask.resize((w,h),Image.Resampling.LANCZOS))>=16
    observed=np.min(np.asarray(image),axis=-1)<245;union=int(np.count_nonzero(expected|observed))
    return float(np.count_nonzero(expected&observed)/union) if union else None

def primitive_facts(rgb,box,drawings,text_spans):
    matches=[];colours=[];nonwhite=int(np.count_nonzero(np.min(rgb,axis=-1)<245))
    for drawing in drawings:
        if max(abs(a-b) for a,b in zip(box,list(drawing['rect'])))>.5:continue
        supports=[]
        for key in ['fill','color']:
            colour=drawing.get(key)
            if colour is None:continue
            support=color_support(rgb,colour)
            if support.get('white_or_no_contrast'):continue
            colours.append(tuple(colour));supports.append({'channel':key,'rgb01':list(colour),'pixels_strict':support['strict_support_pixels'],'pixels_broad':support['broad_support_pixels']})
        matches.append({'paint_order':drawing.get('seqno'),'bbox':list(drawing['rect']),'path_type':drawing['type'],
                        'item_types':dict(collections.Counter(i[0] for i in drawing['items'])),'colours':supports,'simple_fill_foreground_iou':simple_shape_iou(rgb,drawing,box)})
    if not matches:return None,None
    lines=[]
    for drawing in drawings:
        if max(abs(a-b) for a,b in zip(box,list(drawing['rect'])))<=.5:continue
        colour=drawing.get('color')
        if colour is None or not any(max(abs(a-b) for a,b in zip(colour,c))<=.04 for c in colours):continue
        for item in drawing['items']:
            if item[0]!='l':continue
            a,b=item[1],item[2];length=math.hypot(a.x-b.x,a.y-b.y)
            if length>2*max(box[2]-box[0],box[3]-box[1]) and (near_point_to_box(a,box) or near_point_to_box(b,box)):
                lines.append({'a':list(a),'b':list(b),'length':length,'colour':list(colour)})
    symbols=[]
    for span in text_spans:
        actual=span['text'].strip();normal=unicodedata.normalize('NFKC',actual);bbox=span['bbox']
        if not normal or len(normal)>4 or not any(c.isascii() and c.isalpha() for c in normal):continue
        center=pymupdf.Point((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)
        if near_point_to_box(center,box,12):symbols.append({'text':actual,'normal':normal,'bbox':list(bbox),'center_inside':near_point_to_box(center,box,0)})
    summary={'evidence_scope':'original_crop_colour_and_simple_shape_support_only; not semantic judgment',
             'nonwhite_fraction':round(nonwhite/(rgb.shape[0]*rgb.shape[1]),5),
             'source_paths':[{'type':m['path_type'],'items':m['item_types'],'matching_colour_pixels':sum(c['pixels_strict'] for c in m['colours']),
                              'simple_shape_iou':None if m['simple_fill_foreground_iou'] is None else round(m['simple_fill_foreground_iou'],3)} for m in matches[:2]],
             'same_colour_long_strokes_touching_target':len(lines),'separate_source_symbols':[{'text':s['normal'],'inside_target_region':s['center_inside']} for s in symbols[:3]],
             'source_symbols_are_not_claimed_as_target_text':True,'no_colour_match_alone_is_not_deletion_proof':True}
    proof={'matched_paths':matches,'touching_long_same_colour_strokes':lines,'separate_pdf_text_symbols':symbols,'original_crop_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
    return summary,proof
