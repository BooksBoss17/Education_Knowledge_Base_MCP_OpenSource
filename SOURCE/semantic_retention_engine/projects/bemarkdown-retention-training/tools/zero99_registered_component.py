"""Source-backed component protection. Registration follows existing 200DPI/2px crop contract."""

import math,re,unicodedata

import numpy as np

from PIL import Image,ImageDraw

def pixel_frame(box,size,dpi=200,padding=2,page_size=None):
    scale=dpi/72.
    frame=[math.floor(max(0,box[0]*scale-padding)),math.floor(max(0,box[1]*scale-padding)),math.ceil(box[2]*scale+padding),math.ceil(box[3]*scale+padding)]
    if page_size:
        frame[2]=min(frame[2],math.ceil(page_size[0]*scale));frame[3]=min(frame[3],math.ceil(page_size[1]*scale))
    if (frame[2]-frame[0],frame[3]-frame[1])!=tuple(size):return None
    return {'pixel_bounds':frame,'scale':scale,'dpi':dpi,'padding_px':padding,'declared_size_verified':True}

def registered_iou(rgb,drawing,box,page_size):
    h,w=rgb.shape[:2];frame=pixel_frame(box,(w,h),page_size=page_size)
    if frame is None or w*h>262144 or drawing.get('fill') is None or drawing.get('fill_opacity',0)<=0:return None,frame
    items=drawing['items']
    if len(items)==1 and items[0][0]=='re':rect=items[0][1];points=[rect.tl,rect.tr,rect.br,rect.bl]
    elif len(items)==3 and all(i[0]=='l' for i in items):
        points=[i[1] for i in items]+[items[-1][2]]
        if math.hypot(points[0].x-points[-1].x,points[0].y-points[-1].y)>.1:return None,frame
    else:return None,frame
    origin=frame['pixel_bounds'];scale=frame['scale'];mask=Image.new('L',(w*4,h*4),0)
    ImageDraw.Draw(mask).polygon([((p.x*scale-origin[0])*4,(p.y*scale-origin[1])*4) for p in points],fill=255)
    expected=np.asarray(mask.resize((w,h),Image.Resampling.LANCZOS))>=16;actual=np.min(rgb,axis=-1)<245
    union=int(np.count_nonzero(expected|actual));return (float(np.count_nonzero(expected&actual)/union) if union else None),frame

def positive_component(view,paths,nonwhite_pixels):
    if nonwhite_pixels<=0:return None
    facts=view.get('source_primitive_visibility') or {};nodes=view.get('original_MD_text_blocks') or []
    md=unicodedata.normalize('NFKC','\n'.join(n['text'] for n in nodes));box=view['target_bbox']
    symbols=[s for s in facts.get('separate_source_symbols',[]) if re.fullmatch(r'[A-Za-z][0-9]?',s['text'])]
    supported=[p for p in paths if p.get('registered_iou') is not None and p['registered_iou']>=.8
               and p.get('pixels_supported_union',0)/nonwhite_pixels>=.8]
    def mentioned(symbol):return re.search(r'(?<![A-Za-z])'+re.escape(symbol)+r'(?![A-Za-z])',md) is not None
    for p in supported:
        if p['item_types']=={'l':3} and facts.get('same_colour_long_strokes_touching_target',0)>0:
            if any(mentioned(s['text']) for s in symbols) and re.search(r'如图|图中|图示|示意图',md):
                return {'kind':'VISIBLE_SYMBOL_BOUND_CONNECTED_TRIANGLE','basis':'visible registered triangle, attached long same-colour stroke, nearby source symbol also in referenced MD'}
        if p['item_types']=={'re':1} and any(s['inside_target_region'] and mentioned(s['text']) for s in symbols):
            for n in nodes:
                caption=re.fullmatch(r'\s*第\s*(\d+)\s*题\s*',n['text']);b=n['bbox']
                if caption and b[1]>=box[3]-.5 and b[1]-box[3]<=40 and max(box[0]-b[2],b[0]-box[2],0)<=40:
                    number=caption.group(1)
                    if re.search(r'(?:^|\n)\s*'+number+r'\s*[.．、]',md):
                        return {'kind':'VISIBLE_LABELLED_RECTANGLE_IN_NUMBERED_PROBLEM','problem_number':number,'basis':'registered visible rectangle, independent source label inside, matching below-figure question caption and actual question text'}
    return None
