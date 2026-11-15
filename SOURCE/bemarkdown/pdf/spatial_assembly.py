"""Source-bound local formula slots; ordinary prose keeps its original context."""
from dataclasses import replace
from pathlib import Path
import hashlib
import re


def deduplicate(regions):
    kept, seen = [], {}
    for region in regions:
        role = region.normalized_label
        if role in {'image', 'chart'}:
            role = 'figure'
        key = (region.page_index, tuple(region.bbox), role,
               region.crop.get('sha256') or region.crop.get('content_sha256'))
        if key in seen:
            i = seen[key]
            old = kept[i]
            kept[i] = replace(old, metadata={**old.metadata, 'source_aliases':
                [*old.metadata.get('source_aliases', []), region.region_id]})
        else:
            seen[key] = len(kept)
            kept.append(region)
    return kept


def contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def fragment_decision(ink, reference_height):
    """Filter only generated residual slots, never complete layout/formula crops.

    The low-information policy intentionally accepts isolated punctuation loss.
    Continuous strokes (including narrow letters) and horizontal signs survive.
    """
    import numpy as np
    yy, xx = np.nonzero(ink)
    ref = max(1.0, float(reference_height))
    if not len(xx):
        return 'EMPTY', None, {'ink_pixels': 0, 'reference_height': ref}
    a,b,c,d = int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    width,height = c-a,d-b
    core=ink[b:d,a:c]
    longest=0
    for col in core.T:
        current=0
        for pixel in col:
            current=current+1 if pixel else 0
            longest=max(longest,current)
    metrics={'ink_pixels':int(len(xx)), 'ink_width':width,'ink_height':height,
             'reference_height':ref,'longest_vertical_stroke':longest}
    reason=None
    horizontal_sign = width >= .35*ref and width >= 2.5*height
    touches_edge = a == 0 or c == ink.shape[1]
    if not horizontal_sign:
        if width <= .32*ref and height <= .32*ref and len(xx) <= .06*ref*ref:
            reason='ISOLATED_LOW_INFORMATION_MARK'
        elif touches_edge and width <= .22*ref and len(xx) <= .18*ref*ref and longest < .5*ref:
            reason='SPARSE_CUT_EDGE_SLIVER'
    return reason,(a,b,c,d),metrics


def contained_results(regions, values):
    """Suppress only same-page contained formulas with literal output coverage."""
    def lexical(value):
        value = re.sub(r'\s+', '', value).replace(r'^{\prime}', "'")
        return re.sub(r"\{([A-Za-z](?:_\{[^{}]+\})?)\}", r"\1", value)
    normalized = {k: lexical(v) for k, v in values.items()}
    result = {}
    for child in regions:
        value = normalized.get(child.region_id, '')
        if not value:
            continue
        owners = [p for p in regions if p.page_index == child.page_index
                  and p.region_id != child.region_id and p.bbox != child.bbox
                  and contains(p.bbox, child.bbox)
                  and value in normalized.get(p.region_id, '')]
        if owners:
            owner = max(owners, key=lambda p: (p.bbox[2]-p.bbox[0])*(p.bbox[3]-p.bbox[1]))
            result[child.region_id] = owner.region_id
    return result


def prepare_slots(plan, formulas, output_dir):
    """Split only masked parents whose formula ownership is unique and contained.

    Horizontal cuts are blank scanlines. Vertical cuts abut already masked
    rectangles. Low-information residuals follow the explicit filtering policy;
    their original crops and decisions remain available outside the text stream.
    """
    if output_dir is None or plan.policy.policy_id != 'bemarkdown-local-model-api-v1':
        return plan, [], {}, []
    import numpy as np
    from PIL import Image
    suppressed = contained_results(plan.formula_regions, formulas)
    active = [f for f in plan.formula_regions if formulas.get(f.region_id) and f.region_id not in suppressed]
    ownership = {}
    for f in active:
        candidates = [t for t in plan.text_regions if t.page_index == f.page_index and contains((t.bbox[0]-2,t.bbox[1]-2,t.bbox[2]+2,t.bbox[3]+2), f.bbox)]
        if len(candidates) == 1:
            ownership[f.region_id] = candidates[0].region_id
    out, groups, reviews = [], [], []
    target = Path(output_dir)/'_bemarkdown/visual_serial/slots'
    for parent in plan.text_regions:
        local = [f for f in active if ownership.get(f.region_id) == parent.region_id]
        table_overlap = any(t.page_index == parent.page_index and
            min(t.bbox[2],parent.bbox[2]) > max(t.bbox[0],parent.bbox[0]) and
            min(t.bbox[3],parent.bbox[3]) > max(t.bbox[1],parent.bbox[1]) for t in plan.table_regions)
        if table_overlap and local:
            reviews.append({'region_id':parent.region_id,'reason':'SPATIAL_TABLE_PARENT_REVIEW'})
        if not local or table_overlap:
            out.append(parent)
            continue
        with Image.open(parent.crop['path']) as im:
            image = im.convert('RGB')
        ink = np.asarray(image.convert('L')) < 225
        h, w = ink.shape
        x0,y0,x1,y1 = parent.bbox
        sx,sy = w/(x1-x0),h/(y1-y0)
        boxes = {f.region_id:(max(0,int((f.bbox[0]-x0)*sx)),max(0,int((f.bbox[1]-y0)*sy)),
                               min(w,int(round((f.bbox[2]-x0)*sx))),min(h,int(round((f.bbox[3]-y0)*sy)))) for f in local}
        occupied = ink.any(axis=1)
        for a,b,c,d in boxes.values():
            occupied[b:d] = True
        ys = np.flatnonzero(occupied)
        bands = []
        for y in ys:
            if not bands or y-bands[-1][1] > 3:
                bands.append([int(y),int(y)+1])
            else:
                bands[-1][1] = int(y)+1
        # A separate row consisting only of a few dots must not define its own
        # font size. Estimate a floor from normal ink runs in the same parent.
        runs=[];length=0
        for occupied_row in [*ink.any(axis=1),False]:
            if occupied_row:
                length+=1
            elif length:
                if length >= 8:runs.append(length)
                length=0
        parent_reference=float(np.median(runs)) if runs else .5*float(np.median([d-b for a,b,c,d in boxes.values()]))
        rows, fragments, invalid, filtered = [], [], False, []
        candidate_index = 0
        for top,bottom in bands:
            fs = sorted([f for f in local if boxes[f.region_id][1] < bottom and boxes[f.region_id][3] > top],key=lambda f:boxes[f.region_id][0])
            if any(boxes[a.region_id][2] > boxes[b.region_id][0] for a,b in zip(fs,fs[1:])):
                invalid = True
                break
            # Estimate normal text height from sizeable unmasked spans in this
            # row; a tall fraction must not inflate the deletion threshold.
            intervals=[];cursor=0
            for item in [*fs,None]:
                edge=boxes[item.region_id][0] if item else w
                yy,xx=np.nonzero(ink[top:bottom,cursor:edge])
                if len(xx) and xx.max()-xx.min()+1 >= .5*(bottom-top):
                    intervals.append(int(yy.max()-yy.min()+1))
                if item:cursor=boxes[item.region_id][2]
            reference_height=max(parent_reference,float(np.median(intervals)) if intervals else bottom-top)
            row, left = [], 0
            for f in [*fs, None]:
                right = boxes[f.region_id][0] if f else w
                if right > left and ink[top:bottom,left:right].any():
                    rid = f'{parent.region_id}__slot{candidate_index:04d}'
                    candidate_index += 1
                    target.mkdir(parents=True,exist_ok=True)
                    path = target/(rid+'.png')
                    reason,bounds,metrics=fragment_decision(ink[top:bottom,left:right],reference_height)
                    if reason:
                        rejected=target/'filtered'/(rid+'.png')
                        rejected.parent.mkdir(parents=True,exist_ok=True)
                        image.crop((left,top,right,bottom)).save(rejected)
                        filtered.append({'region_id':rid,'parent_region_id':parent.region_id,
                            'reason':reason,'policy':'RESIDUAL_LOW_INFORMATION_V1',
                            'bbox':[x0+left/sx,y0+top/sy,x0+right/sx,y0+bottom/sy],
                            'path':str(rejected),'sha256':hashlib.sha256(rejected.read_bytes()).hexdigest(),
                            'metrics':metrics})
                        if f:
                            row.append(['formula',f.region_id]);left=boxes[f.region_id][2]
                        continue
                    # Padding is white only; never borrow another owner's pixels.
                    # Keep retained model inputs byte-identical to the baseline.
                    # Tightening whitespace changed OCR on legitimate text.
                    cut = image.crop((left,top,right,bottom))
                    padded = Image.new('RGB',(cut.width+12,cut.height+12),'white')
                    padded.paste(cut,(6,6));padded.save(path)
                    crop = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                    fragments.append(replace(parent,region_id=rid,bbox=(x0+left/sx,y0+top/sy,x0+right/sx,y0+bottom/sy),crop=crop,
                        metadata={**parent.metadata,'assembly_parent':parent.region_id}))
                    row.append(['text',rid])
                if f:
                    row.append(['formula',f.region_id]);left=boxes[f.region_id][2]
            rows.append(row)
        if invalid or not rows:
            out.append(parent)
            reviews.append({'region_id':parent.region_id,'reason':'SPATIAL_SLOT_OVERLAP_UNRESOLVED'})
        else:
            out.extend(fragments)
            groups.append({'parent':parent,'rows':rows,'filtered_fragments':filtered})
    return replace(plan,text_regions=tuple(out)),groups,suppressed,reviews


def assemble(groups, texts, formulas, markdown_ids):
    outputs, consumed, provenance = {}, [], []
    for group in groups:
        rows=[]
        for row in group['rows']:
            parts=[]
            for kind,rid in row:
                value = (formulas if kind == 'formula' else texts).get(rid,'').strip()
                parts.append('$'+value+'$' if kind=='formula' and rid not in markdown_ids else value)
                consumed.append(rid)
            rows.append(' '.join(parts))
        parent=group['parent']
        outputs[parent.region_id]='\n\n'.join(rows)
        provenance.append({'parent':parent.to_dict(),'rows':group['rows'],
                           'filtered_fragments':group.get('filtered_fragments',[])})
    return outputs,consumed,provenance


def duplicate_slot_text(regions, values):
    """Equal text is redundant only when its actual ink survives at the same pixels."""
    import numpy as np
    from PIL import Image
    kept, decisions, masks = [], [], {}

    def pixels(r):
        if r.region_id not in masks:
            with Image.open(r.crop['path']) as im:
                a = np.asarray(im.convert('L')) < 225
            if r.metadata.get('assembly_parent'):
                a = a[6:-6,6:-6]
            masks[r.region_id] = a
        return masks[r.region_id]

    for r in sorted(regions,key=lambda r:-(r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1])):
        key=re.sub(r'\s+','',values.get(r.region_id,''))
        if not key:
            continue
        duplicate=None
        for other,other_key in kept:
            if other.page_index!=r.page_index or key!=other_key:
                continue
            if not (r.metadata.get('assembly_parent') or other.metadata.get('assembly_parent')):
                continue
            if min(r.bbox[2],other.bbox[2])<=max(r.bbox[0],other.bbox[0]) or min(r.bbox[3],other.bbox[3])<=max(r.bbox[1],other.bbox[1]):
                continue
            a,b=pixels(r),pixels(other)
            yy,xx=np.nonzero(a)
            if not len(xx):
                continue
            px=r.bbox[0]+(xx+.5)*(r.bbox[2]-r.bbox[0])/a.shape[1]
            py=r.bbox[1]+(yy+.5)*(r.bbox[3]-r.bbox[1])/a.shape[0]
            ox=np.floor((px-other.bbox[0])*b.shape[1]/(other.bbox[2]-other.bbox[0])).astype(int)
            oy=np.floor((py-other.bbox[1])*b.shape[0]/(other.bbox[3]-other.bbox[1])).astype(int)
            if ((ox>=0)&(oy>=0)&(ox<b.shape[1])&(oy<b.shape[0])).all() and b[oy,ox].all():
                duplicate=other;break
        if duplicate:
            decisions.append({'region_id':r.region_id,'retained_region_id':duplicate.region_id,'reason':'DUPLICATE_SLOT_TEXT_PIXELS_CONFIRMED'})
        else:
            kept.append((r,key))
    return decisions


def caption_owners(regions, text):
    owners = {}
    for caption in regions:
        if caption.label != 'figure_title' or not text.get(caption.region_id):
            continue
        a,b,c,d = caption.bbox
        candidates = [r for r in regions if r.page_index == caption.page_index and r.label in {'chart','image'}
                      and r.bbox[0] <= (a+c)/2 <= r.bbox[2]
                      and abs(b-r.bbox[3]) <= 2*(d-b)]
        if len(candidates)==1:
            owners.setdefault(candidates[0].region_id,[]).append(caption.region_id)
    return owners


def complete_figure_edges(plan, renders, output_dir):
    """Extend a clipped bottom edge to the first nearby blank row, without OCR."""
    if not renders or output_dir is None:
        return plan
    import numpy as np
    from PIL import Image
    from io import BytesIO
    changed = {}
    for region in plan.preserved_regions:
        if region.label not in {'chart','image'}:
            continue
        with Image.open(BytesIO(renders[region.page_index].image_bytes)) as im:
            image=im.convert('RGB')
        x0,y0,x1,y1=map(int,region.bbox)
        ink=np.asarray(image.convert('L'))<225
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 < image.height):
            continue
        if not ink[y1-1,x0:x1].any():
            continue
        limit=min(image.height,y1+max(3,min(40,(y1-y0)//6)))
        end=next((y for y in range(y1,limit-1) if not ink[y:y+2,x0:x1].any()),None)
        if end is None:
            continue
        end+=2
        # Another detected object in the extension makes ownership ambiguous.
        if any(r.region_id!=region.region_id and r.page_index==region.page_index
               and min(x1,r.bbox[2])>max(x0,r.bbox[0])
               and min(end,r.bbox[3])>max(y1,r.bbox[1]) for r in plan.all_regions):
            continue
        path=Path(output_dir)/'_bemarkdown/visual_serial/figure_edges'/(region.region_id+'.png')
        path.parent.mkdir(parents=True,exist_ok=True);image.crop((x0,y0,x1,end)).save(path)
        changed[region.region_id]=replace(region,bbox=(x0,y0,x1,end),
            crop={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
            metadata={**region.metadata,'original_bbox':list(region.bbox),'edge_policy':'NEARBY_BLANK_ROWS_V1'})
    return replace(plan,all_regions=tuple(changed.get(r.region_id,r) for r in plan.all_regions),
                   preserved_regions=tuple(changed.get(r.region_id,r) for r in plan.preserved_regions))
