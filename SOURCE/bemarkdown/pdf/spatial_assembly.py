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
        # A thin space before a typeset unit and the font used to print that
        # unit do not change its value. Keep distinct unit symbols distinct.
        value = re.sub(r"~?\\(?:mathrm|text)\{(kg|km/h|cm|mm|m/s|s|Hz|Pa)\}", r"\\unit{\1}", value)
        return re.sub(r"\{([A-Za-z](?:_\{[^{}]+\})?)\}", r"\1", value)
    normalized = {k: lexical(v) for k, v in values.items()}
    result = {}
    for child in regions:
        value = normalized.get(child.region_id, '')
        if not value:
            continue
        owners = [p for p in regions if p.page_index == child.page_index
                  and p.region_id != child.region_id and p.bbox != child.bbox
                  and (contains(p.bbox, child.bbox) or (
                      p.bbox[2]-p.bbox[0] >= 1.5*(child.bbox[2]-child.bbox[0])
                      and contains((p.bbox[0]-1, p.bbox[1]-1, p.bbox[2]+1, p.bbox[3]+1), child.bbox)))
                  and value in normalized.get(p.region_id, '')]
        if owners:
            owner = max(owners, key=lambda p: (p.bbox[2]-p.bbox[0])*(p.bbox[3]-p.bbox[1]))
            result[child.region_id] = owner.region_id
    return result


def merge_continuous_text_rows(parent, rows, fragments, target):
    """Merge only confirmed prose rows; retained slot pixels are the sole ink source."""
    if parent.normalized_label not in {'text', 'content', 'abstract'} or parent.metadata.get('review_reason'):
        return rows, fragments, []
    from PIL import Image
    by_id = {r.region_id: r for r in fragments}
    output_rows, additions, records = [], [], []
    removed = set()

    def build(run, merge_index):
        if len(run) < 2:
            return run, None, None
        regions = [by_id[row[0][1]] for row in run]
        heights = [r.bbox[3]-r.bbox[1] for r in regions]
        reference = sorted(heights)[len(heights)//2]
        if reference <= 0 or any(not .65*reference <= h <= 1.5*reference for h in heights):
            return run, None, None
        for previous, current in zip(regions, regions[1:]):
            gap = current.bbox[1]-previous.bbox[3]
            overlap = min(previous.bbox[2],current.bbox[2])-max(previous.bbox[0],current.bbox[0])
            width = min(previous.bbox[2]-previous.bbox[0],current.bbox[2]-current.bbox[0])
            if (not 0 <= gap <= .9*reference or overlap < .5*width
                    or abs(current.bbox[0]-previous.bbox[0]) > max(2.5*reference,.25*width)):
                return run, None, None
        images = []
        for region in regions:
            if region.metadata.get('review_reason') or region.metadata.get('assembly_parent') != parent.region_id:
                return run, None, None
            with Image.open(region.crop['path']) as source:
                if source.width <= 12 or source.height <= 12:
                    return run, None, None
                images.append(source.convert('RGB').crop((6,6,source.width-6,source.height-6)))
        widths = [r.bbox[2]-r.bbox[0] for r in regions]
        sx = images[0].width/widths[0]; sy = images[0].height/heights[0]
        if (any(abs(image.width/sx-width) > 2 for image,width in zip(images,widths))
                or any(abs(image.height/sy-height) > 2 for image,height in zip(images,heights))):
            for image in images:image.close()
            return run, None, None
        left = min(r.bbox[0] for r in regions); top = min(r.bbox[1] for r in regions)
        right = max(r.bbox[2] for r in regions); bottom = max(r.bbox[3] for r in regions)
        canvas = Image.new('RGB',(int(round((right-left)*sx))+12,int(round((bottom-top)*sy))+12),'white')
        sources = []
        for region, image in zip(regions,images):
            canvas.paste(image,(6+int(round((region.bbox[0]-left)*sx)),6+int(round((region.bbox[1]-top)*sy))))
            image.close()
            sources.append({'region_id':region.region_id,'bbox':list(region.bbox),'page_index':region.page_index,
                            'parent_region_id':parent.region_id,'source_crop_sha256':region.crop['sha256']})
        rid = f'{parent.region_id}__merge{merge_index:04d}'
        target.mkdir(parents=True,exist_ok=True)
        path = target/(rid+'.png'); canvas.save(path); canvas.close()
        crop = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        merged = replace(parent,region_id=rid,bbox=(left,top,right,bottom),crop=crop,
            metadata={**parent.metadata,'assembly_parent':parent.region_id,
                      'line_merge_sources':[s['region_id'] for s in sources],
                      'line_merge':{'parent_region_id':parent.region_id,'page_index':parent.page_index,
                                    'bbox':[left,top,right,bottom],'input_sha256':crop['sha256'],
                                    'sources':sources}})
        record = {'merged_region_id':rid,'parent_region_id':parent.region_id,'page_index':parent.page_index,
                  'bbox':[left,top,right,bottom],'input_sha256':crop['sha256'],'sources':sources}
        return [[['text',rid]]], merged, record

    position = 0
    while position < len(rows):
        if len(rows[position]) != 1 or rows[position][0][0] != 'text' or rows[position][0][1] not in by_id:
            output_rows.append(rows[position]); position += 1; continue
        end = position + 1
        while (end < len(rows) and len(rows[end]) == 1 and rows[end][0][0] == 'text'
               and rows[end][0][1] in by_id):
            end += 1
        # If the next mixed row starts with prose, its leading glyphs are not
        # present in this merge's pixels. The OCR model may nevertheless predict
        # them at the merge boundary (book12/p28: “电压” -> “电压为”), while the
        # retained next slot emits the same word again. Fall back to source cuts.
        if (end < len(rows) and rows[end] and rows[end][0][0] == 'text'
                and any(slot[0] == 'formula' for slot in rows[end])):
            output_rows.extend(rows[position:end])
            position = end
            continue
        replacement, merged, record = build(rows[position:end], len(records))
        output_rows.extend(replacement)
        if merged is not None:
            additions.append(merged); records.append(record)
            removed.update(source['region_id'] for source in record['sources'])
        position = end
    return output_rows, [r for r in fragments if r.region_id not in removed] + additions, records


def _bottom_overhang_line_evidence(parent, formula):
    """Return scale-normalized evidence for a small inline bottom overhang only."""
    import numpy as np
    from PIL import Image

    formula_height = formula.bbox[3]-formula.bbox[1]
    overhang = formula.bbox[3]-parent.bbox[3]
    covered = max(0.0, min(formula.bbox[3], parent.bbox[3])-max(formula.bbox[1], parent.bbox[1]))
    center = (formula.bbox[1]+formula.bbox[3])/2
    # A tall inline fraction may exceed the prose box, but most of its body and
    # its vertical center must still share that box. These ratios are independent
    # of render DPI and leave ample margin around the observed 92%+ true cases.
    if (formula_height <= 0 or covered/formula_height < .75
            or overhang <= 0 or overhang > .25*formula_height
            or not parent.bbox[1] <= center <= parent.bbox[3]):
        return None
    with Image.open(parent.crop['path']) as source:
        ink = np.asarray(source.convert('L')) < 225
    h, w = ink.shape
    sx = w/(parent.bbox[2]-parent.bbox[0])
    sy = h/(parent.bbox[3]-parent.bbox[1])
    # The parent crop is already formula-masked. Exclude the formula columns
    # again to ensure that only adjacent prose can supply same-line evidence.
    x0 = max(0, int((formula.bbox[0]-parent.bbox[0])*sx))
    x1 = min(w, int(np.ceil((formula.bbox[2]-parent.bbox[0])*sx)))
    outside = np.ones(w, dtype=bool)
    outside[x0:x1] = False
    if not outside.any():
        return None
    y0 = max(0, int((max(formula.bbox[1], parent.bbox[1])-parent.bbox[1])*sy))
    y1 = min(h, int(np.ceil((min(formula.bbox[3], parent.bbox[3])-parent.bbox[1])*sy)))
    if y1 <= y0:
        return None
    prose_rows = ink[:, outside].any(axis=1)
    runs = []
    start = None
    for y, occupied in enumerate([*prose_rows, False]):
        if occupied and start is None:
            start = y
        elif not occupied and start is not None:
            runs.append((start, y))
            start = None
    candidates = [(a, b, max(0, min(b, y1)-max(a, y0))) for a, b in runs]
    candidates = [row for row in candidates if row[2] > 0]
    if not candidates:
        return None
    a, b, shared_rows = max(candidates, key=lambda row: row[2])
    # Adjacent prose can occupy only the central part of a tall fraction. A
    # quarter of the covered band is sufficient; isolated touching ink is not.
    if shared_rows < .25*(y1-y0):
        return None
    line_height = (b-a)/sy
    # The bottom excess must also be small relative to the neighboring prose
    # line. Convert pixels back to plan units so raster scale cancels out.
    if line_height <= 0 or overhang > .5*line_height:
        return None
    return {'formula_height': formula_height, 'bottom_overhang': overhang,
            'covered_fraction': covered/formula_height,
            'adjacent_line_height': line_height,
            'same_line_fraction': shared_rows/(y1-y0)}


def prepare_slots(plan, formulas, output_dir, *, line_merge=False):
    """Split masked parents with unique containment or narrow overhang evidence.

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
        # Mask metadata records any positive rectangle intersection and is only
        # a candidate clue. Strict containment remains sufficient; a bottom
        # overhang additionally needs body, same-line prose and order evidence.
        masked = [t for t in plan.text_regions if t.page_index == f.page_index
                  and f.region_id in t.metadata.get('masked_formula_region_ids', [])]
        if masked:
            if len(masked) == 1:
                parent = masked[0]
                contained_here = contains((parent.bbox[0]-2, parent.bbox[1]-2,
                                           parent.bbox[2]+2, parent.bbox[3]+2), f.bbox)
                # Keep the old outer conditions, then only narrow this branch.
                # Top/side overhangs and ambiguous sources remain on the prior path.
                bottom_overhang = (parent.bbox[0]-2 <= f.bbox[0] and
                                   parent.bbox[1]-2 <= f.bbox[1] and
                                   parent.bbox[2]+2 >= f.bbox[2] and
                                   f.bbox[3] > parent.bbox[3]+2)
                if bottom_overhang:
                    parent_source = parent.metadata.get('source_region_id', parent.region_id)
                    reading_clear = parent.reading_order < f.reading_order
                    for other in plan.text_regions:
                        if other.page_index != parent.page_index:
                            continue
                        other_source = other.metadata.get('source_region_id', other.region_id)
                        if other.region_id == parent.region_id or other_source == parent_source:
                            continue
                        overlap_x = min(parent.bbox[2], other.bbox[2])-max(parent.bbox[0], other.bbox[0])
                        same_column = overlap_x > .5*min(parent.bbox[2]-parent.bbox[0],
                                                        other.bbox[2]-other.bbox[0])
                        if not same_column:
                            continue
                        if parent.reading_order < other.reading_order < f.reading_order:
                            reading_clear = False
                            break
                        if (other.reading_order >= parent.reading_order
                                and other.bbox[1] < f.bbox[3]
                                and other.bbox[3] > parent.bbox[3]):
                            reading_clear = False
                            break
                    bottom_overhang = (reading_clear and
                                       _bottom_overhang_line_evidence(parent, f) is not None)
                if contained_here or bottom_overhang:
                    ownership[f.region_id] = parent.region_id
            # A formula masked from multiple columns has no unique prose owner.
            continue
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
        # A tall fraction can bridge the whitespace between two prose rows.
        # Cut at its lower edge only when that edge falls between ink rows and
        # the following text has a substantial run of its own. Keep it full-width.
        cuts = set()
        for x_start, top_edge, x_end, bottom_edge in boxes.values():
            if bottom_edge >= h or bottom_edge - top_edge < 12:
                continue
            if any(b < bottom_edge < d for _, b, _, d in boxes.values()):
                continue
            above = ink[top_edge:bottom_edge].any(axis=1)
            prior = np.flatnonzero(above)
            gap = bottom_edge - top_edge - 1 - prior[-1] if len(prior) else 0
            if gap >= max(4, (bottom_edge-top_edge)*0.12):
                for band_start, band_end in bands:
                    if band_start < bottom_edge < band_end and band_end-bottom_edge >= 0.45*(bottom_edge-top_edge):
                        cuts.add(bottom_edge)
        # Fractions on successive lines can overlap vertically by a few
        # detector pixels. The formula crops stay whole; only the already
        # whitened prose image is split along an empty scanline.
        for top_edge in {b for _, b, _, _ in boxes.values()}:
            crossing = [(b, d) for _, b, _, d in boxes.values() if b < top_edge < d]
            if (crossing and all(d-top_edge <= 0.08*(d-b) for b, d in crossing)
                    and top_edge >= 4 and not ink[top_edge-4:top_edge].any()):
                cuts.add(top_edge)
        if cuts:
            split_bands = []
            for band_start, band_end in bands:
                edges = [band_start, *sorted(y for y in cuts if band_start < y < band_end), band_end]
                split_bands.extend(zip(edges, edges[1:]))
            bands = split_bands
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
            fs = sorted([f for f in local if top <= (boxes[f.region_id][1]+boxes[f.region_id][3])/2 < bottom],key=lambda f:boxes[f.region_id][0])
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
            merges = []
            if line_merge:
                rows, fragments, merges = merge_continuous_text_rows(parent, rows, fragments, target)
            out.extend(fragments)
            groups.append({'parent':parent,'rows':rows,'filtered_fragments':filtered,'line_merges':merges})
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
                           'filtered_fragments':group.get('filtered_fragments',[]),
                           'line_merges':group.get('line_merges',[])})
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
