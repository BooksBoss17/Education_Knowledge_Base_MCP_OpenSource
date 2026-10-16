"""Infer index lanes from source-aligned term/page pairs, not field gaps."""
import re
import statistics


def baseline_rows(blocks):
    tolerance=statistics.median(b['bbox_pdf_pt'][3]-b['bbox_pdf_pt'][1] for b in blocks)*.45
    rows=[];anchor=None
    for block in sorted(blocks,key=lambda b:((b['bbox_pdf_pt'][1]+b['bbox_pdf_pt'][3])/2,b['bbox_pdf_pt'][0])):
        y=(block['bbox_pdf_pt'][1]+block['bbox_pdf_pt'][3])/2
        if anchor is None or y-anchor>tolerance:
            rows.append([]);anchor=y
        rows[-1].append(block)
    return [sorted(row,key=lambda b:b['bbox_pdf_pt'][0]) for row in rows]


def paired_index_columns(blocks):
    pairs=[]
    for row in baseline_rows(blocks):
        for term,page in zip(row,row[1:]):
            name=str(term.get('content',{}).get('text') or '').strip()
            number=str(page.get('content',{}).get('text') or '').strip()
            if (not name or name.isdigit() or re.fullmatch(r'[A-Z]',name)
                    or not re.search(r'\w',name) or not number.isdigit()
                    or term['bbox_pdf_pt'][2]>page['bbox_pdf_pt'][0]):
                continue
            pairs.append((term['bbox_pdf_pt'][0],page['bbox_pdf_pt'][2]))
    if len(pairs)<6:return None
    ordered=sorted(pairs);right=ordered[0][1];splits=[]
    height=statistics.median(b['bbox_pdf_pt'][3]-b['bbox_pdf_pt'][1] for b in blocks)
    for left,end in ordered[1:]:
        if left-right>=max(18,height):splits.append((left+right)/2)
        right=max(right,end)
    if not 1<=len(splits)<=7:return None
    columns=[[] for _ in range(len(splits)+1)]
    for block in blocks:
        box=block['bbox_pdf_pt']
        if any(box[0]<cut<box[2] for cut in splits):return None
        columns[sum(box[0]>=cut for cut in splits)].append(block)
    support=[0]*len(columns)
    for left,right in pairs:
        if any(left<cut<right for cut in splits):return None
        support[sum(left>=cut for cut in splits)]+=1
    if min(support)<3 or min(map(len,columns))<4:return None
    return {'columns':columns,'splits':splits,'pair_counts':support}


def legacy_two_columns(blocks):
    """Preserve the prior TABLE-index fallback when paired evidence is absent."""
    ordered_x=sorted((b['bbox_pdf_pt'][0],b['bbox_pdf_pt'][2]) for b in blocks)
    right=ordered_x[0][1];gaps=[]
    for left,end in ordered_x[1:]:
        if left>right:gaps.append((left-right,(left+right)/2))
        right=max(right,end)
    span=max(x[1] for x in ordered_x)-min(x[0] for x in ordered_x)
    for gap,split in sorted(gaps,reverse=True):
        if gap<max(18,span*.07):break
        columns=[[b for b in blocks if (b['bbox_pdf_pt'][0]<split)==left] for left in (True,False)]
        if min(map(len,columns))<4:continue
        if any(sum(bool(str(b.get('content',{}).get('text') or '').strip())
                   and not str(b.get('content',{}).get('text') or '').strip().isdigit()
                   for b in column)<3 for column in columns):continue
        return {'columns':columns,'splits':[split],'pair_counts':None}
    return None
