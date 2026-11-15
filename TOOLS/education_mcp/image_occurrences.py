"""Recover per-reference ownership from renderer spans and verified review edits.

Deduplicated binary assets are not occurrence identities. Never infer a page from
image numbers, nearest boxes or the order of IR nodes. Carry an original rendered
reference through a review hunk only when that hunk preserves its image references.
"""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote


def text_sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def reference_tokens(text, pattern, destination):
    result=[]
    for match in pattern.finditer(text):
        raw=match.group();prefix=raw.index('](')+2;dest=destination.fullmatch(raw[prefix:-1])
        if dest is None:raise ValueError('Unsupported image reference')
        group=1 if dest.group(1) is not None else 2
        result.append((match.start(),match.end(),match.start()+prefix+dest.start(group),
                       match.start()+prefix+dest.end(group),unquote(dest.group(group))))
    return result


def tokens(text, pattern, destination):
    # A destination span survives edits to alt text without losing its origin.
    return [(row[2],row[3],row[4]) for row in reference_tokens(text,pattern,destination)]


def initial_anchors(original, rendering, tokenize):
    raw=original.encode('utf8')
    if rendering.get('offset_unit')!='UTF8_BYTES' or rendering.get('determinism_sha256')!=text_sha(original):
        raise ValueError('Original Markdown differs from renderer identity')
    spans=rendering.get('node_spans',[]);starts=[];last=0;ids=set()
    for span in spans:
        begin,end=span.get('byte_start'),span.get('byte_end')
        if (type(begin) is not int or type(end) is not int or not last<=begin<end<=len(raw)
                or not span.get('node_id') or span['node_id'] in ids
                or type(span.get('page_index')) is not int or span['page_index']<0):
            raise ValueError('Invalid or overlapping renderer spans')
        if hashlib.sha256(raw[begin:end]).hexdigest()!=span.get('sha256'):
            raise ValueError('Renderer span payload mismatch')
        starts.append(begin);last=end;ids.add(span['node_id'])
    offsets=[0]
    for char in original:offsets.append(offsets[-1]+len(char.encode('utf8')))
    anchors=[]
    for begin,end,name in tokenize(original):
        index=bisect_right(starts,offsets[begin])-1
        span=spans[index] if index>=0 else None
        origin=None
        if span and offsets[end]<=span['byte_end']:
            origin=dict(source_part=f"page:{span['page_index']}",source_document_node_id=span['node_id'],
                        source_document_kind=span.get('kind'),source_node_review_state=span.get('review_state'),
                        bbox_pdf_pt=span.get('bbox_pdf_pt'),source_mapping='RENDERER_SPAN_AND_REVIEW_REPLAY')
        anchors.append(dict(start=begin,end=end,asset_name=name,origin=origin))
    return anchors


def replay(original, current, anchors, journal, tokenize):
    if journal.get('original_sha256')!=text_sha(original):
        raise ValueError('Review journal original identity mismatch')
    text=original
    for revision in journal.get('revisions',[]):
        if revision.get('base_sha256')!=text_sha(text):raise ValueError('Discontinuous review chain')
        for row in revision.get('replacements',[]):
            old,new=row.get('old'),row.get('new')
            if not isinstance(old,str) or not old or not isinstance(new,str) or text.count(old)!=1:
                raise ValueError('Cannot replay exact review replacement')
            begin=text.index(old);end=begin+len(old);delta=len(new)-len(old)
            updated=text[:begin]+new+text[end:]
            affected=any(a['start']<end and a['end']>begin for a in anchors)
            before=([r for r in tokenize(text) if begin<=r[0] and r[1]<=end] if affected else [])
            updated_tokens=tokenize(updated) if affected else []
            after=[r for r in updated_tokens if begin<=r[0] and r[1]<=begin+len(new)]
            remap={}
            if [r[2] for r in before]==[r[2] for r in after]:
                remap={(a[0],a[1]):(b[0],b[1]) for a,b in zip(before,after)}
            following=[]
            for anchor in anchors:
                if anchor['end']<=begin:following.append(anchor)
                elif anchor['start']>=end:
                    following.append(dict(anchor,start=anchor['start']+delta,end=anchor['end']+delta))
                elif (anchor['start'],anchor['end']) in remap:
                    left,right=remap[(anchor['start'],anchor['end'])]
                    following.append(dict(anchor,start=left,end=right))
                else:
                    # A hunk may cut through a destination without changing which
                    # asset it names. Use its untouched prefix/suffix's exact new
                    # coordinate, never the nearest or first matching image.
                    matches=[]
                    for left,right,name in updated_tokens:
                        if name!=anchor['asset_name']:continue
                        prefix_intact=anchor['start']<begin and left==anchor['start']
                        suffix_intact=anchor['end']>end and right==anchor['end']+delta
                        if prefix_intact or suffix_intact:
                            matches.append((left,right))
                    if len(matches)==1:
                        left,right=matches[0]
                        following.append(dict(anchor,start=left,end=right))
                # Changes that cannot preserve an exact original destination anchor
                # lose ownership; no similarity or sequence-only fallback.
            anchors=following;text=updated
        if revision.get('result_sha256')!=text_sha(text):raise ValueError('Review result identity mismatch')
    if text!=current:raise ValueError('Current Markdown differs from replayed edits')
    return anchors


def resolve(package,current,source_sha256,pattern,destination):
    package=Path(package);tokenize=lambda text:tokens(text,pattern,destination)
    try:
        report=json.loads((package/'conversion_report.json').read_text('utf-8-sig'))
        if report.get('source',{}).get('sha256')!=source_sha256:raise ValueError('Renderer source mismatch')
        original=(package/'document.md').read_text('utf8')
        anchors=initial_anchors(original,report.get('markdown_render',{}),tokenize)
        if current!=original:
            anchors=replay(original,current,anchors,json.loads((package/'agent_review.json').read_text('utf-8-sig')),tokenize)
        by_range={(r['start'],r['end'],r['asset_name']):r['origin'] for r in anchors if r['origin']}
        found={r[0]:by_range[(r[2],r[3],r[4])] for r in reference_tokens(current,pattern,destination)
               if (r[2],r[3],r[4]) in by_range}
        return dict(status='VERIFIED',by_start=found)
    except (OSError,ValueError,TypeError,KeyError) as error:
        return dict(status='UNAVAILABLE',by_start={},reason=str(error))
