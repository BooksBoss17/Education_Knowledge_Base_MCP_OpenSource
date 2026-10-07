"""Native PDF context vetoes for image filtering, never guessed asset identity.

All boxes are PDF points. Spatial relationships are protection evidence only;
they do not establish source_region_id/content ownership or text coverage.
"""
from __future__ import annotations

import math
import re

CAPTION = re.compile(r'(?:图\s*\d|第\s*\d+\s*题|[Ff]ig(?:ure)?\s*[.\d])')


def valid_box(box):
    return (isinstance(box,(list,tuple)) and len(box)==4
            and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box)
            and box[0]<box[2] and box[1]<box[3])


def intersects(a,b,pad=0):
    return a[0]-pad<=b[2] and b[0]<=a[2]+pad and a[1]-pad<=b[3] and b[1]<=a[3]+pad


def contains(a,b,pad=0):
    return a[0]-pad<=b[0] and a[1]-pad<=b[1] and a[2]+pad>=b[2] and a[3]+pad>=b[3]


def page_evidence(page):
    words=page.get_text('words')
    lines={}
    for w in words:
        key=(w[5],w[6]);line=lines.setdefault(key,dict(bbox=[w[0],w[1],w[2],w[3]],words=[]))
        box=line['bbox'];box[:]=[min(box[0],w[0]),min(box[1],w[1]),max(box[2],w[2]),max(box[3],w[3])]
        line['words'].append(str(w[4]))
    return dict(page_bounds=list(page.rect),rotation=page.rotation,native_word_count=len(words),
        words=[dict(bbox=list(w[:4]),text=str(w[4])) for w in words],
        captions=[dict(bbox=r['bbox'],text=''.join(r['words'])) for r in lines.values() if CAPTION.search(''.join(r['words']))],
        images=[dict(bbox=list(r['bbox']),number=r.get('number')) for r in page.get_image_info()],
        drawings=[dict(bbox=list(r['rect']),seqno=r.get('seqno')) for r in page.get_drawings()],
        evidence_kind='ACTUAL_NATIVE_PAGE_OBJECTS_NOT_LAYOUT_REGION_IDENTITIES')


def assess(box,evidence):
    if not valid_box(box):return dict(status='UNKNOWN',reasons=['INVALID_OR_MISSING_SOURCE_BOX'],authorized_to_exclude=False)
    if not evidence or not valid_box(evidence.get('page_bounds')):
        return dict(status='UNKNOWN',reasons=['MISSING_SOURCE_PAGE'],authorized_to_exclude=False)
    if evidence.get('rotation',0)!=0:
        return dict(status='UNKNOWN',reasons=['ROTATED_SOURCE_REQUIRES_COORDINATE_REVIEW'],authorized_to_exclude=False)
    if not contains(evidence['page_bounds'],box,pad=1):
        return dict(status='UNKNOWN',reasons=['SOURCE_BOX_OUTSIDE_PAGE'],authorized_to_exclude=False)
    reasons=[];blockers=[]
    if not evidence.get('native_word_count'):
        reasons.append('NATIVE_TEXT_UNAVAILABLE_OR_SCAN')
    for word in evidence.get('words',[]):
        if valid_box(word['bbox']) and intersects(box,word['bbox']):
            reasons.append('SOURCE_TEXT_OVERLAP');blockers.append(dict(type='TEXT',bbox=word['bbox'],text=word['text']))
    for caption in evidence.get('captions',[]):
        if valid_box(caption['bbox']) and intersects(box,caption['bbox'],pad=24):
            reasons.append('NEAR_FIGURE_OR_QUESTION_CAPTION');blockers.append(dict(type='CAPTION',**caption))
    for picture in evidence.get('images',[]):
        other=picture['bbox']
        if not valid_box(other) or not intersects(box,other):continue
        # Equal geometry is not identity: a masked near-white residual may have
        # exactly the same bbox as the meaningful source photograph. Preserve
        # it until ownership/content coverage is actually established.
        same_box = contains(box,other,pad=1) and contains(other,box,pad=1)
        reasons.append('NATIVE_IMAGE_EQUAL_BOX_UNVERIFIED_OWNERSHIP' if same_box else 'OTHER_NATIVE_IMAGE_OVERLAP')
        blockers.append(dict(type='IMAGE',**picture))
    for drawing in evidence.get('drawings',[]):
        other=drawing['bbox']
        if len(other)==4 and all(math.isfinite(v) for v in other) and intersects(box,other,pad=.75) and not contains(box,other,pad=1):
            reasons.append('VECTOR_CONTINUES_OUTSIDE_CANDIDATE');blockers.append(dict(type='VECTOR',**drawing))
    return dict(status='PROTECTED' if reasons else 'NO_KNOWN_NATIVE_CONFLICT',reasons=sorted(set(reasons)),
        blockers=blockers,authorized_to_exclude=False,coverage_status='NOT_VERIFIED',
        limitation='Absence of these vetoes does not prove decoration; source boxes are not region identity.')
