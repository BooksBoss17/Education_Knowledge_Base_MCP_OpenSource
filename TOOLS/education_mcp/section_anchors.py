"""Transport chapter boundary positions, without claiming edited content is unchanged."""
import bisect
import difflib
import hashlib
import json
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def source_anchors(package, current):
    package=Path(package)
    report_path=package/'conversion_report.json'
    unavailable=dict(status='UNAVAILABLE',reason='No persisted renderer spans',nodes={},pages=[])
    if not report_path.is_file():
        return unavailable
    rendering=json.loads(report_path.read_text('utf-8-sig')).get('markdown_render')
    if not isinstance(rendering,dict) or not rendering.get('node_spans'):
        return unavailable
    original=(package/'document.md').read_text('utf-8')
    raw=original.encode('utf-8')
    if rendering.get('offset_unit')!='UTF8_BYTES' or rendering.get('determinism_sha256')!=sha(original):
        raise ValueError('Section renderer identity mismatch')
    offsets=[0]
    for char in original:
        offsets.append(offsets[-1]+len(char.encode('utf-8')))
    points={};first={};previous_end=0
    for span in rendering['node_spans']:
        node=span['node_id'];left,right=span['byte_start'],span['byte_end'];page=span['page_index']
        if node in points or type(page) is not int or page<0 or not 0<=previous_end<=left<right<=len(raw):
            raise ValueError('Invalid section renderer span')
        if hashlib.sha256(raw[left:right]).hexdigest()!=span['sha256']:
            raise ValueError('Section source bytes changed')
        char=bisect.bisect_left(offsets,left)
        if offsets[char]!=left:
            raise ValueError('Section source boundary splits a Unicode character')
        points[node]=dict(position=char,source_node_id=node,source_page_index=page,kind=span.get('kind'),basis='ORIGINAL_RENDERER_BOUNDARY')
        first.setdefault(page,node);previous_end=right
    journal_path=package/'agent_review.json';text=original
    if journal_path.is_file():
        journal=json.loads(journal_path.read_text('utf-8-sig'))
        if journal.get('original_sha256')!=sha(original):
            raise ValueError('Section review original identity mismatch')
        for revision in journal.get('revisions',[]):
            if revision.get('base_sha256')!=sha(text):
                raise ValueError('Section review chain mismatch')
            for edit in revision.get('replacements',[]):
                old,new=edit['old'],edit['new']
                if not old or text.count(old)!=1:
                    raise ValueError('Section review replacement is not unique')
                left=text.index(old);right=left+len(old);delta=len(new)-len(old);matches=None
                for point in points.values():
                    pos=point['position']
                    if pos is None or pos<left:
                        continue
                    if pos>=right:
                        point['position']+=delta
                    elif pos==left:
                        point['position']=left if new else None
                        point['basis']='EXACT_REPLACEMENT_START_BOUNDARY'
                    else:
                        if matches is None:
                            matches=difflib.SequenceMatcher(None,old,new,autojunk=False).get_matching_blocks()
                        relative=pos-left
                        block=next((m for m in matches if m.size>=8 and m.a<=relative<m.a+m.size
                                    and old.count(old[m.a:m.a+m.size])==1 and new.count(old[m.a:m.a+m.size])==1),None)
                        point['position']=left+block.b+relative-block.a if block else None
                        point['basis']='UNIQUE_UNCHANGED_UNICODE_BOUNDARY_CONTEXT' if block else 'EDITED_BOUNDARY_UNAVAILABLE'
                text=text[:left]+new+text[right:]
            if revision.get('result_sha256')!=sha(text):
                raise ValueError('Section review result identity mismatch')
    if text!=current:
        raise ValueError('Section current Markdown replay mismatch')
    lines=[0]+[i+1 for i,c in enumerate(current) if c=='\n'];nodes={}
    for node,point in points.items():
        pos=point['position']
        if pos is None:
            continue
        nodes[node]={k:v for k,v in point.items() if k!='position'}
        nodes[node].update(current_start_line=bisect.bisect_right(lines,pos),at_line_start=pos==0 or current[pos-1:pos]=='\n',status='VERIFIED')
    pages=[nodes.get(node,dict(source_node_id=node,source_page_index=page,status='EDITED_ANCHOR_UNAVAILABLE')) for page,node in sorted(first.items())]
    return dict(status='VERIFIED',nodes=nodes,pages=pages)
