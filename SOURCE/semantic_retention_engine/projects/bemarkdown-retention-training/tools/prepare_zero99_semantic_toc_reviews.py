"""Uniform source-based TOC review inventory; preserve Gold and reuse completed observations."""

import re

from zero99_toc_evidence import toc_evidence

def candidate(row):
    strict=toc_evidence(row)
    if strict:return {'detector':'existing_toc_evidence','uses':strict}
    records=[]
    for use in row['uses']:
        if use['context_status']!='verified' or use['source_window_truncated']:return None
        text='\n'.join(s['text'] for s in use['spans'])
        chapters=set(re.findall(r'第\s*([一二三四五六七八九十百0-9]+)\s*章',text))
        lines=[x.strip() for x in text.splitlines() if x.strip()]
        dotted=[x for x in lines if re.search(r'[·.…]{3,}\s*\d{1,3}\s*$',x)]
        split=[lines[i-1]+' '+x for i,x in enumerate(lines) if i and re.fullmatch(r'\d{1,3}',x)
               and len(re.findall(r'[\u4e00-\u9fff]',lines[i-1]))>=2 and not lines[i-1].startswith('[')]
        header=bool(re.search(r'目\s*录|CONTENTS',text,re.I))
        if not ((header and len(chapters)>=1 and len(dotted)+len(split)>=4) or
                (len(chapters)>=2 and len(dotted)+len(split)>=6)):return None
        records.append({'use_id':use['use_id'],'chapter_count':len(chapters),'page_entry_count':len(dotted)+len(split),
                        'explicit_contents_heading':header,'literal_unresolved_source_markers':text.count('[Unresolved source content]')})
    return {'detector':'source_toc_review_eligibility_not_deletion_rule','uses':records} if records else None
