"""Structural TOC evidence, independent of book/asset identity and Gold."""

import re

def toc_evidence(row):
    coverage=row['coverage']
    if not coverage['enumeration_complete'] or coverage['expected_use_count']!=len(row['uses']):return None
    if set(coverage['covered_use_ids'])!={u['use_id'] for u in row['uses']}:return None
    receipts=[]
    for use in row['uses']:
        if use['context_status']!='verified' or use['source_window_truncated'] or use['selector_version']!='bound-md-full-page-v1':return None
        md='\n'.join(s['text'] for s in use['spans'])
        if md.count('[TARGET_IMAGE]')!=1:return None
        chapters=set(re.findall(r'第\s*([一二三四五六七八九十百0-9]+)\s*章',md))
        lines=[line.strip() for line in md.splitlines() if line.strip()]
        entries=[line for line in lines if re.search(r'(?:[·.…]{3,}\s*|[ \t]+)\d{1,3}\s*$',line)]
        # Some MD exporters place the page number on its own line. Bind it to
        # an immediately preceding readable heading, never an image marker.
        entries += [lines[i-1]+' '+line for i,line in enumerate(lines) if i>0 and re.fullmatch(r'\d{1,3}',line)
                    and len(re.findall(r'[\u4e00-\u9fff]',lines[i-1]))>=2 and not lines[i-1].startswith('[')]
        if len(chapters)<3 or len(entries)<8:return None
        # Structural position alone is not a deletion reason. This only routes a
        # first-pass D real-photo candidate to a separate no-readable-text check.
        receipts.append({'use_id':use['use_id'],'chapter_count':len(chapters),'page_entry_count':len(entries),'page_entries':entries})
    return receipts or None

def no_text_photo(observation):
    return isinstance(observation,dict) and observation.get('kind')=='PHOTO' and isinstance(observation.get('visible_text'),str) and observation['visible_text'].strip()=='' and bool(observation.get('visual_information'))
