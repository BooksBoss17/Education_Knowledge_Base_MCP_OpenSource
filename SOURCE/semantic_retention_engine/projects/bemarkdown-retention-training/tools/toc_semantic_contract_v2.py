"""Check requested role against already-read content; never use labels or sample identity."""

import copy

import re

import unicodedata

from probe_zero99_toc_semantic import SCHEMA, KEEP, DROP, validate_result

DROP_V2=DROP|{'DECORATIVE_BACKGROUND'}

def normalize_result(value, use):
    checked=copy.deepcopy(value)
    if isinstance(checked,dict) and checked.get('basis')=='DECORATIVE_BACKGROUND':checked['basis']='DECORATIVE_SCENE'
    if not validate_result(checked,len(use['source_lines'])):
        return None,'INVALID_OUTPUT_CONTRACT'
    result=copy.deepcopy(value)
    text=unicodedata.normalize('NFKC',result['visible_text']).strip()
    if result['verdict']=='D' and re.sub(r'\s+','',text).casefold() in {'contents','目录'}:
        result.update(verdict='K',basis='FULL_TEXT_TITLE')
        return result,'COMPLETE_TITLE_PROTECTED'
    # An observed index column or entry with dot leaders is positive content evidence,
    # regardless of which incompatible deletion role the language decoder selected.
    numeric_lines=[s.strip() for s in text.splitlines() if re.fullmatch(r'\s*\d{1,3}\s*',s)]
    entry_text=bool(re.search(r'[·.…]{3,}',text) and re.search(r'[\u3400-\u9fffA-Za-z0-9]',text))
    if result['verdict']=='D' and (len(numeric_lines)>=2 or entry_text):
        result.update(verdict='K',basis='TOC_ENTRIES_OR_INDEX')
        return result,'OBSERVED_CONTENT_IS_TEXT_OR_MULTIPLE_INDEX_VALUES_NOT_A_SINGLE_FOOTER'
    if result['verdict']=='D' and result['basis']=='INDEPENDENT_FOOTER_MARK':
        token=re.sub(r'[\s()（）\[\]【】\-—–]','',text)
        singleton=bool(re.fullmatch(r'\d{1,3}|[IVXLCDMivxlcdm]{1,6}',token))
        if not singleton:
            if re.search(r'[·.…]{3,}|[\u3400-\u9fff]{2,}|[A-Za-z]{3,}',text) or len(re.findall(r'\d+',text))>=2:
                result.update(verdict='K',basis='TOC_ENTRIES_OR_INDEX')
                return result,'OBSERVED_CONTENT_IS_TEXT_OR_MULTIPLE_INDEX_VALUES_NOT_A_SINGLE_FOOTER'
            return None,'FOOTER_WITHOUT_SINGLE_READABLE_PAGE_TOKEN'
        if use['target_bbox_pdf_pt'][1] < .9*use['source_page_size_pdf_pt'][1]:
            return None,'FOOTER_POSITION_NOT_ESTABLISHED'
    return result,'UNCHANGED_VALID_RESULT'

def repair_schema(line_count):
    variants=[]
    for verdict,roles in [('K',sorted(KEEP)),('D',sorted(DROP_V2)),('U',['UNCERTAIN'])]:
        schema=copy.deepcopy(SCHEMA)
        schema['properties']['verdict']={'const':verdict}
        schema['properties']['basis']={'type':'string','enum':roles}
        schema['properties']['evidence_line_ids']['items']['maximum']=line_count-1
        if verdict=='D':
            schema['properties']['source_is_toc']={'const':True}
            schema['properties']['evidence_line_ids']['minItems']=1
        variants.append(schema)
    return {'oneOf':variants}
