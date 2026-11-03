"""Observable prose triage; signals request checks, never dictate replacements."""
from __future__ import annotations
import re

SEMANTIC_TRIAGE_VERSION='bounded-prose-triage-v1'


def prose_risk_reasons(text,*,pp_confidence=None,lexical_disagreement=False):
    if not isinstance(text,str) or len(text)>1600 or len(re.findall(r'[\u4e00-\u9fff]',text))<12:
        return ()
    if text.lstrip().startswith(('#','![','<','|','```')) or '\n\n' in text:
        return ()
    reasons=[]
    # Finance-style glyphs embedded inside normal Han prose are a weak OCR
    # confusion signal, not a hardcoded vocabulary correction or sample ID.
    if re.search(r'[\u4e00-\u9fff][壹贰叁肆伍陆柒捌玖][\u4e00-\u9fff]',text):
        reasons.append('UNUSUAL_NUMERAL_GLYPH_IN_PROSE')
    if re.search(r'([\u4e00-\u9fff]{2,6})\1',text):
        reasons.append('POSSIBLE_DUPLICATED_PROSE_SPAN')
    if lexical_disagreement:reasons.append('PROSE_CANDIDATE_DISAGREEMENT')
    if pp_confidence is not None and pp_confidence<.90:
        reasons.append('LOW_LINE_EVIDENCE_PROSE')
    return tuple(reasons)


def prose_regions(content):
    """Offsets into the exact current Markdown, excluding nonprose assets."""
    result=[]
    for match in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)',content):
        text=match.group().rstrip()
        if text.startswith(('#','![','<','|','```','$$','---')) or '![' in text or '<img' in text.lower():continue
        result.append({'start':match.start(),'end':match.start()+len(text),'text':text})
    return result
