"""Locate literal CJK changes needing a focused source check.

This is a review signal, never a claim that the prior OCR reading is correct.
Whitespace/TeX presentation is excluded from this lexical check; mathematical
and numerical safeguards remain the responsibility of their existing checks.
"""
import difflib
import hashlib
import re

VERSION='source-cjk-lexical-target-review-v1'
HAN=re.compile(r'[\u3400-\u9fff\U00020000-\U0003134f]')

def source_lexical_changes(previous,proposed):
    if not previous:return []
    left=[(m.start(),m.group()) for m in HAN.finditer(previous)]
    right=[(m.start(),m.group()) for m in HAN.finditer(proposed)]
    a=''.join(c for _,c in left);b=''.join(c for _,c in right)
    changes=[]
    def span(chars,start,end,text):
        if start==end:
            p=chars[start][0] if start<len(chars) else len(text)
            return [p,p]
        return [chars[start][0],chars[end-1][0]+1]
    for op,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if op=='equal':continue
        changes.append({'operation':op,'old_han':a[i:j],'new_han':b[k:l],
            'previous_character_range':span(left,i,j,previous),
            'proposed_character_range':span(right,k,l,proposed),
            'left_han_anchor':a[max(0,i-6):i],'right_han_anchor':a[j:j+6],
            'previous_sha256':hashlib.sha256(previous.encode('utf-8')).hexdigest(),
            'proposed_sha256':hashlib.sha256(proposed.encode('utf-8')).hexdigest(),
            'guard_version':VERSION,'previous_reading_is_truth':False})
    return changes
