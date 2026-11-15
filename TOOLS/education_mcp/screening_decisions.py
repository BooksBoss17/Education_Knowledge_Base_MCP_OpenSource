"""Reusable source-bound image review decisions for automatic plan construction.

Approval is tied to the source file, page/part, asset bytes and policy. It cannot
spread to a similar image, another page, another book or changed source content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

POLICY='source-bound-decoration-review-v1'


def key(source_sha256,source_part,asset_sha256):
    return hashlib.sha256(json.dumps([POLICY,source_sha256,source_part,asset_sha256],ensure_ascii=False).encode()).hexdigest()


def load(path):
    path=Path(path)
    if not path.exists():return dict(schema=POLICY,decisions={})
    data=json.loads(path.read_text('utf-8'))
    if data.get('schema')!=POLICY or not isinstance(data.get('decisions'),dict):
        raise ValueError('Invalid screening decision ledger')
    for identity,row in data['decisions'].items():
        if key(row['source_sha256'],row['source_part'],row['asset_sha256'])!=identity:
            raise ValueError('Screening decision identity mismatch')
    return data


def record(ledger,source_sha256,references,decisions):
    """Validate caller's explicit source review; returned ledger is immutable copy."""
    if not isinstance(decisions,list) or not 1<=len(decisions)<=100:
        raise ValueError('Expected 1 to 100 image review decisions')
    by_id={r['image_id']:r for r in references}
    next_ledger=json.loads(json.dumps(ledger));seen=set();identity_verdicts={}
    for decision in decisions:
        if set(decision)!={'image_id','decision','source_context_checked','no_meaningful_content',
                           'no_unresolved_content','source_evidence'}:
            raise ValueError('Review requires explicit content protection checks')
        image_id=decision['image_id']
        if image_id not in by_id or image_id in seen:raise ValueError('Unknown or duplicate reviewed image')
        seen.add(image_id);ref=by_id[image_id];part=ref.get('source_part')
        if not isinstance(part,str) or not part:raise ValueError('Cannot reuse review without source-part identity')
        if not re.fullmatch('[0-9a-f]{64}',source_sha256) or not re.fullmatch('[0-9a-f]{64}',ref['sha256']):
            raise ValueError('Invalid source or asset SHA')
        verdict=decision['decision']
        if verdict not in {'KEEP','EXCLUDE_DECORATION','NEEDS_RECOGNITION','REVIEW_UNKNOWN'}:
            raise ValueError('Unknown screening decision')
        if not isinstance(decision['source_evidence'],str) or not decision['source_evidence'].strip():
            raise ValueError('Source evidence is required')
        if any(type(decision[k]) is not bool for k in ['source_context_checked','no_meaningful_content','no_unresolved_content']):
            raise ValueError('Review protection checks must be booleans')
        if verdict=='EXCLUDE_DECORATION' and not all(decision[k] for k in ['source_context_checked','no_meaningful_content','no_unresolved_content']):
            raise ValueError('Cannot exclude unresolved or meaningful content')
        if verdict=='EXCLUDE_DECORATION' and ref.get('screening_protection_reasons'):
            raise ValueError('Conversion still marks this asset as protected or unresolved')
        identity=key(source_sha256,part,ref['sha256'])
        if identity in identity_verdicts and identity_verdicts[identity]!=verdict:
            raise ValueError('Conflicting reviews for the same source asset')
        identity_verdicts[identity]=verdict
        previous=next_ledger['decisions'].get(identity)
        next_ledger.setdefault('history',[]).append(dict(identity=identity,previous=previous,new_decision=verdict,
            source_evidence=decision['source_evidence']))
        next_ledger['decisions'][identity]=dict(source_sha256=source_sha256,source_part=part,asset_sha256=ref['sha256'],
            decision=verdict,source_evidence=decision['source_evidence'],
            source_context_checked=decision['source_context_checked'],no_meaningful_content=decision['no_meaningful_content'],
            no_unresolved_content=decision['no_unresolved_content'])
    return next_ledger


def actions(ledger,source_sha256,references):
    result=[]
    for ref in references:
        part=ref.get('source_part')
        if not part:continue
        if ref.get('screening_protection_reasons'):continue
        identity=key(source_sha256,part,ref['sha256']);row=ledger['decisions'].get(identity)
        if not row or row['decision']!='EXCLUDE_DECORATION':continue
        if not all(row.get(k) is True for k in ['source_context_checked','no_meaningful_content','no_unresolved_content']):
            raise ValueError('Invalid persisted exclusion approval')
        result.append(dict(image_id=ref['image_id'],action='remove_decoration',text='',
            source_evidence=f"Verified screening decision {identity}: {row['source_evidence']}"))
    return result
