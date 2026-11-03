"""Opt-in structure-preserving review policy; never changes selected mathematics."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .adaptive_ocr import RoutingPolicy, canonical


@dataclass(frozen=True)
class BalancedRoutingPolicy(RoutingPolicy):
    version: str = 'bemarkdown-balanced-ocr-b1-v1'

    def select(self, pp, ovis, xiaomi):
        result = super().select(pp, ovis, xiaomi)
        reasons = list(result['review_reasons'])
        pp_text = str(pp.get('text', ''))
        selected = str(result['selected_text'])
        if canonical(pp_text) != canonical(selected):
            reasons.append('STRUCTURE_PRESERVING_DISAGREEMENT_REVIEW')
        if not pp_text.strip() and selected.strip():
            reasons.append('NO_LINE_EVIDENCE_REVIEW')
        if re.search(r'<\s*img\b|!\[[^\]]*\]\s*\(', selected, re.I):
            reasons.append('VISIBLE_TEXT_COVERAGE_REVIEW')
        result['review_reasons'] = list(dict.fromkeys(reasons))
        result['review_required'] = result['review_required'] or any(r.endswith('REVIEW') for r in reasons)
        result['policy_fingerprint'] = self.fingerprint()
        return result
