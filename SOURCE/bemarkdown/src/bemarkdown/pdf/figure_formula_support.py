"""Conservative token support for FormulaNet's auxiliary figure candidates.

Text OCR does not establish two-dimensional mathematical structure. Matching
tokens are only a prerequisite for promoting a candidate, never source accuracy
evidence. Disagreements and unsupported TeX stay pending with raw output retained.
"""
from __future__ import annotations

import re
import unicodedata

from .inline_math_repair import GREEK


_FORMATTING = frozenset({
    'mathrm', 'mathbf', 'mathit', 'mathsf', 'mathtt', 'text', 'textrm',
    'left', 'right', 'vec', 'overrightarrow', 'overleftarrow', 'overline',
    'underline', 'displaystyle', 'textstyle', 'scriptstyle',
    'frac', 'dfrac', 'tfrac',
})
_SYMBOLS = {
    **GREEK, 'nabla': '∇', 'partial': '∂', 'times': '×', 'cdot': '·',
    'div': '÷', 'pm': '±', 'mp': '∓', 'le': '≤', 'leq': '≤',
    'ge': '≥', 'geq': '≥', 'ne': '≠', 'neq': '≠', 'approx': '≈',
    'infty': '∞', 'sqrt': '√', 'sum': '∑', 'int': '∫',
    'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'log': 'log', 'ln': 'ln',
}


def _tokens(value: str) -> str | None:
    text = unicodedata.normalize('NFKC', value).replace('−', '-').replace('µ', 'μ')
    unsupported = False

    def command(match):
        nonlocal unsupported
        name = match.group(1)
        if name in _FORMATTING:
            return ''
        if name in _SYMBOLS:
            return _SYMBOLS[name]
        unsupported = True
        return ''

    text = re.sub(r'\\([A-Za-z]+)', command, text)
    if unsupported:
        return None
    # Plain OCR commonly flattens scripts and fractions. Keep the exact order,
    # case, multiplicity, decimal points and visible operators of its tokens.
    # This cannot verify fraction/script structure or vector scope.
    return ''.join(c for c in text if c.isalnum() or c in '.+-=<>×·÷±∓≤≥≠≈∞√∑∫∇∂')


def figure_formula_content_support(latex: str, evidence: dict) -> dict:
    candidate = _tokens(latex)
    observed = {}
    for provider, row in evidence.items():
        if row and row.get('output_contract_status', 'PASS') == 'PASS':
            value = str(row.get('normalized_text') or '')
            if value.strip():
                observed[provider] = _tokens(value)
    matched = [provider for provider, tokens in observed.items()
               if candidate and tokens == candidate]
    return {
        'version': 'figure-formula-text-token-support-v1',
        'supported': bool(matched),
        'candidate_tokens': candidate, 'provider_tokens': observed,
        'matching_providers': matched,
        'scope': 'Token support only; does not verify mathematical structure or source accuracy.',
    }
