"""Bound FormulaNet disambiguation to single glyphs already seen by text OCR."""
import re

SINGLE_SYMBOL_GATE_VERSION = 'source-single-symbol-family-v2'

# Exact Unicode/TeX spellings of one glyph, not substitutions between visually
# confusable letters. Preserve Greek case and variant forms as separate symbols.
_GREEK_GLYPHS = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'ε': 'varepsilon', 'ϵ': 'epsilon', 'ζ': 'zeta', 'η': 'eta',
    'θ': 'theta', 'ϑ': 'vartheta', 'ι': 'iota', 'κ': 'kappa', 'ϰ': 'varkappa',
    'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'π': 'pi', 'ϖ': 'varpi',
    'ρ': 'rho', 'ϱ': 'varrho', 'σ': 'sigma', 'ς': 'varsigma', 'τ': 'tau',
    'υ': 'upsilon', 'φ': 'varphi', 'ϕ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
    'Γ': 'Gamma', 'Δ': 'Delta', 'Θ': 'Theta', 'Λ': 'Lambda', 'Ξ': 'Xi',
    'Π': 'Pi', 'Σ': 'Sigma', 'Υ': 'Upsilon', 'Φ': 'Phi', 'Ψ': 'Psi', 'Ω': 'Omega',
}
_GREEK_SPELLINGS = {**_GREEK_GLYPHS,
                    **{'\\' + name: name for name in _GREEK_GLYPHS.values()}}


def symbol_family(value):
    value = str(value).strip()
    if value in _GREEK_SPELLINGS:
        return 'greek:' + _GREEK_SPELLINGS[value]
    if not re.fullmatch('[A-Za-z0-9]', value):
        return None
    return 'o' if value in '0Oo' else value.casefold()


def needs_symbol_verification(text_a, confidence_a, text_b=''):
    family = symbol_family(text_a)
    if family is None:
        return False
    if family == 'o':
        return True
    if confidence_a is not None and float(confidence_a) < .7:
        return True
    if family.startswith('greek:'):
        return symbol_family(text_b) != family
    return str(text_a).strip() != str(text_b).strip() and symbol_family(text_b) == family


def accepts_symbol_candidate(text_a, candidate):
    value = str(candidate).strip()
    wrapper = re.fullmatch(r'\\(?:mathrm|mathit|text)\{([^{}]+)\}', value)
    if wrapper:
        value = wrapper.group(1)
    family = symbol_family(text_a)
    return family is not None and symbol_family(value) == family


def select_figure_math_requests(requests, evidence_b):
    selected = []
    for request in requests:
        if request.get('kind') != 'symbol':
            selected.append(request)
            continue
        b = evidence_b.get(request['id'])
        valid_b = b is not None and b.source_crop_sha256 == request['crop_sha256']
        if needs_symbol_verification(request['text_a'], request.get('confidence_a'),
                                     b.normalized_text if valid_b else ''):
            selected.append(request)
    return selected
