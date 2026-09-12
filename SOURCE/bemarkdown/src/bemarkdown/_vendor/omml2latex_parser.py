# Vendored from omml2latex 0.1.1 (Apache-2.0).
# Upstream wheel SHA-256: b24a13f4a14791662972d1b6716e5a1f9114891ccf1a96622f61dbfb1c300bd4
# BeMarkdown modification: split the Python 3.11-invalid f-string expression
# containing a backslash into an equivalent local variable assignment.

"""omml2latex.py — OMML (Office Math Markup Language) → LaTeX 변환기.

출력은 KaTeX-compatible LaTeX math syntax이다.

Public API
----------
convert_omml(node: ET.Element) -> str
    m:oMath 또는 m:oMathPara 엘리먼트를 LaTeX 수식 문자열로 변환한다.
"""

import xml.etree.ElementTree as ET

__all__ = ["convert_omml"]

# ==============================================================================
# Helper / Utility
# ==============================================================================
def _get_tag(node):
    return node.tag.split('}')[-1] if node is not None else ""

def _get_child(node, tag):
    if node is None: return None
    for child in node:
        if _get_tag(child) == tag: return child
    return None

def _get_children(node, tag):
    if node is None: return []
    return [child for child in node if _get_tag(child) == tag]

def _get_val(node, attr='val'):
    if node is None: return None
    # 속성명에 네임스페이스가 붙어있을 경우를 대비해 순회하며 속성명 검사
    # transitional과 strict 스키마 모두에서 val 속성은 동일한 이름이지만, 네임스페이스가 다르므로 (ECMA-376 참고)
    for k, v in node.attrib.items():
        if k == attr or k.endswith(f"}}{attr}"):
            return v
    return None

# ==============================================================================
# Unicode Mapping & Cleanup Table
# ==============================================================================

# ── Unicode Mathematical Alphabet → LaTeX (built once at import time) ─────────
# OMML text nodes sometimes contain Unicode Mathematical Alphabet characters
# (U+1D400–1D7FF) directly instead of using m:rPr style attributes.
# Example: 𝑓 (U+1D453, Math Italic Small F) → f  (LaTeX math mode is italic by default)
#          𝜋 (U+1D70B, Math Italic Pi)       → \pi
#          𝒜 (U+1D49C, Math Script Capital A) → \mathcal{A}
def _build_math_alpha_map() -> dict:
    import unicodedata

    _GREEK_SMALL = {
        'ALPHA': r'\alpha', 'BETA': r'\beta', 'GAMMA': r'\gamma', 'DELTA': r'\delta',
        'EPSILON': r'\varepsilon', 'ZETA': r'\zeta', 'ETA': r'\eta', 'THETA': r'\theta',
        'IOTA': r'\iota', 'KAPPA': r'\kappa', 'LAMDA': r'\lambda', 'MU': r'\mu',
        'NU': r'\nu', 'XI': r'\xi', 'OMICRON': r'\omicron', 'PI': r'\pi', 'RHO': r'\rho',
        'SIGMA': r'\sigma', 'TAU': r'\tau', 'UPSILON': r'\upsilon', 'PHI': r'\varphi',
        'CHI': r'\chi', 'PSI': r'\psi', 'OMEGA': r'\omega',
        # Variant / symbol forms
        'FINAL SIGMA': r'\varsigma',
        'EPSILON SYMBOL': r'\epsilon', 'THETA SYMBOL': r'\vartheta',
        'KAPPA SYMBOL': r'\varkappa', 'PHI SYMBOL': r'\phi', 'RHO SYMBOL': r'\varrho',
        'PI SYMBOL': r'\varpi', 'DIGAMMA': r'\digamma',
        'PARTIAL DIFFERENTIAL': r'\partial',
    }
    _GREEK_CAP = {
        'ALPHA': r'\Alpha', 'BETA': r'\Beta', 'GAMMA': r'\Gamma', 'DELTA': r'\Delta',
        'EPSILON': r'\Epsilon', 'ZETA': r'\Zeta', 'ETA': r'\Eta', 'THETA': r'\Theta',
        'IOTA': r'\Iota', 'KAPPA': r'\Kappa', 'LAMDA': r'\Lambda', 'MU': r'\Mu',
        'NU': r'\Nu', 'XI': r'\Xi', 'OMICRON': r'\Omicron', 'PI': r'\Pi',
        'RHO': r'\Rho', 'SIGMA': r'\Sigma', 'TAU': r'\Tau', 'UPSILON': r'\Upsilon',
        'PHI': r'\Phi', 'CHI': r'\Chi', 'PSI': r'\Psi', 'OMEGA': r'\Omega',
        'NABLA': r'\nabla',
    }

    mapping: dict = {}

    # Standalone math-style letters from the Letterlike Symbols block (U+2100–214F)
    _STANDALONE = {
        '\u210E': 'h',             # ℎ Math Italic h (Planck)
        '\u210F': r'\hbar',        # ℏ
        '\u2113': r'\ell',         # ℓ script l
        '\u212C': r'\mathcal{B}',  # ℬ
        '\u2130': r'\mathcal{E}',  # ℰ
        '\u2131': r'\mathcal{F}',  # ℱ
        '\u210B': r'\mathcal{H}',  # ℋ
        '\u2110': r'\mathcal{I}',  # ℐ
        '\u2112': r'\mathcal{L}',  # ℒ
        '\u2133': r'\mathcal{M}',  # ℳ
        '\u211B': r'\mathcal{R}',  # ℛ
        '\u212D': r'\mathfrak{C}', # ℭ
        '\u210C': r'\mathfrak{H}', # ℌ
        '\u2111': r'\mathfrak{I}', # ℑ
        '\u211C': r'\mathfrak{R}', # ℜ
        '\u2128': r'\mathfrak{Z}', # ℨ
        '\u2102': r'\mathbb{C}',   # ℂ
        '\u210D': r'\mathbb{H}',   # ℍ
        '\u2115': r'\mathbb{N}',   # ℕ
        '\u2119': r'\mathbb{P}',   # ℙ
        '\u211A': r'\mathbb{Q}',   # ℚ
        '\u211D': r'\mathbb{R}',   # ℝ
        '\u2124': r'\mathbb{Z}',   # ℤ
        # Additional letterlike symbols
        '\u2118': r'\wp',          # ℘ Weierstrass p
        '\u2127': r'\mho',         # ℧ mho (inverted omega)
        '\u2135': r'\aleph',       # ℵ aleph
        '\u2136': r'\beth',        # ℶ beth
        '\u2137': r'\gimel',       # ℷ gimel
        '\u2138': r'\daleth',      # ℸ daleth
        '\u210A': 'g',             # ℊ script small g
        '\u212F': 'e',             # ℯ script small e
        '\u2134': 'o',             # ℴ script small o
    }
    mapping.update(_STANDALONE)

    for cp in range(0x1D400, 0x1D800):
        try:
            ch = chr(cp)
            name = unicodedata.name(ch, '')
        except Exception:
            continue
        if not name.startswith('MATHEMATICAL'):
            continue

        # Determine font style (most specific match first)
        style = 'normal'
        for s in ('BOLD ITALIC', 'SANS-SERIF BOLD ITALIC', 'SANS-SERIF ITALIC',
                  'SANS-SERIF BOLD', 'BOLD', 'ITALIC', 'SCRIPT', 'FRAKTUR',
                  'DOUBLE-STRUCK', 'SANS-SERIF', 'MONOSPACE'):
            if s in name:
                style = s.lower().replace(' ', '-')
                break

        if 'CAPITAL' not in name and 'SMALL' not in name:
            # Symbol variants / special constants without CAPITAL/SMALL qualifier
            # e.g. "MATHEMATICAL ITALIC EPSILON SYMBOL", "MATHEMATICAL ITALIC PARTIAL DIFFERENTIAL"
            _STYLE_WORDS = {'MATHEMATICAL', 'BOLD', 'ITALIC', 'SCRIPT', 'FRAKTUR',
                            'DOUBLE-STRUCK', 'SANS-SERIF', 'MONOSPACE'}
            remaining = ' '.join(w for w in name.split() if w not in _STYLE_WORDS)
            cmd = _GREEK_SMALL.get(remaining) or _GREEK_CAP.get(remaining)
            if cmd:
                mapping[ch] = cmd
            continue

        if 'CAPITAL' in name or 'SMALL' in name:
            parts = name.split()
            is_small = 'SMALL' in name
            is_cap   = 'CAPITAL' in name
            base = parts[-1]  # last token = letter name

            if len(base) == 1 and base.isalpha():
                # Dotless i/j (𝚤 𝚥) — must come before the general letter path
                if 'DOTLESS' in name:
                    mapping[ch] = r'\imath' if base == 'I' else r'\jmath'
                    continue
                # Latin letter (A–Z or a–z)
                letter = base.lower() if is_small else base.upper()
            else:
                # Greek or symbol letter name (ALPHA, BETA, PI, EPSILON SYMBOL, …)
                # Reconstruct multi-token names like "EPSILON SYMBOL"
                # by taking everything after CAPITAL/SMALL keyword
                try:
                    idx = max(
                        parts.index('CAPITAL') if is_cap else -1,
                        parts.index('SMALL')   if is_small else -1,
                    )
                    greek_name = ' '.join(parts[idx + 1:])
                except (ValueError, IndexError):
                    continue
                cmd = (_GREEK_CAP if is_cap else _GREEK_SMALL).get(greek_name)
                if cmd:
                    mapping[ch] = cmd
                continue  # handled above; skip Latin logic

            if style == 'italic':
                mapping[ch] = letter          # default math mode is italic
            elif style == 'bold':
                mapping[ch] = f'\\mathbf{{{letter}}}'
            elif style in ('bold-italic', 'sans-serif-bold-italic'):
                mapping[ch] = f'\\boldsymbol{{{letter}}}'
            elif style == 'script':
                mapping[ch] = f'\\mathcal{{{letter}}}'
            elif style == 'fraktur':
                mapping[ch] = f'\\mathfrak{{{letter}}}'
            elif style == 'double-struck':
                mapping[ch] = f'\\mathbb{{{letter}}}'
            elif style in ('sans-serif', 'sans-serif-italic'):
                mapping[ch] = f'\\mathsf{{{letter}}}'
            elif style == 'sans-serif-bold':
                mapping[ch] = f'\\mathbf{{\\mathsf{{{letter}}}}}'
            elif style == 'monospace':
                mapping[ch] = f'\\mathtt{{{letter}}}'

    return mapping


_MATH_ALPHA_TO_LATEX: dict = _build_math_alpha_map()

UNICODE_TO_LATEX = {
    # ASCII special characters that need escaping in KaTeX math mode
    # NOTE: & # $ are intentionally NOT escaped here — OMML m:t text nodes use them
    # as raw LaTeX structural characters (alignment separators, macro args, etc.)
    '%': '\\%',   # comment character — genuinely appears as math symbol (percent sign)
    # Greek lowercase — basic forms
    'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\varepsilon',
    'ζ': '\\zeta', 'η': '\\eta', 'θ': '\\theta', 'ι': '\\iota', 'κ': '\\kappa',
    'λ': '\\lambda', 'μ': '\\mu', 'ν': '\\nu', 'ξ': '\\xi',
    'ο': '\\omicron', # U+03BF GREEK SMALL LETTER OMICRON
    'π': '\\pi', 'ρ': '\\rho', 'σ': '\\sigma', 'τ': '\\tau', 'υ': '\\upsilon',
    'φ': '\\varphi', 'χ': '\\chi', 'ψ': '\\psi', 'ω': '\\omega',
    # Greek lowercase — variant / symbol forms (safety net for PPTX using plain Unicode instead of math-alphabet block)
    'ϑ': '\\vartheta',   # U+03D1 GREEK THETA SYMBOL
    'ϕ': '\\phi',     # U+03D5 GREEK PHI SYMBOL
    'ϖ': '\\varpi',      # U+03D6 GREEK PI SYMBOL
    'ϱ': '\\varrho',     # U+03F1 GREEK RHO SYMBOL
    'ϰ': '\\varkappa',   # U+03F0 GREEK KAPPA SYMBOL
    'ϵ': '\\epsilon', # U+03F5 GREEK LUNATE EPSILON SYMBOL
    'ς': '\\varsigma',   # U+03C2 GREEK SMALL LETTER FINAL SIGMA
    # Greek uppercase — all have KaTeX commands (incl. \Alpha, \Beta, etc.)
    'Α': '\\Alpha', 'Β': '\\Beta', 'Γ': '\\Gamma', 'Δ': '\\Delta',
    'Ε': '\\Epsilon', 'Ζ': '\\Zeta', 'Η': '\\Eta', 'Θ': '\\Theta',
    'Ι': '\\Iota', 'Κ': '\\Kappa', 'Λ': '\\Lambda', 'Μ': '\\Mu',
    'Ν': '\\Nu', 'Ξ': '\\Xi', 'Ο': '\\Omicron', 'Π': '\\Pi',
    'Ρ': '\\Rho', 'Σ': '\\Sigma', 'Τ': '\\Tau', 'Υ': '\\Upsilon',
    'Φ': '\\Phi', 'Χ': '\\Chi', 'Ψ': '\\Psi', 'Ω': '\\Omega',
    
    '∞': '\\infty', '≈': '\\approx', '≠': '\\neq', '≤': '\\leq', '≥': '\\geq',
    '×': '\\times', '÷': '\\div', '±': '\\pm', '·': '\\cdot', '°': '{}^{\\circ}',
    '℃': '{}^{\\circ}\\mathrm{C}', '℉': '{}^{\\circ}\\mathrm{F}',
    '∂': '\\partial', '∇': '\\nabla', 
    '∈': '\\in', '∉': '\\notin', '⊂': '\\subset', '⊃': '\\supset', '⊆': '\\subseteq', '⊇': '\\supseteq',
    '∪': '\\cup', '∩': '\\cap',
    '∧': '\\land', '∨': '\\vee', '∀': '\\forall', '∃': '\\exists', '∄': '\\nexists',
    '∅': '\\emptyset', '←': '\\leftarrow', '→': '\\rightarrow', '↔': '\\leftrightarrow',
    '↑': '\\uparrow', '↓': '\\downarrow', '↕': '\\updownarrow',
    '⇒': '\\Rightarrow', '⇐': '\\Leftarrow', '⇔': '\\Leftrightarrow',
    '⇑': '\\Uparrow', '⇓': '\\Downarrow', '⇕': '\\Updownarrow',
    '⟵': '\\longleftarrow', '⟶': '\\longrightarrow', '⟷': '\\longleftrightarrow',
    '⟸': '\\Longleftarrow', '⟹': '\\Longrightarrow', '⟺': '\\Longleftrightarrow',
    '↗': '\\nearrow', '↘': '\\searrow', '↙': '\\swarrow', '↖': '\\nwarrow',
    '↩': '\\hookleftarrow', '↪': '\\hookrightarrow',
    '↼': '\\leftharpoonup', '↽': '\\leftharpoondown',
    '⇀': '\\rightharpoonup', '⇁': '\\rightharpoondown',
    '⇋': '\\leftrightharpoons', '⇌': '\\rightleftharpoons',
    '↦': '\\mapsto',
    '↤': '\\leftarrow',  # U+21A4 — \mapsfrom not in KaTeX; use \leftarrow
    '↶': '\\curvearrowleft', '↷': '\\curvearrowright',
    '↺': '\\circlearrowleft', '↻': '\\circlearrowright',
    '⇇': '\\leftleftarrows', '⇉': '\\rightrightarrows',
    '⇆': '\\leftrightarrows', '⇄': '\\rightleftarrows',
    '⇈': '\\upuparrows', '⇊': '\\downdownarrows',
    '⇍': '\\nLeftarrow', '⇏': '\\nRightarrow', '⇎': '\\nLeftrightarrow',
    '⇚': '\\Lleftarrow', '⇛': '\\Rrightarrow',
    '⊸': '\\multimap',
    
    # N-ary 및 추가 수학 기호
    '∫': '\\int', '∬': '\\iint', '∭': '\\iiint', '∮': '\\oint', '∯': '\\oiint',
    '∑': '\\sum', '∏': '\\prod', '∐': '\\coprod',
    '⋃': '\\bigcup', '⋂': '\\bigcap', '⋁': '\\bigvee', '⋀': '\\bigwedge',
    '⨁': '\\bigoplus', '⨂': '\\bigotimes', '⨀': '\\bigodot', '⨄': '\\biguplus',

    # Grouping 기호 (문자열 내부에 쓰일 때 맵핑. groupChr 구조가 아닌 경우 대비)
    '⏟': '\\underbrace', '⏞': '\\overbrace',
    '⏜': '\\overgroup',   # U+23DC — \overparen not in KaTeX; use \overgroup
    '⏝': '\\undergroup',  # U+23DD — \underparen not in KaTeX; use \undergroup
    '⌢': '\\frown',        # U+2322 ARC — \widehat is wrong; \frown is correct

    # Additional operators and punctuation commonly found in OMML text nodes
    '∙': '\\cdot',          # U+2219 BULLET OPERATOR (used for dot product / scalar mult)
    '−': '-',               # U+2212 MINUS SIGN → ASCII minus (safe in math mode)
    '…': '\\ldots',         # U+2026 HORIZONTAL ELLIPSIS
    '⋯': '\\cdots',         # U+22EF MIDLINE HORIZONTAL ELLIPSIS
    '⋮': '\\vdots',         # U+22EE VERTICAL ELLIPSIS
    '⋱': '\\ddots',         # U+22F1 DOWN RIGHT DIAGONAL ELLIPSIS
    '⋰': '\\ddots',         # U+22F0 UP RIGHT DIAGONAL ELLIPSIS — \iddots NOT in KaTeX (Issue #1223)
    '≔': '\\coloneqq',      # U+2254 COLON EQUALS
    '≕': '\\eqqcolon',      # U+2255 EQUALS COLON
    '≝': '\\stackrel{\\text{def}}{=}',  # U+225D EQUAL TO BY DEFINITION
    '≜': '\\triangleq',     # U+225C DELTA EQUAL TO

    # Common relations and operators missing from initial set
    '∝': '\\propto',        # U+221D PROPORTIONAL TO
    '∼': '\\sim',           # U+223C TILDE OPERATOR
    '≃': '\\simeq',         # U+2243 ASYMPTOTICALLY EQUAL TO
    '≅': '\\cong',          # U+2245 APPROXIMATELY EQUAL TO
    '≡': '\\equiv',         # U+2261 IDENTICAL TO
    '≪': '\\ll',            # U+226A MUCH LESS-THAN
    '≫': '\\gg',            # U+226B MUCH GREATER-THAN
    '∋': '\\ni',            # U+220B CONTAINS AS MEMBER
    '∌': '\\notni',          # U+220C DOES NOT CONTAIN AS MEMBER — \notni is directly supported by KaTeX
    '∁': '\\complement',    # U+2201 COMPLEMENT
    '∎': '\\blacksquare',   # U+220E END OF PROOF
    '∴': '\\therefore',     # U+2234 THEREFORE
    '∵': '\\because',       # U+2235 BECAUSE
    '∗': '\\ast',           # U+2217 ASTERISK OPERATOR
    '∘': '\\circ',          # U+2218 RING OPERATOR
    '∕': '/',               # U+2215 DIVISION SLASH → plain /
    '∤': '\\nmid',          # U+2224 DOES NOT DIVIDE
    '∥': '\\parallel',      # U+2225 PARALLEL TO
    '∦': '\\nparallel',     # U+2226 NOT PARALLEL TO
    '∠': '\\angle',         # U+2220 ANGLE
    '∡': '\\measuredangle', # U+2221 MEASURED ANGLE
    '∢': '\\sphericalangle',# U+2222 SPHERICAL ANGLE
    '∟': '\\angle',          # U+221F RIGHT ANGLE — \rightangle not in KaTeX
    '∶': ':',               # U+2236 RATIO → colon
    '∷': '\\mathbin{::}',  # U+2237 PROPORTION
    '⊢': '\\vdash',         # U+22A2 RIGHT TACK
    '⊣': '\\dashv',         # U+22A3 LEFT TACK
    '⊤': '\\top',           # U+22A4 DOWN TACK
    '⊥': '\\perp',          # U+22A5 UP TACK (perpendicular)
    '⊨': '\\models',        # U+22A8 TRUE
    '△': '\\triangle',      # U+25B3 WHITE UP-POINTING TRIANGLE

    # More set/order operators
    '≺': '\\prec', '≻': '\\succ', '≼': '\\preceq', '≽': '\\succeq',
    '⊲': '\\vartriangleleft', '⊳': '\\vartriangleright',
    '⊴': '\\trianglelefteq', '⊵': '\\trianglerighteq',
    '⊎': '\\uplus', '⊓': '\\sqcap', '⊔': '\\sqcup',
    '⊕': '\\oplus', '⊖': '\\ominus', '⊗': '\\otimes',
    '⊙': '\\odot', '⊘': '\\oslash', '⊸': '\\multimap',
    '⋅': '\\cdot',          # U+22C5 DOT OPERATOR
    '⋆': '\\star',          # U+22C6 STAR OPERATOR
    '⋄': '\\diamond',       # U+22C4 DIAMOND OPERATOR
    '≀': '\\wr',            # U+2240 WREATH PRODUCT
    '∓': '\\mp',            # U+2213 MINUS-OR-PLUS
    '∆': '\\Delta',         # U+2206 INCREMENT (same as \\Delta)
    '√': '\\surd',          # U+221A SQUARE ROOT sign (without overline)
    '∛': '\\sqrt[3]{}',     # U+221B CUBE ROOT
    '∜': '\\sqrt[4]{}',     # U+221C FOURTH ROOT
    # Logic / negation
    '¬': '\\lnot',          # U+00AC NOT SIGN

    # Letterlike symbols
    'ð': '\\eth',           # U+00F0 LATIN SMALL LETTER ETH (used as math symbol)
    'ⅆ': '\\mathrm{d}',     # U+2146 DOUBLE-STRUCK ITALIC SMALL D (differential)
    '†': '\\dagger',        # U+2020 DAGGER
    '‡': '\\ddagger',       # U+2021 DOUBLE DAGGER

    # Additional relation symbols (confirmed KaTeX-supported)
    '∖': '\\setminus',      # U+2216 SET MINUS
    '∽': '\\backsim',       # U+223D REVERSED TILDE
    '≂': '\\eqsim',         # U+2242 MINUS TILDE
    '≊': '\\approxeq',      # U+224A ALMOST EQUAL OR EQUAL TO
    '≎': '\\Bumpeq',        # U+224E GEOMETRICALLY EQUIVALENT TO
    '≏': '\\bumpeq',        # U+224F DIFFERENCE BETWEEN
    '≐': '\\doteq',         # U+2250 APPROACHES THE LIMIT
    '≑': '\\Doteq',         # U+2251 GEOMETRICALLY EQUAL TO
    '≒': '\\fallingdotseq', # U+2252 APPROXIMATELY EQUAL TO OR IMAGE OF
    '≓': '\\risingdotseq',  # U+2253 IMAGE OF OR APPROXIMATELY EQUAL TO
    '≖': '\\eqcirc',        # U+2256 RING IN EQUAL TO
    '≗': '\\circeq',        # U+2257 RING EQUAL TO
    '≬': '\\between',       # U+226C BETWEEN
    '≳': '\\gtrsim',        # U+2273 GREATER-THAN OR EQUIVALENT TO
    '≶': '\\lessgtr',       # U+2276 LESS-THAN OR GREATER-THAN
    '≷': '\\gtrless',       # U+2277 GREATER-THAN OR LESS-THAN
    '≾': '\\precsim',       # U+227E PRECEDES OR EQUIVALENT TO
    '≿': '\\succsim',       # U+227F SUCCEEDS OR EQUIVALENT TO
    '⊏': '\\sqsubset',      # U+228F SQUARE IMAGE OF
    '⊐': '\\sqsupset',      # U+2290 SQUARE ORIGINAL OF
    '⊑': '\\sqsubseteq',    # U+2291 SQUARE IMAGE OF OR EQUAL TO
    '⊒': '\\sqsupseteq',    # U+2292 SQUARE ORIGINAL OF OR EQUAL TO
    '⊚': '\\circledcirc',   # U+229A CIRCLED RING OPERATOR
    '⊛': '\\circledast',    # U+229B CIRCLED ASTERISK OPERATOR
    '⊝': '\\circleddash',   # U+229D CIRCLED DASH
    '⊩': '\\Vdash',         # U+22A9 FORCES
    '⊪': '\\Vvdash',        # U+22AA TRIPLE VERTICAL BAR RIGHT TURNSTILE
    '⋈': '\\bowtie',        # U+22C8 BOWTIE
    '⋍': '\\backsimeq',     # U+22CD REVERSED TILDE EQUALS
    '⋐': '\\Subset',        # U+22D0 DOUBLE SUBSET
    '⋑': '\\Supset',        # U+22D1 DOUBLE SUPERSET
    '⋔': '\\pitchfork',     # U+22D4 PITCHFORK
    '⋙': '\\ggg',           # U+22D9 VERY MUCH GREATER-THAN
    '⋛': '\\gtreqless',     # U+22DB GREATER-THAN EQUAL TO OR LESS-THAN
    '⋞': '\\curlyeqprec',   # U+22DE EQUAL TO OR PRECEDES
    '⋟': '\\curlyeqsucc',   # U+22DF EQUAL TO OR SUCCEEDS
    '⌣': '\\smile',         # U+2323 SMILE
    '⟂': '\\bot',           # U+27C2 PERPENDICULAR (distinct from ⊥ U+22A5)
    '⩽': '\\leqslant',      # U+2A7D LESS-THAN OR SLANTED EQUAL TO
    '⩾': '\\geqslant',      # U+2A7E GREATER-THAN OR SLANTED EQUAL TO
    '⪅': '\\lessapprox',    # U+2A85 LESS-THAN OR APPROXIMATE
    '⪆': '\\gtrapprox',     # U+2A86 GREATER-THAN OR APPROXIMATE
    '⪋': '\\lesseqqgtr',    # U+2A8B LESS-THAN ABOVE GREATER-THAN ABOVE LESS-THAN
    '⪌': '\\gtreqqless',    # U+2A8C GREATER-THAN ABOVE GREATER-THAN ABOVE LESS-THAN
    '⪕': '\\eqslantless',   # U+2A95 SLANTED EQUAL TO OR LESS-THAN
    '⪖': '\\eqslantgtr',    # U+2A96 SLANTED EQUAL TO OR GREATER-THAN
    '⫅': '\\subseteqq',     # U+2AC5 SUBSET OF ABOVE EQUALS SIGN
    '⫆': '\\supseteqq',     # U+2AC6 SUPERSET OF ABOVE EQUALS SIGN
    '◯': '\\bigcirc',       # U+25EF LARGE CIRCLE
    '∰': '\\oiiint',        # U+2230 VOLUME INTEGRAL (triple contour)
    '⨆': '\\bigsqcup',      # U+2A06 N-ARY SQUARE UNION OPERATOR

    # Additional arrows (confirmed KaTeX-supported)
    '↝': '\\leadsto',            # U+219D RIGHTWARDS WAVE ARROW
    '↚': '\\nleftarrow',         # U+219A NOT LEFTWARDS ARROW
    '↛': '\\nrightarrow',        # U+219B NOT RIGHTWARDS ARROW
    '↞': '\\twoheadleftarrow',   # U+219E
    '↠': '\\twoheadrightarrow',  # U+21A0
    '↢': '\\leftarrowtail',      # U+21A2
    '↣': '\\rightarrowtail',     # U+21A3
    '↫': '\\looparrowleft',      # U+21AB LEFTWARDS ARROW WITH LOOP
    '↬': '\\looparrowright',     # U+21AC RIGHTWARDS ARROW WITH LOOP
    '↮': '\\nleftrightarrow',    # U+21AE
    '↰': '\\Lsh',                # U+21B0
    '↱': '\\Rsh',                # U+21B1
    '↾': '\\upharpoonright',     # U+21BE UPWARDS HARPOON WITH BARB RIGHTWARDS
    '↿': '\\upharpoonleft',      # U+21BF UPWARDS HARPOON WITH BARB LEFTWARDS
    '⇂': '\\downharpoonright',   # U+21C2 DOWNWARDS HARPOON WITH BARB RIGHTWARDS
    '⇃': '\\downharpoonleft',    # U+21C3 DOWNWARDS HARPOON WITH BARB LEFTWARDS
    '⇠': '\\dashleftarrow',      # U+21E0 LEFTWARDS DASHED ARROW
    '⇢': '\\dashrightarrow',     # U+21E2 RIGHTWARDS DASHED ARROW
    '⇝': '\\rightsquigarrow',    # U+21DD
    '⟼': '\\longmapsto',         # U+27FC LONG RIGHTWARDS ARROW FROM BAR

    # Binary operators (confirmed KaTeX-supported)
    '∔': '\\dotplus',        # U+2214 DOT PLUS
    '⊞': '\\boxplus',        # U+229E SQUARED PLUS
    '⊟': '\\boxminus',       # U+229F SQUARED MINUS
    '⊠': '\\boxtimes',       # U+22A0 SQUARED TIMES
    '⊡': '\\boxdot',         # U+22A1 SQUARED DOT OPERATOR
    '⊺': '\\intercal',       # U+22BA INTERCALATE
    '⋇': '\\divideontimes',  # U+22C7 DIVISION TIMES
    '⋉': '\\ltimes',         # U+22C9 LEFT NORMAL FACTOR SEMIDIRECT PRODUCT
    '⋊': '\\rtimes',         # U+22CA RIGHT NORMAL FACTOR SEMIDIRECT PRODUCT
    '⋋': '\\leftthreetimes', # U+22CB LEFT SEMIDIRECT PRODUCT
    '⋌': '\\rightthreetimes',# U+22CC RIGHT SEMIDIRECT PRODUCT
    '⋎': '\\curlyvee',       # U+22CE CURLY LOGICAL OR
    '⋏': '\\curlywedge',     # U+22CF CURLY LOGICAL AND
    '⋒': '\\Cap',            # U+22D2 DOUBLE INTERSECTION
    '⋓': '\\Cup',            # U+22D3 DOUBLE UNION
    '⋢': '\\not\\sqsubseteq', # U+22E2 NOT SQUARE IMAGE OF OR EQUAL TO
    '⋣': '\\not\\sqsupseteq', # U+22E3 NOT SQUARE ORIGINAL OF OR EQUAL TO

    # Negation / not-relation operators (confirmed KaTeX-supported)
    '≁': '\\nsim',           # U+2241 NOT TILDE
    '≄': '\\not\\simeq',     # U+2244 NOT ASYMPTOTICALLY EQUAL TO
    '≇': '\\ncong',          # U+2247 NEITHER APPROXIMATELY NOR ACTUALLY EQUAL TO
    '≉': '\\not\\approx',    # U+2249 NOT ALMOST EQUAL TO — \napprox not in KaTeX
    '≍': '\\asymp',          # U+224D EQUIVALENT TO
    '≢': '\\not\\equiv',     # U+2262 NOT IDENTICAL TO
    '≦': '\\leqq',           # U+2266 LESS-THAN OVER EQUAL TO
    '≧': '\\geqq',           # U+2267 GREATER-THAN OVER EQUAL TO
    '≨': '\\lneqq',          # U+2268 LESS-THAN BUT NOT EQUAL TO
    '≩': '\\gneqq',          # U+2269 GREATER-THAN BUT NOT EQUAL TO
    '≭': '\\not\\asymp',     # U+226D NOT EQUIVALENT TO
    '≮': '\\nless',          # U+226E NOT LESS-THAN
    '≯': '\\ngtr',           # U+226F NOT GREATER-THAN
    '≰': '\\nleq',           # U+2270 NEITHER LESS-THAN NOR EQUAL TO
    '≱': '\\ngeq',           # U+2271 NEITHER GREATER-THAN NOR EQUAL TO
    '≲': '\\lesssim',        # U+2272 LESS-THAN OR EQUIVALENT TO
    '⊀': '\\nprec',          # U+2280 DOES NOT PRECEDE
    '⊁': '\\nsucc',          # U+2281 DOES NOT SUCCEED
    '⊄': '\\not\\subset',    # U+2284 NOT A SUBSET OF — \nsubset NOT in KaTeX
    '⊅': '\\not\\supset',   # U+2285 NOT A SUPERSET OF — \nsupset NOT in KaTeX
    '⊬': '\\nvdash',         # U+22AC DOES NOT PROVE
    '⊭': '\\nvDash',         # U+22AD NOT TRUE
    '⊮': '\\nVdash',         # U+22AE DOES NOT FORCE
    '⊯': '\\nVDash',         # U+22AF NEGATED DOUBLE VERTICAL BAR DOUBLE RIGHT TURNSTILE
    '⊈': '\\nsubseteq',      # U+2288 NEITHER A SUBSET OF NOR EQUAL TO
    '⊉': '\\nsupseteq',      # U+2289 NEITHER A SUPERSET OF NOR EQUAL TO
    '⊊': '\\subsetneq',      # U+228A SUBSET OF WITH NOT EQUAL TO
    '⊋': '\\supsetneq',      # U+228B SUPERSET OF WITH NOT EQUAL TO
    '⋠': '\\npreceq',        # U+22E0 DOES NOT PRECEDE OR EQUAL
    '⋡': '\\nsucceq',        # U+22E1 DOES NOT SUCCEED OR EQUAL
    '⋦': '\\lnsim',          # U+22E6 LESS-THAN BUT NOT EQUIVALENT TO
    '⋧': '\\gnsim',          # U+22E7 GREATER-THAN BUT NOT EQUIVALENT TO
    '⋨': '\\precnsim',       # U+22E8 PRECEDES BUT NOT EQUIVALENT TO
    '⋩': '\\succnsim',       # U+22E9 SUCCEEDS BUT NOT EQUIVALENT TO
    '⋪': '\\ntriangleleft',  # U+22EA NOT NORMAL SUBGROUP OF
    '⋫': '\\ntriangleright', # U+22EB DOES NOT CONTAIN AS NORMAL SUBGROUP
    '⋬': '\\ntrianglelefteq',  # U+22EC NOT NORMAL SUBGROUP OF OR EQUAL TO
    '⋭': '\\ntrianglerighteq', # U+22ED DOES NOT CONTAIN AS NORMAL SUBGROUP OR EQUAL

    # Additional comparison operators
    '⋖': '\\lessdot',        # U+22D6 LESS-THAN WITH DOT
    '⋗': '\\gtrdot',         # U+22D7 GREATER-THAN WITH DOT
    '⋘': '\\lll',            # U+22D8 VERY MUCH LESS-THAN
    '⋚': '\\lesseqgtr',      # U+22DA LESS-THAN EQUAL TO OR GREATER-THAN

    # Musical / suit symbols
    '♠': '\\spadesuit',   # U+2660
    '♡': '\\heartsuit',   # U+2661
    '♢': '\\diamondsuit', # U+2662
    '♣': '\\clubsuit',    # U+2663
    '♭': '\\flat',        # U+266D
    '♮': '\\natural',     # U+266E
    '♯': '\\sharp',       # U+266F

    # Miscellaneous
    '✓': '\\checkmark',   # U+2713 CHECK MARK
    # NOTE: Å (U+00C5), ℇ (U+2107), ℩ (U+2129),
    # ℮ (U+212E), Ⅎ/Ⅎ (U+2132), and rare wave/dashed arrows (↜↝↭↲↳⇜⇠⇢) have
    # no standard KaTeX command — left as raw Unicode (KaTeX handles via direct input).
}

def cleanup_text(text):
    if not text: return ""
    # Remove Unicode invisible operators (function application, etc.)
    for ch in ('\u2061', '\u2062', '\u2063', '\u2064'):
        text = text.replace(ch, '')
    # Non-breaking space → regular space
    text = text.replace('\xa0', ' ')
    # Unicode Mathematical Alphabet → LaTeX (math italic, bold, script, fraktur, etc.)
    # Apply before UNICODE_TO_LATEX to handle math italic Greek before plain Greek
    result = []
    for ch in text:
        mapped = _MATH_ALPHA_TO_LATEX.get(ch)
        if mapped is not None:
            result.append(f' {mapped} ')
        else:
            result.append(ch)
    text = ''.join(result)
    # Common math symbols (Greek, operators, arrows, etc.)
    for k, v in UNICODE_TO_LATEX.items():
        text = text.replace(k, f" {v} ")
    return text

# ==============================================================================
# Simple Types (ST) - Data Extraction
# RNC 파일의 기본 데이터 타입들입니다. 추출은 1:1로 전부 진행하되, 
# LaTeX 생태계와 맞지 않는 부분은 CT(Complex Type) 단계에서 무시([IGNORE])
# ==============================================================================
def parse_m_ST_Integer255(val): 
    # [IGNORE: Word UI 렌더링 전용] 1~255 정수. 주로 "수동 줄바꿈 시 화면의 어느 x좌표에 정렬할 것인가(alnAt)" 같은 GUI 전용 상태값. 
    # 왜 무시하는가? -> LaTeX는 환경(\begin{align} 등) 내에서 '&' 기호를 통해 정렬을 매우 수학적인 구조로 자동 처리하기 때문에 물리적인 인덱스는 의미가 없다.
    return int(val) if val else None

def parse_m_ST_Integer2(val):
    # [IGNORE: LaTeX 자동화 영역] -2~2 정수. "첨자 크기(argSz)를 기본치보다 강제로 더 키울까 줄일까"를 결정.
    # 왜 무시하는가? -> LaTeX의 TeX 엔진은 첨자 중첩 깊이에 따라 크기(\scriptstyle, \scriptscriptstyle)를 자체 로직으로 자동 결정. 
    return int(val) if val else None

def parse_m_ST_SpacingRule(val):
    # [IGNORE: LaTeX 정렬 시스템 위임] 0~4 정수. "행렬 사이 간격을 어떻게 띄울 것인지"에 대한 Word의 규칙 세트 매핑 번호.
    # 왜 무시하는가? -> bmatrix 등 LaTeX 행렬 환경 내부의 고유 Grid Spacing이 수학적 타이포그래피 표준을 따르므로 변경할 필요가 없어서 생략.
    return int(val) if val else None

def parse_m_ST_UnSignedInteger(val):
    # [IGNORE: Word 수동 비율 제어] 강제 여백의 크기나 비율. (위와 동일한 이유로 무시)
    return int(val) if val else None

def parse_m_ST_Char(val):
    """m:ST_Char (maxLength=1) — 수학 괄호/기호 단일 문자.

    유니코드가 직접 저장된 경우(m:val="π" 등)를 cleanup_text()를 통해
    LaTeX 명령어로 변환한다. UNICODE_TO_LATEX 맵에 없는 기호는
    원본 문자를 그대로 반환한다.

    버그 수정: m:CT_Char 경로(nary.chr, groupChr.chr 등)에서
    유니코드가 변환되지 않고 그대로 출력되던 문제 해결.
    """
    if val is None:
        return None
    if val == "":
        return "" 
    converted = cleanup_text(val).strip()
    return converted if converted else val

def parse_s_ST_OnOff(val):
    # s_ST_OnOff (shared-commonSimpleTypes.rnc) — 활성화 유무 플래그.
    # shared 네임스페이스 타입이나 m:CT_OnOff에서 참조하므로 여기서 정의.
    return str(val).lower() in ["1", "true", "on"] if val else False


def parse_s_ST_String(val):
    # s_ST_String (shared-commonSimpleTypes.rnc) — 문자열 그대로 반환. (글꼴 이름 등)
    # shared 네임스페이스 타입이나 m:CT_String, m:CT_Text에서 참조.
    return str(val) if val else ""


def parse_s_ST_XAlign(val):
    # s_ST_XAlign (shared-commonSimpleTypes.rnc) — 엘리먼트 가로 정렬(left, right, center).
    # [IGNORE: 구조적 차이] Word는 칸마다 정렬을 커스텀하지만, LaTeX 표준 행렬 구조는
    # 열(Column) 전체 단위의 정렬을 베이스로 한다. 각 칸에 \hfill을 넣으면 코드가
    # 무거워지므로 생략한다.
    return str(val) if val else None


def parse_s_ST_YAlign(val):
    # s_ST_YAlign (shared-commonSimpleTypes.rnc) — 엘리먼트 세로 정렬(top, center, bot).
    # [IGNORE: 구조적 차이] 위와 동일.
    return str(val) if val else None


def parse_m_ST_Shp(val):
    # [IGNORE: LaTeX 자동화 영역] 괄호가 본문 높이에 "맞춰서 늘어날지(match)" "가운데 정렬될지(centered)" 여부.
    # 왜 무시하는가? -> LaTeX에서는 \left 와 \right 명령어가 수식의 높이(height)를 수학적으로 분석하여 괄호 모양과 크기를 늘려주므로 Word의 결정 힌트가 무의미.
    return str(val) if val else None

def parse_m_ST_FType(val):
    # [IGNORE: 구조적 단순화] 분수(Fraction)를 렌더링할 때 가로줄 긋기, 빗금 긋기, 줄 없애기 등 디자인.
    # 왜 무시하는가? -> 표준 \frac 으로 통일 및 단순화 처리합니다.
    return str(val) if val else None

def parse_m_ST_LimLoc(val):
    # [IGNORE: LaTeX 자동화 영역] 극한 값을 적분 기호 아래위로 붙일지, 옆에 첨자로 붙일지 결정.
    # 왜 무시하는가? -> LaTeX 또한 인라인 모드냐 디스플레이 모드냐에 따라 위치를 알아서 옮긴다. 강제로 넘겨줄 수 있으나, LaTeX 기본 엔진 로직에 맡기는 것이 낫다고 판단하여 생략.
    return str(val) if val else None

def parse_m_ST_TopBot(val):
    # 윗줄(\overline)인지 아랫줄(\underline), 또는 위 괄호인지 아래 괄호인지 구분하는 핵심 변수.
    return str(val) if val else None

def parse_m_ST_Script(val):
    # 수학 폰트 서식(\mathbb 등). 
    return str(val) if val else None

def parse_m_ST_Style(val):
    # 굵게, 이탤릭. 
    return str(val) if val else None

def parse_m_ST_Jc(val):
    # [IGNORE: LaTeX 환경 위임] 방정식 전체를 좌측/중앙/우측에 둘지 결정.
    # 왜 무시하는가? -> 수식 변환본이 들어갈 \begin{equation*} 등의 환경 전체가 문서의 TeX 클래스 설정(fleqn 등)에 의해 정렬되므로 개별 수식에서의 지정은 구조상 무의미.
    return str(val) if val else None

def parse_s_ST_TwipsMeasure(val):
    # s_ST_TwipsMeasure (shared-commonSimpleTypes.rnc) — Twips 단위 측정값.
    # [IGNORE: 완전한 단위 이질성 및 비합리성]
    # Twips는 '1인치의 1/1440'을 뜻하는 Word 기반의 물리적 절대치 단위다.
    # 해상도와 글꼴 환경에 따라 반응형으로 동작해야 하는 TeX 시스템(\quad 등)에
    # 물리적인 Twips 측정값을 주입하면 수식의 유연성이 파괴되므로 무시한다.
    return str(val) if val else None

def parse_m_ST_BreakBin(val):
    # [IGNORE: LaTeX 알고리즘 위임] 
    # 왜 무시하는가? -> Word에서 사용자가 엔터를 치지 않아도 화면이 좁아 수식이 강제 개행될 때 "어느 연산자 기준"으로 줄바꿈할지 결정한다. 
    # LaTeX는 자체적으로 줄바꿈 패널티와 연산자 룰을 보유하고 있으므로, Word의 규칙을 강제할 필요가 없다.
    return str(val) if val else None

def parse_m_ST_BreakBinSub(val):
    # [IGNORE: 위와 동일] 뺄셈 줄바꿈 규칙
    return str(val) if val else None


# ==============================================================================
# Complex Types (CT) - Value Wrappers 
# 단순 어트리뷰트 트리 파싱을 위한 래퍼 계층. 
# ==============================================================================
def parse_m_CT_Integer255(node): return parse_m_ST_Integer255(_get_val(node))
def parse_m_CT_Integer2(node): return parse_m_ST_Integer2(_get_val(node))
def parse_m_CT_SpacingRule(node): return parse_m_ST_SpacingRule(_get_val(node))
def parse_m_CT_UnSignedInteger(node): return parse_m_ST_UnSignedInteger(_get_val(node))
def parse_m_CT_Char(node): return parse_m_ST_Char(_get_val(node))
def parse_m_CT_OnOff(node):
    # ECMA-376: CT_OnOff의 val 속성 기본값은 "1" (요소 존재 = true).
    # node가 None이면 False, node가 있고 val 속성이 없으면 True (presence = on).
    if node is None: return False
    val = _get_val(node)
    if val is None: return True   # 속성 생략 → 기본값 true
    return parse_s_ST_OnOff(val)
def parse_m_CT_String(node): return parse_s_ST_String(_get_val(node))
def parse_m_CT_XAlign(node): return parse_s_ST_XAlign(_get_val(node))
def parse_m_CT_YAlign(node): return parse_s_ST_YAlign(_get_val(node))
def parse_m_CT_Shp(node): return parse_m_ST_Shp(_get_val(node))
def parse_m_CT_FType(node): return parse_m_ST_FType(_get_val(node))
def parse_m_CT_LimLoc(node): return parse_m_ST_LimLoc(_get_val(node))
def parse_m_CT_TopBot(node): return parse_m_ST_TopBot(_get_val(node))
def parse_m_CT_Script(node): return parse_m_ST_Script(_get_val(node))
def parse_m_CT_Style(node): return parse_m_ST_Style(_get_val(node))
def parse_m_CT_OMathJc(node): return parse_m_ST_Jc(_get_val(node))
def parse_m_CT_TwipsMeasure(node): return parse_s_ST_TwipsMeasure(_get_val(node))
def parse_m_CT_BreakBin(node): return parse_m_ST_BreakBin(_get_val(node))
def parse_m_CT_BreakBinSub(node): return parse_m_ST_BreakBinSub(_get_val(node))


# ==============================================================================
# Element Groups (EG) - Routing & Container Macros
# ==============================================================================
def parse_m_EG_ScriptStyle(node):
    if node is None: return {}
    return {
        'scr': parse_m_CT_Script(_get_child(node, 'scr')),
        'sty': parse_m_CT_Style(_get_child(node, 'sty'))
    }

def parse_m_EG_OMathMathElements(node):
    if node is None: return ""
    tag = _get_tag(node)
    
    dispatch_map = {
        'acc': parse_m_CT_Acc, 'bar': parse_m_CT_Bar, 'box': parse_m_CT_Box,
        'borderBox': parse_m_CT_BorderBox, 'd': parse_m_CT_D, 'eqArr': parse_m_CT_EqArr,
        'f': parse_m_CT_F, 'func': parse_m_CT_Func, 'groupChr': parse_m_CT_GroupChr,
        'limLow': parse_m_CT_LimLow, 'limUpp': parse_m_CT_LimUpp, 'm': parse_m_CT_M,
        'nary': parse_m_CT_Nary, 'phant': parse_m_CT_Phant, 'rad': parse_m_CT_Rad,
        'sPre': parse_m_CT_SPre, 'sSub': parse_m_CT_SSub, 'sSubSup': parse_m_CT_SSubSup,
        'sSup': parse_m_CT_SSup, 'r': parse_m_CT_R
    }
    
    if tag in dispatch_map:
        return dispatch_map[tag](node)
    return ""

def parse_m_EG_OMathElements(node):
    return parse_m_EG_OMathMathElements(node)


# ==============================================================================
# Complex Types (CT) - Properties Map
# ==============================================================================
def parse_m_CT_ManualBreak(node):
    if node is None: return {}
    return {'alnAt': parse_m_ST_Integer255(_get_val(node, 'alnAt'))}

def parse_m_CT_CtrlPr(node): 
    # [IGNORE: 순수 어플리케이션(Word) 메타데이터]
    # 이유: 이 속성은 특정 수식을 누가 수정했는지나 단축키 지정 등 '파일 편집 환경의 히스토리'를 저장한 한것.
    # 수식 결과물과는 수학적으로 아무 상관도 없으므로 제외.
    if node is None: return {}
    return {}

def parse_m_CT_RPR(node):
    if node is None: return {}
    res = {
        'lit': parse_m_CT_OnOff(_get_child(node, 'lit')),
        'nor': parse_m_CT_OnOff(_get_child(node, 'nor')),
        'brk': parse_m_CT_ManualBreak(_get_child(node, 'brk')),
        'aln': parse_m_CT_OnOff(_get_child(node, 'aln'))
    }
    res.update(parse_m_EG_ScriptStyle(node))
    return res

def parse_m_CT_OMathArgPr(node): 
    if node is None: return {}
    return {'argSz': parse_m_CT_Integer2(_get_child(node, 'argSz'))}

def parse_m_CT_AccPr(node):
    if node is None: return {}
    return {
        'chr': parse_m_CT_Char(_get_child(node, 'chr')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_BarPr(node):
    if node is None: return {}
    return {
        'pos': parse_m_CT_TopBot(_get_child(node, 'pos')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_BoxPr(node):
    if node is None: return {}
    return {
        'opEmu': parse_m_CT_OnOff(_get_child(node, 'opEmu')),
        'noBreak': parse_m_CT_OnOff(_get_child(node, 'noBreak')),
        'diff': parse_m_CT_OnOff(_get_child(node, 'diff')),
        'brk': parse_m_CT_ManualBreak(_get_child(node, 'brk')),
        'aln': parse_m_CT_OnOff(_get_child(node, 'aln')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_BorderBoxPr(node):
    # [IGNORE: 변환 복잡성 및 출력 품질 관리] 테두리 박스의 선을 일부만 지우거나 대각선으로 취소선을 긋는 등 Word의 자유도 높은 그리기 옵션.
    # 이유: hideTop(윗테두리 가리기), strikeTLBR(대각선으로 취소선 긋기) 등 Word의 그리기 도구.
    # LaTeX로 완전한 이식이 매우 어렵고, 수학적 표현의 명확성에도 크게 기여하지 않으므로 일괄적으로 무시.
    if node is None: return {}
    return {
        'hideTop': parse_m_CT_OnOff(_get_child(node, 'hideTop')),
        'hideBot': parse_m_CT_OnOff(_get_child(node, 'hideBot')),
        'hideLeft': parse_m_CT_OnOff(_get_child(node, 'hideLeft')),
        'hideRight': parse_m_CT_OnOff(_get_child(node, 'hideRight')),
        'strikeH': parse_m_CT_OnOff(_get_child(node, 'strikeH')),
        'strikeV': parse_m_CT_OnOff(_get_child(node, 'strikeV')),
        'strikeBLTR': parse_m_CT_OnOff(_get_child(node, 'strikeBLTR')),
        'strikeTLBR': parse_m_CT_OnOff(_get_child(node, 'strikeTLBR')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_DPr(node):
    if node is None: return {}
    return {
        'begChr': parse_m_CT_Char(_get_child(node, 'begChr')),
        'sepChr': parse_m_CT_Char(_get_child(node, 'sepChr')),
        'endChr': parse_m_CT_Char(_get_child(node, 'endChr')),
        'grow': parse_m_CT_OnOff(_get_child(node, 'grow')),
        'shp': parse_m_CT_Shp(_get_child(node, 'shp')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_EqArrPr(node):
    if node is None: return {}
    return {
        'baseJc': parse_m_CT_YAlign(_get_child(node, 'baseJc')),
        'maxDist': parse_m_CT_OnOff(_get_child(node, 'maxDist')),
        'objDist': parse_m_CT_OnOff(_get_child(node, 'objDist')),
        'rSpRule': parse_m_CT_SpacingRule(_get_child(node, 'rSpRule')),
        'rSp': parse_m_CT_UnSignedInteger(_get_child(node, 'rSp')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_FPr(node):
    if node is None: return {}
    return {
        'type': parse_m_CT_FType(_get_child(node, 'type')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_FuncPr(node):
    if node is None: return {}
    return {'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))}
    
def parse_m_CT_GroupChrPr(node):
    if node is None: return {}
    return {
        'chr': parse_m_CT_Char(_get_child(node, 'chr')),
        'pos': parse_m_CT_TopBot(_get_child(node, 'pos')),
        'vertJc': parse_m_CT_TopBot(_get_child(node, 'vertJc')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_LimLowPr(node):
    if node is None: return {}
    return {'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))}

def parse_m_CT_LimUppPr(node):
    if node is None: return {}
    return {'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))}

def parse_m_CT_MCPr(node):
    if node is None: return {}
    return {
        'count': parse_m_CT_Integer255(_get_child(node, 'count')),
        'mcJc': parse_m_CT_XAlign(_get_child(node, 'mcJc'))
    }

def parse_m_CT_MPr(node):
    if node is None: return {}
    return {
        'baseJc': parse_m_CT_YAlign(_get_child(node, 'baseJc')),
        'plcHide': parse_m_CT_OnOff(_get_child(node, 'plcHide')),
        'rSpRule': parse_m_CT_SpacingRule(_get_child(node, 'rSpRule')),
        'cGpRule': parse_m_CT_SpacingRule(_get_child(node, 'cGpRule')),
        'rSp': parse_m_CT_UnSignedInteger(_get_child(node, 'rSp')),
        'cSp': parse_m_CT_UnSignedInteger(_get_child(node, 'cSp')),
        'cGp': parse_m_CT_UnSignedInteger(_get_child(node, 'cGp')),
        'mcs': parse_m_CT_MCS(_get_child(node, 'mcs')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_NaryPr(node):
    if node is None: return {}
    return {
        'chr': parse_m_CT_Char(_get_child(node, 'chr')),
        'limLoc': parse_m_CT_LimLoc(_get_child(node, 'limLoc')),
        'grow': parse_m_CT_OnOff(_get_child(node, 'grow')),
        'subHide': parse_m_CT_OnOff(_get_child(node, 'subHide')),
        'supHide': parse_m_CT_OnOff(_get_child(node, 'supHide')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_PhantPr(node):
    if node is None: return {}
    return {
        'show': parse_m_CT_OnOff(_get_child(node, 'show')),
        'zeroWid': parse_m_CT_OnOff(_get_child(node, 'zeroWid')),
        'zeroAsc': parse_m_CT_OnOff(_get_child(node, 'zeroAsc')),
        'zeroDesc': parse_m_CT_OnOff(_get_child(node, 'zeroDesc')),
        'transp': parse_m_CT_OnOff(_get_child(node, 'transp')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }
    
def parse_m_CT_RadPr(node):
    if node is None: return {}
    return {
        'degHide': parse_m_CT_OnOff(_get_child(node, 'degHide')),
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_SPrePr(node):
    if node is None: return {}
    return {'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))}
def parse_m_CT_SSubPr(node):
    if node is None: return {}
    return {'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))}
def parse_m_CT_SSupPr(node):
    if node is None: return {}
    return {'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))}

def parse_m_CT_SSubSupPr(node):
    if node is None: return {}
    return {
        'alnScr': parse_m_CT_OnOff(_get_child(node, 'alnScr')), 
        'ctrlPr': parse_m_CT_CtrlPr(_get_child(node, 'ctrlPr'))
    }

def parse_m_CT_OMathParaPr(node):
    if node is None: return {}
    return {'jc': parse_m_CT_OMathJc(_get_child(node, 'jc'))}

def parse_m_CT_MathPr(node): 
    if node is None: return {}
    
    wrapIndent = _get_child(node, 'wrapIndent')
    wrapRight = _get_child(node, 'wrapRight')
    
    return {
        'mathFont': parse_m_CT_String(_get_child(node, 'mathFont')),
        'brkBin': parse_m_CT_BreakBin(_get_child(node, 'brkBin')),
        'brkBinSub': parse_m_CT_BreakBinSub(_get_child(node, 'brkBinSub')),
        'smallFrac': parse_m_CT_OnOff(_get_child(node, 'smallFrac')),
        'dispDef': parse_m_CT_OnOff(_get_child(node, 'dispDef')),
        'lMargin': parse_m_CT_TwipsMeasure(_get_child(node, 'lMargin')),
        'rMargin': parse_m_CT_TwipsMeasure(_get_child(node, 'rMargin')),
        'defJc': parse_m_CT_OMathJc(_get_child(node, 'defJc')),
        'preSp': parse_m_CT_TwipsMeasure(_get_child(node, 'preSp')),
        'postSp': parse_m_CT_TwipsMeasure(_get_child(node, 'postSp')),
        'interSp': parse_m_CT_TwipsMeasure(_get_child(node, 'interSp')),
        'intraSp': parse_m_CT_TwipsMeasure(_get_child(node, 'intraSp')),
        'wrapIndent': parse_m_CT_TwipsMeasure(wrapIndent) if wrapIndent is not None else None,
        'wrapRight': parse_m_CT_OnOff(wrapRight) if wrapRight is not None else None,
        'intLim': parse_m_CT_LimLoc(_get_child(node, 'intLim')),
        'naryLim': parse_m_CT_LimLoc(_get_child(node, 'naryLim'))
    }

# ==============================================================================
# Complex Types (CT) - AST to LaTeX Generation
# ==============================================================================
# [설명: 연산자 속성(~~Pr) 변수들을 변환 구문에서 의도적으로 무시(Ignore)하는 이유]
# AST 생성 단계에서 DPr, FPr, MPr, ArgPr 등 수많은 속성(Pr) 딕셔너리를 파싱해 가져오지만,
# 실제 아래의 함수들에서 LaTeX 문자열을 조립(Return)할 때는 이 속성값 대부분을 안 쓴ㄷ다
# 워드의 'Pr'들은 화면의 픽셀 간격, 요소 정렬(alnScr), 특정 테두리 숨김(hideTop),
# 문자 강제 성장(grow/shp) 등 완벽하게 'GUI 워드프로세서 렌더링용 시각 지시자'이기 때문이다. 
# 
# 반면 LaTeX은 \frac, \begin{bmatrix}, \left( \right), ^, _ 등의 명령어 자체에
# 전문적인 타이포그래피/간격/배치 알고리즘이 내장되어 있다. 이 내장 렌더링을
# 따르는 것이 수학적으로 더 정확하고 코드가 깔끔해진다고 판단하여, Word의 시각적 지시자들은 대부분 무시하고 LaTeX의 기본 렌더링 엔진에 맡기는 전략.
# (단, 괄호기호의 종류(begChr)나, 적분기호 종류(chr) 같이 '수학적 본질'을 결정하는 속성은 사용합니다.)
# ==============================================================================
def parse_m_CT_OMathArg(node):
    if node is None: return ""
    # [IGNORE: argPr] 첨자의 크기를 강제로 키우고 줄이는(argSz) 물리적 비율 속성 무시
    argPr = parse_m_CT_OMathArgPr(_get_child(node, 'argPr')) 
    
    content = ""
    for child in node:
        tag = _get_tag(child)
        if tag not in ('argPr', 'ctrlPr'): 
            content += parse_m_EG_OMathElements(child)
    return content

def parse_m_CT_Text(node):
    if node is None: return ""
    return node.text if node.text else ""

def parse_m_CT_R(node):
    rPr = parse_m_CT_RPR(_get_child(node, 'rPr'))

    content = ""
    for t_node in _get_children(node, 't'):
        raw_text = parse_m_CT_Text(t_node)
        content += cleanup_text(raw_text)

    # m:nor (normal text): 수식 환경 안의 일반 텍스트 — KaTeX \text{}
    # ECMA-376 §22.1.2.77 rPr/nor — scr/sty와 상호 배타적이므로 먼저 검사
    if rPr.get('nor'):
        if '\\' in content:
            # KaTeX parse error 방지: \pi 등 매크로가 포함되어 있으면 \text{} 내부에서 에러 발생
            # 매크로를 허용하는 \mathrm{}을 대신 사용하고 공백을 ~ 로 치환하여 띄어쓰기 유지
            safe_content = content.replace(" ", "~")
            return f"\\mathrm{{{safe_content}}}"
        return f"\\text{{{content}}}"

    # m:lit (literal/non-math-italic): 텍스트를 수식 이탤릭이 아닌 upright 로마체로 표시
    # ECMA-376 §22.1.2.68 rPr/lit — KaTeX에서는 \mathrm{}으로 렌더링
    if rPr.get('lit'):
        return f"\\mathrm{{{content}}}"

    scr = rPr.get('scr')
    if scr == 'fraktur': content = f"\\mathfrak{{{content}}}"
    elif scr == 'double-struck': content = f"\\mathbb{{{content}}}"
    elif scr == 'script': content = f"\\mathcal{{{content}}}"
    elif scr == 'sans-serif': content = f"\\textsf{{{content}}}"
    elif scr == 'monospace': content = f"\\mathtt{{{content}}}"

    sty = rPr.get('sty')
    if sty == 'b': content = f"\\mathbf{{{content}}}"
    elif sty == 'i': content = f"\\mathit{{{content}}}"
    elif sty == 'bi': content = f"\\boldsymbol{{{content}}}"

    return content

def parse_m_CT_F(node):
    fPr = parse_m_CT_FPr(_get_child(node, 'fPr'))
    num = parse_m_CT_OMathArg(_get_child(node, 'num'))
    den = parse_m_CT_OMathArg(_get_child(node, 'den'))
    ftype = fPr.get('type') if fPr else None
    if ftype == 'noBar':
        # noBar: 분수 가로줄 없음 (ECMA-376 §22.1.2.36 m_ST_FType).
        # \binom은 괄호를 추가하므로 잘못됨 — Stirling 수 등 비이항계수 표현에서 틀린 출력.
        # \genfrac{}{}{0pt}{}가 올바른 KaTeX 대응: 괄호 없이 num/den을 세로 배치.
        return f"\\genfrac{{}}{{}}{{0pt}}{{}}{{{num}}}{{{den}}}"
    # skw(빗금), lin(인라인 a/b), bar(기본 수평선) → 모두 \frac으로 처리
    return f"\\frac{{{num}}}{{{den}}}"

def parse_m_CT_Rad(node):
    radPr = parse_m_CT_RadPr(_get_child(node, 'radPr'))
    # degHide: radPr 속성에서 읽어야 함 (ECMA-376 §22.1.2.56 radPr/degHide)
    # deg.strip() 추론은 deg 자식이 항상 존재하기 때문에 잘못된 판단으로 이어짐
    deg_hide = radPr.get('degHide') if radPr else None
    deg = parse_m_CT_OMathArg(_get_child(node, 'deg'))
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    if deg_hide:
        return f"\\sqrt{{{e}}}"
    return f"\\sqrt[{deg}]{{{e}}}" if deg.strip() else f"\\sqrt{{{e}}}"

def parse_m_CT_SSup(node):
    # [IGNORE: sSupPr] 위첨자 메타데이터 무시 (위치는 LaTeX 자동 배치 엔진에 위임)
    sSupPr = parse_m_CT_SSupPr(_get_child(node, 'sSupPr'))
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    sup = parse_m_CT_OMathArg(_get_child(node, 'sup'))
    return f"{{{e}}}^{{{sup}}}"

def parse_m_CT_SSub(node):
    # [IGNORE: sSubPr] 아래첨자 메타데이터 무시
    sSubPr = parse_m_CT_SSubPr(_get_child(node, 'sSubPr'))
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    sub = parse_m_CT_OMathArg(_get_child(node, 'sub'))
    return f"{{{e}}}_{{{sub}}}"

def parse_m_CT_SSubSup(node):
    # [IGNORE: sSubSupPr] 첨자 간 상대적 위치 정렬(alnScr) 무시. LaTeX 기본 커닝(Kerning) 사용
    sSubSupPr = parse_m_CT_SSubSupPr(_get_child(node, 'sSubSupPr'))
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    sub = parse_m_CT_OMathArg(_get_child(node, 'sub'))
    sup = parse_m_CT_OMathArg(_get_child(node, 'sup'))
    return f"{{{e}}}_{{{sub}}}^{{{sup}}}"

def parse_m_CT_SPre(node):
    # [IGNORE: sPrePr] 전위첨자 메타데이터 무시
    # 좌측 위, 좌측 아래 첨자.
    sPrePr = parse_m_CT_SPrePr(_get_child(node, 'sPrePr'))
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    sub = parse_m_CT_OMathArg(_get_child(node, 'sub'))
    sup = parse_m_CT_OMathArg(_get_child(node, 'sup'))
    return f"{{}}_{{{sub}}}^{{{sup}}}{e}"

def parse_m_CT_Nary(node):
    naryPr = parse_m_CT_NaryPr(_get_child(node, 'naryPr'))
    chr_val = naryPr.get('chr') or '∫'
    
    # ECMA-376 shared-math.rnc m_CT_Char: m:chr m:val 속성에는 유니코드 기호가 직접 저장됨
    # parse_m_ST_Char를 거쳤으므로 UNICODE_TO_LATEX 맵 적용 후 값이 들어옴
    # → 아래 맵은 변환된 LaTeX 명령어(\\sum 등)와 원본 유니코드 양쪽을 모두 처리
    char_map = {
        # 단일 적분
        '∫': '\\int',   '\\int': '\\int',
        # 이중/삼중/선분 적분
        '∬': '\\iint',  '\\iint': '\\iint',
        '∭': '\\iiint', '\\iiint': '\\iiint',
        '∮': '\\oint',  '\\oint': '\\oint',
        '∯': '\\oiint', '\\oiint': '\\oiint',
        '∰': '\\oiiint', '\\oiiint': '\\oiiint',
        # 합/곱 계열
        '∑': '\\sum',   '\\sum': '\\sum',
        '∏': '\\prod',  '\\prod': '\\prod',
        '∐': '\\coprod','\\coprod': '\\coprod',
        # 집합 연산
        '⋃': '\\bigcup',  '\\bigcup': '\\bigcup',
        '⋂': '\\bigcap',  '\\bigcap': '\\bigcap',
        '⋁': '\\bigvee',  '\\bigvee': '\\bigvee',
        '⋀': '\\bigwedge','\\bigwedge': '\\bigwedge',
        # 대수 연산
        '⨁': '\\bigoplus',  '\\bigoplus': '\\bigoplus',
        '⨂': '\\bigotimes', '\\bigotimes': '\\bigotimes',
        '⨀': '\\bigodot',   '\\bigodot': '\\bigodot',
        '⨄': '\\biguplus',  '\\biguplus': '\\biguplus',
    }
    latex_char = char_map.get(chr_val, '\\int')
    
    sub = parse_m_CT_OMathArg(_get_child(node, 'sub'))
    sup = parse_m_CT_OMathArg(_get_child(node, 'sup'))
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    
    res = latex_char
    if sub: res += f"_{{{sub}}}"
    if sup: res += f"^{{{sup}}}"
    return f"{res} {{{e}}}"

def parse_m_CT_D(node):
    dPr = parse_m_CT_DPr(_get_child(node, 'dPr'))
    # [설명: DPr 사용 방식] begChr(여는괄호), endChr(닫는괄호), sepChr(구분자)처럼 핵심 '수학 데이터'는 꺼내 쓰지만,
    # [IGNORE: DPr 나머지] grow(본문 크기에 맞춤), shp(가운데 정렬) 등 모양 치장용 속성은 무시 (\left, \right가 알아서 최적화함)
    beg = dPr.get('begChr')
    if beg is None: beg = "("
    end = dPr.get('endChr')
    if end is None: end = ")"
    sep = dPr.get('sepChr')
    if sep is None: sep = "|"
    
    # Map Unicode brackets/fences to LaTeX delimiter names
    _DELIM_MAP = {
        '⌊': '\\lfloor', '⌋': '\\rfloor',
        '⌈': '\\lceil',  '⌉': '\\rceil',
        '⟨': '\\langle', '⟩': '\\rangle',
        '⟦': '[\\![', '⟧': ']\\!]',
        '‖': '\\|',
        '{': '\\{', '}': '\\}',
    }

    def _fmt_delim(ch, side):
        if not ch: return f"\\{side}."
        mapped = _DELIM_MAP.get(ch)
        if mapped:
            return f"\\{side}{mapped}"
        if ch in ('(', ')', '[', ']', '|', '.'):
            return f"\\{side}{ch}"
        # fallback: use the char directly (works with XeLaTeX/LuaLaTeX)
        return f"\\{side}{ch}"

    args = [parse_m_CT_OMathArg(e) for e in _get_children(node, 'e')]
    if sep:
        content = f" {sep} ".join(args)
    else:
        content = " ".join(args)
    left = _fmt_delim(beg, 'left')
    right = _fmt_delim(end, 'right')
    return f"{left} {content} {right}"

def parse_m_CT_Acc(node):
    # AccPr의 핵심인 악센트 기호 문자(chr)는 가져오지만, 에디터 단축키 정보(ctrlPr) 등은 무시
    accPr = parse_m_CT_AccPr(_get_child(node, 'accPr'))
    chr_val = accPr.get('chr', '\u0302') 
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    
    # Combining character → LaTeX accent command (all KaTeX-confirmed unless noted)
    acc_map = {
        '\u0302': '\\hat',              # ̂ COMBINING CIRCUMFLEX ACCENT
        '\u0300': '\\grave',            # ̀ COMBINING GRAVE ACCENT
        '\u0301': '\\acute',            # ́ COMBINING ACUTE ACCENT
        '\u0303': '\\tilde',            # ̃ COMBINING TILDE
        '\u0304': '\\bar',              # ̄ COMBINING MACRON
        '\u0305': '\\overline',         # ̅ COMBINING OVERLINE
        '\u0306': '\\breve',            # ̆ COMBINING BREVE
        '\u0307': '\\dot',              # ̇ COMBINING DOT ABOVE
        '\u0308': '\\ddot',             # ̈ COMBINING DIAERESIS
        '\u030A': '\\mathring',         # ̊ COMBINING RING ABOVE
        '\u030C': '\\check',            # ̌ COMBINING CARON
        '\u033F': '\\overline',         # ̿ COMBINING DOUBLE OVERLINE (no KaTeX equiv → \overline)
        '\u20D0': '\\overleftarrow',    # ⃐ COMBINING LEFT HARPOON ABOVE (approx)
        '\u20D1': '\\vec',              # ⃑ COMBINING RIGHT HARPOON ABOVE (approx → \vec)
        '\u20D6': '\\overleftarrow',    # ⃖ COMBINING LEFT ARROW ABOVE
        '\u20D7': '\\vec',              # ⃗ COMBINING RIGHT ARROW ABOVE
        '\u20DB': '\\dddot',            # ⃛ COMBINING THREE DOTS ABOVE
        '\u20DC': '\\ddddot',           # ⃜ COMBINING FOUR DOTS ABOVE
        '\u20E1': '\\overleftrightarrow', # ⃡ COMBINING LEFT RIGHT ARROW ABOVE
    }
    accent = acc_map.get(chr_val, '\\hat')
    return f"{accent}{{{e}}}"

def parse_m_CT_Bar(node):
    barPr = parse_m_CT_BarPr(_get_child(node, 'barPr'))
    pos = barPr.get('pos') or 'top'
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    return f"\\underline{{{e}}}" if pos == 'bot' else f"\\overline{{{e}}}"

def parse_m_CT_MC(node):
    # [IGNORE: mcPr] 열의 개수(count)나 가로 지정 정렬(mcJc) 무시. LaTeX 앰퍼샌드(&) 환경이 자동 결정
    mcPr = parse_m_CT_MCPr(_get_child(node, 'mcPr'))
    return mcPr

def parse_m_CT_MCS(node):
    return [parse_m_CT_MC(mc) for mc in _get_children(node, 'mc')]

def parse_m_CT_M(node):
    # [IGNORE: mPr] 열/행 간격(cSp, cGp, rSp) 및 베이스라인 정렬 등.
    # 워드의 absolute 수치를 강제하는 것보다 matrix 환경의 자체 그리드 알고리즘에 맡기기로 결정.
    mPr = parse_m_CT_MPr(_get_child(node, 'mPr'))
    rows = [parse_m_CT_MR(mr) for mr in _get_children(node, 'mr')]
    return "\\begin{matrix}\n" + " \\\\\n".join(rows) + "\n\\end{matrix}"

def parse_m_CT_MR(node):
    cols = [parse_m_CT_OMathArg(e) for e in _get_children(node, 'e')]
    return " & ".join(cols)

def parse_m_CT_EqArr(node):
    # [IGNORE: eqArrPr] 방정식 배열의 줄간격 룰(rSpRule) 및 여백 옵션 무시. aligned 환경 기본값 사용
    eqArrPr = parse_m_CT_EqArrPr(_get_child(node, 'eqArrPr'))
    args = [parse_m_CT_OMathArg(e) for e in _get_children(node, 'e')]
    body = " \\\\\n".join(args).rstrip('\n')
    return "\\begin{aligned}\n" + body + "\n\\end{aligned}"

def parse_m_CT_Func(node):
    funcPr = parse_m_CT_FuncPr(_get_child(node, 'funcPr'))
    fName = parse_m_CT_OMathArg(_get_child(node, 'fName')).strip()
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    
    # fName 내부에 \u2061 같은 보이지 않는 문자가 이미 제거된 상태
    # 기본 제공되는 표준 수학 함수들 매핑
    # KaTeX-supported function names (render as \funcname)
    standard_funcs = ['sin', 'cos', 'tan', 'csc', 'sec', 'cot', 'sinh', 'cosh', 'tanh', 'coth', 'arcsin', 'arccos', 'arctan', 'lim', 'log', 'ln', 'min', 'max']
    # Functions NOT in KaTeX — must use \operatorname{}
    operatorname_funcs = ['csch', 'sech']

    # fName 정리 ('sin', 'lim'만 추출하기 위함)
    clean_fName = fName.replace(' ', '').replace('{', '').replace('}', '')

    # 특수한 경우: fName 이 첨자를 포함할 때 (예: \lim_{n \to \infty})
    # 이 부분은 fName 자체가 복잡할 수 있으므로, 단순 문자열 변환보다는 기본 매핑
    if clean_fName in operatorname_funcs:
        return f"\\operatorname{{{clean_fName}}} {{{e}}}"
    if clean_fName in standard_funcs:
        return f"\\{clean_fName} {{{e}}}"
    elif fName.startswith('lim') or fName.startswith('max') or fName.startswith('min'):
        # lim 아래에 첨자가 달린 형태가 fName 쪽으로 파싱되어 들어온 경우 대비
        return f"\\mathop{{{fName}}} {{{e}}}"
        
    return f"\\mathop{{{fName}}} {{{e}}}"

def parse_m_CT_GroupChr(node):
    groupChrPr = parse_m_CT_GroupChrPr(_get_child(node, 'groupChrPr'))
    pos = groupChrPr.get('pos')
    chr_val = groupChrPr.get('chr')
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))

    # ECMA-376 shared-math.rnc m_CT_GroupChrPr: chr는 m:CT_Char (단일 문자)
    # parse_m_ST_Char를 거쳤으므로 유니코드 → LaTeX 변환 후 값이 들어올 수 있음
    # 아래는 원본 유니코드와 변환된 LaTeX 명령어 양쪽을 처리
    _under = {
        '\u23df', '\u23dd', '\u23b5',          # ⏟ ⏝ ⏵ (아래 중괄호/괄호/대괄호)
    }
    _over = {
        '\u23de', '\u23dc', '\u23b4', '\u2322', # ⏞ ⏜ ⏴ ⌢ (위 중괄호/괄호/대괄호/호)
    }

    if chr_val:
        if chr_val in _under:
            return f"\\underbrace{{{e}}}"
        if chr_val in _over:
            return f"\\overbrace{{{e}}}"

    # chr 없거나 맵에 없음: pos로 폴백
    if pos == 'bot':
        return f"\\underbrace{{{e}}}"
    return f"\\overbrace{{{e}}}"


def parse_m_CT_LimLow(node):
    limLowPr = parse_m_CT_LimLowPr(_get_child(node, 'limLowPr'))
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    lim = parse_m_CT_OMathArg(_get_child(node, 'lim'))
    return f"\\mathop{{{e}}}\\limits_{{{lim}}}"

def parse_m_CT_LimUpp(node):
    limUppPr = parse_m_CT_LimUppPr(_get_child(node, 'limUppPr'))
    e = parse_m_CT_OMathArg(_get_child(node, 'e'))
    lim = parse_m_CT_OMathArg(_get_child(node, 'lim'))
    return f"\\mathop{{{e}}}\\limits^{{{lim}}}"

def parse_m_CT_Box(node):
    boxPr = parse_m_CT_BoxPr(_get_child(node, 'boxPr'))
    return parse_m_CT_OMathArg(_get_child(node, 'e'))

def parse_m_CT_BorderBox(node): 
    # [IGNORE 설명 참조 (BorderBoxPr 부분)]
    borderBoxPr = parse_m_CT_BorderBoxPr(_get_child(node, 'borderBoxPr'))
    return f"\\boxed{{{parse_m_CT_OMathArg(_get_child(node, 'e'))}}}"

def parse_m_CT_Phant(node): 
    phantPr = parse_m_CT_PhantPr(_get_child(node, 'phantPr'))
    return f"\\phantom{{{parse_m_CT_OMathArg(_get_child(node, 'e'))}}}"

# ==============================================================================
# Roots Definitions
# ==============================================================================
def parse_m_CT_OMath(node):
    return "".join(parse_m_EG_OMathElements(child) for child in node)

def parse_m_CT_OMathPara(node):
    oMathParaPr = parse_m_CT_OMathParaPr(_get_child(node, 'oMathParaPr'))
    content = "".join(parse_m_CT_OMath(omath) for omath in _get_children(node, 'oMath'))
    return f"$$\n{content}\n$$"

def parse_m_CT_MathPr_Root(node):
    return parse_m_CT_MathPr(node)

# ==============================================================================
# 루트 엘리먼트 래퍼  (shared-math.rnc 1:1 대응)
# m_mathPr    = element mathPr    { m_CT_MathPr }
# m_oMathPara = element oMathPara { m_CT_OMathPara }
# m_oMath     = element oMath     { m_CT_OMath }
# ==============================================================================

def parse_m_mathPr(node):
    """m:mathPr 루트 엘리먼트 래퍼 → parse_m_CT_MathPr 위임.
    shared-math.rnc: m_mathPr = element mathPr { m_CT_MathPr }"""
    return parse_m_CT_MathPr_Root(node)

def parse_m_oMathPara(node):
    """m:oMathPara 루트 엘리먼트 래퍼 → parse_m_CT_OMathPara 위임.
    shared-math.rnc: m_oMathPara = element oMathPara { m_CT_OMathPara }"""
    return parse_m_CT_OMathPara(node)

def parse_m_oMath(node):
    """m:oMath 루트 엘리먼트 래퍼 → 인라인 수식 $...$.
    shared-math.rnc: m_oMath = element oMath { m_CT_OMath }
    직접 호출은 항상 인라인 문맥이므로 $...$ 로 감싼다.
    oMathPara 내부에서 oMath 내용을 가져올 때는 parse_m_CT_OMath를 직접 호출한다."""
    return f"${parse_m_CT_OMath(node)}$"

def convert_omml(node: ET.Element) -> str:
    """Convert an OMML root element to a KaTeX-compatible LaTeX string.

    Args:
        node: An ``m:oMath``, ``m:oMathPara``, or ``m:mathPr`` ``ET.Element``.

    Returns:
        LaTeX math string. ``m:oMathPara`` is wrapped in ``$$...$$`` (display),
        ``m:oMath`` in ``$...$`` (inline), and ``m:mathPr`` returns an empty string.
    """
    tag = _get_tag(node)
    if tag == 'mathPr':    return parse_m_CT_MathPr_Root(node)
    if tag == 'oMathPara': return parse_m_CT_OMathPara(node)
    if tag == 'oMath':     return f"${parse_m_CT_OMath(node)}$"
    return ""


if __name__ == "__main__":
    sample_xml = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
        <m:oMath>
            <m:r>
                <m:rPr><m:scr m:val="double-struck"/><m:sty m:val="b"/></m:rPr>
                <m:t>&#x2061;R&#x2062;α&#x221E;</m:t>
            </m:r>
        </m:oMath>
    </m:oMathPara>"""
    import xml.etree.ElementTree as tree
    print(convert_omml(tree.fromstring(sample_xml)))


