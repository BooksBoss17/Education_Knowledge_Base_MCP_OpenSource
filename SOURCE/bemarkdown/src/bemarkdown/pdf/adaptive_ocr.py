"""Gold-free three-recognizer routing. Never solve or rewrite source mathematics."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping

POLICY_VERSION = 'bemarkdown-pp-ovis-xiaomi-v3'
_SYMBOLS = {'times':'×','cdot':'·','mu':'μ','Delta':'Δ','lambda':'λ','pi':'π',
            'alpha':'α','beta':'β','gamma':'γ','theta':'θ','rho':'ρ','Omega':'Ω',
            'le':'≤','leq':'≤','ge':'≥','geq':'≥','ne':'≠','neq':'≠','pm':'±',
            'ldots':'…','dots':'…','circ':'°','sim':'～','infty':'∞'}
_SUP = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ⁿⁱ','0123456789+-()ni'))
_SUB = dict(zip('₀₁₂₃₄₅₆₇₈₉₊₋₍₎','0123456789+-()'))
_BRACKETS = str.maketrans('（）［］，；：！？µ∆−','()[],;:!?μΔ-')

def canonical(text: str) -> str:
    """Presentation comparison retaining fraction operands and semantic accents.

    This is not a math solver. Unknown commands remain visible; parentheses are
    never removed. FRAC/ACCENT markers cannot collapse numerator/denominator
    boundaries as the former linear slash normalization did.
    """
    text = unicodedata.normalize('NFC', str(text)).translate(_BRACKETS)
    text = re.sub(r'(?m)^\s{0,3}#{1,6}\s+', '', text)
    text = re.sub(r'(?m)^\s*```(?:latex|tex|markdown)?\s*$', '', text)
    text = re.sub(r'\\(?:begin|end)\{doc\}', '', text)
    text = re.sub(r'\\[\[\]() ]|\$+', '', text)
    text = re.sub(r'\\(?:left|right)\b', '', text)
    # Newline commands and explicit typographical glue carry no visible glyph.
    text = re.sub(r'\\\\|\\[,;:!]|\\(?:quad|qquad|thinspace|enspace)\b', '', text)

    def parse(pos: int, stop: bool = False) -> tuple[str, int]:
        out: list[str] = []
        while pos < len(text):
            c = text[pos]
            if c == '}' and stop:
                return ''.join(out), pos + 1
            if c.isspace():
                pos += 1; continue
            if c == '{':
                value, pos = parse(pos + 1, True)
                out.append(value); continue
            if c in '^_':
                value, pos = operand(pos + 1)
                out.append(c + '{' + value + '}'); continue
            if c in _SUP or c in _SUB:
                mapping = _SUP if c in _SUP else _SUB
                marker = '^' if c in _SUP else '_';value = ''
                while pos < len(text) and text[pos] in mapping:
                    value += mapping[text[pos]];pos += 1
                out.append(marker + '{' + value + '}');continue
            if c == '\\':
                command = re.match(r'\\([A-Za-z]+)', text[pos:])
                if command:
                    name = command.group(1);pos += len(command.group())
                    if name in {'frac','dfrac','tfrac'}:
                        a,pos = operand(pos);b,pos = operand(pos)
                        out.append('FRAC(' + a + ')(' + b + ')');continue
                    if name in {'mathrm','textrm','text','mathit','operatorname'}:
                        value,pos = operand(pos);out.append(value);continue
                    if name in {'bar','overline','vec','hat','widehat','sqrt'}:
                        value,pos = operand(pos)
                        semantic = 'bar' if name == 'overline' else 'hat' if name == 'widehat' else name
                        out.append('ACCENT[' + semantic + '](' + value + ')');continue
                    out.append(_SYMBOLS.get(name, '\\' + name));continue
                if pos + 1 < len(text) and text[pos+1] in '%&#_':
                    out.append(text[pos+1]);pos += 2;continue
            # Fold only mathematical italic alphabets, not all Unicode semantics.
            out.append(unicodedata.normalize('NFKC',c) if unicodedata.name(c,'').startswith('MATHEMATICAL ITALIC ') else c)
            pos += 1
        return ''.join(out),pos

    def operand(pos: int) -> tuple[str,int]:
        while pos < len(text) and text[pos].isspace():pos += 1
        if pos >= len(text):return 'MISSING',pos
        if text[pos] == '{':return parse(pos+1,True)
        return text[pos],pos+1
    return parse(0)[0]

def _plain(text: str) -> str:
    # Greek and other Unicode letters are content, never punctuation.
    return ''.join(c for c in canonical(text) if c.isalnum())

def punctuation_only_difference(a: str, b: str) -> bool:
    allowed=set('()[]，,。.;；:：!！?？“”‘’\"\'、…')
    if not re.search(r'[\u4e00-\u9fff]',a+b):return False
    if re.search(r'[=+×÷<>≤≥∫∑√]|[\u0370-\u03ff]',a+b):return False
    for op,i,j,k,l in SequenceMatcher(None,a,b).get_opcodes():
        if op!='equal' and any(c not in allowed for c in a[i:j]+b[k:l]):return False
    return True

def anomalies(text: str, status: str = 'OK') -> tuple[str,...]:
    reasons=[]
    if status not in {'OK','PASS'}:reasons.append('NON_OK_OUTPUT')
    if not text.strip():reasons.append('EMPTY_OUTPUT')
    if any(s in text.lower() for s in ['\\begin{tikz','\\begin{document','i cannot','i am unable','无法识别图像']):
        reasons.append('UNREQUESTED_GENERATED_CONTENT')
    if re.search(r'(.{12,80})\1{3,}',text,re.S):reasons.append('REPEATED_OUTPUT')
    stack=[]
    for c in re.sub(r'\\[{}]', '', text):
        if c=='{':stack.append(c)
        elif c=='}':
            if stack:stack.pop()
            else:reasons.append('UNBALANCED_BRACES');break
    if stack:reasons.append('UNBALANCED_BRACES')
    if 'MISSING' in canonical(text):reasons.append('MISSING_MATH_OPERAND')
    return tuple(dict.fromkeys(reasons))

@dataclass(frozen=True)
class RoutingPolicy:
    version: str = POLICY_VERSION
    pp_fast_path: bool = False
    pp_min_confidence: float = 0.995
    max_easy_text_height: int = 64
    # A mismatch triggers a candidate; it never proves which prediction is right.
    pp_secondary_min_characters: int = 4
    review_all_disagreement: bool = True
    secondary_scope: str = 'bounded_punctuation_or_anomaly'

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def trigger(self, pp: Mapping[str,Any], ovis: Mapping[str,Any], image: Mapping[str,Any]) -> tuple[str,...]:
        reasons=list(anomalies(str(ovis.get('text','')),str(ovis.get('status','OK'))))
        a,b=_plain(str(pp.get('text',''))),_plain(str(ovis.get('text','')))
        if self.secondary_scope=='bounded_punctuation_or_anomaly':
            ac,bc=canonical(str(pp.get('text',''))),canonical(str(ovis.get('text','')))
            structure_free=not any(t in ac+bc for t in ('FRAC(','ACCENT[','^','_','\\'))
            if a==b and ac!=bc and structure_free and abs(len(ac)-len(bc))<=3 and punctuation_only_difference(ac,bc):
                reasons.append('BOUNDED_PUNCTUATION_DISAGREEMENT')
            return tuple(dict.fromkeys(reasons))
        if self.secondary_scope!='all_pp_disagreement':
            raise ValueError('UNKNOWN_SECONDARY_SCOPE')
        if len(a)>=self.pp_secondary_min_characters and a != b:
            reasons.append('PP_TEXT_DISAGREEMENT')
        if canonical(str(pp.get('text',''))) != canonical(str(ovis.get('text',''))) and len(a)<4:
            reasons.append('SHORT_LABEL_DISAGREEMENT')
        return tuple(dict.fromkeys(reasons))

    def easy_text(self, pp: Mapping[str,Any], image: Mapping[str,Any]) -> bool:
        # Production disabled unless a separate calibration artifact enables it.
        if not self.pp_fast_path:return False
        text=str(pp.get('text',''))
        return (not anomalies(text,str(pp.get('status','OK'))) and
                float(pp.get('confidence') or 0)>=self.pp_min_confidence and
                int(image.get('height') or 0)<=self.max_easy_text_height and
                bool(re.fullmatch(r'[\u4e00-\u9fff，。！？、；：“”‘’（）\s]{6,}',text)))

    def select(self, pp: Mapping[str,Any], ovis: Mapping[str,Any], xiaomi: Mapping[str,Any] | None) -> dict[str,Any]:
        """Ovis is retained on unresolved disagreement; no majority hallucination."""
        selected='OvisOCR2';text=str(ovis.get('text',''));reasons=[]
        bad=anomalies(text,str(ovis.get('status','OK')))
        if xiaomi is None and _plain(str(pp.get('text',''))) != _plain(text) and len(_plain(str(pp.get('text',''))))>=4:
            reasons.append('PP_LEXICAL_DISAGREEMENT_REVIEW')
        if xiaomi is not None:
            xt=str(xiaomi.get('text',''));xbad=anomalies(xt,str(xiaomi.get('status','OK')))
            if not xbad and bad:
                selected='Xiaomi-OCR-0';text=xt;reasons.append('PRIMARY_ANOMALY_RECOVERED')
            elif not xbad and canonical(xt)==canonical(text):
                reasons.append('OVIS_XIAOMI_PRESENTATION_AGREEMENT')
            elif not xbad and canonical(xt)==canonical(str(pp.get('text',''))) and not bad:
                xc,oc=canonical(xt),canonical(text)
                # Only restore a small punctuation difference corroborated by
                # the independent line recognizer. Never alter lexical content
                # or a visible mathematical structure on a two-vote basis.
                plain_same=_plain(xt)==_plain(text)
                math_free=not any(token in xc+oc for token in ('FRAC(', 'ACCENT[', '^', '_', '\\'))
                small_change=abs(len(xc)-len(oc))<=3 and SequenceMatcher(None,xc,oc).ratio()>=0.75
                if plain_same and math_free and small_change and punctuation_only_difference(xc,oc):
                    selected='Xiaomi-OCR-0';text=xt
                    reasons.append('LEXICAL_IDENTICAL_PUNCTUATION_CORROBORATED')
                else:
                # Corroboration may flag risk but cannot overwrite mathematical
                # structures on the authority of a line recognizer alone.
                    reasons.append('SECONDARY_PP_AGREEMENT_REVIEW')
            else:reasons.append('MODEL_DISAGREEMENT_REVIEW')
        if bad and selected=='OvisOCR2':reasons.extend(bad)
        review=any(r.endswith('REVIEW') for r in reasons) or (bool(bad) and selected=='OvisOCR2')
        return {'selected_model':selected,'selected_text':text,'review_required':review,
                'review_reasons':reasons,'policy_fingerprint':self.fingerprint()}
