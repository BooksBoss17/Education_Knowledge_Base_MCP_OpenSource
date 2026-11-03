"""Remove isolated empty image markup from an explicitly unverified hypothesis."""
import hashlib,re
from html.parser import HTMLParser
from markdown_it import MarkdownIt
from .markdown_asset_refs import image_references
from .pdf.adaptive_ocr import anomalies

VERSION='native-pending-reference-only-salvage-v1'
def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

class _OnlyImage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.tags=[];self.bad=False
    def handle_starttag(self,tag,attrs):
        self.tags.append((tag,attrs,self.get_starttag_text()))
    def handle_startendtag(self,tag,attrs):self.handle_starttag(tag,attrs)
    def handle_endtag(self,tag):self.bad=True
    def handle_data(self,data):self.bad=self.bad or bool(data.strip())
    def handle_comment(self,data):self.bad=True
    def handle_decl(self,decl):self.bad=True
    def unknown_decl(self,data):self.bad=True

def _empty_image_tag(text):
    if not re.fullmatch(r'''<img(?:\s+(?:src|alt|title)\s*=\s*(?:"[^"]*"|'[^']*'))+\s*/?>''',text,re.I):return False
    parser=_OnlyImage()
    try:parser.feed(text);parser.close()
    except (ValueError,AssertionError):return False
    if parser.bad or len(parser.tags)!=1:return False
    tag,attrs,raw=parser.tags[0]
    names=[k for k,_ in attrs];values=dict(attrs)
    return (tag=='img' and raw==text and len(names)==len(set(names))
        and set(names)<={'src','alt','title'} and bool(values.get('src'))
        and not str(values.get('alt') or '').strip()
        and not str(values.get('title') or '').strip())

def pending_reference_text(raw,status,source_sha):
    """Never validates text accuracy, source coverage, or an OCR candidate vote."""
    if (not isinstance(source_sha,str) or len(source_sha)!=64
            or any(c not in '0123456789abcdef' for c in source_sha)):
        return None
    if anomalies(raw,status)!=('UNREQUESTED_IMAGE_REFERENCE',):return None
    parser=MarkdownIt('commonmark').enable('table')
    tokens=parser.parse(raw)
    if any(t.type in {'blockquote_open','bullet_list_open','ordered_list_open','table_open'} for t in tokens):return None
    spans=[];offsets=[0]
    for line in raw.splitlines(keepends=True):offsets.append(offsets[-1]+len(line))
    blocked=[];html_ranges=[];image_paragraph_ranges=[]
    for token_index,token in enumerate(tokens):
        if token.type in {'fence','code_block'} and token.map:
            blocked.append((offsets[token.map[0]],offsets[token.map[1]]))
        if token.type=='html_block':
            if not _empty_image_tag(token.content.strip()):return None
            if token.map:html_ranges.append((offsets[token.map[0]],offsets[token.map[1]]))
        if (token.type=='inline' and token.map and token.level==1 and token_index>0
                and token_index+1<len(tokens) and tokens[token_index-1].type=='paragraph_open'
                and tokens[token_index+1].type=='paragraph_close'):
            children=[c for c in token.children or [] if c.type not in {'softbreak','hardbreak'}]
            if len(children)==1 and children[0].type=='image':
                image_paragraph_ranges.append((offsets[token.map[0]],offsets[token.map[1]]))
    try:refs=image_references(raw)
    except ValueError:return None
    for ref in refs:
        markup=ref.group()
        if not markup.startswith('![](') or '\n' in markup or (ref.title or '').strip():return None
        if not any(a<=ref.start()<ref.end()<=b for a,b in image_paragraph_ranges):return None
        begin=raw.rfind('\n',0,ref.start())+1
        end=raw.find('\n',ref.end());end=len(raw) if end<0 else end
        if raw[begin:ref.start()].strip() or raw[ref.end():end].strip():return None
        spans.append((ref.start(),ref.end(),'MARKDOWN_EMPTY_IMAGE'))
    position=0
    for line in raw.splitlines(keepends=True):
        stripped=line.strip();start=position+len(line)-len(line.lstrip())
        if stripped.lower().startswith('<img'):
            if any(a<=start<b for a,b in blocked) or not _empty_image_tag(stripped):return None
            if not any(a<=start<start+len(stripped)<=b for a,b in html_ranges):return None
            spans.append((start,start+len(stripped),'HTML_EMPTY_IMAGE'))
        position+=len(line)
    spans=sorted(spans)
    if not spans or any(b>c for (_,b,_),(c,_,_) in zip(spans,spans[1:])):return None
    for start,_,_ in spans:
        math_stack=[]
        for marker in re.findall(r'(?<!\\)(?:\$\$|\$|\\\[|\\\])',raw[:start]):
            if marker==r'\]':
                if not math_stack or math_stack[-1]!=r'\[':return None
                math_stack.pop()
            elif math_stack and marker==math_stack[-1] and marker in {'$','$$'}:math_stack.pop()
            else:math_stack.append(marker)
        if math_stack:return None
    cleaned=raw
    for a,b,_ in reversed(spans):cleaned=cleaned[:a]+cleaned[b:]
    if not cleaned.strip() or anomalies(cleaned,'PASS'):return None
    return {'version':VERSION,'text':cleaned,'source_input_sha256':source_sha,
        'raw_text':raw,'raw_sha256':digest(raw),'cleaned_sha256':digest(cleaned),
        'removed_spans':[{'character_range':[a,b],'utf8_byte_range':[len(raw[:a].encode('utf-8')),len(raw[:b].encode('utf-8'))],
            'markup':raw[a:b],'markup_sha256':digest(raw[a:b]),'kind':kind} for a,b,kind in spans],
        'text_correctness_certified':False,'coverage_certified':False,'candidate_voting_allowed':False,
        'new_model_calls':0}
