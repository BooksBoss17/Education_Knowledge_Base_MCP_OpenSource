"""Organize paginated BeMarkdown content into the textbook-organization-v1 layout."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote
from shutil import copyfile

PAGE = re.compile(r'(?m)^<!-- page:(\d+) -->[ \t]*\r?$')
CHAPTER = re.compile(r'^(第[〇零一二三四五六七八九十百\d]+(?:章|节|单元|篇))\s*(.*)$')
SPECIAL = {'绪论':'INTRODUCTION', '引言':'INTRODUCTION', '课题研究':'APPENDIX','学生实验':'APPENDIX',
           '索引':'BACK_MATTER', '附录':'APPENDIX', '后记':'POSTSCRIPT'}
IMAGE = re.compile(r'!\[(?:\\.|[^\]\\])*\]\(\s*(?:<(?P<angle>[^>\n]+)>|(?P<plain>[^\s)]+))(?:\s+(?:"[^"\n]*"|\'[^\'\n]*\'))?\s*\)')
HEADER = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
TOC = re.compile(r'^(.+?)\s*[.…·\s]+\s*(\d+)\s*$')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read_pages(text, count):
    markers = list(PAGE.finditer(text))
    if [int(m[1]) for m in markers] != list(range(1, count+1)) or text[:markers[0].start()].strip():
        raise ValueError('Physical page anchors do not match the source')
    return [text[m.end():markers[i+1].start() if i+1<count else len(text)].strip('\n')
            for i,m in enumerate(markers)]


def classify(title):
    match = CHAPTER.fullmatch(title)
    if match and not match[1].endswith('节'):
        return 'CHAPTER', match[1]
    for prefix, kind in SPECIAL.items():
        if title == prefix or (prefix=='附录' and title.startswith(prefix)):
            return kind,title
    return None,None


def outline(pages):
    parts = [dict(page=1,line=1,title='封面与目录',kind='FRONT_MATTER')]
    toc = {}
    seen = set()
    for number,body in enumerate(pages,1):
        lines = body.splitlines()
        in_contents = (any(re.fullmatch(r'[#\s]*(目录|Contents)\s*',l,re.I) for l in lines)
                       or sum(bool(re.search(r'[.…·]{2,}\s*\d+\s*$',l)) for l in lines)>=3)
        for i,line in enumerate(lines):
            heading = HEADER.match(line.strip())
            title = (heading[2] if heading else line).strip()
            entry = TOC.fullmatch(title)
            if entry:
                kind,key = classify(entry[1].strip())
                if kind:
                    toc.setdefault(key,entry[1].strip())
                    continue
            kind,key = classify(title)
            if in_contents:
                if kind:toc.setdefault(key,title)
                continue
            if not kind or key in seen or (not heading and title not in toc.values()):
                continue
            if number==1 and i==0:
                continue
            full = toc.get(key,title) if title==key else title
            if full==key and kind=='CHAPTER':
                following=next((x.strip().lstrip('#').strip() for x in lines[i+1:] if x.strip()),'')
                if following and len(following)<28 and not following.startswith(('!', '<')):
                    full=key+' '+following
            seen.add(key)
            start=1 if all(not l.strip() or IMAGE.fullmatch(l.strip()) for l in lines[:i]) else i+1
            parts.append(dict(page=number,line=start,title=full,kind=kind))
    return parts,sorted(set(toc)-seen)


def safe_name(name):
    value=re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')
    if not value or len(value)>160 or value.upper().split('.')[0] in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
        raise ValueError('Unsupported book or section name')
    return value


def asset_path(ref, roots):
    name=Path(unquote(ref))
    if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] not in ('assets','evidence','DIAGRAMS','_bemarkdown'):
        raise ValueError('Unsupported image location: '+ref)
    for root in roots:
        path=(root/name).resolve()
        if path.is_relative_to(root) and path.is_file():
            return path
    raise ValueError('Missing referenced image: '+ref)


def normalize_math(lines):
    """Pair within a paragraph or an intact display block; isolate damaged source."""
    output=list(lines)
    problems=[]
    i=0
    fence=None
    while i<len(lines):
        line=lines[i]
        stripped=line.strip()
        if stripped.startswith(('~~~',chr(96)*3)):
            marker=stripped[0]
            fence=None if marker==fence else marker
            i+=1
            continue
        if fence or '$' not in line:
            i+=1
            continue
        if stripped=='$$':
            end=i+1
            while end<len(lines) and lines[end].strip() and '$' not in lines[end]:
                end+=1
            if end>i+1 and end<len(lines) and lines[end].strip()=='$$':
                output[i],output[end]=r'\[',r'\]'
                i=end+1
                continue
            problems.append(i+1);i+=1;continue
        # Inline code and Markdown image syntax are opaque to math rewriting.
        opaque=[m.span() for m in IMAGE.finditer(line)]
        opaque.extend(m.span() for m in re.finditer(r'(`+).*?\1',line))
        tokens=[m for m in re.finditer(r'(?<!\\)\$+',line)
                if not any(a<=m.start()<b for a,b in opaque)]
        if len(tokens)%2 or any(a.group()!=b.group() or len(a.group())>2
                                for a,b in zip(tokens[::2],tokens[1::2])):
            problems.append(i+1)
        else:
            for a,b in reversed(list(zip(tokens[::2],tokens[1::2]))):
                left,right=(r'\[',r'\]') if a.group()=='$$' else (r'\(',r'\)')
                line=line[:a.start()]+left+line[a.end():b.start()]+right+line[b.end():]
            output[i]=line
        i+=1
    return output,problems


def render(pages, parts, source_name, source_sha, roots, output, approved):
    if not parts or (parts[0]['page'],parts[0]['line'],parts[0]['kind'])!=(1,1,'FRONT_MATTER'):
        raise ValueError('First part must start on the first line of page 1')
    if sum(p['kind']=='FRONT_MATTER' for p in parts)!=1:
        raise ValueError('Expected one front matter part')
    source_lines=[]
    starts={}
    for page,body in enumerate(pages,1):
        normalized,problems=normalize_math(body.split('\n'))
        if problems and approved:
            raise ValueError(f'Math needs source review on page {page}, lines {problems[:5]}')
        for line_no,line in enumerate(normalized,1):
            starts[(page,line_no)]=len(source_lines)
            source_lines.append((page,line))
    bounds=[starts[(part['page'],part['line'])] for part in parts]
    if sorted(set(bounds))!=bounds:
        raise ValueError('Section boundaries must be strictly increasing')
    allowed={'FRONT_MATTER','INTRODUCTION','CHAPTER','APPENDIX','POSTSCRIPT','BACK_MATTER','OTHER'}
    diagrams=output/'DIAGRAMS'
    diagrams.mkdir(parents=True)
    artifacts={}; sources={}; sections=[]
    for i,part in enumerate(parts):
        if part['kind'] not in allowed:
            raise ValueError('Unknown section kind')
        chunk=source_lines[bounds[i]:bounds[i+1] if i+1<len(parts) else None]
        title=part['title']
        name=f'{i:02d}-{safe_name(title)}.md'
        body=[]; fence=None
        opening=True
        title_tail=None
        for source_page,line in chunk:
            if opening and line.strip() and not IMAGE.fullmatch(line.strip()):
                label=line.strip().lstrip('#').strip()
                if label==title:
                    opening=False
                    continue
                if CHAPTER.fullmatch(label) and title.startswith(label+' '):
                    title_tail=title[len(label):].strip()
                    opening=False
                    continue
                opening=False
            if title_tail and line.strip():
                expected=title_tail;title_tail=None
                if line.strip().lstrip('#').strip()==expected:continue
            if re.match(r'^\s*('+'~~~|'+chr(96)+'{3,})',line):
                delimiter=line.strip()[0]
                fence=None if fence==delimiter else delimiter
                body.append(line);continue
            if fence:
                body.append(line);continue
            def image_link(m):
                key=m['angle'] or m['plain']
                if key not in sources:
                    source=asset_path(key,roots)
                    uid=sha(source); file=uid+source.suffix.lower()
                    dest=diagrams/file
                    if not dest.exists():
                        copyfile(source,dest)
                        if sha(dest)!=uid:raise ValueError('Image changed during copy')
                    sources[key]=file;artifacts['DIAGRAMS/'+file]=uid
                group='angle' if m['angle'] else 'plain'
                left=m.start(group)-m.start();right=m.end(group)-m.start()
                return m[0][:left]+'DIAGRAMS/'+sources[key]+m[0][right:]
            if line.count('![')!=len(list(IMAGE.finditer(line))):
                raise ValueError('Unparsed image reference on source page '+str(source_page))
            line=IMAGE.sub(image_link,line)
            h=HEADER.match(line)
            if h:
                sub=h[2]
                if not body and (sub==title or (CHAPTER.fullmatch(sub) and title.startswith(sub+' '))):
                    continue
                level=2 if re.match(r'^(?:第.+节|\d+[.、\s])',sub) else min(4,max(2,len(h[1])))
                line='#'*level+' '+sub
            body.append(line)
        head=dict(schema_version='textbook-organization-v1',book_id=source_sha[:16],
                  source_filename=source_name,source_sha256=source_sha,title=title,
                  part_type=part['kind'],part_order=i,pdf_page_start=chunk[0][0],
                  pdf_page_end=chunk[-1][0],review_status='AGENT_REVIEWED' if approved else 'DRAFT')
        front='---\n'+''.join(f'{k}: {json.dumps(v,ensure_ascii=False)}\n' for k,v in head.items())+'---\n\n'
        path=output/name;path.write_text(front+'# '+title+'\n\n'+'\n'.join(body).strip()+'\n','utf8')
        artifacts[name]=sha(path)
        sections.append(dict(file=name,start=bounds[i],end=bounds[i]+len(chunk),page_start=chunk[0][0],page_end=chunk[-1][0]))
    if not any(p['kind']=='POSTSCRIPT' for p in parts):
        name=f'{len(parts):02d}-后记.md'
        head.update(title='后记',part_type='POSTSCRIPT',part_order=len(parts),pdf_page_start=None,pdf_page_end=None)
        (output/name).write_text('---\n'+''.join(f'{k}: {json.dumps(v,ensure_ascii=False)}\n' for k,v in head.items())+'---\n\n# 后记\n\n<!-- 组织元数据 -->\n原件未包含后记。\n','utf8')
        artifacts[name]=sha(output/name)
        sections.append(dict(file=name,start=len(source_lines),end=len(source_lines),page_start=None,
                             page_end=None,organization_metadata=True))
    return dict(files=artifacts,sections=sections,asset_count=len(artifacts)-len(parts)-(not any(p['kind']=='POSTSCRIPT' for p in parts)))

