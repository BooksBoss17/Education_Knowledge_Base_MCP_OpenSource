"""Source-preserving textbook organization after BeMarkdown review."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote
import uuid

IMAGE = re.compile(r'!\[(?:\\.|[^\]\\])*\]\((?:<[^>\n]+>|[^\n)])+\)')
DEST = re.compile(r'''\s*(?:<([^>]+)>|(\S+?))(?:\s+(?:"[^"]*"|'[^']*'))?\s*''')
KINDS = {'front', 'introduction', 'chapter', 'appendix', 'afterword'}

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def slug(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError('Expected a nonempty title of at most 160 characters')
    result = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '-', value).strip(' .')[:90]
    if not result or result.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1,10)), *(f'LPT{i}' for i in range(1,10))}:
        raise ValueError('Invalid filesystem title')
    return result

def backup(service, source):
    """Copy before conversion; preserve same-named editions under content IDs."""
    identity = digest(source)
    folder = service.workspace_manager.safe_path('Original_Backup/TEXTBOOKS')
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (slug(source.stem) + '__' + identity[:12] + source.suffix.lower())
    if target.exists():
        if digest(target) != identity:
            raise ValueError('Textbook backup collision; existing source is preserved')
    else:
        temporary = folder / ('.' + uuid.uuid4().hex + '.copying')
        shutil.copyfile(source, temporary)
        if digest(temporary) != identity or digest(source) != identity:
            raise ValueError('Source changed while creating textbook backup')
        # A concurrent identical import may already have created this target.
        if target.exists() and digest(target) != identity:
            raise ValueError('Textbook backup collision')
        os.replace(temporary, target)
    return target, identity

def skill(service):
    path = service.mcp_root / 'SKILLS/textbook-import/SKILL.md'
    return dict(name='textbook-import', path=str(path), content=path.read_text(encoding='utf-8'))

def current(service, job_id):
    state = service.state(job_id)
    if state.get('material_type') != 'textbook':
        raise ValueError('Start this import with material_type=textbook')
    package = service.package(job_id)
    source = Path(state['source']).resolve(strict=True)
    if not source.is_relative_to(service.workspace_manager.safe_path('Original_Backup/TEXTBOOKS')) or digest(source) != state['source_sha256']:
        raise ValueError('Textbook backup identity changed')
    path = package / 'document.reviewed.md'
    if not path.exists():
        path = package / 'document.md'
    return state, package, path, path.read_text(encoding='utf-8')

def images(package, text):
    result = []
    for number, match in enumerate(IMAGE.finditer(text), 1):
        raw = match.group()
        # The first ]( belongs to the inline image syntax; optional titles remain supported.
        destination = DEST.fullmatch(raw[raw.index('](') + 2:-1])
        if destination is None:
            raise ValueError('Unsupported image reference: ' + raw[:120])
        name = unquote(destination.group(1) or destination.group(2))
        target = (package / name).resolve(strict=True)
        if not target.is_relative_to(package) or not target.is_file():
            raise ValueError('Image escapes intermediate package')
        from PIL import Image
        with Image.open(target) as picture:
            width, height = picture.size
        result.append(dict(image_id=f'image-{number:05d}', line=text.count('\n', 0, match.start())+1,
                           reference=raw, asset_name=name, width=width, height=height,
                           start=match.start(), end=match.end(), sha256=digest(target)))
    if len(result) != len(re.findall(r'!\[', text)):
        raise ValueError('Unsupported image syntax requires source-based normalization before publication')
    return result

def source_requests(state, package, found):
    """Attach recorded asset provenance; do not infer a source page from image order."""
    path=package/'assets_manifest.jsonl'
    if not path.is_file():
        return
    records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_name={row['relative_path']:row for row in records if row.get('relative_path')}
    pdf=None
    if Path(state['source']).suffix.lower()=='.pdf':
        import fitz
        pdf=fitz.open(state['source'])
    try:
        for row in found:
            record=by_name.get(row['asset_name'])
            if not record:
                continue
            row['source_part']=record.get('source_part')
            part=record.get('source_part') or ''
            if pdf is not None and re.fullmatch(r'page:\d+',part):
                page=int(part.split(':')[1])
                if not 0 <= page < len(pdf):
                    continue
                request=dict(job_id=state['job_id'],kind='image',page=page+1,dpi=288)
                box=record.get('provenance',{}).get('bbox_pdf_pt')
                if isinstance(box,list) and len(box)==4:
                    rect=pdf[page].rect
                    region=[max(0.,min(1.,(box[i]-(rect.x0 if i%2==0 else rect.y0))/(rect.width if i%2==0 else rect.height))) for i in range(4)]
                    if region[0]<region[2] and region[1]<region[3]:
                        request['region']=region
                row['source_request']=request
    finally:
        if pdf is not None:
            pdf.close()

def inspect(service, job_id, offset=0, limit=100):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 500:
        raise ValueError('Invalid inventory window')
    state, package, path, text = current(service, job_id)
    lines = text.splitlines(keepends=True)
    found = images(package, text)
    source_requests(state, package, found)
    headings = [dict(line=i, text=line.strip()[:250]) for i, line in enumerate(lines, 1)
                if re.match(r'^\s*(?:#{1,6}\s|第[一二三四五六七八九十百\d]+[章节编篇]|目\s*录|后\s*记|前\s*言|绪\s*论|附\s*录)', line)]
    return dict(job_id=job_id, title=state.get('book_title', Path(state['source']).stem),
                base_sha256=digest(path), total_lines=len(lines), total_images=len(found),
                headings=headings[offset:offset+limit], headings_total=len(headings),
                images=[{k:v for k,v in row.items() if k not in ('start','end')} for row in found[offset:offset+limit]],
                next_offset=offset+limit if offset+limit < max(len(found),len(headings)) else None,
                backup=str(state['source']), skill=skill(service),
                review_views=['handoff','issues'], note='Inventory is evidence for planning, not automatic semantic classification.')

def contact_sheet(service, job_id, offset=0, limit=12):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 24:
        raise ValueError('Contact sheet accepts 1..24 images')
    from PIL import Image, ImageDraw
    _, package, _, text = current(service, job_id)
    found = images(package, text)
    rows = found[offset:offset+limit]
    if not rows:
        raise ValueError('No images at this offset')
    canvas = Image.new('RGB', (1200, 260*((len(rows)+2)//3)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x, y = (i%3)*400, (i//3)*260
        with Image.open(package / row['asset_name']) as original:
            picture = original.convert('RGB')
            picture.thumbnail((380,220))
            canvas.paste(picture, (x+(400-picture.width)//2,y+28))
        draw.text((x+8,y+5), f"{row['image_id']} L{row['line']} {row['width']}x{row['height']}", fill='black')
    buffer = io.BytesIO()
    canvas.save(buffer, format='PNG')
    return dict(total_images=len(found), next_offset=offset+len(rows) if offset+len(rows)<len(found) else None,
                note='Output-asset thumbnails. Zoom individual assets and verify original pages before removing or transcribing.'), buffer.getvalue()

def organize(service, job_id, action='inspect', base_sha256=None, plan=None, offset=0, limit=100):
    if action == 'inspect':
        return inspect(service, job_id, offset, limit)
    if action == 'lines':
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError('Invalid line window')
        _, _, path, text = current(service, job_id)
        lines = text.splitlines()
        return dict(base_sha256=digest(path), lines=[{'line':i+1,'text':lines[i]} for i in range(offset,min(len(lines),offset+limit))],
                    total_lines=len(lines),next_offset=offset+limit if offset+limit<len(lines) else None)
    if action not in {'preview','publish'}:
        raise ValueError('Use inspect, contact_sheet, preview or publish')
    state, package, path, text = current(service, job_id)
    if digest(path) != base_sha256:
        raise ValueError('Reviewed Markdown changed; inspect and rebuild the plan')
    if not isinstance(plan, dict) or set(plan) != {'title','sections','image_actions','review'}:
        raise ValueError('Plan requires title, sections, image_actions and review')
    review = plan['review']
    if not isinstance(review, dict) or not review.get('source_evidence') or type(review.get('images_checked')) is not bool or not isinstance(review.get('unresolved'), list):
        raise ValueError('Review requires source_evidence, images_checked boolean, unresolved list')
    if action == 'publish' and not review['images_checked']:
        raise ValueError('Visual image review must be complete before knowledge-base publication')
    for issue in review['unresolved']:
        if not isinstance(issue, dict) or issue.get('severity') not in {'minor','major'} or not issue.get('description'):
            raise ValueError('Unresolved issue requires severity minor/major and description')
    if action == 'publish' and any(i['severity']=='major' for i in review['unresolved']):
        raise ValueError('Major unresolved content issues prevent knowledge-base publication')
    title = slug(plan['title'])
    lines = text.splitlines(keepends=True)
    sections = plan['sections']
    if not isinstance(sections, list) or not sections or len(sections)>150:
        raise ValueError('Expected ordered sections covering the whole document')
    starts = []
    for index, section in enumerate(sections):
        if set(section) != {'title','kind','start_line','source_evidence'} or section['kind'] not in KINDS or not section['source_evidence']:
            raise ValueError('Each section needs title, kind, start_line and original-source evidence')
        slug(section['title'])
        start = section['start_line']
        if type(start) is not int or not 1 <= start <= len(lines) or (starts and start<=starts[-1]):
            raise ValueError('Section boundaries must be increasing valid line numbers')
        starts.append(start)
    if starts[0] != 1 or sections[0]['kind'] != 'front' or not any(s['kind']=='chapter' for s in sections):
        raise ValueError('Start with front matter at line 1 and include actual chapters')
    if sum(s['kind']=='front' for s in sections)!=1 or sum(s['kind']=='afterword' for s in sections)>1:
        raise ValueError('Duplicate front matter or afterword')
    if any(s['kind']=='afterword' for s in sections[:-1]):
        raise ValueError('Afterword must be the last section')
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1]+len(line))
    found = images(package,text)
    by_id = {r['image_id']:r for r in found}
    actions = {}
    if not isinstance(plan['image_actions'], list):
        raise ValueError('image_actions must be a list; omitted images are preserved')
    for row in plan['image_actions']:
        if set(row) != {'image_id','action','text','source_evidence'} or row['image_id'] not in by_id or row['image_id'] in actions or not row['source_evidence']:
            raise ValueError('Each image action needs a unique current image_id, action, text and source evidence')
        if row['action'] not in {'remove_decoration','transcribe_heading'}:
            raise ValueError('Only confirmed decoration removal or heading transcription is supported')
        if not isinstance(row['text'],str) or (row['action']=='remove_decoration' and row['text']) or (row['action']=='transcribe_heading' and (not row['text'].strip() or '\n' in row['text'] or len(row['text'])>500 or '![' in row['text'])):
            raise ValueError('Invalid image replacement text')
        actions[row['image_id']] = row
    book_id = title+'__'+state['source_sha256'][:12]
    root = service.workspace_manager.safe_path('KNOWLEDGE_BASE/TEXTBOOKS')
    target = root / book_id
    diagrams = service.workspace_manager.safe_path('KNOWLEDGE_BASE/TEXTBOOKS/DIAGRAMS') / book_id
    plan_id = hashlib.sha256(json.dumps(dict(base=base_sha256,plan=plan),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    work = service.workspace_manager.safe_path('tmp/textbook-imports') / plan_id[:16]
    staged_book, staged_images = work/book_id, work/'DIAGRAMS'/book_id
    staged_book.mkdir(parents=True,exist_ok=True)
    staged_images.mkdir(parents=True,exist_ok=True)
    patches = []
    image_records = []
    for row in found:
        decision = actions.get(row['image_id'])
        if decision:
            replacement = decision['text']
        else:
            source = package / row['asset_name']
            name = row['sha256'][:24] + source.suffix.lower()
            destination = staged_images / name
            if not destination.exists():
                shutil.copyfile(source,destination)
            assert digest(destination)==row['sha256']
            raw = row['reference']
            prefix=raw.index('](')+2
            destination_match=DEST.fullmatch(raw[prefix:-1])
            group=1 if destination_match.group(1) is not None else 2
            link='../DIAGRAMS/' + quote(book_id,safe='') + '/' + name
            replacement=raw[:prefix+destination_match.start(group)]+link+raw[prefix+destination_match.end(group):]
        patches.append((row['start'],row['end'],replacement))
        image_records.append(dict(image_id=row['image_id'], sha256=row['sha256'], action=decision['action'] if decision else 'preserve', source_evidence=decision.get('source_evidence') if decision else None))
    outputs = []
    for index, section in enumerate(sections):
        begin = offsets[section['start_line']-1]
        end = offsets[starts[index+1]-1] if index+1<len(starts) else len(text)
        body = text[begin:end]
        for left,right,replacement in reversed(patches):
            if left < end and right > begin:
                if left < begin or right > end:
                    raise ValueError('Section boundary splits an image reference')
                body=body[:left-begin]+replacement+body[right-begin:]
        filename=f"{index:02d}-{slug(section['title'])}.md"
        if not body.strip():
            raise ValueError('Section is empty after image actions')
        header='---\n'+ '\n'.join(f'{k}: {json.dumps(v,ensure_ascii=False)}' for k,v in {
            'title':plan['title']+' - '+section['title'], 'type':'textbook',
            'source':os.path.relpath(state['source'],target).replace('\\','/'),
            'source_sha256':state['source_sha256']}.items())+'\n---\n\n'
        output=staged_book/filename
        output.write_text(header+body,encoding='utf-8',newline='')
        outputs.append(dict(file=filename,sha256=digest(output),start_line=section['start_line'],end_line=starts[index+1]-1 if index+1<len(starts) else len(lines)))
    manifest=dict(schema='education-textbook-import-v1',job_id=job_id,source=state['source'],source_sha256=state['source_sha256'],reviewed_sha256=base_sha256,plan_id=plan_id,plan=plan,sections=outputs,images=image_records,output=str(target),diagrams=str(diagrams))
    dump(work/'manifest.json',manifest)
    if action=='publish':
        if digest(path)!=base_sha256:
            raise ValueError('Reviewed Markdown changed during organization')
        # Check all existing destinations before moving either staged directory.
        for original, destination in ((staged_book,target),(staged_images,diagrams)):
            service.workspace_manager.safe_path(destination.relative_to(service.workspace).as_posix())
            if destination.exists():
                expected={p.name:digest(p) for p in original.iterdir() if p.is_file()}
                actual={p.name:digest(p) for p in destination.iterdir() if p.is_file()}
                if expected!=actual or any(p.is_dir() for p in destination.iterdir()):
                    raise ValueError('Existing textbook differs; preserve it and choose another title')
        for original,destination in ((staged_images,diagrams),(staged_book,target)):
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():
                os.replace(original,destination)
        record=service.workspace_manager.safe_path('.education-mcp/textbook-imports')/(book_id+'.json')
        dump(record,manifest)
    return dict(status='PUBLISHED' if action=='publish' else 'PREVIEW',output=str(target),preview=str(work),sections=outputs,
                image_actions=len(actions),preserved_images=len(found)-len(actions),review=review,
                coverage='All reviewed Markdown lines retained in order, except the explicit image-reference actions',plan_id=plan_id)
