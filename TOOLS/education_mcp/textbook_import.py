"""Durable preparation, evidence review and single-directory publication of PDF textbooks."""
from pathlib import Path
import base64
import hashlib
import json
import os
import re
import runpy
import shutil
import tempfile
import uuid

FMT=runpy.run_path(str(Path(__file__).with_name('textbook_format.py')))
sha=FMT['sha']
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
atomic=runpy.run_path(str(Path(__file__).with_name('atomic_files.py')))['atomic_json']


def slug(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError('Expected a nonempty title of at most 160 characters')
    result = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '-', value).strip(' .')[:90]
    if not result or result.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1,10)), *(f'LPT{i}' for i in range(1,10))}:
        raise ValueError('Invalid filesystem title')
    return result


def backup(service, source):
    """Copy before conversion; preserve same-named editions under content IDs."""
    identity = sha(source)
    folder = service.workspace_manager.safe_path('Original_Backup/TEXTBOOKS')
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (slug(source.stem) + '__' + identity[:12] + source.suffix.lower())
    if target.exists():
        if sha(target) != identity:
            raise ValueError('Textbook backup collision; existing source is preserved')
    else:
        temporary = folder / ('.' + uuid.uuid4().hex + '.copying')
        shutil.copyfile(source, temporary)
        if sha(temporary) != identity or sha(source) != identity:
            raise ValueError('Source changed while creating textbook backup')
        # A concurrent identical import may already have created this target.
        if target.exists() and sha(target) != identity:
            raise ValueError('Textbook backup collision')
        os.replace(temporary, target)
    return target, identity


def skill(service):
    for name in ('textbook-import',):
        path = service.mcp_root / 'SKILLS' / name / 'SKILL.md'
        if path.is_file():
            return dict(name=name, path=str(path), content=path.read_text(encoding='utf-8'))
    raise RuntimeError('TEXTBOOK_INSTRUCTIONS_NOT_INSTALLED')


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def checked_path(service, value):
    path=Path(value).resolve(strict=True)
    if not any(path.is_relative_to(root) for root in service.allowed):
        raise ValueError('Input outside configured workspace or input roots')
    return path


def verify_package(package, report, source):
    manifest=read(package/'package_manifest.json')
    if not manifest.get('complete') or manifest['source']['sha256']!=sha(source) or report['source']['sha256']!=manifest['source']['sha256']:
        raise ValueError('Package incomplete or source identity mismatch')
    for filename,key in [('document.md','document_md_sha256'),('conversion_report.json','conversion_report_sha256'),('assets_manifest.jsonl','assets_manifest_sha256')]:
        if sha(package/filename)!=manifest['integrity'][key]:
            raise ValueError('Package identity mismatch: '+filename)
    return manifest


def catalog(service):
    path=service.workspace/'.education-mcp'/'textbook-inputs.json'
    return read(path) if path.exists() else {}


def load_input(service, args):
    reg=catalog(service)
    if 'job_id' in args:
        package=service.package(args['job_id'])
    elif args.get('package_path'):
        package=checked_path(service,args['package_path'])
    else:
        supplied=checked_path(service,args['source'])
        source_key=sha(supplied)
        entry=reg.get(source_key)
        cache=service.workspace_manager.safe_path('tmp/textbook-import-v1')/('source-'+source_key+'.json')
        if not entry and cache.is_file():entry=read(cache)
        if not entry:
            for record in sorted(service.jobs.glob('*/state.json'),key=lambda f:f.stat().st_mtime_ns,reverse=True):
                value=read(record)
                if value.get('status')=='SUCCEEDED' and value.get('source_sha256')==source_key:
                    known=value.get('conversion',{}).get('package_path')
                    if known and Path(known).is_dir():
                        entry={'package':known}
                        break
        if not entry:return None,dict(status='CONVERSION_REQUIRED',source=str(supplied))
        package=checked_path(service,entry['package'])
    report=read(package/'conversion_report.json')
    source=checked_path(service,args.get('source') or report['source']['path'])
    manifest=verify_package(package,report,source)
    key=manifest['source']['sha256']
    entry=reg.get(key,{})
    md=package/'document.md'
    if entry.get('original_sha')==sha(md) and entry.get('reading'):
        md=checked_path(service,entry['reading'])
        if sha(md)!=entry.get('reading_sha'):raise ValueError('Reading revision changed')
    elif (package/'document.reviewed.md').exists():
        md=checked_path(service,package/'document.reviewed.md')
        journal=read(package/'agent_review.json')
        if journal['revisions'][-1]['result_sha256']!=sha(md):raise ValueError('Reviewed text mismatch')
    import fitz
    with fitz.open(source) as pdf:
        page_count=len(pdf)
        native_text=[p.get_text() for p in pdf]
    pages=FMT['read_pages'](md.read_text('utf8'),page_count)
    parts,missing=FMT['outline'](pages)
    issues=[]
    def issue(page,kind,detail):
        page=int(page or 0)
        if not 0<=page<=len(pages):page=0
        issues.append(dict(id=len(issues)+1,page=page,kind=kind,detail=detail))
    for item in report.get('review_items',[]):
        issue(item.get('page'),'conversion',item)
    if report.get('review_item_count',0)>len(report.get('review_items',[])):
        issue(0,'report_count_gap',{'reported':report['review_item_count'],'listed':len(report.get('review_items',[]))})
    if entry.get('findings'):
        if entry.get('reading_sha')!=sha(md):
            issue(0,'review_version_conflict','Reconcile the registered audit with this reading version')
        else:
            file=checked_path(service,entry['findings'])
            if sha(file)!=entry['findings_sha']:raise ValueError('Registered audit changed')
            for item in read(file):
                issue(item.get('page') if isinstance(item.get('page'),int) else 0,'audit',item)
    if missing or not any(p['kind']=='CHAPTER' for p in parts):
        issue(0,'outline',dict(unmatched_toc=missing,reason='Verify chapter boundaries against source'))
    for n,body in enumerate(pages,1):
        native=''.join(native_text[n-1].split())
        for line in body.splitlines():
            compact=''.join(line.strip('# ').split())
            for match in re.finditer(r'([\u3400-\u9fff]{4,})(\d+(?:\.\d+)?)',compact):
                prefix=match[1][-10:]
                values=re.findall(re.escape(prefix)+r'(\d+(?:\.\d+)?)',native)
                if len(values)==1 and values[0]!=match[2]:
                    issue(n,'numeric_mismatch',dict(context=prefix,reading=match[2],pdf_text=values[0]))
        _,invalid_math=FMT['normalize_math'](body.split('\n'))
        if invalid_math:
            issue(n,'math_delimiters',dict(lines=invalid_math,reason='Unpaired delimiters; compare the source'))
    assets={}
    roots=[md.parent,package]
    manifest_assets={r['relative_path']:r.get('content_sha256') for r in
                     (json.loads(l) for l in (package/'assets_manifest.jsonl').read_text('utf8').splitlines() if l.strip()) if r.get('relative_path')}
    for body in pages:
        for match in FMT['IMAGE'].finditer(body):
            ref=match['angle'] or match['plain']
            if ref in assets:continue
            image=FMT['asset_path'](ref,roots);value=sha(image)
            if ref in manifest_assets and manifest_assets[ref] and manifest_assets[ref]!=value:
                raise ValueError('Original asset identity changed: '+ref)
            assets[ref]=dict(path=str(image),sha=value)
    return dict(formatter_sha=sha(Path(__file__).with_name('textbook_format.py')),
                importer_sha=sha(Path(__file__)),source=str(source),source_sha=key,package=str(package),reading=str(md),reading_sha=sha(md),
                report_sha=sha(package/'conversion_report.json'),pages=pages,parts=parts,issues=issues,
                decisions={},assets=assets,title=FMT['safe_name'](source.stem),approved=None),None


def signature(state):
    return digest({k:v for k,v in state.items() if k not in ('approved','published','previewed','observed','final_review')})


def summary(state):
    open_ids=[i['id'] for i in state['issues'] if str(i['id']) not in state['decisions']]
    return dict(revision=signature(state),pages=len(state['pages']),parts=state['parts'],
                asset_count=len(state['assets']),pending=len(open_ids),total_issues=len(state['issues']),
                approved=bool(state['approved']),output=state.get('published'))


def check_inputs(state):
    for key,path in [('source_sha',state['source']),('reading_sha',state['reading']),
                     ('report_sha',Path(state['package'])/'conversion_report.json')]:
        if sha(path)!=state[key]:raise ValueError('Input changed; prepare again')
    pkg=Path(state['package']);verify_package(pkg,read(pkg/'conversion_report.json'),Path(state['source']))
    for a in state['assets'].values():
        if sha(a['path'])!=a['sha']:raise ValueError('Reviewed asset changed; prepare again')


def files_at(path):
    return {p.relative_to(path).as_posix():sha(p) for p in path.rglob('*') if p.is_file()}


def call(service, lock, action, **args):
    work=service.workspace_manager.safe_path('tmp/textbook-import-v1')
    work.mkdir(parents=True,exist_ok=True)
    if action=='prepare':
        state,reply=load_input(service,args)
        if reply:return reply
        ident=digest(state)
    else:
        ident=args.get('import_id','')
        if not re.fullmatch('[0-9a-f]{64}',ident):raise ValueError('Invalid import_id')
    # One lock covers edits and publication, including separate imports with the same target.
    with lock(work/'import.lock',service.closing):
        folder=work/ident;path=folder/'state.json'
        if action=='prepare' and not path.exists():
            folder.mkdir(exist_ok=True)
            atomic(path,state)
        state=read(path)
        if state.get('formatter_sha')!=sha(Path(__file__).with_name('textbook_format.py')) or state.get('importer_sha')!=sha(Path(__file__)):
            raise ValueError('Candidate version changed; prepare again')
        if action=='prepare':
            atomic(work/('source-'+state['source_sha']+'.json'),dict(package=state['package']))
        if action in ('prepare','inspect'):
            return dict(import_id=ident,**summary(state))
        if action=='pending':
            offset=int(args.get('offset',0));limit=int(args.get('limit',12))
            if offset<0 or not 1<=limit<=50:raise ValueError('Invalid page window')
            rows=[i for i in state['issues'] if str(i['id']) not in state['decisions']]
            return dict(revision=signature(state),total=len(rows),items=rows[offset:offset+limit])
        if action=='evidence':
            page=args['page']
            if not 1<=page<=len(state['pages']):raise ValueError('Invalid PDF page')
            if sha(state['source'])!=state['source_sha']:raise ValueError('Source changed')
            import fitz
            with fitz.open(state['source']) as pdf:
                p=pdf[page-1];rect=p.rect
                x0,y0,x1,y1=args.get('region',[0,0,1,1])
                if not (0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('Invalid page region')
                clip=fitz.Rect(rect.x0+x0*rect.width,rect.y0+y0*rect.height,rect.x0+x1*rect.width,rect.y0+y1*rect.height)
                scale=min(3,1600/max(clip.width,clip.height))
                image=p.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=clip,alpha=False).tobytes('png')
            body=state['pages'][page-1]
            state.setdefault('observed',{})[str(page)]=digest(body)
            atomic(path,state)
            return {'content':[{'type':'text','text':json.dumps(dict(page=page,markdown=body,source_sha=state['source_sha']),ensure_ascii=False)},
                               {'type':'image','mimeType':'image/png','data':base64.b64encode(image).decode()}]}
        if args.get('revision')!=signature(state):raise ValueError('Stale revision; inspect again')
        if action=='flag':
            page=args['page'];note=args['source_evidence'].strip()
            if not note or not 1<=page<=len(state['pages']):raise ValueError('Issue needs source location')
            state['issues'].append(dict(id=len(state['issues'])+1,page=page,kind='agent',detail=note))
            state['approved']=None;atomic(path,state);return summary(state)
        if action=='resolve':
            ids=args['issue_ids'];by_id={i['id']:i for i in state['issues']}
            if not ids or len(set(ids))!=len(ids) or any(i not in by_id or str(i) in state['decisions'] for i in ids):
                raise ValueError('Unknown or already resolved issue')
            page=args['page'];note=args['source_evidence'].strip()
            if not note or not 1<=page<=len(state['pages']) or any(by_id[i]['page'] not in (0,page) for i in ids):
                raise ValueError('Provide a matching source page and observation')
            changes=args.get('replacements',[]);outline=args.get('parts')
            if args.get('decision') not in ('keep','correct') or (args['decision']=='correct')!=bool(changes or outline):
                raise ValueError('A correction requires an edit; a keep decision requires evidence only')
            body=state['pages'][page-1]
            if state.get('observed',{}).get(str(page))!=digest(body):raise ValueError('Inspect evidence for this page revision')
            for row in changes:
                old,new=row['old'],row['new']
                if not old or old==new or body.count(old)!=1:raise ValueError('Edit target must be unique on page')
                body=body.replace(old,new,1)
            if changes or outline:
                for k in list(state['decisions']):
                    if outline or state['decisions'][k]['page']==page:del state['decisions'][k]
                state['pages'][page-1]=body
                if outline:state['parts']=outline
            for i in ids:
                state['decisions'][str(i)]=dict(page=page,evidence=note,decision=args['decision'],replacements=changes)
            state['approved']=None;atomic(path,state)
            return summary(state)
        if action not in ('preview','approve','publish'):raise ValueError('Unknown import operation')
        check_inputs(state)
        pending=summary(state)['pending']
        if action!='preview' and pending:raise ValueError(f'{pending} unresolved review items')
        if action=='publish' and state.get('approved')!=signature(state):
            raise ValueError('Final review required for current revision')
        target=service.workspace_manager.safe_path('KNOWLEDGE_BASE/TEXTBOOKS')/state['title']
        target.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='render-',dir=folder))
        try:
            manifest=FMT['render'](state['pages'],state['parts'],Path(state['source']).name,state['source_sha'],
                                   [Path(state['reading']).parent,Path(state['package'])],stage,action!='preview')
            check_inputs(state)
            allowed_assets={a['sha'] for a in state['assets'].values()}
            if any(h not in allowed_assets for f,h in manifest['files'].items() if f.startswith('DIAGRAMS/')):
                raise ValueError('Image missing from reviewed input inventory')
            if action=='preview':
                preview=folder/('preview-'+signature(state))
                if preview.exists() and files_at(preview)!=manifest['files']:
                    preview=folder/('preview-'+stage.name)
                if not preview.exists():stage.rename(preview)
                state['previewed']=dict(revision=signature(state),folder=str(preview),files=manifest['files'])
                atomic(path,state)
                return dict(**summary(state),preview=str(preview),sections=manifest['sections'])
            if action=='approve':
                proof=state.get('previewed') or {}
                if not isinstance(proof,dict) or proof.get('revision')!=signature(state):raise ValueError('Preview this revision first')
                if files_at(Path(proof['folder']))!=proof['files']:raise ValueError('Preview changed; render again')
                if not args.get('source_evidence','').strip():raise ValueError('Final review statement required')
                state['final_review']=args['source_evidence'].strip()
                state['approved']=signature(state);atomic(path,state)
                return summary(state)
            atomic(folder/'manifest.json',manifest)
            if target.exists():
                if files_at(target)!=manifest['files']:raise ValueError('Existing textbook differs; publication conflict')
            else:
                stage.rename(target)
            state['published']=str(target);atomic(path,state)
            return dict(status='PUBLISHED',**summary(state))
        finally:
            if stage.exists():shutil.rmtree(stage)

