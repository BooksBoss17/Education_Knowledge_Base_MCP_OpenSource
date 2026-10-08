"""Prepare source-only classifier inputs from an unchanged BeMarkdown PDF package.

This adapter does not read evaluation labels, previous predictions, or category folders.
The inference-facing selectors are the frozen, separately vendored functions.
"""
import argparse,copy,hashlib,importlib.util,io,json,math,mimetypes,re,shutil,sys,time
from pathlib import Path
from urllib.parse import unquote,urlsplit

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
def emit(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def build(package,job_root,libraries,source_pdf=None):
    from PIL import Image
    import numpy as np
    from scipy.fft import dctn
    import pymupdf
    started=time.monotonic();package=Path(package).resolve(strict=True);R=Path(job_root).resolve();libraries=Path(libraries).resolve()
    P=R/'projects/bemarkdown-retention-training';tools=libraries/'projects/bemarkdown-retention-training/tools'
    sys.path.insert(0,str(tools));sys.path.insert(0,str(libraries/'projects/bemarkdown-retention-training/reports'))
    from contracts import validate_input,digest
    from dev_calibration import policy_fingerprint
    from build_bound_md_probe import context_view
    from extreme_aspect_padding_candidate import pad_extreme
    core=module('portable_source_core',libraries/'projects/education_mcp/context_retention.py')
    engine=module('portable_source_context',libraries/'projects/education_mcp/retention_context_v8_7.py')
    selector=module('portable_source_selector',libraries/'tmp/retention-v6-20261003/source_builder_v29.py')
    original_report=read(package/'conversion_report.json')
    assert original_report['source']['type']=='PDF','SOURCE_PDF_REQUIRED'
    text=(package/'document.md').read_bytes().decode('utf-8')
    # Reject nonlocal/escaping image destinations before the legacy reader opens files.
    for match in core.IMAGE.finditer(text):
        token=match.group();inside=token[token.index('](')+2:-1];dest=core.DEST.fullmatch(inside)
        assert dest is not None,'UNSUPPORTED_IMAGE_REFERENCE'
        name=unquote(dest.group(1) if dest.group(1) is not None else dest.group(2))
        assert not urlsplit(name).scheme and not Path(name).is_absolute(),'EXTERNAL_IMAGE_REFERENCE'
        assert (package/name).resolve(strict=True).is_relative_to(package),'IMAGE_REFERENCE_ESCAPES_PACKAGE'
    prepared=core.prepare(package)
    pdf=Path(source_pdf or prepared['source']['path']).resolve(strict=True)
    assert sha(pdf)==prepared['source']['sha256'],'SOURCE_PDF_HASH_MISMATCH'
    shadow=R/'source-package';shadow.mkdir(parents=True,exist_ok=False)
    shutil.copy2(package/'document.md',shadow/'document.md')
    shutil.copy2(pdf,shadow/'source.pdf')
    report=copy.deepcopy(original_report);report['source']['path']=str(shadow/'source.pdf')
    write(shadow/'conversion_report.json',report)
    for asset in prepared['assets']:
        source=Path(asset['path']).resolve(strict=True);assert source.is_relative_to(package)
        target=shadow/asset['asset_name'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert sha(target)==asset['asset_sha256']
        asset['path']=str(target)
    prepared.update(package=str(shadow),source=report['source'])
    book_id='book-'+prepared['source']['sha256'][:12]
    prepared_path=R/'source-inputs.json';write(prepared_path,prepared)
    features={};resolver={};rows=[];bindings=[];asset_map=[]
    objects=R/'tmp/rt-train/objects';objects.mkdir(parents=True,exist_ok=True)
    context=engine.Context(prepared)
    try:
        for asset in sorted(prepared['assets'],key=lambda x:x['asset_name'].casefold()):
            image_sha=asset['asset_sha256'];source=Path(asset['path']);sample_id=book_id+':'+asset['asset_name']
            if image_sha not in features:
                data=source.read_bytes();assert hashlib.sha256(data).hexdigest()==image_sha
                object_path=objects/(image_sha+source.suffix.lower());object_path.write_bytes(data)
                with Image.open(io.BytesIO(data)) as image:
                    rgb=image.convert('RGB');width,height=rgb.size;pixels=hashlib.sha256()
                    for y in range(0,height,64):
                        with rgb.crop((0,y,width,min(height,y+64))) as strip:pixels.update(strip.tobytes())
                    arr=np.asarray(rgb.resize((128,128),Image.Resampling.LANCZOS),dtype=np.uint8).copy()
                    gray=np.asarray(rgb.resize((32,32),Image.Resampling.LANCZOS).convert('L'),dtype=np.float32)
                    coef=dctn(gray,norm='ortho')[:8,:8];bits=(coef>np.median(coef[1:,:])).reshape(-1)
                    features[image_sha]={'image_sha256':image_sha,'pixel_sha256':pixels.hexdigest(),'width':width,'height':height,
                        'phash64':str(sum(int(v)<<i for i,v in enumerate(bits))),'mean_rgb':arr.mean(axis=(0,1)).tolist(),
                        'gray_std':float(np.std(gray)),'object_path':str(object_path)}
                resolver['sha256:'+image_sha]=str(object_path)
            feature=features[image_sha];uses=[];proofs=[]
            for occurrence in asset['occurrences']:
                facts,proof=selector.build(context,prepared,asset,occurrence,None,max_chars=2400)
                spans=[];refs=[]
                for card in facts['figure_attribution']['records']:
                    matches=[n for n in context.pages.get(occurrence['page'],[]) if n['text']==card['text'] and list(n['bbox_pdf_pt'])==list(card['bbox'])]
                    assert matches,'SOURCE_CARD_WITHOUT_MD_SPAN'
                    node=min(matches,key=lambda n:n['byte_start']);start,end=node['byte_start'],node['byte_end']
                    content=context.raw[start:end].decode('utf-8');assert content==card['text']
                    spans.append({'span_id':f'md-{start}-{end}','start_byte':start,'end_byte':end,'text':content,
                        'text_sha256':hashlib.sha256(content.encode()).hexdigest(),'source_page_index':occurrence['page'],
                        'relation_to_use':'same_page','binding_evidence_refs':[]})
                    for ref in card['figure_refs']:
                        assignment=ref['assignment']
                        if assignment not in {'TARGET_CAPTION_IMAGE','TARGET_REGION_OR_COMPONENT','TARGET_GROUP_CANDIDATE','OTHER_IMAGE_REGION','UNRESOLVED'}:assignment='UNRESOLVED'
                        refs.append({'figure':str(ref['figure']),'assignment':assignment})
                has_text=any(n['kind'] not in ('IMAGE','OTHER') and n['text'].strip() for n in context.pages.get(occurrence['page'],[]))
                status='verified' if spans else ('missing' if has_text else 'verified_empty')
                use={'use_id':occurrence['id'],'page_index':occurrence['page'],'context_status':status,'spans':spans,
                     'figure_refs':refs,'selector_version':'source_v29_actual_md_no_gold_v1','source_window_truncated':status=='missing'}
                selected={(s['start_byte'],s['end_byte']) for s in spans}
                omitted=[{'start_byte':n['byte_start'],'end_byte':n['byte_end']} for n in proof['page_nodes'] if (n['byte_start'],n['byte_end']) not in selected]
                proofs.append({'use_id':occurrence['id'],'source_doc_sha256':prepared['markdown_sha256'],'occurrence':occurrence,
                    'source_usage_sha256':digest(facts),'caption_bindings':proof['caption_bindings'],'omitted_md_ranges':omitted,
                    'selection_limits':{'max_chars':2400,'max_nodes':5,'span_truncation':False},'layout_fact_scope':'source-derived, not Gold; not model-visible'})
                uses.append(use)
            row={'schema_version':'1.0','sample_id':sample_id,'book_id':book_id,'synthetic':False,
                 'image':{'ref':'sha256:'+image_sha,'sha256':image_sha,'mime':mimetypes.guess_type(feature['object_path'])[0],
                          'width':feature['width'],'height':feature['height']},
                 'document':{'ref':str(shadow/'document.md'),'sha256':prepared['markdown_sha256']},'scope':'asset_use_bag','uses':uses,
                 'coverage':{'expected_use_count':len(uses),'covered_use_ids':[u['use_id'] for u in uses],'enumeration_complete':True}}
            validate_input(row);rows.append(row);bindings.append({'sample_id':sample_id,'uses':proofs,'actual_md_verified':True})
            asset_map.append({'sample_id':sample_id,'asset_name':asset['asset_name'],'original_path':str(package/asset['asset_name']),'image_sha256':image_sha})
    finally:context.close()
    assert rows,'NO_IMAGE_REFERENCES'
    emit(P/'data/prepared/v2/inputs.jsonl',rows);emit(P/'data/prepared/v2/binding-proof.jsonl',bindings)
    emit(P/'data/manifests/image-features.jsonl',features.values())
    write(P/'data/manifests/source-inventory.json',{'books':[{'book_id':book_id,'inputs_json':str(prepared_path),
          'declared_source_book':original_report['source'].get('file_name',pdf.name),'markdown_sha256':prepared['markdown_sha256']}]})
    write(R/'asset-map.json',asset_map)
    first=[];bound=[];bound_proofs=[];docs=R/'tmp/rt-train/runs/input-preparation/docs';docs.mkdir(parents=True,exist_ok=True)
    raw=(shadow/'document.md').read_bytes();proof_by_id={x['sample_id']:x for x in bindings}
    for number,original in enumerate(rows):
        row=copy.deepcopy(original)
        if max(row['image']['width'],row['image']['height'])/min(row['image']['width'],row['image']['height'])>200:
            with Image.open(resolver[row['image']['ref']]) as image:padded,geometry=pad_extreme(image,32)
            buf=io.BytesIO();padded.save(buf,format='PNG');data=buf.getvalue();h=hashlib.sha256(data).hexdigest()
            target=objects/(h+'.png');target.write_bytes(data)
            row['image'].update(ref='sha256:'+h,sha256=h,mime='image/png',width=padded.width,height=padded.height);resolver[row['image']['ref']]=str(target)
        validate_input(row);first.append(row)
        changed=copy.deepcopy(row);content=bytearray();by_use={u['use_id']:u for u in proof_by_id[row['sample_id']]['uses']}
        for use in changed['uses']:
            text,receipt=context_view(raw,use,by_use[use['use_id']])
            if text is None:text='\n'.join(s['text'] for s in use['spans'])
            else:use['selector_version']='bound-md-full-page-v1'
            start=len(content);value=text.encode('utf-8');content.extend(value);end=len(content);content.extend(b'\n\n')
            if value:use['spans']=[{'span_id':f'derived-md-{start}-{end}','start_byte':start,'end_byte':end,'text':text,
                'text_sha256':hashlib.sha256(value).hexdigest(),'source_page_index':use['page_index'],'relation_to_use':'same_page','binding_evidence_refs':[]}]
            bound_proofs.append({'sample_id':row['sample_id'],'use_id':use['use_id'],'receipt':receipt})
        path=docs/f'{number:05d}.md';path.write_bytes(content);changed['document']={'ref':str(path),'sha256':sha(path)}
        validate_input(changed);bound.append(changed)
    out=docs.parent;emit(out/'first.inputs.jsonl',first);emit(out/'bound.inputs.jsonl',bound)
    emit(P/'data/splits/v2/resolver.jsonl',[{'image_ref':ref,'object_path':path} for ref,path in sorted(resolver.items())])
    emit(out/'bound-proof.jsonl',bound_proofs)
    # Repetition and locality are source features; no model predictions enter this computation.
    nodes={}
    for node in report['markdown_render']['node_spans']:
        if node['kind'] in ('IMAGE','OTHER'):continue
        value=raw[node['byte_start']:node['byte_end']];assert not node.get('sha256') or hashlib.sha256(value).hexdigest()==node['sha256']
        text=value.decode('utf-8')
        if text.strip():nodes.setdefault(node['page_index'],[]).append({'bbox':node['bbox_pdf_pt'],'text':text,
                    'byte_start':node['byte_start'],'byte_end':node['byte_end']})
    with pymupdf.open(shadow/'source.pdf') as pdf_doc:sizes={i:[p.rect.width,p.rect.height] for i,p in enumerate(pdf_doc)}
    def distance(a,b):return max(a[0]-b[2],b[0]-a[2],0)+max(a[1]-b[3],b[1]-a[3],0)
    locality=[]
    for row in rows:
        feature=features[row['image']['sha256']];similar=set();exact=set();aspect=feature['width']/feature['height']
        for other in rows:
            other_feature=features[other['image']['sha256']];other_uses=proof_by_id[other['sample_id']]['uses']
            if other_feature['pixel_sha256']==feature['pixel_sha256']:exact.update(u['occurrence']['page'] for u in other_uses)
            if abs(math.log((other_feature['width']/other_feature['height'])/aspect))>math.log(1.15):continue
            if (int(feature['phash64'])^int(other_feature['phash64'])).bit_count()>4:continue
            if max(abs(a-b) for a,b in zip(feature['mean_rgb'],other_feature['mean_rgb']))>25:continue
            similar.update(u['occurrence']['page'] for u in other_uses)
        positions=[]
        for use in proof_by_id[row['sample_id']]['uses']:
            occurrence=use['occurrence'];box=occurrence['bbox'];width,height=sizes[occurrence['page']]
            nearby=sorted(nodes.get(occurrence['page'],[]),key=lambda n:distance(box,n['bbox']))
            close=[n for n in nearby if distance(box,n['bbox'])<=32];short=[n for n in close if len(''.join(n['text'].split()))<=14]
            positions.append({'use_id':use['use_id'],'page_index':occurrence['page'],'bbox':box,'page_size':[width,height],
                'header_band':box[3]<=height*.12,'footer_band':box[1]>=height*.90,'close_text_count':len(close),
                'short_close_texts':short,'nearest_texts':[{**n,'gap_pt':distance(box,n['bbox'])} for n in nearby[:3]],
                'whole_page_text_chars':sum(len(n['text']) for n in nearby),'within_32pt_text_chars':sum(len(n['text']) for n in close)})
        locality.append({'sample_id':row['sample_id'],'uses':positions,'distinct_exact_pixel_pages':len(exact),
            'distinct_similar_visual_pages':len(similar),'all_uses_in_margin':bool(positions) and all(u['header_band'] or u['footer_band'] for u in positions),
            'all_uses_have_close_short_text':bool(positions) and all(u['short_close_texts'] for u in positions),'new_images_decoded':0})
    feature_root=R/'tmp/rt-train/z99/locality-v1';emit(feature_root/'features.jsonl',locality)
    write(feature_root/'freeze.json',{'status':'SOURCE_FEATURES_FROZEN','features_sha256':sha(feature_root/'features.jsonl'),'gold_opened':False})
    result={'status':'SOURCE_ONLY_INPUTS_PREPARED','count':len(rows),'book_id':book_id,'source_sha256':prepared['source']['sha256'],
        'markdown_sha256':prepared['markdown_sha256'],'first_inputs':str(out/'first.inputs.jsonl'),'bound_inputs':str(out/'bound.inputs.jsonl'),
        'resolver':str(P/'data/splits/v2/resolver.jsonl'),'features':str(feature_root/'features.jsonl'),
        'features_sha256':sha(feature_root/'features.jsonl'),'elapsed_seconds':time.monotonic()-started,'gold_opened':False,
        'policy_fingerprints':{'0.8':policy_fingerprint(.8),'0.5':policy_fingerprint(.5)}}
    write(R/'source-preparation.json',result);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True);ap.add_argument('--job-root',required=True)
    ap.add_argument('--libraries',required=True);ap.add_argument('--source-pdf')
    args=ap.parse_args();print(json.dumps(build(args.package,args.job_root,args.libraries,args.source_pdf),ensure_ascii=True))
