"""Construct fresh branch inputs from source metadata and this run's predictions only."""
import argparse,copy,json,re
from pathlib import Path
import probe_gguf_image_runtime as G
import retention_inference as N
from contracts import validate_input,source_complete
from build_zero99_locality_probe import local_view
from build_zero99_spatial_probe import layout_view
from prepare_zero99_semantic_toc_reviews import candidate

def emit(path,rows):
    with path.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def mask_source_links(value):
    # Preserve captions while removing filenames and generated generic image labels.
    def mask(m):
        alt=m.group(1)
        if alt in ('表格原图','表格视觉保留','保留原图'):alt=''
        return '[来源中的其他图片占位'+('：'+alt if alt else '')+']'
    return re.sub(r'!\[([^\]]*)\]\([^)]*\)',mask,value)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--stage',choices=['locality','toc','body','dependency'],required=True)
    ap.add_argument('--baseline',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    P,R=G.P,G.R;config=G.read(a.config);out=Path(a.out).resolve();out.mkdir(exist_ok=False)
    baseline={x['sample_id']:x for x in N.A.read_rows(a.baseline)}
    bound=list(N.A.read_rows(config['bound_inputs']))
    assert set(baseline)=={x['sample_id'] for x in bound}
    first={x['sample_id']:x for x in N.A.read_rows(config['firstpass_predictions'])}
    guard={x['sample_id']:x for x in N.A.read_rows(config['guard_predictions'])}
    features_path=Path(config['features'])
    assert N.file_sha(features_path)==config['features_sha256']
    features={x['sample_id']:x for x in N.A.read_rows(features_path)}
    original={x['sample_id']:x for x in N.A.read_rows(P/'data/prepared/v2/inputs.jsonl')}
    bindings={x['sample_id']:x for x in N.A.read_rows(P/'data/prepared/v2/binding-proof.jsonl')}
    resolver={x['image_ref']:x['object_path'] for x in N.A.read_rows(P/'data/splits/v2/resolver.jsonl')}
    books={x['book_id']:x for x in G.read(P/'data/manifests/source-inventory.json')['books']}
    if a.stage=='locality':
        selected=[x for x in bound if first[x['sample_id']]['raw_decision']=='K'
                  and features[x['sample_id']]['distinct_similar_visual_pages']>=4
                  and features[x['sample_id']]['all_uses_have_close_short_text']]
    elif a.stage=='toc':selected=[x for x in bound if baseline[x['sample_id']]['final_decision']=='K' and candidate(x)]
    elif a.stage=='body':
        selected=[x for x in bound if baseline[x['sample_id']]['final_decision']=='K'
                  and guard[x['sample_id']]['guard_triggered'] and guard[x['sample_id']]['form']=='PHOTOGRAPH' and not candidate(x)]
    else:
        previous=G.read(config['body_manifest'])
        selected=[x for x in previous if baseline[x['sample_id']]['final_decision']=='D']
        G.write(out/'manifest.json',selected)
        G.write(out/'input-freeze.json',{'stage':a.stage,'count':len(selected),'manifest_sha256':N.file_sha(out/'manifest.json'),
              'model_response_cache_used':False,'source_manifest':config['body_manifest'],'gold_opened':False})
        print(json.dumps({'stage':a.stage,'count':len(selected)}));return
    cache={};items=[];proofs=[]
    if a.stage=='locality':(out/'docs').mkdir()
    for number,row in enumerate(selected):
        sid,book=row['sample_id'],row['book_id']
        assert source_complete(row)
        if book not in cache:
            prepared_path=R/books[book]['inputs_json'];prepared=json.loads(prepared_path.read_text(encoding='utf-8-sig'))
            conversion_path=Path(prepared['package'])/'conversion_report.json'
            conversion=G.read(conversion_path)
            doc=Path(original[sid]['document']['ref']);doc=doc if doc.is_absolute() else P/doc
            assert N.file_sha(doc)==original[sid]['document']['sha256']
            cache[book]={'raw':doc.read_bytes(),'nodes':conversion['markdown_render']['node_spans'],
                 'peers':[o for asset in prepared['assets'] for o in asset['occurrences']],
                 'prepared_sha256':N.file_sha(prepared_path),'conversion_sha256':N.file_sha(conversion_path)}
        source=cache[book];mapping={x['use_id']:x for x in bindings[sid]['uses']}
        if a.stage=='locality':
            row=copy.deepcopy(row);content=bytearray()
            for use in row['uses']:
                text,selected_nodes=local_view(mapping[use['use_id']],source['raw'],source['nodes'],features[sid])
                value=text.encode('utf-8');start=len(content);content.extend(value);end=len(content);content.extend(b'\n\n')
                use['spans']=[{'span_id':f'local-{number}-{start}','start_byte':start,'end_byte':end,'text':text,
                    'text_sha256':__import__('hashlib').sha256(value).hexdigest(),'source_page_index':use['page_index'],
                    'relation_to_use':'same_page','binding_evidence_refs':[]}]
                use['selector_version']='repeated-region-bounded-md-v1'
                proofs.append({'sample_id':sid,'use_id':use['use_id'],'nodes':selected_nodes})
            doc=out/'docs'/f'{number:05d}.md';doc.write_bytes(content)
            row['document']={'ref':str(doc),'sha256':N.file_sha(doc)};validate_input(row);items.append(row);continue
        document=Path(row['document']['ref']);document=document if document.is_absolute() else P/document
        assert N.file_sha(document)==row['document']['sha256'];raw=document.read_bytes()
        positions={x['use_id']:x for x in features[sid]['uses']};uses=[]
        for use in row['uses']:
            lines=[]
            if a.stage=='toc':
                nodes=[]
                for node in source['nodes']:
                    if node['page_index']!=use['page_index'] or node['kind'] in ('IMAGE','OTHER'):continue
                    value=source['raw'][node['byte_start']:node['byte_end']]
                    if node.get('sha256'):assert __import__('hashlib').sha256(value).hexdigest()==node['sha256']
                    nodes.append({'bbox':node['bbox_pdf_pt'],'text':value.decode('utf-8')})
                proofs.append({'sample_id':sid,'use_id':use['use_id'],'nodes':nodes,'conversion_sha256':source['conversion_sha256']})
            if a.stage=='body':
                spatial,proof=layout_view(source['raw'],use,mapping[use['use_id']]['occurrence'],source['nodes'],source['peers'],text_budget=1400,max_nodes=24)
                lines.append(spatial);proofs.append({'sample_id':sid,'use_id':use['use_id'],'proof':proof})
            for span in use['spans']:
                assert raw[span['start_byte']:span['end_byte']].decode('utf-8')==span['text']
                lines.extend(s.strip() for s in span['text'].splitlines() if s.strip())
            if a.stage=='body':lines=[mask_source_links(s) for s in lines]
            pos=positions[use['use_id']]
            uses.append({'use_id':use['use_id'],'source_lines':lines,'target_bbox_pdf_pt':pos['bbox'],
                'source_page_size_pdf_pt':pos['page_size'],'context_status':use['context_status']})
        im=original[sid]['image'];path=Path(resolver[im['ref']]);path=path if path.is_absolute() else R/path
        assert N.file_sha(path)==im['sha256']
        items.append({'sample_id':sid,'image_path':str(path),'image':im,'source_document':row['document'],
                      'coverage':row['coverage'],'uses':uses})
    if a.stage=='locality':emit(out/'locality.inputs.jsonl',items)
    else:G.write(out/'manifest.json',items)
    G.write(out/'binding-proof.json',proofs)
    target=out/('locality.inputs.jsonl' if a.stage=='locality' else 'manifest.json')
    G.write(out/'input-freeze.json',{'stage':a.stage,'count':len(items),'artifact_sha256':N.file_sha(target),
         'source_features_sha256':config['features_sha256'],'baseline_sha256':N.file_sha(a.baseline),
         'model_response_cache_used':False,'gold_opened':False,'source_builder_sha256':N.file_sha(__file__)})
    print(json.dumps({'stage':a.stage,'count':len(items)}))

if __name__=='__main__':main()
