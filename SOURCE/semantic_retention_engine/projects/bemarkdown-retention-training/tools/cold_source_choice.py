"""CPU only: scan source positions for all current deletions, never open labels."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import fitz
sys.path.insert(0,str(Path(__file__).resolve().parent))
from source_choice_option import option_witnesses
R=Path(__file__).resolve().parents[3];P=R/'projects/bemarkdown-retention-training'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--baseline',required=True);a=ap.parse_args()
    start=time.monotonic();out=Path(a.run_dir).resolve();out.mkdir(exist_ok=True)
    base=Path(a.baseline).resolve()
    assert sha(base)==read(base.with_suffix('.freeze.json'))['prediction_file_sha256']
    delete={x['sample_id'] for x in rows(base) if x['final_decision']=='D'}
    originals={x['sample_id']:x for x in rows(P/'data/prepared/v2/inputs.jsonl')}
    bindings={x['sample_id']:x for x in rows(P/'data/prepared/v2/binding-proof.jsonl')}
    books=read(P/'data/manifests/source-inventory.json')['books']
    write(out/'preregistration.json',{'baseline':str(base),'baseline_sha256':sha(base),
      'predicate':'every current D; any verified use with four distinct image-valued options is necessary',
      'code_sha256':sha(__file__),'rule_sha256':sha(P/'tools/source_choice_option.py'),
      'gold_opened':False,'only_positive_K_changes':True,'delete_candidates':len(delete)})
    decisions=[];audits=[]
    for book in books:
        selected=[sid for sid in delete if originals[sid]['book_id']==book['book_id']]
        if not selected:continue
        prepared=read(R/book['inputs_json']);pdf=Path(prepared['source']['path'])
        assert sha(pdf)==prepared['source']['sha256']
        occurrences=[o for a in prepared['assets'] for o in a['occurrences']]
        by_page={}
        for o in occurrences:by_page.setdefault(o['page'],[]).append(o)
        pages={u['page_index'] for sid in selected for u in originals[sid]['uses']}
        proof_by_page={}
        with fitz.open(pdf) as doc:
            for page_index in sorted(pages):
                page=doc[page_index];text=page.get_text()
                proof_by_page[page_index]=option_witnesses(page.get_text('words'),by_page.get(page_index,[])) if any(t in text for t in ('选择题','选项','下列')) else {}
        for sid in sorted(selected):
            witnessed=[]
            for u in bindings[sid]['uses']:
                occ=u['occurrence'];proof=proof_by_page.get(occ['page'],{}).get(occ['id'])
                if proof:witnessed.append({'use_id':u['use_id'],'page_index':occ['page'],'proof':proof})
            if witnessed:decisions.append({'sample_id':sid,'final_decision':'K','proofs':witnessed,'source_pdf_sha256':prepared['source']['sha256']})
        audits.append({'book_id':book['book_id'],'candidates':len(selected),'pages_scanned':len(pages),'source_pdf_sha256':prepared['source']['sha256']})
    write(out/'decisions.json',decisions)
    write(out/'freeze.json',{'status':'SOURCE_CHOICE_POSITIVE_PROOFS_FROZEN','decisions_sha256':sha(out/'decisions.json'),
      'protected':len(decisions),'all_deletions_examined':len(delete),'audits':audits,'gold_opened':False,
      'elapsed_seconds':time.monotonic()-start,'image_pixels_opened':0})
    print(json.dumps({'protected':len(decisions),'all_deletions_examined':len(delete),'seconds':time.monotonic()-start}))

if __name__=='__main__':main()
