"""One finite table-model stage; no gold or LLM context is accepted."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    assert cfg['schema']=='isolated-table-model-stage-v1' and sha(__file__)==cfg['worker_sha256']
    assert sha(sys.executable)==cfg['python_executable_sha256']
    assert sha(Path(cfg['snapshot_src'])/'bemarkdown/pdf/table_staged_runtime.py')==cfg['coordinator_sha256']
    sys.path[:0]=[cfg['snapshot_src'],*cfg['dependency_paths']]
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    from bemarkdown.pdf.table_runtime import RegistryBackedTableEvidenceProvider
    from bemarkdown.pdf_content_router import PaddleXPdfOcrRuntime
    model=cfg['model_id'];results={};started=time.monotonic()
    if model=='pp-ocrv6':
        runtime=PaddleXPdfOcrRuntime(**cfg['registry_options'])
        try:
            for row in cfg['rows']:
                assert sha(row['path'])==row['sha256']==row['crop']['content_sha256']
                results[row['key']]=runtime.recognize(row['crop'])
            identity=runtime.fingerprint()
        finally:runtime.close()
    else:
        if model=='pp-lcnet-x1-0-table-cls':fields=('label_names','scores')
        elif model in {'slanext-wired','slanext-wireless'}:fields=('structure','bbox','structure_score')
        elif model in {'rt-detr-l-wired-table-cell-det','rt-detr-l-wireless-table-cell-det'}:fields=('boxes',)
        else:raise RuntimeError('UNREGISTERED_TABLE_STAGE:'+model)
        runtime=RegistryBackedTableEvidenceProvider(Path(a.output).parent,**cfg['registry_options'])
        try:
            for row in cfg['rows']:
                assert sha(row['path'])==row['sha256']
                results[row['key']]=runtime._predict(model,Path(row['path']),fields)
            identity=runtime._fingerprints[model]
        finally:runtime.close()
    for row in cfg['rows']:assert sha(row['path'])==row['sha256']
    Path(a.output).write_text(json.dumps({'schema':cfg['schema'],'model_id':model,
        'request_sha256':sha(a.manifest),'results':results,'identity':identity,
        'elapsed_seconds':time.monotonic()-started,'pid':os.getpid(),'gold_read':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
