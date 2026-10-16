"""Batch table inputs by model, with each heavy model in a short-lived process."""
from __future__ import annotations
import copy,hashlib,json,os,subprocess,sys,time,tempfile
from pathlib import Path
from PIL import Image
from .table_runtime import RegistryBackedTableEvidenceProvider
from ..pdf_document_ir import document_asset_source_ref
from ..pdf_table_engine import normalize_table_classification

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def ocr_input(asset,block):
    path=Path(document_asset_source_ref(asset)).resolve(strict=True)
    with Image.open(path) as image:width,height=image.size
    bbox=block['bbox_pdf_pt']
    return {'path':str(path),'content_sha256':asset['content_sha256'],'width':width,'height':height,
            'bbox_pdf_pt':list(bbox),'scale_x':width/(bbox[2]-bbox[0]),'scale_y':height/(bbox[3]-bbox[1])}
def ocr_key(crop):return key({k:crop[k] for k in ('content_sha256','width','height','bbox_pdf_pt','scale_x','scale_y')})

class ReplayTableOCR:
    def __init__(self,results,identity):self.results=results;self.identity=identity
    def recognize(self,crop):return copy.deepcopy(self.results[ocr_key(crop)])
    def fingerprint(self):return copy.deepcopy(self.identity)
    def close(self):pass

class StagedTableEvidenceProvider(RegistryBackedTableEvidenceProvider):
    def __init__(self,work_root,*,execute=None,before_stages=None,**options):
        super().__init__(work_root,**options)
        self.execute=execute or subprocess.run;self.before_stages=before_stages
        self.stage_records=[];self.stage_cache={};self.prepared=None;self.stage_root=None;self.closed=False
        self.prepared_consumers=set()

    def _save_stage_state(self):
        (self.work_root/'table-stage-controller.json').write_text(json.dumps({
            'version':'table-model-stages-v1','stages':self.stage_records},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    def _run_stage(self,model_id,rows):
        worker=Path(__file__).with_name('workers')/'table_stage.py'
        if self.stage_root is None:self.stage_root=Path(tempfile.mkdtemp(prefix='table-stages-'))
        folder=self.stage_root/str(len(self.stage_records))
        folder.mkdir(parents=True,exist_ok=False)
        request={'schema':'isolated-table-model-stage-v1','model_id':model_id,'rows':rows,
            'registry_options':{k:str(v) if v is not None else None for k,v in self.registry_options.items()},
            'snapshot_src':str(Path(__file__).resolve().parents[2]),
            'dependency_paths':[str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir()],
            'worker_sha256':sha(worker),'coordinator_sha256':sha(__file__)}
        executable=getattr(sys,'_base_executable',sys.executable)
        request['python_executable_sha256']=sha(executable)
        manifest=folder/'request.json';output=folder/'output.json'
        manifest.write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        # Direct base interpreter avoids a Windows venv redirector surviving a timeout.
        started=time.monotonic()
        record={'model_id':model_id,'inputs':len(rows),'status':'RUNNING',
                'request_path':str(manifest),'request_sha256':sha(manifest),'output_path':str(output)}
        self.stage_records.append(record);self._save_stage_state()
        try:
            with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
                run=self.execute([executable,'-I',str(worker),'--manifest',str(manifest),'--output',str(output)],
                    stdout=stdout,stderr=stderr,timeout=900,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            record['child_exit_code']=run.returncode
            assert run.returncode==0,'TABLE_MODEL_STAGE_FAILED:'+model_id+':'+str(run.returncode)
            result=json.loads(output.read_text(encoding='utf-8'))
            assert result['request_sha256']==sha(manifest) and result['model_id']==model_id
            assert set(result['results'])=={r['key'] for r in rows},'TABLE_MODEL_STAGE_POPULATION_CHANGED'
            for row in rows:assert sha(row['path'])==row['sha256']
            record.update(status='COMPLETED',output_sha256=sha(output),child_exited_before_next_stage=True)
        except BaseException as exc:
            record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc));raise
        finally:
            record['seconds']=time.monotonic()-started;self._save_stage_state()
        return result

    def prepare_batch(self,items):
        if self.closed:raise RuntimeError('TABLE_STAGED_PROVIDER_CLOSED')
        descriptors=[];unique={};ocr_rows={}
        for block,asset in items:
            path=Path(document_asset_source_ref(asset)).resolve(strict=True);path.relative_to(self.work_root)
            digest=sha(path);assert digest==asset['content_sha256']
            descriptors.append((block['document_id'],block['node_id'],block['page_index'],digest,list(block['bbox_pdf_pt'])))
            unique.setdefault(digest,{'key':digest,'sha256':digest,'path':str(path)})
            crop=ocr_input(asset,block);oid=ocr_key(crop)
            ocr_rows.setdefault(oid,{'key':oid,'sha256':digest,'path':str(path),'crop':crop})
        signature=key(descriptors)
        if self.prepared is not None:
            assert self.prepared==signature,'TABLE_PREPARED_COHORT_CHANGED'
            return
        if self.before_stages is not None:self.before_stages()
        classifier='pp-lcnet-x1-0-table-cls'
        first=self._run_stage(classifier,list(unique.values()))
        self.stage_cache[classifier]=first['results'];self._fingerprints[classifier]=first['identity']
        branches={'wired':[],'wireless':[]}
        for uid,row in unique.items():
            raw=first['results'][uid]['raw']
            normalized=normalize_table_classification(raw.get('label_names',[]),raw.get('scores',[]))
            branches[normalized['selected_branch'].lower()].append(row)
        for branch,rows in branches.items():
            if not rows:continue
            for model_id in ('slanext-'+branch,'rt-detr-l-'+branch+'-table-cell-det'):
                result=self._run_stage(model_id,rows)
                self.stage_cache[model_id]=result['results'];self._fingerprints[model_id]=result['identity']
        result=self._run_stage('pp-ocrv6',list(ocr_rows.values()))
        self._ocr=ReplayTableOCR(result['results'],result['identity'])
        self._counts['model_loads']=len(self.stage_records)-1
        self.prepared=signature;self.prepared_consumers={key(d) for d in descriptors}

    def __call__(self,block,asset):
        if self.closed or self.prepared is None:raise RuntimeError('TABLE_STAGES_NOT_PREPARED_OR_CLOSED')
        descriptor=(block['document_id'],block['node_id'],block['page_index'],asset['content_sha256'],list(block['bbox_pdf_pt']))
        if key(descriptor) not in self.prepared_consumers:raise RuntimeError('TABLE_CONSUMER_NOT_PREPARED')
        return super().__call__(block,asset)

    def _predict(self,model_id,path,fields):
        if self.prepared is None:raise RuntimeError('TABLE_STAGES_NOT_PREPARED')
        value=self.stage_cache[model_id][sha(path)]
        assert set(value['raw']).issubset(fields),'TABLE_STAGE_OUTPUT_FIELDS_CHANGED'
        return copy.deepcopy(value)

    def _load_model(self,model_id):raise RuntimeError('TABLE_COORDINATOR_MUST_NOT_LOAD_MODELS')
    def metrics(self):
        return {**super().metrics(),'process_isolated_model_stages':self.stage_records,
                'resource_strategy':'table-model-stages-v1','local_batch_size':1}
    def close(self):
        super().close();self.closed=True
