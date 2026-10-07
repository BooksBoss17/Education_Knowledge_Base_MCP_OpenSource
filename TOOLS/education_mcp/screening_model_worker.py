"""Frozen, label-blind local feature extraction for asset-screening experiments.

Does not score truth, train templates, edit source images, or download models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--kind',choices=['wemm','siglip2'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resource-profile',choices=['6gb','10gb'],default='6gb')
    args=parser.parse_args()
    if sha(args.manifest)!=args.manifest_sha256:
        raise ValueError('Frozen input manifest mismatch')
    if args.output.exists():
        raise ValueError('Output must be new; preserve prior runs')
    model_path=args.model_path.resolve(strict=True)
    rows=json.loads(args.manifest.read_text('utf-8'))['rows']
    if len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate sample identity')
    for row in rows:
        if sha(row['path'])!=row['sha256']:
            raise ValueError(f"Source changed: {row['sample_id']}")
    args.output.mkdir(parents=True)
    (args.output/'vectors').mkdir()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    events=args.output/'events.jsonl'
    def event(event_type,**values):
        with events.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time=time.time(),event=event_type,**values),ensure_ascii=False)+'\n')
    event('START',manifest_sha256=args.manifest_sha256,kind=args.kind)
    started=time.perf_counter()
    try:
        import numpy as np
        import torch
        from PIL import Image,ImageOps
        from transformers import AutoModel,AutoProcessor
        torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        import runpy
        pixels=runpy.run_path(str(Path(__file__).with_name('screening_pixels.py')))
        resources=runpy.run_path(str(Path(__file__).with_name('screening_resources.py')))
        free_bytes,total_bytes=torch.cuda.mem_get_info()
        budget=resources['plan_budget'](args.resource_profile,total_bytes/2**20,(total_bytes-free_bytes)/2**20)
        torch.cuda.set_per_process_memory_fraction(budget['allocator_fraction'],device=0)
        torch.cuda.reset_peak_memory_stats()
        write(args.output/'resource-plan.json',budget)
        event('RESOURCE_ADMITTED',**budget)
        model_files={p.relative_to(model_path).as_posix():sha(p) for p in sorted(model_path.rglob('*'))
                     if p.is_file() and p.suffix in {'.json','.py','.safetensors','.bin','.model'}}
        fingerprint=hashlib.sha256(json.dumps(model_files,sort_keys=True).encode()).hexdigest()
        write(args.output/'model-files.json',model_files)
        loading=time.perf_counter()
        processor=AutoProcessor.from_pretrained(str(model_path),trust_remote_code=args.kind=='wemm',local_files_only=True)
        import psutil
        cpu_mib=min(49152,int(psutil.virtual_memory().available/2**20)-4096)
        if cpu_mib<2048:
            raise RuntimeError('WeMM host-memory budget unavailable for explicit weight offload')
        model=AutoModel.from_pretrained(str(model_path),trust_remote_code=args.kind=='wemm',local_files_only=True,
            dtype=torch.bfloat16 if args.kind=='wemm' else torch.float32,
            device_map='auto',max_memory={0:budget['gpu_weight_budget_mib']*2**20,'cpu':cpu_mib*2**20},
            offload_buffers=True,offload_folder=str(args.output/'offload')).eval()
        device_map={k:str(v) for k,v in getattr(model,'hf_device_map',{}).items()}
        if not device_map or any(v=='disk' for v in device_map.values()):
            raise RuntimeError('WeMM placement unavailable within GPU/host-memory budget; disk offload not allowed')
        write(args.output/'device-map.json',device_map)
        torch.cuda.synchronize();load_seconds=time.perf_counter()-loading
        cache={};records=[]
        event('MODEL_READY',fingerprint=fingerprint,load_seconds=load_seconds,device=torch.cuda.get_device_name(0))
        for row in rows:
            item_start=time.perf_counter();identity=row['sha256']
            cache_hit=identity in cache
            if not cache_hit:
                with Image.open(row['path']) as original:
                    im=pixels['prepare_image'](original)
                with torch.inference_mode():
                    if args.kind=='wemm':
                        message=[{'role':'user','content':[{'type':'image','image':im}]}]
                        template=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=False)
                        inputs=processor(text=template,images=[im],return_tensors='pt').to('cuda')
                        vector=model.embedding(**inputs).float()
                    else:
                        inputs=processor(images=[im],return_tensors='pt').to('cuda')
                        vector=model.get_image_features(**inputs)
                        if not isinstance(vector,torch.Tensor):vector=vector.pooler_output
                        vector=vector.float()
                    vector=torch.nn.functional.normalize(vector,dim=-1)[0].cpu().numpy()
                torch.cuda.synchronize()
                if vector.ndim!=1 or not np.isfinite(vector).all() or np.linalg.norm(vector)<.99:
                    raise ValueError('Invalid model feature vector')
                vector_path=args.output/'vectors'/f'{identity}.npy'
                np.save(vector_path,vector,allow_pickle=False)
                cache[identity]=dict(vector_path=vector_path.relative_to(args.output).as_posix(),vector_sha256=sha(vector_path))
            record=dict(sample_id=row['sample_id'],asset_sha256=identity,cache_hit=cache_hit,
                        end_to_end_seconds=time.perf_counter()-item_start,**cache[identity])
            records.append(record);event('ASSET_ENCODED',**record)
            write(args.output/'checkpoint.json',dict(status='RUNNING',completed=len(records),requested=len(rows)))
        # Inputs are checked again: avoid accepting features of a file modified during the run.
        for row in rows:
            if sha(row['path'])!=row['sha256']:raise ValueError('Source changed during extraction')
        write(args.output/'results.json',dict(status='SUCCESS',quality_status='NOT_SCORED',kind=args.kind,
            manifest_sha256=args.manifest_sha256,model_path=str(model_path),model_fingerprint=fingerprint,
            preprocessing=pixels['PREPROCESSING'],
            torch=torch.__version__,device=torch.cuda.get_device_name(0),cpu_fallback=False,
            network_model_download=False,load_seconds=load_seconds,wall_seconds=time.perf_counter()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            resource_plan=budget,device_map=device_map,
            cpu_weight_offload=any(v=='cpu' for v in device_map.values()),records=records))
        write(args.output/'checkpoint.json',dict(status='SUCCESS',completed=len(records),requested=len(rows)))
        event('SUCCESS',samples=len(records),unique_assets=len(cache))
    except Exception as exc:
        event('FAILED',error=str(exc),traceback=traceback.format_exc())
        write(args.output/'checkpoint.json',dict(status='FAILED',error=str(exc)))
        raise
    files={p.relative_to(args.output).as_posix():sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    write(args.output/'checksums.json',dict(files=files,self_excluded=True))
    print(json.dumps(dict(status='SUCCESS',samples=len(rows),kind=args.kind,output=str(args.output))))


if __name__=='__main__':main()
