"""Standalone local-only GPU worker for pinned Ovis/Xiaomi payloads.

Executed as a file in a separate managed Torch environment so importing the
production Paddle process never imports Torch or depends on a Developer path.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PROMPT = ('Transcribe all visible text in the entire image faithfully in reading order. '
          'Preserve every visible number, sign, label, arrow and punctuation. '
          'Preserve two-dimensional mathematical structure using LaTeX. '
          'Do not infer missing content or correct the source. '
          'Output only the transcription, with no explanation.')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.new')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    os.replace(tmp,path)

def load_image_preprocessing():
    """Load the bundled CPU helper in this standalone managed environment."""
    path = Path(__file__).resolve().parents[2] / 'blank_images.py'
    spec = spec_from_file_location('_bemarkdown_model_image_preprocessing', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('IMAGE_PREPROCESSING_MODULE_UNAVAILABLE')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path



def run(manifest,output):
    started=time.perf_counter();manifest=Path(manifest);output=Path(output)
    payload=json.loads(manifest.read_text(encoding='utf8'))
    root=Path(payload['model_root']).resolve();model=json.loads((root/'MODEL_MANIFEST.json').read_text(encoding='utf8'))
    if model['model_id'] not in {'ovis-ocr2','xiaomi-ocr-0'}:raise ValueError('QWEN_MODEL_ID_INVALID')
    if model['model_fingerprint']!=payload['model_fingerprint']:raise ValueError('QWEN_MODEL_IDENTITY_MISMATCH')
    source_root=Path(payload['crop_root']).resolve();rows=payload['rows']
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('QWEN_DUPLICATE_INPUT_ID')
    for row in rows:
        path=Path(row['path']).resolve();path.relative_to(source_root)
        if sha(path)!=row['crop_sha256']:raise ValueError('QWEN_CROP_SHA_MISMATCH')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText,AutoProcessor
    if not torch.cuda.is_available():raise RuntimeError('GPU_REQUIRED_NO_CPU_FALLBACK')
    total=torch.cuda.get_device_properties(0).total_memory
    fraction=min(1.0,payload.get('vram_limit_mib',2816)*1024**2/total)
    torch.cuda.set_per_process_memory_fraction(fraction,0);torch.cuda.reset_peak_memory_stats()
    preprocessing, preprocessing_path = load_image_preprocessing()
    identity={'model_id':model['model_id'],'display_name':model['display_name'],
        'model_fingerprint':model['model_fingerprint'],'revision':model['upstream_revision'],
        'worker_sha256':sha(__file__),'prompt':PROMPT,'device':'cuda:0','dtype':'bfloat16',
        'max_new_tokens':payload.get('max_new_tokens',2048),'do_sample':False,'enable_thinking':False,
        'image_max_pixels':1048576,'batch_size':1,'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','pillow','huggingface-hub']},
        'image_preprocessing': {'source_sha256':sha(preprocessing_path),
            'policy':preprocessing.ASPECT_PADDING_POLICY,
            'max_aspect_ratio':preprocessing.ASPECT_PADDING_MAX_RATIO,
            'minimum_short_edge':preprocessing.ASPECT_PADDING_MIN_SHORT_EDGE,
            'maximum_derived_pixels':preprocessing.ASPECT_PADDING_MAX_PIXELS}}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    processor=AutoProcessor.from_pretrained(root,local_files_only=True)
    processor.image_processor.size={**dict(processor.image_processor.size),'longest_edge':1048576}
    load=time.perf_counter()
    network=AutoModelForImageTextToText.from_pretrained(root,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0').eval()
    load_seconds=time.perf_counter()-load;results=[]
    cache=output.parent/'prediction-cache';cache.mkdir(exist_ok=True)
    cached=0;fresh=0
    for row in rows:
        key=hashlib.sha256((row['crop_sha256']+fingerprint).encode()).hexdigest()
        cp=cache/(key[:32]+'.json')
        if cp.exists() and not payload.get('force_inference',False):
            result=json.loads(cp.read_text(encoding='utf8'))
            if result.get('cache_key')==key and result.get('output_contract_status')=='PASS':
                result.update(sample_id=row['sample_id'],source_crop_sha256=row['crop_sha256'],cache_hit=True)
                results.append(result);cached+=1;continue
        fresh+=1;begin=time.perf_counter()
        result={'sample_id':row['sample_id'],'source_crop_sha256':row['crop_sha256'],
            'model_id':model['display_name'],'model_fingerprint':model['model_fingerprint'],
            'runtime_fingerprint':fingerprint,'cache_key':key,'cache_hit':False,'identity':identity}
        image = None
        try:
            with Image.open(row['path']) as source:
                prepared, transform = preprocessing.prepare_model_image(source)
            try:
                image = prepared.convert('RGB')
            finally:
                prepared.close()
            result['image_input_transform'] = transform
            result['model_input_rgb_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
            messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda:0')
            torch.cuda.synchronize()
            with torch.inference_mode():generated=network.generate(**inputs,do_sample=False,max_new_tokens=identity['max_new_tokens'],use_cache=True,eos_token_id=processor.tokenizer.eos_token_id)
            torch.cuda.synchronize();new=generated[:,inputs['input_ids'].shape[-1]:]
            tokens=new[0].tolist();text=processor.batch_decode(new,skip_special_tokens=True)[0]
            eos=processor.tokenizer.eos_token_id
            eos_values={eos} if isinstance(eos,int) else set(eos or [])
            complete=any(t in eos_values for t in tokens)
            result.update(raw_text=text,raw_output=text,normalized_output=text,generated_token_count=len(tokens),
                terminated_by_eos=complete,output_contract_status='PASS' if complete and text.strip() else 'TRUNCATED' if not complete else 'EMPTY_OUTPUT',runtime_failure=False)
            del inputs,generated,new
        except Exception as exc:
            result.update(raw_text='',raw_output='',normalized_output='',output_contract_status='RUNTIME_FAILURE',runtime_failure=True,error=f'{type(exc).__name__}:{exc}')
            if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
        finally:
            if image is not None:
                image.close()
        result['latency_seconds']=time.perf_counter()-begin
        atomic_json(cp,result);results.append(result)
        # Durable checkpoint per item, preserving failure diagnostics.
        atomic_json(output,results)
    atomic_json(output,results)
    atomic_json(output.with_suffix('.runtime.json'),{'identity':identity,'runtime_fingerprint':fingerprint,
        'model_load_seconds':load_seconds,'total_wall_seconds':time.perf_counter()-started,
        'fresh_units':fresh,'cached_units':cached,'peak_vram_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'load_count':1,'unload_count':1,'gpu_only':True,
        'process_exit_unload':True,'failures':sum(r['runtime_failure'] for r in results)})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.manifest,args.output)
