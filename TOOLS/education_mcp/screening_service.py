"""Asynchronous optional WeMM screening using the existing shared GPU queue."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import time
import uuid

HERE=Path(__file__).resolve().parent
SIM=runpy.run_path(str(HERE/'screening_similarity.py'))
FEATURES=runpy.run_path(str(HERE/'asset_screening.py'))
RESOURCES_PATH=HERE/'screening_resources.py'
RESOURCES=runpy.run_path(str(RESOURCES_PATH))
PIXELS_PATH=HERE/'screening_pixels.py'
PIXELS=runpy.run_path(str(PIXELS_PATH))


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(uuid.uuid4().hex[:12]+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def compute(service,run_id,run_root,manifest,model_path,conversion_lock,profile='6gb'):
    """One local process owns GPU; source/model changes fail closed."""
    state_path=run_root/'state.json'
    gpu=dict(profile=profile,limit_mib=RESOURCES['LIMITS'][profile],peak_mib=0,
             samples=0,measurement_errors=0,budget_exceeded=False,scope='WHOLE_DEVICE_SAMPLED')
    try:
        with conversion_lock(service.jobs.parent/'gpu.lock',service.closing):
            if service.closing.is_set():raise RuntimeError('Service closing')
            write(state_path,dict(status='RUNNING',started=time.time()))
            command=[str(service.python),'-X','utf8',str(HERE/'screening_model_worker.py'),
                '--manifest',str(manifest),'--manifest-sha256',SIM['sha'](manifest),
                '--model-path',str(model_path),'--kind','wemm','--output',str(run_root/'extraction'),
                '--resource-profile',profile]
            write(run_root/'command.json',dict(command=command,created=time.time()))
            with (run_root/'stdout.log').open('w',encoding='utf-8') as out,(run_root/'stderr.log').open('w',encoding='utf-8') as err:
                process=subprocess.Popen(command,stdout=out,stderr=err,cwd=run_root)
                service.processes[run_id]=process
                try:
                    deadline=time.monotonic()+7200
                    with (run_root/'gpu-samples.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                        while process.poll() is None:
                            if service.closing.is_set():raise RuntimeError('Screening interrupted by service shutdown')
                            if time.monotonic()>deadline:raise RuntimeError('Screening exceeded execution limit')
                            used=service.workspace_manager.sample_gpu_usage()
                            log.write(json.dumps(dict(time=time.time(),used_mib=used,limit_mib=gpu['limit_mib']))+'\n')
                            if used is None:
                                gpu['measurement_errors']+=1
                                if gpu['measurement_errors']>=3:raise RuntimeError('Screening GPU measurements unavailable')
                            else:
                                gpu['samples']+=1;gpu['peak_mib']=max(gpu['peak_mib'],used)
                                if used>gpu['limit_mib']:
                                    gpu['budget_exceeded']=True
                                    raise RuntimeError('Screening GPU budget exceeded; owned worker stopped')
                            time.sleep(.25)
                    code=process.returncode
                except Exception:
                    if process.poll() is None:process.kill()
                    process.wait(timeout=20)
                    raise
                finally:service.processes.pop(run_id,None)
            if code:raise RuntimeError(f'Screening worker exited {code}; see stderr.log')
            write(state_path,dict(status='SUCCESS',finished=time.time(),gpu_usage=gpu))
    except Exception as exc:
        write(state_path,dict(status='FAILED',error=str(exc),finished=time.time(),gpu_usage=gpu))


def reusable_pixel_run(service, identity, package, template_sha):
    """Find a sealed or live same-package pixel superset, never old decisions."""
    required={tuple(row) for row in identity['assets']}
    compatible=('source_sha256','resource_profile','resource_policy_sha256',
                'preprocessing','pixel_policy_sha256','worker_sha256')
    candidates=[]
    for root in sorted((service.jobs.parent/'as').glob('*')):
        if not root.is_dir() or not (root/'identity.json').is_file():
            continue
        old=json.loads((root/'identity.json').read_text('utf-8'))
        if any(old.get(key)!=identity[key] for key in compatible):
            continue
        if old.get('cache_schema')=='pixel-set-v1':
            if old.get('model_fingerprint')!=identity['model_fingerprint'] or old.get('package_root')!=str(package.resolve()):
                continue
        elif old.get('cache_schema') is not None or old.get('templates_sha256')!=template_sha:
            continue
        assets={tuple(row) for row in old['assets']}
        if not required<=assets:
            continue
        cache_id=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()
        if root.name!=cache_id[:24]:
            raise ValueError('Screening cache identity collision or corruption')
        status=json.loads((root/'state.json').read_text('utf-8'))['status']
        if status!='SUCCESS' and not (status in {'QUEUED','RUNNING'} and 'screen-'+cache_id in service.futures):
            continue
        rows=json.loads((root/'manifest.json').read_text('utf-8'))['rows']
        if any(not Path(row['path']).resolve().is_relative_to(package.resolve()) for row in rows):
            continue
        candidates.append((0 if status=='SUCCESS' else 1,len(assets),cache_id,old))
    if candidates:
        _,_,cache_id,old=min(candidates,key=lambda item:item[:3])
        return cache_id,old
    return None


def request(service,job_id,flow,conversion_lock,*,offset=0,limit=100,view='all'):
    """Polling the same current source/Markdown resumes the same feature run."""
    if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=500:
        raise ValueError('Invalid screening window')
    if view not in {'all','candidates','triage'}:raise ValueError('Invalid screening view')
    state,package,path,text=flow['current'](service,job_id)
    source_sha=state['source_sha256']
    if SIM['sha'](state['source'])!=source_sha:raise ValueError('Source changed before screening')
    base_sha=SIM['sha'](path);found=flow['images'](package,text)
    if not found:
        return dict(status='SUCCESS',screening_model='wemm',images_total=0,distinct_images=0,groups=[],actual_exclusions=0)
    template_path=HERE/'screening_templates.json'
    templates=json.loads(template_path.read_text('utf-8'))
    if templates.get('status') == 'NOT_CONFIGURED_PUBLIC':
        return dict(status='NOT_CONFIGURED', screening_model='wemm', actual_exclusions=0, reason='Provide licensed positive/protected retrieval templates; private templates are not distributed')
    if templates.get('preprocessing')!=PIXELS['PREPROCESSING']:
        raise ValueError('Screening template preprocessing mismatch; rebuild or verify templates first')
    manager=getattr(service,'workspace_manager',None)
    profile=(manager.state()['gpu'].get('profile') if manager is not None else state.get('resource_profile')) or '6gb'
    if profile not in RESOURCES['LIMITS']:raise ValueError('Invalid screening GPU profile')
    model_path=service.mcp_root/'MODELS/WeMM-Embedding-4B'
    if not model_path.is_dir():raise ValueError('Registered WeMM model is unavailable')
    identity=dict(cache_schema='pixel-set-v1',source_sha256=source_sha,package_root=str(package.resolve()),
        model_fingerprint=templates['model_fingerprint'],
        resource_profile=profile,resource_policy_sha256=SIM['sha'](RESOURCES_PATH),
        preprocessing=PIXELS['PREPROCESSING'],pixel_policy_sha256=SIM['sha'](PIXELS_PATH),
        worker_sha256=SIM['sha'](HERE/'screening_model_worker.py'),
        assets=[list(row) for row in sorted({(r['asset_name'],r['sha256']) for r in found})])
    cache_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    reuse=reusable_pixel_run(service,identity,package,SIM['sha'](template_path))
    if reuse is not None:
        cache_id,identity=reuse
    run_root=service.jobs.parent/'as'/cache_id[:24];manifest=run_root/'manifest.json'
    if (run_root/'identity.json').exists() and json.loads((run_root/'identity.json').read_text('utf-8'))!=identity:
        raise ValueError('Screening cache identity collision or corruption')
    run_id='screen-'+cache_id;state_path=run_root/'state.json'
    previous=json.loads(state_path.read_text('utf-8')) if state_path.exists() else None
    if previous and previous['status']=='SUCCESS':
        result,records,vectors=SIM['load_extraction'](run_root/'extraction',SIM['sha'](manifest))
        if result['model_fingerprint']!=templates['model_fingerprint']:
            raise ValueError('WeMM model changed relative to screening templates')
        if result.get('preprocessing')!=templates['preprocessing']:
            raise ValueError('Extraction preprocessing does not match screening templates')
        expected={r['sha256'] for r in found}
        if not expected<=set(vectors):raise ValueError('Screening vectors do not cover current images')
        cached_count=len(vectors)
        vectors={key:vectors[key] for key in expected}
        scores=SIM['rank_queries'](vectors,templates['templates'],model_fingerprint=result['model_fingerprint'],
            cosine_min=templates['cosine_min'],margin_min=templates['margin_min'])
        flow['source_requests'](state,package,found)
        output=FEATURES['screen'](package,found,source_sha256=source_sha,markdown_sha256=base_sha,
            source_path=state['source'],similarity_by_sha=scores,offset=offset,limit=limit,view=view)
        output.update(status='SUCCESS',screening_model='wemm',cache_id=cache_id,
            vector_cache_asset_count=cached_count,requested_distinct_images=len(expected),
            extraction_load_seconds=result['load_seconds'],extraction_wall_seconds=result['wall_seconds'])
        return output
    if previous and previous['status']=='FAILED':return dict(**previous,cache_id=cache_id)
    if previous and run_id not in service.futures:
        return dict(status='INTERRUPTED',cache_id=cache_id,reason='Prior worker is no longer owned by this service; preserved for review')
    if not previous:
        run_root.mkdir(parents=True,exist_ok=False)
        unique={r['sha256']:r for r in found}
        write(manifest,dict(rows=[dict(sample_id=s,path=str(package/r['asset_name']),sha256=s) for s,r in unique.items()]))
        write(run_root/'identity.json',identity)
        write(state_path,dict(status='QUEUED',created=time.time()))
        service.futures[run_id]=service.pool.submit(compute,service,run_id,run_root,manifest,model_path,conversion_lock,profile)
        previous=dict(status='QUEUED')
    progress=run_root/'extraction/checkpoint.json'
    return dict(**previous,cache_id=cache_id,base_sha256=base_sha,
        vector_cache_asset_count=len({row[1] for row in identity['assets']}),
        requested_distinct_images=len({r['sha256'] for r in found}),
        progress=json.loads(progress.read_text('utf-8')) if progress.exists() else None,
        poll_action='textbook_organize(action=screen_images, screening_model=wemm)',actual_exclusions=0)
