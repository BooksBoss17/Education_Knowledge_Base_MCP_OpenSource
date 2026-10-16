"""Sequential post-OCR orchestration inside the existing managed task."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import time

from .image_screening import select_assets,export_screened,read,write,sha,POLICY

def wait_classifier_capacity(mcp_root,context,work):
    """Reserve for driver/physical overhead while retaining the same safety floors."""
    import psutil
    from .host_memory_policy import current_memory_policy,sample_device,evaluate_admission
    helpers=runpy.run_path(str(Path(mcp_root)/'TOOLS/runtime_manager/bootstrap.py'))
    client=helpers['existing_client'](mcp_root);policy=current_memory_policy()
    deadline=time.monotonic()+policy.get('stage_wait_timeout_seconds',60)
    requirement=dict(extra_private_bytes=int(6*1024**3),extra_physical_bytes=int(6*1024**3),extra_gpu_mib=5.6*1024)
    while True:
        job=client.status(context['job_id'])
        if job['state']!='RUNNING':raise RuntimeError('CLASSIFICATION_PARENT_NOT_RUNNING')
        owner=psutil.Process(job['process_identity']['pid'])
        family={p.pid:p for p in [owner,*owner.children(recursive=True),psutil.Process(client.ping()['identity']['pid'])]}
        private=working=0
        for process in family.values():
            try:
                memory=process.memory_info();private+=memory.private;working+=memory.rss
            except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
        observation=sample_device();observation.update(group_current_bytes=private,group_working_bytes=working)
        verdict=evaluate_admission(policy,observation,requirement)
        write(Path(work)/'classification-admission.json',dict(verdict=verdict,requirement=requirement,observation=observation))
        if verdict['decision']=='ADMIT':return verdict
        if verdict['decision']=='CAPACITY_INSUFFICIENT' or time.monotonic()>=deadline:return verdict
        time.sleep(policy.get('stage_wait_poll_seconds',1))

def classifier_view(package,folder):
    """Bind derived pagination explicitly; never change original source type."""
    package=Path(package).resolve(strict=True);folder=Path(folder)
    report=read(package/'conversion_report.json')
    if report['source']['type']=='PDF':
        pdf=Path(report['source']['path'])
        expected=report['source']['sha256']
    elif report['source']['type']=='IMAGE' or report.get('input_transform',{}).get('route')=='DOCX_SCREENSHOT_PDF':
        pdf=package/'_bemarkdown/review_source.pdf'
        if not pdf.is_file():pdf=package/'debug/source-pagination.pdf'
        expected=(report['input_transform']['renderer'].get('pdf_sha256')
                  or report['input_transform'].get('intermediate_pdf',{}).get('sha256'))
    else:raise ValueError('SCREENING_SOURCE_PDF_UNAVAILABLE')
    if not pdf.is_file() or sha(pdf)!=expected:raise ValueError('SCREENING_SOURCE_PDF_HASH_MISMATCH')
    if folder.exists():
        binding=read(folder/'classification-view.json')
        if binding['original_markdown_sha256']!=sha(package/'document.md') or binding['pdf_sha256']!=expected:
            raise ValueError('CLASSIFICATION_VIEW_CHANGED')
        return folder
    folder.mkdir(parents=True)
    for name in ['document.md','assets_manifest.jsonl']:
        shutil.copyfile(package/name,folder/name)
    if (package/'assets').is_dir():shutil.copytree(package/'assets',folder/'assets')
    shutil.copyfile(pdf,folder/'source.pdf')
    original=report['source']
    report['source']=dict(type='PDF',path=str((folder/'source.pdf').resolve()),file_name='source.pdf',
                          name='source.pdf',sha256=expected,size_bytes=pdf.stat().st_size)
    report['classification_view_original_source']=original
    write(folder/'conversion_report.json',report)
    write(folder/'classification-view.json',dict(original_package=str(package),original_source=original,
        original_markdown_sha256=sha(package/'document.md'),pdf_sha256=expected,
        transformation='READ_ONLY_DERIVED_PDF_CONTEXT_VIEW'))
    return folder

def screen_package(package,*,mcp_root,work_root):
    package=Path(package).resolve(strict=True);mcp_root=Path(mcp_root).resolve(strict=True);work=Path(work_root).resolve()
    work.mkdir(parents=True,exist_ok=True)
    manifest_rows=[json.loads(line) for line in (package/'assets_manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not manifest_rows:
        value=dict(status='COMPLETE_NO_IMAGES',classified_count=0,excluded_count=0,protected_count=0)
        write(work/'receipt.json',value)
        return dict(package_path=str(package),image_screening=value)
    source_report=read(package/'conversion_report.json');source_type=source_report['source']['type']
    if source_type not in {'PDF','IMAGE'} and source_report.get('input_transform',{}).get('route')!='DOCX_SCREENSHOT_PDF':
        value=dict(status='DEFERRED_SOURCE_FORMAT',reason='SOURCE_PDF_CONTEXT_NOT_AVAILABLE',source_type=source_type,
                   classified_count=0,excluded_count=0,retained_count=len(manifest_rows),needs_review=True)
        write(work/'receipt.json',value)
        return dict(package_path=str(package),image_screening=value)
    selection=select_assets(package);write(work/'selection.json',selection)
    if not selection['candidate_count']:
        result=dict(status='COMPLETE_NO_CLASSIFICATION_NEEDED',protected_count=selection['protected_count'],
                    classified_count=0,excluded_count=0,raw_package_path=str(package))
        write(work/'receipt.json',result)
        return dict(package_path=str(package),image_screening=result)
    helpers=runpy.run_path(str(mcp_root/'TOOLS/runtime_manager/bootstrap.py'))
    context=helpers['trusted_context'](mcp_root)
    if not context:raise RuntimeError('SCREENING_REQUIRES_MANAGED_TASK')
    admission=wait_classifier_capacity(mcp_root,context,work)
    if admission['decision']!='ADMIT':
        value=dict(status='DEFERRED_RESOURCES',protected_count=selection['protected_count'],
            classified_count=0,excluded_count=0,candidate_count=selection['candidate_count'],needs_review=True,admission=admission)
        write(work/'receipt.json',value)
        return dict(package_path=str(package),image_screening=value)
    # Short, task-owned path required by the frozen Windows native runtime.
    short_key=hashlib.sha256((context['job_id']+str(package)+selection['markdown_sha256']).encode()).hexdigest()[:12]
    view_root=Path(os.environ['LOCALAPPDATA'])/'BSR'/'v'/short_key
    view=classifier_view(package,view_root)
    tool=mcp_root/'TOOLS/semantic_retention'
    request=dict(package_dir=str(view),source_pdf=str(view/'source.pdf'),
                 asset_names=[r['asset_name'] for r in selection['assets'] if not r['protected']])
    write(work/'classifier-request.json',request)
    runtime=runpy.run_path(str(tool/'bootstrap.py'))['runtime_config'](tool)
    result_path=work/'classifier-result.json'
    started=time.perf_counter()
    if not result_path.exists():
        command=[runtime['inference_python'],'-I','-X','utf8',str(tool/'inline_entry.py'),
                 '--request',str(work/'classifier-request.json'),'--result',str(result_path)]
        environment=os.environ.copy();environment.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        with (work/'classifier.stdout.log').open('wb') as out,(work/'classifier.stderr.log').open('wb') as err:
            completed=subprocess.run(command,env=environment,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,timeout=86400)
        if completed.returncode or not result_path.is_file():
            raise RuntimeError('INTEGRATED_CLASSIFIER_FAILED; raw package retained; see '+str(work/'classifier.stderr.log'))
    classified=read(result_path)
    if classified['parent_manager_job']!=context['job_id']:raise RuntimeError('CLASSIFIER_OWNER_CHANGED')
    receipt=dict(status='COMPLETE',policy=POLICY,semantic_job=classified['job_id'],elapsed_seconds=time.perf_counter()-started,
        original_package=str(package),selection_sha256=sha(work/'selection.json'),results_sha256=sha(result_path),
        supervisor_all_owned_exited=classified['receipt']['supervisor_all_owned_exited'])
    if not receipt['supervisor_all_owned_exited']:raise RuntimeError('CLASSIFIER_MODELS_NOT_RELEASED')
    destination=package.parent/'screened'/package.name
    applied=export_screened(package,destination,selection,classified['rows'],receipt)
    write(work/'receipt.json',dict(receipt,output=applied))
    return applied

def finish_conversion(result,*,mcp_root,work_root):
    """Single and batch fronts share the same post-conversion stage."""
    if 'documents' in result:
        for row in result['documents']:
            if row.get('status')!='PUBLISHED':continue
            updated=screen_package(row['result']['package_path'],mcp_root=mcp_root,
                                   work_root=Path(work_root)/f"document-{row['index']:04d}")
            row['result'].update(updated)
        result['image_screening_mode']='INTEGRATED_SEQUENTIAL'
    elif result.get('success'):
        result.update(screen_package(result['package_path'],mcp_root=mcp_root,work_root=work_root))
    return result

def screen_existing_conversion(package,source,output_root,*,mcp_root):
    from .production import PackageValidator
    package=Path(package).resolve(strict=True);source=Path(source).resolve(strict=True)
    PackageValidator().require_valid(package)
    manifest=read(package/'package_manifest.json');report=read(package/'conversion_report.json')
    if sha(source)!=manifest['source']['sha256']:raise ValueError('EXISTING_OCR_SOURCE_MISMATCH')
    output_root=Path(output_root).resolve();destination=output_root/package.name
    if destination.exists():raise FileExistsError('Existing output requires original job attachment, not resubmission')
    if destination==package or destination.is_relative_to(package):raise ValueError('EXISTING_OCR_OUTPUT_MUST_BE_SEPARATE')
    output_root.mkdir(parents=True,exist_ok=True);shutil.copytree(package,destination)
    value=dict(success=True,package_path=str(destination),document_id=package.name,source_sha256=manifest['source']['sha256'],
        source_type=manifest['source']['type'],quality_status=report['quality_status'],review_item_count=report['review_item_count'],
        ocr_reused=True,ocr_source_package=str(package),automatic_external_completion=False)
    return finish_conversion(value,mcp_root=mcp_root,work_root=output_root/'.image-screening')
