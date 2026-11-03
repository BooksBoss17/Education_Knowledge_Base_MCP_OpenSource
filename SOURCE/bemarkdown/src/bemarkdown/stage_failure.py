"""Persist native loading failures without importing inference frameworks."""
from __future__ import annotations
import hashlib,json,os,re,shutil,tempfile,time
from pathlib import Path

def retain_stage_failure(folder, *, stage, exit_code=None, error=None):
    folder=Path(folder)
    root=Path(os.environ.get('BEMARKDOWN_DIAGNOSTICS_ROOT',
        str(Path(os.environ.get('LOCALAPPDATA',Path.home()/'.local/share'))/'BeMarkdown/diagnostics')))
    root.mkdir(parents=True,exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix='stage-failure-',dir=root))
    records=[]
    for source in folder.iterdir():
        if not source.is_file():continue
        # Retain protocol/configuration and native logs, not input image copies.
        if source.suffix.lower() not in {'.json','.jsonl','.log'}:continue
        target=destination/source.name
        shutil.copy2(source,target)
        records.append({'name':source.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    message='\n'.join(path.read_text(encoding='utf-8',errors='replace')
        for name in ('stderr.log','worker.log') if (path:=folder/name).exists())
    is_1455=bool(re.search(r'(?:os error|WinError|winerror[=: ]+)\s*1455',message,re.I))
    error_type=type(error).__name__ if error is not None else None
    admission_refused=error_type in {'ForecastAdmissionError','ResourceAdmissionError'}
    classification=('PROJECTED_RESOURCE_ADMISSION_REFUSED' if getattr(error,'projected',None) is not None else
                    'FORECAST_QUALIFICATION_REFUSED' if error_type=='ForecastAdmissionError' else
                    'RESOURCE_ADMISSION_REFUSED' if admission_refused else
                    'WINDOWS_1455_LOADING_FAILURE_CAUSE_UNDETERMINED' if is_1455 else
                    'UNDETERMINED_STAGE_FAILURE')
    report={'schema':'native-stage-failure-v1','stage':stage,'exit_code':exit_code,
        'os_error':1455 if is_1455 else None,'classification':classification,
        'error_type':error_type,'admission_refused':admission_refused,
        'projected_resources':getattr(error,'projected',None),
        'resource_cause_confirmed':False,'error':str(error) if error else None,
        'original_diagnostics':str(folder),'persistent_diagnostics':str(destination),
        'files':records,'recorded_epoch':time.time(),
        'retry_count':0,'retry_eligible':False,
        'retry_reason':'No independently verified resource improvement; automatic retry is not eligible.'}
    (destination/'failure.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report
