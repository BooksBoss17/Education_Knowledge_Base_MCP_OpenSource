"""Opt-in task-scoped admission before loading a model; no GPU imports here."""
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path

CONTEXT_ENV='BEMARKDOWN_STAGE_ADMISSION_CONTEXT'

def require_stage_admission(stage: str) -> None:
    raw=os.environ.get(CONTEXT_ENV)
    if not raw:
        return
    context=json.loads(raw)
    runner=Path(context['runner_path'])
    if hashlib.sha256(runner.read_bytes()).hexdigest()!=context['runner_sha256']:
        raise RuntimeError('STAGE_ADMISSION_RUNNER_CHANGED')
    process=subprocess.run([context['python'],str(runner),'--caller-pid',str(os.getpid()),'--stage',stage],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if process.returncode:
        raise RuntimeError(f'STAGE_ADMISSION_REFUSED:{stage}:exit={process.returncode}:journal={context["journal_path"]}')
