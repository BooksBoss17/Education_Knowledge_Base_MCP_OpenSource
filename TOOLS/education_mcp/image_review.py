"""Queue one generated paragraph-image through the existing PDF converter."""
import hashlib
import io
import json
from pathlib import Path
import time
import uuid


def convert_text_image(service, job_id, asset_name, retry, atomic_json):
    # Caller holds the parent's review lock. No separate model/environment.
    package = service.package(job_id)
    asset = (package / asset_name).resolve(strict=True)
    if not asset.is_relative_to(package / 'assets') or not asset.is_file():
        raise ValueError('Choose a generated image inside the parent assets directory')
    if type(retry) is not bool:
        raise ValueError('retry must be a boolean')
    from PIL import Image
    payload = asset.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    parent = service.state(job_id)
    identity = dict(parent_job_id=job_id, parent_source_sha256=parent.get('source_sha256'),
                    asset_name=asset.relative_to(package).as_posix(), asset_sha256=digest,
                    operation='paragraph-image-pdf-v1')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Keep Windows temporary filenames below MAX_PATH; full identity is checked
    # in the ledger rather than relying on the shortened directory hash.
    root = service.jobs.parent / 'img' / key[:20]
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / 'conversion.json'
    if ledger.exists():
        saved = json.loads(ledger.read_text(encoding='utf-8'))
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError('Derived-image identity collision')
        existing = service.state(saved['job_id'])
        if existing['status'] not in {'FAILED', 'CANCELLED', 'INTERRUPTED'} or not retry:
            return dict(**existing, reused=True,
                        next_step='Poll the same child job, inspect its text, then review the parent image reference. Failed jobs require explicit retry=true.')
    with Image.open(io.BytesIO(payload)) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Choose a single-frame paragraph image')
        rgba = image.convert('RGBA')
        # White compositing preserves transparent text/line art for PDF input.
        rgb = Image.new('RGB', rgba.size, 'white')
        rgb.paste(rgba, mask=rgba.getchannel('A'))
        buffer = io.BytesIO()
        rgb.save(buffer, format='PNG')
    import fitz
    pdf_path = root / 'paragraph.pdf'
    if pdf_path.exists() and not ledger.exists():
        raise ValueError('Unregistered derived PDF exists; inspect interrupted conversion before retry')
    if not pdf_path.exists():
        with fitz.open() as pdf:
            page = pdf.new_page(width=rgb.width / 3, height=rgb.height / 3)
            page.insert_image(page.rect, stream=buffer.getvalue())
            pdf.save(pdf_path)
    elif ledger.exists() and saved.get('derived_source_sha256') != hashlib.sha256(pdf_path.read_bytes()).hexdigest():
        raise ValueError('Derived paragraph PDF changed; retain evidence and inspect before retry')
    child = uuid.uuid4().hex
    directory = service.jobs / child
    directory.mkdir()
    state = dict(job_id=child, status='QUEUED', source=str(pdf_path),
                 source_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                 created_at=time.time(), server_pid=__import__('os').getpid(),
                 derivation=identity,
                 scope='One paragraph-image conversion; parent Markdown unchanged until explicit review')
    atomic_json(directory / 'state.json', state)
    atomic_json(ledger, dict(job_id=child, derived_source_sha256=state['source_sha256'], **identity))
    service.futures[child] = service.pool.submit(service.convert, child)
    return dict(**state, reused=False,
                next_step='Poll child job, inspect Markdown and unresolved items; use source-supported bemarkdown_review on the parent. Do not discard meaningful diagrams from mixed images.')
