"""Prepare a hash-bound input for a durable paragraph-image conversion."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import uuid


MANAGED = runpy.run_path(str(Path(__file__).with_name('managed_bridge.py')))
REPLACE = runpy.run_path(str(Path(__file__).with_name('atomic_files.py')))['replace_path']
RETRYABLE = {'FAILED', 'CANCELLED', 'INTERRUPTED', 'WAIT_EXPIRED', 'CAPACITY_INSUFFICIENT'}
NEXT_STEP = 'Poll the same child job, inspect the output and source evidence, then explicitly review the parent image reference.'


def bind_asset(service, job_id, asset_name):
    package = service.package(job_id)
    asset = (package / asset_name).resolve(strict=True)
    if not asset.is_relative_to(package / 'assets') or not asset.is_file():
        raise ValueError('Choose a generated image inside the parent assets directory')
    payload = asset.read_bytes()
    parent = service.state(job_id)
    identity = dict(parent_job_id=job_id, parent_source_sha256=parent.get('source_sha256'),
                    asset_name=asset.relative_to(package).as_posix(),
                    asset_sha256=hashlib.sha256(payload).hexdigest(),
                    operation='paragraph-image-pdf-v1')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return payload, identity, key


def verify_derivation(service, request):
    identity = request['derivation']
    _, current, key = bind_asset(service, identity['parent_job_id'], identity['asset_name'])
    expected = service.jobs.parent / 'img' / key[:20] / 'paragraph.pdf'
    if current != identity or Path(request['source']) != expected:
        raise ValueError('Paragraph image or its parent changed before conversion')


def write_pdf(payload, target):
    import fitz
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Choose a single-frame paragraph image')
        rgba = image.convert('RGBA')
        rgb = Image.new('RGB', rgba.size, 'white')
        rgb.paste(rgba, mask=rgba.getchannel('A'))
        buffer = io.BytesIO()
        rgb.save(buffer, format='PNG')
    temporary = target.with_name('t-' + uuid.uuid4().hex[:12] + '.pdf')
    with fitz.open() as pdf:
        page = pdf.new_page(width=rgb.width / 3, height=rgb.height / 3)
        page.insert_image(page.rect, stream=buffer.getvalue())
        pdf.save(temporary)
    REPLACE(temporary, target)


def convert_text_image(service, job_id, asset_name, retry, atomic_json):
    # The parent review lock serializes the lookup, preparation and submission.
    if type(retry) is not bool:
        raise ValueError('retry must be a boolean')
    payload, identity, key = bind_asset(service, job_id, asset_name)
    root = service.jobs.parent / 'img' / key[:20]
    if root.resolve() != root:
        raise ValueError('Derived-image storage escaped its assigned directory')
    scope_key = 'paragraph-image:' + key
    managed = MANAGED['paragraph_job'](service, scope_key)
    retry_of = None
    if managed:
        prior = MANAGED['status'](service, managed['job_id'])
        if prior['status'] not in RETRYABLE or not retry:
            return dict(prior, reused=True, next_step=NEXT_STEP)
        retry_of = managed['job_id']

    manifest = root / 'derivation.json'
    pdf = root / 'paragraph.pdf'
    root.mkdir(parents=True, exist_ok=True)
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError('Derived-image identity collision')
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if digest != saved['derived_source_sha256']:
            raise ValueError('Derived paragraph PDF changed')
    else:
        write_pdf(payload, pdf)
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()

    # Persist the derived-source binding before submitting. The manager's
    # scope index recovers a lost submission response without a second job.
    if not manifest.is_file():
        atomic_json(manifest, dict(derived_source_sha256=digest, **identity))
    result = MANAGED['submit_paragraph'](service, pdf, digest, identity, scope_key, retry_of)
    return dict(result, reused=False, next_step=NEXT_STEP)
