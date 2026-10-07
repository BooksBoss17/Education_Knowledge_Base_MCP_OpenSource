"""Group and triage output images without confusing candidates with permission.

Image equality saves inspection work; it does not make an image disposable.
The caller supplies parsed Markdown references so every occurrence survives.
"""
from __future__ import annotations

import hashlib
from collections import Counter
import json
from pathlib import Path
import re
import runpy
import time

from PIL import Image, ImageOps

POLICY_VERSION = 'asset-screening-v2'
FEATURE_VERSION = 'pillow-rgb-256-histogram-v3-full-resolution-white-proof'
PROTECTED_ADAPTERS = {'CONTENT_REVIEW_REQUIRED', 'TABLE_ENGINE', 'FORMULA_ENGINE'}
PROTECTED_KINDS = {'FORMULA', 'TABLE', 'EQUATION', 'MATH', 'TEXT_IMAGE'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def visual_features(path):
    """Cheap candidate features, never an automatic content-deletion rule."""
    with Image.open(path) as original:
        if getattr(original, 'n_frames', 1) != 1:
            return {'status': 'UNSUPPORTED_MULTIFRAME'}
        im = ImageOps.exif_transpose(original).convert('RGBA')
        # Inspect original pixels before compositing/resizing. Even one faint
        # pixel or alpha variation prevents machine omission.
        extrema = im.getextrema()
        pixel_empty = all(channel == (255, 255) for channel in extrema)
        background = Image.new('RGBA', im.size, 'white')
        background.alpha_composite(im)
        im = background.convert('RGB')
        width, height = im.size
        im.thumbnail((256, 256))
        count = im.width * im.height
        colors = im.point([value // 16 * 16 for value in range(256)] * 3).getcolors(4096)
        dark = sum(im.convert('L').histogram()[:140])
    counts = sorted((n for n, _ in colors), reverse=True)
    return dict(status='OK', width=width, height=height, measured_pixels=count,
                dark_fraction=dark/count,
                top2_color_fraction=sum(counts[:2])/count,
                feature_version=FEATURE_VERSION,
                empty_pixel_proof=dict(version='opaque-exact-white-v1',
                    full_resolution=True, rgba_extrema=extrema,
                    full_resolution_opaque_white=pixel_empty))


def protection_reasons(record):
    """Only use recorded semantics; no guessed region identity from coordinates."""
    reasons = []
    provenance = record.get('provenance') or {}
    if record.get('adapter') in PROTECTED_ADAPTERS or provenance.get('adapter') in PROTECTED_ADAPTERS:
        reasons.append('CONTENT_RECOGNITION_OR_REVIEW_ROUTE')
    for value in (record.get('semantic_type'), record.get('kind'), provenance.get('semantic_type')):
        if isinstance(value, str) and value.upper() in PROTECTED_KINDS:
            reasons.append('PROTECTED_CONTENT_KIND')
    if record.get('review_state') not in (None, '', 'NONE', 'RESOLVED'):
        reasons.append('UNRESOLVED_REVIEW_STATE')
    if not record:
        reasons.append('MISSING_ASSET_PROVENANCE')
    return sorted(set(reasons))


def screen(package, references, *, source_sha256, markdown_sha256, offset=0, limit=100,
           source_path=None, similarity_by_sha=None, view='all'):
    """Return paged SHA groups and whole-package counts; write nothing.

    A group may mix different page roles. All occurrence identities and all
    protection reasons are retained. Low-information pixels only set priority.
    """
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 500:
        raise ValueError('Invalid screening window')
    if view not in {'all', 'candidates', 'triage'}:
        raise ValueError('Invalid screening view')
    started = time.perf_counter()
    package = Path(package).resolve(strict=True)
    manifest = package/'assets_manifest.jsonl'
    records = {}
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                name = row.get('relative_path')
                if name in records:
                    raise ValueError('Ambiguous duplicate asset manifest path')
                if name:
                    records[name] = row
    groups = {}
    checked = {}
    reference_counts=Counter(ref['asset_name'] for ref in references)
    for ref in references:
        name = ref['asset_name']
        path = (package/name).resolve(strict=True)
        if not path.is_relative_to(package) or not path.is_file():
            raise ValueError('Screening image escapes package')
        if name not in checked:
            checked[name] = digest(path)
        actual = checked[name]
        if actual != ref['sha256']:
            raise ValueError('Image changed during screening')
        record = records.get(name, {})
        declared = record.get('content_sha256') or record.get('sha256')
        if declared and declared != actual:
            raise ValueError('Asset manifest SHA mismatch')
        group = groups.setdefault(actual, dict(asset_sha256=actual, representative_asset=name,
            occurrences=[], protection_reasons=[], authorized_to_exclude=False,
            coverage_status='NOT_VERIFIED', proposed_action='KEEP_PENDING_REVIEW'))
        mapped=ref.get('source_mapping')=='RENDERER_SPAN_AND_REVIEW_REPLAY'
        unique=reference_counts[name]==1
        part=ref.get('source_part',record.get('source_part')) if mapped or unique else None
        box=ref.get('bbox_pdf_pt',(record.get('provenance') or {}).get('bbox_pdf_pt')) if mapped or unique else None
        group['occurrences'].append(dict(image_id=ref['image_id'], asset_name=name, line=ref['line'],
            source_part=part, asset_uid=record.get('asset_uid'),
            source_region_ids=None if mapped or not unique else record.get('source_region_ids'),
            source_content_id=None if mapped or not unique else record.get('source_content_id'),
            source_document_node_id=ref.get('source_document_node_id'),source_mapping=ref.get('source_mapping'),
            source_document_kind=ref.get('source_document_kind'),source_node_review_state=ref.get('source_node_review_state'),
            source_request=ref.get('source_request') if mapped or unique else None,
            bbox_pdf_pt=box,
            empty_provenance_valid=(declared == actual and record.get('source_type') == 'PDF'
                and record.get('semantic_type') == 'PDF_IMAGE' and record.get('status') == 'RESOLVED'
                and isinstance(part, str) and part.startswith('page:'))))
        group['protection_reasons'].extend(protection_reasons(record))
        group['protection_reasons'].extend(ref.get('screening_protection_reasons',[]))
        if not mapped and not unique:
            group['protection_reasons'].append('SOURCE_OCCURRENCE_UNRESOLVED')
    ordered = list(groups.values())
    candidate_count = 0
    for group in ordered:
        feature = visual_features(package/group['representative_asset'])
        group['features'] = feature
        low_information = (feature.get('status') == 'OK' and feature['dark_fraction'] < .001
                           and feature['top2_color_fraction'] > .95)
        group['protection_reasons'] = sorted(set(group['protection_reasons']))
        group['empty_proof'] = dict(
            **feature.get('empty_pixel_proof', {}),
            eligible=(bool(feature.get('empty_pixel_proof', {}).get('full_resolution_opaque_white'))
                and not group['protection_reasons']
                and all(o['empty_provenance_valid'] for o in group['occurrences'])))
        group['rule_candidate'] = low_information
        similarity = (similarity_by_sha or {}).get(group['asset_sha256'])
        group['similarity'] = similarity
        candidate = low_information or bool(similarity and similarity.get('template_candidate'))
        # Source protection must not hide the flag that needs contextual review.
        group['triage_reasons'] = (['LOW_INFORMATION_PIXELS'] if low_information else []) + (
            ['SIMILAR_TO_DECORATION_TEMPLATE'] if similarity and similarity.get('template_candidate') else [])
        group['classification'] = ('CONTENT_REVIEW' if group['protection_reasons'] else
            'EMPTY_ASSET' if group['empty_proof']['eligible'] else
            'DECORATION_CANDIDATE' if candidate else 'UNKNOWN')
        if group['empty_proof']['eligible']:
            group['proposed_action'] = 'OMIT_EMPTY_REFERENCE'
            group['exclusion_basis'] = 'FULL_RESOLUTION_OPAQUE_WHITE_PIXELS'
        candidate_count += group['classification'] == 'DECORATION_CANDIDATE'
        group['occurrence_count'] = len(group['occurrences'])
    if source_path is not None:
        apply_source_guards(ordered,source_path,source_sha256)
    candidate_count = sum(g['classification']=='DECORATION_CANDIDATE' for g in ordered)
    triage=[g for g in ordered if g['triage_reasons'] and g['classification']!='EMPTY_ASSET']
    visible=(ordered if view=='all' else triage if view=='triage' else
             [g for g in ordered if g['classification']=='DECORATION_CANDIDATE'])
    return dict(schema='education-asset-screening-v1', policy_version=POLICY_VERSION,
        source_sha256=source_sha256, base_sha256=markdown_sha256,
        images_total=len(references), distinct_images=len(ordered),
        repeated_occurrences=len(references)-len(ordered), candidate_groups=candidate_count,
        triage_groups=len(triage),triage_occurrences=sum(g['occurrence_count'] for g in triage),
        actual_exclusions=0, view=view, view_groups=len(visible), groups=visible[offset:offset+limit],
        automatic_omission_groups=sum(g['empty_proof']['eligible'] for g in ordered),
        automatic_omission_occurrences=sum(len(g['occurrences']) for g in ordered if g['empty_proof']['eligible']),
        automatic_empty_actions=_empty_actions(ordered),
        next_offset=offset+limit if offset+limit < len(visible) else None,
        elapsed_seconds=time.perf_counter()-started,
        limits='Pixel rules only prioritize candidates. Repeated SHA does not authorize deletion; '
               'unread formulas, tables, figures and source coverage remain protected.')


def apply_source_guards(groups,source_path,expected_sha256):
    """Evaluate every occurrence; one conflicted occurrence protects the group."""
    source_path=Path(source_path).resolve(strict=True)
    if digest(source_path)!=expected_sha256:raise ValueError('Source changed before screening')
    guard=runpy.run_path(str(Path(__file__).with_name('screening_source_guard.py')))
    candidates=[g for g in groups if g['classification']=='DECORATION_CANDIDATE']
    if not candidates:return
    if source_path.suffix.lower()!='.pdf':
        for group in candidates:
            group['source_guard']=dict(status='UNKNOWN',reasons=['SOURCE_CONTEXT_NOT_NATIVE_PDF'])
        return
    import fitz
    with fitz.open(source_path) as document:
        pages={}
        for group in candidates:
            results=[]
            for occurrence in group['occurrences']:
                part=occurrence.get('source_part') or ''
                match=re.fullmatch(r'page:(\d+)',part)
                page_index=int(match.group(1)) if match else None
                if page_index is not None and 0<=page_index<len(document):
                    if page_index not in pages:pages[page_index]=guard['page_evidence'](document[page_index])
                    evidence=pages[page_index]
                else:evidence=None
                results.append(dict(image_id=occurrence['image_id'],page_index=page_index,
                    **guard['assess'](occurrence.get('bbox_pdf_pt'),evidence)))
            group['source_guard']=dict(status='PROTECTED' if any(r['status']=='PROTECTED' for r in results) else
                'UNKNOWN' if any(r['status']=='UNKNOWN' for r in results) else 'NO_KNOWN_NATIVE_CONFLICT',occurrences=results)
            if group['source_guard']['status']=='PROTECTED':
                group['classification']='CONTENT_REVIEW'
                group['protection_reasons'].append('SOURCE_CONTEXT_CONFLICT')


def empty_reference_actions(package, references, *, source_sha256, markdown_sha256):
    """Prove empty pixels again at publication; never remove source assets.

    This is a content-free reference omission, not a semantic decoration verdict.
    Original conversion review items and source coverage status stay unchanged.
    """
    result = screen(package, references, source_sha256=source_sha256,
                    markdown_sha256=markdown_sha256, limit=500)
    return result['automatic_empty_actions']


def _empty_actions(groups):
    actions = []
    for group in groups:
        if not group['empty_proof']['eligible']:
            continue
        evidence = json.dumps(dict(policy=POLICY_VERSION,asset_sha256=group['asset_sha256'],
            pixel_proof=group['empty_proof'],scope='Omit empty reference only; retain original asset and unresolved source reviews'),sort_keys=True)
        for occurrence in group['occurrences']:
            actions.append(dict(image_id=occurrence['image_id'],action='remove_empty',
                                text='',source_evidence=evidence))
    return actions
