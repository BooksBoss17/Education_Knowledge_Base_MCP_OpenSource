"""Bind a verified residual image transcript to an existing complete text node.

Some PDF renders deliberately omit semantic formula owners. Their visible text
can already be delivered by a full body node. This module proves that relation;
it never inserts the masked formula into the image transcription or certifies it.
"""
import copy
import hashlib
import json
import math
import re

VERSION = 'source-residual-existing-delivery-v1'
_MATH = re.compile(r'\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]*\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]')


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _geometry_sha(node):
    return _sha(json.dumps({k:node.get(k) for k in ('node_id','kind','page_index','bbox_pdf_pt','visibility','content')}
        | {'route':node.get('provenance',{}).get('route_provenance'),
           'route_id':node.get('provenance',{}).get('route_id')}, sort_keys=True, ensure_ascii=False))


def _visible_text(value):
    value = re.sub(r'\s+', ' ', value).strip()
    # Preserve ASCII word/number separators; ignore only layout whitespace.
    return re.sub(r'(?<![A-Za-z0-9]) | (?![A-Za-z0-9])', '', value)


def _rect(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        rect = tuple(float(x) for x in value)
    except (ValueError, TypeError):
        return None
    return rect if all(math.isfinite(x) for x in rect) and rect[2] > rect[0] and rect[3] > rect[1] else None


def _same_rect(a, b):
    a, b = _rect(a), _rect(b)
    return bool(a and b and all(abs(x-y) <= .001 for x,y in zip(a,b)))


def _contains(outer, inner):
    outer, inner = _rect(outer), _rect(inner)
    return bool(outer and inner and outer[0]-.001 <= inner[0] and outer[1]-.001 <= inner[1]
                and outer[2]+.001 >= inner[2] and outer[3]+.001 >= inner[3])


def find_residual_text_coverage(document, image, answer):
    """Return a unique, source-bound coverage proof, or None for normal delivery."""
    from .pdf.adaptive_ocr import canonical
    if image.get('kind') != 'IMAGE' or answer.get('status') != 'SUCCESS' or answer.get('uncertain'):
        return None
    evidence = answer.get('repair_result', {})
    if not evidence.get('source_verified') or evidence.get('evidence_kind') != 'SOURCE_IMAGE_VERIFIED':
        return None
    transcript = answer.get('markdown', '')
    task = answer.get('repair_task', {})
    source_sha = document.get('source', {}).get('sha256') or document.get('source', {}).get('source_sha256')
    if not transcript.strip() or not source_sha or task.get('source_sha256') != source_sha:
        return None
    original_sha, actual_sha = answer.get('original_image_sha256'), answer.get('actual_input_sha256')
    if not original_sha or not actual_sha:
        return None
    uid = image.get('content', {}).get('asset_uid', '')
    if uid.startswith('asset-sha256-') and uid != 'asset-sha256-'+original_sha:
        return None
    route = image.get('provenance', {}).get('route_provenance', {})
    owners = route.get('source_graphics_ownership', {}).get('excluded_owners', [])
    rendering = route.get('render_crop', {}).get('source_graphics_render', {})
    masks = rendering.get('semantic_image_excluded_regions_pdf_pt', [])
    if not owners or rendering.get('method') != 'MUPDF_SOURCE_GRAPHICS_WITH_SEMANTIC_IMAGE_OWNERSHIP' or len(masks) != len(owners):
        return None
    page = image.get('page_index')
    formulas = []
    for owner in owners:
        if owner.get('output_kind') != 'FORMULA' or owner.get('page_index') != page:
            return None
        if sum(_same_rect(owner.get('bbox_pdf_pt'), mask) for mask in masks) != 1:
            return None
        nodes = [n for n in document['blocks'] if n.get('kind') == 'FORMULA'
                 and n.get('page_index') == page and n.get('visibility', 'VISIBLE') == 'VISIBLE'
                 and n.get('provenance', {}).get('route_id') == owner.get('route_id')
                 and _same_rect(n.get('bbox_pdf_pt'), owner.get('bbox_pdf_pt'))]
        if len(nodes) != 1 or not nodes[0].get('content', {}).get('latex', '').strip():
            return None
        formulas.append((owner, nodes[0]))
    if len({n['node_id'] for _,n in formulas}) != len(formulas):
        return None
    candidates = []
    for body in document['blocks']:
        if body.get('kind') not in {'TEXT','TITLE','CAPTION'} or body.get('page_index') != page or body.get('visibility', 'VISIBLE') != 'VISIBLE':
            continue
        if not _contains(image.get('bbox_pdf_pt'), body.get('bbox_pdf_pt')):
            continue
        if any(not _contains(body.get('bbox_pdf_pt'), owner.get('bbox_pdf_pt')) for owner,_ in formulas):
            continue
        text = body.get('content', {}).get('text', '')
        spans, excluded = list(_MATH.finditer(text)), []
        for owner, node in formulas:
            matches = [s for s in spans if canonical(s.group()) == canonical(node['content']['latex'])]
            if len(matches) != 1:
                break
            span = matches[0]
            excluded.append({'formula_node_id':node['node_id'], 'formula_latex_sha256':_sha(node['content']['latex']),
                'formula_geometry_sha256':_geometry_sha(node),
                'route_id':owner['route_id'], 'bbox_pdf_pt':owner['bbox_pdf_pt'],
                'body_character_range':[span.start(),span.end()], 'body_formula_sha256':_sha(span.group())})
        else:
            ordered = sorted(excluded, key=lambda r:r['body_character_range'])
            cursor, pieces = 0, []
            for record in ordered:
                a,b = record['body_character_range']
                if a < cursor:
                    break
                pieces.append(text[cursor:a]);cursor=b
            else:
                pieces.append(text[cursor:]); residual=''.join(pieces)
                if _visible_text(residual) == _visible_text(transcript):
                    candidates.append({'body_node_id':body['node_id'], 'body_text_sha256':_sha(text),
                        'body_geometry_sha256':_geometry_sha(body),
                        'visible_residual_normalized_sha256':_sha(_visible_text(residual)), 'excluded_owners':ordered})
    if len(candidates) != 1:
        return None
    return {'version':VERSION, 'scope':'ONLY_VISIBLE_RESIDUAL_TEXT', 'asset_uid':uid,
            'image_node_id':image['node_id'], 'page_index':page, 'source_document_sha256':source_sha,
            'image_geometry_sha256':_geometry_sha(image),
            'original_image_sha256':original_sha, 'actual_input_sha256':actual_sha,
            'visible_transcript_sha256':_sha(transcript), 'source_review':copy.deepcopy(answer.get('source_review', {})),
            'excluded_formulas_newly_certified':False, 'full_page_image_transcription_claimed':False,
            'body_text_newly_modified':False, **candidates[0]}


def bind_residual_render_coverage(document, rendered):
    """Seal the actual emitted body bytes; reject stale proofs instead of hiding them."""
    nodes = {n['node_id']:n for n in document['blocks']}
    data = rendered['markdown'].encode('utf-8')
    links = []
    for image in document['blocks']:
        proof = image.get('provenance', {}).get('residual_text_coverage')
        if not proof:
            continue
        source_sha = document.get('source', {}).get('sha256') or document.get('source', {}).get('source_sha256')
        if source_sha != proof['source_document_sha256'] or _geometry_sha(image) != proof['image_geometry_sha256']:
            raise ValueError('RESIDUAL_DELIVERY_SOURCE_CHANGED')
        body = nodes.get(proof['body_node_id'])
        if not body or _sha(body.get('content', {}).get('text', '')) != proof['body_text_sha256'] or _geometry_sha(body) != proof['body_geometry_sha256']:
            raise ValueError('RESIDUAL_DELIVERY_BODY_CHANGED')
        for owner in proof['excluded_owners']:
            formula = nodes.get(owner['formula_node_id'])
            if not formula or _sha(formula.get('content', {}).get('latex', '')) != owner['formula_latex_sha256'] or _geometry_sha(formula) != owner['formula_geometry_sha256']:
                raise ValueError('RESIDUAL_DELIVERY_OWNER_CHANGED')
        spans = [s for s in rendered['node_spans'] if s['node_id'] == proof['body_node_id']]
        if len(spans) != 1:
            raise ValueError('RESIDUAL_DELIVERY_BODY_NOT_UNIQUELY_RENDERED')
        span = spans[0]; a,b = span['byte_start'],span['byte_end']
        if not 0 <= a <= b <= len(data) or hashlib.sha256(data[a:b]).hexdigest() != span['sha256']:
            raise ValueError('RESIDUAL_DELIVERY_RENDER_BYTES_CHANGED')
        from .pdf_output_audit import CleanHandoffRenderer
        expected, _ = CleanHandoffRenderer()._render_block(body, {})
        if data[a:b].decode('utf-8') != (expected.strip() or '[Unresolved source content]'):
            raise ValueError('RESIDUAL_DELIVERY_EMITTED_BODY_MISMATCH')
        links.append({**copy.deepcopy(proof), 'markdown_sha256':hashlib.sha256(data).hexdigest(),
                      'body_render_span':copy.deepcopy(span)})
    return links
