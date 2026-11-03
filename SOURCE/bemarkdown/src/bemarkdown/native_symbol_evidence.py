"""Source-bound DOCX alternative text corroboration for short visible symbols.

Alternative text alone never authorizes transcription. The optional gate keeps
the image and requires a verified primary reading plus matching local readings.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
import zipfile
from pathlib import Path

from .pdf.adaptive_ocr import canonical

VERSION = 'native-alt-symbol-corroboration-v1'
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
      'pic':'http://schemas.openxmlformats.org/drawingml/2006/picture',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'mc':'http://schemas.openxmlformats.org/markup-compatibility/2006'}


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def collect_docx_native_alts(source, source_sha256, assets, package):
    """Read direct body drawing descriptors tied to the exact emitted media.

    Derived regions, grouped pictures, external relationships, and conditional
    XML branches are deliberately unsupported. Descriptors stay out of prompts.
    """
    source, package = Path(source), Path(package)
    if source.suffix.lower() != '.docx':
        return {}
    if _sha(source) != source_sha256:
        raise ValueError('NATIVE_ALT_SOURCE_CHANGED')
    if not assets:
        return {}
    from lxml import etree
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    records = {}
    with zipfile.ZipFile(source) as archive:
        if not {'word/document.xml','word/_rels/document.xml.rels'} <= set(archive.namelist()):
            return {}
        xml = archive.read('word/document.xml')
        rels = archive.read('word/_rels/document.xml.rels')
        root = etree.fromstring(xml, parser)
        relationships = {r.get('Id'): dict(r.attrib) for r in etree.fromstring(rels, parser)}
        media_hashes = {}
        for asset in assets:
            locator = str(asset.get('source_locator', ''))
            if not re.fullmatch(r'/w:document/w:body/.*/w:drawing(?:\[\d+\])?', locator):
                continue
            if not asset.get('relative_path') or not asset.get('relationship_id'):
                continue
            nodes = root.xpath(locator, namespaces=NS)
            if len(nodes) != 1 or nodes[0].xpath('ancestor::mc:AlternateContent | .//mc:AlternateContent', namespaces=NS):
                continue
            node = nodes[0]
            graphics = node.xpath('.//a:graphicData', namespaces=NS)
            if (len(graphics) != 1 or graphics[0].get('uri') != NS['pic']
                    or len(graphics[0]) != 1 or graphics[0][0].tag != '{'+NS['pic']+'}pic'
                    or len(node.xpath('.//pic:pic', namespaces=NS)) != 1):
                continue
            if any(etree.QName(e).localname in {'sp','grpSp','txBody','txbx','txbxContent','wgp','wsp','svgBlip'}
                   for e in node.iter() if isinstance(e.tag,str)):
                continue
            # The bitmap must represent the whole visible picture, not the
            # uncropped fallback for a transformed or conditionally drawn view.
            if any(any(value not in {'0','0.0'} for value in e.attrib.values())
                   for e in node.xpath('.//a:srcRect', namespaces=NS)):
                continue
            if any(e.get('rot','0') not in {'0','0.0'} or e.get('flipH','0') not in {'0','false'}
                   or e.get('flipV','0') not in {'0','false'} for e in node.xpath('.//a:xfrm', namespaces=NS)):
                continue
            if node.xpath('.//a:custGeom', namespaces=NS) or any(e.get('prst') != 'rect' for e in node.xpath('.//a:prstGeom', namespaces=NS)):
                continue
            embeds = node.xpath('.//a:blip/@r:embed', namespaces=NS)
            if len(embeds) != 1 or embeds[0] != asset['relationship_id']:
                continue
            relation = relationships.get(embeds[0], {})
            if relation.get('TargetMode') == 'External' or not relation.get('Type', '').endswith('/image'):
                continue
            media = posixpath.normpath(posixpath.join('word', relation.get('Target', '')))
            if not media.startswith('word/media/') or media not in archive.namelist():
                continue
            if media not in media_hashes:
                media_hashes[media] = hashlib.sha256(archive.read(media)).hexdigest()
            original_sha = _sha(package / asset['relative_path'])
            if media_hashes[media] != original_sha:
                continue
            descriptions = [str(s).strip() for s in node.xpath(
                './/wp:docPr/@descr | .//pic:cNvPr/@descr', namespaces=NS) if str(s).strip()]
            if not descriptions or any(len(s) > 256 for s in descriptions):
                continue
            normalized = {canonical(s) for s in descriptions}
            if len(normalized) != 1 or not next(iter(normalized)):
                continue
            records[asset['asset_uid']] = {
                'version':VERSION, 'scope':'DIRECT_WHOLE_BODY_MEDIA', 'source_sha256':source_sha256,
                'asset_uid':asset['asset_uid'],
                'source_part':'word/document.xml', 'source_xml_sha256':hashlib.sha256(xml).hexdigest(),
                'relationships_sha256':hashlib.sha256(rels).hexdigest(), 'source_locator':locator,
                'xml_object_sha256':hashlib.sha256(etree.tostring(node)).hexdigest(),
                'relationship_id':embeds[0], 'media_part':media, 'original_image_sha256':original_sha,
                'descr_values':descriptions, 'canonical_text':next(iter(normalized)),
                'duplicate_attributes_are_one_source':True, 'metadata_alone_is_not_visual_proof':True,
            }
    return records


def qualify_native_symbol(task, owners, prepared, primary, local):
    """Resolve only the isolated-symbol hold; never replace the primary text."""
    from .asset_local_evidence import candidate_conflicts, independent_review_reasons
    reasons = []
    def require(ok, reason):
        if not ok:
            reasons.append(reason)
    require(prepared.get('sha256') == task.input_sha256 and not prepared.get('blank_source')
            and prepared.get('recipe', {}).get('no_pixels_cropped') is True, 'FULL_VISIBLE_INPUT_REQUIRED')
    proofs = [o.get('native_alt_evidence') for o in owners]
    native = set()
    for owner, proof in zip(owners, proofs):
        valid = bool(proof and proof.get('version') == VERSION and proof.get('scope') == 'DIRECT_WHOLE_BODY_MEDIA'
                     and proof.get('source_sha256') == task.source_sha256
                     and proof.get('asset_uid') == str(owner.get('id', '')).removeprefix('asset:')
                     and proof.get('original_image_sha256') == owner.get('image_sha256'))
        require(valid, 'ALL_OWNER_NATIVE_SOURCE_BINDINGS_REQUIRED')
        if valid:
            values = {canonical(s) for s in proof.get('descr_values', []) if str(s).strip()}
            require(len(values) == 1 and proof.get('canonical_text') in values, 'NATIVE_ATTRIBUTES_CHANGED_OR_INCONSISTENT')
            native.update(values)
    require(bool(owners) and len(native) == 1, 'ALL_OWNER_NATIVE_TEXT_AGREEMENT_REQUIRED')
    expected = next(iter(native)) if len(native) == 1 else ''
    require(bool(re.fullmatch(r'[A-Za-zΑ-Ωα-ω0-9+−\-().,=°]{1,4}', expected))
            and len(re.findall(r'[A-Za-zΑ-Ωα-ω0-9]', expected)) <= 2, 'SHORT_SYMBOL_SCOPE_REQUIRED')
    require(primary.get('status') == 'SUCCESS' and primary.get('uncertain') is False
            and primary.get('source_verified') is True and primary.get('text_presence') == 'PRESENT'
            and primary.get('visual_content_kind') in {'PURE_TEXT','PURE_FORMULA'}
            and primary.get('task_id') == task.task_id and primary.get('task_fingerprint') == task.fingerprint,
            'VERIFIED_CERTAIN_PURE_PRIMARY_REQUIRED')
    require(bool(expected) and canonical(str(primary.get('text', ''))) == expected, 'PRIMARY_NATIVE_DISAGREEMENT')
    local = local or {}
    require(local.get('input_sha256') == task.input_sha256, 'LOCAL_INPUT_BINDING_REQUIRED')
    coverage = local.get('source_coverage', {})
    require(coverage.get('input_sha256') == task.input_sha256 and coverage.get('geometry_available') is True
            and not coverage.get('reasons'), 'BOUND_COMPLETE_LOCAL_COVERAGE_REQUIRED')
    require(set(local.get('decision', {}).get('review_reasons', [])) == {'ISOLATED_SYMBOL_SOURCE_REVIEW'},
            'NO_OTHER_LOCAL_DECISION_RISKS_ALLOWED')
    require(set(independent_review_reasons(local)) == {'ISOLATED_SYMBOL_SOURCE_REVIEW'},
            'NO_OTHER_INDEPENDENT_RISKS_ALLOWED')
    candidates = local.get('candidates', {})
    for name in ('PP','Ovis','Xiaomi'):
        candidate = candidates.get(name)
        if name == 'Xiaomi' and candidate is None:
            continue
        require(bool(candidate) and candidate.get('status') in {'OK','PASS'}
                and bool(str(candidate.get('text', '')).strip()), name+'_VALID_READING_REQUIRED')
        require(bool(candidate) and canonical(str(candidate.get('text', ''))) == expected, name+'_NATIVE_DISAGREEMENT')
    require(not candidate_conflicts(local, str(primary.get('text', ''))), 'LOCAL_CONFLICT_MUST_REMAIN_REVIEW')
    return {'version':VERSION, 'accepted':not reasons, 'reasons':list(dict.fromkeys(reasons)),
            'source_sha256':task.source_sha256, 'input_sha256':task.input_sha256,
            'primary_task_fingerprint':task.fingerprint, 'primary_text_sha256':hashlib.sha256(str(primary.get('text','')).encode()).hexdigest(),
            'native_canonical_text':expected, 'source_records':proofs,
            'resolved_risks':['ISOLATED_SYMBOL_SOURCE_REVIEW'] if not reasons else [],
            'preserve_image':True, 'independent_human_review_claim':False,
            'policy_sha256':_sha(__file__),
            'visual_evidence_sha256':hashlib.sha256(json.dumps(local,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
