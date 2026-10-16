"""Read-only trace of DOCX source images, including producer-recorded exclusions.

An exclusion receipt records a decision, not evidence that the decision is correct.
It never supplies transcript text or certifies page geometry.
"""
import hashlib
import posixpath
from pathlib import Path
from zipfile import ZipFile
from lxml import etree

VERSION = 'docx-source-disposition-v1'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
V = 'urn:schemas-microsoft-com:vml'


def trace_docx_source_dispositions(source, report, assets, *, text_source_locations=()):
    source = Path(source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if report.get('source', {}).get('sha256') != source_sha:
        raise ValueError('SOURCE_DISPOSITION_DOCUMENT_CHANGED')
    observations = []
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    with ZipFile(source) as package:
        parts = [n for n in package.namelist() if n == 'word/document.xml' or
                 (n.startswith(('word/header', 'word/footer')) and n.endswith('.xml'))]
        for part in parts:
            root = etree.fromstring(package.read(part), parser)
            rel_part = posixpath.join(posixpath.dirname(part), '_rels', posixpath.basename(part) + '.rels')
            rels = {}
            if rel_part in package.namelist():
                rel_root = etree.fromstring(package.read(rel_part), parser)
                for rel in rel_root:
                    if rel.get('TargetMode') != 'External':
                        rels[rel.get('Id')] = posixpath.normpath(posixpath.join(posixpath.dirname(part), rel.get('Target', '')))
            for node in root.iter():
                rid = node.get('{' + R + '}embed') if node.tag == '{' + A + '}blip' else (
                    node.get('{' + R + '}id') if node.tag == '{' + V + '}imagedata' else None)
                if not rid or rid not in rels:
                    continue
                parent = node
                for ancestor in node.iterancestors():
                    if ancestor.tag in {'{' + W + '}drawing', '{' + W + '}pict'}:
                        parent = ancestor
                        break
                locator = root.getroottree().getpath(parent)
                media_part = rels[rid]
                if media_part not in package.namelist():
                    raise ValueError('SOURCE_DISPOSITION_MEDIA_MISSING')
                branch = next((etree.QName(a).localname for a in node.iterancestors()
                               if etree.QName(a).localname in {'Choice', 'Fallback'}), None)
                observations.append({'source_part': part, 'source_locator': locator,
                    'relationship_id': rid, 'media_part': media_part,
                    'source_media_sha256': hashlib.sha256(package.read(media_part)).hexdigest(),
                    'compatibility_branch': branch})
    exclusions = report.get('blank_images_removed', [])
    for row in observations:
        part, locator = row['source_part'], row['source_locator']
        matching = [r for r in exclusions if r.get('source_locator') == locator and
                    (r.get('source_part') in {None, part})]
        location_sources = {r['source_part'] for r in observations if r['source_locator'] == locator}
        ambiguous = any(not r.get('source_part') for r in matching) and len(location_sources) != 1
        emitted = [r['asset_uid'] for r in assets if r.get('source_part') == part and
                   (r.get('source_locator') == locator or (r.get('source_locator') or '').startswith(locator + '/'))]
        text_emitted = (part, locator) in text_source_locations
        state = ('CONFLICTING_PRODUCER_DISPOSITIONS' if matching and (emitted or text_emitted) else
                 'AMBIGUOUS_STORY_PART' if ambiguous else
                 'PRODUCER_RECORDED_EXCLUSION' if matching else
                 'PRODUCER_EMITTED_ASSET' if emitted else
                 'PRODUCER_EMITTED_TEXT' if text_emitted else 'UNRESOLVED_SOURCE_DISPOSITION')
        row.update(disposition=state, exclusion_receipts=matching, emitted_asset_uids=emitted,
                   text_emitted=text_emitted, exclusion_correctness_verified=False,
                   rendered_occurrence_count_certified=False)
    return {'version': VERSION, 'source_sha256': source_sha, 'observations': observations,
            'gold_read': False, 'markdown_modified': False}
