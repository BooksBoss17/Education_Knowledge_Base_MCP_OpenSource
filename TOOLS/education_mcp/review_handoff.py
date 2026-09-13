"""Compact, source-bound residual OCR evidence; never claims error certainty."""
from __future__ import annotations
from pathlib import Path
import copy
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from posixpath import normpath


def summarize_handoff(handoff):
    """Keep every review item and source request; fold pixel-level boundary logs."""
    result = copy.deepcopy(handoff)
    result['schema'] = 'education-mcp-residual-ocr-handoff-summary-v1'
    result['full_evidence_view'] = 'handoff'
    result['summary_scope'] = 'All text candidates, table tasks and image-region reviews are retained. Only pixel-level boundary diagnostics are folded; use handoff for those details.'
    for row in result.get('image_region_reviews', []):
        details = row.pop('boundary_evidence', None)
        if details is None:
            continue
        methods = []
        def visit(value):
            if isinstance(value, dict):
                if value.get('schema'):
                    summary = {key:child for key,child in value.items()
                               if key in {'schema','status'} or isinstance(child, bool)}
                    if 'blocking_regions' in value:
                        summary['blocking_regions'] = value['blocking_regions']
                    if summary not in methods:
                        methods.append(summary)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(details)
        row['boundary_summary'] = dict(methods=methods, pixel_diagnostics_view='handoff')
    return result


def _units(value):
    if isinstance(value, dict):
        candidate = value.get("agent_review_candidate")
        if isinstance(candidate, dict):
            yield candidate, value.get("resolver", {})
        for key, child in value.items():
            if key != "agent_review_candidate":
                yield from _units(child)
    elif isinstance(value, list):
        for child in value:
            yield from _units(child)


def _overlap(a, b):
    return len(a) == len(b) == 4 and min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _crop(box, width, height):
    if len(box) != 4 or width <= 0 or height <= 0:
        return None
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    # Small context margin; never silently reinterpret pixel coordinates as PDF points.
    return [max(0, left / width - .005), max(0, top / height - .005), min(1, right / width + .005), min(1, bottom / height + .005)]


def build_handoff(source, markdown, report, document_ir=None):
    source = Path(source)
    document_ir = document_ir or {}
    source_type = source.suffix.lower()
    pagination = source_type == '.docx' and report.get('input_transform', {}).get('route') == 'DOCX_SCREENSHOT_PDF'
    pages = {p["page_index"]: p for p in document_ir.get("pages", [])}
    spans = {s["node_id"]: s for s in report.get("markdown_render", {}).get("node_spans", [])}
    original_text = markdown.decode("utf-8")
    media_by_relationship, media_sizes = {}, {}
    if source_type == ".docx":
        from PIL import Image
        import io
        with zipfile.ZipFile(source) as archive:
            rels = ET.fromstring(archive.read("word/_rels/document.xml.rels")) if "word/_rels/document.xml.rels" in archive.namelist() else []
            for rel in rels:
                if rel.get("TargetMode") == "External":
                    continue
                target = normpath("word/" + rel.get("Target", ""))
                if not target.startswith("word/media/") or target not in archive.namelist():
                    continue
                media_by_relationship[rel.get("Id")] = target
                try:
                    with Image.open(io.BytesIO(archive.read(target))) as image:
                        media_sizes[target] = image.size
                except (OSError, ValueError):
                    pass
    buckets = {}

    def collect(tree, owner=None, asset=None):
        for candidate, resolver in _units(tree):
            task_id = candidate.get("task_id") or hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()[:24]
            entry = buckets.setdefault(task_id, dict(candidate=candidate, resolver=resolver, owners=[], asset=asset))
            if owner is not None and owner not in entry["owners"]:
                entry["owners"].append(owner)

    for block in document_ir.get("blocks", []):
        collect(block, owner=block)
    for asset in report.get("assets", {}).get("records", []):
        collect(asset.get("provenance", {}), asset=asset)
    for source_image in report.get("image_content", {}).get("sources", []):
        collect(source_image.get("provenance", {}), asset=source_image)
    # Word debug DocumentIR is unnecessary: its asset report already retains unit evidence.
    items = []
    for task_id, raw in sorted(buckets.items()):
        candidate, resolver, asset = raw["candidate"], raw["resolver"], raw["asset"]
        box = candidate.get("bbox", [])
        evidence = {}
        for provider, record in candidate.get("evidence", {}).items():
            if isinstance(record, dict):
                evidence[provider] = {k: record.get(k) for k in ("model_id", "text", "normalized_text", "output_contract_status")}
        crop_provenance = candidate.get("source_provenance", {}).get("crop", {})
        page_index = crop_provenance.get("page_index")
        if page_index is None:
            match = re.search(r":page:(\d+)$", candidate.get("page_id", ""))
            page_index = int(match[1]) if match else None
        item = dict(task_id=task_id, reason=candidate.get("reason"), resolution_status=resolver.get("resolution_status"), evidence=evidence,
                    all_three_outputs_valid=len(evidence) == 3 and all(v.get("output_contract_status") == "PASS" for v in evidence.values()),
                    crop_sha256=candidate.get("primary_crop_sha256") or candidate.get("crop_sha256"),
                    original_bbox=box, markdown_contexts=[])
        if source_type == ".pdf" or pagination:
            page = pages.get(page_index, {})
            region = _crop(box, page.get("width_pt", 0), page.get("height_pt", 0))
            item.update(coordinate_space="PDF_POINTS", source_request=dict(kind="image", page=page_index + 1, dpi=288) if page_index is not None else None)
            if region and item["source_request"]:
                item["source_request"]["region"] = region
            if pagination:
                item['coordinate_space'] = 'DOCX_PAGINATION_PDF_POINTS'
                if item['source_request'] and region:
                    item['source_request']['source_view'] = 'pagination'
                else:
                    item['source_request'] = None
            for owner in raw["owners"]:
                if owner.get("page_index") != page_index or not _overlap(box, owner.get("bbox_pdf_pt", [])):
                    continue
                span = spans.get(owner.get("node_id"))
                if not span:
                    continue
                start, end = span["byte_start"], span["byte_end"]
                item["markdown_contexts"].append(dict(node_id=owner["node_id"], character_offset=len(markdown[:start].decode("utf-8")),
                                                      character_length=len(markdown[start:end].decode("utf-8")), original_span_sha256=span.get("sha256")))
        else:
            media = media_by_relationship.get((asset or {}).get("relationship_id"))
            dimensions = media_sizes.get(media, (0, 0))
            region = _crop(box, *dimensions)
            item.update(coordinate_space="ORIGINAL_DOCX_IMAGE_PIXELS", asset_name=(asset or {}).get("relative_path"),
                        source_request=dict(kind="image", image_name=media) if media else None)
            if region and item["source_request"]:
                item["source_request"]["region"] = region
            if item["asset_name"]:
                for match in re.finditer(r'!\[[^\]]*\]\(' + re.escape(item["asset_name"]) + r'(?:\s+"[^"]*")?\)', original_text):
                    item["markdown_contexts"].append(dict(character_offset=match.start(), character_length=len(match[0]), original_text=match[0]))
        if source_type == ".docx" and not item["markdown_contexts"]:
            item["markdown_contexts"] = _source_image_contexts(asset or {}, box, original_text)
        item["markdown_location_available"] = bool(item["markdown_contexts"])
        item["source_location_available"] = item["source_request"] is not None
        items.append(item)
    items.sort(key=lambda item: ((item["source_request"] or {}).get("page", 0), (item["source_request"] or {}).get("image_name", ""),
                                 (item["original_bbox"] or [0, 0])[1], (item["original_bbox"] or [0])[0], item["task_id"]))
    content_reviews = _table_content_reviews(source, markdown, report, document_ir,
                                            media_by_relationship, media_sizes)
    return dict(schema="education-mcp-residual-ocr-handoff-v1", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                original_markdown_sha256=hashlib.sha256(markdown).hexdigest(), candidate_count=len(items),
                candidate_basis="Recorded three-model disagreement or runtime failure; candidates are not independently proven recognition errors. Agreement can still be wrong, and a candidate can be correct or blank.",
                evidence_available=bool(document_ir) if pagination else source_type == ".docx" or bool(document_ir),
                scope="Three-model text candidates are in items, table content tasks in content_reviews, and region inspection in image_region_reviews. Formula/table/layout review markers are separate, not silently treated as OCR disagreements.",
                items=items, image_region_reviews=_image_region_reviews(report, media_by_relationship, media_sizes, document_ir),
                content_reviews=content_reviews, content_review_count=len(content_reviews))


def _table_contexts(markup, markdown, span=None):
    if not markup:
        return []
    original = markdown.decode('utf-8')
    start_char, end_char = 0, len(original)
    basis = 'EXACT_TABLE_CONTENT'
    if span:
        start, end = span.get('byte_start'), span.get('byte_end')
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(markdown):
            fragment = markdown[start:end]
            if not span.get('sha256') or hashlib.sha256(fragment).hexdigest() == span['sha256']:
                try:
                    start_char = len(markdown[:start].decode('utf-8'))
                    end_char = start_char + len(fragment.decode('utf-8'))
                    basis = 'EXACT_TABLE_CONTENT_WITH_NODE_SPAN'
                except UnicodeDecodeError:
                    start_char, end_char = 0, len(original)
    matches = list(re.finditer(re.escape(markup), original[start_char:end_char]))
    return [dict(character_offset=start_char + match.start(), character_length=len(markup),
                 original_text=markup, original_span_sha256=hashlib.sha256(markup.encode('utf-8')).hexdigest(),
                 mapping_basis=basis, unique_in_document=len(matches) == 1)
            for match in matches]


def _table_content_reviews(source, markdown, report, document_ir, media_by_relationship, media_sizes):
    pagination = source.suffix.lower() == '.docx' and report.get('input_transform', {}).get('route') == 'DOCX_SCREENSHOT_PDF'
    pages = {p['page_index']: p for p in document_ir.get('pages', [])}
    spans = {s['node_id']: s for s in report.get('markdown_render', {}).get('node_spans', [])}
    owners = [(block, None) for block in document_ir.get('blocks', [])] if source.suffix.lower() == '.pdf' or pagination else [
        (block, image) for image in report.get('image_content', {}).get('sources', [])
        for block in image.get('table_review_blocks', [])]
    tasks = []
    for block, image in owners:
        if block.get('kind') != 'TABLE':
            continue
        if block.get('visibility') in {'SUPPRESSED_DUPLICATE', 'SUPPRESSED_FROM_MAIN_BODY'}:
            continue
        content = block.get('content', {})
        table = content.get('table_ir') or {}
        quality = table.get('quality', {})
        cells = [c for c in table.get('cells', []) if c.get('review_state', 'NONE') != 'NONE']
        unassigned = table.get('unassigned_formula_regions', [])
        if not cells and not unassigned and (block.get('review_state') == 'NONE'
                or quality.get('status') == 'STRUCTURED_TABLE'):
            continue
        box = block.get('bbox_pdf_pt') or []
        page_index = block.get('page_index')
        if image is None:
            page = pages.get(page_index, {})
            region = _crop(box, page.get('width_pt', 0), page.get('height_pt', 0))
            request = dict(kind='image', page=page_index+1, dpi=288, region=region) if region and isinstance(page_index, int) else None
            coordinates = 'PDF_POINTS'
            if pagination:
                coordinates = 'DOCX_PAGINATION_PDF_POINTS'
                if request:
                    request['source_view'] = 'pagination'
            identity = [str(source), page_index, block.get('node_id')]
        else:
            media = media_by_relationship.get(image.get('relationship_id'))
            region = _crop(box, *media_sizes.get(media, (0, 0)))
            request = dict(kind='image', image_name=media, region=region) if media and region else None
            coordinates = 'ORIGINAL_DOCX_IMAGE_PIXELS'
            identity = [str(source), image.get('source_part'), image.get('relationship_id'),
                        image.get('source_locator'), block.get('node_id')]
        markup = content.get('table_serialization', {}).get('content_body') or ''
        contexts = _table_contexts(markup, markdown, spans.get(block.get('node_id')) if image is None else None)
        task = dict(
            task_id='table-content-' + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24],
            kind='TABLE_CONTENT', node_id=block.get('node_id'),
            status='REVIEW_REQUIRED', recognition_success=False,
            review_semantics='Unresolved table content or structure; not proof that every listed cell is wrong.',
            reason_codes=sorted(set(quality.get('reason_codes', [])) | set(block.get('provenance', {}).get('review_reasons', []))),
            table_structure_status=quality.get('status', content.get('source_status', 'UNKNOWN')),
            coordinate_space=coordinates, original_bbox=copy.deepcopy(box),
            source_request=request, source_location_available=request is not None,
            markdown_contexts=contexts, markdown_location_available=bool(contexts),
            markdown_location_status='UNIQUE' if len(contexts) == 1 else 'AMBIGUOUS' if contexts else 'UNAVAILABLE',
            cells=[{k: copy.deepcopy(c[k]) for k in (
                'cell_id', 'row', 'col', 'rowspan', 'colspan', 'text', 'bbox_pdf_pt',
                'review_state', 'content_type', 'formula_region_evidence') if k in c} for c in cells],
            unassigned_formula_regions=copy.deepcopy(unassigned))
        if image is not None:
            task['source_locator'] = image.get('source_locator')
            task['relationship_id'] = image.get('relationship_id')
        tasks.append(task)
    return tasks


def _image_region_reviews(report, media_by_relationship, media_sizes=None, document_ir=None):
    items = []
    seen = set()

    def append_review(review, request):
        # Repeated IR references share a task; distinct DOCX insertions do not.
        identity = (review.get('review_id') or json.dumps(review, sort_keys=True),
                    review.get('source_part'), review.get('source_locator'),
                    review.get('relationship_id'), review.get('media_part'))
        if identity in seen:
            return
        seen.add(identity)
        items.append({**review, 'source_request': request,
                      'scope': 'Region localization and crop inspection; not a text OCR failure.'})

    media_sizes = media_sizes or {}
    document_ir = document_ir or {}
    for review in report.get("image_content", {}).get("region_reviews", []):
        media = media_by_relationship.get(review.get("relationship_id"))
        request = dict(kind='image', image_name=media) if media else None
        width, height = media_sizes.get(media, (0, 0))
        region = _crop(review.get('bbox_pixels', []), width, height)
        if request and region:
            request['region'] = region
        append_review(review, request)
    pages = {p['page_index']: p for p in document_ir.get('pages', [])}
    for block in document_ir.get('blocks', []):
        provenance = block.get('provenance', {}).get('route_provenance', {})
        reviews = list(provenance.get('boundary_reviews', []))
        if provenance.get('source_scope_review'):
            reviews.append(provenance['source_scope_review'])
        for review in reviews:
            index = review['page_index']
            page = pages.get(index, {})
            region = _crop(review['source_bbox_pdf_pt'], page.get('width_pt', 0), page.get('height_pt', 0))
            request = dict(kind='image', page=index+1, dpi=288)
            if report.get('input_transform', {}).get('route') == 'DOCX_SCREENSHOT_PDF':
                request['source_view'] = 'pagination'
            if region:
                request['region'] = region
            append_review(review, request)
    return items


def _source_image_contexts(source_image, candidate_box, markdown):
    contexts = []
    seen = set()
    for locator in source_image.get("markdown_locators", []):
        text = locator.get("text")
        if not text or not _overlap(candidate_box, locator.get("bbox_pixels", [])):
            continue
        matches = list(re.finditer(re.escape(text), markdown))
        for match in matches:
            key = (match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            contexts.append(dict(character_offset=match.start(), character_length=len(text),
                original_text=text, mapping_basis="EXACT_RECOGNIZED_IMAGE_REGION_TEXT",
                unique_in_document=len(matches) == 1, original_bbox=locator["bbox_pixels"]))
    return contexts
