"""Local API adapters for images and unresolved formula previews in native Word."""
from __future__ import annotations

import hashlib
import re
import time
import zipfile
from pathlib import Path

from .formula_review import FormulaReview, _initialize_report, _iter_formula_nodes
from .ir import HyperlinkNode, ImageContentNode, ImageNode, TableNode
from .image_input import IMAGE_SUFFIXES, InvalidRasterImageError
from .pdf.local_model_api import OcrApiRuntime, read_config


class NativeFormulaApi(FormulaReview):
    """Reuse conservative formula validation; cache only inside one document."""

    def __init__(self, config_path, runtime=None):
        super().__init__()
        self.api = runtime or OcrApiRuntime('formula', read_config(config_path)['roles']['formula'])

    def process(self, document, output_dir, report):
        from PIL import Image
        started = time.perf_counter()
        nodes = [n for n in _iter_formula_nodes(document)
                 if n.status == 'RENDERED_FALLBACK' and n.rendered_ref]
        summary = _initialize_report(report)
        summary['candidates'] = len(nodes)
        groups, results = {}, []
        for node in nodes:
            path = (output_dir/node.rendered_ref).resolve()
            if not path.is_relative_to(output_dir.resolve()) or not path.is_file():
                results.append(self._failed_result(node, report, 'Formula preview missing or outside package'))
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, {'path': path, 'nodes': []})['nodes'].append(node)
        summary['unique_png'] = len(groups)
        summary['deduplicated_occurrences'] = sum(len(g['nodes'])-1 for g in groups.values())
        if groups:
            # Protocol/network errors fail the job; a successful empty prediction
            # is handled by the safety gate and preserves the original preview.
            try:
                outputs = self.api.recognize_batch([{'path': str(g['path']), 'sha256': sha}
                                                   for sha, g in groups.items()])
            finally:
                self.api.close()
            identity = self.api.identity
            self._runtime_fingerprint = {
                **identity, 'model_identifier': identity['model_id'],
                'model_sha256': identity['model_fingerprint']}
            for (sha, group), raw in zip(groups.items(), outputs, strict=True):
                with Image.open(group['path']) as image:
                    width, height = image.size
                # Native serializer owns the math delimiters.
                raw = raw.strip()
                for left, right in (('$$','$$'), ('\\[','\\]'), ('\\(','\\)'), ('$','$')):
                    if raw.startswith(left) and raw.endswith(right) and len(raw) >= len(left)+len(right):
                        raw = raw[len(left):-len(right)].strip()
                        break
                key = hashlib.sha256((sha+identity['model_fingerprint']).encode()).hexdigest()
                for index, node in enumerate(group['nodes']):
                    results.append(self._evaluate_result(node, report, raw_latex=raw,
                        png_sha256=sha, width=width, height=height, cache_key=key, cache_hit=index>0))
        summary.update(model_id=self.api.identity.get('model_id', 'api:formula'),
                       runtime_fingerprint=self._runtime_fingerprint,
                       inference_calls=len(self.api.metrics['requests']))
        report['timing']['formula_ocr_inference_seconds'] = time.perf_counter()-started
        report['runtime']['vision_or_ocr_used'] = bool(groups)
        self._finalize_document_report(report, results)
        return results


def _inline_lists(document):
    def nested(children):
        yield children
        for child in children:
            if isinstance(child, HyperlinkNode):
                yield from nested(child.children)
    for block in document.blocks:
        if isinstance(block, TableNode):
            for row in block.rows:
                for cell in row:
                    for paragraph in cell.blocks:
                        yield from nested(paragraph.children)
        else:
            yield from nested(block.children)


def word_picture_input(path, owners, output_dir, occurrence, relationship_id=None):
    """Rasterize a rectangular Word picture occurrence without hidden source pixels."""
    import math
    from PIL import Image, ImageOps
    from .namespaces import NS
    from .image_input import MAX_IMAGE_PIXELS

    pictures = [pic for owner in owners for pic in owner.xpath("self::pic:pic | .//pic:pic", namespaces=NS)]
    if relationship_id is not None:
        pictures = [pic for pic in pictures if pic.xpath(
            "./pic:blipFill/a:blip/@r:embed", namespaces=NS) == [relationship_id]]
    if len(pictures) != 1:
        raise ValueError("WORD_PICTURE_OWNER_AMBIGUOUS")
    pic = pictures[0]
    group_ratio = 1.
    for group in pic.xpath("ancestor::wpg:wgp | ancestor::a:grpSp", namespaces=NS):
        if any(child.tag not in {f"{{{NS['wpg']}}}cNvGrpSpPr", f"{{{NS['wpg']}}}grpSpPr",
                                  f"{{{NS['pic']}}}pic"} for child in group):
            raise ValueError("WORD_GROUP_COMPLEX_CONTENT")
        transform = group.find("wpg:grpSpPr/a:xfrm", NS)
        if transform is None or any(transform.get(k, "0") not in {"0", "false", ""}
                                    for k in ("rot", "flipH", "flipV")):
            raise ValueError("WORD_GROUP_TRANSFORM_UNSUPPORTED")
        outer, inner = transform.find("a:ext", NS), transform.find("a:chExt", NS)
        if outer is None or inner is None:
            raise ValueError("WORD_GROUP_EXTENT_MISSING")
        dims = [float(e.get(k, "0")) for e in (outer, inner) for k in ("cx", "cy")]
        if not all(math.isfinite(v) and v > 0 for v in dims):
            raise ValueError("WORD_GROUP_EXTENT_INVALID")
        group_ratio *= (dims[0]/dims[2])/(dims[1]/dims[3])
        own_transform = pic.find("pic:spPr/a:xfrm", NS)
        def bounds(element):
            offset, extent = element.find("a:off", NS), element.find("a:ext", NS)
            if offset is None or extent is None:
                raise ValueError("WORD_GROUP_POSITION_MISSING")
            x, y = float(offset.get("x")), float(offset.get("y"))
            return x, y, x+float(extent.get("cx")), y+float(extent.get("cy"))
        if own_transform is None:
            raise ValueError("WORD_GROUP_POSITION_MISSING")
        own_box = bounds(own_transform)
        for sibling in group.findall("pic:pic", NS):
            if sibling is pic:
                continue
            sibling_transform = sibling.find("pic:spPr/a:xfrm", NS)
            if sibling_transform is None or float(sibling_transform.get("rot","0")) % 21600000:
                raise ValueError("WORD_GROUP_SIBLING_TRANSFORM_UNSUPPORTED")
            other = bounds(sibling_transform)
            if min(own_box[2],other[2]) > max(own_box[0],other[0]) and min(own_box[3],other[3]) > max(own_box[1],other[1]):
                raise ValueError("WORD_GROUP_OVERLAPPING_PICTURES")
    if pic.xpath("./pic:blipFill/a:tile | ./pic:spPr/a:custGeom", namespaces=NS):
        raise ValueError("WORD_PICTURE_FILL_OR_GEOMETRY_UNSUPPORTED")
    geometry = pic.find("pic:spPr/a:prstGeom", NS)
    if geometry is not None and geometry.get("prst") != "rect":
        raise ValueError("WORD_PICTURE_NONRECTANGULAR")
    fill_rect = pic.find("pic:blipFill/a:stretch/a:fillRect", NS)
    if fill_rect is not None and any(float(v) != 0 for v in fill_rect.attrib.values()):
        raise ValueError("WORD_PICTURE_FILL_RECT_UNSUPPORTED")
    source_rect = pic.find("pic:blipFill/a:srcRect", NS)
    fractions = [float(source_rect.get(k, "0"))/100000 if source_rect is not None else 0.
                 for k in ("l", "t", "r", "b")]
    if not all(math.isfinite(v) and -1 <= v <= 1 for v in fractions):
        raise ValueError("WORD_PICTURE_CROP_RANGE")
    xfrm = pic.find("pic:spPr/a:xfrm", NS)
    angle = float(xfrm.get("rot", "0"))/60000 if xfrm is not None else 0.
    if not math.isfinite(angle):
        raise ValueError("WORD_PICTURE_ROTATION_INVALID")
    flip_h = xfrm is not None and xfrm.get("flipH") in {"1", "true"}
    flip_v = xfrm is not None and xfrm.get("flipV") in {"1", "true"}
    with Image.open(path) as original:
        if original.width*original.height > MAX_IMAGE_PIXELS or getattr(original, "n_frames", 1) != 1:
            raise ValueError("WORD_PICTURE_SIZE_UNSUPPORTED")
        image = original.convert("RGBA")
    width, height = image.size
    left, top, right, bottom = fractions
    box = (math.ceil(left*width), math.ceil(top*height),
           math.floor((1-right)*width), math.floor((1-bottom)*height))
    cw, ch = box[2]-box[0], box[3]-box[1]
    if cw <= 0 or ch <= 0 or cw*ch > MAX_IMAGE_PIXELS:
        raise ValueError("WORD_PICTURE_EMPTY_OR_EXCESSIVE_CROP")
    visible = Image.new("RGBA", (cw, ch), "white")
    intersection = (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))
    if intersection[2] > intersection[0] and intersection[3] > intersection[1]:
        visible.alpha_composite(image.crop(intersection), (intersection[0]-box[0], intersection[1]-box[1]))
    # Preserve source sampling density while applying the displayed aspect ratio.
    extent = xfrm.find("a:ext", NS) if xfrm is not None else None
    if extent is not None:
        cx, cy = float(extent.get("cx", "0")), float(extent.get("cy", "0"))
        if not (math.isfinite(cx) and math.isfinite(cy) and cx > 0 and cy > 0):
            raise ValueError("WORD_PICTURE_EXTENT_INVALID")
        ratio = cx/cy * group_ratio
        target = (max(cw, math.ceil(ch*ratio)), max(ch, math.ceil(cw/ratio)))
        if target[0]*target[1] > MAX_IMAGE_PIXELS:
            raise ValueError("WORD_PICTURE_EXTENT_TOO_LARGE")
        if target != visible.size:
            visible = visible.resize(target, Image.Resampling.LANCZOS)
    if flip_h:
        visible = ImageOps.mirror(visible)
    if flip_v:
        visible = ImageOps.flip(visible)
    if angle % 360:
        radians = math.radians(angle)
        rotated_w = abs(visible.width*math.cos(radians))+abs(visible.height*math.sin(radians))
        rotated_h = abs(visible.width*math.sin(radians))+abs(visible.height*math.cos(radians))
        if math.ceil(rotated_w)*math.ceil(rotated_h) > MAX_IMAGE_PIXELS:
            raise ValueError("WORD_PICTURE_ROTATION_TOO_LARGE")
        visible = visible.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor="white")
    target = output_dir/"_bemarkdown"/"word_appearance"/f"{occurrence:06d}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    visible.convert("RGB").save(target)
    receipt = dict(policy="WORD_VISIBLE_RECTANGULAR_PICTURE_V1",
                   original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   original_size=[width,height], crop_box=list(box), rotation_degrees=angle,
                   flip_h=flip_h, flip_v=flip_v, rendered_size=list(visible.size),
                   path=target.relative_to(output_dir).as_posix(), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    return target, receipt


def word_group_input(archive_path, source_part, owner, output_dir, occurrence):
    """Composite the visible, non-overlapping pictures of a Word group."""
    import math
    import posixpath
    from PIL import Image
    from lxml import etree
    from .namespaces import NS
    from .image_input import MAX_IMAGE_PIXELS

    groups = owner.xpath(".//wpg:wgp", namespaces=NS)
    if len(groups) != 1:
        raise ValueError("WORD_GROUP_COMPLEX_CONTENT")
    group = groups[0]
    pictures = group.findall("pic:pic", NS)
    if not pictures:
        raise ValueError("WORD_GROUP_EMPTY")
    transform = group.find("wpg:grpSpPr/a:xfrm", NS)
    if transform is None:
        raise ValueError("WORD_GROUP_EXTENT_MISSING")
    origin, inner, outer = (transform.find("a:"+name, NS) for name in ("chOff", "chExt", "ext"))
    if any(item is None for item in (origin, inner, outer)):
        raise ValueError("WORD_GROUP_EXTENT_MISSING")
    gx, gy = float(origin.get("x")), float(origin.get("y"))
    gw, gh = float(inner.get("cx")), float(inner.get("cy"))
    ratio = (float(outer.get("cx"))/gw)/(float(outer.get("cy"))/gh)
    members = []
    rels_name = posixpath.join(posixpath.dirname(source_part), "_rels", posixpath.basename(source_part)+".rels")
    with zipfile.ZipFile(archive_path) as archive:
        relationships = etree.fromstring(archive.read(rels_name),
            parser=etree.XMLParser(resolve_entities=False, no_network=True))
        targets = {r.get("Id"):r for r in relationships}
        for index, pic in enumerate(pictures):
            rid = pic.find("pic:blipFill/a:blip", NS).get("{"+NS["r"]+"}embed")
            rel = targets[rid]
            if rel.get("TargetMode") == "External":
                raise ValueError("WORD_GROUP_EXTERNAL_IMAGE")
            part = posixpath.normpath(posixpath.join(posixpath.dirname(source_part), rel.get("Target")))
            if not part.startswith("word/"):
                raise ValueError("WORD_GROUP_IMAGE_OUTSIDE_DOCUMENT")
            payload = archive.read(part)
            source = output_dir/"_bemarkdown"/"word_appearance"/f"group-{occurrence}-{index}"/Path(part).name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(payload)
            rendered, receipt = word_picture_input(source, [owner], source.parent, index, rid)
            xfrm = pic.find("pic:spPr/a:xfrm", NS)
            if float(xfrm.get("rot","0")) % 21600000:
                raise ValueError("WORD_GROUP_ROTATED_MEMBER")
            offset, extent = xfrm.find("a:off", NS), xfrm.find("a:ext", NS)
            box = ((float(offset.get("x"))-gx)*ratio, float(offset.get("y"))-gy,
                   float(extent.get("cx"))*ratio, float(extent.get("cy")))
            with Image.open(rendered) as image:
                members.append((image.convert("RGB"), box, receipt, part))
    density = max(max(image.width/box[2], image.height/box[3]) for image, box, _, _ in members)
    width, height = math.ceil(gw*ratio*density), math.ceil(gh*density)
    if width <= 0 or height <= 0 or width*height > MAX_IMAGE_PIXELS:
        raise ValueError("WORD_GROUP_EXTENT_TOO_LARGE")
    canvas = Image.new("RGB", (width,height), "white")
    receipts = []
    for image, (x,y,w,h), receipt, part in members:
        size = (max(1,round(w*density)), max(1,round(h*density)))
        canvas.paste(image.resize(size, Image.Resampling.LANCZOS), (round(x*density),round(y*density)))
        receipts.append(dict(media_part=part, original_sha256=receipt["original_sha256"],
                             crop_box=receipt["crop_box"]))
    target = output_dir/"_bemarkdown"/"word_appearance"/f"{occurrence:06d}.png"
    canvas.save(target)
    return target, dict(policy="WORD_VISIBLE_GROUP_V1", members=receipts,
                        rendered_size=[width,height], path=target.relative_to(output_dir).as_posix(),
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest())


class NativeImageApi:
    def __init__(self, visual_runtime):
        self.visual_runtime = visual_runtime

    def process(self, document, output_dir, report, **_):
        summary = report['image_content'] = {'route': 'LOCAL_MODEL_API', 'converted_images': 0,
            'review_required': 0, 'sources': [], 'region_reviews': []}
        pattern = re.compile(r'!\[(?P<alt>(?:\\.|[^\]\\])*)\]\((?P<path>assets/[^\s)]+)(?:\s+"[^"\n]*")?\)')
        # The native scanner deliberately exports original media. A cropped,
        # rotated or flipped Word occurrence must not silently OCR hidden pixels.
        appearances = {}
        from lxml import etree
        from .namespaces import NS
        with zipfile.ZipFile(report['source']['path']) as archive:
            for part in {n.source_part for children in _inline_lists(document)
                         for n in children if isinstance(n, ImageNode)}:
                if part in archive.namelist():
                    appearances[part] = etree.fromstring(archive.read(part),
                        parser=etree.XMLParser(resolve_entities=False,no_network=True))
        image_index = 0
        for children in list(_inline_lists(document)):
            for position, node in enumerate(children):
                if not isinstance(node, ImageNode) or not node.asset_path or node.external_target:
                    continue
                path = (output_dir/node.asset_path).resolve()
                if not path.is_relative_to(output_dir.resolve()) or not path.is_file():
                    raise ValueError('DOCX_IMAGE_SOURCE_MISSING_OR_OUTSIDE_PACKAGE')
                tree=appearances.get(node.source_part)
                owners=tree.xpath(node.source_locator,namespaces=NS) if tree is not None and node.source_locator else []
                appearance=[]
                for owner in owners:
                    appearance.extend(owner.xpath('.//a:srcRect | .//a:xfrm',namespaces=NS))
                transformed=any(any(value not in ('0','false','') for key,value in item.attrib.items()
                    if key in {'l','t','r','b','rot','flipH','flipV'}) for item in appearance)
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    summary['review_required'] += 1
                    summary['region_reviews'].append({'source_part':node.source_part,
                        'source_locator':node.source_locator,'media_part':node.media_part,
                        'reason':'NON_RASTER_SOURCE_PRESERVED_WITHOUT_OCR'})
                    continue
                appearance_receipt = None
                if transformed:
                    try:
                        if len(owners) == 1 and len(owners[0].xpath(".//pic:pic", namespaces=NS)) > 1:
                            path, appearance_receipt = word_group_input(report['source']['path'], node.source_part,
                                owners[0], output_dir, image_index)
                        else:
                            path, appearance_receipt = word_picture_input(path, owners, output_dir, image_index, node.relationship_id)
                    except (ValueError, OSError) as exc:
                        summary['review_required'] += 1
                        summary['region_reviews'].append({'source_part':node.source_part,
                            'source_locator':node.source_locator,'media_part':node.media_part,
                            'reason':'WORD_IMAGE_APPEARANCE_REQUIRES_REVIEW_ORIGINAL_PRESERVED',
                            'detail':str(exc)})
                        continue
                    # Any later fallback must show the visible occurrence too.
                    node.asset_path = path.relative_to(output_dir).as_posix()
                    node.content_sha256 = appearance_receipt['sha256']
                    node.mime_type, node.extension = 'image/png', '.png'
                    node.render_method = 'WORD_VISIBLE_RECTANGULAR_PICTURE_V1'
                    node.provenance = {**(node.provenance or {}), 'word_appearance':appearance_receipt}
                index = image_index
                image_index += 1
                package = output_dir/'_bemarkdown'/'embedded'/f'{index:06d}'
                package.mkdir(parents=True)
                try:
                    result = self.visual_runtime.convert_image(path, package, document_id=f'embedded-{index:06d}')
                except InvalidRasterImageError as exc:
                    summary['review_required'] += 1
                    summary['region_reviews'].append({'source_part':node.source_part,
                        'source_locator':node.source_locator,'media_part':node.media_part,
                        'reason':'INVALID_RASTER_SOURCE_PRESERVED','detail':str(exc)})
                    continue
                stage_identities = result.report['visual_serial']['stage_identities']
                if any(identity.get('invoked') for identity in stage_identities.values()):
                    report['runtime']['vision_or_ocr_used'] = True
                markdown = (package/'document.md').read_text(encoding='utf-8')
                markdown = re.sub(r'<!-- page:\d+ -->', '', markdown).strip()
                assets = {}
                def bind(match):
                    asset = (package/match['path']).resolve()
                    if not asset.is_relative_to((package/'assets').resolve()) or not asset.is_file():
                        raise ValueError('NESTED_ASSET_MISSING_OR_OUTSIDE_PACKAGE')
                    token = f'@@BEMARKDOWN_EMBEDDED_ASSET_{index}_{len(assets)}@@'
                    assets[token] = ImageNode(asset.relative_to(output_dir).as_posix(), node.source_part,
                        node.relationship_id, node.media_part, alt=match['alt'],
                        source_locator=f'{node.source_locator}/region[{len(assets)}]',
                        provenance={'embedded_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                    return token
                markdown = pattern.sub(bind, markdown)
                if not markdown.strip():
                    # An empty layout/text result cannot erase an embedded image.
                    summary['review_required'] += 1
                    continue
                children[position] = ImageContentNode(markdown, assets, node.source_part,
                    node.source_locator, provenance={'image_source_policy': 'WORD_VISIBLE_RECTANGULAR_PICTURE_V1' if appearance_receipt else 'ORIGINAL_EMBEDDED_MEDIA', 'word_appearance': appearance_receipt})
                summary['converted_images'] += 1
                summary['review_required'] += result.report.get('review_item_count', 0)
                summary['sources'].append({'source_part': node.source_part,
                    'source_locator': node.source_locator, 'media_part': node.media_part,
                    'stage_identities': stage_identities, 'word_appearance': appearance_receipt})
        return summary
