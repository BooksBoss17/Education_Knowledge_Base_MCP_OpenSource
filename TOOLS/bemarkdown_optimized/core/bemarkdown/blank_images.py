"""Exact-color exclusion and lossless preparation of model-only image inputs."""
import hashlib
import math
from pathlib import Path


ASPECT_PADDING_POLICY = 'aspect-padding-cpu-candidate-v1'
ASPECT_PADDING_MAX_RATIO = 200
ASPECT_PADDING_MIN_SHORT_EDGE = 32
ASPECT_PADDING_MAX_PIXELS = 40_000_000


def prepare_model_image(image):
    """Pad extreme ratios without resizing, cropping, or deciding retention.

    The caller owns the returned PIL image. Source files and delivery assets
    remain untouched. Oversized derived images fail before allocating pixels.
    """
    from PIL import Image

    if getattr(image, 'n_frames', 1) != 1:
        raise ValueError('MULTIFRAME_MODEL_INPUT_UNSUPPORTED')
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError('INVALID_MODEL_IMAGE_SIZE')
    metadata = {
        'policy': ASPECT_PADDING_POLICY,
        'changed': False,
        'original_size': [width, height],
        'output_size': [width, height],
        'resized_or_cropped': False,
    }
    if max(width, height) / min(width, height) <= ASPECT_PADDING_MAX_RATIO:
        return image.copy(), metadata

    needed = max(ASPECT_PADDING_MIN_SHORT_EDGE,
                 math.ceil(max(width, height) / ASPECT_PADDING_MAX_RATIO))
    output_size = ((max(width, needed), height) if width < height
                   else (width, max(height, needed)))
    if output_size[0] * output_size[1] > ASPECT_PADDING_MAX_PIXELS:
        raise ValueError('EXTREME_ASPECT_PADDING_PIXEL_BUDGET_EXCEEDED')
    left = (output_size[0] - width) // 2
    top = (output_size[1] - height) // 2
    with image.convert('RGBA') as rgba:
        original_sha = hashlib.sha256(rgba.tobytes()).hexdigest()
        padded = Image.new('RGBA', output_size, (255, 255, 255, 255))
        # No alpha mask: preserve invisible RGB and partially opaque pixels.
        padded.paste(rgba, (left, top))
    metadata.update({
        'changed': True,
        'output_size': list(output_size),
        'original_region': [left, top, left + width, top + height],
        'original_rgba_sha256': original_sha,
        'effective_rgba_sha256': hashlib.sha256(padded.tobytes()).hexdigest(),
    })
    return padded, metadata


def blank_reason(path):
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(path) as image:
            # Do not infer an entire animation's content from its first frame.
            if getattr(image, 'n_frames', 1) != 1:
                return None
            rgba = image.convert('RGBA')
            extrema = rgba.getextrema()
            if extrema[3][1] == 0:
                return 'FULLY_TRANSPARENT'
            if all(low == high for low, high in extrema):
                return 'SINGLE_COLOR'
            colors = rgba.getcolors(maxcolors=2)
            if colors is not None and len(colors) == 2:
                # Exact colors only; preserve any visible black, including
                # black-on-white and black-on-transparent text or line art.
                if not any(color[:3] == (0, 0, 0) and color[3] > 0
                           for _, color in colors):
                    return 'TWO_COLOR_NO_BLACK'
    except (OSError, ValueError, UnidentifiedImageError):
        pass
    return None


def prune_docx(document, output_dir, report):
    from .ir import ImageNode, ImageContentNode, HyperlinkNode, TableNode
    root = Path(output_dir).resolve()
    removed = report.setdefault('blank_images_removed', [])
    cache = {}

    def reason(node):
        if not node.asset_path:
            return None
        path = (root / node.asset_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return None
        if path not in cache:
            cache[path] = blank_reason(path)
        found = cache[path]
        if found:
            removed.append(dict(source_ref=node.asset_path, source_locator=node.source_locator, reason=found))
        return found

    def inlines(nodes):
        kept = []
        for node in nodes:
            if isinstance(node, ImageNode) and reason(node):
                continue
            if isinstance(node, HyperlinkNode):
                node.children = inlines(node.children)
            if isinstance(node, ImageContentNode):
                for token, asset in list(node.assets.items()):
                    if reason(asset):
                        node.markdown = node.markdown.replace(token, '')
                        del node.assets[token]
            kept.append(node)
        return kept

    def blocks(items):
        for block in items:
            if isinstance(block, TableNode):
                for row in block.rows:
                    for cell in row:
                        blocks(cell.blocks)
            else:
                block.children = inlines(block.children)
    blocks(document.blocks)


def prune_pdf(document, source_ref):
    """Remove blank IMAGE blocks before rendering and visible asset numbering.

    Formula/table/text fallback nodes retain their unrecognized-content status.
    Their image is evidence for an unresolved recognition task, not decoration.
    """
    assets = {str(a['asset_uid']): a for a in document.get('assets', [])}
    removed = []
    kept = []
    cache = {}
    for block in document.get('blocks', []):
        uid = str(block.get('content', {}).get('asset_uid') or '')
        asset = assets.get(uid)
        reason = None
        if block.get('kind') == 'IMAGE' and asset:
            ref = source_ref(asset)
            if ref:
                if str(ref) not in cache:
                    cache[str(ref)] = blank_reason(ref)
                reason = cache[str(ref)]
        if reason:
            removed.append(dict(node_id=block.get('node_id'), asset_uid=uid, reason=reason,
                                content_sha256=asset.get('content_sha256')))
        else:
            kept.append(block)
    document['blocks'] = kept
    return removed
