"""Bounded, invocation-local layout precomputation for raster batches.

One layout actor processes a small window, then exits before content models
start. Each item still traverses the normal single-document conversion path.
No GPU model is resident across a content-model stage and no cross-run cache
is trusted. Render pixels, original bytes and derived document identity are
verified again when consuming a prepared prediction.
"""
from __future__ import annotations

import copy
import hashlib
import io
import time
from pathlib import Path

from .image_input import IMAGE_SUFFIXES, prepared_raster_pdf


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _pixels(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


class LayoutBatchCleanupError(RuntimeError):
    """Model ownership could not be released; all later stages must stop."""


class PreparedLayout:
    """Read-only, one-source adapter; mismatches fail without fresh inference."""
    def __init__(self, item):
        self.item = item
        self.consumed = False

    def load(self):
        if _sha(self.item['source']) != self.item['source_sha256']:
            raise RuntimeError('LAYOUT_BATCH_SOURCE_CHANGED')
        return copy.deepcopy(self.item['identity'])

    def predict_image(self, image, *, document_id, page_index, page_render_identity):
        if self.consumed:
            raise RuntimeError('LAYOUT_BATCH_PREDICTION_ALREADY_CONSUMED')
        actual = (document_id, page_index, page_render_identity, list(image.size), _pixels(image))
        expected = tuple(self.item[name] for name in
                         ('document_id', 'page_index', 'render_sha256', 'size', 'pixels_sha256'))
        if actual != expected:
            raise RuntimeError('LAYOUT_BATCH_RENDER_BINDING_MISMATCH')
        self.load()
        self.consumed = True
        return copy.deepcopy(self.item['value'])

    def unload(self):
        return {'unload_status': 'PRECOMPUTED_LAYOUT_ACTOR_ALREADY_EXITED'}


class LayoutBatchPreparer:
    """Keep only detection records for at most four images, never image pixels."""
    def __init__(self, size=1, *, registry_options=None, runtime_factory=None):
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 4:
            raise ValueError('layout_batch_size must be an integer from 1 to 4')
        self.size = size
        self.options = dict(registry_options or {})
        self.runtime_factory = runtime_factory
        self.items = {}
        self.errors = {}
        self.timings = {}
        self.metrics = {'enabled': size > 1, 'window_size': size,
                        'prepared_inputs': 0, 'layout_actor_windows': 0,
                        'preparation_seconds': 0.0,
                        'scope': 'RASTER_LAYOUT_ONLY_SINGLE_IMAGE_INFERENCE'}

    def prepare(self, sources, *, continue_on_error=False):
        from PIL import Image
        from .production import build_document_id
        from .pdf_layout_runtime import ProductionPdfPageRenderer
        from .pdf_layout_runtime import CAPTURE_FLOOR
        from .isolated_native_backends import NativeBackendFactory

        self.items.clear()
        self.errors.clear()
        self.timings.clear()
        if self.size == 1:
            return
        sources = list(sources)
        if len(sources) > self.size:
            raise ValueError('LAYOUT_BATCH_WINDOW_EXCEEDED')
        if len({Path(source).resolve() for source in sources}) != len(sources):
            raise ValueError('LAYOUT_BATCH_DUPLICATE_SOURCE')
        actor = None
        factory = None
        started = time.perf_counter()
        try:
            for source in sources:
                source = Path(source).resolve()
                if source.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                item_started = time.perf_counter()
                try:
                    digest = _sha(source)
                    with prepared_raster_pdf(source, expected_source_sha256=digest) as (derived, _):
                        renders = ProductionPdfPageRenderer().render(derived)
                        if len(renders) != 1:
                            raise RuntimeError('LAYOUT_BATCH_EXPECTED_ONE_PAGE')
                        render = renders[0]
                        document_id = build_document_id(derived)
                        if actor is None:
                            if self.runtime_factory is not None:
                                actor = self.runtime_factory()
                            else:
                                factory = NativeBackendFactory(**self.options)
                                actor = factory.layout(capture_floor=CAPTURE_FLOOR)
                            self.metrics['layout_actor_windows'] += 1
                        identity = actor.load()
                        with Image.open(io.BytesIO(render.image_bytes)) as opened:
                            with opened.convert('RGB') as image:
                                value = actor.predict_image(image, document_id=document_id,
                                    page_index=render.page_index,
                                    page_render_identity=render.render_sha256)
                                item = dict(source=str(source), source_sha256=digest,
                                    document_id=document_id, page_index=render.page_index,
                                    render_sha256=render.render_sha256, size=list(image.size),
                                    pixels_sha256=_pixels(image), identity=copy.deepcopy(identity),
                                    value=value)
                        if _sha(source) != digest:
                            raise RuntimeError('LAYOUT_BATCH_SOURCE_CHANGED')
                        self.items[source] = item
                        self.metrics['prepared_inputs'] += 1
                except Exception as error:
                    self.errors[source] = error
                    # A failed remote actor is terminal, but prepared earlier inputs
                    # remain usable. Never restart inference automatically here.
                    if actor is not None:
                        try:
                            actor.unload()
                        except BaseException as cleanup_error:
                            self.items.clear()
                            raise LayoutBatchCleanupError('LAYOUT_BATCH_CLEANUP_UNCONFIRMED') from cleanup_error
                        finally:
                            actor = None
                    if not continue_on_error:
                        break
                finally:
                    self.timings[source] = time.perf_counter() - item_started
        finally:
            try:
                try:
                    if actor is not None:
                        actor.unload()
                finally:
                    if factory is not None:
                        factory.close()
            except BaseException as cleanup_error:
                self.items.clear()
                raise LayoutBatchCleanupError('LAYOUT_BATCH_CLEANUP_UNCONFIRMED') from cleanup_error
            finally:
                self.metrics['preparation_seconds'] += time.perf_counter() - started

    def pdf_factory(self, source, **options):
        source = Path(source).resolve()
        if source in self.errors:
            raise self.errors[source]
        if source not in self.items:
            return None
        from .pdf.production_runtime import create_production_pdf_runtime
        item = self.items[source]
        return lambda: create_production_pdf_runtime(
            **self.options, **options, layout_runtime_factory=lambda: PreparedLayout(item))
