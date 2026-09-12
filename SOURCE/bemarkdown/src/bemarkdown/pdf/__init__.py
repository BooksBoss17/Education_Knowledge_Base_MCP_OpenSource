"""Production PDF modules and explicitly separated experimental evidence code.

The modular production seam does not import PaddleOCR-VL or the Phase 7 agent
stack.  The frozen experiment remains importable from
:mod:`bemarkdown.pdf_dual_paddle_evidence` only when explicitly selected.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .modular_pipeline import ModularPdfPipeline, ModularPipelineResult
    from .object_profiler import PDFObjectProfile, PDFObjectProfiler, ProfileState
    from .pipeline_mode import PDFPipelineMode, resolve_pdf_pipeline_mode

__all__ = [
    "ModularPdfPipeline",
    "ModularPipelineResult",
    "PDFObjectProfile",
    "PDFObjectProfiler",
    "PDFPipelineMode",
    "ProfileState",
    "resolve_pdf_pipeline_mode",
]


def __getattr__(name: str) -> Any:
    """Keep the public convenience exports without eager circular imports."""

    if name in {"ModularPdfPipeline", "ModularPipelineResult"}:
        from . import modular_pipeline

        return getattr(modular_pipeline, name)
    if name in {"PDFObjectProfile", "PDFObjectProfiler", "ProfileState"}:
        from . import object_profiler

        return getattr(object_profiler, name)
    if name in {"PDFPipelineMode", "resolve_pdf_pipeline_mode"}:
        from . import pipeline_mode

        return getattr(pipeline_mode, name)
    raise AttributeError(name)
