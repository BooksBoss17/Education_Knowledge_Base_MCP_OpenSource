"""Public BeMarkdown production interface.

Developer census and benchmark exports remain available lazily for source-tree
compatibility, but importing :mod:`bemarkdown` never loads those harnesses.
"""

from importlib import import_module
from typing import Any

from .batch import convert_documents
from .formula_ocr import FormulaOcrAdapter, FormulaOcrSafetyGate, FormulaOcrState
from .formulanet_runtime import FormulaOcrOutputValidator, OcrVerdict
from .mtef_cache import MtefCacheContext
from .package import DocxResourceLimits, InvalidDocxError, ResourceLimitError
from .pipeline import ConversionResult, convert_docx
from .production import (
    PackageConversionResult,
    PackageValidationResult,
    cleanup_staging,
    convert_document,
    validate_package,
)
from .validator import FormulaStructuralValidator, FormulaVerdict

_LAZY_DEVELOPER_EXPORTS = {
    "DocxCensusResult": (".docx_census", "DocxCensusResult"),
    "FormulaNetBenchmarkResult": (
        ".formulanet_benchmark",
        "FormulaNetBenchmarkResult",
    ),
    "run_docx_formula_census": (".docx_census", "run_docx_formula_census"),
    "run_formulanet_benchmark": (
        ".formulanet_benchmark",
        "run_formulanet_benchmark",
    ),
}


def __getattr__(name: str) -> Any:
    """Load legacy developer exports only when a caller explicitly requests one."""

    target = _LAZY_DEVELOPER_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "ConversionResult",
    "DocxCensusResult",
    "DocxResourceLimits",
    "FormulaNetBenchmarkResult",
    "FormulaOcrAdapter",
    "FormulaOcrOutputValidator",
    "FormulaOcrSafetyGate",
    "FormulaOcrState",
    "FormulaStructuralValidator",
    "FormulaVerdict",
    "InvalidDocxError",
    "MtefCacheContext",
    "OcrVerdict",
    "PackageConversionResult",
    "PackageValidationResult",
    "ResourceLimitError",
    "cleanup_staging",
    "convert_document",
    "convert_documents",
    "convert_docx",
    "run_docx_formula_census",
    "run_formulanet_benchmark",
    "validate_package",
]
__version__ = "0.2.0"
