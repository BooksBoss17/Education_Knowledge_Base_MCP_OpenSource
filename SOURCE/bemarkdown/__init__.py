"""BeMarkdown local API production interface."""
from .next_runtime import convert_document, convert_documents
from .package_io import validate_package, PackageConversionResult, PackageValidationResult, cleanup_staging
from .package import DocxResourceLimits, InvalidDocxError, ResourceLimitError
__version__ = "0.4.6+localapi.8.review.4"
