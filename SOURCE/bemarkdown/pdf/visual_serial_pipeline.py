"""Visual-only serial transcription contracts.

This module deliberately contains no PDF text extraction and no model SDK
imports.  It is the seam between a layout adapter and the two managed OCR
stages used by the new route:

    PP-DocLayoutV3 -> Xiaomi-OCR-0 (formula regions) -> OvisOCR3 (other text)

The planner removes formula-overlapping text regions before the text stage is
started.  The executor keeps the two stages strictly serial and records the
actual stage order in its result.  Model-specific adapters can be supplied by
the production runtime without changing these invariants.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .visual_content_guard import MATH_SPAN, clean_text_output, duplicate_text_regions, output_problem


VISUAL_SERIAL_SCHEMA = "bemarkdown-visual-serial-transcription-v1"
VISUAL_SERIAL_POLICY = "bemarkdown-v3-xiaomi-formula-ovisocr3-text-v1"
LOCAL_API_POLICY = "bemarkdown-local-model-api-v1"
LAYOUT_MODEL_ID = "pp-doclayout-v3"
FORMULA_MODEL_ID = "xiaomi-ocr-0"
# Keep the registry spelling used by the installed OvisOCR3 payload.  The
# display name is OvisOCR3, but the developer registry uses the compact id.
TEXT_MODEL_ID = "ovisocr3"

_FORMULA_LABELS = frozenset(
    {"formula", "display_formula", "inline_formula", "formula_number", "equation"}
)
_TEXT_LABELS = frozenset(
    {
        "text",
        "paragraph_title",
        "doc_title",
        "abstract",
        "content",
        "figure_title",
        "reference",
        "reference_content",
        "footnote",
        "header",
        "footer",
        "aside_text",
        "number",
    }
)
# Tables and algorithm diagrams stay as bounded visual assets until a
# separately verified serializer exists. Sending them through the ordinary
# text stage would recreate the high-memory, low-quality TABLE_HTML candidate.
_PRESERVED_VISUAL_LABELS = frozenset({"table", "algorithm"})
_INLINE_FORMULA_PATTERNS = (
    re.compile(r"\\\[(?:.|\n)*?\\\]"),
    re.compile(r"\\\((?:.|\n)*?\\\)"),
    re.compile(r"\$(?:[^$\n]|\\.)+\$"),
)
_MIXED_OUTPUT_WRAPPER_PATTERNS = (
    re.compile(r"```(?:latex|tex)?\s*(.*?)```", re.IGNORECASE | re.DOTALL),
    re.compile(r"\\\[(.*?)\\\]", re.DOTALL),
    re.compile(r"\\\((.*?)\\\)", re.DOTALL),
    re.compile(r"\$\$(.*?)\$\$", re.DOTALL),
    re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", re.DOTALL),
)


class VisualSerialContractError(ValueError):
    """The visual-only route would violate a hard routing invariant."""


@dataclass(frozen=True, slots=True)
class VisualSerialPolicy:
    """Frozen behavior of the new route."""

    schema: str = VISUAL_SERIAL_SCHEMA
    policy_id: str = VISUAL_SERIAL_POLICY
    layout_model_id: str = LAYOUT_MODEL_ID
    formula_model_id: str = FORMULA_MODEL_ID
    text_model_id: str = TEXT_MODEL_ID
    table_model_id: str = ""
    parse_pdf_native: bool = False
    image_screening: bool = False
    final_review: bool = False
    serial_stages: bool = True
    # Retained for old serialized policy readers; current masking handles every
    # positive intersection after checking the corresponding formula result.
    formula_overlap_threshold: float = 0.25
    prompt_schema: str = "bemarkdown-visual-serial-prompts-v2"
    formula_prompt_fingerprint: str = ""
    text_prompt_fingerprint: str = ""
    prompt_source_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != VISUAL_SERIAL_SCHEMA:
            raise VisualSerialContractError("VISUAL_SERIAL_SCHEMA_MISMATCH")
        api_mode = self.policy_id == LOCAL_API_POLICY
        if self.table_model_id and (not api_mode or self.table_model_id != "api:table"):
            raise VisualSerialContractError("VISUAL_SERIAL_TABLE_ROLE_BINDING_INVALID")
        if self.policy_id not in {VISUAL_SERIAL_POLICY, LOCAL_API_POLICY}:
            raise VisualSerialContractError("VISUAL_SERIAL_POLICY_UNKNOWN")
        if not api_mode and self.layout_model_id != LAYOUT_MODEL_ID:
            raise VisualSerialContractError("VISUAL_SERIAL_LAYOUT_MODEL_MISMATCH")
        if not api_mode and self.formula_model_id != FORMULA_MODEL_ID:
            raise VisualSerialContractError("VISUAL_SERIAL_FORMULA_MODEL_MISMATCH")
        if not api_mode and self.text_model_id != TEXT_MODEL_ID:
            raise VisualSerialContractError("VISUAL_SERIAL_TEXT_MODEL_MISMATCH")
        if self.parse_pdf_native:
            raise VisualSerialContractError("VISUAL_SERIAL_NATIVE_PDF_PARSING_FORBIDDEN")
        if self.image_screening or self.final_review:
            raise VisualSerialContractError("VISUAL_SERIAL_EXTRA_STAGE_FORBIDDEN")
        if not self.serial_stages:
            raise VisualSerialContractError("VISUAL_SERIAL_STAGES_MUST_BE_SERIAL")
        if not 0.0 < self.formula_overlap_threshold <= 1.0:
            raise VisualSerialContractError("VISUAL_SERIAL_OVERLAP_THRESHOLD_INVALID")
        if api_mode:
            # These are role bindings; the responding model and its prompt
            # fingerprint are recorded in stage_identities after inference.
            if (self.layout_model_id, self.formula_model_id, self.text_model_id) != (
                "api:layout", "api:formula", "api:text"
            ) or self.prompt_schema != LOCAL_API_POLICY:
                raise VisualSerialContractError("VISUAL_SERIAL_API_ROLE_BINDING_INVALID")
            return
        from .visual_serial_prompts import (
            PROMPT_SCHEMA,
            prompt_fingerprint,
            prompt_source_sha256,
        )

        if self.prompt_schema != PROMPT_SCHEMA:
            raise VisualSerialContractError("VISUAL_SERIAL_PROMPT_SCHEMA_MISMATCH")
        if not self.formula_prompt_fingerprint:
            object.__setattr__(self, "formula_prompt_fingerprint", prompt_fingerprint(self.formula_model_id))
        if not self.text_prompt_fingerprint:
            object.__setattr__(self, "text_prompt_fingerprint", prompt_fingerprint(self.text_model_id))
        if not self.prompt_source_sha256:
            object.__setattr__(self, "prompt_source_sha256", prompt_source_sha256())
        for value in (
            self.formula_prompt_fingerprint,
            self.text_prompt_fingerprint,
            self.prompt_source_sha256,
        ):
            if len(value) != 64:
                raise VisualSerialContractError("VISUAL_SERIAL_PROMPT_FINGERPRINT_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "layout_model_id": self.layout_model_id,
            "formula_model_id": self.formula_model_id,
            "text_model_id": self.text_model_id,
            "table_model_id": self.table_model_id,
            "parse_pdf_native": self.parse_pdf_native,
            "image_screening": self.image_screening,
            "final_review": self.final_review,
            "serial_stages": self.serial_stages,
            "formula_overlap_threshold": self.formula_overlap_threshold,
            "formula_mask_policy": "ANY_POSITIVE_INTERSECTION_WITH_USABLE_RESULT_V2",
            "prompt_schema": self.prompt_schema,
            "formula_prompt_fingerprint": self.formula_prompt_fingerprint,
            "text_prompt_fingerprint": self.text_prompt_fingerprint,
            "prompt_source_sha256": self.prompt_source_sha256,
        }


@dataclass(frozen=True, slots=True)
class VisualRegion:
    region_id: str
    label: str
    bbox: tuple[float, float, float, float]
    crop: Mapping[str, Any]
    page_index: int = 0
    reading_order: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise VisualSerialContractError("VISUAL_REGION_ID_REQUIRED")
        if len(self.bbox) != 4 or self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise VisualSerialContractError("VISUAL_REGION_BBOX_INVALID")
        if not self.crop.get("path") or len(str(self.crop.get("sha256") or self.crop.get("content_sha256") or "")) != 64:
            raise VisualSerialContractError("VISUAL_REGION_CROP_SHA256_REQUIRED")

    @property
    def normalized_label(self) -> str:
        return self.label.strip().lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "label": self.label,
            "bbox": list(self.bbox),
            "crop": dict(self.crop),
            "page_index": self.page_index,
            "reading_order": self.reading_order,
            "metadata": dict(self.metadata),
        }


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _overlap_fraction(first: Sequence[float], second: Sequence[float]) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    smaller = min(_area(first), _area(second))
    return overlap / smaller if smaller else 0.0


def _as_region(value: VisualRegion | Mapping[str, Any], index: int) -> VisualRegion:
    if isinstance(value, VisualRegion):
        return value
    crop = dict(value.get("crop") or {})
    if "sha256" not in crop and crop.get("content_sha256"):
        crop["sha256"] = crop["content_sha256"]
    bbox = tuple(float(item) for item in value.get("bbox") or value.get("bbox_pdf_pt") or ())
    return VisualRegion(
        region_id=str(value.get("region_id") or value.get("id") or f"region-{index:06d}"),
        label=str(value.get("label") or value.get("raw_label") or "unknown"),
        bbox=bbox, crop=crop, page_index=int(value.get("page_index", 0)),
        reading_order=int(value.get("reading_order", index)), metadata=dict(value.get("metadata") or {}),
    )


@dataclass(frozen=True, slots=True)
class VisualSerialPlan:
    policy: VisualSerialPolicy
    all_regions: tuple[VisualRegion, ...]
    formula_regions: tuple[VisualRegion, ...]
    text_regions: tuple[VisualRegion, ...]
    preserved_regions: tuple[VisualRegion, ...]
    formula_overlap_removed: tuple[str, ...] = ()
    table_regions: tuple[VisualRegion, ...] = ()
    table_owned_regions: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.policy.schema,
            "policy": self.policy.to_dict(),
            "all_region_ids": [r.region_id for r in self.all_regions],
            "formula_region_ids": [r.region_id for r in self.formula_regions],
            "text_region_ids": [r.region_id for r in self.text_regions],
            "table_region_ids": [r.region_id for r in self.table_regions],
            "table_owned_regions": dict(self.table_owned_regions),
            "preserved_region_ids": [r.region_id for r in self.preserved_regions],
            "formula_text_overlap_ids": list(self.formula_overlap_removed),
            "regions": [r.to_dict() for r in self.all_regions],
        }


def build_visual_serial_plan(
    regions: Sequence[VisualRegion | Mapping[str, Any]],
    *,
    policy: VisualSerialPolicy | None = None,
) -> VisualSerialPlan:
    """Partition layout regions into formula, text and preserved visual inputs."""

    policy = policy or VisualSerialPolicy()
    normalized = sorted(
        (_as_region(value, index) for index, value in enumerate(regions)),
        key=lambda row: (row.page_index, row.reading_order, row.region_id),
    )
    ids = [row.region_id for row in normalized]
    if len(ids) != len(set(ids)):
        raise VisualSerialContractError("VISUAL_REGION_IDS_MUST_BE_UNIQUE")
    from .spatial_assembly import deduplicate
    normalized = deduplicate(normalized)
    from .table_content import table_ownership
    tables, owned = table_ownership(normalized) if policy.table_model_id else ((), {})
    table_ids = {r.region_id for r in tables}
    formulas = tuple(row for row in normalized if row.normalized_label in _FORMULA_LABELS and row.region_id not in owned)
    formula_ids = {row.region_id for row in formulas}
    texts: list[VisualRegion] = []
    preserved: list[VisualRegion] = []
    overlap_removed: list[str] = []
    for row in normalized:
        if row.region_id in formula_ids or row.region_id in owned or row.region_id in table_ids:
            continue
        if row.normalized_label in _PRESERVED_VISUAL_LABELS:
            preserved.append(row)
            continue
        if row.normalized_label in _TEXT_LABELS:
            overlaps = [
                formula for formula in formulas
                if formula.page_index == row.page_index
                and _overlap_fraction(row.bbox, formula.bbox) > 0
            ]
            if overlaps:
                # The text model must never see a crop that contains a formula
                # region. Preserve the layout crop as provenance for downstream
                # assembly, while sending only the disjoint set to OvisOCR3.
                overlap_removed.append(row.region_id)
                continue
            texts.append(row)
        else:
            preserved.append(row)
    text_inputs = tuple(row for row in texts if not row.metadata.get("formula_overlap_removed"))
    return VisualSerialPlan(
        policy, tuple(normalized), formulas, text_inputs, tuple(preserved), tuple(overlap_removed), tables, owned
    )


def materialize_masked_text_plan(
    plan: VisualSerialPlan,
    output_dir: str | Path,
    *,
    accepted_formula_ids: set[str] | None = None,
    mask_kind: str = "formula",
) -> VisualSerialPlan:
    """Create residual text crops with formula pixels masked out.

    V3 sometimes emits a broad text box around a formula.  Dropping that box
    avoids duplicate formula recognition but can lose nearby prose.  This
    helper derives a new crop for each such box, paints the intersecting
    formula rectangles white, and gives the derived region a new ID so the
    formula/text disjointness check remains exact.  The original crop and
    formula IDs stay in metadata for provenance.
    """

    if not plan.formula_overlap_removed:
        return plan
    from PIL import Image, ImageDraw

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formula_by_page = {}
    for formula in plan.formula_regions:
        if accepted_formula_ids is None or formula.region_id in accepted_formula_ids:
            formula_by_page.setdefault(formula.page_index, []).append(formula)
    removed = set(plan.formula_overlap_removed)
    derived: list[VisualRegion] = list(plan.text_regions)
    for original in plan.all_regions:
        if original.region_id not in removed:
            continue
        source = Path(str(original.crop["path"])).resolve()
        try:
            with Image.open(source) as source_image:
                image = source_image.convert("RGB")
                width, height = image.size
                draw = ImageDraw.Draw(image)
                masked_ids: list[str] = []
                region_width = original.bbox[2] - original.bbox[0]
                region_height = original.bbox[3] - original.bbox[1]
                if region_width <= 0 or region_height <= 0:
                    raise VisualSerialContractError("VISUAL_RESIDUAL_CROP_GEOMETRY_INVALID")
                for formula in formula_by_page.get(original.page_index, ()):
                    left = max(original.bbox[0], formula.bbox[0])
                    top = max(original.bbox[1], formula.bbox[1])
                    right = min(original.bbox[2], formula.bbox[2])
                    bottom = min(original.bbox[3], formula.bbox[3])
                    if right <= left or bottom <= top:
                        continue
                    masked_ids.append(formula.region_id)
                    draw.rectangle(
                        (
                            max(0, int((left - original.bbox[0]) / region_width * width)),
                            max(0, int((top - original.bbox[1]) / region_height * height)),
                            min(width - 1, math.ceil((right - original.bbox[0]) / region_width * width) - 1),
                            min(height - 1, math.ceil((bottom - original.bbox[1]) / region_height * height) - 1),
                        ),
                        fill="white",
                    )
                if not masked_ids:
                    derived.append(original)
                    continue
                derived_id = f"{original.region_id}__residual" if mask_kind == "formula" else f"{original.region_id}__table_residual"
                target = output_dir / f"{derived_id}.png"
                image.save(target, format="PNG")
                crop = {
                    "path": str(target),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "source_path": str(original.crop.get("source_path") or source),
                    "source_sha256": str(original.crop.get("source_sha256") or original.crop.get("sha256") or original.crop.get("content_sha256")),
                    "mask_policy": mask_kind.upper()+"_RECTANGLES_WHITE_V1",
                    "masked_"+mask_kind+"_region_ids": masked_ids,
                }
                derived.append(
                    VisualRegion(
                        region_id=derived_id,
                        label=original.label,
                        bbox=original.bbox,
                        crop=crop,
                        page_index=original.page_index,
                        reading_order=original.reading_order,
                        metadata={
                            **dict(original.metadata),
                            "source_region_id": original.metadata.get("source_region_id", original.region_id),
                            mask_kind+"_masked": True,
                            "masked_"+mask_kind+"_region_ids": masked_ids,
                        },
                    )
                )
                image.close()
        except OSError as exc:
            raise VisualSerialContractError(f"VISUAL_RESIDUAL_CROP_FAILED:{source}") from exc
    derived.sort(key=lambda row: (row.page_index, row.reading_order, row.region_id))
    return VisualSerialPlan(
        policy=plan.policy,
        all_regions=plan.all_regions,
        formula_regions=plan.formula_regions,
        text_regions=tuple(derived),
        preserved_regions=plan.preserved_regions,
        formula_overlap_removed=plan.formula_overlap_removed,
        table_regions=plan.table_regions,
        table_owned_regions=plan.table_owned_regions,
    )


def prepare_text_inputs(plan: VisualSerialPlan, formulas: Mapping[str, str],
                        output_dir: str | Path | None) -> VisualSerialPlan:
    """Mask only usable formula results; recover failed formula pixels once."""
    accepted = {key for key, value in formulas.items() if not output_problem(value, formula=True)}
    if plan.formula_overlap_removed:
        if output_dir is None:
            raise VisualSerialContractError("VISUAL_SERIAL_RESIDUAL_CROP_ROOT_REQUIRED")
        plan = materialize_masked_text_plan(plan, Path(output_dir) / "_bemarkdown/visual_serial/masked_crops",
                                            accepted_formula_ids=accepted)
    texts = list(plan.text_regions)
    for formula in plan.formula_regions:
        if formula.region_id in accepted:
            continue
        a, b, c, d = formula.bbox
        area = (c-a)*(d-b)
        covering = []
        for text in texts:
            if text.page_index != formula.page_index:
                continue
            x0, y0, x1, y1 = text.bbox
            intersection = max(0, min(x1,c)-max(x0,a))*max(0, min(y1,d)-max(y0,b))
            if area > 0 and intersection / area >= 0.98:
                covering.append(text)
        if covering:
            continue
        texts.append(replace(formula, region_id=formula.region_id+"__text_fallback", label="text",
                             metadata={**dict(formula.metadata), "fallback_formula_region_id": formula.region_id,
                                       "review_reason": "FORMULA_TEXT_FALLBACK_REVIEW"}))
    return replace(plan, text_regions=tuple(sorted(texts, key=lambda r: (r.page_index, r.reading_order, r.region_id))))


def add_page_fallbacks(regions: Sequence[Any], renders: Sequence[Any], output_dir: Any) -> list[Any]:
    """At most one text request per page without detected transcribable regions."""
    normalized = [_as_region(value, i) for i, value in enumerate(regions)]
    if output_dir is None:
        return normalized
    from io import BytesIO
    from PIL import Image
    folder = Path(output_dir) / "_bemarkdown/visual_serial/fallback_pages"
    for index, render in enumerate(renders):
        page = [r for r in normalized if r.page_index == index]
        if any(r.normalized_label in _FORMULA_LABELS | _TEXT_LABELS | _PRESERVED_VISUAL_LABELS for r in page):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"page-{index}.png"
        with Image.open(BytesIO(render.image_bytes)) as image:
            image.save(path, format="PNG")
            box = (0., 0., float(image.width), float(image.height))
        reason = "NO_LAYOUT_REGIONS_TEXT_FALLBACK_REVIEW" if not page else "IMAGE_ONLY_PAGE_TEXT_FALLBACK_REVIEW"
        normalized.append(VisualRegion(region_id=f"p{index:04d}__page_text_fallback", label="text", bbox=box,
                          page_index=index, reading_order=0,
                          crop={"path":str(path), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()},
                          metadata={"review_reason":reason, "preserve_source":True}))
    return normalized


class VisualSerialStageError(RuntimeError):
    """A model stage failed or returned a result with the wrong cardinality."""


def strip_explicit_formula_fragments(value: Any) -> tuple[Any, int]:
    """Remove only explicitly delimited LaTeX fragments from text output."""

    if not isinstance(value, str):
        return value, 0
    cleaned = value
    removed = 0
    cleaned, removed = MATH_SPAN.subn("", cleaned)
    return cleaned, removed


def strip_mixed_serialization_wrappers(value: Any) -> tuple[Any, int]:
    """Remove model-added math wrappers while preserving mixed TeX content."""

    if not isinstance(value, str):
        return value, 0
    cleaned = value
    removed = 0
    for pattern in _MIXED_OUTPUT_WRAPPER_PATTERNS:
        def replace(match: re.Match[str]) -> str:
            nonlocal removed
            body = match.group(1)
            # A lone dollar sign in ordinary text is not a math wrapper.
            if pattern.pattern.startswith("(?<!\\$)") and not re.search(
                r"\\|[{}_^=]|\b(?:frac|times|div|sqrt|pi|sin|cos|tan)\b",
                body,
            ):
                return match.group(0)
            removed += 1
            return body

        cleaned = pattern.sub(replace, cleaned)
    return cleaned.strip(), removed


def normalize_formula_output(value: Any) -> Any:
    """Remove only whole-output display wrappers from formula OCR.

    Xiaomi occasionally emits a display wrapper even though the route stores
    LaTeX directly.  Removing a paired outer wrapper preserves the formula
    content while keeping the serialized value compatible with the package
    and scorer contracts.  Inline delimiters inside the expression are left
    untouched and malformed/unpaired wrappers fail closed.
    """

    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]).strip()
    wrappers = ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$"))
    changed = True
    while changed:
        changed = False
        spans = list(MATH_SPAN.finditer(cleaned))
        if spans and (len(spans) != 1 or spans[0].span() != (0, len(cleaned))):
            break
        for left, right in wrappers:
            if cleaned.startswith(left) and cleaned.endswith(right) and len(cleaned) >= len(left) + len(right):
                cleaned = cleaned[len(left) : -len(right)].strip()
                changed = True
                break
    return cleaned


def formula_output_is_markdown(value: str) -> bool:
    """Keep marked math spans and ordinary prose outside TeX groups verbatim."""
    if MATH_SPAN.search(value):
        return True
    depth = 0
    for piece in re.findall(r"\\[A-Za-z]+|\\.|[{}]|[^\\{}]+", value):
        if piece == "{":
            depth += 1
        elif piece == "}":
            depth -= 1
        elif depth == 0 and not piece.startswith("\\"):
            if re.search(r"[\u3400-\u9fff]|\b[A-Za-z]{2,}\s+[A-Za-z]{2,}\b", piece):
                return True
    return False


class VisualSerialRuntimeAdapter:
    """Production adapter boundary for the visual-only route.

    The adapter owns page-image layout inference, crop materialization and
    package assembly.  It is intentionally separate from the legacy
    ``ModularPdfPipeline``: that pipeline starts with PDF source inspection and
    therefore cannot be used for this route.  A concrete adapter must provide
    ``convert_visual`` and may use :class:`VisualSerialExecutor` for the two
    OCR stages.
    """

    def convert_visual(self, **kwargs: Any) -> Any:  # pragma: no cover - contract only
        raise NotImplementedError("VISUAL_SERIAL_RUNTIME_ADAPTER_NOT_IMPLEMENTED")


class VisualSerialOrchestrator(VisualSerialRuntimeAdapter):
    """Model-agnostic orchestration for one visual document.

    Concrete V3/Xiaomi/Ovis adapters are injected.  Keeping orchestration
    model-agnostic lets the production process enforce routing invariants
    without importing Torch or reading PDF-native content.  ``result_factory``
    is responsible for materializing the normal BeMarkdown package result.
    """

    def __init__(
        self,
        *,
        layout_runtime: Any,
        formula_runtime: Any,
        text_runtime: Any,
        result_factory: Callable[..., Any],
        policy: VisualSerialPolicy | None = None,
        table_runtime: Any = None,
    ) -> None:
        self.policy = policy or VisualSerialPolicy()
        self.layout_runtime = layout_runtime
        self.result_factory = result_factory
        self.executor = VisualSerialExecutor(
            formula_runtime=formula_runtime,
            text_runtime=text_runtime,
            table_runtime=table_runtime,
            policy=self.policy,
        )

    def convert_visual(self, **kwargs: Any) -> Any:
        renders = tuple(kwargs.get("renders") or ())
        if not renders:
            raise VisualSerialContractError("VISUAL_SERIAL_NO_PAGE_RENDERS")
        detect = getattr(self.layout_runtime, "detect", None)
        if not callable(detect):
            raise VisualSerialContractError("VISUAL_SERIAL_LAYOUT_ADAPTER_INVALID")
        output_dir = kwargs.get("output_dir")
        try:
            if "output_dir" in inspect.signature(detect).parameters:
                regions = detect(renders, output_dir=output_dir)
            else:
                regions = detect(renders)
        finally:
            close_layout = getattr(self.layout_runtime, "close", None)
            if callable(close_layout):
                close_layout()
        regions = add_page_fallbacks(regions, renders, output_dir)
        plan = build_visual_serial_plan(regions, policy=self.policy)
        from .spatial_assembly import complete_figure_edges
        plan = complete_figure_edges(plan, renders, output_dir)
        from .text_overlap import repair_text_overlap
        plan, overlap_decisions = repair_text_overlap(plan, renders, output_dir)
        stage_result = self.executor.execute(plan, output_dir=output_dir)
        stage_result['text_overlap_adjustments'] = overlap_decisions
        stage_result['review_items'].extend(
            {'region_id': row['region_id'], 'reason': 'TEXT_EDGE_OVERLAP_REVIEW',
             'detail': row['reason'], 'owner_region_id': row['owner_region_id']}
            for row in overlap_decisions if row['action'] == 'PRESERVED'
        )
        plan = self.executor.effective_plan
        assert_visual_serial_result(stage_result)
        runtimes = {
            "layout": self.layout_runtime,
            "formula": self.executor.formula_runtime,
            "text": self.executor.text_runtime,
        }
        if self.executor.table_runtime is not None:
            runtimes["table"] = self.executor.table_runtime
        stage_identities = {
            name: dict(getattr(runtime, "identity", {}) or {})
            for name, runtime in runtimes.items()
        }
        stage_metrics = {
            name: dict(getattr(runtime, "metrics", {}) or {})
            for name, runtime in runtimes.items()
        }
        payload = {
            "schema": self.policy.schema,
            "policy": self.policy.to_dict(),
            "plan": plan.to_dict(),
            "stage_result": stage_result,
            "stage_identities": stage_identities,
            "stage_metrics": stage_metrics,
            "source": str(kwargs.get("source") or ""),
            "document_id": str(kwargs.get("document_id") or ""),
        }
        return self.result_factory(payload=payload, plan=plan, stage_result=stage_result, **kwargs)


class VisualSerialExecutor:
    """Run formula OCR then text OCR with a hard no-overlap boundary."""

    def __init__(self, *, formula_runtime: Any, text_runtime: Any, table_runtime: Any = None, policy: VisualSerialPolicy | None = None) -> None:
        self.policy = policy or VisualSerialPolicy()
        self.formula_runtime = formula_runtime
        self.text_runtime = text_runtime
        self.table_runtime = table_runtime
        if bool(self.policy.table_model_id) != (table_runtime is not None):
            raise VisualSerialContractError("VISUAL_SERIAL_TABLE_RUNTIME_REQUIRED")
        self._stage_trace: list[dict[str, Any]] = []

    @staticmethod
    def _run(runtime: Any, regions: Sequence[VisualRegion]) -> list[Any]:
        if not regions:
            return []
        crops = []
        for region in regions:
            crop = dict(region.crop)
            # Keep answer-free serialization metadata beside the crop so a
            # concrete OCR adapter can select the matching text contract.
            # Geometry and source hashes remain owned by the crop itself.
            serialization_profile = region.metadata.get("serialization_profile")
            if serialization_profile is not None:
                crop["serialization_profile"] = serialization_profile
            crops.append(crop)
        method = getattr(runtime, "recognize_batch", None)
        values = list(method(crops) if callable(method) else [runtime.recognize(crop) for crop in crops])
        if len(values) != len(regions):
            raise VisualSerialStageError("VISUAL_SERIAL_STAGE_CARDINALITY_MISMATCH")
        return values

    @staticmethod
    def _close(runtime: Any) -> None:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()

    def execute(self, plan: VisualSerialPlan, *, output_dir: str | Path | None = None) -> dict[str, Any]:
        if plan.policy != self.policy:
            raise VisualSerialContractError("VISUAL_SERIAL_POLICY_MISMATCH")
        self._stage_trace = []
        started = time.perf_counter()
        formula_started = time.perf_counter()
        try:
            formula_values = self._run(self.formula_runtime, plan.formula_regions)
        finally:
            self._close(self.formula_runtime)
        formula_elapsed = time.perf_counter() - formula_started
        self._stage_trace.append({"stage": "FORMULA" if self.policy.policy_id == LOCAL_API_POLICY else "XIAOMI_FORMULA", "count": len(formula_values), "seconds": formula_elapsed})
        raw_formulas = {r.region_id: v for r, v in zip(plan.formula_regions, formula_values, strict=True)}
        normalized_formulas = {key: normalize_formula_output(value) for key, value in raw_formulas.items()}
        problems = {key: output_problem(value, formula=True) for key, value in normalized_formulas.items()}
        formulas = {key: value if not problems[key] else "" for key, value in normalized_formulas.items()}
        # Some mixed crops are transcribed without math delimiters. Keep the
        # literal text when math and prose cannot be separated reliably.
        formula_markdown_ids = [key for key, value in formulas.items() if formula_output_is_markdown(value)]
        plan = prepare_text_inputs(plan, formulas, output_dir)
        tables, raw_tables, table_reviews = {}, {}, []
        if self.table_runtime is not None:
            from .table_content import normalize_table
            table_started = time.perf_counter()
            try:
                table_values = self._run(self.table_runtime, plan.table_regions)
            finally:
                self._close(self.table_runtime)
            for region, value in zip(plan.table_regions, table_values, strict=True):
                raw_tables[region.region_id] = value
                tables[region.region_id], problem = normalize_table(value)
                table_reviews.append({"region_id":region.region_id,
                    "reason":problem or "TABLE_CONTENT_AND_STRUCTURE_REVIEW"})
            self._stage_trace.append({"stage":"TABLE", "count":len(table_values), "seconds":time.perf_counter()-table_started})
            # Preserve cell context for tables. Only successful table pixels
            # are removed from any larger prose box after table OCR completes.
            accepted_tables=tuple(r for r in plan.table_regions if tables.get(r.region_id))
            overlapping=tuple(r.region_id for r in plan.text_regions if any(
                r.page_index==t.page_index and _overlap_fraction(r.bbox,t.bbox)>0 for t in accepted_tables))
            if overlapping:
                if output_dir is None:raise VisualSerialContractError("VISUAL_SERIAL_RESIDUAL_CROP_ROOT_REQUIRED")
                masked=materialize_masked_text_plan(replace(plan,all_regions=plan.text_regions,
                    formula_regions=accepted_tables,formula_overlap_removed=overlapping,
                    text_regions=tuple(r for r in plan.text_regions if r.region_id not in overlapping)),
                    Path(output_dir)/"_bemarkdown/visual_serial/table_masked_crops",mask_kind="table")
                plan=replace(plan,text_regions=masked.text_regions)
        from .spatial_assembly import prepare_slots, assemble
        plan, assembly_groups, contained_formulas, assembly_reviews = prepare_slots(plan, formulas, output_dir)
        self.effective_plan = plan
        text_started = time.perf_counter()
        try:
            raw_text_values = self._run(self.text_runtime, plan.text_regions)
        finally:
            self._close(self.text_runtime)
        text_values: list[Any] = []
        text_formula_fragments_removed = 0
        mixed_serialization_wrappers_removed = 0
        decisions = []
        review_items = [{"region_id":key, "reason":reason, "raw_output":raw_formulas[key]}
                        for key, reason in problems.items() if reason]
        review_items.extend(table_reviews)
        review_items.extend(assembly_reviews)
        suppressed = []
        for region, value in zip(plan.text_regions, raw_text_values, strict=True):
            # Mixed-serialization targets explicitly require the upstream TeX
            # spans to remain in the text stream.  Applying the ordinary-text
            # cleanup here would contradict that contract and silently delete
            # the very tokens the prompt asked the recognizer to preserve.
            if region.metadata.get("assembly_parent"):
                # Slot OCR still sees undetected math; preserve its delimiters.
                cleaned, removed = str(value).strip(), 0
                if cleaned.startswith('```') and cleaned.endswith('```'):
                    cleaned = '\n'.join(cleaned.splitlines()[1:-1]).strip()
            elif region.metadata.get("serialization_profile") == "UPSTREAM_TEX_MIXED_V1":
                cleaned, removed = strip_mixed_serialization_wrappers(value)
                mixed_serialization_wrappers_removed += removed
                removed = 0
            else:
                cleaned, notes, removed = clean_text_output(value, region, plan.formula_regions, formulas)
                decisions.extend(notes)
                review_items.extend(note for note in notes if note["reason"] != "DUPLICATE_FORMULA_SOURCE_CONFIRMED")
                if removed and not str(cleaned).strip():
                    suppressed.append(region.region_id)
            reason = output_problem(cleaned)
            if reason and region.region_id not in suppressed:
                review_items.append({"region_id":region.region_id, "reason":reason, "raw_output":value})
                if reason in {"EMPTY_MARKUP_OUTPUT", "EMPTY_MATH_BODY", "UNPAIRED_MATH_DELIMITER"}:
                    cleaned = ""
            if region.metadata.get("review_reason"):
                review_items.append({"region_id":region.region_id, "reason":region.metadata["review_reason"]})
            text_values.append(cleaned)
            text_formula_fragments_removed += removed
        text_elapsed = time.perf_counter() - text_started
        duplicate_notes = duplicate_text_regions(plan.text_regions, dict(zip(
            (r.region_id for r in plan.text_regions), text_values, strict=True)))
        decisions.extend(duplicate_notes)
        suppressed.extend(note["region_id"] for note in duplicate_notes)
        self._stage_trace.append({"stage": "TEXT" if self.policy.policy_id == LOCAL_API_POLICY else "OVISOCR3_TEXT", "count": len(text_values), "seconds": text_elapsed})
        text_map = dict(zip((r.region_id for r in plan.text_regions), text_values))
        from .spatial_assembly import duplicate_slot_text
        slot_duplicates = duplicate_slot_text(plan.text_regions, text_map) if assembly_groups else []
        decisions.extend(slot_duplicates)
        suppressed.extend(note['region_id'] for note in slot_duplicates)
        assembly_groups = [{**g, 'rows':[[slot for slot in row if slot[1] not in suppressed] for row in g['rows']]} for g in assembly_groups]
        assembled, consumed, assembly_provenance = assemble(assembly_groups, text_map, formulas, formula_markdown_ids)
        for group in assembly_groups:
            if any(not str(text_map.get(rid, '')).strip()
                   for row in group['rows'] for kind,rid in row if kind == 'text'):
                review_items.append({'region_id':group['parent'].region_id,'reason':'ASSEMBLY_TEXT_FRAGMENT_EMPTY_SOURCE_PRESERVED'})
        return {
            "spatial_assembly": assembly_provenance,
            "assembled_text": assembled,
            "assembly_consumed_ids": consumed,
            "contained_formula_aliases": contained_formulas,
            "schema": self.policy.schema,
            "policy": self.policy.to_dict(),
            "formula": formulas,
            "formula_markdown_ids": formula_markdown_ids,
            "table": tables,
            "raw_table": raw_tables,
            "table_input_provenance": {r.region_id:r.to_dict() for r in plan.table_regions},
            "table_owned_regions": dict(plan.table_owned_regions),
            "text": {region.region_id: value for region, value in zip(plan.text_regions, text_values, strict=True)},
            "raw_formula": raw_formulas,
            "raw_text": {r.region_id:v for r,v in zip(plan.text_regions,raw_text_values,strict=True)},
            "review_items": review_items,
            "content_decisions": decisions,
            "suppressed_text_region_ids": suppressed,
            "text_input_provenance": {r.region_id:{"bbox":list(r.bbox), "crop":dict(r.crop), "metadata":dict(r.metadata)} for r in plan.text_regions},
            "preserved_region_ids": [region.region_id for region in plan.preserved_regions],
            "stage_trace": list(self._stage_trace),
            "serial": True,
            "formula_text_input_disjoint": not ({r.region_id for r in plan.formula_regions} & {r.region_id for r in plan.text_regions}),
            "formula_text_input_disjoint_scope": "REGION_IDS; failed formulas may enter text fallback",
            "text_formula_fragments_removed": text_formula_fragments_removed,
            "mixed_serialization_wrappers_removed": mixed_serialization_wrappers_removed,
            "image_screening_called": False,
            "final_review_called": False,
            "elapsed_seconds": time.perf_counter() - started,
        }


def assert_visual_serial_result(result: Mapping[str, Any]) -> None:
    policy_id = result.get("policy", {}).get("policy_id")
    if result.get("schema") != VISUAL_SERIAL_SCHEMA or policy_id not in {VISUAL_SERIAL_POLICY, LOCAL_API_POLICY}:
        raise VisualSerialContractError("VISUAL_SERIAL_RESULT_SCHEMA_INVALID")
    if result.get("serial") is not True or result.get("formula_text_input_disjoint") is not True:
        raise VisualSerialContractError("VISUAL_SERIAL_RESULT_NOT_SERIAL_OR_DISJOINT")
    stages = [row.get("stage") for row in result.get("stage_trace", [])]
    expected = ["FORMULA", "TABLE", "TEXT"] if result.get("policy", {}).get("table_model_id") else (["FORMULA", "TEXT"] if policy_id == LOCAL_API_POLICY else ["XIAOMI_FORMULA", "OVISOCR3_TEXT"])
    if stages != expected:
        raise VisualSerialContractError("VISUAL_SERIAL_STAGE_ORDER_INVALID")
    if result.get("image_screening_called") or result.get("final_review_called"):
        raise VisualSerialContractError("VISUAL_SERIAL_FORBIDDEN_STAGE_RECORDED")

