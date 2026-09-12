"""Stable PDF pipeline-mode contract shared by production and experiments."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import load_workspace_config

PDF_PIPELINE_MODE_ENV = "BEMARKDOWN_PDF_PIPELINE_MODE"


class PDFPipelineMode(str, Enum):
    MODULAR_PRIMARY = "modular_primary"
    PADDLE_VL_EXPERIMENTAL = "paddle_vl_experimental"
    DUAL_PADDLE_EVIDENCE_EXPERIMENTAL = "dual_paddle_evidence_experimental"
    NATIVE_FIRST_ADAPTIVE_EXPERIMENTAL = "native_first_adaptive_experimental"

    # Import-compatible attribute aliases for frozen Phase 7 code.
    LEGACY_MODULAR = "modular_primary"  # noqa: PIE796 - compatibility alias
    DUAL_PADDLE_EVIDENCE = (  # noqa: PIE796 - compatibility alias
        "dual_paddle_evidence_experimental"
    )
    NATIVE_FIRST_ADAPTIVE = (  # noqa: PIE796 - compatibility alias
        "native_first_adaptive_experimental"
    )


_LEGACY_VALUE_ALIASES = {
    "legacy_modular": PDFPipelineMode.MODULAR_PRIMARY.value,
    "dual_paddle_evidence": PDFPipelineMode.DUAL_PADDLE_EVIDENCE_EXPERIMENTAL.value,
    "native_first_adaptive": PDFPipelineMode.NATIVE_FIRST_ADAPTIVE_EXPERIMENTAL.value,
}


def resolve_pdf_pipeline_mode(
    value: str | PDFPipelineMode | None,
    *,
    config_path: str | Path | None = None,
) -> PDFPipelineMode:
    """Resolve explicit > environment > workspace config > production default."""

    selected: str | PDFPipelineMode | None = value
    if selected is None:
        selected = os.environ.get(PDF_PIPELINE_MODE_ENV)
    if selected is None:
        _, config = load_workspace_config(config_path)
        selected = _configured_default(config)
    if selected is None:
        selected = PDFPipelineMode.MODULAR_PRIMARY.value
    if isinstance(selected, PDFPipelineMode):
        return selected
    selected = _LEGACY_VALUE_ALIASES.get(str(selected), str(selected))
    try:
        return PDFPipelineMode(selected)
    except ValueError as exc:
        raise ValueError(f"UNSUPPORTED_PDF_PIPELINE_MODE:{selected}") from exc


def pipeline_mode_inventory() -> dict[str, Any]:
    """Return the auditable authority/status inventory without loading any model."""

    return {
        "schema": "bemarkdown-pdf-pipeline-mode-inventory-v1",
        "default": PDFPipelineMode.MODULAR_PRIMARY.value,
        "modes": [
            {
                "mode": PDFPipelineMode.MODULAR_PRIMARY.value,
                "authority": "PRODUCTION_CANDIDATE",
                "default": True,
                "paddleocr_vl": "NOT_LOADED",
                "pp_doclayout_v3": "NOT_LOADED",
            },
            {
                "mode": PDFPipelineMode.PADDLE_VL_EXPERIMENTAL.value,
                "authority": "EXPERIMENTAL_OPTIONAL_PARSER",
                "default": False,
            },
            {
                "mode": PDFPipelineMode.DUAL_PADDLE_EVIDENCE_EXPERIMENTAL.value,
                "authority": "DIAGNOSTIC_ONLY",
                "default": False,
            },
            {
                "mode": PDFPipelineMode.NATIVE_FIRST_ADAPTIVE_EXPERIMENTAL.value,
                "authority": "DIAGNOSTIC_ONLY",
                "default": False,
            },
        ],
        "legacy_aliases": dict(sorted(_LEGACY_VALUE_ALIASES.items())),
    }


def modular_primary_residency_plan() -> dict[str, Any]:
    """Describe specialist residency while proving Phase 7 models are absent."""

    return {
        "schema": "bemarkdown-modular-primary-residency-plan-v1",
        "mode": PDFPipelineMode.MODULAR_PRIMARY.value,
        "stages": [
            {
                "stage": "SPECIALIST_LAYOUT",
                "resident_models": ["PP-DocLayout_plus-L"],
                "activation": "ALWAYS_ON",
            },
            {
                "stage": "SPECIALIST_TEXT",
                "resident_models": ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"],
                "activation": "SCANNED_OR_UNTRUSTED_TEXT_ONLY",
            },
            {
                "stage": "SPECIALIST_FORMULA",
                "resident_models": ["PP-FormulaNet_plus-L"],
                "activation": "FORMULA_REGION_ALWAYS",
            },
            {
                "stage": "SPECIALIST_TABLE",
                "resident_models": ["Table recognition v2", "TableEngine"],
                "activation": "TABLE_REGION_ALWAYS",
            },
        ],
        "paddleocr_vl_load_count": 0,
        "pp_doclayout_v3_load_count": 0,
        "vl_runtime_required": False,
        "vision_agent_enabled": False,
        "all_models_simultaneously_resident": False,
    }


def _configured_default(config: dict[str, Any]) -> str | None:
    pipeline = config.get("pdf_pipeline")
    if pipeline is None:
        return None
    if not isinstance(pipeline, dict):
        raise TypeError("pdf_pipeline configuration must be a table")
    selected = pipeline.get("default")
    if selected is None:
        return None
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError("pdf_pipeline.default must be a non-empty string")
    return selected
