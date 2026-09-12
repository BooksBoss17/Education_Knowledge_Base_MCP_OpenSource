"""Research-only contracts and metrics for the Phase 6B-2C3 OvisOCR2 benchmark.

This module deliberately has no Torch or Transformers dependency.  Production
BeMarkdown can test the evidence rules without loading or downloading a model;
the isolated research runner imports the model runtime only inside its worker.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_ID = "ATH-MaaS/OvisOCR2"
MODEL_REVISION = "65c619d374b55d4152e85150fc1b003700bc1f0c"

# Exact official model-card prompt at the pinned revision.  The leading newline
# is part of the official string and therefore part of the frozen hash.
P0_OFFICIAL = (
    "\nExtract all readable content from the image in natural human reading order and "
    "output the result as a single Markdown document. For charts or images, represent "
    "them using an HTML image tag: <img "
    'src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, '
    "bottom are bounding box coordinates scaled to [0, 1000). Format formulas as "
    "LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as "
    "standard Markdown. Preserve the original text without translation or paraphrasing."
)
P1_FORMULA = (
    "Extract the mathematical formula from this image.\n"
    "Output only the formula as LaTeX.\n"
    "Do not add explanations or Markdown code fences."
)

ACCEPTABLE_LABELS = frozenset({"CORRECT", "MINOR_ERROR"})
MATERIAL_LABELS = frozenset({"MAJOR_ERROR", "UNUSABLE"})
CERTAIN_LABELS = ACCEPTABLE_LABELS | MATERIAL_LABELS
PROMPT_IDS = ("P0", "P1")


@dataclass(frozen=True)
class SecondaryFormulaCandidate:
    """A conservative parse result which never repairs formula semantics."""

    status: str
    latex: str | None
    reason: str | None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_contract() -> dict:
    return {
        "schema": "bemarkdown-ovisocr2-prompt-contract-v0",
        "frozen_before_first_inference": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompts": {
            "P0": {
                "kind": "OFFICIAL_STYLE_BASELINE",
                "text": P0_OFFICIAL,
                "utf8_sha256": sha256_text(P0_OFFICIAL),
                "source": "README.md at pinned model revision",
            },
            "P1": {
                "kind": "FORMULA_FOCUSED_MINIMAL",
                "text": P1_FORMULA,
                "utf8_sha256": sha256_text(P1_FORMULA),
                "source": "Phase 6B-2C3 frozen task contract",
            },
        },
        "selection_policy": "REPORT_SEPARATELY_NO_PER_SAMPLE_ORACLE",
        "prompt_engineering_after_freeze": False,
    }


def generation_contract() -> dict:
    return {
        "schema": "bemarkdown-ovisocr2-generation-contract-v0",
        "do_sample": False,
        "decoding": "GREEDY_DETERMINISTIC",
        "max_new_tokens": 512,
        "max_new_tokens_rationale": (
            "Fixed deterministic formula-crop safety bound after a pre-benchmark P1 smoke "
            "failed to emit EOS for five minutes under the model-card page limit of 16384."
        ),
        "temperature": None,
        "batch_size": 1,
        "dtype": "bfloat16",
        "enable_thinking": False,
        "local_files_only": True,
        "processor_policy": {
            "min_pixels": 448 * 448,
            "max_pixels": 2880 * 2880,
            "official_baseline": True,
        },
    }


_FENCE = re.compile(r"\A```(?:latex|tex|math)?[ \t]*\r?\n(.*?)\r?\n```\Z", re.DOTALL)
_DELIMITERS = (
    ("$$", "$$"),
    (r"\[", r"\]"),
    (r"\(", r"\)"),
    ("$", "$"),
)


def _strip_one_explicit_delimiter(value: str) -> str | None:
    for opening, closing in _DELIMITERS:
        if value.startswith(opening) and value.endswith(closing):
            payload = value[len(opening) : len(value) - len(closing)].strip()
            if payload:
                return payload
    return None


def parse_secondary_formula(raw_output: str) -> SecondaryFormulaCandidate:
    """Extract one unambiguous payload using delimiter removal only.

    The function intentionally does not balance braces, replace symbols, remove
    tokens, or otherwise turn an invalid recognition into a plausible formula.
    """

    value = raw_output.strip()
    if not value:
        return SecondaryFormulaCandidate("SECONDARY_OUTPUT_EMPTY", None, "EMPTY_OUTPUT")

    fence = _FENCE.fullmatch(value)
    if fence:
        value = fence.group(1).strip()
        if not value:
            return SecondaryFormulaCandidate(
                "SECONDARY_OUTPUT_AMBIGUOUS", None, "EMPTY_MARKDOWN_FENCE"
            )
    elif "```" in value:
        return SecondaryFormulaCandidate(
            "SECONDARY_OUTPUT_AMBIGUOUS", None, "MIXED_OR_MULTIPLE_MARKDOWN_FENCES"
        )

    payload = _strip_one_explicit_delimiter(value)
    if payload is not None:
        # Delimiters inside the unique payload indicate multiple formula blocks,
        # not a single candidate that can safely support agreement.
        if any(token in payload for pair in _DELIMITERS for token in pair):
            return SecondaryFormulaCandidate(
                "SECONDARY_OUTPUT_AMBIGUOUS", None, "MULTIPLE_FORMULA_DELIMITERS"
            )
        return SecondaryFormulaCandidate("SECONDARY_OUTPUT_PARSED", payload, None)

    has_markdown_structure = bool(
        re.search(r"(?m)^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|<\/?(?:table|img)\b)", value)
    )
    mixed_content_tokens = ("$$", "$", r"\[", r"\]", "<think>", "</think>", "\n code")
    if has_markdown_structure or any(token in value for token in mixed_content_tokens):
        return SecondaryFormulaCandidate(
            "SECONDARY_OUTPUT_AMBIGUOUS", None, "FORMULA_MIXED_WITH_DOCUMENT_CONTENT"
        )

    # Plain output is allowed by P1.  Preserve every non-outer character.
    return SecondaryFormulaCandidate("SECONDARY_OUTPUT_PARSED", value, None)


class ImmutableRawOutputStore:
    """Write raw output exactly once, before any parsing or normalization."""

    def __init__(self, root: Path):
        self.root = root

    def write_once(self, *, prompt_id: str, crop_sha256: str, record: Mapping) -> Path:
        if prompt_id not in PROMPT_IDS:
            raise ValueError(f"Unknown prompt ID: {prompt_id}")
        raw_output = record.get("raw_output")
        if not isinstance(raw_output, str):
            raise TypeError("raw_output must be an immutable string field")
        target = self.root / prompt_id.lower() / f"{crop_sha256}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        payload["raw_output_sha256"] = sha256_text(raw_output)
        payload["raw_output_immutable"] = True
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
        return target


def derive_current_consensus_pass_subset(
    truth_rows: Iterable[Mapping], consensus_rows: Iterable[Mapping]
) -> list[dict]:
    """Derive the deployment-relevant subset from frozen C3 evidence only."""

    consensus_by_route = {row["route_id"]: row for row in consensus_rows}
    subset = []
    for truth in truth_rows:
        score = consensus_by_route[truth["route_id"]]
        c3 = score["strategies"]["C3_E0_E1_E2_E3"]
        if c3["consensus_status"] == "CONSENSUS_PASS":
            subset.append(
                {
                    "reference_id": truth["reference_id"],
                    "route_id": truth["route_id"],
                    "content_id": truth["content_id"],
                    "formulanet_label": truth["human_quality_label"],
                    "c3_consensus_status": c3["consensus_status"],
                    "c3_reason_codes": list(c3.get("reason_codes", [])),
                }
            )
    return sorted(subset, key=lambda row: row["reference_id"])


def formula_model_cross_table(review_rows: Iterable[Mapping], prompt_id: str) -> dict:
    """Build the FormulaNet x Ovis table without uncertain rows in the denominator."""

    if prompt_id not in PROMPT_IDS:
        raise ValueError(f"Unknown prompt ID: {prompt_id}")
    counts: Counter[str] = Counter()
    certain = 0
    for row in review_rows:
        formula_label = row["formulanet_label"]
        ovis_label = row[f"ovis_{prompt_id.lower()}_label"]
        if formula_label not in CERTAIN_LABELS:
            continue
        certain += 1
        formula_group = (
            "ACCEPTABLE" if formula_label in ACCEPTABLE_LABELS else "MATERIAL"
        )
        if ovis_label in ACCEPTABLE_LABELS:
            ovis_group = "ACCEPTABLE"
        elif ovis_label in MATERIAL_LABELS:
            ovis_group = "MATERIAL"
        else:
            ovis_group = "AMBIGUOUS_OR_UNAVAILABLE"
        counts[f"FORMULANET_{formula_group}__OVIS_{ovis_group}"] += 1
    return {
        "prompt_id": prompt_id,
        "certain_formula_truth_rows": certain,
        "counts": dict(sorted(counts.items())),
    }


def label_counts(review_rows: Iterable[Mapping], prompt_id: str) -> dict:
    labels = Counter(row[f"ovis_{prompt_id.lower()}_label"] for row in review_rows)
    certain = sum(labels[label] for label in CERTAIN_LABELS)
    acceptable = sum(labels[label] for label in ACCEPTABLE_LABELS)
    material = sum(labels[label] for label in MATERIAL_LABELS)
    return {
        "prompt_id": prompt_id,
        "counts": dict(sorted(labels.items())),
        "certain_reviewed": certain,
        "acceptable": acceptable,
        "material": material,
        "acceptable_rate": acceptable / certain if certain else None,
        "material_error_rate": material / certain if certain else None,
    }


def assert_no_sample_specific_rules(source: str) -> None:
    forbidden = (
        r"formula-reference-\d{4}",
        r"content-route-[0-9a-f]{20}",
        r"content-[0-9a-f]{20}",
    )
    hits = [pattern for pattern in forbidden if re.search(pattern, source)]
    if hits:
        raise AssertionError(f"Sample-specific identifiers are forbidden: {hits}")


def dataclass_dict(value: SecondaryFormulaCandidate) -> dict:
    return asdict(value)
