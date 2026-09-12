"""Deterministic content-type authority and heading resolution policies."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, ClassVar

TYPED_AUTHORITY_SCHEMA = "bemarkdown-typed-authority-policy-v1"


class ContentType(str, Enum):
    TEXT = "TEXT"
    FORMULA = "FORMULA"
    TABLE = "TABLE"
    HEADING = "HEADING"
    IMAGE = "IMAGE"
    READING_ORDER = "READING_ORDER"


@dataclass(frozen=True, slots=True)
class AuthorityDecision:
    schema: str
    content_type: ContentType
    primary_source: str
    accepted_content: Any
    supporting_content: Any
    authority_state: str
    risk_reasons: tuple[str, ...]
    automatic_overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["content_type"] = self.content_type.value
        result["risk_reasons"] = list(self.risk_reasons)
        return result


@dataclass(frozen=True, slots=True)
class HeadingDecision:
    state: str
    score: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def _canonical(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sensitive_tokens(value: Any) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value or ""))
    pattern = re.compile(
        r"(?:\d+(?:\.\d+)?)|(?:m/s(?:\^?[23])?)|(?:[A-Za-z]+(?:/[A-Za-z]+)?)|"
        r"(?:[=<>≤≥±+\-×÷^_²³₀-₉])|(?:[α-ωΑ-Ω])"
    )
    return tuple(pattern.findall(text))


class HeadingResolver:
    """Resolve heading state from independent layout and native font signals."""

    HEADING_LABELS: ClassVar[set[str]] = {
        "doc_title",
        "paragraph_title",
        "section_title",
        "figure_title",
        "title",
        "heading",
    }

    def resolve(
        self,
        *,
        layout_v3_label: str | None,
        plus_l_label: str | None,
        native_font_size: float | None,
        body_font_size: float | None,
        native_font_flags: int | None,
        vl_label: str | None,
        geometry_relation: str | None = None,
    ) -> HeadingDecision:
        reasons: list[str] = []
        votes = 0
        for name, label in (
            ("LAYOUT_V3", layout_v3_label),
            ("PLUS_L", plus_l_label),
        ):
            if str(label or "").lower() in self.HEADING_LABELS:
                votes += 1
                reasons.append(f"{name}_HEADING_LABEL")
        if (
            native_font_size is not None
            and body_font_size is not None
            and native_font_size >= body_font_size * 1.25
        ):
            votes += 1
            reasons.append("NATIVE_FONT_SIZE_EMPHASIS")
        if native_font_flags is not None and native_font_flags & 16:
            votes += 1
            reasons.append("NATIVE_FONT_BOLD")
        if geometry_relation in {"SECTION_START", "TITLE_ABOVE_CONTENT"}:
            votes += 1
            reasons.append("HEADING_GEOMETRY_RELATION")
        vl_heading = str(vl_label or "").lower() in self.HEADING_LABELS
        if vl_heading:
            reasons.append("VL_HEADING_LABEL_NON_AUTHORITATIVE")
        if votes >= 3:
            state = "HEADING_CONFIRMED"
        elif votes >= 2:
            state = "HEADING_PLAUSIBLE"
        elif vl_heading != (votes > 0):
            state = "HEADING_CONFLICT"
        else:
            state = "NOT_HEADING"
        return HeadingDecision(state=state, score=votes, reasons=tuple(reasons))


def calibrate_formula_authority(pairs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    feature_counts: dict[str, Counter[str]] = {}
    rows = list(pairs)
    for row in rows:
        if row.get("unverifiable"):
            bucket = "unverifiable"
        elif row.get("vl_correct") and row.get("formulanet_correct"):
            bucket = "both_correct"
        elif row.get("vl_correct"):
            bucket = "vl_only_correct"
        elif row.get("formulanet_correct"):
            bucket = "formulanet_only_correct"
        else:
            bucket = "both_wrong"
        counts[bucket] += 1
        for feature in (
            "renderable",
            "token_exact",
            "subscript",
            "superscript",
            "glyph_sensitive",
            "chinese_subscript",
            "long_formula",
            "multiline_formula",
        ):
            if row.get(feature):
                feature_counts.setdefault(feature, Counter())[bucket] += 1
    ordered = {
        name: counts[name]
        for name in (
            "both_correct",
            "vl_only_correct",
            "formulanet_only_correct",
            "both_wrong",
            "unverifiable",
        )
    }
    if counts["formulanet_only_correct"] > counts["vl_only_correct"]:
        selected = "FORMULANET_PRIMARY"
    elif counts["vl_only_correct"] > counts["formulanet_only_correct"]:
        selected = "VL_PRIMARY"
    else:
        selected = "CO_EVIDENCE"
    return {
        "schema": "bemarkdown-formula-authority-calibration-v1",
        "pair_count": len(rows),
        "counts": ordered,
        "features": {
            name: dict(sorted(values.items()))
            for name, values in sorted(feature_counts.items())
        },
        "selected_authority": selected,
        "selection_basis": "FROZEN_PAIRWISE_TRUTH_COUNTS",
    }


def pair_formula_occurrences_by_source_order(
    *,
    truth_occurrences: Iterable[Mapping[str, Any]],
    evidence_occurrences: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair only source-ordered display formulas with exact page cardinality.

    Candidate text is deliberately excluded from the pairing decision.  A page is
    eligible only when every formula evidence occurrence is an exact
    ``display_formula`` crop and its cardinality equals the frozen display-truth
    cardinality.  Inline truth and ambiguous pages remain explicitly unverifiable.
    """

    truth_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in truth_occurrences:
        truth_by_page[str(row["page_id"])].append(dict(row))
    for row in evidence_occurrences:
        evidence_by_page[str(row["page_id"])].append(dict(row))

    result: list[dict[str, Any]] = []
    for page_id in sorted(truth_by_page):
        truths = sorted(
            truth_by_page[page_id],
            key=lambda row: (int(row["truth_order"]), str(row["truth_item_id"])),
        )
        display_truths = [
            row
            for row in truths
            if str(row.get("display_or_inline", "")).upper() == "DISPLAY"
        ]
        candidates = sorted(
            evidence_by_page.get(page_id, []),
            key=lambda row: (int(row["source_order"]), str(row["node_id"])),
        )
        source_orders = [int(row["source_order"]) for row in candidates]
        exact_page_pairing = bool(display_truths) and (
            len(candidates) == len(display_truths)
            and len(source_orders) == len(set(source_orders))
            and all(
                str(row.get("label", "")).lower() == "display_formula"
                and row.get("source_crop_is_exact") is True
                and bool(row.get("source_crop_ref"))
                and bool(row.get("source_crop_sha256"))
                for row in candidates
            )
        )
        candidate_by_truth_order = (
            {
                int(truth["truth_order"]): candidate
                for truth, candidate in zip(display_truths, candidates, strict=True)
            }
            if exact_page_pairing
            else {}
        )
        for truth in truths:
            base = {
                **truth,
                "page_id": page_id,
            }
            if str(truth.get("display_or_inline", "")).upper() != "DISPLAY":
                result.append(
                    {
                        **base,
                        "unverifiable": True,
                        "pairing_reason": "INLINE_TRUTH_HAS_NO_EXACT_FORMULA_CROP",
                    }
                )
                continue
            candidate = candidate_by_truth_order.get(int(truth["truth_order"]))
            if candidate is None:
                result.append(
                    {
                        **base,
                        "unverifiable": True,
                        "pairing_reason": (
                            "DISPLAY_OCCURRENCE_CARDINALITY_OR_LABEL_MISMATCH"
                        ),
                    }
                )
                continue
            result.append(
                {
                    **candidate,
                    **base,
                    "unverifiable": False,
                    "pairing_reason": (
                        "SOURCE_ORDER_EXACT_DISPLAY_CARDINALITY_SAME_ALIGNED_CROP"
                    ),
                }
            )
    return result


class TypedAuthorityPolicy:
    """Select authority by content type; never by majority vote."""

    def __init__(self, *, formula_authority: str = "CO_EVIDENCE") -> None:
        self.formula_authority = formula_authority

    def resolve(
        self,
        *,
        content_type: ContentType,
        native: Any,
        vl: Any,
        specialist: Any,
        native_reliable: bool = False,
    ) -> AuthorityDecision:
        reasons: list[str] = []
        support = specialist if specialist is not None else vl
        if content_type is ContentType.IMAGE:
            primary, accepted, state = "SOURCE_ASSET", native, "SOURCE_ASSET_AUTHORITY"
        elif (
            content_type is ContentType.TEXT and native_reliable and native is not None
        ):
            primary, accepted, state = "NATIVE_TEXT", native, "SOURCE_NATIVE_ACCEPTED"
            if vl is not None and _sensitive_tokens(native) != _sensitive_tokens(vl):
                reasons.append("SENSITIVE_CONTENT_CONFLICT")
        elif content_type is ContentType.TEXT:
            primary, accepted, state = "PADDLEOCR_VL", vl, "VISUAL_PRIMARY"
        elif content_type is ContentType.FORMULA:
            if self.formula_authority == "FORMULANET_PRIMARY":
                primary, accepted = "PP_FORMULANET_PLUS_L", specialist
            elif self.formula_authority == "VL_PRIMARY":
                primary, accepted = "PADDLEOCR_VL", vl
            else:
                primary, accepted = "CO_EVIDENCE", vl
            state = "FORMULA_CALIBRATED_POLICY"
            if (
                vl is not None
                and specialist is not None
                and _canonical(vl) != _canonical(specialist)
            ):
                reasons.append("FORMULA_EVIDENCE_CONFLICT")
        elif content_type is ContentType.TABLE:
            primary, accepted, state = (
                "PADDLEOCR_VL",
                vl,
                "TABLE_PRIMARY_WITH_STRUCTURAL_SPECIALIST",
            )
        elif content_type is ContentType.HEADING:
            primary, accepted, state = (
                "HEADING_RESOLVER",
                native or vl,
                "MULTI_SIGNAL_HEADING",
            )
        else:
            primary, accepted, state = (
                "READING_ORDER_RESOLVER_V2",
                native or vl,
                "PRECEDENCE_GRAPH",
            )
        if accepted is None:
            reasons.append("PRIMARY_EVIDENCE_UNAVAILABLE")
        return AuthorityDecision(
            schema=TYPED_AUTHORITY_SCHEMA,
            content_type=content_type,
            primary_source=primary,
            accepted_content=accepted,
            supporting_content=support,
            authority_state=state,
            risk_reasons=tuple(reasons),
        )


def typed_authority_policy_document(formula_authority: str) -> dict[str, Any]:
    return {
        "schema": TYPED_AUTHORITY_SCHEMA,
        "automatic_majority_vote": False,
        "policies": {
            "TEXT": "NATIVE_RELIABLE_ELSE_VL_WITH_PP_OCR_SPECIALIST",
            "FORMULA": formula_authority,
            "TABLE": "VL_PRIMARY_TABLEENGINE_STRUCTURAL_SPECIALIST",
            "HEADING": "DETERMINISTIC_MULTI_SIGNAL_HEADING_RESOLVER",
            "IMAGE": "SOURCE_ASSET_AUTHORITY",
            "READING_ORDER": "DETERMINISTIC_PRECEDENCE_GRAPH_V2",
        },
    }
