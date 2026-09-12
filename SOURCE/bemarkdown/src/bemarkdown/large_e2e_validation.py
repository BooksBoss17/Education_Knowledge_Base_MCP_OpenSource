"""Validation-only contracts for large BeMarkdown end-to-end attempts.

The helpers in this module do not execute or alter production routing.  They
provide fail-closed cohort, truth-isolation, Agent-scope, accounting, handoff,
and metric checks for validation harnesses.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

from .agent_task_contract import AgentTaskV2

__all__ = ["AgentTaskV2"]

AGENT_STATUSES = {
    "AGENT_REQUIRED_ALL_DIFFER",
    "AGENT_REQUIRED_RUNTIME_FAILURE",
}
AUTO_CONSENSUS_STATUSES = {"CONSENSUS_AB", "CONSENSUS_AC", "CONSENSUS_BC"}
AGENT_RESULT_STATUSES = {"TRANSCRIBED", "UNRESOLVED", "SOURCE_UNREADABLE"}


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cohort_lock(
    pages: Sequence[Mapping[str, Any]], *, cohort_id: str
) -> dict[str, Any]:
    rows = [dict(row) for row in pages]
    identities = [str(row["page_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("DUPLICATE_COHORT_PAGE_ID")
    ordered = sorted(rows, key=lambda row: str(row["page_id"]))
    documents = sorted({str(row["document_id"]) for row in ordered})
    payload = {
        "schema": "bemarkdown-large-e2e-cohort-lock-v1",
        "cohort_id": cohort_id,
        "document_count": len(documents),
        "page_count": len(ordered),
        "documents": documents,
        "pages": ordered,
    }
    payload["lock_sha256"] = semantic_sha256(payload)
    return payload


@dataclass(slots=True)
class TruthIsolationGuard:
    """Keep reference truth unavailable until outputs are SHA-frozen."""

    truth_reads: int = 0
    output_lock: dict[str, str] = field(default_factory=dict)
    unlocked: bool = False

    def freeze_outputs(self, outputs: Mapping[str, str]) -> dict[str, Any]:
        if not outputs or any(len(value) != 64 for value in outputs.values()):
            raise ValueError("PRE_TRUTH_OUTPUT_LOCK_INVALID")
        self.output_lock = dict(sorted(outputs.items()))
        return {
            "schema": "bemarkdown-pre-truth-final-output-lock-v1",
            "outputs": self.output_lock,
            "lock_sha256": semantic_sha256(self.output_lock),
        }

    def unlock(self) -> None:
        if not self.output_lock:
            raise RuntimeError("TRUTH_UNLOCK_REQUIRES_FINAL_OUTPUT_LOCK")
        self.unlocked = True

    def record_truth_read(self) -> None:
        if not self.unlocked:
            raise RuntimeError("REFERENCE_TRUTH_ACCESS_FORBIDDEN")
        self.truth_reads += 1


@dataclass(frozen=True, slots=True)
class AgentResultV2:
    task_id: str
    status: Literal["TRANSCRIBED", "UNRESOLVED", "SOURCE_UNREADABLE"]
    transcribed_text: str
    normalized_text: str
    source_crop_sha256: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if self.status not in AGENT_RESULT_STATUSES:
            raise ValueError("AGENT_RESULT_STATUS_INVALID")
        if self.status == "TRANSCRIBED" and not self.transcribed_text:
            raise ValueError("AGENT_TRANSCRIPTION_EMPTY")
        if len(self.source_crop_sha256) != 64:
            raise ValueError("AGENT_RESULT_CROP_SHA_INVALID")
        value = asdict(self)
        value["schema"] = "bemarkdown-agent-result-v2"
        value["warnings"] = list(self.warnings)
        return value


def build_agent_inventory(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    eligible = [row for row in records if _resolution_status(row) in AGENT_STATUSES]
    forbidden = [
        row for row in records if _resolution_status(row) in AUTO_CONSENSUS_STATUSES
    ]
    return {
        "schema": "bemarkdown-agent-task-inventory-v2",
        "candidate_count": len(eligible),
        "auto_consensus_count": len(forbidden),
        "candidate_region_ids": sorted(
            str(row.get("region_id") or row.get("route_id")) for row in eligible
        ),
    }


def apply_agent_result(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_region_id: str,
    expected_crop_sha256: str,
    result: AgentResultV2,
) -> list[dict[str, Any]]:
    if result.status != "TRANSCRIBED":
        return [copy.deepcopy(dict(row)) for row in rows]
    if result.source_crop_sha256 != expected_crop_sha256:
        raise ValueError("AGENT_SOURCE_IDENTITY_MISMATCH")
    output = [copy.deepcopy(dict(row)) for row in rows]
    flat_matches = [
        row
        for row in output
        if str(row.get("region_id") or row.get("route_id")) == target_region_id
    ]
    bounded_matches: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for row in output:
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        evidence = provenance.get("three_model_text_evidence")
        if not isinstance(evidence, Mapping):
            continue
        units = evidence.get("bounded_units")
        if not isinstance(units, list):
            continue
        for unit in units:
            if not isinstance(unit, Mapping):
                continue
            request = unit.get("request")
            if (
                isinstance(request, Mapping)
                and str(request.get("region_id") or "") == target_region_id
            ):
                bounded_matches.append((row, unit))
    if len(flat_matches) + len(bounded_matches) != 1:
        raise ValueError("AGENT_TARGET_CARDINALITY_INVALID")
    if flat_matches:
        target = flat_matches[0]
        if _resolution_status(target) not in AGENT_STATUSES:
            raise ValueError("AUTO_CONSENSUS_MUTATION_FORBIDDEN")
        target["text"] = result.transcribed_text
        target["selected_text"] = result.transcribed_text
        provenance = target.setdefault("provenance", {})
        provenance["agent_result"] = result.to_dict()
        target["agent_result"] = result.to_dict()
        return output

    target, unit = bounded_matches[0]
    request = unit["request"]
    resolver = unit.get("resolver")
    if (
        not isinstance(resolver, Mapping)
        or str(resolver.get("resolution_status") or "") not in AGENT_STATUSES
    ):
        raise ValueError("AUTO_CONSENSUS_MUTATION_FORBIDDEN")
    if str(request.get("crop_sha256") or "") != expected_crop_sha256:
        raise ValueError("AGENT_SOURCE_IDENTITY_MISMATCH")
    provenance = target.setdefault("provenance", {})
    agent_results = provenance.setdefault("agent_unit_results", {})
    agent_results[target_region_id] = result.to_dict()
    units = provenance["three_model_text_evidence"]["bounded_units"]
    assembled = []
    for current in units:
        region_id = str(current["request"]["region_id"])
        override = agent_results.get(region_id)
        if override and override.get("status") == "TRANSCRIBED":
            assembled.append(str(override["transcribed_text"]))
        else:
            assembled.append(str(current["resolver"].get("selected_text") or ""))
    target["text"] = "\n".join(assembled)
    if "selected_text" in target:
        target["selected_text"] = target["text"]
    return output


def compare_cohort_membership(
    frozen: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare only stable page identities; never inspect truth-bearing payloads."""

    expected_ids = {str(row["page_id"]) for row in frozen.get("pages", ())}
    observed_ids = {str(row["page_id"]) for row in observed.get("pages", ())}
    missing = sorted(expected_ids - observed_ids)
    extra = sorted(observed_ids - expected_ids)
    return {
        "schema": "bemarkdown-large-e2e-v2-cohort-parity-v1",
        "expected_page_count": len(expected_ids),
        "observed_page_count": len(observed_ids),
        "membership_parity": len(missing) == 0 and len(extra) == 0,
        "missing_page_ids": missing,
        "extra_page_ids": extra,
        "gate": "PASS" if not missing and not extra else "FAIL",
    }


def validate_agent_merge(
    tasks: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_ids = [str(row["task_id"]) for row in tasks]
    result_ids = [str(row.get("task_id") or "") for row in results]
    expected = set(expected_ids)
    observed = set(result_ids)
    duplicates = sorted(
        task_id for task_id, count in Counter(result_ids).items() if count > 1
    )
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    invalid_status = sum(
        str(row.get("status") or "") not in AGENT_RESULT_STATUSES for row in results
    )
    counts = Counter(str(row.get("status") or "") for row in results)
    gate = (
        len(expected_ids) == len(expected)
        and not duplicates
        and not missing
        and not unknown
        and invalid_status == 0
    )
    return {
        "schema": "bemarkdown-agent-merge-validation-v2",
        "expected": len(expected_ids),
        "completed": counts["TRANSCRIBED"],
        "unresolved": counts["UNRESOLVED"],
        "source_unreadable": counts["SOURCE_UNREADABLE"],
        "missing": len(missing),
        "duplicate": len(duplicates),
        "unknown": len(unknown),
        "invalid_status": invalid_status,
        "missing_task_ids": missing,
        "duplicate_task_ids": duplicates,
        "unknown_task_ids": unknown,
        "gate": "PASS" if gate else "FAIL",
    }


def count_auto_consensus_mutations(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> int:
    before_rows = {
        str(row.get("region_id") or row.get("route_id")): dict(row)
        for row in before
        if _resolution_status(row) in AUTO_CONSENSUS_STATUSES
    }
    after_rows = {
        str(row.get("region_id") or row.get("route_id")): dict(row)
        for row in after
        if str(row.get("region_id") or row.get("route_id")) in before_rows
    }
    return sum(
        route_id not in after_rows
        or semantic_sha256(row) != semantic_sha256(after_rows[route_id])
        for route_id, row in before_rows.items()
    )


def summarize_three_model_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    a_attempts: int,
    b_attempts: int,
    c_attempts: int,
    request_count: int | None = None,
    request_terminal_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    statuses = (
        "CONSENSUS_AB",
        "CONSENSUS_AC",
        "CONSENSUS_BC",
        "AGENT_REQUIRED_ALL_DIFFER",
        "AGENT_REQUIRED_RUNTIME_FAILURE",
        "FAILED_PRESERVE_INPUT",
    )
    route_counts = Counter(_resolution_status(row) for row in rows)
    route_population = len(rows)
    population = route_population if request_count is None else int(request_count)
    counts = (
        Counter({status: int(request_terminal_counts.get(status, 0)) for status in statuses})
        if request_terminal_counts is not None
        else route_counts
    )
    terminal_total = sum(counts[status] for status in statuses)
    agent_required = counts["AGENT_REQUIRED_ALL_DIFFER"] + counts[
        "AGENT_REQUIRED_RUNTIME_FAILURE"
    ]
    auto_consensus = (
        counts["CONSENSUS_AB"] + counts["CONSENSUS_AC"] + counts["CONSENSUS_BC"]
    )
    route_terminal_total = sum(route_counts[status] for status in statuses)
    route_agent_required = route_counts["AGENT_REQUIRED_ALL_DIFFER"] + route_counts[
        "AGENT_REQUIRED_RUNTIME_FAILURE"
    ]
    gate = (
        a_attempts == population
        and b_attempts == population
        and terminal_total == population
        and 0 <= c_attempts <= population
        and route_terminal_total == route_population
    )
    return {
        "schema": "bemarkdown-large-e2e-v2-three-model-metrics-v2",
        "ocr_text_routes": route_population,
        "three_model_request_count": population,
        "a_attempts": int(a_attempts),
        "b_attempts": int(b_attempts),
        "c_attempts": int(c_attempts),
        "terminal_counts": {status: counts[status] for status in statuses},
        "route_terminal_counts": {
            status: route_counts[status] for status in statuses
        },
        "c_trigger_rate": c_attempts / population if population else None,
        "auto_consensus_rate": auto_consensus / population if population else None,
        "agent_required_rate": agent_required / population if population else None,
        "agent_required_request_rate": (
            agent_required / population if population else None
        ),
        "agent_required_route_rate": (
            route_agent_required / route_population if route_population else None
        ),
        "gate": "PASS" if gate else "FAIL",
    }


def agent_workload_metrics(
    *,
    candidate_page_ids: Sequence[str],
    total_pages: int,
    ocr_text_routes: int,
    all_text_routes: int,
) -> dict[str, Any]:
    if total_pages < 1 or ocr_text_routes < 0 or all_text_routes < 0:
        raise ValueError("E2E_AGENT_WORKLOAD_INPUT_INVALID")
    candidates = len(candidate_page_ids)
    pages = len({str(value) for value in candidate_page_ids})
    return {
        "schema": "bemarkdown-large-e2e-v2-agent-workload-v1",
        "candidate_count": candidates,
        "candidates_per_page": candidates / total_pages,
        "pages_requiring_agent": pages,
        "pages_requiring_agent_rate": pages / total_pages,
        "agent_required_per_ocr_text": (
            candidates / ocr_text_routes if ocr_text_routes else None
        ),
        "agent_required_per_all_text": (
            candidates / all_text_routes if all_text_routes else None
        ),
    }


def content_accounting(
    *,
    source_units: int,
    primary_consumptions: Sequence[str],
    unrouted_ids: Sequence[str],
) -> dict[str, Any]:
    counts = Counter(str(value) for value in primary_consumptions)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    consumed = len(counts)
    unaccounted = max(0, int(source_units) - consumed - len(set(unrouted_ids)))
    return {
        "schema": "bemarkdown-large-e2e-content-accounting-v1",
        "source_units": int(source_units),
        "primary_consumed": consumed,
        "duplicate_primary_consumption": len(duplicates),
        "duplicate_ids": duplicates,
        "unrouted_primary_candidate": len(set(unrouted_ids)),
        "unaccounted_source_unit": unaccounted,
        "silent_drop": unaccounted,
        "gate": "PASS" if not duplicates and not unrouted_ids and not unaccounted else "FAIL",
    }


def validate_handoff_portability(value: Any) -> dict[str, Any]:
    violations: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            if PureWindowsPath(item).is_absolute() or PurePosixPath(item).is_absolute():
                violations.append(path)
            if "tmp/" in item.replace("\\", "/").casefold():
                violations.append(path)
            if "truth" in item.casefold() or "agent_scratch" in item.casefold():
                violations.append(path)

    visit(value, "$")
    unique = sorted(set(violations))
    return {
        "schema": "bemarkdown-large-e2e-handoff-integrity-v1",
        "violation_count": len(unique),
        "violations": unique,
        "gate": "PASS" if not unique else "FAIL",
    }


def performance_metrics(*, wall_seconds: float, pages: int) -> dict[str, Any]:
    if wall_seconds < 0 or pages < 1:
        raise ValueError("E2E_PERFORMANCE_INPUT_INVALID")
    return {
        "wall_seconds": wall_seconds,
        "pages": pages,
        "seconds_per_page": wall_seconds / pages,
        "pages_per_second": pages / wall_seconds if wall_seconds else None,
    }


def _resolution_status(row: Mapping[str, Any]) -> str:
    direct = row.get("resolution_status")
    if direct:
        return str(direct)
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        return str(row.get("status") or "")
    evidence = provenance.get("three_model_text_evidence")
    if not isinstance(evidence, Mapping):
        return str(row.get("status") or "")
    resolver = evidence.get("resolver")
    if not isinstance(resolver, Mapping):
        return str(row.get("status") or "")
    return str(resolver.get("resolution_status") or "")


def _contains_truth(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "truth" in str(key).casefold() or _contains_truth(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_truth(item) for item in value)
    return False
