"""Production-owned truth-blind Agent handoff contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class AgentTaskV2:
    task_id: str
    document_id: str
    page_id: str
    region_id: str
    source_candidate_id: str
    bbox: tuple[float, float, float, float]
    primary_crop_ref: str
    primary_crop_sha256: str
    route_kind: str
    text_track: str
    reason: Literal["ALL_DIFFER", "RUNTIME_FAILURE"]
    source_provenance: Mapping[str, Any]
    evidence: Mapping[str, Any] = field(default_factory=dict)
    optional_context_crop_ref: str | None = None
    optional_context_crop_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = "bemarkdown-agent-task-v2"
        value["bbox"] = list(self.bbox)
        if _contains_truth(value):
            raise ValueError("AGENT_TASK_TRUTH_LEAK")
        if len(self.primary_crop_sha256) != 64:
            raise ValueError("AGENT_TASK_CROP_SHA_INVALID")
        return value


def _contains_truth(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "truth" in str(key).casefold() or _contains_truth(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_truth(item) for item in value)
    return False
