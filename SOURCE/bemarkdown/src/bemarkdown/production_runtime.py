from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import shutil
import threading
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from types import TracebackType
from typing import Any, Self

PERFORMANCE_TRACE_SCHEMA = "bemarkdown-performance-trace-v0"
PERFORMANCE_TRACE_REQUIRED_SPANS = (
    "document open",
    "page source extraction",
    "DisplayList creation",
    "native text extraction",
    "page render",
    "page encode",
    "layout model load",
    "layout inference",
    "fusion",
    "router",
    "OCR det",
    "OCR rec",
    "FormulaNet",
    "Table classifier",
    "SLANeXt",
    "RT-DETR",
    "Table OCR",
    "DocumentIR assembly",
    "Draft render",
    "asset hash",
    "asset hardlink/copy",
    "manifest serialization",
    "directory staging/rename",
    "Audit package generation",
    "Agent queue wait",
    "Agent provider wall",
    "patch apply",
    "machine validation",
    "Lean Handoff render",
)

ARTIFACT_CACHE_NAMES = frozenset(
    {
        "RenderCache",
        "SourceEvidenceCache",
        "LayoutCache",
        "OCRCache",
        "FormulaCache",
        "TableCache",
        "AuditPackageCache",
    }
)


def _canonical_value(value: Any, exclude_keys: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item, exclude_keys)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in exclude_keys
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, exclude_keys) for item in value]
    if isinstance(value, set):
        normalized = [_canonical_value(item, exclude_keys) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if value == 0.0:
            return 0.0
    return value


def canonical_json(value: Any, *, exclude_keys: set[str] | None = None) -> str:
    normalized = _canonical_value(value, frozenset(exclude_keys or set()))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def semantic_sha256(value: Any, *, exclude_keys: set[str] | None = None) -> str:
    return hashlib.sha256(
        canonical_json(value, exclude_keys=exclude_keys).encode("utf-8")
    ).hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    for attempt in range(3):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.05)


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(value)
    for attempt in range(3):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.05)


@dataclass
class _OpenSpan:
    trace: PerformanceTrace
    name: str
    attributes: dict[str, Any]
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_span_id: str | None = None
    output_units: int | float | None = None
    bytes_out: int | None = None
    cache_status: str | None = None
    model_load_count: int | None = None
    queue_wait_seconds: float | None = None
    _started_wall: float = 0.0
    _started_cpu: float = 0.0
    _started_utc: str = ""

    def __enter__(self) -> Self:
        self.parent_span_id = self.trace._parent_span_id()
        self._started_wall = self.trace._wall_clock()
        self._started_cpu = self.trace._cpu_clock()
        self._started_utc = self.trace._utc_now()
        self.trace._push(self.span_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        ended_wall = self.trace._wall_clock()
        ended_cpu = self.trace._cpu_clock()
        self.trace._pop(self.span_id)
        resources = self.trace._sample_resources()
        row = {
            "schema": PERFORMANCE_TRACE_SCHEMA,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "status": "ERROR" if exc is not None else "OK",
            "error": f"{type(exc).__name__}: {exc}" if exc is not None else None,
            "started_at": self._started_utc,
            "wall_seconds": max(0.0, ended_wall - self._started_wall),
            "cpu_seconds": max(0.0, ended_cpu - self._started_cpu),
            "input_units": self.attributes.get("input_units"),
            "output_units": self.output_units
            if self.output_units is not None
            else self.attributes.get("output_units"),
            "bytes_in": self.attributes.get("bytes_in"),
            "bytes_out": self.bytes_out
            if self.bytes_out is not None
            else self.attributes.get("bytes_out"),
            "batch_size": self.attributes.get("batch_size"),
            "queue_wait_seconds": self.queue_wait_seconds
            if self.queue_wait_seconds is not None
            else self.attributes.get("queue_wait_seconds"),
            "cache_status": self.cache_status
            if self.cache_status is not None
            else self.attributes.get("cache_status"),
            "model_load_count": self.model_load_count
            if self.model_load_count is not None
            else self.attributes.get("model_load_count"),
            "peak_rss_bytes": resources.get("peak_rss_bytes"),
            "gpu_memory_bytes": resources.get("gpu_memory_bytes"),
            "gpu_utilization_percent": resources.get("gpu_utilization_percent"),
            "disk_read_bytes": resources.get("disk_read_bytes"),
            "disk_write_bytes": resources.get("disk_write_bytes"),
            "attributes": {
                key: copy.deepcopy(value)
                for key, value in self.attributes.items()
                if key
                not in {
                    "input_units",
                    "output_units",
                    "bytes_in",
                    "bytes_out",
                    "batch_size",
                    "queue_wait_seconds",
                    "cache_status",
                    "model_load_count",
                }
            },
        }
        self.trace._append(row)
        return False


class PerformanceTrace:
    """Thread-safe production trace whose unavailable resource fields stay null."""

    def __init__(
        self,
        *,
        wall_clock: Callable[[], float] = time.perf_counter,
        cpu_clock: Callable[[], float] = time.process_time,
        utc_now: Callable[[], str] | None = None,
        resource_sampler: Callable[[], Mapping[str, int | float | None]] | None = None,
    ):
        self._wall_clock = wall_clock
        self._cpu_clock = cpu_clock
        self._utc_now = utc_now or (lambda: datetime.now(UTC).isoformat())
        self._resource_sampler = resource_sampler
        self._rows: list[dict[str, Any]] = []
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._rows)

    def span(self, name: str, **attributes: Any) -> _OpenSpan:
        if not name.strip():
            raise ValueError("PERFORMANCE_SPAN_NAME_REQUIRED")
        return _OpenSpan(self, name, dict(attributes))

    def record_not_measured(self, name: str, *, reason: str) -> None:
        if not name.strip() or not reason.strip():
            raise ValueError("NOT_MEASURED_SPAN_NAME_AND_REASON_REQUIRED")
        self._append(
            {
                "schema": PERFORMANCE_TRACE_SCHEMA,
                "span_id": uuid.uuid4().hex,
                "parent_span_id": None,
                "name": name,
                "status": "NOT_MEASURED",
                "error": None,
                "started_at": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "input_units": None,
                "output_units": None,
                "bytes_in": None,
                "bytes_out": None,
                "batch_size": None,
                "queue_wait_seconds": None,
                "cache_status": None,
                "model_load_count": None,
                "peak_rss_bytes": None,
                "gpu_memory_bytes": None,
                "gpu_utilization_percent": None,
                "disk_read_bytes": None,
                "disk_write_bytes": None,
                "attributes": {"reason": reason},
            }
        )

    def record_measurement(
        self,
        name: str,
        *,
        wall_seconds: float,
        cpu_seconds: float | None = None,
        **attributes: Any,
    ) -> None:
        if wall_seconds < 0 or (cpu_seconds is not None and cpu_seconds < 0):
            raise ValueError("PERFORMANCE_MEASUREMENT_DURATION_INVALID")
        resources = self._sample_resources()
        self._append(
            {
                "schema": PERFORMANCE_TRACE_SCHEMA,
                "span_id": uuid.uuid4().hex,
                "parent_span_id": None,
                "name": name,
                "status": "OK",
                "error": None,
                "started_at": None,
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "input_units": attributes.pop("input_units", None),
                "output_units": attributes.pop("output_units", None),
                "bytes_in": attributes.pop("bytes_in", None),
                "bytes_out": attributes.pop("bytes_out", None),
                "batch_size": attributes.pop("batch_size", None),
                "queue_wait_seconds": attributes.pop("queue_wait_seconds", None),
                "cache_status": attributes.pop("cache_status", None),
                "model_load_count": attributes.pop("model_load_count", None),
                "peak_rss_bytes": resources.get("peak_rss_bytes"),
                "gpu_memory_bytes": resources.get("gpu_memory_bytes"),
                "gpu_utilization_percent": resources.get("gpu_utilization_percent"),
                "disk_read_bytes": resources.get("disk_read_bytes"),
                "disk_write_bytes": resources.get("disk_write_bytes"),
                "attributes": copy.deepcopy(attributes),
            }
        )

    def _stack(self) -> list[str]:
        stack = getattr(self._local, "span_stack", None)
        if stack is None:
            stack = []
            self._local.span_stack = stack
        return stack

    def _parent_span_id(self) -> str | None:
        stack = self._stack()
        return stack[-1] if stack else None

    def _push(self, span_id: str) -> None:
        self._stack().append(span_id)

    def _pop(self, span_id: str) -> None:
        stack = self._stack()
        if not stack or stack[-1] != span_id:
            raise RuntimeError("PERFORMANCE_SPAN_STACK_CORRUPT")
        stack.pop()

    def _append(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._rows.append(row)

    def _sample_resources(self) -> dict[str, int | float | None]:
        unavailable = {
            "peak_rss_bytes": None,
            "gpu_memory_bytes": None,
            "gpu_utilization_percent": None,
            "disk_read_bytes": None,
            "disk_write_bytes": None,
        }
        if self._resource_sampler is None:
            return unavailable
        try:
            sampled = dict(self._resource_sampler())
        except Exception:  # noqa: BLE001 - metrics failure must not fail conversion
            return unavailable
        return {key: sampled.get(key) for key in unavailable}

    def summary(self) -> dict[str, Any]:
        rows = self.rows
        stages: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {
                "span_count": 0,
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "input_units": 0,
                "output_units": 0,
                "bytes_in": 0,
                "bytes_out": 0,
                "queue_wait_seconds": 0.0,
            }
        )
        for row in rows:
            stage = stages[row["name"]]
            stage["span_count"] += 1
            for key in ("wall_seconds", "cpu_seconds", "queue_wait_seconds"):
                stage[key] += float(row.get(key) or 0)
            for key in ("input_units", "output_units", "bytes_in", "bytes_out"):
                stage[key] += int(row.get(key) or 0)
        normalized = {
            name: {
                key: round(value, 9) if isinstance(value, float) else value
                for key, value in metrics.items()
            }
            for name, metrics in sorted(stages.items())
        }
        return {
            "schema": "bemarkdown-performance-summary-v0",
            "span_count": len(rows),
            "error_count": sum(row["status"] == "ERROR" for row in rows),
            "stage_totals": normalized,
            "required_span_coverage": {
                name: name in normalized for name in PERFORMANCE_TRACE_REQUIRED_SPANS
            },
        }

    def write(self, output_dir: str | Path) -> dict[str, str]:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = self.rows
        trace_path = output_dir / "performance_trace.jsonl"
        _atomic_write_text(
            trace_path,
            "".join(canonical_json(row) + "\n" for row in rows),
        )
        summary = self.summary()
        summary_path = output_dir / "performance_summary.json"
        stage_path = output_dir / "stage_breakdown.json"
        queue_path = output_dir / "queue_wait_breakdown.json"
        _atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        _atomic_write_text(
            stage_path,
            json.dumps(summary["stage_totals"], ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write_text(
            queue_path,
            json.dumps(
                {
                    name: values["queue_wait_seconds"]
                    for name, values in summary["stage_totals"].items()
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        return {
            "performance_trace": str(trace_path),
            "performance_summary": str(summary_path),
            "stage_breakdown": str(stage_path),
            "queue_wait_breakdown": str(queue_path),
        }


class PerformanceCorpusSelector:
    """Deterministic nested Quick/Medium/Formal selection from source-only tags."""

    REQUIRED_TAGS = (
        "native_text",
        "scan",
        "formula_heavy",
        "table",
        "image",
        "multi_column",
        "mixed",
    )
    TARGETS = (("Quick", 10), ("Medium", 50), ("Formal", 100))

    @staticmethod
    def _page_id(page: Mapping[str, Any]) -> str:
        return f"{page['document_id']}:{int(page['page_index'])}"

    @classmethod
    def _sort_key(cls, page: Mapping[str, Any]) -> tuple[str, str, int]:
        identity = {
            "source_sha256": page["source_sha256"],
            "document_id": page["document_id"],
            "page_index": int(page["page_index"]),
        }
        return (
            semantic_sha256(identity),
            str(page["document_id"]),
            int(page["page_index"]),
        )

    def select(self, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        pages = [copy.deepcopy(dict(page)) for page in candidates]
        if len(pages) < 100:
            raise ValueError("PERFORMANCE_CORPUS_REQUIRES_AT_LEAST_100_PAGES")
        page_ids = [self._page_id(page) for page in pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("PERFORMANCE_CORPUS_DUPLICATE_PAGE")
        if any(
            not page.get("source_sha256")
            or not isinstance(page.get("source_tags"), list)
            for page in pages
        ):
            raise ValueError("PERFORMANCE_CORPUS_SOURCE_FIELDS_REQUIRED")
        ordered = sorted(pages, key=self._sort_key)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        result: dict[str, Any] = {}
        for profile, target in self.TARGETS:
            for tag in self.REQUIRED_TAGS:
                if any(tag in page["source_tags"] for page in selected):
                    continue
                candidate = next(
                    (
                        page
                        for page in ordered
                        if self._page_id(page) not in selected_ids
                        and tag in page["source_tags"]
                    ),
                    None,
                )
                if candidate is not None:
                    selected.append(candidate)
                    selected_ids.add(self._page_id(candidate))
            for page in ordered:
                if len(selected) >= target:
                    break
                page_id = self._page_id(page)
                if page_id not in selected_ids:
                    selected.append(page)
                    selected_ids.add(page_id)
            if len(selected) < target:
                raise ValueError(f"PERFORMANCE_CORPUS_TARGET_UNAVAILABLE:{profile}")
            profile_pages = selected[:target]
            tags = Counter(
                tag for page in profile_pages for tag in page.get("source_tags", [])
            )
            result[profile] = {
                "page_count": target,
                "pages": [
                    {
                        **page,
                        "page_id": self._page_id(page),
                    }
                    for page in profile_pages
                ],
                "source_tag_counts": dict(sorted(tags.items())),
            }
        return result

@dataclass(frozen=True)
class ArtifactCacheKey:
    namespace: str
    source_sha256: str
    page_index: int | None = None
    crop_sha256: str | None = None
    render_config: Mapping[str, Any] = field(default_factory=dict)
    model_fingerprint: str | None = None
    policy_version: str | None = None
    backend_profile: str = "portable_fp32"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.namespace not in ARTIFACT_CACHE_NAMES:
            raise ValueError(f"UNKNOWN_ARTIFACT_CACHE_NAMESPACE:{self.namespace}")
        if len(self.source_sha256) != 64:
            raise ValueError("SOURCE_SHA256_REQUIRED")

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "source_sha256": self.source_sha256,
            "page_index": self.page_index,
            "crop_sha256": self.crop_sha256,
            "render_config": dict(self.render_config),
            "model_fingerprint": self.model_fingerprint,
            "policy_version": self.policy_version,
            "backend_profile": self.backend_profile,
            "extra": dict(self.extra),
        }

    @property
    def fingerprint(self) -> str:
        return semantic_sha256(self.to_dict())


class BeMarkdownArtifactCache:
    """Content-addressed cache with fail-closed integrity validation."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def _paths(self, key: ArtifactCacheKey) -> tuple[Path, Path]:
        entry = self.root / key.namespace / key.fingerprint[:2] / key.fingerprint
        return entry / "payload.bin", entry / "metadata.json"

    def put_bytes(
        self,
        key: ArtifactCacheKey,
        payload: bytes,
        *,
        semantic_sha256_value: str | None = None,
    ) -> dict[str, Any]:
        payload_path, metadata_path = self._paths(key)
        payload_sha = hashlib.sha256(payload).hexdigest()
        metadata = {
            "schema": "bemarkdown-artifact-cache-entry-v0",
            "key": key.to_dict(),
            "key_fingerprint": key.fingerprint,
            "payload_sha256": payload_sha,
            "semantic_sha256": semantic_sha256_value or payload_sha,
            "bytes": len(payload),
        }
        _atomic_write_bytes(payload_path, payload)
        _atomic_write_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        with self._lock:
            self._counts["writes"] += 1
            self._counts["bytes_written"] += len(payload)
        return {
            **metadata,
            "payload_path": str(payload_path),
            "metadata_path": str(metadata_path),
        }

    def get_bytes(
        self,
        key: ArtifactCacheKey,
        *,
        expected_semantic_sha256: str | None = None,
    ) -> bytes | None:
        payload_path, metadata_path = self._paths(key)
        if not payload_path.is_file() or not metadata_path.is_file():
            with self._lock:
                self._counts["misses"] += 1
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload = payload_path.read_bytes()
            valid = (
                metadata.get("schema") == "bemarkdown-artifact-cache-entry-v0"
                and metadata.get("key_fingerprint") == key.fingerprint
                and metadata.get("key") == key.to_dict()
                and metadata.get("payload_sha256") == hashlib.sha256(payload).hexdigest()
                and (
                    expected_semantic_sha256 is None
                    or metadata.get("semantic_sha256") == expected_semantic_sha256
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
            payload = b""
        with self._lock:
            if not valid:
                self._counts["misses"] += 1
                self._counts["integrity_failures"] += 1
                return None
            self._counts["hits"] += 1
            self._counts["bytes_read"] += len(payload)
        return payload

    def put_json(self, key: ArtifactCacheKey, value: Any) -> dict[str, Any]:
        payload = canonical_json(value).encode("utf-8")
        return self.put_bytes(
            key,
            payload,
            semantic_sha256_value=semantic_sha256(value),
        )

    def get_json(self, key: ArtifactCacheKey) -> Any | None:
        payload = self.get_bytes(key)
        if payload is None:
            return None
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            with self._lock:
                self._counts["hits"] -= 1
                self._counts["misses"] += 1
                self._counts["integrity_failures"] += 1
            return None

    def metrics(self) -> dict[str, int]:
        with self._lock:
            counts = dict(self._counts)
        return {
            "hits": counts.get("hits", 0),
            "misses": counts.get("misses", 0),
            "writes": counts.get("writes", 0),
            "bytes_read": counts.get("bytes_read", 0),
            "bytes_written": counts.get("bytes_written", 0),
            "integrity_failures": counts.get("integrity_failures", 0),
            "stale_hits_returned": 0,
        }


class ImmutableAssetStore:
    """Single immutable content store; Handoff materializes links only at finalize."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, Path] = {}
        self._counts: Counter[str] = Counter()

    def ingest_bytes(self, value: bytes, *, suffix: str = "") -> dict[str, Any]:
        suffix = suffix if not suffix or suffix.startswith(".") else f".{suffix}"
        digest = hashlib.sha256(value).hexdigest()
        asset_uid = f"sha256-{digest}"
        path = self.root / digest[:2] / f"{digest}{suffix.lower()}"
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("IMMUTABLE_ASSET_STORE_COLLISION")
            self._counts["deduplicated"] += 1
        else:
            _atomic_write_bytes(path, value)
            self._counts["ingested"] += 1
            self._counts["bytes_ingested"] += len(value)
        self._records[asset_uid] = path
        return {
            "asset_uid": asset_uid,
            "sha256": digest,
            "bytes": len(value),
            "path": str(path),
        }

    def ingest_file(self, source: str | Path) -> dict[str, Any]:
        source = Path(source).resolve()
        return self.ingest_bytes(source.read_bytes(), suffix=source.suffix)

    def finalize(self, asset_uid: str, target: str | Path) -> dict[str, Any]:
        source = self._records.get(asset_uid)
        if source is None or not source.is_file():
            raise KeyError(f"UNKNOWN_ASSET_UID:{asset_uid}")
        target = Path(target).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != asset_uid.removeprefix("sha256-"):
                raise FileExistsError(f"ASSET_TARGET_CONFLICT:{target}")
            return {"source": str(source), "target": str(target), "method": "existing"}
        try:
            os.link(source, target)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            method = "copy"
        self._counts[method] += 1
        self._counts[f"{method}_bytes"] += target.stat().st_size
        return {"source": str(source), "target": str(target), "method": method}

    def metrics(self) -> dict[str, int]:
        return dict(sorted(self._counts.items()))


class DocumentWindowScheduler:
    def __init__(self, *, max_pages: int = 24, byte_budget: int = 512 * 1024**2):
        if not 1 <= max_pages <= 32:
            raise ValueError("WINDOW_PAGE_LIMIT_MUST_BE_1_TO_32")
        if byte_budget <= 0:
            raise ValueError("WINDOW_BYTE_BUDGET_MUST_BE_POSITIVE")
        self.max_pages = max_pages
        self.byte_budget = byte_budget

    def plan(self, pages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted((dict(page) for page in pages), key=lambda page: int(page["page_index"]))
        if len({int(page["page_index"]) for page in ordered}) != len(ordered):
            raise ValueError("DUPLICATE_PAGE_INDEX")
        windows: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        current_bytes = 0

        def close() -> None:
            nonlocal current, current_bytes
            if not current:
                return
            windows.append(
                {
                    "window_id": f"window-{len(windows) + 1:06d}",
                    "pages": current,
                    "estimated_bytes": current_bytes,
                }
            )
            current = []
            current_bytes = 0

        for page in ordered:
            estimate = int(page.get("estimated_bytes") or 0)
            if estimate < 0:
                raise ValueError("PAGE_ESTIMATED_BYTES_INVALID")
            if estimate > self.byte_budget:
                raise ValueError("SINGLE_PAGE_EXCEEDS_WINDOW_BYTE_BUDGET")
            if current and (
                len(current) >= self.max_pages or current_bytes + estimate > self.byte_budget
            ):
                close()
            current.append(page)
            current_bytes += estimate
        close()
        flattened = [int(page["page_index"]) for window in windows for page in window["pages"]]
        if flattened != [int(page["page_index"]) for page in ordered]:
            raise RuntimeError("WINDOW_PAGE_COVERAGE_INVALID")
        return windows


class ByteBudget:
    """Blocking byte budget used by source/render queues instead of item counts."""

    def __init__(self, capacity_bytes: int):
        if capacity_bytes <= 0:
            raise ValueError("BYTE_BUDGET_MUST_BE_POSITIVE")
        self.capacity_bytes = capacity_bytes
        self.used_bytes = 0
        self.peak_bytes = 0
        self._condition = threading.Condition()

    def acquire(self, size_bytes: int, *, timeout: float | None = None) -> None:
        if not 0 <= size_bytes <= self.capacity_bytes:
            raise ValueError("ITEM_EXCEEDS_BYTE_BUDGET")
        started = time.monotonic()
        with self._condition:
            while self.used_bytes + size_bytes > self.capacity_bytes:
                remaining = None
                if timeout is not None:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("BYTE_BUDGET_WAIT_TIMEOUT")
                self._condition.wait(remaining)
            self.used_bytes += size_bytes
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)

    def release(self, size_bytes: int) -> None:
        with self._condition:
            if size_bytes < 0 or size_bytes > self.used_bytes:
                raise ValueError("BYTE_BUDGET_RELEASE_INVALID")
            self.used_bytes -= size_bytes
            self._condition.notify_all()


@dataclass
class ModelHandle:
    family: str
    instance: Any
    loaded_at: float
    last_used_at: float
    pin_count: int = 0
    use_count: int = 0


class ModelPool:
    def __init__(
        self,
        factories: Mapping[str, Callable[[], Any]],
        *,
        idle_timeout_seconds: float = 900,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._factories = dict(factories)
        self.idle_timeout_seconds = idle_timeout_seconds
        self._clock = clock
        self._handles: dict[str, ModelHandle] = {}
        self._counts: Counter[str] = Counter()
        self._eviction_reasons: Counter[str] = Counter()
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def acquire(self, family: str) -> Iterator[Any]:
        with self._lock:
            handle = self._handles.get(family)
            if handle is None:
                factory = self._factories.get(family)
                if factory is None:
                    raise KeyError(f"MODEL_FACTORY_NOT_REGISTERED:{family}")
                instance = factory()
                now = self._clock()
                handle = ModelHandle(family, instance, now, now)
                self._handles[family] = handle
                self._counts["load_count"] += 1
                if self._counts[f"family_load:{family}"]:
                    self._counts["reload_count"] += 1
                self._counts[f"family_load:{family}"] += 1
            handle.pin_count += 1
            handle.use_count += 1
            handle.last_used_at = self._clock()
        try:
            yield handle.instance
        finally:
            with self._lock:
                handle.pin_count -= 1
                handle.last_used_at = self._clock()

    def get(self, family: str) -> Any:
        """Return a resident model without transferring lifecycle ownership."""

        with self.acquire(family) as instance:
            return instance

    def _evict(self, family: str, reason: str) -> bool:
        handle = self._handles.get(family)
        if handle is None or handle.pin_count:
            return False
        close = getattr(handle.instance, "close", None) or getattr(handle.instance, "unload", None)
        if close is not None:
            close()
        del self._handles[family]
        self._counts["eviction_count"] += 1
        self._eviction_reasons[reason] += 1
        return True

    def evict_idle(self) -> list[str]:
        now = self._clock()
        with self._lock:
            candidates = sorted(
                family
                for family, handle in self._handles.items()
                if not handle.pin_count
                and now - handle.last_used_at >= self.idle_timeout_seconds
            )
            return [family for family in candidates if self._evict(family, "IDLE_TIMEOUT")]

    def evict_for_memory(self, required_families: set[str] | None = None) -> list[str]:
        required_families = required_families or set()
        with self._lock:
            candidates = sorted(
                (
                    handle
                    for family, handle in self._handles.items()
                    if family not in required_families and not handle.pin_count
                ),
                key=lambda handle: (handle.last_used_at, handle.family),
            )
            return [
                handle.family
                for handle in candidates
                if self._evict(handle.family, "MEMORY_PRESSURE")
            ]

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "load_count": self._counts["load_count"],
                "eviction_count": self._counts["eviction_count"],
                "reload_count": self._counts["reload_count"],
                "resident_families": sorted(self._handles),
                "eviction_reasons": dict(sorted(self._eviction_reasons.items())),
                "family_load_count": {
                    key.removeprefix("family_load:"): value
                    for key, value in sorted(self._counts.items())
                    if key.startswith("family_load:")
                },
            }


class ModelResidencyManager:
    def __init__(
        self,
        *,
        total_vram_bytes: int,
        reserve_bytes: int | None = None,
        envelope: Mapping[str, Mapping[str, int | None]] | None = None,
    ):
        if total_vram_bytes <= 0:
            raise ValueError("TOTAL_VRAM_MUST_BE_POSITIVE")
        default_reserve = max(int(total_vram_bytes * 0.20), int(1.5 * 1024**3))
        self.total_vram_bytes = total_vram_bytes
        self.reserve_bytes = reserve_bytes if reserve_bytes is not None else default_reserve
        if not 0 < self.reserve_bytes < total_vram_bytes:
            raise ValueError("VRAM_RESERVE_INVALID")
        self.usable_bytes = total_vram_bytes - self.reserve_bytes
        self.envelope = {family: dict(values) for family, values in (envelope or {}).items()}

    def policy_for(self, family: str, *, stage: str, route_population: int) -> str:
        family_lower = family.lower()
        stage_lower = stage.lower()
        if family_lower in {"ocr", "formula", "formulanet"} and route_population > 0:
            return "resident"
        if family_lower == "layout" and stage_lower == "layout":
            return "resident"
        if family_lower.startswith("table") or family_lower in {"slanext", "rt-detr"}:
            return "stage-resident"
        return "stage-resident" if route_population else "evicted"

    def fits(self, families: Sequence[str]) -> bool:
        total = 0
        for family in families:
            peak = self.envelope.get(family, {}).get("peak_batch_vram_bytes")
            if peak is None:
                return False
            total += int(peak)
        return total <= self.usable_bytes


class GpuResourceScheduler:
    """One-process, one-worker GPU arbitration with no CPU fallback route."""

    def __init__(self, *, device: str = "gpu:0"):
        if not device.startswith("gpu:"):
            raise ValueError("GPU_SCHEDULER_REQUIRES_GPU_DEVICE")
        self.device = device
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._active_family: str | None = None

    def run(self, family: str, operation: Callable[[], Any]) -> Any:
        if not family.strip():
            raise ValueError("GPU_MODEL_FAMILY_REQUIRED")
        wait_started = time.perf_counter()
        with self._lock:
            waited = time.perf_counter() - wait_started
            self._counts["calls"] += 1
            self._counts["queue_wait_microseconds"] += int(waited * 1_000_000)
            self._active_family = family
            try:
                return operation()
            finally:
                self._active_family = None

    def metrics(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "worker_concurrency": 1,
            "calls": self._counts["calls"],
            "queue_wait_seconds": round(
                self._counts["queue_wait_microseconds"] / 1_000_000, 6
            ),
            "cpu_fallback_count": 0,
            "active_family": self._active_family,
        }


class FormulaBatchPlanner:
    def __init__(self, batch_sizes: Mapping[str, int] | None = None):
        self.batch_sizes = dict(
            batch_sizes or {"small": 8, "medium": 4, "large_wide": 1}
        )
        if set(self.batch_sizes) != {"small", "medium", "large_wide"}:
            raise ValueError("FORMULA_BUCKET_BATCH_SIZES_INCOMPLETE")
        if any(value <= 0 for value in self.batch_sizes.values()):
            raise ValueError("FORMULA_BUCKET_BATCH_SIZE_INVALID")

    @staticmethod
    def bucket(crop: Mapping[str, Any]) -> str:
        width = int(crop["width"])
        height = int(crop["height"])
        if width <= 0 or height <= 0:
            raise ValueError("FORMULA_CROP_GEOMETRY_INVALID")
        area = width * height
        aspect = max(width / height, height / width)
        # A short inline expression can be wide without being a large tensor.
        if area <= 80_000 and max(width, height) <= 600:
            return "small"
        if area >= 400_000 or aspect >= 4.0 or max(width, height) >= 900:
            return "large_wide"
        return "medium"

    def plan(self, crops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for crop in crops:
            grouped[self.bucket(crop)].append(dict(crop))
        order = {"large_wide": 0, "medium": 1, "small": 2}
        return [
            {
                "bucket": bucket,
                "batch_size": self.batch_sizes[bucket],
                "items": sorted(items, key=lambda item: str(item.get("id") or item.get("crop_sha256"))),
            }
            for bucket, items in sorted(grouped.items(), key=lambda pair: order[pair[0]])
        ]


class BatchSizeController:
    def __init__(self, candidates: Mapping[str, Sequence[int]]):
        self.candidates = {
            family: sorted({int(size) for size in sizes if int(size) > 0})
            for family, sizes in candidates.items()
        }
        if any(not sizes for sizes in self.candidates.values()):
            raise ValueError("BATCH_CANDIDATES_REQUIRED")
        self._oom_history: Counter[str] = Counter()
        self._previous_peak: dict[tuple[str, int], int | None] = {}
        self._counts: Counter[str] = Counter()

    def choose(
        self,
        family: str,
        *,
        preferred: int | None = None,
        vram_budget_bytes: int | None = None,
    ) -> int:
        sizes = self.candidates[family]
        ceiling = preferred or sizes[-1]
        safe = [size for size in sizes if size <= ceiling]
        if vram_budget_bytes is not None:
            safe = [
                size
                for size in safe
                if self._previous_peak.get((family, size)) in {None, 0}
                or int(self._previous_peak[(family, size)] or 0) <= vram_budget_bytes
            ]
        if not safe:
            raise RuntimeError("NO_SAFE_BATCH_SIZE")
        return safe[-1]

    def record_peak(self, family: str, batch_size: int, peak_vram_bytes: int | None) -> None:
        self._previous_peak[(family, batch_size)] = peak_vram_bytes

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        message = str(exc).lower()
        return isinstance(exc, MemoryError) or "out of memory" in message or "oom" in message

    def execute(
        self,
        family: str,
        items: list[Any],
        infer: Callable[[list[Any], int], list[Any]],
        *,
        preferred: int | None = None,
        vram_budget_bytes: int | None = None,
    ) -> list[Any]:
        initial = self.choose(
            family, preferred=preferred, vram_budget_bytes=vram_budget_bytes
        )
        try:
            result = infer(items, initial)
            self._counts["inference_count"] += 1
            return result
        except Exception as exc:
            if not self._is_oom(exc):
                raise
            self._oom_history[family] += 1
            self._counts["oom_count"] += 1
            lower = [size for size in self.candidates[family] if size < initial]
            if not lower:
                raise RuntimeError("GPU_OOM_NO_LOWER_BATCH") from exc
            retry_size = lower[-1]
            self._counts["oom_retry_count"] += 1
            try:
                result = infer(items, retry_size)
                self._counts["inference_count"] += 1
                return result
            except Exception as retry_exc:
                if self._is_oom(retry_exc):
                    self._oom_history[family] += 1
                    self._counts["oom_count"] += 1
                    raise RuntimeError("GPU_OOM_AFTER_SINGLE_RETRY") from retry_exc
                raise

    def metrics(self) -> dict[str, Any]:
        return {
            "inference_count": self._counts["inference_count"],
            "oom_count": self._counts["oom_count"],
            "oom_retry_count": self._counts["oom_retry_count"],
            "cpu_fallback_count": 0,
            "oom_history": dict(sorted(self._oom_history.items())),
        }


@dataclass(frozen=True)
class InferenceBackendProfile:
    name: str
    precision: str
    hpi: bool
    is_default: bool
    output_class: str
    optimizations: tuple[str, ...] = ()
    support_reason: str | None = None

    def __post_init__(self) -> None:
        allowed = {"SAFE_OPTIMIZATION", "OUTPUT_CHANGING_OPTIMIZATION", "UNSUPPORTED"}
        if self.output_class not in allowed:
            raise ValueError("BACKEND_OUTPUT_CLASS_INVALID")
        if self.is_default and self.output_class != "SAFE_OPTIMIZATION":
            raise ValueError("OUTPUT_CHANGING_PROFILE_CANNOT_BE_DEFAULT")
        if self.is_default and self.precision != "FP32":
            raise ValueError("DEFAULT_PROFILE_MUST_REMAIN_FP32")

    @classmethod
    def portable_fp32(cls) -> InferenceBackendProfile:
        return cls("portable_fp32", "FP32", False, True, "SAFE_OPTIMIZATION")

    @classmethod
    def optimized_fp32(cls, *, promoted: bool = False) -> InferenceBackendProfile:
        return cls(
            "optimized_fp32",
            "FP32",
            False,
            promoted,
            "SAFE_OPTIMIZATION",
            (
                "multiprocessing_source",
                "displaylist_reuse",
                "artifact_cache",
                "safe_batching",
                "model_residency",
                "windowed_pipeline",
                "audit_overlap",
            ),
        )

    @classmethod
    def hpi_experimental(
        cls, *, supported: bool, reason: str | None = None
    ) -> InferenceBackendProfile:
        return cls(
            "hpi_experimental",
            "FP32",
            True,
            False,
            "OUTPUT_CHANGING_OPTIMIZATION" if supported else "UNSUPPORTED",
            support_reason=reason,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_source_row(
    document: Any,
    item: Mapping[str, Any],
    page_index: int,
    *,
    opened_seconds: float,
) -> dict[str, Any]:
    import fitz

    dpi = int(item["dpi"])
    try:
        if page_index < 0 or page_index >= document.page_count:
            raise IndexError(f"page {page_index} outside 0..{document.page_count - 1}")
        page = document[page_index]
        display_started = time.perf_counter()
        display_list = page.get_displaylist()
        display_seconds = time.perf_counter() - display_started
        text_started = time.perf_counter()
        native_text_flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
        if page.rotation:
            text_page = page.get_textpage(flags=native_text_flags)
            textpage_from_displaylist = False
            textpage_fallback_reason = "PAGE_ROTATION_REQUIRES_PAGE_TEXT_MATRIX"
        else:
            text_page = fitz.TextPage(
                display_list.get_textpage(flags=native_text_flags)
            )
            textpage_from_displaylist = True
            textpage_fallback_reason = None
        source_evidence = json.loads(text_page.extractJSON())
        text_seconds = time.perf_counter() - text_started
        scale = dpi / 72.0
        render_started = time.perf_counter()
        pixmap = display_list.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        render_seconds = time.perf_counter() - render_started
        encode_started = time.perf_counter()
        render_png = pixmap.tobytes("png")
        encode_seconds = time.perf_counter() - encode_started
        render_sha = hashlib.sha256(render_png).hexdigest()
        semantic = {
            "source_sha256": item["source_sha256"],
            "page_index": page_index,
            "source_evidence": source_evidence,
            "render_sha256": render_sha,
            "dpi": dpi,
            "colorspace": "RGB",
            "alpha": False,
        }
        return {
            **semantic,
            "semantic_sha256": semantic_sha256(semantic),
            "render_png": render_png,
            "render_bytes": len(render_png),
            "displaylist_reused": True,
            "textpage_from_displaylist": textpage_from_displaylist,
            "textpage_fallback_reason": textpage_fallback_reason,
            "worker_pid": os.getpid(),
            "source_path": str(item["source_path"]),
            "timings": {
                "document open": opened_seconds,
                "DisplayList creation": display_seconds,
                "native text extraction": text_seconds,
                "page render": render_seconds,
                "page encode": encode_seconds,
            },
        }
    except Exception as exc:  # noqa: BLE001 - process boundary needs serializable error
        return {
            "page_index": page_index,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _page_source_process_group(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    import fitz

    source = Path(str(item["source_path"]))
    indices = [int(index) for index in item["page_indices"]]
    try:
        opened_at = time.perf_counter()
        with fitz.open(source) as document:
            opened_seconds = time.perf_counter() - opened_at
            return [
                _page_source_row(
                    document,
                    item,
                    page_index,
                    opened_seconds=opened_seconds if ordinal == 0 else 0.0,
                )
                for ordinal, page_index in enumerate(indices)
            ]
    except Exception as exc:  # noqa: BLE001 - process boundary needs serializable error
        return [
            {
                "page_index": page_index,
                "error": f"{type(exc).__name__}: {exc}",
            }
            for page_index in indices
        ]


class PageSourceWorker:
    """Process-isolated PyMuPDF source worker with one DisplayList per page."""

    def __init__(self, *, processes: int = 1, dpi: int = 200):
        if processes <= 0:
            raise ValueError("SOURCE_WORKER_PROCESS_COUNT_INVALID")
        if dpi != 200:
            raise ValueError("PRODUCTION_RENDER_DPI_MUST_REMAIN_200")
        self.processes = processes
        self.dpi = dpi

    def process(self, source: str | Path, page_indices: Sequence[int]) -> list[dict[str, Any]]:
        rows = self.process_many(
            [{"source_path": str(Path(source).resolve()), "page_indices": list(page_indices)}]
        )
        return sorted(rows, key=lambda row: int(row["page_index"]))

    def process_many(
        self, documents: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        all_items: list[dict[str, Any]] = []
        expected: list[tuple[str, int]] = []
        for request in documents:
            source = Path(str(request["source_path"])).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            source_sha = _sha256_file(source)
            indices = [int(index) for index in request["page_indices"]]
            if len(indices) != len(set(indices)):
                raise ValueError("DUPLICATE_SOURCE_PAGE_WORK")
            ordered_indices = sorted(indices)
            expected.extend((str(source), index) for index in ordered_indices)
            worker_count = min(self.processes, max(1, len(ordered_indices)))
            chunk_size = max(1, math.ceil(len(ordered_indices) / worker_count))
            all_items.extend(
                {
                    "source_path": str(source),
                    "source_sha256": source_sha,
                    "page_indices": ordered_indices[start : start + chunk_size],
                    "dpi": self.dpi,
                }
                for start in range(0, len(ordered_indices), chunk_size)
            )

        rows: list[dict[str, Any]] = []
        if self.processes == 1 or len(all_items) <= 1:
            for item in all_items:
                rows.extend(_page_source_process_group(item))
        else:
            with ProcessPoolExecutor(
                max_workers=min(self.processes, len(all_items)),
                mp_context=get_context("spawn"),
            ) as executor:
                futures = {
                    executor.submit(_page_source_process_group, item): item
                    for item in all_items
                }
                for future in as_completed(futures):
                    try:
                        rows.extend(future.result())
                    except Exception as exc:  # noqa: BLE001
                        item = futures[future]
                        rows.extend(
                            {
                                "source_path": item["source_path"],
                                "page_index": page_index,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            for page_index in item["page_indices"]
                        )
        failures = [row for row in rows if row.get("error")]
        if failures:
            details = "; ".join(
                f"source={row.get('source_path')} page={row['page_index']} {row['error']}"
                for row in sorted(
                    failures,
                    key=lambda row: (str(row.get("source_path")), row["page_index"]),
                )
            )
            raise RuntimeError(f"PAGE_SOURCE_WORK_FAILED:{details}")
        ordered = sorted(
            rows, key=lambda row: (str(row["source_path"]), int(row["page_index"]))
        )
        observed = [(str(row["source_path"]), int(row["page_index"])) for row in ordered]
        if observed != sorted(expected):
            raise RuntimeError("PAGE_SOURCE_COVERAGE_INVALID")
        return ordered
