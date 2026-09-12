from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import io
import json
import os
import re
import struct
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from lxml import etree

from .formula import FormulaConversion, convert_mathml
from .validator import (
    FormulaIssue,
    FormulaStructuralValidator,
    FormulaValidation,
    FormulaVerdict,
)

MTEF_CACHE_SCHEMA_VERSION = 1
MTEF_CONVERSION_CONTRACT = "bemarkdown-mtef-conversion-v1"
MTEF_VALIDATOR_CONTRACT = "bemarkdown-formula-structural-validator-v1"
MATHTYPEJX_REVISION = "7d90e7274c85cf56ac28d4d15e593044693d7e70"
MTEF_COMPONENT = "mathtypejx 0.1.0 + mathml2latex 0.2.12"
_OLE_STREAM_NAMES = ("Equation Native", "EquationNative", "Equation")
_OLE_HEADER = struct.Struct("<H I H I I I I I")
_CACHE_MODES = {"off", "memory", "persistent"}


def _dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _default_contract() -> dict[str, str | int]:
    return {
        "schema_version": MTEF_CACHE_SCHEMA_VERSION,
        "mathtypejx_revision": MATHTYPEJX_REVISION,
        "mathml2latex_version": "0.2.12",
        "beautifulsoup4_version": _dependency_version("beautifulsoup4"),
        "lxml_version": _dependency_version("lxml"),
        "conversion_contract": MTEF_CONVERSION_CONTRACT,
        "validator_contract": MTEF_VALIDATOR_CONTRACT,
    }


@dataclass(frozen=True)
class MtefCacheResolution:
    conversion: FormulaConversion
    validation: FormulaValidation | None
    mtef_sha256: str | None
    cache_key: str | None
    cache_level: str
    cacheable: bool
    timings: dict[str, float]
    l2_write: bool = False
    l2_write_failure: bool = False
    l2_corruption: bool = False
    warnings: tuple[str, ...] = ()


@dataclass
class _ProfiledConversion:
    conversion: FormulaConversion
    timings: dict[str, float] = field(default_factory=dict)
    cacheable: bool = True


@dataclass
class _OleExtraction:
    payload: bytes | None
    semantic_state: str
    timings: dict[str, float]
    error: str | None = None


class MtefCacheContext:
    """Single-thread, caller-owned content-addressed MTEF conversion cache."""

    def __init__(
        self,
        *,
        mode: str = "memory",
        persistent_dir: str | Path | None = None,
        memory_store: dict[str, dict[str, Any]] | None = None,
        contract_overrides: dict[str, str | int] | None = None,
        converter: Callable[[bytes], FormulaConversion | _ProfiledConversion]
        | None = None,
        validator: FormulaStructuralValidator | None = None,
        max_ole_bytes: int = 64 * 1024 * 1024,
        max_mtef_payload_bytes: int = 16 * 1024 * 1024,
    ):
        if mode not in _CACHE_MODES:
            raise ValueError("mtef cache mode must be 'off', 'memory', or 'persistent'")
        self.mode = mode
        self.persistent_dir = Path(
            persistent_dir
            if persistent_dir is not None
            else Path.home() / ".cache" / "bemarkdown" / "mtef"
        )
        self.memory_store = memory_store if memory_store is not None else {}
        self.contract = _default_contract()
        self.contract.update(contract_overrides or {})
        contract_json = json.dumps(
            self.contract, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        self.contract_fingerprint = hashlib.sha256(
            contract_json.encode("utf-8")
        ).hexdigest()
        self.converter = converter or _convert_payload_profiled
        self.validator = validator or FormulaStructuralValidator()
        self.max_ole_bytes = max_ole_bytes
        self.max_mtef_payload_bytes = max_mtef_payload_bytes
        self.warnings: list[str] = []
        self._counts: Counter[str] = Counter()
        self._payload_occurrences: Counter[str] = Counter()
        self._samples: dict[str, list[float]] = {
            "uncached_conversion_seconds": [],
            "l1_hit_seconds": [],
            "l2_hit_seconds": [],
            "cache_lookup_seconds": [],
            "l2_read_seconds": [],
            "l2_write_seconds": [],
        }
        self._stage_totals: Counter[str] = Counter()

    def convert_ole(
        self, ole_data: bytes, source_metadata: dict[str, Any]
    ) -> MtefCacheResolution:
        operation_started = time.perf_counter()
        if len(ole_data) > self.max_ole_bytes:
            return self._resource_limit_resolution(
                f"OLE input byte limit exceeded: {len(ole_data)} > {self.max_ole_bytes}",
                operation_started,
            )
        extraction = _extract_mtef_payload(ole_data)
        timings = dict(extraction.timings)
        if extraction.payload is None:
            conversion = FormulaConversion(
                None,
                "FAILED_PRESERVED",
                MTEF_COMPONENT,
                [],
                "mathtypejx returned no MathML",
                semantic_state=extraction.semantic_state,
            )
            timings["total_seconds"] = time.perf_counter() - operation_started
            result = MtefCacheResolution(
                conversion,
                None,
                None,
                None,
                "miss",
                False,
                timings,
            )
            self._record(result)
            return result

        if len(extraction.payload) > self.max_mtef_payload_bytes:
            return self._resource_limit_resolution(
                "MTEF payload byte limit exceeded: "
                f"{len(extraction.payload)} > {self.max_mtef_payload_bytes}",
                operation_started,
                timings,
            )

        sha_started = time.perf_counter()
        mtef_sha = hashlib.sha256(extraction.payload).hexdigest()
        timings["mtef_sha_seconds"] = time.perf_counter() - sha_started
        cache_key = self._cache_key(mtef_sha)
        self._payload_occurrences[mtef_sha] += 1

        lookup_started = time.perf_counter()
        cached = self.memory_store.get(cache_key) if self.mode != "off" else None
        timings["cache_lookup_seconds"] = time.perf_counter() - lookup_started
        if cached is not None:
            result = self._resolution_from_entry(
                cached,
                source_metadata,
                mtef_sha,
                cache_key,
                "l1",
                timings,
                operation_started,
            )
            self._record(result)
            return result

        corruption = False
        operation_warnings: list[str] = []
        if self.mode == "persistent":
            read_started = time.perf_counter()
            cached, warning = self._read_l2(mtef_sha, cache_key)
            timings["l2_read_seconds"] = time.perf_counter() - read_started
            if warning:
                corruption = True
                operation_warnings.append(warning)
                self.warnings.append(warning)
            if cached is not None:
                self.memory_store[cache_key] = cached
                result = self._resolution_from_entry(
                    cached,
                    source_metadata,
                    mtef_sha,
                    cache_key,
                    "l2",
                    timings,
                    operation_started,
                    l2_corruption=corruption,
                    warnings=tuple(operation_warnings),
                )
                self._record(result)
                return result

        conversion_started = time.perf_counter()
        profiled = self._run_converter(extraction.payload)
        timings.update(profiled.timings)
        validation_started = time.perf_counter()
        validation = self._validate(profiled.conversion, source_metadata)
        timings["formula_validation_seconds"] = (
            time.perf_counter() - validation_started
        )
        timings["uncached_conversion_seconds"] = (
            time.perf_counter() - conversion_started
        )

        entry = self._entry(
            mtef_sha,
            cache_key,
            profiled.conversion,
            validation,
            cacheable=profiled.cacheable,
        )
        wrote = False
        write_failure = False
        if profiled.cacheable and self.mode != "off":
            self.memory_store[cache_key] = entry
            if self.mode == "persistent":
                write_started = time.perf_counter()
                try:
                    self._write_l2(mtef_sha, entry)
                    wrote = True
                except OSError as exc:
                    write_failure = True
                    warning = (
                        f"Could not write MTEF cache entry {self.entry_path(mtef_sha)}: "
                        f"{exc}"
                    )
                    operation_warnings.append(warning)
                    self.warnings.append(warning)
                finally:
                    timings["l2_write_seconds"] = (
                        time.perf_counter() - write_started
                    )

        timings["total_seconds"] = time.perf_counter() - operation_started
        result = MtefCacheResolution(
            profiled.conversion,
            validation,
            mtef_sha,
            cache_key,
            "miss",
            profiled.cacheable,
            timings,
            l2_write=wrote,
            l2_write_failure=write_failure,
            l2_corruption=corruption,
            warnings=tuple(operation_warnings),
        )
        self._record(result)
        return result

    def _resource_limit_resolution(
        self,
        error: str,
        operation_started: float,
        timings: dict[str, float] | None = None,
    ) -> MtefCacheResolution:
        conversion = FormulaConversion(
            None,
            "FAILED_PRESERVED",
            MTEF_COMPONENT,
            [],
            error,
            semantic_state="unresolved",
        )
        resolved_timings = dict(timings or {})
        resolved_timings["total_seconds"] = time.perf_counter() - operation_started
        result = MtefCacheResolution(
            conversion,
            None,
            None,
            None,
            "miss",
            False,
            resolved_timings,
            warnings=(error,),
        )
        self.warnings.append(error)
        self._record(result)
        return result

    def semantic_sha256(self, ole_data: bytes) -> str:
        extraction = _extract_mtef_payload(ole_data)
        if extraction.payload is None:
            raise ValueError(extraction.error or "OLE has no usable Equation Native payload")
        return hashlib.sha256(extraction.payload).hexdigest()

    def entry_path(self, mtef_sha256: str) -> Path:
        return (
            self.persistent_dir
            / self.contract_fingerprint[:16]
            / mtef_sha256[:2]
            / f"{mtef_sha256}.json"
        )

    def snapshot(self) -> dict[str, Any]:
        occurrences = sum(self._payload_occurrences.values())
        unique = len(self._payload_occurrences)
        result: dict[str, Any] = {
            "schema": "bemarkdown-mtef-cache-stats-v1",
            "mode": self.mode,
            "worker_count": 1,
            "contract": dict(self.contract),
            "contract_fingerprint": self.contract_fingerprint,
            "occurrences": occurrences,
            "unique_mtef_sha256": unique,
            "duplicates": max(0, occurrences - unique),
            "full_conversion_calls": self._counts["full_conversion_calls"],
            "l1_hits": self._counts["l1_hits"],
            "l1_misses": self._counts["l1_misses"],
            "l2_hits": self._counts["l2_hits"],
            "l2_misses": self._counts["l2_misses"],
            "l2_writes": self._counts["l2_writes"],
            "l2_corruptions": self._counts["l2_corruptions"],
            "l2_write_failures": self._counts["l2_write_failures"],
            "non_cacheable": self._counts["non_cacheable"],
            "warnings": list(self.warnings),
            "stage_seconds": {
                key: round(value, 9) for key, value in sorted(self._stage_totals.items())
            },
            "top_repeated_payloads": [
                {"sha256_prefix": sha[:16], "occurrences": count}
                for sha, count in self._payload_occurrences.most_common(20)
                if count > 1
            ],
        }
        for name, values in self._samples.items():
            result[name.removesuffix("_seconds")] = _distribution(values)
        if self.mode == "persistent":
            files = list(self._contract_cache_files())
            sizes = [path.stat().st_size for path in files]
            result["storage"] = {
                "directory": str(self.persistent_dir.resolve()),
                "entry_count": len(files),
                "bytes": sum(sizes),
                "average_entry_bytes": round(mean(sizes), 3) if sizes else 0.0,
            }
        return result

    def clear_persistent(self) -> int:
        """Delete only valid JSON entries for this contract; leave other data alone."""

        removed = 0
        for path in list(self._contract_cache_files()):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def _cache_key(self, mtef_sha: str) -> str:
        value = f"{self.contract_fingerprint}:{mtef_sha}"
        return hashlib.sha256(value.encode("ascii")).hexdigest()

    def _run_converter(self, payload: bytes) -> _ProfiledConversion:
        try:
            converted = self.converter(payload)
            if isinstance(converted, _ProfiledConversion):
                return converted
            if not isinstance(converted, FormulaConversion):
                raise TypeError("MTEF converter returned an unsupported result")
            return _ProfiledConversion(converted)
        except Exception as exc:  # noqa: BLE001 - environment failures remain misses
            return _ProfiledConversion(
                FormulaConversion(
                    None,
                    "FAILED_PRESERVED",
                    MTEF_COMPONENT,
                    [],
                    str(exc),
                    semantic_state="unknown",
                ),
                cacheable=False,
            )

    def _validate(
        self, conversion: FormulaConversion, source_metadata: dict[str, Any]
    ) -> FormulaValidation | None:
        if conversion.latex is None and conversion.intermediate is None:
            return None
        return self.validator.validate(conversion, source_metadata=source_metadata)

    def _entry(
        self,
        mtef_sha: str,
        cache_key: str,
        conversion: FormulaConversion,
        validation: FormulaValidation | None,
        *,
        cacheable: bool,
    ) -> dict[str, Any]:
        validation_dict = validation.to_dict() if validation else None
        validator_evidence = None
        if validation_dict:
            validator_evidence = copy.deepcopy(validation_dict["evidence"])
            validator_evidence.pop("source_metadata", None)
        return {
            "schema_version": MTEF_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "mtef_sha256": mtef_sha,
            "contract_fingerprint": self.contract_fingerprint,
            "mathml": conversion.intermediate,
            "latex": conversion.latex,
            "conversion_status": conversion.status,
            "conversion_component": conversion.component,
            "conversion_warnings": list(conversion.warnings),
            "conversion_error": conversion.error,
            "semantic_state": conversion.semantic_state,
            "validator_verdict": validation_dict["verdict"] if validation_dict else None,
            "validator_issues": validation_dict["issues"] if validation_dict else [],
            "validator_evidence": validator_evidence,
            "created_with": dict(self.contract),
            "cacheable": cacheable,
        }

    def _resolution_from_entry(
        self,
        entry: dict[str, Any],
        source_metadata: dict[str, Any],
        mtef_sha: str,
        cache_key: str,
        level: str,
        timings: dict[str, float],
        operation_started: float,
        *,
        l2_corruption: bool = False,
        warnings: tuple[str, ...] = (),
    ) -> MtefCacheResolution:
        conversion = FormulaConversion(
            entry["latex"],
            entry["conversion_status"],
            entry["conversion_component"],
            list(entry["conversion_warnings"]),
            entry["conversion_error"],
            entry["mathml"],
            entry["semantic_state"],
        )
        validation = None
        if entry["validator_verdict"] is not None:
            evidence = copy.deepcopy(entry["validator_evidence"])
            evidence["source_metadata"] = dict(source_metadata)
            validation = FormulaValidation(
                FormulaVerdict(entry["validator_verdict"]),
                tuple(FormulaIssue(**issue) for issue in entry["validator_issues"]),
                evidence,
            )
        timings[f"{level}_hit_seconds"] = time.perf_counter() - operation_started
        timings["total_seconds"] = time.perf_counter() - operation_started
        return MtefCacheResolution(
            conversion,
            validation,
            mtef_sha,
            cache_key,
            level,
            True,
            timings,
            l2_corruption=l2_corruption,
            warnings=warnings,
        )

    def _read_l2(
        self, mtef_sha: str, cache_key: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        path = self.entry_path(mtef_sha)
        if not path.is_file():
            return None, None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            required = {
                "schema_version",
                "cache_key",
                "mtef_sha256",
                "contract_fingerprint",
                "mathml",
                "latex",
                "conversion_status",
                "conversion_component",
                "conversion_warnings",
                "conversion_error",
                "semantic_state",
                "validator_verdict",
                "validator_issues",
                "validator_evidence",
                "created_with",
                "cacheable",
            }
            if not isinstance(entry, dict) or not required.issubset(entry):
                raise ValueError("missing required fields")
            if entry["schema_version"] != MTEF_CACHE_SCHEMA_VERSION:
                raise ValueError("wrong schema version")
            if entry["cache_key"] != cache_key:
                raise ValueError("wrong cache key")
            if entry["mtef_sha256"] != mtef_sha:
                raise ValueError("wrong MTEF SHA-256")
            if entry["contract_fingerprint"] != self.contract_fingerprint:
                raise ValueError("wrong contract fingerprint")
            if entry["created_with"] != self.contract:
                raise ValueError("wrong dependency contract")
            if entry["cacheable"] is not True:
                raise ValueError("entry is marked non-cacheable")
            _validate_entry_shape(entry)
            return entry, None
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return None, f"Ignoring corrupt MTEF cache entry {path}: {exc}"

    def _write_l2(self, mtef_sha: str, entry: dict[str, Any]) -> None:
        path = self.entry_path(mtef_sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(
                    entry,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _contract_cache_files(self):
        root = self.persistent_dir / self.contract_fingerprint[:16]
        if not root.is_dir():
            return iter(())
        return root.glob("*/*.json")

    def _record(self, result: MtefCacheResolution) -> None:
        if result.cache_level == "l1":
            self._counts["l1_hits"] += 1
        else:
            if self.mode != "off":
                self._counts["l1_misses"] += 1
            if result.cache_level == "l2":
                self._counts["l2_hits"] += 1
            else:
                if self.mode == "persistent":
                    self._counts["l2_misses"] += 1
                self._counts["full_conversion_calls"] += 1
        if result.l2_write:
            self._counts["l2_writes"] += 1
        if result.l2_write_failure:
            self._counts["l2_write_failures"] += 1
        if result.l2_corruption:
            self._counts["l2_corruptions"] += 1
        if not result.cacheable:
            self._counts["non_cacheable"] += 1
        for key, value in result.timings.items():
            self._stage_totals[key] += value
            if key in self._samples:
                self._samples[key].append(value)


def _extract_mtef_payload(ole_data: bytes) -> _OleExtraction:
    timings: dict[str, float] = {}
    open_started = time.perf_counter()
    try:
        import olefile

        ole = olefile.OleFileIO(io.BytesIO(ole_data))
    except Exception as exc:  # noqa: BLE001 - malformed OLE is input data
        timings["ole_open_seconds"] = time.perf_counter() - open_started
        return _OleExtraction(None, "unknown", timings, str(exc))
    timings["ole_open_seconds"] = time.perf_counter() - open_started

    native = None
    try:
        lookup_seconds = 0.0
        read_seconds = 0.0
        for stream_name in _OLE_STREAM_NAMES:
            lookup_started = time.perf_counter()
            exists = ole.exists(stream_name)
            lookup_seconds += time.perf_counter() - lookup_started
            if not exists:
                continue
            read_started = time.perf_counter()
            try:
                native = ole.openstream(stream_name).read()
                read_seconds += time.perf_counter() - read_started
                break
            except Exception:  # noqa: BLE001 - try the pinned aliases in order
                read_seconds += time.perf_counter() - read_started
                continue
        timings["equation_stream_lookup_seconds"] = lookup_seconds
        timings["equation_stream_read_seconds"] = read_seconds
        if native is None:
            return _OleExtraction(None, "unknown", timings, "Equation Native stream missing")
    except Exception as exc:  # noqa: BLE001 - malformed OLE is input data
        return _OleExtraction(None, "corrupted", timings, str(exc))
    finally:
        ole.close()

    payload_started = time.perf_counter()
    payload = native
    try:
        if len(native) < _OLE_HEADER.size:
            raise ValueError("Equation Native stream is shorter than its header")
        cb_hdr, _, _, cb_object, *_ = _OLE_HEADER.unpack_from(native, 0)
        if cb_hdr == _OLE_HEADER.size:
            payload = native[cb_hdr:]
            if 0 < cb_object <= len(payload):
                payload = payload[:cb_object]
    except (struct.error, ValueError) as exc:
        timings["mtef_payload_extraction_seconds"] = (
            time.perf_counter() - payload_started
        )
        return _OleExtraction(None, "corrupted", timings, str(exc))
    timings["mtef_payload_extraction_seconds"] = time.perf_counter() - payload_started
    if len(payload) < 2 or payload[0] not in {3, 5}:
        return _OleExtraction(None, "corrupted", timings, "invalid MTEF payload")
    return _OleExtraction(payload, "present", timings)


def _convert_payload_profiled(payload: bytes) -> _ProfiledConversion:
    timings: dict[str, float] = {}
    parse_started = time.perf_counter()
    equation = _parse_payload(payload)
    timings["mtef_parse_seconds"] = time.perf_counter() - parse_started
    if equation is None:
        return _ProfiledConversion(
            FormulaConversion(
                None,
                "FAILED_PRESERVED",
                MTEF_COMPONENT,
                [],
                "mathtypejx returned no MathML",
                semantic_state="corrupted",
            ),
            timings,
        )

    mathml_started = time.perf_counter()
    mathml = _build_mathml(equation)
    timings["mtef_to_mathml_seconds"] = time.perf_counter() - mathml_started
    if not mathml:
        return _ProfiledConversion(
            FormulaConversion(
                None,
                "FAILED_PRESERVED",
                MTEF_COMPONENT,
                [],
                "mathtypejx returned no MathML",
                semantic_state="corrupted",
            ),
            timings,
        )

    latex_started = time.perf_counter()
    conversion = convert_mathml(mathml, component=MTEF_COMPONENT)
    timings["mathml_to_latex_seconds"] = time.perf_counter() - latex_started
    return _ProfiledConversion(conversion, timings)


def _parse_payload(payload: bytes) -> dict[str, Any] | None:
    from mathtypejx.mtef.mathml import _skip_stream_header
    from mathtypejx.mtef.stream import ByteStream

    version = payload[0]
    stream = ByteStream(payload)
    _skip_stream_header(stream, version)
    try:
        if version == 5:
            from mathtypejx.mtef.records5 import parse_equation

            equation = parse_equation(stream)
        else:
            from mathtypejx.mtef.records3 import parse_equation_v3

            equation = parse_equation_v3(stream)
    except Exception:  # noqa: BLE001 - mirror pinned mathtypejx fallback
        try:
            stream = ByteStream(payload)
            _skip_stream_header(stream, 5 if version == 3 else 3)
            if version == 5:
                from mathtypejx.mtef.records3 import parse_equation_v3

                equation = parse_equation_v3(stream)
            else:
                from mathtypejx.mtef.records5 import parse_equation

                equation = parse_equation(stream)
        except Exception:  # noqa: BLE001 - deterministic malformed payload
            return None
    if equation is not None:
        equation.setdefault("mtef_version", version)
    return equation


def _build_mathml(equation: dict[str, Any]) -> str | None:
    from mathtypejx.mtef.builder import build_mtef_xml
    from mathtypejx.mtef.chars import replace as chars_replace
    from mathtypejx.mtef.mathml import _xslt_transform
    from mathtypejx.mtef.mover import move as mover_move

    xml_root = build_mtef_xml(equation)
    mover_move(xml_root)
    chars_replace(xml_root)
    mathml = _xslt_transform(xml_root)
    if mathml is None:
        return None
    try:
        mathml_ns = "http://www.w3.org/1998/Math/MathML"
        cleaned = re.sub(r"<\?xml[^?]*\?>", "", mathml).strip()
        out_root = etree.fromstring(cleaned.encode("utf-8"))
        for element in out_root.iter():
            tag = element.tag
            if isinstance(tag, str) and "}" in tag:
                element.tag = tag.split("}", 1)[1]
            for attribute in list(element.attrib):
                if attribute == "xmlns":
                    value = element.attrib[attribute]
                    if value == "" or (element is not out_root and value == mathml_ns):
                        del element.attrib[attribute]
                elif attribute.startswith("xmlns:"):
                    del element.attrib[attribute]
        if "xmlns" not in out_root.attrib:
            out_root.attrib["xmlns"] = mathml_ns
        result = etree.tostring(out_root, encoding="unicode", pretty_print=False)
        if "xmlns=" not in result:
            result = result.replace('<math ', f'<math xmlns="{mathml_ns}" ', 1)
            result = result.replace("<math>", f'<math xmlns="{mathml_ns}">', 1)
        return result
    except Exception:  # noqa: BLE001 - mirror pinned mathtypejx fallback
        return mathml


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_seconds": None, "median_seconds": None, "p95_seconds": None, "max_seconds": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    return {
        "count": len(ordered),
        "mean_seconds": round(mean(ordered), 9),
        "median_seconds": round(median(ordered), 9),
        "p95_seconds": round(ordered[p95_index], 9),
        "max_seconds": round(ordered[-1], 9),
    }


def _validate_entry_shape(entry: dict[str, Any]) -> None:
    for key in ("mathml", "latex", "conversion_error", "validator_verdict"):
        if entry[key] is not None and not isinstance(entry[key], str):
            raise TypeError(f"{key} has the wrong type")
    for key in (
        "cache_key",
        "mtef_sha256",
        "contract_fingerprint",
        "conversion_status",
        "conversion_component",
        "semantic_state",
    ):
        if not isinstance(entry[key], str):
            raise TypeError(f"{key} has the wrong type")
    if not isinstance(entry["conversion_warnings"], list) or not all(
        isinstance(value, str) for value in entry["conversion_warnings"]
    ):
        raise TypeError("conversion_warnings has the wrong type")
    if not isinstance(entry["validator_issues"], list) or not all(
        isinstance(issue, dict)
        and {"code", "message", "layer"}.issubset(issue)
        and all(
            issue.get(key) is None or isinstance(issue.get(key), str)
            for key in ("code", "message", "layer", "location")
        )
        for issue in entry["validator_issues"]
    ):
        raise TypeError("validator_issues has the wrong type")
    verdict = entry["validator_verdict"]
    if verdict is not None:
        FormulaVerdict(verdict)
        if not isinstance(entry["validator_evidence"], dict):
            raise TypeError("validator_evidence has the wrong type")
    elif entry["validator_evidence"] is not None:
        raise TypeError("validator_evidence must be null without a verdict")
    if not isinstance(entry["created_with"], dict):
        raise TypeError("created_with has the wrong type")
