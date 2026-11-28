"""Versioned, model-independent loopback HTTP contract for visual transcription.

Only image bytes cross the boundary. No server filesystem paths or Python
objects are part of the protocol. Inference is never retried automatically.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import time
import uuid
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .visual_serial_pipeline import LOCAL_API_POLICY, VisualSerialOrchestrator, VisualSerialPolicy

SCHEMA = LOCAL_API_POLICY
ROLES = ("layout", "formula", "table", "text")
LEGACY_ROLES = frozenset({"layout", "formula", "text"})
MAX_BYTES = 64 * 1024 * 1024
MAX_ITEMS = 128
SESSION_SCHEMA = "bemarkdown-model-session-v2"


def request_batches(items, *, max_items=MAX_ITEMS, max_bytes=MAX_BYTES, max_pixels=None):
    """Bound transport batches independently of backend GPU minibatch size."""
    if type(max_items) is not int or not 1 <= max_items <= MAX_ITEMS:
        raise ModelApiError("MODEL_API_BATCH_ITEMS_INVALID")
    if type(max_bytes) is not int or not 1024 <= max_bytes <= MAX_BYTES:
        raise ModelApiError("MODEL_API_BATCH_BYTES_INVALID")
    if max_pixels is not None and (type(max_pixels) is not int or max_pixels <= 0):
        raise ModelApiError("MODEL_API_BATCH_PIXELS_INVALID")
    batch, byte_count, pixel_count = [], 512, 0
    for item in items:
        item_bytes = len(json.dumps(item, ensure_ascii=False).encode("utf-8")) + 2
        pixels = 0
        if max_pixels is not None:
            from PIL import Image
            with Image.open(BytesIO(base64.b64decode(item["image_base64"], validate=True))) as image:
                pixels = image.width * image.height
        if item_bytes + 512 > max_bytes or (max_pixels is not None and pixels > max_pixels):
            raise ModelApiError("MODEL_API_SINGLE_ITEM_EXCEEDS_BATCH_LIMIT")
        if batch and (len(batch) >= max_items or byte_count + item_bytes > max_bytes
                      or (max_pixels is not None and pixel_count + pixels > max_pixels)):
            yield batch
            batch, byte_count, pixel_count = [], 512, 0
        batch.append(item)
        byte_count += item_bytes
        pixel_count += pixels
    if batch:
        yield batch


class ModelApiError(ValueError):
    pass


class SessionExitPending(ModelApiError):
    """The server retains ownership of a closing worker."""


class ConfirmedSessionTimeout(ModelApiError):
    """Local server confirmed a timed-out request after worker cleanup."""


def api_policy(*, table_enabled=False):
    return VisualSerialPolicy(policy_id=SCHEMA, layout_model_id="api:layout",
                              formula_model_id="api:formula", text_model_id="api:text",
                              prompt_schema=SCHEMA, table_model_id="api:table" if table_enabled else "")


def encoded_image(data):
    return {"image_base64": base64.b64encode(data).decode("ascii"),
            "sha256": hashlib.sha256(data).hexdigest()}


def decoded_image(item):
    try:
        data = base64.b64decode(item["image_base64"], validate=True)
        if not data or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("image hash mismatch")
        from PIL import Image
        with Image.open(BytesIO(data)) as image:
            image.verify()
        return data
    except Exception as exc:
        raise ModelApiError("MODEL_API_INVALID_IMAGE") from exc


def local_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ModelApiError("MODEL_API_LOOPBACK_HTTP_URL_REQUIRED")
    return value


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or frozenset(config.get("roles", {})) not in {frozenset(ROLES), LEGACY_ROLES}:
        raise ModelApiError("MODEL_API_CONFIG_SCHEMA_OR_ROLES_INVALID")
    for role in config["roles"]:
        spec = config["roles"][role]
        local_url(spec["url"])
        timeout = spec.get("timeout_seconds", 600)
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ModelApiError("MODEL_API_TIMEOUT_INVALID")
        env = spec.get("token_env")
        if env and not os.environ.get(env):
            raise ModelApiError(f"MODEL_API_TOKEN_ENV_MISSING:{env}")
        if spec.get("session_mode", "auto") not in {"auto", "off", "required"}:
            raise ModelApiError("MODEL_API_SESSION_MODE_INVALID")
    return config


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelApiError("MODEL_API_REDIRECT_FORBIDDEN")


class LocalModelApi:
    def __init__(self, role, spec):
        # A runtime is one immutable service binding. Keep a public compatibility
        # view, but never let mutation change an existing session or cache scope.
        self.role, self.spec = role, copy.deepcopy(dict(spec))
        self._role_snapshot = role
        self._config_snapshot = copy.deepcopy(self.spec)
        token_env = self._config_snapshot.get("token_env")
        self._token = os.environ.get(token_env) if token_env else None
        self.identity = {"role": role, "endpoint": local_url(spec["url"]), "invoked": False}
        self.metrics = {"requests": []}
        self.capability = None
        self.discovery_done = False
        self.session_id = None
        self.stage_id = None
        self.stage_requests = 0
        self.stage_items = 0
        suffix = f"/v1/{role}"
        self.base_url = spec["url"][:-len(suffix)] if spec["url"].endswith(suffix) else None

    def _assert_config_fixed(self):
        if self.role != self._role_snapshot or self.spec != self._config_snapshot:
            raise ModelApiError("MODEL_API_CONFIG_CHANGED_DURING_DOCUMENT")

    def _http(self, url, payload=None, *, missing_ok=False):
        headers = {"Content-Type": "application/json"}
        token_env = self._config_snapshot.get("token_env")
        if token_env:
            if self._token is None:
                raise ModelApiError(f"MODEL_API_TOKEN_ENV_MISSING:{token_env}")
            headers["Authorization"] = "Bearer " + self._token
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        if body is not None and len(body) > MAX_BYTES:
            raise ModelApiError("MODEL_API_REQUEST_TOO_LARGE")
        try:
            with build_opener(ProxyHandler({}), _NoRedirect()).open(
                    Request(local_url(url), body, headers),
                    timeout=self._config_snapshot.get("timeout_seconds", 600)) as response:
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ModelApiError("MODEL_API_RESPONSE_TOO_LARGE")
            return json.loads(data)
        except HTTPError as exc:
            if missing_ok and exc.code in {404, 405, 501}:
                return None
            detail = exc.read(8192).decode("utf-8", errors="replace")
            if exc.code == 500 and self.base_url and url == self.base_url + "/v2/session/close":
                try:
                    pending = json.loads(detail) == {"error": "MODEL_SESSION_EXIT_PENDING"}
                except ValueError:
                    pending = False
                if pending:
                    raise SessionExitPending("MODEL_SESSION_EXIT_PENDING") from exc
            if exc.code == 500 and self.base_url and url == self.base_url + "/v2/session/infer" and self.role == "text" and (
                    self.capability or {}).get("confirmed_timeout_split") is True:
                try:
                    error = json.loads(detail)
                except ValueError:
                    error = None
                if error == {"error":"MODEL_SESSION_REQUEST_TIMEOUT"}:
                    raise ConfirmedSessionTimeout("MODEL_SESSION_REQUEST_TIMEOUT") from exc
            raise ModelApiError(f"MODEL_API_HTTP_ERROR:{self.role}:{exc.code}:{detail}") from exc

    def _progress(self, state, request_id=None, model_unloaded=False):
        from ..model_session import record_progress
        record_progress({"role": self.role, "stage_id": self.stage_id, "session_id": self.session_id,
                         "state": state, "request_id": request_id, "completed_requests": self.stage_requests,
                         "completed_items": self.stage_items, "model_unloaded": model_unloaded})

    def metrics_checkpoint(self):
        cache = self.metrics.get("text_result_cache", {})
        return (len(self.metrics["requests"]), len(self.metrics.get("sessions", [])), dict(cache))

    def metrics_since(self, checkpoint):
        requests, sessions, previous_cache = checkpoint
        metrics = {**self.metrics, "requests": self.metrics["requests"][requests:]}
        if "sessions" in metrics:
            metrics["sessions"] = metrics["sessions"][sessions:]
        if "text_result_cache" in metrics:
            cache = dict(metrics["text_result_cache"])
            for key in ("occurrences", "model_generations", "hits", "evictions",
                        "identity_bypass_occurrences", "empty_results_not_cached"):
                cache[key] -= previous_cache.get(key, 0)
            metrics["text_result_cache"] = cache
        identity = {**self.identity, "invoked": bool(metrics["requests"])}
        return identity, metrics

    def discover(self):
        self._assert_config_fixed()
        if self.discovery_done:
            return
        mode = self._config_snapshot.get("session_mode", "auto")
        if mode == "off" or self.base_url is None:
            if mode == "required":
                raise ModelApiError("MODEL_API_SESSION_ENDPOINT_REQUIRED")
            self.discovery_done = True
            return
        value = self._http(self.base_url + "/v1/capabilities", missing_ok=True)
        if value is not None:
            if (not isinstance(value, dict) or value.get("schema") != "bemarkdown-model-api-capabilities-v1"
                    or not isinstance(value.get("roles"), list) or self.role not in value["roles"]):
                raise ModelApiError("MODEL_API_CAPABILITIES_INVALID")
            session_roles = value.get("session_roles", value["roles"])
            if not isinstance(session_roles, list) or any(role not in value["roles"] for role in session_roles):
                raise ModelApiError("MODEL_API_CAPABILITIES_SESSION_ROLES_INVALID")
            if value.get("session_schema") == SESSION_SCHEMA and self.role in session_roles:
                for key, minimum, maximum in (("max_items", 1, MAX_ITEMS), ("max_bytes", 1024, MAX_BYTES)):
                    if type(value.get(key)) is not int or not minimum <= value[key] <= maximum:
                        raise ModelApiError("MODEL_API_CAPABILITIES_LIMIT_INVALID")
                self.capability = value
        if mode == "required" and self.capability is None:
            raise ModelApiError("MODEL_API_SESSION_UNSUPPORTED")
        self.discovery_done = True

    def _session_call(self, action, payload=None):
        return self._http(self.base_url + "/v2/session/" + action,
                          {**(payload or {}), "schema": SESSION_SCHEMA, "session_id": self.session_id})

    def _open_session(self):
        if self.session_id is not None:
            return
        self.session_id = uuid.uuid4().hex
        try:
            value = self._session_call("open", {"role": self.role})
            if (value.get("schema") != SESSION_SCHEMA or value.get("session_id") != self.session_id
                    or value.get("role") != self.role or value.get("state") != "OPEN"):
                raise ModelApiError("MODEL_API_SESSION_OPEN_INVALID")
        except BaseException:
            self.close()
            raise

    def batches(self, items):
        self.discover()
        limits = self.capability or {}
        default_count = limits.get("layout_batch_items", 1) if self.role == "layout" else MAX_ITEMS
        return request_batches(items,
            max_items=min(self._config_snapshot.get("batch_items", default_count), limits.get("max_items", MAX_ITEMS)),
            max_bytes=min(self._config_snapshot.get("batch_bytes", MAX_BYTES), limits.get("max_bytes", MAX_BYTES)),
            max_pixels=self._config_snapshot.get("batch_pixels"))

    def request(self, items):
        try:
            return self._request_once(items)
        except ConfirmedSessionTimeout:
            if len(items) < 2:
                raise
            # Previous successful batches are already in the running converter. Only
            # retry this failed batch, once in two smaller parts, on a fresh session.
            self._close_session()
            middle = len(items) // 2
            self.metrics.setdefault("timeout_splits", []).append({"count":len(items),"parts":[middle,len(items)-middle]})
            return self._request_once(items[:middle]) + self._request_once(items[middle:])

    def _request_once(self, items):
        self._assert_config_fixed()
        self.discover()
        if self.stage_id is None:
            self.stage_id = uuid.uuid4().hex
        request_id = uuid.uuid4().hex
        payload = {"schema": SCHEMA, "role": self.role, "request_id": request_id, "items": items}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(body) > MAX_BYTES or not 0 < len(items) <= MAX_ITEMS:
            raise ModelApiError("MODEL_API_REQUEST_TOO_LARGE")
        started = time.perf_counter()
        self._progress("RUNNING", request_id)
        try:
            if self.capability is not None:
                self._open_session()
                try:
                    result = self._session_call("infer", payload)
                except (TimeoutError, ConnectionError, URLError):
                    status = self._session_call("status", {"request_id": request_id})
                    from ..model_session import payload_digest
                    receipt = status.get("receipt", {})
                    if (status.get("session_id") != self.session_id or status.get("request_state") != "COMPLETE"
                            or receipt.get("payload_sha256") != payload_digest(payload)):
                        raise ModelApiError("MODEL_API_REQUEST_COMPLETION_UNKNOWN")
                    result = receipt["result"]
            else:
                result = self._http(self._config_snapshot["url"], payload)
        except ConfirmedSessionTimeout:
            self._progress("REQUEST_FAILED", request_id)
            raise
        except Exception as exc:
            self._progress("REQUEST_FAILED", request_id)
            raise ModelApiError(f"MODEL_API_REQUEST_FAILED:{self.role}:{type(exc).__name__}:{exc}") from exc
        expected_schema = SESSION_SCHEMA if self.session_id else SCHEMA
        released = (result.get("request_complete") is True and result.get("model_unloaded") is False
                    and result.get("released") is False and result.get("session_id") == self.session_id
                    if self.session_id else result.get("released") is True)
        if (result.get("schema") != expected_schema or result.get("role") != self.role
                or result.get("request_id") != request_id or not released):
            raise ModelApiError("MODEL_API_RESPONSE_BINDING_OR_RELEASE_INVALID")
        rows = result.get("items")
        expected = [item["id"] for item in items]
        if (not isinstance(rows, list) or len(rows) != len(expected)
                or {row.get("id") for row in rows} != set(expected)):
            raise ModelApiError("MODEL_API_RESPONSE_ITEM_IDS_INVALID")
        identity = result.get("identity", {})
        if (not identity.get("model_id") or len(identity.get("model_fingerprint", "")) != 64):
            raise ModelApiError("MODEL_API_MODEL_IDENTITY_REQUIRED")
        stable = {k: v for k, v in identity.items() if k != "prompt_variants_used"}
        if hasattr(self, "_stable_identity") and self._stable_identity != stable:
            raise ModelApiError("MODEL_API_MODEL_CHANGED_DURING_DOCUMENT")
        self._stable_identity = stable
        self.identity = {**identity, "role": self.role,
                         "endpoint": self._config_snapshot["url"], "invoked": True}
        self.metrics["requests"].append({"request_id": request_id, "count": len(items),
                                          "wall_seconds": time.perf_counter() - started,
                                          "server": result.get("metrics", {})})
        self.stage_requests += 1
        self.stage_items += len(items)
        self._progress("BATCH_COMPLETE", request_id, model_unloaded=self.session_id is None)
        by_id = {row["id"]: row for row in rows}
        return [by_id[key] for key in expected]

    def _close_session(self):
        if self.session_id is not None:
            started = time.perf_counter()
            for attempt in range(4):
                try:
                    value = self._session_call("close")
                    break
                except SessionExitPending:
                    self._progress("EXIT_PENDING")
                    if attempt == 3:
                        raise
                    time.sleep(1)
            if (value.get("schema") != SESSION_SCHEMA or value.get("session_id") != self.session_id
                    or value.get("model_unloaded") is not True or value.get("state") != "CLOSED"):
                raise ModelApiError("MODEL_API_SESSION_UNLOAD_UNCONFIRMED")
            self.metrics.setdefault("sessions", []).append({**value, "close_wall_seconds": time.perf_counter()-started})
            self.session_id = None

    def close(self):
        self._close_session()
        if self.stage_id is not None:
            self._progress("STAGE_COMPLETE", model_unloaded=True)
            self.stage_id = None
            self.stage_requests = self.stage_items = 0


class LayoutApiRuntime(LocalModelApi):
    def detect(self, renders, *, output_dir):
        from PIL import Image
        crop_root = Path(output_dir) / "_bemarkdown/visual_serial/api_crops"
        crop_root.mkdir(parents=True, exist_ok=True)
        regions = []
        by_id = {str(render.page_index): render for render in renders}
        if len(by_id) != len(renders):
            raise ModelApiError("MODEL_API_DUPLICATE_PAGE_IDS")
        items = ({"id": str(render.page_index), **encoded_image(render.image_bytes)} for render in renders)
        predictions = ((by_id[row["id"]], row) for batch in self.batches(items) for row in self.request(batch))
        for render, result in predictions:
            with Image.open(BytesIO(render.image_bytes)) as source:
                image = source.convert("RGB")
            try:
                rows = result.get("regions")
                if not isinstance(rows, list):
                    raise ModelApiError("MODEL_API_LAYOUT_REGIONS_REQUIRED")
                for index, row in enumerate(rows):
                    box = row.get("bbox", [])
                    if (len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                            or not 0 <= box[0] < box[2] <= image.width
                            or not 0 <= box[1] < box[3] <= image.height
                            or not isinstance(row.get("label"), str) or not row["label"]
                            or not isinstance(row.get("reading_order"), int)):
                        raise ModelApiError("MODEL_API_LAYOUT_REGION_INVALID")
                    # Preserve all pixels covered by a fractional box.
                    box = [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]
                    region_id = f"p{render.page_index:04d}-r{index:04d}"
                    crop_path = crop_root / f"{region_id}.png"
                    with image.crop(tuple(box)) as crop:
                        crop.save(crop_path, format="PNG")
                    regions.append({"region_id": region_id, "label": row["label"], "bbox": box,
                                    "page_index": render.page_index, "reading_order": row["reading_order"],
                                    "crop": {"path": str(crop_path),
                                             "sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest()},
                                    "metadata": {"render_sha256": render.render_sha256,
                                                 "score": row.get("score")}})
            finally:
                image.close()
        return regions


class OcrApiRuntime(LocalModelApi):
    def __init__(self, role, spec):
        super().__init__(role, spec)
        enabled = role == "text" and spec.get("text_result_cache", "on") in {True, "on"}
        entries = spec.get("text_result_cache_entries", 4096)
        characters = spec.get("text_result_cache_characters", 16 * 1024 * 1024)
        if type(entries) is not int or entries <= 0 or type(characters) is not int or characters <= 0:
            raise ModelApiError("MODEL_API_TEXT_CACHE_LIMIT_INVALID")
        self._cache_enabled = enabled
        self._cache_entries = OrderedDict()
        self._cache_limits = (entries, characters)
        self._cache_characters = 0
        self.metrics["text_result_cache"] = {
            "enabled": enabled, "scope": "document", "occurrences": 0,
            "model_generations": 0, "hits": 0, "evictions": 0,
            "identity_bypass_occurrences": 0, "empty_results_not_cached": 0,
            "entries": 0, "characters": 0, "max_entries": entries, "max_characters": characters,
        }

    def _cache_scope(self):
        identity = getattr(self, "_stable_identity", None)
        if not identity or not identity.get("prompt_fingerprint") or "max_new_tokens" not in identity:
            return None
        scope = {"role": self._role_snapshot, "spec": self._config_snapshot, "identity": identity}
        return hashlib.sha256(json.dumps(scope, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()

    def _cache_store(self, key, text):
        size = len(text)
        entries, characters = self._cache_limits
        while self._cache_entries and (len(self._cache_entries) >= entries
                                      or self._cache_characters + size > characters):
            _, old = self._cache_entries.popitem(last=False)
            self._cache_characters -= len(old)
            self.metrics["text_result_cache"]["evictions"] += 1
        if size > characters:
            return
        self._cache_entries[key] = text
        self._cache_characters += size

    def recognize_batch(self, crops):
        # Check before lookup: a mutated endpoint must never observe a cache hit.
        self._assert_config_fixed()

        def items():
            for index, crop in enumerate(crops):
                data = Path(crop["path"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != crop["sha256"]:
                    raise ModelApiError("MODEL_API_CROP_HASH_MISMATCH")
                yield {"id": str(index), **encoded_image(data),
                       "serialization_profile": crop.get("serialization_profile", "default")}

        if not self._cache_enabled:
            output = []
            for batch in self.batches(items()):
                for row in self.request(batch):
                    if not isinstance(row.get("text"), str):
                        raise ModelApiError("MODEL_API_OCR_TEXT_REQUIRED")
                    output.append(row["text"])
            self.metrics["text_result_cache"]["occurrences"] += len(output)
            self.metrics["text_result_cache"]["model_generations"] += len(output)
            return output

        output = [None] * len(crops)
        stats = self.metrics["text_result_cache"]
        stats["occurrences"] += len(crops)
        for original in self.batches(items()):
            scope = self._cache_scope()
            groups = OrderedDict()
            for item in original:
                key = (item["sha256"], item["serialization_profile"])
                if scope is not None and key in self._cache_entries:
                    text = self._cache_entries.pop(key)
                    self._cache_entries[key] = text
                    output[int(item["id"])] = text
                    stats["hits"] += 1
                else:
                    groups.setdefault(key, []).append(item)
            representatives = [members[0] for members in groups.values()]
            rows = self.request(representatives) if representatives else []
            deferred = []
            for (key, members), row in zip(groups.items(), rows, strict=True):
                text = row.get("text")
                if not isinstance(text, str):
                    raise ModelApiError("MODEL_API_OCR_TEXT_REQUIRED")
                output[int(members[0]["id"])] = text
                scope = self._cache_scope()
                has_content = bool(text.strip())
                reusable = scope is not None and has_content
                if reusable:
                    self._cache_store(key, text)
                    for item in members[1:]:
                        output[int(item["id"])] = text
                        stats["hits"] += 1
                else:
                    stats["identity_bypass_occurrences"] += (len(members) if scope is None else 0)
                    stats["empty_results_not_cached"] += (1 if scope is not None and not has_content else 0)
                    deferred.extend(members[1:])
            for batch in self.batches(iter(deferred)):
                for item, row in zip(batch, self.request(batch), strict=True):
                    if not isinstance(row.get("text"), str):
                        raise ModelApiError("MODEL_API_OCR_TEXT_REQUIRED")
                    output[int(item["id"])] = row["text"]
            stats["model_generations"] += len(representatives) + len(deferred)
        stats["entries"] = len(self._cache_entries)
        stats["characters"] = self._cache_characters
        return output


def create_api_runtime_factory(config_path):
    config = read_config(config_path)

    def factory():
        from .visual_package import _write_visual_package
        return VisualSerialOrchestrator(
            layout_runtime=LayoutApiRuntime("layout", config["roles"]["layout"]),
            formula_runtime=OcrApiRuntime("formula", config["roles"]["formula"]),
            text_runtime=OcrApiRuntime("text", config["roles"]["text"]),
            table_runtime=OcrApiRuntime("table", config["roles"]["table"]) if "table" in config["roles"] else None,
            result_factory=_write_visual_package, policy=api_policy(table_enabled="table" in config["roles"]))
    return factory
