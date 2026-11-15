"""Versioned, model-independent loopback HTTP contract for visual transcription.

Only image bytes cross the boundary. No server filesystem paths or Python
objects are part of the protocol. Inference is never retried automatically.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .visual_serial_pipeline import LOCAL_API_POLICY, VisualSerialOrchestrator, VisualSerialPolicy

SCHEMA = LOCAL_API_POLICY
ROLES = ("layout", "formula", "table", "text")
LEGACY_ROLES = frozenset({"layout", "formula", "text"})
MAX_BYTES = 64 * 1024 * 1024
MAX_ITEMS = 128


class ModelApiError(ValueError):
    pass


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
    return config


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelApiError("MODEL_API_REDIRECT_FORBIDDEN")


class LocalModelApi:
    def __init__(self, role, spec):
        self.role, self.spec = role, dict(spec)
        self.identity = {"role": role, "endpoint": local_url(spec["url"]), "invoked": False}
        self.metrics = {"requests": []}

    def request(self, items):
        request_id = uuid.uuid4().hex
        payload = {"schema": SCHEMA, "role": self.role, "request_id": request_id, "items": items}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(body) > MAX_BYTES or not 0 < len(items) <= MAX_ITEMS:
            raise ModelApiError("MODEL_API_REQUEST_TOO_LARGE")
        headers = {"Content-Type": "application/json"}
        if self.spec.get("token_env"):
            headers["Authorization"] = "Bearer " + os.environ[self.spec["token_env"]]
        started = time.perf_counter()
        try:
            opener = build_opener(ProxyHandler({}), _NoRedirect())
            with opener.open(Request(self.spec["url"], body, headers),
                             timeout=self.spec.get("timeout_seconds", 600)) as response:
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ModelApiError("MODEL_API_RESPONSE_TOO_LARGE")
            result = json.loads(data)
        except HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", errors="replace")
            raise ModelApiError(f"MODEL_API_HTTP_ERROR:{self.role}:{exc.code}:{detail}") from exc
        except Exception as exc:
            raise ModelApiError(f"MODEL_API_REQUEST_FAILED:{self.role}:{type(exc).__name__}:{exc}") from exc
        if (result.get("schema") != SCHEMA or result.get("role") != self.role
                or result.get("request_id") != request_id or result.get("released") is not True):
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
        self.identity = {**identity, "role": self.role, "endpoint": self.spec["url"], "invoked": True}
        self.metrics["requests"].append({"request_id": request_id, "count": len(items),
                                          "wall_seconds": time.perf_counter() - started,
                                          "server": result.get("metrics", {})})
        by_id = {row["id"]: row for row in rows}
        return [by_id[key] for key in expected]

    def close(self):
        # The API contract requires resources to be released before responding.
        pass


class LayoutApiRuntime(LocalModelApi):
    def detect(self, renders, *, output_dir):
        from PIL import Image
        crop_root = Path(output_dir) / "_bemarkdown/visual_serial/api_crops"
        crop_root.mkdir(parents=True, exist_ok=True)
        regions = []
        for render in renders:
            result = self.request([{"id": str(render.page_index), **encoded_image(render.image_bytes)}])[0]
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
    def recognize_batch(self, crops):
        output, batch, size = [], [], 0

        def flush():
            if batch:
                for row in self.request(batch):
                    if not isinstance(row.get("text"), str):
                        raise ModelApiError("MODEL_API_OCR_TEXT_REQUIRED")
                    output.append(row["text"])

        for index, crop in enumerate(crops):
            data = Path(crop["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != crop["sha256"]:
                raise ModelApiError("MODEL_API_CROP_HASH_MISMATCH")
            item = {"id": str(index), **encoded_image(data),
                    "serialization_profile": crop.get("serialization_profile", "default")}
            item_size = len(json.dumps(item).encode("utf-8"))
            if batch and (len(batch) >= MAX_ITEMS or size + item_size > MAX_BYTES - 4096):
                flush()
                batch, size = [], 0
            batch.append(item)
            size += item_size
        flush()
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
