"""Durable request lookup for an explicitly linked, identical managed retry."""
import hashlib
import json
import os
from pathlib import Path
import re

from .model_session import payload_digest, write_json


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RetryReceipts:
    def __init__(self, session):
        self.session = session
        self.previous = []
        self.assets = None
        self.weights_hash = None
        self.weights = None
        jobfile = session.journal.parent.parent / "request.json"
        if not jobfile.is_file():
            return
        current = json.loads(jobfile.read_bytes())
        if current.get("job_id") != os.environ.get("EKB_MANAGED_JOB_ID"):
            return
        spec = json.loads(session.config_path.read_bytes()).get("roles", {}).get(session.role, {})
        if not spec.get("model_root"):
            return
        model = (session.config_path.parent / spec["model_root"]).resolve()
        weights = model / "model.safetensors"
        if not weights.is_file():
            return
        # The native backend already hashes the main weights on load. Bind the rest
        # of the processor/model configuration as well, without rehashing weights on every item.
        parts = {p.relative_to(model).as_posix():file_hash(p) for p in sorted(model.rglob("*"))
                 if p.is_file() and p != weights and "__pycache__" not in p.parts}
        self.assets = payload_digest(parts)
        self.weights = weights
        seen = {current["job_id"]}
        prior = current.get("retry_of")
        while isinstance(prior, str) and re.fullmatch(r"[0-9a-f]{32}", prior) and prior not in seen:
            seen.add(prior)
            folder = jobfile.parent.parent / prior
            parent_request = folder / "request.json"
            if not parent_request.is_file():
                break
            previous = json.loads(parent_request.read_bytes())
            if previous.get("job_id") != prior or not all(
                    current.get(k) and previous.get(k) == current[k]
                    for k in ("workspace", "payload", "program_fingerprint", "config_fingerprint",
                              "dedupe_fingerprint", "registry_fingerprint")):
                break
            for opened in (folder / "model_api_sessions").glob("*/OPENED.json"):
                info = json.loads(opened.read_bytes())
                if info.get("role") == session.role and info.get("config_sha256") == session.config_sha256:
                    self.previous.append(opened.parent)
            prior = previous.get("retry_of")

    def envelope(self, payload, result):
        return {"payload_sha256": payload_digest(payload), "result": result,
                "recovery": {"key": self.key(payload), "assets": self.assets,
                             "result_sha256": payload_digest(result)}}

    def unindexed(self, folder, payload):
        # The durable result is committed before its lookup index. Read a matching
        # envelope directly after interruption, leaving the parent's receipts immutable.
        for path in folder.glob("[0-9a-f]" * 64 + ".json"):
            raw = path.read_bytes()
            saved = json.loads(raw)
            result = saved["result"]
            binding = saved.get("recovery", {})
            if binding and binding.get("key") != self.key(payload):
                continue
            if saved["payload_sha256"] != payload_digest(dict(payload, request_id=result["request_id"])):
                continue
            if (binding.get("assets") != self.assets or
                    binding.get("result_sha256") != payload_digest(result)):
                raise ValueError("MODEL_SESSION_RECOVERY_NEEDS_VERIFIED_RECEIPT")
            return {"receipt": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                    "assets": self.assets, "weights": result.get("identity", {}).get("model_fingerprint")}
        return None

    def key(self, payload):
        return payload_digest({"schema":payload["schema"], "role":payload["role"], "items":payload["items"]})

    def record(self, payload, receipt_file, result):
        if self.assets is None:
            return
        weight = result.get("identity", {}).get("model_fingerprint")
        if not isinstance(weight, str) or not re.fullmatch(r"[0-9a-f]{64}", weight):
            return
        index = self.session.journal / "by-input"
        index.mkdir(exist_ok=True)
        write_json(index / (self.key(payload) + ".json"), dict(receipt=receipt_file.name, sha256=file_hash(receipt_file),
                                                            assets=self.assets, weights=weight))

    def recover(self, payload):
        if not self.previous:
            return None
        name = self.key(payload) + ".json"
        for folder in self.previous:
            index = folder / "by-input" / name
            item = json.loads(index.read_bytes()) if index.is_file() else self.unindexed(folder, payload)
            if item is None:
                continue
            if item["assets"] != self.assets or not re.fullmatch(r"[0-9a-f]{64}\.json", item["receipt"]):
                continue
            if self.weights_hash is None:
                self.weights_hash = file_hash(self.weights)
            if item["weights"] != self.weights_hash:
                continue
            raw = (folder / item["receipt"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("MODEL_SESSION_RECOVERY_RECEIPT_CHANGED")
            saved = json.loads(raw)
            result = saved["result"]
            if (saved["payload_sha256"] != payload_digest(dict(payload,request_id=result["request_id"]))
                    or result.get("session_id") != folder.name or result.get("role") != payload["role"]
                    or result.get("request_complete") is not True or result["identity"].get("model_fingerprint") != self.weights_hash):
                raise ValueError("MODEL_SESSION_RECOVERY_BINDING_INVALID")
            return dict(result, session_id=self.session.session_id, request_id=payload["request_id"],
                        metrics={"recovered":True, "load_count":0, "load_seconds":0, "worker_seconds":0})
        return None
