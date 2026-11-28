"""Recovery of a writer interruption and verification of named Windows Job ownership."""
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bemarkdown.model_session import write_json
from bemarkdown.session_recovery import RetryReceipts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TOOLS/runtime_manager/src"))
from ekb_model_runtime.run_manager.process import OwnedProcess, WindowsJob


def test_recover_receipt_committed_before_index(tmp_path, monkeypatch):
    model = tmp_path / "weights"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"fixture weights")
    config = tmp_path / "config.json"
    write_json(config, {"roles": {"text": {"model_root": str(model)}}})
    old, new = "1" * 32, "2" * 32
    binding = dict(workspace="test", payload={"source_sha256": "bound"},
                   program_fingerprint="program", config_fingerprint="config",
                   dedupe_fingerprint="dedupe", registry_fingerprint="registry")

    def setup(job, parent):
        journal = tmp_path / job / "model_api_sessions" / ("a" * 32 if parent is None else "b" * 32)
        journal.mkdir(parents=True)
        write_json(journal.parent.parent / "request.json", dict(binding, job_id=job, retry_of=parent))
        write_json(journal / "OPENED.json", dict(role="text", config_sha256="matching"))
        return SimpleNamespace(journal=journal, role="text", config_sha256="matching",
                               config_path=config, session_id=journal.name)

    first, second = setup(old, None), setup(new, old)
    payload = {"schema": "test", "role": "text", "request_id": "old", "items": [{"id": "1"}]}
    reply = dict(request_id="old", session_id=first.session_id, role="text", request_complete=True,
                 items=[{"id": "1", "text": "saved result"}],
                 identity={"model_fingerprint": hashlib.sha256(weights.read_bytes()).hexdigest()})
    monkeypatch.setenv("EKB_MANAGED_JOB_ID", old)
    prior = RetryReceipts(first)
    receipt = first.journal / (hashlib.sha256(b"old").hexdigest()+".json")
    envelope = prior.envelope(payload, reply)
    write_json(receipt, envelope)
    monkeypatch.setenv("EKB_MANAGED_JOB_ID", new)
    retry = RetryReceipts(second)
    got = retry.recover(dict(payload, request_id="new"))
    assert got["items"] == reply["items"] and got["session_id"] == second.session_id
    assert got["metrics"]["load_count"] == 0
    assert retry.recover(dict(payload, request_id="new", items=[{"id": "2"}])) is None
    envelope["result"]["items"][0]["text"] = "changed"
    write_json(receipt, envelope)
    with pytest.raises(ValueError, match="NEEDS_VERIFIED_RECEIPT"):
        retry.recover(dict(payload, request_id="new"))


@pytest.mark.skipif(os.name != "nt", reason="Requires Windows Job objects")
def test_owned_tree_query_reports_live_and_confirmed_exit(tmp_path):
    name = f"Local\\EKB-Release-Test-{os.getpid()}-{tmp_path.name}"
    owned = OwnedProcess([sys.executable, "-c", "import time; time.sleep(30)"],
                         cwd=tmp_path, environment=os.environ.copy(), folder=tmp_path, job_name=name)
    try:
        owned.resume()
        assert owned.identity["pid"] in WindowsJob.members_after_restart(name)
    finally:
        assert owned.close()["exit_confirmed"]
    assert WindowsJob.members_after_restart(name) == []
