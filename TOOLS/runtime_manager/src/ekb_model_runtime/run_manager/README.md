# User-level run manager API

Windows implementation, standard library only. No model/framework imports. The
package is included by the existing `ekb-model-runtime` setuptools package
discovery; it can also be shipped as the unchanged `ekb_model_runtime` package
inside a self-contained installed tool. Production paths derive from `__file__`
and the installation registry. The client imports on Python 3.10; a canonical
controlled installation interpreter launches the daemon.

## Registration and API

The installer/host creates an installation-owned JSON file:

```json
{
  "schema": "ekb-run-manager-registry-v1",
  "resource_policy": {
    "module_path": "C:/installed/core/bemarkdown/host_memory_policy.py",
    "sha256": "actual-file-sha256",
    "gpu_index": 0,
    "mode": "auto"
  },
  "programs": {
    "education-mcp-convert": {
      "python": "C:/controlled/runtime/python.exe",
      "script": "C:/installed/TOOLS/education_mcp/managed_worker.py",
      "script_sha256": "actual-file-sha256",
      "installed_root": "C:/installed",
      "arguments": ["--manager-job-id", "{job_id}", "--manager-request", "{request_path}"],
      "resource_kind": "gpu",
      "gpu_key": "GPU:0",
      "requirement": {"extra_private_bytes": 0, "extra_physical_bytes": 0, "extra_gpu_mib": 0},
      "run_timeout_seconds": 86400
    }
  }
}
```

`python_sha256` may additionally pin the registered interpreter. Its actual SHA
is always part of the registration fingerprint. Only the three literal template
fields `{job_id}`, `{request_path}`, `{workspace}` are supported. Clients cannot
submit executable paths, argv, environment overrides, shell commands or secret
fields. Program and policy source hashes are checked before execution. All
frontends from one installation must use the same registry and canonical daemon
interpreter; program interpreters may differ.

```python
from ekb_model_runtime.run_manager import ensure_daemon, RunManagerClient, process_identity

client = ensure_daemon(registry_path=registry, python=canonical_python)
client.register_workspace(workspace, ["education-mcp-convert"], input_roots=[source_root])
job = client.submit("education-mcp-convert", workspace,
                    {"source": source, "source_sha256": source_sha, "options": options},
                    scope_key="source-and-request-scope", requirement=requirement,
                    wait_timeout=900, config_fingerprint=config_sha)
client.status(job["job_id"])                 # get(...) is an alias
client.list(states=["RUNNING"], limit=20)    # 1..20 records
client.adopt(job["job_id"], "new-session")
client.cancel(job["job_id"], reason="user", wait_seconds=5)
client.lookup(program_id, workspace, job["public_id"])
client.find_scope(program_id, workspace, scope_key)
```

`RunManagerClient(root)` reconnects directly; it never starts/restarts a daemon.
Default root is `%LOCALAPPDATA%/BeMarkdown/run-manager`, optionally selected by
`EKB_RUN_MANAGER_HOME` or an explicit root. `status`, `lookup` and `find_scope`
return durable SQLite records without scanning the client's recent-task list.
`find_scope` returns the latest record in that scope or `None`; callers must not
confuse observing an old completion with requesting a new run. `shutdown()` is
drain-only administration and refuses while registered tasks are active.

## Atomic job identity and worker context

`submit` first uses `BEGIN IMMEDIATE` plus a partial unique SQLite index. No
worker output, backup, package or job directory is initialized before this
registration. Identical active requests return the original broker-generated
32-hex `job_id` with `deduplicated=True`. `public_id=job_id[:12]` is stable; public
prefix collisions are rejected. Do not place a fresh random public ID in the
payload before deduplication. Request, configuration, registered program and
resource-policy definition fingerprints form the registration identity; the
effective frozen hardware policy also binds `execution_fingerprint` before
startup. `input_roots` are frozen from registration into each request.

An identical `INTERRUPTED` request returns that record and `retry_required=True`.
Only `retry_of=old_job_id` explicitly authorizes a new attempt. A new explicit
submit after `SUCCEEDED`, `FAILED`, `CANCELLED`, `WAIT_EXPIRED` or
`CAPACITY_INSUFFICIENT` creates a new ID/output scope. Completed predictions are
never silently reused. `retry_of` can optionally link another failed terminal
attempt. Client transport code does not automatically resubmit requests.

The fixed worker receives a JSON `request_path` containing `job_id`, `public_id`,
`workspace`, `input_roots`, `payload`, `job_folder`, `manager_root`, `policy`,
`requirement` and fingerprints. The environment supplies discovery hints
`EKB_RUN_MANAGER_HOME`, `EKB_MANAGED_JOB_ID`, and the complete frozen
`BEMARKDOWN_HOST_MEMORY_POLICY_JSON`. These are not ownership proof. Before
entering an internal managed path:

```python
identity = process_identity(os.getpid())
proof = RunManagerClient(request["manager_root"]).owns_process(
    request["job_id"], identity["pid"], identity["created"])
assert proof["owned"]
```

The daemon verifies creation time through a pinned process handle and real
membership of its retained Windows Job. Nested stage descendants are covered.
Unknown/unverifiable identities return no proof; they are never killed by PID
or process image name.

## Queue, resources, persistence and recovery

States: `QUEUED`, `WAITING_RESOURCES`, `STARTING`, `RUNNING`,
`CANCEL_REQUESTED`, then `SUCCEEDED`, `FAILED`, `CANCELLED`, `INTERRUPTED`,
`CAPACITY_INSUFFICIENT` or `WAIT_EXPIRED`. The initial implementation is
conservative: **one controlled heavy task globally**, including CPU probes.
FIFO queue selection uses `created ASC`; history listing uses newest first.
Cancellation remains observable while a task waits. Waits and runs have explicit
finite deadlines. Requirements exceeding capacity are terminal, not infinite
waits. Already running allocations are measured live, not reserved twice.

The registered shared policy module implements `sample_device`,
`resolve_device_policy`, `evaluate_admission`, `check_current`. Each worker Job
has a hard private-commit limit equal to the policy project cap minus a 128 MiB
controller allowance. This allowance is a conservative scheduling reserve, not
a measured universal daemon footprint. Continuous sampled project private/
working-set counts include **the manager plus the complete owned worker tree**;
whole-device GPU, physical free and commit headroom remain separate observations.
Policy/observation failure stops only the owned Job. Resource summaries preserve
sample count, sampled private/working-set/GPU peaks and minimum physical/commit
free values across log rotation. Failed-allocation Windows Job peak counters
are not presented as actual sampled usage. Very short tasks may have zero
resource samples and carry no sampled-peak claim.

Workers are created suspended, assigned to an owned kill-on-close Windows Job,
registered with PID/creation time/command/source/configuration fingerprints, and
only then resumed. Cancelling or losing the manager closes that owned tree,
including nested native-model Jobs. A cleanup that cannot confirm exit blocks
the global queue; no replacement worker is silently launched.

The daemon uses a user mutex, detached startup and mandatory
`CREATE_BREAKAWAY_FROM_JOB`. If the client's exclusive Job disallows breakaway,
cold bootstrap fails `DAEMON_BREAKAWAY_DENIED`; it never falls back to a daemon
bound to that client. A shared outer host Job may remain. The real Windows test
checks separation from an explicit client-only Job, kills that Job, and confirms
the same manager/task can still be adopted. This is not a promise to outlive an
OS/container-wide Job that intentionally contains all user processes.

Clients disconnecting do not cancel tasks. After daemon crash, previously
starting/running records become `INTERRUPTED`; they are observed rather than
rerun. Queued work that never started retains its queue/deadline. A changed
daemon/registry requires a drain and explicit restart, not replacement of a
running owner. Source changes while queued fail before program execution.

IPC binds only `127.0.0.1`. Each connection uses a fresh random challenge and
HMAC-authenticated JSON request/response, with no pickle or token on the wire.
The manager root ACL grants only the object owner and SYSTEM. Authentication
keys and environments are not logged. There are at most 16 request-handler
threads, 2 MiB messages and 64 KiB payloads. stdout/stderr/resource logs rotate
at 1 MiB with two backups per stream; per-job database events retain the latest
200 entries. Task metadata/history is durable and is not automatically deleted.

## Verification boundary

The scoped suite uses actual lightweight Windows processes with a registered
fake resource-policy sampler. It covers competing bootstrap clients, atomic
deduplication, reconnect/adopt, explicit completed reruns, cross-workspace FIFO,
nested Job ownership/cancel, manager crash, interrupted retries, resource wait/
capacity/cancel/stop, bounded logs, source changes, authenticated JSON rejection,
Windows sharing conflicts, permitted client-Job breakaway and denied breakaway.
It does not load models or repeat the frozen 434-item evaluation. Real MCP/CLI/
WeMM/Semantic bridges and hardware-policy/model validation belong to host
integration acceptance; arbitrary manually started scripts are not managed.
