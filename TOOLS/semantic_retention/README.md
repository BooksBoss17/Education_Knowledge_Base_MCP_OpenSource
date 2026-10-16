# Semantic Retention MCP

本目录提供独立语义筛图接口；当前 BeMarkdown 在转换后调用它生成派生包。安装与集成行为见[公开版使用说明](../../docs/MANAGED_CONVERSION.md)。独立接口仍不直接修改原包，冻结指标仅适用于原来明确的评测口径。

Frozen B200 + Qwen3.5-9B semantic image-retention scheme. This independent stdio
MCP tool reads an existing BeMarkdown PDF conversion package and its matching
source PDF. It does not change the BeMarkdown conversion route, modify Markdown,
delete images, or load a model until an explicit classification job starts.

Set `SEMANTIC_RETENTION_ALLOWED_ROOTS` to a JSON array of absolute allowed input
directories. Start `powershell.exe -NoProfile -File <tool directory>/launch.ps1`.
The launcher reads the installed host-runtime receipt under LOCALAPPDATA; no
DEVELOPER checkout or development virtual environment is required.

Tools:

- `semantic_retention_doctor`: `deep=true` verifies all model/native hashes.
- `semantic_retention_start`: required `package_dir`; optional `source_pdf` and
  `asset_names` (exact package-relative image names). Package must contain
  document.md, conversion_report.json, and original image assets. The PDF must
  match the recorded hash. Returns a job ID immediately.
- `semantic_retention_status`: poll the original 12-hex `job_id`; includes manager policy, queue, identity and terminal state without repeating inference.
- `semantic_retention_cancel`: cancel a queued or running managed job and its owned process tree.
- `semantic_retention_results`: read completed results with `job_id`, `offset`,
  and `limit` (1–200). K = KEEP; D = EXCLUDE_CANDIDATE; U = REVIEW_KEEP.

Candidate runtime integration: one heavy GPU job runs at a time across all registered MCP instances through the persistent per-user run manager. An identical active request reuses its public job ID across frontend reconnects. A new explicit start after completion creates a new job. Jobs and diagnostic logs
are saved in LOCALAPPDATA/BSR/j. Failure retains evidence and never deletes source
content. Results are suggestions for the caller; U must remain retained pending
review. There is no automatic deletion operation. For an explicit retry of an interrupted managed task, pass its previous 12-hex ID as `retry_job_id` to `semantic_retention_start`; other terminal tasks can also carry this optional retry lineage;
the frozen engine may recover only verified completed stages within its own run.

Runtime: Windows x64, installed source Python with PyMuPDF/NumPy/Pillow/SciPy,
separate verified inference Python with Torch/Transformers, and pinned CUDA
llama.cpp. Native binaries/host environments and model bytes are release-managed
local assets, not ordinary Git payloads. `SCHEME.json` binds the two model
manifests, engine archive and profiles. B200 processor assets are tokenizer/config
files only; base safetensors are not needed.

Qualification: 8,767 assets / 12 books, current semantic-v2 evaluation labels,
0 false exclusions, 130 false retentions, 0 unresolved; accuracy 98.5172%,
retention precision 97.1188%. Four books individually have precision below 97%.
99% was not achieved. Zero false exclusions describes this evaluated dataset,
not a guarantee for unseen books. Semantic-v2 includes 28 authorized criterion
revisions; original gold/images/Markdown/training labels remain unchanged.
The identical outputs have 28 false exclusions against original gold and 5
against semantic-v1. Those label revisions are not model improvements.

The logical cold run required one allocation-failure recovery: execution
5290.921 s (includes failed attempt, excludes manual repair), wall 7095.657 s.
Conservative classifier private bound 5.98354 GiB, working bound 4.881 GiB,
whole-card peak 5589 MiB. These are full benchmark measurements, distinct from
the four-asset publication smoke. The runtime candidate uses a device-adaptive owned-process guard and a frozen policy snapshot,
system headroom floors and one-model-at-a-time policy. Resource stops preserve
diagnostics; physical RAM, committed memory and VRAM are different quantities.

License: AGPL-3.0-or-later for the combined tool; Apache-2.0 for the Qwen/B200 model assets;
see LICENSE and THIRD_PARTY_NOTICES.md. Private textbook images, source Markdown,
evaluation labels, and sample-derived templates are excluded from the tool.

Managed runtime installation requires the sibling `TOOLS/runtime_manager` bundle. Jobs and original paginated results remain in `LOCALAPPDATA/BSR/j`; the manager keeps its SQLite queue and bounded diagnostics in its user runtime home. A queued source mutation fails before model loading. Capacity limits report `CAPACITY_INSUFFICIENT`; temporary pressure reports `WAITING_RESOURCES`, with a 900-second queue deadline. This integration does not change inference precision, model selection or native worker arguments. Developer source changes alone do not publish this candidate.
