# Native optimized conversion engine

`convert.py --source INPUT --output-root OUTPUT --mcp-root MCP --caller-pid PID --debug --json`

The default route uses the accepted R6 native core for PDF, DOCX and single-frame PNG/JPEG/WebP/BMP/TIFF. It calls the public core API with automatic external completion disabled. It does not read the private optimization profile or automatically start Flash, Completion or DS.

The existing managed interpreter remains the dependency/bootstrap environment. Its installed wheel metadata is unchanged; `core/` is the verified engine overlay. The entire owned conversion Job, entry controller, MCP caller and known venv launcher are charged under the existing 6 GiB committed-memory and whole-card 6144 MiB GPU limits, physical-free 8 GiB and commit-headroom 9 GiB floors. Before a heavy stage starts, the accepted packaged forecast also reserves its measured increment. An identity-checked child admission rechecks live resources without reserving its already admitted load twice.

The caller's output root contains the normal source-hash-bound conversion package and `.ocr-work/` supervision evidence. Heavy-stage processes belong to the task Job; the MCP caller is observed and reserved but never placed in that kill-on-close Job.

`legacy_completion/` preserves the complete previous optional payload, including its original core, route manifest and bounded review scripts. It is never selected automatically. Use that retained entry only after separately authorizing its external completion behavior. This publication does not call it or remove the existing optional review tools.


使用说明：[提速参数与批处理](PERFORMANCE.md) · [内置图片筛选](IMAGE_SCREENING.md)。开发报告和验收收据保存在 DEVELOPER。
