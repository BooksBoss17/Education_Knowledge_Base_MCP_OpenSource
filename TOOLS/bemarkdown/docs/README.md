# BeMarkdown 0.2.0 PDF Publication Tool

This payload contains an immutable BeMarkdown wheel under `dist/`, the frozen
single-full-runtime locks under `runtime/`, isolated-runtime bootstrap scripts,
and a portable configuration template. The sibling `../../MODELS/` directory
owns model assets. No Skill, Agent runtime, or MCP server is required.

The Formal MCP publication keeps the existing nine-model Paddle suite at
`runtime/model_suite.json` and adds the eleven-model PDF production suite at
`runtime/pdf_production_suite.json`. The Tool resolves every model through the
Formal model registry/model root. It never downloads a model or resolves a
Developer cache during production conversion.

On Windows x64, run `scripts/bootstrap.ps1` with an explicit CPython 3.11
executable. Bootstrap creates host-generated state under the short
`%LOCALAPPDATA%\BeMarkdown\runtimes\<wheel-sha-prefix>` path by default. It
never installs into system Python, edits `PATH`, or modifies the registry.

Public CLI:

```text
bemarkdown doctor --json --deep
bemarkdown convert source.docx --json
bemarkdown convert source.pdf --json
bemarkdown validate <package> --json
bemarkdown cleanup-staging --json
```

DOCX remains `PRODUCTION_READY_INTERNAL`. PDF is `PUBLISHED_INTERNAL`, while
`independent_quality_validation=PENDING_ZCODE`; this release does not claim a
PDF quality pass. Handwriting remains unsupported, formula semantic
auto-correction remains disabled, and the frozen spatial ordering contract is
unchanged.
