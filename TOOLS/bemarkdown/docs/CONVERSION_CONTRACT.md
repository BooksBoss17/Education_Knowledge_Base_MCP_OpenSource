# BeMarkdown Package Contract v1

Status: frozen for DOCX and PDF production.

Contract identifier: `bemarkdown-package-v1`.

## Product boundary

A BeMarkdown Package is a source-faithful intermediate representation produced
before document classification, knowledge extraction, and knowledge-base
admission. Conversion success is not knowledge-base admission success.

This layer may restore source structure, normalize representation, convert
formulas to LaTeX, preserve visual assets, and record technical provenance. It
must not split question types, rebuild textbook chapters, generate RAG chunks,
add teaching labels, rewrite or summarize content, deduplicate source meaning,
or map content to knowledge points. BeMarkdown never writes directly to the
formal `KNOWLEDGE_BASE`.

`CLEAN`, `COMPLETED_WITH_WARNINGS`, and `COMPLETED_WITH_REVIEW_ITEMS` are all
complete packages. A later consumer may reject warnings or review items for
formal admission. `FAILED` is never published as a complete package.

## Package topology

Every published directory is named by its `document_id` and contains:

```text
<document_id>/
  package_manifest.json
  document.md
  assets_manifest.jsonl
  conversion_report.json
  assets/                       # optional when no resolved visual assets exist
    image_000001.ext
    image_000002.ext
```

`assets_manifest.jsonl` and `assets/` follow frozen Asset Reference Contract v1.
An empty manifest is valid; `assets/` may be absent when no resolved assets
exist. The source document is not copied into the package.

The same topology is source-format agnostic. Current source metadata may use
`DOCX` or `PDF`; `SCANNED_PDF` remains a source classification rather than a
different package shape. Package-level files and integrity rules are unchanged.

## Document identity

`document_id` is:

```text
<sanitized-source-stem>__<first-12-lowercase-hex-of-source-sha256>
```

The stem is Unicode NFC, replaces Windows-forbidden/control characters with
underscores, collapses whitespace to underscores, protects reserved Windows
device names, and is capped at 80 code points. Empty stems become `document`.
The 12-hex prefix is frozen by this contract. Repeating the same source file
produces the same ID; different bytes of the same filename produce different
IDs.

## Authoritative package manifest

`package_manifest.json` is the small consumer entrypoint, not an engineering
report. It contains:

```json
{
  "package_contract": "bemarkdown-package-v1",
  "document_id": "source__0123456789ab",
  "source": {
    "type": "DOCX",
    "name": "source.docx",
    "sha256": "..."
  },
  "converter": {
    "name": "bemarkdown",
    "version": "0.2.0"
  },
  "quality_status": "CLEAN",
  "artifacts": {
    "markdown": "document.md",
    "assets_manifest": "assets_manifest.jsonl",
    "conversion_report": "conversion_report.json"
  },
  "integrity": {
    "document_md_sha256": "...",
    "assets_manifest_sha256": "...",
    "conversion_report_sha256": "..."
  },
  "complete": true
}
```

The manifest is written only after Markdown, asset finalization, asset manifest,
conversion report, and payload validation finish. It is therefore the final
logical completion file in staging. Consumers must require the exact contract,
required artifact map, matching hashes, valid Asset Contract invariants, and
`complete=true`.

## Quality status

- `CLEAN`: no known warning or review item.
- `COMPLETED_WITH_WARNINGS`: complete and consumable, with non-blocking warnings.
- `COMPLETED_WITH_REVIEW_ITEMS`: complete, with unresolved/unsupported visuals,
  OCR review/rejection, or preserved OCR inference failure.
- `FAILED`: not publishable.

## Atomic publish and replacement

Production writes only to `<output_root>/.staging/<unique-job-id>/<document_id>`.
It validates the payload, writes the final manifest, validates the complete
package, and then performs a same-volume directory rename to
`<output_root>/<document_id>`.

Replacement never deletes the old package before new conversion completes. The
old directory is renamed to a job-local backup, the validated staging package
is renamed to final, and failures restore the backup. A cleanup failure after a
successful publish leaves an auditable staging backup rather than invalidating
the new complete package. Residual staging jobs are never consumer packages.

## Consumer validation

`validate_package(path)` and `bemarkdown validate <path>` check:

- contract, document ID/directory identity, required files, and `complete=true`;
- SHA-256 for Markdown, asset manifest, and conversion report;
- continuous `image_000001..image_XXXXXX` ordering and unique occurrence UIDs;
- resolved asset existence, SHA, MIME, extension, and Markdown reference;
- unresolved markers and the absence of invented files;
- asset directory/manifest equality;
- formula conservation and a publishable quality status.

Package retention, material classification, and knowledge-base admission remain
the responsibility of later orchestration.
