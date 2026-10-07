# BeMarkdown Production Runtime

## Output-root resolution

The production API is:

```python
convert_document(
    source,
    output_root=None,
    *,
    formula_ocr="auto",
    mtef_cache="persistent",
)
```

The same interface dispatches by source extension. `.docx` uses the existing
DOCX producer; `.pdf` uses the frozen modular PDF producer. Both publish the
same BeMarkdown Package v1 topology. The CLI equivalent is:

```text
bemarkdown convert source.docx --json
bemarkdown convert source.pdf --json
bemarkdown convert-batch first.docx second.pdf --json
```

Resolution priority is:

1. explicit API/CLI `output_root`;
2. `BEMARKDOWN_OUTPUT_ROOT` environment override;
3. `workspace.output_root` in `BEMARKDOWN_CONFIG` or discovered
   `bemarkdown.toml`;
4. `<current-directory>/tmp/bemarkdown` portable fallback.

Relative configuration paths resolve against the configuration file. This
checkout's `bemarkdown.toml` resolves to:

```text
<knowledge-base-root>\tmp\bemarkdown
```

The absolute machine path is configuration, not library source. Tests,
benchmarks, and engineering gates always pass an explicit temporary/artifact
root and do not write production packages into DEVELOPER artifacts by default.

## CLI and exit codes

```text
bemarkdown convert <source> [--output-root ROOT] [--json] [--no-replace]
bemarkdown convert-batch <source> [<source> ...] [--output-root ROOT] [--json] [--continue-on-error]
bemarkdown validate <package> [--json]
bemarkdown cleanup-staging [--output-root ROOT] [--older-than-hours 24] [--json]
```

Machine JSON is emitted to stdout; logging is emitted to stderr. `--debug`
enables debug logging and IR evidence. Stable exits are:

```text
0  package published or validation succeeded
2  invalid/unsupported input
3  environment or output-root configuration failure
4  conversion failure
5  package validation, existing-package, or publish failure
```

Warnings and review items still exit 0 when a complete package is published.

`convert-batch` accepts the same conversion options as `convert`, processes
inputs sequentially in one Python host, and publishes the ordinary independent
package for each file. The corresponding Python API is
`convert_documents(sources, output_root=None, continue_on_error=False, reuse_formula_model=True, **conversion_options)`.
It returns `bemarkdown-batch-conversion-v1` with ordered document records,
`PUBLISHED`/`FAILED`/`NOT_RUN` statuses, counts and timings. A failed document
does not undo previously published packages. By default later inputs are not
attempted; `--continue-on-error` attempts them. The CLI returns the first failed
document's exit category even if later documents succeed.

Batch timings cover the API's work, all attempted files, initialization and final
shared-model teardown. They exclude interpreter startup before API entry.
Per-file timings end after that package is published and exclude later shared
teardown, so use the batch total for throughput. They distinguish the first
conversion from subsequent conversions.

By default one FormulaNet model may remain owned by this serial batch across
PDF and screenshot-DOCX files. Each request revalidates the model registry;
changed verified configuration replaces the previous model after its borrowers
release it. The model is bound to its construction thread and released at batch
completion or interruption. An inference failure retires it on release. All
predictions are recomputed, and other models retain their normal lifecycle.
Native-DOCX fallback model lifecycle remains per document; its route releases
any idle shared PDF model before loading native-DOCX models. Use
`--no-reuse-formula-model` / `reuse_formula_model=False` for per-file PDF model
construction. Custom injected PDF/formula factories retain caller control.

The batch result's `formula_model_lifecycle` records actual model loads/unloads,
borrows/releases and inference inputs. Each PDF pipeline report records its own
borrow lifecycle instead of counting a borrow as a new load or an ordinary
release as a physical unload. Resource cleanup errors return nonzero JSON
errors; already published packages remain available. Existing MTEF cache settings
are controlled by the same conversion options.

PP-OCR detection groups unchanged crop files only when the existing detector
preprocessor produces equal tensor shapes. Groups use at most32 images and
1048576 total input pixels; a larger single image retains its scalar path.
No replacement resize/padding is introduced. Unknown preprocessors keep scalar
detection. Input order is restored before the existing recognition batch8 and
source-ink coverage rules. Execution metadata records requested/actual batch
sizes, shape groups, pixel budget and detector batch count. Batch numerics may
change detector scores or raw polygons, so complete-file regression remains
required; grouping alone is not an accuracy guarantee.

The compatibility `convert_docx(source, output, debug=False)` interface remains
available for tests and developer tools.

## Input and XML guards

DOCX preflight requires a regular `.docx` file, ZIP signature, valid central
directory, safe unique part names, `[Content_Types].xml`, `_rels/.rels`,
`word/document.xml`, the main-document content type, and the root internal
office-document relationship. Absolute, traversal, drive-letter, UNC, and
backslash ZIP member names are rejected.

The lxml parser explicitly disables DTD loading, entity resolution, network
access, recovery, and huge-tree mode. Any DTD declaration is rejected. Invalid
main XML fails conversion; malformed optional relationship, style, numbering,
or header data is isolated and reported where source content can still be
preserved.

All OOXML external relationships are provenance-only. Local drives, UNC paths,
HTTP(S), and other external targets are never fetched by default.

## Default resource limits

Defaults are configurable through `DocxResourceLimits`:

| Limit | Default |
|---|---:|
| archive bytes | 512 MiB |
| part count | 10,000 |
| total uncompressed bytes | 512 MiB |
| one XML part | 32 MiB |
| one media part | 128 MiB |
| one OLE part | 64 MiB |
| one other part | 128 MiB |
| member compression ratio | 200:1 |
| image width / height | 50,000 / 50,000 px |
| image pixels | 250,000,000 |

Raster dimensions are read from headers before decoding pixels. The WMF
inspector separately caps input bytes, embedded comment bytes, and record count.
The renderer rejects abnormal physical dimensions or base bitmap allocations;
supersampling may still be safely reduced for a normal geometry. MTEF parsing
caps OLE and extracted payload bytes and converts limit failures into preserved
object-level failures.

The frozen 169-DOCX resource census and chosen headroom are recorded in the
Phase 4A evidence. No frozen sample reaches a default.

## Failure behavior

- FormulaNet is lazy. Zero OCR candidates never load the model. Model-load,
  runtime, or prediction failures preserve source PNGs as
  `OCR_INFERENCE_FAILED_PRESERVE_IMAGE` and publish review packages.
- Corrupt/unreadable/unwritable persistent MTEF cache entries are warnings and
  cache misses; cache is not required for correctness.
- Root package damage fails without a final directory. Object-level image/OLE
  damage is preserved or explicitly reported when safe.
- Disk, manifest, validation, or publish failures leave no new fake final.
  Failed replacements retain or restore the previous complete package.
- `cleanup-staging` removes only immediate job directories older than an
  explicit threshold. It does not implement package retention.

The Formal 0.2.0 runtime is a single Windows x64 / CPython 3.11 environment.
It keeps PaddlePaddle GPU 3.2.2, PaddleX 3.7.2, CUDA 12.6, cuDNN 9.9.0.52, and
GPU 0 with no CPU fallback. The text ensemble adds the frozen Torch
2.13.0+cu130, torchvision 0.28.0+cu130, Transformers 5.15.1, and Accelerate
1.14.0 overlay. `cu130` is the Torch provider wheel ABI; it does not upgrade
the Paddle CUDA baseline.

DOCX remains `PRODUCTION_READY_INTERNAL`. PDF is `PUBLISHED_INTERNAL` with
independent quality validation `PENDING_ZCODE`. Publication does not assert a
PDF quality pass.
