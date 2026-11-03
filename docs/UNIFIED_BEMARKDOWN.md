# Unified BeMarkdown 0.3.0

One installed wheel now supplies the conversion, validation and managed CLI. TOOLS/bemarkdown_optimized is retired; all supported entrypoints use TOOLS/bemarkdown.

Run setup.ps1 to prepare and verify the new versioned runtime. Existing dependency environments and model bytes can be reused, and old environments remain available for rollback. Restart the MCP client after the persistent queue drains.

python -m bemarkdown convert INPUT --mcp-root MCP --output-root OUTPUT --json uses the persistent resource-managed route. Use --no-image-screening only when explicitly requesting OCR without classification. Formula/cache options are forwarded; the accepted managed execution recipe remains 6gb with device-adaptive admission. Python library calls remain low-level functions for embedding and testing.

The optional legacy external-completion payload is archived outside the active distribution and never selected automatically. Source and wheel are byte-bound by the release manifest. This packaging release does not change model weights, precision or resource protection floors.

Historical documentation describes earlier releases; use this page and MANAGED_CONVERSION.md for current paths. Source/installation tests are distinct from content accuracy.
