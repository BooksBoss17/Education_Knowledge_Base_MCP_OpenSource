param([switch]$Doctor)
$ErrorActionPreference = 'Stop'
$spec = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'SCHEME.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$receipt = Join-Path $env:LOCALAPPDATA ('BeMarkdownSemanticRetention\runtimes\' + $spec.runtime_id + '\runtime.json')
$pointer = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) '.local\semantic-retention-runtime.json'
if (Test-Path -LiteralPath $pointer) { $receipt = (Get-Content -LiteralPath $pointer -Raw -Encoding UTF8 | ConvertFrom-Json).runtime_path }
if ($env:SEMANTIC_RETENTION_RUNTIME_CONFIG) { $receipt = $env:SEMANTIC_RETENTION_RUNTIME_CONFIG }
$runtime = Get-Content -LiteralPath $receipt -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Doctor) { & $runtime.source_python -I -X utf8 (Join-Path $PSScriptRoot 'server.py') --doctor }
else { & $runtime.source_python -I -X utf8 (Join-Path $PSScriptRoot 'server.py') }
exit $LASTEXITCODE
