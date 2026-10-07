param(
    [Parameter(Mandatory = $true)]
    [string]$Python311,
    [ValidateSet('core', 'full')]
    [string]$Profile = 'full',
    [string]$RuntimeDir,
    [string]$Wheelhouse
)

$ErrorActionPreference = 'Stop'
$toolRoot = Split-Path -Parent $PSScriptRoot
$arguments = @(
    (Join-Path $PSScriptRoot 'bootstrap_runtime.py'),
    '--tool-root', $toolRoot,
    '--profile', $Profile
)
if ($RuntimeDir) { $arguments += @('--runtime-dir', $RuntimeDir) }
if ($Wheelhouse) { $arguments += @('--wheelhouse', $Wheelhouse) }
& $Python311 @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
