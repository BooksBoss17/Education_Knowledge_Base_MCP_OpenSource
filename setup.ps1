[CmdletBinding()]
param(
    [switch]$AcceptExternalLicenses,
    [ValidateSet('all','conversion','none')][string]$Models = 'all',
    [string]$Workspace,
    [string]$RuntimeDirectory,
    [string]$ModelCache,
    [string]$PythonExe
)
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'The current full release is validated on Windows x64 only.' }
if (-not $AcceptExternalLicenses) {
    Write-Host 'Full installation may download NVIDIA CUDA/cuDNN and Microsoft Visual C++ runtime components under their upstream proprietary terms.'
    Write-Host "Read $PSScriptRoot\docs\LICENSE_AUDIT.md before continuing."
    $answer = Read-Host 'Accept these external license terms and continue? [y/N]'
    if ($answer -notin @('y','Y','yes','YES')) { throw 'Installation cancelled before external component installation.' }
}
function Find-Python311 {
    $candidates = @()
    if ($PythonExe) { $candidates += $PythonExe }
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $candidate = & $launcher.Source -3.11 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0) { $candidates += $candidate }
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    $candidates += "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        $probe = & $candidate -c 'import sys,struct; print("OK" if sys.version_info[:2]==(3,11) and struct.calcsize("P")==8 else "NO")' 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe -eq 'OK') { return $candidate }
    }
    return $null
}
function Install-WingetPackage([string]$Id, [switch]$UserScope) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) { throw "Install Windows App Installer (winget), or install $Id from its official site, then rerun setup." }
    $arguments = @('install','--id',$Id,'--exact','--source','winget','--silent','--accept-package-agreements','--accept-source-agreements','--disable-interactivity')
    if ($UserScope) { $arguments += @('--scope','user') }
    & $winget.Source @arguments
    if ($LASTEXITCODE -ne 0) { throw "Installation of $Id failed. Check installer privileges or install from the official site, then rerun setup." }
}
$python = Find-Python311
if (-not $python) {
    Install-WingetPackage -Id 'Python.Python.3.11' -UserScope
    $python = Find-Python311
}
if (-not $python) { throw 'CPython 3.11 x64 could not be located. Rerun with -PythonExe.' }
$vcFiles = @("$env:WINDIR\System32\msvcp140.dll", "$env:WINDIR\System32\vcruntime140_1.dll")
if (@($vcFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }).Count -gt 0) {
    Install-WingetPackage -Id 'Microsoft.VCRedist.2015+.x64'
}
$officeCandidates = @($env:BEMARKDOWN_SOFFICE, "$env:ProgramFiles\LibreOffice\program\soffice.com", "${env:ProgramFiles(x86)}\LibreOffice\program\soffice.com")
$officeCommand = Get-Command soffice -ErrorAction SilentlyContinue
if ($officeCommand) { $officeCandidates += $officeCommand.Source }
$office = $officeCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
if (-not $office) { Install-WingetPackage -Id 'TheDocumentFoundation.LibreOffice' }
$arguments = @("$PSScriptRoot\scripts\setup.py", '--accept-external-licenses', '--models', $Models)
if ($Workspace) { $arguments += @('--workspace', $Workspace) }
if ($RuntimeDirectory) { $arguments += @('--runtime-dir', $RuntimeDirectory) }
if ($ModelCache) { $arguments += @('--model-cache', $ModelCache) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'MCP setup failed. Read .local/logs and rerun to resume.' }
