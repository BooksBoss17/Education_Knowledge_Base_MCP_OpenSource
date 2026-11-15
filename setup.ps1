param([Parameter(Mandatory=$true)][string]$ApiConfig,[Parameter(Mandatory=$true)][string]$Workspace)
$ErrorActionPreference="Stop"
py -3.11 "$PSScriptRoot/scripts/setup.py" --api-config $ApiConfig --workspace $Workspace
exit $LASTEXITCODE
