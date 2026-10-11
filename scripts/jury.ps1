<#
.SYNOPSIS
Official Artemis-X release workflows. See docs/JURY_QUICKSTART.md.
.EXAMPLE
    .\scripts\jury.ps1 -Step preflight
.EXAMPLE
    .\scripts\jury.ps1 -Step test
.EXAMPLE
    .\scripts\jury.ps1 -Step start
.EXAMPLE
    .\scripts\jury.ps1 -Step stop
#>
[CmdletBinding()]
param(
    [ValidateSet('preflight', 'unit', 'test', 'start', 'stop', 'demo', 'status')]
    [string] $Step = 'preflight',
    [switch] $ReuseDashboard
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot 'ground_station/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    Write-Error 'Prepare Python first: uv sync --project ground_station --locked --group dev'
    exit 2
}
$juryArgs = @((Join-Path $PSScriptRoot 'jury.py'), $Step)
if ($ReuseDashboard) { $juryArgs += '--reuse-dashboard' }
& $pythonExe @juryArgs
exit $LASTEXITCODE
