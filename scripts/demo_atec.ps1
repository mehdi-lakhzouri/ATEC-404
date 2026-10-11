<#
.SYNOPSIS
    Prepare or launch one Artemis-X demo service per PowerShell terminal.
.DESCRIPTION
    Default/check and status are read-only. Launch steps are explicit, run in the
    foreground, and never kill an existing process or overwrite a previous take.
    See docs/DEMO_ATEC_26.md for the filming and seven-minute presentation script.
.EXAMPLE
    .\scripts\demo_atec.ps1 -Step check
.EXAMPLE
    .\scripts\demo_atec.ps1 -Step link -RunName prise-01
#>
[CmdletBinding()]
param(
    [ValidateSet('check', 'database', 'ground', 'link', 'inference', 'dashboard', 'webots', 'status')]
    [string] $Step = 'check',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')]
    [string] $RunName = ('demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string] $WebotsExe,
    [string] $Device
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$mlPython = Join-Path $repoRoot 'backend/.venv/Scripts/python.exe'
$groundPython = Join-Path $repoRoot 'ground_station/.venv/Scripts/python.exe'
$world = Join-Path $repoRoot 'artemis-webots/worlds/artemis_r4_monitor.wbt'
$runDir = Join-Path $repoRoot ('artemis-webots/output/' + $RunName)
if (-not $WebotsExe) {
    $webotsInstall = if ($env:WEBOTS_HOME) { $env:WEBOTS_HOME } else { 'C:\Program Files\Webots' }
    $WebotsExe = Join-Path $webotsInstall 'msys64/mingw64/bin/webots.exe'
}

function Require-File([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing file: $Path. See docs/DEMO_ATEC_26.md." }
}

function Require-Command([string] $Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { throw "Missing command: $Name." }
}

function Get-PortOwner([int] $Port, [string] $Protocol = 'TCP') {
    if ($Protocol -eq 'UDP') {
        @(Get-NetUDPEndpoint -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -eq $Port })
    } else {
        @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -eq $Port })
    }
}

function Require-FreePort([int] $Port, [string] $Protocol = 'TCP') {
    $owners = @(Get-PortOwner $Port $Protocol)
    if ($owners.Count) {
        $processIds = ($owners.OwningProcess | Sort-Object -Unique) -join ', '
        throw "$Protocol port $Port is already used (PID $processIds). Reuse the verified service or close it yourself; nothing was stopped."
    }
}

function Require-Dependencies([string] $Python, [string[]] $Modules) {
    Require-File $Python
    & $Python -c 'import importlib.util, sys; sys.exit(not all(importlib.util.find_spec(m) for m in sys.argv[1:]))' @Modules
    if ($LASTEXITCODE -ne 0) { throw "Missing dependencies in $Python. Prepare the environment before filming." }
}

function Require-Model {
    $weights = Join-Path $repoRoot 'models/artemis_rock_yolo11n.pt'
    $configPath = Join-Path $repoRoot 'artifacts/provenance/FROZEN_MODEL_CONFIG.json'
    Require-File $weights
    Require-File $configPath
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    if (-not $config.frozen -or (Get-FileHash -LiteralPath $weights -Algorithm SHA256).Hash -ne $config.model.sha256) {
        throw 'Frozen model integrity check failed. Do not train or replace weights for the demo.'
    }
}

function Require-Assets {
    Require-File $world
    $content = Get-Content -LiteralPath $world -Raw
    $refs = [regex]::Matches($content, '"(\.\./assets/[^"\r\n]+)"')
    if (-not $refs.Count) { throw 'No local scene assets found in the monitor world.' }
    foreach ($ref in $refs) {
        Require-File (Join-Path (Split-Path -Parent $world) $ref.Groups[1].Value)
    }
}

function Read-GroundHealth {
    (Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/health' -TimeoutSec 4).data
}

function Require-Ground {
    $health = Read-GroundHealth
    if ($health.mongodb -ne 'up' -or -not $health.receiver.enabled) {
        throw 'Ground station must have MongoDB up AND its UDP receiver enabled. An archive-only API is not sufficient.'
    }
}

function With-Environment([hashtable] $Values, [scriptblock] $Action) {
    $previous = @{}
    try {
        foreach ($key in $Values.Keys) {
            $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
            [Environment]::SetEnvironmentVariable($key, $Values[$key], 'Process')
        }
        & $Action
    } finally {
        foreach ($key in $previous.Keys) {
            [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
        }
    }
}

function Show-Status {
    foreach ($port in @(3000, 8000, 5761, 27017)) {
        $owners = @(Get-PortOwner $port)
        Write-Host ("TCP {0}: {1}" -f $port, $(if ($owners.Count) { 'listening; PID ' + (($owners.OwningProcess | Sort-Object -Unique) -join ', ') } else { 'not started' }))
    }
    foreach ($port in @(5790, 5791, 5792, 5793, 5794)) {
        $owners = @(Get-PortOwner $port 'UDP')
        Write-Host ("UDP {0}: {1}" -f $port, $(if ($owners.Count) { 'bound; PID ' + (($owners.OwningProcess | Sort-Object -Unique) -join ', ') } else { 'not started' }))
    }
    try {
        $health = Read-GroundHealth
        Write-Host ("Ground: {0}; MongoDB: {1}; receiver enabled: {2}; valid packets: {3}" -f $health.status, $health.mongodb, $health.receiver.enabled, $health.receiver.valid)
    } catch { Write-Host 'Ground health unavailable (not started or unreachable).' }
    Write-Host 'Port presence alone does not prove the identity or readiness of a service.'
}

Push-Location $repoRoot
try {
    switch ($Step) {
        'check' {
            Require-Command 'docker'
            Require-Command 'node'
            Require-Command 'pnpm'
            $nodeVersion = & node --version
            if ([int]($nodeVersion.TrimStart('v').Split('.')[0]) -lt 24) { throw 'Node >= 24 is required.' }
            Require-File (Join-Path $repoRoot 'docker/.env')
            Require-File (Join-Path $repoRoot 'ground_station/.env')
            Require-File (Join-Path $repoRoot 'frontend/node_modules/.modules.yaml')
            Require-File $WebotsExe
            Require-Dependencies $mlPython @('torch', 'ultralytics', 'numpy')
            Require-Dependencies $groundPython @('uvicorn', 'fastapi', 'pymongo', 'pydantic_settings')
            Require-Model
            Require-Assets
            & docker info --format '{{.ServerVersion}}'
            if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop first.' }
            Write-Host 'Preflight OK: environments, frozen weights and local world assets exist. No service was started.'
            Show-Status
        }
        'database' {
            Require-Command 'docker'
            Require-File (Join-Path $repoRoot 'docker/.env')
            & docker compose -f docker/compose.yml up -d mongodb
            if ($LASTEXITCODE -ne 0) { throw 'MongoDB startup failed.' }
            & docker compose -f docker/compose.yml ps mongodb
            Write-Host 'Wait for healthy before starting ground; no volume was deleted.'
        }
        'ground' {
            Require-File (Join-Path $repoRoot 'ground_station/.env')
            Require-Dependencies $groundPython @('uvicorn', 'fastapi', 'pymongo')
            Require-FreePort 8000
            Require-FreePort 5790 'UDP'
            With-Environment @{
                ARTEMIS_GROUND_RECEIVER_ENABLED = 'true'; ARTEMIS_GROUND_UDP_HOST = '127.0.0.1'
                ARTEMIS_GROUND_UDP_PORT = '5790'; ARTEMIS_GROUND_ACK_TO = '127.0.0.1:5794'
                ARTEMIS_GROUND_EMULATOR_CONTROL = '127.0.0.1:5792'; ARTEMIS_GROUND_NETWORK_CONTROL_ENABLED = 'true'
            } {
                & $groundPython -m uvicorn artemis_ground.main:app --app-dir ground_station --host 127.0.0.1 --port 8000
                if ($LASTEXITCODE -ne 0) { throw 'Ground station exited with an error.' }
            }
        }
        'link' {
            Require-File $mlPython
            Require-Ground
            foreach ($port in @(5791, 5792, 5794)) { Require-FreePort $port 'UDP' }
            $logPath = Join-Path $runDir 'link.jsonl'
            $statsPath = Join-Path $runDir 'link-stats.json'
            if ((Test-Path -LiteralPath $logPath) -or (Test-Path -LiteralPath $statsPath)) {
                throw 'This take already has link evidence. Choose a new -RunName.'
            }
            New-Item -ItemType Directory -Path $runDir -Force | Out-Null
            & $mlPython scripts/lunar_link.py serve --reverse --preset NORMAL --log $logPath --stats $statsPath
            if ($LASTEXITCODE -ne 0) { throw 'Lunar link exited with an error.' }
        }
        'inference' {
            Require-Dependencies $mlPython @('torch', 'ultralytics', 'numpy')
            Require-Model
            Require-FreePort 5761
            $inferArgs = @('scripts/inference_server.py', '--port', '5761')
            if ($Device) { $inferArgs += @('--device', $Device) }
            & $mlPython @inferArgs
            if ($LASTEXITCODE -ne 0) { throw 'Inference server exited with an error.' }
        }
        'dashboard' {
            Require-Command 'pnpm'
            Require-FreePort 3000
            Push-Location (Join-Path $repoRoot 'frontend')
            try {
                & pnpm --filter web dev
                if ($LASTEXITCODE -ne 0) { throw 'Dashboard exited with an error.' }
            } finally { Pop-Location }
        }
        'webots' {
            Require-File $WebotsExe
            Require-Assets
            Require-Ground
            if (-not @(Get-PortOwner 5761).Count) { throw 'Start inference on TCP 5761 first and wait for its ready message.' }
            $network = (Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/network/status' -TimeoutSec 4).data
            if ($network.preset -notin @('NORMAL', 'RESTORE')) { throw 'Set the real emulator to NORMAL or RESTORE before a new take.' }
            Require-FreePort 5793 'UDP'
            if (Get-Process -Name webots -ErrorAction SilentlyContinue) { throw 'Webots is already open. Close it yourself before launching the environment-configured demo.' }
            $tourDir = Join-Path $runDir 'tour'
            $panelDir = Join-Path $runDir 'monitor-panels'
            if ((Test-Path -LiteralPath $tourDir) -or (Test-Path -LiteralPath $panelDir)) {
                throw 'This take already has rover evidence. Choose a new -RunName.'
            }
            New-Item -ItemType Directory -Path $runDir -Force | Out-Null
            With-Environment @{
                ARTEMIS_MODE = 'demo'; ARTEMIS_INFERENCE = '1'; ARTEMIS_INFER_PORT = '5761'
                ARTEMIS_MAPPING = '1'; ARTEMIS_MONITOR = '1'; ARTEMIS_MONITOR_PANELS = 'rgb,ai,depth,map'
                ARTEMIS_TELEMETRY = '1'; ARTEMIS_TELEMETRY_HOST = '127.0.0.1'; ARTEMIS_TELEMETRY_PORT = '5791'
                ARTEMIS_TELEMETRY_DTN = '1'; ARTEMIS_TELEMETRY_ACK_PORT = '5793'; ARTEMIS_TELEMETRY_SESSION = $null
                ARTEMIS_WEBOTS_OUT = $tourDir; ARTEMIS_WEBOTS_QUIT = '0'; ARTEMIS_FAULTS = $null
                ARTEMIS_MAP_RESET_AT = $null; ARTEMIS_DTN_MAX_ENTRIES = $null; ARTEMIS_DTN_MAX_BYTES = $null
                ARTEMIS_MONITOR_SAVE = $panelDir; ARTEMIS_MONITOR_SAVE_AT = '20,45'
                ARTEMIS_PROFILE = $null; ARTEMIS_MEMTRACE = $null; VIRTUAL_ENV = $null
            } {
                Write-Host "Outputs: $runDir. Webots opens PAUSED; press Run when ready to record."
                # Visible GUI is intentional: the operator is filming this window.
                Start-Process -FilePath $WebotsExe -ArgumentList @('--mode=pause', ('"' + $world + '"')) -Wait
            }
        }
        'status' { Show-Status }
    }
} finally { Pop-Location }
