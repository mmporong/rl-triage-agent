[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('prepare-inputs','smoke','execute')][string]$Mode,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$Anchor,
    [Parameter(Mandatory=$true)][string]$Protocol,
    [string]$Manifest = '',
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$BatchManifest = '',
    [string]$BatchResult = '',
    [string]$TargetRecord = '',
    [string]$Approval = '',
    [string]$SmokeRoot = '',
    [switch]$Execute
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
$runner = Join-Path $repo 'evals/lineage_recovery_execution_v1.py'
$lab = Join-Path $HOME 'IsaacLab'
$isaac = Join-Path $lab '_isaac_sim/python.bat'
$lease = Join-Path $HOME 'CodexSignals/Use-GpuLease.ps1'
$owner = 'rl-triage-lineage-sources-v1'

switch ($Mode) {
    'prepare-inputs' {
        $arguments = @('-B',$runner,'prepare-inputs','--protocol',$Protocol,
            '--batch-manifest',$BatchManifest,'--batch-result',$BatchResult,
            '--target-record',$TargetRecord,'--out',$Output,'--anchor',$Anchor,
            '--lab',$lab,'--isaac-python',$isaac)
    }
    'smoke' {
        $arguments = @('-B',$runner,'smoke','--manifest',$Manifest,'--out',$Output)
    }
    'execute' {
        $arguments = @('-B',$runner,'execute','--manifest',$Manifest,
            '--approval',$Approval,'--smoke-root',$SmokeRoot,'--out',$Output)
    }
}
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; phase=$Mode; executable=$python; arguments=$arguments;
        owner=$owner; gpuWaitSeconds=600; smokeWatchdogSeconds=300;
        trainWatchdogSeconds=1800; evalWatchdogSeconds=600;
        totalWatchdogSeconds=21600; modelCalls=0; paidCloudStarts=0} |
        ConvertTo-Json -Depth 5
    return
}
foreach ($path in @($python,$runner,$isaac,$lease,$Protocol)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "필수 파일 없음: $path" }
}
if (Test-Path -LiteralPath $Output) { throw '기존 recovery 폴더는 덮어쓰지 않습니다' }
if ($Mode -eq 'prepare-inputs') {
    foreach ($path in @($BatchManifest,$BatchResult,$TargetRecord)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "prepare-inputs 필수 파일 없음: $path" }
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'recovery CPU 입력 준비 실패' }
    return
}
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) { throw "manifest 없음: $Manifest" }
if ($Mode -eq 'execute') {
    foreach ($path in @($Approval,(Join-Path $SmokeRoot 'result.json'))) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "execute prerequisite 없음: $path" }
    }
}
Push-Location $repo
try {
    & $lease -Owner $owner -FilePath $python -ArgumentList $arguments -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw "recovery $Mode 중단(exit=$LASTEXITCODE). 자동 재시도하지 않습니다" }
} finally {
    Pop-Location
}
