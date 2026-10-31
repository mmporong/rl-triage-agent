[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Protocol,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$Anchor,
    [Parameter(Mandatory=$true)][string]$Queue,
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SmokeCheckpointSha256,
    [switch]$Execute
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
$runner = Join-Path $repo 'evals/runtime_observation_recovery_v1.py'
$lab = Join-Path $HOME 'IsaacLab'
$isaac = Join-Path $lab '_isaac_sim/python.bat'
$lease = Join-Path $HOME 'CodexSignals/Use-GpuLease.ps1'
$arguments = @('-B',$runner,'run','--protocol',$Protocol,'--anchor',$Anchor,
    '--queue',$Queue,'--out',$Output,'--lab',$lab,'--isaac-python',$isaac,
    '--smoke-checkpoint-sha256',$SmokeCheckpointSha256)
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; executable=$python; arguments=$arguments;
        owner='rl-triage-lineage-sources-v1'; nimMaxRequests=18;
        totalWallSeconds=28800; brevStarts=0; tokenfactoryRequests=0} |
        ConvertTo-Json -Depth 5
    return
}
foreach ($path in @($python,$runner,$isaac,$lease,$Protocol,(Join-Path $Queue 'session.json'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "필수 파일 없음: $path" }
}
if (Test-Path -LiteralPath $Output) { throw '기존 실행 폴더를 덮어쓰지 않습니다' }
Push-Location $repo
try {
    & $lease -Owner 'rl-triage-lineage-sources-v1' -FilePath $python -ArgumentList $arguments -WaitTimeoutSeconds 28800
    if ($LASTEXITCODE -ne 0) { throw "runtime observation 실행 중단(exit=$LASTEXITCODE)" }
} finally {
    Pop-Location
}
