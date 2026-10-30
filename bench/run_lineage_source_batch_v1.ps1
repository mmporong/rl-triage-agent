[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Anchor,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('smoke','all')][string]$Mode = 'smoke',
    [string]$SmokeResult = '',
    [switch]$Execute
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
$runner = Join-Path $repo 'evals/lineage_source_batch_v1.py'
$lab = Join-Path $HOME 'IsaacLab'
$isaac = Join-Path $lab '_isaac_sim/python.bat'
$lease = Join-Path $HOME 'CodexSignals/Use-GpuLease.ps1'
$runArgs = @('-B',$runner,'execute','--lab',$lab,'--anchor',$Anchor,'--output',$Output,
             '--isaac-python',$isaac,'--mode',$Mode)
if ($SmokeResult) { $runArgs += @('--smoke-result',$SmokeResult) }
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; executable=$python; arguments=$runArgs;
        owner='rl-triage-lineage-sources-v1'; paidCloudStarts=0; API=0} | ConvertTo-Json -Depth 4
    return
}
foreach ($path in @($python,$runner,$isaac,$lease)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "필수 파일 없음: $path" }
}
if (Test-Path -LiteralPath $Output) { throw '기존 실행 폴더는 덮어쓰지 않습니다' }
Push-Location $repo
try {
    & $python -B $runner verify --lab $lab --anchor $Anchor
    if ($LASTEXITCODE -ne 0) { throw '실행 전 source·protocol 검증 실패' }
    & $lease -Owner 'rl-triage-lineage-sources-v1' -FilePath $python -ArgumentList $runArgs -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw "source 수집 중단(exit=$LASTEXITCODE). 같은 폴더를 재실행하지 않습니다" }
} finally {
    Pop-Location
}
