[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$Anchor,
    [Parameter(Mandatory=$true)][string]$Protocol,
    [Parameter(Mandatory=$true)][string]$SourceRoot,
    [Parameter(Mandatory=$true)][string]$Output,
    [switch]$Execute
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
$runner = Join-Path $repo 'evals/lineage_fixed_eval_batch_v2.py'
$lab = Join-Path $HOME 'IsaacLab'
$isaac = Join-Path $lab '_isaac_sim/python.bat'
$lease = Join-Path $HOME 'CodexSignals/Use-GpuLease.ps1'
$common = @('--lab',$lab,'--protocol',$Protocol,'--anchor',$Anchor)
$prepareArgs = @('-B',$runner,'prepare-inputs') + $common + @(
    '--source-root',$SourceRoot,'--out',$Output,'--isaac-python',$isaac)
$executeArgs = @('-B',$runner,'execute') + $common + @('--manifest',(Join-Path $Output 'manifest.json'))
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; executable=$python; prepare=$prepareArgs; execute=$executeArgs;
        owner='rl-triage-lineage-sources-v1'; phase='fixed_evaluation_v2_gpu_wait';
        prepareBeforeLease=$true; executeInsideLease=$true; admissionWaitMaxSeconds=600;
        wholeControllerSeconds=3600; sourceJobsRequired=18; evaluationJobs=18;
        trainingUpdates=0; API=0; paidCloudStarts=0} | ConvertTo-Json -Depth 5
    return
}
if (Test-Path -LiteralPath $Output) { throw '기존 평가 폴더는 덮어쓰지 않습니다' }
foreach ($path in @($python,$runner,$isaac,$lease,$Protocol,(Join-Path $SourceRoot 'result.json'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "필수 파일 없음: $path" }
}
Push-Location $repo
try {
    & $python -B $runner verify @common
    if ($LASTEXITCODE -ne 0) { throw 'v2 고정 평가 source·protocol·anchor 검증 실패' }
    & $python @prepareArgs
    if ($LASTEXITCODE -ne 0) { throw 'source 완료·체크포인트·v2 manifest 준비 실패' }
    & $lease -Owner 'rl-triage-lineage-sources-v1' -FilePath $python `
        -ArgumentList $executeArgs -WaitTimeoutSeconds 3600
    if ($LASTEXITCODE -ne 0) { throw "v2 고정 평가 중단(exit=$LASTEXITCODE). 같은 폴더를 재실행하지 않습니다" }
} finally {
    Pop-Location
}
