[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ExecutionAnchor,
    [Parameter(Mandatory=$true)][string]$Protocol,
    [Parameter(Mandatory=$true)][string]$OriginRoot,
    [Parameter(Mandatory=$true)][string]$ContinuationRoot,
    [Parameter(Mandatory=$true)][string]$Output,
    [switch]$Execute
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
$runner = Join-Path $repo 'evals/lineage_source_continuation_v2.py'
$lab = Join-Path $HOME 'IsaacLab'
$isaac = Join-Path $lab '_isaac_sim/python.bat'
$lease = Join-Path $HOME 'CodexSignals/Use-GpuLease.ps1'
$manifest = Join-Path $Output 'continuation-manifest.json'
$common = @('--origin-root',$OriginRoot,'--continuation-root',$ContinuationRoot,'--lab',$lab,
            '--protocol',$Protocol,'--execution-anchor',$ExecutionAnchor)
$prepareArgs = @('-B',$runner,'prepare-inputs') + $common + @('--out',$Output,'--isaac-python',$isaac)
$executeArgs = @('-B',$runner,'execute') + $common + @('--manifest',$manifest)
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; prepareExecutable=$python; prepareArguments=$prepareArgs;
        leaseOwner='rl-triage-lineage-sources-v1'; executeArguments=$executeArgs;
        reusedJobs=16; remainingJobs=2; parentWallSeconds=5400; admissionWaitSeconds=1200;
        retries=0; API=0} | ConvertTo-Json -Depth 5
    return
}
foreach ($path in @($python,$runner,$isaac,$lease,$Protocol,$OriginRoot,$ContinuationRoot)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "필수 경로 없음: $path" }
}
if (Test-Path -LiteralPath $Output) { throw '기존 v2 continuation 폴더는 덮어쓰지 않습니다' }
Push-Location $repo
try {
    & $python @prepareArgs
    if ($LASTEXITCODE -ne 0) { throw 'v2 continuation 입력 준비 검증 실패' }
    & $lease -Owner 'rl-triage-lineage-sources-v1' -FilePath $python -ArgumentList $executeArgs -WaitTimeoutSeconds 5400
    if ($LASTEXITCODE -ne 0) { throw "v2 continuation 중단(exit=$LASTEXITCODE). 자동 재시도하지 않습니다" }
} finally {
    Pop-Location
}
