[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Tag = 'selection_comparison_s2170_20261023',
    [string]$Approver = 'codex-main-user-delegated'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runner = Join-Path $repo 'evals\selection_comparison_demo_v1.py'
$isaac = Join-Path $HOME 'IsaacLab'
$lease = Join-Path $HOME 'CodexSignals\Use-GpuLease.ps1'
$arguments = @('-B', $runner, 'execute', '--isaaclab', $isaac, '--tag', $Tag,
               '--approver', $Approver, '--approve-registered-actions')
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; lease=$lease; executable=$python; arguments=$arguments;
        api_http_cap=60; full_training_runs_max=10; fixed_eval_stages_max=9;
        owned_parent_wall_cap_s=16200; overall_elapsed_cap_s=18000;
        identity_attested=$false} | ConvertTo-Json -Depth 4
    return
}
foreach ($required in @($python, $runner, $lease)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "필수 파일 없음: $required" }
}
Push-Location $repo
try {
    & $python -B $runner verify --isaaclab $isaac
    if ($LASTEXITCODE -ne 0) { throw '커밋·사전등록 검증 실패' }
    & $python -B $runner fresh-seeds --isaaclab $isaac
    if ($LASTEXITCODE -ne 0) { throw 'seed 재검사 실패' }
    & $python -B $runner gpu-gate --isaaclab $isaac
    if ($LASTEXITCODE -ne 0) { throw 'GPU 입장 확인 실패' }
    & $lease -Owner 'rl-triage-selection-comparison-v1' -FilePath $python -ArgumentList $arguments -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw "선택 비교 중단(exit=$LASTEXITCODE). 같은 tag를 다시 실행하지 않는다." }
} finally {
    Pop-Location
}
