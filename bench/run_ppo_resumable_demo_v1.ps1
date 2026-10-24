[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Tag = 'ppo_resumable_s2200_20261025',
    [string]$Approver = 'codex-main-user-delegated'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runner = Join-Path $repo 'evals\ppo_resumable_demo_v1.py'
$isaac = Join-Path $HOME 'IsaacLab'
$lease = Join-Path $HOME 'CodexSignals\Use-GpuLease.ps1'
$arguments = @('-B', $runner, 'execute', '--isaaclab', $isaac, '--tag', $Tag,
               '--approver', $Approver, '--approve-registered-actions')
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; executable=$python; lease=$lease; arguments=$arguments;
        api_cap=20; identity_attested=$false;
        resume='같은 명령; paused_infra는 원장 not_before 이후에만 HTTP 재시도';
        gpu_gate='공용 임대 안에서 매 단계 strict 검사';
        unknown='미저장 응답·중단 GPU 단계는 자동 반복하지 않음'} | ConvertTo-Json -Depth 4
    return
}
foreach ($required in @($python, $runner, $lease)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "필수 파일 없음: $required" }
}
Push-Location $repo
try {
    & $python -B $runner verify --isaaclab $isaac
    if ($LASTEXITCODE -ne 0) { throw '커밋·사전등록 검증 실패' }
    & $lease -Owner 'rl-triage-ppo-resumable-v1' -FilePath $python -ArgumentList $arguments -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw '실행 실패. 원장·미완료 단계를 확인하며 자동 재시도하지 않는다.' }
} finally {
    Pop-Location
}
