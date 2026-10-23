[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$FactoryOpsCompleteAck,
    [string]$Tag = 'ppo_ratio_repair_s2165_20261023',
    [string]$Approver = 'codex-main-user-delegated'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runner = Join-Path $repo 'evals\ppo_ratio_repair_demo_v4.py'
$isaac = Join-Path $HOME 'IsaacLab'
$lease = Join-Path $HOME 'CodexSignals\Use-GpuLease.ps1'
$arguments = @('-B', $runner, 'execute', '--isaaclab', $isaac, '--tag', $Tag,
               '--approver', $Approver, '--approve-registered-actions')
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; lease=$lease; executable=$python; arguments=$arguments;
        required_ack='FactoryOpsCompleteAck'; required_gpu_gate='standalone_strict_clear';
        approvals='등록 검사와 교정을 운영자 위임으로 승인·소비'; identity_attested=$false;
        api_request_cap=4} | ConvertTo-Json -Depth 4
    return
}
if (-not $FactoryOpsCompleteAck) { throw 'FactoryOps 완료 ACK 뒤에만 실행할 수 있다.' }
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
    if ($LASTEXITCODE -ne 0) { throw 'standalone strict GPU clear 확인 실패' }
    & $lease -Owner 'rl-triage-ppo-ratio-repair-v4' -FilePath $python -ArgumentList $arguments -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw "PPO ratio C4 실행 중단(exit=$LASTEXITCODE). 같은 tag를 다시 실행하지 않는다." }
} finally {
    Pop-Location
}
