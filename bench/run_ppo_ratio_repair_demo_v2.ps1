[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Tag = 'ppo_ratio_repair_s2163_20261023',
    [string]$Approver = 'codex-main-user-delegated'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runner = Join-Path $repo 'evals\ppo_ratio_repair_demo_v2.py'
$isaac = Join-Path $HOME 'IsaacLab'
$lease = Join-Path $HOME 'CodexSignals\Use-GpuLease.ps1'
$arguments = @('-B', $runner, 'execute', '--isaaclab', $isaac, '--tag', $Tag,
               '--approver', $Approver, '--approve-registered-actions')
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; lease=$lease; executable=$python; arguments=$arguments;
        approvals='등록 검사와 교정을 운영자 위임으로 승인·소비'; identity_attested=$false; api_request_cap=8} | ConvertTo-Json -Depth 4
    return
}
foreach ($required in @($python, $runner, $lease)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "필수 파일 없음: $required" }
}
Push-Location $repo
try {
    & $python -B $runner verify --isaaclab $isaac
    if ($LASTEXITCODE -ne 0) { throw '커밋·사전등록 검증 실패' }
    & $lease -Owner 'rl-triage-ppo-ratio-repair' -FilePath $python -ArgumentList $arguments -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw "PPO ratio 종단 실행 중단(exit=$LASTEXITCODE). 같은 tag를 다시 실행하지 않는다." }
} finally {
    Pop-Location
}
