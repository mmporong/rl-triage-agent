[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Tag = 'residual_v2_20261017',
    [string]$Approver = 'claude-main-user-delegated'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runner = Join-Path $repo 'evals\residual_demo_v2.py'
$isaac = Join-Path $HOME 'IsaacLab'
$lease = Join-Path $HOME 'CodexSignals\Use-GpuLease.ps1'
$arguments = @('-B', $runner, 'execute', '--isaaclab', $isaac, '--tag', $Tag,
               '--approver', $Approver, '--approve-registered-actions')
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; seeds=@(2120,2121); api_cap=20; lease=$lease;
        executable=$python; arguments=$arguments; identity_attested=$false;
        approval='등록 검사·교정별 위임 승인과 일회 소비';
        selection='모델 선택과 등록 규칙 전환을 구분하고 승인 receipt로 결합'} | ConvertTo-Json -Depth 4
    return
}
foreach ($required in @($python, $runner, $lease)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "필수 파일 없음: $required" }
}
Push-Location $repo
try {
    & $python -B $runner verify --isaaclab $isaac
    if ($LASTEXITCODE -ne 0) { throw '커밋·사전등록 검증 실패' }
    & $lease -Owner 'rl-triage-residual-v2' -FilePath $python -ArgumentList $arguments -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw "등록 규칙 전환 고리 중단(exit=$LASTEXITCODE). 기존 tag로 다시 실행하지 않는다." }
} finally {
    Pop-Location
}
