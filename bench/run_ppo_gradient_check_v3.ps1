[CmdletBinding()]
param([switch]$Execute, [string]$Tag = 'ppo_gradient_v3_20261017')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
$runner = Join-Path $repo 'evals/ppo_gradient_check_v3.py'
$isaac = Join-Path $HOME 'IsaacLab'
$lease = Join-Path $HOME 'CodexSignals/Use-GpuLease.ps1'
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; seeds=@(2136,2137,2138); max_gpu_runs=8; api_cap=0;
        normal_pairs=3; fault='inverse_importance_ratio_v1'; lease=$lease} | ConvertTo-Json
    return
}
foreach ($path in @($python,$runner,$lease)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "필수 파일 없음: $path" }
}
Push-Location $repo
try {
    & $python -B $runner verify --isaaclab $isaac
    if ($LASTEXITCODE -ne 0) { throw '커밋·사전등록 결합 검증 실패' }
    & $lease -Owner 'rl-triage-ppo-gradient-v3' -FilePath $python -ArgumentList @('-B',$runner,'run','--tag',$Tag,'--isaaclab',$isaac) -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw 'PPO 계산 검사 배치 중단. 기존 태그를 재사용하지 않는다.' }
} finally { Pop-Location }
