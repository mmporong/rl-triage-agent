[CmdletBinding()]
param([switch]$Execute, [ValidatePattern('^ppo_gradient_v4_[0-9]{8}[a-z0-9_]{0,4}$')][string]$Tag = 'ppo_gradient_v4_20261017', [ValidateSet('DryRun','Batch','All')][string]$Phase = 'All')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
$runner = Join-Path $repo 'evals/ppo_gradient_check_v4.py'
$isaac = Join-Path $HOME 'IsaacLab'
$lease = Join-Path $HOME 'CodexSignals/Use-GpuLease.ps1'
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; seeds=@(2140,2141,2142); dry_seed=2139; phase=$Phase; max_gpu_runs=@{DryRun=1; Batch=8; All=9}[$Phase]; api_cap=0;
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
    $command = @{DryRun='dry-run'; Batch='run'; All='run-all'}[$Phase]
    & $lease -Owner 'rl-triage-ppo-gradient-v4' -FilePath $python -ArgumentList @('-B',$runner,$command,'--tag',$Tag,'--isaaclab',$isaac) -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw 'PPO 계산 검사 배치 중단. 기존 태그를 재사용하지 않는다.' }
} finally { Pop-Location }
