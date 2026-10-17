[CmdletBinding()]
param(
    [switch]$Execute,
    [ValidatePattern('^combination_anymal_u1_s2151_20261017$')]
    [string]$Tag = 'combination_anymal_u1_s2151_20261017'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv/Scripts/python.exe'
$runner = Join-Path $repo 'evals/combination_trial_v2.py'
$isaac = Join-Path $HOME 'IsaacLab'
$lease = Join-Path $HOME 'CodexSignals/Use-GpuLease.ps1'
if (-not $Execute) {
    [pscustomobject]@{
        mode = 'preview'; tag = $Tag; source_seed = 2151; restart_seed = 3151
        maximum_gpu_runs = 5; api_calls = 0; conditional_restart = $true
        maximum_parent_wall_s = 2700; maximum_owned_gpu_process_wall_s = 2400
        watchdogs_s = @{ train = 900; fixed_eval = 300 }
        stop_when_source_failure_absent = $true; expected_parent_wall_s = @{ absent = 1100; ready = 1700 }
    } | ConvertTo-Json -Depth 4
    return
}
foreach ($path in @($python, $runner, $lease)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "필수 파일 없음: $path" }
}
Push-Location $repo
try {
    & $python -B $runner verify --isaaclab $isaac
    if ($LASTEXITCODE -ne 0) { throw '조합 사전등록·source 결합 검증 실패' }
    & $lease -Owner 'rl-triage-combination-trial-v2' -FilePath $python -ArgumentList @(
        '-B', $runner, 'execute', '--tag', $Tag, '--isaaclab', $isaac
    ) -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw '조합 실행 중단. 기존 tag를 재사용하지 않는다.' }
} finally {
    Pop-Location
}
