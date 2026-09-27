# 벤치마크 v2 단독 변경 측정 배치(변경 20개 x seed 2). 이미 있는 결과는 건너뛴다.
param([int[]]$Seeds = @(42, 7), [int]$MaxIterations = 100)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$runCase = Join-Path $PSScriptRoot 'run_case.ps1'
$singles = & uv run --no-sync python (Join-Path $PSScriptRoot 'catalog_v2.py') singles | ConvertFrom-Json
foreach ($s in $Seeds) { foreach ($c in $singles) {
    $tele = Join-Path $repo "bench\runs\$($c.case_id)_s$s.telemetry.json"
    if (Test-Path $tele) { Write-Output "skip $($c.case_id)_s$s"; continue }
    try { & $runCase -CaseId $c.case_id -Seed $s -MaxIterations $MaxIterations -Overrides @($c.overrides) }
    catch { Write-Output "FAIL $($c.case_id)_s$s : $_" }
} }
Write-Output "singles done"
