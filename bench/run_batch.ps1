# 결함 주입 벤치마크 전체 배치. 이미 텔레메트리가 있는 실행은 건너뛴다.
param([int[]]$Seeds = @(42, 7), [int]$MaxIterations = 100)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$runCase = Join-Path $PSScriptRoot 'run_case.ps1'
$cases = Get-Content (Join-Path $PSScriptRoot 'cases.json') -Raw | ConvertFrom-Json
$benign = @('env.rewards.dof_acc_l2.weight=-3.0e-07','agent.algorithm.learning_rate=0.0012','agent.save_interval=25',
            'env.rewards.action_rate_l2.weight=-0.012','env.commands.base_velocity.resampling_time_range=[8.0,8.0]',
            'env.rewards.flat_orientation_l2.weight=-3.0')
$jobs = @()
foreach ($s in $Seeds) { $jobs += ,@("baseline", $s, @()) ; $jobs += ,@("benign_all", $s, $benign) }
foreach ($c in $cases) { foreach ($s in $Seeds) { $jobs += ,@($c.case_id, $s, @($c.overrides)) } }
foreach ($j in $jobs) {
    $id, $seed, $ov = $j
    $tele = Join-Path $repo "bench\runs\${id}_s$seed.telemetry.json"
    if (Test-Path $tele) { Write-Output "skip ${id}_s$seed"; continue }
    try { & $runCase -CaseId $id -Seed $seed -MaxIterations $MaxIterations -Overrides $ov }
    catch { Write-Output "FAIL ${id}_s$seed : $_" }
}
Write-Output "batch done"
