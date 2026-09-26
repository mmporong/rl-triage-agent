# 결함 주입 벤치마크 한 건을 Isaac Lab에서 학습하고 텔레메트리 JSON을 만든다.
# 사용: pwsh bench/run_case.ps1 -CaseId c01 -Seed 42 -Overrides @('env.actions.joint_pos.scale=1.0')
param(
    [Parameter(Mandatory)] [string]$CaseId,
    [int]$Seed = 42,
    [int]$NumEnvs = 1024,
    [int]$MaxIterations = 100,
    [string[]]$Overrides = @(),
    [string]$Task = 'Isaac-Velocity-Flat-Unitree-Go2-v0',
    [string]$IsaacLab = "$HOME\IsaacLab"
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$runName = "${CaseId}_s$Seed"
$outDir = Join-Path $repo 'bench\runs'
$logFile = Join-Path $repo "bench\private\logs\$runName.log"
New-Item -ItemType Directory -Force -Path (Split-Path $logFile) | Out-Null

Push-Location $IsaacLab
try {
    $trainArgs = @('-p', 'scripts\reinforcement_learning\rsl_rl\train.py',
        '--task', $Task, '--num_envs', $NumEnvs, '--max_iterations', $MaxIterations,
        '--seed', $Seed, '--headless', '--run_name', $runName) + $Overrides
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & .\isaaclab.bat @trainArgs *> $logFile
    $exit = $LASTEXITCODE
    $sw.Stop()
    # Isaac Lab 2.1.1 train.py는 --experiment_name을 반영하지 않아 태스크 기본 실험 폴더에 기록된다.
    $runDir = Get-ChildItem "logs\rsl_rl" -Directory | ForEach-Object { Get-ChildItem $_.FullName -Directory } |
        Where-Object { $_.Name -like "*_$runName" } | Sort-Object Name | Select-Object -Last 1
    if (-not $runDir) { throw "실행 폴더 없음 (exit=$exit). 로그: $logFile" }
    $telemetry = Join-Path $outDir "$runName.telemetry.json"
    & .\isaaclab.bat -p (Join-Path $repo 'bench\extract_tb.py') $runDir.FullName $telemetry *>> $logFile
    $meta = [ordered]@{
        case_id = $CaseId; seed = $Seed; num_envs = $NumEnvs; max_iterations = $MaxIterations
        task = $Task; train_exit_code = $exit; wall_time_s = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        run_dir = $runDir.FullName
    }
    $meta | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $repo "bench\private\$runName.meta.json")
    Write-Output "$runName exit=$exit wall=$($meta.wall_time_s)s -> $telemetry"
}
finally { Pop-Location }
