# 결함 주입 벤치마크 한 건을 Isaac Lab에서 학습하고 텔레메트리 JSON을 만든다.
# 사용: pwsh bench/run_case.ps1 -CaseId c01 -Seed 42 -Overrides @('env.actions.joint_pos.scale=1.0')
param(
    [Parameter(Mandatory)] [string]$CaseId,
    [int]$Seed = 42,
    [int]$NumEnvs = 1024,
    [int]$MaxIterations = 100,
    [string[]]$Overrides = @(),
    [string]$Task = 'Isaac-Velocity-Flat-Unitree-Go2-v0',
    [string]$IsaacLab = "$HOME\IsaacLab",
    [string]$OutDir = ''  # 텔레메트리 폴더(기본 bench\runs). 벤치 밖 실험은 다른 폴더에 둔다
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$runName = "${CaseId}_s$Seed"
$outDir = if ($OutDir) { $OutDir } else { Join-Path $repo 'bench\runs' }
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$logFile = Join-Path $repo "bench\private\logs\$runName.log"
New-Item -ItemType Directory -Force -Path (Split-Path $logFile) | Out-Null

Push-Location $IsaacLab
try {
    # isaaclab.bat(cmd)는 '='와 ','를 인자 구분자로 쪼개 hydra override를 깨뜨린다.
    # Isaac Sim 번들 python.bat에 인자마다 큰따옴표를 씌운 한 줄로 넘긴다.
    $trainArgs = @('scripts\reinforcement_learning\rsl_rl\train.py',
        '--task', $Task, '--num_envs', $NumEnvs, '--max_iterations', $MaxIterations,
        '--seed', $Seed, '--headless', '--run_name', $runName) + $Overrides
    $argLine = ($trainArgs | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' }) -join ' '
    $pythonBat = Join-Path $IsaacLab '_isaac_sim\python.bat'
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $pythonBat -ArgumentList $argLine -WorkingDirectory $IsaacLab -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    $exit = $proc.ExitCode
    $sw.Stop()
    # Isaac Lab 2.1.1 train.py는 --experiment_name을 반영하지 않아 태스크 기본 실험 폴더에 기록된다.
    $runDir = Get-ChildItem "logs\rsl_rl" -Directory | ForEach-Object { Get-ChildItem $_.FullName -Directory } |
        Where-Object { $_.Name -like "*_$runName" } | Sort-Object Name | Select-Object -Last 1
    if (-not $runDir) { throw "실행 폴더 없음 (exit=$exit). 로그: $logFile" }
    $telemetry = Join-Path $outDir "$runName.telemetry.json"
    & .\isaaclab.bat -p (Join-Path $repo 'bench\extract_tb.py') $runDir.FullName $telemetry *>> $logFile
    # isaaclab.bat은 추출이 성공해도 0이 아닌 종료 코드를 남길 때가 있어 결과 파일로 판정한다.
    if (-not (Test-Path $telemetry)) { throw "텔레메트리 추출 실패 (bat exit=$LASTEXITCODE). 로그: $logFile" }
    $global:LASTEXITCODE = 0  # 위 판정 뒤 bat의 거짓 실패 코드가 호출자(eval_bridge 등)에 남지 않게 지운다
    $meta = [ordered]@{
        case_id = $CaseId; seed = $Seed; num_envs = $NumEnvs; max_iterations = $MaxIterations
        task = $Task; train_exit_code = $exit; wall_time_s = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        run_dir = $runDir.FullName
    }
    $meta | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $repo "bench\private\$runName.meta.json")
    Write-Output "$runName exit=$exit wall=$($meta.wall_time_s)s -> $telemetry"
}
finally { Pop-Location }
