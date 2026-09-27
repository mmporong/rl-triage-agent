# 학습된 체크포인트를 Isaac Lab play.py로 재생해 영상으로 남긴다(데모용).
# 사용: & bench/record_play.ps1 -RunName c04_s42 -Overrides @(...) -OutName c04_fault
param(
    [Parameter(Mandatory)] [string]$RunName,
    [string[]]$Overrides = @(),
    [Parameter(Mandatory)] [string]$OutName,
    [int]$NumEnvs = 16,
    [int]$VideoLength = 300,
    [string]$Task = 'Isaac-Velocity-Flat-Unitree-Go2-Play-v0',
    [string]$IsaacLab = "$HOME\IsaacLab"
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$runDir = Get-ChildItem "$IsaacLab\logs\rsl_rl\unitree_go2_flat" -Directory | Where-Object Name -like "*_$RunName" |
    Sort-Object Name | Select-Object -Last 1
if (-not $runDir) { throw "실행 폴더 없음: $RunName" }
$ckpt = Get-ChildItem $runDir.FullName -Filter 'model_*.pt' | Sort-Object { [int]($_.BaseName -replace 'model_', '') } |
    Select-Object -Last 1
$playArgs = @('scripts\reinforcement_learning\rsl_rl\play.py', '--task', $Task, '--num_envs', $NumEnvs,
    '--headless', '--video', '--video_length', $VideoLength,
    '--load_run', $runDir.Name, '--checkpoint', $ckpt.Name) + $Overrides
$argLine = ($playArgs | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' }) -join ' '
$log = Join-Path $repo "bench\private\logs\play_$OutName.log"
$proc = Start-Process -FilePath (Join-Path $IsaacLab '_isaac_sim\python.bat') -ArgumentList $argLine `
    -WorkingDirectory $IsaacLab -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$video = Get-ChildItem (Join-Path $runDir.FullName 'videos\play') -Filter '*.mp4' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime | Select-Object -Last 1
if ($proc.ExitCode -ne 0 -or -not $video) { throw "재생 실패 exit=$($proc.ExitCode). 로그: $log" }
$dst = Join-Path $repo "docs\media\$OutName.mp4"
New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
Copy-Item $video.FullName $dst -Force
Write-Output "saved $dst ($($ckpt.Name))"
