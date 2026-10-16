[CmdletBinding()]
param([switch]$Execute)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runner = Join-Path $repo 'evals\anymal_termination_smoke_v1.py'
$isaac = Join-Path $HOME 'IsaacLab'
$lease = Join-Path $HOME 'CodexSignals\Use-GpuLease.ps1'
$arguments = @('-B', $runner, 'execute', '--isaaclab', $isaac, '--tag', 'anymal_f6_smoke_s2100_20261017')
if (-not $Execute) {
    [pscustomobject]@{mode='preview'; lease=$lease; executable=$python; arguments=$arguments} | ConvertTo-Json -Depth 4
    return
}
foreach ($required in @($python, $runner, $lease)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "필수 파일 없음: $required" }
}
Push-Location $repo
try {
    & $python -B $runner verify --isaaclab $isaac
    if ($LASTEXITCODE -ne 0) { throw '커밋·사전등록 검증 실패' }
    & $lease -Owner 'rl-triage-anymal-f6-smoke' -FilePath $python -ArgumentList $arguments -WaitTimeoutSeconds 21600
    if ($LASTEXITCODE -ne 0) { throw 'smoke 중단. 기존 결과를 보존하며 같은 tag를 재실행하지 않는다.' }
} finally { Pop-Location }
