# Isaac 승인 고리 실행 사전등록

기존 순위를 재생한 뒤 승인 원장부터 실제 Isaac probe, 갱신, 종료, 비용 감사까지 기록한다. 최초 사전등록은 `bench/protocols/live_loop_v1.json`이며 종료 시점 계측 보완은 `live_loop_v2.json`에 남긴다. **이 문서·코드·입력 해시를 커밋한 뒤 측정한다.** 기존 결과와 probe 수식·canonical args·임계값·결함 크기는 바꾸지 않는다.

## 사례와 절차

사례는 seed 2027의 `h01`~`h06`, `baseline_p0c`, `e01` 총 8건이다. 전체 결함 6건을 포함하므로 과거 전수가 확정하지 못한 `h04_s2027`도 포함된다. 사례 교체·성공 사례만 선별·인프라 실패 뒤 무조건 재시도는 하지 않는다. 종료 상태·오차·예산 초과를 모두 남긴다.

- 결함 사례: 이미 저장된 `mode=agent` 순위 상위 3개. NONE: 기존 범주 나열 순서 상위 3개. 새 모델 호출은 없다.
- 최대 probe 4개. 모든 사례의 첫 제안을 거절한 뒤 다음 제안부터 승인·소비·실행한다. 거절은 관측이나 probe 예산으로 세지 않는다.
- 승인 기록의 담당자는 `codex_operator_user_authorized_test`다. 사용자가 위임한 로컬 실행 시험이며 실제 사람의 능동 검토 시간은 0이다. 승인 신원 인증이나 사람 사용성 평가를 주장하지 않는다.
- `h04_s2027`의 첫 실행은 원시 출력·시간 sidecar를 쓴 뒤 receipt 전에 별도 worker 프로세스를 종료 코드 75로 끝낸다. 저장된 고아를 `recover`로 한 번 닫고 계속한다. probe 재실행은 없다.
- 모든 사례에서 소비 전에 감사 계약을 등록하고 종료 뒤 `finalize --abstain`에 해당하는 감사를 수행한다. 수리·재학습·새 행동 평가는 하지 않는다.

## 재현 판정

이전 일괄 probe와 같은 checkpoint SHA256·seed를 요구한다. 실시간 `classify` 결과와 기존 결과를 같은 기준 실행 측정값으로 비교하며 **판정 일치율 100%**를 목표로 한다. 수치 일치는 상대 오차 `1e-3`, 절대 오차 `1e-8`로 별도 판정한다. 수치 오차가 커도 임계값을 바꾸지 않는다.

수치 목표는 동결 `classify`가 읽는 필드다: `noise_ratio`, `critic_change`, 보상 재계산 오차 두 항, `low_speed_saturation`, `mismatch_count`, `timeout_ratio`. 그 밖의 원시 측정값도 양쪽을 저장하되 허용 오차 통과의 분모에는 넣지 않는다. 이전 일괄 실행은 모든 probe가 환경 256개를 공유했고 단일 실행은 기존 canonical args에 따라 64개 또는 256개를 쓴다. 환경 수를 비교 결과에 맞춰 늘리지 않는다. 따라서 질량 최솟값·최댓값 같은 부가 통계가 같다고 가정하지 않는다.

기술적 수용 조건은 승인→소비→receipt 기록 4건 이상, 감사 입력 오류 0건, 같은 checkpoint·seed, 판정 100% 일치, 중단 복구의 추가 실행 0건, 비용 8필드 누락 없음, NONE 틀린 확정 0건이다. 모든 8건을 보고하고 수치 허용치·판정·예산 조건은 각각 통과 여부를 표시한다. 원인 확정률·일반화·모델 우위·GPU 절감의 평가가 아니다.

## 비용과 근거

`loop.py run --metered`는 새 관측 wrapper를 통해 동결된 `probes.py`를 호출한다. wrapper는 원래 `measure` 반환값·예외·RNG·reset 순서를 보존하고, `measure` 안의 성공한 `env.step` 호출만 센다. 스텝은 실제 호출 횟수 × 실제 환경 수이며, 환경 생성·reset·물리 내부 substep은 이 필드에 포함하지 않는다. 읽기 전용 probe의 성공한 전이는 0이다. 해당 과정의 CPU·GPU 벽시계 비용은 포함된다.

- 원시 probe: `evals/results/<tag>/probes/<case>__<probe>.json`.
- Isaac·driver 자원 기록: 같은 tag의 `resources/`. wrapper 원본 계측·작업 목록·로그는 비공개 실행 폴더에 보존한다.
- 프로세스 실행 시간·CPU·스텝·출력 해시: 같은 tag의 `execution/`. 복구는 이 시간으로 원래 실행을 한 번만 계산한다. sidecar 없는 과거 파일은 기동 시간을 알 수 없어 비용을 null로 남긴다.
- worker 프로세스 CPU: 같은 tag의 `workers/`. 사례별 입력·계약·추가 비용·timeline·audit·결과는 `cases/<case>/`.

비용 벡터는 모델 호출·입출력 토큰·GPU 벽시계 초·CPU 초·사람 검토 초·개별 실행 벽시계 초·제어 전이 수다. 모델·사람 검토가 없는 이번 범위에서는 해당 값이 0이다. CPU는 운영자 Python, worker Python, 실행 driver Python, Isaac Python 프로세스의 계측 구간 합이다. Isaac CPU는 환경 close까지 먼저 저장하고 app close가 Python으로 돌아오면 추가 sidecar의 종료 후 값을 사용한다. 돌아오지 않으면 app close 중 CPU는 포함하지 않으며 `app_close_cpu_included=false`와 계측 범위를 표시한다. batch launcher 셸과 외부 서비스 CPU도 계측 범위 밖이다. GPU 벽시계는 기동·load·reset·측정·close를 포함하는 실행 구간이며 장치 busy 시간이 아니다. 실행별 벽시계 합과 사례 전체 경과 시간을 따로 기록한다. offline finalize·파일 저장 이후 비용은 실행 범위 밖이다.

공학 한도는 사례당 GPU 벽시계 480초, CPU·개별 실행 합·전체 경과 각각 900초, 전이 1,228,800회다. 실행 전 예상 성능이나 소요 시간으로 읽지 않는다. 기존 probe별 승인 한도도 유지하고 초과 관측은 unknown으로 남긴다. 정상 반복 자료는 추가하지 않으며 기존 상대 여유값과 정상 변동 0을 사용한다. 절대 추종 상한은 고정 명령 격자의 제자리 오차 절반, 생존 하한 0.9, 낙상 상한 0.05로 사전등록한다.

감사기의 `approval_identity_attested`, `repair_execution_verified`, `probe_execution_checkpoint_verified`는 기존 의미대로 false다. 실행 harness의 checkpoint·출력 해시 대조는 별도 자료 연결이며 호스트 서명이 아니다. 수리를 하지 않았으므로 행동 회복을 주장하지 않는다.

## 실행

PowerShell에서 저장소로 이동하고, 커밋된 사전등록을 사용한다. GPU·다른 Isaac·Ollama 확인은 실행기가 수행하며 사용 중이면 측정을 시작하지 않는다.

```powershell
cd "$HOME/rl-triage-agent"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
.venv/Scripts/python.exe evals/live_loop_evidence.py run `
  --prereg bench/protocols/live_loop_v2.json --tag <새-tag> --case <아직-시작하지-않은-case>
```

이미 있는 고리·결과는 덮어쓰지 않는다. 관측 불일치·예산 초과는 원시 기록과 함께 보고한다.

## 종료 시점 계측 보완

v1의 첫 `h01_s2027/P_noise` 측정은 원시 값이 이전과 일치했지만 Isaac 종료 뒤 계측 파일이 남지 않았다. wrapper 외부 finally에만 저장하던 경로를 환경 종료 직후 저장하도록 보완했다. 운영자 출력에서도 cp949가 em dash를 인코딩하지 못해 중단됐으므로 실행 환경을 UTF-8로 고정한다. 이 변경은 측정 수식·RNG·인자·순위·사례·threshold를 바꾸지 않는다.

첫 원장 receipt의 unknown·GPU 18.4초와 초기 CPU·스텝·timeline 누락은 보존한다. 같은 probe를 다시 재서 성공 결과로 바꾸지 않는다. 이후 사례와 남은 probe만 새 계측으로 이어가며, 사전등록 조건별 실패와 인프라 누락을 결과표에 포함한다. 원시 classify 일치와 receipt 판정 일치를 따로 보고한다.

v2는 최초 report·execution·worker·감사 계약·고리 사전등록·기준·중단 state 스냅샷의 SHA와 원장 LF prefix의 길이·행 수·SHA를 고정한다. 기존 원장에 append한 뒤에도 최초 prefix가 같아야 한다. `evals/resume_live_loop_evidence.py`는 이 자료와 커밋 코드를 검증한 뒤 현재 pending부터 이어가고, 최초 CPU·스텝과 전체 timeline 누락을 유지한 audit/result를 새로 쓴다.

```powershell
cd "$HOME/rl-triage-agent"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
.venv/Scripts/python.exe evals/resume_live_loop_evidence.py `
  --prereg bench/protocols/live_loop_v2.json --tag live_loop_20261009 --case h01_s2027
```

## 측정 결과: live_loop_20261009

v1 실행 코드는 `0d712a8`, 종료 시점 보완과 나머지 실행 코드는 `eb568e4`다. 두 사전등록 모두 해당 측정 전에 커밋했다. [요약](../evals/results/live_loop_20261009/summary_v2.json)과 [비용표](../evals/results/live_loop_20261009/costs_v2.csv)는 저장된 8건의 원장·계약·출력·감사를 GPU 없이 다시 대조한 결과다. 원장 receipt ID와 결과표는 1:1이며 최초 실패 자료는 v2의 파일 SHA 및 원장 prefix 대조를 통과했다. 최초 집계 `summary.json`·`costs.csv`는 보존하고 판정에는 분모·예산 검증을 보완한 v2 집계를 사용한다.

| 사례 | 고리 종료 | GPU 벽시계 초 | CPU 초 | 제어 전이 수 | receipt 일치 | 수치 허용치 |
|---|---|---:|---:|---:|---|---|
| h01_s2027 | unidentifiable | 37.6 | 누락 | 누락 | 실패 | 통과 |
| h02_s2027 | confirmed: reward | 18.3 | 24.0625 | 51,200 | 통과 | 통과 |
| h03_s2027 | unidentifiable | 44.4 | 59.90625 | 179,200 | 통과 | 통과 |
| h04_s2027 | unidentifiable | 40.5 | 54.5625 | 102,400 | 통과 | 통과 |
| h05_s2027 | confirmed: exploration | 21.2 | 28.140625 | 51,200 | 통과 | 실패 |
| h06_s2027 | unidentifiable | 19.2 | 25.125 | 51,200 | 통과 | 통과 |
| baseline_p0c_s2027 | unidentifiable | 18.7 | 26.296875 | 51,200 | 통과 | 통과 |
| e01_s2027 | unidentifiable | 35.9 | 51.265625 | 102,400 | 통과 | 통과 |

승인·소비·receipt는 각각 12건이고 거절은 8건이다. 8건 모두 종료했으며 pending·고아 요청과 감사 입력 오류는 0건이다. 출력의 checkpoint SHA·seed와 사전등록 입력은 12/12건 일치했다. 원시 classify는 12/12건, 실제 receipt 판정은 11/12건 일치했다. 사례 단위 receipt 일치는 7/8건이다. 확정 2건, 식별 불가 6건, 틀린 확정 0/8건이며 NONE 틀린 확정은 0/1건이다. 이는 기존 순위와 설계자 결합 사례의 실행 시험이므로 독립 정확도 평가로 쓰지 않는다.

수치 허용치는 필드 18/19개, probe 11/12개, 사례 7/8건에서 통과했다. `h05/P_noise.noise_ratio`는 이전 `0.05030973255634308`, 이번 `0.05025530606508255`로 상대 차이 0.1081828%다. 사전등록 0.1%를 넘었으나 두 원시 판정은 모두 abnormal이다. 오차 원인은 확인하지 않았으며 재측정·허용치 변경·probe 수정은 하지 않았다.

`h04/P_noise` worker는 출력 저장 뒤 receipt 전에 종료 코드 75로 중단됐고 `recover`를 한 번 수행했다. 복구 전후 execution 기록 수는 각각 6개, 출력 해시는 같고 복구 후 고아는 없다. 복구 경로가 실행기를 호출하지 않고 새 실행 기록을 만들지 않아 추가 GPU 벽시계와 전이 수는 각각 0이다. 장치 busy 시간의 별도 측정은 아니다.

GPU 벽시계 합은 **235.8초**다. 비용 8필드가 모두 있는 사례는 7/8건이다. `h01` 최초 인프라 실패 때문에 전체 CPU·개별 실행 벽시계·전이 수 합은 `null`이다. 기록이 있는 부분만의 합은 CPU 295.78125초, 개별 실행 벽시계 243.3685807초, 전이 640,000회이며 전체 합으로 해석하지 않는다. 새 자원 기록 11개 모두 `app_close_cpu_included=false`다. 환경 종료 이후 Isaac app close의 CPU, 셸·외부 서비스 CPU는 포함하지 않는다.

| 사전등록 수용 조건 | 관측 | 결과 |
|---|---|---|
| 종료 고리 4건 이상 | 8건 | 통과 |
| finalize 입력 오류 0건 | 0건 | 통과 |
| receipt 판정 전부 일치 | 11/12건 | 실패 |
| 수치 필드 전부 허용치 안 | 18/19개 | 실패 |
| 프로세스 시작부터 receipt까지 시간 전부 기록 | 11/12건 | 실패 |
| 중단 복구 1회·추가 GPU 실행 0 | 1회·0초 | 통과 |
| 모든 사례 비용 8필드 누락 없음 | 7/8건 | 실패 |
| NONE 틀린 확정 0건 | 0/1건 | 통과 |
| 사례별 비용 한도 모두 확인·통과 | 7건 within, h01 unknown | 실패 |
| 사례 전체 경과 한도 모두 확인·통과 | 7건 within, h01 unknown | 실패 |
| probe 개수·개별 승인 한도 모두 통과 | 8/8건 | 통과 |

**전체 수용 조건은 미충족(`accepted=false`)이다.** 실행·복구 기록은 남았지만 receipt 재현성과 비용 계측 완전성을 모두 만족하지 못했다. 신규 모델 호출·입출력 토큰·사람 검토 시간은 각각 0이다. 따라서 LLM을 새로 호출하는 진단 전체 비용, 사람 승인 사용성, 수리 후 행동 회복, 규칙 대비 모델 우위나 GPU 절감은 이 결과의 측정 범위에 없다.

사전등록 한도는 사례별로 적용한다. 8건 합계에는 별도 한도를 등록하지 않았으므로 합계 `budget_status`는 `not_evaluated`다. 각 사례의 비용·전체 경과·probe 한도는 위 표와 사례별 audit에서 확인한다. GPU 없는 재검산은 다음과 같이 수행한다.

```powershell
cd "$HOME/rl-triage-agent"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
.venv/Scripts/python.exe evals/live_loop_summary.py `
  --prereg bench/protocols/live_loop_v2.json --tag live_loop_20261009 --check-only
```
