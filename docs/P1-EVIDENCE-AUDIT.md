# P1 행동·비용 감사 계약 (2026-10-09)

P1의 승인 원장과 고정 평가를 연결한다. 기존 P0-B1/P0-B2/P0-C 판정·프롬프트·도구·결과 파일은 바꾸지 않는다. 이 단계의 산출물은 **offline 감사 도구와 테스트**이며 새 Isaac 실행, 모델 성능 또는 비용 절감의 실측 결과가 아니다.

## 1. 구현 범위

| 파일 | 역할 |
|---|---|
| `src/rl_triage/behavior_oracle.py` | 기존 `recovery.compare/verdict/derive_bands`에 사전등록한 절대 행동 기준을 추가 |
| `src/rl_triage/experiment_cost.py` | 모든 시도 비용 합산, 누락 보존, 다음 시도의 예산 확인, 동일 정보·계약·한도 검사 |
| `evals/audit_experiment.py` | `register`, `finalize`, `budget`, `compare` CLI |
| `evals/fixed_eval.py` | 기존 계산에 조건별 원시 합계·UTC 시작/종료 시각을 추가하는 출력 확장 |

```mermaid
flowchart LR
  A[진단·probe 사전등록] --> B[감사 계약 등록]
  B --> C[기존 사람 승인 probe 고리]
  C --> D[원장·고정 평가·비용 감사]
  D --> E[행동 변화와 예산 판정]
```

GPU 실행은 기존 실행기가 맡는다. 감사 CLI는 GPU·네트워크·모델을 쓰거나 승인·실험 실행을 수행하지 않는다. `budget`은 다음 실행의 추정 상한을 검사하는 명령이며, 기존 실행기를 자동 차단하는 훅이나 GPU 강제 종료 기능은 아니다.

## 2. 사전등록과 행동 판정

`register`는 읽은 원장에 소비된 probe가 있으면 거부한다. 정상 기준·정상 반복 자료·절대 행동 기준·비용 한도·공통 입력 스냅샷·고리 사전등록과 판정 코드의 해시를 저장한다. 출력은 새 파일로만 쓴다. 원장 소비와 계약 생성 사이에 분산 잠금을 제공하지 않으므로, 한 호스트 운영자가 **등록 완료 후 probe 실행** 순서를 지킨다. 새 평가에 쓸 코드와 계약은 결과를 보기 전에 커밋한다.

행동 판정은 두 조건을 모두 요구한다.

1. 정상 기준 대비 기존 RC 상대 기준: 생존, 낙상, 선속도·회전 추종 오차.
2. 별도 사전등록한 절대 기준: 생존 비율 하한, 낙상 상한, 선속도·회전 RMSE 상한.

학습 보상 크기는 판정하지 않는다. 제자리 정책보다 느슨한 추종 오차 상한과, 절대 기준을 통과하지 못하는 정상 기준은 계약 등록 단계에서 거부한다. 정상 반복 자료가 없으면 변동 0에 기존 `RC.MARGIN`을 더한다. 이를 정상 변동의 실측으로 표현하지 않는다.

프로토콜 파일의 LF SHA256과 task·eval seed·환경 수·명령 격자·관측 시간·실행 환경 버전이 일치해야 한다. 체크포인트는 고리 사전등록과 개입 전 고정 평가 사이에서 일치해야 한다. `action_scale`은 기존 고정 평가의 정책 인터페이스이므로 실행마다 다를 수 있고 감사 결과에 기록한다. 살아 있는 스텝의 RMSE가 낮아도 낙상·생존 기준이 나쁘면 healthy로 판정하지 않는다.

새 고정 평가 출력은 조건마다 `num_envs`, `fall_count`, `alive_steps_sum`, `lin_error_sq_sum`, `yaw_error_sq_sum`을 기록한다. 이를 합산해 낙상률·생존 시간·전체 RMSE를 다시 계산하고 조건별 지표와도 대조한다. float32 합산 순서 차이는 상대 오차 1e-5, 절대 오차 1e-8(생존 초는 1e-4)로 허용한다. 부분 누락·범위 위반·집계 모순은 거부한다.

원시 합계가 없는 과거 자료는 조건 평균 낙상률·조건별 RMSE 범위·생존 하한까지만 확인한다. 읽기 호환을 위해 행동 라벨은 반환하지만 `metrics_verified=false`이며 완전한 비교에 쓰지 않는다. 기준·정상 보정·후보 자료가 모두 원시 통계 검증을 통과해야 true다. 기존 파일을 새 형식으로 꾸며 쓰지 않고, 새 평가 태그에서 원시 합계를 얻는다.

| 개입 전 | 개입 후 | 행동 결과 |
|---|---|---|
| unhealthy | healthy | `recovered` |
| unhealthy | unhealthy | `not_recovered` |
| healthy | healthy | `unnecessary_intervention` |
| healthy | unhealthy | `regression` |
| 무관 | 결과 없음 | `undetermined` |
| 무관 | 개입 보류를 명시 | `abstained` |

입력 조건·해시·필수 지표가 잘못됐으면 입력 오류로 거부한다. 결과 미수신은 `undetermined`다. 행동 회복만으로 유일 원인이나 수리 개입의 인과성을 확정하지 않는다.

## 3. 승인 원장과 비용

`finalize`는 기존 `evals/loops/<case>/`의 네 파일을 읽는다: `prereg.json`, `reference.json`, `state.json`, `ledger.jsonl`. 승인 → 소비 → receipt 순서, 사전등록·canonical args digest, 현재 가설을 가르는 `next_probe` 선택 규칙, 열린 요청 1개 규칙을 검사한다. receipt 관측을 측정·임계값으로 대조하고 가설·관측·거절·갱신 이력·대기 요청·종료 상태를 원장에서 복원해 `state.json`과 맞춘다. 중복 receipt나 승인 없는 소비는 거부한다. 소비 후 receipt가 없으면 재실행하지 않고 `unknown` 시도와 비용 누락을 보존한다.

비용 벡터는 `model_calls`, `input_tokens`, `output_tokens`, `gpu_wall_s`, `cpu_s`, `human_review_s`, `execution_wall_s`, `simulator_steps`다. 개수 지표는 정수다. 성공·실패·timeout·중단·거절을 모두 기록하고 시도 ID 중복은 거부한다.

- `totals`: 지표가 하나라도 미기록이면 `null`.
- `known_totals`: 알려진 부분 합. 누락이 있어도 이 합이 한도를 넘으면 `exceeded`.
- `missing_attempts`: 지표별 누락 시도 목록. 한도 지표가 미기록이면 예산 판정은 `unknown`.
- `gpu_wall_s`: 기존 receipt의 `gpu_s`, 즉 Isaac 시작을 포함한 GPU 실행 프로세스의 벽시계 구간. GPU 장치 busy 시간은 아니다.
- `execution_wall_s`: 개별 실행 구간의 합. 병렬 실행의 전체 경과 시간으로 바꾸어 읽지 않는다.
- `human_review_s`: 능동 검토 실측값. 승인 대기 시간을 사람의 작업 시간으로 채우지 않는다.

기존 probe 경로는 모델을 호출하지 않으므로 해당 시도의 모델 호출·토큰은 0이다. 진단 모델, 분석 CPU, 평가, 실패한 호출, 사람 검토 비용은 `--events`로 별도 제공한다. 시도 비용 객체의 누락은 0으로 추정하지 않는다. 같은 시도 ID를 제공하면 미기록 지표만 보충하고 receipt의 알려진 값·실행 상태를 바꾸려 하면 거부한다.

```json
[
  {"attempt_id":"<case>:<request_id>","status":"completed",
   "costs":{"cpu_s":8.2,"human_review_s":3.1,"simulator_steps":51200}},
  {"attempt_id":"diagnosis-1","status":"timeout",
   "costs":{"model_calls":1,"input_tokens":2300,"output_tokens":0,
            "gpu_wall_s":0,"cpu_s":0.2,"human_review_s":0,
            "execution_wall_s":120,"simulator_steps":0}}
]
```

위 숫자는 **파일 형식 예시**다. timeout 호출의 출력 토큰이 확인되지 않았다면 예시의 0 대신 `null`을 쓴다. 시뮬레이터 스텝도 설정값을 곱한 추정치로 채우지 않고 측정한 값을 쓴다. 측정 출처는 sidecar 파일이나 실행 로그로 보관한다. 통화 환산이나 비용 벡터의 단일 점수화는 하지 않는다.

전체 작업 시간은 `--timeline`의 timezone을 포함한 `started_at`, `finished_at`으로 별도 계산한다. 계약 등록과 모든 원장 이벤트가 그 구간 안에 있어야 하며 미래 종료 시각은 거부한다. `--wall-budget`으로 등록한 한도가 없거나 timeline이 없으면 전체 시간의 예산 판정은 `unknown`이다. 원장 첫 이벤트~마지막 이벤트의 `observed_loop_span_s`는 진단과 최종 평가 전체 시간을 대신하지 않는다.

전체 시간의 완전성을 확인하려면 모든 비용 시도에 `stage`, `started_at`, `finished_at`을 넣는다. probe 시각은 원장의 소비·receipt에서 가져오며 보충 자료로 바꿀 수 없다. `diagnosis` 단계의 receipt가 필요하고, 개입 후 결과를 붙일 때는 `repair`와 `fixed_eval` receipt도 필요하다. 수리 receipt의 `artifact_sha256`은 평가 정책 checkpoint SHA, 고정 평가 receipt의 값은 결과 JSON 파일의 원본 바이트 SHA여야 한다. 새 평가 결과의 시작·종료 시각이 receipt와 전체 구간 안에 있어야 한다. 한 단계라도 빠지거나 산출물 hash가 맞지 않으면 전체 시간의 예산 판정은 `unknown`이다. 시각이나 실행 시간 자체가 모순되면 입력 오류로 거부한다.

외부 receipt 예시의 `costs`에는 같은 비용 벡터를 넣고, 실제 측정값이 없으면 null로 남긴다.

```json
{"attempt_id":"fixed-eval-1","stage":"fixed_eval","status":"completed",
 "started_at":"<UTC 시작 시각>","finished_at":"<UTC 종료 시각>",
 "artifact_sha256":"<평가 JSON 원본 바이트 SHA256>","costs":{}}
```

이 자료 연결은 호스트 서명이나 승인 권한의 증명이 아니다. 수리 실행의 승인·소비 검증은 별도 경계로 남긴다.

## 4. 사용 순서

PowerShell에서 저장소로 이동한다.

```powershell
cd "$HOME/rl-triage-agent"
$env:PYTHONUTF8 = "1"
```

`evals/loop.py start`로 고리를 만든 뒤, 아직 probe를 실행하기 전에 등록한다. `<...>`는 실제 자료 경로로 바꾼다. 기존 결과 태그와 구분해 `evidence_*` 접두어를 쓴다.

```powershell
.venv/Scripts/python.exe evals/audit_experiment.py register `
  --loop-dir evals/loops/<case> --reference <walking-reference.json> `
  --gates <absolute-gates.json> --limits <cost-limits.json> `
  --information <common-raw-input-snapshot.json> --method agent `
  --wall-budget <seconds> --out evals/results/evidence_<tag>/contract.json
```

`absolute-gates.json`에는 `min_survival_fraction`, `max_fall_rate`, `max_lin_vel_rmse_mps`, `max_yaw_rate_rmse_radps` 네 값을 명시한다. 테스트의 0.9·0.05·0.5 m/s·0.5 rad/s는 fixture용 공학 기준 예시이며 기존 P0-C 판정을 바꾸거나 보류 세트에 맞춰 고른 기준이 아니다. 실제 평가 기준은 정상 자료와 목적에 따라 새 결과를 보기 전에 고정한다.

`cost-limits.json`에는 비용 벡터의 한도를 넣는다. 일부만 등록할 수 있지만 기록된 비용 벡터 전체가 완전해야 `compare`가 완전한 비용 비교를 허용한다. 공통 입력 파일에는 양쪽에 줄 같은 정밀도·시계열·도구 정보의 스냅샷을 둔다. 파일 해시 일치만으로 모델이 실제로 같은 입력을 받았다는 사실은 증명되지 않는다.

기존 사람 승인 고리와 별도 승인된 수리·고정 평가를 마친 뒤 감사한다.

```powershell
.venv/Scripts/python.exe evals/audit_experiment.py finalize `
  --loop-dir evals/loops/<case> --contract evals/results/evidence_<tag>/contract.json `
  --reference <walking-reference.json> --before <failed-fixed-eval.json> `
  --after <repaired-fixed-eval.json> --events <measured-attempt-costs.json> `
  --timeline <whole-run-timeline.json> --out evals/results/evidence_<tag>/audit.json
```

개입을 보류하면 `--after` 대신 `--abstain`을 쓴다. 둘 다 없으면 개입 후 결과 미수신으로 처리한다. 평가 실행기의 GPU 호출을 이 명령이 대신하지 않는다.

```powershell
.venv/Scripts/python.exe evals/audit_experiment.py budget `
  --events <attempts-so-far.json> --limits <cost-limits.json> `
  --next-cost <next-attempt-upper-bound.json> --out <new-budget-check.json>

.venv/Scripts/python.exe evals/audit_experiment.py compare `
  --reports <agent-audit.json> <baseline-audit.json> --out <new-comparison.json>
```

종료 코드: 0 저장 완료, 2 입력 오류·덮어쓰기 거부, 3 예산 차단 또는 비교 조건 미충족. `budget`·`compare`는 차단 이유를 출력 파일에 저장하고 3을 반환한다. 호출 측은 0일 때만 다음 실행이나 비교 주장을 진행한다. `finalize`는 초과·불명을 포함한 관측 기록을 저장하는 명령이므로, 종료 코드 0이 회복 또는 예산 준수의 뜻은 아니다.

## 5. 비교와 남은 검증 경계

`compare`는 같은 케이스·개입 전 평가 파일·공통 입력 SHA·판정 계약·비용 한도를 요구한다. 미완료 고리, probe 횟수·요청 예산 초과, 누락된 비용 벡터, 전체 시간의 초과·불명, 원시 행동 통계 미검증은 비교 완료로 인정하지 않는다. `F_failed`, `H_healthy`, `N_all`과 행동 회복·불필요 개입·악화·보류·불명 분자를 따로 기록한다. 결과 미수신 healthy 사례는 불필요 개입의 행동 분자에 넣지 않는다. 검증된 실행 기반 `unnecessary_H/N`은 recovery와 마찬가지로 null이다. 현재 명령은 방식당 감사 결과 하나를 비교하며, 여러 케이스의 벤치 전체 집계는 별도 작업이다.

현재 원장의 receipt에는 실행된 checkpoint SHA가 없다. 따라서 고리 사전등록과 개입 전 고정 평가의 SHA 연결은 검사하지만 `probe_execution_checkpoint_verified=false`다. 수리 실행의 승인·소비·실행 receipt를 이 도구가 검증하지 않으므로 `repair_execution_verified=false`, `validated_recovery_F=null`이다. `behavior_recovered_F`를 검증된 수리 성공 수로 바꾸어 쓰지 않는다. 승인 담당자 이름도 호스트 서명으로 인증한 신원은 아니므로 `approval_identity_attested=false`다.

같은 입력 스냅샷 SHA와 완전한 비용 기록으로 `comparable=true`가 나오더라도 범위는 등록한 자료·비용의 비교다. 모델 요청/응답 trace와 공통 입력 전달을 검증한 것으로 확대하지 않는다(`input_delivery_verified=false`). 이 결과를 AI 우위·일반화·수리 인과성의 증거로 쓰려면 실제 trace와 새 보류 평가가 필요하다.

검증 파일은 `tests/test_behavior_oracle.py`, `tests/test_experiment_cost.py`, `tests/test_experiment_audit.py`다. CLI 테스트는 모델 관련 import·네트워크를 막고 GPU를 숨긴 하위 프로세스에서 기존 `Ledger`와 fixture 고정 평가를 연결한다. 저장된 걷는 정상 기준과 과거 제자리 기준의 호환성도 읽기 전용으로 확인한다. Isaac에서 새 수리를 성공시켰다는 검증은 아니다.

`evals/fixed_eval.py`는 기존 metric 수식·계산 순서·명령·프로토콜을 보존하고 출력 필드만 추가했다. 실제 GPU에서 기존 기준 정책의 RMSE 일치와 새 원시 합계의 재계산 일치는 다음 고정 평가에서 확인한다. 현재 호스트의 다른 페인이 P0-C 학습·probe 측정을 맡고 있어 감사 구현이 별도 GPU 작업을 시작하지 않는다.

후속 실측 담당은 현재 Herdr `w1:p5`다. 이 페인은 P0-C 고정 평가 전에 `baseline_s42`를 새 태그에서 재평가해 기존 선속도 RMSE `1.1754993595071246`과 대조하기로 했다. 완료 조건은 기존 지표 일치와 새 원시 합계의 지표 재계산 통과다. 아직 결과를 수신하지 않았으므로 GPU 호환성 검증 완료로 기록하지 않는다. 수신할 산출물은 새 태그의 `runs/baseline_s42.json`과 실행 로그이며, 기존 `p0b2_fixed_eval_20261008` 자료를 덮어쓰지 않는다. 다른 호스트에서는 pane ID를 다시 조회한다.
