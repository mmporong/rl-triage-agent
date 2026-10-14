# 두 번째 로봇 task 이식: ANYmal-C flat

이 문서의 수치는 `evals/build_anymal_doc.py`가 단계별 사전등록 프로토콜과 결과 파일에서 만든다. 손으로 고치지 않는다.

Go2 flat에서 만든 고리를 다른 제조사 로봇 task `Isaac-Velocity-Flat-Anymal-C-v0`에서 같은 코드로 돌렸다. 고리는 동결 probe 6종, 고정 평가 계약, 등록 교정, 추종 경고 R1, 모델 결정 경로를 말한다. 단계마다 실행 전에 프로토콜을 커밋했고, 앞 단계가 pass일 때만 다음 단계를 사전등록했다.

- **단계 0 관문:** 새 seed 정상 학습 1회로 걷기·probe 측정·절대 기준 probe·학습 비용을 확인한다.
- **단계 1 밴드:** 정상 seed 2개를 더한 정상 실행 3개로 판정 밴드를 정하고, 비 기준 probe와 경고 R1이 정상 실행끼리 오경보를 내지 않는지 본다.
- **단계 2 F1 종단 시연:** 보상 좌표 결함 하나를 시연 v4와 같은 흐름으로 끝까지 연결한다.

범위:
- 결함·probe·교정을 같은 설계자가 만들었다. 로봇 하나·seed 몇 개의 기록이다.
- 진단 정확도·교정 성공률·일반화·다른 방법 대비 우위·GPU 절감을 주장하지 않는다. Go2 결과와 수치를 비교하지 않는다(task·로봇이 다르다).
- 승인은 사용자가 위임한 운영자 승인 또는 사람 승인이다. 승인자 문자열만 기록하며 신원 인증이 아니다.

## 바꾼 것과 그대로 둔 것

| 부품 | 이 task에서 | Go2와 같은가 |
|---|---|---|
| 학습 | `bench/train_with_fault.py`(`TRIAGE_FAULT`, `--task Isaac-Velocity-Flat-Anymal-C-v0`), 등록 학습 설정 그대로(action scale 0.5, push·COM 무작위화 켬) | 진입은 같고 task만 다르다 |
| 고정 평가 | `fixed_eval_v1_anymal_c`: 명령 격자 26개, 지평 1000 step, 평가 seed 2026이 Go2 계약과 같다. 평가 환경에서만 학습 이벤트 push_robot·base_com을 끈다(`evals/fixed_eval_task.py`) | 계약은 같고 평가 환경 이벤트 둘을 끈다 |
| probe 6종 | 동결 `evals/probes.py`의 task 상수만 바꾸는 구동기(`evals/probes_task.py`, 고리 측정은 `evals/metered_probes_task.py`). 측정 수식·인자·임계값은 그대로 | 같다 |
| 판정 밴드 | 이 task의 정상 실행 3개(단계 1)로 P0-C와 같은 방식으로 정했다 | 방식만 같고 값은 새로 정했다 |
| 경고 R1 | 규칙 상수 그대로, 참조는 이 task의 정상 실행 3개 | 규칙은 같다 |
| 진단 packet | 사실·카탈로그·임계값·예산은 같고 task 문구만 ANYmal-C | 문구 외 같다 |
| 교정·원장·결정 경로 | 시연 v4의 회복 방식 규칙·등록 표·교정 원장·검증 피드백 1회 재요청 | 같다 |

## 단계별 결과

| 단계 | 질문 | 결과 | 실험 커밋 | 공개 결과 |
|---|---|---|---|---|
| 0 관문 | 정상 학습 1회가 걷고 probe가 재지는가 | `pass` | `1cd510c` | `evals/results/anymal_c_gate_20261014/result.json` |
| 1 밴드 | 정상 실행 3개로 밴드를 정하고 오경보가 없는가 | `pass` | `f85c085` | `evals/results/anymal_c_band_20261014/result.json` |
| 2 F1 시연 | 결함 하나를 끝까지 연결하는가 | `recovered`, 시연 성공 | `a6be963` | `evals/results/anymal_c_e2e_f1_s2047_20261014/demo_result.json` |

## 단계 0: 이식 관문 (seed 2043)

결과 **`pass`**(예측 pass, 적중). 사전등록 결과 문구: "ANYmal-C flat 정상 학습(새 seed 하나, 4096 env × 300회)이 고정 평가에서 걸었고, 동결 probe 6종이 모두 재졌으며 절대 기준 4종이 normal이었고, 학습 1회 비용이 계획 안이었다. 단계 1로 간다. 단일 seed 이식 관문이며 진단 성능·일반화가 아니다."

- 유효 조건 8/8 참. 학습 프로세스의 결함 readback은 `NONE`이었다. 평가 환경 readback에서 push_robot·base_com은 학습 설정에서 켜져 있다가 평가 환경에서 꺼졌다.
- 고정 평가: 정지 대비 선속도 오차 비 0.155, 낙상 비율 0.013, 생존 19.74 / 20.0 s → 걷는다(기준 0.5 미만·0.05 이하).
- probe 6종이 모두 측정됐다. 절대 기준 4종(P_noise·P_reward·P_physics·P_episode)이 모두 normal이다.
- 비 기준 probe 측정값(판정은 단계 1): P_value critic 변화 1.778, P_torque 저속 포화 2.1e-06.
- 비용(부모 프로세스 벽시계, 기동 포함): 학습 555.4 s(반복당 1.851 s), 고정 평가 54.2 s, probe 6종 97.6 s. 모델 호출 0회.

## 단계 1: 정상 실행 3개와 판정 밴드 (seed 2043, 2044, 2045)

결과 **`pass`**(예측 pass, 적중). 사전등록 결과 문구: "ANYmal-C flat 정상 실행 셋(새 seed 셋)이 모두 걸었고, 동결 비 기준 probe와 추종 경고 R1이 정상 실행끼리 오경보를 내지 않았다. 이 셋으로 P0-C 방식 밴드를 정했다. 단계 2로 간다. 정상 실행 변동 기록이며 진단 성능·일반화가 아니다."

| seed | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 대비 비 | critic 변화 | 저속 포화 | R1 |
|---|---|---|---|---|---|---|---|---|
| 2043 | 0.182 | 0.119 | 0.013 | 19.74 | 0.155 | 1.778 | 2.1e-06 | 울리지 않음 |
| 2044 | 0.189 | 0.124 | 0.028 | 19.47 | 0.161 | 1.835 | 2.1e-06 | 울리지 않음 |
| 2045 | 0.176 | 0.108 | 0.033 | 19.37 | 0.149 | 1.746 | 4.2e-06 | 울리지 않음 |

- 사전등록 기준 4개 중 4개가 참이다.
  - 참: seed 2044·2045 정상 정책이 고정 평가에서 걷는다(정지 대비 오차 비 < 0.5, 낙상 ≤ 0.05)
  - 참: seed 2044·2045의 절대 기준 probe 4종이 normal
  - 참: 비 기준 probe 순서쌍 12개가 모두 normal
  - 참: 정상 실행 셋 모두에서 R1이 울리지 않는다
- 비 기준 probe를 순서쌍 6개마다 동결 classify에 넣은 12개 중 12개가 normal이다. R1이 울린 정상 실행은 0개다(경고 조건을 채운 반복 수 seed별 0, 0, 0).
- 판정 밴드(순서쌍 6개, 여유 비 0.10·낙상 0.05): 생존 비 ≥ 0.881, 선속도 오차 비 ≤ 1.177, yaw 오차 비 ≤ 1.249, 낙상 비율 ≤ 0.083. 단계 2 판정에 이 값을 쓴다.
- 비용(부모 프로세스 벽시계): seed 2044 학습 533.8 s·고정 평가 53.7 s·probe 89.1 s; seed 2045 학습 547.8 s·고정 평가 72.2 s·probe 98.0 s. 모델 호출 0회.

## 단계 2: 보상 좌표 결함(F1) 종단 시연 (seed 2047)

- 사전등록 `bench/protocols/e2e_demo_anymal_c_f1.json`(LF SHA256 `db13cc86…`), 실행기 `evals/e2e_demo_anymal_c.py`, 교정 실행기 `bench/demo_correction_anymal_c.py`.
- 결함 `F1`(`bench/hidden_faults.py`, 동결), 사례 `anymal_c_h02_s2047`·`anymal_c_none_s2047`, 교정 재학습 seed 3047.
- 흐름은 시연 v4와 같다: 경고 R1 → 고정 평가 → 새 Nemotron 판단(검증 피드백 재요청 최대 1회) → 승인된 검사 → 확정 → 교정 제안(같은 재요청) → 승인된 교정 → 재학습 → 고정 평가.
- 판정 밴드와 R1 참조는 단계 1 결과에서 오고 SHA로 묶었다(`evals/results/anymal_c_band_20261014/result.json`). 단계 1이 pass가 아니면 시작하지 않도록 사전등록했다.
- 회복 방식 규칙: reward에서 정지 대비 오차 비 ≥ 0.9이면 restart, 아니면 resume(Go2 시연과 같은 규칙).
- 사전 예측:
  - 학습 실패: F1 정책이 같은 seed NONE 대비 unhealthy이고 정지 대비 오차 비 ≥ 0.9(Go2 F1 실행에서 본 양상을 옮긴 예측이다. 이 task에서 F1 학습은 처음이다)
  - 경고 R1: R1이 F1에 울리고 NONE에는 울리지 않는다(단계 1에서 정상 실행끼리 울리지 않았다는 결과가 전제다)
  - 첫 판단: 첫 판단 최종 결정 action=continue, 상위 3가설에 reward 포함
  - 검사 고리: P_reward abnormal 뒤 reward 확정(예산 4 이내). F1은 보상 함수 본문을 바꾸므로 task와 무관하게 문서 정의 재계산과 기록 보상이 어긋난다고 본다
  - 특이성 관문: 게이트 없음(reward)
  - 회복 방식 규칙: reward에서 정지 대비 오차 비 ≥ 0.9 → restart
  - 교정 결과: 교정 정책 recovered

### 결과

결과는 **`recovered`, 시연 성공**이다. 사전등록 결과 문구: "새 판단으로 시작한 승인된 검사가 등록 결함 메커니즘을 확정했고, 승인된 교정 정책이 사전등록 기준 정책 대비 고정 행동 허용 범위를 통과했다."

| 결정 단계 | 1차 유효 | 재요청 | 최종 유효 | 요청 수 | 토큰(입력·출력) | 대화 전체 |
|---|---|---|---|---|---|---|
| 첫 판단 | 예 | 0회 | 예 | 1 | 2,915·197 | 3.73 s |
| 교정 제안 | 예 | 0회 | 예 | 1 | 1,351·173 | 3.75 s |

| 단계 | 기록 |
|---|---|
| 정상·결함 학습 | 학습 벽시계 NONE 618.2 s · F1 582.4 s |
| 경고 R1 | F1 경고(첫 경고 iteration 76), NONE 경고 없음 |
| 고정 평가 | NONE 보행 통과, F1 unhealthy(낙상 · 선속도 실패) → `failure_present` |
| Nemotron 첫 판단 | `continue`. 순위 reward > actuator > exploration > optimizer > physics > termination, probe `P_reward`(동결 규칙 대안 `P_noise`). 인용 근거 `telemetry:case:Train/mean_reward` 8.568, `telemetry:reference:Train/mean_reward` 20.474 |
| 검사 `8a8e2015dc2f` | `P_reward` abnormal(track_lin_vel_xy_exp_rel_error 0.307), GPU 벽시계 23.1 s |
| 고리 종료 | `confirmed`, 남은 가설 reward (검사 1회, 예산 4) |
| 특이성 관문 | 게이트 없음(reward), 통과로 기록 |
| Nemotron 교정 제안 | 검증 accepted, 규칙과 일치. 요청은 `restore_body_frame_reward` + `restart`(예산 2400 s) |
| 교정 요청 `7b09e4761730` | receipt `recovered`, gpu_s 654.6 |
| 교정 재학습·고정 평가 | restart, healthy(네 검사 모두 통과), 정지 대비 비 0.169 |

고정 평가:

| 정책 | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 대비 비 |
|---|---|---|---|---|---|
| NONE s2047 | 0.193 | 0.112 | 0.032 | 19.40 | 0.164 |
| F1 s2047 | 1.194 | 0.133 | 0.129 | 17.85 | 1.015 |
| restart 교정 | — | — | — | — | 0.169 |

사전 예측 7개 중 적중 7, 빗나감 0, 그 단계까지 가지 못함 0:
- 학습 실패: 적중
- 경고 R1: 적중
- 첫 판단: 적중
- 검사 고리: 적중
- 특이성 관문: 적중
- 회복 방식 규칙: 적중
- 교정 결과: 적중

성공 조건 12/12 참.

비용(부모 프로세스 벽시계):
- 고리 검사: `P_reward` GPU 벽시계 23.1 s·CPU 28.8 s·시뮬레이터 51,200 step
- 정상 기준 probe 6종 102.8 s
- 교정 학습 586.6 s, 교정 정책 고정 평가 68.0 s

해석 경계:
- 두 번째 로봇 task(ANYmal-C flat)의 새 seed 2047 합성 결함(F1) 1건을 시연 v4와 같은 흐름으로 끝까지 연결한 통합 시연이다. 결함과 probe는 같은 설계자가 만들었고, 진단 정확도·일반화·교정 성공률·GPU 절감의 근거가 아니며 Go2 결과와 수치를 비교하지 않는다.
- 판정 밴드는 정상 실행 3개로 정했다. 밴드 폭은 이 세 실행의 변동에 기댄다.

## 주장하지 않는 것

- 진단 정확도·우월성
- 일반화
- 교정 성공률
- GPU 절감
- 이미 본 seed 결과와 합친 분모
- Go2 시연 수치와의 비교
- 승인자 문자열을 신원 인증으로 표기
- 운영자 승인을 사람 승인으로 표기
- 재요청이 진단 내용(순위·probe)을 좋게 했다
- 정상 실행 3개(단계 0·1)와 시연 seed 하나(단계 2)로 이 task 전반의 동작을 말하지 않는다.

## 근거 파일

| 근거 | 파일 | LF SHA256 |
|---|---|---|
| 단계 0 프로토콜 | `bench/protocols/anymal_c_gate_v1.json` | `35c2b743e848…` |
| 단계 0 결과 | `evals/results/anymal_c_gate_20261014/result.json` | `989c81ed2b8d…` |
| 단계 1 프로토콜 | `bench/protocols/anymal_c_band_v1.json` | `054f288b4248…` |
| 단계 1 결과 | `evals/results/anymal_c_band_20261014/result.json` | `9db68f194fad…` |
| 단계 2 프로토콜 | `bench/protocols/e2e_demo_anymal_c_f1.json` | `db13cc86b03d…` |
| 고정 평가 프로토콜 | `bench/protocols/fixed_eval_v1_anymal_c.json` | `d5dbc5c92bb6…` |
| 단계 2 시연 결과 | `evals/results/anymal_c_e2e_f1_s2047_20261014/demo_result.json` | `e1a8f6045062…` |

## 재현

```powershell
.\.venv\Scripts\python.exe evals/build_anymal_doc.py --demo evals/results/anymal_c_e2e_f1_s2047_20261014/demo_result.json --check
```
