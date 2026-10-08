# P0-A2 결정적 기준선 계약 (2026-10-08)

LLM 없이 같은 입력으로 과제 B(원인을 모르는 실패의 메커니즘 순위)를 얼마나 푸는지 잰다. [AI-DAY-2026.md](AI-DAY-2026.md)의 철회 조건을 가장 싸게 확인하는 단계다([IMPLEMENTATION-ORDER.md](IMPLEMENTATION-ORDER.md) 3절).

- 구현: `src/rl_triage/rules.py`(규칙), `evals/baselines.py`(실행·기록), `tests/test_rule_baselines.py`.
- 동결: 이 문서와 위 파일을 커밋한 뒤 seed 123에 적용한다. `evals/baselines.py`는 규칙·입력 파일에 커밋하지 않은 변경이 있으면 dev 밖 seed를 거부한다.
- 채점: 에이전트와 같은 `rl_triage.scoring.score_blind`. 결과 jsonl은 `evals/replay.py`로 다시 채점할 수 있다.
- 모델·네트워크·GPU를 쓰지 않는다.

## 1. 기준선

| mode | 입력 정보 | 질문 |
|---|---|---|
| `rule_prior` | 없음(dev 정답 빈도순 고정) | 텔레메트리를 안 봐도 몇 개 맞나 |
| `rule_features` | 과제 B 에이전트와 같음: 실패·기준 텔레메트리, 기준 params | 정규화 특징 + 고정 규칙이 에이전트만큼 맞히나 |
| `rule_template` | 위 + dev seed의 정답 범주 | 같은 결함 템플릿을 다시 알아보는 것만으로 맞나(seed holdout의 일반화 시험 여부) |
| `rule_b0` | 실패 실행 설정과 기준 설정의 차이 | 설정을 볼 수 있으면 LLM 없이 끝나나(2절 B) |

`rule_template`은 평가하는 seed의 정답을 쓰지 않는다. dev seed끼리는 서로의 정답만 쓰고(7↔42), seed 123은 7·42의 정답을 쓴다. `rule_b0`는 정보 조건이 달라 에이전트와 같은 칸으로 비교하지 않는다.

## 2. 정규화 특징

같은 seed 기준 실행 대비 학습 후반 20% 평균의 비다. case_id는 쓰지 않는다.

| 특징 | 정의 | 이유 |
|---|---|---|
| `len_ratio` | `Train/mean_episode_length` 비 | 생존 시간 |
| `rate_ratio[항]` | `Episode_Reward/항` 비 ÷ `len_ratio` | 보상 항은 에피소드 합 ÷ 제한 시간이라 빨리 넘어지면 모두 작아진다. 초당 비로 바꾼다 |
| `timeout_frac` | time_out ÷ (time_out + base_contact) | 종료 항은 비율이 아니라 리셋 개수다 |
| `len_pinned` | (전체 최대 길이 − 후반 평균) ÷ 후반 평균 | 모든 에피소드가 같은 길이에서 끝나는지 |
| `noise_ratio`, `vf_ratio` | `Policy/mean_noise_std`, `Loss/value_function` 비 | 탐색·가치 학습 |
| `tracking_rate` | 양수 가중 `track_*` 항의 `rate_ratio` 최대 | 넘어지는 정책이 정상보다 초당 추종 보상을 더 받을 수 없다. 더 크면 시간 간격이 가정과 다르다 |
| `positive_term_rate`, `torque_rate`, `max_rate`, `dominance`, `sign_flips` | 양수 가중 항 최대, `dof_torques_l2`, 전체 최대, 1위/2위, 부호 반전 항 | 보상 정의·하중 변화 |

## 3. 규칙

강한 증거 3점, 보조 증거 2점을 범주별로 더한다. 동점과 발화하지 않은 범주는 `rule_prior` 순서(reward, optimizer, physics, actuator, exploration, termination)로 정한다. "무너짐"은 `len_ratio < 0.5`, "즉시 붕괴"는 `len_ratio ≤ 0.022`다.

| 규칙 | 범주 | 점수 | 조건 |
|---|---|---|---|
| R1 | termination | 3 | 무너짐, `timeout_frac ≥ 0.5`, `len_pinned ≤ 0.1` |
| R2 | exploration | 3 | `noise_ratio ≤ 0.16` |
| R3 | optimizer | 3 | 살아남음, `vf_ratio ≤ 0.12` |
| R4 | reward | 3 | 보상 항 부호 반전 |
| R5 | physics | 3 | 무너짐, R1 아님, `tracking_rate ≥ 2.0` |
| R6 | reward | 3 | `dominance ≥ 20` |
| R7 | reward | 2 | 살아남음, `positive_term_rate ≥ 2.3` |
| R8 | actuator | 2 | 즉시 붕괴 |
| R9 | physics | 2 | 무너짐, 즉시 붕괴 아님, `timeout_frac < 0.5`, `torque_rate ≥ 1.7` |
| R10 | optimizer | 2 | 즉시 붕괴 아님, `vf_ratio ≥ 2.2`, `max_rate ≤ 79` |

**임계값 절차.** dev seed 7·42에서 규칙이 맞아야 할 사례(양성)의 가장 약한 값과 맞지 않아야 할 사례(음성)의 가장 강한 값 사이 기하 평균을 유효숫자 두 자리로 정했다. 음성은 규칙의 다른 조건을 통과하면서 점수가 더 높은 규칙이 이미 정답을 정하지 않는 dev 사례다. 0.5·0.1·0.5는 의미로 정한 값이다.

| 임계값 | 값 | dev 양성 | dev 음성 |
|---|---|---|---|
| `noise_collapse` | 0.16 | 0.038 | 최소 0.677 |
| `value_loss_collapse` | 0.12 | 0.017 | 최소 0.796 |
| `tracking_rate_up` | 2.0 | 최소 3.43 | 최대 1.16 |
| `single_term_dominance` | 20 | 최소 36.6 | 최대 11.2 |
| `positive_term_up` | 2.3 | 최소 4.66 | 최대 1.16 |
| `immediate_collapse` | 0.022 | 최대 0.0100 | 최소 0.0501 |
| `torque_rate_up` | 1.7 | 최소 2.89 | 최대 1.03 |
| `value_loss_up` | 2.2 | 최소 3.76 | 최대 1.32 |
| `value_loss_up_max_rate` | 79 | 최대 10.2 | 최소 620 |

테스트가 확인하는 것: 규칙 10개가 dev에서 모두 한 번 이상 발화한다. 다른 범주 규칙이 발화한 dev 사례에서는 정답 범주 점수가 더 높다. 각 임계값을 0.8배·1.25배로 옮겨도 dev 1위가 바뀌지 않는다. 정상 실행끼리 비교하면 아무 규칙도 발화하지 않는다.

**B0.** 기준 params에 case의 override를 적용해 바뀐 키를 구한다. 실행 식별 값(`agent.run_name`, `agent.seed`, `env.seed`)은 뺀다. 키 접두어로 범주를 정하고(`env.rewards.`→reward, `env.actions.`→actuator, `agent.policy.init_noise_std`→exploration, `agent.algorithm.`→optimizer, `env.sim.`·`env.events.`→physics, `env.episode_length_s`→termination 등, `rules.KEY_MECHANISM`), 범주 없는 키(체크포인트 주기·명령 샘플링)는 순위에 넣지 않는다. 변경 크기는 같은 부호 수치면 |ln(새 값/이전 값)|, 부호 반전·0 경계면 무한대다.

## 4. 설계자가 미리 알던 seed 123 정보

[IMPLEMENTATION-ORDER.md](IMPLEMENTATION-ORDER.md) 2절 C는 seed 123 trace에서 다음 값을 인용한다. 규칙 작성자는 이 값을 보고 규칙을 썼다.

- c02: 에피소드 길이 비 0.214, `dof_torques_l2` 비 0.269. 초당 비로 바꾸면 약 1.26으로 R9 임계값 1.7보다 작다.
- c10: `dof_torques_l2` 비 0.043.
- c08: value loss 비 0.016.

임계값은 3절 절차로 dev 값만 써서 정했고, 이 값에 맞춰 바꾸지 않았다. 사전 예상: seed 123 c02는 R9로 잡히지 않는다. 그 밖의 seed 123 텔레메트리는 동결 전에 열지 않았다.

이 정보 때문에 `rule_features`의 seed 123 점수는 규칙에 유리하게 기울 수 있다. 규칙이 에이전트보다 낮으면 에이전트 쪽 근거가 되고, 같거나 높으면 주장을 좁힌다. 최종 판단은 P0-C의 새 holdout에서 한다.

## 5. 해석 규칙 (seed 123 결과 전에 고정)

| 결과 | 해석과 조치 |
|---|---|
| `rule_features` top-1 ≥ 에이전트 6/10 | AI-DAY-2026.md 철회 조건: "에이전트가 규칙보다 잘 맞힌다"는 주장을 접고 README 과제 B 설명을 좁힌다 |
| `rule_features` top-1 < 6/10 | 같은 정보의 고정 규칙보다 에이전트가 높았다고 기록한다. 표본 10개라 유의성은 주장하지 않는다 |
| `rule_template` top-1 ≥ 9/10 | seed holdout은 같은 템플릿을 다시 알아보는 시험이라 일반화 근거가 아니다. P0-C 새 holdout의 필요 근거로 기록한다 |
| `rule_b0` top-1 = 10/10 | 설정을 볼 수 있는 조건에서는 현재 결함이 LLM 없이 풀린다. P0-C holdout은 설정 diff로 설명되지 않는 결함으로 구성한다 |
| `rule_prior` top-1과 대조군 2/10 | 대조군이 텔레메트리를 안 보는 빈도순보다 낮거나 같으면 그대로 기록한다 |

## 6. dev 결과

규칙을 dev에 맞춰 정했으므로 dev 점수는 성능 근거가 아니다. 결과는 `evals/results/p0a2_dev_20261008/`에 남긴다.

| mode | seed 7 top-1 / top-2 | seed 42 top-1 / top-2 |
|---|---|---|
| `rule_prior` | 3/10 / 5/10 | 3/10 / 5/10 |
| `rule_features` | 10/10 / 10/10 | 10/10 / 10/10 |
| `rule_template` | 10/10 / 10/10 | 10/10 / 10/10 |
| `rule_b0` | 10/10 / 10/10 | 10/10 / 10/10 |

B0 입력 대조(`--run-params`, Windows 로컬 Isaac Lab 로그): dev 실패 실행 20개 모두 실제 `params/env.yaml`·`agent.yaml`의 diff가 기준 params + override의 diff와 같았다(실행 식별 값 제외).

## 7. 한계

- 결함은 같은 10개 템플릿이고 seed만 다르다. Go2 flat 한 task, 100회 학습이다.
- 규칙은 사람이 dev를 보고 썼다. 새 결함 가족에 대한 일반화는 이 단계에서 재지 않는다.
- 에이전트 결과는 과거 실행(`evals/results/heldout_blind`)이고 다시 돌리지 않았다.
