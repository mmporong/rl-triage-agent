# P0-B1 행동 기반 회복 판정 계약 (2026-10-08)

현행 회복 판정(`src/rl_triage/eval_bridge.py`의 `RECOVERY_BAND`)은 `Train/mean_reward` 비를 쓴다. 보상 가중치를 바꾸는 개입은 이 비율을 직접 움직이므로 판정이 개입에 오염된다([IMPLEMENTATION-ORDER.md](IMPLEMENTATION-ORDER.md) 2절 E). 이 단계는 Isaac을 다시 돌리지 않고 저장된 학습 텔레메트리에 보상 크기와 무관한 판정을 적용해, 현행 판정과 라벨이 달라지는 실행을 기록한다.

- 구현: `src/rl_triage/recovery.py`(판정), `evals/recovery_relabel.py`(적용·기록).
- 계약(이 문서와 위 코드)을 먼저 커밋한 뒤 전체 실행에 적용한다. 결과는 새 tag 폴더에 둔다.
- `eval_bridge.py`의 판정은 이 단계에서 바꾸지 않는다. 고정 평가 조건 실측(P0-B2) 뒤에 교체 여부를 정한다.
- 모델·네트워크·GPU를 쓰지 않는다.

## 1. 실행 설정

각 텔레메트리의 실행 이름으로 적용된 override를 찾고, 같은 seed 기준 params에 적용해 실행 설정을 얻는다.

| 이름 | override |
|---|---|
| `baseline_s<N>` | 없음 |
| `benign_all_s<N>` | `bench/catalog.py` BENIGN 6개 전부(`bench/run_batch.ps1`과 같음) |
| `c<NN>_s<N>` | `bench/cases.json`의 해당 case |
| `c<NN>_revert_ch<K>_s<N>` | 해당 case에서 K번째 override를 뺀 것 |
| `v2_S<NN>_s<N>` | `bench/catalog_v2.py` CANDIDATES의 해당 override |

같은 학습 실행을 가리키는 텔레메트리(`run_dir_name` 동일, `s02_baseline_s42`와 `baseline_s42`)는 하나만 센다.

## 2. 행동 지표

| 지표 | 정의 | 근거 |
|---|---|---|
| `survival_s` | `Train/mean_episode_length` × step_dt(= sim.dt × decimation) | 스텝 수는 sim.dt·decimation 변경에 따라 의미가 바뀐다 |
| `fall_frac` | base_contact ÷ (base_contact + time_out) | `Episode_Termination/<항>`은 리셋 개수다([termination_manager.py v2.1.1](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab/isaaclab/managers/termination_manager.py)) |
| `err_xy`, `err_yaw` | `Metrics/base_velocity/error_vel_*` × (재샘플 최대 시간 ÷ step_dt) ÷ 에피소드 스텝 수 | `UniformVelocityCommand._update_metrics`는 스텝 오차를 `max_command_step`으로 나눠 에피소드 동안 누적한다(v2.1.1 `velocity_command.py` 111~121행). 짧게 끝난 에피소드는 오차 값도 작게 기록된다 |

같은 seed 기준 실행 대비 `survival_ratio`, `err_xy_ratio`, `err_yaw_ratio`와 절대값 `fall_frac`을 쓴다. 학습 보상(`Train/mean_reward`, `Episode_Reward/*`)과 정책 노이즈는 쓰지 않는다.

## 3. 허용 범위와 판정

**정상 실행 비교값**: 서로 다른 seed의 기준 실행끼리(양방향) + `benign_all_s<N>` 대 같은 seed 기준 실행.

**허용 범위**: 비 지표는 1에서 가장 먼 정상 비교값까지의 거리에 0.10을 더한다. 생존 비는 아래쪽, 오차 비는 위쪽만 본다. 낙상 비율은 정상 최댓값에 0.05를 더한다. 범위는 실행 때 정상 실행에서 계산해 결과 파일에 남긴다.

**판정**

| 라벨 | 조건 |
|---|---|
| `unhealthy` | 생존 비·낙상 비율·오차 비 중 하나라도 범위 밖 |
| `undetermined` | 실행의 제한 시간이 기준(20초)보다 짧아 기준 시간 동안 버티는지 관측할 수 없고, 나머지 지표는 범위 안 |
| `healthy` | 모두 범위 안이고 제한 시간이 기준 이상 |

현행 판정과의 비교: `recovered=true`↔`healthy`, `recovered=false`↔`unhealthy`가 일치이고, 나머지(특히 `undetermined`)는 모두 변경으로 기록한다. v2 단독 변경 라벨(`bench/catalog_v2.py label`)도 같은 규칙(두 seed 모두 healthy면 benign, 모두 unhealthy면 harmful, 갈리면 ambiguous, 하나라도 undetermined면 undetermined)으로 다시 매긴다.

## 4. 계약 전에 본 값

- seed 7·42 실행과 v2 실행(seed 123 제외)의 행동 지표를 먼저 계산해 보고 여유값(0.10, 0.05)을 정했다. seed 123 실행은 P0-A2 holdout 적용 전이라 열지 않았다.
- [확인함] 정상 기준 실행도 100회 학습 후 선속도 추종 오차가 0.72~0.73 m/s(seed 7·42 후반 20%)다. 명령은 x·y 각각 ±1 m/s 균등분포라 제자리에 서 있을 때의 평균 오차가 약 0.77 m/s다. 정상 기준 실행은 넘어지지 않고 서 있지만 속도 명령은 거의 따르지 못하는 학습 초기 정책이다. 그래서 이 판정은 사실상 "넘어지지 않고 버티는가"를 본다. 걷기 품질은 학습을 더 오래 한 고정 평가 조건(P0-B2)에서 본다.

## 5. 한계

- 학습 텔레메트리는 각 실행 자신의 환경(제한 시간·명령 주기)에서 기록된다. 고정 평가 조건 실측이 아니다.
- 정상 실행 비교값이 seed 3개·무해 변경 실행 3개뿐이다. 같은 seed 반복 실행 변동은 없다(`s02_baseline_s42`는 `baseline_s42`와 같은 실행).
- 여유값은 판단으로 정했다. 다른 값이면 경계 근처 실행의 라벨이 바뀔 수 있다. 결과 파일에 비교값을 모두 남긴다.
- [확인함] Isaac Lab v2.1.1에서 `extras["log"]`는 리셋이 있는 스텝에서만 새로 채워지고(`manager_based_rl_env.py` 369행), rsl_rl은 매 스텝 그 값을 모은다. 리셋이 없는 스텝에서는 직전 리셋의 오차·종료 값이 다시 집계되어, 이 값들은 에피소드 평균이 아니라 시간 가중 평균에 가깝다. 에피소드 길이(`Train/mean_episode_length`)는 rsl_rl이 끝난 에피소드로 따로 세므로 둘의 가중 방식이 다르다. 고정 평가(P0-B2)는 환경 상태에서 직접 센다.
