# U1 보상 함수군 교정 범위 확장

U1은 외부 공개 결함의 의미를 Isaac Lab에 이식해 root 몸체 속도를 잘못 읽도록 네 보상 함수를 바꾼 사례다. 기존 seed 2040 시연은 `track_lin_vel_xy_exp` 한 항만 복원했다. `P_reward`에서 선속도와 yaw 추종 항이 모두 이상으로 측정돼 범위 점검이 `scope_insufficient`를 냈고, 교정 뒤 재학습 정책도 `not_recovered`였다. 이 기록은 [기존 U1 결과](../evals/results/e2e_demo_u1_s2040_20261013/demo_result.json)에 그대로 남아 있다.

이번 확장은 그 결과를 본 뒤 설계한 개발 등록이다. 알려진 U1 함수군 네 항을 모두 원본 구현으로 복원하는 `restore_reward_family_scope`를 [프로토콜](../bench/protocols/u1_scope_extension_v1.json)에 고정했다. 모델 판단 고리는 비활성이며 API 호출은 0회다.

## 범위 점검과 등록 항

`P_reward`가 측정하는 항과 소스·런타임 readback으로 확인하는 항을 구분한다.

| 함수 | 역할 | 이번 등록의 근거 |
|---|---|---|
| `track_lin_vel_xy_exp` | 선속도 추종 | `P_reward` 상대 오차 0.3904, 소스·readback |
| `track_ang_vel_z_exp` | yaw 추종 | `P_reward` 상대 오차 1.3198, 소스·readback |
| `lin_vel_z_l2` | 수직 속도 penalty | 소스·readback |
| `ang_vel_xy_l2` | roll·pitch 각속도 penalty | 소스·readback |

`P_reward`는 앞의 추종 보상 두 항만 측정했다. penalty 두 항은 probe로 이상을 측정한 항이 아니다. U1 주입 뒤 네 함수가 원본과 달라지고, 교정 뒤 네 함수가 모두 원본 지문으로 돌아오며, 환경의 활성 reward 항이 복원된 함수에 연결됐는지를 실행 계약으로 확인했다.

## seed와 실행 역할

| seed | 역할 | 학습 조건 |
|---:|---|---|
| 2040 | 기존 좁은 교정의 역사 결과 | 한 항 복원 뒤 `not_recovered` |
| 2051 | 새 실패 원본과 NONE 대조 | NONE·U1 각각 처음부터 300 update, NONE 함수군 교정 대조도 같은 seed로 300 update |
| 3051 | 확장 교정 정책 | U1 주입 뒤 네 항 복원, 처음부터 300 update |
| 2026 | 고정 행동 평가 seed | 학습 seed와 별개이며 모든 정책에 같은 26개 명령 조건 적용 |

학습은 환경 4096개, 환경당 rollout 24 step, 300 update로 고정했다. 평가는 `fixed_eval_v1`의 26개 명령 조건, 조건당 40환경, 1000 step, action scale 0.25를 재사용했다. source 실패는 seed 2051의 NONE과 U1을 비교하고, 회복은 seed 2051 NONE을 기준으로 seed 3051 확장 교정 정책을 판정한다.

## 결과

| 정책 | 학습 seed | 고정 평가 판정 | 선속도 RMSE (m/s) | 회전속도 RMSE (rad/s) | 생존 (s) | 낙상률 |
|---|---:|---|---:|---:|---:|---:|
| source NONE | 2051 | 기준·보행 확인 | 0.1463 | 0.2538 | 20.0 | 0.0 |
| source U1 | 2051 | `unhealthy` | 1.2232 | 5.5929 | 20.0 | 0.0 |
| NONE + 함수군 교정 | 2051 | `healthy`·보행 확인 | 0.1463 | 0.2538 | 20.0 | 0.0 |
| U1 + 네 항 복원 후 재학습 | 3051 | `healthy`, `recovered` | 0.1395 | 0.2352 | 20.0 | 0.0 |

[사실] seed 2051에서 NONE은 걸었고 U1은 선속도와 회전속도 오차 기준을 통과하지 못해 source failure가 확인됐다. NONE에 함수군 교정을 적용했을 때 바뀐 함수가 없었고, 최종 체크포인트 SHA도 source NONE과 같았다. 이 대조는 `healthy`와 보행 조건을 모두 통과했다.

[사실] seed 3051의 확장 교정은 U1이 바꾼 네 함수만 복원했고 등록 밖 함수는 바꾸지 않았다. 300 update 재학습 뒤 정책은 seed 2051 NONE 대비 네 행동 허용 조건을 모두 통과해 `recovered`로 판정됐다. 공개 요약은 [U1 범위 확장 결과](../evals/results/u1_scope_extension_s2051_20261015/summary.json)에 있다.

[추론] 기존 한 항 교정의 `not_recovered`와 이번 네 항 교정의 `recovered`는 교정 범위가 행동 회복에 영향을 줬다는 가설과 맞는다. 두 실행은 학습 seed와 실행 계약이 다르고, 이번 실험에는 seed 3051에서 한 항만 복원한 대조가 없다. 따라서 차이를 네 항 복원의 인과 효과로 분리할 수 없다.

## 처음 쓰는 seed에서 다시 확인

seed 2051 결과를 확인한 뒤 같은 함수군 교정을 source seed 2062와 restart seed 3062에 고정했다. 이 후속 확인도 개발 측정이다. 모델 판단 고리는 쓰지 않았고 API 호출은 0회다.

| 정책 | 학습 seed | 고정 평가 판정 | 선속도 RMSE (m/s) | 회전속도 RMSE (rad/s) | 생존 (s) | 낙상률 |
|---|---:|---|---:|---:|---:|---:|
| source NONE | 2062 | 기준·보행 확인 | 0.1365 | 0.2945 | 20.0 | 0.0 |
| source U1 | 2062 | `unhealthy` | 1.2411 | 5.2369 | 20.0 | 0.0 |
| NONE + 함수군 교정 | 2062 | `healthy`·보행 확인 | 0.1365 | 0.2945 | 20.0 | 0.0 |
| U1 + 네 항 복원 후 재학습 | 3062 | `healthy`, `recovered` | 0.1372 | 0.2965 | 20.0 | 0.0 |

[사실] seed 2062의 source NONE은 걸었고 source U1은 선속도와 회전속도 오차 조건을 통과하지 못했다. 네 항 교정을 NONE에 적용한 음성 대조는 원본 NONE과 같은 체크포인트로 `healthy`를 유지했다. seed 3062 교정 재학습 정책도 네 행동 조건을 모두 통과해 `recovered`로 판정됐다. [새 seed 확인 결과](../evals/results/novel_u1_scope_confirmation_s2062_20261016/summary.json)에는 원본 SHA와 고정 평가 수치를 함께 기록했다.

[추론] 2051과 2062에서 같은 판정이 나온 사실은 넓힌 함수군 교정이 한 번의 seed에만 묶인 현상이라는 우려를 줄인다. 두 source seed와 두 restart seed의 개발 측정이므로 성공률이나 일반화 근거로 쓰지는 않는다. 같은 seed에서 한 항 교정과 네 항 교정을 비교하지 않아 회복 원인도 분리되지 않았다.

## 해석 범위

이 결과는 알려진 U1 함수군을 본 뒤 만든 단일 결함 개발 사례다. source seed 2051·2062와 교정 재학습 seed 3051·3062를 확인했지만, 같은 seed에서 좁은 교정과 확장 교정을 비교하지 않았다. 모델이 교정을 발견하거나 선택한 실행도 아니고, 전체 모델 승인 고리를 검증한 결과도 아니다. 다른 보상 결함이나 로봇에 적용하려면 별도 등록과 평가가 필요하다.
