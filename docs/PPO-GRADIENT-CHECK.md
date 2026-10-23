# PPO 계산 불변식 검사

## 확인할 질문

PPO의 epoch 수와 미니배치 수가 맞더라도 clipped surrogate 계산이 틀릴 수 있다. 이 검사는 갱신 횟수와 미니배치 수를 세는 기존 검사에 gradient 불변식을 더해 그런 오류를 구분한다.

이번 개발 측정은 `inverse_ratio` 인공 오류를 썼다. 이 오류는 optimizer step과 미니배치 수를 바꾸지 않으면서 PPO ratio 계산을 뒤집는다.

## 검사식

한 미니배치의 표본 수를 $N$, advantage를 $A_i$, 새 정책과 이전 정책의 log probability 차이로 만든 비율을 $r_i$라 두면 다음과 같다.

$$
r_i = \exp(\log \pi_\theta(a_i\mid s_i) - \log \pi_{old}(a_i\mid s_i))
$$

RSL-RL의 최소화 loss에서 clipping이 gradient를 통과시키는 조건은 다음 둘 중 하나다. clip 범위는 \(\epsilon=0.2\)로 고정했다.

$$
(A_i \ge 0 \land r_i \le 1 + \epsilon)
\quad \lor \quad
(A_i < 0 \land r_i \ge 1 - \epsilon)
$$

그때 새 log probability에 대한 표본별 기대 gradient는 다음과 같다.

$$
g_i^{expected} =
\begin{cases}
-A_i r_i, & \text{clipping이 gradient를 통과시키는 경우} \\
0, & \text{그 밖의 경우}
\end{cases}
$$

PyTorch가 돌려준 평균 loss gradient에 \(N\)을 곱한 값과 \(g_i^{expected}\)를 비교한다. 허용치는 `atol=1e-6`, `rtol=1e-5`다. 한 표본이라도 이 범위를 벗어나면 해당 update의 `ppo_surrogate_gradient`가 `abnormal`이 된다. `optimizer_step_count`와 `minibatch_count`는 각각 update당 20회를 기대한다.

## 등록한 실행 순서

결과를 보기 전에 다음 순서와 판정 기준을 고정했다.

1. seed 2143에서 1 update 선행 점검을 실행한다.
2. seed 2144, 2145, 2146에서 정상 PPO의 계측 OFF와 ON을 짝지어 실행한다. 학습 state와 loss가 같아야 하며, 세 PPO 갱신 구간 시간 합의 ON/OFF 비율은 각 seed에서 1.5 이하여야 한다. 이 시간에는 시작, rollout, 최종 상태 저장 구간이 들어가지 않는다.
3. seed 2144에서 `inverse_ratio` 오류를 실행한다. 횟수 검사는 `normal`, gradient 검사는 `abnormal`이어야 한다.
4. 같은 seed에서 원래 ratio 계산을 복원한다. 세 검사가 모두 `normal`이고 학습 state와 loss가 정상 대조와 같아야 한다.

본배치는 정상 OFF/ON 세 쌍, 오류 한 건, 복원 한 건으로 모두 8회다. 모델 API는 쓰지 않는다.

## 결과

판정은 `confirmed`였고 8/8 실행이 완료됐다. 등록한 조건 일곱 개가 모두 참이었다.

| 확인 항목 | 결과 |
| --- | --- |
| 정상 계측 비간섭 | seed 2144, 2145, 2146의 OFF/ON state와 loss가 각각 일치 |
| 정상 계측 비용비 | 세 PPO 갱신 구간 시간 합 기준 0.955, 1.155, 1.182로 세 seed 모두 상한 1.5 이내 |
| 정상 검사 | 세 seed 모두 `normal` |
| `inverse_ratio` 횟수 검사 | optimizer step과 미니배치 수 모두 `normal` |
| `inverse_ratio` gradient 검사 | `abnormal`, 첫 불일치 update 0, 검사한 batch 60 |
| 복원 검사 | optimizer step, 미니배치 수, gradient 모두 `normal` |
| 복원 학습 state와 loss | seed 2144 정상 대조와 모두 일치 |

공개 결과에는 집계와 각 실행의 manifest, 학습 증거, 관측 record, 검사 결과가 들어 있다.

- [집계 결과](../evals/results/ppo_gradient_v5_20261017/result.json)
- [공개 artifact](../evals/results/ppo_gradient_v5_20261017/)
- [사전등록 프로토콜](../bench/protocols/ppo_gradient_check_v5.json)
- [v4와 v5의 비용 판정](../evals/results/ppo_gradient_v5_20261017/v4_comparison.json)
- [CPU 호출 profile](../evals/results/ppo_gradient_v5_20261017/cpu_profile.json)
- [결과 재생 화면](replay/ppo-gradient.html)

## 관측 비용과 구현 변경

v4는 미니배치마다 GPU 값을 읽어 gradient 요약을 만들었다. 이 방식은 반복적인 host synchronization을 만들 수 있는 구조다. 다만 비용 초과분에서 이 구조의 기여도는 GPU trace로 분리하지 않았다. 등록한 비용 상한을 적용했을 때 세 PPO 갱신 구간 시간 합의 ON/OFF 비율이 1.564, 1.522, 1.632로 세 seed 모두 상한 1.5를 넘었다.

이 결과를 본 뒤 관측 구현만 바꿨다. v5는 backward hook에서 tensor snapshot을 모으고, update가 끝날 때 같은 device의 값을 묶어 CPU로 옮긴다. 지표, 허용치, 실행 순서와 비용 상한은 바꾸지 않았다.

Isaac Python의 같은 CPU fixture에서 3 update를 profile한 호출 수는 다음과 같다. 이 profile은 GPU 본배치 판정과 분리한 개발 측정이며, [공개 profile](../evals/results/ppo_gradient_v5_20261017/cpu_profile.json)에 함수별 caller와 측정 소스 SHA를 함께 기록했다.

| 호출 | 계측 OFF | v4 ON | v5 ON |
| --- | ---: | ---: | ---: |
| Tensor `.item()` | 726 | 2,211 | 726 |
| Tensor `.cpu()` 전체 | 85 | 85 | 88 |
| 이 중 관측 구현의 `.cpu()` | 0 | 0 | 3 |

v5에서 늘어난 `.cpu()` 세 번은 update당 한 번씩 관측 tensor를 묶어 읽은 호출이다.

v5 본배치에서 세 비용비가 모두 등록 상한 안에 들었고, 오류와 복원 판정도 유지됐다.

## 측정 범위

이 3 update 결과는 설계자가 만든 `inverse_ratio` 오류를 Go2 PPO 학습에 넣어 확인한 개발 측정이다. 외부 PPO 버그를 재현한 결과가 아니다. 이 검사는 학습 계산의 이상 여부를 기록하며, 정책 행동의 회복은 측정하지 않았다.

## Nemotron 3.5 Super VL과 연결한 종단 실행 v2

별도 사전등록에서는 새 source seed 2163의 정상 정책과 `inverse_ratio` 정책을 각각 300회 학습한 뒤 진단·승인 검사·등록 교정·재학습·고정 평가로 이어지는 흐름을 고정했다. 요청 모델은 `nvidia/nemotron-3.5-super-vl-120b-a12b`이며 반환 별칭 `private/nvidia/nemotron-3.5-super-120b-a12b`도 등록했다. 이 판은 앞서 본 파일럿 seed 2162의 관측과 기존 실행의 기록을 바탕으로 설계한 개발 확인이다.

26개 명령의 source 고정 평가에서는 정상 정책이 건강 기준을 통과했고, 오류 정책은 선속도와 yaw 추종 기준을 통과하지 못했다.

| 지표 | 정상 | `inverse_ratio` |
| --- | ---: | ---: |
| 선속도 RMSE (m/s) | 0.139 | 2.313 |
| yaw rate RMSE (rad/s) | 0.294 | 1.326 |
| 낙상 비율 | 0 | 0 |
| 평균 생존 시간 (s) | 20 | 20 |

모델은 보상을 1위로 두고 `P_reward`를 선택했다. 첫 응답은 등록표와 다른 예상 검사값으로 `D_EXPECTED`가 발생했고, 고정 피드백 한 번 뒤 유효해졌다. `P_reward`는 정상으로 나와 보상 가설이 제외됐고, 이후 등록 규칙이 선택한 `P_value`와 `P_physics`도 정상이었다. 네 번째 `P_episode`는 실행 후 환경 검증을 통과하지 못해 `unknown`으로 기록됐다. 종단 결과는 `invalid`이며 교정·재학습과 회복 판정은 수행하지 않았다.

모델 요청은 상한 8회 중 2회였다. 실행 receipt에서 확인한 부모 프로세스 벽시계 시간은 완료 단계 1,912.890초와 미수용 검사 단계 110.688초로, 합계 2,023.578초다. 화면 앱과 GPU를 공유하는 조건이며 GPU 커널 시간이나 독점 사용 시간으로 해석하지 않는다.

근거: [실행 요약](../evals/results/ppo_ratio_repair_s2163_20261023/summary.json), [26조건 평가](../evals/results/ppo_ratio_repair_s2163_20261023/evaluations.json), [모델 기록](../evals/results/ppo_ratio_repair_s2163_20261023/model_evidence.json), [검사·교정 원장](../evals/results/ppo_ratio_repair_s2163_20261023/probe_correction.json), [출처 결합](../evals/results/ppo_ratio_repair_s2163_20261023/provenance.json), [사전등록](../bench/protocols/ppo_ratio_repair_demo_v2.json).
