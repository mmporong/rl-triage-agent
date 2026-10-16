# PPO update 계약 계측

동결 행동 probe는 학습된 정책의 출력과 체크포인트를 검사하지만, PPO가 설정된 횟수만큼 실제로 갱신됐는지는 보여주지 않는다. `evals/optimizer_observability.py`는 update마다 미니배치 수와 `optimizer.step()` 호출 수를 기록해 실행 전 manifest의 기대값과 대조한다. R1은 RSL-RL #38을 재현한 결함으로, 설정된 PPO epoch 수를 무시한다.

## 검사 계약

[프로토콜 v1](../bench/protocols/optimizer_observability_v1.json)은 `learning_epochs=5`, `mini_batches=4`에서 update당 20회가 실행돼야 한다고 고정한다. 각 arm은 환경 4096개로 새 학습을 3 update만 실행한다. manifest는 arm·seed·task·예상 횟수와 소스 SHA를 담고, 각 JSONL record가 그 manifest를 가리킨다.

검사는 등록된 기본 단위를 순서대로 조립한다. 지금 최종 판정에 쓰는 항목은 미니배치 수와 optimizer step 수다.

| 기본 단위 | 현재 상태 | 쓰임 |
|---|---|---|
| optimizer step 수 | 판정 | 기대 횟수와 실제 호출 수 비교 |
| 미니배치 수 | 판정 | 기대 횟수와 실제 생성 수 비교 |
| Adam step delta | 조건부 판정 | 첫 update는 이전 Adam step이 없어 `unknown` |
| gradient norm | 기록 | `clip_grad_norm` 뒤, optimizer step 직전 값을 임계값 없이 보존 |
| learning rate | 기록 | step별 learning rate 궤적 보존 |
| approximate KL | `unavailable` | RSL-RL 2.3.3의 지역 변수라 현재 공개 hook에서 읽지 못함 |
| clip fraction | `unavailable` | RSL-RL 2.3.3이 계산 결과를 기록하지 않음 |

두 횟수가 모든 관측 update에서 기대값과 같으면 `normal`, 하나라도 다르면 `abnormal`이다. update 누락이나 계측 오류가 있으면 `unknown`으로 남긴다. 여기서 `normal`은 PPO update 횟수 계약이 맞았다는 뜻이며 정책이나 다른 결함이 정상이라는 판정이 아니다.

## 개발 측정 결과

R1과 NONE의 seed 2042 비교, NONE과 숨은 결함 h01~h06의 seed 2026~2028을 합쳐 23개 실행을 측정했다. 모두 3 update의 실행·계측 계약을 마쳤다.

| 묶음 | 실행 수 | `normal` | `abnormal` | `unknown` |
|---|---:|---:|---:|---:|
| R1 비교: NONE·R1, seed 2042 | 2 | 1 | 1 | 0 |
| 횟수 특이성 확인: NONE·h01~h06, seed 2026~2028 | 21 | 21 | 0 | 0 |
| 합계 | 23 | 22 | 1 | 0 |

[사실] NONE seed 2042는 runtime 설정 `learning_epochs=5`, `mini_batches=4`였고 세 update 모두 미니배치와 optimizer step이 각각 20회였다. R1은 runtime에서 `learning_epochs=1`, `mini_batches=4`로 읽혔고 세 update 모두 두 횟수가 각각 4회였다. 첫 불일치는 update 0이었다. [공개 요약](../evals/results/optimizer_observability_gpu_20261015/summary.json)에 실행별 판정이 있다.

[사실] seed 2026~2028의 NONE과 h01~h06은 21개 실행 모두 횟수 계약이 `normal`이었다. 이는 이 결함들이 정상이라는 뜻이 아니라, 등록된 다른 결함이 update 횟수를 바꾸지 않았다는 관찰이다.

[추론] 짧은 update 계측은 R1처럼 반복 횟수를 바꾸는 구현 결함을 행동 평가 전에 드러낼 후보 검사다. 이 개발 측정은 이미 본 R1의 seed 2042 한 개를 비교했다. 모델이 이 검사를 선택하는 진단 고리는 다시 실행하지 않았고, 이 결과만으로 다른 PPO 구현 결함을 찾는다고 볼 수 없다.

## 처음 쓴 seed에서 확인

R1 결과를 본 뒤 검사 규칙을 고정하고, 처음 쓰는 seed로 개발 확인을 한 번 더 진행했다. R1과 짝을 맞춘 NONE은 seed 2060, 별도 정상 특이성 확인은 seed 2061을 썼다. 각 검사는 새 학습의 PPO update 세 개만 계측했다. 300-update 학습은 체크포인트 생성과 정상 종료 확인에 사용했으며, 아래 횟수는 그 학습 전체를 센 값이 아니다.

| 사례 | seed | update별 기대 step | update별 관측 step | 판정 |
|---|---:|---:|---:|---|
| NONE | 2060 | 20·20·20 | 20·20·20 | `normal` |
| R1 | 2060 | 20·20·20 | 4·4·4 | `abnormal` |
| 별도 NONE 특이성 확인 | 2061 | 20·20·20 | 20·20·20 | `normal` |

R1은 첫 update부터 설정된 `learning_epochs=5`, `mini_batches=4`의 기대치 20회와 달랐다. NONE 두 실행은 세 update 모두 기대 횟수와 같았다. 세 사례의 사전 예측도 관측 판정과 일치했다. 공개 결과와 원본 SHA는 [새 seed 확인 결과](../evals/results/novel_optimizer_confirmation_20261016/summary.json)에 묶었다.

이 확인은 optimizer epoch/update 횟수 검사에 한정된다. 모델이 검사나 회복 방식을 고르는 고리는 실행하지 않았다. 처음 쓴 R1 seed 한 개와 NONE 두 실행의 개발 결과이므로 다른 PPO 결함, 일반화, 비용 절감이나 모델 우월의 근거로 쓰지 않는다.

## 검사에서 교정·회복까지 연결한 실행

새 seed 2071에서는 갱신 횟수 검사를 진단 고리에 넣고 epoch 복원을 등록해, 원인 확인부터 교정 재학습과 고정 평가까지 마쳤다. [종단 결과](../evals/results/r1_repair_s2071_20261016/summary.json)는 `outcome=recovered`, `demo_success=true`이며 사전등록 성공 조건 13개를 모두 통과했다. [사전등록 v2](../bench/protocols/r1_repair_demo_v2.json)는 앞선 R1 결과를 본 뒤 만들었고, 알려진 결함 유형을 새 seed 하나로 확인한 개발 시연이다.

기존 결정 모델 `nvidia/nemotron-3-super-120b-a12b`는 처음에 reward를 1위, optimizer를 4위로 두고 `P_reward`를 제안했다. 검사 결과가 normal이어서 reward 가설을 제거했고, 이후 등록된 선택 규칙에 따라 검사를 이어 갔다. 마지막 갱신 횟수 검사에서 optimizer의 update-count 하위 가설이 확정됐다. 첫 검사 선택은 모델이, 관측에 따른 가설 제거와 후속 검사 순서는 등록 규칙이 담당했다.

| 순서 | 승인 검사 | 관측 | 제거된 가설 |
|---|---|---|---|
| 1 | `P_reward` | `normal` | reward |
| 2 | `P_value` | `normal` | optimizer-value |
| 3 | `P_physics` | `normal` | physics |
| 4 | `P_episode` | `normal` | termination |
| 5 | `P_noise` | `normal` | exploration |
| 6 | `P_optimizer_updates` | `abnormal` | actuator |

새 검사는 승인 후 같은 구성과 seed로 별도 초기화한 PPO update 세 번을 관측했다. R1의 optimizer step과 미니배치는 매번 4회, NONE 대조는 매번 20회였다. 원 결함 checkpoint를 불러오는 검사가 아니며, checkpoint SHA는 계보를 연결한다. 교정은 `learning_epochs=1`을 등록값 5로 복원하고 seed 3071에서 300 update를 새로 학습했다. 교정 학습 첫 세 update에서는 각각 20회가 기록됐다.

| 고정 평가 정책 | 학습 seed | 수평 속도 RMSE (m/s) | 회전 속도 RMSE (rad/s) | 평균 생존 (s) | 낙상률 |
|---|---:|---:|---:|---:|---:|
| 정상 기준 NONE | 2071 | 0.1464 | 0.2779 | 20.0 | 0 |
| R1 결함 | 2071 | 1.1692 | 0.7926 | 20.0 | 0 |
| epoch 복원 후 새 학습 | 3071 | 0.1482 | 0.2695 | 20.0 | 0 |

교정 정책은 26조건 고정 평가에서 정상 기준 대비 네 행동 조건을 통과했다. 생존 시간과 낙상률이 같아도 결함 정책의 명령 추종은 실패했고, 교정 후에는 추종 오차도 허용 범위 안에 들어왔다. 진단과 교정 모델 응답은 각각 첫 응답에서 유효해 총 2회 호출했으며 피드백 재요청은 사용하지 않았다.

이 판은 기존 top-3 가설·검사 예산 4 대신 모든 범주를 내부 7가설로 펼치고 최대 7검사를 허용한다. 기존 여섯 검사의 임계값과 가설 갱신·확정 함수는 유지했다. 교정 정책의 비교 기준은 시연 v4와 같은 원 seed 2071의 NONE 정책이며, R1에서 이어 학습과 재학습을 비교한 실행은 아니다.

V1(seed 2070)은 학습 전 실행 폴더 생성 누락으로 invalid였으며, 해당 기록을 보존하고 새 판·새 seed로 실행했다.
