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

[추론] 짧은 update 계측은 R1처럼 반복 횟수를 바꾸는 구현 결함을 행동 평가 전에 드러낼 후보 검사다. 현재 근거는 이미 본 결함과 seed의 개발 측정이며, R1은 한 seed만 비교했다. 모델이 이 검사를 선택하는 진단 고리는 다시 실행하지 않았고, 이 결과만으로 다른 PPO 구현 결함을 찾는다고 볼 수 없다.
