# 등록 회복 비교 V12 환경 종료 증거 재시도 사전등록

## 목적

V11의 `case-07/one_shot_llm`은 300 iteration 학습과 checkpoint 저장을 마쳤지만, 환경 종료 시점의 final readback이 봉인되지 않아 고정 평가를 시작하지 못했다. 이 실행은 simulator SDK 1회와 API 0회를 소비했으며 정책 결과는 unknown이다. 기존 결과와 요청 기록은 보존하고 완료 학습이나 회복 결과로 import하지 않는다.

V12는 정책 평가 결과를 보기 전에 확인된 인프라 실패에 대한 새 실행이다. 새 재시작 seed를 아래처럼 고정한다.

| case | resume seed | restart seed |
| --- | ---: | ---: |
| case-07 | 2666 | 6066 |
| case-08 | 2667 | 6067 |

## 불변 조건

- 계획 스키마는 `registered_recovery_comparison_plan_v6`이다.
- 완료된 motor 효과 8개와 기존 registry entry whitelist를 그대로 import한다.
- 새 회복 arm은 7개이며 `case-07/iterative_agent` HOLD를 유지한다.
- V11에서 소비한 SDK 1회를 과거 17회에 더해 prior 18회로 계산한다.
- 새 학습·고정 평가 예상은 14 SDK, 새 primary 상한은 14회, 누적 primary 상한은 32회다.
- CPU checkpoint 7회, telemetry 7회, 전체 owned launch 상한은 28회, API 상한은 0이다.
- source seed와 resume seed `2666`, `2667`은 바꾸지 않는다.
- 학습은 300 iteration을 유지한다.
- V11 checkpoint와 학습 시간은 비용 기록에는 남기지만 완료 학습, 정책 평가, 회복 결과로 합산하지 않는다.
- primary 누적 상한 32회가 모두 닫힌 뒤 필요한 post 확인은 별도 사전등록·별도 SDK 1회로 다루며 primary 상한을 소급해 늘리지 않는다.

## V11 실패 분류

- 학습 child는 natural exit 0과 cleanup을 남겼다.
- final checkpoint `model_299.pt`의 SHA는 공개 정책에 고정한다.
- final environment readback이 없어 학습 결과를 완료 import로 인정하지 않는다.
- 고정 평가는 시작하지 않았고 정책 결과는 unknown이다.
- 요청은 `ambiguous_consumed_side_effect`로 닫혔으며 자동 재실행을 허용하지 않은 기록을 보존한다.

## 실행 조건

- 공개 `registered_epoch_shutdown_retry_v1.json`의 V11 source·registration·plan·terminal·resource·dispatch error·checkpoint·supervisor SHA가 모두 일치해야 한다.
- 두 새 seed는 실행 직전 기존 seed guard를 통과해야 한다.
- training binding, 환경 recipe의 `training_seed`, 계획의 `recovery_policy.seeds`가 같은 restart seed를 가리켜야 한다.
- 8개 import, 7개 fresh, 1개 HOLD와 rotation 순서가 일치해야 한다.
- 위 결합이 하나라도 다르거나 새 seed가 이미 사용된 것으로 확인되면 실행하지 않는다.
- V12는 V11의 학습 checkpoint를 이어받지 않으며 새 학습으로 시작한다.

## 결과 문구

- 7개 fresh arm과 8개 import, 1개 HOLD가 모두 계약대로 닫히기 전에는 비교 완료를 주장하지 않는다.
- V11은 `학습 완료, 정책 평가 미실행, 결과 unknown`으로만 기록한다.
- 등록 범위 밖의 일반화, 우월성, 절감 효과를 주장하지 않는다.
