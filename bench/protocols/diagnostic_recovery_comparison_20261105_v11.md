# 등록 회복 비교 V11 재시작 seed 사전등록

## 목적

V10 작업 준비와 병렬로 실행된 CPU preflight가 `case-07`의 재시작 seed `5666`을 사용 기록에 남겼다. 이 기록은 학습 시작 전 발생했으며 simulator SDK 실행과 정책 실행은 각각 0회였다. 기존 기록과 seed guard는 보존하고 우회하거나 삭제하지 않는다.

V11은 아직 결과를 보지 않은 새 재시작 seed를 아래처럼 고정한다.

| case | resume seed | restart seed |
| --- | ---: | ---: |
| case-07 | 2666 | 5866 |
| case-08 | 2667 | 5867 |

## 불변 조건

- 계획 스키마는 `registered_recovery_comparison_plan_v5`를 유지한다.
- 완료된 motor 효과 8개와 검증 proof를 그대로 import한다.
- 새 회복 arm은 7개이며 `case-07/iterative_agent` HOLD를 유지한다.
- 학습은 300 iteration, GPU SDK 예상 14회, 새 상한 15회, 누적 상한 32회다.
- CPU checkpoint 7회, telemetry 7회, 전체 owned launch 29회, API 상한 0을 유지한다.
- source seed와 resume seed `2666`, `2667`은 바꾸지 않는다.
- V10은 CPU preflight 통과 뒤 작업 준비의 seed 검사에서 거부된 기록으로 보존하며 새 회복 결과로 합산하지 않는다.

## 판정

- 두 새 seed는 실행 직전 기존 seed guard를 다시 통과해야 한다.
- seed 검사와 작업 준비를 먼저 마친 뒤 CPU preflight를 순차 수행하고, 두 작업을 병렬로 실행하지 않는다.
- training binding, 환경 recipe의 `training_seed`, 계획의 `recovery_policy.seeds`가 같은 restart seed를 가리켜야 한다.
- 위 결합이 하나라도 다르거나 새 seed가 이미 사용된 것으로 확인되면 실행하지 않는다.
- V11은 인프라 단계에서 결과를 보지 못한 재시도이며 기존 V10 결과를 대체하거나 성공으로 간주하지 않는다.
