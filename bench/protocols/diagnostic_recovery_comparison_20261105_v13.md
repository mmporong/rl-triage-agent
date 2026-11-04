# 등록 회복 비교 V13 요청 namespace 재시도 사전등록

## 목적

V12는 첫 fresh arm을 시작하기 전에 이전 V11과 같은 전역 `run_name`을 재사용해 종료됐다. simulator SDK와 API는 사용하지 않았고 학습·고정 평가도 시작하지 않았다. V12 기록과 실패 원문은 보존하며, CPU preflight에 기록된 seed 6066과 기존 등록 기록을 보존하며 freshness 검사를 완화하지 않는다.

V13은 정책 결과를 보기 전에 확인된 인프라 실패에 대한 새 등록 실행이다. source·resume seed와 기존 완료 motor 8개는 유지하고 restart seed만 아래처럼 새로 고정한다.

| case | resume seed | restart seed |
| --- | ---: | ---: |
| case-07 | 2666 | 6266 |
| case-08 | 2667 | 6267 |

## 불변 조건

- 계획 스키마는 `registered_recovery_comparison_plan_v7`이다.
- 완료 motor 효과 8개, fresh 7개, `case-07/iterative_agent` HOLD 1개와 rotation 순서를 유지한다.
- V11에서 소비한 SDK 1회를 포함한 prior는 18회다. V12는 SDK 0회이므로 prior를 늘리지 않는다.
- 새 예상은 학습·고정 평가 14 SDK이고 primary 누적 상한은 32회다.
- CPU checkpoint 7회, telemetry 7회, 전체 owned launch 상한 28회, API 상한 0을 유지한다.
- 일곱 fresh 학습은 300 iteration이며 진단 원문·모델 기록·회복 기준을 바꾸지 않는다.
- primary가 닫힌 뒤의 post 확인 1 SDK는 별도 등록이며 primary 상한에 포함하지 않는다.

## V12 실패 결합

- 공개 계획에는 V12 실패 context SHA만 고정하고 개인·비공개 경로를 넣지 않는다.
- 실행 준비 job은 context와 error·resource·supervisor·plan·preflight 원문 SHA를 모두 재검증해야 한다.
- V12의 consumed ledger/request를 다시 소비하지 않는다. 새 session의 독립 요청은 같은 ID 바이트를 가질 수 있으며, 학습 이름은 검증 seed와 환경 recipe로 분리한다.
- 새 session·plan·tag와 새 seed 6266·6267만 사용한다.

## 결과 문구

- V12는 `SDK 0, 학습·평가 미실행, request namespace 충돌`로만 기록한다.
- 8개 import, 7개 fresh, 1개 HOLD가 계약대로 닫히기 전 비교 완료를 주장하지 않는다.
- 등록 범위 밖의 일반화, 우월성, 절감 효과를 주장하지 않는다.
