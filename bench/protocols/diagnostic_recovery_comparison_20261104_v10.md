# 등록 회복 비교 v10: 완료된 모터 효과 보존과 epoch 실행 복구

## 상태와 목적

이 판은 v9 실행을 결과 선택 없이 이어 간다. v9는 case-05·case-06의 모터 결함 8개 효과를 모두 완료했고 결과는 모두 `not_recovered`였다. case-07 `fixed_sequence`는 학습 SDK가 시작되기 전에 owned 실행 폴더가 없어 중단됐으며, 해당 요청은 재사용하지 않는다. 학습이 시작되지 않았으므로 원래 restart seed는 소비되지 않았다.

목적은 완료된 8개 효과를 C4 증명으로 보존하고, 남은 7개 회복 효과만 새 요청과 `v10_epoch_repair` 태그로 실행하는 것이다. case-07 `iterative_agent`의 모델 HOLD는 회복 권한 없이 유지한다.

## 고정 입력과 실행 범위

- 완료 효과 프로필: `motor_completed_v1`
- 공개 등록표: `bench/protocols/registered_completed_motor_effects_v1.json`
- GPU 입장 프로필: `registered_desktop_shared_retry_v1`
- 학습 횟수: 300
- 모델 판단: 기존 저장 응답만 사용, 새 API 호출 0
- 완료 효과 가져오기: case-05·case-06의 네 방법, 합계 8
- 새 회복 효과: case-07의 `fixed_sequence`·`static_rule`·`one_shot_llm`, case-08의 네 방법, 합계 7
- 회복 권한 없음: case-07 `iterative_agent`, 합계 1

등록표는 완료 효과 결과와 무관하게 첫 두 case의 8개 전체를 고정한다. proof 본문이 스스로 기대값을 정할 수 없도록 artifact·case·method·proof SHA·계보를 별도 공개 파일에 고정하고, C4 검증기로 과거 source→registration→현재 HEAD 조상 관계와 등록 blob을 다시 확인한다.

## 자원 상한

- 과거 회복 SDK 소비: 17
- 새 회복 SDK 예상: 14
- 새 회복 SDK 상한: 15
- 누적 회복 SDK 상한: 32
- 새 CPU checkpoint: 7
- 새 CPU telemetry: 7
- 새 owned child 상한: 29
- API 상한: 0

가져온 8개 효과의 새 dispatch SDK는 0이며, 과거 비용은 한 번만 기록한다. v9의 case-07 실패는 SDK 0으로 기록하고 누적 상한을 늘리지 않는다.

## 순서와 비교 해석

V5 작업 생성기는 각 case의 등록 `rotation` 순서를 실제 arm 순서로 사용하며 실행기는 16행 전체의 정확한 순서를 검사한다. V1~V4 과거 작업은 당시 전역 방법 순서를 사용했으므로 그대로 수용한다.

따라서 v9까지의 시간 자료와 v10 시간 자료는 순서 설계가 섞인 기술 통계다. 과거 자료가 counterbalanced였다고 쓰지 않으며 방법별 속도의 인과 효과를 주장하지 않는다. 진단·검사·교정·회복 결과는 각 등록 arm의 결과로만 보고한다.

## 성공·중단 기준

- 준비 성공: 8 import / 7 fresh recovery / 1 HOLD가 정확히 구성되고 등록 rotation 순서와 일치한다.
- 실행 성공: 완료 효과 8개는 새 SDK 없이 재료화되고, 남은 7개만 새 요청을 소비한다.
- 중단: 등록표·proof·source closure·순서·자원 상한·승인 경계가 다르면 실행 전에 거부한다.
- 인프라 실패: 새 요청의 근거를 보존하고 자동 재실행하지 않는다.

