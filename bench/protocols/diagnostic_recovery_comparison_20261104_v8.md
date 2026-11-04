# 교정·회복 비교 v8 인프라 후속 사전등록

- parent plan schema는 `registered_recovery_comparison_plan_v4`, tag는 `v8_resumable`이다.
- 완료 import는 `case-05/fixed_sequence`와 `case-05/static_rule` 두 개다.
  - fixed는 기존 v5 C3 proof_v1을 C4의 새 whitelist 검증 뒤 C3 verifier로 위임한다. v7의 imported result를 다시 source로 삼지 않는다.
  - static은 v7 actual result에서 만든 proof_v2를 C4가 검증한다.
- v7 static training-only import는 제거한다.
- v7 one-shot은 iteration 528/599, `model_500.pt`, completion receipt 없음이므로 interrupted로 보존하고 import하지 않는다. 요청을 다시 소비하지 않는다.
- 새로 실행할 confirmed arm은 13개다. 기존 model HOLD 1개를 포함한 16-grid 구조는 유지한다.
- 실제 과거 recovery SDK 소비는 5개다: v4 training 1, v5 fixed 1, v5 static training 1, v7 static fixed 1, v7 one-shot interrupted 1.
- 새 기대 SDK 26, 새 cap 27, 누적 32다. CPU checkpoint 16 + telemetry 16을 유지해 total owned cap은 59다.
- API cap 0, 300 updates, 같은 seed/band/model 원문을 유지한다.
- 새 13개 arm은 기존 `registered_recovery_protocol_v3`와 `registered_desktop_shared_baseline_v1`을 사용하므로 backend 변경이 필요 없다.
- 정책/optimizer 전체 상태, 환경 전체 상태, RNG 전체 상태 복원은 주장하지 않는다.

## 검증 경계

- history source는 source→registration→HEAD ancestor와 source anchor의 closure 135개 Git blob exact SHA로 검증한다.
- 현재 working tree를 역사 source 권한 근거로 사용하지 않는다.
- checkpoint/config/SDK/runtime/correction/actual receipt는 기존 backend 및 post-observation verifier로 현재 실제 파일을 다시 검증한다.
- history helper는 read-only이고 실행·승인 callback이 없다.


## 실행 수명과 소비 경계

- v7은 부모·임대 자식·Kit가 OS 관측에서 부재이며 종료 영수증이 없다. 종료 원인은 미확정이다. 완료된 두 effect와 중단 학습의 원문·비용은 보존한다. 종료 코드나 소유 GPU wall을 추정하지 않는다.
- 새 controller는 도구 부모 Job 밖 base Python supervisor에서 시작한다. supervisor는 기존 WindowsJob 실행기를 재사용해 임대 wrapper와 그 실행 트리만 소유하고, 공용 Use-GpuLease를 거친다. 운영자 wrapper의 전체 상한은14,400초이며 원래 arm별5,000초 상한과 SDK/API cap도 각각 유지한다. 이 wrapper는 SDK 실행 수가 아니다.
- base supervisor가 Job 밖에서 시작해 부모 반환 뒤18.005655초 정상 종료한 CPU 시험과, 별도 소유 Job에 worker를 배정한 뒤3.25초·exit0·트리 cleanup을 확인한 CPU 시험을 수행했다. 이 CPU 증거는 GPU 배치의 내구성이나 회복 효과를 입증하지 않는다.
- 중단 요청을 reset하거나 재소비하지 않는다. 새 출력·새 request에서 원래 source checkpoint로부터 등록된300회를 다시 수행한다. 부분 checkpoint에서 남은 횟수만 이어 전체 환경/RNG 연속성을 가정하지 않는다. 중단 시도는 별도 실패 비용으로 남긴다.
- 이전 시도·준비·queue·실패 비용을 전체 관측에서 제외하지 않는다. 완료 effect의 과거 학습·평가 비용은 전체 effect에 한 번, 새 dispatch 비용은 별도로 계산한다. 종료 영수증이 없는 소비는 execution_unsettled와 비용 unknown이며 not_started 또는0으로 바꾸지 않는다.
- 공유 데스크톱 관측이므로 장치 전체 전력/energy·wall은 독점 성능 측정이 아니다. 방법별 정확도·회복·HOLD와 비용을 함께 보고하고, 회복 없는 낮은 비용을 절감 성공으로 해석하지 않는다.
- 모터 사후 매핑 검사는 이 비교에 끼워 넣지 않는다. 필요하면 별도 source/사전등록·남은 상한 확인 뒤 추가하며, 기존 post_review와 회복 결과는 바꾸지 않는다.
