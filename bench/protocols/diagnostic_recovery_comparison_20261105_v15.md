# 별도 자원 측정의 모터 학습 로그 이름 결합

이 판은 [V14 자원 비교 사전등록](diagnostic_recovery_comparison_20261105_v14.md)의 과학적 입력·순서·교정·학습 예산·판정·분모를 유지한다. V14 실행 계획은 `83ab63651f373854f1e0e435a41ade710e4823c9`, 소스는 `2eeb296cf1e93d6f570a52fd926d4af223f3d890`에 보존한다. V14는 실제 입력의 CPU 확인에서 멈췄으며 GPU·API·승인 소비는 모두 0회다. 새 행동 평가 결과를 보고 바꾸는 판이 아니다.

## 확인한 실행 전 문제

V14는 템플릿 준비 5.841488초, job 준비 311.805374초를 마쳤다. 그 뒤 실제 15개 요청의 로그 경로를 확인하던 중 첫 모터 arm의 이름이 과거 학습 로그와 일치해 `FileExistsError`로 중단했다. 원래 15개 회복 원장에는 `proposed` 이벤트만 있었고, 학습·검사·교정 실행은 시작하지 않았다. 이 CPU 실패·시간·원장을 보존하고 완료 효과로 세지 않는다.

원인은 모터 학습기가 의미상 request ID를 전역 RSL 로그 이름으로 쓰는 것이었다. 같은 원천·교정을 별도 자원 측정에서 다시 실행해도 그 ID는 같을 수 있다. 과거 로그를 지우거나 덮지 않고 새 측정의 소스 결합을 이름에 반영한다.

## 새 계약

- 새 plan schema: `registered_recovery_comparison_plan_v9`.
- plan profile: `balanced_resource_remeasurement_namespaced_v1`.
- 기존 V1∼V8 계획과 모터 binding v1의 의미는 유지한다. 새 V9에서만 모터 binding `registered_actuator_recovery_binding_v2`를 요구한다.
- binding v2의 `namespace_profile`은 `balanced_resource_remeasurement_v1`이다. 이는 새 로그 이름 구성요소의 등록 ID이며 모델이 정하는 인자가 아니다.
- 실효 로그 이름은 source commit SHA, 원천 checkpoint SHA, 학습 seed, 원래 request ID, namespace profile의 정규 JSON SHA로 결정한다. 원래 request ID는 접두사와 원장에 남긴다. 실행 경로·시각·임의 난수·사용자 경로를 이름 재료로 쓰지 않는다.
- 변경되는 것은 학습 로그 이름이다. 승인 request/effect, source checkpoint, 모델의 진단, 교정 ID·방식, seed·300갱신·환경·고정 평가·행동 기준은 그대로다. 모터의 이어 학습 seed는 2664·2665, epoch 재학습 후보는 6466·6467이다.
- 등록 후 job 준비 직전에 같은 core guard로 새 재학습 seed의 사용 기록을 확인한다. 사용 이력이 확인되면 멈추고 사유를 남긴다. 이 문서로 임의 seed 변경을 허용하지 않는다.

새 job은 16행의 등록 rotation, 새 효과 15개, 원래 `case-07/iterative_agent` HOLD 1개, 과거 효과 import 0개를 유지한다. 실제 요청 15개의 실효 로그 이름이 서로 다르고 기존 로그에 없음을 CPU에서 확인한다. 모터와 epoch 각각의 실제 소비 함수로 SDK 실행 직전까지 확인하되 이 확인은 GPU 측정으로 세지 않는다.

## 한도와 결과 해석

V14는 GPU 예산을 소비하지 않았다. V15의 새 상한은 V14와 같은 SDK 32회, CPU checkpoint 16회·telemetry 16회, owned 합계 64회, API 0회다. V14와 V15의 미실행 한도를 더해 64 SDK로 늘리지 않는다. 기존 primary 32+post 1과 새 측정의 합계 상한은 여전히 65회다. 예상 실제 SDK는 15학습+15평가=30회이며, 실제 영수증에서 계산하기 전에는 소비량으로 쓰지 않는다.

V14의 준비·실패 CPU 비용은 기존 이력에 한 번 더하고, V15의 새 준비·실행 비용과 구분한다. 공통 원천·진단·기존 효과를 반복 가산하지 않는다. 부모·owned·SDK·CPU·장치 표본의 중첩 범위를 유지하고 미측정 비용은 null이다. V15의 결과가 양성·음성·unknown이든 원래 진단과 V13 결과를 바꾸지 않는다. 이미 본 원천의 개발 자원 확인이며 새 결함·새 진단 일반화나 모델 우위 시험으로 부르지 않는다.

실제 GPU 전 소스·계획·예측·한도를 커밋한다. 공용 lease와 등록 admission, SDK/부모 watchdog, 요청의 한 번 소비, 실패 시 중단·원문 보존, GPU 배치 중 source/Git 고정은 V14와 같다. 추가 GPU·API·Brev Start·설치·WSL restart·외부 제출 권한을 이 문서에서 만들지 않는다.
