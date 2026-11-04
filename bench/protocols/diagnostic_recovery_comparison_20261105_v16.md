# 고정 학습 manifest를 보존하는 자원 비교 재시도

이 판은 [V14 비교 범위](diagnostic_recovery_comparison_20261105_v14.md)의 네 원천·네 방식·등록 rotation·같은 교정·300갱신·고정 평가·행동 기준을 유지한다. V14의 CPU 이름 충돌과 [V15 namespace 판](diagnostic_recovery_comparison_20261105_v15.md)의 학습 전 스키마 실패를 모두 보존한다. 모델 응답을 다시 받거나 행동 결과를 보고 기준을 고르는 재시도가 아니다.

## V15에서 확인한 실패와 소비

V15는 첫 `case-05/fixed_sequence`에서 등록 SDK 자식 1회를 실행했으나, 고정 학습기가 manifest의 추가 필드 3개를 거부했다. Simulator 앱·학습·고정 평가는 시작하지 않았다. SDK owned wall은 0.610초, 그 구간을 포함한 outer 부모 경과는 63.492919초였으며 종료와 소유 프로세스 정리를 확인했다. 두 시간을 더하지 않는다. 원래 첫 요청의 `unknown`과 나머지 미시작 행을 보존한다.

공개 근거는 `evals/results/balanced_recovery_v15_20261105_manifest_failure_lf_v2/summary.json`, raw LF SHA256은 `60992cb88d95ab15ab65e275cd35e9f2157122cd1e43ff4f646a754df31412f0`이다. 최초 CRLF 판은 보존하고 Git LF 바이트에 맞는 별도 판으로 재결합했으며 JSON 값과 실제 실행은 같다. 이 파일과 V15 소스·등록·실행·비용은 변경하지 않는다. App 시작 전 실패도 등록 SDK 시도 1회로 비용에 포함한다. 이를 정책의 미회복이나 새 회복 결과로 바꾸지 않는다.

## 수정할 소비 계약

- 새 계획 schema는 `registered_recovery_comparison_plan_v10`, profile은 `balanced_resource_manifest_retry_v1`이다.
- 새 모터 binding은 `registered_actuator_recovery_binding_v3`다. binding v1·v2와 계획 V1∼V9의 의미는 보존한다.
- 고정 `bench/train_with_correction.py`와 그 `MANIFEST_FIELDS`는 수정하지 않는다. 새 manifest는 기존 23개 필드만 가지며, namespace를 위한 키를 추가하지 않는다.
- 원래 요청 이름·실효 이름·namespace profile·binding SHA·manifest SHA는 별도 `run-namespace.json`에 둔다. schema는 `registered_actuator_run_namespace_v1`이며, 정확한 필드는 `schema`, `namespace_profile`, `original_run_name`, `effective_run_name`, `binding_sha256`, `manifest_sha256`이다.
- 부모와 자식은 sidecar와 binding·manifest의 SHA 및 결정적 실효 이름을 다시 검증한다. 부모는 자기 원래 request ID와의 일치도 확인한다. sidecar는 출처 결합 자료이며 실행 승인 파일을 대신하지 않는다.
- namespace의 재료는 V15와 같은 source commit·원천 checkpoint SHA·seed·원래 request ID·등록 profile이다. 로그를 덮거나 시각·난수로 기존 경로를 피하지 않는다.
- 실제 CPU 확인은 adapter의 자체 검증에서 끝내지 않는다. 고정 학습기의 `execute`를 소비해 스키마·cwd·protocol SHA·실행 전 설정을 검증한 뒤, 상류 `train.py` 실행 경계에서 멈춘다. 이 경계 확인과 실제 SDK Python의 ABI 확인은 CPU 사전 확인이며 App·학습·회복 증거가 아니다.

## 입력·분모·새 seed

원천 `case-05`∼`case-08`, reference, 원래 진단 manifest·검사 receipt·모델 응답, 회복 방식과 판단 규칙은 그대로다. 계획 16행 중 새 효과 15개와 원래 `case-07/iterative_agent` HOLD 1개를 유지한다. historical effect import와 모델 재요청은 0이다.

모터 이어 학습 seed는 원천 2664·2665를 유지한다. epoch 재학습 후보는 6666·6667로 실행 기록을 검사하고 등록한다. 앞선 6466·6467에서는 학습·평가를 시작하지 않았지만 CPU 실행 경계 확인 기록을 보존하며 이번 판의 새 학습 seed와 구분한다. 후보의 사용 이력이 있으면 이 판을 실행하지 않고 사유를 남긴다. 임의 대체 seed나 의미 실패 재요청은 없다. 새로운 행동 결과는 아직 관측하지 않았다.

## 남은 한도와 비용

기존 primary 32 SDK+post 1 SDK에 V15 실패 1 SDK를 더해 이미 소비한 합은 34회다. V16의 상한은 새 SDK 31회, CPU checkpoint 16회·telemetry 16회, registered owned 작업 합 63회, API 0회다. 합계 SDK 상한은 65회로 유지한다. 예정된 15학습+15평가가 모두 끝나면 새 실제 SDK는 30회지만, 이 예상치를 소비량으로 쓰지 않는다. CPU 사전 확인용 SDK 인터프리터와 App을 여는 등록 학습·평가 작업도 따로 기록한다.

V14·V15 준비·CPU 실패·V15 SDK 실패 및 부모 경과를 이전 비용에 보존한다. 새 준비와 실제 실행은 다른 구간으로 기록하고, 동시·중첩 구간을 더하지 않는다. 기존 공통 원천·진단·효과를 새 arm마다 재가산하지 않는다. 장치 표본은 coverage·누락·공유 부하를 남기며 provider·프로세스 비용으로 바꾸지 않는다. 전체 시간·사람 시간·가격·실청구 등 확보하지 못한 값은 null이다.

각 회복 부모 5,000초·학습 2,700초·고정 평가 900초·전체 owned 감독 14,400초의 상한은 유지한다. 공용 lease, 등록 admission, 별도 승인·한 번 소비, 실패 시 배치 중단, GPU 중 source/Git 고정도 유지한다. V15에서 소비된 요청은 다시 실행하지 않고 V16의 별도 등록 요청으로 시작한다. 새 실행·판정·양성 효과를 원래 결과로 합산하지 않는다.

완료 조건은 같은 입력·도구·예산·종료 기준과 실제 회전 순서에서 결과와 비용을 확보하는 것이다. 회복 0건·차이 없음·HOLD·unknown도 그대로 보고한다. 이미 본 원천의 개발 자원 확인이며 새 결함 일반화나 모델 우위로 부르지 않는다.
