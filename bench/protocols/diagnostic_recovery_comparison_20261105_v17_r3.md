# V17 r3: 완료 학습 재사용의 실행기 소스 결합

V17 r2의 입력 생성은 GPU·API 실행 전에 `prior training source missing`으로 중단됐다. 실제 backend가 요구한 이전 학습 검증기와 continuation V3·V4의 소스·테스트 6개가 등록 source closure에서 빠져 있었다. 원래 등록과 부분 입력, 준비 실패 영수증은 보존한다. 이 준비의 부모 경과 339.791100초와 현재 프로세스 CPU 256.25초는 별도 준비 비용이다.

이 판은 다음 기존 파일을 source closure에 추가한다. 해당 검증기의 조건을 완화하거나 소스 내용을 바꾸지 않는다.

- `evals/registered_prior_training_v1.py`, `tests/test_registered_prior_training_v1.py`
- `evals/registered_recovery_continuation_v3.py`, `tests/test_registered_recovery_continuation_v3.py`
- `evals/registered_recovery_continuation_v4.py`, `tests/test_registered_recovery_continuation_v4.py`

계획을 파일로 쓰기 전에 실제 backend의 extension 검증을 완료 학습 재사용 V4 경로와 기존 fresh V3 경로에 적용한다. 필수 소스 목록을 fixture에서 대신 선언하지 않는다. 실제 r2 closure를 음성 대조로 사용해 누락을 거부하고 보완한 결합을 수용하는지 확인한다. 이후 실제 private 입력 생성과 CPU 소비 확인도 수행한다.

고정된 후처리 V2와 그 테스트·CPU 상한 정정 문서도 새 closure에 포함한다. 상한 각 8회를 기대 횟수로 바꾸지 않으며 실제 checkpoint·telemetry 횟수를 보고한다.

seed 6666·6667, 원천·참조·진단·교정·300회 학습·회복 기준·순서, 완료 효과 7개와 완료 학습 1개 재사용, 원래 HOLD 1개를 유지한다. 신규 SDK 예상 15/상한 16·누적 상한 65·API 0·기존 GPU 입장·승인·단회 소비 조건도 그대로다. CPU 비실행 seed 증거는 앞서 고정한 개별 원문 하나만 분리한다.

새 등록은 미사용 실행 폴더에서 시작한다. 이전 학습을 다시 실행하거나 이전 unknown을 덮어쓰지 않는다. 실제 평가 결과가 없었던 준비 실패를 의미 재시도로 부르지 않는다. 결과가 미회복이어도 그대로 보고하며 이전 실패·준비·재개 비용을 보존한다.
