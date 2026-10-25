# Brev 영상 자료 수집 사전등록 v8

v7은 점검 seed 2206의 정상 정책 20회 학습을 마쳤다. SDK가 반환한 시각 정보는
등록된 11필드와 `execOut`이었으며, 실행 포트를 분리한 기존 시각 검증을 통과했다.
네 캡처의 물리 시각과 로봇 상태는 유지됐지만 A→B→A의 마지막 화면이 첫 A와
일치하지 않았다. 회수한 RGB에서 바닥 재질 차이가 보이며, 자산 준비와 렌더 이력의
영향은 아직 분리하지 못했다. 앱 종료는 USD callback에서 반환하지 않아 부모의
제한 시간으로 끝났다. GPU 해제를 확인했으며 기존 결과와 사전등록은 보존한다.
본 학습·고정 평가·추가 모델 호출은 모두 0이다.

## 실행 결합

- 소스: `vision_brev_source_20261025_v8.json`.
- 렌더: `vision_brev_render_20261025_v9.json`.
- 실행 전 점검: `vision_brev_preflight_20261025_v8.json`.
- runtime: `vision-source-20261025h`.
- 핵심 seed: 2210·2211. 별도 점검 seed: 2207. 실행 직전 미사용을 다시 확인한다.
- 실행 진입점: `bench/run_vision_brev_annotated_v2.py`.

## 준비와 캡처

새 준비 함수는 환경과 render product 생성 뒤 자산 준비 상태를 읽고, 물리 step을
호출하지 않는 제한된 앱 갱신으로 준비 완료를 기다린다. 준비 전후 상태·시각과 SDK
loading/streaming 값, 렌더 설정을 보존한다. 자산 이벤트의 시작·완료·중단 개수를
대조하고 loading/streaming idle이 세 번 연속 관측될 때만 준비된다. 앱 갱신은 최대
240회이며, 동기 갱신 호출 사이에서 30초 polling budget을 검사한다. 한 호출 안의
정지를 30초에 끊는 watchdog는 아니며 기존 부모의 900초 제한을 유지한다.
각 앱 갱신은 설치된 Isaac Lab `SimulationContext.render()`와 같은
`/app/player/playSimulations=False` 범위에서 수행하고, `finally`에서 원래 값을
복원한다. 갱신 전·중·후 readback을 준비 이력에 기록한다. timeline과 캡처 인자는
변경하지 않는다. 설정 또는 복원 검증이 실패하면 준비도 실패한다.
준비 시간 초과나 상태 변화는 실패이고 실패 때의 readback도 보존한다.
카메라 위치를 적용한 뒤 `reset_renderer_accumulation()`을 호출하고 기존 수동 캡처를
실행한다. 원래 profile, 물리 조건, `rt_subframes=1`, 시각 허용 오차와 정확한 A→B→A
픽셀 일치 기준은 유지한다. 비교가 맞을 때까지 캡처를 반복해 결과를 선택하지 않는다.

이 경로는 SDK의 자산 준비와 렌더 누적 초기화 API에 근거한 수정이다. 통과하더라도
자산 준비 지연과 렌더 이력 중 하나를 단독 원인으로 확정하지 않는다. 실패하면
원문과 HOLD를 남기고 본 수집은 시작하지 않는다.

## 프로세스 종료

새 worker는 CPU supervisor와 SDK child를 분리한다. launcher 인자는 기존 profile과 같은
`fast_shutdown=False`이며 앱 생성 뒤 실제 설정값을 확인한다. 캡처 종료 후 결과·실패
중 하나를 먼저 기록하고, 별도 종료 정책이 `/app/fastShutdown`을 false에서 true로
바꾼다. 전후 readback과 job SHA, 먼저 기록한 terminal evidence SHA를 별도 영수증으로
봉인한 뒤 SDK close를 호출한다. 기존 profile의 launcher 필드는 시작 인자이고, 이 새
종료 단계의 전환은 해당 영수증과 source 결합으로 구분한다. supervisor는
같은 프로세스 그룹의 child가 끝날 때까지 기다린다. child 종료 코드가 0이고 원본 job이
변하지 않았으며 결과·모든 산출물의 bytes와 SHA, 종료 정책 영수증이 맞고 실패 기록이 없을 때만 0을
반환한다. 그 밖의 경우에는 실패 영수증과 비정상 종료 코드를 남긴다.

이는 extension별 정상 종료를 증명하는 경로가 아니다. SDK의 빠른 프로세스 종료를
사용하면서 실패 코드와 산출물 검증을 바깥에서 보존하는 경로다. 정책 설정이나 기록이
실패해도 SDK close를 시도하며 유효한 종료 영수증이 없으면 성공으로 처리하지 않는다.
supervisor는 프로세스를
강제 종료하지 않는다. 기존 부모 실행기의 timeout·소유권·GPU 해제와 결과 의미 검증은
유지하며, child가 살아 있는 동안 파일만 보고 완료로 처리하지 않는다.

## 예측과 판정

예측은 준비 완료와 렌더 누적 초기화 뒤 기존 A→B→A 검사를 통과하고, SDK child의
종료를 확인한 supervisor가 유효한 결과를 반환하는 것이다. 성공 조건은 시각·물리
상태·정확한 픽셀 일치·완전한 산출물·프로세스 종료·GPU 해제 전체다. 준비 기록이나
child exit 0만으로 성공하지 않는다. 점검의 `model_use_ready`는 계속 false다.

## 회수한 원문과 실행 영수증 결합

SDK child가 종료한 뒤 supervisor는 준비 상태, 캡처별 누적 초기화, 반환 시각 원문,
smoke RGB와 종료 정책을 `job.execution-evidence.json`에 봉인한다. smoke는 네 캡처,
녹화는 625프레임의 정확한 파일 집합을 요구한다. 시각 원문은 smoke report 또는 record
trace의 해당 캡처와 대조한다. 실행 영수증 v2의 `auxiliary`가 manifest SHA를,
기존 attestation이 실행 영수증 SHA를 결합한다. 모델 입력의 최상위 원본 참조에도
`execution_evidence`가 필요하다. 회수할 때 stage 전체와 각 원문을 보존하며 누락이나
해시 불일치가 있으면 입력을 만들지 않는다.

이 결합을 위해 기존 backend·attestation·회수·선택 입력 검증기에 새 실행 schema 분기를
추가한다. 기존 v1 실행 영수증의 필드와 판정은 유지하고, supervisor source를 포함한
새 판은 v2 영수증과 전체 보조 원문을 요구한다. source 목록·새 테스트를 이번 사전등록에
다시 결합한다. 과거 프로토콜과 결과 파일은 바꾸지 않는다.

그 밖의 예측·결과 문구·교정 승인 경계는 [v7](vision_brev_source_20261025_v7.md)을
따른다. 정상·결함·교정 각 두 건, 환경 4,096개·300회 갱신 여섯 건의 상한은 같다.
점검용 20회 학습은 정책 성능 표본에 합산하지 않는다. 같은 설치 환경과 VM을 재사용하고,
비용은 v2의 실제 시작 요청부터 계산한다. 이전 점검·대기·실패·회수를 포함한 compute
상한 20 USD는 유지한다.
