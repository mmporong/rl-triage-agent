# Brev 영상 자료 수집 사전등록 v6

v5는 점검 seed 2204의 정상 정책 20회 학습을 완료했으나 첫 프레임 예약 대기에서
렌더 제한 시간을 넘겨 HOLD로 끝났다. 종료 확인에서 소유 GPU 프로세스는 없었다.
핵심 학습·고정 평가·모델 호출은 실행하지 않았다. 기존 protocol과 결과는 보존한다.

실제 실행이 선택한 `isaaclab.python.headless.rendering.kit`은 Replicator의
`Orchestrator.enabled`를 끈다. 해당 SDK의 `_set_tick_graph()`는 이 값이 false이면
캡처 이벤트를 만들지 않으며, 두 번의 스택 관측은 첫 이벤트를 기다리는 같은 지점에 있었다.
새 판은 이 실행 설정을 명시하고 읽어 확인한다. 물리·시각·픽셀 판정 기준은 유지한다.

- 소스: `vision_brev_source_20261025_v6.json`.
- 렌더: `vision_brev_render_20261025_v7.json`.
- 실행 전 점검: `vision_brev_preflight_20261025_v6.json`.
- runtime: `vision-source-20261025f`.
- 핵심 seed: 2210·2211. 별도 점검 seed: 2205. 실행 직전 미사용 검사를 다시 수행한다.
- 실행 진입점: `bench/run_vision_brev_annotated_v1.py`.

앱과 Isaac 노드 확장을 시작한 뒤
`/exts/omni.replicator.core/Orchestrator/enabled=true`와
`/omni/replicator/captureOnPlay=false`를 설정하고 읽은 값이 일치하는지 검사한다.
첫 설정은 수동 캡처 이벤트를 허용하고 두 번째는 timeline 재생에 따른 자동 캡처를 막는다.
설정 확인에 실패하면 녹화에 들어가지 않고 실패 기록을 남긴다.
캡처 전에 job 옆의 보조 기록을 봉인해 외부 시간 제한 종료에도 준비 증거를 보존한다.
기존 worker v2에 이 준비 단계를 연결하고, 새 helper·worker·CPU 회귀를 새 source와
controller 해시에 결합한다. 기존 v5의 결합 해시와 실행 기록은 변경하지 않는다.

예측은 이 설정으로 첫 캡처 예약 대기가 끝나는 것이다. 통과 조건은 기존 A→B→A
동기화 검사와 종료 증거 전체이며 설정 확인만으로 통과하지 않는다.
실패하면 HOLD를 보존한다. 점검 통과 전에는 본 수집을 시작하지 않으며
`model_use_ready=false`를 유지한다. 실제 모델·정책 회복 효과는 이 점검의 평가 대상이 아니다.

그 밖의 예측·결과 문구·교정 승인 경계는 [v5](vision_brev_source_20261025_v5.md)를 따른다.
정상·결함·교정 각 두 건, 환경 4,096개·300회 갱신 여섯 건의 상한은 같다.
점검용 20회 학습은 정책 성능 표본에 합산하지 않는다.
같은 VM과 설치 환경을 재사용하며 비용은 v2의 실제 시작 요청부터 계산한다.
기존 점검·대기·실패·회수를 포함한 compute 상한 20 USD를 유지한다.
