# Brev 영상 자료 수집 사전등록 v5

v4는 점검용 seed 2203의 정상 정책 20회 학습을 완료했으나, 렌더 점검에서
`IsaacReadTimes` 등록을 찾지 못해 HOLD로 끝났다. 핵심 학습과 고정 평가는 실행하지 않았다.
기존 사전등록·실패 원문·종료 기록을 보존하고, 새 runtime `vision-source-20261025e`에서
확장 등록을 확인하는 worker를 사용한다.

- 소스: `vision_brev_source_20261025_v5.json`.
- 렌더: `vision_brev_render_20261025_v6.json`.
- 실행 전 점검: `vision_brev_preflight_20261025_v5.json`.
- 실행 진입점: `bench/run_vision_brev_annotated_v1.py`.
- 핵심 seed: 2210·2211. 별도 점검 seed: 2204. 실행 직전 미사용 검사를 다시 수행한다.

새 worker는 앱 시작 뒤 `isaacsim.core.nodes`를 활성화하고 앱을 한 번 갱신한 다음
`rgb`, `IsaacReadTimes`, `ReferenceTime` 등록을 확인한다. 등록 확인에 성공한 경우에만
기존 record 또는 렌더 동기화 검사를 실행한다. 실패하면 원문을 보존하고 HOLD로 멈춘다.
등록 확인 기록은 보조 파일이며, 기존 result·trace·동기화 판정 형식은 유지한다.
새 worker·backend·진입점 및 CPU 회귀를 소스와 controller 해시에 결합한다.

이 판의 예측·성공/실패 기준·결과 문구·교정 승인 경계는
[v4](vision_brev_source_20261025_v4.md)를 유지한다.
예측은 확장 등록 확인을 거쳐 렌더 점검이 완료되는 것이다. 통과 여부는 실제
A→B→A 동기화와 종료 증거로 판정하며, 등록 확인만으로 통과 처리하지 않는다.
통과 전에는 본 수집을 시작하지 않고 `model_use_ready=false`를 유지한다.
동기화 실패는 그대로 보존하며 허용 오차·픽셀·시각 기준을 바꾸지 않는다.

정상·결함·교정 각 두 건, 환경 4,096개·300회 갱신 여섯 건 상한은 동일하다.
별도 20회 점검은 환경 실행 확인이며 정책 성능 표본에 합산하지 않는다.
같은 소유 VM을 재사용하고 비용 시작 시각은 v2의 실제 시작 요청으로 유지한다.
이전 점검·실패·대기·회수를 포함한 전체 경과 시간에 20 USD compute 상한을 적용한다.
새 자원 증거는 이 판의 소스 protocol SHA를 결합하되 비용 시계를 재시작하지 않는다.
