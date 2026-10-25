# Brev 영상 자료 수집 사전등록 v9

v8은 학습 시작 전에 backend 필수 소스 집합과 사전등록 집합이 달라 중단됐다.
학습·캡처·고정 평가·API 호출은 없었고 기존 원문과 판을 보존한다.
v9은 누락된 테스트 파일을 실행 소스 목록에도 결합한다. 준비 단계에서 실제 backend의
소스 대조를 수행하며, 생성된 protocol을 사용하는 factory 경계 회귀를 추가한다.

소스는 `vision_brev_source_20261025_v9.json`, 렌더는
`vision_brev_render_20261025_v10.json`, 점검은
`vision_brev_preflight_20261025_v9.json`이다. runtime은 `vision-source-20261025i`다.
점검 seed 2207과 본 seed 2210·2211은 실행 직전에 미사용을 다시 확인한다.

실행 코드와 renderer profile, 준비·시각·물리 상태·정확한 픽셀 일치·종료·회수 기준은
[v8](vision_brev_source_20261025_v8.md)을 따른다. 예측·판정 문구와 학습·API 상한도 같다.
비용은 같은 VM의 원래 시작 시각부터 계산하고 compute cap 20 USD를 유지한다.
점검 전체가 통과하기 전에는 본 학습이나 모델 입력을 시작하지 않는다.
