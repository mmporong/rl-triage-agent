# Brev 영상 자료 수집 사전등록 v10

v9에서 자산 준비는 35회 update·2.15초에 완료됐고 물리 상태가 유지됐다.
SDK 종료도 시간 초과 없이 16.48초에 끝났다. A→B→A 카메라 복귀 화면은
픽셀 완전 일치 조건을 통과하지 못했다. 기존 원문·실패 판정·비용은 보존한다.
점검용 seed 2207은 소비됐으며 본 학습·고정 평가·모델 요청은 실행하지 않았다.

## 변경과 예측

실측 렌더 설정은 DLSS(`/rtx/post/aa/op=3`)였다. 설치된 Replicator 1.11.35는
비연속 데이터 생성에 프레임 누적이 없는 FXAA를 권장한다.
`rep.settings.set_render_rtx_realtime(antialiasing="FXAA")` 한 호출로 변경하고
AA=2, limitedOps=false, RayTracedLighting의 실제 readback을 검증한다.
설정 적용 전후와 캡처마다 읽은 값을 원문 manifest에 결합한다.
카메라 위치·학습·물리·시각·해상도·프레임 수·rt_subframes=1·delta_time=0·
pause_timeline=false는 유지한다. sampled lighting과 denoiser는 바꾸지 않는다.

가설은 DLSS 시간 누적이 화면 복귀 차이의 원인 중 하나라는 것이다.
FXAA가 완전한 픽셀 재현성을 보장한다는 주장은 하지 않는다.
이 변경으로 A→B→A의 동일 상태 화면이 완전히 같아질 것으로 예측하며,
기존 정확한 픽셀 일치와 시각·물리·종료·회수 조건을 모두 통과해야 본 수집을 허용한다.

## 실행·판정

- source `vision_brev_source_20261025_v10.json`, render `vision_brev_render_20261025_v11.json`,
  preflight `vision_brev_preflight_20261025_v10.json`, runtime `vision-source-20261025j`.
- 새 점검 seed 2208과 본 seed 2210·2211의 미사용을 실행 직전에 재검사한다.
- 동일 VM 원래 시작 시각부터 compute cap 20 USD. 핵심 학습 상한 6건 중 이번 source는 4건,
  교정 2건은 승인·검사 receipt가 있는 후속 단계에서만 허용한다. API 상한도 유지한다.
- 통과 문구: “FXAA로 고정한 카메라 점검에서 동일 상태 복귀 화면과 시각·물리·종료 조건을 통과했다.”
- 실패 문구: “FXAA만 적용한 점검에서도 동기화 캡처의 등록 조건을 충족하지 못해 본 수집을 보류했다.”
- 실패하면 같은 판에서 설정을 탐색하거나 허용 오차·subframes를 바꾸지 않는다.
  추가 설계는 원문 분석과 별도 사전등록 뒤 판단한다.

## 근거와 결합

- [Replicator 1.11.35 API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.11.35/source/extensions/omni.replicator.core/docs/API.html)
- 설치 소스 `omni/replicator/core/scripts/settings.py` SHA256
  `1c7e8096cfa82c0efb3d998ba39307609a55c1aa4cb01335e3ce8af431f0a5ea`, 332~360행.
- 설계자는 v9 실패 화면과 설정을 본 뒤 이 변경을 만들었다. 개발 환경 적합성 점검이며 모델 성능 측정이 아니다.
- 기존 capture/readiness/evidence/backend 파일은 새 판에서 위 설정 적용·검증만 추가했다.
  실행 전 소스·controller/test LF SHA를 다시 결합하며 기존 protocol·결과는 덮어쓰지 않는다.
