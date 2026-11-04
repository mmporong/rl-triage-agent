# 교정·회복 비교 v9: 불완전한 GPU 관측의 제한 재조회

## 변경 범위와 실행 전 예측

- v8 등록은 보존한다. v8의 네 운영자 시도는 학습·고정평가·API 소비 전에 종료됐다. a01·a02·a04는 GPU 관측 결합 실패였고, a03은 실행 출력 폴더 누락으로 lease 전에 종료됐다.
- XML process, GPU Engine counter, CIM process 조회는 서로 다른 시점의 관측이다. PID가 도중에 종료되거나 생성되면 결합에 실패할 수 있다. `nvgwls.exe`의 역할이나 안전성은 확정하지 않으며 허용 목록에 추가하지 않는다.
- 예측: 새 fresh 관측에서 기존 shared-v1 조건을 모두 만족하면 실행을 이어갈 수 있다. 일관된 관측을 얻지 못하면 효과 실행 없이 거부한다. 재조회 성공률이나 추가 회복 성공을 미리 주장하지 않는다.
- 새 profile은 `registered_desktop_shared_retry_v1`, parent schema는 기존 `registered_recovery_comparison_plan_v4`, tag는 `v9_consistent_observation`이다.
- 한 입장당 조회 최대 12회, 관측 재조회 창 60초, 재조회 간격 1초다. 진행 중인 OS 조회가 창을 넘겨 반환하면 그 관측을 acceptance로 사용하지 않는다. 외부 운영자와 arm의 기존 watchdog도 유지한다.
- 유효 lease, source/policy SHA, GPU·메모리·known workload·unknown identity 검사는 기존 shared-v1 그대로다. 불완전한 PID 관측만 재조회하며, 유효한 거부 관측이나 정책/소스/lease 오류는 성공으로 바꾸지 않는다. 모든 시도·실패·CPU·wall을 기록한다.
- 최종 fresh 관측이 기존 shared-v1의 accepted일 때만 다음 단계를 허용한다. 이전 accepted 관측을 다음 실행에 재사용하지 않는다. 거부 원인 삭제, 프로세스 종료, 기존 결과 수정은 없다.

## 과학 조건과 비용 결합

- v8과 동일한 완료 effect 두 개를 기존 C4 whitelist/역사 Git blob/원본 실행 영수증으로 재검증해 import한다. 이력 proof나 학습 결과를 다시 봉인하지 않는다.
- 미완료 학습을 import하지 않는다. v7 one-shot의 부분 checkpoint/환경/RNG 연속성을 가정하지 않는다.
- 16 arm 중 완료 import 2개, 새 confirmed arm 13개와 기존 model HOLD 1개다. 같은 입력·300 updates·seed·교정·고정평가 band·모델 원문을 유지한다.
- 과거 recovery SDK 소비 5개, 새 기대 26개, 새 cap 27개, 누적 상한 32개다. CPU checkpoint/telemetry 각 16개, total owned cap 59, API cap 0, arm당 5,000초, outer controller 14,400초를 유지한다.
- v8 입장 거부/운영자 실패, 준비 CPU, 대기·재조회 비용을 기록하고 전체 비용에서 빼지 않는다. SDK 소비와 운영자 시도를 구분한다. 종료 영수증 없는 v7 소비의 비용은 unknown을 유지한다.
- 공유 데스크톱에서 측정한 장치 전체 energy/wall은 독점 GPU 성능 근거가 아니다. WDDM의 process별 GPU memory `N/A`는 unknown이며 장치 전체 memory와 구분한다.
- 재조회는 자원 관측 복구이고, 모델 답이나 과학 판정의 재시도가 아니다. 회복·진단·HOLD 분모와 등록 판정은 바뀌지 않는다.

## 판정 문구

- 최종 입장 accepted: "기존 공유 입장 조건을 만족한 fresh 관측에서 실행했다."
- 재조회 소진/유효 거부: "입장 조건을 확인하지 못해 다음 실행을 시작하지 않았다."
- 회복 결과와 비교: 기존 등록된 recovered/not_recovered/HOLD 문구를 유지한다. 비용이 낮아도 회복하지 못했으면 절감 성공이라고 쓰지 않는다.

## 근거

- [NVIDIA NVML device process queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html): process 목록은 동적이며 compute와 graphics 조회는 별도다.
- [NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html): Windows WDDM process별 GPU memory 미제공과 통합 process 표의 타입을 설명한다.
- [Microsoft GPUView](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/using-gpuview): GPU 이벤트 시간축을 분석하는 진단 도구다. 성능 counter만으로 프로세스 실행 권한을 증명하지 않는다.

이 문서는 새 측정 전 등록이며 실행 완료나 회복 효과의 증거가 아니다. 외란·기립·계단 영상은 이 비교에 추가하지 않는다.
