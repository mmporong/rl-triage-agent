# 회복 구간 실행 순서와 자원 비교 보완 사전등록

## 목적과 기존 결과

기존 V13의 진단·교정 결과와 원문은 보존한다. 전체 회복 분모 16행은 미회복 15행·원래 진단 HOLD 1행이었으며, 모터 8행은 과거 완료 효과를 검증해 가져온 것이다. 이 8개의 실제 실행은 등록한 회전 순서를 소비하지 않았다. 진단 비교 24행과 V13 새 epoch 효과 7개의 순서는 별도 기록에 남아 있다. 이 문서는 알려진 음성 결과를 양성으로 바꾸기 위한 재시도가 아니라, 같은 진단 선택에서 회복 구간의 순서·비용 조건을 맞추는 별도 측정을 등록한다.

- 보존하는 결과: `evals/results/recovery_comparison_s2664_s2667_20261105_v13/summary.json`.
- 위 파일의 raw SHA256: `1b28d0e0b98788508d65bbd1edfe3d8763ac5c6d5905715512ff403106f6c011`.
- 이미 본 source와 진단·회복 결과를 사용한 개발 자원 확인이다. 새 보류 계보·독립 진단 성능 평가로 세지 않는다.
- 기존 모델 응답·검사 결과·HOLD를 그대로 소비한다. 모델 재요청과 새 API 호출은 0이다. 과거 NIM 응답을 새 호출로 표시하지 않는다.
- 기존 결과·프로토콜·비용을 덮지 않는다. 기존 모터 효과 8개를 새 측정 결과로 가져오거나, 새 값으로 교체하지 않는다.

## 비교 단위와 순서

대상은 기존 `case-05`∼`case-08` 네 원천과 같은 reference다. source checkpoint, 진단 manifest, 실제 확인 receipt, 도구·registry, 회복 방식, 300갱신, 고정 26조건 평가와 행동 자격·상대 band를 동일하게 유지한다. source/runtime closure와 모든 입력 SHA는 실행 계획에서 고정한다. 사후 truth는 채점에만 쓰고 교정 승인에 전달하지 않는다.

| 사례 | 실행 순서 |
|---|---|
| case-05 | fixed_sequence → static_rule → one_shot_llm → iterative_agent |
| case-06 | static_rule → one_shot_llm → iterative_agent → fixed_sequence |
| case-07 | one_shot_llm → iterative_agent → fixed_sequence → static_rule |
| case-08 | iterative_agent → fixed_sequence → static_rule → one_shot_llm |

계획·job·각 실행 전 검증에서 위 순서를 강제한다. 과거 completed-effect/training import는 허용하지 않는다. 원래 case-07 반복 에이전트의 진단 HOLD는 새로운 학습이나 권한 없이 그대로 종결한다. 따라서 계획 16행 중 실제 신규 효과는 15개, 원래 HOLD는 1개로 예상한다. 16행·실행된 효과·HOLD를 각각 보고하며 반복 arm을 독립 seed로 세지 않는다.

| 확인된 원인 | 교정·회복 방식 | 학습 seed |
|---|---|---|
| case-05 actuator | torque-speed 함수 복원 후 이어 학습 300갱신 | 원천 2664 |
| case-06 actuator | torque-speed 함수 복원 후 이어 학습 300갱신 | 원천 2665 |
| case-07 optimizer count | configured epoch만 복원 후 재학습 300갱신 | 후보 6466 |
| case-08 optimizer count | configured epoch만 복원 후 재학습 300갱신 | 후보 6467 |

새 재학습 seed는 실행 기록을 검사하는 원래 core seed guard로 등록 전과 job 준비 직전에 확인한다. 사용 이력이 있으면 이 판을 실행하지 않고 사유를 보존한다. 임의 대체 seed는 없다. 모터 이어 학습 seed는 같은 원천에서 시작하는 의도적인 짝 조건이며 새 원천으로 표시하지 않는다. 재학습에서도 기존 보상 좌표계 등 교정 대상 밖 구성을 보존한다.

## 예측과 판정

1. 같은 source·교정·예산에서 방식에 따른 회복 우위를 예상하지 않는다. 기존 미회복과 같은 결과도 유효하다. 새 재학습 seed의 결과는 아직 관측하지 않았다.
2. 원래 진단 HOLD는 유지될 것으로 예상한다. 형식 실패·미확정을 재호출해 없애지 않는다.
3. 각 짝의 학습·고정 평가·실제 적용·확인 receipt·상태 복원·종료 증거를 검증한 뒤에만 recovered/not_recovered를 반환한다. 실행 실패와 unknown은 회복 실패로 임의 치환하지 않는다.
4. epoch 교정 첫 세 갱신의 step 수를 다시 검사하고, 원인 확인이 없는 남은 행동 실패는 조사 또는 참조 점검으로 보낸다. 행동 실패가 추가 교정 권한을 만들지 않는다.
5. 비교는 이 네 원천의 기술 통계다. 같은 결과의 wall·장치 표본 차이와 서로 다른 endpoint/HOLD까지의 비용을 분리한다. 작은 표본·동일 seed 반복·warm cache·장치 경합 때문에 인과적 절감·일반화를 주장하지 않는다.

## 실행·비용 상한

이 판은 기존 primary 32 SDK와 별도 post 1 SDK를 바꾸지 않는다. 새 상한은 GPU SDK 32회, CPU checkpoint 16회, CPU telemetry 16회, owned 자식 합계 64회, API 0회다. 예정된 15효과가 모두 끝나면 새 GPU SDK는 학습 15+고정 평가 15=30회다. 실패 때에도 사용량을 세고 상한을 늘리지 않는다. primary·post·이번 보완의 합계 GPU SDK 상한은 65회다. 개발·원천·진단을 포함한 앞선 캠페인 146회는 별도 보존하며 이 판의 분모로 바꾸지 않는다.

각 회복 부모 상한은 5,000초, 학습 2,700초, 고정 평가 900초, 전체 owned 감독 상한은 14,400초다. 전체 감독 밖의 준비·검증·lease 대기는 별도 계측한다. 같은 예산과 종료 계약을 네 방식 모두에 적용한다. 애매하게 소비된 요청은 unknown으로 닫고 재실행하지 않는다. 실패하면 남은 배치를 멈추며, 인프라 수정·추가 실행이 필요할 경우 기존 원문과 소비를 보존한 새 등록을 요구한다.

공용 lease owner는 `rl-triage-lineage-sources-v1`, admission은 등록된 `registered_desktop_shared_retry_v1`이다. CPU의 실제 소비 입력·최종 protocol·run namespace·SDK 직전 경로 검증을 마친 뒤 들어간다. GPU 배치 중 source/Git을 바꾸지 않는다. 새 Brev Start·설치·WSL restart·TokenFactory 접근·외부 제출은 없다.

준비·출처 검증·queue·승인·dispatch·SDK 학습/평가·CPU 보조·후속 검토·실패/재개·부모 전체 경과를 기록한다. 중첩 구간을 더하지 않는다. 장치 전력·사용률·메모리·energy는 실제 표본 범위와 coverage를 남기며 프로세스 전용·provider 비용으로 귀속하지 않는다. GPU kernel 시간·사람 검토·실청구·가격 근거를 얻지 못한 항목은 null이다. 예전 누락 비용을 이번 측정으로 채우지 않는다.

## 공개 결과

새 plan schema는 `registered_recovery_comparison_plan_v8`, profile은 `balanced_resource_remeasurement_v1`이다. 기존 공개 결과와 다른 새 폴더로 16행·실제 순서·원래 진단 SHA·새 effect·준비/실행 비용·unknown을 내보낸다. 이번 결과를 기존 진단 시간과 더한 값은 여러 단계의 결합값으로 표시하며 한 번에 실행한 사용자 종단 시간으로 부르지 않는다. 목표는 비교 조건과 근거를 완성하는 것이며 양성 회복·모델 우위를 완료 조건으로 두지 않는다.
