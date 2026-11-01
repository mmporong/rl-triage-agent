# 같은 입력의 네 방식 진단 비교

두 개발 계보에서 같은 수치와 검사 권한을 제공했을 때, Nemotron 3.5는 수치만 받든 프레임을 함께 받든 `P_reward` 짝 검사 한 번으로 보상 결함을 확정했다. 고정 순서는 세 번으로 확정했고, `registry_next`는 세 번 뒤 HOLD였다. 프레임 추가에 따른 개선은 확인되지 않았다.

이 비교는 이미 결과를 확인한 seed 2430·2431의 F1 자료로 실행기를 연결한 개발 측정이다. 새 계보 본평가나 모델 버전 비교가 아니며, 독립 사례는 두 계보다.

## 입력과 실행

- 네 방식: `registry_next`, `fixed_eligible`, 같은 3.5의 `logs_only`, `logs_plus_frames`.
- 공통 입력: 학습 수치 48개와 같은 Play의 수치 54개. 프레임 추가 방식에만 case 3장·reference 3장을 전달했다.
- 검사 권한과 예산: 같은 등록 검사 7종, arm당 최대 짝 검사 3회. 인자·기대값·승인·원인 확정은 호스트가 담당한다.
- 검사 실행에 필요한 runtime 사실 16개는 backend에만 결합했다. 두 모델 입력에서는 모두 제외했다.
- 모델: `nvidia/nemotron-3.5-super-vl-120b-a12b`, temperature 0, reasoning effort `none`, max tokens 1024. 요청마다 현재 등록 행동·근거 ID로 JSON schema를 생성했다.
- 실행 상한: API 12회, 짝 검사 24회, SDK 역할 실행 상한 48회. 실제 소비는 API 4회·짝 검사 16회·SDK 역할 32회다. 자동 재시도는 없었다.
- 정상 대조 8행은 정상 source를 host gate로 보류했다. API·검사 소비는 0이며 모델의 정상 식별로 세지 않는다.
- 새 학습·교정·회복 실험은 수행하지 않았다.

## 실제 결과

| 방식 | 올바른 확정 | HOLD | seed별 짝 검사 수 | API 요청 합 | SDK 역할 실행 합 | 역할 실행 wall 합 |
|---|---:|---:|---|---:|---:|---:|
| registry_next | 0/2 | 2 | 3 / 3 | 0 | 12 | 268.858초 |
| fixed_eligible | 2/2 | 0 | 3 / 3 | 0 | 12 | 334.391초 |
| 3.5 수치 | 2/2 | 0 | 1 / 1 | 2 | 4 | 74.578초 |
| 같은 3.5 수치+프레임 | 2/2 | 0 | 1 / 1 | 2 | 4 | 72.610초 |

역할 실행 wall은 자식 시작부터 종료와 소유 프로세스 정리 확인까지의 시간이다. GPU kernel 시간이나 과금 GPU 시간이 아니다. 모델 요청 wall 합은 수치 방식 3.683초, 프레임 방식 5.118초다. 전체 controller wall은 3,546.235초이며 입력·소스 검증과 영수증 재생도 포함한다. 검사 수만으로 전체 진단 시간을 환산하지 않는다.

두 seed에서 `registry_next`의 순서는 `P_value → P_physics → P_episode`, 고정 순서는 `P_episode → P_noise → P_reward`였다. 두 모델 방식은 모두 `P_reward`를 선택했다. 검사 결과를 받은 뒤 호스트가 등록된 보상 범주를 확정했다. 오확정·모델 계약 무효·provider unknown·실행 unknown은 관측되지 않았다.

## 영상과 불확실성

수치와 프레임 방식의 원인 확정·검사 수가 두 seed에서 같았다. 프레임 방식의 두 응답도 각각 학습 사실 48개와 8개를 인용했으며, raw frame ID와 Play 수치 인용은 0개였다. 따라서 이 실행을 영상이 더 나은 검사를 고른 증거로 사용하지 않는다. 프레임을 포함한 요청이 유효 응답과 실제 검사로 연결됐다는 범위만 확인했다.

2/2 분율의 Wilson 95% 구간은 약 0.342~1.000이며 0/2는 0~0.658이다. 영상 유무 간 불일치 짝은 0개라 paired exact p는 계산하지 않았다. 네 방식의 8행이나 정상 host gate 8행을 독립 정책 16개로 세지 않는다.

이 개발 사례에서는 같은 확정 결과에 도달하는 데 고정 순서보다 모델 방식이 적은 검사를 사용했다. 이 관측을 일반적 LLM 우위, 3 대비 3.5의 개선, 영상 효과 또는 새로운 회복 결과로 확대하지 않는다. 원래의 여러 결함 유형 본평가는 별도 계보와 사전등록으로 남아 있다.

## 근거와 재현

- [공개 요약](../evals/results/local_pair_four_arm_development_20261101_v1/summary.json), [사례별 결과](../evals/results/local_pair_four_arm_development_20261101_v1/cases.json), [출처 결합](../evals/results/local_pair_four_arm_development_20261101_v1/provenance.json), [파일 SHA](../evals/results/local_pair_four_arm_development_20261101_v1/manifest.json).
- [v2 사전등록](../bench/protocols/local_pair_four_arm_s2430_s2431_20261101_v2.json), 실행 commit `7787655e9003d9852b6f44f9d91d03c3ffc42b04`.
- [진단 driver](../evals/local_pair_four_arm_v1.py), [검사 호스트](../evals/local_pair_probe_host_v1.py), [NIM 원장](../evals/local_pair_nim_transport_v1.py), [실행기](../bench/run_local_pair_four_arm_v1.py), [공개 검증기](../evals/export_local_pair_four_arm_v1.py).
- 공개 검증은 모델 요청·응답·선택·승인·실제 검사·원장 prefix·후속 후보·최종 판정을 재생한다. 기존 host의 child wall null 값은 보존하고, raw 역할 영수증과 `owned-result`를 대조한 시간을 별도 필드로 기록했다.
- v1의 검사 전 실패는 [OPS-132](TROUBLESHOOTING.md#ops-132-네-방식-실행기의-gpu-lease-id-형식-연결)에 보존했다. 기존 결과와 응답을 덮어쓰지 않았다.

공개 파일에는 raw 프레임·모델 응답·개인 경로가 포함되지 않는다. 원문을 보유한 환경에서만 실제 영수증 전체 재생과 API/GPU 실행을 할 수 있다. 공개 JSON만으로 새 실험을 실행할 권한은 생기지 않는다.
