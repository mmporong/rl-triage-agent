# 영상 검사 선택 비교의 실행 경계

이 실행기는 동결한 텍스트·영상·결합 입력 24개를 WSL에서 순서대로 호출한다.
원본 Play와 렌더 확인 기록을 검증한 뒤 기존 선택 호출기의 요청 원장을 사용한다.
실제 비교에는 새로 수집한 자료와 커밋된 입력별 사전등록이 필요하다.
이 문서는 실행 코드의 계약을 설명하며 영상의 선택 개선 효과를 보고하지 않는다.

## 구성

| 파일 | 역할 |
| --- | --- |
| [입력 검증](../evals/vision_selection_sources_v1.py) | 원본·V01 계약·선택 snapshot·동결 capsule을 대조 |
| [환경 원문 기록](../bench/vision_render_context_v1.py) | 선택한 Isaac Python과 GPU UUID·이름·드라이버·Torch CUDA 정보를 결합 |
| [실행 순서 기록](../bench/vision_render_attestation_v1.py) | 환경 수집 뒤 실행하고 source 전후·실행·결과 SHA를 봉인 |
| [WSL 실행기](../evals/vision_selection_worker_v1.py) | 커밋·입력·호출 순서·프로세스 소유권을 검사하고 기존 caller를 실행 |
| [입력 구성기](../evals/vision_selection_materialize_v1.py) | 네 source readiness에서 두 구간·세 입력 조건을 만들고 production 입력 검증기로 다시 확인 |
| [실행 결과 검증](../evals/vision_selection_execution_v1.py) | 호출 상태·남은 명령과 실제 Brev 검사 원문 여섯 종의 결합을 검증 |
| [종단 요청 계약](../evals/vision_selection_e2e_v1.py) | 비교 결과를 보기 전에 두 결합 입력을 고정하고 검사·교정·후속 관찰의 조건을 검사 |
| [종단 호출기](../evals/vision_selection_e2e_caller_v1.py) | 승인·소비와 부분 실행 원문을 보존하고 다음 단계 한 개를 진행 |
| [Brev 실행 연결](../evals/vision_selection_brev_adapter_v1.py) | 등록 검사·교정·고정 평가·녹화의 원문과 source 결합을 대조 |
| [종단 CLI](../evals/run_vision_selection_e2e_v1.py) | 검증·남은 명령 조회·한 단계 실행·결과 요약 |
| [기존 비교 방법](../bench/protocols/vision_selection_comparison_v1.json) | 두 source seed·두 정책·두 구간의 세 입력 조건과 판정 기준 |

모델은 등록 검사와 현재 입력에 보이는 근거 ID를 고른다. 이 실행기는 검사 실행,
승인, 교정, 재학습을 요청하지 않는다. 선택 결과를 실행으로 옮길 때는 기존 live
activation과 승인·receipt 검사를 별도로 거쳐야 한다.

## 사전등록과 입력

실행 protocol은 방법 protocol과 source protocol의 상대경로·LF SHA, 정확한 24행,
저장소 source/evidence LF SHA, `input_locations_sha256`를 담는다.
실행 protocol 자신의 SHA와 source anchor는 커밋 후 실행 인자로 전달한다.
이 방식으로 자기 파일의 SHA를 파일 안에 넣는 순환을 피한다.

`input_locations`는 비공개 파일이다. 실행 ID와 고정 순서의 call ID마다 source manifest와
capsule의 저장소 상대경로·원본 SHA를 담는다. source manifest는 snapshot, V01 contract,
raw trace, camera audit, model readiness의 바이트를 결합한다. 공개 protocol에는
비공개 경로 대신 위치 파일 전체의 SHA가 들어간다.

입력 검증은 다음 연결을 확인한다.

1. source 학습의 intent·raw·receipt·runtime manifest가 등록 action, seed, arm과 같다.
2. 녹화 job·protocol·admission·실행 receipt·video·trace·두 구간의 measurement와 spec이 같다.
3. GPU 입장·종료 원문, 자원·비용 proof와 lease 범위가 기록된 실행과 같다.
4. V01이 보관한 video·checkpoint·measurement와 원본 녹화의 SHA가 같다.
5. 625개 trace 행의 physics step·시각·camera를 대조한 camera audit가 같다.
6. snapshot에서 재계산한 선택 상태, V01에서 재생성한 bundle·messages가 capsule과 같다.

원격 job과 receipt 안의 경로는 과거 실행의 provenance로 보존한다. 로컬 검증에서는
비공개 상대경로 표로 받은 파일을 읽는다. 원격 가상환경이나 원격 소스 경로를
검증 PC에서 열거나 원본 기록의 경로를 덮어쓰지 않는다.

## 렌더 확인과 환경 결합

원본 녹화 결과의 `gpu_smoke_required_before_model_use=true`와
`zero_update_pixel_lag_confirmed=false`는 유지한다.
별도 readiness 검증이 다음 조건을 모두 만족할 때만 해당 입력을 허용한다.

- 같은 고정 capture 코드·Play profile·source snapshot·GPU에서 record와 smoke를 실행
- 두 실행의 context 원문에서 GPU 이름·드라이버·Torch 버전·built CUDA·device capability 일치
- smoke의 warmup–A–B–A 순서, timing, 고정 camera offset과 보존한 A/B/A RGB 바이트 일치
- readiness receipt 전체가 원본 파일에서 재계산한 값과 일치

실행 wrapper는 GPU 임대와 입장검사 아래에서 source 확인 → context 수집 → 실행 →
source 재확인 → 봉인 순서를 고정한다. context가 끝난 시각, 실행 시작·종료, 봉인 시각이
순서대로 admission과 등록 deadline 안에 있어야 한다. 실행·결과·job SHA도 함께 검증한다.
외부 입력으로 임의 실행 함수를 받는 CLI는 두지 않는다.

context를 만들지 않은 기존 녹화는 이 경로에서 사용 가능 상태가 되지 않는다.
수집 실행기에 이 기록을 연결하고 실제 GPU에서 확인하는 단계는 별도로 남아 있다.

readiness 범위는 `registered_renderer_combination_only`다.
기존 v1 smoke의 warmup RGB와 state before/after 원문은 해당 산출물에 남지 않는다.
후검증은 동결 assessor의 통과 기록과 보존 RGB를 대조하며, 보존되지 않은 상태를
다시 계산했다고 표현하지 않는다. 보편적인 pixel lag 보장도 하지 않는다.

[새 atomic capture 경로](VISION-ATOMIC-CAPTURE.md)는 RGB·semantic mask·카메라 행렬·시각을 하나의 Replicator
`Writer.write(data)`에서 복사한다. 정지 화면 검사에는 프레임마다 바뀌는 표식을 넣어
이전 RGB나 semantic mask가 섞이면 거부하고, 정책 녹화에는 표식을 넣지 않는다.
배경 픽셀 전체의 일치와 프레임의 시각 결합을 구분하며 기존 결과의 판정은 유지한다.

각 callback의 배열과 metadata를 보존하고, 실행 manifest를 읽을 때 카메라·시각·RGB·
물리 상태와 원문을 다시 대조한다. readiness와 renderer 설정, SDK 종료 및 GPU 해제 검사는
같이 유지한다. 새 worker·backend는 별도 source 결합을 사용한다. 수집·원문 대조의 CPU 검증을 통과했으며
실제 Isaac 호환성 검사를 통과하기 전에는 모델 입력을 내보내지 않는다.

## 호출과 재개

비교 실행기는 `comparison`만 허용한다. 논리 선택 24개, 전역 요청 최대 72회이며
선택당 최대 3회 안에서 의미 피드백 1회와 인프라 재시도 1회만 허용한다.
SDK 재시도는 0회다. 이 상한은 기존 core 배정의 비교 몫이며 다른 단계의 몫을 사용하지 않는다.

`verify`는 Git ancestor·결합 blob·현재 파일·입력들을 검사한 뒤 기존 caller의 plan을 만든다.
`advance`는 `setsid`로 실행한 Linux 프로세스에서 등록 call 하나를 진행한다.
키는 WSL 환경에서 읽으며 원문을 출력하거나 입력 파일에 넣지 않는다.

| 상태 | 종료 코드 | 처리 |
| --- | ---: | --- |
| `choice_ready` | 0 | 유효한 선택을 보존 |
| `terminal_hold` | 2 | 모델 보류 또는 계약 무효를 보존 |
| `paused_infra` | 3 | 등록한 대기·재시도 규칙으로 재개 |
| `blocked_pending` | 4 | 미완료 요청이나 활성·stale 소유 기록을 보존 |

완료 응답은 기존 원장을 재검증해 사용한다. 미완료 요청은 재전송하지 않는다.
다른 프로세스를 종료하거나 stale 소유 파일을 지워 재시작하지 않는다.
실행 receipt는 지정한 stage 아래에 추가 기록한다.

검증 대상은 구현 계약과 CPU 개발 fixture다. 새 GPU 자료, 입력별 비교 결과,
검사 지지와 회복 효과는 실제 사전등록 실행 뒤에 보고한다.

## 검사 결과에서 교정 요청까지

입력 구성기는 네 source의 readiness를 확인해 두 구간의 V01 입력 여덟 개와 세 입력 조건의
capsule 24개를 만든다. 작성한 protocol과 위치 파일은 실제 비교 실행기의 입력 검증기로
다시 확인한다. 녹화 checkpoint와 학습 telemetry, 같은 seed의 NONE 기준이 달라지면 거부한다.

Brev 검사 근거는 job, worker, execution, backend result, report jobs, report 여섯 원문을
결합한다. 각 파일의 SHA뿐 아니라 task, checkpoint, source, 실행 protocol, 완료·종료·GPU 정리와
판정값도 대조한다. source 데이터 protocol과 검사 실행 protocol은 각각 검증한다.

종단 요청 계약은 비교의 canonical 결과 위치에 완료 응답이나 결과가 생기기 전에
두 seed의 F1 첫 구간 결합 입력을 고정한다. 학습 job, 녹화 job, 영상 계보와 CORRECTED job은
각각 원문과 등록 training plan에서 읽는다. 문자열을 덧붙여 교정 job을 만들지 않는다.

모델이 선택한 유효 검사는 NONE/F1 쌍 검사의 후보가 된다. `P_reward`에서 NONE normal과
F1 abnormal 영수증이 같은 seed·checkpoint에 결합된 경우에만 보상 복원과 재학습의 승인
evidence를 만든다. 승인과 소비 뒤의 evidence가 같아야 CORRECTED 요청이 열린다.
후속 영상 관찰은 고정 평가의 회복 판정을 바꾸지 않는다.

이 상태 전이는 CPU 계약으로 검증했다. API 32회는 초기 선택과 후속 관찰이 공유하며,
한 상태 조회마다 모델 명령 하나만 반환해 다음 호출 전에 원장을 다시 확인한다.
종단 호출기는 등록 실행 protocol의 source·backend·검증기와 일치하는 작업만 연결한다.
검사 쌍은 실행 명령에 결합된 승인을 소비한 뒤 진행한다. 교정 평가에 필요한 NONE 정책은
검증된 완료 기록에서 읽으며, 기록이 없으면 새 기준 학습을 시작하지 않고 보류한다.

완료 단계는 원문과 승인 원장을 대조해 재사용한다. 부분 실행은 완료한 arm의 원문과 GPU
시간을 보존하고 보류하며, 이미 소비한 요청을 다시 실행하지 않는다. 임대가 끝난 뒤에도
완료 기록을 조회할 수 있지만 새 실행에는 현재 임대·시간·비용 확인이 필요하다.

종단 CLI는 `verify`, `remaining_commands`, `advance`, `summarize`를 제공한다.
교정 후 녹화는 등록 job·checkpoint·고정 평가와 대조하고 V01 입력을 만든 뒤 관찰 호출에
전달한다. 이 경로는 CPU 회귀와 저장된 Brev 검사 원문으로 검증했다. 실제 수집한 새 영상의
결합과 GPU 교정·후속 관찰 실행은 남아 있으며, 현재 검증은 새 종단 회복 결과를 뜻하지 않는다.
