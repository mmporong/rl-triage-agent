# Brev 녹화와 렌더 확인 연결

새 수집 실행기는 정상·결함 정책의 Play 녹화와 렌더 동기화 확인을 연결한다.
두 실행의 환경·소스·결과 원문을 검증한 뒤 모델 입력 준비 파일을 만든다.
현재 검증 범위는 CPU 개발 fixture와 실행 계약이다. 새 Isaac 수집과 모델 비교는 아직 실행하지 않았다.

## 구성과 순서

| 구성 | 역할 |
| --- | --- |
| [수집 controller](../bench/run_vision_brev_render_v2.py) | 기존 source 배치, 별도 smoke 원장, 준비 파일 회수를 연결 |
| [렌더 backend](../evals/vision_brev_render_backend_v2.py) | record와 smoke를 환경 수집·실행 순서 봉인 안에서 실행 |
| [CPU 최종화](../evals/vision_render_readiness_v2.py) | 회수한 원문에서 준비 상태를 재검증하고 새 파일로 보존 |
| [입력 검증](../evals/vision_selection_sources_v1.py) | 원본 학습·자원·녹화·smoke·선택 입력의 연결을 확인 |

기존 source 순서는 학습 네 건, 짝 평가 두 건, 녹화 네 건이다.
학습 계획의 `NONE`·`F1`·`CORRECTED` 각 두 건과 환경 4,096개·300회 갱신은 유지한다.
교정 학습은 후속 고리에서 승인 근거를 소비한 경우에만 허용한다.

짝 평가 뒤에는 정상 정책의 기존 걷기 기준과 결함 정책의 고정 평가를 확인한다.
정상이 기준을 통과하고 결함은 회복 기준을 통과하지 못한 경우에 다음 수집으로 진행한다.
결함 정책도 기준을 통과하면 진단할 source 실패가 없는 결과로 남긴다.
미확정 결과는 `HOLD`이며 seed를 추가하지 않는다.

녹화가 끝나면 각 checkpoint에 대응하는 smoke 네 건을 별도 원장으로 실행한다.
한 번의 `advance`는 한 건만 진행한다. smoke를 기존 source action 목록이나 학습 원장에 추가하지 않는다.

## 실행과 재개

record와 smoke 모두 같은 GPU 잠금 안에서 소스 확인, GPU·Torch 환경 기록,
고정 worker 실행, 소스 재확인, 결과 봉인을 순서대로 수행한다.
worker는 기존 [Play worker](../bench/vision_play_worker_v1.py)를 사용하며
등록 모드에 따라 녹화 또는 카메라 A→B→A 동기화 확인을 실행한다.
자식은 시작 장벽에서 기다리고, 부모가 PID·process group·시작 시각을 확인한 뒤 worker를 실행한다.
소유권 확인이 실패하면 대기 중인 자식을 회수한다.

완료 실행은 context·attestation·job·실행·결과·artifact를 다시 대조한다.
검증된 완료 응답을 재생할 때는 새 context나 worker를 실행하지 않는다.
진행 중인 intent, 불완전한 attestation, 실패 기록은 보존한다.
기존 execution에 새 context를 붙이거나 불명확한 실행을 자동 반복하지 않는다.
유효한 완료 attestation이 있고 시간·비용 상한도 남아 있으면 새 GPU 실행 없이 원장을 마감한다.

각 smoke도 시작 전에 현재 resource proof, lease, deadline과 비용 여유를 확인한다.
전체 상한에는 기존 source action의 최대 예약액과 smoke 네 건의 최대 예약액이 들어가야 한다.
smoke 비용은 context 이전 intent부터 검증·봉인 이후 receipt까지의 경과 시간으로 계산한다.
VM 준비·대기·회수 비용은 기존 생명주기 원장에 포함하며 종료 전 비용 상태는 `open`이다.

## 자료 회수와 준비 상태

원격 job 안의 절대 경로는 당시 실행의 기록으로 보존한다.
회수 뒤에는 저장소 상대경로와 SHA를 담은 비공개 위치 표로 로컬 원본을 가리킨다.
원문 경로를 로컬 경로로 덮어쓰지 않는다.

네 smoke 원장까지 완료되면 `export`가 준비 자료의 위치 표를 만든다.
마지막 receipt 뒤에 중단됐어도 보관 원문으로 이 CPU 단계를 재개할 수 있다.
source 평가와 원장의 시간·비용을 다시 검증하며, VM을 다시 시작하거나 현재 lease를 요구하지 않는다.
평가 기준 파일이 LF 원문 해시로 등록됐다면 CRLF 사본에서 LF 바이트를 복원해 같은 해시인지 확인한다.
이는 줄바꿈 차이를 처리하는 경계이며, 평가 기준값은 등록 원문과 같아야 한다.

최종화 도구는 GPU 앱이나 모델 API를 실행하지 않는다. 보관한 resource proof를 원래 발급
시점의 기록으로 검증하며, 새 GPU 실행 권한으로 사용하지 않는다. source protocol의 LF SHA,
학습 action, seed freshness, checkpoint, 녹화와 smoke의 환경·소스·artifact를 함께 확인한다.
freshness 확인 시각은 해당 자원 시작 이후이며 등록 학습 시작 이전이어야 한다.
사용 진입점은 수집 controller의 `finalize`다. render protocol과 네 smoke 원장의 SHA,
시간·비용·자원 결합을 검사한 뒤 쌍별 검증을 호출한다.

같은 checkpoint의 record와 smoke가 검증된 경우에만 `model_use_ready=true` 파일을 만든다.
범위는 `registered_renderer_combination_only`다. warmup RGB나 state 원문처럼 기존 결과에
보존되지 않은 자료를 다시 계산했다고 표현하지 않는다. 준비 상태와 영상의 진단 기여,
확인 검사 지지, 원인 확정, 회복은 각각 별도 판정이다.

## CPU 확인

```powershell
cd "$HOME/rl-triage-agent"
uv run --no-project --python 3.10.12 --with pytest --with pyyaml --with numpy python -m pytest -q tests/test_vision_brev_render_backend_v2.py tests/test_run_vision_brev_render_v2.py tests/test_vision_render_readiness_v2.py
```

POSIX 소유권과 자식 정리 경로는 Linux에서도 확인한다.
실제 GPU 실행은 현재 잔액·견적, 새 seed 확인, source·render protocol 커밋과 자원 입장 뒤에 진행한다.
