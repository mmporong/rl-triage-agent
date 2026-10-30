# 새 정책 수집과 진단 비교

Go2 평지 과제에서 새 seed의 정책 10개와 같은 seed의 정상 참조를 수집한 뒤, 고정 평가와 세 검사 선택 방식을 비교한다. 수집·행동 평가·모델 제안·검사 확정은 별도 단계이며, source 수집 완료를 진단이나 회복 성공으로 집계하지 않는다.

## 등록 범위

| 사례 | seed 후보 | source | 정상 참조 |
|---|---|---|---|
| 정상 2개 | 2301·2302 | NONE | 자신 |
| 보상 2개 | 2303·2304 | F1 | 각 seed의 NONE |
| 모터 2개 | 2305·2306 | F2 | 각 seed의 NONE |
| 보상과 PPO epoch 2개 | 2307·2308 | F1+R1 | 각 seed의 NONE |
| 등록 교정 밖 2개 | 2309·2310 | F5·F3 | 각 seed의 NONE |

총 18회 학습이며 모두 4096개 환경, 300회 갱신, 새 초기화를 쓴다. 결함 강도와 동결 결함 구현은 바꾸지 않는다. seed 2399의 NONE 1회 갱신은 실행 계층 사전 확인용이며 비교 사례에 합산하지 않는다.

결함 유형은 개발에 사용한 범주다. 새 seed는 새 결함 유형을 뜻하지 않는다. 실행 전 검사 범위는 현재 저장소의 실행 기록과 지정 Isaac Lab 로그이며, 다른 장비의 미보관 기록까지 미노출을 증명하지 않는다. 설계자가 기존 비교와 같은 seed 대조 결과를 본 뒤 만든 실행기다.

같은 seed 정상 참조를 두는 이유는 [3221 대조 실험](SAME-SEED-RECOVERY-CONTROL.md)에 있다. 교정 뒤 정책이 원래 대조군의 기준을 넘지 못했으나, 같은 seed의 정상 학습도 같은 체크포인트와 평가값을 만들었다. 이 수집기는 seed가 다른 대조군에만 의존하는 비교를 피한다.

## 실행

Windows PowerShell과 설치된 Isaac Lab을 사용한다. 공용 lease는 자식 실행 전후에 소유자·PID·heartbeat·배타 lock을 확인한다. 각 자식은 기존 Windows Job 실행기로 관리하며 실패하면 다음 학습을 시작하지 않는다.

등록 v2는 Windows 가상환경 Python 중계 프로세스를 실제 부모 계층으로 확인한다. GPU 표시 앱에는 경로가 확인된 Windows 설정 창 두 개와 Codex 표시 도우미를 추가했으며 계산 작업·엔진·메모리 차단 기준은 유지한다. v1은 CPU 입장 확인 뒤 학습·모델 호출 없이 보존했다.

```powershell
cd "$HOME/rl-triage-agent"
$anchor = git rev-parse HEAD
$evidence = Join-Path $HOME 'rl-triage-source-evidence-v1'

# 실제 등록 source·protocol·Isaac 파일을 CPU에서 확인
& ./.venv/Scripts/python.exe -B evals/lineage_source_batch_v1.py verify --lab "$HOME/IsaacLab" --anchor $anchor

# 사전 실행: 새 폴더만 허용하며 GPU lease를 취득·해제
& ./bench/run_lineage_source_batch_v1.ps1 -Anchor $anchor -Output "$evidence/smoke" -Mode smoke -Execute

# 완료 원문을 재검증해 사전 실행을 반복하지 않고 source 18개 수집
& ./bench/run_lineage_source_batch_v1.ps1 -Anchor $anchor -Output "$evidence/sources" -Mode all -SmokeResult "$evidence/smoke" -Execute
```

등록 파일과 묶인 소스가 해당 커밋에 있어야 한다. 결합 밖의 후속 커밋은 허용하며 결합 파일이 달라지면 멈춘다. 위 경로에 이미 결과가 있으면 새 출력 경로가 필요하다. 소비된 seed를 같은 tag로 다시 학습하는 자동 재시도는 없다.

학습 자식은 실제 `PPO.update()` 반환 횟수와 `learn()` 반환을 확인하고 체크포인트 SHA를 앱 종료 전에 봉인한다. `Runner.load()` 호출을 거부해 기존 가중치의 이어 학습을 막는다. 부모는 종료 코드 0뿐 아니라 완료 증거·체크포인트 바이트·소유 프로세스 정리를 검증한다. 체크포인트 바이트 중복은 보고하며 다른 seed로 바꾸지 않는다.

상한은 학습별 부모 벽시계 1800초, 배치 전체 21600초다. 예상 소요 시간이나 GPU 커널 시간 측정값을 뜻하지 않는다. 이 판의 유료 클라우드 GPU Start와 모델 API 예산은 모두 0이다.

## 후속 평가와 분리

수집 후에는 동결된 26조건 평가와 기존 밴드로 정상 참조의 유효성과 결함의 실패 존재를 확인해야 한다. 정상 실패·결함 실패 부재가 있으면 해당 사례와 제외 이유를 보존하고 대체 seed를 찾지 않는다. 계획 10개와 유효 비교 짝 수를 함께 보고한다.

그다음 등록 순서·고정 실행 가능 순서·Nemotron 3.5 수치 입력을 같은 검사·판단 예산에서 비교하는 별도 판을 등록한다. 영상으로 더 나은 검사를 골랐다는 개발 근거가 아직 없어 시각 진단 arm은 보류한다. [VL 렌더 무결성 검사](VISUAL-INTEGRITY-OBSERVER.md)의 결과는 영상 입력을 가리는 기능의 근거이며 진단 선택 이득의 근거로 합산하지 않는다.

## 완료 source 재사용

[재개 실행기](../evals/lineage_source_continuation_v1.py)는 완료된 13개 source를 검증해 재사용하고 원래 순서의 나머지 5개만 실행한다. 적용 대상은 [재개 등록](../bench/protocols/lineage_source_continuation_20261031_v1.json)에 묶인 실행이다. 중단 stage에 GPU 입장 기록만 있고 학습 자식이 시작되지 않았는지 확인한다. 기존 실패 결과·accepted·체크포인트·TensorBoard 원문은 읽기만 한다.

새 실행에는 새 출력 폴더와 등록 anchor가 필요하다. 원래 source anchor와 재개 실행기의 anchor를 나눠 기록하며, 결과에도 두 실행의 provenance를 남긴다. GPU 입장 규칙은 유지한다. 이미 끝난 학습의 반복과 자동 재시도는 허용하지 않는다.

```powershell
cd "$HOME/rl-triage-agent"
& ./bench/run_lineage_source_continuation_v1.ps1 -ExecutionAnchor $anchor -Protocol ./bench/protocols/lineage_source_continuation_20261031_v1.json -PriorRoot "$evidence/sources" -Output "$evidence/continuation" -Execute
```

`PriorRoot`는 등록한 13개 완료 원문과 실패 기록이 있는 폴더여야 한다. 임의의 다른 중단 배치에는 이 등록을 적용할 수 없다. 새 실행은 학습별 1800초와 나머지 배치 9000초 상한을 둔다.

## 고정 평가와 세 정책의 비교

[고정 평가 실행기](../evals/lineage_fixed_eval_batch_v1.py)는 완료 source 18개의 체크포인트를 하나의 owned worker에서 평가한다. 평가 seed 2026, 명령 26조건, 원래 밴드를 유지한다. source의 학습 결함은 fixed 환경에 재주입하지 않고 같은 seed 정상 정책과 행동을 비교한다. [PowerShell 실행기](../bench/run_lineage_fixed_eval_v1.ps1)가 입력 검증 뒤 공용 lease 취득·해제를 처리한다. 정상 참조 실패와 source 실패 부재는 후속 진단에서 제외하되 계획된 사례 수에 남긴다.

[진단 비교 실행기](../evals/lineage_comparison_batch_v1.py)는 `registry_next`, `fixed_eligible`, Nemotron 3.5의 `model_adaptive`를 같은 공개 수치·검사·예산으로 비교한다. 사례마다 정책 순서를 순환하고, 정책별 검사 한도는 3회다. 정상 source는 호스트 anomaly gate로 개입을 막으므로 이 결과를 모델만의 정상 판별 성능으로 쓰지 않는다.

- [비교 등록](../bench/protocols/lineage_comparison_20261031_v1.json): 계획 10사례와 정책 3개의 전 결과를 보존한다. 복합 결함의 첫 구성요소 확정은 부분 진단이며 전체 원인 확정으로 집계하지 않는다.
- [paired 검사 등록](../bench/protocols/lineage_paired_probes_20261031_v1.json): reference와 case를 각각 측정하고 부모가 하나의 논리 검사 영수증으로 정산한다. `P_value`의 `model_0.pt`도 별도 SHA로 묶는다.
- [NIM 호출 등록](../bench/protocols/lineage_nim_transport_20261031_v1.json): 모델 판단 slot 30개, slot당 피드백·인프라 요청을 포함한 최대 3회, 전체 최대 90요청이다. 이전 72회 초안이나 120회 원장을 늘린 판이 아니다.

NIM worker는 요청별 영구 dispatch claim을 남겨 불명확한 송신을 다시 보내지 않는다. 닫힌 queue에는 새 요청을 넣지 않으며, 부모는 실제 probe 영수증이 없으면 확정하지 않는다. 무효 모델 응답을 규칙 선택으로 바꾸지 않는다. TokenFactory의 별도 비교 응답은 이 NIM 고리에 치환하지 않는다.

주지표는 같은 예산 내 올바른 구성요소 확정 수다. 오확정·HOLD·정상 개입·모델 계약 유효율과 실제 검사 시도·성공·unknown을 나눠 보고한다. API 지연, owned child 시간, 부모 벽시계도 구분하며 비용이 미확정이면 알려진 소계만 남긴다. 이 비교 판은 첫 진단 epoch까지이며 교정·재학습·두 번째 epoch는 별도 등록이 필요하다.

구현과 CPU 회귀는 실제 검사·모델 비교 결과를 대신하지 않는다. 실행별 완료/HOLD/무효 기록으로 측정 상태를 확인하며, 결과가 나오기 전에 모델 우위나 회복 효과를 주장하지 않는다.
