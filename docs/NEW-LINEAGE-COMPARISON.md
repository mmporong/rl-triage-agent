# 새 정책 수집과 진단 비교

Go2 평지 과제에서 새 seed의 정책과 같은 seed 정상 참조, 총 18개를 학습하고 고정 평가를 마쳤다. 계획한 10사례 모두 진단 비교 입력으로 사용할 수 있다. 수집·행동 평가·모델 제안·검사 확정은 별도 단계이며, 학습과 평가 완료를 진단이나 회복 성공으로 집계하지 않는다.

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

## 수집과 행동 평가 결과

[공개 결과](../evals/results/lineage_source_fixed_dev_20261031_v1/summary.json)에는 학습 18개와 고정 평가 18개의 체크포인트·보고서 SHA를 담았다. [사례별 gate](../evals/results/lineage_source_fixed_dev_20261031_v1/gate.json)를 같은 원문으로 재계산해 일치 여부를 확인했다. 평가 seed 2026, 명령 26조건과 원래 밴드를 유지했다.

| 구분 | 사례 수 | 고정 평가 뒤 처리 |
|---|---:|---|
| 정상 자기 대조 | 2 | healthy, 호스트가 개입을 막음 |
| 보상·모터·보상+PPO epoch 결함 | 6 | 행동 실패 확인, 등록 진단 고리 대상 |
| 등록 교정 밖 F5·F3 | 2 | 행동 실패 확인, 원인 진단은 허용하되 교정은 HOLD |
| 정상 참조 실패·실패 부재·측정 미확정 | 0 | 제외 사례 없음 |

모든 정상 참조가 보행 기준을 통과했고, 체크포인트 중복은 없었다. 결함 표시는 주입 등록 정보이며 검사가 원인을 확정했다는 뜻은 아니다. 이 수집·평가 단계의 모델 API 요청은 0회다. 첫 진단 비교는 아래 결과에 기록했으며 교정 뒤 회복 여부는 별도 단계다.

## 후속 진단과 분리

동결된 26조건 평가와 기존 밴드로 정상 참조의 유효성과 결함의 실패 존재를 확인한다. 정상 실패·결함 실패 부재가 있으면 해당 사례와 제외 이유를 보존하고 대체 seed를 찾지 않는다. 이번 묶음의 계획 사례와 유효 비교 짝은 모두 10개다.

그다음 등록 순서·고정 실행 가능 순서·Nemotron 3.5 수치 입력을 같은 검사·판단 예산에서 비교하는 별도 판을 등록한다. 영상으로 더 나은 검사를 골랐다는 개발 근거가 아직 없어 시각 진단 arm은 보류한다. [VL 렌더 무결성 검사](VISUAL-INTEGRITY-OBSERVER.md)의 결과는 영상 입력을 가리는 기능의 근거이며 진단 선택 이득의 근거로 합산하지 않는다.

## 완료 source 재사용

[재개 실행기](../evals/lineage_source_continuation_v2.py)는 두 실행에서 완료한 16개 source를 검증해 재사용하고 마지막 정상·탐색 결함 정책만 학습한다. 적용 대상은 [재개 등록 v2](../bench/protocols/lineage_source_continuation_20261031_v2.json)에 묶인 실행이다. 각 중단 stage에 GPU 입장 기록만 있고 학습 자식이 시작되지 않았는지 확인한다. 기존 실패 결과·accepted·체크포인트·TensorBoard 원문은 읽기만 한다.

새 실행에는 새 출력 폴더와 등록 anchor가 필요하다. 원래 source anchor와 재개 실행기의 anchor를 나눠 기록하며, 결과에 모든 실행의 provenance를 남긴다. [GPU 입장 대기](../evals/lineage_gpu_wait_v1.py)는 기존 기준으로 연속된 clear 관측 두 번을 확인한다. 일시적인 busy에는 최대 600초 기다리고, 기준을 통과하지 못하면 학습을 시작하지 않는다. 이미 끝난 학습의 반복과 자동 재시도는 허용하지 않는다.

```powershell
cd "$HOME/rl-triage-agent"
& ./bench/run_lineage_source_continuation_v2.ps1 -ExecutionAnchor $anchor -Protocol ./bench/protocols/lineage_source_continuation_20261031_v2.json -OriginRoot "$evidence/sources" -ContinuationRoot "$evidence/continuation" -Output "$evidence/final-two" -Execute
```

`OriginRoot`는 첫 13개, `ContinuationRoot`는 다음 3개 완료 원문과 실패 기록이 있는 폴더여야 한다. 임의의 다른 중단 배치에는 이 등록을 적용할 수 없다. 새 실행은 학습별 1800초와 대기를 포함한 전체 5400초 상한을 둔다.

## 고정 평가와 세 정책의 비교

[고정 평가 실행기 v2](../evals/lineage_fixed_eval_batch_v2.py)는 완료 source 18개의 체크포인트를 하나의 owned worker에서 평가한다. 평가 seed 2026, 명령 26조건, 원래 밴드를 유지한다. source의 학습 결함은 fixed 환경에 재주입하지 않고 같은 seed 정상 정책과 행동을 비교한다. [PowerShell 실행기](../bench/run_lineage_fixed_eval_v2.ps1)가 입력 검증 뒤 공용 lease 취득·해제를 처리한다. GPU 대기도 원래 전체 3600초 안에 포함하고 자식 실행 한도를 남은 시간에 맞춘다. 정상 참조 실패와 source 실패 부재는 후속 진단에서 제외하되 계획된 사례 수에 남긴다.

[진단 비교 실행기](../evals/lineage_comparison_batch_v1.py)는 `registry_next`, `fixed_eligible`, Nemotron 3.5의 `model_adaptive`를 같은 공개 수치·검사·예산으로 비교한다. 사례마다 정책 순서를 순환하고, 정책별 검사 한도는 3회다. 정상 source는 호스트 anomaly gate로 개입을 막으므로 이 결과를 모델만의 정상 판별 성능으로 쓰지 않는다.

- [비교 등록](../bench/protocols/lineage_comparison_20261031_v1.json): 계획 10사례와 정책 3개의 전 결과를 보존한다. 복합 결함의 첫 구성요소 확정은 부분 진단이며 전체 원인 확정으로 집계하지 않는다.
- [paired 검사 등록](../bench/protocols/lineage_paired_probes_20261031_v1.json): reference와 case를 각각 측정하고 부모가 하나의 논리 검사 영수증으로 정산한다. `P_value`의 `model_0.pt`도 별도 SHA로 묶는다.
- [NIM 호출 등록](../bench/protocols/lineage_nim_transport_20261031_v1.json): 모델 판단 slot 30개, slot당 피드백·인프라 요청을 포함한 최대 3회, 전체 최대 90요청이다. 이전 72회 초안이나 120회 원장을 늘린 판이 아니다.

NIM worker는 요청별 영구 dispatch claim을 남겨 불명확한 송신을 다시 보내지 않는다. 닫힌 queue에는 새 요청을 넣지 않으며, 부모는 실제 probe 영수증이 없으면 확정하지 않는다. 무효 모델 응답을 규칙 선택으로 바꾸지 않는다. TokenFactory의 별도 비교 응답은 이 NIM 고리에 치환하지 않는다.

주지표는 같은 예산 내 올바른 구성요소 확정 수다. 오확정·HOLD·정상 개입·모델 계약 유효율과 실제 검사 시도·성공·unknown을 나눠 보고한다. API 지연, owned child 시간, 부모 벽시계도 구분하며 비용이 미확정이면 알려진 소계만 남긴다. 이 비교 판은 첫 진단 epoch까지이며 교정·재학습·두 번째 epoch는 별도 등록이 필요하다.

구현과 CPU 회귀는 실제 검사·모델 비교 결과를 대신하지 않는다. 실행별 완료/HOLD/무효 기록으로 측정 상태를 확인하며, 결과가 나오기 전에 모델 우위나 회복 효과를 주장하지 않는다.

## 첫 진단 비교 결과

[공개 집계](../evals/results/lineage_diagnosis_dev_20261031_v1/summary.json)는 10사례와 세 선택 방식의 30행을 모두 보존한다. 각 방식의 분모는 10이며, 완료는 결과 기록의 완료를 뜻한다.

| 선택 방식 | 올바른 원인 확정 | HOLD | 정상 무개입 | 오확정 | 논리 검사 | 모델 요청 |
|---|---:|---:|---:|---:|---:|---:|
| 등록 순서 `registry_next` | 0 | 8 | 2 | 0 | 16 | 0 |
| 실행 가능 고정 순서 `fixed_eligible` | 2 | 6 | 2 | 0 | 15 | 0 |
| Nemotron 3.5 `model_adaptive` | 2 | 6 | 2 | 0 | 13 | 13 |

보상 결함 seed 2303·2304에서 모델은 `P_episode normal → P_reward abnormal`의 두 검사로 원인을 확정했다. 고정 순서는 같은 원인을 세 검사로 확정했고, 등록 순서는 세 검사 한도에서 HOLD였다. 이 두 사례에서 필요한 검사 수의 차이를 확인했으며 전체 확정 건수는 모델과 고정 순서가 같았다. 정상 두 사례의 무개입은 호스트 gate의 결과다.

실행 중 미등록 Windows 서비스의 GPU 컨텍스트가 나타나 입장 검사에서 26역할을 차단했다. 이 역할들은 Isaac 자식을 시작하지 못했고, 원래 `unknown`과 HOLD를 유지했다. 별도 한 역할에서는 자식의 exit0·정리 뒤 부모 검증이 `PermissionError`를 기록했다. 오류 메시지와 파일명이 원문에 없어 그 원인은 확정하지 못했다. 해당 보고서를 나중에 읽을 수 있었어도 원래 판정을 고치지 않았다.

모델 요청 13회의 토큰 합은 35,986, API 응답 시간 합은 91.757초다. 소유 자식 시간은 누락 26역할이 있어 전체값을 `null`로 두고 알려진 소계 2,106.187초만 기록했다. 이 결과로 전체 진단 속도나 GPU 절감을 계산하지 않는다. 공개 관측에는 측정 전 예외와 측정 뒤 검증 오류를 구분하고, 영수증이 없으면 그 사실을 표시했다.

이 판의 입력은 수치 자료다. 기존 3 모델과의 비교, 영상의 검사 선택 이득, 교정·재학습·새 회복을 포함하지 않는다. [행별 결과와 관측](../evals/results/lineage_diagnosis_dev_20261031_v1/manifest.json)은 원래 실행의 HOLD와 비용 미확정을 보존한다.

## 첫 교정의 별도 실행

[첫 교정 등록 v2](../bench/protocols/lineage_recovery_execution_20261031_v2.json)는 모델 응답 전에 정한 보상 사례와 복합 사례, 등록 순서와 모델 선택 방식만 대상으로 삼는다. 유효한 검사 영수증으로 확정된 후보만 승인·소비하고, 미확정 후보는 학습하지 않는다. 같은 seed 정상 고정 평가를 참조하고, 교정 뒤 300회 새 초기화 학습과 동결 26조건 평가로 회복 여부를 판정한다.

[실행기 v2](../evals/lineage_recovery_execution_v2.py)는 실제 3회 갱신 사전 실행을 먼저 요구한다. 첫 판은 학습 전 owned 실행 폴더 누락으로 중단돼 보존했고, v2는 호출자가 이 폴더를 배타 생성하도록 맞췄다. 이 사전 실행은 인프라 확인이며 회복 성공 사례가 아니다. 이 판은 첫 교정까지이고, 두 번째 교정은 유효한 미회복 영수증을 얻은 뒤 새로 등록해야 한다.

## 새 계보의 첫 교정 회복

[결과](../evals/results/lineage_first_recovery_s2303_20261031_v1/summary.json)는 Go2 보상 결함 seed 2303에서 `recovered`를 기록했다. Nemotron 3.5의 `P_episode → P_reward` 제안과 실제 paired 검사 영수증으로 보상 원인을 확정한 뒤, 호스트가 등록된 `restore_body_frame_reward`를 선택했다. 승인·소비 기록을 거쳐 같은 seed로 새 초기화 학습 300회를 완료하고 동결 26조건 평가를 통과했다.

| 고정 평가 지표 | 결함 정책 | 교정 뒤 정책 | 같은 seed 정상 참조 |
|---|---:|---:|---:|
| 수평 속도 추종 RMSE, m/s | 1.182110 | 0.175828 | 0.175829 |
| yaw 각속도 추종 RMSE, rad/s | 0.798553 | 0.264138 | 0.264138 |
| 낙상률 | 4.04% | 0% | 0% |
| 평균 생존, 초 | 19.484789 | 20 | 20 |

[결함 정책 원문](../evals/results/lineage_source_fixed_dev_20261031_v1/reports/f1-s2303.json), 교정·정상 평가 원문과 각 SHA는 공개 결과에 연결했다. 회복 판정의 추종 오차 비율 두 항목·낙상·생존 네 기준을 모두 통과했다. 교정 체크포인트는 같은 seed 정상 참조와 SHA256이 같아 정상 정책을 재현한 결과다.

계획한 두 사례×두 선택 방식의 4행 중 검사로 확정된 교정 후보는 1행이었다. 해당 후보는 회복했고, 등록 순서의 보상 사례 1행과 두 방식의 복합 사례 2행은 HOLD로 유지했다. 총 1회 학습과 1회 평가를 수행했으며 나머지 3행의 학습은 0회다. 이 결과의 범위는 사전에 정한 첫 F1 사례와 같은 seed 대조다.

학습의 소유 자식 벽시계는 488.516초, 평가는 44.578초였다. 학습 뒤 임대 파일 읽기 오류로 평가가 중단된 기록을 보존했고, [재개 등록 v4](../bench/protocols/lineage_recovery_continuation_20261031_v4.json)에서 완료 학습·체크포인트·consumed 원장을 검증해 평가만 이어갔다. 재개 시 추가 학습·새 모델 호출·새 교정 승인은 모두 0이다. [재개 실행기](../evals/lineage_recovery_continuation_v4.py)는 원래 학습 anchor와 새 등록 anchor를 나눠 검증한다.

이번 고리의 모델 입력은 수치 자료이며, 기존 3 모델과의 비교나 영상의 검사 선택 이득을 측정한 결과는 아니다. 전체 10사례의 비교 결과와 기존 HOLD도 유지한다.

## 복합 사례의 첫 교정

[seed 2307의 별도 결과](../evals/results/lineage_compound_s2307_20261031_v1/first-correction/summary.json)는 보상 좌표계와 PPO 갱신 횟수 결함이 함께 있는 사례에서 첫 교정만으로 회복되지 않았음을 기록한다. 원래 비교는 GPU 입장에서 막혀 검사 자식이 시작되지 않았으므로, [별도 등록](../bench/protocols/lineage_compound_retry_20261031_v1.json)으로 같은 입력과 검사 한도를 다시 실행했다. 이미 본 인프라 실패 뒤의 개발 측정이며 원래 30행 결과는 유지한다.

등록 순서는 `P_value → P_physics → P_episode` 세 검사가 모두 normal인 뒤 HOLD였다. Nemotron 3.5는 `P_episode normal → P_reward abnormal`로 보상 성분을 확정했다. 이 확정은 복합 결함 전체의 진단이 아니다. 호스트가 등록 교정 `restore_body_frame_reward`를 선택했고, [첫 교정 실행기](../evals/lineage_compound_first_recovery_v1.py)가 승인·소비 뒤 같은 seed로 새 초기화 학습을 수행했다.

| 고정 평가 지표 | 복합 결함 정책 | 보상만 교정한 뒤 | 같은 seed 정상 참조 |
|---|---:|---:|---:|
| 수평 속도 추종 RMSE, m/s | 1.177695 | 1.172526 | 0.162601 |
| yaw 각속도 추종 RMSE, rad/s | 0.790999 | 0.764480 | 0.295032 |
| 낙상률 | 0% | 0% | 0% |
| 평균 생존, 초 | 20 | 20 | 20 |

첫 교정은 300회 학습과 동결 26조건 평가를 완료했지만 추종 오차 두 기준을 통과하지 못해 `not_recovered`였다. 낙상과 생존만 보면 이 실패를 놓칠 수 있다. 새 체크포인트·텔레메트리·미회복 영수증을 [후속 관측 계약](../evals/lineage_post_correction_inputs_v1.py)으로 묶어 두 번째 진단의 입력으로 사용한다. 이미 복원한 보상 성분은 적용 교정 영수증을 근거로 후보에서 제외하며, 숨은 결함 정답은 모델에 주지 않는다.

## 미회복 뒤 두 번째 진단

[후속 관측 집계](../evals/results/lineage_compound_s2307_20261031_v1/followup/summary.json)에서 세 paired 검사는 `P_episode normal → P_noise normal → P_physics normal`이었다. 모델은 세 검사 한도에서 `P_optimizer_updates`를 선택하지 않았다. 남은 후보는 모터·가치 갱신·갱신 횟수 세 가지였으며, 선택기의 최종 상태는 예산을 소진한 `unidentifiable`이었다. 진단은 HOLD이고 두 번째 교정·재학습은 수행하지 않았다.

[모델 원문과 선택](../evals/results/lineage_compound_s2307_20261031_v1/followup/model.json)에는 실제 NIM 응답 세 건을 연결했다. 세 번째 선택은 앞선 두 검사 결과를 인용했지만, 남은 원인을 확정하는 검사로 이어지지 못했다. 관측을 인용했다는 사실과 올바른 검사 선택은 구분한다.

첫 교정 학습의 [사후 계측](../evals/results/lineage_compound_s2307_20261031_v1/followup/optimizer-monitor.json)은 앞선 세 update 모두 예상 optimizer step 20회 대비 실제 4회를 기록했다. 이 계측은 모델 입력 fact에 포함되지 않았고, 진단 고리의 승인된 `P_optimizer_updates` 검사로 합산하지 않았다. 남은 실패의 다음 개발 대상은 이처럼 이미 측정된 학습 루프 관측을 진단 입력에 연결하는 것이다. 연결 뒤 검사 선택과 회복이 나아지는지는 새 등록으로 확인해야 한다.

두 완료 검사와 세 모델 응답은 [재개 실행기](../evals/lineage_post_diagnosis_resume_v1.py)에서 검증해 재사용했고, 마지막 검사만 새로 실행했다. 추가 API·학습은 없었다. 다만 controller가 세 번째 normal 뒤에도 상태가 `open`이어야 한다고 검사해 종료 처리에서 실패했다. 실제 선택기 상태는 예산 소진을 나타내는 `unidentifiable`이었다. 원래 실패를 완료 결과로 바꾸지 않고, [CPU 집계기](../evals/lineage_post_diagnosis_finalize_v1.py)가 완료된 12사건 원장과 양쪽 실측을 재검산했다. 공개 결과의 `controller_status=invalid_finalization`과 `analysis_status=complete`는 이 차이를 보존한다.

이번 실행은 보상 성분 교정 뒤에도 실패가 남는 것을 잡았고, 확정되지 않은 두 번째 수리를 막았다. 복합 사례 회복 성공이나 모델 버전 간 정확도 차이의 근거로 쓰지 않는다.
