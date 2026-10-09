# T2: healthy 입력에서 고리를 열지 않는 게이트 사전등록

상태는 **T2 게이트 단독 실행 완료**다. [입력·게이트 사전등록](../bench/protocols/t2_no_intervention_v1.json)은 이미 관측한 개발 보정 자료 X2·X3와 같은 기준 실행 NONE을 고정한다. [실행 코드 사전등록](../bench/protocols/t2_gate_execution_v1.json)은 원래 사전등록 SHA와 게이트 CLI·추가 의존 코드의 SHA를 고정하며 `0df6e9d`에 커밋한 뒤 측정했다. 입력·선정·행동 기준은 그대로이며 T1 원인 식별 결과와 분리한다.

## 입력과 선정 근거

모든 입력은 [T1 보정 기록](../evals/results/t1_calibration_20261009/t1_calibration.json)에서 온다. X2·X3의 메커니즘 재현은 [smoke](../evals/results/t1_smoke_20261009/summary.json)에 있고, 상류 크기 그대로 학습한 checkpoint의 고정 행동 평가는 개발 보정에서 healthy였다. 결함 크기·probe 수식·임계값을 바꾸지 않는다.

| T2 사례 | 고정 평가 입력 | 같은 seed 기준 | 기존 개발 관측: 선속도 오차 비 | 기존 개발 판정 |
|---|---|---|---:|---|
| X2 | t1cal_X2_s7 | p0ccal_NONE_s7 | 0.9767 | healthy |
| X3 | t1cal_X3_s7 | t1cal_X3_REF_s7 | 1.0035 | healthy |
| NONE | p0ccal_NONE_s7 | p0ccal_NONE_s7 | 자기 비교 | 기준 확인 |

숫자는 새 T2 측정이 아닌 기존 개발 관측이다. NONE 자기 비교는 독립 정상 반복으로 세지 않는다. 학습 seed는 7, 고정 평가 seed는 2026이며 환경 1,040개·1,000스텝·20초·기존 명령 격자를 유지한다. 사전등록에는 평가 JSON의 LF SHA256, 기록된 checkpoint SHA256, 프로토콜·판정 코드 SHA256이 들어 있다. 이번 사전등록에서 checkpoint 파일을 다시 읽어 검증한 것은 아니다.

## 먼저 확인하는 행동 게이트

1. 코드·입력 해시와 평가 프로토콜·seed·명령 격자·환경 수·관측 시간이 같은지 확인한다. 기준 실행도 절대 행동 기준을 통과해야 한다. 입력 불일치나 검증 실패는 healthy로 세지 않는다.
2. 기존 `behavior_oracle`의 상대 행동 허용 범위와 절대 행동 기준을 함께 쓴다. 정상 반복은 추가하지 않으며 기준 자기 비교 + 기존 `RC.MARGIN`으로 상대 범위를 고정한다. 정상 변동을 실측했다는 뜻은 아니다.
3. healthy이면 `do_not_open_loop`로 종료한다. 모델·probe 호출, 승인 소비·receipt, 설정 변경·재학습을 시작하지 않는다.
4. unhealthy이면 T2 정상 입력 범위 밖으로 기록하고 멈춘다. undetermined나 입력 오류이면 `stop_undetermined`로 멈춘다. 이후 진단 실행은 별도 사전등록 대상으로 남긴다.

절대 기준은 생존 비율 0.9 이상, 낙상 비율 0.05 이하, 선속도 RMSE 0.5883484054 m/s 이하, 회전속도 RMSE 0.4160251472 rad/s 이하다. 두 오차 상한은 동결 명령 격자의 제자리 오차 절반이다. 상대 기준은 생존 비 0.9 이상·추종 오차 비 1.1 이하이며 낙상 상한은 각 기준의 낙상 비율 + 0.05다. X3의 상대 낙상 상한 약 0.0885와 절대 상한 0.05를 함께 적용한다.

## 향후 검증과 문구

향후 실행은 사전등록 커밋 뒤 별도 새 tag에 기록한다. 입력 해시 확인, 기준 유효성, 행동 판정, 고리 생성 여부, 모델·probe·승인·소비·receipt·개입 수, 게이트 CPU·벽시계 비용을 남긴다. 수용 조건은 선택한 3건 모두 동결 healthy 게이트를 통과하고, 모델·probe·승인 소비·receipt·개입 각각 0건인 것이다. 파일 읽기와 행동 판정 비용을 0으로 가정하지 않는다. 게이트는 Isaac 프로세스를 시작하지 않으므로 그 실행의 GPU 벽시계는 0이다.

전부 통과하면 "이미 healthy로 관측한 개발 입력 3건에서 게이트가 추가 진단·개입을 시작하지 않았다"로 쓴다. 일부 실패하면 각 입력의 불일치·기준 실패·unhealthy·undetermined를 그대로 분리한다. 개발 입력 선정에 결과를 이미 사용했으므로 독립 보류 세트 일반화, 원인 식별 성공, 상류 버그의 영향이 항상 없다는 주장으로 확대하지 않는다.

## 게이트 실행 하네스

`evals/t2_no_intervention.py`는 사전등록·코드가 커밋됐고 해시가 같은지 확인한 뒤 평가 JSON의 행동 값을 판정한다. 모델·Isaac 실행 코드를 가져오지 않으며 기존 승인 원장 등 `evals/loops/`의 모든 파일을 실행 전후 해시로 대조한다. 결과는 사전등록의 `planned_tag`에 한 번만 쓴다. 다른 tag나 기존 결과 폴더는 판정 전에 거부한다. 이번 검증 범위는 이 게이트 CLI이며 일반 `loop.py` 호출의 라우팅을 바꾸지 않는다.

기존 보정 JSON의 조건별 원시 합계 통계와 전체 행동 지표를 기존 `behavior_oracle`로 대조한다. 기록된 checkpoint SHA도 사전등록과 비교하며 새 checkpoint 파일 검증이나 새 행동 평가로 표현하지 않는다. 비용은 사전 확인부터 판정·원장 스냅샷까지 게이트 Python의 CPU·벽시계다. Python 기동·import, Git 자식 프로세스 CPU, 결과 저장 비용은 포함하지 않는다.

```powershell
cd "$HOME/rl-triage-agent"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
.venv/Scripts/python.exe evals/t2_no_intervention.py `
  --prereg bench/protocols/t2_gate_execution_v1.json --tag t2_gate_20261009
```

위 tag의 결과가 이미 있으므로 같은 명령의 재실행은 판정 전에 거부된다. 다음 실행은 새 사전등록 파일에 새 planned tag를 고정하고 커밋한 뒤 진행한다.

## 실행 결과: t2_gate_20261009

[보정 결과 JSON](../evals/results/t2_gate_20261009/summary_v2.json)에 3건의 입력 검증·기준 계약·판정·비용·원장 스냅샷을 저장했다. 원본 `summary.json`도 보존한다.

| 사례 | 입력 SHA·기준 유효성 | 행동 판정 | 종료 결정 |
|---|---|---|---|
| X2 | 통과 | healthy | do_not_open_loop |
| X3 | 통과 | healthy | do_not_open_loop |
| NONE | 통과 | healthy | do_not_open_loop |

3/3건이 동결 게이트를 통과했고 고리를 열지 않았다. 모델 호출·probe 실행·승인·소비·receipt·개입은 각각 0건이다. 기존 고리 파일 32개의 실행 전후 LF SHA가 같으며 새로운 고리는 만들지 않았다. 새 Isaac 실행·GPU 벽시계·제어 전이는 각각 0이다. 게이트 Python CPU는 0.03125초, 사전 확인부터 종료 스냅샷까지 벽시계는 0.0905301999초다. 이 비용 범위에 결과 저장·Python 기동·import·Git 자식 CPU는 포함하지 않는다.

사전등록 수용 조건은 모두 통과했다(`accepted=true`). 세 평가 모두 조건별 원시 합계가 있고 검산을 통과해 `metrics_verified=true`다. "이미 healthy로 관측한 개발 입력 3건에서 게이트가 추가 진단·개입을 시작하지 않았다"는 게이트 동작 결과이며, 새 행동 평가·독립 일반화·원인 식별 성공을 뜻하지 않는다.

실행 사전등록의 `behavior_metric_scope`와 원본 결과의 `claim_limits`에는 원시 통계가 없다는 잘못된 설명이 있었다. 실제 세 행의 assessment는 처음부터 모두 `metrics_verified=true`였다. 입력·판정 규칙은 원시 통계가 없어야 한다는 조건을 요구하지 않았으므로 게이트 판정은 영향을 받지 않았다. 사전등록·실행 코드·원본 결과는 보존하고, 실행 후 읽기 전용 검산으로 세 assessment가 원시 입력과 같은지 확인한 뒤 `summary_v2.json`의 설명만 보정했다. 원본 SHA와 보정 이유를 `annotation_correction`에 넣었으며 측정 시각·행·비용·횟수·판정·수용 결과는 원본과 같다. 새 게이트 측정이나 GPU 실행은 하지 않았다.
