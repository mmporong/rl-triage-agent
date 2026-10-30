# Nemotron 3.5 영상 입력의 검사 선택 가치 사례

**결론부터:** 등록 결함 사례에서 Nemotron 3.5의 이미지 입력이 검사 선택에 더하는 값은 현재 근거로 성립하지 않는다. 이미지로만 원인이 갈리는 사례를 찾지 못했다. 이 문서의 생성기는 그 사실을 해결로 표시하지 않으며, 영상 조건은 render-sync가 확인된 F2 프레임을 새로 수집하기 전까지 차단한다. 지금 실행할 수 있는 비교는 같은 수치에서 3.5와 3 Super의 차이(모델 변경 효과)다.

모든 개발 계보는 이미 본 자료다. 아래 수치는 결과 파일에서 계산한 개발 관찰이며 성능·우위·절감 근거가 아니다.

## 먼저 가른 것

| 후보 | 이미지 없이 풀리는가 | 이미지로 구별되는가 | 판단 |
|---|---|---|---|
| F1 보상 결함 | 풀린다. 기존 텍스트 첫 선택 `P_reward`, 같은 checkpoint receipt abnormal([s2034](../evals/results/e2e_demo_v3_1_f1_s2034_20261013/demo_result.json), [s2080](../evals/results/vl_aux_f1_s2080_20261016/demo_result.json)) | 제자리 화면. B4의 F1 원래 결함 영상 4개 중 3개가 `in_place=yes`([개별 판정](../evals/results/nemotron_endpoint_bottlenecks_20261016_v3/decisions.json)) | 반복하지 않는다. 특이도 대조군으로만 쓴다 |
| F2 모터 결함의 첫 검사 선택 | 정보는 있다. 3 Super는 세 계보 모두 `P_reward`를 먼저 골랐고(normal), `P_noise` normal 뒤 `P_torque` abnormal로 세 번째 검사에서 확정됐다([s2033](../evals/results/e2e_demo_v3_1_f2_s2033_20261013/demo_result.json), [s2046](../evals/results/e2e_demo_v4_f2_s2046_20261014/demo_result.json), [s2027 active](../evals/results/active_20261011_v1/cases/h06_s2027/model/initial.decision.json)). 그러나 같은 학습 수치에서 F2 범위가 F1·F5와 겹치지 않는 필드가 7개다 | 결함을 다시 적용한 Play(v9)는 몸체와 관절이 모두 정지했다. F1 제자리와 같은 부류이며 render-sync가 확인된 F2 case 프레임이 없다 | **가장 유망한 사례로 구현.** 예상 B=C, 열린 질문은 A-B |
| F2 정상 환경의 움직임/지표 불일치(B4 `motor_before`) | 정상 환경 고정 평가에서 F2만 후진을 부분 추종했다. 진단 전에 계산되는 고정 평가 수치에 이미 있다 | 10/13 Windows stock Play 녹화는 render-sync 미확인이고 같은 Play 상태 기록이 없다. 등록 녹화 명령(전진 1.0 m/s)에서는 F2도 정지한다 | 판정 입력으로 쓰지 않는다. 새 수집 필요 |
| 일시적 동작 이상(외부 원인 X2·X3 등), R1·복합 2차 진단 | — | 이 저장소에 해당 결함의 render-sync 영상과 같은 Play 상태가 없다. PPO update 횟수 결함은 시각 특징 근거가 없다 | 새 수집 필요로만 표시한다 |

같은 seed NONE 대비 학습 기록 비율(`mean_last_20pct`)에서 F2 범위가 F1·F5와 겹치지 않는 필드다. `bench/runs_p0c`의 이미 본 26개 파일에서 다시 계산한다. 임계값이나 선택 규칙을 만든 것이 아니다.

| 필드 | F2 | F1 | F5 |
|---|---|---|---|
| `Episode_Reward/dof_acc_l2` | 1.622~2.097 (6) | 0.509~0.748 (7) | 1.061~1.159 (3) |
| `Episode_Reward/lin_vel_z_l2` | 0.965~1.030 (6) | 0.564~0.768 (7) | 0.867~0.926 (3) |
| `Episode_Reward/track_ang_vel_z_exp` | 0.897~0.913 (6) | 0.923~0.995 (7) | 0.994~1.025 (3) |
| `Episode_Reward/track_lin_vel_xy_exp` | 0.237~0.265 (6) | 0.216~0.223 (7) | 0.267~0.296 (3) |
| `Loss/learning_rate` | 1.234~1.381 (6) | 1.550~1.953 (7) | 1.391~1.413 (3) |
| `Metrics/base_velocity/error_vel_yaw` | 1.316~1.373 (6) | 1.024~1.271 (7) | 0.907~1.023 (3) |
| `Train/mean_reward` | 0.278~0.313 (6) | 0.331~0.372 (7) | 0.364~0.384 (3) |

화면 쪽 근거는 다음과 같다.

- [v9 같은 Play 수치](../evals/results/agentic_motor_capture_20261028_v9/window_metrics.json): F2 몸체 수평 속도 0.0018(s2033)·0.0033(s2046) m/s, 정상 1.04·1.09 m/s. 1~6초 12관절 |관절속도| 평균은 F2 0.018·0.058 rad/s, 정상 2.78·2.40 rad/s([facts](../evals/results/agentic_motor_capture_20261028_v9/motor_capture_facts.json)). v9 네 영상은 렌더 갱신 문제로 모델 입력이 보류돼 있다.
- [P0-C 고정 평가](../evals/results/p0c_fixed_eval_20261009/runs/)와 두 F2 시연: 후진 `[-1,0,*]` 세 조건 중 최소 정규화 오차(rmse/명령 크기)는 F2 0.172~0.264(5/5), F1·F4·F5 0.820~1.002(9/9). 전진 `[1,0,0]`은 F2도 1.002~1.006이다. 녹화 명령은 [`record_agentic_state_sidecar_v1.py`](../bench/record_agentic_state_sidecar_v1.py)의 `lin_vel_x=(1.0, 1.0)`이다.

## 비교 축

| 조건 | 입력 | 지금 상태 |
|---|---|---|
| A | 3 Super + 생산 선택 계약의 수치 화면(학습 기록 24필드×case/reference, `action_scale`) | 실행 대기. 옛 모델 호출이 안 되면 판정불가 |
| B | 3.5 + A와 바이트가 같은 capsule | 실행 대기 |
| C | 3.5 + 같은 수치 + render-sync가 확인된 case·reference 프레임 6장 | 차단(새 수집 필요) |
| D | 등록 규칙 `registry_next`·`fixed_eligible` | CPU 계산 완료. 첫 선택 뒤 receipt가 없어 pending |

B-C가 영상 기여, A-B가 모델 변경 효과다. 같은 provider·호출기·세 필드 계약·검사 예산 7을 쓰고 모델 ID만 바꾼다. 새 seed 단계에서는 같은 V01 contract의 `text_only`(A·B)와 `text_and_video`(C)를 쓰며, 같은 Play 수치는 세 조건에 모두 들어간다. 과거 다른 계약의 실행(위 3 Super 첫 선택)은 A 결과로 합치지 않고 설계 노출로만 기록한다.

## 생성기

[`evals/nemotron35_value_cases_v1.py`](../evals/nemotron35_value_cases_v1.py)는 계보마다 판정 파일(실패 라벨과 고정 평가 보고서 SHA), 학습 기록, checkpoint, 같은 checkpoint의 과거 검사 receipt, 영상 후보와 렌더 근거를 결합한다. 숫자 입력은 기존 [`selection_comparison_view_v1.py`](../evals/selection_comparison_view_v1.py)의 packet→snapshot→messages를 그대로 써서 A·B 공용 capsule로 저장한다. 새 모델 호출 틀은 만들지 않았다.

- 오류로 거부: 다른 계보의 판정 파일(라벨), 다른 seed 학습 기록, 다른 checkpoint의 검사 원문·영상, 다른 영상을 가리키는 render receipt, 사전등록에 없는 입력, 바이트가 바뀐 입력, 미실행 DRAFT가 아닌 사전등록, 준비본과 다른 manifest·capsule.
- 불수용: 렌더 감사의 stale 영상, render-sync receipt가 없는 영상. 현재 v9 네 영상은 불수용이고 v11 정상 s2033 reference만 수용된다.
- 보류: A·B `pending_api`, C `blocked_new_collection_required`, receipt가 없는 후속 `pending_receipt`. 무효 응답은 HOLD이며 규칙 선택으로 바꾸지 않는다.
- `--score`는 사전등록 규칙으로 축별 결과를 분류한다. 등록 확인 검사를 고르고 같은 checkpoint의 실제 receipt가 abnormal일 때만 맞힌 것으로 세며, 선택 변화만으로 개선으로 세지 않는다.

```bash
PY=.venv/Scripts/python.exe   # 또는 uv run python
PYTHONUTF8=1 PYTHONPATH=src $PY evals/nemotron35_value_cases_v1.py --prepare --output <새-폴더>
PYTHONUTF8=1 PYTHONPATH=src $PY evals/nemotron35_value_cases_v1.py --check --manifest <새-폴더>/manifest.json
PYTHONUTF8=1 PYTHONPATH=src $PY evals/nemotron35_value_cases_v1.py --describe
PYTHONUTF8=1 PYTHONPATH=src $PY -m pytest -q -p no:cacheprovider tests/test_nemotron35_value_cases_v1.py
```

현재 준비 결과: 네 계보 모두 C 차단, D의 `registry_next`는 `P_value` 뒤 `P_physics` receipt가 없어 pending, `fixed_eligible`은 `P_episode` receipt가 없어 pending, 전체 결과 `inconclusive`, 외부 호출 0.

## 사전등록 DRAFT

[`bench/protocols/nemotron35_value_cases_20261030_v1.json`](../bench/protocols/nemotron35_value_cases_20261030_v1.json)은 미실행 DRAFT다. 결과 전에 다음을 고정했다.

- 사전 진술: 영상 선택 가치 `not_established`, 예상 B=C.
- 단위는 계보(case×training seed). 맞힘은 등록 확인 검사 선택+같은 checkpoint receipt abnormal. 영상 축은 프레임 인용이 있어야 한다.
- 축별 결과 `supported`(새 독립 계보 2개 이상, 반대 방향 0), `development_observation_only`, `not_supported`, `regression`, `inconclusive`와 각 문구.
- 개발 단계 전제: A·B·D만 실행, API 논리 호출 8회(요청 상한 24), GPU·학습 0. 새 seed 단계 전제: seed 2개 이상, seed마다 NONE·F1·F2 학습과 render-sync 녹화, 선택된 검사 실행, API 논리 호출 6회.

## 대안: 화면 주장과 시뮬레이션 상태의 불일치 탐지

영상 검사 선택 대신 제안하는 기능은 하나다. 3.5가 프레임에서 로봇 보임·걷기·제자리를 주장하고, 같은 Play의 몸체 속도·명령과 모순되면 진단을 HOLD하고 등록 render-sync 재검사를 요청한다. 렌더된 화면이 시뮬레이션 상태를 보여 주는지는 상태 수치만으로 알 수 없으므로 영상 관찰이 필요하다.

| 방법 | 기존 근거에서 할 수 있는 것 | 한계 |
|---|---|---|
| 수치 상태 규칙 | 같은 Play 속도로 움직임 판정 | 렌더 결함을 볼 수 없어 v9 stale 영상도 통과 |
| 간단한 영상 규칙(렌더 감사 mask) | v9 정상 두 정책: mask가 있던 분리 표본 19개 중 초기 root 근처 17·현재 0. [v11](../evals/results/agentic_motor_render_preflight_20261028_v11/summary.json): 현재 근처 23/23 | 카메라 파라미터와 로봇 색 가정 필요. 정지 정책은 판정 불가 |
| 3.5 화면 주장+상태 | 아직 측정 없음 | 정지 정책은 마찬가지로 판정 불가 |

이동 정책에서는 mask 규칙이 이미 stale을 가르므로 3.5에 남는 값은 카메라 보정·색 mask 없이 다른 로봇·지형(원래 PC의 H1·Go2 계단 영상)에서도 같은 판정을 내는지뿐이며 근거는 아직 없다. 평가 라벨은 render-sync receipt와 위치 감사에서만 만든다. 기존 영상 7개(이동+stale 2, 이동+render-sync 2, 정지 3)로 검출·오경보·unclear를 mask 규칙과 같은 표로 세고, 계단 영상이 들어오면 같은 방식으로 늘린다. stale 영상은 이 탐지의 양성 시험 자료로만 쓰고 진단·검사 선택 입력으로 쓰지 않는다. 생성기의 `alternative_evidence`가 이 근거 파일을 SHA로 다시 읽는다.

## 데이터 노출과 범위

설계자는 B4 영상 관찰 18건, F2·F1 계보의 기존 첫 선택과 receipt, P0-C 고정 평가의 방향별 오차, 학습 기록 비율, v9 관절 수치, 렌더 감사, 3.5 fresh-frame s2221 응답(telemetry 인용 6, frame 인용 0)을 본 뒤 사례와 사전 진술을 정했다. 실패 지표를 숨겨 영상에 유리한 입력을 만들지 않았으며, 생산 선택 화면에 고정 평가 방향별 수치가 없는 것은 [`BASE_FACT_IDS`](../evals/selection_comparison_view_v1.py)의 정의 그대로다. 실제 모델·GPU 결과가 없으므로 개선·우위·절감·일반화·새 회복을 완료로 쓰지 않는다.
