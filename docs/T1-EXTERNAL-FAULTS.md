# T1 외부 원인 결함 안전성 시험 — 사전등록

작성 2026-10-09. 이 문서·주입 코드·판정 코드는 X 결함의 어떤 학습보다 먼저 커밋한다. 커밋 뒤에 바꾸면 무엇을 왜 바꿨는지 7절 아래에 기록한다. 배경 조사는 [NEXT-STEPS-20261009.md](NEXT-STEPS-20261009.md) 1절, 순서는 [IMPLEMENTATION-ORDER.md](IMPLEMENTATION-ORDER.md) 7절 T1이다.

## 1. 목적과 주장 경계

P1-A probe 고리는 P0-C 숨은 결함 18건 중 17건(전수)·14건(에이전트 상위 3 가르기)을 틀린 확정 없이 확정했다. probe와 결함을 같은 사람이 만들었으므로 이 결과는 "고리가 동작한다"까지만 말한다.

T1은 원인이 공개 이슈·PR로 확정된 Isaac Lab 결함을 넣고, **동결된 probe 고리가 틀린 원인을 확정하지 않는지**를 잰다. 동결 probe 6종은 이 결함들을 겨누지 않는다(아래 2절 "probe 대응"). 따라서 기대 결과는 식별이 아니라 "식별 불가로 멈춤"이다.

- 주 지표: 선택 방식별 **오답 확정 수**(목표 0).
- 주장하지 않는 것: 외부 결함을 진단한다는 일반화, 검출력, 비용 절감.

## 2. 후보와 주입

대상은 Isaac Lab v2.1.1(`90b79bb`), Isaac Sim 4.5, `Isaac-Velocity-Flat-Unitree-Go2-v0`, rsl_rl 2.3.3, 4096 env × 300회(P0-C와 같은 예산)다. 주입 코드는 [bench/external_faults.py](../bench/external_faults.py)다. `bench/hidden_faults.py`가 `X`로 시작하는 `TRIAGE_FAULT`를 이 모듈로 넘기므로 학습과 probe가 같은 경로로 주입된다.

| ID | 상류 근거 | 넣는 것 | 크기 출처 | 사전등록 범주 | params diff |
|---|---|---|---|---|---|
| X1 | [#7311](https://github.com/isaac-sim/IsaacLab/issues/7311)(closed). 누적 코드는 [v2.2.1](https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/source/isaaclab/isaaclab/envs/mdp/events.py)(`0f00ca2`)에 출하 | `base_com`을 reset 모드로 켜고 `randomize_rigid_body_com` 본문을 v2.2.1 그대로(현재 COM을 읽어 더함)로 교체 | velocity `EventCfg.base_com` 기본값: x·y ±5 cm, z ±1 cm | physics | 짝 기준(X1_REF)과 없음. 순정 대비 `events.base_com` 보임 |
| X1_REF | develop(`6681b64`, 2026-10-08)의 수정: 첫 호출의 COM을 기본값으로 저장하고 매번 거기에 더함 | X1과 같은 설정. 본문을 수정 의미로 교체(v2.1.1 API로 옮김) | 같음 | — (기준) | — |
| X2 | [#7613](https://github.com/isaac-sim/IsaacLab/issues/7613)(closed), 수정 [PR #7614](https://github.com/isaac-sim/IsaacLab/pull/7614)(2026-09-11 병합) | `scene.contact_forces.history_length = 0`(`ContactSensorCfg` 기본값). v2.1.1은 history > 0일 때만 매 물리 스텝 갱신 | 설정 한 줄(크기 없음) | termination | 순정 대비 한 줄 보임 |
| X3 | [#915](https://github.com/isaac-sim/IsaacLab/issues/915)의 원인, 수정 [PR #1584](https://github.com/isaac-sim/IsaacLab/pull/1584)(`574abc4`, 2025-01-02 병합) | `push_robot`을 켜고 `push_by_setting_velocity` 본문에서 #1584가 바꾼 한 줄만 되돌림(`vel_w += 표본` → `vel_w[:] = 표본`) | velocity `EventCfg.push_robot` 기본값: 10~15초, x·y ±0.5 m/s | physics | 짝 기준(X3_REF)과 없음. 순정 대비 `events.push_robot` 보임 |
| X3_REF | v2.1.1 함수 그대로(#1584 이후) | X3과 같은 설정 | 같음 | — (기준) | — |

- 이벤트 항은 velocity 기본 `EventCfg()`에서 가져온다. 설계자가 크기를 고르지 않는다.
- X1을 "v2.1.1의 결함"이라고 하지 않는다. v2.1.1 함수는 모든 env 행에 더해(`coms[:, body_ids, :3] +=`) 일부 env만 reset되면 shape가 맞지 않는다. X1은 **v2.2.1~v2.3.x에 출하된 함수 하나를 v2.1.1에 옮긴 재현**이다.
- X1_REF는 develop의 클래스형 수정을 v2.1.1의 함수형 API로 옮긴 것이다(기본 COM은 자산 객체에 저장). 상류 코드 그대로가 아니라는 점을 결과와 함께 적는다.
- `train.py`는 `gym.make` 뒤에 params를 dump하므로 바꾼 설정은 params에 남는다. 짝 사이 diff는 실행 이름뿐이어야 한다(5절 검사).
- 주입 확인: 환경 생성 뒤 이벤트 관리자·센서 설정을 읽어 반영되지 않았으면 학습 전에 멈춘다(`after_env`). 확인 줄은 비공개 로그에만 남는다.

**probe 대응(사전 추론).** `P_physics`는 마찰·질량·물리 dt만 읽고 COM·push를 읽지 않는다. `P_episode`는 time_out으로 끝난 에피소드 길이를 본다. X2가 base_contact 종료를 늘리면 time_out 자체가 드물어 판정이 unknown이나 normal로 나올 수 있다. 결함을 겨누는 probe가 없으므로 정답 확정이 나와도 "그 probe가 그 결함을 겨누지 않았다"를 함께 쓴다.

**제외한 후보.** X4 `feet_air_time` 첫 접촉 누락(#7283)은 기존 모든 실행에 대칭으로 들어 있고 우리 설정에서 효과가 작을 것으로 본다. T2의 "원인 있음 + 행동 정상" 사례로 쓴다. 그 밖의 제외 사유는 NEXT-STEPS 1.2절 표를 따른다.

## 3. D1 smoke — 메커니즘 재현

[bench/smoke_external.py](../bench/smoke_external.py)가 결함마다 Isaac 프로세스 하나(64 env, 학습·정책 없음)로 잰다. 결과는 `evals/results/t1_smoke_<date>/`에 남는다.

| 측정 | 실행 | 절차 | 재현 기준 |
|---|---|---|---|
| X1 Bug 1(누적) | X1_REF·X1 | 짝수·홀수 env를 번갈아 reset 20회(env마다 10회). 첫 reset 전 COM 대비 x·y 편차의 최댓값 | X1에서 env 50% 이상이 ±5 cm 밖, X1_REF는 0% |
| X1 Bug 2(텔레포트 손실) | NONE·X1_REF·X1 | 로봇을 x로 3 m 옮긴 뒤 모두 reset, 물리 4스텝 뒤 x 위치 | X1_REF 또는 X1에서 env 50% 넘게 2 m 밖에 남음. NONE은 5% 이하여야 측정이 유효 |
| X2 접촉력 고착 | NONE·X2 | 뒤집어 base를 바닥에 닿게 한 뒤(정책 주기로 읽음) 마지막 읽기 직후 1.5 m 공중으로 옮기고, 4스텝마다 5회 읽음 | 닿았던 env 중 50% 이상이 공중 5회 모두 > 1 N(base_contact 임계값), NONE은 0%. 두 실행 모두 50% 이상 닿아야 유효 |
| X3 덮어쓰기 | X3_REF·X3 | 서 있는 로봇에 x 1 m/s를 주고 push 이벤트를 모든 env에 건 뒤 1스텝 | X3_REF 평균 vx가 1.0±0.15, X3 평균 vx가 0±0.15 |

처리(NEXT-STEPS 1.3절 판정표):

- X1: Bug 1 재현·Bug 2 없음 → 짝 설계. Bug 1·2 모두 재현 → 기준을 순정 Go2 flat으로 바꾸고, 결함 쪽 params diff에 `events.base_com`이 보인다고 기록하고 진행. Bug 1 없음 → 제외. NONE 텔레포트가 안 되면 측정 절차를 고친다(크기·결함 정의는 바꾸지 않는다).
- X2: 재현 → 포함. 재현 안 됨 → 제외. 바닥 접촉을 만들지 못하면 측정 절차를 고친다.
- X3: 재현 → 포함. 아니면 제외.

## 4. D2 보정 — 상류 크기에서 깨지는가

dev seed 7에서 `t1cal_<X>_s7`과 짝 기준 `t1cal_<X>_REF_s7`을 학습한다(`python bench/run_t1.py cal`). X2와 순정 기준을 쓰는 X1의 기준은 P0-C 보정의 `p0ccal_NONE_s7`이다(같은 경로·seed·예산, 다시 학습하지 않음). 최종 체크포인트를 `fixed_eval_v1`로 평가하고 [evals/t1_calibration.py](../evals/t1_calibration.py)로 판정한다.

- 깨짐: 짝 기준 대비 P0-C 보정 규칙 그대로(`recovery.compare`·`verdict`, 정상 변동 0). 생존 비 < 0.90, 낙상 > 기준 + 0.05, 선속도 오차 비 > 1.10, 회전 오차 비 > 1.10 중 하나.
- **크기를 키우지 않는다.** 상류 기본 크기에서 깨지지 않으면 제외한다. 외부 결함의 크기를 설계자가 고르면 시험의 의미가 사라진다.
- 짝 기준의 타당성: 기준 실행이 걸어야 한다. 선속도 RMSE < 제자리 RMSE × 0.5, 낙상 ≤ 0.05. 아니면 그 후보를 제외한다.
- 텔레메트리만 보고 결함을 알아볼 수 있는지는 기준에 넣지 않는다.

## 5. D3 holdout

- seed 2026·2027·2028. 학습 seed는 P0-C holdout과 같고 결함은 새것이다. X2의 기준은 P0-C의 `baseline_p0c_s<seed>`를 그대로 쓴다.
- 사례 ID `e01~`은 남은 결함에 비밀 seed로 섞어 대응한다(`python bench/t1_cases.py --faults ...`, 정답표 `bench/private/answer_key_t1.json`). 짝 기준 실행은 `ref_<사례>_s<seed>`다. 학습 출력에는 결함 이름을 쓰지 않는다. 정답표는 평가가 끝난 뒤 공개한다.
- 학습: `python bench/run_t1.py holdout`. 텔레메트리는 `bench/runs_t1/`.
- 짝마다 params를 P0-C와 같은 방식(`evals/baselines.py --run-params`의 `run_params_check`)으로 대조해 실행 식별 값 말고 차이가 없는지 확인한다(X2와 순정 기준 X1은 2절의 예외를 기록).
- seed마다 짝 기준이 걷는지(4절 기준) 확인한다. 걷지 않는 seed의 짝은 채점에서 빼고 이유를 남긴다.
- 고정 평가에서 결함 실행이 짝 기준 대비 unhealthy가 아니면 "실패 없음"으로 따로 표시한다.

## 6. D4 측정

- 고정 평가: 모든 holdout 실행을 `fixed_eval_v1`로 평가한다(결함 없는 순정 환경, P0-C와 같음).
- probe: 각 실행을 그 실행과 같은 결함 환경에서 6종 한 번씩 잰다(`evals/run_probes.py`, 새 tag). 판정은 사례마다 정답표의 기준 실행 측정값 대비 `probe_loop.classify`다(`evals/loop_compare.py`가 `reference`를 읽음).
- 선택 방식: exhaustive, fixed, random(seed 0~4), discriminate:rules(동결 P0-A2 `rule_features` 상위 3), discriminate:agent(P0-C와 같은 프롬프트·도구·모델의 상위 3).
- 순위 입력: 규칙·에이전트가 보는 작업공간은 사례 텔레메트리와 그 사례의 기준 텔레메트리다. 작업공간 생성 코드는 holdout 학습 뒤, 순위 실행 전에 커밋한다. 에이전트 프롬프트·도구는 바꾸지 않는다.
- 결과는 새 `evals/results/t1_*` 폴더에 쓰고 기존 결과를 덮어쓰지 않는다.

## 7. 판정·해석 규칙

- 사례별 결과: 정답 확정 / 오답 확정 / 식별 불가 / 근거 없음(none_supported). 고정 평가에서 "실패 없음"이면 함께 표시한다.
- 주 지표: 방식별 오답 확정 수(목표 0). 분모는 채점한 결함 실행 수다.
- 문구는 아래 D4 측정 전 분기표를 따른다. 오답이 1건 이상이면 그 수·사례·해당 probe 판정 규칙을 한계로 공개한다. 정답 확정이 나와도 그 probe가 그 결함을 겨누지 않았다는 사실을 함께 쓴다.
- 이 단계에서 probe 정의·임계값을 추가하거나 고치지 않는다. 고치면 그 사실과 이유를 이 문서에 기록하고, 고친 뒤의 수치를 T1 결과로 쓰지 않는다.

### D4 측정 전 결과 문구 등록 (2026-10-09)

D3 학습 중이며 holdout 고정 평가·probe 측정 전인 시점에 아래 해석을 등록한다. 보정 통과 후보는 X1 한 종류이고 예정된 결함 실행은 seed 3개다. `N`은 기준 보행 조건을 충족해 채점한 짝 수, `F`는 그중 고정 평가에서 unhealthy인 결함 실행 수, `k`는 선택 방식별 오답 확정 수다. 기준이 걷지 않는 짝은 제외 이유와 함께 남긴다.

| 고정 평가 결과 | 사용할 문구와 분모 |
|---|---|
| 유효한 기준 없음(`N=0`) | "기준 보행 조건을 충족한 짝이 없어 T1을 채점하지 못했다." 강건성·오답 0을 주장하지 않는다. |
| 모두 실패 없음(`N>0, F=0`) | "채점 가능한 N개 실행 모두 사전등록한 unhealthy 기준을 넘지 않았다. 이 task·학습 예산·seed에서 학습 실패 재현은 0/N이었다." probe 결과는 실패 없음 조건의 오답 확정 k/N으로 표시하며, 실패 진단 성능으로 해석하지 않는다. |
| 일부 실패 없음(`0<F<N`) | "학습 실패 재현 F/N, 실패 없음 (N-F)/N." 전체 오답 확정 k/N과 함께 unhealthy·실패 없음 집단의 오답 수와 각 분모를 나눠 적는다. |
| 모두 unhealthy(`F=N>0`) | "채점한 N개 실행 모두 사전등록한 unhealthy 기준을 넘었다. 동결 probe 고리의 오답 확정은 k/N이었다." |

`N=0` 분기가 모든 오답 수 문구보다 우선한다. 이때 오답 비율은 정의하지 않는다. `N>0`인 경우에만 아래 오답 수 조건을 적용하며, 나머지 기록·해석 제한은 모든 경우에 적용한다.

- `k=0`: "이 표본에서 오답 확정 0/N"으로 한정한다. 한 결함 종류의 seed 반복이며 다른 결함·task에 대한 보장이나 통계적 우위를 주장하지 않는다.
- `k>0`: 방식·사례·오답 범주·확정에 사용된 probe·원시 측정값·동결 판정 규칙을 공개한다. probe나 임계값을 고쳐 T1 수치를 다시 만들지 않는다.
- 정답 범주를 확정한 경우에도 probe가 COM 누적 결함을 겨누지 않았음을 적는다. 범주 일치를 외부 결함의 메커니즘 식별로 표현하지 않는다.
- 식별 불가와 `none_supported`는 따로 센다. 누락·실패한 probe는 unknown으로 남기고, 실행 자체가 빠진 사례를 분모에서 조용히 없애지 않는다.
- random의 5개 순서 반복은 순서별 N건으로 표시한다. 합계 5N개 셀을 독립 학습 표본으로 세지 않는다.
- probe의 `elapsed_s`만으로 GPU 절감이나 종단 비용 절감을 주장하지 않는다. 에이전트와 규칙의 우위도 주장하지 않는다.

**동결 파일(이 커밋 기준, LF 정규화 SHA256).**

| 파일 | SHA256 |
|---|---|
| `src/rl_triage/probe_loop.py` | `0f2da7fc753efaed3bc0c07d5ff699c7beea51630df8a474daf4d1c478d131e1` |
| `evals/probes.py` | `8bbe5bc5daaa94a2fbe88ea34dfb23fa298a1f43c4734cf4584fd6b790c16601` |

판정 코드(`bench/smoke_external.py`의 `judge`, `evals/t1_calibration.py`)와 주입 코드(`bench/external_faults.py`)도 이 커밋 기준으로 동결한다.

## 8. 위협 요인

- 주입 코드는 probe를 만든 사람이 옮겼다. 결함의 의미와 크기는 상류에서 왔지만, 짝 기준 함수(X1_REF)는 옮긴 코드다.
- 사전등록 범주는 개입 관점으로 한 사람이 정했다. X2는 원인 위치로는 센서·물리 백엔드다.
- X2는 params diff에 보이므로 설정 diff 규칙이 쉽게 맞힐 수 있다. T1의 질문은 probe 고리의 오답 확정이지 설정 diff의 정확도가 아니다.
- 표본은 결함 최대 3종 × seed 3이다. 오답 0도 "이 표본에서 없었다"까지만 말한다.
- 학습 seed를 P0-C와 공유한다. 결함과 기준(X1·X3)은 새로 학습한다.

## 9. 실행 순서

```bash
python bench/smoke_external.py --tag t1_smoke_<date>                       # D1(GPU 수 분)
python bench/run_t1.py cal --faults <smoke 통과>                           # D2 학습
python evals/run_fixed_eval.py --tag t1_calibration_<date> --runs t1cal_... p0ccal_NONE_s7
python evals/t1_calibration.py evals/results/t1_calibration_<date> --smoke evals/results/t1_smoke_<date>/summary.json
python bench/t1_cases.py --faults <보정 통과> [--stock-reference X1]      # 비밀 대응
python bench/run_t1.py holdout                                             # D3 학습
python evals/run_fixed_eval.py --tag t1_fixed_eval_<date> --runs ...       # D4
python evals/run_probes.py --tag t1_probes_<date> --runs ...
python evals/loop_compare.py --probes evals/results/t1_probes_<date> --tag t1_loop_compare_<date> \
    --key bench/private/answer_key_t1.json --ranking rules=... --ranking agent=...
```

## 10. 결과

### D1 smoke (2026-10-09)

사전등록 커밋 `41b3642`의 코드로 64 env에서 실행했다. 결과는 [t1_smoke_20261009/summary.json](../evals/results/t1_smoke_20261009/summary.json)이며 누락 실행은 0/6이다.

| 후보 | 실측 | 3절 판정 |
|---|---|---|
| X1 | COM 범위 이탈 X1 100%, X1_REF 0%; 텔레포트 손실 NONE·X1_REF·X1 모두 0% | Bug 1 재현, Bug 2 없음 → 짝 설계 유지 |
| X2 | 바닥 접촉 NONE·X2 모두 100%; 공중 5회 잔류 접촉력 NONE 0%, X2 100% | 포함 |
| X3 | push 뒤 평균 vx: X3_REF 0.9967 m/s, X3 -0.0118 m/s | 포함 |

크기·주입·판정·probe 파일은 수정하지 않았다. D2는 X1·X2·X3에 대해 4096 env × 300회, dev seed 7로 진행한다. D1의 메커니즘 재현만으로 학습 실패나 진단 성능을 주장하지 않는다.

### D2 보정 (2026-10-09)

5개 새 학습을 모두 4096 env × 300회, seed 7로 마쳤다(exit 0, 최종 체크포인트 `model_299.pt`). X2의 기준은 사전등록대로 기존 `p0ccal_NONE_s7`을 재사용했다. 6개 정책의 `fixed_eval_v1` 결과와 동결 판정은 [t1_calibration_20261009/t1_calibration.json](../evals/results/t1_calibration_20261009/t1_calibration.json)에 있다. 모든 기준 실행이 4절의 보행 조건을 충족했다.

| 후보 | 선속도 오차 비 | 회전 오차 비 | 생존 비 | 낙상 비율(결함 / 기준) | 판정 |
|---|---:|---:|---:|---:|---|
| X1 | 1.1031 | 1.0373 | 1.0000 | 0 / 0 | 선속도 오차 비 > 1.10 → 포함 |
| X2 | 0.9767 | 0.9642 | 1.0000 | 0 / 0 | 상류 크기에서 unhealthy가 아니므로 제외 |
| X3 | 1.0035 | 1.0322 | 0.9998 | 0.03846 / 0.03846 | 상류 크기에서 unhealthy가 아니므로 제외 |

X1의 선속도 RMSE는 기준 0.147392 m/s, 결함 0.162590 m/s다. 비율이 임계값을 0.0031만 넘으므로 큰 성능 붕괴로 해석하지 않는다. 이는 dev seed 7의 포함 판정이며, holdout 결과를 대신하지 않는다.

실행 식별 값을 제외한 params diff는 X1·X3의 짝에서 0개, X2와 순정 기준 사이에서 사전등록한 `env.scene.contact_forces.history_length` 한 줄(3 → 0)이다. 주입·크기·probe·판정 소스의 LF 정규화 해시는 `41b3642`와 일치한다. 판정 결과의 `git_dirty: true`는 커밋 전 결과 폴더도 검사에 포함된 상태를 기록한 값이며 수정하지 않았다.

D3는 X1 한 종류의 seed 2026·2027·2028과 각 짝 기준만 학습한다(결함 3건, 기준 3건). X2·X3의 크기를 키우거나 probe를 조정하지 않는다.

### D3 holdout 학습 (2026-10-09)

[bench/runs_t1](../bench/runs_t1/)의 holdout 6개 실행을 모두 4096 env × 300회로 마쳤다. 각 실행은 exit 0, 24개 지표 × 300시점, `model_299.pt`가 있으며 세 짝 모두 실행 식별 값 외 params diff가 0개다.

| seed | 기준 학습 wall time(s) | 결함 학습 wall time(s) |
|---|---:|---:|
| 2026 | 614.7 | 674.5 |
| 2027 | 592.8 | 733.1 |
| 2028 | 873.6 | 1871.3 |

시간 합계는 5360.0초다. 마지막 seed 실행 중 다른 프로세스의 GPU 메모리 점유를 관찰했으므로 이 시간표로 처리 속도나 비용 우위를 비교하지 않는다. 보행 조건·실패 없음 판정은 아래 D4 고정 평가 기록으로 확인한다.

학습 뒤 추가한 [build_workspace_t1.py](../bench/build_workspace_t1.py)는 사례별 기준 텔레메트리·params를 등록하고, [t1_rankings.py](../evals/t1_rankings.py)는 동결 규칙과 P0-C 에이전트를 같은 입력에 적용한다. 정답표는 작업공간에 복사하지 않는다. 아래 입력·코드를 커밋한 뒤 순위 실행을 시작한다.

```bash
python bench/build_workspace_t1.py register
python bench/build_workspace_t1.py build
python evals/t1_rankings.py --mode rules --tag t1_rules_20261009
TRIAGE_ANALYSIS_SANDBOX=required uv run --no-sync python evals/t1_rankings.py --mode agent --tag t1_agent_20261009
python evals/t1_holdout_summary.py evals/results/t1_fixed_eval_20261009
```

에이전트 실행은 Linux에서 정답표·`/proc/self/environ` 읽기 거부를 확인한 뒤 시작한다. 프롬프트·도구·주입·probe·임계값은 바꾸지 않는다. D4 결과 문구는 선행 커밋 `1299579`에 등록했다.

### D4 고정 평가·순위 (2026-10-09)

입력·실행 코드 커밋 `29ce990` 뒤 고정 평가 6개와 각 방식의 순위 3개를 실행했다. 다른 GPU 작업이 해제된 것을 확인한 뒤 Isaac 평가를 시작했다. 결과 문구는 측정 전 커밋 `1299579`의 `0<F<N` 분기에 해당한다. **유효한 짝 3개 중 unhealthy 2/3, 실패 없음 1/3**이다. 여기서 unhealthy는 사전등록한 상대 오차 기준을 넘었다는 뜻이며, 낙상이나 보행 중단을 뜻하지 않는다.

[holdout_summary.json](../evals/results/t1_fixed_eval_20261009/holdout_summary.json)의 원시 통계·체크포인트 SHA는 6/6 검증했고, 기준 보행 조건은 3/3 충족했다. 평가 조건은 `fixed_eval_v1`, 1040 env, 1000 step, 평가 seed 2026, action scale 0.25다. 모든 실행의 낙상 비율은 0, 평균 생존 시간은 20.0초다.

| 학습 seed | 선속도 오차 비 | 회전 오차 비 | 고정 평가 판정 |
|---|---:|---:|---|
| 2026 | 1.0432 | 1.0575 | healthy, 실패 없음 |
| 2027 | 0.8764 | 1.2435 | 회전 오차 비 > 1.10, unhealthy |
| 2028 | 1.1479 | 1.0233 | 선속도 오차 비 > 1.10, unhealthy |

순위는 [규칙 기록](../evals/results/t1_rules_20261009/rankings.json)과 [에이전트 기록](../evals/results/t1_agent_20261009/rankings.json)에 있다. 각 방식은 계획 3개·완료 3개·시도 3개로 인프라 재시도 없이 끝났다. 에이전트 모델은 P0-C와 같은 `nvidia/nemotron-3-super-120b-a12b`다. Landlock canary는 3/3 통과했고 분석 호출 14/14에 Landlock이 적용됐다.

| 방식 | seed별 상위 세 범주(2026 / 2027 / 2028) | 순위 실행 경과 시간 합(s) |
|---|---|---:|
| 규칙 | optimizer, reward, physics / 동일 / 동일 | 0.200282 |
| 에이전트 | exploration, actuator, optimizer / exploration, optimizer, reward / exploration, actuator, optimizer | 724.2 |

두 방식의 입력 해시 13개와 코드 해시 10개가 일치한다. 이 순위 표는 원인 확정 결과가 아니며, 경과 시간은 순위 실행의 기록이다. probe의 측정 시간이나 전체 작업 비용을 대신하지 않고, GPU 절감·LLM 우위로 해석하지 않는다.

### D4 probe·선택 방식 비교 (2026-10-09)

[t1_probes_20261009/probes](../evals/results/t1_probes_20261009/probes/)의 6개 실행 × 6종 측정은 36/36 정상 종료했고 체크포인트 SHA가 실제 `model_299.pt`와 모두 일치했다. 세 결함 실행의 probe 판정 18개는 전부 normal이었다. 동결 파일 8개의 LF 정규화 바이트는 `41b3642`와 일치한다. 결함 크기·probe·임계값은 변경하지 않았다.

[loop_compare.json](../evals/results/t1_loop_compare_20261009/loop_compare.json)과 [사례별 기록](../evals/results/t1_loop_compare_20261009/loop_compare.jsonl)의 결과는 다음과 같다. **이 표본에서 오답 확정은 0/3이지만 원인 범주 확정도 0/3이다.** 동결 probe 고리는 COM 누적 결함을 식별하지 못했고, 모든 사례에서 후보 가설의 근거가 없다는 `none_supported`로 끝났다. 이를 식별 불가(`unidentifiable`)나 진단 성공으로 바꾸어 표현하지 않는다.

| 방식 | 채점 수 | 정답 확정 | 오답 확정 | 식별 불가 | none_supported | unhealthy 오답 | healthy 오답 | 평균 probe 수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 전수 | 3 | 0 | 0 | 0 | 3 | 0/2 | 0/1 | 6 |
| 고정 순서 | 3 | 0 | 0 | 0 | 3 | 0/2 | 0/1 | 6 |
| random 0·1·2·3·4 각각 | 각 3 | 각 0 | 각 0 | 각 0 | 각 3 | 각각 0/2 | 각각 0/1 | 각각 6 |
| 에이전트 순위 가르기 | 3 | 0 | 0 | 0 | 3 | 0/2 | 0/1 | 3 |
| 규칙 순위 가르기 | 3 | 0 | 0 | 0 | 3 | 0/2 | 0/1 | 3 |

unhealthy 집단은 seed 2027·2028의 2건, healthy 집단은 seed 2026의 1건이다. 각 방식에서 unhealthy의 `none_supported`는 2/2, healthy는 1/1이다. random은 같은 세 학습 결과를 다섯 순서로 재생한 15개 셀이며 독립 학습 표본 15개가 아니다.

전체 36개 probe의 `measure()` 구간 합은 397.8초다. 비교 JSON의 방식별 측정 시간은 이 결과표에서 선택한 probe의 시간을 더한 값이며 기동·reset·checkpoint 로딩·모델 순위·사람 검토 비용을 포함하지 않는다. 이 시간으로 GPU 절감 또는 종단 비용 우위를 주장하지 않는다. 비교 결과의 `git_dirty: true`는 probe 결과 폴더가 미커밋 상태인 시점의 기록이며 바꾸지 않았다.

모든 순위·고정 평가·probe·선택 방식 평가가 끝난 뒤 [정답표](../bench/answer_key_t1.json)를 공개 사본으로 고정했다. 비공개 원본과 바이트가 같고 LF SHA256은 `d8cd3f3b82b315ea78dcfaf7e3b36664d8cd3df8cf8d58b90076615a2681d483`이다. 공개 사본으로 재계산하려면 기존 결과와 다른 새 tag를 사용한다.

```bash
python evals/loop_compare.py --probes evals/results/t1_probes_20261009 --tag t1_replay_<new-tag> \
    --key bench/answer_key_t1.json --ranking rules=evals/results/t1_rules_20261009 \
    --ranking agent=evals/results/t1_agent_20261009
```

표본은 외부 결함 한 종류의 seed 반복 3건이다. 다른 결함·task에 대한 보장, 통계적 우위, LLM 필요성의 증거로 확대하지 않는다.
