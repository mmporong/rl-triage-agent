# P0-C 새 holdout 계약 (2026-10-08, 초안)

P0-A2에서 v1 결함은 설정 diff(B0)와 같은 템플릿 재인식(최근접 dev 사례)으로 10/10 풀렸다. P0-B2 파일럿에서는 1024 env × 100회 기준 정책이 걷지 않는다는 것이 드러났다. 이 단계는 두 문제를 고친 새 holdout을 만든다([IMPLEMENTATION-ORDER.md](IMPLEMENTATION-ORDER.md) 2절 B, 3절 P0-C).

- 결함: `bench/hidden_faults.py`(정의), `bench/train_with_fault.py`(부트스트랩), `bench/run_case.ps1 -Fault`
- 어느 케이스에 어느 결함이 들어갔는지는 `bench/private/`에만 두고 평가가 끝난 뒤 공개한다.
- 학습·고정 평가는 Windows 로컬 RTX 3060에서 한다(로컬 GPU 허용, 2026-10-08). 에이전트·LLM 기준선 평가는 NVIDIA API 호출이 필요해 별도 승인을 받는다.

## 1. 학습 조건

Go2 flat, **4096 env × 300회**(Isaac Lab 기본값). P0-B2 파일럿에서 이 예산의 기준 정책은 150회부터 걷는다(선속도 RMSE 0.25 대 제자리 1.18). 기준 실행도 `TRIAGE_FAULT=NONE`으로 같은 부트스트랩을 거친다.

## 2. 결함 6종 (원인 범주마다 하나)

| 결함 | 범주 | 구현 | 설정 diff에 안 드러나는 이유 |
|---|---|---|---|
| F1 | reward | `track_lin_vel_xy_exp` 본문이 몸체 좌표 대신 월드 좌표 선속도를 씀 | 함수 객체·이름·모듈 그대로, 본문(`__code__`)만 바뀜 |
| F2 | actuator | DC 모터 토크-속도 곡선의 속도 항 5배 | 액추에이터 클래스 메서드. 설정 값(effort·velocity limit) 그대로 |
| F3 | exploration | 학습 중 행동 샘플링 노이즈 0.05 고정. 기록되는 `Policy/mean_noise_std`는 분포의 std라 정상처럼 보임 | rsl_rl 메서드 |
| F4 | optimizer | critic 기울기 0. `value_loss_coef`는 1.0 그대로 | rsl_rl 러너 생성 뒤 hook |
| F5 | physics | 환경 생성 뒤 로봇 접촉 재질 정·동마찰 0.25배 | params는 설정 객체를 dump한다. 결함은 설정 객체가 아니라 시뮬레이터의 재질 값을 바꾼다 |
| F6 | termination | `time_out`이 최대 길이의 5%에서 발생(보정 1차의 10%에서 키움). `episode_length_s`는 20초 그대로 | 함수 본문만 바뀜 |

부트스트랩은 환경 생성 직전·직후에 결함을 넣고, 넣었다는 확인 줄을 비공개 로그에만 남긴다. `bench/run_case.ps1`은 그 줄이 없으면 실패로 끝낸다.

**크기 보정:** 결함 크기(5배, 0.05, 0.25배, 10%)는 dev seed 7에서만 보정한다. 보정 판정은 결과를 보기 전에 다음과 같이 고정한다.

- seed 7의 NONE 실행과 결함 실행의 최종 체크포인트를 `fixed_eval_v1`로 평가한다(`evals/results/p0c_calibration_<date>/`).
- 결함 실행이 NONE 대비 아래 중 하나면 "깨짐"이다. P0-B2 판정 함수와 여유값을 쓰고, 정상 변동은 0으로 둔다(같은 seed 재학습이 동일했으므로).
  - 생존 비 < 0.90
  - 낙상 비율 > NONE + 0.05
  - 선속도 오차 비 > 1.10
  - 회전 오차 비 > 1.10
- 깨지지 않으면 seed 7에서만 크기를 한 단계 키운다(5배→10배, 0.05→0.02, 0.25배→0.1배, 10%→5%). 다시 학습하고 시도마다 기록한다. 두 번째 시도에도 깨지지 않으면 그 결함은 holdout에서 뺀다. F1·F4는 조정할 크기가 없어 첫 시도에 깨지지 않으면 뺀다.
- 텔레메트리만 보고 결함을 알아볼 수 있는지는 보정 기준에 넣지 않는다. 진단 난이도를 보고 크기를 고르면 holdout이 설계자 쪽으로 기운다.

**보정 기록**

1차(`evals/results/p0c_calibration_20261009/`, seed 7, 4096 env × 300회): NONE은 낙상 0, 선속도 RMSE 0.145, 회전 RMSE 0.298이다.

| 결함 | 생존 비 | 낙상 | 선속도 오차 비 | 회전 오차 비 | 판정 |
|---|---|---|---|---|---|
| F1 | 1.000 | 0 | 8.09 | 2.49 | 깨짐 |
| F2 | 1.000 | 0 | 7.93 | 1.66 | 깨짐 |
| F3 | 0.065 | 1.000 | 8.65 | 6.93 | 깨짐 |
| F4 | 0.559 | 0.484 | 8.23 | 3.44 | 깨짐 |
| F5 | 1.000 | 0 | 8.10 | 2.74 | 깨짐 |
| F6(10%) | 1.000 | 0 | 0.92 | 0.95 | 안 깨짐 |

F6은 규칙대로 5%로 키워 seed 7에서 다시 학습한다(2차). 10% 조건(2초 에피소드)으로 학습한 정책도 20초 고정 평가에서 기준만큼 걷는다.

## 3. holdout

- seed: 2026, 2027, 2028(이전에 쓰지 않은 값). seed마다 기준 실행 1개 + 결함 6개.
- 케이스 ID `h01`~`h06`은 비밀 seed로 섞어 결함에 대응시키고 모든 seed에 같은 대응을 쓴다(v1과 같은 방식). 실행 이름 `h0N_s<seed>`에는 결함 정보가 없다.
- 결함 실행의 고정 평가가 healthy면 그 실행은 진단 점수에서 빼고 따로 보고한다(실패하지 않은 실행의 원인 순위는 정의되지 않는다).
- B0 입력 대조(`evals/baselines.py --run-params`)로 holdout 실패 실행의 설정 diff가 실행 식별 값 말고는 비어 있음을 확인한다.

## 4. 비교 방법 (평가 전에 커밋)

| 방법 | 입력 | 비용 |
|---|---|---|
| `rule_prior`, `rule_features`, `rule_template` | P0-A2 동결 규칙 그대로(v1 dev로 정한 것, 수정하지 않음) | 없음 |
| `rule_b0` | 설정 diff. 구조상 빈 diff라 빈도순과 같아야 함 | 없음 |
| 단일 프롬프트 LLM, 전체 시계열 | 에이전트와 같은 텔레메트리 전부를 한 번에 | NVIDIA API |
| 전체 에이전트 | 기존 도구 | NVIDIA API |

**에이전트 프롬프트는 바꾸지 않는다.** 설계자가 숨은 결함을 아는 상태라 지금 고치면 그 정보가 들어갈 수 있다. 바꾼 것은 도구 경계뿐이다.

**정답 파일 경계.** 같은 실행 위치(WSL에서 본 이 checkout)에 비공개 정답표가 있다. 에이전트가 임의 파일을 열 수 있는 통로는 `run_analysis`(에이전트가 쓴 파이썬 코드 실행) 하나다. 다른 도구는 작업공간 밖 경로를 거부한다(`_safe`).

- `run_analysis` 하위 프로세스는 텔레메트리를 읽은 뒤 Linux Landlock으로 스스로를 묶는다. 파이썬 설치 경로 읽기와 `scratch` 쓰기만 남는다.
- 정답표·저장소·`/mnt/c`·`/proc`은 열 수 없고, 부모의 환경변수(API 키)도 넘겨받지 않는다.
- p0c 평가는 `TRIAGE_ANALYSIS_SANDBOX=required`에서만 돈다.
- 평가 직전 에이전트와 같은 실행 경로(`uv run --no-sync`, 같은 환경변수)의 하위 프로세스에서 분석 코드로 정답표와 `/proc/self/environ`을 열어 보는 canary를 돌린다. 둘 다 거부되고 `sandbox`가 `landlock`이어야 평가를 시작하고, 결과는 `canary_seed<N>.json`에 남긴다.
- `run_analysis` 결과의 첫 키 `sandbox`가 trace에 남는다. 에이전트 실행마다 trace의 분석 호출 수와 Landlock이 걸린 호출 수를 행에 기록하고, 하나라도 다르면 평가를 멈춘다.
- 테스트는 샌드박스를 끄면 canary가 실패를 드러내는 것까지 확인한다. 네트워크는 Landlock ABI 3에서 막지 못한다.

평가 명령(WSL, 키는 `~/.config/nvidia/env`):

```bash
cd /mnt/c/Users/<user>/rl-triage-agent && . ~/.config/nvidia/env
export UV_PROJECT_ENVIRONMENT=~/.venvs/rl-triage TRIAGE_ANALYSIS_SANDBOX=required
uv run --no-sync python evals/run_eval.py --bench p0c --task blind --seed 2026 --mode agent control_full --tag p0c_holdout_<date>
```

## 5. 설계자가 아는 것

결함 정의는 이 문서와 코드에 공개돼 있고, 설계자가 썼다. P0-A2 규칙은 이 결함을 보기 전에 동결됐으므로 규칙 쪽에는 유리한 정보가 없다. 에이전트 프롬프트를 이 단계에서 고치면 설계자가 결함을 아는 상태에서 고친 것이므로 그 사실을 기록한다.
