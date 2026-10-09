# 구현 순서 보정 (2026-10-08)

[CLAUDE-HANDOFF.md](CLAUDE-HANDOFF.md)의 P0~RC 수용 기준은 그대로 둔다. 이 문서는 그 위에서 **순서와 범위**를 보정한다. 근거는 2026-10-08 코드 대조와 공개 출처 조사다.

- 사용자 지시(2026-10-08): Windows에서 Claude 세션을 열어 이어서 진행하고, Claude가 총괄을 맡는다.
- 기준 커밋: 이 문서를 추가한 커밋. 코드 감사 기준은 계속 `58bafa3`이다.
- 이 문서 작성 중 제품 코드·모델 호출·학습·GPU 실행은 하지 않았다. 아래 "확인함"은 저장 파일을 읽어 계산한 결과이고, "미검증"은 실행하지 않은 것이다.

## 1. 시작 점검 (Windows 세션 W0)

Windows 로컬 사본(`%USERPROFILE%\rl-triage-agent`, [HANDOFF.md](HANDOFF.md) 6행)은 이 저장소 원격보다 오래됐을 수 있고 비공개 자료(`bench/private/`)를 가진 유일한 위치일 수 있다. 구현 전에 상태부터 맞춘다.

```powershell
cd $HOME\rl-triage-agent
git status --short --branch
git log -3 --oneline
git stash list
git fetch origin
git log --oneline HEAD..origin/main
git log --oneline origin/main..HEAD
```

1. 작업 트리가 깨끗하고 뒤처지기만 했으면 `git merge --ff-only origin/main`.
2. 로컬 커밋·미커밋 변경·stash가 있으면 되돌리거나 덮지 않는다. 목록을 사용자에게 보고하고 처리 방법을 정한 뒤 진행한다.
3. `bench/private/`는 이름·크기·SHA256 목록만 만든다. 내용은 공개 저장소에 올리지 않는다.
4. `bench/private/answer_key.json`이 있으면 `bench/answer_key.json`과 SHA256을 비교한다. 다르면 P0-A를 멈추고 차이를 보고한다(과거 점수가 다른 정답표로 매겨졌을 가능성).
5. 기준 실행 params 존재 확인: `bench/private/baseline_s{7,42,123}.meta.json`이 가리키는 run 폴더의 `params/env.yaml`·`agent.yaml`.
6. 실행 환경 버전 기록: Isaac Sim 경로·버전, `%USERPROFILE%\IsaacLab`의 `git describe --tags`, `rsl_rl`·`torch` 버전, `nvidia-smi`의 GPU·드라이버.

공개 가능한 결과(버전, 해시 일치 여부)는 P0-A 커밋의 실행 manifest에 넣는다. 사용자 경로·계정이 들어간 원본은 `bench/private/`에만 둔다.

## 2. 계획에 없던 개선점

### A. 정답 유출 경계 — P0-A에 포함

- [사실] `run_analysis`(`src/rl_triage/triage_tools.py:136`)는 에이전트가 쓴 Python을 `subprocess.run([sys.executable, script], cwd=scratch)`로 실행한다(156행). `_safe()`(34행)는 도구가 만드는 경로에만 적용되고, 에이전트 코드의 파일 열기는 막지 않는다.
- [사실] 커널 차단(Landlock)은 OpenShell 안에서만 동작한다. 과거 채점 평가는 호스트 NAT로 돌았다([CLAUDE-HANDOFF.md](CLAUDE-HANDOFF.md) 3절). 같은 checkout에 `bench/answer_key.json`·`bench/cases.json`이 있다.
- [사실] `evals/results/traces/` 50개에서 `answer_key`·`cases.json`·`../..` 문자열을 검색하면 0건이다. 과거 유출 증거는 없다.
- 대응: P0-A는 (1) blind workspace에 case 정보·override·정답이 없음을 검사하고, (2) trace 검색을 자동 검사로 만든다. 새 채점 평가(P0-C) 전에는 정답 파일이 없는 실행 위치(OpenShell 또는 정답을 뺀 별도 checkout)에서 에이전트를 돌리고, 정답 파일 읽기 시도가 실패하는 canary 검사를 수용 기준에 넣는다.

### B. 설정 diff로 풀리는 결함 — P0-C 설계 변경

- [사실] Isaac Lab v2.1.1 `scripts/reinforcement_learning/rsl_rl/train.py` 177~180행은 모든 실행의 `params/env.yaml`·`agent.yaml`을 저장한다([원문](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/scripts/reinforcement_learning/rsl_rl/train.py#L177-L180)).
- [사실] 보류 seed 123의 physics·optimizer 결함 4종(c02 몸통 질량, c10 `sim.dt`, c06 `value_loss_coef`, c08 `gamma`)은 모두 hydra override 설정값이다(`bench/answer_key.json`, `bench/catalog.py`).
- [추론] 실제 사용 조건에서는 이 4종이 YAML diff 한 번에 드러난다. Task B는 실패 실행 params를 감춘 조건이라(`bench/build_workspace.py`) "현실에서는 diff로 끝난다"는 반론에 답하지 못한다. 변경 목록을 준 Task A는 두 방식 모두 포화였다(`evals/results/heldout_changes/seed123.jsonl` top-1 10/10 대 10/10).
- 대응: P0-C 기준선에 **B0 설정 diff 사전 점검**(LLM 없음)을 넣는다. 새 holdout은 설정 diff로 설명되지 않는 결함으로 구성한다. 후보: 프레임워크 의미 변경([Isaac Lab 릴리스 노트](https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html) v2.0.2의 actuator 한계 의미 변경), CPU/GPU 물리 백엔드 차이, 이벤트·랜덤화 동작, `env.yaml` 밖의 자산 변경. (2026-10-09 정정: 처음 인용한 [#915](https://github.com/isaac-sim/IsaacLab/issues/915)는 IsaacLab·IsaacGym 학습 성능 비교 질문이고 이벤트 타이밍 이슈가 아니다. 외부 원인 후보는 [NEXT-STEPS-20261009.md](NEXT-STEPS-20261009.md) 1절.)

### C. 보상 항 단위 오독 — P1-A와 P0-A2

- [사실] `Episode_Reward/<항>`은 에피소드 합 평균을 `max_episode_length_s`(Go2 flat 20초)로 나눈 값이다([reward_manager.py v2.1.1](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab/isaaclab/managers/reward_manager.py)). 빨리 넘어지면 모든 항이 같이 작아진다. `Episode_Termination/*`은 비율이 아니라 리셋된 env 개수다([termination_manager.py v2.1.1](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/source/isaaclab/isaaclab/managers/termination_manager.py)).
- [사실] 에이전트는 c02에서 episode length 비 0.214와 `dof_torques_l2` 비 0.269를 나란히 적고 actuator로 판단했고(`evals/results/traces/blind_seed123_c02_1790482327.jsonl`), c10에서는 torque 비 0.043을 "very weak torques"로 읽었다(`…_c10_1790483734.jsonl`). (2026-10-08 P0-A2 확인: c02의 0.269는 에이전트가 쓴 숫자이고 텔레메트리 값은 0.658이다.)
- [사실] c08(`gamma=0.5`) 근거 문장에 "value loss ratio 0.016"이 있었는데도 actuator를 1위로 냈다(`…_c08_1790483263.jsonl`).
- 대응: 생존 시간으로 나눈 보상 항, 리셋 수로 나눈 종료 개수, step_dt로 환산한 에피소드 길이를 도구가 계산해 제공한다. 같은 특징을 쓰는 고정 규칙 기준선과 비교한다.

### D. 실행 환경 고정 — P0-B2 전에 결정

- [사실] 기존 벤치는 Windows 네이티브 Isaac Sim 4.5·Isaac Lab 2.1.1·RTX 3060 12GB에서 PowerShell 실행기(`bench/run_case.ps1`)로 만들었다([HANDOFF.md](HANDOFF.md) 55행).
- [사실] `bench/runs/*.telemetry.json` 79개의 키는 `run_dir_name`·`summary`·`series`뿐이다. 버전·장치·params 해시가 없다.
- 대응: 새 실행부터 manifest(Isaac Sim/Lab 버전, GPU·드라이버, params SHA256, seed, 실행 명령)를 텔레메트리 옆에 저장한다. 비교하는 실행끼리 호스트·버전을 섞지 않는다. 호스트 후보는 4절.

### E. 행동 기반 회복 판정의 일부는 기존 데이터로 가능 — P0-B 앞당김

- [사실] `RECOVERY_BAND`(`src/rl_triage/eval_bridge.py:29`)에 `Train/mean_reward`(0.5~1.8배)가 들어 있다. 보상 가중치를 바꾸는 개입은 이 비율을 움직이므로 회복 판정이 개입 자체에 오염된다.
- [사실] 텔레메트리에 보상 가중치와 무관한 지표가 이미 있다: `Metrics/base_velocity/error_vel_xy`, `Metrics/base_velocity/error_vel_yaw`, `Episode_Termination/base_contact`, `Train/mean_episode_length`.
- 대응: Isaac 재실행 없이 기존 79개 실행으로 행동 기반 판정을 계산하고, 현행 판정과 라벨이 달라지는 실행을 기록한다(P0-B1). 고정 평가 조건 실측은 P0-B2.

### F. 외부에서 원인이 정해진 결함 — 선택

- "제작자가 만든 인공 결함" 반론에 대비해, 원인이 공개 이슈·릴리스 노트로 확정된 사례(B절 후보)를 holdout에 몇 건 넣는다. Isaac 재현이 필요하므로 일정이 허락할 때만 한다.

### G. 재채점 집계 규칙

- [확인함] 집계 규칙 "(seed, case, mode)별 마지막 비인프라 행"으로 저장 결과를 다시 세면 아래 값이 나온다. 공개 `bench/answer_key.json`으로 `ranking`에서 다시 계산한 top-1·top-2는 blind 비인프라 65행 모두 저장값과 일치했다.

| 결과 폴더 | 전체 행 | 인프라 행 | agent top-1 / top-2 | control top-1 / top-2 |
|---|---|---|---|---|
| `blind_v1` (dev seed 7·42) | 47 | 7 | 10/20 / 12/20 | 8/20 / 10/20 |
| `heldout_blind` (seed 123) | 22 | 2 | 6/10 / 6/10 | 2/10 / 6/10 |

- P0-A는 이 계산을 테스트로 고정하고 규칙을 문서에 명시한다. 이 수치는 과거 결과의 재현이며 새 성능 근거가 아니다.

## 3. 구현 순서

| 순서 | 작업 | 수용 기준 | 실행 위치 | 외부 비용 |
|---|---|---|---|---|
| **W0** | 1절 시작 점검 | Windows 사본 상태·정답표 해시·기준 params·버전 기록 | Windows | 없음 |
| **P0-A** | ① 저장 결과 재채점 진입점(G절 규칙, 공개 정답표, 입력 SHA·코드 SHA 기록) ② `evals/run_eval.py:180`의 `bench/private/answer_key.json` 참조 정리 ③ 기준 실행 params를 공개 reference로 고정해 private 없이 workspace 생성(사용자 경로·계정 문자열 검사 후 커밋) ④ A절 유출 경계 검사 ⑤ 누락 파일·잘못된 JSON·모르는 case·빈 ranking 실패 검사 | 임시 폴더의 깨끗한 clone에서 키·네트워크·GPU 없이 G절 표를 재현. 재채점 경로가 `openai`·NAT를 import하지 않음. blind workspace에 정답 정보 없음 | Windows 또는 Linux (③은 params가 있는 Windows) | 없음 |
| **P0-A2** | 결정적 기준선: B0 설정 diff 규칙, C절 정규화 특징 규칙 | 규칙은 dev seed 7·42만 보고 고정·커밋한 뒤 seed 123에 적용. agent 6/10·control 2/10과 같은 표로 기록. 규칙이 동등 이상이면 [AI-DAY-2026.md](AI-DAY-2026.md) 철회 조건에 따라 주장 범위를 좁힌다 | 어디서나 | 없음 |
| **P0-B1** | E절 행동 기반 회복 판정을 기존 79개 실행에 적용 | 판정 계약(지표·구간·reference 변동 처리)을 먼저 커밋하고, 현행 `RECOVERY_BAND`와 라벨이 바뀐 실행 목록 기록 | 어디서나 | 없음 |
| **결정 게이트** | 4절 호스트·버전 결정, manifest 형식 확정 | 사용자 결정 | — | — |
| P0-B2 | 고정 평가 조건에서 회복 판정 실측 | 기준 seed 반복 변동 포함 | 결정된 호스트 | GPU |
| P0-C | 정보·예산을 맞춘 기준선(B0, 정규화 특징 규칙, 전체 시계열 단일 프롬프트, 전체 agent)과 B절 기준 새 holdout | 프롬프트·도구·판정 평가 전 커밋, 새 `--tag`, 정답 없는 실행 위치 | 결정된 호스트 + NVIDIA API | GPU·API, 사용자 승인 |
| P1-A~C | CLAUDE-HANDOFF 5절 순서(가설·실험 선택에 C절 특징 반영, 승인 소비·receipt, 최소 변화 쌍) | CLAUDE-HANDOFF 기준 | 결정된 호스트 | GPU·API |
| RC | 독립 재현·고정 데모·Q&A | CLAUDE-HANDOFF 기준 | — | — |

**순서를 바꾼 이유.** 비용이 들지 않는 P0-A·A2·B1로 "규칙만으로 얼마나 풀리는가"와 "회복 라벨이 얼마나 바뀌는가"를 먼저 잰다. 결과가 나쁘면 GPU·API를 쓰기 전에 주장 범위를 좁힌다. AI-DAY-2026.md의 철회 조건을 가장 싸게 검증하는 순서다.

## 4. 실행 호스트 후보

| 호스트 | 확인된 사실 | 장점 | 확인할 것 |
|---|---|---|---|
| Windows RTX 3060 12GB | Isaac Sim 4.5·Isaac Lab 2.1.1, 1024 env × 100 iter 약 2.5분(HANDOFF.md 55행), PowerShell 실행기 있음 | 기존 79개 실행과 같은 호스트·버전. 비용 없음 | 현재도 같은 버전인지(W0 6번) |
| Brev L40S 48GB (Linux) | Isaac Lab v2.1.1 `docker/.env.base`가 `nvcr.io/nvidia/isaac-sim:4.5.0` 기반([원문](https://github.com/isaac-sim/IsaacLab/blob/v2.1.1/docker/.env.base)). NGC 공개 태그는 `isaac-lab` 2.1.0·2.2.0·2.3.x 등으로 2.1.1 전용 이미지는 없고 `isaac-sim` 4.5.0은 있다(2026-10-08 레지스트리 조회). Isaac Sim 문서에 Brev L40S 배포 절차가 있다([Isaac Sim 6.1 Brev](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_advanced_cloud_setup_brev.html)) | 메모리·속도 여유, 많은 seed 병렬 | 계정 로그인·크레딧 잔액·시간당 요금, v2.1.1 태그로 이미지 빌드, Linux 실행기(`run_case.ps1` 이식), GPU가 달라 reference를 같은 호스트에서 다시 만들어야 함 |
| Linux 노트북 RTX 5050 8GB | Isaac Sim 5.1·Isaac Lab 2.3 | — | 기존 벤치와 버전이 달라 비교 실행에 쓰지 않음 |

[권고] P0-B2는 Windows에서 한다(기존 실행과 같은 조건). P0-C holdout의 실행 수가 일정 안에 Windows로 감당되지 않으면 Brev로 옮기고, 그때는 reference와 모든 비교 실행을 Brev에서 새로 만든다. Brev 크레딧은 GPU 인스턴스 비용이며 NVIDIA API(Nemotron 호출) 비용과 별개다.

## 5. 운영

- 산출물마다 입력 SHA·코드 SHA·실행 명령·실행 위치를 남긴다. `evals/results/**`는 덮어쓰지 않고 새 `--tag`를 쓴다.
- 공개 저장소다. `bench/private/`·키·사용자 경로·비공개 현장 원문을 커밋하지 않는다. 커밋 전 `git diff --cached`로 경로·계정 문자열을 확인한다.
- 각 단계는 작성 → 관련 검증 → 수정 → 재검증 → 별도 검수 → 경로별 커밋 순서로 닫는다. push는 그 세션에서 사용자 요청을 받은 뒤 한다.
- GPU 학습·API 호출·Brev 인스턴스 생성은 사용자 승인 범위를 확인한 뒤 시작한다.

## 6. 진행 상태

| 단계 | 상태 | 근거·남은 제약 |
|---|---|---|
| W0 | 완료(2026-10-08, Windows) | 작업 트리 깨끗, `be8cae9`로 fast-forward. 비공개·공개 정답표 동일(원본 바이트와 LF 정규화 모두). 기준 params 3개 존재: seed 42는 비공개 meta가 없고 텔레메트리 `run_dir_name`(s02_baseline_s42)의 실행 폴더로 확인했다. 버전·해시는 `bench/reference/manifest.json`의 `w0_check`. 버전은 10/8 호스트 값이며 학습 당시 기록이 아니고, 학습 때 import된 torch 빌드는 미확인. `bench/private/` 이름·크기·SHA256 목록은 비공개 폴더 안에만 저장 |
| P0-A | 완료(`0f112ca`) | 진입점 `evals/replay.py`, 공용 채점 `src/rl_triage/scoring.py`, 유출 검사 `src/rl_triage/leakcheck.py`, 공개 params `bench/reference/params/`, 테스트 `tests/test_offline_replay.py`. clean clone 재현 기록 `evals/results/replay_p0a_20261008/replay.json`(G절 표 일치, 저장값 불일치 0, trace 50개 표식 0) |
| P0-A2 | 완료(동결 `eb318b4` 뒤 seed 123 적용) | 계약·결과 [P0-A2-BASELINES.md](P0-A2-BASELINES.md). seed 123 top-1: 고정 규칙 10/10, 최근접 dev 사례 10/10, 설정 diff 10/10, 빈도순 3/10 (에이전트 6/10, 대조군 2/10). 에이전트 정확도 우위 주장 철회, seed holdout은 템플릿 재인식 시험으로 판정. 기록 `evals/results/p0a2_{dev,holdout}_20261008/`, `replay_p0a2_20261008/` |
| P0-B1 | 완료(계약 `ad894bc` 뒤 적용) | 계약·결과 [P0-B1-RECOVERY.md](P0-B1-RECOVERY.md). 실행 78개 중 19개 라벨 변경. 결함 실행 30개: unhealthy 21, healthy 6(c03 3개 등), undetermined 3(c01). 정상 기준 실행도 속도 명령을 거의 따르지 못해 100회 학습 텔레메트리의 행동 판정은 낙상 여부 위주다. 기록 `evals/results/p0b1_relabel_20261008/` |
| P0-B2 | 완료(계약 `5264dac` 뒤 평가, Windows 로컬 GPU) | 계약·결과 [P0-B2-FIXED-EVAL.md](P0-B2-FIXED-EVAL.md). 고정 26개 명령 격자에서 100회 기준 정책의 추종 오차는 제자리와 같다(1.175 대 1.177). 결함 실행 30개 중 unhealthy 15. 같은 seed 재학습은 체크포인트 동일. 4096 env × 300회 기준은 걷는다(0.154). P0-C 학습 예산 결정 근거. 기록 `evals/results/p0b2_{fixed_eval,pilot}_20261008/` |
| P0-C | 완료(2026-10-09) | 계약·결과 [P0-C-HOLDOUT.md](P0-C-HOLDOUT.md) 6절. 숨은 결함 6종 × seed 3, 4096 env × 300회. 결함 실행 18개 모두 고정 평가 unhealthy, params diff 18/18 비어 있음. top-1: 빈도순·설정 diff 3/18, 동결 규칙 5/18, 최근접 v1 사례 6/18, 단일 프롬프트(전체 시계열) 2/18, 에이전트 8/18. 표본이 작아 우위 주장은 하지 않음. 정답표 공개 `bench/answer_key_p0c.json`, 재채점 `evals/results/replay_p0c_20261009/` |
| P1-A | holdout 비교 완료 | [P1-A-LOOP.md](P1-A-LOOP.md) 7절. probe 6종(dev에서 네 차례 정의 수정)으로 전수 17/18, 고정 순서 17/18(평균 3.4개, 틀린 확정 1), 에이전트 순위 가르기 14/18(평균 2.1개, 틀린 확정 0). probe 작성자가 결함 작성자라 고리 동작 확인으로만 쓴다 |

사용자 결정(2026-10-08): 주 task는 Go2 flat을 유지한다(이족 전환 안 함). 실행은 Windows 로컬(RTX 3060)에서 하며 로컬 GPU 실행은 허용됐다. 이것으로 3절 결정 게이트의 호스트는 Windows로 정해졌다. Brev는 필요할 때 따로 요청한다. NVIDIA API 호출은 이 허용 범위에 넣지 않았으므로 P0-C 평가 전에 승인을 받는다.

P0-A에서 남긴 제약:

- `evals/results/v1_nim_client/seed42.jsonl`·`evals/results/smoke_c01_seed42.jsonl`은 `infra_error` 필드가 없어 G절 규칙으로 재채점하지 않고 입력 오류로 거부한다. README 과제 A dev 대조군 20/20 중 seed 42 절반이 이 파일에 의존한다.
- 유출 검사는 작업공간 입력과 저장 trace만 본다. 같은 checkout의 정답 파일 읽기를 막는 canary 검사는 P0-C 수용 기준으로 남아 있다(2절 A).
- 이 Windows 사본은 추적 텍스트 파일 132개가 CRLF(인덱스는 LF)다. replay는 CRLF→LF 정규화 SHA를 기록해 clean checkout과 같은 값을 남긴다.

## 7. 다음 단계 (2026-10-09 조사)

조사 보고서 [NEXT-STEPS-20261009.md](NEXT-STEPS-20261009.md)의 우선 작업이다. 수용 검사는 보고서 2.2절.

| 순위 | 작업 | 왜 지금 | 비용 |
|---|---|---|---|
| T1 | 외부 원인 결함 안전성 시험(2절 F) — 사전등록(2026-10-09, [T1-EXTERNAL-FAULTS.md](T1-EXTERNAL-FAULTS.md)), smoke·학습은 GPU 대기 | "probe와 결함을 같은 사람이 만들었다"는 반론에 답하는 유일한 실험. 공개 이슈로 원인이 확정된 결함(후보 X1 COM 무작위화 누적 #7311, X2 접촉력 고착 #7613, X3 push 덮어쓰기)으로, 동결 probe가 틀린 확정 없이 식별 불가로 멈추는지를 사전등록하고 잰다 | GPU 3~4시간 |
| T2 | P1-C 보류·최소 변화 쌍 | NONE·체크포인트 SHA 불일치·승인 뒤 사전등록 변경·unknown·범주 밖 결함에서 고리가 멈추는지, always-act·always-abstain과 비교 | GPU 거의 0 |
| T3 | P1-B 복구 — 완료(2026-10-09) | effect key 중복 거부, 고아 요청 탐지, 산출물로 재실행 없이 receipt·없으면 unknown과 새 승인, 예산 초과 unknown(`probe_loop.Ledger`, `loop.py recover`). 관문 5곳 무력화 시 테스트 실패 확인 | GPU 0 |
| T4 | RC 독립 재현·고정 데모·Q&A | 깨끗한 clone 한 명령 재계산, 5·10·15분 데모(진단 → 사전등록 → 거절 → 승인 → probe → 확정 → 회복 판정), 녹화본, 철회한 주장·설계자 결합·표본 크기 Q&A | GPU 수 분 |
| T5 | LLM 역할 측정(조건부) | 에이전트 순위 가르기와 규칙 순위 가르기를 같은 결과표에서 비교(P1-A 7절에 1차 결과) | API |

행사 정보(2026-10-09 공개 자료): 11/9 컨퍼런스, 11/10 DLI 워크숍·인증시험. 해커톤 Top 5 피칭의 일시·길이·형식·심사 기준은 공개 자료에서 찾지 못했다. 주최 측에 일시·길이·실시간 데모 허용·심사 기준·발표작 변경 승인을 확인해야 한다.
