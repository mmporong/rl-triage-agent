# RL Triage Agent — Isaac Lab 학습 실패 원인 추적 에이전트

NVIDIA Isaac Lab에서 로봇 보행 정책 학습이 기준 실행과 다르게 무너졌을 때, 여러 설정 변경 중 **어느 변경이 원인인지** 텔레메트리로 찾아내고, 사람이 승인할 **다음 실험 하나**를 사전등록하는 에이전트다. 에이전트는 설정·안전 게이트·관절 한계·보상을 직접 고칠 수 없다. 그 경계는 NVIDIA OpenShell 정책과 `openshell-prover` 증명으로 강제한다.

> Status: 개발 중(Korea Agentic AI Hackathon 2026 예선). 아래 "측정 전"으로 표시한 항목은 아직 실행하지 않았다.

## 왜 필요한가

로봇 RL 엔지니어는 한 번에 여러 설정을 바꾸고 학습을 돌린다. 학습이 무너지면 보상·물리·액추에이터·탐색·최적화·계측 중 어디가 원인인지 모른 채 같은 계열의 개입을 반복하기 쉽다. 이 저장소 작성자의 실제 Go2 학습([isaac-walk-rl](https://github.com/mmporong/isaac-walk-rl))에서도 안전 게이트 실패에 명령 축소 계열 개입이 3회 연속 기각됐고, 같은 증상은 12일 전 계측에 이미 남아 있었다.

## 동작

```
목표("이 학습이 왜 무너졌나")
  → Nemotron 3 Super 계획 (NeMo Agent Toolkit tool_calling_agent)
  → 도구: list_changes / telemetry_overview / get_series / run_analysis(샌드박스 파이썬) / query_ledger
  → write_preregistration: 의심 변경, 전체 순위, 가설, 단일 실험 변수, 판정 기준
  → (사람 승인) 재평가 브리지가 의심 변경 하나만 되돌려 Isaac Lab에서 재학습 → 회복 여부 판정
```

| NVIDIA 구성요소 | 쓰임 |
|---|---|
| Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`, build.nvidia.com) | 계획·도구 호출·추론 |
| NeMo Agent Toolkit 1.9 (`nvidia-nat`) | 에이전트 워크플로(`configs/triage_workflow.yml`), 도구 6개 등록(`src/rl_triage/nat_functions.py`) |
| OpenShell | 샌드박스 정책(`policies/triage_agent.yaml`), 현장 경계(`policies/site_boundary.yaml`), Policy Advisor 권한 요청 |
| openshell-prover | 권한 확장 제안을 경계와 SMT로 비교, 반례 출력(`evals/prove_policies.py`) |
| NVIDIA Skills | `skills-lock.json`(nemo-rl-auto-research, nemoclaw-user-guide, nemotron-policy-generator), 자체 스킬 `skills/isaaclab-rl-triage` |
| Isaac Lab 2.1.1 / Isaac Sim 4.5 | 결함 주입 벤치마크와 재평가 재학습(`Isaac-Velocity-Flat-Unitree-Go2-v0`) |

## 결함 주입 벤치마크

- 결함 10종(보상·액추에이터·탐색·최적화·물리·종료)을 Go2 flat 기본 학습에 hydra override로 주입한다. 각 케이스는 해로운 변경 1개 + 무해한 변경 2개를 비밀 seed로 섞었다(`bench/catalog.py`, `bench/cases.json`).
- 1024 환경 × 100 iteration × seed 2개(42, 7), 기준 실행과 무해 변경 전부 적용 실행 포함 총 24회.
- 벤치마크 성립 확인: 회복 판정 기준(`eval_bridge.RECOVERY_BAND`)에서 무해 실행 2/2는 회복, 결함 실행 20/20은 미회복.
- 루프 검증(에이전트 평가 아님, `evals/results/bridge_smoke.json`): c01 seed 42에서 수동 사전등록 2건을 브리지로 승인·재학습했다. 원인 변경을 되돌리면 회복(에피소드 길이 비 0.999, 보상 비 1.024), 무해 변경을 되돌리면 미회복(에피소드 길이 비 0.05). 틀린 진단은 재학습이 걸러낸다.
- 정답표는 평가가 끝난 뒤 `bench/answer_key.json`으로 공개한다.

### 결과 — 측정 전

| 방식 | top-1 원인 적중 | 비고 |
|---|---|---|
| 무작위 | 33% (1/3) | 이론값 |
| 단일 프롬프트 Nemotron 3 Super (같은 입력) | 측정 전 | 대조군 |
| RL Triage Agent (NAT 도구 호출) | 측정 전 | 프롬프트·도구는 평가 전 커밋 `4acb1ed`에 동결 |

## 보안 경계

| 위협 | 통제 | 검증 |
|---|---|---|
| 게이트를 통과시키려 설정·관절 한계·보상 수정(명세 우회) | 설정 경로가 샌드박스 파일 정책에 없음, 재평가 브리지에 설정 API 없음(405) | `tests/test_security.py`, prover `bad_patch_config` 반례 `PUT /config/` |
| 체크포인트·로봇 모델 외부 반출 | 네트워크는 `integrate.api.nvidia.com`의 `POST /v1/chat/completions`만 | prover `bad_exfil_checkpoint` 반례 `huggingface.co:443` |
| GPU 재학습 남용 | 재평가는 사전등록 필수 + 사람 CLI 승인, HTTP 승인 경로 없음 | `tests/test_security.py` |
| API 키 노출 | OpenShell provider가 요청에만 키 주입 | 측정 전(샌드박스 실행 필요) |
| 커널 수준 차단(기준 설정 쓰기, 외부 전송, 클라우드 메타데이터) | Landlock·seccomp·네트워크 정책 | `tests/sandbox/` — 측정 전(샌드박스 전용) |

prover 결과(`evals/results/policy_proofs.json`): 에이전트 정책과 재평가 요청은 `within_boundary` → 사람 검토, 설정 변경·외부 반출 요청은 `exceeds_boundary` → 자동 거절, 기준 설정 쓰기 요청은 prover가 증명하지 못해(`unsupported`) → fail-closed 자동 거절.

## 실행

```bash
uv sync
uv run pytest -q tests                              # 단위·보안 테스트
uv run python evals/prove_policies.py               # OpenShell 정책 증명(WSL/Linux에 openshell-prover 필요)
uv run python bench/build_workspace.py 42           # 벤치마크 텔레메트리 → 에이전트 작업공간
export NVIDIA_API_KEY=nvapi-...                     # build.nvidia.com
uv run python evals/run_eval.py --seed 42 --mode both
uv run python -m rl_triage.eval_bridge serve        # 재평가 브리지(Isaac Lab 호스트)
uv run python -m rl_triage.eval_bridge list         # 사람: 대기 요청 확인
uv run python -m rl_triage.eval_bridge approve <id> # 사람: 승인 → 재학습 → 회복 판정
```

## 한계

- 벤치마크 순환성: 결함 카탈로그와 에이전트를 같은 사람이 만들었다. 완화: 무해 변경과 섞고 비밀 seed로 섞음, 에이전트 프롬프트·도구를 평가 전에 커밋으로 동결, 같은 모델·입력의 단일 프롬프트 대조군.
- 스킬 `isaaclab-rl-triage`의 메커니즘-흔적 표는 벤치마크 점검 뒤 작성했다. 그래서 정량 결과는 스킬을 쓰지 않는 동결 에이전트로만 계산한다.
- 탐색 노이즈를 거의 0으로 만든 결함(H03)은 보상이 오히려 높다. 이 케이스는 "실패"가 아니라 "기준 대비 이상"이다.
- 100 iteration 짧은 학습이다. 장기 학습에서만 드러나는 결함은 다루지 않는다.
- 네트워크 규칙은 허용된 실행 파일의 자식 프로세스에도 적용된다. 그래서 `run_analysis`가 실행한 코드도 NVIDIA 추론 API 경로에는 접근할 수 있다.
