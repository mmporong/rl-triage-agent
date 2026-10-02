# 인계 문서 (2026-10-02, Claude Code → Codex)

## 1. 지금 상태

- 예선 제출 완료 2026-09-27(팀 Mollet, 2인). **본선 진출 발표 2026-10-02**, 본선 10/7(수) 09:00~19:00 오프라인 원데이(10팀, 미션 당일 공개, 팀당 Brev L40S 크레딧 최대 $1,000, 피칭 최대 5분), 파이널 11/10 NVIDIA AI Day Seoul.
- 공개 저장소 https://github.com/mmporong/rl-triage-agent, 제출 시점 커밋 `8d0e7ad`. 로컬 `%USERPROFILE%\rl-triage-agent`.
- 조사·결정 기록과 제출 요약: `%USERPROFILE%\hackathon-specialist\reports\nvidia-agentic-hackathon-2026\summary.md`(비공개 저장소 mmporong/hackathon-specialist). 결과가 나오면 그 파일의 "결과" 절에 이유 한 줄을 추가해 커밋·push한다.

## 2. 무엇을 만들었나

Isaac Lab RSL-RL 학습이 기준 실행과 다르게 무너졌을 때, 텔레메트리로 원인을 좁히고 설정을 고치는 대신 다음 실험 하나를 사전등록하는 에이전트.

| 구성 | 위치 |
|---|---|
| 도구 구현(프레임워크 독립) | `src/rl_triage/triage_tools.py` |
| NeMo Agent Toolkit 함수 7개 | `src/rl_triage/nat_functions.py` |
| 워크플로(과제 A: 바꾼 설정 중 원인) | `configs/triage_workflow.yml` |
| 워크플로(과제 B: 원인 모름 메커니즘) | `configs/triage_blind.yml` |
| 셸 에이전트용 CLI | `src/rl_triage/cli.py` |
| 사람 승인형 Isaac 재학습 브리지 | `src/rl_triage/eval_bridge.py` |
| OpenShell 정책·provider·제안 | `policies/` (`site_boundary.yaml`, `triage_agent.yaml`, `providers/nvidia.yaml`, `proposals/`) |
| prover 증명 | `evals/prove_policies.py` → `evals/results/policy_proofs.json` |
| 샌드박스 이미지 | `docker/Dockerfile.sandbox`, `docker/Dockerfile.workspace`, `scripts/sandbox_up.sh` |
| 벤치마크 | `bench/catalog.py`(결함 10종), `bench/cases.json`, `bench/runs/*.telemetry.json`, `bench/answer_key.json`(공개) |
| 평가 | `evals/run_eval.py` → `evals/results/<tag>/seed<N>.jsonl`, 실행 기록 `evals/results/traces/` |
| 자체 스킬 | `skills/isaaclab-rl-triage/` (NVIDIA 스킬 형식, 서명 없음) |
| 데모 | `docs/demo/build_slides.py`, `docs/demo/make_video.sh` → `docs/demo/rl_triage_demo.mp4`, Isaac 녹화 `bench/record_play.ps1` → `docs/media/` |
| 제출물 | `submission/form_answers.txt`, `submission/build_pdf.py`, `submission/NVIDIA 해커톤_Mollet_RL Triage Agent.pdf` |

## 3. 결과 (제출 시점)

| 과제 | 세트 | 에이전트 | 같은 모델 단일 프롬프트 |
|---|---|---:|---:|
| A. 바꾼 설정 중 원인 | 개발 seed 42·7 / 보류 seed 123 | 20/20 / 10/10 | 20/20 / 10/10 |
| B. 원인 모름 메커니즘 top-1 | 개발 / 보류 | 10/20 / 6/10 | 8/20 / 2/10 |

- 보류 세트 차이 n=10, p≈0.125(유의하지 않음). physics·optimizer 결함은 두 방식 모두 top-1 0건.
- 재학습 검증: 원인 되돌리기 회복(에피소드 길이 비 0.999), 무해 변경 되돌리기 미회복(0.05).
- 샌드박스 커널 차단 6/6, prover 5/5 기대 일치.
- 동결 커밋: `4acb1ed`(과제 A 프롬프트), `517f613`(과제 B), `c7e8cd9`(v1.1 출력 한도 16384).

## 4. 실행 환경 (Windows 11 + WSL2)

- **API 키**: WSL Ubuntu(24.04)의 `~/.config/nvidia/env`에만 있다. Windows 환경변수에는 없다. 그래서 에이전트 평가는 WSL에서 돌린다.
- **WSL 파이썬 환경**: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/rl-triage`. 예시:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"; . ~/.config/nvidia/env
  cd /mnt/c/Users/LIMMM/rl-triage-agent
  export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/rl-triage UV_LINK_MODE=copy PYTHONUTF8=1
  uv run --no-sync python evals/run_eval.py --task blind --seed 123 --mode both --tag <새태그>
  ```
- **Windows 파이썬 환경**: 저장소 `.venv`(테스트·작업공간 생성·prover 호출용). `PYTHONUTF8=1 uv run --no-sync pytest -q tests` → 13 passed, 6 skipped(샌드박스 전용).
- **OpenShell**: WSL Ubuntu에 0.1.1(snap, 게이트웨이 `127.0.0.1:17670`), `openshell-prover` 0.1.1은 `/usr/bin`. provider 프로필은 `policies/providers/nvidia.yaml`을 import한 상태.
- **샌드박스 `rl-triage`가 2026-10-02 기준 Error 상태다.** 다시 만들려면 WSL에서 `bash /mnt/c/Users/LIMMM/rl-triage-agent/scripts/sandbox_up.sh 42`. 샌드박스 안 실행은 `openshell sandbox exec -n rl-triage --env HOME=/tmp -- bash -c 'cd /sandbox/app && nat run --config_file configs/triage_workflow.yml --input "..."'`, 커널 테스트는 `OPENSHELL_SANDBOX=1 python -m pytest -q -p no:cacheprovider tests/sandbox`.
- **Isaac Lab**: Windows 네이티브, Isaac Sim 4.5 `E:\IsaacSim\isaac-sim-4.5.0`, Isaac Lab 2.1.1 `%USERPROFILE%\IsaacLab`, RTX 3060 12GB. 학습 1회(1024 env × 100 iter) 약 2.5분.

## 5. 반복해서 걸리는 함정

1. `isaaclab.bat`은 hydra override의 `=`와 `,`를 인자 구분자로 쪼갠다 → `_isaac_sim\python.bat`에 인자마다 큰따옴표를 씌워 `Start-Process`로 넘긴다(`bench/run_case.ps1`).
2. Isaac Lab 2.1.1 `train.py`는 `--experiment_name`을 무시한다. 결과는 `logs/rsl_rl/unitree_go2_flat/*_<run_name>`에서 찾는다.
3. 카메라 녹화 시 기본 Vulkan 렌더러가 access violation → `--kit_args=--/app/vulkan=false --/app/window/hideUi=true`(D3D12). `play.py --checkpoint`는 전체 경로를 받는다.
4. NAT `_type: nim`은 Nemotron 도구 호출 응답을 빈 응답으로 잃었다(20회 중 13회) → `_type: openai` + `base_url`로 같은 모델을 부른다.
5. Nemotron 3 Super는 추론 모델이라 `max_tokens` 4096에서 잘린다 → 16384 + `truncation_retry`.
6. NVIDIA 무료 엔드포인트가 자주 "Service temporarily overloaded" → `run_eval.py`가 인프라 오류만 30s·60s·120s 대기 후 재시도.
7. 정책의 실행 파일 경로는 심링크가 아닌 실제 경로(`/usr/bin/python3.12`). 네트워크 규칙은 자식 프로세스에도 적용된다.
8. OpenShell 업로드도 샌드박스 정책을 따른다 → 작업공간은 업로드가 아니라 이미지 레이어로 넣는다(`Dockerfile.workspace`). 이미지에 `sandbox` 사용자가 있어야 한다.
9. prover는 경로 포함관계를 증명하지 못하면 `unsupported`를 낸다 → 게이트는 fail-closed로 거절.
10. NemoClaw 블루프린트는 OpenShell 0.0.116 고정, WSL에서는 네이티브 Docker Engine을 preflight가 막는다(`host.platform.wsl_native_docker_unqualified`, Docker Desktop 필요). Ubuntu-22.04 배포판에 시도 흔적(Docker Engine, systemd, NemoClaw CLI)이 남아 있다.
11. 보상 비율 기반 회복 판정은 보상 가중치를 바꾸는 변경을 오판한다(벤치 v2 폐기 이유). 가중치 불변 지표(추종 보상÷가중치, base_contact, 에피소드 길이)를 쓴다.

## 6. 다음 할 일 후보

1. 본선 결과 확인(10/2) → hackathon-specialist `summary.md` 결과 절 갱신.
2. 진출 시 본선 준비:
   - 약점 보강: physics·optimizer 결함 구분용 도구(관절 토크·접촉력·가치 손실 곡선 비교, 행동 지표 분해). 보강 후 성능은 **새 seed 보류 세트**로만 보고.
   - 샌드박스 재생성과 5분 피칭 리허설(데모 영상 50초 + 라이브 1케이스).
   - 당일 미션이 다른 도메인이어도 재사용할 부분: OpenShell 정책·prover 게이트, 사전등록·승인 루프, 평가 하네스.
   - 선택: Docker Desktop 환경에서 NemoClaw Hermes 하네스 연동(README `Harness` 칸을 Hermes로 채울 수 있음).
3. 탈락 시: README에 결과를 적지 않아도 된다. summary.md에 이유만 기록.
