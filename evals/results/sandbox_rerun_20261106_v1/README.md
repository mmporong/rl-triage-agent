# 결선 전 OpenShell 샌드박스 재실행 (2026-11-06, Windows 11 + WSL2)

예선 기록(`evals/results/sandbox_kernel_tests.txt`, 2026-09-27, 커널 시험 6/6)을 결선 시점에 다시 실행한 기록이다. 기존 기록은 수정하지 않았다.

로그 머리줄의 시각은 KST다. NAT·supervisor·gateway가 직접 찍은 줄은 UTC다(9시간 차이).

로그 읽는 법:
- `outside_sandbox_api_check.log`, `nemotron35_c03_*.log`의 `$` 줄은 실행한 명령을 사람이 읽도록 옮긴 표시다. 따옴표가 빠지거나 겹쳐 있어 그대로 붙여 넣으면 동작하지 않는다.
  - API 확인은 JSON 본문을 작은따옴표로 감싸 `-d '{"model":"<ID>","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8}'`로 보냈다.
  - 3.5 입구 준비의 task context는 `{"schema": "nemotron35_task_context_v1", "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "observed_problem": "Training run c03 failed its gate."}`이다.
- `outside_sandbox_api_check.log`의 출력은 걸러 낸 값이다. 모델 목록은 개수와 대상 ID 존재 여부만, 호출은 HTTP 코드와 응답 앞 300자만 남겼다.
- `nemotron35_c03_*.log`의 `exit=`는 샌드박스 안 명령의 종료 코드다(명령 문자열에 `$?`를 이스케이프해 샌드박스 안에서 평가되게 했다).

## 결과

| 단계 | 결과 | 파일 |
|---|---|---|
| 베이스 이미지 `rl-triage-sandbox:0.1` 빌드 | 성공, 114초 | `base_image_build.log` |
| provider 프로필 lint·import (`policies/providers/nvidia.yaml`) | 성공 | `provider_profile.log` |
| `scripts/sandbox_up.sh 42` 1차 | 실패: supervisor가 게이트웨이에 접속하지 못함 | `sandbox_up_attempt1_loopback.log` |
| `scripts/sandbox_up.sh 42` 2차 (따라 하기 3단계 설정 뒤) | 성공: 마지막 줄 `ls /sandbox/workspace/cases` 결과 10개. 이후 `openshell sandbox list`에서 Ready | `sandbox_up.log`, `gateway_state_after_reconfig.log` |
| 커널 시험 `tests/sandbox` | **8 passed in 0.52s** | `sandbox_kernel_tests.txt` |
| 샌드박스 안 `NVIDIA_API_KEY` | 자리표시자 `openshell:re…`(길이 57), 실제 키 아님 | `key_placeholder_check.txt`, `key_placeholder_check_script.txt` |
| 사례 c03 `nat run` (`configs/triage_workflow.yml`) | **실패**: NVIDIA API `404 Function … Not found for account`. 사전등록 파일 없음 | `nat_run_c03.log` |
| 같은 키로 샌드박스 밖 호출 | 3 Super·nano 3 30B는 404, 3.5 Super VL은 200 | `outside_sandbox_api_check.log` |
| 사례 c03 Nemotron 3.5 입구 `--prepare-only` | 성공, `ready_for_execute: true`, 모델 호출 0 | `nemotron35_c03_prepare.log` |
| 사례 c03 Nemotron 3.5 입구 `--execute` | **실패**(모델 호출 전): 이미지에 `/sandbox/app/evals/grounded_active_model_call.py`가 없음 | `nemotron35_c03_execute.log` |

커널 시험 8개는 `tests/sandbox/test_kernel_boundary.py`에 있다. 예선 6개에 2026-10-09 커밋 `f36929e`에서 추가된 2개(진단 출력 쓰기 허용, 작업공간 루트 쓰기 차단)가 더해졌다.

### `nat run` 실패에 대해

- 확인함(`outside_sandbox_api_check.log`):
  - 공개 모델 목록(`GET /v1/models`, 인증 없음) 84개에 세 모델 ID가 모두 있다.
  - 같은 키로 샌드박스 밖에서 호출하면 `nvidia/nemotron-3-super-120b-a12b`와 `nvidia/nemotron-nano-3-30b-a3b`는 404, `nvidia/nemotron-3.5-super-vl-120b-a12b`는 200이다.
  - 따라서 404는 샌드박스 탓이 아니고, 이 키의 계정에서 특정 모델 ID를 호출할 수 없는 문제다. 키 자체는 유효하다.
- 추정: 샌드박스 안 404 응답도 계정을 지목하므로, 프록시가 자리표시자를 실제 키로 바꿔 보낸 것으로 본다.
- 공개 사본에서는 응답의 계정 ID를 `<redacted>`로 가렸다(`nat_run_c03.log` 5곳, `outside_sandbox_api_check.log`). `nat_run_c03.log` 원본은 저장소 밖에 보관한다.
- 3 Super 접근이 확인되면 같은 명령(`nat_run_c03.log` 2행)으로 다시 실행하고, 결과는 새 `_v2` 폴더에 남긴다.

### Nemotron 3.5 입구로 대신 시도한 것

- 문서화된 `python -m rl_triage.nemotron35_entry_v1`([docs/NEMOTRON35-ENTRY.md](../../../docs/NEMOTRON35-ENTRY.md), `configs/triage_nemotron35_v1.yml`)을 샌드박스 안에서 시도했다. 입력은 `--run-dir /sandbox/workspace/cases/c03`이다.
- 이 입구는 `triage_workflow.yml`과 도구·지시가 다른 별도 workflow다. 성공했더라도 예선 c03 기록과 같은 실험은 아니다.
- `--prepare-only`는 통과했다. `--execute`는 workflow에 들어가기 전에 `evals/grounded_active_model_call.py`를 불러오다 멈췄다. `docker/Dockerfile.sandbox`가 `src`·`configs`·`tests`만 복사하고 `evals/`는 넣지 않기 때문이다. 모델 호출은 없었다.
- 이 경로를 샌드박스에서 쓰려면 이미지에 필요한 `evals/` 모듈을 넣어야 한다. 이번에는 이미지를 바꾸지 않았다.

## 환경 (로그 없음, 실행 중 확인한 값)

- Windows 11 Home, WSL2 Ubuntu 24.04.4(systemd), Docker Desktop 29.5.2(WSL integration: Ubuntu)
- OpenShell 0.1.1: 공식 릴리스 v0.1.1의 `openshell_0.1.1-1_amd64.deb`. SHA-256 `2360c9ee…3325`가 공식 `openshell-checksums-sha256.txt`와 일치했다(`install_openshell.sh`가 설치 전에 `sha256sum -c`로 확인한다). 이 패키지에 `openshell`, `openshell-gateway`, `openshell-prover`(모두 `/usr/bin`)가 들어 있다.
- `docs/HANDOFF.md`의 예선 환경은 snap이었다. snap은 스토어 채널만 따르고 버전 고정을 지원하지 않아(공식 `install.sh` v0.1.1 usage) 이번에는 deb를 썼다.

## 따라 하기

모든 WSL 명령은 저장소 루트에서 실행한다. 이 PC에서는 Windows 사본을 WSL에서 `/mnt/c/Users/<user>/…/rl-triage-agent`로 열어 같은 `workspace/seed42`를 썼다.

1. **설치(sudo).** `bash evals/results/sandbox_rerun_20261106_v1/install_openshell.sh`. sudo 단계(deb 설치, docker 그룹 추가, linger 설정)를 이 스크립트 하나로 모았다. 이 deb는 게이트웨이 사용자 서비스를 켜지 않는다(deb의 postinst 주석, `dpkg-deb -e`로 확인, 로그 없음).
2. **WSL 다시 시작.** systemd 사용자 관리자가 docker 그룹을 갖게 하려면 필요하다. 이 PC에서는 `wsl --terminate Ubuntu` 뒤 Ubuntu 안 `/usr/bin/docker`의 대상(`/mnt/wsl/docker-desktop/cli-tools/…`)이 사라져 docker를 쓸 수 없었다(로그 없음). Docker Desktop 종료 → `wsl --shutdown` → Docker Desktop 시작 순서로 복구했다.
3. **게이트웨이(Docker Desktop용 설정).** 이 단계의 명령은 실행 당시 로그를 남기지 않았다. 아래는 정리본이고, 결과 상태는 `gateway_state_after_reconfig.log`에 있다.
   - 원인: 기본 설정은 게이트웨이를 `127.0.0.1:17670`에 띄우고, supervisor 컨테이너(host 네트워크)도 `127.0.0.1`로 접속한다. Docker Desktop에서 컨테이너의 host 네트워크는 Ubuntu가 아니라 Docker Desktop VM이어서 1차 시도가 `failed to connect to OpenShell server`로 실패했다. docker 드라이버 문서(`crates/openshell-driver-docker/README.md` v0.1.1)의 "Set `grpc_endpoint` when the gateway is not on the Docker daemon host"에 따라 다음을 설정했다.
   ```bash
   IP=$(ip -4 -o addr show eth0 | awk '{print $4}' | cut -d/ -f1)
   systemctl --user stop openshell-gateway
   mv ~/.local/state/openshell/tls ~/.local/state/openshell/tls.bak-20261106
   openshell-gateway generate-certs --output-dir ~/.local/state/openshell/tls \
     --server-san host.openshell.internal --server-san "$IP"
   # gateway.toml.example의 <WSL_IP>를 $IP로 바꿔 ~/.config/openshell/gateway.toml에 저장(권한 600)
   systemctl --user daemon-reload
   systemctl --user enable --now openshell-gateway
   openshell gateway add https://127.0.0.1:17670 --local --name openshell   # 이미 등록돼 있으면 먼저 openshell gateway remove openshell
   openshell status   # Status: Connected, Version: 0.1.1
   ```
   - WSL NAT IP는 WSL을 다시 시작하면 바뀔 수 있다. 바뀌면 인증서와 `grpc_endpoint`를 다시 맞춰야 한다.
4. **키.** 키 파일 `~/.config/nvidia/env`(`export NVIDIA_API_KEY=...`, 권한 600)는 사용자가 직접 쓴다. 저장소에는 두지 않는다.
5. **provider 프로필.** `openshell provider profile lint -f policies/providers/nvidia.yaml`, 이어서 `openshell provider profile import -f policies/providers/nvidia.yaml --global` (`provider_profile.log`)
6. **베이스 이미지.** `docker build -f docker/Dockerfile.sandbox -t rl-triage-sandbox:0.1 .` (`base_image_build.log`). 저장소에 원문 명령이 없어 Dockerfile 경로에 맞춰 정했다.
7. **작업공간.** Windows PowerShell 또는 Git Bash에서 저장소 `.venv`로 `uv run --no-sync python bench/build_workspace.py 42` → `workspace/seed42`(cases 10). WSL에서 같은 폴더가 보이면 된다.
8. **샌드박스·시험.** `bash scripts/sandbox_up.sh 42`, 커널 시험, `nat run`은 각 로그 2행의 `$` 줄 그대로 실행했다. 키 확인은 첫 실행(`key_placeholder_check.txt`) 때 명령 원문을 남기지 않았다. 같은 로직을 `key_placeholder_check.sh`로 정리해 다시 실행했고(`key_placeholder_check_script.txt` 2행), 판정이 같았다.

## 사용자 홈 설정 변경과 원복

저장소 밖에서 바꾼 것은 다음뿐이다.

| 바꾼 것 | 원복 |
|---|---|
| `~/.config/openshell/gateway.toml` 새로 만듦 | 파일을 지우고 `systemctl --user restart openshell-gateway` |
| `~/.local/state/openshell/tls`를 IP SAN 포함으로 다시 만듦. 원래 것은 `~/.local/state/openshell/tls.bak-20261106` | 게이트웨이를 멈추고 `tls`를 지운 뒤 백업을 `tls`로 되돌리고 다시 시작. 그다음 `openshell gateway remove openshell` 뒤 `openshell gateway add https://127.0.0.1:17670 --local --name openshell` |
| `openshell gateway add`로 CLI 등록(`~/.config/openshell/gateways/openshell`) | `openshell gateway remove openshell` |
| `systemctl --user enable openshell-gateway` | `systemctl --user disable --now openshell-gateway` |
| `install_openshell.sh`: deb 설치, docker 그룹, linger | `sudo dpkg -r openshell`, `sudo gpasswd -d $USER docker`, `sudo loginctl disable-linger $USER` |

## 실행하지 않은 것

- `openshell-prover`로 `evals/prove_policies.py`를 다시 증명하는 것은 이번 범위가 아니다.
- 성공한 `nat run`과 사전등록 JSON은 없다(위 404와 이미지 누락).
