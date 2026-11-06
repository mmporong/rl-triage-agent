# Windows 깨끗한 clone 재현 (2026-11-06)

저장소를 처음 받는 사람의 조건으로 Windows에서 README [Quickstart](../README.md#quickstart)와 [Execution and verification](../README.md#execution-and-verification)의 CPU 명령을 다시 실행했다. README에 없는 `pytest` 수집 검사도 추가했다. 모델 API·GPU·Isaac Lab·OpenShell sandbox는 쓰지 않았고 새 학습이나 새 성능 수치를 만들지 않았다. 저장된 결과를 다시 계산하거나 검사하기만 했다.

같은 clone에서 수정 전 `a6cc301`(origin/main)과 수정 후 `db59134`를 같은 스크립트로 실행했다. 원문 로그의 사용자 경로를 가린 공개 사본과 SHA-256은 [`evals/results/windows_clean_clone_repro_20261106/`](../evals/results/windows_clean_clone_repro_20261106/manifest.json)에 있다.

## 1. 환경

| 항목 | 값 |
|---|---|
| OS·셸 | Windows 11 Home 10.0.26200, Git Bash(MINGW64). 아래 명령은 Git Bash 기준이다 |
| Git | 2.46.1.windows.1, 전역 `core.autocrlf=true` |
| uv | 0.13.7. `python -m pip install --user uv`로 설치했으며 `uv.exe`는 사용자 Python `Scripts` 폴더에 생겨 PATH에 직접 추가했다. 공식 설치 스크립트를 써도 된다 |
| Python | uv가 PC에 이미 있던 Python 3.12.7(conda)로 `.venv`를 만들었다. pytest 9.1.1, nvidia-nat 1.9.0 |
| clone 위치 | 사용자 홈 아래 ASCII 경로 |
| WSL | Ubuntu 배포판은 있으나 `openshell-prover`·`openshell`은 설치돼 있지 않다 |
| 없는 도구 | `ffprobe`(4.4절) |

`.gitattributes`가 `* text=auto eol=lf`라서 `core.autocrlf=true`여도 추적 텍스트 파일은 LF로 체크아웃됐다. 바이트를 보존하는 `-text` 예외와 `eol=crlf`인 `*.ps1`은 제외다. 한글·OneDrive 경로의 기존 작업 사본에서도 1~4번 명령은 같은 판정이었다. 그 실행 로그는 공개 기록에 넣지 않았으므로 참고용이다.

## 2. 따라 하기

```bash
cd ~
git clone https://github.com/mmporong/rl-triage-agent.git rl-repro
cd rl-repro
uv sync
export PYTHONUTF8=1
uv run python evals/replay.py evals/results/blind_v1 evals/results/heldout_blind \
  evals/results/heldout_changes --traces evals/results/traces
uv run python evals/prove_policies.py
git status --short                     # prove_policies가 추적 파일을 바꿨는지 확인
uv run python evals/correct_v2.py replay
uv run python evals/build_replay_page.py --check
uv run pytest --collect-only -q tests  # README에 없는 추가 검사
```

PowerShell에서는 `export PYTHONUTF8=1` 대신 `$env:PYTHONUTF8 = "1"`을 쓰고, `\` 줄 이음 없이 한 줄로 입력한다. `PYTHONUTF8=1`은 [HANDOFF.md](HANDOFF.md) 52행의 Windows 실행 방식을 따른 것이며, 설정하지 않은 실행은 확인하지 않았다.

실제 측정에는 위 명령과 6·7번 테스트를 묶은 [`run_all.sh`](../evals/results/windows_clean_clone_repro_20261106/run_all.sh)를 썼다(`--no-sync`, 명령별 exit·초 기록). 공개 사본은 로그 위치만 `REPRO_LOGS` 변수로 바꿨다. 소요 시간은 이 PC에서 다른 pytest 실행과 동시에 잰 벽시계 값이라 기준값이 아니다.

## 3. 결과

| # | 명령 | 기대 | 수정 전 `a6cc301` | 수정 후 `db59134` |
|---|---|---|---|---|
| 1 | `evals/replay.py …` | blind_v1·heldout_blind는 [IMPLEMENTATION-ORDER.md](IMPLEMENTATION-ORDER.md) 2절 G 표와 같고, 세 폴더 모두 저장값과 `mismatches=0` | blind_v1 agent top-1 10/20·top-2 12/20, control 8/20·10/20 / heldout_blind agent 6/10·6/10, control 2/10·6/10 / heldout_changes 10/10 동률 / trace 71개 `leak hits=0` / `status=pass` (1초) | 같음 (4초) |
| 2 | `evals/prove_policies.py` | prover가 없으면 판정 없음을 알리고 저장 기록을 보존 | `TypeError: unsupported format string passed to NoneType.__format__`, exit 1. 표 출력 없음. 추적 파일 `policies/candidates/*.merged.yaml` 4개가 줄끝만 CRLF로 바뀜 (24초) | 다섯 후보 `result=unavailable`, exit 1, `policy_proofs.json` 갱신 안 함, `git status` 변화 없음 (22초) |
| 3 | `evals/correct_v2.py replay` | `status: pass` | `correct_v2_replay_v1` `status: pass`, 사례 6개(proposed 3, HOLD 3) (2초) | 같음 (2초) |
| 4 | `evals/build_replay_page.py --check` | `{"status": "current"}` | `{"status": "stale"}`, exit 1 (229초) | `{"status": "current"}`, exit 0 (103초) |
| 5 | `uv run pytest --collect-only -q tests` | 수집 오류 0 | `ModuleNotFoundError: No module named 'evals'` 수집 오류 5건으로 `Interrupted`, exit 2. 오류 파일 5개를 뺀 9422개 수집 (207초) | 9516개 수집, exit 0 (79초) |
| 6 | 재생 데이터 snapshot 테스트 3개(`test_replay_page.py`·`test_additional_checks_replay_v2.py`·`test_ppo_gradient_replay_v1.py`의 current 검사) | 통과 | 3개 모두 실패, exit 1 (166초) | 3개 통과, exit 0 (160초) |
| 7 | `pytest tests/test_prove_policies.py` | 통과 | 파일 없음(이 커밋에서 추가) | 3개 통과, exit 0 (8초) |

## 4. 발견과 조치

### 4.1 `uv run pytest`가 루트 모듈을 찾지 못함 — `33fff4f`

- 원인: 일부 테스트가 `from evals import …`를 쓰는데 `uv run pytest`는 저장소 루트를 `sys.path`에 넣지 않는다. `python -m pytest`는 현재 디렉터리를 넣어 통과한다.
- HANDOFF 52행의 `PYTHONUTF8=1 uv run --no-sync pytest -q tests` → `13 passed, 6 skipped(샌드박스 전용)`은 2026-10-02 기록이다. 수집 오류를 낸 테스트 파일 5개는 그 뒤에 추가됐다(가장 이른 `tests/test_active_replay.py`가 2026-10-11).
- 조치: `pyproject.toml`에 `[tool.pytest.ini_options] pythonpath = ["."]`를 넣었다. 다른 pytest 설정 파일이 없어 rootdir와 `tests/conftest.py` 동작은 그대로다.

### 4.2 prover가 없을 때 `prove_policies.py`가 중단되고 추적 파일을 바꿈 — `42432af`

- 원인: Windows에서 이 스크립트는 `wsl -d Ubuntu --exec bash -lc "openshell-prover check …"`를 호출한다. prover가 없으면 `result`가 `None`이 되고, 출력의 `:<17` 서식이 `TypeError`를 낸다. `Path.write_text`는 Windows 기본 줄끝(CRLF)으로 merged 후보를 다시 쓴다. README는 이 명령에 WSL과 prover가 필요하다는 점을 적지 않는다.
- 조치:
  - 판정이 없는 후보는 `result=unavailable`로 출력한다. 하나라도 판정이 없으면 저장된 `evals/results/policy_proofs.json`을 덮어쓰지 않고 exit 1로 끝낸다.
  - `wsl`·`bash` 실행 파일이 없어 생기는 `OSError`도 판정 없음으로 처리한다.
  - 두 파일 쓰기를 `newline="\n"`으로 고정했다.
  - 테스트 `tests/test_prove_policies.py` 3개를 추가했다. 수정 전 코드에서는 실패하고 수정 후에는 통과한다.
- 미검증: prover가 설치된 WSL/Linux에서의 실제 증명은 실행하지 않았다.
- 저장된 기록(커밋 `39c7f9e`, 2026-09-26)은 확인했다. 다섯 후보가 모두 기대와 같고 세 후보가 `auto_reject`였다.
  - `bad_exfil_checkpoint`: huggingface.co:443 반례
  - `bad_patch_config`: host.openshell.internal:8750 `PUT /config/` 반례
  - `bad_write_reference`: `unsupported`

### 4.3 재생 데이터의 소스 해시가 CRLF 기준임 — `2650473`, `db59134`

- 원인: 커밋된 재생 데이터 세 파일의 소스 SHA-256 59개가, LF인 git blob을 CRLF로 바꾼 바이트의 해시였다. 그래서 깨끗한 checkout에서 `--check`가 stale을 내고 snapshot 테스트가 실패한다.
  - `docs/replay/data.js`: 51개
  - `additional-checks-v2-data.js`: 2개
  - `ppo-gradient-data.js`: 6개
- 확인: clone 루트에서 `uv run --no-sync python evals/results/windows_clean_clone_repro_20261106/crlf_check.py a6cc301 db59134`를 실행했다. [`crlf_check.py`](../evals/results/windows_clean_clone_repro_20261106/crlf_check.py)가 git blob으로 59개 모두 "새 값 = blob 해시, 옛 값 = CRLF 변환 해시"임을 확인했다. 다른 변경은 0줄이다([로그](../evals/results/windows_clean_clone_repro_20261106/after_db59134/crlf_check.log)).
  - 예: `evals/results/p0c_fixed_eval_20261009/runs/h02_s2026.json`은 blob이 `01b29b3f…`, 옛 기록이 `f7e2ebab…`다.
- 조치: 각 생성기로 최신 main에서 다시 만들었다. data.js(`2650473`)와 나머지 두 파일(`db59134`)은 단독 커밋이다.
- 정규화하지 않은 이유: `evals/replay.py`는 CRLF→LF 정규화 SHA를 기록하지만([IMPLEMENTATION-ORDER.md](IMPLEMENTATION-ORDER.md) 6절), 재생 데이터 생성기는 영상 등 이진 파일까지 원 바이트로 해시하고 manifest의 원 바이트 해시와 대조한다. 그래서 줄끝 정규화를 넣지 않았다.
- 재발 방지: 추적 텍스트 파일이 CRLF인 작업 사본에서는 생성하지 않는다. 생성 전에 `file evals/results/p0c_fixed_eval_20261009/runs/h02_s2026.json`에 `CRLF`가 나오지 않는지 확인한다.
- 범위 밖: `build_anymal_doc.py`·`build_d3_doc.py`의 `--check`도 stale이었다. 해시 문제가 아니라 문서 내용이 다르다. 현재 ANYmal 문서는 `build_anymal_doc_v2.py`가 만들므로 손대지 않았다.

### 4.4 환경 의존 실패

- `tests/test_additional_checks_replay_v2.py::test_actual_media_is_fully_verified_before_commit`는 `ffprobe`가 없어 `FileNotFoundError`로 실패한다. 수정 전후 모두 같다.
- 수정 전 원본 작업 사본에서 시작한 전체 실행의 첫 5%에도 실패·오류가 있었다. 수정 후 `tests/test_a*.py` 부분 실행에서도 같은 위치에 같은 형태로 나타났다. 원인은 분류하지 않았다. 두 실행 모두 중간에 멈췄고 로그를 공개 기록에 넣지 않았으므로 참고용이다.

## 5. 실행하지 않은 것

- pytest 전체 실행. 이 PC에서는 진행이 느려 5% 근처에서 중단했다(참고용, 로그 미공개). 3절 6번의 관련 테스트와 `tests/test_prove_policies.py`·`test_replay_page.py`·`test_offline_replay.py`·`test_additional_checks_replay_v2.py`·`test_ppo_gradient_replay_v1.py`만 실행했다.
- prover 실제 증명, OpenShell sandbox 커널 검사, NVIDIA API 호출, Isaac Lab 학습·평가.
- README의 `bench/extract_tb.py`·`nemotron35_entry_v1 --prepare-only` 경로. 기존 Isaac 학습 실행 폴더가 필요하다.
- `PYTHONUTF8` 없이 실행, PowerShell에서 실행.
- `git clone`·`uv sync` 소요 시간. 로그로 남기지 않았다.
