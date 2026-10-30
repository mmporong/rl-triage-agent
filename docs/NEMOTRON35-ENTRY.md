# Nemotron 3.5 사용자 실행 입구

`src/rl_triage/nemotron35_entry_v1.py`는 사용자가 자기 Isaac Lab RSL-RL 학습 로그로 Nemotron 3.5 grounded 진단을 시작하는 모듈 CLI다. 기존 [grounded 단일 실행](GROUNDED-SINGLE-RUN.md)의 도구 다섯 개와 저장 계약을 그대로 쓰고, 모델과 전송 설정만 새 설정 `configs/triage_nemotron35_v1.yml`로 분리했다. 동결된 3 Super 설정 `configs/triage_single_run_grounded_active.yml`은 바꾸지 않는다.

출력은 가설과 다음 검사 사전등록 한 건이다. 정상 기준은 항상 `absent`이며, 원인 확정·검사 승인·검사 실행·GPU·교정은 하지 않는다.

## 입력

`--run-dir`에는 `bench/extract_tb.py`가 만든 `telemetry.json`이 있어야 한다. TensorBoard 이벤트만 있으면 Isaac Python에서 먼저 변환한다.

```text
<run-dir>/
  telemetry.json        # 필수. {"summary": {...}, "series": {...}}
  configuration.json    # 선택. context.json이 있을 때만 사용
  context.json          # 선택. single_run_context_v1
  model_*.pt, events.*  # 읽지 않고 복사하지 않으며 모델에 보내지 않음
```

`configuration.json`과 `context.json`의 형식은 [설정 근거 안내](SINGLE-RUN-CONTEXT.md)와 같다. `context.json` 없이 있는 `configuration.json`은 사용하지 않고 보고서에 표시한다.

`--task-context`는 다음 세 키만 받는다. `healthy_reference`, `answer_key` 같은 다른 키는 거절한다. `observed_problem`은 사용자 서술이며 모델에는 검증되지 않은 데이터로 전달한다.

```json
{"schema": "nemotron35_task_context_v1", "task": "Isaac-Velocity-Rough-H1-v0",
 "observed_problem": "학습 후반 reward가 떨어진다"}
```

`--out`은 아직 없는 새 폴더여야 하고 `--run-dir`과 겹칠 수 없다. 입력은 임시 폴더에서 검증한 뒤 `--out`으로 한 번에 옮긴다. 검증에 실패하면 `--out`을 만들지 않는다.

## 실행

`--prepare-only`는 CPU에서 입력 계약, 설치 의존성, 설정을 확인한다. 키가 없어도 실행되며 모델을 호출하지 않는다. 설정 확인에는 실제 NAT 1.9 `load_config`를 사용한다.

```bash
python -m rl_triage.nemotron35_entry_v1 --run-dir <run> --task-context task.json --out <새 폴더> --prepare-only
```

`--execute`는 같은 준비를 새 `--out`에 한 뒤 NAT workflow 한 세션을 실행한다. 키는 환경변수 `NVIDIA_API_KEY`에서만 읽고, 출력·기록에는 존재 여부만 남긴다. 키가 없으면 모델을 호출하지 않고 `HOLD`로 끝난다. 기존 WSL 환경의 예는 다음과 같다.

```bash
cd /mnt/c/Users/<user>/rl-triage-agent
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/rl-triage UV_LINK_MODE=copy PYTHONUTF8=1
. "$HOME/.config/nvidia/env"
uv run --no-sync python -m rl_triage.nemotron35_entry_v1 --run-dir <run> --task-context <task.json> \
  --out <새 폴더> --execute
```

| 출력 | 내용 |
|---|---|
| `entry_packet.json` | 입력·task context·설정 SHA, catalog SHA, 모델에 보낼 사용자 메시지 |
| `entry_report.json` | 의존성·설정·키 존재 여부, catalog 크기, 읽지 않은 run-dir 항목, `ready_for_execute` |
| `workspace/` | 복사한 입력과 `reference_status: absent` reference. 저장 도구의 기록도 여기에 남는다 |
| `execution/record.json` | `--execute`의 상태, 오류 단계, NAT 이벤트의 LLM 호출·token 합계, 도구 예산 |

종료 코드는 준비·저장 성공 0, 의존성·설정 미충족이나 `HOLD` 1, 입력 거절 2다.

## 판정과 역할

모델은 근거 그룹을 고르고 가설과 다음 검사를 제안한다. 전제 값·관계·사전 예측의 구조는 기존 등록 규칙이 저장 시 검증한다. 저장된 `next_check`의 `registered_status`는 기존 registry 결과를 그대로 옮긴다. 현재 host 실행기는 stock Go2 개발 재현만 지원하므로 사용자 실행의 검사는 `host_execution_unsupported`가 된다. 이 입구는 어떤 경우에도 승인·소비·실행하지 않고 `check_execution`을 `HOLD`로 기록한다.

저장 성공은 `hypotheses_only`이다. 도구 오류, 예산 초과, catalog 변경, LLM 호출 수 기록 누락, 출력 파일의 키 문자열 발견은 `HOLD` 또는 `invalid`이며 재시도하지 않는다. NAT 이벤트는 provider가 반환한 모델 ID를 담지 않으므로 `provider_model`은 `not_recorded`다.

## 요청 계약

| 항목 | 값과 근거 |
|---|---|
| 모델 | `nvidia/nemotron-3.5-super-vl-120b-a12b`, `https://integrate.api.nvidia.com/v1` |
| 재시도 | `max_retries: 0`, `do_auto_retry: false`. stub의 500 응답에서 HTTP 요청 1회로 중단 |
| 추론 | `extra_body: {reasoning_effort: none}`. 기존 3.5 단일 호출 설정과 같은 값 |
| 실제 body 키 | `model`, `messages`, `tools`, `temperature`, `max_completion_tokens`, `stream`, `stream_options`, `reasoning_effort` |
| `stop` | 보내지 않는다. 공식 API reference는 `stop`을 문서화된 필드로 기술하며, 이 저장소에 미지원 근거는 없다 |

[공식 API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b-infer)의 요청 schema는 `additionalProperties: false`다. NAT 1.9의 LangChain 경로는 `max_tokens: 16384`를 `max_completion_tokens`로 보내며, 이 이름은 공식 schema에 없다. 같은 문서의 LangChain 예시도 같은 변환을 사용한다. 이 저장소의 [V3 요청 기록](../evals/results/nemotron_endpoint_bottlenecks_20261016_v3/requests.json)에서 3.5 모델의 B3 NAT 요청 11건은 모두 `ok`였다. 이 관측은 당시 hosted API의 수용 여부이며 향후 동작을 보장하지 않는다. `frequency_penalty`처럼 schema에 없는 설정을 추가하면 설정 검사가 거절한다.

[V3 비교](NEMOTRON-ENDPOINT-COMPARISON-V3.md)에서 이 요청 ID의 반환 모델명은 `private/nvidia/nemotron-3.5-super-120b-a12b`였다. 서비스된 가중치의 동일성은 확정하지 않았다.

## 검증 범위

`tests/test_nemotron35_entry_v1.py`는 CPU에서 다음을 확인한다. 모든 fixture는 `synthetic_cpu`이며 실제 학습 로그가 아니다.

- 다른 cwd에서 `--help`와 module import가 NAT·openai·httpx·torch·Isaac을 불러오지 않는다.
- 다른 cwd에서 상대 경로 `--prepare-only`가 최소 입력을 받는다.
- 손상된 telemetry, context만 있는 입력, task context의 누락·추가 키를 거절하고 `--out`을 남기지 않는다.
- `--out`의 겹침·덮어쓰기, 없는 `--run-dir`, 링크 입력을 거절한다. 링크 회귀는 링크 권한이 없는 Windows 계정에서 skip된다.
- 키 값은 stdout·stderr·출력 파일·요청 body에 없고 `Authorization` 헤더에만 있다.
- local stub transport로 실제 NAT workflow 한 세션을 실행해, 6개 요청의 body 키·모델·`stop` 부재·checkpoint 바이트 부재·도구 예산을 확인한다. stub의 도구 순서는 모델 판단이 아니다.
- provider 500에서 요청 1회로 중단하고, 키가 없으면 요청을 만들지 않는다.

실제 NVIDIA API 호출, 실제 사용자 로그, provider 반환 모델·token 수용, 진단 품질은 아직 확인하지 않았다.
