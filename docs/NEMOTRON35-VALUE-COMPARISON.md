# Nemotron 3.5 동일 수치 A/B 비교 실행 고리

[P5 사례 입력](NEMOTRON35-VALUE-CASES.md)의 같은 수치 capsule로 A(3 Super)와 B(Nemotron 3.5)를 호출하고, 검증된 응답을 P5 `score()`로 판정하는 실행기다. 이 문서 시점에는 실제 API 호출이 없다. A·B 결과는 `pending_api`, 전체 판정은 `inconclusive`다. 영상 조건 C는 계속 `blocked_new_collection_required`이며 이 실행기가 다루지 않는다.

대상은 이미 본 개발 계보 4개(F2 s2033·s2046, F1 s2034·s2080)뿐이다. 논리 호출 8개, 등록 요청 상한 24이며 slot 재시도는 등록하지 않았다. 새 독립 seed 2개 이상의 확인 실험은 별도 사전등록으로 남긴다. 일반 모델 우위, 영상 기여, 비용 절감, 새 회복을 주장하지 않는다.

## 구성

- 실행기: [`evals/nemotron35_value_compare_call_v1.py`](../evals/nemotron35_value_compare_call_v1.py)
- 사전등록(DRAFT, 미실행·미승인): [`bench/protocols/nemotron35_value_compare_call_20261030_v1.json`](../bench/protocols/nemotron35_value_compare_call_20261030_v1.json). 새 소스·테스트의 LF SHA, P5 네 파일 SHA, P5 protocol·manifest digest를 결합한다.
- 회귀: [`tests/test_nemotron35_value_compare_call_v1.py`](../tests/test_nemotron35_value_compare_call_v1.py)

## slot과 순서

`--prepare`는 P5 `prepare()`와 `check()`로 manifest·capsule을 다시 검증한 뒤 8개 slot을 write-once `plan.json`으로 고정한다. P5 계보 순서가 짝수면 A→B, 홀수면 B→A다.

| 순서 | 계보 | 조건 |
|---|---|---|
| 1·2 | F2 s2033 | A → B |
| 3·4 | F2 s2046 | B → A |
| 5·6 | F1 s2034 | A → B |
| 7·8 | F1 s2080 | B → A |

같은 계보의 두 slot은 같은 capsule 파일(messages bytes·SHA), provider, request params, 세 필드 도구 계약, 검사 예산을 쓰고 model ID만 다르다. params는 기존 호출기의 `feedback_model_call.request_params()`와 같다(temperature 0, max_tokens 16384, stream false, reasoning_effort none). capsule을 다시 만들거나 telemetry·config 사실을 빼지 않는다.

## 실행 계약

- `--execute`는 메인 PC가 만든 별도 실행 등록(`--registration`)과 호스트 키가 있을 때만 보낸다. 등록이 없거나 키 환경변수가 비었거나 등록 platform이 다르면 호출 전에 HOLD한다. 키는 `NVIDIA_API_KEY` 이름으로만 읽고 값·헤더·nonce·예외 문자열을 기록하지 않는다.
- slot마다 journal에 intent를 먼저 fsync한 뒤 한 번 보낸다. provider 원문(`raw/`), 요청 receipt(`requests/`), 선택 결과(`results/`)를 각각 write-once로 쓰고 응답 event로 journal을 닫는다. journal은 줄마다 앞줄 SHA를 이어 변조를 막는다.
- receipt에는 logical call ID·attempt·model·capsule SHA·params SHA·request digest·protocol SHA·source SHA·plan digest·실행 등록 SHA·provider 종류·시작/종료 시각·latency·HTTP 상태·원문 SHA·usage·반환 모델·finish_reason을 남긴다.
- 완료 slot은 원문 SHA와 등록값이 맞으면 다시 보내지 않는다. intent만 있고 응답이 없으면 `unknown_pending`으로 실행 전체를 HOLD하며 자동 재전송하지 않는다. `execute.lock`이 남아 있으면 HOLD한다.
- `--resume`은 완료 원문을 다시 검증해 재생할 뿐 provider를 받지 않는다.

| 상태 | 조건 |
|---|---|
| `valid` | HTTP 200, 등록 반환 모델, `finish_reason=stop`, 세 필드·근거·eligible 검사 통과 |
| `semantic_invalid` | HTTP 200 응답이 세 필드 계약 위반(HOLD) |
| `infrastructure_failed` | 전송 오류, 404·410 외 HTTP 오류, 원문·usage 계약 위반, 등록 밖 반환 모델, 빈 응답·잘림(HOLD) |
| `model_unavailable` | HTTP 404·410. A가 이 상태면 A-B는 `inconclusive` |
| `unknown_pending` | intent만 있고 응답 기록 없음(실행 HOLD) |

응답 검증은 생산 화면과 같은 `residual_selection_v2.format_review`와 `selection_comparison_view_v1.select`의 근거·eligible 검사다. `select()`와 달리 무효 응답을 등록 규칙 선택으로 바꾸지 않는다.

`--score`는 plan·P5 manifest·journal·원문·receipt·usage·latency·결과를 다시 검증한 뒤 P5 `CHOICES_SCHEMA` 파일을 만들고 P5 `score()`를 부른다. 결과는 journal 끝 SHA별 `scores/` 폴더에 남는다. 등록 확인 검사를 골랐고 같은 checkpoint receipt가 abnormal인 경우만 맞힌 것으로 세며, 선택 변화만으로 개선으로 세지 않는다. 테스트 stub의 receipt는 synthetic으로 표시되고 실제 측정 상태로 바뀌지 않는다.

## 확인 명령

```bash
PYTHONUTF8=1 PYTHONPATH=src uv run python -m pytest -q -p no:cacheprovider tests/test_nemotron35_value_compare_call_v1.py
PYTHONUTF8=1 PYTHONPATH=src uv run python evals/nemotron35_value_compare_call_v1.py --prepare --run <새-실행-폴더>
PYTHONUTF8=1 PYTHONPATH=src uv run python evals/nemotron35_value_compare_call_v1.py --resume --run <새-실행-폴더>
PYTHONUTF8=1 PYTHONPATH=src uv run python evals/nemotron35_value_compare_call_v1.py --score --run <새-실행-폴더>
```

세 명령은 API 키와 네트워크 없이 동작한다. 실제 전송은 메인 PC가 사전등록을 커밋하고 실행 등록 파일을 만든 뒤 키가 있는 호스트에서 수행한다.

```bash
PYTHONUTF8=1 PYTHONPATH=src uv run python evals/nemotron35_value_compare_call_v1.py --execute --run <실행-폴더> --registration <실행-등록.json>
```

실행 등록 schema는 `nemotron35_value_compare_registration_v1`이며 `kind=registered_real_run`, 이 사전등록의 LF SHA, `--prepare`가 출력한 plan·P5 manifest digest, 실행 소스 commit, `approval_granted=true`와 승인 주체, 요청 상한 24·논리 호출 8, 호스트(`key_env`, `platform`)를 담는다. 실행기는 새 실행 폴더의 journal과 원문만 쓰며 기존 API10/12 원장, provider 1/2 상태, TokenFactory 작업은 읽거나 재설정하지 않는다.
