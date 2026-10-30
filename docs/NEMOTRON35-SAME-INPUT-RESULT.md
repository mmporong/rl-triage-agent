# Nemotron 3.5 동일 수치 개발 비교 결과

같은 수치 capsule을 A(3 Super)와 B(Nemotron 3.5)에 보내는 등록 실행을 2026-10-30에 완료했다. 이미 본 F2 두 계보와 F1 두 계보를 대상으로 논리 호출 8개를 한 번씩 보냈다. A는 네 호출 모두 HTTP 404로 `model_unavailable`, B는 네 호출 모두 형식 검증을 통과했다. A 응답이 없으므로 모델 변경 축은 `inconclusive`다. 영상 조건 C도 실행하지 않아 영상 기여 축은 `inconclusive`다.

공개 결과는 [`evals/results/nemotron35_same_input_dev_20261030_v1`](../evals/results/nemotron35_same_input_dev_20261030_v1)에 있다. `summary.json`은 집계, `calls.json`은 원문 본문을 제외한 호출별 검증 결과, `source-map.json`은 원문 SHA 연결을 담는다. provider raw body, 헤더, 키, private 경로는 공개하지 않는다.

## 등록과 실행 경계

- source commit: `15b597fdd8bf9d3008472ba977406e3af91a3aa1`
- 실행 등록 commit: `287d994b803dfb7f2961a1193356e2975fa6ef19`
- protocol SHA-256: `6a0c545676947b1dee9850377d1a3f16989fae0b159fa08dfa50156ec13c466d`
- plan digest: `be20997514c7ae05f0bfeb2d108600b4f2c29f466fd8ed514dfab9e5184f89af`
- P5 manifest digest: `1bd0c8888af2305f6782175b0ed17f178289df243cf530318160c56278e1b57c`
- 실제 요청 8/등록 상한 24, 재시도 0

계보별 순서는 A→B, B→A, A→B, B→A로 교차했다. 같은 계보의 A와 B는 capsule SHA와 params SHA가 같고 model ID만 다르다. 기존 API10/12 원장과 provider 1/2 상태는 이 실행에 섞지 않았다.

## 관측

| arm | 요청 모델 | 상태 | 선택 | usage |
|---|---|---|---|---|
| A | `nvidia/nemotron-3-super-120b-a12b` | 4/4 HTTP 404, `model_unavailable` | 응답 없음 | provider가 제공하지 않아 미제공 |
| B | `nvidia/nemotron-3.5-super-vl-120b-a12b` | 4/4 `valid` | 4/4 `P_episode` | 입력 10,738 · 출력 260 · 합계 10,998 tokens |

B의 실제 반환 모델 별칭은 `private/nvidia/nemotron-3.5-super-120b-a12b`였다. `valid`는 JSON·fact ID·eligible probe·반환 모델·usage 계약이 맞았다는 뜻이다. 등록된 확인 검사는 F2에서 `P_torque`, F1에서 `P_reward`였으나 B는 네 계보 모두 `P_episode`를 골랐다. 따라서 형식 유효와 좋은 검사 선택은 구분해야 하며, 이번 네 선택 중 등록 확인 검사를 고른 횟수는 0이다.

catalog에 두 model ID가 모두 나타나는 사실은 A inference 성공 근거가 아니다. A의 과거 응답을 이번 실행 결과로 대체하지 않았고, 404 이유를 모델 성능 실패로 해석하지 않았다. A 사용량도 B 사용량과 합산하지 않았다.

이 결과는 이미 본 개발 계보 네 개에 한정된다. B 응답의 형식 유효성은 확인됐지만 A/B 우위, 영상이 검사 선택을 개선했다는 주장, 비용 절감, 일반화, 새 회복은 모두 성립하지 않는다. 별도 GPU 회복 결과와도 합산하지 않는다. A가 같은 provider에서 다시 제공되면 새 등록 실행으로 비교를 다시 닫아야 한다.

## 재생 검증

```bash
PYTHONUTF8=1 PYTHONPATH=src uv run python evals/nemotron35_value_compare_call_v1.py \
  --resume --run <보관된-실행-폴더>
PYTHONUTF8=1 PYTHONPATH=src uv run python evals/nemotron35_value_compare_call_v1.py \
  --score --run <보관된-실행-폴더>
```

공개 exporter는 원문 SHA, receipt SHA, source SHA, protocol·plan·P5 digest, A/B 동일 capsule·params, 호출 수와 cap을 다시 확인한 뒤 요약을 만든다. 같은 출력 폴더에서 다시 실행하면 바이트가 같을 때만 통과한다.
