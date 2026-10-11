# 정상 기준 없는 실행의 근거 검증

`configs/triage_single_run_grounded.yml`은 한 실행의 텔레메트리 요약과 선택된 설정값을 읽고, 복수 가설의 전제와 다음 검사 제안을 저장한다. Nemotron과 NeMo Agent Toolkit을 사용하며, 출력은 `hypotheses_only`다.

설정값이 인용과 일치하는지, 그 값이 `eq`, `ne`, `lt`, `lte`, `gt`, `gte` 관계를 만족하는지 코드가 확인한다. 이 확인은 가설의 원인 설명이나 검사 예측의 과학적 타당성을 입증하지 않는다. 자유 설명에는 `model_narrative_unverified`, 검사 예측에는 `prediction_unverified`가 남는다.

## 입력과 실행

[기존 설정 근거 안내](SINGLE-RUN-CONTEXT.md)에 따라 별도 workspace를 준비한다. `reference/telemetry.json`은 `{"reference_status":"absent","summary":{},"series":{}}`이며, 각 사례는 `telemetry.json`과 선택 사항인 `configuration.json`, `context.json`을 갖는다.

```bash
cd "$HOME/rl-triage-agent"
export TRIAGE_WORKSPACE="$PWD/workspace/single_run"
uv run python - <<'PY'
import json
from rl_triage.grounded_evidence import evidence_catalog
print(json.dumps(evidence_catalog("example"), ensure_ascii=False, indent=2))
PY
```

catalog 조회에는 모델, API 키, GPU가 필요하지 않다. 모델 흐름은 기존 WSL/Linux API 환경에서 실행한다. `example`은 준비한 case ID로 바꾼다.

```bash
cd "$HOME/rl-triage-agent"
export TRIAGE_WORKSPACE="$PWD/workspace/single_run"
. "$HOME/.config/nvidia/env"
uv run nat run --config_file configs/triage_single_run_grounded.yml \
  --input "Assess case example, cite the catalog, and register one read-only next check."
```

이 흐름은 `grounded_evidence_catalog`와 `write_grounded_assessment` 두 도구를 사용한다. 결과는 `preregistrations/grounded_assessments/<case_id>.json`에 한 번 저장한다. 모델이 만든 코드나 셸 명령을 실행하는 도구는 이 흐름에 포함되지 않는다.

## 저장할 때 확인하는 내용

| 확인 항목 | 동작 |
|---|---|
| 원본 결속 | context가 없어도 텔레메트리 원본 바이트의 SHA를 제공한다. context가 있으면 설정·텔레메트리 SHA를 다시 대조한다 |
| 제공 범위 | 등록한 설정 fact만 반환한다. 텔레메트리는 원본 summary의 고정 필드를 반환하며, 샘플 곡선으로 기존 summary를 재계산하지 않는다 |
| 자료 없음 | `missing`, 명시적 `null`, 실제 숫자 `0`을 구분한다. 미제공 전제를 인용하면 `HOLD`이며 평가 파일을 만들지 않는다 |
| 가설 전제 | 고유 ID, 연속 순위, 근거 ID·타입·단위·관계를 검증한다. 거짓 관계와 알 수 없는 ID는 저장을 거절한다 |
| 다음 검사 | 등록된 read-only 검사와 measurement ID만 사용한다. 선택한 가설 쌍마다 한 측정 이상에서 사전 예측이 서로 배타적이어야 한다 |
| 실행 경계 | 검사 선택은 `approval_pending`, 측정은 `not_observed`, 판단은 `hold`로 남는다. 이 도구가 Isaac를 실행하거나 승인을 소비하지 않는다 |
| 변경과 덮어쓰기 | 조회 뒤 원본이 바뀌면 저장을 거절한다. 기존 평가 파일을 덮어쓰지 않는다 |

전제의 형태는 다음과 같다. `evidence_id`와 `unit`은 받은 catalog의 값을 사용한다.

```json
{
  "evidence_id": "configuration::success_weight",
  "relation": "gt",
  "operand": {"value": 0, "unit": "configuration_value"}
}
```

가설에는 `id`, `rank`, `narrative`, `premises`가 필요하다. 다음 검사는 `registry_id`와 `hypothesis_predictions`를 받는다. 각 가설의 예측은 registry에 있는 measurement를 한 번씩 포함하고, 측정별 `relation`과 `operand`를 기록한다. 두 가설이 일부 측정에서 같은 값을 예측해도, 다른 측정에서 배타적인 예측을 제시하면 구조 검증을 통과한다.

## 실행 상태를 읽는 수집기

`src/rl_triage/runtime_observers.py`의 `RuntimeReadbackCollector`는 완료된 제어 스텝의 상태를 읽도록 만든 코드다. 등록한 env index와 샘플 수, 호출자가 control/physics index와 episode·reset phase라고 제공한 값, task/checkpoint/config/observer SHA에 묶인다. 실제 Isaac clock·phase·reward 출처의 결속은 launcher와 receipt에서 검증해야 한다. 수집기는 env를 생성하거나 스텝·reset·보상 재계산을 실행하지 않는다.

| Registry | 읽는 값 | 해석 범위 |
|---|---|---|
| `runtime_com_state` | root COM 높이·수직 속도, body-local COM 좌표 최대 절댓값 | 원래 COM과의 차이나 실패 원인을 자동 판정하지 않는다 |
| `runtime_reward_window` | 호출자가 완료된 `env.step` 반환값으로 제공한 reward 평균·최솟값 | reward term readback은 가중치가 적용되고 dt 적분 전인 별도 근거다 |
| `runtime_contact_window` | contact force norm 최대, nonzero contact가 있는 env/sample 비율 | norm `> 0`은 접촉 존재이며 실패 임계값이 아니다. auto-reset 뒤 contact는 unavailable이다 |

수집기는 자료 없음·비유한 값·읽기 오류를 0으로 채우지 않는다. 측정 phase와 유효/미제공 표본 수를 기록하며, 원인·정상성·회복·프로세스 정리 확정은 모두 false다. 종료 확인은 부모 receipt가 수행한다. 모델 선택을 소비하는 승인 고리와 새 진단 효과는 별도로 검증해야 한다.

개발 입력의 CPU JSON/NAT round-trip은 무손실로 확인했다. provider의 context·token 수용과 새 모델 호출 완료는 아직 검증하지 않았다.

실제 SDK 연결은 별도 `bench/protocols/runtime_readback_smoke_v1.json`으로 고정한다. stock Go2의 기존 체크포인트로 16개 env를 20 control step 실행하며, 그중 4개 env의 값을 읽는다. 새 학습이나 결함 주입은 포함하지 않는다. 결과별 문구도 실행 전에 프로토콜에 기록한다.

```powershell
cd "$HOME/rl-triage-agent"
& .venv/Scripts/python.exe bench/run_runtime_readback_smoke.py --tag readback_sdk_v1
```

새 tag의 `bench/private/runtime_readback/<tag>`에 manifest·설정 snapshot·body와 부모 receipt가 남는다. 실행기는 커밋된 소스·SDK·체크포인트 해시를 대조하고, Windows Job에 연결한 프로세스만 추적한다. SDK clock·완료 보상·재설정 phase·coverage와 PID 생성 시각을 대조한 뒤 종료를 확인한다. 이 smoke의 판정 범위는 API 연결이며, AI 판단 변화와 정책 회복 검증은 별도로 필요하다.

[소스 커밋 `724aefd`의 실측 요약](../evals/results/runtime_readback_sdk_v1_20261011/result.json)과 [원시 body](../evals/results/runtime_readback_sdk_v1_20261011/body.json)는 20 control step·80 physics step·4개 env의 80표본을 보존한다. 7개 측정 모두 유효한 값 80개·미제공 0개였고, 부모 wall은 13.078초였다. 원시 재집계와 해시·프로세스 종료의 별도 검수도 통과했다. 이 실행은 모델이 검사를 선택한 사례가 아니며, 정책 건강·원인·회복은 판정하지 않았다.

## 참고한 연구와 코드

[ReAct](https://arxiv.org/abs/2210.03629)의 행동·관측 연결과 [CRITIC](https://arxiv.org/abs/2305.11738)의 외부 도구 피드백을 참고해, 근거 조회와 코드 검증을 분리했다. [Reflexion](https://arxiv.org/abs/2303.11366)의 피드백은 문맥 갱신에 관한 방법이다. 고정 API 모델의 가중치가 자동으로 학습된다고 설명하지 않는다.

[Bayesian experimental design](https://doi.org/10.1214/23-STS915)은 다음 측정의 판별력과 비용을 고려할 근거다. 현재 검증기는 사전 예측의 구조와 배타성을 확인하며, 보정한 prior·likelihood나 EIG를 계산하지 않는다. [RLExplorer](https://arxiv.org/abs/2410.04322)의 RL 실행 흔적 진단도 참고했지만, 해당 논문의 성능을 이 도구의 성능으로 사용하지 않는다.

관측 API는 [Isaac Lab v2.1.1](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.1.1)의 [articulation data](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/assets/articulation/articulation_data.py)와 [reward manager](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/managers/reward_manager.py)를 근거로 구현했다. sensor 설정의 존재와 actor 입력·실제 접촉은 각각 확인해야 한다.

후속 효과 평가는 [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264), [Empirical Design in RL](https://arxiv.org/abs/2304.01315), [Deep RL that Matters](https://arxiv.org/abs/1709.06560)의 실험 설계 문제를 반영해, 같은 정보·예산의 규칙 비교와 invalid/HOLD 분모를 보존한다. 이미 본 사례의 통합 동작은 개발 회귀이며 일반화 근거가 아니다. 검사 선택의 유용성과 독립 행동 지표로 측정한 정책 회복을 따로 평가한다.

참고한 공식 구현은 [ReAct](https://github.com/ysymyth/ReAct/tree/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9), [CRITIC](https://github.com/microsoft/ProphetNet/tree/5cf70eb41cdaa1d8faa3e1265d95ee5792d49a53/CRITIC), [Reflexion](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0), [SWE-agent ACI](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/docs/background/aci.md), [RLiable](https://github.com/google-research/rliable/tree/3ccd9f4dea577a04d3d2b557f259aac08badbd81)다. 실행 인터페이스와 평가 관행을 참고했으며 새 프레임워크 의존성을 추가하지 않았다.

## 실행 장애 기록

모델 worker가 결과 파일을 남기지 못하거나 Brev 전송이 실패하면, 새 실행은 관측한 종료 코드·timeout 여부·stdout/stderr 길이와 해시·등록된 WSL 오류 코드를 보존한다. 명령, 키, 환경변수와 출력 원문은 진단 필드에 복사하지 않는다. 문자열 출력의 해시는 디코딩한 텍스트의 UTF-8 해시이며, 바이트 출력의 원본 해시와 구분한다.

이 기록은 앞으로의 장애를 관측하는 기능이다. stderr가 보존되지 않은 과거 실행의 원인을 소급해서 확정하지 않는다.
