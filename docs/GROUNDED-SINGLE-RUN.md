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
from rl_triage.grounded_evidence import evidence_catalog_group_index
print(json.dumps(evidence_catalog_group_index("example"), ensure_ascii=False, indent=2))
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

이 흐름은 아래 네 도구를 사용한다. 결과는 `preregistrations/grounded_assessments/<case_id>.json`에 한 번 저장한다. 모델이 만든 코드나 셸 명령을 실행하는 도구는 이 흐름에 포함되지 않는다.

1. `grounded_evidence_group_index`: 값 없이 telemetry tag와 설정 경로 앞 세 구간으로 만든 그룹을 반환한다. payload는 65,536 UTF-8 byte를 넘으면 거절하며, 크기·해시·조회 상태 metadata는 별도로 반환한다.
2. `grounded_evidence_group_page`: 모델이 선택한 그룹에서 최대 64개 ID·경로·타입·단위를 읽는다. 미조회 그룹과 행 수를 기록한다.
3. `grounded_evidence_details`: 선택한 ID 최대 16개의 원문 행을 읽고 현재 전체 catalog SHA를 대조한다. host가 새 선택 receipt를 `preregistrations/evidence_selections/<case_id>__<selection_id>.json`에 한 번 저장하고 `selection_binding`을 반환한다.
4. `write_grounded_assessment`: 받은 `selection_binding`을 필수로 제출한다. 저장된 receipt·case·catalog·source SHA를 대조하고, 전제의 ID가 해당 선택 집합 안에 있을 때만 원본 값·예측 구조를 검증해 저장한다.

모델 지시는 그룹 최대 두 개와 상세 조회 한 번을 사용하도록 되어 있다. 부분 조회로 원인을 모두 검토했다고 표현하면 안 된다. 기존 전체 catalog CPU API와 NAT 등록 함수도 유지한다. 이 변경은 입력 전달 범위를 줄이는 기능이며, provider의 token 수용·비용·진단 효과를 입증하는 결과는 아니다.

## 저장할 때 확인하는 내용

| 확인 항목 | 동작 |
|---|---|
| 원본 결속 | context가 없어도 텔레메트리 원본 바이트의 SHA를 제공한다. context가 있으면 설정·context·텔레메트리 SHA를 다시 대조하며, 선택 fact가 없어도 설정·context SHA를 유지한다 |
| 근거 선택 | host가 발급해 저장한 선택 receipt와 최대 16개 ID를 필수로 대조한다. 조회하지 않은 ID를 전제로 쓰면 저장을 거절한다 |
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

별도 [base COM reset 관측기](../src/rl_triage/reset_com_observer.py)는 유일한 `base` body의 초기 COM snapshot을 저장하고, 호출자가 완료한 reset 뒤 fresh PhysX 값을 읽는다. x·y 편차는 원시 표본으로, 최대 절댓값과 허용 범위 이탈 env 비율은 집계 측정으로 구분한다. 다른 body의 정상 COM offset이 base 변화를 가리지 않게 한다. 샘플의 단위·reset ID·유효 수와 부분 coverage를 보존한다. 이 관측기는 reset이나 step을 실행하지 않으며, 초기 snapshot 시점과 reset 완료 여부는 caller 주장이다.

[SDK 실행기](../bench/run_base_com_reset.py)는 [고정 프로토콜](../bench/protocols/base_com_reset_observation_v3.json)의 64개 env에서 짝수·홀수 ID를 번갈아 20회 reset한다. 각 env의 reset은 10회다. X1과 X1_REF는 별도 프로세스에서 실행하며, 학습·정책 추론·추가 `env.reset()`은 하지 않는다. 부모가 전체 초기 COM snapshot의 해시, 원시 편차 재집계, 원본 설정과 이벤트 관리자의 해결된 설정, SDK clock·episode buffer의 연속성과 프로세스 종료를 대조한다. 이벤트 관리자는 설정을 복사해 body 이름과 ID를 해결하므로 두 설정을 별도 snapshot으로 보존한다. 고정 SDK는 첫 `step()` 전에 `reset_buf`를 만들지 않으므로, 이 절차에서는 그 속성의 부재를 명시해 검증한다.

```powershell
cd "$HOME/rl-triage-agent"
& .venv/Scripts/python.exe bench/run_base_com_reset.py --tag base_com_ref_v3 --arm X1_REF
& .venv/Scripts/python.exe bench/run_base_com_reset.py --tag base_com_fault_v3 --arm X1
```

소스와 프로토콜을 커밋한 뒤 실행하며, tag마다 `bench/private/base_com_reset/<tag>`에 설정·원시 표본·부모 receipt를 한 번 저장한다. 실행 중인 계산 작업이 있으면 시작을 거절한다. 이 실행은 reset 동작을 읽는 계측 검증이며, 기존 세 registry의 실행 스텝 수집과 별개다. 모델의 검사 선택·원인 판정·정책 회복 효과는 아직 검증하지 않았다.

base COM 계측의 v2 게이트는 기존에 등록한 화면 앱에 Antigravity 편집기를 추가한 `base_com_reset_readback_v2`를 사용한다. 프로세스 이름과 이미지 경로의 일치, 알려진 계산 작업과 Compute 엔진의 차단, GPU 메모리 여유 기준을 유지한다. GPU를 독점하지 않으므로 성능·비용 비교에 쓰지 않는다. 기존 녹화기의 기본 v1 게이트와 원래 프로토콜은 보존한다.

개발 입력의 CPU JSON/NAT round-trip은 무손실로 확인했다. provider의 context·token 수용과 새 모델 호출 완료는 아직 검증하지 않았다.

현재 소스의 SDK 연결 계약은 별도 `bench/protocols/runtime_readback_smoke_v3.json`에 고정한다. 이전 프로토콜은 보존하고 새 소스 해시를 새 버전에 결속한다. schema와 측정 크기·기준·결과 문구는 유지하며, 이 smoke는 녹화기의 기본 v1 게이트를 사용한다. stock Go2의 기존 체크포인트로 16개 env를 20 control step 실행하며, 그중 4개 env의 값을 읽는다. 새 학습이나 결함 주입은 포함하지 않는다.

```powershell
cd "$HOME/rl-triage-agent"
& .venv/Scripts/python.exe bench/run_runtime_readback_smoke.py --tag readback_sdk_v3 --protocol-version 3
```

새 tag의 `bench/private/runtime_readback/<tag>`에 manifest·설정 snapshot·body와 부모 receipt가 남는다. 실행기는 커밋된 소스·SDK·체크포인트 해시를 대조하고, Windows Job에 연결한 프로세스만 추적한다. SDK clock·완료 보상·재설정 phase·coverage와 PID 생성 시각을 대조한 뒤 종료를 확인한다. 이 smoke의 판정 범위는 API 연결이며, AI 판단 변화와 정책 회복 검증은 별도로 필요하다.

과거 v1 재현은 소스 커밋 `724aefdfc765f0c8de385737fac656b201457cfa`의 checkout과 새 tag를 사용한다. 현재 소스에서 옛 프로토콜을 선택해도 해시가 다르면 실행을 거절한다. 현재 소스 결속은 실행 준비이며, 아래 v1 실측을 새 버전의 실행 결과로 세지 않는다.

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
