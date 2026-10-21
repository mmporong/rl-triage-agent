# Nebius x NVIDIA 출품 준비

확인일: 2026-10-21. 대상: 기존 RL Triage Agent 개선본, Physical AI 트랙.
이번 사용자 요청에서 참가 방향을 선택했으며 계정 등록·Devpost 접수·최종 제출은 아직 완료하지 않았다.

## 재출품 판정

[공식 규정 §4](https://nebiusglobalaihackathon.devpost.com/rules)는 신규·기존 프로젝트를 허용한다.
접수기간 전부터 존재한 프로젝트는 2026-08-26 09:00 PDT 이후 상당한 개선과 그 설명이 필요하다.
다른 해커톤의 출품 이력 자체를 금지하는 조항은 확인되지 않았다. 이번 대회 안에서 복수 출품할 때
서로 실질적으로 달라야 한다는 조건과 타 행사 출품 이력을 혼동하지 않는다.

사용자는 이 프로젝트가 단독 저작물이라고 확인했다. 공개 라이선스는 Apache-2.0이다.
이전 대회 약관 원문은 현재 공개 접근으로 확보하지 못했으므로 그 계약의 제한이 없다고 단정하지 않는다.
이번 규정의 사전 개발 지원 제한은 Sponsor인 Nebius B.V.와 Administrator인 Devpost에 관한 것이다.
NVIDIA가 이전 행사에 참여했다는 이유만으로 이 제한을 확대하지 않는다.

## 이전 제출과 개선 근거

이전 제출 기준점은 `8d0e7ad`(2026-09-27), 이번 준비의 코드 기준점은
`f0ec12947095246a246a0e497d847d4e9cf4e665`(2026-10-20)다.

| 구분 | 이전 제출 | 현재 공개 코드·근거 |
| --- | --- | --- |
| 사용자 흐름 | 텔레메트리 진단과 다음 실험 등록 | 승인한 probe → 등록 교정 → 재학습 → 고정 행동 평가. [종단 시연](E2E-REWARD-DEMO.md) |
| 남은 결함 | 제출 시점 고도화 대상 | 첫 교정 뒤 미회복을 다시 진단하고 두 번째 교정을 검증. [결과](RESIDUAL-DIAGNOSIS-RESULT.md), [공개 JSON](../evals/results/residual_success_public_20261017/summary.json) |
| 로봇 범위 | 기존 Go2 실험 | ANYmal-C 이식 및 실행 기록. [이식 근거](ANYMAL-C-PORT.md) |
| 적용 범위 | 등록된 교정의 범위 제한 | 로봇·결함·교정 조합의 적용성 계약을 진단 고리에 연결. [계약](COMBINATION-APPLICABILITY.md) |
| 이번 추가분 | NVIDIA API 실행 설정 | Nebius Token Factory 전용 설정 2개. 실제 Nebius API 실행은 미검증 |

위 실행 결과는 저장소에 보존된 기존 NVIDIA 경로의 측정 기록이다. 이번 준비 과정에서 GPU 실험을 다시 돌리거나
같은 수치가 Nebius에서도 재현됐다고 주장하지 않는다. 소수 개발 사례를 일반화 성능이나 비용 우위로 쓰지 않는다.

## 계정 없이 진행하는 프로젝트 작업

현재 계정·카드 등록은 보류하고 입력 준비와 검사 기능부터 개발한다. 공개 telemetry JSON으로
동작을 확인한 뒤 사용자 실행의 JSON을 같은 경로에 넣을 수 있다.

```bash
cd "$HOME/rl-triage-nebius-2026"
uv run rl-triage prepare-run --telemetry bench/runs/c01_s42.telemetry.json \
  --workspace workspace/imported-c01 --case c01
uv run rl-triage inspect-run c01 --workspace workspace/imported-c01 \
  --report workspace/c01-input-report.json
```

지표·선택 설정·입력 해시를 검사하며, AI 진단과 실제 Nebius 호출은 아직 수행하지 않는다.
입력 형식과 설정 선택은 [단일 실행 입력 안내](SINGLE-RUN-CONTEXT.md)를 따른다.

## 계정 준비와 모델 확인

1. [Token Factory 가입](https://tokenfactory.nebius.com/) 후 API 키를 발급한다. 공식 [결제 안내](https://docs.tokenfactory.nebius.com/other-capabilities/billing-new)는 가입 완료에 카드 등록을 요구하므로 이 단계는 사용자 요청에 따라 나중에 진행한다.
2. [행사 리소스](https://nebiusglobalaihackathon.devpost.com/resources)의 폼과 Builders Program에서
   각각 안내하는 Token Factory 25달러 크레딧을 신청한다. 실제 발급과 한국 거주 계정 개설은 아직 확인하지 않았다.
3. 키는 저장소·채팅·명령 기록에 넣지 않고 현재 터미널 환경에 넣는다.

```bash
cd "$HOME/rl-triage-nebius-2026"
uv sync
read -r -s -p 'Nebius API key: ' NEBIUS_API_KEY
export NEBIUS_API_KEY
printf '\n'
```

모델은 공식 제공 안내가 있는 `nvidia/nemotron-3-super-120b-a12b`를 후보로 설정했다.
계정에서 모델 가용성과 도구 호출 지원을 먼저 확인한다. 아래 명령은 추론 대신 모델 카탈로그를 조회한다.

```bash
uv run python - <<'PY'
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
    max_retries=0,
    timeout=30,
)
catalog = client.models.list(extra_query={"verbose": "true"})
matches = [model.model_dump() for model in catalog.data
           if model.id == "nvidia/nemotron-3-super-120b-a12b"]
if not matches:
    raise SystemExit("이 계정의 모델 목록에서 Nemotron 3 Super를 찾지 못했습니다.")
print(json.dumps(matches, ensure_ascii=False, indent=2))
PY
```

## Nebius 워크플로 실행

아래 명령은 준비한 벤치마크 텔레메트리에 대해 실제 유료 추론을 요청할 수 있다. 크레딧·한도를 확인한 뒤 실행한다.
도구를 사용한 진단과 다음 실험 등록까지 수행하며 Isaac 재학습·실물 로봇 조작을 시작하지 않는다.

```bash
uv run python bench/build_workspace.py 42
export TRIAGE_WORKSPACE="$PWD/workspace/seed42"
export TRIAGE_ANALYSIS_SANDBOX=required
uv run nat run --config_file configs/triage_workflow_nebius.yml \
  --input "Triage failed training case_id=c01. Rank the causes from telemetry and preregister one next experiment."
```

건강한 기준 실행이 없는 단일 실행 평가는 [기존 입력 형식](SINGLE-RUN-CONTEXT.md)으로 작업공간을 준비하고
`configs/triage_single_run_nebius.yml`을 선택한다.

두 Nebius 설정은 모델 제공자의 URL과 자격증명 환경변수만 바꾼다. 과거 평가용 YAML·프롬프트·도구·판정 기록은
이번 제공자 설정의 실행 증거로 대체하지 않는다. 모델 목록에 있어도 실제 도구 호출 성공은 별도 실행으로 검증한다.
기존 OpenShell 정책은 NVIDIA 호스트에만 맞춰져 있다. 이 문서의 Nebius 명령은 호스트 실행용이며,
OpenShell용 허용 호스트·provider를 검증한 것으로 표시하지 않는다.

## 제출 전에 남은 일

- Nebius 계정·크레딧·API 키, 모델 가용성과 도구 호출 확인.
- 실제 입력 → Nebius 모델 호출 → 도구 분석 → 결과 저장의 새 실행 기록과 설정·코드 SHA 보존.
- Physical AI 영상에는 실물 또는 핵심 시뮬레이션 모듈의 동작 장면 1분 이상을 포함하고 전체는 3분 미만으로 제작.
- 공개 코드·라이선스·실행 README, 영문 설명·영상/번역·기술 피드백을 준비.
- 공개 저장소의 단독 저작권 및 제3자 모델·코드·에셋 라이선스를 제출 구성에 맞춰 대조.
- Devpost 등록·최종 제출과 개인정보·약관 동의는 실제 수용 상태를 확인한 뒤 완료 처리.

제출 마감은 **2026-10-31 02:00 KST**이며 [공식 일정](https://nebiusglobalaihackathon.devpost.com/details/dates)은
2026-10-30 10:00 PDT로 표시한다. 계정이 없는 현재 상태는 `selected / account_pending`이며 `ready_to_submit`이 아니다.

## 공식 기술 근거

- [Token Factory Quickstart](https://docs.tokenfactory.nebius.com/quickstart): OpenAI 호환 URL과 `NEBIUS_API_KEY`.
- [모델 카탈로그 조회](https://docs.tokenfactory.nebius.com/api-reference/examples/list-of-models): 계정별 모델 ID와 상세 속성 확인.
- [Nemotron 3 Super 제공 안내](https://nebius.com/blog/posts/nemotron3-super-now-available): 공개 모델 제공 사실. 사용자 계정의 호출 성공 증거는 아니다.
- [행사 FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs): Nebius 런타임 호출 요건과 Physical AI 시뮬레이션 허용.
