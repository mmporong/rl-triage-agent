# Nebius x NVIDIA 출품 준비

확인일: 2026-10-25. 대상: 기존 RL Triage Agent 개선본, Physical AI 트랙.
계정 로그인과 제한된 API 연동 시험을 완료했다. Devpost 접수·최종 제출은 아직 완료하지 않았다.

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
| 이번 추가분 | NVIDIA API 실행 설정 | Nebius Token Factory 전용 설정 2개와 [추론·도구 호출 2회 시험](../evals/results/nebius_smoke_public_20261025T152338+0900/README.md). 전체 NAT 워크플로·교정 루프의 Nebius 실행은 미검증 |

위 실행 결과는 저장소에 보존된 기존 NVIDIA 경로의 측정 기록이다. 이번 준비 과정에서 GPU 실험을 다시 돌리거나
같은 수치가 Nebius에서도 재현됐다고 주장하지 않는다. 소수 개발 사례를 일반화 성능이나 비용 우위로 쓰지 않는다.

## 입력 준비와 검사

공개 telemetry JSON으로 입력 준비와 검사 기능을 확인했다. 사용자 실행의 JSON도 같은 경로에 넣을 수 있다.

```bash
cd "$HOME/rl-triage-nebius-2026"
uv run rl-triage prepare-run --telemetry bench/runs/c01_s42.telemetry.json \
  --workspace workspace/imported-c01 --case c01
uv run rl-triage inspect-run c01 --workspace workspace/imported-c01 \
  --report workspace/c01-input-report.json
```

위 두 명령은 지표·선택 설정·입력 해시를 검사하며 모델 API를 호출하지 않는다.
입력 형식과 설정 선택은 [단일 실행 입력 안내](SINGLE-RUN-CONTEXT.md)를 따른다.

## 계정 준비와 모델 확인

1. [Token Factory](https://tokenfactory.nebius.com/) 로그인과 카드 등록 상태를 확인했다. 2026-10-25 체험 잔액은 $1, 잔여 29일이었다. 제한 시험 후 시험용 키를 폐기했고 로컬 보안 파일의 키 할당도 제거했다. 현재 계속 사용할 키는 없다.
2. [행사 리소스](https://nebiusglobalaihackathon.devpost.com/resources)의 폼과 Builders Program에서
   각각 안내하는 Token Factory 25달러 크레딧을 확인했다. 폼의 `NEBIUS-DEVPOST-GLOBAL26`은 실제 코드를 받기 위한 활성화 코드다. 신청·약관 동의·크레딧 발급은 아직 수행하지 않았다.
3. 키는 저장소·채팅·명령 기록에 넣지 않고 현재 터미널 환경에 넣는다.

```bash
cd "$HOME/rl-triage-nebius-2026"
uv sync
read -r -s -p 'Nebius API key: ' NEBIUS_API_KEY
export NEBIUS_API_KEY
printf '\n'
```

모델 `nvidia/nemotron-3-super-120b-a12b`를 계정 카탈로그에서 확인했고 OpenAI 호환 API의 도구 호출도 시험했다.
아래 명령은 추론 대신 모델 카탈로그를 조회한다. 폐기된 시험 키로는 인증되지 않으므로 새 실행을 승인하고 새 키를 준비한 경우에만 사용한다.

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
이번 제공자 설정의 실행 증거로 대체하지 않는다. 제한된 OpenAI 호환 도구 호출 시험은 전체 NAT 워크플로·교정 루프의 실행 검증을 대신하지 않는다.
기존 OpenShell 정책은 NVIDIA 호스트에만 맞춰져 있다. 이 문서의 Nebius 명령은 호스트 실행용이며,
OpenShell용 허용 호스트·provider를 검증한 것으로 표시하지 않는다.

## 2026-10-25 제한 시험과 비용 차단

공개 c01 입력으로 모델의 도구 호출 → 로컬 지표 조회 → 후속 응답을 2회 요청으로 확인했다.
입력 1,225토큰·출력 803토큰과 현재 단가로 계산한 금액은 $0.0010902다. 실행 전 예약 상한 합계는
$0.0131442로, 사용자에게 제시한 $0.05 상한 이하였다. 자동 재시도는 끄고 호출 수를 2회로 제한했다.
이는 접속 시험이며 진단 정확도 평가가 아니다. 모델의 표본 단위 해석에 미확인 표현이 있어 원 응답과 한계를 함께 기록했다.

시험용 키 폐기 뒤 같은 키의 모델 조회가 HTTP 401로 거부되는 것을 확인했다. 프로젝트의 API 키 목록은 비었고,
로컬 키 할당도 제거했다. 지속 실행하는 추론 프로세스·GPU·전용 endpoint는 생성하지 않았다.

카드가 등록된 계정의 자동 청구 자체를 비활성화한 상태는 아니다. 공식 결제 문서는 음수 잔액의 자동 청구를 설명하며,
확인한 설정 화면에서는 달러 지출 hard cap을 찾지 못했다. 추가 비용 방지는 이번 시험 키의 사용 경로를 없앤 범위다.
새 키나 Playground 사용을 다시 허용하면 별도 사용량이 발생할 수 있다. Usage 화면에 이번 요청이 아직 반영되지 않아
위 금액은 토큰 기반 계산치이며 확정 청구액은 아니다.

근거: [시험 요약](../evals/results/nebius_smoke_public_20261025T152338+0900/summary.json),
[폐기 확인](../evals/results/nebius_smoke_public_20261025T152338+0900/revocation.json),
[과금 경계](../evals/results/nebius_smoke_public_20261025T152338+0900/account-safety.json),
[미신청 크레딧](../evals/results/nebius_smoke_public_20261025T152338+0900/credit-offers.json).

## 제출 전에 남은 일

- 무료 크레딧 신청·발급과 다음 실행의 비용 한도, 새 API 키 준비.
- 전체 NAT 워크플로·교정 루프의 Nebius 실행 기록과 설정·코드 SHA 보존.
- Physical AI 영상에는 실물 또는 핵심 시뮬레이션 모듈의 동작 장면 1분 이상을 포함하고 전체는 3분 미만으로 제작.
- 공개 코드·라이선스·실행 README, 영문 설명·영상/번역·기술 피드백을 준비.
- 공개 저장소의 단독 저작권 및 제3자 모델·코드·에셋 라이선스를 제출 구성에 맞춰 대조.
- Devpost 등록·최종 제출과 개인정보·약관 동의는 실제 수용 상태를 확인한 뒤 완료 처리.

제출 마감은 **2026-10-31 02:00 KST**이며 [공식 일정](https://nebiusglobalaihackathon.devpost.com/details/dates)은
2026-10-30 10:00 PDT로 표시한다. 현재 상태는 `selected / bounded_api_smoke_complete / credentials_revoked`이며 `ready_to_submit`이 아니다.

## 공식 기술 근거

- [Token Factory Quickstart](https://docs.tokenfactory.nebius.com/quickstart): OpenAI 호환 URL과 `NEBIUS_API_KEY`.
- [모델 카탈로그 조회](https://docs.tokenfactory.nebius.com/api-reference/examples/list-of-models): 계정별 모델 ID와 상세 속성 확인.
- [Nemotron 3 Super 제공 안내](https://nebius.com/blog/posts/nemotron3-super-now-available): 공개 모델 제공 사실. 사용자 계정의 호출 성공 증거는 아니다.
- [행사 FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs): Nebius 런타임 호출 요건과 Physical AI 시뮬레이션 허용.
