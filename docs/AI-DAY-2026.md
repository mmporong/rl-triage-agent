# AI Day 2026 고도화 계획

2026-10-08의 기술 감사와 계획이다. 제품 구현·새 평가·학습 실행 결과가 아니다. 기준 코드 HEAD는 `58bafa3c48e364f7dfeef6b77dc2dd009a3a3c32`이며 기존 결과 파일을 보존한다.

권고는 **RL 실패 설명에서 다음 실험 선택과 회복 검증까지 이어지는 시스템**으로 발전시키는 것이다. 기존 Nemotron·NeMo Agent Toolkit·Isaac Lab·OpenShell 기반을 재사용하며 이름 추가나 UI 확대를 기술 기여로 세지 않는다.

## 행사 조건

[NVIDIA AI Day Seoul](https://www.nvidia.com/ko-kr/ai-days/)은 11/9~10 코엑스 행사로 피지컬 AI·에이전틱 AI·AI 인프라를 다룬다. 사용자는 10/7 `hangul-donghaeng` 발표로 상위 5팀 선정과 11/10 발표 예정이라고 알려줬다. 사용자가 10/8에 RL 프로젝트로 변경 가능하다고 추가 확인했다. 개별 슬롯·새 심사표는 확인되지 않았다. 기존 해커톤의 배점·5분 조건을 새 행사에 자동 적용하지 않는다.

## 우선 보정할 공백

| 우선 | 현재 코드/기록의 공백 | 바꿀 파일 후보 | 완료 기준 |
|---|---|---|---|
| P0 | 공개 workspace 생성과 eval이 로컬 params/private answer key에 의존; quickstart workspace 경로 불일치 | `bench/build_workspace.py`, `evals/run_eval.py`, `README.md`, 공개 reference fixture | clean checkout에서 API 없이 replay·과거 점수 재계산 |
| P0 | `recovery_verdict`가 total reward ratio를 사용하며 HANDOFF의 가중치 불변 설명과 불일치 | `src/rl_triage/eval_bridge.py`, `docs/HANDOFF.md`, `bench/` 평가 계약 | 고정 평가 환경·독립 행동 지표·기준 seed 변동으로 회복 판정; 기존 label 재검토 |
| P0 | agent는 시계열/계산 도구, 대조군은 요약만 받아 ‘same inputs’가 부정확 | `evals/run_eval.py`, `README.md`, 새 평가 manifest | 정보·호출 예산을 맞춘 기준선과 모든 실패·재시도 기록 |
| P1 | physics/optimizer 원인 구별 미해결, 동일 10개 템플릿의 seed holdout | `bench/catalog.py`, telemetry 추출, `src/rl_triage/triage_tools.py` | 토크/포화/접촉/KL/value 등 구별 가능한 계측과 새 결함/task holdout |
| P1 | Task B 진단은 순위 저장까지, 사전등록의 예상 signature·acceptance가 실행 판정에 반영되지 않음 | `src/rl_triage/{triage_tools,eval_bridge}.py`, workflow | 가설→단일 변수 실험→승인→관측→가설 갱신·회복 결과 |
| P1 | 점수화 eval은 host NAT 실행; 기본 sandbox의 diagnoses 쓰기·자원 한도 증거 부족 | `evals/run_eval.py`, `policies/`, sandbox tests | 동일 경계 안 평가와 deny/예산/timeout/중복 실행/보존 receipt |

이 목록은 후속 구현 대상으로만 기록한다. 현재 product source와 기존 README의 과거 주장은 이번 계획 작성에서 수정하지 않았다.

## 평가 계약

- 기존 공개 seed는 앞으로 개발용이다. prompt/도구/정답 기준 변경을 평가 전에 커밋하고 새 holdout을 봉인한다.
- 기준선은 규칙, 동일 모델 요약, 동일 모델 전체 시계열, 고정 feature+모델, 전체 agent로 나눈다. 시계열·분석·새 계측 제거와 동일 호출 예산 ablation을 둔다.
- case×training seed를 실험 단위로 두고 새 강도·결함 가족·복합 원인·원인 없음·자료 부족·가능한 두 번째 task를 구분한다. 같은 입력의 agent 재호출을 독립 RL 표본으로 세지 않는다.
- 주 지표는 원인 확정까지 실험 횟수·GPU 분·잘못된 개입률·회복률·abstention, 보조 지표는 top-1/2·latency/token·사람 검토 시간이다. 주 지표와 실패/재시도 처리를 사전 고정한다.
- 총 보상과 학습 설정에 독립적인 고정 평가 조건을 사용한다. 표본 수·통계 방식은 pilot 분산과 GPU 예산을 확인한 뒤 정하며 유의성 확보를 약속하지 않는다.
- [rliable 논문](https://arxiv.org/abs/2108.13264)의 불확실성 보고, [Isaac Lab 계측](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/training_guide.html), [NAT trajectory/profiling](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/develop/docs/source/improve-workflows/evaluate.md)을 참고한다. 최신 upstream CLI·기능이 현재 설치 lock에서 지원되는지 별도 확인한다.

현재 Task B 보류는 6/10 대 2/10, top-2는 6/10 동률이며 top-1 차이는 p≈0.125다. 일반화·유의한 우월성·GPU 절감의 증거가 아니다. 두 재학습 smoke는 전체 결함의 회복 입증이 아니다.

## 범위와 철회 조건

최소 데모는 한 task에서 실패→두 경쟁 가설→권한 거절/사람 승인→한 실험→회복 또는 비회복→근거 trace를 보여준다. 학습 영상 재생은 재생으로 표시한다. 실물 로봇 제어·자동 무제한 튜닝·새 모델 훈련·다중 클러스터 운영은 기본 범위 밖이다.

[DrEureka](https://eureka-research.github.io/dr-eureka/)는 LLM 보상·domain randomization 설계, [Episode Inspector](https://github.com/VShirokun/RL-Episode-Inspector)는 replay·reward 분석 선행 대안이다. 새로운 기여는 viewer나 자동 reward 생성 자체가 아니라 실패를 가르는 실험과 검증 가능한 실행이다.

같은 정보·예산에서 규칙/feature 기준선이 동등 이상이면 agent 우월성 주장을 철회하고 결정적 진단 도구로 단순화한다. 전체 GPU/모델/사람 비용이 줄지 않으면 절감 문구를 제거한다. 새 결함을 구별하지 못하면 지원 task·결함을 한정한다. 회복 oracle 변경으로 label이 바뀌면 과거 점수를 재평가한다.

## 제안 일정과 다음 시작점

10/8~13 공개 재현·oracle → 10/14~21 계측·기준선·holdout → 10/22~29 승인 실험 loop → 10/30~11/6 비교·독립 재현·RC → 11/7 이후 실제 발표 길이에 맞춰 고정 데모·영상 대안·Q&A를 준비한다. 이는 계획이며 실측 소요 시간이 아니다.

다음 개발은 P0 하나부터 열고 새 결과를 새 tag에 저장한다. runtime·모델·정책 버전을 pin하고 어제 발표작의 OpenShell 0.0.116과 이 프로젝트의 과거 0.1.1 인계를 혼합하지 않는다. 기술 감사/후보 10개/현장 회고는 hackathon-specialist 저장소의 `reports/nvidia-ai-day-2026-rl-triage-20261008/`에 보존한다.

## 최신 논문·방법론을 반영한 보정

2026-10-08 Peer_1·Peer_2의 원문 14개 후속 조사에 따른 계획 보정이다. 제품 코드·모델·GPU 실험 결과는 추가하지 않았다. 기존 감사와 평가 기록을 보존한다.

- **신규성 주장 축소:** [EvalXRL](https://arxiv.org/html/2608.17524v1)은 가설 갱신·수리 loop와 불변 평가 함수를 제안한 가까운 선행이다. 계획 논문이며 검증된 성능 결과는 아니다. loop나 독립 평가 자체를 최초로 주장하지 않고 Isaac의 관측 제약·승인/복구·전체 비용 조건에서의 추가 가치를 검증한다.
- **반증 가능한 사전등록:** 각 가설의 예상·반증 signature, tag/시간 구간/reference SHA를 고정하고 실험 관측이 판정·갱신에 쓰이도록 연결한다. 현행 문자열 저장만으로 완성됐다고 하지 않는다. 회복은 유일 원인 확정과 구분하고, 구별 불가면 식별 불가로 종료한다.
- **blind 실험 연결 공백:** 현행 approve는 suspected_change_id를 case.overrides 위치로 대응시킨다. Task B의 mechanism 순위를 이 경로의 완성으로 간주하지 않는다. 공개 허용 실험 목록과 canonical args로 blind 가설을 연결하고, 숨은 정답·변경 목록이 선택 근거에 유출되지 않도록 별도 설계한다.
- **작은 실험 후보와 공정 대조:** [보상 진단 연구](https://arxiv.org/html/2605.28918v1)의 정적 분류표·동일 총 훈련 예산 Best-of-K·metrics-only를 추가한다. 무작위/최저 비용/전문가 고정 순서/모델 한 번 선택/적응 선택을 비교하고 계측·동적 진단·결과 갱신을 제거해 기여를 나눈다.
- **확률 미확인 시 휴리스틱:** [BoxingGym](https://arxiv.org/html/2501.01540v2)·[BED-LLM](https://arxiv.org/html/2508.21184v3)은 가설과 실험 선택의 참고다. RL 개입의 likelihood·prior·모델 endpoint 확률 지원이 미검증이므로 현재는 가설 구별 휴리스틱으로 명시한다. Bayesian/EIG 수치와 비용 절감은 검증 전 주장하지 않는다.
- **실행/보류 쌍 평가:** [AgentAbstain](https://arxiv.org/html/2607.10059v1)을 참고해 계측 누락/오래됨·승인 불일치를 최소 변화 쌍으로 평가한다. always-act/always-abstain·고정 gate와 비교하며 coverage–risk와 유효 사례 회복을 함께 본다. 자기 confidence만으로 승인하지 않는다.
- **승인 소비·복구 계약:** [CapLease](https://arxiv.org/html/2608.01710v1)는 요청 ID 중복과 의미상 같은 행동의 재발급을 구분한다. proposal digest·approval event·canonical args·effect key·예산·receipt·불명 결과 조회를 설계한다. 신뢰된 원장과 멱등 sink의 가정 없이 외부 효과 exactly-once를 보장하지 않으며 새 SDK 설치를 전제하지 않는다.
- **실제 사용 경로 검수:** [agent eval 지침](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)·[Building to the Test](https://arxiv.org/html/2606.28430v1)에 따라 결과 상태와 trace를 대조한다. oracle·승인 gate·관측 갱신을 무력화했을 때 해당 수용 검사가 실패하는지 확인한다. 테스트 개수나 파일 존재를 작업 성공으로 대신하지 않는다.

행사 전에는 한 task·구별 가능한 두 원인·승인 실험 하나·회복/비회복/보류/식별 불가·공정 replay로 범위를 고정한다. 두 번째 task, formal Bayesian 실험 설계, 범용 scientist·큰 병렬 탐색·새 모델 훈련은 행사 후 검증 범위다. 정적 결정 트리가 동등 이상이거나 전체 비용 이득이 없으면 agent 우월성·절감 문구를 철회한다.

Q/O는 수용 후 다음 작업을 공급하는 기존 계약을 유지하되 실제 inbox→ACK→검수→수용→공급과 재시작·중복·stale 결과를 검증한다. [Scaling Agent Systems](https://arxiv.org/html/2512.08296v3)가 보여주듯 병렬화 이득은 과제 구조에 달려 있다. 같은 모델·총 슬롯·예산에서 단일 O와 Q/O를 비교하고 Q 비용을 포함한다. 미실측 속도 향상을 약속하지 않는다.

전체 근거·반례·Peer 결과는 별도 저장소의 `$HOME/hackathon-specialist/reports/nvidia-ai-day-2026-rl-triage-methods-20261008/report.md`에 보존한다. 이 절은 후속 구현 입력이며 새 제품 gate를 닫은 기록이 아니다.
