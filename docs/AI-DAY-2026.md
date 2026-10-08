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
