# Claude 인계: RL Triage AI Day 고도화

이 문서는 2026-10-08 조사와 기술 감사를 개발 작업으로 이어가는 시작점이다. 현재까지 완료된 것은 연구·회고·운영 스킬·계획 문서이며, 아래 제품 고도화는 미구현이다.

## 1. 목표와 행사 맥락

사용자는 10/7 해커톤에서 [한글동행](https://github.com/mmporong/hangul-donghaeng)으로 상위 5팀에 선정됐고 11/10 NVIDIA AI Day 발표 예정이라고 전달했다. RL 프로젝트로 발표작을 바꿀 수 있다는 사용자 확인도 받았다. 공개 행사 안내는 [11/9~10 코엑스](https://www.nvidia.com/ko-kr/ai-days/)이며 개별 발표 길이·새 심사표는 미확인이다. 이전 해커톤의 5분·40/20/20/10/10 배점을 승계하지 않는다.

개발 대상은 이 저장소의 RL Triage다. 권고 목표는 **Isaac Lab의 실패 가설을 가르는 다음 실험 하나를 선택하고, 사람 승인 후 관측으로 판단을 갱신하며, 회복 여부를 독립 지표로 검증하는 서비스**다. 일반 챗봇·viewer·도구 이름 추가만으로 기술 기여를 주장하지 않는다.

행사 전 범위는 한 Isaac task·구별 가능한 두 원인·승인 실험 하나·회복/비회복/보류/식별 불가·재현 가능한 trace다. 범용 AI scientist·새 모델 훈련·큰 병렬 tree·자유로운 보상 코드 수정·실물 로봇·다중 클러스터는 기본 범위 밖이다.

## 2. 읽을 순서와 정본

1. 저장소 루트의 [AGENTS.md](../AGENTS.md)를 따른다. 해당 지침에 따라 [과거 HANDOFF](HANDOFF.md)를 먼저 읽는다.
2. [AI Day 계획](AI-DAY-2026.md)과 이 문서를 읽고 현재 구현·미구현을 구분한다.
3. 첫 작업에 필요한 소스와 결과 파일만 읽는다. 이번 인계를 이유로 전체 논문 조사·아이디어 10개 생성을 반복하지 않는다.

코드 감사 기준은 `58bafa3c48e364f7dfeef6b77dc2dd009a3a3c32`다. 계획 커밋 `a3c6a89`·`882c4f9`는 문서 변경이며 제품 코드는 그대로다. 이후 구현 시 현재 HEAD·diff·환경 버전을 새로 기록한다.

상세 연구·현장 회고는 비공개 `mmporong/hackathon-specialist` 저장소에 보존한다. 그 저장소의 접근권한이 없어도 이 문서와 공개 소스로 첫 P0를 진행할 수 있다. 비공개 현장 원문·키·계정 설정을 공개 저장소로 복사하지 않는다.

## 3. 현재 확인된 사실과 주장 제한

| 사실 | 코드·기록 | 의미 |
|---|---|---|
| Task B 과거 top-1은 6/10 대 2/10, top-2는 6/10 동률, p≈0.125 | 기존 README·heldout 결과 | 유의한 우월성·일반화·비용 절감 증거 아님 |
| agent는 전체 시계열·계산 도구, 기존 대조군은 요약 입력 | `evals/run_eval.py` | 정보가 일치한 비교가 아니며 새 공정 기준선 필요 |
| workspace 생성·평가가 로컬 params/private 자료에 의존 | `bench/build_workspace.py`, `evals/run_eval.py` | clean checkout offline replay를 우선 완성 |
| 회복 판정은 학습 보상 비율을 포함 | `src/rl_triage/eval_bridge.py:28,45` | 과거 HANDOFF의 가중치 불변 설명은 현행 코드와 다름 |
| 예상 signature·acceptance gate는 저장만 하고 실행 후 판정에 미연결 | `src/rl_triage/triage_tools.py:171`, `eval_bridge.py:82` | 관측 후 가설 갱신 loop 미완성 |
| blind Task B는 mechanism 순위를 저장; approve는 change ID→case.overrides 대응 | `triage_tools.py:199`, `eval_bridge.py:87` | 공개 허용 실험 목록/canonical args 연결 필요; 숨은 정답 유출 금지 |
| submit마다 새 UUID; 파일 상태 확인 뒤 실행 | `eval_bridge.py:61,82` | 의미상 동일 실험의 승인 소비·중복 효과·crash 복구 보장 미확인 |
| 예전 scored eval은 host NAT 실행 | 기존 평가 코드·감사 | 전체 평가가 sandbox 내부에서 수행됐다고 확대하지 않음 |

동시 승인·중복 효과·재시작 결함을 이번 조사에서 실행해 재현하지 않았다. 코드 구조에서 위험을 추론한 것이다. 기존 두 재학습 smoke를 전체 결함의 회복으로 확대하지 않는다. 운영 planner의 과거 32 tests는 RL 제품 테스트 수가 아니다.

## 4. 첫 작업: API 없이 공개 replay부터

첫 구현 단위는 **P0-A: 공개 clean checkout에서 저장된 telemetry를 읽고 과거 점수를 재계산하는 offline replay**로 고정한다. 새 모델·Isaac 학습·sandbox 생성이 필요 없는 범위부터 끝낸다.

- 입력: 공개된 telemetry·reference·결과 파일과 공개 재배포가 가능한 최소 fixture. 로컬 private 파일·숨은 정답·개인 params를 실행 전제로 두지 않는다.
- 출력: 실행 가능한 진입점, 입력/코드 버전과 실제 SHA, 계산 결과, 재현 명령, 실패 이유. fixture/replay임을 표시한다.
- 작업 후보: `bench/build_workspace.py`, `evals/run_eval.py`, 공개 fixture/평가 계약, 관련 테스트, 대표 README. 실제 확인 뒤 필요한 파일만 수정한다.
- 모델에 전달되는 blind 입력과 채점 정답을 분리한다. 공개된 정답을 evaluator가 읽는 것과 agent 입력에 넣는 것은 별개다.
- 수용: 깨끗한 checkout에서 credential·네트워크·GPU 없이 같은 저장 결과를 재계산하고, 누락 파일·잘못된 입력·정답 유출 경계를 검증한다.
- 현재 존재하지 않는 offline CLI flag를 문서에 먼저 만들지 않는다. 소스를 확인하고 구현한 실제 명령만 README에 적는다.
- 작성→관련 검증→실패 수정→재검증→별도 검수→경로별 로컬 커밋까지 진행한다. 환경 blocker가 나와도 실행 가능한 로컬 작업을 계속한다.

기존 코드의 읽기 전용 시작 예시다. 경로는 실제 clone 위치에 맞춘다.

```bash
cd "$HOME/rl-triage-agent"
pwd
git status --short --branch
git log -3 --oneline
git diff --stat
```

저장소가 없다면 공개 clone부터 시작한다.

```bash
cd "$HOME"
git clone https://github.com/mmporong/rl-triage-agent.git
cd "$HOME/rl-triage-agent"
```

## 5. 다음 작업과 의존성

| 순서 | 작업 | 수용 기준 |
|---|---|---|
| P0-B | 학습 보상과 분리한 고정 행동 회복 oracle | 고정 평가 조건·reference 변동·관측 지표로 재계산; 바뀐 과거 label 기록 |
| P0-C | 정보·총예산을 맞춘 기준선과 새 holdout | 규칙/동일 모델 전체 시계열/고정 feature/전체 agent 비교; 실패·탈락 후보·재시도 비용 포함 |
| P1-A | 경쟁 가설과 유한 실험 선택 | 예상·반증 signature·tag/시간 구간/reference SHA를 사전 고정하고 실제 관측으로 갱신 |
| P1-B | 승인 소비·실행 receipt·불명 결과 복구 | proposal digest·approval event·canonical args·effect key·예산·checkpoint 연결; timeout을 실패로 단정하지 않음 |
| P1-C | 최소 변화 쌍·반례·실제 사용 경로 검수 | 정상/누락/오래된 관측/승인 불일치, 회복/비회복/보류/식별 불가를 구분 |
| RC | 독립 재현·고정 데모·영상 대안·Q&A | 실제 승인된 환경의 실행 근거와 replay를 구분; 제품 gate 미완료를 연구 승인으로 닫지 않음 |

P0-A는 다른 P0의 완성을 주장하지 않는다. 프롬프트·도구·판정 기준을 평가 전에 커밋하고 새 holdout/new tag를 사용한다. 이미 본 seed를 최종 미관측 평가로 재사용하지 않는다. 과거 `evals/results/**`는 덮어쓰지 않는다.

## 6. 최신 연구를 적용할 방법

- **신규성:** EvalXRL의 loop·불변 평가 함수도 선행이다. 이번 프로젝트의 기여는 실제 관측·승인·전체 비용 조건에서 실험 선택의 추가 가치를 입증하는 후보이며 신규성 확정이 아니다.
- **실험 선택:** 무작위/최저 비용/전문가 고정 순서/모델 한 번 선택/적응 선택을 비교한다. 정적 분류표·metrics-only·같은 총 훈련 예산 Best-of-K를 빼지 않는다. 비용·관측 갱신·새 계측·동적 진단을 제거해 기여를 나눈다.
- **Bayesian/EIG:** RL 개입 likelihood·prior·endpoint 확률 지원이 미검증이므로 지금은 가설 구별 휴리스틱으로 명시한다. 정밀 posterior/EIG와 보장된 위험을 만들지 않는다.
- **보류 판단:** always-act/always-abstain·고정 gate와 비교하고 coverage–risk와 유효 사례 회복을 함께 본다. 자기 confidence는 승인 권한이 아니다.
- **승인/복구:** 요청 ID 중복 제거와 의미상 같은 행동의 승인 소비를 분리한다. 멱등 sink나 이에 준하는 효과 조정 없이 외부 효과 exactly-once를 보장하지 않는다.
- **검수:** 결과 상태·trace·실제 소비 경로를 확인한다. oracle·승인 gate·관측 갱신을 무력화했을 때 해당 수용 검사가 실패해야 한다. 작성자와 검수자의 이름 차이만으로 오류 독립성을 주장하지 않는다.

실험 단위는 case×training seed다. 모델 재호출을 독립 RL 표본으로 세지 않는다. 주 지표는 회복·잘못된 개입·식별까지 실험 수/GPU 분이며 모델 토큰·분석 CPU·사람 검토·총지연도 기록한다. 표본수는 pilot 분산과 예산 확인 뒤 정한다.

정적 방식이 동등 이상이면 결정적 진단 도구로 단순화한다. 총비용이 줄지 않으면 절감 문구를 제거한다. 허용 관측·개입으로 구별할 수 없으면 식별 불가로 끝낸다. 회복만으로 유일 원인을 확정하지 않는다.

## 7. 질문 세션과 오케스트레이터

사용자가 별도 세션 운영을 요청하면 **Q는 질문·읽기 전용 snapshot+decision inbox, O는 단일 원장 작성·수용·배분·통합**을 맡긴다. 질문만 들어오면 진행 중 작업을 멈추지 않는다. 변경/stop은 event ID·원본 revision을 보존하고 O가 최신 상태에서 적용/ACK한다.

작업마다 목표·입력 SHA·read/write 소유 경로·의존성·수용 기준·실제 runtime 대상·attempt를 기록한다. `ready_for_review≠accepted`이며 O는 검수 수용 뒤 의존 작업을 열고 빈 실제 슬롯에 준비 작업을 공급한다. 완료 알림 전달만으로 끝내지 않고 결과 파일·근거·검수를 회수한다.

실측 runtime 한도를 따르고 Q/O/검수가 같은 pool이면 모두 포함한다. 이전 Codex surface는 root 포함 4슬롯이었지만 Claude의 현재 한도는 새로 확인한다. Codex native API·OMX Team이 Claude에서 지원된다고 가정하지 않고 실제 설치된 도구·스킬·메시지 표면을 쓴다. 세션/스킬/모델 설정을 임의로 교체하지 않는다.

ACK 유실·중복/순서 역전·stale 입력·O 재시작/이전 owner 잔존·취소 후 늦은 완료·파일 충돌·검수 기아·수용 뒤 후속 공급을 연습한다. 원장 파일·planner만으로 자동 wake를 주장하지 않는다. 진행 증거가 있는 작업을 polling timeout만으로 중단하지 않는다.

같은 과제·모델·총 슬롯·예산·질문 시나리오에서 단일 O와 Q/O를 비교하고 Q 비용도 센다. ready→start·artifact_ready→accepted·accepted→dispatch·결정 지연·stale 답변·재작업·총 완료 시간/비용을 기록한다. 이득이 없으면 상시 Q 대신 필요할 때 snapshot을 조회한다.

## 8. 환경·권한·보존

과거 RL 환경은 OpenShell 0.1.1·NAT 1.9.0·Isaac Lab 2.1.1/Sim 4.5였고, 한글동행의 OpenShell 0.0.116과 다르다. 과거 HANDOFF의 WSL 경로·sandbox Error·API 설정·학습 소요를 현재 환경의 사실로 추정하지 않는다. 설치 lock·현재 실행 경로를 확인한 뒤 버전별 지원 절차를 쓴다.

안전한 로컬 구현·테스트·수정·재검증·커밋은 질문으로 중단하지 않는다. 실제 NVIDIA/Isaac 호출·원격 서비스 조작·비용 발생 행동은 현재 세션의 사용자 승인 범위를 확인한다. 키 값을 화면·로그·브라우저·저장소로 노출하지 않고 계정/키 파일 탐색으로 blocker를 우회하지 않는다. 기존 서비스·gateway·다른 작업자의 변경을 보존한다.

이 인계 작성에서는 제품 코드·모델 호출·학습·원격 실행·sandbox 생성·새 테스트를 수행하지 않았다. 새 기능·성능·운영 속도 PASS를 선언하지 않는다.

## 9. 원문 14개와 상세 기록

2026-10-08 Peer_1·Peer_2가 원문을 확인하고 root가 결과를 회수·주요 내용을 대조했다. 여러 자료는 preprint이며 공개를 동료 심사 완료로 간주하지 않는다. 출처의 성능을 우리 제품 결과로 옮기지 않는다.

| 원문 | 적용·한계 |
|---|---|
| [EvalXRL v1](https://arxiv.org/html/2608.17524v1) | 계획 논문; loop·불변 평가도 선행 |
| [Diagnostic-Driven Reward Refinement v1](https://arxiv.org/html/2605.28918v1) | 정적 진단·예산 기준선; 희소 구조화 중심, 보행 일반화 제한 |
| [AI Scientist-v2 v1](https://arxiv.org/html/2504.08066v1) | 단계별 중단·반복 검증 참고; 범용 scientist 확대 보류 |
| [BoxingGym v2](https://arxiv.org/html/2501.01540v2) | 실험 선택/모델 발견 분리; 알려진 생성모델·자원 제약 미반영 |
| [BED-LLM v3](https://arxiv.org/html/2508.21184v3) | 가설·관측 갱신; 문답 likelihood를 RL로 전용하지 않음 |
| [AgentRx v2](https://arxiv.org/html/2602.02475v2) | 도구 agent trace 귀속; RL 물리 인과 확정과 구분 |
| [AgentAbstain v1](https://arxiv.org/html/2607.10059v1) | 실행/보류 쌍; 위험 보정 보장 아님 |
| [MAST v3](https://arxiv.org/html/2503.13657v3) | 설계·조정·검증 실패 분류; 우리 Q/O 인과 효과 미검증 |
| [Scaling Agent Systems v3](https://arxiv.org/html/2512.08296v3) | 과제별 병렬화 손익; 구판 180구성/4벤치마크와 혼합 금지 |
| [Anthropic agent evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | 결과·trace·시행 격리·회귀 평가; 공식 엔지니어링 경험 |
| [CapLease v1](https://arxiv.org/html/2608.01710v1) | 승인 소비·정규화·멱등 sink 가정; 새 SDK 채택 결정 아님 |
| [Agentic Verifier v1](https://arxiv.org/html/2602.04254v1) | 후보 차이를 드러내는 입력; 코딩 결과를 RL 성능으로 전용 금지 |
| [Spec Kit Agents v1](https://arxiv.org/html/2604.05278v1) | 저장소 근거·검증의 품질/시간 교환; 모든 작업에 전체 SDD 강제하지 않음 |
| [Building to the Test v1](https://arxiv.org/html/2606.28430v1) | 실제 사용 경로·무력화 검사; 한 과제의 존재 사례 |

권한이 있으면 비공개 저장소의 다음 경로를 추가로 읽는다. 원문 캡처 SHA는 normalized acquisition 파일의 해시이며 전체 논문/PDF 해시가 아니다.

- [기존 기술 감사·후보 검토](https://github.com/mmporong/hackathon-specialist/blob/main/reports/nvidia-ai-day-2026-rl-triage-20261008/report.md)
- [10/7 병목 회고](https://github.com/mmporong/hackathon-specialist/blob/main/reports/nvidia-ai-day-2026-rl-triage-20261008/postmortem.md)
- [최신 논문·방법론 대조](https://github.com/mmporong/hackathon-specialist/blob/main/reports/nvidia-ai-day-2026-rl-triage-methods-20261008/report.md)
- [Q/O 운영 계약](https://github.com/mmporong/hackathon-specialist/blob/main/skills/hackathon-specialist/references/session-orchestration.md)
- [읽기 전용 배분 planner](https://github.com/mmporong/hackathon-specialist/blob/main/skills/hackathon-specialist/scripts/field_queue.py)

## 10. Claude에 전달할 시작 요청

> 이 저장소의 AGENTS.md에 따라 docs/HANDOFF.md를 먼저 읽고, docs/AI-DAY-2026.md와 docs/CLAUDE-HANDOFF.md를 읽어 현재 상태를 확인해 줘. 첫 작업은 P0-A 공개 offline replay다. 현재 cwd·git status와 필요한 코드만 확인하고, API·GPU·네트워크 없이 재현할 입력/출력·수용 기준을 고정한 뒤 로컬 구현·검증·수정·별도 검수·커밋까지 진행해 줘. 기존 결과·다른 변경은 보존하고, 다음 P0의 완료나 제품 성능을 미리 주장하지 마. 실제 외부 실행이 필요한 단계는 승인 범위와 구체 blocker를 구분해 남겨 줘.
