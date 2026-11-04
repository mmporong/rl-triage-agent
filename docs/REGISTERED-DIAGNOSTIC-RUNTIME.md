# 등록된 검사로 학습 실패 조사하기

새 조사 진입점은 입력 검증, 다음 검사 제안, 승인, 실행, 중단 후 결과 접수, 내보내기를 같은 세션 원장에 남긴다. 모델은 검사 ID와 근거 fact ID를 고른다. 실행 인자와 권한은 등록된 호스트가 검증한다.

개발 대조에서는 정상·모터 함수 결함·PPO 갱신 횟수 오류·계산 오류를 Isaac 실행으로 관측했다. 모터 두 건은 부모 정규화기를 수정한 뒤 보관 원문으로 다시 검증했다. 새 원천 20학습·20고정평가 뒤 네 방식의 진단 비교 24행과 교정·평가 비교 16행을 완료했다. 진단에서 네 방식 모두 전체 원인 집합을 2/6사례에서 확인했고, 교정 비교는 미회복15행·HOLD1행이었다. [중단된 첫 비교](../evals/results/diagnostic_comparison_s2660_s2669_20261104_v3_incomplete/interpretation.md)는 별도 이력으로 보존한다. [완료한 진단 결과](../evals/results/diagnostic_comparison_s2664_s2669_20261104_v4/summary.json)의 부분 확인과 HOLD를 전체 원인 확인으로 합치지 않는다.

## 지원하는 확인 검사

| 검사 | 관측 | 확인하는 범위 |
|---|---|---|
| `P_actuator_mapping` | runtime DCMotor의 등록 속도 격자에서 토크 제한 함수 읽기 | 해당 함수·설정의 불일치. 모든 모터 고장을 제외하지 않는다. |
| `P_optimizer_count` | 짧은 PPO 실행의 실제 optimizer step·미니배치 수 | 등록한 갱신 횟수 불변식 |
| `P_optimizer_gradient` | PPO clipped-surrogate의 계산 gradient 대조 | 등록한 계산 불변식 |

`P_torque`의 이상은 증상이다. 정책이나 다른 학습 오류 때문에 토크가 달라질 수 있으므로 그 값만으로 actuator 원인을 확정하지 않는다.

`runtime_confirmation_v1`은 위 세 도구만 노출한다. 다른 원인 가설은 남겨 두며 세 검사가 모두 정상이어도 정책 전체의 정상이나 원인 부재를 확정하지 않는다. 전체 registry는 기존 검사를 포함한 별도 `full` 프로파일로 유지한다.

새 runtime 검사는 등록된 source 메커니즘을 재현한다. optimizer는 새로 초기화한 짧은 실행이고 actuator는 정책 추론 없는 함수 읽기다. 체크포인트 SHA는 원천 식별에 결합하지만 이 검사에서 그 가중치를 로드하지 않는다. 과거 체크포인트가 어떤 갱신을 거쳤는지 역으로 증명하거나 새 정책 회복을 측정한 것으로 쓰지 않는다.

## 세션 사용

저장소에서 가상환경 Python을 활성화한 뒤 도움말을 확인한다.

```powershell
cd "$HOME/rl-triage-agent"
.\.venv\Scripts\python.exe -m rl_triage.investigate_v2 --help
```

명령은 `prepare`, `next`, `approve`, `dispatch`, `settle`, `status`, `export`다. `--session`은 세션 폴더, `--input`은 해당 명령의 JSON 입력이다. 기본 backend는 `registered_diagnostic_v1`이며 임의 Python 모듈이나 셸 명령을 입력받지 않는다. `investigate_v2`는 기존 실행 함수를 재사용하면서 모델 HOLD와 아직 접수하지 못한 도구 실행 상태를 구분한다.

`prepare`는 등록된 case/reference의 고정 26조건 평가, source manifest, 기준·프로토콜 SHA, 검사 registry, 실제 artifact resolver를 재검증한다. `next`는 제안만 남기며 실행하지 않는다. `approve`는 검사·사례·원천·체크포인트·request·effect·registry에 결합된 호스트 승인 파일을 요구한다. `dispatch`는 그 승인을 한 번 소비한 뒤에만 등록된 자식 실행기로 넘어간다.

입력은 등록된 공개 export와 호스트 resolver를 사용한다. 임의 사용자 로그를 자동으로 이 스키마에 맞추는 importer는 포함하지 않는다. 실제 artifact 경로와 승인 파일은 비공개 실행 입력에 두고, 공개 등록에는 논리 ID와 SHA를 남긴다.

실행 중 중단된 요청은 재실행하지 않는다. `status`로 상태를 확인하고 `settle`에서 원래 host receipt·실행·출력 증거를 재검증하거나 unknown으로 닫는다. 모델 형식 위반, 전송 오류, 실제 실행 실패는 따로 남기며 자동으로 새 요청을 보내지 않는다.

모델과 호스트는 같은 `field_contract`의 필수 필드·최소 인용 수·허용 fact ID를 사용한다. HOLD도 근거 ID를 최소1개 포함한다. 앞선 판에서 빈 목록 예시와 호스트의 최소1개 조건이 달랐던 응답4건은 보존하며 모델의 판단 품질로 평가하지 않는다. 수정 계약은 미실행 사례에만 적용한다.

CPU fixture backend는 `--backend cpu_fixture_v1 --fixture`를 함께 지정한 개발 검사에만 쓴다. fixture의 결과와 0초 GPU 비용을 실제 실행으로 내보내지 않는다.

## 확인한 원인에서 교정으로 이어가기

같은 진입점의 `recovery-*` 명령으로 교정 요청과 후속 평가를 관리한다. `recovery-prepare`의 `--session`은 완료한 조사 폴더다. 입력 JSON에는 새 교정 폴더를 가리키는 `folder`와 등록된 `recovery_payload`를 둔다. 이후 명령의 `--session`은 그 교정 폴더다.

| 명령 | 하는 일 |
|---|---|
| `recovery-prepare` | 확정 검사 원문, 원천·체크포인트, 교정 등록, 실행 프로토콜을 재검증한다. |
| `recovery-propose` | 등록 교정과 회복 방식으로 요청을 남긴다. |
| `recovery-approve` | request·effect·사례·체크포인트·교정·예산에 결합된 별도 승인 파일을 검증한다. |
| `recovery-dispatch` | 승인을 한 번 소비한 뒤 학습과 고정 평가를 실행한다. |
| `recovery-settle` | 중단된 실행의 원래 결과를 접수하거나 unknown으로 닫는다. 학습을 다시 시작하지 않는다. |
| `recovery-status` | 승인 대기, 실행 중, 완료 상태를 조회한다. |
| `recovery-post` | 새 체크포인트와 실제 고정 평가 수치, 교정 기록, 비용을 내보낸다. |
| `recovery-review` | 등록된 원천·교정 후 근거·호스트 정책을 다시 확인하고 다음 판단을 제안한다. 새 모델·GPU 실행이나 승인을 만들지 않는다. |

새 실행 경로는 모터 함수 교정 후 이어 학습, PPO epoch 복원 후 재학습, PPO surrogate 계산 복원 후 재학습을 지원한다. 기존 교정 표는 보존하고 optimizer 두 항을 새 등록 표에 추가했다. 이번 네 방식 비교에서 실행한 교정은 모터와 epoch 복원이다. 계산 복원 경로의 구현을 이번 비교의 실측으로 쓰지 않는다.

epoch 교정의 새 실행기는 원천의 등록 보상·모터 구성을 보존한다. 복합 결함에서 epoch를 고치며 보상 함수까지 함께 기본값으로 바꾸지 않도록 원천 설정·SDK 코드 계약·새 runtime 지문을 결합한다. 과거 학습의 runtime 지문은 확보되지 않았으므로, 등록 구성의 출처와 새 학습에서 확인한 구현을 구분한다.

교정 후 수치는 새 체크포인트의 평가 원문에서 읽는다. 교정 전 검사 기록을 새 정책의 확인 검사로 재사용하지 않는다. 교정 후 원인 확인 기록이 없으면 unknown을 유지하며 추가 교정을 승인하지 않는다.

## 개입을 막아야 하는 경우

정상 여부는 caller의 `healthy=true` 플래그로 정하지 않는다. 같은 등록 기준으로 case/reference 평가 원문을 검증하고 정상 행동 묶음이 통과하면 호스트가 조사를 막는다. 이는 모델의 정상 식별 성공과 구분한다.

원천과 교정 후 평가에는 기존 `walks()`의 절대 추종·낙상 조건도 적용한다. 기준 정책이 이 조건을 통과하지 못하면 `reference_unqualified`로 남긴다. 두 정지 정책의 상대 오차 비율이 1이라는 이유로 정상이나 회복을 판정하지 않는다. 등록된 collection의 평가를 사용하며 과거 collection의 같은 case ID로 대신하지 않는다.

참조 누락·불일치, unknown 검사, 원인 미확정은 HOLD다. 원인 receipt가 확인돼도 `proposal_eligible`일 뿐 실행 승인을 만들지 않는다. 교정 뒤 확인 검사가 정상인데 행동 기준만 미달하는 분기는 등록된 참조 점검 상태로 넘긴다. 필요한 교정 후 수치가 export에 없으면 결과 라벨에서 수치를 만들어 채우지 않고 unknown을 유지한다.

새 체크포인트의 행동 평가와 교정 적용 기록만으로 원인 확인 검사를 통과했다고 쓰지 않는다. 실제 후속 확인 계약이 없는 입력은 unknown/HOLD를 유지한다. CPU fixture의 확인 영수증은 실제 교정 후 확인으로 받지 않는다.

optimizer 교정의 확인은 학습 첫 3갱신 기록에서 다시 계산한다. 일부 구성요소만 확인했다면 참조 점검을 먼저 제안하되 미관측 검사도 남긴다. 이를 전체 원인이 배제됐다는 뜻으로 쓰지 않는다. 실제 모터 mapping 후속 관측이 없는 경우는 readback 정상만으로 확인을 통과시키지 않는다.

## 에이전트 사용 비용 비교

| 방식 | 다음 검사 선택 | 모델 호출 |
|---|---|---|
| `fixed_sequence` | 등록 순서 | 0 |
| `static_rule` | 같은 검사 기대값과 남은 가설을 쓰는 명시 규칙 | 0 |
| `one_shot_llm` | 최초 입력에서 순서를 한 번 고정 | 조사 필요 시 최대 1 |
| `iterative_agent` | 완료된 관측을 받아 다음 검사 선택 | 등록된 한도 이내 |

네 방식은 같은 원천·참조·사실·도구·완료 기준·상한에서 비교한다. 정상 호스트 차단은 모두에 적용하고, API 응답을 받지 않은 방식을 모델 정상 식별로 세지 않는다. 작은 등록 카탈로그에서 명시 규칙이 더 빠르거나 결과가 같을 가능성을 포함한다.

시간은 공통 학습 자료 확보, 출처 검증, 대기, API, SDK 시작, 검사, 실패·재개, 전체 경과로 나눈다. 실제 진단에 쓴 비용과 교정·회복 확인 비용도 분리한다. 실패와 HOLD를 포함한 전체 비용을 먼저 보고하고, 같은 성공 상태끼리의 비용은 보조 표로 제시한다.

학습·고정 평가의 Simulator 실행과 CPU 체크포인트 검사·학습 로그 추출은 따로 센다. 전체 경과 시간에는 준비와 대기도 포함된다. 세부 구간의 합이 전체 경과 시간과 같다고 가정하지 않는다. 같은 원천 확보 비용이나 같은 실행 영수증을 방식별로 중복 합산하지 않는다.

교정 dispatch는 공용 lease 입장과 sampler 시작 뒤 원장을 소비하고, 실행 직전 같은 lease인지 다시 확인한다. 종료 시 실패를 포함한 자식 실행과 장치 표본을 봉인한다. 과거 실행의 정산은 당시 입장 기록으로 검증한다. 측정하지 않은 lease 대기는 0으로 채우지 않는다.

교정 controller의 단계별 시작·종료는 별도 기록한다. 측정한 단계의 경과 합과 첫 시작부터 마지막 종료까지의 구간을 구분하고, 중간 대기를 지우지 않는다. 종료 기록이 없거나 시계가 일치하지 않는 구간은 unknown이다. 이 시간에 dispatch 경과를 다시 더하지 않는다.

모델 원장은 예약, 전송 결과 반환, 확인된 HTTP 응답, 토큰 사용량을 구분한다. 요청한 모델명과 서버가 반환한 모델명도 보존한다. 응답이 남지 않은 예약의 소비량은 미확정으로 두고 재호출하지 않는다. 공개 집계기는 실패·HOLD·미실행을 분모에서 빼지 않으며, 진단 성공을 회복 성공으로 바꾸지 않는다.

형식 무효나 전송 실패 때문에 모델이 HOLD한 경우를 정상 판별 성공으로 세지 않는다. 정상 호스트 차단, 유효한 모델 HOLD, 근거가 부족한 HOLD를 구분한다. 무결성 오류로 멈춘 비교 CLI는 비영 종료 코드를 반환하고, 실패한 자식의 실행·비용·장치 표본도 보존한다.

[공통 원천 비용](../evals/results/source_common_cost_s2660_s2669_20261104_v1/summary.json)은 학습20회·고정평가20회에 대한 별도 집계다. 부모 구간11,065.078초와 소유 Simulator wall 합10,691.327초는 더하지 않는다. 장치 전체 에너지228.9201Wh는 다른 부하를 포함하며 방식·프로세스별로 귀속하지 않는다. CPU 텔레메트리 추출20회의 경과 시간이 원문에 없어 null로 남겼다. 이 공통 비용을 네 방식마다 반복 가산하지 않는다.

중단된 v3의 API4회는 입력2,614·출력128토큰이었다. [NVIDIA의 usage 지표](https://docs.nvidia.com/aiperf/dev/reference/ai-perf-metrics-reference)는 API가 보고한 토큰 수이며 청구 영수증이나 provider GPU 시간이 아니다. [공식 모델 페이지](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b?section=deploy)의 무료 endpoint 안내와 [체험 Credits 규정](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf)은 계정별 실청구0을 증명하지 않는다. 확인한 공식 페이지에서 이 endpoint의 토큰당 단가를 확보하지 못해 달러 환산·실청구·provider GPU 시간은 null로 둔다.

`gpu_s` 원장의 새 runtime 값은 lease를 보유한 owned 자식의 경과 시간이다. GPU kernel이 실제로 작동한 시간과 같지 않다. 별도 sampler는 장치 전체 사용률·메모리·전력을 기록하고, 전력 적분은 표본 coverage·누락 구간과 함께 장치 전체 추정치로 표시한다. 호스팅 모델의 GPU 시간, 사람의 검토 시간, 실제 청구를 확보하지 못하면 null로 남긴다. 모델 무호출의 요청·토큰 0과 API의 실제 청구 미확인은 다른 값이다.

평가 설계는 [AI Agents That Matter](https://arxiv.org/abs/2407.01502), seed·구현 변동은 [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), 작은 표본 해석은 [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)를 참고한다. GPU 필드의 의미와 지원 범위는 [NVIDIA SMI 문서](https://docs.nvidia.com/deploy/nvidia-smi/)를 따른다. 논문의 결과를 이 프로젝트의 측정 효과로 옮기지 않는다.
