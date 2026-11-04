# 등록된 검사로 학습 실패 조사하기

새 조사 진입점은 입력 검증, 다음 검사 제안, 승인, 실행, 중단 후 결과 접수, 내보내기를 같은 세션 원장에 남긴다. 모델은 검사 ID와 근거 fact ID를 고른다. 실행 인자와 권한은 등록된 호스트가 검증한다.

현재 관련 CPU 회귀와 실제 공개 자료의 재검증은 통과했다. 새 실행 경로의 Isaac 측정과 네 방식의 비교는 사전등록 준비 중이다. 아래 기능 설명을 새 GPU 성과나 비용 절감 결과로 해석하지 않는다.

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
.\.venv\Scripts\python.exe -m rl_triage.investigate --help
```

명령은 `prepare`, `next`, `approve`, `dispatch`, `settle`, `status`, `export`다. `--session`은 세션 폴더, `--input`은 해당 명령의 JSON 입력이다. 기본 backend는 `registered_diagnostic_v1`이며 임의 Python 모듈이나 셸 명령을 입력받지 않는다.

`prepare`는 등록된 case/reference의 고정 26조건 평가, source manifest, 기준·프로토콜 SHA, 검사 registry, 실제 artifact resolver를 재검증한다. `next`는 제안만 남기며 실행하지 않는다. `approve`는 검사·사례·원천·체크포인트·request·effect·registry에 결합된 호스트 승인 파일을 요구한다. `dispatch`는 그 승인을 한 번 소비한 뒤에만 등록된 자식 실행기로 넘어간다.

입력은 등록된 공개 export와 호스트 resolver를 사용한다. 임의 사용자 로그를 자동으로 이 스키마에 맞추는 importer는 포함하지 않는다. 실제 artifact 경로와 승인 파일은 비공개 실행 입력에 두고, 공개 등록에는 논리 ID와 SHA를 남긴다.

실행 중 중단된 요청은 재실행하지 않는다. `status`로 상태를 확인하고 `settle`에서 원래 host receipt·실행·출력 증거를 재검증하거나 unknown으로 닫는다. 모델 형식 위반, 전송 오류, 실제 실행 실패는 따로 남기며 자동으로 새 요청을 보내지 않는다.

CPU fixture backend는 `--backend cpu_fixture_v1 --fixture`를 함께 지정한 개발 검사에만 쓴다. fixture의 결과와 0초 GPU 비용을 실제 실행으로 내보내지 않는다.

## 개입을 막아야 하는 경우

정상 여부는 caller의 `healthy=true` 플래그로 정하지 않는다. 같은 등록 기준으로 case/reference 평가 원문을 검증하고 정상 행동 묶음이 통과하면 호스트가 조사를 막는다. 이는 모델의 정상 식별 성공과 구분한다.

참조 누락·불일치, unknown 검사, 원인 미확정은 HOLD다. 원인 receipt가 확인돼도 `proposal_eligible`일 뿐 실행 승인을 만들지 않는다. 교정 뒤 확인 검사가 정상인데 행동 기준만 미달하는 분기는 등록된 참조 점검 상태로 넘긴다. 필요한 교정 후 수치가 export에 없으면 결과 라벨에서 수치를 만들어 채우지 않고 unknown을 유지한다.

## 에이전트 사용 비용 비교

| 방식 | 다음 검사 선택 | 모델 호출 |
|---|---|---|
| `fixed_sequence` | 등록 순서 | 0 |
| `static_rule` | 같은 검사 기대값과 남은 가설을 쓰는 명시 규칙 | 0 |
| `one_shot_llm` | 최초 입력에서 순서를 한 번 고정 | 조사 필요 시 최대 1 |
| `iterative_agent` | 완료된 관측을 받아 다음 검사 선택 | 등록된 한도 이내 |

네 방식은 같은 원천·참조·사실·도구·완료 기준·상한에서 비교한다. 정상 호스트 차단은 모두에 적용하고, API 응답을 받지 않은 방식을 모델 정상 식별로 세지 않는다. 작은 등록 카탈로그에서 명시 규칙이 더 빠르거나 결과가 같을 가능성을 포함한다.

시간은 공통 학습 자료 확보, 출처 검증, 대기, API, SDK 시작, 검사, 실패·재개, 전체 경과로 나눈다. 실제 진단에 쓴 비용과 교정·회복 확인 비용도 분리한다. 실패와 HOLD를 포함한 전체 비용을 먼저 보고하고, 같은 성공 상태끼리의 비용은 보조 표로 제시한다.

`gpu_s` 원장의 새 runtime 값은 lease를 보유한 owned 자식의 경과 시간이다. GPU kernel이 실제로 작동한 시간과 같지 않다. 별도 sampler는 장치 전체 사용률·메모리·전력을 기록하고, 전력 적분은 표본 coverage·누락 구간과 함께 장치 전체 추정치로 표시한다. 호스팅 모델의 GPU 시간, 사람의 검토 시간, 실제 청구를 확보하지 못하면 null로 남긴다. 모델 무호출의 요청·토큰 0과 API의 실제 청구 미확인은 다른 값이다.

평가 설계는 [AI Agents That Matter](https://arxiv.org/abs/2407.01502), seed·구현 변동은 [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), 작은 표본 해석은 [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)를 참고한다. GPU 필드의 의미와 지원 범위는 [NVIDIA SMI 문서](https://docs.nvidia.com/deploy/nvidia-smi/)를 따른다. 논문의 결과를 이 프로젝트의 측정 효과로 옮기지 않는다.
