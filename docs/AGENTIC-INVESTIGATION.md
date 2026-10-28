# 다단계 조사 행동 계약

`evals/agentic_investigation_contract_v1.py`와 `evals/agentic_investigation_loop_v1.py`는 기존 검사·교정 controller 위에 자료 조회와 다음 행동 제안을 연결한다. 새 증거 어댑터와 호출기는 실제 원본·프레임 검증, API 예약, 완료 응답 재생을 연결한다. GPU 검사·교정 backend까지 이어지는 새 고리의 종단 실행은 아직 준비 중이다. 기존 [3.5 종단 실행](VISION-TRANSFERRED-E2E-DEVELOPMENT.md)의 첫 검사 선택과 후속 관찰 결과를 이 고리의 실제 실행 성과로 합산하지 않는다.

## 행동과 실행 경계

모델 출력은 `action`, `target`, `fact_ids` 세 필드다.

| 행동 | 허용되는 대상 | 호스트 확인 |
|---|---|---|
| `read_evidence` | 등록된 학습 자료 ID | 제공 자료의 출처와 공개 fact 형식 |
| `inspect_frames` | 등록된 영상 자료 ID | 원본 frame ID·시각, 누락·오결합 여부 |
| `run_probe` | 현재 후보를 가르거나 단일 후보를 확인하는 등록 검사 | 기존 controller의 정해진 인자·승인·일회 소비·raw 결과 |
| `propose_correction` | 현재 확인된 성분의 등록 교정 | 실제 이상 검사와 교정 승인·원장·고정 평가 |
| `hold` | 대상 없음 | 이후 요청을 만들지 않음 |

행동 선택에는 실행 승인이나 임의 인자를 넣을 수 없다. 무효 응답은 HOLD로 끝나며 규칙 선택으로 바꾸지 않는다. 호스트의 원인·회복 판정은 기존 controller를 재사용한다. 이 판의 교정 제안은 기존 controller에 등록된 보상 좌표계·PPO epoch 교정 범위다. 다른 범주의 검사에서 원인이 확인돼도 교정이 등록되지 않았다면 교정 후보를 제공하지 않는다.

자료 조회는 공개 사실을 추가할 수 있지만 기존 사실을 덮어쓸 수 없다. 내부 경로·정답 필드와 비유한 수치는 거부한다. 실제 artifact 검증과 모델에 제공할 자료의 선택은 호출자가 기존 입력 수용 경계를 통해 수행해야 한다. 이 CPU 어댑터만으로 원본 자료가 인증됐다고 판단하지 않는다.

## 결과를 받아 이어가는 방식

1. 기존 source/계보에 결합된 snapshot으로 조사 context를 만든다.
2. 모델이 공개 사실을 인용해 다음 행동을 제안한다.
3. 호스트는 그 행동의 대상·현재 상태·예산을 검증한다. 미완료 행동이 있으면 새 행동을 받지 않는다.
4. 실제 검사 seal의 원문 SHA·승인·소비·측정 결과를 기존 controller로 다시 검증하고 다음 snapshot에 연결한다.
5. 교정 후에는 원본·참조 checkpoint, 교정 원장과 고정 평가를 검증한다. 회복이면 종료한다.
6. 첫 교정의 검증된 미회복 receipt가 있으면 두 번째 epoch로 간다. 이전 checkpoint의 영상 자료 목록은 비우고 새 계보의 자료를 별도로 수용해야 한다. 서로 다른 교정 최대 두 개와 기존 receipt 경계는 유지한다.

`Investigation` 객체는 한 프로세스의 상태 어댑터다. `agentic_investigation_evidence_v1.py`는 등록된 같은 checkpoint의 학습 기록·원본 프레임만 조회하고, 조사용 system prompt 한 개를 유지한다. `agentic_investigation_call_v1.py`는 각 요청의 공개 메시지와 history를 동결해 기존 `model_decision_journal_v1.py`로 예약·응답·재생을 관리한다. 의미 무효는 재요청 없이 HOLD이며, 완료 여부를 모르는 요청은 pending으로 남긴다.

`agentic_investigation_runtime_v1.py`는 판단·승인·실행 요청·완료 영수증을 저장해 프로세스 재시작 뒤에도 같은 작업을 조회한다. 실행 요청 기록이 있으면 새 작업을 제출하지 않으며, 검증된 검사 결과를 받은 뒤 다음 모델 입력을 만든다. 교정 미회복 뒤에는 새 checkpoint에 결합된 두 번째 고리만 허용한다. 실행 callback과 승인은 호스트가 명시해야 하며, 원문·승인 원장·영수증을 기존 controller가 다시 검증한다. CPU 재시작·변조 회귀를 통과했으며, 이 어댑터를 이용한 새 GPU 종단 실행은 아직 측정하지 않았다. 기존 controller의 Windows 실행 영수증과 Brev의 POSIX 영수증은 서로 다른 계약이므로 같은 형식으로 간주하지 않는다.

`agentic_brev_probe_evidence_v1.py`는 Brev F1/NONE 원본의 job·worker·report·POSIX 실행·GPU·임대·소스·checkpoint를 다시 읽는 검사 검증기다. 원격 경로는 원문에 보존하고 등록한 root-map으로만 보관 사본을 찾는다. 기존 여섯 검사와 optimizer 세 update의 측정 규칙을 재사용한다. 과거 session ID만 남은 보관 결과는 부족한 소유 process·GPU·임대 근거를 채우지 않고 HOLD로 둔다.

`agentic_brev_controller_scope_v1.py`는 이 새 원문을 기존 승인·seal·조사 고리에 연결한다. 공개 등록의 필수 소스·검증 helper·Git 조상과 blob을 확인하고, 같은 thread의 context 안에서만 새 GPU 계약을 연다. 새 계약은 합성 시험에서도 승인 원장을 소비하기 전에 소스 등록을 요구하며, 합성 입력을 실제 입력으로 바꾸지 않는다. 종료 시 기존 controller를 복원한다. 완료 원문의 신뢰 digest·예상 probe·operation·checkpoint는 별도의 호스트 봉인에서 받아야 한다. 이 연결은 CPU 검증 단계이며 실제 검사 실행·교정 backend·새 회복 결과를 제공하지 않는다. 호스트는 지원하지 않는 교정 요청을 승인 전에 HOLD로 닫아야 한다.

`bench/run_agentic_brev_probe_unit_v1.py`는 명시적으로 승인된 검사 한 건을 기존 POSIX backend에 전달하는 실행 단위다. 기본 동작은 준비·조회이며, 같은 operation의 불완전 기록을 만나면 재실행하지 않는다. 기존 controller로 승인·소비 원장과 등록 소스를 다시 검사하고, 원래 job·실행·GPU·소스 원문을 새 증거 검증기가 확인한 뒤에만 완료를 기록한다. CPU 관련 회귀 98개를 통과했으며, 실제 사용에는 호스트가 유지하는 OS 잠금과 별도 검사 권한 증명이 필요하다. 완료 reference가 없는 첫 대조 검사와 Windows·POSIX 간 승인 운반은 별도 연결 전까지 HOLD다. 이 단위의 CPU 통과를 새 GPU 검사 완료로 세지 않는다.

## 개발 호출 경로

`run_agentic_development_v1.py`는 자료 조회 결과를 다음 판단 입력에 추가하고, 등록 검사 제안 또는 HOLD에서 멈춘다. 검사와 교정은 실행하지 않는다. Windows 호스트가 API를 예약한 뒤 WSL의 기존 provider를 호출하며, 자격증명은 WSL 환경에만 남는다. 응답이 불명확하게 끊기면 자동 재전송하지 않는다.

[개발 사전등록](../bench/protocols/agentic_investigation_development_20261028_v1.json)은 이미 본 정상·보상 결함 자료 세 입력과 최대 네 행동, 별도 API 12요청을 고정한다. 첫 두 입력은 기존 학습 수치를 제공하고, 세 번째는 설정 수치만 먼저 제공해 자료 조회 경로를 확인한다. 프레임 조회는 선택 사항이다. 이 자료를 새 결함이나 독립 보류 사례로 세지 않는다.

기존 core 120회 배정의 소비·상한은 보존한다. 과거 일부 상태 조회 기록을 회수하지 못해 잔여량을 확정하지 못했으므로 그 잔여를 새 단계에 재사용하지 않는다. 새 단계 12회를 더한 배정 상한 합계는 132회이며 실제 소비 합계가 아니다. 이후 비교 호출은 이 등록에 포함하지 않는다.

실행은 등록 커밋이 현재 HEAD의 조상이고 결합 파일 blob이 같을 때만 허용한다. 비공개 실행 계획에는 원본 경로·해시·이전 소비 감사·새 출력 위치를 결합한다.

### 첫 개발 측정

[v1 결과](../evals/results/agentic_investigation_dev_20261028_v1/summary.json)는 완료 응답 3건 중 유효 0건, 무효 3건으로 모두 HOLD였다. 학습 수치가 이미 제공된 두 입력은 사용할 수 없는 `read_evidence`에 `target: null`을 반환했고, 설정만 제공한 입력은 등록 대상을 골랐으나 `fact_ids`를 비웠다. 입력의 허용 대상·필수 근거 조건과 모델 원문을 대조해 계약 위반으로 분류했다.

API 소비는 별도 12회 상한 중 3회, pending 0건이다. 완료 응답 재생은 추가 호출 0회였다. 프레임 조회·확인 검사·교정은 실행되지 않았으므로 영상 효과·원인 확정·회복을 판정하지 않는다. 다음 계약 판에서도 이 응답을 유효한 답으로 보정하지 않으며, 새 호출은 별도 사전등록 뒤 같은 개발 예산의 잔여 안에서 진행한다.

### v2 출력 계약 패키지

[v2 등록](../bench/protocols/agentic_investigation_development_20261028_v2.json)은 같은 세 개발 입력에 명시적 prompt와 판단별 `response_format: json_schema`를 함께 적용한다. 현재 허용 행동·대상·공개 fact ID를 enum으로 전달하고 필수 세 필드·배열 길이를 명시한다. 기존 action 검증기가 대상과 행동의 관계, 빈 근거·중복 근거를 계속 확인한다. NVIDIA의 [해당 hosted API](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b-infer)와 [모델별 가이드](https://docs.nvidia.com/nim/large-language-models/2.0.13/get-started/advanced/get-started-nemotron-3.5-super-vl.html)에 JSON Schema 지원이 명시돼 있다. 개별 키워드의 hosted 강제 여부와 이 프로젝트의 유효율 개선은 실제 호출에서 확인한다.

v1의 잘못된 세 답은 v2에서도 무효다. 기존 API 원장·소비 3회를 고정하고 같은 12회 상한의 잔여 9회만 사용한다. 실제 WSL 요청·응답의 메시지와 schema params 해시를 대조한다. prompt와 schema가 함께 바뀌므로 이 개발 결과를 prompt 단독 효과로 해석하지 않는다.

[v2 실측](../evals/results/agentic_investigation_dev_20261028_v2/summary.json)은 추가 요청 4건 모두 호스트 계약 검증을 통과했다. 세 입력 모두 검사 제안에 도달했고, 설정만 제공한 입력에서는 실제 학습 기록 조회 뒤 반환된 fact ID를 인용해 다음 검사를 제안했다.

| 같은 개발 입력 3건 | 검사 제안까지 도달 | 유효 응답 | 프레임 조회 |
|---|---:|---:|---:|
| v1 | 0/3 | 0/3 | 0 |
| v2 prompt + JSON Schema | 3/3 | 4/4 | 0 |

정상 입력과 설정부터 시작한 보상 결함 입력은 `P_episode`, 학습 수치가 제공된 보상 결함 입력은 `P_reward`를 제안했다. 확인 검사를 실행하지 않았으므로 이를 올바른 원인 진단으로 세지 않는다. 두 판을 합친 API 소비는 7/12, pending 0건이며 완료 응답 재생은 추가 0회다. 이번 기록은 조회·다음 행동 연결과 계약 유효성을 확인한 개발 측정이다. 시각 기여·회복·미사용 입력 성능 및 JSON Schema 각 키워드의 강제 여부는 별도 확인이 필요하다.

## 영상과 같은 시각의 관절 상태

`bench/record_agentic_state_sidecar_v1.py`는 기존 native 녹화의 프레임 callback에서 simulation time·step과 관절 위치·속도·계산 토크·적용 토크·가용 접촉 상태를 기록한다. 원본에 없는 값은 미가용 사유로 남긴다. 접촉 센서의 lazy update를 일으키는 읽기는 건너뛰며, 별도 시각이나 행동 정답을 만들지 않는다. 녹화 원문과 checkpoint·frame·시각을 맞춰 확인한 뒤 사용할 수 있다.

`evals/agentic_state_facts_v1.py`는 이 결합을 확인하고 기존 두 window의 선언 시각 여섯 개씩에서 몸체 좌표의 명령·속도·벡터 추종 오차와 관절·토크 기록을 요약한다. 없는 값은 미가용으로 남기고, 명령 크기가 0인 상대 오차는 정의하지 않는다. 이 수치는 모델 주장의 인용 근거용이며 보행·넘어짐·원인·회복 라벨을 만들지 않는다. 현재 관절·토크 회귀는 합성 입력으로 검증했으며 새 모터 영상의 관측 결과로 세지 않는다.

`bench/agentic_fault_playback_v1.py`는 새 protocol에 등록한 Go2 F2를 App 생성 뒤 환경 생성 전에 적용하고, 녹화가 끝나거나 실패하면 원래 모터 함수를 복원한다. F2는 전체 DCMotor의 속도 항을 바꾸는 결함이다. 한쪽 다리 고장으로 해석하지 않는다. 학습 당시 결함을 재현한 Play의 관절 상태는 정상 환경에서 수행한 기존 고정 평가와 구분해 기록한다.

두 기능은 CPU callback·실패 봉인·source/checkpoint 검증을 통과했다. 새 모터 checkpoint 등록과 실제 Isaac/Brev 녹화·비간섭 확인은 후속 단계다. 기록기만으로 영상의 진단 기여가 확인된 것은 아니다.

`bench/agentic_play_job_v1.py`는 새 checkpoint 목록·F2/NONE 쌍·runtime 소스·촬영 profile을 등록한다. `resource_lease()`가 자원별 OS 잠금과 owner start identity·heartbeat를 유지하고, 부모·AABA child·녹화 child·최종 수용 시 같은 임대를 확인한다. 동결 대상에는 잠금·checkpoint·provider 검증 helper도 포함한다. 시험용 process callback을 주입한 결과는 실제 실행으로 표시하지 않는다.

`bench/run_agentic_capture_unit_v1.py`는 임대 context 안에서 fresh provider/GPU 원문을 확인한 뒤 NONE 정책의 1-frame 상태 smoke 또는 등록 정책 하나의 native 녹화를 실행한다. 상태 smoke는 exit code 0에 더해 결과 파일·실패 파일 부재·checkpoint·상태 기록·소유 child 종료·GPU 유휴를 확인한다. 각 VM의 새 프로토콜과 source를 커밋하고 실제 원격 runtime을 대조한 뒤 실행해야 한다. CPU 승인과 실제 GPU 수용을 구분한다.

[첫 수집 시도](../evals/results/agentic_motor_capture_20261028_v1/summary.json)는 기동 대기 상한과 Brev control-plane의 TLS hostname 불일치로 Isaac 실행 전에 종료됐다. 녹화·학습·모델 호출은 0건이며 두 자원의 `STOPPED`를 확인했다. [새 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v2.json)은 같은 네 정책을 기존 VM 한 대에서 순차 수집한다. 각 녹화의 frame·AABA·child 상한은 유지하며, read-only 상태 조회만 제한 횟수 안에서 다시 시도한다.

[순차 수집 v2](../evals/results/agentic_motor_capture_20261028_v2/summary.json)는 원격 접속 뒤 cache 경로 조건에서 중단됐고 정지를 확인했다. [v3 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v3.json)은 이전 성공 기록에 명시된 VM별 HOME 상대 repo·Lab·Python 경로를 사용한다. 환경이 사라졌다고 단정하거나 재설치하지 않으며, 실제 존재 상태를 판정 전에 기록한다. 원격 본문 종료 코드를 별도 marker로 확인하고 같은 요청의 재실행을 거부한다.

[v3의 실제 boot](../evals/results/agentic_motor_capture_20261028_v3/summary.json)는 cache와 source·package pin 확인을 통과했다. CLI가 성공 marker 뒤에 붙인 자원 이름 한 줄을 host가 거부해 녹화 전에 종료했다. [v4 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v4.json)은 등록된 이름 한 줄만 선택적으로 처리하며 단일 marker·비영 종료·예상 밖 출력 거부는 유지한다.

[v4 실행](../evals/results/agentic_motor_capture_20261028_v4/summary.json)은 boot 확인과 네 checkpoint 복사를 통과한 뒤 상태 smoke의 잠금 취득에서 중단됐다. 고정 cache 잠금 경로의 조상에 symlink 또는 reparse point가 있어 거부됐으며, 구체적인 링크 대상은 아직 관측하지 않았다. GPU 입장·Isaac 실행·녹화는 시작하지 않았고 자원 정지를 확인했다. 후속 adapter는 alias와 canonical 경로가 같은 자원 잠금을 사용하게 하며 기존 검증을 유지한다.

```powershell
cd "$HOME/rl-triage-agent"
.venv/Scripts/python.exe evals/run_agentic_development_v1.py run --plan <등록된-비공개-plan.json의-전체경로>
```

## 확인한 범위

CPU 회귀는 다음 흐름을 기존 controller의 raw 파일·승인 원장·검사/교정 seal과 함께 실행한다.

- 정상 입력에 대한 명시적 HOLD와 추가 실행 없음.
- 원본 프레임 근거 조회 후 등록 검사 제안.
- 첫 정상 검사 뒤 다른 검사를 제안하고, 실제 이상 receipt에서 교정 후보가 열림.
- 첫 교정 미회복 → 출력 checkpoint에 결합된 두 번째 고리 → 다른 교정 후 회복 판정.
- 무효 JSON·없는 근거·미등록 행동·승인 필드 주입·seal 변조·타 계보·unknown receipt 거부.
- 운영 모드에서 합성 CPU fixture 결과 거부.

검증 명령:

```powershell
cd "$HOME/rl-triage-agent"
.venv/Scripts/python.exe -m pytest -q tests/test_agentic_investigation_v1.py
.venv/Scripts/python.exe -m pytest -q tests/test_agentic_investigation_call_v1.py tests/test_agentic_development_v1.py
```

합성 CPU 경로의 통과는 실제 Nemotron 선택, 실제 학습 회복, 시각 정보의 추가 효과를 의미하지 않는다. 이 세 결과는 새 사전등록과 실제 실행에서 각각 측정한다.
