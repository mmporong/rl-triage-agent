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

`bench/run_agentic_brev_probe_parent_v1.py`는 별도로 등록한 검사쌍 승인 원장을 검증해 대조 NONE→사례 F1 순서로 실행하는 부모다. 원문 모델 응답과 history는 보존하고 등록한 사본 경로만 해석한다. 현재 boot의 L40S UUID·활성 OS 잠금·입장 관찰·두 작업의 원문을 결합하며, 지원 probe는 공개 등록의 명시적 부분집합으로 제한한다. 종료 불명 자식은 격리하고 같은 작업을 다시 제출하지 않는다. 부모 CPU 회귀 66개와 저장된 모델 응답의 POSIX 운반 검증을 통과했으며, 새 GPU 검사와 기존 조사 고리에 대한 완료 seal 반영은 후속 단계다.

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

새 `agentic_play_job_v2.py`·`run_agentic_play_record_v2.py`·`run_agentic_capture_unit_v2.py`는 고정 HOME cache alias를 canonical 경로에 결합하는 어댑터다. 부모와 SDK·AABA 자식이 같은 자원별 lock inode를 사용하며, 기존 owner start identity·heartbeat·소스 검증을 유지한다. GPU 입장 전에 alias와 실제 경로 관측을 기록한다. 실제 WSL의 관련 회귀 25개를 통과했고 Windows 전용 1개는 제외됐다. 이 결과는 클라우드 VM에서의 새 녹화 완료를 뜻하지 않는다.

`run_agentic_capture_unit_v3.py`는 공급자 조회와 현재 입장 검증의 시각을 구분한다. 공급자 원문은 30초 신선도로 검사하고, GPU query 뒤 실제 현재시각에서 활성 임대와 resource proof를 검증한다. 조회·query·임대 취득·입장 시각과 원문 SHA를 별도 receipt에 보존한다. 이전 실행기는 수정하지 않는다.

`record_agentic_state_sidecar_v2.py`와 `run_agentic_capture_unit_v4.py`는 상태 시험의 SDK 종료 증거를 연결한다. 결과·sidecar·세션 정상 종료를 검증한 뒤 기존 fast-shutdown 설정을 적용하고, 부모는 자연 exit 0·소유 자식 정리·체크포인트·종료 증거 PID의 일치를 추가 확인한다. 비영 종료를 성공으로 바꾸지 않는다. 종료 정책 helper를 포함한 소스 141개를 결합하며, 새 child·부모·입장 경계의 WSL 회귀 39개를 통과했다. 실제 SDK 충돌의 개선 여부는 새 GPU 실행으로 판정한다.

`agentic_motor_capture_acceptance_v1.py`는 명시한 phase·checkpoint 등록과 회수 archive·C/D 사본·native·sidecar·같은 Play의 trace를 대조한다. 네 정책이 모두 기술 수용돼야 공개 후보 상태 facts를 만들며, 기존 false 승인 플래그와 행동 라벨 부재를 유지한다. 관련 Python 3.10 CPU 회귀 26개를 통과했다. 실제 자료 수용과 모델 사용은 별도 단계다.

[첫 수집 시도](../evals/results/agentic_motor_capture_20261028_v1/summary.json)는 기동 대기 상한과 Brev control-plane의 TLS hostname 불일치로 Isaac 실행 전에 종료됐다. 녹화·학습·모델 호출은 0건이며 두 자원의 `STOPPED`를 확인했다. [새 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v2.json)은 같은 네 정책을 기존 VM 한 대에서 순차 수집한다. 각 녹화의 frame·AABA·child 상한은 유지하며, read-only 상태 조회만 제한 횟수 안에서 다시 시도한다.

[순차 수집 v2](../evals/results/agentic_motor_capture_20261028_v2/summary.json)는 원격 접속 뒤 cache 경로 조건에서 중단됐고 정지를 확인했다. [v3 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v3.json)은 이전 성공 기록에 명시된 VM별 HOME 상대 repo·Lab·Python 경로를 사용한다. 환경이 사라졌다고 단정하거나 재설치하지 않으며, 실제 존재 상태를 판정 전에 기록한다. 원격 본문 종료 코드를 별도 marker로 확인하고 같은 요청의 재실행을 거부한다.

[v3의 실제 boot](../evals/results/agentic_motor_capture_20261028_v3/summary.json)는 cache와 source·package pin 확인을 통과했다. CLI가 성공 marker 뒤에 붙인 자원 이름 한 줄을 host가 거부해 녹화 전에 종료했다. [v4 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v4.json)은 등록된 이름 한 줄만 선택적으로 처리하며 단일 marker·비영 종료·예상 밖 출력 거부는 유지한다.

[v4 실행](../evals/results/agentic_motor_capture_20261028_v4/summary.json)은 boot 확인과 네 checkpoint 복사를 통과한 뒤 상태 smoke의 잠금 취득에서 중단됐다. 고정 cache 잠금 경로의 조상에 symlink 또는 reparse point가 있어 거부됐으며, 구체적인 링크 대상은 아직 관측하지 않았다. GPU 입장·Isaac 실행·녹화는 시작하지 않았고 자원 정지를 확인했다. 후속 adapter는 alias와 canonical 경로가 같은 자원 잠금을 사용하게 하며 기존 검증을 유지한다.

[v5 수집 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v5.json)은 이 어댑터와 입력 복사 전 POSIX 잠금 사전 검사를 결합한다. 기존 두 seed·네 정책·625프레임·50fps·AABA 조건은 유지하고, 이전 실패의 배정 상한을 보존한 새 $5 구간으로 실행한다. 실제 녹화 성공 여부는 새 실행 원문으로 판정한다.

[v5 실행](../evals/results/agentic_motor_capture_20261028_v5/summary.json)은 두 프로토콜의 실제 POSIX 잠금 검사를 통과했지만, 이후 입장 검사에서 과거 공급자 조회 시각을 현재 임대 검증 시각으로 재사용해 중단됐다. 영상·상태 frame·학습·모델 호출은 0건이며 정지와 원문 회수를 확인했다. 후속 판은 App 실행 전에 전체 입장 proof의 생성과 소비까지 확인한다.

[v6 수집 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v6.json)은 새 unit v3와 두 종류의 시각을 분리한 receipt를 결합한다. 같은 네 정책·촬영 조건을 유지하며, 기존 실패의 배정 상한을 보존한 새 $5 구간과 필수 소스 135개를 고정한다. 생성된 원격 입장 program의 proof 소비 경로도 실제 OS 잠금을 사용하는 CPU 회귀에서 확인했다.

[v6 실행](../evals/results/agentic_motor_capture_20261028_v6/summary.json)은 두 전체 입장 검사를 통과하고 상태 1프레임을 봉인했지만 자식이 SIGSEGV로 종료해 실패로 유지됐다. 세션과 앱 종료 중 충돌한 위치는 기존 기록만으로 확정하지 못했다. 새 종료 단계 기록은 그 위치를 구분하며, 전체 정책 영상은 아직 생성하지 않았다.

[v7 수집 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v7.json)은 상태 시험을 봉인 검증 후 SDK fast shutdown을 적용하는 새 자식과 연결한다. 자연 exit 0·소유 자식 정리·종료 증거 PID 일치를 요구하며, 네 정책과 촬영 조건은 유지한다. 관련 host WSL 회귀 105개를 통과했고 필수 소스 141개를 결합했다. 이전 배정 상한을 보존한 새 $5 구간으로, 실제 종료와 영상 수용 여부는 실행 뒤 판정한다.

[v7 실행](../evals/results/agentic_motor_capture_20261028_v7/summary.json)은 두 전체 입장 검사를 통과했지만 입력 전송의 HTTP 401 응답으로 App 전에 종료됐다. 부분 원문 39파일의 사본 해시와 자원 정지를 확인했다. [v8 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v8.json)은 같은 입력의 기존 원격 cache를 SHA 대조해 재사용하며, 회수 경로도 Start 전에 확인한다. 새 경계의 WSL 회귀 122개를 통과했고 촬영 조건은 유지한다. 실제 SDK 종료 개선 여부는 아직 미측정이다.

[v8 실행](../evals/results/agentic_motor_capture_20261028_v8/summary.json)에서는 원격 입력을 재사용했고 상태 시험 한 건이 봉인·세션 종료·fast shutdown·자연 exit 0·소유 자식 정리를 통과했다. 이어진 녹화는 AABA 자식의 exit 0 뒤 GPU 관찰 원문 참조 누락으로 중단됐다. 새 runner v3는 임대 검사를 유지하면서 기존 reader의 현재 `last_receipt`를 전달한다. 관련 새 경계의 WSL 회귀 23개를 통과했으며, 전체 영상 수집은 아직 미완료다.

[v9 수집 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v9.json)은 이 runner와 unit v5를 소스 145개에 결합한다. 같은 네 정책·원격 입력 재사용·종료 증거·촬영 조건을 유지하며 추가 $5 상한으로 실행한다. 등록 host의 WSL 회귀 122개를 통과했고, 실제 전체 녹화는 새 원문으로 판정한다.

## 모터 영상과 상태의 실제 수집

[v9 실제 결과](../evals/results/agentic_motor_capture_20261028_v9/summary.json)는 이미 본 seed 2033·2046의 정상·F2 정책 네 개를 모두 기술 수용했다. 각 영상은 625프레임·50fps·12.5초이며 관절·기록 토크·몸체 속도를 같은 Play의 프레임과 결합했다. 원문 7,895파일의 C/D 사본 해시를 대조하고 VM 정지를 확인했다. 학습과 모델 호출은 0건이다.

1–6초의 고정된 여섯 sample에서 수평 명령 크기는 모두 1.0 m/s였다. 아래는 같은 sample의 몸체 수평 속도 크기와 명령 벡터 추종 오차 평균이다. 전체 영상 평균이나 행동 판정 임계값은 아니다. [전체 수치와 fact 출처](../evals/results/agentic_motor_capture_20261028_v9/window_metrics.json).

| seed | 정책 | 수평 속도 크기 평균 (m/s) | 추종 오차 평균 (m/s) | 같은 Play 영상 |
|---|---|---:|---:|---|
| 2033 | 정상 | 1.0384 | 0.0776 | [영상](media/agentic_motor_capture_20261028_v9/motor-s2033-none.mp4) |
| 2033 | 모터 결함 | 0.0018 | 0.9984 | [영상](media/agentic_motor_capture_20261028_v9/motor-s2033-f2.mp4) |
| 2046 | 정상 | 1.0882 | 0.1085 | [영상](media/agentic_motor_capture_20261028_v9/motor-s2046-none.mp4) |
| 2046 | 모터 결함 | 0.0033 | 0.9972 | [영상](media/agentic_motor_capture_20261028_v9/motor-s2046-f2.mp4) |

이 자료는 등록된 모터 변경을 재적용한 Play 관측이다. 별도 pristine 고정 평가와 구분하며, 모터 변경은 12관절 전체에 적용됐다. 기술 수용 뒤에도 원래 승인 플래그·행동 라벨 부재를 보존했다. 이 수집에서는 Nemotron을 호출하지 않았으므로 시각 정보의 진단 기여를 판정하지 않는다. [영상·checkpoint·상태 출처](../evals/results/agentic_motor_capture_20261028_v9/source_manifest.json) · [공개 facts](../evals/results/agentic_motor_capture_20261028_v9/motor_capture_facts.json).

처음 화면 확인에서는 정상 seed 2033의 3초 프레임에 로봇이 보였지만, 6초·11초 검사 프레임에서는 보이지 않았다. 이후 네 정책 각 26개 표본의 CameraParams와 sim root를 대조했다. 렌더 카메라 위치는 trace의 현재 위치와 최대 약 5.3e-15 m 차이였으며, 밝은 무채색 픽셀의 중심은 현재 root보다 초기 root 투영에 가까웠다. 정상 두 정책에서 초기·현재 root 투영이 100픽셀 이상 떨어진 표본 46개 중 이 mask가 있던 19개를 보면, 초기 투영 30픽셀 안은 17개, 현재 투영 30픽셀 안은 0개였다. mask는 로봇의 시각적 근사이며 행동 정답 라벨은 아니다.

이 관찰은 카메라 추종 변경보다 articulation 렌더 상태 갱신을 점검할 근거다. `sim.forward()` 호출 전후의 물리 상태 동일성을 검사하는 새 경로를 준비하며, SDK 내부 원인과 수정 효과는 GPU 시험 전까지 미확정이다. 네 영상 모두 모델 입력 사용을 보류하고 기존 원문·기술 수용·상태 수치는 보존한다. [최초 화면 관찰](../evals/results/agentic_motor_capture_20261028_v9/visibility_observation.json) · [후속 판정과 근거](../evals/results/agentic_motor_capture_20261028_v9/render_visibility_followup.json) · [104개 표본](../evals/results/agentic_motor_capture_20261028_v9/render_visibility_evidence.json).

새 [runner v4](../bench/run_agentic_play_record_v4.py)와 [unit v6](../bench/run_agentic_capture_unit_v6.py)는 녹화 child의 매 capture 직전에만 fabric 활성·물리 snapshot·`forward()`·snapshot 불변을 검사한다. AABA와 기존 카메라·native frame 조건은 유지한다. sidecar 봉인과 앱 종료 전에 render-sync 기록을 남기고, 부모가 프레임 수와 job SHA를 대조한다. 새 WSL 회귀 22개를 통과했으며 실제 렌더 반영은 정상 한 정책의 새 GPU 시험으로 확인한다.

[정상 한 정책의 렌더 시험 등록](../bench/protocols/agentic_motor_render_preflight_phase_20261028_v10.json)은 같은 seed 2033 checkpoint와 625프레임·50fps 설정을 사용한다. 전체 입장·상태 시험 후 정상 정책만 녹화하며 추가 compute 상한은 $5다. 26개 고정 표본에서 초기·현재 root 투영이 100픽셀 넘게 떨어진 표본을 최소 10개 확보하고, 그중 90% 이상이 현재 투영 30픽셀 안이어야 한다. 초기 투영 30픽셀 안 표본은 0개, 모든 표본의 고정 mask는 200픽셀 이상을 요구한다. 이는 이미 본 영상에서 만든 렌더 시험 기준이며 행동 정답이나 모델 성능 지표가 아니다. 원격 렌더 관련 Lab 파일 다섯 개도 새 pin으로 확인한다.

v10은 VM이 준비 상태에 이르지 못해 boot·App·녹화 전에 종료됐다. [새 자원의 v11 등록](../bench/protocols/agentic_motor_render_preflight_phase_20261028_v11.json)은 생성 원문에 결합한 L40S 한 대를 사용한다. 환경 복원과 이번 렌더 시험은 최초 Create 요청부터 계산하는 같은 $10 상한을 공유한다. 정상 정책 하나의 625프레임과 렌더 판정 기준은 v10과 같으며, 동결 소스 151개와 실제 환경 pin을 확인한 뒤 실행한다. CPU 환경 복원과 실행기 회귀 28개 통과는 GPU 렌더 통과를 뜻하지 않는다.

[v11 실제 결과](../evals/results/agentic_motor_render_preflight_20261028_v11/summary.json)는 정상 seed 2033 한 정책에서 등록 조건 11개를 통과했다. 625회 `forward()` 전후의 물리 snapshot이 모두 같았고, native·sidecar·trace·playback과 실제 POSIX 임대를 검증했다. 26개 표본 중 초기·현재 root 투영이 100픽셀 넘게 떨어진 23개 모두에서 밝은 형상의 중심이 현재 root 30픽셀 안에 있었으며, 초기 root 30픽셀 안 표본은 0개였다. 전체 표본의 최소 mask는 969픽셀, 카메라·trace 위치의 최대 차이는 약 5.33×10⁻¹⁵m였다. 상한이나 분석법은 실행 뒤 바꾸지 않았다.

[새 정상 영상](media/agentic_motor_render_preflight_20261028_v11/normal-s2033.mp4)은 640×480·50fps·625프레임·12.5초이며 [3초](media/agentic_motor_render_preflight_20261028_v11/frame-3s.png)·[6초](media/agentic_motor_render_preflight_20261028_v11/frame-6s.png)·[11초](media/agentic_motor_render_preflight_20261028_v11/frame-11s.png) 원본 추출 화면도 함께 보관한다. 원문 2,005파일의 C/D 해시가 일치했고 VM 정지를 확인했다. 생성·설치·대기·시험·정지까지의 견적 환산은 약 $2.943이며 실제 청구액은 아니다. 이미 본 정상 정책 하나의 렌더 확인 결과다. 기존 결함 영상의 모델 입력 보류와 원래 승인 플래그는 유지하며, 결함 재녹화와 모델 판단 기여는 후속 실행에서 확인한다.

새 [F1 등록 계약](../bench/agentic_play_job_v3.py)과 [녹화 unit v7](../bench/run_agentic_capture_unit_v7.py)은 같은 렌더 갱신·임대·native·sidecar·AABA 경로에 정상/F1 정책 쌍을 연결한다. checkpoint 행 전체를 기존 historical catalog와 대조하며, 학습의 보상 결함을 Play 때 다시 주입하지 않는다. 자식은 원본 보상·모터 함수의 출처를 확인하고 부모도 원본 모터 함수 조건을 검사한다. WSL CPU 회귀 44개를 통과했으며 F1 실제 재녹화와 화면 수용은 새 사전등록 뒤 확인한다.

[독립 F1 수집 환경 등록](../bench/protocols/fresh_brev_l40s_f1_environment_20261028d.json)은 기존 검사 자원과 별도로 AWS 단일 L40S 환경 하나를 준비한다. 생성·고정 환경 복원과 후속 녹화가 같은 $10 상한을 공유하며, 새 자원 ID와 별도 촬영 등록을 결합하기 전에는 Isaac·정책 녹화·모델 호출을 허용하지 않는다.

## 실제 검사 결과를 다음 판단에 연결하는 운반 계층

새 [호스트](../evals/agentic_brev_runtime_host_v1.py)는 정상 참조 원문을 검증한 뒤 case 검사 승인을 소비하고, case 원문이 검증·봉인돼야 다음 조사 snapshot을 반환한다. [mailbox transport](../evals/agentic_brev_mailbox_transport_v1.py)와 [IO service v2](../evals/agentic_brev_mailbox_service_v2.py)는 등록한 원격 helper만 호출한다. IO 응답을 받았다는 이유로 원문 검증을 생략하지 않는다.

기존 intent가 남은 요청은 조회로만 재개하며 새 실행·승인을 만들지 않는다. 서비스는 VM을 시작하거나 정지하지 않으며, 공개 사전등록 시각부터 비용·기한을 계산한다. 실제 raw SHA와 등록 root를 검사해 새 파일만 회수한다. 동결한 네 모듈·네 테스트의 메인 WSL 통합 75개를 통과했으며, 실제 Brev 검사쌍과 후속 모델 판단은 새 실행으로 확인한다.

[운반 helper v2](../bench/agentic_brev_probe_transport_remote_v2.py)는 원래 E13 검사 root를 유지하고 unit 부모의 outcome·evidence·request를 별도 metadata layer로 회수한다. 두 map과 원래 request·protocol·binding의 SHA를 archive에 묶고, 필수 원문이 등록 밖에 있거나 서로 다르면 거부한다. 실제 부모 실행과 같은 CPU 폴더 구조에서 양 arm의 metadata 여섯 경로와 원래 검사 근거를 결합하는 WSL 회귀 28개를 통과했다. 부분 회수와 중복 요청은 새 실행이나 승인으로 전환하지 않는다.

[텍스트 전용 후속 등록](../bench/protocols/agentic_runtime_text_s2221_20261028_v1.json)은 이미 본 F1 seed 2221의 실제 학습 기록을 조회한 뒤 실제 검사쌍과 후속 모델 판단을 연결한다. 학습 자료 alias 하나만 제공하고 프레임과 과거 모델 판단 prefix는 재사용하지 않는다. 기존 API 원장의 7건을 보존하며 전체 상한 12회 안에서 새 판단은 최대 4번이다. 정상 참조·case 검사·승인·소비·원문 봉인 뒤에만 다음 snapshot을 연다. 무효 판단과 지원 밖 교정 요청은 HOLD한다. 이 등록의 성공 범위는 검사 원문과 다음 판단의 연결이며 회복·시각 효과·새 seed 성능과 구분한다.

실제 모델 두 판단은 학습 기록 조회와 `P_reward` 검사 선택으로 유효했다. 첫 Parent 등록은 명령 canonical hash의 생성·소비 규약 차이로 GPU 전에 차단됐으며, [새 Parent 판](../bench/protocols/agentic_runtime_text_s2221_pair_20261028_v2.json)은 같은 모델 선택과 고정 인자를 유지하고 hash 규약을 맞춘다. [차단 원문 요약](../evals/results/agentic_runtime_text_pair_registration_20261028_v1/summary.json)을 보존하며 이 재등록에 새 모델 호출은 쓰지 않는다.

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
