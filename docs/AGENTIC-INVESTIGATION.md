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

## 개발 호출 경로

`run_agentic_development_v1.py`는 자료 조회 결과를 다음 판단 입력에 추가하고, 등록 검사 제안 또는 HOLD에서 멈춘다. 검사와 교정은 실행하지 않는다. Windows 호스트가 API를 예약한 뒤 WSL의 기존 provider를 호출하며, 자격증명은 WSL 환경에만 남는다. 응답이 불명확하게 끊기면 자동 재전송하지 않는다.

[개발 사전등록](../bench/protocols/agentic_investigation_development_20261028_v1.json)은 이미 본 정상·보상 결함 자료 세 입력과 최대 네 행동, 별도 API 12요청을 고정한다. 첫 두 입력은 기존 학습 수치를 제공하고, 세 번째는 설정 수치만 먼저 제공해 자료 조회 경로를 확인한다. 프레임 조회는 선택 사항이다. 이 자료를 새 결함이나 독립 보류 사례로 세지 않는다.

기존 core 120회 배정의 소비·상한은 보존한다. 과거 일부 상태 조회 기록을 회수하지 못해 잔여량을 확정하지 못했으므로 그 잔여를 새 단계에 재사용하지 않는다. 새 단계 12회를 더한 배정 상한 합계는 132회이며 실제 소비 합계가 아니다. 이후 비교 호출은 이 등록에 포함하지 않는다.

실행은 등록 커밋이 현재 HEAD의 조상이고 결합 파일 blob이 같을 때만 허용한다. 비공개 실행 계획에는 원본 경로·해시·이전 소비 감사·새 출력 위치를 결합한다.

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
