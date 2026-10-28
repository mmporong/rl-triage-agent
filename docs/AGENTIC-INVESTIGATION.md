# 다단계 조사 행동 계약

`evals/agentic_investigation_contract_v1.py`와 `evals/agentic_investigation_loop_v1.py`는 기존 검사·교정 controller 위에 자료 조회와 다음 행동 제안을 연결하는 CPU 어댑터다. 실제 모델 호출기와 GPU backend의 추가 결합은 아직 완료하지 않았다. 기존 [3.5 종단 실행](VISION-TRANSFERRED-E2E-DEVELOPMENT.md)의 첫 검사 선택과 후속 관찰 결과를 이 어댑터의 실제 실행 성과로 합산하지 않는다.

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

`Investigation` 객체는 한 프로세스의 CPU 상태 어댑터다. 모델 요청 예약·재시도·프로세스 재시작의 지속 저장은 실제 caller 통합에서 기존 원장과 연결해야 한다. 해당 통합을 구현하기 전에는 이 객체를 새 운영 실행기로 사용하지 않는다.

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
```

합성 CPU 경로의 통과는 실제 Nemotron 선택, 실제 학습 회복, 시각 정보의 추가 효과를 의미하지 않는다. 이 세 결과는 새 사전등록과 실제 실행에서 각각 측정한다.
