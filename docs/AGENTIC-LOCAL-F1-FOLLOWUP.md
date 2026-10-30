# 로컬 F1 검사 후속 고리

이 개발 측정은 Nemotron 3.5가 고른 검사를 실제 Isaac Lab 실행으로 확인하고, 그 결과를 다음 판단에 다시 넣는 구간을 기록한다. 대상은 이미 본 seed 2221의 Unitree Go2 F1 보상 결함이다.

첫 모델 판단은 학습 telemetry 6개를 근거로 `P_reward`를 골랐다. 프레임 인용은 없었다. 이어서 Windows Isaac Lab에서 NONE과 F1 체크포인트를 같은 조건으로 실행했다. 환경 256개를 200 step 동안 측정한 결과, 선형 속도 추종 보상의 상대 오차는 NONE `1.6901418784520673e-08`, F1 `0.30962626162916423`이었다. 등록 임계값 `0.05`를 적용하면 NONE은 normal, F1은 abnormal이다.

검사 뒤 첫 후속 호출인 API 11은 `ConnectionResetError`로 끝났다. 의미 판단이 나오기 전의 인프라 실패이므로 모델 오진으로 세지 않았다. 같은 messages와 params를 보존한 등록 재시도 API 12는 `observation:0:P_reward:outcome`과 학습 telemetry를 인용해 `restore_body_frame_reward`를 제안했다. 계약 검증 결과는 valid, 고리 상태는 `choice_ready`였다.

| 단계 | 관측 결과 | 다음 상태 |
| --- | --- | --- |
| 모델의 검사 선택 | `P_reward`, telemetry 6개, frame 0개 | 로컬 검사 실행 |
| NONE/F1 검사 | normal / abnormal | 보상 결함 근거 추가 |
| 후속 모델 판단 | `restore_body_frame_reward` 제안 | 승인 대기 |

여기서 끝난 범위는 교정 제안까지다. 승인, 교정 실행, 재학습, 회복 평가는 아직 수행하지 않았다. 같은 seed를 사용한 개발 측정이므로 처음 보는 결함에 대한 일반화, LLM 우위, 영상이 검사 선택을 개선했다는 주장에도 쓰지 않는다. 공개 수치와 계보는 [`evals/results/agentic_local_f1_probe_followup_s2221_20261030_v1`](../evals/results/agentic_local_f1_probe_followup_s2221_20261030_v1/)에 있다.
