# 실제 프레임 복사본의 품질·승인 가드 확인

기존 시뮬레이션 프레임 두 장의 원본을 보존하고, 통제 복사본에서 흐림과 decoder 사용 불가 조건을 만들었다. 품질 가드는 해당 프레임을 모델 입력에서 제외했고 승인·실행 권한은 생성하지 않았다. GPU busy 관측의 callback 계약도 원래 host 승인 전에 HOLD하는지 확인했다.

| CPU 소비 조건 | 품질값 / 관측 | 모델에 전달할 이미지 | 결과 |
|---|---|---:|---|
| 원본 두 프레임 | 903.5998 / 1078.9633 | 2 | 품질 available |
| 통제 blur 복사본 | 0.7152 / 0.7003 | 0 | visual unknown, HOLD |
| 통제 decoder-unavailable | metric 없음 | 0 | decode 격리, HOLD |
| 통제 busy callback | busy 관측 1회 | 해당 없음 | host approval 0회, HOLD |

품질값은 FFmpeg grayscale의 4-neighbor Laplacian population variance이며 단위는 intensity²다. 기존에 등록한 floor 10을 유지했다. 원본 영상의 의미나 보행 상태를 판정한 측정은 아니다. 실행 전후 원문 28파일의 SHA가 같았고 API·GPU·승인·dispatch는 모두 0회였다.

손상 JPEG는 상위 입력 계약이 먼저 거절했다. 따라서 자연 발생한 decode 실패가 품질 adapter까지 도달했다고 보고하지 않는다. decoder 사용 불가와 busy 관측은 통제 주입이며 새 API 장애나 실제 GPU 충돌을 발생시킨 결과가 아니다. 기존 API prefix 7쌍은 읽기 검증만 했고, 현재 부재한 snapshot ledger를 재구성하거나 재개하지 않았다.

이 확인은 품질·승인 경계의 CPU 개발 측정이다. 시각·수치 의미 충돌과 API 오류의 전체 실행 matrix는 아직 완료되지 않았다. [공개 기술 요약](../evals/results/agentic_input_guard_cpu_development_20261101_v1/summary.json), [출처 결합](../evals/results/agentic_input_guard_cpu_development_20261101_v1/provenance.json), [품질 가드](../evals/agentic_frame_quality_guard_v1.py), [실행 가드](../evals/agentic_runtime_input_guard_v1.py).
