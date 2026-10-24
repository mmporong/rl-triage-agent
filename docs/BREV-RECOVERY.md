# Brev의 등록 F1 실행

L40S에서 정상·결함 정책의 학습과 고정 평가를 마쳤다. 모델이 선택한 첫 검사는 normal이었고, 다음 모델 요청이 HTTP 500을 두 번 반환해 고리는 `HOLD`로 끝났다. 교정과 교정 후 회복 평가는 요청하지 않았다.

[공개 집계](../evals/results/brev_recovery_v4_s2196_20261024/summary.json) · [사전등록](../bench/protocols/brev_recovery_v4.json) · [검사](../evals/results/brev_recovery_v4_s2196_20261024/probes.json) · [모델 호출](../evals/results/brev_recovery_v4_s2196_20261024/model_calls.json) · [공개 파일 해시](../evals/results/brev_recovery_v4_s2196_20261024/manifest.json)

## 실행 조건

소스와 프로토콜은 측정 전 커밋 `fa0c8f2707eaee93c84ff273a4510c164ef3fd90`으로 고정했다. 기존 F1 결과와 고리 구현을 본 뒤 설계한 클라우드 이식 개발 측정이다. 등록한 보상 좌표계 결함을 사용했으며, 처음 보는 결함에 대한 대응을 평가한 실행은 아니다.

| 항목 | 고정 조건 |
| --- | --- |
| 작업 | `Isaac-Velocity-Flat-Unitree-Go2-v0` |
| 인스턴스 | Stop 지원 `l40s-48gb.1x` |
| 정상·F1 정책 | 같은 source seed 2196, 각각 처음부터 300회, 환경 4,096개 |
| dry-run | 별도 seed 2197, 정상 20회 |
| 환경 | Python 3.10.12, Isaac Sim 4.5.0.0, Isaac Lab `90b79bb2d44feb8d833f260f2bf37da3487180ba`, PyTorch 2.7.0+cu128, RSL-RL 2.3.3 |
| 행동 평가 | 기존 26개 명령·판정 밴드, 기존 `fixed_eval_v1` |
| 모델·실행 경계 | WSL에서만 모델 호출, 검사 선택 뒤 호스트가 인자 생성·승인·영수증 검증 |
| 상한 | 검사 7종 중 최대 7회, 등록 보상 교정 최대 1회, API 요청 총 20회 |

실행 전 로컬·클라우드 기록에서 seed 사용 여부를 확인했다. 각 학습·검사·평가 전후에 소스, 선택한 Python 환경과 패키지 버전을 확인했다. 체크포인트와 원본 출력의 해시, 자식 종료·정리 기록도 결합했다.

## 고정 평가와 판단

[정상 26조건 원본 지표](../evals/results/brev_recovery_v4_s2196_20261024/source_fixed_none.json) · [F1 26조건 원본 지표](../evals/results/brev_recovery_v4_s2196_20261024/source_fixed_f1.json) · [학습 영수증 요약](../evals/results/brev_recovery_v4_s2196_20261024/training.json)

| 지표 | 정상 NONE | F1 |
| --- | ---: | ---: |
| 선속도 추종 RMSE (m/s) | 0.139937 | 1.178762 |
| 회전속도 추종 RMSE (rad/s) | 0.287154 | 0.733081 |
| 넘어짐 비율 | 0 | 0 |
| 평균 생존 (초) | 20 | 20 |

F1 정책은 생존을 유지했지만 이동·회전 추종 기준을 통과하지 못해 진단 대상으로 들어갔다. 이후 모델이 `P_episode`를 선택했고, 정상·F1 정책의 검사 결과는 normal이었다. 이 결과로 보상 원인을 확정하지 않았다.

다음 판단 요청에서 HTTP 500 `InternalServerError`가 반환됐다. 사전등록한 인프라 재시도 한 번도 같은 상태 코드로 끝났다. 고리는 `model_unavailable` 사유로 HOLD했고, 종료 재감사도 같은 최종 상태를 확인했다. 서버 내부 원인은 이 응답 기록으로 가리지 못했다.

| 고리 기록 | 결과 |
| --- | --- |
| API 요청 | 3/20, 완료 3, 미완료 예약 0 |
| 유효한 모델 선택 | 1건, `P_episode` |
| 완료한 검사 | 1건, normal |
| 교정 승인·재학습·교정 후 평가 | 모두 0건 |
| 최종 판정 | HOLD, `demo_success=false` |

요청 모델은 `nvidia/nemotron-3.5-super-vl-120b-a12b`이다. 첫 성공 응답의 반환 모델명은 `private/nvidia/nemotron-3.5-super-120b-a12b`였다. 두 이름을 별도 필드로 보존했으며, 이 실행에서 이미지 입력은 사용하지 않았다.

## 증거와 자원 기록

봉인된 판단 대기 상태와 같은 API 원장에서 실행을 완료했다. 완료된 학습과 평가는 반복하지 않았다. HOLD가 확정된 뒤에는 추가 모델 호출이나 교정 요청을 만들지 않았다.

[원본 해시와 공개본의 대응](../evals/results/brev_recovery_v4_s2196_20261024/provenance.json)은 회수한 원본 묶음·개별 파일 해시와 공개 파일 해시를 구분한다. 공개 JSON에는 사용자 경로와 비공개 파일 경로를 넣지 않았다.

원본 250개 파일을 회수하고 해시를 확인한 뒤 VM을 Stop했으며, STOPPED 상태를 확인했다.

[자원 기록](../evals/results/brev_recovery_v4_s2196_20261024/resource.json)의 견적 기준 실행 비용은 약 $3.55다. 설치·대기·회수 구간을 포함하며, 공급자의 실제 청구액과 저장소 과금은 확인하지 않았다.
