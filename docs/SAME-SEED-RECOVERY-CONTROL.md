# 같은 seed 정상 대조로 확인한 회복 판정

보상 결함을 고친 뒤 학습한 seed 3221 정책은 원래 고정 평가에서 `not_recovered`였다. 별도로 같은 seed의 정상 정책을 300회 학습하자, 기존 교정 정책과 checkpoint 바이트 및 26조건 평가 수치가 모두 같았다. 정상 대조도 기존 seed 2221 참조에 대한 회전 오차 기준을 넘었다.

이 결과는 교정 코드가 남아서 회복에 실패했다는 해석과 구분해야 한다. 이번 조건에서 교정 실행은 같은 seed의 정상 학습 산출물을 재현했다. 원래 판정은 유지하며, 이 대조군을 새 모델 회복 성공에 더하지 않는다.

## 결과 전에 고정한 비교

seed 3221의 이전 결과를 본 뒤 설계한 개발 대조 실험이다. [사전등록](../bench/protocols/local_same_seed_none_control_s3221_v1.json)은 source commit `01f67a7`을 결합했고, 등록 commit `a84d0a3` 뒤에 학습을 시작했다. 학습은 NONE arm, source checkpoint 없음, seed 3221, 4096환경, 300회, 환경당 24스텝, action scale 0.25였다. 자동 재시도와 추가 모델 호출은 없었다.

| 등록 비교 | 판정 규칙 | 실측 |
| --- | --- | --- |
| 정상 3221 / 기존 정상 2221 | 기존 회복 기준을 그대로 적용 | 기준 미통과: yaw 비율 1.619850 > 1.1719 |
| 기존 교정 3221 / 새 정상 3221 | 같은 기준을 적용하는 보조 비교 | 통과: 선형·회전 오차 비율과 생존 비율 각각 1.0 |
| 위 두 비교의 결합 | 첫 비교 실패·두 번째 통과이면 `seed_or_command_coverage_consistent` | 해당 분기, `causal_claim=false` |

다른 두 분기도 결과 전에 고정했다. 정상 3221이 기존 참조를 통과하고 교정 3221이 새 정상 대조와의 비교에 실패하면 `correction_path_influence_possible`, 그 밖이나 참조 무효이면 `inconclusive`다. 같은 seed의 독립 학습이 기존 교정 checkpoint와 바이트까지 같을 가능성은 허용하되, 실제 300회 learn 반환·NONE manifest·재개하지 않은 설정·새 실행 증거를 요구했다.

## 세 정책의 고정 평가

모든 정책은 eval seed 2026, 동일한 속도 명령 26개, 1040환경, 1000스텝으로 평가했다. 기존 두 정책의 결과는 보존된 원문을 사용했고 정상 3221만 새로 평가했다.

| 정책 | 선형 속도 RMSE (m/s) | 회전 속도 RMSE (rad/s) | 평균 생존 (s) | 낙상률 |
| --- | ---: | ---: | ---: | ---: |
| 기존 정상 NONE 2221 | 0.148631697 | 0.177752913 | 20.000000000 | 0 |
| 기존 교정 CORRECTED 3221 | 0.144703292 | 0.287933012 | 19.989365234 | 0.000961538 |
| 새 정상 NONE 3221 | 0.144703292 | 0.287933012 | 19.989365234 | 0.000961538 |

교정 3221과 새 정상 3221의 checkpoint SHA256은 모두 `d370e76c2184b86b108c80318081c5d5b6bf3c59c7846e6c2c48b2a1a46bab5a`다. 실행 폴더는 다르고 새 정상 학습은 source checkpoint를 읽지 않았다. 새 정상 학습의 learn 시간은 454.014739초였다.

저장된 환경 설정은 같았고, 학습 설정 차이는 실행 이름 하나였다. 보상 함수 `track_lin_vel_xy_exp`의 최종 코드 SHA도 두 실행에서 같았으며, 각각의 readback에서 남은 결함은 없었다. 저장된 설정 비교는 [별도 파일](../evals/results/local_same_seed_none_control_s3221_20261030_v1/configuration_comparison.json)에 있다.

## 확인한 범위와 다음 판단

- **사실:** 이 seed와 환경에서 교정 정책은 새 정상 학습과 같은 산출물이다. 기존의 다른 seed 참조와 비교하면 정상 정책도 회전 기준을 통과하지 못한다.
- **추론:** 관찰된 미회복은 seed 또는 학습 중 명령 커버리지 차이와 일치한다. 이 실험만으로 어느 요소가 회전 약점을 만들었는지는 확정하지 못한다.
- **유지하는 판정:** [원래 RI213](AGENTIC-LOCAL-F1-RECOVERY.md)의 `not_recovered`와 임계값을 바꾸지 않는다. 보조 비교 통과를 모델의 새 회복 성공이나 영상의 선택 이득으로 계산하지 않는다.
- **후속 기능:** 회복 판정에 참조 seed 차이와 정상 대조 결과를 함께 제시한다. seed가 다른 정상 참조 하나를 쓸 때의 변동 범위는 새 seed의 정상 실행들로 검증한 뒤 고정해야 한다. 이번 결과를 본 뒤 임계값을 넓히지는 않는다.

## 근거와 실행 경계

[결과 요약](../evals/results/local_same_seed_none_control_s3221_20261030_v1/summary.json), [78조건 행](../evals/results/local_same_seed_none_control_s3221_20261030_v1/conditions.csv), [소스·원문 결합](../evals/results/local_same_seed_none_control_s3221_20261030_v1/provenance.json), [공개 파일 SHA](../evals/results/local_same_seed_none_control_s3221_20261030_v1/manifest.json)에 수치를 보존했다. 78행은 세 정책의 조건별 기록이며, 새 학습과 새 평가는 각각 한 건이다.

[실행기](../bench/local_same_seed_none_control_v1.py)는 실제 repo/Lab/package 소스 결합을 확인하고 공용 GPU 임대 안에서 학습과 평가를 순차 실행했다. NONE 전용 [종료 검증](../bench/local_same_seed_none_terminal_v1.py)은 학습 완료 영수증과 SDK 종료 순서를 확인한다. Python 3.10 관련 15개 테스트와 Ruff가 통과했고, 실제 실행도 exit 0으로 끝났다. 유료 Brev와 추가 API는 사용하지 않았다.
