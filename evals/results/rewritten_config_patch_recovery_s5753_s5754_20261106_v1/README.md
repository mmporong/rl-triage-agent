# 기록된 재작성 패치의 새 개발 seed 행동 평가

결과: **recovered** (seed 5753: recovered, seed 5754: recovered).

이미 기록된 CPU 모델 응답의 재작성 소스를 재사용했으며, 이 GPU 평가에서 새 모델 API 호출은 0회였다. 결함 어댑터가 생략한 값에는 epoch뿐 아니라 gamma와 entropy 계수도 포함되므로, 결과를 epoch 하나의 인과 효과로 설명하지 않는다. 정상 대조, 원 결함 재현 여부, 패치 정책의 Fixed26 판정을 seed별로 분리해 보존했다.
