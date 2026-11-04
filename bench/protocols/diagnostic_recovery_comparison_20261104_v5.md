# 교정 비교 v5: 완료 학습 보존과 고정 평가 후속 실행

## 실행 전 고정

- v2의 사례·방식·보류 규칙·교정 선택·300 update·고정 평가 밴드·예측·판정 문구를 유지한다. 모델 응답 재호출은 없다.
- v3는 첫 owned 실행 폴더 생성 전 중단됐다. v4에서는 case-05/fixed_sequence의 모터 교정 이어 학습이 완료됐지만 고정 평가는 시작되지 않았다. 두 판의 원문과 비용은 보존한다.
- 완료 학습은 source seed 2664, iteration 299에서 정책·optimizer를 이어받아 598까지 실행했다. owned wall은 515.3279999999795초, 최종 checkpoint SHA는 `3aaeabbc6a00f4fcb34fe13dff12fdadb557a031fc7ff581d6436b4452f5a3e7`다. 환경 전체 상태와 checkpoint RNG를 복원한 실행이 아니다.
- 학습 완료·체크포인트를 확인한 뒤 만든 후속 실행이다. 고정 평가 결과는 아직 없으며, 본 판은 같은 원천에서 교정 효과를 확인하는 후속 비교다. 새 독립 사례로 세거나 이전 학습을 반복하지 않는다.

## 실행 계약

1. 새 `registered_recovery_comparison_plan_v2`와 arm별 `registered_recovery_protocol_v2`를 커밋한 뒤 실행한다. 기존 판은 수정하지 않는다.
2. `case-05/fixed_sequence`의 정확한 진단 manifest·source checkpoint·seed·task·교정·설정·완료 receipt·소비 원장이 proof와 일치할 때만 기존 학습 결과를 참조한다. 새 요청은 고정 평가부터 시작하며 닫힌 이전 요청을 재소비하지 않는다. 다른 방식·같은 checkpoint로 재사용 범위를 넓히지 않는다.
3. 나머지 14개 확정 arm은 등록된 학습과 고정 평가를 실행한다. 모델 보류 1개는 교정하지 않는다. COMPOUND에서 보상 함수의 기존 상태를 보존하고, epoch 교정만으로 모든 결함이 고쳐졌다고 판정하지 않는다.
4. 공용 GPU lease를 유지한다. 새 `registered_desktop_shared_v2`는 같은 시점의 NVIDIA XML·CIM 신원, 명령어, 프로세스 유형을 검증한 등록 데스크톱 C+G 종류만 허용한다. 계산 작업·미등록 신원·불일치·활성 GPU 엔진·정책 상한 초과는 계속 거부한다. 운영 scope에는 fixture callback을 받지 않는다.
5. 원래 SDK 상한 32 중 v4 학습 1회가 소비됐다. 새 상한은 31회, 예측은 29회(학습 14+고정 평가 15)다. CPU checkpoint·telemetry 상한은 각16, 새 전체 owned 상한은63이다. arm wall 상한5000초·API0을 유지한다.

## 비용과 결과 문구

- 새 resource receipt에는 이번 dispatch에서 생긴 자식만 센다. 재사용 arm의 새 SDK는 고정 평가 1회다.
- 전체 교정 effect 비용에는 이전 학습 515.328초와 새 고정 평가 비용을 함께 포함한다. 새 dispatch 비용·이전 학습 비용·전체 effect 비용을 분리한다. 전체 시도 합계에는 이전 학습을 두 번 더하지 않는다.
- 구현·대기 공백은 알고리즘 실행 시간이나 GPU 시간으로 바꾸지 않는다. 기록이 없는 CPU·provider GPU·실청구는 null로 둔다.
- 고정 평가를 통과하면 해당 등록 조건에서 회복, 실패하면 미회복, 근거 부족이나 예산 초과면 unknown으로 적는다. 보류·인프라 중단 때문에 줄어든 실행을 절감이나 우월성 근거로 쓰지 않는다.
