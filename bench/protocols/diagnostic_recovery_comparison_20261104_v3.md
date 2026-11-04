# 후속 교정 비교: 저장된 모델 HOLD 연결 사전등록 v3

S4a의 미실행 사례 속행과 S4b의 판정·예측·예산은 [v2](diagnostic_recovery_comparison_20261104_v2.md)를 유지한다. 이 판은 S4a v4에서 저장된 모델 HOLD와 검사 원장의 open이 구분된다는 사실을 확인한 뒤 만든 후속 소비 수정이다. 앞선 S4b v1·v2 등록은 실행하지 않은 판으로 보존하며 새 v3 계획을 실행 전에 커밋한다.

## 확인한 연결 문제

모델이 같은 검사를 다시 선택하면 검사 원장은 새 요청을 만들지 않고 모델 결과를 HOLD로 반환한다. 그러나 원장에 남은 검사 예산이 있으면 status는 open이다. S4a는 이를 model_hold로 마감했지만 S4b 소비기는 검사 원장의 status만 읽어 diagnostic_not_terminal 재개 대기를 만들 수 있었다. case-07 반복 에이전트의 저장된 응답에서 이 차이를 확인했다. 모델 응답을 바꾸거나 재호출하지 않는다.

## 실행 전 고정하는 수정

- 검사 원장의 open과 실제 저장된 모델 terminal_hold가 함께 있을 때만 원래 모델 validator를 저장 응답으로 재생한다. 새 transport를 제공하지 않으며, 저장된 terminal 뒤 새 API 예약을 거부하는 기존 원장을 사용한다.
- context·messages·params·응답 영수증·terminal 재계산 일치가 확인되어 model_hold를 반환한 경우에만 후속 상태를 model_hold로 읽는다. 새 원인 확정·검사 요청·승인·GPU 실행을 만들지 않는다. pending 검사·terminal 없음·choice_ready는 바꾸지 않는다.
- 변조·불일치·미완료 API 원장은 거부한다. 원본 terminal·응답·검사 원장과 과거 결과 파일은 보존한다.
- 같은 함수를 S4b 준비기와 controller가 사용한다. 기존 core seed 검사 helper와 그 테스트도 새 source closure에 포함한다. 이 helper는 기존과 같이 접근 금지 provider subtree를 메타데이터 검사·하위 열거 전에 제외한다. 기존 제외 경계·seed 판정은 바꾸지 않는다.

## 유지하는 비교와 결과 문구

case-05~08 × 네 방식의 계획 분모는 16이다. 원인이 확인된 arm만 등록 교정으로 이어진다. 모델 HOLD arm은 회복 시도 0·HOLD로 남기고, 회복 실패나 성공에 합치지 않는다. 비용이 적어도 문제를 해결하지 못한 HOLD를 절감 성공으로 부르지 않는다.

모터 이어 학습, epoch 교정 재학습, 300갱신, 기존 seed, Simulator 최대32·CPU checkpoint16·CPU telemetry16·전체 owned64, arm당5000초, API0을 유지한다. COMPOUND에서는 보상 결함을 보존하므로 epoch 교정 후 행동 실패가 남을 수 있다는 예측을 유지한다. 원천·S4a·과거 실패 비용은 별도로 한 번씩 보고한다.
