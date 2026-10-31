# Nemotron 3.5의 추론 요청 모드를 바꾼 개발 비교

같은 입력에서 근거 조회나 검사를 고른 첫 응답은 기존 요청 설정5/12, 추론 활성화 설정11/12였다. 양쪽 모두16/16 응답이 유효했다. 이어서 각 사례의 첫 번째 실제 응답을 검사 고리에 연결한 개발 비교에서는 올바른 범주 확정이2/6과5/6이었다. 후자의2건은 복합 결함의 일부 확정이며, 새 교정·회복은 실행하지 않았다.

## 무엇을 바꿨나

기존 호출은 `extra_body.reasoning_effort="none"`이었다. [NVIDIA Build의 같은 hosted 모델 예시](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b)는 `chat_template_kwargs.enable_thinking=true`를 사용한다. [NIM 모델별 문서](https://docs.nvidia.com/nim/large-language-models/2.0.13/get-started/advanced/get-started-nemotron-3.5-super-vl.html)는 `low`, `medium`, `high` 추론 강도를 설명한다. 자체 배포 NIM 문서가 hosted API의 동작까지 보장하지는 않으므로 실제 반환값으로 사용 여부를 확인했다.

| 항목 | `legacy_none` | `enabled_low` |
|---|---|---|
|요청 모델|같은 Nemotron 3.5 Super VL|같은 모델|
|추론 관련 요청|`reasoning_effort=none`|`enable_thinking=true`, `reasoning_effort=low`|
|나머지 요청|temperature0, max_tokens16384, stream=false|동일|
|메시지|case별 동일한 원문과 canonical digest|동일|
|피드백·재시도|0|0|

모델을3에서3.5로 바꾼 비교가 아니다. 같은3.5 endpoint에 보낸 요청 모드의 비교다. 추론과 최종 답변은 출력 한도를 함께 쓰므로, `finish_reason`과 최종 JSON 유효성을 별도로 확인했다. 추론 본문은 판단 근거나 공개 산출물로 쓰지 않는다.

## 입력과 실행 순서

[사전등록 v2](../bench/protocols/nemotron_reasoning_options_development_20261101_v2.json)는 이미 본 seed2320·2321의 정상·F1·R1·복합 조건8개를 사용한다. 각 입력에서 두 모드를 두 번씩 호출해 총32회다. 순서는 case마다 `none → low → low → none`으로 고정했다. 두 반복이나 같은 seed 안의 결함 조건을 독립된 새 사례로 세지 않는다.

이 비교의 정상2개는 모델의 제안만 관찰하는 대조다. 원래 진단 고리의 정상 host gate는 그대로이며 실제 검사·교정 권한은 없다. 모델에는 gate 판정이나 결함 정답을 보내지 않고 동일한 O2 메시지만 보냈다.

첫 실행의 요청 전 해시 검사 무효 기록은 API0회로 보존했다. 메시지 파일의 바이트 SHA와 canonical 메시지 digest를 분리한 v2에서만 측정을 수행했다.

## 실제 결과

[공개 결과](../evals/results/nemotron_reasoning_options_development_20261101_v1/summary.json)의 값은 원문32개와 요청 intent, mode별 인자, 입력 SHA, O2 검증 결과를 대조해 계산했다.

| 지표 | `legacy_none` | `enabled_low` |
|---|---:|---:|
|전체 단일 응답 유효|16/16|16/16|
|결함 입력의 조회·검사 제안|5/12|11/12|
|결함 입력의 조회 제안|2/12|9/12|
|결함 입력의 검사 제안|3/12|2/12|
|결함 입력의 HOLD|7/12|1/12|
|정상 입력의 검사 제안|0/4|0/4|
|정상 입력의 조회 제안|0/4|3/4|
|양의 추론 토큰 반환|0/16|16/16|
|평균 API 응답 구간(s)|3.68|19.95|
|입력 토큰 합|73,850|73,850|
|출력 토큰 합|473|42,290|
|입력·출력 토큰 합|74,323|116,140|

추론 활성화의 변화는 주로 곧바로 멈추는 대신 **등록된 optimizer 기록을 읽겠다는 제안**이었다. 검사 제안 자체는3회에서2회로 줄었다. 정상 대조에서도 조회가 늘었으므로, 조사 행동이 많다는 이유만으로 더 정확하거나 효율적이라고 해석하지 않는다. 실제 시스템에서는 정상 gate가 이런 호출과 검사를 막는다.

반환된 추론 토큰은 기존 설정에서16회 모두0, 활성화 설정에서16회 모두 양수였다. 활성화 설정은 답변 대기와 출력 토큰 사용도 늘었다. 위 시간은 이32요청의 관찰값이며 전체 진단 시간·탐지 속도·GPU 커널 시간은 아니다.

## 실제 조회·검사 뒤의 확정 비교

[후속 사전등록](../bench/protocols/reasoning_diagnosis_study_20261101_v1.json)은 위32회 중 각 case·mode의 첫 반복 응답을 고정했다. 이미 응답을 본 뒤 설계한 개발 비교이며, 좋은 반복으로 대체하지 않았다. 결함6개를 두 모드로 실행하고, 정상2개는 두 모드 모두 기존 host gate에서 끝냈다. 첫 응답은 새로 호출하지 않았으며 이후 판단만 같은 모드로 호출했다.

[공개 결과](../evals/results/reasoning_diagnosis_study_development_20261101_v1/summary.json)는 원응답·후속 요청과 응답·승인·검사 영수증·상태 전이를 대조한다. 두 모드의 도구, 입력 checkpoint, 정상 참조, 행동4회·검사3회 상한은 같다. HOLD를 강제로 검사로 바꾸지 않았다.

| 지표 | `legacy_none` | `enabled_low` |
|---|---:|---:|
|결함 입력|6|6|
|올바른 범주 확정|2|5|
|그중 복합 결함의 일부 확정|0|2|
|HOLD|4|1|
|오확정|0|0|
|추가 모델 호출|2|6|
|승인된 paired 검사|3|6|
|새 Isaac 실행|4|2|
|정상 host gate 이후 모델·검사|0|0|

| source | `legacy_none` | `enabled_low` |
|---|---|---|
|2320 F1|HOLD|기록 횟수 정상 후 HOLD|
|2320 R1|optimizer-update-count 확정|같은 하위형 확정|
|2320 복합|HOLD|optimizer-update-count 일부 확정|
|2321 F1|reward 확정|reward 확정|
|2321 R1|reward 검사 정상 후 HOLD|기록 횟수 검사로 optimizer-update-count 확정|
|2321 복합|HOLD|optimizer-update-count 일부 확정|

2321 R1에서 기존 설정은 새 reward 검사를 한 쌍 실행한 뒤 HOLD였다. 추론 활성화는 보관된 갱신 기록을 조회하고 CPU 횟수 검사를 선택해 확정했다. 새 Isaac 실행 수는 이 실행에서 관찰한 횟수이며, 원 학습 계측 비용·NVIDIA 서버의 추론 계산·전체 GPU 비용을 측정한 값은 아니다. 추가 모델 호출은 오히려 늘었다.

원32회와 후속8회는 별도 호출이다. 후속 표의 시간은 필요할 때 기록된 첫 응답 이후의 부모 실행 구간으로 읽어야 하며, 첫 API 대기까지 포함한 연속 종단 시간으로 해석하지 않는다. 원문의 `sdk_role_executions=18`은 paired9회에 역할2개를 곱한 보수 집계다. 공개본은 영수증을 확인해 기록 재사용6쌍의 SDK0회와 새 검사3쌍의 SDK6회를 분리했다.

## 채택 범위와 다음 검사

사전등록한 개발 후보 조건을 통과했으나 기본 모드는 바꾸지 않았다. 같은 두 seed의 등록 결함을 본 뒤 만든 판이므로, 새로운 seed에서 설정과 예측을 고정해 확인해야 한다. 단일 응답 두 반복이나 같은 seed의 여러 결함을 독립된 새 사례로 세지 않는다. 기존3 모델 대비 우위, 영상에 따른 개선, 새 회복의 근거도 아니다.

2320 F1은 두 모드 모두 HOLD였다. 이를 가른 [별도 사전등록 감사](../bench/protocols/diagnostic_coverage_audit_20261101_v1.json)에서 운영자가 고정한 `P_reward`를 한 쌍 실행했다. [결과는 abnormal](../evals/results/diagnostic_coverage_audit_c02_20261101_v1/summary.json)이었고, xy 보상 상대오차는 case0.319069·reference1.70127e-8이었다. 거의 정지한 정책이라 검사가 normal일 수 있다는 사전 가설은 기각됐다. 이 사례는 등록 검사로 구분할 수 있었으나 모델이 그 검사를 선택하지 않고 보류한 경우다.

감사는 모델 호출0회·새 Isaac 실행2회였으며, 선택 주체는 `operator_registry_coverage_audit`다. 기존 모델의 HOLD와5/6 집계는 바꾸지 않는다. 다른 검사의 결과나 새 교정·회복도 추정하지 않는다.

[후속 제안 기능](../evals/model_hold_followup_v1.py)은 이런 유효 HOLD를 보존하면서 남은 예산의 등록 검사만 제안한다. 부모 결과 SHA, 닫힌 상태와 직전 상태, 모델 응답, source·checkpoint를 검증한다. c02에서는 `P_reward`, c07의 기존 설정에서는 갱신 기록 조회를 CPU에서 제안했다. 승인·실행·교정·자동 재개 권한은 모두 false다. 이는 원래 모델 선택과 분리한 기능이며, 실행 효과를5/6 결과에 더하지 않는다.
