# 영상 근거로 다음 검사 고르기

영상 선택 접속부는 같은 진단 상태를 세 입력으로 준비한다. 모델은 등록된 검사 하나와
화면에 주어진 근거 ID를 고른다. 선택이 유효하면 기존 실행기가 검사 인자와 승인 후보를 만든다.
영상만으로 원인을 확정하거나 교정을 요청하는 경로는 없다.

| 입력 | 모델이 받는 관찰 자료 | 공통으로 받는 내용 |
| --- | --- | --- |
| `text_only` | 같은 Play의 시각·명령·운동 수치와 학습 기록·완료된 검사 사실 | 원인 후보·실행 가능한 검사·검사 비용·남은 예산 |
| `video_only` | 같은 Play의 시각·명령·운동 수치와 실제6프레임 | 동일 |
| `text_and_video` | 같은 Play 수치, 학습 기록, 실제6프레임 | 동일 |

같은 Play의 운동 기록이 없으면 수치는 null이다. 별도로 수행한 고정 평가를 영상 시점의
측정값으로 넣지 않는다. 영상 선택 비교에는 앞선 모델의 행동 요약을 추가하지 않는다.
프레임과 수치는 [영상 입력 계약](VISION-EVIDENCE-CONTRACT.md)으로 검증한다.
명령 좌표계와 운동 좌표계는 입력 계약의 `body`·`world` 값을 그대로 보존하며 좌표 변환을
만들지 않는다. 모델에게는 입력 조건 이름이나 근거 개수 집계를 보내지 않는다.

세 입력은 같은 Play 수치와 missingness를 공유한다. 따라서 비교 대상은 순수 픽셀만의 효과가
아니라, 같은 수치 맥락에 raw-frame 근거 묶음을 추가했을 때 검사 선택이 어떻게 달라지는지다.

## 선택과 실행의 경계

출력은 `action`, `probe`, `fact_ids` 세 필드다. 모델이 실행 인자나 기대값을 덧붙이거나,
보이지 않은 근거를 인용하거나, 실행할 수 없는 검사를 고르면 무효다.
`hold`는 유효한 판단 보류이며 실행 후보를 만들지 않는다. 무효도 HOLD로 끝나고
등록 규칙의 선택으로 바꾸지 않는다.

프레임의 근거 ID는 `vision:frame:f01` 같은 이름이다. 이 인용은 선택 이유를 연결할 뿐,
독립된 물리 검사나 원인 확정 사실로 세지 않는다. 걷는 모습과 속도 명령 추종·회복도 구분한다.

선택 기록은 영상·정책·계보, 현재 진단 상태, 모델 메시지와 선택을 SHA로 묶는다.
영상 바이트나 진단 원장이 바뀌면 이전 기록을 실행에 사용할 수 없다. 검증된 선택만 기존
`selection_comparison_controller_v1.request_spec()`으로 넘기며, 승인과 검사 결과 receipt는
기존 실행 경계를 따른다. 승인 전에는 `approval_granted=false`다.

`seal_bridge()`가 반환한 canonical path와 SHA를 한 reference로 보존해야 한다. 이후
`verify_bridge()`와 `host_request_spec()`은 이 reference 전체를 요구하며 path만 받지 않는다.
상위 caller 원장은 reference SHA를 부모 실행 상태에 결합해야 한다. receipt 파일을 다른
유효 선택으로 다시 만들거나 공백만 바꿔도 원래 reference로는 승인 후보를 만들 수 없다.
symlink와 hardlink receipt, 중복 JSON key, NaN 같은 비표준 수치도 거부한다.

## 구현과 검증 범위

- [`vision_selection_contract_v1.py`](../evals/vision_selection_contract_v1.py): 입력 준비·검증, 모델 메시지, 세 필드 검증과 기존 선택 형식으로의 변환.
- [`vision_selection_loop_v1.py`](../evals/vision_selection_loop_v1.py): 변경 불가 선택 기록, 현재 상태 재검사와 승인 후보 준비.
- [`vision_selection_call_v1.py`](../evals/vision_selection_call_v1.py): 입력·메시지·실행 순서를 SHA로 고정하고 단계별 공유 요청 상한 안에서 판단 기록을 재생한다.

CPU 회귀는 세 입력의 같은 운영 상태와 실행 인자, 프레임·정책 결합, 라벨 누출,
허위 근거·추가 인자 거부, 유효 HOLD와 무효 HOLD, 기록 변조와 오래된 상태 거부를 확인한다.

```powershell
cd "$HOME/rl-triage-agent"
python -m pytest -q tests/test_vision_selection_contract_v1.py tests/test_vision_selection_loop_v1.py tests/test_vision_selection_call_v1.py
```

호출기의 `freeze()`는 현재 입력을 검증해 bundle·실제 이미지 메시지·capsule을 동결한다.
`run()`은 그 바이트와 요청 원장만 읽는다. 완료 응답을 다시 조회해도 새 요청을 보내지 않으며,
미완료 예약은 자동 재전송하지 않는다. `activate()`는 실행 후보를 준비하기 전에 현재 상태와
영상 입력을 다시 비교한다. 그동안 원장이 바뀌었으면 이전 판단을 승인 후보로 사용하지 않는다.

비교 단계는 공유 상한72회, 종단 단계32회, 보류 단계10회이며 논리 판단 하나의 최대 호출은3회다.
실제 네트워크 함수는 제공하지 않는다. 별도 실행기는 모델·소스·사전등록·입력 manifest를
검증한 뒤 provider를 주입해야 한다. 이 접속부의 CPU 검증과 실제 모델·GPU 결과는 별개다.

## 고정한 비교 방법

[비교 사전등록](../bench/protocols/vision_selection_comparison_v1.json)은 새 source seed 두 개의
정상·등록 F1 정책을 대상으로 한다. 각 정책의 두 시간 구간은 같은 계보이며 독립 사례로 세지 않는다.
8입력×3조건의 선택을 보존하고, F1 입력의 `P_reward` 선택 변화와 같은 체크포인트의 승인된
검사 결과를 함께 집계한다. 다른 검사를 골랐다는 이유만으로 오진으로 세지 않는다.

양쪽 source seed에서 등록된 선택 증가 기준을 충족하고, 반대 방향 변화가 없으며 실제 검사로
지지될 때만 `selection_supported`로 쓴다. 그 밖의 결과는 고정한
`observation_only`·`regression`·`inconclusive` 정의를 따른다. 정상 입력에는 유일한 정답 검사를
붙이지 않고 선택·HOLD와 근거 없는 확정·교정 여부를 보고한다.

이 문서는 방법 고정이다. 새 학습 전에 seed·GPU·가격·잔액·비용 상한을 다른 실행 protocol로
고정하고, 모델 호출 전에는 정확한 영상·정책·동기화 기록·24개 capsule 목록을 커밋해야 한다.
기존 G1 개발 영상은 이 비교에 넣지 않는다. 아직 영상의 선택 개선이나 교정 뒤 회복을 측정하지 않았다.
