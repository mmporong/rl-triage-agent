# 전송 입력의 교정·후속 관찰 개발 실행

[후속 렌더 감사](../evals/results/vision_render_integrity_audit_20261028_v1/summary.json)는 교정 영상 두 개에서 현재 sim root와 렌더 형상의 불일치를 확인했다. 해당 영상·후속 관찰 이미지는 새 시각 근거 사용을 보류하고 재녹화 대상으로 둔다. 아래 모델 응답과 영상 원문은 보존한다. 시뮬레이션 상태로 계산한 고정 평가와 회복 판정은 이 영상 점검으로 바꾸지 않는다.

## 실행 전 고정한 범위

이 판은 seed 2220·2221의 등록 F1을 사용한 개발 실행이다. 두 정책은 이미
[24입력 비교](../evals/results/vision_transferred_selection_dev_20261027_v1/summary.json)에 사용됐다.
그 비교의 결과를 본 뒤 종단 연결을 준비했다. 처음 보는 결함이나 별도 보류 평가로 세지 않는다.

처음의 비교 전 준비 가드를 만족하지 못한 v1 초안은 미실행 기록으로 보존한다.
v2는 원래 방법 커밋 `3531039`와 비교 사전등록 `8dddefb`에 고정된
`input-03`·`input-07`, 첫 구간의 결합 입력을 그대로 사용한다.
대상·seed·checkpoint·capsule을 비교 응답에 따라 바꾸지 않는다.

## 예측과 판정

실행 전 예측은 두 대상 모두 모델이 `P_reward`를 선택하고, 새 승인 검사에서 NONE은
`normal`, F1은 `abnormal`이며, 보상 함수를 복원한 restart 300회 뒤 고정 평가를 통과한다는 것이다.
이는 검증할 예측이다. 다른 검사 선택, HOLD, 미회복도 결과로 남기며 대상을 교체하지 않는다.

| 단계 | 완료 기준 | 충족하지 못했을 때 |
|---|---|---|
| 첫 선택 | 같은 capsule에 대한 새 E2E 모델 판단과 활성 bridge가 유효 | 등록 피드백 이후에도 무효이거나 미확정 요청이면 HOLD |
| 확인 검사 | 선택이 유효한 `P_reward`이며 새 승인·소비 뒤 실제 두 arm 원문이 NONE normal/F1 abnormal | 다른 검사 선택·검사 미확정이면 교정 요청 없음 |
| 교정 | 현재 선택·검사·checkpoint에 결합된 승인 한 번을 소비 | 승인 재사용·다른 계보이면 실행 거부 |
| 재학습·평가 | 각 대상 CORRECTED 4096 환경×300회, 기존 26조건·회복 밴드 통과 | `not_recovered` 또는 실행 근거 미확정으로 기록 |
| 교정 후 녹화 | 검증된 새 checkpoint에서 같은 Play의 AABA·625프레임·PTS·정리 통과 | 영상·후속 관찰 완료로 표시하지 않음 |
| 보조 관찰 | 실제 프레임·수치 ID 인용 계약을 만족하고 원문·요청 해시를 기록 | 관찰 무효/HOLD로 기록, 고정 평가 판정은 유지 |

회복은 고정 평가로 판정한다. 두 사례의 종단 완료·회복·관찰 유효 여부를 각각 보고한다.
모델은 첫 검사와 근거 ID를 선택하고, 확인된 등록 결함의 교정과 restart는 호스트 규칙이 정한다.
후속 영상 관찰은 `used_for_decision=false`다. 같은 seed의 restart checkpoint가 NONE과
바이트까지 같으면 그 사실도 기록한다.

## 실제 종단 결과

[공개 결과](../evals/results/vision_transferred_e2e_dev_20261027_v1/summary.json)에서
Nemotron 3.5 Super VL의 첫 판단은 두 seed 모두 유효했고 `P_reward`를 선택했다.
모델이 고른 근거 ID는 telemetry 항목뿐이었다. 새 승인 검사 두 쌍은 각각 NONE
`normal`, F1 `abnormal`이었다.

첫 교정 실행에서 남은 두 `unknown` receipt는 원문 그대로 보존했다. 이미 끝난 300회
학습은 두 seed 모두 다시 실행하지 않았다. completed-training resume에서 seed 2220은
새 paired 고정 평가를 한 번 실행했고, seed 2221은 이미 완료된 평가 원문을 재검증했다.
기존 26조건 평가의 aggregate 판정은 두 seed 모두 `recovered`였다. 두 restart 300회
checkpoint는 각각 같은 seed의 NONE checkpoint와 바이트까지 같았다. 이는 등록 F1에서
정상 학습 경로가 복원됐다는 개발 근거이며, 다른 결함이나 처음 보는 seed의 결과로 확대하지 않는다.

검증된 교정 checkpoint 두 개는 같은 Play 설정으로 각각 625프레임·50fps native 녹화를
마쳤다. [종단 manifest](../evals/results/vision_transferred_e2e_dev_20261027_v1/e2e-manifest.json)는
첫 선택, 검사, 승인·소비, 학습 재개, 고정 평가와 영상 계보를 한 묶음으로 연결한다.
[재생 화면](replay/transferred-vision-e2e-v1.html)은 이 순서를 보여 준다. 후속 영상 관찰은
고정 평가의 회복 판정에 사용하지 않는다.

VM이 `STOPPED`가 된 뒤 WSL에서 실행한 Nemotron 3.5 Super VL 보조 관찰은 2/2
`observation_ready`였다. 두 교정 영상 모두 `walking=yes`, `in_place=no`, `falling=no`였고,
인용한 프레임 1~6의 ID와 시각이 유효했다. 이 단계는 API 2/6회를 썼으며 원격 보조 호출은
0회였다. 첫 판단 2회까지 합친 E2E 원장은 4/32회다. `used_for_decision=false`이므로 이
묘사는 고정 평가 판정이나 검사 선택을 바꾸지 않는다.

이 결과는 등록 F1과 이미 비교에 사용한 seed 2220·2221을 연결한 개발 실행이다. 앞선
24입력 비교의 판정은 계속 `observation_only`이고 선택 변경은 0이다. 영상이 검사 선택을
개선했다거나 모델이 규칙보다 낫다는 근거로 사용하지 않는다.

## 자원과 재현 경계

- 새 source 학습은 0건, CORRECTED 학습은 최대 2건이다. source 네 건과 합쳐 주 계획의 학습 상한 6건을 유지한다.
- 비교 API 72회 몫을 재사용하지 않는다. 새 E2E 원장과 WSL 중계 원장의 상한은 각각 같은 실제 요청 32회를 덮는다. 이 둘의 요청 수를 더해 중복 집계하지 않는다.
- 같은 L40S 설치 디스크를 새 Start 뒤 재사용한다. 자원·GPU UUID·유휴 원문은 해당 boot에서 새로 확인한다.
- 최초 실행 배정은 $5였다. 남은 native 녹화·산출물 보존과 Stop 지연을 포함한
  [v6 자원 등록](../bench/protocols/vision_transferred_native_stop_reserve_20261027_v6.json)은
  종료 준비 reserve $0.5를 더해 이 lifecycle 상한을 $5.5로 고정했다. 실제 견적은
  $5.198280, 이전 단계를 포함한 누적 견적은 $33.409058이며 프로그램 상한은 $35였다.
  VM은 `2026-10-27T15:13:17Z`에 `STOPPED`로 확인됐다. 이 값은 청구서 확정액과 구분한다.
- API 키는 기존 WSL 환경에 둔다. Brev caller의 write-once 요청을 WSL에서 검증해 호출하고, 같은 요청 ID·SHA에 결합된 응답만 전달한다. pending 요청은 재전송하지 않는다.
- 체크포인트 경로를 담은 bound 녹화 protocol은 비공개로 보존한다. 공개 Git에는 template와 bound 원문·두 checkpoint·교정 결과·승인 receipt의 해시를 봉인하고, 실제 녹화 전에 그 commit을 검증한다.

공개 결과에는 모델 유효율, 선택·HOLD, 새 검사 결과, 교정 승인·소비, 회복, 후속 관찰,
API 사용과 실제 실행 시간을 분리한다. 이 실행을 영상의 검사 선택 개선 근거로 사용하지 않는다.
## native 녹화 직전 자원 확인

긴 입력·승인 검증이 끝난 뒤 [짧은 handoff](../evals/vision_corrected_native_handoff_v2.py)가 120초 ticket을 봉인했다. 호스트는 ticket 원문 SHA를 별도로 보관하고, 녹화 직전에 얻은 공급자·GPU·유휴 원문으로 기존 30초 입장 검사를 통과시켰다. native 결과를 검증한 뒤에만 고리 cache에 기록했으며, 두 교정 정책의 녹화가 이 경계를 통과했다. 후속 모델 관찰은 cache를 다시 검증해 사용하되 회복 판정에는 관여하지 않는다.
