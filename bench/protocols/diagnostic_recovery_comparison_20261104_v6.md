# 교정 비교 v6: 완료 단계 가져오기와 제한된 GPU 입장 대기

## 유지하는 실험 범위

- v2의 사례4개·방식4개·전체16arm, 회전 순서, seed, 300 update, 교정 선택, 고정 평가 밴드, 모델 보류 규칙·예측·판정 문구를 유지한다. 모델 호출은0이다.
- v5는 두 arm 뒤에 중단됐다. case-05/fixed_sequence는 이전 학습515.328초와 고정 평가47.078초를 완료해 not_recovered였다. case-05/static_rule은 별도 학습518.516초를 완료했지만 고정 평가 전에 GPU 사용률2%가 등록 상한1%를 넘었다. 원문·중단·요청·비용을 보존한다.
- 이 판은 이미 본 완료 효과와 학습을 가져오는 인프라 후속 실행이다. 새 독립 표본으로 세거나 기존 부정 결과를 대체하지 않는다. 첫 완료 효과를 다시 실행하지 않고, static의 학습을 반복하지 않는다. 최종 checkpoint 바이트가 같아도 다른 방식의 고정 평가 보고서를 가져다 쓰지 않는다.

## 두 종류의 명시적 가져오기

1. 완료 효과: case-05/fixed_sequence의 원 진단 세션·계획·등록·source checkpoint·학습·고정 평가·최종 checkpoint·승인·소비·완료 receipt·자원·후속 판단을 다시 검증한다. source closure는 실행 당시 commit의 Git blob과 대조한다. 검증된 효과를 새 결과에 투영하되 새 승인·dispatch·SDK 실행을 만들지 않는다.
2. 완료 학습: case-05/static_rule의 원 세션·source checkpoint·seed·task·교정·설정·300 update·Windows Job 자연 종료·원 소비 요청·닫힌 unknown 상태와 최종 checkpoint를 다시 검증한다. 새 요청에서 fixed 단계만 실행한다. 이전 요청을 재소비하지 않으며, fixed 결과가 생기기 전 회복 라벨을 만들지 않는다.
3. 나머지13개 확정 arm은 학습과 고정 평가를 실행한다. 기존 HOLD1은 개입하지 않는다. 공개 plan은 두 proof의 논리 identity와 SHA만 포함하고 물리 경로는 private resolver에 둔다.

## GPU 입장

- 기존 profile과 상한을 바꾸지 않는다. 새 opt-in `registered_desktop_shared_wait_v1`만 사용한다.
- 거부 사유가 사용률 상한 초과 하나뿐일 때 최대60초·최대12조회·조회 사이2초 간격으로 기다린다. 매번 원 production reader로 live lease와 같은 fresh XML/CIM 신원을 확인한다.
- 계산 작업·활성 엔진·신원 불일치·자료 부족·lease 오류·다른 거부 사유는 즉시 거부한다. 제한 시간 뒤의 accepted 관측도 승인하지 않는다. 운영 scope에 fixture callback을 받지 않는다.
- 모든 재조회와 대기 wall·CPU를 남긴다. 대기는 SDK GPU 작업으로 바꾸지 않으며 device-wide sampling과 구분한다. 다른 사용자 프로세스를 종료하지 않는다.

## 예산과 비용

- 원 SDK 상한32 중 학습2회와 고정 평가1회, 총3회를 소비했다. 새 상한29, 예상 새 SDK27(static fixed1 + 나머지13×2), 예상 누적30이다. CPU checkpoint16·telemetry16 상한은 유지하며 새 owned 상한은61이다.
- 16arm 분모를 줄이지 않는다. 완료 효과를 새로운 성공으로 더하지 않는다. 각 방식의 전체 effect 비용은 그 방식의 과거 실행과 새 실행을 한 번씩만 포함한다.
- 완료 효과 import의 새 SDK/CPU 자식0은 실행 부재의 이산 사실이다. 측정하지 않은 새 wall·GPU 시간은 null로 둔다. 기존562.406초를0으로 바꾸거나 새 dispatch 비용으로 포장하지 않는다.
- static의 전체 effect는 기존 학습518.516초와 새 fixed 비용이다. 새 dispatch resource는 이번 fixed만 센다. 전체 시도 집계에는 v3~v5의 준비·거부·자원과 이번 실행을 보존하고 동일 owned execution을 중복 합산하지 않는다.
- 구현 공백·대기 창·실제 활성 명령 시간·owned SDK wall·controller CPU·device-wide energy를 구분한다. provider GPU·실청구·미기록 CPU 값은 null이다. HOLD나 인프라 중단으로 낮아진 비용을 절감 또는 우월성 근거로 쓰지 않는다.
