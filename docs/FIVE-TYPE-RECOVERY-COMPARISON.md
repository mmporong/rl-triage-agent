# 다섯 정책 비교에서 이어진 회복 고리

등록된 네 고리의 최종 결과는 회복1개·HOLD3개다. 원래 원천은 F1과 복합 결함의
두 계보이며, 단일 복합 모델 경로의 회복을 전체 선택기 우위로 해석하지 않는다.

## 무엇을 이어서 확인했나

[다섯 정책 진단 비교](FIVE-TYPE-DIAGNOSIS-COMPARISON.md)는 새 원천 정책 10개에
정책 5개를 적용해 50행을 종결했다. 기술적으로 비교 가능한 결함 7쌍에서 Nemotron
3.5 수치 경로와 고정 순서는 각각 원인군 3개를 확인했고, 등록 판정은
`inconclusive`였다. 두 방식이 함께 확인한 3개 사례에서는 짝 검사를 각각 3회와
9회 사용했다. 이 하위집합은 전체 성공률이나 GPU 시간 비교가 아니다.

그 비교에서 사전에 고른 F1과 F1+R1 원천에 고정 순서와 수치 모델 방식을 적용해
네 회복 고리를 만들었다. 각 고리는 서로 다른 FIRST 학습 seed와 checkpoint를 쓴다.
따라서 고정/모델 행의 오차 차이는 같은 입력을 둔 선택기 효과가 아니다.

## FIRST: 네 행 모두 미회복

FIRST는 등록된 첫 교정 뒤 fresh 300 iteration 학습과 26조건 고정 평가를 수행했다.
네 행 모두 `not_recovered`였다.

| 원천/방식 | 학습 seed | 선속도 RMSE (m/s) | yaw RMSE (rad/s) | FIRST |
|---|---:|---:|---:|---|
| F1 / 고정 순서 | 4552 | 0.160730 | 0.294768 | not_recovered |
| F1 / 3.5 수치 | 4553 | 0.149227 | 0.272229 | not_recovered |
| F1+R1 / 고정 순서 | 4556 | 1.154934 | 0.752231 | not_recovered |
| F1+R1 / 3.5 수치 | 4557 | 1.176983 | 0.817835 | not_recovered |

F1 두 행은 오차가 작아졌지만 등록된 회복 기준을 모두 통과하지 못했다. 이 결과를
회복으로 바꾸거나, 남은 결함이 확인되지 않은 F1 행에 새 원인을 붙이지 않는다.

## SECOND: 네 진단과 한 후보

SECOND는 각 행에 최대 세 번의 등록 검사를 적용했다. 고정 순서도 같은 세 검사
상한을 썼다. 수치 모델 경로에서는 모델이 다음 검사를 골랐고, 호스트가 입력 결합과
허용 검사를 검증했다.

| 원천/방식 | 검사 관측 | 진단 종단 | 후속 교정 후보 |
|---|---|---|---|
| F1 / 고정 순서 | P_episode normal → P_noise normal → P_torque normal | HOLD | 없음 |
| F1 / 3.5 수치 | P_episode normal → P_optimizer_updates normal → P_value normal | HOLD | 없음 |
| F1+R1 / 고정 순서 | P_episode normal → P_noise normal → P_torque normal | HOLD | 없음 |
| F1+R1 / 3.5 수치 | P_episode normal → P_value normal → P_optimizer_updates abnormal | validated candidate | PPO epoch 복원 |

같은 세 검사 상한에서 고정 순서는 복합 사례의 남은 R1을 확인하지 못했다. 이 한
개발 사례는 모델이 선택한 갱신 횟수 검사로 후속 후보를 만든 기록이다. 모델 우위,
새 유형에 대한 일반화, 3.5 버전의 개선률로 확대하지 않는다.

benchmark 후보 승인은 `scoring.correct_any_family`, `original_faults`,
`remaining_faults` 대조도 함께 소비한다. 이 정답은 모델 입력에 들어가지 않았다.
따라서 이 실행을 정답이 없는 사용자 자료의 자동 수리로 설명하지 않는다.

갱신 횟수 검사의 raw record 첫 세 행은 case `[4, 4, 4]`, reference
`[20, 20, 20]`이었다. 교정 뒤 새 training observer의 첫 세 행은
`[20, 20, 20]`을 기록했다. 원 records SHA-256은 case
`b3e3436729faab67063f008f24076de15569a1f41aca00478a1a546502ce83ac`,
reference
`6568e75facbece6d96b9ec26e012c68a212f0f371a67b166040fe22175c711d7`,
새 observer는
`efb9f83c4475b1cf353d53cf09c2bae8fbf822ca60be95d42a5ffd0d63695a2b`다.

## 최종 고정 평가

유일한 후속 후보는 seed 5557, BOTH fresh 300 iteration 학습과 fixed26 평가를
마쳤다. 결과는 `recovered`였고 네 등록 check가 모두 true였다.

| 최종 행 | 선속도 RMSE (m/s) | yaw RMSE (rad/s) | 평균 생존 (s) | 낙상률 | outcome |
|---|---:|---:|---:|---:|---|
| F1+R1 / 3.5 수치 / epoch 복원 | 0.142376 | 0.290958 | 20.0 | 0.0 | recovered |

네 고리 중 한 고리의 등록 결과다. 독립 원천은 F1과 F1+R1 두 개이고, 회복 행은
복합 원천 한 개다. 3.5 버전 효과, 동일 checkpoint 선택기의 인과 효과, 다른 규칙보다
나은 성능을 뜻하지 않는다. 다른 세 고리는 HOLD로 남는다.

## 실행 수와 현재 상태

- FIRST는 별도 SDK 역할 8개를 사용했다.
- SECOND는 이전 완료 역할 2개와 새 probe 역할 22개, 합계 24개를 보존했다.
- Nemotron 요청은 6회였고 여섯 응답 모두 유효했다.
- V9는 모델이나 probe를 다시 호출하지 않았다.
- 후보 하나의 fresh 300 iteration 학습과 fixed26 평가에 새 SDK 역할 2개를
  사용했다. SECOND 누적은 26이며 등록 cap 32 안이다.
- 최신 회복 평가의 NIM 요청은6회이고 앞선 진단 평가30회와 합쳐36회다.
- V9 학습 재개에서 추가한 API 요청은0회다.

운영 CPU 사전 검증은 후보 네 개와 typed training component 결합을 확인하고
1305.116초에 통과했다. 이 시간은 모델 요청이나 GPU 학습의 가속 수치가 아니다.
이전 실패 원문 902개와 누적 parent wall 10783.745328초도 별도로 보존됐다. V9의
`second_parent_wall_s` 1865.516초는 새 continuation 범위만 나타내므로 전체 실행
시간 단축으로 해석하지 않는다.

## 이전 비교와 함께 읽는 법

R3의 50행, 비교 가능한 7쌍의 3 대 3, 공통 성공 3건의 검사 3 대 9,
`inconclusive` 판정은 그대로다. 영상 추가는 확인 수를 늘리지 않았고 프레임 사실
인용은 0이었다. 정상 shadow 제안 4건은 호스트가 막았으며 정상 식별 성공으로 세지
않는다. F2 다섯 정책이 놓친 두 사례에서 사후 P_torque 대조가 2/2 abnormal을
관측했지만, 정답을 알고 고른 개발 대조이므로 원래 정책 성과에 합산하지 않는다.

원 leaf receipt SHA-256은
`d166bb71155542f73bbb41f63137e89dca8dfc77d9c89ba550d6743d5cf4a4a1`,
닫힌 실행 lease SHA-256은
`ae2be3105f48d7b6493d2bb65011b2f05e57094ccd44134a3026bea3274116ed`다.
[공개 결과](../evals/results/five_type_recovery_s4552_s5557_20261101_v1/summary.json)와
[검사 관측·갱신 횟수·비용](../evals/results/five_type_recovery_observations_20261102_v1/summary.json)을
별도 manifest와 파일 SHA로 연결했다. 검사 source는 같은 설정으로 새로 수행한
3 update 관측이며 checkpoint의 과거 실행 이력을 가중치에서 추정한 값이 아니다.
FIRST와 SECOND 학습의 첫3회 계측은 각각4회와20회를 기록했다. 300회 전체의
optimizer 횟수를 모두 계측했다고 주장하지 않는다.

새 학습의 owned wall은474.546초, 고정 평가는43.468초였다. 이는 process 시작부터
종료·정리까지의 시간이며 GPU 커널 시간이 아니다. 조건부 회복 영상은 별도 녹화
등록으로 연결하며 수치 회복 판정에 영상을 사용하지 않는다.

## 같은 Play 조건의 정책 재생

[정상·복합 결함·두 교정 뒤 비교 영상](media/five_type_recovery_s5557_20261102_v1/comparison.mp4)은 기존 원천13·14의 정상/복합 정책과 새 회복5557 정책을 가로로 편집한 자료다. 각 원본은625프레임·50fps·12.5초·640×480이며, 카메라·명령1m/s·제어 seed42 조건을 유지했다. 재생 중 결함을 새로 주입하거나 액추에이터를 바꾸지 않았다.

새 회복 정책은 공용 GPU 임대에서1프레임 smoke1회와 전체 녹화1회를 정상 종료했다. 실제 SDK 실행은2회이며 모델/API·새 학습은0회다. render forward625회, 물리 상태 비간섭, native/sidecar/PTS·checkpoint·종료 증거는 원 검증기로 대조했다. 영상의 정상적인 종료·정리와 학습 회복 판정은 별도 근거다.

[정상 원본](media/five_type_recovery_s5557_20261102_v1/normal.mp4)·[복합 결함 원본](media/five_type_recovery_s5557_20261102_v1/compound.mp4)·[두 교정 뒤 원본](media/five_type_recovery_s5557_20261102_v1/after-two-corrections.mp4)은 byte copy이며, [출처 manifest](media/five_type_recovery_s5557_20261102_v1/source_manifest.json)에 SHA를 담았다. `comparison.mp4`는 라벨이 있는 파생 편집물이다. manifest의 GPU0/API0은 CPU에서 수행한 공개 변환 단계의 사용량이며, 앞서 수행한 실제 녹화 SDK2회를0으로 바꾸지 않는다. 원래 capture/model/decision 승인 플래그는false로 보존했다. 이 재생을3.5 버전이나 VL 방식의 우위, 새로운 회복 판정으로 합산하지 않는다.
