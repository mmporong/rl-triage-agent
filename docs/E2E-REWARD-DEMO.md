# 새 seed 결함 종단 시연 (보상·모터)

학습 실패 하나를 끝까지 연결해 같은 사례 이름과 요청 ID로 기록한 통합 시연이다.

- **흐름:** 경고 → 고정 평가 → 새 Nemotron 판단 → 승인된 검사 → 원인 확정 → 교정 제안 → 승인된 교정 → 재학습 → 고정 평가
- **보상 결함(F1):**
  - v2(seed 2030): 끝까지 진행했다(`recovered`).
  - NAT 결정 경로(seed 2034): 첫 판단과 교정 제안을 NeMo Agent Toolkit 도구 호출 경로로 냈다(`recovered`).
- **모터 결함(F2):**
  - v3.1(seed 2033): 첫 판단의 틀린 1위(reward)를 승인된 검사가 바로잡고 끝까지 진행했다(`recovered`).
  - v4(seed 2046): 두 모델 결정 단계에 검증 피드백 1회 재요청을 사전등록하고 끝까지 진행했다(`recovered`). 두 결정 모두 1차 응답이 유효해 재요청 경로는 실제로 쓰이지 않았다.
- 같은 입력으로 잰 개발 측정 두 가지를 함께 적는다.
  - NAT 도구 호출 경로 비교(네 판 중 네 번째 판이 채택 조건을 충족했다)
  - 공식 Isaac Lab 디버깅 스킬 유무 프롬프트 비교

이 기록의 범위는 다음과 같다.

- **이미 본 결함 종류(F1 보상의 world frame 속도 사용, F2 모터 토크-속도 곡선)의 새 seed 기록이다.** 결함·probe·교정 등록을 같은 설계자가 만들었다.
- **승인은 사용자가 위임한 운영자 승인이며 사람 승인이 아니다.** 승인자 문자열만 기록하며 신원 인증이 아니다.
- **진단 정확도·교정 성공률·일반화·다른 방법 대비 우위·GPU 절감을 주장하지 않는다.** 시연끼리 합친 분모를 만들지 않는다.

## 요약

| 기록 | 대상 | 결과 | 공개 결과 |
|---|---|---|---|
| 보상 종단 시연 v2 | seed 2030, 교정 재학습 seed 3030 | `recovered`, 시연 성공. 성공 조건 11/11, 사전 예측 6/6 | `evals/results/e2e_reward_s2030_20261013/demo_result.json` |
| 모터 종단 시연 v3.1 | seed 2033, 교정 재학습 seed 3033 | `recovered`, 시연 성공. 성공 조건 12/12, 사전 예측 7/7 | `evals/results/e2e_demo_v3_1_f2_s2033_20261013/demo_result.json` |
| 모터 종단 시연 v4(검증 피드백 1회 재요청) | seed 2046, 교정 재학습 seed 3046 | `recovered`, 시연 성공. 성공 조건 12/12, 사전 예측 7/7. 첫 판단·교정 제안 모두 1차 유효, 재요청 0회(재요청 경로는 쓰이지 않음) | `evals/results/e2e_demo_v4_f2_s2046_20261014/demo_result.json` |
| NAT 결정 경로 보상 종단 시연 | seed 2034, 교정 재학습 seed 3034 | `recovered`, 시연 성공. 성공 조건 12/12, 사전 예측 7/7 | `evals/results/e2e_demo_v3_1_f1_s2034_20261013/demo_result.json` |
| NAT 경로 비교 v4 | seed 2031 모터 결함의 두 모델 결정 입력을 5회씩 | `adopt`. 유효 결정: 첫 판단 5/5, 교정 제안 5/5 | `evals/results/e4b_nat_compare_v4/summary.json` |
| NAT 경로 비교 v1~v3 | 판마다 다른 고정 입력 | 세 판 모두 `do_not_adopt`(아래 "NAT 경로 비교") | `evals/results/e4b_e2e_reward_s2029_20261013/summary.json`, `evals/results/e4b_nat_compare_v2/summary.json`, `evals/results/e4b_nat_compare_v3/summary.json` |
| 스킬 유무 프롬프트 비교 | 이미 본 사례 h02_s2027, 조건별 15회 | 알려진 실패 모드 해당 0/30(포함) 대 2/30(미포함), 기대 행동 26/50 대 24/50. 우위 근거 아님 | `evals/results/skill_ab_v1/summary.json` |
| 결정 반복 일관성(N2) | 시연·개발 packet 11개(진단 6·교정 5), 단일 호출 5회씩 55회(새 측정). NAT v4는 E4b v4 기록 재사용 | 유효 출력의 결정은 반복 내내 같았고, pass^k를 낮춘 것은 계약 위반 출력 6/55. 평균 pass^4 진단 0.60·교정 0.64로 사전등록 하한 0.8 미만(`reconsider`). 정확도·우위 근거 아님 | `evals/results/n2_pass_k_v1/summary.json`, `docs/N2-PASS-K.md` |

## 시연 흐름과 공통 조건

모든 시연은 실행 전에 프로토콜을 커밋했다.

| 시연 | 프로토콜 | 커밋 | LF SHA256 |
|---|---|---|---|
| 보상 v1 | `bench/protocols/e2e_reward_demo_v1.json` | `1a6461e` | `a916049d…` |
| 보상 v2 | `bench/protocols/e2e_reward_demo_v2.json` | `8eade58` | `b6b86b04…` |
| 모터 v3 | `bench/protocols/e2e_demo_v3_f2.json` | `5769970` | `a9a634ef…` |
| 모터 v3.1 | `bench/protocols/e2e_demo_v3_1_f2.json` | `8f4b092` | `1275ec61…` |
| 보상 NAT 결정 경로(v3.1) | `bench/protocols/e2e_demo_v3_1_f1.json` | `2d83b16` | `16d07abd…` |
| 모터 v4(검증 피드백 1회 재요청) | `bench/protocols/e2e_demo_v4_f2.json` | `4c90dcc` | `f166882d…` |

아래 표는 보상 시연 v1·v2의 조건이다. v2에서 바뀐 항목은 "v1에서 v2로 바뀐 점"에, 모터 시연과 NAT 결정 경로 시연의 다른 조건은 각 절에 있다.

| 항목 | 값 |
|---|---|
| 과제·학습량 | Isaac Lab v2.1.1 Go2 flat, 4096 env × 300회. NONE(정상)과 F1(결함)을 같은 seed로 새로 학습 |
| 결함 | F1: 추종 보상이 몸체 좌표계 대신 world frame 속도를 쓴다(`bench/hidden_faults.py` 동결 정의) |
| 교정 재학습 seed | 실패 seed + 1000. 같은 seed로 처음부터 학습하면 정상 학습을 바이트 단위로 재현하므로(v4 seed 2026·2028) 다른 seed를 쓴다 |
| 경고 R1 | `error_vel_xy` 20회 이동 평균이 같은 iteration 정상 참조(seed 2026~2028) 최대값의 2배를 넘는 상태가 iteration 60 이후 10회 연속. 기록만 하고 판정·게이트에 쓰지 않는다 |
| 고정 평가 | `fixed_eval_v1`(26개 명령 조건, 1040 env, 20 s). 판정은 같은 seed 정상 정책 대비 p0c 허용 범위(생존 비 ≥ 0.9, 선속도 오차 비 ≤ 1.1513, yaw 오차 비 ≤ 1.1719, 낙상 ≤ 0.05) |
| 첫 판단 | Nemotron(`nvidia/nemotron-3-super-120b-a12b`) 단일 호출, temperature 0. 사례 이름·정답·결함 정의를 넣지 않는다. 인프라 오류 재시도 최대 1회, 의미 재시도 0회 |
| 검사 고리 | 상위 3가설, 예산 probe 4개. 모델 probe가 상위 3가설을 가르면 그 probe, 아니면 동결 규칙 probe. 관측 뒤 모델을 다시 부르지 않는다 |
| 교정 | 등록 교정 `restore_body_frame_reward`. 회복 방식은 결정적 규칙(보상 결함이고 정지 비율 ≥ 0.9이면 restart)이 정한다. 교정 제안 Nemotron 호출은 검증 결과와 함께 근거로만 원장에 붙는다. 교정 요청 예산 2400 s 이상 |
| 결과 | 결과 범주 14개와 성공 조건 11개의 문구를 사전등록했다. 결과가 `recovered`이고 성공 조건이 모두 참일 때만 시연 성공 |

정지 비율은 같은 고정 평가의 선속도 RMSE를 제자리 정책의 RMSE로 나눈 값이다. 1에 가까우면 명령 추종이 없다는 뜻이며, 정지 자체를 증명하지 않는다.

`P_reward`는 학습 중 기록된 추종 보상 항을 몸체 좌표계 속도로 다시 계산한 값과 비교한다. 지표는 평균 절대 차이를 재계산 값의 평균 절대값으로 나눈 상대 오차다.

## 승인 표기

모든 시연의 승인(검사 요청, 교정 요청)은 사용자가 위임한 운영자 승인이다.
- 승인자 문자열은 `claude_operator_user_authorized`이다.
- 원장 기록, 요청당 1회 소비, receipt는 사람 승인일 때와 같은 방식으로 남는다.
- 결과 파일에는 `approver_identity_attested: false`가 적힌다.
- 저장소의 원장·실행기에는 자동 승인 경로가 없다. 승인은 매번 원장 승인 명령으로 기록됐다.

승인 방식은 시연마다 다르다(운영 기록 기준).

| 시연 | 검사 요청 | 교정 요청 |
|---|---|---|
| 보상 v1·v2(seed 2029·2030) | 운영자가 요청마다 등록 여부·예산·제안 출처·근거를 점검해 운영 메모에 남긴 뒤 승인 | 같은 방식 |
| 모터 v3(seed 2031) | 운영 스크립트가 고리가 제안한 등록 검사를 차례로 승인. 승인 직전 GPU 상태를 메모에 기록했고, 검사별 예산 재확인은 하지 않았다 | 원장 거부로 승인 단계에 가지 않음 |
| 모터 v3.1·NAT 결정 경로(seed 2033·2034) | seed 2031과 같은 방식 | 운영 스크립트가 다음을 모두 확인한 경우에만 승인: 등록 교정, 회복 방식이 규칙 값과 같음, 모델 제안 검증 통과·규칙 일치, 체크리스트 미충족 0, 예산 하한(2400 s) 이상 |
| 모터 v4(seed 2046) | 운영자 세션이 요청마다 등록 probe·예산·1회 소비 조건을 확인한 뒤 원장 승인 명령으로 한 건씩 승인 | 운영자 세션이 등록 교정·예산·1회 소비 조건을 확인한 뒤 원장 승인 명령으로 승인 |

seed 2031·2033·2034의 제안→승인 시각 차이가 0~12 s로 짧은 것은 이 방식 때문이다(seed 2031 `P_torque`의 77분은 운영자 세션 중단). 이 승인은 요청별 검토 시간을 뜻하지 않는다.

모터 v4(seed 2046)의 제안→승인 시각 차이는 원장 기록으로 검사 6~8 s, 교정 8 s였다. 운영자 승인의 기록 시각 차이이므로 요청별 검토 시간으로 쓰지 않는다.

v2·v3·v3.1·v4 사전등록은 승인을 "사용자가 위임한 운영자 승인 또는 사람 승인(승인자 문자열 기록, 신원 인증 아님)"으로 적었다. v1 사전등록은 승인 주체를 사람으로 적었다(아래 "사전등록과 다른 점").

## 시연 v2 결과 (seed 2030)

결과는 **`recovered`, 시연 성공**이다. 사전등록 결과 문구: "새 Nemotron 판단으로 시작한 승인된 검사가 보상 결함을 확정했고, 승인된 restart 교정 정책이 같은 seed 정상 정책 대비 고정 행동 허용 범위를 통과했다."

모든 단계가 실험 커밋 `8eade58` 하나로 기록됐다. 시각은 한국 시간(KST)이며 공개 원장(`evals/loops/h02_s2030/`)에서 읽었다.

| 단계 | 기록 |
|---|---|
| 정상·결함 학습 | 두 학습 모두 300회 정상 종료 |
| 경고 R1 | F1 경고(첫 경고 iteration 135), NONE 경고 없음 |
| 고정 평가 | NONE 보행 통과, F1 unhealthy → 실패 존재 |
| Nemotron 첫 판단 | 유효한 `continue`. 순위 reward > actuator > exploration > optimizer > physics > termination, probe `P_reward`(동결 규칙 대안은 `P_noise`) |
| 검사 요청 `d22f3fc90eb2` | 13:17:22 제안, 13:17:32 승인·소비, 13:17:51 receipt. `P_reward` abnormal(`track_lin_vel_xy_exp` 상대 오차 0.358, 임계 0.05. `track_ang_vel_z_exp`는 1.7e-08) |
| 고리 종료 | reward 확정(검사 1회, actuator·exploration 기각) |
| Nemotron 교정 제안 | `restore_body_frame_reward` + `restart`. 검증 accepted, 체크리스트 미충족 0, 결정적 규칙과 일치 |
| 교정 요청 `99207d20248c` | 13:18:16 제안, 13:18:26 승인, 13:18:32 소비, 13:27:59 receipt `recovered` |
| 교정 재학습(seed 3030)·고정 평가 | healthy(네 검사 모두 통과) |

고정 평가:

| 정책 | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 비율 | 판정 |
|---|---|---|---|---|---|---|
| NONE s2030 | 0.147 | 0.296 | 0 | 20.0 | 0.125 | 보행 통과(기준) |
| F1 s2030 | 1.197 | 0.665 | 0.023 | 19.71 | 1.018 | unhealthy(`err_xy_ratio`·`err_yaw_ratio` 실패) |
| restart 교정 s3030 | — | — | — | — | 0.125 | healthy(생존·낙상·선속도·yaw 검사 통과) |

restart 교정 정책은 공개 결과에 판정과 정지 비율만 남는다.

사전 예측 6개는 모두 적중했다.
- F1이 같은 seed NONE 대비 unhealthy이고 정지 비율 ≥ 0.9
- R1이 F1에 울리고 NONE에는 울리지 않음
- 첫 판단 `continue`, 상위 3가설에 reward 포함
- `P_reward` abnormal 뒤 reward 확정
- 규칙이 restart를 고름
- seed 3030 restart 교정 정책 recovered

예측 적중은 성공 조건과 따로 기록한다.

성공 조건 11개도 모두 참이다. 그중 하나는 "모든 단계가 실행 전에 커밋된 같은 프로토콜과 하나의 실험 커밋에서 돌았다"이다.

비용은 부모 프로세스 벽시계(Isaac 기동·정리 포함)이며 GPU 전용 시간이 아니다.

| 항목 | 실측 | 사전 추정 범위 |
|---|---|---|
| 정상 학습 300회 | 639.0 s | 593.0~802.3 s |
| 결함 학습 300회 | 512.9 s | 453.4~672.3 s |
| 정상 기준 probe 6종 | 87.9 s | 80~180 s |
| `P_reward` 검사 | GPU 벽시계 19.5 s, CPU 25.9 s, 시뮬레이터 51,200 step | 18~110 s |
| Nemotron 첫 판단 | 2.49 s, 입력 2,829·출력 185 토큰 | 진단 1회 |
| Nemotron 교정 제안 | 1.95 s, 입력 1,347·출력 174 토큰 | 교정 1회 |
| restart 교정 학습 | 508.8 s | 453.4~802.3 s |
| 교정 정책 고정 평가 | 48.4 s | 46.2~89.5 s |

교정 요청 receipt의 gpu_s는 557.2(학습과 평가 부모 벽시계의 합)이며, 예산 2400 s 안이다. 승인 대기 시간(제안→승인)은 검사·교정 모두 10 s였다. 운영자 승인의 기록 시각 차이이므로 사람 검토 시간으로 쓰지 않는다.

[비교 영상](media/go2-e2e-e2e_reward_s2030_20261013/comparison.mp4)과 [GIF](media/go2-e2e-e2e_reward_s2030_20261013/comparison.gif)는 세 정책을 왼쪽부터 같은 Play 조건으로 나란히 재생한다.
- 결함 정책(F1 s2030)
- 승인된 restart 교정 정책(s3030)
- 같은 seed 정상 정책(NONE s2030)

재생 조건은 `Isaac-Velocity-Flat-Unitree-Go2-Play-v0`, 재생 seed 42, env 1개, 같은 카메라 위치다. 비교 영상은 5.98초·299프레임, GIF는 6초·90프레임이다. 클립별 원본도 함께 있다.
- [before_fault.mp4](media/go2-e2e-e2e_reward_s2030_20261013/before_fault.mp4)
- [after_restart_fix.mp4](media/go2-e2e-e2e_reward_s2030_20261013/after_restart_fix.mp4)
- [reference_clean.mp4](media/go2-e2e-e2e_reward_s2030_20261013/reference_clean.mp4)

[출처 기록](media/go2-e2e-e2e_reward_s2030_20261013/source_manifest.json)에는 클립별 체크포인트·설정·파일 해시와 실행 정리 검증이 있다. 결함·정상 클립의 체크포인트는 위 고정 평가의 체크포인트와 같다. **회복 판정은 위 고정 평가 수치로 하며, 영상은 판정 근거가 아닌 시각 자료다.**

## 시연 v1 결과 (seed 2029)

결과는 **`correction_unknown`**이다. 사전등록 결과 문구: "승인된 교정 실행이 실패·예산 초과·검증 불일치로 끝나 회복 여부를 판정하지 않았다. 다음 성공 조건을 충족하지 못했다: restart_recovered, costs_recorded"

교정 제안까지는 v2와 같은 경로로 진행했다.
- R1 첫 경고 iteration 136
- 고정 평가에서 F1 unhealthy, 정지 비율 1.005
- 첫 판단은 reward 1위·`P_reward` 제안(규칙 대안 `P_noise`)
- `P_reward` abnormal, 상대 오차 0.289
- reward 확정
- 교정 제안 accepted·규칙 일치

### 교정 단계에서 일어난 일

시각은 KST이며 공개 원장(`evals/loops/h02_s2029/`)과 저장소 커밋 이력에서 읽었다.

| 시각 | 사건 |
|---|---|
| 10:34:48 | 교정 요청 `3f595749f005` 소비, seed 3029 restart 교정 학습 시작 |
| 10:38:42 | 결합 밖 파일 커밋 `63ab9ba` |
| 10:42:05 | 결합 밖 파일 커밋 `aa640d5` |
| 10:44:06 | receipt `unknown`(exit 1, gpu_s 없음) |

[사실] v1의 restart 실행기는 교정 요청에 기록한 커밋과 검증 시점의 HEAD가 같아야 했다. 교정 학습은 300회를 자연 종료했지만, 학습 뒤 검증에서 HEAD가 바뀌어 거부됐다. 그래서 고정 평가를 하지 않았고 교정 결과 파일도 만들어지지 않았다. 묶인 소스 파일 자체는 바뀌지 않았다(v2 프로토콜 `changes_from_v1`에 같은 내용).

[사실] `costs_recorded`가 미충족인 이유는 receipt가 없어서가 아니다. 이 조건은 교정 결과 파일의 restart 학습·평가 시간을 요구하는데, 그 결과 파일이 없었다.

이 실패는 운영 실수였다. v1 실행 문서는 실행 중 결합 밖 파일의 커밋을 막지 않았다.

고정 평가:

| 정책 | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 비율 | 판정 |
|---|---|---|---|---|---|---|
| NONE s2029 | 0.141 | 0.296 | 0 | 20.0 | 0.120 | 보행 통과(기준) |
| F1 s2029 | 1.183 | 0.573 | 0 | 20.0 | 1.005 | unhealthy(`err_xy_ratio`·`err_yaw_ratio` 실패) |
| restart 교정 s3029 | — | — | — | — | — | 판정 없음(고정 평가 미실행) |

v1이 끝난 뒤 운영자가 seed 3029 교정 정책을 같은 고정 평가로 따로 잰 보조 측정이 있다. 이 측정은 사전등록 실행기 밖에서 사후에 한 것이다. 회복 판정·성공 조건·예측 적중에 쓰지 않으며, v1 결과는 `correction_unknown` 그대로다. 공개 결과 파일이 없으므로 이 문서에는 수치를 싣지 않는다.

성공 조건 11개 가운데 9개가 참이고 `restart_recovered`·`costs_recorded`가 거짓이다. v1의 `protocol_committed`는 프로토콜 SHA만 비교해, 단계 기록의 커밋이 셋(`1a6461e`·`bbd2ed5`·`afa9b06`)이어도 참이었다.

사전 예측은 다섯 개가 적중했다. restart 회복 예측은 결과 파일에 `false`로 남았지만, 판정 자체가 없어 적중·불적중 어느 쪽으로도 세지 않는다.

비용(부모 프로세스 벽시계):
- 정상 학습 593.0 s, 결함 학습 528.6 s
- 정상 기준 probe 91.7 s
- `P_reward` 검사 GPU 벽시계 20.9 s
- Nemotron 첫 판단 2.36 s(입력 2,834·출력 212 토큰)
- 교정 제안 2.41 s(입력 1,311·출력 174 토큰)
- restart 교정 학습·평가 시간은 결과에 없다

## v1에서 v2로 바뀐 점

| 항목 | v1 | v2 |
|---|---|---|
| seed / 재학습 seed | 2029 / 3029 | 2030 / 3030 |
| 소스 결합 | 단계마다 그 시점 HEAD를 기록·대조. restart는 요청 시점 커밋과 검증 시점 HEAD가 같아야 함 | 실험 커밋 고정(아래 설명) |
| `protocol_committed` | 단계 기록의 프로토콜 SHA가 하나 | 프로토콜 SHA와 실험 커밋이 모두 하나 |
| 승인 문구 | 사람 승인 전제 | "사용자가 위임한 운영자 승인 또는 사람 승인(승인자 문자열 기록, 신원 인증 아님)" |
| 금지 주장 | 정확도·우월성, 일반화, 교정 성공률, GPU 절감, 이미 본 seed와 합친 분모 | 여기에 더해: v1 사후 보조 측정을 회복 판정으로 인용, 승인자 문자열을 신원 인증으로 표기, 운영자 승인을 사람 승인으로 표기 |
| 결과 스키마 | `e2e_reward_demo_result_v1` | `e2e_reward_demo_result_v2`, restart 결과 `e2e_restart_correction_result_v2` |
| 운영 규칙 | 실행 중 결합 밖 커밋 허용 | 모든 GPU 단계(학습 2회, 고정 평가, 기준 probe, 고리 probe, restart 교정) 실행 중 커밋 금지 |

실험 커밋 고정은 다음과 같다.
- begin이 기록한 HEAD가 실험 커밋이다.
- 이후 모든 단계와 restart 교정(부모의 실행 전·후 검사, 학습 자식의 manifest 검사)은 두 조건이 모두 맞을 때만 진행하고, 실험 커밋을 기록한다.
  - 실험 커밋이 HEAD의 조상이다.
  - 결합 파일 blob의 LF 정규화 SHA256이 두 커밋에서 같다.

v1 실행기와 v1 프로토콜은 v1 공개 결과의 소스 결합을 지키려고 바꾸지 않았다. v2는 `evals/e2e_reward_demo_v2.py`, `bench/demo_restart_correction_v2.py`로 따로 만들었다. 비용 기록 조건(`costs_recorded`)을 포함한 나머지 성공 조건의 뜻은 v1과 같다.

## 모터 결함 종단 시연 (F2)

같은 흐름을 이미 본 다른 결함 종류 F2의 새 seed에 적용했다. F2는 DC 모터 토크-속도 곡선의 속도 항을 5배로 키워 낮은 관절 속도에서 토크가 포화되게 한다(`bench/hidden_faults.py` 동결 정의).

- **v3.1(seed 2033):** `recovered`, 시연 성공. 성공 조건 12/12, 사전 예측 7/7.
- **v4(seed 2046):** 두 모델 결정 단계에 검증 피드백 1회 재요청을 사전등록했다. `recovered`, 시연 성공. 성공 조건 12/12, 사전 예측 7/7. 두 결정 모두 1차 응답이 유효해 재요청은 0회였다.
- **v3(seed 2031):** 교정 요청을 원장에 올리는 단계에서 시연 실행기 결함으로 멈췄다. 결과는 `harness_failure`이다. 이 범주는 v3 사전등록에 없었고 v3.1 도구로 사후 기록했다. 교정·회복은 판정하지 않았다.

### 보상 시연과 다른 조건

| 항목 | 값 |
|---|---|
| 결함 | F2: DC 모터 토크-속도 곡선의 속도 항 5배(낮은 관절 속도에서 토크 포화) |
| 확정 probe | `P_torque`: 낮은 관절 속도에서 토크가 포화되는 비율. 판정선은 3 × max(같은 seed 정상 기준 측정, 0.01)이고 넘으면 abnormal. 아래 시연들은 모두 정상 기준이 0.01보다 작아 판정선이 0.03이었다 |
| 특이성 관문 | 고리가 actuator를 확정한 뒤 결함 정책에서 `P_noise`·`P_value`를 따로 잰다. 둘 다 normal이어야 교정 단계로 가고, 아니면 `confirmation_not_specific`으로 끝낸다. 고리 밖 사전등록 측정이라 승인 원장을 거치지 않는다 |
| 회복 방식 규칙 | actuator → restart(아래 회복 방식 규칙). 보상 결함 규칙(정지 비율 ≥ 0.9이면 restart, 아니면 resume)은 그대로다 |
| 교정 | 등록 교정 `restore_motor_torque_speed`. restart 재학습 seed는 실패 seed + 1000 |
| 첫 판단·교정 제안 | v3·v3.1은 단일 호출(`decision_path: single`). v4는 같은 단일 호출에 검증 피드백 1회 재요청(`single_feedback`, 아래 v4 절) |
| 결과 범주·성공 조건 | v3 15개·12개, v3.1·v4 16개(`harness_failure` 추가)·12개. 성공 조건에 "확정 메커니즘의 특이성 관문 통과"가 더해졌다 |

특이성 관문의 근거는 이미 본 holdout·dev 실행 31개다. `P_torque` abnormal은 F2 4/4뿐 아니라 F3 4/4·F4 1/4에서도 나왔고, `P_noise`·`P_value` normal을 함께 요구하면 F2 4/4, 그 밖 0/27이었다. 이미 본 seed에서 사후에 고른 시연 게이트이며 진단 정확도의 근거가 아니다.

**회복 방식 규칙.** 모터 결함을 고친 뒤 300회 이어 학습(resume)한 결과를 이미 본 두 seed에서 쟀다([프로토콜](../bench/protocols/training_correction_motor_v1.json)).

| seed | 교정 뒤 이어 학습 선속도 RMSE | 같은 갱신량 정상 이어 학습 | 비 | 라벨 |
|---|---|---|---|---|
| 2026 | 0.225 m/s | 0.143 m/s | 1.57 | `not_recovered` |
| 2028 | 0.208 m/s | 0.133 m/s | 1.57 | `not_recovered` |

- 중간 체크포인트의 선속도 오차는 두 seed 모두 단조로 내려갔다. 이어 학습 갱신 0·51·151·251·299회에서 seed 2026은 1.017·0.877·0.613·0.383·0.225, seed 2028은 1.016·0.945·0.753·0.356·0.208이었다. 두 seed 모두 예산 부족 후보로 기록됐다.
- 사전 기준에 따른 규칙 갱신 결정은 `change`다. 다음 시연에서 actuator에도 restart를 권고하는 규칙과 restart arm을 사전등록해 시험한다.
- 결과 문구: "이번 실험은 restart를 시험하지 않았고 이어 학습의 예산 부족 가능성도 남으므로 restart를 택할 근거로 쓰지 않는다." 모터 시연의 restart 교정이 그 시험이다.

### 시연 v3.1 결과 (seed 2033)

결과는 **`recovered`, 시연 성공**이다. 사전등록 결과 문구: "새 판단으로 시작한 승인된 검사가 등록 결함 메커니즘을 확정했고, 승인된 교정 정책이 사전등록 기준 정책 대비 고정 행동 허용 범위를 통과했다."

모든 단계가 실험 커밋 `8f4b092` 하나로 기록됐다. 시각은 KST이며 공개 원장(`evals/loops/h06_s2033/`)에서 읽었다.

| 단계 | 기록 |
|---|---|
| 정상·결함 학습 | 두 학습 모두 300회 단계 기록 |
| 경고 R1 | F2 경고(첫 경고 iteration 136), NONE 경고 없음 |
| 고정 평가 | NONE 보행 통과, F2 unhealthy → 실패 존재 |
| Nemotron 첫 판단 | 유효한 `continue`. 순위 reward > actuator > exploration > optimizer > physics > termination, probe `P_reward`(동결 규칙 대안은 `P_noise`). 인용 근거는 학습 평균 보상 10.61(결함) 대 33.95(정상) |
| 검사 `aac77ab711ac` | 20:07:43 제안·승인, 20:07:44 소비, 20:08:04 receipt. `P_reward` normal(`track_lin_vel_xy_exp` 상대 오차 1.8e-08, 임계 0.05) → reward 기각 |
| 검사 `c324747ced04` | 20:08:04 제안, 20:08:05 승인·소비, 20:08:25 receipt. `P_noise` normal(noise ratio 0.9997) → exploration 기각 |
| 검사 `095e95df5fae` | 20:08:25 제안, 20:08:26 승인·소비, 20:08:58 receipt. `P_torque` abnormal(포화 비율 0.0633, 판정선 0.03. 정상 기준은 0.00078) |
| 고리 종료 | actuator 확정(검사 3회, 예산 4) |
| 특이성 관문 | 통과. `P_noise` normal(0.9997), `P_value` normal(critic 변화 1.931, 정상 기준 1.555) |
| Nemotron 교정 제안 | `restore_motor_torque_speed` + `restart`. 검증 accepted, 체크리스트 미충족 0, 결정적 규칙과 일치 |
| 교정 요청 `1e21c7720f44` | 20:10:16 제안·승인, 20:10:31 소비, 20:20:52 receipt `recovered` |
| 교정 재학습(seed 3033)·고정 평가 | healthy(네 검사 모두 통과) |

첫 판단은 reward를 1위로 두고 `P_reward`를 골랐다. 승인된 `P_reward` 검사가 normal로 나와 reward가 기각됐고, 동결 고리 규칙이 다음 검사 `P_noise`를 제안했다. 관측 뒤에는 모델을 다시 부르지 않는다. `P_noise` normal로 exploration이 기각됐고, `P_torque` abnormal로 actuator가 확정됐다. 첫 판단의 상위 3가설 안에 actuator가 2위로 있었기 때문에 가능한 경로였다. seed 2031에서도 같은 순서였다.

고정 평가:

| 정책 | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 비율 | 판정 |
|---|---|---|---|---|---|---|
| NONE s2033 | 0.140 | 0.292 | 0 | 20.0 | 0.119 | 보행 통과(기준) |
| F2 s2033 | 1.048 | 0.461 | 0 | 20.0 | 0.891 | unhealthy(`err_xy_ratio`·`err_yaw_ratio` 실패) |
| restart 교정 s3033 | — | — | — | — | 0.124 | healthy(생존·낙상·선속도·yaw 검사 통과) |

restart 교정 정책은 공개 결과에 판정과 정지 비율만 남는다.

사전 예측 7개는 모두 적중했다.
- F2가 같은 seed NONE 대비 unhealthy
- R1이 F2에 울리고 NONE에는 울리지 않음
- 첫 판단 `continue`, 상위 3가설에 actuator 포함
- `P_torque` abnormal 뒤 actuator 확정
- `P_noise`·`P_value` normal
- 규칙이 restart를 고름
- seed 3033 restart 교정 정책 recovered

성공 조건 12개도 모두 참이다.

비용(부모 프로세스 벽시계):
- 정상 학습 553.6 s, 결함 학습 592.4 s, 정상 기준 probe 92.1 s
- 고리 검사 GPU 벽시계: `P_reward` 20.7 s, `P_noise` 20.7 s, `P_torque` 32.4 s
  - CPU: 26.7 s·27.2 s·44.5 s
  - 시뮬레이터: 51,200·51,200·128,000 step
- 특이성 관문 20.7 s
- Nemotron 첫 판단 2.40 s(입력 2,845·출력 186 토큰), 교정 제안 1.89 s(입력 1,284·출력 157 토큰)
- restart 교정 학습 541.1 s, 교정 정책 고정 평가 54.9 s. 교정 요청 receipt의 gpu_s는 596.0이며 예산 2400 s 안이다.
- 승인 대기(제안→승인)는 검사 0~1 s, 교정 0 s였다. 운영자 승인의 기록 시각 차이이므로 사람 검토 시간으로 쓰지 않는다.

### 시연 v4 결과 (seed 2046, 검증 피드백 1회 재요청)

v3.1과 같은 흐름·규칙·특이성 관문·원장에서 두 모델 결정 단계(첫 판단, 교정 제안)의 호출 방식 하나만 바꿨다.

- 근거: [반복 결정 측정의 검증 피드백 재요청 arm](N2-PASS-K.md#검증-피드백-재요청-arm)이 이미 본 packet 11개 × 5회에서 사전등록 개선 기준을 넘었다(최종 무효 1/55). 그 결과 문구대로 시연 결정 단계에 넣기 전에 새로 사전등록했다. 사전등록은 그 요약이 `improved`일 때만 만들어지고 요약 파일을 SHA로 묶는다.
- 호출: `evals/feedback_model_call.py`. 1차 호출은 v3.1 단일 호출과 같다. 기존 계약 검증기가 거절하면 문제 코드와 고정 규칙 설명(packet별 값·정답 없음)을 붙여 같은 대화로 한 번만 다시 요청하고, 그 결과를 같은 검증기로 최종 판정한다.
- 실행기는 호출 기록을 packet·메시지·요청 설정에 묶어 다시 검증한다. 1차·최종 판정을 검증기로 다시 계산하고, 피드백 문장이 문제 코드에서 만든 고정 문장인지 확인한다.
- 재요청 뒤에도 무효면 v3.1과 같게 처리하도록 사전등록했다.
  - 첫 판단: `model_invalid`로 끝낸다. 규칙 probe로 대체하지 않는다.
  - 교정 제안: 거절된 제안을 근거로 붙인 채 규칙의 교정·회복 방식으로 요청을 올린다. 성공 조건 "교정 제안 검증 통과"가 거짓이 되어 시연 성공이 되지 않는다.
- 결정 단계마다 1차 유효 여부·재요청 횟수·최종 유효·문제 코드·재요청 비용을 따로 기록했다. 재요청 발생 여부는 예측하지 않았다.
- 사전등록 `bench/protocols/e2e_demo_v4_f2.json`(`4c90dcc`, LF SHA256 `f166882d…`), 실행기 `evals/e2e_demo_v4.py`, 교정 실행기 `bench/demo_correction_v4.py`. 결함·규칙·관문·교정·성공 조건·예측의 틀은 v3.1과 같다.

결과는 **`recovered`, 시연 성공**이다. 사전등록 결과 문구: "새 판단으로 시작한 승인된 검사가 등록 결함 메커니즘을 확정했고, 승인된 교정 정책이 사전등록 기준 정책 대비 고정 행동 허용 범위를 통과했다."

**두 결정 모두 1차 응답이 계약을 통과해 재요청은 한 번도 일어나지 않았다.** 이 시연에서 재요청 경로는 사전등록만 됐고 실제로 쓰이지 않았다. 따라서 이 결과는 재요청이 무효 응답을 고친다는 근거가 아니며, 결정 단계의 동작은 v3.1 단일 호출과 같았다.

| 결정 단계 | 1차 유효 | 재요청 | 최종 유효 | 요청 수 | 토큰(입력·출력) | 대화 전체 |
|---|---|---|---|---|---|---|
| 첫 판단 | 예 | 0회 | 예 | 1 | 2,837·186 | 2.92 s |
| 교정 제안 | 예 | 0회 | 예 | 1 | 1,281·157 | 2.60 s |

모든 단계가 실험 커밋 `4c90dcc` 하나로 기록됐다. 시각은 KST이며 공개 원장(`evals/loops/h06_s2046/`)에서 읽었다.

| 단계 | 기록 |
|---|---|
| 정상·결함 학습 | 두 학습 모두 300회 단계 기록 |
| 경고 R1 | F2 경고(첫 경고 iteration 136), NONE 경고 없음 |
| 고정 평가 | NONE 보행 통과, F2 unhealthy → 실패 존재 |
| Nemotron 첫 판단 | 1차 응답이 유효한 `continue`. 순위 reward > actuator > exploration > optimizer > physics > termination, probe `P_reward`(동결 규칙 대안은 `P_noise`). 인용 근거는 학습 평균 보상 10.38(결함) 대 34.52(정상) |
| 검사 `58ff030724c2` | 17:52:59 제안, 17:53:07 승인, 17:53:13 소비, 17:53:32 receipt. `P_reward` normal(`track_lin_vel_xy_exp` 상대 오차 1.8e-08, 임계 0.05) → reward 기각 |
| 검사 `7d6dc0dbcd98` | 17:53:32 제안, 17:53:38 승인·소비, 17:53:57 receipt. `P_noise` normal(noise ratio 0.9997) → exploration 기각 |
| 검사 `781074bea4b5` | 17:53:57 제안, 17:54:03 승인·소비, 17:54:33 receipt. `P_torque` abnormal(포화 비율 0.0813, 판정선 0.03. 정상 기준은 0.00046) |
| 고리 종료 | actuator 확정(검사 3회, 예산 4) |
| 특이성 관문 | 통과. `P_noise` normal(0.9997), `P_value` normal(critic 변화 1.537, 정상 기준 1.502) |
| Nemotron 교정 제안 | 1차 응답이 `restore_motor_torque_speed` + `restart`. 검증 accepted, 체크리스트 미충족 0, 결정적 규칙과 일치 |
| 교정 요청 `7742cd4e1de3` | 17:55:36 제안, 17:55:44 승인, 17:55:58 소비, 18:05:21 receipt `recovered` |
| 교정 재학습(seed 3046)·고정 평가 | healthy(네 검사 모두 통과) |

검사 순서는 v3.1 seed 2033과 같았다. 첫 판단은 reward를 1위로 두고 `P_reward`를 골랐고, 승인된 `P_reward`·`P_noise` 검사가 normal로 reward와 exploration을 기각한 뒤 `P_torque` abnormal로 actuator가 확정됐다. 관측 뒤에는 모델을 다시 부르지 않는다.

고정 평가:

| 정책 | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 비율 | 판정 |
|---|---|---|---|---|---|---|
| NONE s2046 | 0.144 | 0.291 | 0 | 20.0 | 0.123 | 보행 통과(기준) |
| F2 s2046 | 0.997 | 0.453 | 0 | 20.0 | 0.847 | unhealthy(`err_xy_ratio`·`err_yaw_ratio` 실패) |
| restart 교정 s3046 | — | — | — | — | 0.130 | healthy(생존·낙상·선속도·yaw 검사 통과) |

사전 예측 7개는 모두 적중했다.
- F2가 같은 seed NONE 대비 unhealthy
- R1이 F2에 울리고 NONE에는 울리지 않음
- 첫 판단(최종 결정) `continue`, 상위 3가설에 actuator 포함
- `P_torque` abnormal 뒤 actuator 확정
- `P_noise`·`P_value` normal
- 규칙이 restart를 고름
- seed 3046 restart 교정 정책 recovered

성공 조건 12개도 모두 참이다.

비용(부모 프로세스 벽시계):
- 정상 학습 519.9 s, 결함 학습 534.2 s, 정상 기준 probe 82.1 s
- 고리 검사 GPU 벽시계: `P_reward` 19.3 s, `P_noise` 19.0 s, `P_torque` 29.7 s
  - CPU: 25.4 s·25.1 s·41.5 s
  - 시뮬레이터: 51,200·51,200·128,000 step
- 특이성 관문 19.3 s
- Nemotron 첫 판단·교정 제안은 위 표와 같다(재요청 추가 비용 0)
- restart 교정 학습 494.9 s, 교정 정책 고정 평가 47.8 s. 교정 요청 receipt의 gpu_s는 542.7이며 예산 2400 s 안이다.

해석 경계:
- 새 seed 하나의 통합 시연이다. 재요청 경로는 이 시연에서 쓰이지 않았으므로, 그 경로가 실제 고리에서 무효 응답을 고치는지는 아직 관측하지 않았다.
- 같은 seed를 재요청 없이 돌린 대조는 없다. N2 arm의 개선 판정은 이미 본 packet의 개발 측정이고, 이 시연 결과와 합쳐 분모를 만들지 않는다.
- 결함·검사·교정을 같은 설계자가 만들었다. 진단 정확도·교정 성공률·일반화·절감의 근거가 아니다.

### 시연 v3 결과 (seed 2031)

결과는 **`harness_failure`**이다. 기록 문구: "시연 실행기 결함으로 등록된 다음 단계를 진행할 수 없어 시연을 멈췄다. 사건(단계·오류·원인·승인 소비 없음)을 기록했고 교정·회복 결과를 판정하지 않는다. v3 사전등록에 없던 범주를 v3.1 도구로 사후 기록했다."

모든 단계가 실험 커밋 `5769970` 하나로 기록됐다. 시각은 KST이며 공개 원장(`evals/loops/h06_s2031/`)에서 읽었다.

| 단계 | 기록 |
|---|---|
| 정상·결함 학습 | 두 학습 단계 기록 |
| 경고 R1 | F2 경고(첫 경고 iteration 136), NONE 경고 없음 |
| 고정 평가 | NONE 보행 통과, F2 unhealthy → 실패 존재 |
| Nemotron 첫 판단 | 유효한 `continue`. 순위 reward > actuator > exploration > optimizer > physics > termination, probe `P_reward`(동결 규칙 대안은 `P_noise`). 인용 근거는 학습 평균 보상 9.91(결함) 대 34.04(정상) |
| 검사 `3c919574ef61` | 16:32:48 제안, 16:32:58 승인·소비, 16:33:20 receipt. `P_reward` normal(상대 오차 1.7e-08) → reward 기각 |
| 검사 `1422dd3bd3eb` | 16:33:20 제안, 16:33:32 승인·소비, 16:33:53 receipt. `P_noise` normal(noise ratio 1.0004) → exploration 기각 |
| 검사 `4f5d51855a04` | 16:33:53 제안, 17:51:19 승인·소비, 17:51:51 receipt. `P_torque` abnormal(포화 비율 0.0784, 판정선 0.03. 정상 기준은 0.00055) |
| 고리 종료 | actuator 확정(검사 3회) |
| 특이성 관문 | 통과. `P_noise` normal(1.0004), `P_value` normal(critic 변화 1.617, 정상 기준 1.501) |
| Nemotron 교정 제안 | `restore_motor_torque_speed` + `restart`. 시연 등록 표로 다시 검증해 accepted, 체크리스트 미충족 0, 결정적 규칙과 일치 |
| 교정 요청 원장 기록 | 거부: "actuator에 등록되지 않은 회복 방식 'restart'". 원장 기록·승인 소비 없음 |

고정 평가:

| 정책 | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 비율 | 판정 |
|---|---|---|---|---|---|---|
| NONE s2031 | 0.150 | 0.268 | 0 | 20.0 | 0.127 | 보행 통과(기준) |
| F2 s2031 | 1.025 | 0.490 | 0.030 | 19.74 | 0.871 | unhealthy(`err_xy_ratio`·`err_yaw_ratio` 실패) |
| restart 교정 s3031 | — | — | — | — | — | 판정 없음(교정 미실행) |

사전 예측 7개 중 판정까지 간 5개(실패 존재, R1, 첫 판단, 고리 확정, 특이성)는 모두 적중했다. 나머지 둘("규칙이 restart를 고름", "restart 교정 정책 recovered")은 교정 단계에 가지 못해 어느 쪽으로도 세지 않는다.

**하네스 실패와 원인.**

- [사실] 시연 v3 실행기는 교정 요청의 제안(`evals/e2e_demo_v3.py:694`)·조회·실행에서 `evals/correct.py`의 기본 원장을 썼다. 기본 원장은 기본 등록 표를 쓰며, 그 표의 actuator 회복 방식은 resume뿐이다.
- [사실] 교정 packet과 교정 제안 검증은 시연 등록 표(actuator restart·resume)를 썼다. 그래서 규칙과 모델이 모두 restart를 냈는데 원장이 거부했다. 보상 결함은 기본 등록 표도 restart·resume을 모두 허용하므로 이 거부가 나지 않는다.
- [사실] v3 테스트는 restart 경로의 제안 → 승인 → 실행을 실제 원장 파일로 지나가지 않아 이 결함을 놓쳤다.
- [사실] 원장 파일(`evals/loops/h06_s2031/correction_ledger.jsonl`)은 만들어지지 않았고 승인 소비도 없었다.
- 멈춘 시연은 v3.1 실행기(`3869743`)의 `record-harness-failure`로 종료 기록을 남겼다. 기록 내용은 멈춘 단계 `correction-propose`, 오류 문자열 원문, 원인 `evals/e2e_demo_v3.py:694`(실험 커밋 `5769970`의 내용과 같음을 확인), 승인 소비 없음, 고리 상태 `confirmed`, "v3.1 도구로 사후 기록"이다.
- v3 요약 도구는 이 범주를 모르므로 공개 요약은 `harness-summary`로 만들었다. 이 요약은 진단·고리·특이성·교정 제안까지를 담고 교정·회복을 판정하지 않는다.
- v3 사전등록 결과 범주 15개 중 어느 것도 이 시연에 붙이지 않았다. 종료 기록이 있으면 v3 실행기는 다음 단계를 거부한다.

비용(부모 프로세스 벽시계):
- 정상 학습 605.0 s, 결함 학습 622.8 s, 정상 기준 probe 94.7 s
- 고리 검사 GPU 벽시계: `P_reward` 22.1 s, `P_noise` 21.4 s, `P_torque` 31.5 s
- 특이성 관문 20.0 s
- Nemotron 첫 판단 3.02 s(입력 2,849·출력 197 토큰), 교정 제안 1.50 s(입력 1,317·출력 165 토큰)
- 승인 대기(제안→승인)는 검사 순서대로 10 s, 12 s, 77분 26초였다. 운영 메모에 따르면 `P_torque` 승인은 운영자 세션 중단으로 늦어졌다. 검사는 승인·소비(17:51:19) 뒤에 실행됐으므로 대기 시간은 측정값과 판정에 들어가지 않는다.

### v3에서 v3.1로 바뀐 점

| 항목 | v3 | v3.1 |
|---|---|---|
| seed / 재학습 seed | F2 2031 / 3031(F1 2032 / 3032 등록, 미실행) | F2 2033 / 3033, F1 2034 / 3034 |
| 교정 원장 | 제안·조회·실행에 `correct.py` 기본 등록 표 원장(actuator: resume만) | 원장 함수 하나가 시연 등록 표(actuator: restart·resume)로 제안·승인·거절·소비·실행·조회를 모두 한다 |
| 승인·거절 | `evals/correct.py approve`·`reject` | `evals/e2e_demo_v3_1.py approve-correction`·`reject-correction`(시연 등록 표로 만든 요청은 `correct.py approve`의 기본 표 원장과 등록 표 digest가 달라 거부된다) |
| 교정 실행 | `correct.py run`, 실행기 `bench/demo_correction_v3.py` | `run_correction`(같은 순서: 고아 확인 → 승인 소비 → 실행 → receipt), 실행기 `bench/demo_correction_v3_1.py` |
| 실행 중 끊긴 요청 | 닫는 명령 없음 | `recover-correction`: 재실행 없이 닫는다. 교정 결과 파일이 있으면 스키마·태그·요청·회복 방식·실험 커밋을 대조해 그 판정을, 없으면 unknown |
| 교정 예산 하한(2400 s) | 승인 뒤 교정 실행 단계에서 검사 | 제안 단계에서 원장에 쓰기 전에 검사 |
| 종료 뒤 원장 쓰기 | 막지 않음 | 교정 근거·제안·승인·거절을 거부 |
| 결과 범주 | 15개 | 16개(`harness_failure`) |
| 결과 스키마 | `e2e_demo_result_v3` | `e2e_demo_result_v3_1`, 교정 결과 `e2e_correction_result_v3_1` |

`harness_failure`는 v3.1에서 사전등록한 범주다. 다음 조건이 모두 맞을 때만 기록한다.
- 종료 기록과 교정 단계 기록이 없다.
- 멈춘 단계가 마지막 기록 단계 바로 다음 등록 단계이거나, 특이성 뒤의 교정 명령 중 하나다.
- 교정 원장에 승인·거절·소비·실행·보류 사건이 없다.
- 원인 파일이 시연 실험 커밋에 같은 내용으로 있다.
- 오류 문자열에 절대 경로가 없다.
- 기록 도구가 커밋돼 있다.

v3 실행기와 프로토콜은 seed 2031 공개 기록의 소스 결합 때문에 바꾸지 않았다. v3.1 회귀 테스트는 F2·F1 restart 경로를 실제 원장 파일로 제안부터 receipt까지 지나간다. 같은 구동을 v3 실행기에 하면 seed 2031과 같은 거부가 재현된다(`tests/test_e2e_demo_v3_1.py`).

## NAT 결정 경로 보상 결함 종단 시연 (seed 2034)

NAT 경로 비교 v4가 채택 조건을 충족한 뒤, 채택 규칙("다음 seed 시연의 결정 단계")에 따라 첫 판단과 교정 제안을 NeMo Agent Toolkit 도구 호출 경로로 낸 시연이다. 실행기·흐름·판정은 모터 v3.1과 같고, 결함은 F1(보상)이다.

결과는 **`recovered`, 시연 성공**이다. 성공 조건 12/12, 사전 예측 7/7. 모든 단계가 실험 커밋 `2d83b16` 하나로 기록됐다.

| 항목 | 값 |
|---|---|
| 결정 경로 | `nat_v4`: `evals/nat_decision_call_v4.py`(WSL). 프로토콜이 NAT 경로 비교 v4 공개 요약(`adopt`)을 SHA로 묶는다 |
| 첫 판단 도구 | 사실 id 목록·사실 값·probe 카탈로그 조회, 제출 전 계약 점검(실행당 최대 2회), 제출. 실수 사실 값은 유효숫자 6자리 문자열로 보여 주고, 보여 준 표기와 같은 근거만 저장 값으로 바꿔 공개 검증기로 검증한다 |
| 교정 제안 도구 | 사실·교정 등록 표·스킬 체크리스트 조회, 점검, 제출. null 칸이 없는 제출 스키마를 공개 계약 형태로 바꾸고, packet에 실린 시연 등록 표로 검증한다 |
| 재시도 | 인프라 오류만 다시 호출한다(첫 판단은 사전등록 상한 1회). 의미 재시도 없음 |
| 회복 방식 규칙 | 보상 결함 규칙(정지 비율 ≥ 0.9이면 restart, 아니면 resume) |
| 특이성 관문 | 보상 메커니즘에는 등록 관문이 없어 측정 없이 기록만 한다 |

시각은 KST이며 공개 원장(`evals/loops/h02_s2034/`)에서 읽었다.

| 단계 | 기록 |
|---|---|
| 정상·결함 학습 | 두 학습 모두 300회 단계 기록 |
| 경고 R1 | F1 경고(첫 경고 iteration 135), NONE 경고 없음 |
| 고정 평가 | NONE 보행 통과, F1 unhealthy → 실패 존재 |
| NAT 첫 판단 | 유효한 `continue`. 순위 reward > exploration > optimizer > physics > actuator > termination, probe `P_reward`(동결 규칙 대안은 `P_value`). 인용 근거는 학습 평균 보상 11.02(결함) 대 33.27(정상). LLM 호출 10회, 계약 점검 1회 |
| 검사 `b7dcc63c9289` | 20:45:41 제안·승인·소비, 20:46:03 receipt. `P_reward` abnormal(`track_lin_vel_xy_exp` 상대 오차 0.352, 임계 0.05) |
| 고리 종료 | reward 확정(검사 1회) |
| NAT 교정 제안 | `restore_body_frame_reward` + `restart`. 검증 accepted, 체크리스트 미충족 0, 결정적 규칙과 일치. LLM 호출 7회, 계약 점검 1회 |
| 교정 요청 `7bdcd38f0a16` | 20:47:07 제안·승인, 20:47:19 소비, 20:57:02 receipt `recovered` |
| 교정 재학습(seed 3034)·고정 평가 | healthy(네 검사 모두 통과) |

고정 평가:

| 정책 | 선속도 RMSE (m/s) | yaw rate RMSE (rad/s) | 낙상 비율 | 생존 (s) | 정지 비율 | 판정 |
|---|---|---|---|---|---|---|
| NONE s2034 | 0.145 | 0.293 | 0 | 20.0 | 0.123 | 보행 통과(기준) |
| F1 s2034 | 1.173 | 0.641 | 0 | 20.0 | 0.997 | unhealthy(`err_xy_ratio`·`err_yaw_ratio` 실패) |
| restart 교정 s3034 | — | — | — | — | 0.124 | healthy(생존·낙상·선속도·yaw 검사 통과) |

사전 예측 7개(실패 존재·정지 비율 ≥ 0.9, R1, 첫 판단 `continue`·상위 3가설에 reward, `P_reward` abnormal 뒤 reward 확정, 특이성 관문 없음, 규칙 restart, restart 교정 정책 recovered)는 모두 적중했다. 성공 조건 12개도 모두 참이다.

비용(부모 프로세스 벽시계):
- 정상 학습 569.1 s, 결함 학습 559.9 s, 정상 기준 probe 93.3 s
- `P_reward` 검사 GPU 벽시계 21.8 s, CPU 28.0 s, 시뮬레이터 51,200 step
- NAT 첫 판단 28.7 s(LLM 호출 10회 합계 입력 47,359·출력 1,632 토큰)
- NAT 교정 제안 18.8 s(LLM 호출 7회 합계 입력 22,247·출력 1,171 토큰)
- restart 교정 학습 520.9 s, 교정 정책 고정 평가 50.2 s. 교정 요청 receipt의 gpu_s는 571.1이며 예산 2400 s 안이다.
- 승인 대기(제안→승인)는 검사·교정 모두 0 s였다. 운영자 승인의 기록 시각 차이이므로 사람 검토 시간으로 쓰지 않는다.

NAT 결정 경로를 쓴 첫 시연이며, 같은 seed를 단일 호출로 돌린 대조는 없다. 단일 호출 시연(보상 v2, 모터 v3.1)의 첫 판단 입력 토큰은 2,829·2,845였다. NAT는 근거를 도구로 나눠 조회하고 점검으로 고칠 기회가 있어 조건이 다르므로, 비용·결과 차이를 우위나 절감으로 읽지 않는다.

## NAT 경로 비교 (E4b)

종단 시연의 두 모델 결정(첫 판단, 교정 제안)과 같은 고정 입력을 NeMo Agent Toolkit 1.9 도구 호출 워크플로에 반복해 넣고, 계약상 유효한 결정이 얼마나 나오는지 쟀다. GPU를 쓰지 않는 개발 측정이다.

네 판 모두 공통으로 다음을 따랐다.
- 실행 전에 프로토콜을 커밋했다.
- 종류별 5회 실행하고, 인프라 오류만 1회 재시도했다.
- 입력 packet은 SHA로 묶었다.
- 채택 규칙: 다음 seed 시연에 NAT 결정을 쓰려면 채택 세트에서 두 종류 모두 인프라 오류를 뺀 실행이 4 이상이고, 그중 유효 결정이 4 이상이어야 한다.
- 일치율은 채택 조건에 쓰지 않는다(단일 호출이 정답이 아니다).
- 판마다 채택 입력을 새로 정했고, 판끼리 분모를 합치지 않는다.

| 같은 것 | 다른 것 |
|---|---|
| 모델, temperature 0, `reasoning_effort: none` | NAT는 사실·probe 카탈로그·교정 등록 표·체크리스트를 도구로 여러 턴에 나눠 조회한다. 단일 호출은 packet 전체를 한 번에 받는다 |
| 같은 packet | NAT 시스템 프롬프트는 계약 프롬프트 뒤에 도구 사용 규칙을 붙인다 |
| 같은 판단 계약 프롬프트와 검증기 | v2부터 NAT는 제출 전 계약 점검 도구로 실행당 최대 2회 사유를 받아 고칠 수 있다 |

| 판 | 프로토콜(커밋, LF SHA256) | 바꾼 것 | 채택 입력 | 첫 판단 유효 | 교정 제안 유효 | 판정 |
|---|---|---|---|---|---|---|
| v1 | `e4b_nat_compare_v1.json`(`bbd2ed5`, `0063a5fb…`) | — | 보상 시연 v1(seed 2029) | 3/5 | 0/5 | `do_not_adopt` |
| v2 | `e4b_nat_compare_v2.json`(`88ede63`, `e106460d…`) | 입력 스키마 설명에 형식 규칙, 제출 전 계약 점검 도구 | 보상 시연 v2(seed 2030) | 5/5 | 0/5 | `do_not_adopt` |
| v3 | `e4b_nat_compare_v3.json`(`d330c82`, `14293333…`) | 교정 제출 스키마에서 null 칸을 없앰(propose·hold 판별 칸, 쓰지 않는 칸은 빈 문자열) | 이미 본 사례 h02_s2027(공개 기록에서 만든 packet) | 2/5 | 5/5 | `do_not_adopt` |
| v4 | `e4b_nat_compare_v4.json`(`b128030`, `b51fabfb…`) | 첫 판단: 실수 사실 표기 고정·표기 일치 치환. 교정: packet 등록 표로 검증 | 사전등록 뒤 시작한 첫 모터 시연(seed 2031)의 packet | 5/5 | 5/5 | `adopt` |

채택 판정에 쓰지 않은 세트는 다음과 같다.
- v2: seed 2029 입력(별도 분모). 첫 판단 5/5, 교정 제안 0/5
- v3: 교정 개발 세트. seed 2030 입력 5/5, seed 2029 입력 5/5, h05_s2027(등록 밖 메커니즘, 보류 분기) 4/5
- v4: 개발 세트 h02_s2027. 첫 판단 3/5. 무효 2회의 오류 유형은 `ValueError` 1회, `ToolInvocationError` 1회다.

판마다 무효의 모양과 원인 추정:

1. **v1.**
   - [사실] 첫 판단 무효 2회는 expected_outcomes에 메커니즘 6개 전부를 적었다. 계약은 상위 3가설의 기대 결과만 적게 한다(v2 프로토콜 `v1_failures_fixed_for`).
   - [사실] 교정 제안 무효 5회는 모두 `hold_reason`에 문자열 `"None"`을 넣은 채 교정을 제안했다. 4회는 체크리스트 한 칸에 사실 id 여러 개를 쉼표로 이어 적어서도 거절됐고, 1회는 체크리스트 칸에 교정 id를 적었다.
   - [사실] 유효한 첫 판단 3회 중 1회는 `P_episode`를 골랐고 상위 3가설 모두에 normal을 기대했다.
   - [추론] 종단 시연 실행기였다면 상위 3가설을 가르지 못하는 probe로 보고 동결 규칙 probe로 대체했을 것이다.
   - [추론] 도구 스키마 설명이 부족했다. v2에서 형식 규칙을 필드 설명에 넣고 점검 도구를 더했다.
2. **v2.**
   - [사실] 교정 제출·점검 인자 20개 중 19개의 `hold_reason`이 문자열 `"None"`이었다.
   - [사실] 같은 호출의 ATIF 궤적(파싱된 LLM 도구 호출 인자)과 도구 기록(도구가 받은 인자)이 호출마다 같았다.
   - [사실] langchain은 도구 인자 문자열을 `json.loads`로 파싱하므로 JSON `null`은 `None`이 되고 `"None"`으로 바뀌지 않는다. 같은 계약의 단일 호출은 `null`을 썼다.
   - [추론] `"None"`은 제공자 쪽 도구 호출 인자 생성·파싱 단계에서 생긴다. 원시 응답을 남기지 않아 모델과 제공자 파서를 가르지 못했다. v3에서 null 칸이 없는 스키마로 바꿨다.
3. **v3.**
   - [사실] 채택 입력의 첫 판단 무효 3회는 모두 근거 claim `telemetry:reference:Policy/mean_noise_std`의 값이 0.3697160104910533이었다. packet 사실 값 0.36971601049105324와 다른 double이다.
   - [사실] 공개 검증기는 claim 값을 사실 값과 `==`로 비교한다. 유효 2회는 17자리 실수 claim 4개를 같은 double로 인용했다.
   - [추론] 17자리 실수 표기를 옮기는 과정에서 마지막 자리가 바뀌었다. 원시 응답이 없어 모델과 제공자 파서를 가르지 못한다.
   - v4에서 표기를 고정했다. 허용 오차를 쓰지 않은 이유는 두 가지다. 검증기의 "보여 준 근거를 그대로 인용" 의미가 유지되고, 실패를 본 뒤 오차 크기를 정하지 않아도 된다.
4. **v4.** 채택 세트 두 종류 모두 5/5 유효로 `adopt`다. 사전등록 문구: "다음 seed 시연의 결정 단계에 쓸 수 있다. 정확도·우위의 근거가 아니다."
   - 유효 실행 중 단일 호출 결정과의 일치:
     - 첫 판단: 행동 5/5, 1위 3/5, 상위 3 집합 0/5, probe 1/5
     - 교정 제안: 결정·교정·회복 방식·미충족 체크리스트 모두 5/5
   - 비용(중앙값):
     - 첫 판단: LLM 호출 8회, 입력 34,900·출력 1,384 토큰, 17.6 s
     - 교정 제안: LLM 호출 7회, 입력 22,377·출력 1,242 토큰, 16.2 s
     - 같은 입력의 단일 호출: 2,849·197 토큰 3.02 s, 1,317·165 토큰 1.50 s

해석 경계:
- 판마다 고정 입력을 반복 호출한 개발 측정이다. 판마다 입력이 달라, 다음 판의 유효 결정 수를 앞 판 원인 추론의 검증으로 쓰지 않는다.
- NAT는 근거를 도구로 나눠 조회하고 단일 호출은 한 번에 받으므로, 일치율·비용 차이에는 이 제공 방식 차이가 섞여 있다.
- 판정은 각 판의 워크플로 구성(도구 설명·스키마·표기 포함)에 대한 기록이며 NAT 자체의 성능 평가가 아니다.
- v4 채택으로 NAT 경로를 쓴 시연은 위 "NAT 결정 경로 보상 결함 종단 시연(seed 2034)" 한 건이다.
- 같은 packet을 단일 호출로도 5회 반복한 짝 비교와 반복 일관성(pass^k)은 `docs/N2-PASS-K.md`에 있다.
  - v4 채택 세트(seed 2031)의 첫 판단은 5회 모두 유효했지만 1위가 reward 3 · optimizer 1 · termination 1로 갈렸다(pass^4 0).
  - 채택 조건은 유효 수만 보므로 판정은 그대로 `adopt`다.
  - 같은 packet의 단일 호출은 유효 4회의 1위가 모두 reward였고, 1회는 근거 claim이 빈 무효 결정이었다.

## 공식 디버깅 스킬 유무 프롬프트 비교

공식 Isaac Lab 디버깅 스킬(`isaaclab-debugging-rl-training`) 텍스트를 시스템 프롬프트에 넣은 경우와 뺀 경우를 비교했다. 스킬 텍스트는 SKILL.md·reference.md·examples.md다. 측정 대상은 모델이 진술한 다음 단계 계획이 스킬의 Known failure modes와 Expected behavior 규칙에 몇 번 해당하는가다.

- 프로토콜: `bench/protocols/skill_ab_v1.json`(`63ab9ba`, LF SHA256 `8e35975b…`)
- 사례: 이미 본 사례 h02_s2027
- 질의: 스킬 evaluations.md의 질의 3개
- 반복: 각 5회, 조건별 15회
- 모델 설정: 같은 모델, temperature 0, `reasoning_effort: none`

| 조건 | 응답(채점 가능) | 알려진 실패 모드 해당 | 기대 행동 포함 | 입력 토큰 중앙값 | 벽시계 중앙값 |
|---|---|---|---|---|---|
| 스킬 포함 | 15(15) | 0/30 | 26/50 | 2,411 | 4.1 s |
| 스킬 미포함 | 15(15) | 2/30 | 24/50 | 968 | 5.0 s |

규칙 판정 항목만 셌다. 설명 항목(조건별 30개)은 사람 채점이 필요하며, 조건을 가린 채점이 아직 끝나지 않았다.

공개 요약의 공통 문구: "이미 본 한 사례의 고정 입력을 반복 호출해 모델이 진술한 계획을 센 개발 측정이다. 실행한 행동이나 진단의 맞고 틀림은 재지 않았다. 스킬 예시가 채점 기준과 겹치고 두 조건이 같은 행동 목록을 받으므로, 건수 차이는 기록일 뿐 스킬의 효과·우위·일반화의 근거가 아니다."

## 사전등록과 다른 점

1. **v1 승인 주체.**
   - v1 프로토콜은 승인을 사람이 하는 것으로 적었다. 해당 문구는 `loop.approval`, 교정 단계 "approve(사람)", 결과 범주 `recovered`의 "사람 승인 검사"·"사람이 승인한 restart 교정"이다.
   - 실제 두 승인은 사용자가 위임한 운영자 승인이다(사용자 결정 2026-10-13).
   - 판정 조건은 승인 기록의 존재와 요청당 1회 소비이므로 판정에는 영향이 없다.
   - 사전등록 결과 문구를 인용하더라도 사람 승인이라고 쓰지 않는다. v2는 승인 문구를 사전등록 단계에서 고쳤다.
2. **v1 단계별 소스 커밋.**
   - 실행 중 결합 밖 파일의 커밋이 들어가 단계 기록의 커밋이 `1a6461e` → `bbd2ed5` → `afa9b06`으로 바뀌었다.
   - 단계마다 묶인 소스·프로토콜 SHA를 다시 검사해 통과했으므로 결합 파일은 바뀌지 않았다.
   - 교정 학습 중의 커밋 두 건이 restart 검증 거부로 이어졌다.
3. **v1 사후 보조 측정.** seed 3029 교정 정책의 고정 평가는 사전등록 실행기 밖에서 사후에 했다. 판정·성공 조건·예측에 쓰지 않는다.
4. **NAT 경로 비교 무효 원인 표기.** 네 판 모두 실행 절차는 사전등록대로 적용했다. 무효 결정의 모양(6개 기대 결과, 문자열 `"None"`, 다른 double의 실수 claim)은 사실이다. 그 원인(도구 스키마 설명 부족, 제공자 쪽 인자 생성·파싱, 실수 표기 옮김)은 추론이다.
5. **보상 v2.** 실행·판정 절차에서 사전등록과 다른 점은 없다. 승인은 사전등록 문구대로 사용자가 위임한 운영자 승인으로 했다.
6. **모터 v3 결과 범주.**
   - seed 2031 시연은 v3 사전등록 결과 범주 15개 어디에도 해당하지 않는 실행기 결함으로 멈췄다.
   - `harness_failure` 범주는 v3.1에서 사전등록했고, 이 시연에는 v3.1 도구로 사후 기록했다(기록에 그렇게 적힌다).
   - v3 결과로 세지 않으며, 교정·회복 예측 둘은 판정하지 않는다.
7. **모터 시연 seed 교체.** 실행기 결함을 고치려면 시연 중간에 코드를 바꿔야 했다. 같은 시연에서 그렇게 하지 않고, 새 실행기(v3.1)와 새 seed(2033)로 다시 사전등록했다. seed 2031과 2033의 결과를 합친 분모를 만들지 않는다.
8. **NAT 경로 비교 v4 채택 입력.** 사전등록 규칙("사전등록 뒤 시작한 첫 모터 시연", 종류마다 packet과 단일 호출 기록이 있으면 묶는다)대로 seed 2031의 packet을 묶었다. 하네스 결함은 두 packet과 단일 호출 기록이 만들어진 뒤에 일어났고, 종료 기록은 묶는 조건에 필요하지 않았다. 사전등록과 다른 점은 없다.
9. **모터 v3.1(seed 2033)·NAT 결정 경로(seed 2034).** 실행·판정 절차에서 사전등록과 다른 점은 없다. 운영 기록 하나를 남긴다.
   - seed 2033의 첫 정상 학습 시도(18:54)는 GPU 입장 검사에서 사용자 브라우저의 GPU 프로세스 때문에 거부됐다(exit 3). 아무것도 기록하지 않았고 단계도 소비하지 않았다.
   - 사용자 승인으로 그 프로세스를 닫은 뒤 19:44에 같은 단계부터 진행했다.
   - 사전등록한 입장 대기 규칙이 설계대로 동작한 것이므로 사전등록과 다른 점으로 세지 않는다. seed 2034에는 해당 사건이 없다.

## 주장하지 않는 것

- 진단 정확도, 교정 성공률, 일반화. 이미 본 결함 종류의 새 seed 기록이며, 시연끼리 합친 분모를 만들지 않는다.
- 모터 결함에서 고리가 첫 판단의 틀린 1위 가설(reward)을 바로잡은 것은 seed 2031·2033·2046 세 건의 고리 동작 기록이며 진단 정확도가 아니다. 특이성 관문은 이미 본 seed에서 사후에 고른 게이트이며 다른 결함과의 구별 성능 근거가 아니다.
- 모터 결함 restart 교정의 일반적 회복. seed 2033·2046 두 건이며, 회복 방식 규칙의 근거 실험은 resume만 시험했다.
- 검증 피드백 재요청의 효과. seed 2046 시연에서 재요청은 0회라 그 경로가 쓰이지 않았고, 같은 seed를 재요청 없이 돌린 대조도 없다.
- NAT 결정 경로의 우위. seed 2034 한 건의 성공이며 같은 seed의 단일 호출 대조가 없다. NAT는 근거를 도구로 나눠 조회하고 점검으로 고칠 기회가 있어 조건이 다르며, 입력 토큰을 더 많이 썼다.
- `harness_failure` 사후 기록을 v3 사전등록 결과로 쓰지 않는다.
- 다른 방법 대비 우위. 첫 판단이 reward를 1위로 둔 것은 결함·probe를 같은 설계자가 만든 조건의 결과다. 교정 제안 모델은 결정적 규칙의 결과를 볼 수 있으므로, 규칙과 일치했다는 기록을 독립 검증으로 쓰지 않는다. 교정과 회복 방식은 규칙이 정했다.
- GPU 절감. 위 시간은 부모 프로세스 벽시계이며 비교 대상이 없다.
- 사람 승인·신원 인증. 승인은 사용자가 위임한 운영자 승인이다.
- 실험 커밋 고정은 결합 파일이 그대로인지 확인하는 장치다. 실행 프로세스의 신원이나 실행 환경 전체를 인증하지 않는다.

## 근거 파일

- v2:
  - 프로토콜 `bench/protocols/e2e_reward_demo_v2.json`
  - 실행기 `evals/e2e_reward_demo_v2.py`, `bench/demo_restart_correction_v2.py`
  - 결과 `evals/results/e2e_reward_s2030_20261013/demo_result.json`, `evals/results/e2e_reward_s2030_20261013/runs/`
  - 검사 `evals/results/e2e_reward_s2030_20261013_loop/`
  - 기준 probe `evals/results/e2e_reward_s2030_20261013_ref/`
  - 원장 `evals/loops/h02_s2030/`
  - 학습 텔레메트리 `bench/runs_p0c/{baseline_p0c_s2030,h02_s2030}.telemetry.json`
  - 시연 영상과 출처 기록 `docs/media/go2-e2e-e2e_reward_s2030_20261013/`
- v1:
  - 프로토콜 `bench/protocols/e2e_reward_demo_v1.json`
  - 실행기 `evals/e2e_reward_demo.py`, `bench/demo_restart_correction.py`
  - 결과 `evals/results/e2e_reward_s2029_20261013/demo_result.json`, `evals/results/e2e_reward_s2029_20261013/runs/`
  - 검사 `evals/results/e2e_reward_s2029_20261013_loop/`
  - 기준 probe `evals/results/e2e_reward_s2029_20261013_ref/`
  - 원장 `evals/loops/h02_s2029/`
- 모터 v3.1(seed 2033):
  - 프로토콜 `bench/protocols/e2e_demo_v3_1_f2.json`
  - 실행기 `evals/e2e_demo_v3_1.py`, `bench/demo_correction_v3_1.py`
  - 결과 `evals/results/e2e_demo_v3_1_f2_s2033_20261013/demo_result.json`, `evals/results/e2e_demo_v3_1_f2_s2033_20261013/runs/`
  - 검사 `evals/results/e2e_demo_v3_1_f2_s2033_20261013_loop/`, 기준 probe `evals/results/e2e_demo_v3_1_f2_s2033_20261013_ref/`, 특이성 `evals/results/e2e_demo_v3_1_f2_s2033_20261013_spec/`
  - 원장 `evals/loops/h06_s2033/`
  - 학습 텔레메트리 `bench/runs_p0c/{baseline_p0c_s2033,h06_s2033}.telemetry.json`
- 모터 v4(seed 2046, 검증 피드백 1회 재요청):
  - 프로토콜 `bench/protocols/e2e_demo_v4_f2.json`
  - 실행기 `evals/e2e_demo_v4.py`, `bench/demo_correction_v4.py`, 결정 호출 `evals/feedback_model_call.py`, 테스트 `tests/test_e2e_demo_v4.py`
  - 결과 `evals/results/e2e_demo_v4_f2_s2046_20261014/demo_result.json`(결정 단계별 기록 `model_feedback`), `evals/results/e2e_demo_v4_f2_s2046_20261014/runs/`
  - 검사 `evals/results/e2e_demo_v4_f2_s2046_20261014_loop/`, 기준 probe `evals/results/e2e_demo_v4_f2_s2046_20261014_ref/`, 특이성 `evals/results/e2e_demo_v4_f2_s2046_20261014_spec/`
  - 원장 `evals/loops/h06_s2046/`
  - 학습 텔레메트리 `bench/runs_p0c/{baseline_p0c_s2046,h06_s2046}.telemetry.json`
- 모터 v3(seed 2031):
  - 프로토콜 `bench/protocols/e2e_demo_v3_f2.json`
  - 실행기 `evals/e2e_demo_v3.py`, `bench/demo_correction_v3.py`
  - 하네스 실패 요약 `evals/results/e2e_demo_v3_f2_s2031_20261013/harness_failure_summary.json`, 고정 평가 `evals/results/e2e_demo_v3_f2_s2031_20261013/runs/`
  - 검사 `evals/results/e2e_demo_v3_f2_s2031_20261013_loop/`, 기준 probe `evals/results/e2e_demo_v3_f2_s2031_20261013_ref/`, 특이성 `evals/results/e2e_demo_v3_f2_s2031_20261013_spec/`
  - 원장 `evals/loops/h06_s2031/`
  - 학습 텔레메트리 `bench/runs_p0c/{baseline_p0c_s2031,h06_s2031}.telemetry.json`
- NAT 결정 경로 보상 시연(seed 2034):
  - 프로토콜 `bench/protocols/e2e_demo_v3_1_f1.json`
  - 실행기 `evals/e2e_demo_v3_1.py`, NAT worker `evals/nat_decision_call_v4.py`, 도구 `src/rl_triage/nat_demo_tools_v4.py`, 설정 `configs/nat_demo_v4_diagnosis.yml`·`configs/nat_demo_v4_correction.yml`
  - 결과 `evals/results/e2e_demo_v3_1_f1_s2034_20261013/demo_result.json`, `evals/results/e2e_demo_v3_1_f1_s2034_20261013/runs/`
  - 검사 `evals/results/e2e_demo_v3_1_f1_s2034_20261013_loop/`, 기준 probe `evals/results/e2e_demo_v3_1_f1_s2034_20261013_ref/`
  - 원장 `evals/loops/h02_s2034/`
  - 학습 텔레메트리 `bench/runs_p0c/{baseline_p0c_s2034,h02_s2034}.telemetry.json`
- 모터 회복 방식 규칙: `bench/protocols/training_correction_motor_v1.json`, `evals/results/motor_w1_s2026_20261013/`, `evals/results/motor_w1_s2028_20261013/`
- NAT 경로 비교:
  - v1 `bench/protocols/e4b_nat_compare_v1.json`, `evals/results/e4b_e2e_reward_s2029_20261013/summary.json`
  - v2 `bench/protocols/e4b_nat_compare_v2.json`, `evals/results/e4b_nat_compare_v2/summary.json`
  - v3 `bench/protocols/e4b_nat_compare_v3.json`, `evals/results/e4b_nat_compare_v3/summary.json`
  - v4 `bench/protocols/e4b_nat_compare_v4.json`, `evals/results/e4b_nat_compare_v4/summary.json`
- 스킬 유무 비교: `bench/protocols/skill_ab_v1.json`, `evals/results/skill_ab_v1/summary.json`
- 결정 반복 일관성(N2): `bench/protocols/n2_pass_k_v1.json`, `evals/n2_pass_k.py`, `evals/results/n2_pass_k_v1/summary.json`, 테스트 `tests/test_n2_pass_k.py`, 문서 `docs/N2-PASS-K.md`
- 테스트:
  - 보상 시연 `tests/test_e2e_reward_demo.py`, `tests/test_e2e_reward_demo_v2.py`
  - 모터·NAT 결정 경로 시연 `tests/test_e2e_demo_v3.py`, `tests/test_e2e_demo_v3_1.py`, `tests/test_e2e_demo_v3_1_nat.py`
  - NAT 경로 비교 `tests/test_e4b_compare.py`, `tests/test_e4b_compare_v2.py`, `tests/test_e4b_compare_v3.py`, `tests/test_e4b_compare_v4.py`, `tests/test_nat_demo.py`, `tests/test_nat_demo_v2.py`, `tests/test_nat_demo_v3.py`, `tests/test_nat_demo_v4.py`
  - 스킬 유무 비교 `tests/test_skill_ab_compare.py`
