# 실패한 정책의 교정 후 이어 학습

`evals/run_training_correction.py`는 이미 실패한 Go2 정책을 체크포인트에서 이어 학습해 등록 교정의 실행 가능성을 확인한다. 호스트가 교정을 선택하는 개발 실험이며, 에이전트의 선택 정확도나 보류 세트 성능 평가가 아니다. 기존 진단 CLI에 자동 교정 기능을 추가하지 않는다.

## 입력과 비교

Isaac Lab v2.1.1, Isaac Sim 4.5, RSL-RL 2.3.3을 사용한다. 원본은 P0-C에서 이미 본 seed 2027의 `model_299.pt`다. 원래 학습 설정, 체크포인트, 고정 행동 평가와 요약의 SHA를 프로토콜에 묶는다. 원본 파일과 결함 정의는 보존한다.

| 사례 | 교정 대상 | 결함을 유지하는 대조 교정 |
|---|---|---|
| `case_01` | 보상의 몸체 좌표계 사용 | 모터 곡선 복원 |
| `case_02` | 모터 토크·속도 곡선 | 설정된 제한 시간 복원 |
| `case_03` | 설정된 제한 시간 | 보상의 몸체 좌표계 사용 |

각 사례는 정상 기준, 잘못된 교정, 등록 교정의 세 조건으로 구성된다. 뒤의 두 조건은 같은 실패 체크포인트의 정책과 optimizer 상태에서 시작한다. 정상 기준도 같은 seed의 정상 체크포인트에서 이어 학습한다. 환경 수 4096, 환경당 rollout 24 step, 추가 PPO update 300회를 고정한다. 최종 파일은 RSL-RL의 저장 iteration 규칙에 따라 `model_598.pt`다. 중간 체크포인트 중 좋은 결과를 고르지 않는다.

이어 학습은 정책과 optimizer 상태의 복원이다. 시뮬레이터 상태와 이전 RNG 상태는 복원하지 않는다. 새 실행의 시작 CPU·CUDA RNG 해시가 교정·대조 사이에서 같아야 하며, `init_at_random_ep_len=True`를 모든 조건에 적용한다. 실행 순서는 정상 기준 → 잘못된 교정 → 등록 교정이다. GPU 연산의 비결정성이 있으므로 단일 실행의 차이를 확정적 인과 효과나 일반화로 표현하지 않는다.

## 교정 범위와 실행 검증

`bench/correction_adapter.py`가 순정 SDK 소스의 선택된 함수 구현을 보관한다. 환경을 만들기 전에 기존 결함을 주입하고 선택된 구현 하나만 복원한다. 보상·종료 함수는 기존 함수 객체의 코드만 복원하고, 모터는 선택된 클래스 메서드를 복원한다. 잘못된 교정은 원래 결함을 제거하지 않아야 한다.

저장된 설정과 새 학습 설정은 환경 설정이 같아야 한다. Agent 설정에서는 실행 이름과 재개 인자, 추가 update 수만 변경할 수 있다. 체크포인트를 불러온 뒤 실제 정책·optimizer 지문을 파일 내부 상태와 대조한다. 성공한 환경 step 수, 최종 체크포인트 내부 iteration, PID와 생성 시각, 자연 종료와 프로세스 정리를 확인한다.

`bench/check_correction_cpu.py`는 실제 SDK 함수 소스를 추출해 CPU에서 계산한다. 올바른 교정과 잘못된 교정의 readback을 확인하는 검사이며 Isaac 시뮬레이션이나 보행 회복 측정은 아니다. 여섯 검사 결과의 PASS, 소스·런타임 일치, 결과 SHA를 프로토콜에 묶고 GPU 실행 전에 다시 확인한다.

## 결과 기준

기존 `fixed_eval_v1`의 26개 명령 조건, 1040개 환경, 1000 step 평가를 그대로 사용한다. 보상 크기로 회복을 판정하지 않는다. P0-C 요약에 저장된 행동 허용 범위를 사전등록하며 측정 후 조정하지 않는다.

주 기준은 같은 양을 이어 학습한 정상 정책이다. 원본 정상 정책과의 비교도 함께 기록한다. 새 정상 기준이 원본 정상 범위와 보행 조건을 통과하지 못하면 교정 효과를 판정하지 않는다.

| 상태 | 의미 |
|---|---|
| `invalid` | 출처·초기 상태·개입·학습량·평가·종료 검증 실패 |
| `source_failure_absent` | 원본 정책에서 등록 실패가 확인되지 않음 |
| `baseline_invalid` | 새 정상 기준의 행동이 기준을 통과하지 못함 |
| `no_failure_observed` | 결함을 유지한 대조도 정상 범위를 통과함 |
| `sham_shifted` | 대조의 실패 지표가 원본 실패 지표와 겹치지 않음 |
| `no_recovery` | 교정한 정책이 행동 기준을 통과하지 못함 |
| `restoration_observed_single_run` | 교정한 정책은 통과하고, 대조는 원본과 겹치는 실패 지표를 유지함 |

마지막 상태도 이미 본 단일 seed의 합성 결함 개발 결과다. AI의 진단·교정 선택, 실제 사용자 실패의 해결, 통계적 일반화, GPU 비용 절감을 증명하지 않는다.

## 개발 실행 결과

`case_01`의 등록 실행 `reward_resume_dev3_20261013`은 `no_recovery`였다. 세 조건 모두 20초 생존, 낙상률 0이었지만 등록 교정 후 정책의 선속도·회전속도 오차가 정상 기준의 허용 범위를 통과하지 못했다.

| 조건 | 선속도 RMSE (m/s) | 회전속도 RMSE (rad/s) | 생존 (s) | 낙상률 |
|---|---:|---:|---:|---:|
| 정상 기준 | 0.13186171244395944 | 0.1721144763503927 | 20.0 | 0.0 |
| 잘못된 교정 | 1.2004034794554577 | 0.537657828779045 | 20.0 | 0.0 |
| 등록 교정 | 1.160934033395102 | 0.6699384014437818 | 20.0 | 0.0 |

학습과 고정 평가를 합친 세 조건의 parent wall time은 1606.500999999931초다. 이 값의 범위는 각 프로세스의 parent launch부터 종료와 소유 작업 정리 확인까지다. GPU 커널 시간이나 다른 방법 대비 GPU 절감량이 아니다. 공개 결과는 `evals/results/reward_resume_dev3_20261013/`에 있으며, 원본 검증 자료는 `bench/private/training_correction/reward_resume_dev3_20261013/`에 보존한다.

`case_02`의 등록 실행 `motor_resume_dev_20261013`도 `no_recovery`였다. 등록 교정의 추종 오차는 대조보다 낮았지만 같은 예산의 정상 정책에서 정한 허용 범위를 통과하지 못했다. 아래 차이는 이미 본 seed의 단일 비교이며 반복 실험의 효과 추정이 아니다.

| 조건 | 선속도 RMSE (m/s) | 회전속도 RMSE (rad/s) | 생존 (s) | 낙상률 |
|---|---:|---:|---:|---:|
| 정상 기준 | 0.13186171244395944 | 0.1721144763503927 | 20.0 | 0.0 |
| 잘못된 교정 | 0.8262102916253836 | 0.30237012299637317 | 20.0 | 0.0 |
| 등록 교정 | 0.1883595651968174 | 0.26787660476962194 | 20.0 | 0.0 |

교정·대조의 초기 정책, optimizer 상태와 새 실행 RNG 해시는 일치했다. 세 조건의 학습·평가·종료 검증은 모두 통과했고, parent wall time의 합은 1712.2979999997187초였다. [공개 결과](../evals/results/motor_resume_dev_20261013/result.json)는 원시 보고서와 비용을 다시 계산할 수 있다.

`case_03`의 등록 실행 `timeout_resume_dev_20261013`도 `no_recovery`였다. 추종 오차는 대조보다 낮았지만 정상 범위를 통과하지 못했고, 낙상률은 대조보다 높았다.

| 조건 | 선속도 RMSE (m/s) | 회전속도 RMSE (rad/s) | 생존 (s) | 낙상률 |
|---|---:|---:|---:|---:|
| 정상 기준 | 0.13186171244395944 | 0.1721144763503927 | 20.0 | 0.0 |
| 잘못된 교정 | 0.20440836940238738 | 0.2924766986075353 | 19.794288330078125 | 0.014423076994717121 |
| 등록 교정 | 0.17559853700574732 | 0.24793178411414357 | 19.4297119140625 | 0.03846153989434242 |

초기 상태·RNG 일치와 세 조건의 실행 검증을 통과했고 parent wall time의 합은 1628.8900000001304초였다. [공개 결과](../evals/results/timeout_resume_dev_20261013/result.json)에 원시 평가와 검증 요약이 있다. 세 개발 사례 모두 정상 범위 회복을 확인하지 못했으므로, 이 결과로 자동 회복 기능이나 교정 성공률을 주장하지 않는다.

## 공개 결과와 미디어 순서

공개본은 절대 경로, PID, stdout·stderr 같은 비공개 실행 정보를 제외하고 원시 고정 평가, 초기 상태 지문 요약, 체크포인트·설정 SHA, 검증된 parent wall time을 담는다. `replay`는 공개 파일 SHA와 수치를 다시 계산할 뿐 실행 프로세스의 신원을 재인증하지 않는다. `source_commit_sha`는 실험 실행과 판정 코드의 커밋이며, 후속 재생 코드는 별도의 `publisher_sources` LF-SHA 집합으로 결합한다.

미디어 프로토콜은 실험 HEAD를 유지한 상태에서 준비한다. 그 뒤 공개 결과·publisher·기록기·프로토콜을 커밋하고, 변경된 HEAD에서 `record`를 실행한다. 기록기는 고정된 공개 artifact와 실험 커밋의 Git blob을 검증하므로 실험 커밋과 기록기 커밋이 같을 필요는 없다. 순서를 바꾸면 `prepare` 또는 `record`의 소스 결합 검사가 실패한다.

```powershell
cd "$HOME/rl-triage-agent"
$tag = 'reward_resume_dev3_20261013'

# 실험 HEAD를 유지한 채 공개본을 새 폴더에 한 번만 생성하고 수치 재생을 확인한다.
.\.venv\Scripts\python.exe evals/publish_training_correction.py export $tag --isaaclab "$HOME/IsaacLab"
.\.venv\Scripts\python.exe evals/publish_training_correction.py replay "evals/results/$tag"

# 같은 실험 HEAD에서 미디어 프로토콜을 준비한다.
.\.venv\Scripts\python.exe bench/record_correction_media.py prepare $tag --isaaclab "$HOME/IsaacLab"

# 여기서 코드·공개 결과·프로토콜을 검증하고 커밋한다. record는 그 커밋 뒤에 실행한다.
.\.venv\Scripts\python.exe bench/record_correction_media.py record $tag --isaaclab "$HOME/IsaacLab"
```

`record`는 GPU를 사용하는 Isaac Play 재생이다. 다른 GPU 작업이 있으면 기다리고, 기존 공개 결과나 미디어 프로토콜을 덮어쓰지 않는다. 생성된 영상은 정책 재생 자료이며 `no_recovery` 결과를 성공 사례처럼 바꾸지 않는다.

첫 사례의 [비교 영상](media/go2-correction-reward_resume_dev3_20261013/comparison.mp4)과 [GIF](media/go2-correction-reward_resume_dev3_20261013/comparison.gif)는 교정 전·등록 교정 후·대조 정책을 같은 조건으로 재생한다. 각 원본은 5.98초·299프레임이며, GIF는 같은 장면에서 90프레임을 사용한다. [출처 기록](media/go2-correction-reward_resume_dev3_20261013/source_manifest.json)에 세 체크포인트, 소스, 실행 종료 검증, 파일 해시가 있다. 행동 판정은 이 재생 화면이 아닌 위 고정 평가의 원시 수치로 계산한다.

모터 사례의 [비교 영상](media/go2-correction-motor_resume_dev_20261013/comparison.mp4)과 [GIF](media/go2-correction-motor_resume_dev_20261013/comparison.gif)도 같은 세 조건을 재생한다. 원본과 비교 영상은 5.98초·299프레임, GIF는 6초·90프레임이다. [출처 기록](media/go2-correction-motor_resume_dev_20261013/source_manifest.json)은 case_02의 `no_recovery` 판정과 원본 체크포인트를 결합한다. README의 미리보기는 이 모터 사례이며, 첫 보상 사례의 영상도 보존한다.

## 회복 방식 비교 v4

v3 `case_01`(seed 2027)에서는 보상 결함 F1을 고친 뒤 300회 이어 학습해도 정상 범위로 돌아오지 않았다. 학습 기록상 F1 정책은 명령을 따르지 않는 상태였고, 그 상태에서는 몸체 좌표와 월드 좌표의 추종 보상 차이가 작아 교정 신호가 약했을 가능성이 있다. v3 중단 규칙에 따라 seed 2027의 재시도나 예산 확대는 하지 않고, 회복 방식을 비교하는 질문을 새로 사전등록했다.

> 보상 결함을 고친 뒤, 이어 학습·탐색 재설정 이어 학습·처음부터 재학습 중 무엇이 같은 300회 예산으로 정상 범위에 돌아오는가?

### 입력과 arm

입력은 P0-C에서 이미 본 seed 2026·2028의 F1 실행(`h02_s2026`, `h02_s2028`)과 정상 실행(`baseline_p0c_s2026`, `baseline_p0c_s2028`)이다. 원본 `model_299.pt`, 학습 설정, 메타데이터, 고정 평가의 SHA와 원본 판정을 [프로토콜 v4](../bench/protocols/training_correction_v4.json)에 묶었다. 등록 교정은 `restore_body_frame_reward` 하나이며 v3와 같은 CPU 검사 행렬을 SHA로 다시 결합했다.

seed마다 아래 순서로 네 arm을 실행한다. 환경 수 4096, 환경당 rollout 24 step, 학습 300회는 모든 arm이 같다.

| arm | 시작 상태 | 결함·교정 | 최종 체크포인트 | 주 기준 |
|---|---|---|---|---|
| `clean_resume_reference` | 정상 `model_299.pt`의 정책·optimizer | 없음 | `model_598.pt` | — (이어 학습 arm의 기준) |
| `resume_corrected` | F1 `model_299.pt`의 정책·optimizer | F1 주입 뒤 교정 | `model_598.pt` | `clean_resume_reference` |
| `resume_corrected_std_reset` | `resume_corrected`와 같고 action noise std만 재설정 | F1 주입 뒤 교정 | `model_598.pt` | `clean_resume_reference` |
| `restart_corrected` | 체크포인트 없이 같은 seed로 새 초기화 | F1 주입 뒤 교정 | `model_299.pt` | 원본 정상 실행의 기존 고정 평가 |

`resume_corrected_std_reset`은 체크포인트를 불러와 정책·optimizer 지문을 검증한 직후, 같은 파라미터 객체에 in-place로 `init_noise_std`(1.0)를 채우고 그 파라미터의 Adam 모멘트만 지운다. 다른 파라미터의 optimizer 상태와 param_groups는 재설정 시점에 바꾸지 않는다. 전후 std 값과 평균, 모멘트 유무, 정책·optimizer 지문, std 항목만 뺀 기대 optimizer 지문을 `initial.json`에 남기고 부모가 대조한다.

`restart_corrected`의 대조는 원본 F1 실행(같은 seed, 같은 300회, 교정 없음)의 기존 고정 평가를 재사용하며 다시 학습하지 않는다. 원본 실행의 초기 정책 상태는 기록돼 있지 않으므로, 새 초기화의 지문과 RNG만 기록하고 원본과 같은지는 실행 중에 확인하지 않는다.

### 판정과 사전 예측

처리 arm마다 고정 평가 26조건의 행동 지표를 주 기준과 비교해 `recovered`(healthy), `not_recovered`(unhealthy), `invalid`로 표시한다. 이어 학습 기준은 제자리 대비 걷고 원본 정상 정책 대비 healthy여야 유효하며, 재학습 기준인 원본 정상 정책은 제자리 대비 걸어야 유효하다. 기준이 무효이거나 실행·검증 계약이 맞지 않으면 그 arm은 `invalid`다. 한 seed의 네 arm은 등록 순서로 실행하고 첫 실패에서 멈추며, 완료된 arm까지만 판정한다. 두 resume 처리 arm의 학습 직전 RNG 해시가 다르면 std 재설정 arm만 `invalid`로 둔다. 같은 프로토콜로 한 seed를 다시 실행하지 않는다.

실행 전 예측은 `restart_corrected`가 두 seed 모두 `recovered`, `resume_corrected`가 두 seed 모두 `not_recovered`이며, `resume_corrected_std_reset`은 예측하지 않는 탐색 arm이다. v3 학습 기록에서 F1 교정·대조 arm의 std는 0.38~0.55로 정상 기준 0.27~0.36보다 높았으므로, std가 낮아 탐색이 부족하다는 설명은 약하다.

`standstill_ratio`는 같은 고정 평가의 선속도 RMSE를 제자리 정책의 RMSE로 나눈 값이다. 1에 가까우면 명령 추종이 없다는 뜻이며 정지 자체를 증명하지 않는다. 실행 전에 계산한 원본 값은 아래와 같고, 사전 예측(F1 0.9 이상, 정상 0.3 이하)을 두 seed 모두 충족했다.

| seed | 원본 F1 | 원본 정상 | 규칙 권고 |
|---|---:|---:|---|
| 2026 | 0.9967 | 0.1256 | restart |
| 2028 | 0.9889 | 0.1195 | restart |

회복 방식 규칙은 "결함 메커니즘이 보상이고 원본 F1의 standstill_ratio가 0.9 이상이면 restart, 아니면 resume"이다. seed 2027 단일 관찰에서 만든 규칙이므로 v4 결과로 적중 여부만 기록하며 판정 라벨에는 쓰지 않는다. 권고 arm이 회복하고 대안 arm이 회복하지 않으면 `hit`, 둘 다 회복하면 `both_recovered`, 권고 arm이 회복하지 않으면 `recommended_not_recovered`, 어느 쪽이든 `invalid`이면 `undetermined`다.

모든 resume arm에 같게 적용되는 교란이 있다. RSL-RL 2.3.3 PPO는 resume 뒤 learning_rate를 설정값에서 다시 시작하고 adaptive schedule이 첫 update에서 optimizer lr을 덮어쓴다(rsl_rl #221). actor·critic·std는 Adam 하나와 전역 gradient clip(1.0)을 공유한다.

### 결과

| seed | arm | 판정 | 선속도 RMSE (m/s) | 회전속도 RMSE (rad/s) | standstill_ratio | 낙상률 |
|---|---|---|---:|---:|---:|---:|
| 2026 | 이어 학습 정상 기준 | — | 0.143 | 0.233 | 0.122 | 0.038 |
| 2026 | 고쳐서 이어 학습 | `not_recovered` | 1.156 | 0.377 | 0.982 | 0.000 |
| 2026 | 이어 학습 + std 재설정 | `not_recovered` | 1.115 | 0.295 | 0.948 | 0.001 |
| 2026 | 고쳐서 처음부터 재학습 | `recovered` | 0.148 | 0.299 | 0.126 | 0.000 |
| 2028 | 이어 학습 정상 기준 | — | 0.133 | 0.224 | 0.113 | 0.038 |
| 2028 | 고쳐서 이어 학습 | `not_recovered` | 1.137 | 0.252 | 0.966 | 0.000 |
| 2028 | 이어 학습 + std 재설정 | `not_recovered` | 1.155 | 0.184 | 0.981 | 0.000 |
| 2028 | 고쳐서 처음부터 재학습 | `recovered` | 0.141 | 0.279 | 0.119 | 0.000 |

| seed | 사전 예측 일치 | 규칙 권고 · 결과 | 학습·평가 parent wall (s) |
|---|---|---|---:|
| 2026 | 예측한 두 arm 모두 일치(std 재설정은 예측 없음) | restart · `hit` | 2426.624 |
| 2028 | 예측한 두 arm 모두 일치(std 재설정은 예측 없음) | restart · `hit` | 2464.908 |

두 seed 모두 사전 예측대로 고쳐서 이어 학습한 정책은 정상 범위에 돌아오지 않았고(선속도 오차 1.156·1.137 m/s, 이어 학습 정상 기준 0.143·0.133 m/s), 행동 노이즈 std를 초기값으로 되돌린 이어 학습도 돌아오지 않았다(1.115·1.155 m/s). 처음부터 재학습한 arm은 `recovered`였다. 이 arm의 최종 체크포인트 SHA는 두 seed 모두 원본 정상 실행의 `model_299.pt`와 같았다(seed 2026 `2e1e84cd…`, seed 2028 `d8c67eec…`). 교정한 코드가 정상 학습 코드와 같아져 같은 seed의 정상 학습을 바이트 그대로 재현한 것이므로, 재학습의 회복은 예상된 확인이며 새로운 회복 효과로 해석하지 않는다. 회복 방식 규칙의 권고(restart)는 두 seed에서 `hit`였다. seed 2027 하나에서 만든 규칙이 같은 결함의 이미 본 두 seed에서 맞았다는 관찰이며 일반 규칙의 근거가 아니다. 공개 결과는 [seed 2026](../evals/results/reward_v4_s2026_20261013/result.json)과 [seed 2028](../evals/results/reward_v4_s2028_20261013/result.json)에 있다.

같은 seed 재학습이 정상 학습을 재현한다는 점 때문에, 이후 [종단 시연](../evals/results/e2e_reward_s2030_20261013/demo_result.json)은 재학습 seed를 실패 seed와 다르게(실패 seed 2030, 재학습 seed 3030) 사전등록했다. 그 단일 실행에서 restart 교정 정책은 같은 seed 정상 정책 대비 healthy였다(standstill_ratio 0.125). 체크포인트 재현이 아닌 재학습의 한 사례이며, 반복 실험의 회복률이 아니다.

parent wall time은 각 arm의 부모 프로세스 실행부터 종료와 소유 작업 정리 확인까지를 더한 값이며 GPU 커널 시간이나 다른 방법 대비 절감량이 아니다.

### 해석 경계

각 칸은 이미 본 seed의 합성 결함에 대한 단일 실행이다. 같은 방식의 반복 분산을 재지 않았으므로 arm 사이 차이를 효과 크기나 일반화로 표현하지 않는다. 재학습 arm의 회복은 "보상 결함은 처음부터 다시 학습해야 한다"는 일반 규칙의 근거가 아니며, 회복하지 않은 이어 학습 arm은 300회 예산과 이 seed에서의 관찰이다. 회복 방식은 호스트가 사전등록한 비교이며 모델이 교정이나 회복 방식을 고른 결과가 아니다. 결함 체크포인트의 학습 이력 전체와 시뮬레이터·RNG 상태를 복원하지 않으므로 정확히 같은 학습 궤적의 재개로 부르지 않는다.

### 공개 결과와 미디어

v3 실행 코드는 공개 결과 재생의 소스 결합을 지키려고 바꾸지 않았다. v4는 [실행기](../evals/run_training_correction_v4.py)와 [학습 진입점](../bench/train_with_recovery_mode.py)이 v3 함수를 재사용한다. 공개본과 미디어 프로토콜은 결합 파일이 실험 커밋과 현재 HEAD에서 같고 실험 커밋이 HEAD의 조상일 때 실험 커밋 기준으로 다시 검증하므로, 실험 뒤 코드 커밋으로 HEAD가 바뀌어도 만들 수 있다.

```powershell
cd "$HOME/rl-triage-agent"
# 실행(seed마다 한 번). 다른 GPU 작업이 있으면 시작하지 않는다.
.\.venv\Scripts\python.exe evals/run_training_correction_v4.py verify
.\.venv\Scripts\python.exe evals/run_training_correction_v4.py execute --seed 2026 --tag reward_v4_s2026_20261013
.\.venv\Scripts\python.exe evals/run_training_correction_v4.py execute --seed 2028 --tag reward_v4_s2028_20261013

# 공개 수치 묶음을 새 폴더에 한 번만 만들고 재생한다.
.\.venv\Scripts\python.exe evals/publish_training_correction_v4.py export reward_v4_s2026_20261013 --isaaclab "$HOME/IsaacLab"
.\.venv\Scripts\python.exe evals/publish_training_correction_v4.py export reward_v4_s2028_20261013 --isaaclab "$HOME/IsaacLab"
.\.venv\Scripts\python.exe evals/publish_training_correction_v4.py replay evals/results/reward_v4_s2026_20261013
.\.venv\Scripts\python.exe evals/publish_training_correction_v4.py aggregate evals/results/reward_v4_s2026_20261013 evals/results/reward_v4_s2028_20261013

# 미디어 프로토콜(개정판 r2)을 만든 뒤 공개 결과와 함께 커밋하고, 그 다음에 녹화한다.
.\.venv\Scripts\python.exe bench/record_correction_media_v4.py prepare reward_v4_s2026_20261013 --isaaclab "$HOME/IsaacLab"
.\.venv\Scripts\python.exe bench/record_correction_media_v4.py prepare reward_v4_s2028_20261013 --isaaclab "$HOME/IsaacLab"
.\.venv\Scripts\python.exe bench/record_correction_media_v4.py record reward_v4_s2026_20261013 --isaaclab "$HOME/IsaacLab"
.\.venv\Scripts\python.exe bench/record_correction_media_v4.py record reward_v4_s2028_20261013 --isaaclab "$HOME/IsaacLab"

# 공개 결과를 커밋한 뒤 발표 재생 화면 데이터를 다시 만든다.
.\.venv\Scripts\python.exe evals/build_replay_page.py
```

공개 묶음은 seed마다 완료된 arm의 원시 고정 평가, 초기 상태 지문 요약(std 재설정 readback 포함), 학습 receipt 요약, 그 seed의 입력 SHA와 판정을 담고 절대 경로·PID·로그는 제외한다. 일부 arm만 끝난 seed도 완료된 arm까지 공개한다. `replay`는 공개 파일의 해시와 수치 판정을 다시 계산할 뿐 실행 프로세스의 신원을 재인증하지 않는다.

비교 영상은 원본 F1, 고쳐서 이어 학습, 고쳐서 재학습, std 재설정, 이어 학습 정상 기준, 원본 정상 정책의 여섯 정책을 같은 stock Play 조건으로 재생해 2×3 격자로 합친다. seed 2026의 [비교 영상](media/go2-correction-reward_v4_s2026_20261013/comparison.mp4)·[GIF](media/go2-correction-reward_v4_s2026_20261013/comparison.gif)·[출처 기록](media/go2-correction-reward_v4_s2026_20261013/source_manifest.json)과 seed 2028의 [비교 영상](media/go2-correction-reward_v4_s2028_20261013/comparison.mp4)·[GIF](media/go2-correction-reward_v4_s2028_20261013/comparison.gif)·[출처 기록](media/go2-correction-reward_v4_s2028_20261013/source_manifest.json)이 있다. 비교 영상은 1920×720·299프레임·5.98초, GIF는 960×360·90프레임·6초다. 두 seed 모두 4.5초 지점에서 원본 F1·고쳐서 이어 학습·std 재설정 정책은 출발점 근처에서 몸을 낮춘 채 머물고, 재학습·이어 학습 정상 기준·원본 정상 정책은 걸어서 멀어진다. 재생 한 번의 관찰이며 판정은 위 고정 평가 수치로만 한다.

미디어 프로토콜은 개정판 r2(`bench/protocols/training_correction_media_v4_<tag>_r2.json`)를 쓴다. 같은 tag의 r1은 원 실행 seed를 담은 생성 키가 Play 재생 seed(42)를 덮어써 `record`가 녹화 전에 거부했고(GPU 미사용), 파일은 바꾸지 않은 채 r2의 `supersedes`에 r1의 SHA와 거부 이유를 결합했다. r2는 원 실행 seed를 `source_seed`로 따로 둔다.

## 재설정 개입 v2

v4에서 보상 결함을 고친 뒤 이어 학습한 두 seed가 모두 회복하지 않았고, action noise std 재설정도 결과를 바꾸지 못했다. 후속 개발 측정은 체크포인트의 어떤 상태가 이어 학습을 막는지 좁히려고 세 개입을 [프로토콜 v2](../bench/protocols/recovery_reset_v2.json)에 고정했다. 입력은 이미 본 F1 seed 2026·2028이며, 각 arm은 결함을 고친 뒤 300회 이어 학습하고 같은 고정 행동 평가로 판정했다.

| arm | 재설정 범위 | seed 2026 | seed 2028 | 두 seed 회복 |
|---|---|---|---|---:|
| `optimizer_only` | 정책 텐서를 유지하고 공유 Adam 상태 제거 | `not_recovered` | `not_recovered` | 0/2 |
| `critic_reset` | critic의 모든 Linear와 해당 Adam 상태 재설정 | `not_recovered` | `not_recovered` | 0/2 |
| `actor_last_layer_reset` | actor 마지막 Linear와 해당 Adam 상태 재설정 | `recovered` | `not_recovered` | 1/2 |

[사실] 여섯 실행은 모두 출처·개입·학습량·평가·종료 계약을 통과했다. `optimizer_only`와 `critic_reset`은 두 seed에서 정상 이어 학습 기준의 행동 허용 범위를 통과하지 못했다. `actor_last_layer_reset`은 seed 2026에서 통과했지만 seed 2028에서는 통과하지 못했다. 공개 요약은 [재설정 개입 결과](../evals/results/reset_interventions_v2_20261015/summary.json)에 있다.

[추론] actor 마지막 층 상태가 seed 2026의 회복을 막았을 가능성은 후속 가설이다. seed 2028에서 같은 개입이 회복하지 않았으므로 가소성 상실이나 특정 층을 실패 원인으로 확정할 수 없다. 사전등록한 `mixed` 규칙에 따라 회복 방식은 바꾸지 않고 메커니즘을 미확정으로 둔다. 이 비교는 이미 본 합성 결함 두 seed의 개발 측정이며, 회복 규칙 변경이나 비용 효과의 근거가 아니다.

## 새 seed 출력층 재설정 확인

v2에서 seed 2026은 회복하고 seed 2028은 회복하지 않은 혼합 결과를 본 뒤, actor 마지막 Linear와 해당 Adam 상태만 재설정하는 조건을 고정했다. 처음 쓰는 F1 seed 2063·2064에서 같은 결함 체크포인트로 `continuation`과 `actor_last_layer_reset`을 짝지었다. 두 arm은 결함을 고친 뒤 각각 300회 이어 학습했으며, actor 재설정 arm은 critic과 policy std를 유지했다.

| seed | arm | 판정 | 선속도 RMSE (m/s) | 회전속도 RMSE (rad/s) | 낙상률 |
|---|---|---|---:|---:|---:|
| 2063 | 그대로 이어 학습 | `not_recovered` | 1.119 | 0.404 | 0.031 |
| 2063 | actor 마지막 층 재설정 | `recovered` | 0.155 | 0.140 | 0.038 |
| 2064 | 그대로 이어 학습 | `not_recovered` | 1.175 | 0.457 | 0.000 |
| 2064 | actor 마지막 층 재설정 | `recovered` | 0.169 | 0.276 | 0.000 |

새 seed 두 개에서는 actor 마지막 층 재설정이 2/2 `recovered`, 그대로 이어 학습은 0/2 `recovered`였다. 등록 조건을 충족했으므로 이 개입을 조건부 후보로 남긴다. 기존 seed 2026·2028의 결과는 1/2였으며, 이 기록도 바꾸지 않는다. 공개 결과에는 [요약](../evals/results/novel_actor_reset_confirmation_20261016/summary.json), seed별 행동 지표, 개입 readback과 원본 SHA가 들어 있다.

이 결과만으로 전체 회복 규칙을 자동 변경하지 않는다. 기존 혼합 결과를 본 뒤 설계한 두 seed의 개발 확인이므로 실패 원인, 일반화, 비용 절감이나 모델 우월을 확정하지 않는다. 메커니즘은 계속 미확정으로 둔다.

## 액추에이터 곡선 검사

`src/rl_triage/actuator_envelope.py`의 `inspect_actuator_envelope`는 정책 rollout 없이 등록 속도 격자에서 DC motor `_clip_effort`의 양·음 포화 출력을 읽고 설정에서 계산한 토크–속도 곡선과 비교한다. 측정 중 교체한 `_joint_vel`은 `finally`에서 원 객체로 복원한다. 설정·shape·유한성 검증이 실패하면 곡선 이상으로 확정하지 않고 `invalid`를 반환한다.

이 함수는 device 중립의 제한된 곡선 readback이다. 반환하는 source SHA는 호출자가 제시한 값이며 함수가 파일 결합을 검증하지 않는다.

`bench/actuator_envelope_runtime.py`와 `bench/run_actuator_envelope_runtime.py`는 이 함수를 Isaac Sim에서 생성한 Go2 DCMotor 객체에 연결하는 별도 실행기다. 정책 rollout 없이 13개 속도점과 12개 관절을 읽으며, NONE·F2·F3·F4 조건의 소스·런타임·상태 복원·자연 종료를 검증한다. 부모 프로세스가 측정 곡선과 판정을 다시 계산한다. Isaac Sim 4.5.0 버전을 확인하지만 정확히 같은 rc/build 바이너리까지 보증하지 않는다.

`actuator_envelope_dev3_20261013`은 네 조건 모두 유효한 측정과 실행 검증을 통과했다. 등록된 합성 결함의 개발 검사이며, 학습된 정책이나 모델을 호출하지 않았다.

| 조건 | 곡선 판정 | 사전 예상 일치 | parent wall (s) |
|---|---|---|---:|
| NONE: 순정 | `normal` | 예 | 11.750 |
| F2: 모터 곡선 결함 | `curve_difference` | 예 | 11.391 |
| F3: 탐색 결함 | `normal` | 예 | 11.547 |
| F4: 최적화 결함 | `normal` | 예 | 11.656 |

예를 들어 +6 rad/s의 양(+) 방향 최대 토크는 순정·F3·F4에서 약 18.8 N·m, F2에서는 0 N·m였다. +18 rad/s의 F2 출력은 양·음 포화 요청 모두 −47 N·m로, 설정된 절댓값 제한 23.5 N·m를 벗어났다. 이 값은 실제 런타임 함수의 등록 격자 출력이며 실물 모터 측정이 아니다. 네 조건의 실행·소유 작업 정리 시간 합은 46.344초다. GPU 커널 시간이나 GPU 절감량으로 해석하지 않는다.

[프로토콜 v3](../bench/protocols/actuator_envelope_runtime_v3.json)는 환경 종료·GC·body 발행 뒤 Isaac Sim의 기본 fast shutdown을 호출하도록 고정한다. body는 측정 완료와 종료 대기만 나타낸다. 부모가 자연 종료, 정수 exit code 0, 소유 Job 비어 있음, fatal/traceback 부재, body·manifest SHA가 결합된 종료 인계 marker를 모두 확인해야 최종 통과다. 확장 모듈 각각의 정상 종료를 증명하는 계약은 아니다. 앞선 v1·v2의 인프라 오류 기록은 `invalid`로 보존했다.

[공개 측정 결과](../evals/results/actuator_envelope_dev3_20261013/result.json)는 PID·절대 경로·로그 원문을 제외한 네 조건의 곡선, 상태, 소스 해시와 시간을 담는다. 아래 명령은 공개 파일의 해시·수치·판정을 재계산하며 원래 실행 프로세스의 신원을 재인증하지 않는다.

```powershell
cd "$HOME/rl-triage-agent"
.\.venv\Scripts\python.exe evals/publish_actuator_envelope.py replay evals/results/actuator_envelope_dev3_20261013
```

`src/rl_triage/mechanism_registry.py`는 기존 검사를 보존하면서 별도 레지스트리와 승인 원장에 digest를 결합한다. 정책에 따라 달라지는 `P_torque`를 메커니즘 확정에서 제외하는 구성을 제공한다. 새 곡선 검사는 아직 에이전트의 검사 선택 경로에 연결하지 않았다. 곡선 차이는 등록 입력 격자의 관찰이며 원인 확정이나 LLM 우위 근거가 아니다.

## 실행

아래 명령은 Windows의 저장소와 Isaac Lab 설치를 전제로 한다. 프로토콜에 묶인 기존 체크포인트와 원본 학습 기록이 로컬에 있어야 한다.

```powershell
cd "$HOME/rl-triage-agent"
# 최초 프로토콜 작성 시에만: 새 CPU 결과 폴더에서 검사 후 코드와 함께 커밋한다.
$matrix = @{ F1=@('restore_body_frame_reward','restore_motor_torque_speed'); F2=@('restore_motor_torque_speed','restore_configured_timeout'); F6=@('restore_configured_timeout','restore_body_frame_reward') }
foreach ($fault in $matrix.Keys) {
    foreach ($index in 0,1) {
        $suffix = @('match','wrong')[$index]
        & "$HOME/IsaacLab/_isaac_sim/python.bat" bench/check_correction_cpu.py --isaaclab "$HOME/IsaacLab" --fault $fault --correction $matrix[$fault][$index] --output "bench/private/correction_cpu/$($fault.ToLower())_$suffix.json"
        if ($LASTEXITCODE -ne 0) { throw 'CPU correction check failed' }
    }
}
& "$HOME/IsaacLab/_isaac_sim/python.bat" bench/check_correction_cpu.py --isaaclab "$HOME/IsaacLab" --contract-output bench/private/correction_cpu/source_contract.json
.\.venv\Scripts\python.exe evals/run_training_correction.py prepare --contract bench/private/correction_cpu/source_contract.json
# 커밋된 소스·프로토콜·기존 입력을 확인한다.
.\.venv\Scripts\python.exe evals/run_training_correction.py verify
# 다른 GPU 작업이 없을 때 등록된 한 사례를 새 태그로 실행한다.
.\.venv\Scripts\python.exe evals/run_training_correction.py execute --case case_01 --tag correction_reward_dev
```

이미 작성된 프로토콜은 `prepare`로 덮어쓰지 않는다. 결과와 실행 로그는 `bench/private/training_correction/<tag>/`에 보존한다. 한 프로토콜의 같은 사례를 의미 재시도하지 않는다. GPU 점유를 확인해 실행 전 대기하며, 다른 세션의 프로세스를 종료하지 않는다. 승인 기록은 사용자가 허용한 개발 실험의 호스트 기록이고, 사람의 신원을 인증하는 기능은 아니다.
