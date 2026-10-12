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
