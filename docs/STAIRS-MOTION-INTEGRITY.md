# 이족·사족 계단 재생과 자세 검사

H1의 상행·하행 두 영상은 기존 진행·높이 기준을 통과했고, 추가 자세 검사에서도 몸체의 위쪽 축이 아래로 향한 표본은 없었다. Go2 상행은 기존에 실패했다. Go2 하행은 거리·높이 기준상 `completed`였지만 마지막 구간에서 몸체가 뒤집혀 있어, 안정적인 계단 시연 완료로 수용하지 않는다.

아래는 기존 0.25m/s Go2 영상의 감사다. 후속 [동일 정책 명령 대조](GO2-STAIRS-COMMAND-CONTRAST.md)에서 0.75m/s 상행·하행이 등록 기준을 통과했으며 기존 실패는 보존했다. 네 영상은 공식 기학습 정책 재생으로, RL Triage의 계단 진단·교정·회복이나 새 학습 성과로 세지 않는다.

## 기존 결과와 추가 관측

| 영상 | 기존 판정 | 전진 / 높이 변화 | body-up z 최솟값 | 음수 표본 | 현재 해석 |
|---|---|---|---:|---:|---|
| H1 상행 | completed | 2.977m / +0.579m | 0.987284 | 0 | 기존 재생 통과, 뒤집힘 미관측 |
| H1 하행 | completed | 2.827m / −0.673m | 0.988620 | 0 | 기존 재생 통과, 뒤집힘 미관측 |
| Go2 상행, 5cm 단차 | failed | 1.076m / −0.057m | 0.845712 | 0 | 진행·상승 실패 보존 |
| Go2 하행, 5cm 단차 | completed | 2.494m / −0.301m | −0.583399 | 13 | 기존 완료와 upright 관측 충돌 |

Go2 하행의 첫 음수 표본은 12.26초이며 마지막 표본도 −0.583399다. 같은 마지막 프레임의 영상에서도 뒤집힌 자세가 보인다. `terminated=false`였다는 사실은 보행 성공을 보장하지 않는다. 원래 task의 종료 조건은 시간 제한과 base contact이며, 별도의 upright 종료 조건은 없다.

## 검사 정의

[새 CPU 검사](../evals/stairs_motion_integrity_v1.py)는 manifest·trace SHA와 checkpoint, frame 0~625, 50Hz·12.5초를 결합한다. `root_quat_w`의 wxyz 순서와 유한 값·0이 아닌 norm을 확인하고 다음 값을 계산한다.

```text
body_up_world_z = 1 - 2 * (x² + y²) / (w² + x² + y² + z²)
```

음수는 몸체의 위쪽 축이 world 아래쪽을 향한 관측이다. 기존 `completed`와 이 관측이 함께 있으면 `upright_completion_contradicted`로 기록한다. 기존 실패는 유지한다. 음수 표본이 없다는 결과도 보행 안정성 전체의 증명으로 쓰지 않는다. 누락된 quaternion이나 잘못된 SHA를 값 0으로 채우지 않고 거부한다.

이 검사는 결과를 본 뒤 만든 사후 개발 감사다. 모델이 문제를 발견한 것처럼 보고하지 않으며, 원래 manifest·trace·영상·판정을 수정하지 않는다. 후속 재생의 수용 기준에는 실행 전에 이 검사를 포함한다.

## 후속 대조의 근거

0.25m/s는 공식 Go2 rough 학습 명령 범위 안이고 코드상 정지 명령으로 바뀌는 값도 아니다. 그렇다고 모든 계단에서 그 속도의 성공이 보장되지는 않는다. 따라서 속도와 계단 접근 상태의 상호작용은 남은 가설이다. 단차와 seed를 계속 바꾸며 성공 영상을 고르는 방식은 사용하지 않는다.

정책·명령·height scanner·action 의미를 확인한 뒤, 같은 checkpoint·지형·초기 상태에서 명령만 바꾸는 좁은 대조가 후속 후보다. 새 실행 전 입력·판정·상한을 등록하고 상행과 하행의 실패를 모두 보존한다. [공식 고정 버전의 명령·관측 설정](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py), [Go2 설정](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py).

## 공개 근거와 CPU 재현

- [감사 결과](../evals/results/stairs_motion_integrity_20261101_v1/audit.json), [출처](../evals/results/stairs_motion_integrity_20261101_v1/provenance.json), [파일 SHA](../evals/results/stairs_motion_integrity_20261101_v1/manifest.json).
- [H1 원래 영상·측정](media/stairs_h1_s172_20261030_v1/README.md).
- Go2 원문 manifest·trace는 공개 감사 폴더의 `sources/`에 바이트 그대로 보존했다. 이 폴더에는 Go2 영상이 포함되지 않는다.

저장소 루트에서 다음 명령은 새 출력에 네 원본의 감사를 재생한다. GPU·모델 API·Isaac 실행은 필요하지 않다.

```powershell
cd "$HOME/rl-triage-agent"
.\.venv\Scripts\python.exe -B evals/stairs_motion_integrity_v1.py --spec evals/results/stairs_motion_integrity_20261101_v1/inputs.json --out "$HOME/rl-triage-stairs-audit-new"
```
