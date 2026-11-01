# Go2 계단: 같은 정책에서 명령 속도만 바꾼 대조

공식 Go2 rough 정책을 같은 계단·초기 상태에서 재생했을 때, 0.25m/s에서는 상행이 진행하지 못하고 하행은 마지막에 뒤집혔다. 명령을 0.75m/s로 바꾼 두 재생은 등록한 진행·높이·자세 기준을 통과했다. 정책을 새로 학습하거나 고치지 않았다.

## 영상과 결과

네 조건 모두 625프레임·50fps·12.5초로 녹화됐고, render forward 전후 물리 상태가 같았다. 기술 종료·원본 프레임 encoder·checkpoint·실제 실행 자식의 정리 증거도 대조했다.

| 방향·명령 | 전진 | 높이 변화 | 최소 body-up z | 음수 표본 | 수용 | 원본 영상 |
|---|---:|---:|---:|---:|---|---|
| 상행 0.25m/s | 1.076m | −0.057m | 0.845712 | 0 | 불수용: 진행·상승 실패 | [MP4](../evals/results/go2_stairs_command_contrast_s200_20261101_v1/media/quadruped-stairs-up-step-0p05-cmd-0p25.mp4) |
| 하행 0.25m/s | 2.494m | −0.301m | −0.583399 | 13 | 불수용: 뒤집힘 | [MP4](../evals/results/go2_stairs_command_contrast_s200_20261101_v1/media/quadruped-stairs-down-step-0p05-cmd-0p25.mp4) |
| 상행 0.75m/s | 8.968m | +0.338m | 0.945781 | 0 | 통과 | [MP4](../evals/results/go2_stairs_command_contrast_s200_20261101_v1/media/quadruped-stairs-up-step-0p05-cmd-0p75.mp4) |
| 하행 0.75m/s | 8.867m | −0.361m | 0.979429 | 0 | 통과 | [MP4](../evals/results/go2_stairs_command_contrast_s200_20261101_v1/media/quadruped-stairs-down-step-0p05-cmd-0p75.mp4) |

같은 방향의 두 조건은 frame 0의 상태 전체가 같고 checkpoint·지형·runtime 결합도 같다. 기존 0.25m/s 상행·하행의 trace SHA는 이번 재생과 일치했다. 따라서 기존 실패가 재현됐다는 근거와 명령 변경 후 관측을 함께 남겼다.

수용 조건은 녹화·종료 증거에 더해 진행·높이 기준 통과, `termination=false`, `body_up_world_z` 음수 표본 0개다. 음수 없음만으로 모든 보행 안정성을 입증하지는 않는다. 네 조건 중 수용은 두 개이므로 전체 사전등록 판정은 `partial`이다.

## 무엇을 확인했나

고정한 조건은 seed 200, 공식 checkpoint SHA `f2aa77bf0349c10ec2a00071162e37123d0cf80f3447c3989c17a618e1b24ad2`, 5cm 단차·30cm 폭, Isaac Lab `90b79bb2d44feb8d833f260f2bf37da3487180ba`다. 명령 0.25/0.75m/s와 두 방향을 실행 전에 고정했고 추가 속도 탐색이나 자동 재시도는 하지 않았다.

0.25m/s는 공식 학습 명령 범위 안이며 정지 명령으로 치환되는 값도 아니다. 범위 안이라는 사실이 계단 성공을 보장하지는 않는다. 이번 대조는 이 정책·지형·초기 상태에서 명령 선택이 결과를 바꿨음을 보여준다. 저속 실패의 내부 학습 메커니즘이나 다른 계단·seed에서의 성공률을 규명한 결과는 아니다.

이 자료는 이미 본 정책의 단일 계보 개발 재생이다. 모델 API·새 학습·교정·회복 실행은 모두 0이다. H1과 Go2의 상행·하행 영상을 확보했지만, 이를 RL Triage가 두 로봇의 계단 고장을 수리한 성과로 표현하지 않는다. [H1 기존 영상](media/stairs_h1_s172_20261030_v1/README.md), [기존 Go2 자세 감사](STAIRS-MOTION-INTEGRITY.md).

## 재현과 근거

- [실행 전 등록](../bench/protocols/go2_stairs_command_contrast_s200_20261101_v1.json), 등록 commit `fc15d6ce193781d4244e6cae8b9bdc558cd0853d`.
- [공개 요약](../evals/results/go2_stairs_command_contrast_s200_20261101_v1/summary.json), [조건별 결과](../evals/results/go2_stairs_command_contrast_s200_20261101_v1/cases.json), [출처](../evals/results/go2_stairs_command_contrast_s200_20261101_v1/provenance.json), [영상·JSON SHA](../evals/results/go2_stairs_command_contrast_s200_20261101_v1/manifest.json).
- [재생 adapter](../bench/run_go2_stairs_velocity_v1.py), [원문 검증·공개 exporter](../evals/export_go2_stairs_command_contrast_v1.py).
- SDK 실행 4회는 모두 자연 종료·exit 0·소유 Job 정리를 확인했다. 자식 wall 합은 270.734초이며 시작부터 종료·정리 확인까지의 시간이다. GPU kernel 시간이나 학습 비용 절감으로 해석하지 않는다.

기존 실패 원문을 덮어쓰지 않았다. 공개 영상은 원본 MP4를 바이트 그대로 복사했으며 속도 변경·편집·성공 구간 선택은 없다.
