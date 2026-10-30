# 이족·계단 녹화를 공통 진단 입력으로 읽기

[`evals/rough_play_state_adapter_v1.py`](../evals/rough_play_state_adapter_v1.py)는 원래 PC의 계단 녹화기
`run_stairs_playback_v2.py`가 쓴 H1·Go2 상행/하행 clip을 공통 진단 입력용 관측 수치로 읽는다.
CPU에서 실행하며 Isaac·torch·모델 API를 부르지 않는다. 보행 영상이나 녹화기 라벨을 RL Triage 교정 성공으로 바꾸지 않는다.

녹화기는 저장소에 넣지 않았다. [사전등록 후보 protocol](../bench/protocols/rough_play_state_contract_20261030_v1.json)이
녹화기 sha256(`6858d737…5b5a`)과 녹화기가 실제로 쓰는 spec·manifest·trace 필드를 고정한다.
상태는 `unexecuted_preregistration_candidate`다. 실제 계단 trace는 아직 없고 synthetic_cpu fixture로만 검사했다.

## 입력

clip 하나는 녹화기 `--out` 폴더와 그 `--spec` 파일의 쌍이다.

| 파일 | 녹화기가 쓰는 내용 |
| --- | --- |
| spec | robot_role·direction·task·seed·checkpoint SHA·runner/Isaac Lab source SHA·capture 고정값 |
| `intent.json` | spec SHA, checkpoint SHA, `new_training=0`, `model_api=0`, 진단·회복 주장 없음 |
| `manifest.json` | spec 사본, 지형·명령 설정, 실행 Isaac Lab source, 녹화기 assessment, video·trace SHA, 카메라·render 동기화 |
| `shutdown/01~04-*.json` | app 시작 → result-ready(manifest SHA) → close 진입 → close 반환 |
| `state-trace.jsonl` | frame별 `root_pos_w`, `root_quat_w`, `root_lin_vel_w`, `terrain_origin_w`, `terminated` |
| `video/*.mp4` | RecordVideo 영상 하나 |

`failure.json`이 있으면 실패로 봉인된 녹화이므로 읽지 않는다.

## 교차 검증

하나라도 다르면 `FactsError`로 부분 결과 없이 멈춘다.

| 대조 | 확인 |
| --- | --- |
| spec ↔ protocol | schema·키, 등록 네 조합의 robot_role/direction, robot task 표, capture 고정값, runner LF SHA = 녹화기 SHA, Isaac Lab source 목록 |
| intent ↔ spec | spec SHA, checkpoint SHA, 관측 전용 재생 표시 |
| manifest ↔ spec | clip_id·robot·direction·task·source·checkpoint, direction별 지형·기하, seed·명령, 실행 Isaac Lab source, 카메라, render 동기화, RecordVideo 기록 |
| marker ↔ manifest | 네 marker의 순서·pid·시각, result-ready의 manifest SHA |
| video·trace ↔ manifest | clip 안 상대경로, mp4 하나, 파일 SHA |
| trace ↔ assessment | `root_dx_m`·`root_dz_m`·`termination_seen`·`trace_rows`. frame 순서, `sim_time_s = frame / 50`, done 위치, 단일 terrain origin |
| clip 묶음 | clip_id와 (robot, direction) 중복 없음, 같은 robot은 같은 checkpoint, 다른 robot은 다른 checkpoint, 같은 Isaac Lab commit |

protocol의 `registered_checkpoints`가 null이 아니면 robot별 checkpoint SHA도 대조한다. 지금은 공식 checkpoint SHA를 이 PC에서 확인할 수 없어 null이다.

## 수치와 가용성

trace에 기록된 값만 쓴다. 좌표는 모두 world다.

| fact | 정의 |
| --- | --- |
| `root.world_displacement` | 사용 구간 첫·끝 행의 `root_pos_w` 차와 xy norm |
| `root.world_xy_path_length` | 인접 행 xy 거리의 합 |
| `root.height_above_terrain_origin` | `root_pos_w.z - terrain_origin_w.z`. 발밑 지면 높이가 아니다 |
| `root.world_horizontal_speed`, `root.world_vertical_velocity` | `root_lin_vel_w` 성분 통계 |
| `command.configured` | manifest의 명령 설정값. 좌표계와 frame별 명령은 기록되지 않는다 |
| `command.tracking_error` | 항상 unavailable |
| `episode.done`, `trace.coverage` | done 여부·frame, 행 수 |

joint·contact·학습 telemetry·종료 원인·영상 내용은 기록에 없으므로 `field_availability`에 `unavailable: 이유`로 남기고 0으로 채우지 않는다.
`root_quat_w`는 기록되지만 성분 순서가 선언되지 않아 수치로 쓰지 않는다.

## 판정하지 않는 것

- **추종 오차.** 명령 좌표계, frame별 명령, quaternion 성분 순서가 없어 world 속도를 몸체 좌표로 바꾸지 않는다.
- **넘어짐.** done은 종료 항을 기록하지 않는다. done 행은 Isaac Lab이 같은 step에서 reset한 뒤의 상태일 수 있어 이동·높이·속도 수치에서 뺀다.
- **계단 성공·교정 성공.** root 높이 변화와 이동량은 수치로만 남긴다. 녹화기 outcome(`fall`·`completed`·`failed`)은 녹화기 상수 규칙의 라벨로 옮기고 `used_for_decision: false`다. 어댑터는 임계값을 만들지 않는다.
- **모델 해석.** 이 입력에는 영상 모델 해석이 없다. simulator fact와 녹화기 라벨만 센다.

## 적용성

| 구분 | 상태 |
| --- | --- |
| 진단 준비도 | `observation_only`. rough 계단 task에 등록된 정상 기준·가설·판정 규칙이 없다 |
| 검사 적용성 | `probe_loop.PROBES` 여섯 개 모두 HOLD. flat task 고리에서 정의·검증된 probe다 |
| 교정 적용성 | `correction_registry.REGISTRY` 교정 모두 HOLD. `check_repair`는 원장에 쓰지 않고 HOLD를 돌려준다 |

rough task에 기존 flat probe·교정을 자동 적용하지 않는다. 요청 수는 0이다.

## 실행

실제 clip 네 개가 도착하면 저장소 루트에서 실행한다. 출력 폴더는 새로 만들며 이미 있으면 거부한다.

```bash
python evals/rough_play_state_adapter_v1.py --clip "$H1_UP" "$H1_UP_SPEC" --clip "$H1_DOWN" "$H1_DOWN_SPEC" --clip "$GO2_UP" "$GO2_UP_SPEC" --clip "$GO2_DOWN" "$GO2_DOWN_SPEC" --out evals/results/rough_play_state_20261030_v1
```

각 변수는 녹화기 호출의 `--out` 폴더와 `--spec` 파일이다. 공개 출력 `rough_play_facts.json`에는 경로를 넣지 않고,
입력 경로는 `rough_play_facts_metadata.json`에만 둔다. checkpoint SHA를 등록하려면 v1을 고치지 않고 새 protocol 파일을 만들어 `--protocol`로 넘긴다.

개발 회귀:

```bash
python -m pytest -q tests/test_rough_play_state_adapter_v1.py
```

Trace 행에는 clip·robot 식별자가 없어 trace·manifest·marker를 함께 바꾼 위조를 이 어댑터가 잡지 못한다. 녹화 직후 manifest SHA를 따로 기록하면 이 경우를 막을 수 있다. Done 행이 reset 뒤 상태인지는 Isaac Lab v2.1.1 소스로 이 PC에서 확인하지 않았고, 녹화기 assessment의 `root_dx_m`·`root_dz_m`은 이 행을 포함한다. 녹화기의 `record_video_contract`는 실행 step이 아니라 고정값(625 step)으로 계산되므로 done이 있으면 실제 영상 frame 수와 다를 수 있으며, 어댑터는 영상을 디코딩하지 않고 SHA만 대조한다. 실제 계단 trace와 공식 checkpoint SHA는 아직 없다. 위 검사는 synthetic_cpu fixture의 CPU 회귀이며 실제 녹화의 성공이나 실행 준비 완료를 뜻하지 않는다.
