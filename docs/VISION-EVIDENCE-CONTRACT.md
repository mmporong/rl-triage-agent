# 영상과 운동 측정 입력 계약

`evals/vision_evidence_contract_v1.py`는 로봇 영상에서 프레임 여섯 장을 추출하고,
정책 checkpoint와 같은 Play 실행의 운동 기록을 묶어 모델 입력을 준비한다.
영상 준비와 검증은 CPU에서 수행하며 모델 API를 호출하지 않는다.

## 입력과 출력

`prepare(spec_path, output_dir)`는 새 출력 폴더에 원본 영상, checkpoint,
추출 프레임, 측정 기록과 `contract.json`을 보관한다. 이미 있는 출력은 덮어쓰지 않는다.
`verify(contract_path)`는 파일 해시와 입력 결합을 다시 확인한다.
`public_model_view(contract_path)`는 검증된 이미지와 수치로 모델 메시지를 만든다.

입력 명세의 schema는 `vision_evidence_prepare_v1`이다.

| 필드 | 내용 |
| --- | --- |
| `play_run_id` | 영상과 측정 기록을 연결하는 실행 ID |
| `source_lineage_id`, `parent_source_lineage_id` | 데이터 분할에 사용할 정책 계보와 부모 계보. 부모가 없으면 null |
| `video`, `checkpoint` | 파일 경로, 공개용 논리 ID, 실행 ID |
| `frames` | 추출을 요청할 시각 여섯 개, 실행 ID, 원본 영상 SHA |
| `measurement` | 같은 실행의 측정 파일 정보. 기록이 없으면 null |
| `play` | task, 카메라, 명령 이름과 좌표계, Play 설정 SHA |

프레임 파일을 임의로 지정하는 필드는 받지 않는다. 준비기가 원본 영상에서 추출하고
추출 기록에 도구 버전과 시각을 남긴다. 생성한 JPEG는 해시와 디코딩 검사를 거친다.
요청 시각과 실제 원본 frame PTS를 나눠 기록하고, 차이가 고정 허용값 0.05초를 넘으면 거부한다.
첫 원본 frame PTS를 영상 상대시각 0으로 두며, 운동 sample은 선택된 frame의 실제 상대시각과 일치해야 한다.
모델에 노출하는 명령 이름은 `base_velocity`, 명령 좌표계는 `body`, 운동 좌표계는
`body` 또는 `world`로 제한한다.

## 측정과 해석의 경계

운동 기록에는 프레임 시점의 명령·실제 선속도·회전속도·자세를 넣는다.
측정 파일의 실행 ID, 영상·checkpoint SHA, 좌표계와 sample 시각이 입력과 맞아야 한다.
영상과 별도로 실행한 고정 평가 결과는 이 운동 기록으로 사용할 수 없다.
기록이 없으면 해당 수치는 `null`이며 0이나 추정값으로 채우지 않는다.

모델 메시지에는 이미지, 프레임 ID·시각과 허용된 수치만 넣는다.
실행·정책 계보, 파일명, checkpoint 해시와 평가 정답은 로컬 provenance에 둔다.
여섯 프레임은 한 영상의 표본으로 세며, 모델 설명을 독립 측정으로 추가하지 않는다.
이 입력 계약은 원인 확정이나 회복 판정을 수행하지 않는다.

## 실행

FFmpeg와 ffprobe가 PATH에 있어야 한다. 입력 명세와 원본 자료를 준비한 뒤 실행한다.

```powershell
cd "$HOME/rl-triage-agent"
python -m evals.vision_evidence_contract_v1 prepare --spec input/vision_spec.json --output artifacts/vision-input-001
python -m evals.vision_evidence_contract_v1 verify --contract artifacts/vision-input-001/contract.json
python -m evals.vision_evidence_contract_v1 model-view --contract artifacts/vision-input-001/contract.json --output artifacts/vision-input-001/model_view.json
```

출력 묶음에는 원본 정책과 영상이 포함된다. 공개할 때에는 별도 내보내기에서
데이터 공개 범위와 개인 경로를 확인한다. 모델 입력 파일에도 실제 이미지가 포함된다.

개발 회귀는 다음 명령으로 실행한다.

```powershell
cd "$HOME/rl-triage-agent"
python -m pytest -q tests/test_vision_evidence_contract_v1.py
```

검증 범위는 파일·시각·출처 결합과 모델 입력 준비다. 영상 관찰의 정확도,
검사 선택의 기여와 교정 후 회복은 각 단계의 사전등록된 실행에서 평가한다.
