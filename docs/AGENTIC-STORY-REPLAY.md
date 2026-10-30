# 근거로 따라가는 결과 재생 화면 (agentic story v1)

[재생 화면](replay/agentic-story-v1.html)은 검사 선택 → 검사 결과 → 판단 변화 → 교정 → 회복/HOLD를 공개 결과 원문으로 다시 보여 준다. 이미 본 결함 종류의 개발 기록이며 진단 정확도·교정 성공률·일반화·GPU 절감의 근거가 아니다.

## 만들기와 최신성 확인

```bash
PYTHONUTF8=1 PYTHONPATH=src python evals/build_agentic_story_replay_v1.py          # HTML 생성
PYTHONUTF8=1 PYTHONPATH=src python evals/build_agentic_story_replay_v1.py --check  # 원문과 같으면 exit 0, 낡았으면 1
PYTHONUTF8=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_agentic_story_replay_v1.py
```

빌더는 커밋된 공개 파일만 읽는다. 결과 원문이 커밋 전이거나 작업 트리 바이트가 커밋과 다르면 생성을 멈추고, 영상·계단 슬롯은 대기·불일치로 남긴다. GPU·모델 API·네트워크를 쓰지 않는다.

## 화면과 원문

| 화면 | 원문 | 표시 범위 |
|---|---|---|
| R1 PPO epoch 결함, seed 2071 | `evals/results/r1_repair_s2071_20261016/{summary,probe_and_correction,model_evidence}.json` | 첫 검사 `P_reward`만 model, 이후 다섯 검사는 rule(`registered_next`·`registered_new_probe`). 회복 recovered |
| F1+R1 복합 결함 v1, seed 2110·2111 | `evals/results/residual_failure_public_20261017_v2/{summary,model_contract_audit}.json` | 첫 교정 not_recovered → 두 번째 진단 invalid(`D_EXPECTED`) → HOLD |
| F1+R1 복합 결함 v2, seed 2120·2121 | `evals/results/residual_success_public_20261017/{summary,model_audit}.json` | 두 고리 모두 model 선택, 첫 교정 not_recovered → 두 번째 교정 뒤 recovered. 최종 체크포인트가 같은 seed 정상과 같음 |
| U1 함수군 교정, seed 2051·2062 | `evals/results/{u1_scope_extension_s2051_20261015,novel_u1_scope_confirmation_s2062_20261016}/` | 모델 호출 0회, 검사 선택 주체 host, recovered |
| 새 3.5 fresh-frame, seed 2221 | `evals/results/fresh_frame_probe_choice_dev_s2221_20261029_v1/`, `fresh_frame_runtime_{prefix,operation}_*`, `agentic_f1_fresh_render_dev_20261029h_v1/summary.json`, `fresh_brev_render_creation_20261029f_v1/summary.json`, `docs/media/agentic_f1_fresh_render_dev_20261029h_v1/` | `P_reward` 선택과 telemetry 인용 6·frame 인용 0까지. 검사·판단·교정·회복은 pending. 이전 렌더 환경 생성은 실행 실패로 표시 |
| H1·Go2 계단 슬롯 4개 | `docs/media/**/manifest.json` 중 `stairs_playback_manifest_v2` | 커밋된 manifest와 같은 SHA의 영상만 ready로 계산하며 나머지는 대기 또는 불일치 |
| 원문 결합 | 위 파일 전체 | 경로·sha256·커밋 상태·모델 실행 |

칸마다 `S번호#JSON pointer`를 붙인다. 테스트는 모든 칸의 pointer 값을 원문에서 다시 읽어 화면 값과 비교한다.

## 표시 규칙

- **선택 주체.** 검사 선택 칸의 model/rule/host는 원문 필드와 값의 쌍(`SELECTOR_BASIS`)으로만 정한다. 등록되지 않은 쌍이면 생성을 멈춘다. R1에서 모델의 첫 선택 뒤 규칙이 고른 검사를 model로 바꾸면 `check_story`가 거부한다.
- **모델 실행 분리.** 원문마다 읽은 모델 실행(3 Super, 3.5, 모델 호출 0회)을 기록한다. 한 화면에 다른 실행의 원문이 섞이면 거부한다. U1과 옛 3 Super 결과는 3.5 경로의 회복으로 합산하지 않는다.
- **3.5 경로.** 첫 모델 선택 뒤 단계에는 pending·HOLD·실행 실패 외의 배지를 붙이지 않는다. 원문에 검사 실행·원인 확정·교정·새 회복이 생기면 빌더가 멈추므로 화면 계약을 먼저 갱신한다.
- **상태 배지.** confirmed·not_recovered·recovered·HOLD·실행 실패·pending과 검사 결과 normal·abnormal을 구분한다. 회복 배지는 원문 값을 그대로 쓴다.
- **영상.** 커밋된 파일이 manifest SHA와 같을 때만 재생 요소를 넣는다. 파일이 없으면 대기, 바이트가 다르면 불일치로 자리만 남긴다. [렌더 감사](../evals/results/vision_render_integrity_audit_20261028_v1/summary.json)의 stale-render 영상 SHA와 같으면 불일치로 둔다. 영상 역할은 입력 녹화(정상·F1 결함)와 계단 재생뿐이며 교정 영상 역할은 없다.
- **계단 슬롯.** 커밋된 `stairs_playback_manifest_v2` manifest의 robot·방향·task가 등록값과 같고, manifest 폴더 기준 `video.path` 영상의 SHA가 같을 때만 ready다. 같은 슬롯의 manifest가 둘이면 불일치다. ready여도 공식 rough 정책 재생이며 교정 영상이나 계단 성공 판정이 아니다.
- **외부 리소스.** HTML은 CDN·웹 폰트·http(s) 주소를 참조하지 않는다. 영상은 `../media/...` 저장소 상대 경로로만 연다.

## 확인 기록 (2026-10-30)

- CPU: `tests/test_agentic_story_replay_v1.py` 9개 통과. 기존 `test_build_transferred_vision_replay_v1.py`와 `test_replay_page.py`의 Sources 판정 테스트 8개 통과. `--check` current.
- 브라우저: 이 PC의 Chrome 확장은 localhost 서버에 연결하지 못했다(`ERR_CONNECTION_REFUSED`). 대신 headless Edge를 1920×1080으로 띄워 CDP로 확인했다. 8개 화면 모두 overflow 0, 잘린 칸 0. 외부 요청 0. 영상 2개 모두 metadata 로드(12.5초, 640×480), 3초 seek, 재생 중 시간 증가를 확인했다. Range 요청을 지원하지 않는 `python -m http.server`에서는 seek가 0초로 돌아가므로 Range를 지원하는 정적 서버로 확인했다.

이 화면은 이미 본 seed의 개발 기록을 다시 보여 줄 뿐 새 실험 결과가 아니다. 3.5 경로의 검사·교정·회복과 영상의 검사 선택 기여는 아직 측정하지 않았다. H1·Go2 계단 슬롯의 ready 수는 HTML을 생성한 시점에 커밋된 manifest와 영상 SHA를 기준으로 계산한다. 새 계단 자료를 커밋한 뒤에는 HTML을 다시 생성해야 한다.
