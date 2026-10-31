# 실패 진단과 재발 방지 기록

이 문서는 반복 실패를 찾고 다음 검증을 정하는 색인이다. 구현 오류, 환경 조건,
모델 판단 실패와 회복 미확정을 구분한다. 상태는 CPU 검증과 실제 실행 확인을 나눠 적는다.
검사 하나가 통과해도 전체 녹화나 회복이 완료된 것으로 바꾸지 않는다.

## 모델 판단과 회복에서 남은 문제

| ID | 실제로 막힌 지점 | 해본 방법과 결과 | 확인한 것과 남은 판단 | 근거 |
|---|---|---|---|---|
| RL-01 | 보상 결함을 고쳐도 이어 학습으로 회복하지 못함 | 이어 학습·노이즈 std 재설정은 두 seed에서 미회복. optimizer·critic 재설정도 미회복. actor 마지막 층 재설정은 300회에서 기존 seed 중 하나만 회복했고 새 seed 두 개는 회복 | 등록 회복 규칙은 유지한다. 가소성 상실·critic 오보정·탐색 붕괴 중 하나를 원인으로 확정할 증거는 부족하다 | [회복 개입과 결과](TRAINING-CORRECTION.md) |
| RL-02 | actor 재설정의 학습 예산을 줄이면 회복하지 못함 | 별도 두 seed에서 100·150·200회 모두 미회복, restart 300회는 두 seed 모두 회복 | 이 결과로 짧은 회복 경로를 등록하지 않는다. 300회 actor 개입의 회복과 계산 비용 감소는 별도 질문이다 | [단축 예산 결과](../evals/results/actor_reset_budget_public_20261017/summary.json) |
| RL-03 | 동결 여섯 검사가 카탈로그 밖 결함을 모두 정상으로 읽음 | R1에서는 `none_supported`로 멈췄다. 그 결과를 본 뒤 optimizer 갱신 횟수 검사와 epoch 복원을 등록해 새 seed 종단 실행에서 회복 | 특정 확장 등록의 개발 확인이다. 처음 보는 다른 optimizer 결함까지 구분한다는 근거는 없다 | [원래 R1 중단](../evals/results/upstream_r1_loop_20261014/result.json), [검사와 수리](OPTIMIZER-OBSERVABILITY.md) |
| RL-04 | PPO 계산 오류 종단 실행에서 모델 판단이 계약을 어김 | 처음 답과 등록된 피드백 재요청 답이 모두 무효여서 검사·교정 전에 `model_invalid`로 중단 | 행동 실패는 평가로 확인됐다. 해당 실행의 진단·수리는 완료되지 않았다. 무효 답을 규칙 결정으로 대체하지 않았다 | [종단 실행의 판단 보류](PPO-GRADIENT-CHECK.md), [결과](../evals/results/ppo_ratio_repair_s2165_20261023/summary.json) |
| RL-05 | 올바른 원인 후보에도 존재하지 않는 근거를 붙임 | D3에서 원인 변경을 1위로 고른 답이 없는 diff 줄을 인용해 무효였다. hunk와 실제 코드 문자열을 검증하는 계약을 별도로 만들었다 | 원인 순위와 근거 유효성을 분리한다. 새 모델의 출력·근거·도구 인자 비교도 등록한 해결 기준을 충족하지 못했다 | [D3](D3-MULTI-CHANGE.md), [모델 비교](NEMOTRON-ENDPOINT-COMPARISON-V3.md) |
| RL-06 | 다른 로봇에 같은 결함을 넣어도 실패 시연이 성립하지 않음 | ANYmal-C F2는 적용 불가, F6는 source 행동 실패가 없어 종단 시연을 중단 | 실행·판정 오류로 취급하지 않는다. 활성 함수·주입 효과·source 실패·교정 범위를 적용성 계약으로 먼저 확인한다 | [ANYmal-C](ANYMAL-C-PORT.md), [조합 적용성](COMBINATION-APPLICABILITY.md) |
| RL-07 | 모델의 검사 선택이 규칙보다 나은지 결론을 내리지 못함 | seed 2171의 한 복합 사례에서 세 선택 방식은 회복했고 한 방식은 인프라 재시도 소진 후 HOLD였다 | 한 사례의 서로 다른 arm으로 선택 정책의 우위를 주장하지 않는다. 영상 추가 효과도 별도 사전등록 비교 전에는 미측정이다 | [선택기 비교](SELECTION-COMPARISON.md), [영상 선택 비교](VISION-INSPECTION-SELECTION.md) |
| RL-08 | 프레임 근거를 더해도 첫 검사 선택이 달라지지 않음 | seed 2220·2221의 정상·F1, 두 구간·세 입력 조건에서 24/24 유효 응답이 모두 `P_reward`를 선택했다. 실행 검사는 정상과 F1을 구분했다 | 사전등록 판정은 `observation_only`다. 이 등록 결함 비교에서 영상에 따른 선택 개선이 없었다. 정상 창의 검사 요청과 오진 확정은 구분한다 | [실측 비교](../evals/results/vision_transferred_selection_dev_20261027_v1/summary.json) |
| RL-09 | VL이 예상 수치 위치를 영상 관찰처럼 답하며 렌더 오류를 놓침 | 좌표와 여섯 이미지를 함께 준 V1은 8/8 형식 유효였지만 오류 응답 4개도 모두 일치로 판정했다. V2는 좌표를 숨기고 단일 이미지의 bbox만 읽어 호스트가 비교했으며, 오류 프레임 4/4를 격리하고 수정 프레임 4/4를 유지했다 | 이미 본 4녹화·3checkpoint의 개발 확인이다. 좌표 비공개·단일 이미지·detail 생략을 함께 바꿔 개별 원인은 분리하지 못했다. PNG 형식이나 반환 alias만으로 이미지 미처리를 단정하지 않는다 | [실패·보완·소비 차단](VISUAL-INTEGRITY-OBSERVER.md) |

## 클라우드 영상 수집

기준 환경은 Isaac Sim 4.5, Replicator 1.11.35, Python 3.10이다. 다른 버전에서는
설치된 SDK의 payload와 종료 동작을 다시 확인한다. 아래 상태는 새 데이터 수집 전체의 완료 판정과 별개다.

### CAP-01: RGB Writer 값이 배열이 아니라고 실패

- 현상: 첫 Writer 콜백에서 `TypeError: Writer rgb은 ndarray여야 한다`가 발생했다.
- 조건: custom Writer가 `data_structure = "annotator"`를 사용한다.
- 확인된 원인: 이 모드에서 SDK는 RGB를 `{"data": ndarray}`로 감싼다.
  기존 decoder와 테스트 fixture는 RGB 배열이 바깥에 놓이는 legacy 형태를 가정했다.
- 함께 확인한 계약: segmentation은 `{"data": mask, "idToLabels": mapping}` 형태다.
  `{"data": mask, "info": ...}` 형태와 구분해야 한다. RenderProduct 해상도는 NumPy 배열도 올 수 있다.
- 조치: SDK 형식의 fixture로 RGB·분할 데이터 검증을 고치고, 해상도 수치는 정규화하되 원문 형식을 journal에 남긴다.
  같은 콜백·RenderProduct·시각 결합과 배열 dtype·shape 검사를 유지한다.
- 상태: 실제 실패 원인과 SDK 소스 계약을 확인했다. 수정의 CPU 검증과 새 GPU 확인 결과는 실행별 기록에 남긴다.
  실패한 실행의 RGB 내부 dtype·shape는 기록되지 않았으므로 추정값을 결과로 쓰지 않는다.
- 재발 방지: legacy 형태를 거부하는 검사와 실제 SDK 형태의 fixture를 함께 둔다.
  첫 콜백과 실패 콜백에는 픽셀·라벨 본문을 제외한 keys·type·shape·dtype 구조를 write-once로 저장한다.
- 코드·검사: [decoder](../bench/vision_writer_payload_v1.py),
  [SDK 형식 회귀](../tests/test_vision_writer_nonlegacy_v1.py),
  [콜백 진단 기록](../tests/test_vision_capture_descriptor_v1.py).
- 이슈/PR 판단: 확인된 범위는 프로젝트 adapter와 fixture의 결함이다.
  이 기록만으로 Replicator 자체의 버그라고 보고하지 않는다.

### CAP-02: 준비 확인 횟수를 너무 빨리 소진

- 현상: asset 준비 확인 240회를 약 3.36초에 소진하고 `poll_budget_exhausted`로 끝났다.
- 시도: 전체 대기 상한과 확인 횟수를 유지한 채 확인 간격을 배분했다.
- 실제 후속 결과: 40회·약 4.65초에 연속 idle 조건을 통과했다. 이후 CAP-01에서 중단했다.
- 원인 범위: 빠른 polling이 대기 구간을 소진한 사실을 확인했다. 모든 asset 지연의 원인이 같다는 뜻은 아니다.
- 재발 방지: update 수와 경과 시간, loading·streaming 상태, 물리 상태 불변 여부를 함께 기록한다.
- 코드·검사: [준비 장벽](../bench/vision_capture_readiness_v1.py),
  [준비 장벽 회귀](../tests/test_vision_capture_readiness_v1.py).

### CAP-03: 전송 중 자원 관측의 유효기간 만료

- 현상: 두 번의 원격 복사가 약 37.53초 걸려 30초 freshness 검사에서 GPU 작업 전에 멈췄다.
- 시도: provider 원문과 호출 자료를 단일 전송에 결합했다. 30초 기준은 유지했다.
- 실제 후속 결과: no-GPU 전송 확인의 조회 응답부터 실행 응답까지 약 2.10초였고 bootstrap·등록 단계도 통과했다.
- 재발 방지: 조회 시각, 실제 CLI 인자와 stdout·stderr, 전송 snapshot의 해시를 묶는다.
  CLI 내부 재시도가 같은 작업을 다시 전달해도 미완료 intent를 실행 완료로 추정하지 않는다.
- 코드·검사: [단일 전송 경계](../bench/vision_onehop_transfer_v1.py),
  [전송·중복 실행 회귀](../tests/test_vision_onehop_transfer_v1.py).

### CAP-04: marker의 출력 prim 형식 불일치

- 현상: marker binding이 SDK가 반환한 prim 목록을 찾지 못해 멈췄다.
- 확인된 원인: `get_output_prims()` 반환값은 `{"prims": [Usd.Prim]}` mapping이다.
- 조치: 단일 `Usd.Prim`, 현재 stage, Imageable 여부와 자손의 physics API 부재를 검사한다.
- 상태: 실제 후속 실행이 marker binding을 지나 Writer 콜백까지 도달했다.
- 재발 방지: 문자열 경로나 다른 stage의 prim을 거부하고 실제 in-memory USD prim 회귀를 유지한다.
- 코드·검사: [marker binding](../bench/vision_atomic_capture_v1.py),
  [USD prim 회귀](../tests/test_vision_atomic_capture_v1.py).

### CAP-05: 시각 필드와 픽셀 최신성을 같은 것으로 취급

- 현상: 시각 adapter 검사는 통과해도 카메라 A→B→A의 픽셀 비교가 실패했다.
- 해본 방법: renderer accumulation 초기화와 FXAA 설정의 실제 readback을 확인했다.
  FXAA readback 통과만으로 픽셀 비교 실패가 해결되지는 않았다.
- 현재 조치: RGB·분할·카메라·시각을 같은 Writer 콜백으로 수집하고,
  동기화 검사에서 프레임별 marker code를 함께 검사한다.
- 상태: 새 atomic 경로의 전체 GPU 동기화 통과는 미확인이다.
  픽셀 차이의 원인을 시간 누적, 장면 변경, renderer 지연 중 하나로 확정하지 않았다.
- 재발 방지: timestamp 일치, 픽셀 최신성, 물리 비간섭을 따로 검사한다.
  기준을 느슨하게 바꿔 촬영 성공으로 처리하지 않는다.
- 코드·검사: [atomic 동기화](../bench/vision_frame_sync_v1.py),
  [smoke 판정](../bench/vision_atomic_smoke_v1.py),
  [FXAA 설정](../bench/vision_fxaa_policy_v1.py), [수집 계약](VISION-BREV-DATA.md).
- 이슈/PR 판단: 원인 미확정이다. RL 없이 재현되는 최소 장면과 실제 callback 원문을 확보한 뒤 SDK 이슈 여부를 판단한다.

### CAP-06: Linux renderer 라이브러리와 annotator 등록

- 현상: renderer 자식 종료 또는 capture annotator 등록 실패로 촬영을 완료하지 못했다.
- 조치: renderer 공유 라이브러리 검사를 분리하고, Isaac node 확장 활성화와
  `rgb`·`IsaacReadTimes`·`ReferenceTime` 등록 증거를 촬영 전에 저장한다.
- 상태: 후속 실행이 등록과 준비 장벽을 지나 Writer 콜백까지 도달했다.
  새로운 VM의 전체 환경 복원은 별도로 검증한다.
- 재발 방지: 학습 가능 여부로 renderer 준비 완료를 추정하지 않는다.
- 코드·검사: [라이브러리 검사](../bench/check_vision_renderer_libraries_v1.py),
  [등록 확인 worker](../bench/vision_play_worker_v2.py),
  [감독 worker](../bench/vision_play_worker_v3.py).

### CAP-07·08: 첫 semantic 출력이 비고 같은 프레임 콜백이 반복됨

- 현상: 준비 장벽을 통과한 뒤 RGB는 `uint8[480,640,4]`였지만 semantic은
  `float64[0]`이었다. 기존 Writer는 첫 decode 오류를 저장하고 이후 콜백 처리를 중단했다.
- 후속 진단: 같은 Writer를 유지하며 두 event의 콜백 네 개를 모두 기록했다.
  frame 32에서는 빈 semantic 뒤 `uint32[480,640]` 출력이 왔고,
  frame 33에서는 유효한 출력 두 개가 왔다. 두 event 모두 물리 상태는 바뀌지 않았다.
- 확인한 범위: 이 실행의 semantic 출력은 계속 비어 있지 않았다.
  초기 출력의 준비 시점과 첫 오류를 유지하는 처리 방식이 기존 실패에 관여한다는 설명을 지지한다.
  같은 프레임 번호의 두 출력이 픽셀까지 같은지는 이 진단으로 확인하지 못했다.
- 다음 검증: 준비용 출력과 측정용 출력을 구분하고, 중복 후보의 RGB·분할·카메라·시각 원문 해시를 비교한다.
  해시가 다르거나 시점 근거가 부족하면 중복으로 합치지 않는다.
  이 진단의 정상 종료를 카메라/marker 동기화나 전체 녹화 성공으로 처리하지 않는다.
- 함께 확인한 병목: 소스 검증을 Git blob 일괄 조회로 바꾼 뒤 Brev의 입장 모의 검증은
  프로토콜 네 번과 현재 파일·런타임 검사 등을 포함해 약 0.487초였다.
  proof 발급부터 실행 시작까지 약 0.016초였으며 freshness 30초 기준은 유지했다.
- 근거: [개발 진단 결과](../evals/results/vision_semantic_sdk_dev_20261026r/summary.json),
  [진단 사전등록](../bench/protocols/vision_semantic_sdk_20261026_v2.json).
- 원문 해시를 추가한 후속 진단에서는 frame 60과 61에 정상 콜백이 하나씩 왔다.
  각 콜백의 14개 항목 해시를 남겼지만, 같은 프레임의 비교쌍이 없어 동등성 결과는
  `insufficient`였다. 이 결과를 중복 정규화의 승인으로 쓰지 않는다.
  [후속 진단 결과](../evals/results/vision_semantic_fingerprint_dev_20261026s/summary.json).
- 수정 확인: 한 Writer에서 준비 출력과 측정 네 장을 분리한 새 경로가 실제 GPU smoke를 통과했다.
  준비 frame 32, 측정 frame 33~36의 콜백 다섯 개에서 journal 네 개를 만들었다.
  기존 marker·시각·A-A-B-A 픽셀·물리 불변·정리 기준을 유지했고, 회수한 원문에서도 재검증했다.
  이번 실행에 빈 출력은 없었다. 이 결과는 정지 장면 동기화 확인이며 정책 전체 녹화는 다음 단계다.
  [준비 분리 개발 결과](../evals/results/vision_prepared_capture_dev_20261026t/summary.json) ·
  [사전등록](../bench/protocols/vision_prepared_capture_20261026_v1.json).
- 이슈/PR 판단: 첫 오류 이후 출력을 버리는 프로젝트 처리와 SDK 초기 출력을 구분해 기록한다.
  SDK 자체의 결함으로 확정하지 않는다.

### CAP-13·14: 전체 녹화 전 파일 오류와 최초 예외 소실

- 현상: 첫 전체 녹화는 `run_real`의 `FileNotFoundError`로 Writer를 붙이기 전에 멈췄다.
  SDK가 예외 타입만 남겼고, 부모가 실패 보고서와 부분 보고서를 혼재 오류로 처리해 최초 경로와 스택을 가렸다.
- 수정: 영상 인코더의 실행 파일·버전을 미리 확인하고, 최초 예외의 메시지·스택을 앱 종료 전에 저장한다.
  실패 보고서와 부분 보고서를 하나의 실패 증거로 검증하며, 종료 코드가 0이어도 실패 기록이 있으면 거부한다.
- 실행 연결: 인코더 준비 스크립트의 개행 escape 오류와 bootstrap 출력 필드 불일치를 수정했다.
  생성된 shell과 내장 Python 자체를 구문 검사하고, 실제 bootstrap 출력으로 입장 검증을 왕복한다.
  실패한 준비 명령의 원문은 유지하며, 등록한 복구 명령의 성공 증거가 있을 때만 후속 변경을 허용한다.
- 확인: 새 판은 실제 정책 녹화와 로컬 원문 재검증을 통과했다. 이전 `FileNotFoundError`의 대상은
  원래 스택이 없어 확정하지 않는다. 이 결과를 모델의 진단·회복 성능으로 집계하지 않는다.
- 근거: [실패 결과](../evals/results/vision_prepared_record_dev_20261026u/summary.json) ·
  [후속 녹화 결과](../evals/results/vision_prepared_record_dev_20261026v/summary.json).

### CPU-05: 비용 원장과 전이 참조를 중복 소비로 처리함

- 현상: 최신 Stop 영수증을 누적 비용 원장과 이전 판의 전이 참조에 함께 기록한 새 판이 실행 준비에서 거부됐다.
  GPU·학습·API 실행 전이었다.
- 원인: 두 참조의 출현 횟수를 합쳐 하나만 허용했다. 원장 내 중복과 같은 영수증의 전이 참조는 구분해야 한다.
- 수정 확인: 원장 마지막 항목이 유일하고 전이 참조의 해시가 같아야 통과하도록 분리했다.
  이중 원장·해시 충돌·순서 오류는 거부한다. 실제 사전등록 입력의 준비 생성과
  Windows·Linux Python 3.10의 관련 CPU 검사 각 21개를 통과했다. 새 GPU 실행 결과는 별도로 남긴다.
- 근거: [학습·평가 접두 실행 사전등록](../bench/protocols/vision_source_prefix_20261026_v1.json).

### CAP-15: renderer 검사와 정책별 입력 승인의 범위가 다름

- 확인한 차이: 앞선 정지 장면 smoke는 checkpoint를 불러오지 않았다. 이 결과만으로 다른 정책의 영상 입력을 승인할 수 없다.
- 구현: 새 정책별 smoke는 실제 환경 관측으로 정책 행동을 계산하고 환경을 한 번 진행한 뒤 reset한다.
  이후 기존 A-A-B-A 검사 기준을 적용한다. 완료 표시만 읽는 대신 녹화·검사·checkpoint·자원·학습·고정 평가의
  원문을 다시 검증하며, 기존 실행에 없던 attestation을 만들어 채우지 않는다.
- 확인 범위: 최초 오류 보존, 종료 코드 0의 실패 거부, 준비 완료 표시 위조·영상 교체·근거 변조 거부를
  CPU 회귀로 확인했다. 후속 checkpoint 검사의 실제 실패와 새 순서의 검증은 아래 CAP-16에서 구분한다.
- 코드: [정책별 smoke](../bench/run_vision_checkpoint_smoke_v1.py) ·
  [입력 승인 검증](../evals/vision_record_source_readiness_v1.py).

### TRAIN-01: 학습 실행 이름의 두 검증 계약이 다름

- 현상: 첫 source 학습 예약 뒤 worker가 약 1.37초 만에 종료 코드 1을 반환했다.
  stdout은 비었고 stderr에는 `ValueError` 타입만 남았다. 부모는 `InvalidExecutionEvidence`로 보류했다.
- 원인: worker job은 `vision-s2210-none`을 허용하지만, 다음 residual 학습 매니페스트는
  실행 이름에 영문·숫자·밑줄만 허용한다. 매니페스트 생성이 Isaac 실행보다 앞에 있어 학습은 시작되지 않았다.
- 증거 분리: 소스 전후 일치, worker 정리, GPU idle은 통과했다. worker 완료 결과와 학습 매니페스트는 없었다.
  따라서 모델의 진단이나 학습 회복 실패로 집계하지 않는다.
- 후속 수정: 새 판은 두 이름 계약을 학습 전에 함께 검증하고, 실패 메시지·traceback을 원문으로 보존한다.
  기존 실패는 유지하며 새 seed에서 같은 학습·평가 기준을 사용한다.
- 근거: [실패 결과](../evals/results/vision_source_prefix_dev_20261026w/summary.json) ·
  [worker job 계약](../bench/brev_recovery_worker_v2.py) ·
  [학습 매니페스트 계약](../bench/residual_training_v1.py).

### OPS-15: 소스 앵커와 프로토콜 체크아웃의 구별

- 현상: 소스와 프로토콜을 별도 커밋으로 고정한 뒤 등록 단계가 완료되지 않았다. 등록 스크립트는 소스 앵커와 HEAD의 동등성을 요구했다.
- 수정: 실행 프로토콜이 있는 체크아웃, 소스 앵커의 조상 관계, 결합된 소스 blob 일치를 구분해 검사한다. 새 등록 이후 학습 네 건과 고정 평가 두 건이 완료됐다.
- 확인 범위: [source 결과](../evals/results/vision_source_prefix_dev_20261026x/summary.json). 앞선 등록 실패는 학습·모델 판단 실패로 집계하지 않는다.

### OPS-16: 긴 실행과 조회 기록의 atomic intent 혼동

- 현상: 학습·평가가 끝났고 native 명령도 종료 코드 0을 반환했지만, 호스트 완료 검증이 atomic intent의 유일성 검사에서 실패했다.
- 원인: 실행 중 생성된 별도 상태 조회 intent까지 같은 실행의 후보로 셌다. 완료 명령의 snapshot과 일치하는 intent는 한 건이었다.
- 복구: 그 한 건으로 동결한 packet·script·stdout·결과 envelope 검증을 다시 수행했고, 원문 437파일의 해시를 확인했다. 원격 action 재실행과 이전 unknown 기록 변경은 없었다.
- 후속 경계: 실행과 상태 조회의 호스트 기록 폴더를 분리한다. 실행 성공, 호스트 전달 검증, 회수 완료를 별도 상태로 남긴다.

### CAP-16: 정책 검사는 통과했으나 캡처 arm에 callback 두 건이 들어옴

- 관찰: 새 source 정책의 checkpoint 로딩, 환경 관측 `1×48`, 정책 행동 `1×12`, 환경 1 step은 통과했다.
  준비 callback 뒤 첫 캡처 arm에 유효 callback 두 건이 들어와 exact-one 검사가 실패했다.
  child 종료 코드가 0이어도 저장된 최초 오류와 실패 envelope를 근거로 성공을 거부했다.
- 보존: cleanup은 통과했고 GPU compute process는 남지 않았다. 나머지 정책 세 개와 전체 녹화·모델 호출은 시작하지 않았다.
  원문 475파일을 회수하고 해시를 확인했다. 두 callback의 RGB 바이트 동일성과 캡처 전 정책 step의 인과는 미확정이다.
- 새 구현: 같은 환경에서 checkpoint를 로딩한 뒤 엄격 A-A-B-A 캡처, Writer detach, 정책 1 step, 환경 종료 순서로 실행한다.
  중복 callback을 버리거나 허용하지 않으며, 새 v2 schema로 기존 실행과 구분한다.
  CPU 회귀를 통과한 뒤 새 사전등록으로 실행했지만, 정책 step에 도달하기 전 같은 arm 중복이 재현됐다.
  따라서 캡처 전 정책 step만 옮기면 해결된다는 가설은 지지되지 않는다.
- 다음 분해: callback 발생 경계를 기록하면서 custom-event Writer와 수동 Writer 예약을 비교한다.
  `trigger=None`과 `schedule_write()`는 설치 버전의 [공식 API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.11.35/source/extensions/omni.replicator.core/docs/API.html)가 지원한다.
  두 callback의 발생 경로나 SDK 결함 여부는 이 문서만으로 확정하지 않는다.
- 근거: [실패 요약](../evals/results/vision_checkpoint_smoke_dev_20261026y/summary.json),
  [새 smoke 실행기](../bench/run_vision_checkpoint_smoke_v2.py),
  [새 입력 검증기](../evals/vision_record_source_readiness_v2.py),
  [순서 변경 뒤 실제 결과](../evals/results/vision_checkpoint_smoke_dev_20261026z/summary.json).

### OPS-17: 준비 검사의 source 필드와 Python 경로 불일치

- 관찰: 새 준비 스크립트가 source 프로토콜에 없는 `source_commit` 필드를 검사해 GPU 실행 전에 멈췄다.
  후속 조정 스크립트는 시스템 Python에서 필요한 패키지를 찾지 못했다.
- 수정: source 코드 앵커와 프로토콜 커밋을 구분하고, 설치된 환경의 Python으로 검증했다.
  이미 복사한 원문은 다시 쓰지 않고 해시를 확인했으며 누락된 결합 기록만 생성했다.
- 확인 범위: 실제 등록 복구가 통과한 뒤 checkpoint 캡처까지 도달했다. 이후 실패는 위 CAP-16의 callback 문제이며 등록 실패와 구분한다.
  기존 실패 기록은 보존하고, 해소 영수증이 일치하는 실패 한 건에만 후속 실행을 허용한다.

### CAP-17: 허용된 준비 callback과 진단 fingerprint의 입력 범위

- 관찰: callback 경계 진단은 custom-event 방식에서 준비 callback 두 건과 첫 arm callback 두 건을 기록했다.
  첫 준비 callback은 기존 mux가 허용하는 `bootstrap_empty`였으나, 진단 fingerprint는 유효한 semantic mask만 받아 계측 오류를 남겼다.
  그 결과 `diagnostic_complete=false`가 되어 수동 예약 방식은 시작하지 않았다.
- 확인: 첫 arm의 두 callback은 frame 59이며 두 번째 `orchestrator.step()` 내부에서 발생했다.
  원문 전체 SHA는 서로 다르다. 이 판은 RGB 등 성분별 SHA를 저장하지 않아 차이가 난 성분은 확인하지 못했다.
- 후속 수정: 원문 fingerprint와 semantic 유효성 검사를 분리하고, 준비 빈 mask·유효 준비·같은 arm의 중복이라는 실제 입력 순서를 회귀에 넣는다.
  기존 mux의 exact-one 기준과 실패 결과는 유지한다.
- 실제 후속 결과: 수정판은 custom-event와 수동 예약 모두 `diagnostic_complete=true`를 남겼다.
  준비 빈 callback의 계측 오류는 해소됐으나 두 방식 모두 첫 arm에 유효 callback 두 건이 들어와 캡처는 거부됐다.
  첫 callback의 semantic mask·labels는 직전 prepare와 같은 SHA였고 RGB·시각은 현재 arm의 값이었다.
  두 번째 callback은 RGB·시각·카메라 등 나머지 성분이 같고 semantic 세 성분의 SHA만 바뀌었다.
  Writer 예약 방식만으로는 해결되지 않았으며, 마커 변경 뒤 의미 분할 출력의 갱신 경로로 다음 검사를 좁혔다.
  SDK 내부 노드의 인과는 아직 미확정이다. [두 방식의 실제 결과](../evals/results/vision_callback_boundary_dev_20261026ab/summary.json).
- 보존: 원문 427파일을 회수했으며 종료 자료를 포함한 사본 443파일의 SHA를 확인했다.
  소유 프로세스와 GPU가 비어 있음을 두 번 확인한 뒤 VM을 정지했다. 새 학습·모델 호출·전체 녹화는 0건이다.
  [진단 결과](../evals/results/vision_callback_boundary_dev_20261026aa/summary.json).

### OPS-18: 회수 identity와 자원 binding의 필드 구분

- 관찰: 강한 회수 검증 뒤 종료 승인 검사가 VM ID를 찾지 못했다.
- 원인과 수정: 회수 검증기의 identity 반환값은 protocol·proof·lease·run의 네 필드이며, VM ID와 이름은 검증된 transfer binding에 있다.
  새 종료 승인기는 두 자료를 각각 확인하고 네 공통 필드를 대조한다. 기존 원문·Start·GPU·비용 검증은 유지한다.
- 확인: 해당 실행에서 종료 승인과 소유 VM의 `STOPPED` 확인까지 완료했다. 이전 실행기는 보존했다.

## 새 실패를 남길 때

실행별 내부 기록은 다음 항목을 포함한다. 공개 이슈나 PR에는 재현에 필요한 코드·환경·오류만 옮긴다.
원래 결과는 보존하고 새 실행은 새 tag로 남긴다.

1. 실행 tag, source commit, protocol 해시와 실제 runtime 버전.
2. 실패 단계와 최초 예외. SDK 종료나 누락된 결과 파일에서 생긴 후속 예외는 별도 표시.
3. 실행 명령·stdout·stderr·결과·원장 원문 경로와 해시.
4. 확인된 원인, 경쟁 가설, 아직 읽지 못한 원문.
5. 시도한 변경과 CPU/GPU 결과. 실패한 시도도 보존.
6. 최소 재현 입력과 regression test. fixture가 SDK 계약을 반영하는지 확인.
7. 학습·API 소비, 소유 프로세스 정리, 산출물 회수·해시 검증, VM 종료와 비용 범위.
8. 다음 한 가지 진단, 통과 기준과 멈춤 조건. 상위 SDK 보고 후보인지 프로젝트 수정인지 구분.

기록의 `CPU 통과`, `실제 후속 단계 도달`, `전체 실행 완료` 상태를 구분한다.
모델의 무효 답, 적용 불가, source 실패 부재와 `HOLD`를 모두 같은 실행 오류로 합치지 않는다.


### CAP-18: 의미 분할 사전 갱신 뒤에도 동일 callback 두 건

- 관찰: seed 2220 F1 checkpoint에서 marker 변경 뒤 Writer를 예약하지 않은 delta-zero render를 한 번 수행했다. 이 구간은 callback 증가 0, physics SHA·simulation time·marker 상태 불변을 통과했다.
- 실제 캡처: 이어진 arm0에 frame 60의 유효 callback 두 건이 들어왔다. 두 건의 전체 payload와 14개 성분 SHA가 모두 같았지만 exact-one 조건은 실패했다. callback은 버리지 않았고 입력 승인·모델 사용은 하지 않았다.
- 확인 범위: 앞선 AB의 서로 다른 semantic mask 현상과 이번 동일 payload 중복은 구분한다. 이번 한 checkpoint의 결과만으로 사전 렌더의 일반적 효과나 SDK 내부 원인을 확정하지 않는다.
- 실행: 31.112초, 새 학습·모델 API·전체 녹화 0. 자식 exit 0은 성공으로 받아들이지 않았고, 부모 cleanup과 GPU process 부재는 확인했다. SDK environment close는 실패 기록을 유지했다.
- 다음 조치: marker·renderer 동기화 기준을 유지하며 Writer dispatch의 중복 경로와 단일 callback을 보장하는 수집 방법을 분리한다. 같은 후보를 반복하지 않는다.
- 근거: [실제 결과](../evals/results/vision_semantic_settle_dev_20261026ad/summary.json).


### CAP-19: 기본 Writer와 step 단독 경로에서 준비 callback 0건

- 관찰: 같은 seed 2220 F1 checkpoint에서 Writer를 기본 인자로 attach하고 event·schedule 실호출을 제거했다. prepare의 renderer step은 한 번이었으나 callback이 한 건도 없어 prepare가 실패했다. arm과 정책 추론 확인은 실행되지 않았다.
- 확인 범위: 앞선 manual 경로의 중복과 다른 결과다. callback 0을 중복 해소나 캡처 성공으로 해석하지 않는다. `captureOnPlay=False`는 readback했고 실제 dispatch 수는 결과에 남겼다.
- 실행: 31.120초, 새 학습·모델 API·전체 녹화 0, 부모 cleanup 확인, 종료 뒤 GPU process 없음. 자식 exit 0은 성공으로 받아들이지 않았다.
- 다음 조치: 설치된 동일 버전 SDK 원문에서 기본 trigger의 활성화 조건을 확인한 뒤 다음 후보를 정한다. 같은 후보를 반복하지 않는다.
- 근거: [실제 결과](../evals/results/vision_frame_trigger_dev_20261026ae/summary.json).

### OPS-29: 준비 렌더의 카메라 설정 계약

- 준비 프레임을 분리한 새 진단은 렌더 함수에 전체 Play 설정을 넘겨 `rt_subframes` 조회에서 실패했다. 해당 함수는 `profile.camera`를 입력으로 받는다.
- 실제 렌더 step과 callback은 모두 0건이므로 Writer 가설은 평가되지 않았다. 실패 원문과 종료 증거를 보존했으며, 다음 판은 중첩 카메라 설정을 소비하는 실제 렌더 함수로 회귀를 확인한다.
- 근거: [실행 결과](../evals/results/vision_frame_primed_dev_20261027af/summary.json).

### OPS-31: 클라우드 기동 상태와 실행 준비 상태

- `RUNNING` 조회 뒤에도 건강 상태가 `UNAVAILABLE`이거나 다음 조회가 `STARTING`일 수 있었다. 상태 문자열 하나만으로 원격 실행을 시작하면 초기화 경계에서 중단된다.
- 실행기는 `RUNNING`, build `COMPLETED`, shell `READY`, health `HEALTHY`를 간격을 둔 두 조회에서 확인하고 단계 입장에서도 다시 확인한다. 비동기 Stop도 접수와 완료를 구분한다.
- 수정판은 이 경계를 넘어 초기화·등록·준비 검사와 실제 렌더링까지 도달했다. 기동 중단 두 건은 렌더링 결과와 구분해 보존하고 비용에 포함했다.

### OPS-33: 두 입장 검사의 배타 기록 경로 충돌

- 실행 입장과 후보 입장은 같은 Start의 비용을 각각 확인한다. 두 검사가 동일 파일을 배타 생성하면 두 번째 검사에서 `FileExistsError`가 난다. 해당 실행은 후보 렌더링 전에 중단됐다.
- 기록을 `lifecycle_admission`과 `candidate_admission` 단계별 경로로 분리했다. 같은 Start에서 실제 예산 함수를 두 번 호출하는 통합 회귀와 상한 초과 거부를 확인했으며, 실제 후속 실행도 두 입장을 통과했다. 기존 기록은 덮어쓰지 않는다.
- 근거: [실제 렌더 결과와 선행 중단 요약](../evals/results/vision_frame_primed_dev_20261027ag/summary.json).

### CAP-20: 기본 Writer의 baseline 뒤에도 동일 callback 두 건

- 관찰: 기본 Writer를 attach한 뒤 baseline 렌더 한 번을 분리했다. 이 구간은 callback 0, physics SHA와 PhysX 시각 불변을 통과했다. 다음 prepare 렌더 한 번에는 frame 59, reference time `[59, 30]`인 유효 callback 두 건이 들어왔다.
- 확인: 두 payload 전체와 14개 성분 SHA가 모두 같았다. event·schedule 실호출은 0이었고 baseline 포함 renderer step은 총 두 번이었다. 콜백은 삭제하거나 병합하지 않았으며 exact-one 조건은 실패로 보존했다.
- 실행: 32.123초, AABA arm·정책 추론 확인 미도달, 새 학습·모델 API·전체 녹화 0. 부모 cleanup과 GPU process 부재를 확인하고 소유 VM의 정지를 완료했다. SDK 환경 close의 후속 오류도 원문에 남아 있다.
- 판단: 준비 프레임을 분리하는 것만으로 단일 전달을 확보하지 못했다. SDK 내부 원인은 미확정이다. Writer trigger 변형을 더 반복하지 않고, annotator 출력과 frame·시각·카메라·마커·물리 상태를 보존하는 별도 수집 경계를 검증한다. 새 방식의 성공은 아직 확인되지 않았다.
- 근거: [실제 결과](../evals/results/vision_frame_primed_dev_20261027ag/summary.json).

### OPS-36: 새 수집 단계와 기존 실행기 명령의 연결

- Writer 없는 수집 진단은 초기화·등록·준비를 마쳤으나, 새 `annotator_pull` 명령이 기존 실행기의 등록 목록에 없어 후보 시작 전에 거부됐다. GPU 자식·렌더·학습·모델 호출은 0건이다.
- 새 어댑터는 수집 단계만 기존 `record` 경로에 연결하고 예산 검사의 후보 이름도 함께 변환한다. 기존 명령 목록은 변경하지 않는다. 회귀는 어댑터를 가짜 성공 함수로 대체하지 않고 기존 실행기를 거쳐 `record`까지 도달하는지 확인한다.
- 원본 자료 452파일과 종료 기록을 회수했고 소유 VM의 정지를 확인했다. 이 결과로 수집 방식의 성공·실패를 판정하지 않는다. [실행 결과](../evals/results/vision_annotator_pull_dev_20261027ah_finalized/summary.json).

### CAP-21: 명시 호출 횟수와 실제 Kit 갱신 횟수의 구분

- 코드 확인에서 수집기의 갱신 카운터가 명시한 `app.update()` 호출만 셌다. 이 값은 렌더 함수 내부에서 일어나는 Kit 갱신 횟수를 나타내지 않는다.
- 새 판은 native IApp의 `get_update_number()`를 사용한다. API가 없거나 잘못된 값을 반환하면 거부하며, 렌더 구간의 갱신과 연속 annotator 읽기 구간의 갱신을 나눠 기록한다. 읽기 사이의 추가 갱신은 거부한다.
- CPU 회귀와 실제 수집 판정을 구분한다. 명시 호출 0을 실제 갱신 0으로 해석하거나 원자성을 주장하지 않는다.

- 실제 후속 결과: native 카운터를 쓰는 수집기는 F1 checkpoint에서 A-A-B-A 네 장면을 통과했다. frame은 57·58·59·60으로 증가했고 각 렌더 구간의 Kit 갱신은 1회, 연속 읽기 구간은 0회였다. 물리 상태·시각 불변, 마커 0·1·2·3, 카메라 전환, detach 뒤 정책 한 step과 자원 정리도 통과했다. 실행은 33.731초였으며 전체 정책 영상·모델 입력은 아직 승인하지 않았다. [실제 수집 진단](../evals/results/vision_annotator_pull_dev_20261027ai/summary.json).


### OPS-38: 설정 완료 응답의 스키마 불일치

- 첫 정책 녹화의 환경 설정은 source 359개·이전 정지 기록 22개를 확인하고 `complete`를 반환했다. 그러나 클라이언트가 응답 스키마에 남은 다른 이름을 기대해 실행 전 검사 단계로 넘어가지 못했다.
- GPU 수집·새 학습·모델 API·전체 녹화는 모두 0건이다. 수집기의 프레임 판정 결과와 구분하며 VM 정지를 확인했다. 원문 메타데이터 202개를 보존하고 사본 해시를 대조했다.
- 다음 경계는 실제 설정 응답을 입력으로 쓰는 `prepare()` 통합 회귀다. 기존 판정 기준은 유지하고, 인프라 재시도에는 이 시작의 비용 추정 $0.058680을 원래 batch·전체 상한에 합산한다.
- 근거: [실행·정지·비용 요약](../evals/results/vision_pull_record_dev_20261027aj_infra/summary.json).


### OPS-39·40: 준비 모드 검사와 실패 회수 명령의 연결

- 수정된 설정 응답 검증과 dry-run·bootstrap·등록을 통과한 뒤, 준비 스크립트가 등록 모드 `record`와 다른 이전 모드를 요구해 실패했다. 생성 코드를 컴파일하는 것만으로는 이 분기 불일치를 확인하지 못했다. 후속 receiver가 반환한 `stage_executed=false`는 첫 실행 부재가 아니라 intent만 남은 실패의 재실행 거부다.
- 이어 제어기가 광고한 `snapshot`·`failure-snapshot`이 6개 실행 stage 목록으로 전달돼 회수가 중단됐다. 등록된 실패 snapshot 경로로 복구해 원문 468개를 회수했고, 사본 472개와 최종 host 메타데이터 311개의 해시를 대조한 뒤 소유 VM 정지를 완료했다.
- 정책 녹화·새 학습·모델 호출은 0건이다. 다음 준비 검사에서는 생성한 모드 검사 코드를 실행하며, 전체 제어 명령의 소비 경로와 누적 인프라 비용을 함께 확인한다. 기존 실행·원문·판정 기준은 보존한다.
- 근거: [시도와 회수·정지 요약](../evals/results/vision_pull_record_dev_20261027aj_retry1/summary.json).

### CAP-25: 진단 산출물 저장 실패와 원오류 보존

- 설정·등록·준비 검사를 통과하고 정상 checkpoint의 환경·정책 로드와 annotator readiness까지 도달했으나, AABA 보고서 저장에서 `ndarray` JSON 직렬화 오류가 보존됐다. record 자식과 두 번째 checkpoint는 시작하지 않았다. 오류 뒤 종료 지연으로 약 400초의 부모 timeout도 발생했다.
- 코드와 CPU 재현에서 AABA 출력 폴더가 없으면 NPZ 저장이 실패하고, collection에 남은 배열이 다음 JSON 저장 오류를 일으켜 앞선 오류를 가릴 수 있음을 확인했다. 실제 원문에는 직렬화 오류만 남아 내부 최초 오류 전체를 복원할 수는 없다. 최초 resource freshness는 통과했지만 이후 record 단계의 freshness 재검증은 미도달 경로다.
- 원문 502개, 회수 사본 506개와 최종 host 메타데이터 331개의 해시를 대조했고 VM 정지를 확인했다. 새 학습·모델 호출·전체 정책 영상은 0건이다. 새 실행판은 진단·영상 출력 경로를 분리하고, 원시 NPZ와 최초 오류를 앱 종료 전에 보존하는 경계를 검증한다.
- 근거: [실행·회수·정지 결과](../evals/results/vision_pull_record_dev_20261027aj_retry2/summary.json).

### CAP-26: 진단 보고서와 앱 종료 래퍼의 terminal 계약

- 후속 실행은 AABA 네 장면의 NPZ·영수증·JSON 보고서를 저장했고, 회수 후 보고서와 배열 metadata·해시 대조도 통과했다. CAP-25의 저장 경계 수정은 실제 실행에서 확인됐다. 그러나 자식 종료가 완료되지 않아 약 410.74초 뒤 부모 정리로 끝났고 record 앱은 시작하지 않았다.
- RL 환경 종료와 앱 프로세스 종료는 별개다. 실제 factory의 `AppProxy`는 close 전에 `result.json` 또는 `failure.json` 정확히 하나를 요구하지만, AABA 성공 출력에는 진단 보고서만 있었다. 이 경로는 종료 정책을 적용할 수 없으며, SDK close가 반환해도 보류된 정책 오류 때문에 exit 0이 될 수 없다.
- `wait_for_replicator=False`는 이미 적용돼 있었다. SDK close 내부의 긴 대기 위치는 stack 근거가 없어 미확정이며, 이를 NVIDIA SDK 결함으로 단정하지 않는다. 새 경계는 AABA 원문을 검증하는 종료 정책과 phase별 영수증을 사용하고, 보고서·종료 정책·child exit·정리 증거를 나눠 검증한다.
- SDK close의 Python 반환을 정상 종료의 필수 조건으로 삼지 않는다. fast shutdown에서는 프로세스가 먼저 끝날 수 있으므로, 종료 진입 증거와 부모가 관찰한 자연 exit 0·자원 정리를 필수로 검증하고 Python 반환은 관측됐을 때만 기록한다. [NVIDIA의 4.5 종료 동작 답변](https://forums.developer.nvidia.com/t/how-to-keep-a-standalone-script-running-after-calling-simulationapp-close/331754).
- 원문 522개, 회수 사본 526개와 host 메타데이터 334개를 대조했고 소유 VM은 정지했다. 새 학습·모델 API·전체 영상은 0건이다. [실제 결과](../evals/results/vision_pull_record_dev_20261027ak/summary.json) · [종료 계약 진단](../evals/results/vision_pull_record_dev_20261027ak/termination_contract_diagnosis.json).

### CAP-27: 앱 부팅 전에 가져온 SDK 모듈

- 후속 실행은 정상 checkpoint의 AABA 자식이 시작된 지 약 0.43초 만에 `ModuleNotFoundError: No module named 'carb'`로 종료됐다. 최초 traceback과 exit 1, 정리 완료 기록이 남아 있다. 본 녹화와 두 번째 checkpoint는 시작하지 않았다.
- 전용 factory가 `AppLauncher` 생성 전에 `carb.settings`를 import한 것이 원인이다. 앱 부팅이 제공하는 모듈을 먼저 불러 생긴 실행 코드의 순서 오류이며, SDK의 영상 수집이나 종료 정책을 평가한 결과는 아니다.
- 단독 fake app 회귀로는 부팅 순서를 확인하지 못했다. 새 회귀는 `carb`가 없는 별도 Python 프로세스에서 factory를 실행하고, 앱 생성 뒤에만 해당 모듈을 사용할 수 있게 한다. 기존 실행과 원문은 보존한다.
- 원문 525개, 회수 사본 529개와 host 메타데이터 339개의 SHA를 대조했고 소유 VM의 정지를 확인했다. 새 학습·모델 API·전체 영상은 0건이다. [실제 결과](../evals/results/vision_pull_record_dev_20261027al/summary.json).

### CAP-28: 부모와 Play worker의 출력 폴더 생성 충돌

- 다음 실행은 앱 부팅 뒤 AABA 네 장면과 NPZ·원시 영수증 각 네 개를 저장했다. 종료 정책·close 진입·부모가 관찰한 자연 exit 0·정리·GPU 관찰의 결합도 회수 후 검증을 통과했다. CAP-26과 CAP-27의 수정은 이 범위에서 실제 확인됐다. SDK가 Python으로 반환한 영수증은 없으며, 이는 fast shutdown 계약에서 허용하는 경로다.
- 이어 시작한 record 자식이 `FileExistsError`를 남겼다. 부모는 AABA 뒤 record output 폴더를 생성하고, 자식의 job 검증은 그 폴더가 있어야 통과하지만, Play worker는 같은 폴더를 `exist_ok=False`로 생성한다. 서로 다른 세 경계의 폴더 소유권 계약이 맞지 않았다.
- 새 판은 Play worker가 폴더를 생성하도록 맞춘다. 부모는 기존 output을 거부하고, 유효한 AABA continuation을 검증한 자식만 생성 경계에 도달한다. 이미 존재하는 결과·symlink를 지우거나 재사용해 통과시키지 않는다. CPU 통합 회귀와 실제 625프레임 완료는 구분한다.
- 원문 556개, 회수 사본 560개와 host 메타데이터 341개의 SHA를 대조했고 소유 VM은 정지했다. 새 학습·모델 API·완료 영상은 0건이다. [실제 결과](../evals/results/vision_pull_record_dev_20261027am/summary.json).

### OPS-42: 이전 수집의 자원 기한과 후속 촬영

- 후속 촬영이 자원 등록에서 중단됐다. 이전 소스 수집의 자원 기한은 06:00 UTC였고 새 인스턴스 시작은 06:00:57 UTC였다. 원격 오류 기록에서 기한 부족으로 자원 증명 발급이 거절된 것을 확인했다. 녹화·새 학습·모델 API 호출은 0건이다.
- 자동 정리 호출의 이름도 허용 목록과 달라 최초 정지가 막혔다. GPU와 소유 자식 프로세스가 없음을 확인한 뒤 허용된 정지 호출을 사용했고, 읽기 복구를 포함한 모든 가동의 정지를 확인했다. 기한 오류와 정리 호출 오류를 구분해 기록한다.
- 후속 판은 기존 소스 데이터의 출처와 새 촬영의 자원 유효기간을 별도로 결합한다. 미확정 실행에서도 소유 작업이 남아 있으면 중지하지 않는 정리 경계를 CPU 회귀로 확인했으며, 실제 후속 실행은 별도로 판정한다. [실행 결과](../evals/results/vision_pull_record_dev_20261027an/summary.json).

### OPS-44: 정지된 인스턴스에 대한 정리용 원격 호출

- 후속 실행은 기동 요청 뒤 `STARTING`에서 `STOPPED`로 바뀌었고, 정해진 횟수 안에 준비 조건을 충족하지 못했다. 영상 수집·새 학습·모델 호출은 시작하지 않았다. 최초 기동이 준비되지 않은 이유는 확인되지 않았다.
- 정리 함수는 소유 작업의 부재를 확인하려고 `brev exec`을 사용했다. 그 호출의 출력은 인스턴스를 시작한다고 표시했고, 이후 원격 유휴 상태를 확인한 뒤 정지했다. 이 호출은 정지 상태를 확인하는 용도로 사용할 때도 기동 부수효과가 생길 수 있다.
- 원격 명령 전에 대상과 최신 상태를 확인하고, 이미 정지된 경우 원격 호출 없이 그 증거를 남기는 경계를 추가한다. [실행 결과](../evals/results/vision_pull_record_dev_20261027ao/summary.json).
- 후속 실행도 준비 상태에 도달하지 못했지만, 수정된 정리 경계는 `STOPPED`를 두 차례 확인하고 원격 명령·추가 정지 요청 없이 끝났다. `brev` 0.6.335의 [공개 코드](https://github.com/brevdev/brev-cli/blob/v0.6.335/pkg/cmd/exec/exec.go#L184-L211)도 SSH 실패 뒤 상태가 `STOPPED`이면 기동 함수를 호출한다. 최초 기동이 실패한 공급자 측 이유는 이 코드만으로 알 수 없다. [후속 실행 결과](../evals/results/vision_pull_record_dev_20261027ao_retry1/summary.json).

## OPS-45: 생성 직전 catalog에서 선택 GPU type이 사라짐

- 보존한 견적에는 `massedcompute_L40S`가 있었지만, 생성 직전 같은 조회의 결과에는 없었다. 생성 명령 전에 중단했으며 새 인스턴스·학습·영상 수집·모델 호출은 없었다. catalog에서 빠진 공급자 측 이유는 확인하지 못했다.
- 다른 공급자의 L40S·200 GB 디스크·새 견적을 별도 판으로 고정한다. 생성 전 최신 catalog 검사와 정확한 소유 ID 확인은 유지한다. [중단 기록](../evals/results/vision_environment_transfer_20261027a/summary.json).

## OPS-48: 정규화한 시간 값을 검사한 뒤 원문을 소비자에 반환

- 새 L40S에서 설치 버전과 의존성 검증이 통과했고, AABA 입장 검사를 마친 뒤 첫 정책 sample을 수집했다. 수집 원문의 `IsaacReadTimes`에는 시각 11필드와 실행 포트 `execOut`이 있었다.
- producer는 기존 정규화 함수를 호출했지만 반환값에는 12필드 원문을 넣었다. 시각 11필드를 요구하는 소비자가 첫 sample을 거부했다. 원문에는 오류를 가릴 필드와 값이 남았고, GPU 자식 정리도 확인됐다. 완성 영상과 모델 입력은 생성되지 않았으며 결함 정책은 실행하지 않았다.
- 원문·NPZ·해시는 유지하고, 소비자와 manifest 비교에는 같은 정규화 값을 쓰도록 새 판을 준비한다. 알 수 없는 필드와 시각 불일치의 거부 기준은 유지한다. [설치 검증](../evals/results/vision_environment_transfer_20261027b/summary.json), [첫 촬영 실패](../evals/results/vision_transferred_pull_record_dev_20261027b_v1/summary.json).

### OPS-49: 녹화 함수의 5키 반환값과 worker의 8키 artifact 계약

- 다음 실행은 정상 checkpoint의 AABA와 625프레임 수집을 마치고 영상·trace·두 measurement·두 spec을 생성했다. `build_trace_and_v01_specs`는 이 5개 파일의 mapping과 primary video를 별도 값으로 반환하지만, worker는 8키 artifact mapping을 가정해 후처리에서 실패했다. 영상 경로 변경 문제로 해석하지 않는다.
- v3 adapter는 5키와 별도 영상을 output 경계의 단일 regular file로 검사하고, trace에 내장된 encoder·timing receipt를 write-once 파일로 봉인한다. 영상 SHA·trace·PTS·receipt 결합과 기존 8키 검증을 통과한 뒤에만 manifest 단계로 넘기며, 기존 파일과 symlink는 거부한다.
- 실제 625프레임 형태의 CPU 회귀와 원문 사본의 사후 검증은 통과했지만 원래 v2 실행 상태는 실패로 유지한다. 새 v3 두 정책의 실제 성공 여부는 별도 실행 결과로 판정한다. 결함 정책은 실행하지 않았고 새 학습·모델 API 호출은 0건이다. [v2 실행 결과](../evals/results/vision_transferred_pull_record_dev_20261027b_v2/summary.json).

### OPS-50: 재기동한 VM의 GPU 신원과 작업 소유권

- 같은 공급자 VM을 Stop/Start한 뒤 GPU UUID가 바뀌었다. 과거 boot의 UUID를 요구하던 환경 진입 검사는 새 GPU 작업 전에 중단했다. 별도 검사에서는 OS의 Python 기반 서비스도 학습 작업으로 분류했다.
- 공급자 VM ID·이름·type·새 견적과 설치 버전을 먼저 확인하고, 해당 boot의 실제 L40S UUID와 유휴 증거를 새로 결합했다. 프로세스 소유권은 등록된 작업 경로로 검사한다. 같은 boot 안에서 실행 전후 GPU 신원을 확인하는 조건은 유지한다.
- 수정한 환경 진입 뒤 seed 2221 두 정책의 native 녹화를 완료했다. 환경 재설치는 하지 않았다. [데이터 경로와 완료 범위](VISION-BREV-DATA.md#전송-정책-네-개의-입력선택-비교).

### OPS-51: 증분 bundle은 불완전한 Git history를 복원하지 못함

- 확인 검사 실행 전 source snapshot이 과거 학습 commit을 읽지 못해 중단했다. 실행 기록은 `launch.session_id=null`, 자식 exit code 없음, `wall_s=0.00545`초였다. 이 시도에서 검사는 시작하지 않았다.
- 원격 저장소에는 최신 commit 일부만 있었고, 증분 bundle은 이미 있다고 가정한 과거 object를 채우지 않았다. [완전한 history를 사용하는 새 시도](../bench/protocols/vision_transferred_probe_history_retry1_20261027.json)를 등록했다. 실제 backend의 source snapshot을 승인 소비 전에 확인한 뒤 같은 두 pair를 실행했다.
- 두 pair 모두 NONE `normal`, F1 `abnormal`로 완료했다. 최초 중단·승인 소비 기록은 보존했다. [원문 해시와 검사 결과](../evals/results/vision_transferred_selection_dev_20261027_v1/probe-pairs.json).

### OPS-52: 공유 API 원장과 미시작 decision의 구분

- 첫 응답을 완료하면 공유 API manifest가 생기지만 다음 decision manifest는 아직 없다. 동결 inspector는 이 정상 미시작 상태를 부분 생성으로 거부했다. 최초 stack trace는 별도 파일로 보존되지 않았다.
- 활성 호출과 미응답 reservation이 없음을 확인한 뒤 등록된 빈 decision manifest만 준비했다. 기존 응답은 재사용했고, 이 준비 과정의 API 호출은 0건이었다. 이후 비교는 24/24 완료했다.
- [새 읽기 전용 inspector](../evals/vision_selection_inspection_v2.py)는 미시작 상태를 구분한다. 실제 부분 기록·소유 manifest 누락·변조·미완료 reservation은 계속 거부하며 조회 중 파일을 생성하지 않는다. [실제 caller 기반 회귀](../tests/test_vision_selection_inspection_v2.py)는 첫 terminal 뒤 다음 decision이 없는 경우와 미완료 요청의 재전송 차단을 확인한다. 완료한 비교에는 새 inspector를 소급 적용하지 않았다.

### OPS-53: Stop 요청 성공과 공급자 정지 완료는 다른 상태

- seed 2221 촬영·비교 원문을 회수하고 GPU 유휴 상태를 확인한 뒤 Stop 명령은 exit 0으로 끝났지만, 공급자는 약 8분 46초 동안 `STOPPING`을 반환했다. 첫 40회 관찰 한도 초과도 보존했다. 이 상태를 정지 완료로 기록하지 않았다.
- Stop을 다시 보내거나 VM에 접속하지 않고 목록 조회만 이어가 `STOPPED`를 확인했다. 내부 지연 원인은 확인되지 않았다. 다음 실행도 새 Start 요청과 두 번의 READY 관찰을 요구한다.
- 비용 상한 추정에는 학습·촬영 시간뿐 아니라 시작·설정·복사·CPU 대기·정지 확인까지의 전체 구간을 포함한다. [정지 관찰과 비용 범위](../evals/results/vision_transferred_pull_record_dev_20261027c_v1/compute-reconciliation.json).

### OPS-54: 전송 사본·relay 참조·요청 폴더의 사전 준비

- E2E v2의 첫 입력 승인은 native 원시 영수증의 hardlink를 거부했다. 전송 mirror 7,728개를 바이트 해시가 같은 독립 파일로 바꿨다. 이어 relay 설정을 저장소 상대경로로 준 host scope가 절대 참조를 요구하는 소비자에서 거부돼, 원문을 보존하고 새 scope를 만들었다.
- 첫 모델 선택은 spool root가 준비되지 않아 `SpoolIntegrityError`로 멈췄다. Journal 예약 1건과 relay allocation은 남았지만 spool 요청 파일과 provider 호출은 없었다. 이 예약을 응답으로 채우거나 다시 보내지 않았다. 새 학습도 시작하지 않았다.
- [새 실행판 v3](../bench/protocols/vision_transferred_e2e_20261027_v3.json)는 같은 두 입력·모델 인자·학습 상한을 유지하며 요청 폴더를 먼저 준비한다. [이전 시도와 비용 연결](../bench/protocols/vision_transferred_e2e_20261027_v3.infra.json)에 지연 사유와 새 종료 시각을 등록했다. 비용 시작 시각과 $5 구간·$35 전체 상한은 그대로다.
- 이 준비 단계에서 Brev CLI는 원격 비영 종료 뒤 SSH 연결 확인과 명령 재실행을 보였다. 부수효과가 있는 host 준비 명령은 오류 원문과 실패 상태를 반환하는 외부 envelope로 감싸며, 이미 생성된 원장·결과를 덮어쓰지 않는다.

### OPS-55: 완료 학습과 저장소 내부 증거 경로의 연결

- 전송 E2E seed 2220의 CORRECTED 학습은 300회를 마치고 정리도 완료했지만, 다음 `training_evidence`가 Isaac Lab 로그의 외부 checkpoint 경로를 저장소 상대 참조로 바꾸지 못해 `HOLD`로 끝났다. 고정 평가는 실행하지 않았다. 원래 승인 소비와 unknown receipt, 완료 학습 원문을 보존한다.
- 이 checkpoint는 기존 seed 2220 NONE과 SHA가 같다. 별도 정책이나 처음 보는 결함의 새 증거로 세지 않는다. 새 학습을 반복하지 않고 원문 해시를 확인한 저장소 내부 사본과 평가 전용 재개 경로를 준비한다.
- [다음 학습의 경로 등록](../bench/protocols/vision_transferred_portable_lab_layout_20261027_v1.json)은 같은 Isaac Lab commit을 저장소 내부 실제 디렉터리에 복원한다. 인터프리터·학습 인자·소스는 유지하며 환경을 재설치하지 않는다. 이 기록만으로 회복을 판정하지 않는다.

### OPS-56: 완료 고정 평가의 arm 이름과 고리 검증기의 이름 대응

- seed 2221의 교정 학습과 고정 평가 자식은 정상 종료했다. backend는 `NONE`·`CORRECTED`라는 이름의 원문 report를 생성했지만, 고리 adapter는 mapping 키만 `none`·`corrected`로 바꾼 뒤 report 본문의 이름도 소문자라고 검증해 `HOLD`를 반환했다. 이 시도의 unknown receipt와 평가 원문은 보존한다.
- 재개 판은 원문의 arm 이름을 명시한 대응표로 검증하고, 26조건·checkpoint·프로토콜·수치 재집계·실행 및 원장 SHA 검증을 유지한다. 원본 report를 수정하거나 완료 학습을 반복하지 않는다. seed 2221은 이미 완료된 평가를 읽는 개발 측정이며, seed 2220의 누락된 평가는 별도 등록 후 한 번만 실행한다.

### OPS-57: 교정 녹화 adapter와 RUN5의 helper 계약

- 두 교정 정책의 고정 평가를 확인한 뒤 첫 native 녹화가 `AttributeError`로 중단됐다. 교정 JOB adapter가 RUN5의 `write_new`를 제공하지 않았다. 원문에서 자식 launch 파일 0개와 GPU 유휴 상태를 확인했으며, 이 시도는 완성 영상으로 세지 않는다.
- [새 adapter](../bench/run_vision_corrected_pull_record_v2.py)는 필요한 파일·해시 helper와 v2 자식 진입점을 제공한다. 기존 녹화 intent·stage·실패는 보존하고 새 판의 경로를 사용한다. 30초 자원 확인, 120초 ticket, AABA, 625프레임·50 fps와 원문·PTS 검증은 유지한다.
- [후속 단계 등록](../bench/protocols/vision_transferred_native_continuation_20261027_v4.json)은 녹화와 보조 관찰만 허용한다. 원래 Start부터의 비용 시계와 $5 구간·$35 전체 상한을 유지하며, 학습·고정 평가·초기 선택을 다시 실행하지 않는다.

### OPS-58: 녹화 권한의 기한과 단계별 재개

- 녹화 준비 중 기존 host scope가 만료돼 새 ticket 발급이 거절됐다. 완료한 학습·고정 평가와 실패한 handoff를 보존하고, 녹화만 허용하는 새 scope·등록 판으로 이어 갔다. 30초 자원 확인과 120초 ticket 기준은 유지했다.
- 두 교정 정책 모두 625프레임·50fps 녹화를 마쳤다. 마지막 녹화 시작 전에 등록한 [v6 종료 준비 예산](../bench/protocols/vision_transferred_native_stop_reserve_20261027_v6.json)은 같은 Start의 비용 시계를 유지하며 구간 상한을 $5.5로 정했다. 새 GPU 작업의 입장 기한은 늘리지 않았다.
- 원문 4,246개를 두 저장 위치에서 해시 대조한 뒤 Stop을 요청했다. 이후 원격 실행 없이 상태 조회만으로 `STOPPED`를 확인했다. 견적 비용은 구간 $5.198280, 프로그램 누적 $33.409058이며 청구서 확정액은 아니다. [종단 공개 결과](../evals/results/vision_transferred_e2e_dev_20261027_v1/summary.json).

### OPS-59: 모델 호출 원장과 공개 변환기의 해시 대상

- 보관 영상의 후속 관찰 두 건은 유효했으나, 공개 변환기가 `calls_sha256` 대조에서 중단됐다. 원장은 공급자 응답 배열을 줄바꿈을 포함한 정규 JSON으로 해시하고, 관찰 엔진은 별칭 검증 정보를 더한 호출 목록을 반환한다. 서로 다른 대상을 같은 해시로 검증한 것이 원인이었다.
- 변환기는 원장의 응답·예약·event 결합을 확인하고, 원시 응답과 엔진의 정규화 결과를 구분해 검증한다. 실제 원장 형식의 회귀를 추가하며 실패한 변환 경로는 보존한다. 모델 재호출과 실행 결과 변경은 하지 않는다.

### OPS-60: 다단계 행동 계약의 대상과 필수 근거

- 새 조사 계약의 첫 개발 호출은 정상·보상 결함의 이미 본 입력 3건에서 모두 HOLD로 끝났다. 학습 수치가 있는 두 입력은 허용 대상이 빈 `read_evidence`에 `target: null`을 반환했고, 설정만 있는 입력은 빈 `fact_ids`를 반환했다. [모델 응답·검증 결과](../evals/results/agentic_investigation_dev_20261028_v1/summary.json)를 보존한다. 확인된 실패는 모델의 대상·근거 계약 위반이며 GPU backend는 실행되지 않았다.
- API 3회 소비와 완료 응답 재생의 추가 0회를 확인했다. 의미 오류는 인프라 재시도로 처리하지 않았다. 다음 판은 빈 대상 목록의 행동을 선택할 수 없고, null은 HOLD에만 쓰며, 근거 ID는 현재 공개 사실에서 하나 이상 골라야 한다는 계약을 명확히 전달한다. 잘못된 출력을 실행기가 보정하지 않는다.
- [v2 개발 결과](../evals/results/agentic_investigation_dev_20261028_v2/summary.json)는 같은 입력에서 명시적 prompt와 판단별 JSON Schema를 함께 사용해 4응답 모두 유효, 세 입력 모두 검사 제안에 도달했다. 한 입력은 학습 기록 조회 후 다음 판단을 이어 갔다. 기존3+추가4=총7/12요청이며 프레임·확인 검사·교정은 실행하지 않았다. 계약 연결의 개선과 원인 진단·시각 기여의 개선은 구분한다.

### OPS-61: Brev 기동 접수·TLS 오류·정지 완료의 구분

- [모터 영상 수집 v1](../evals/results/agentic_motor_capture_20261028_v1/summary.json)은 두 Start 요청이 접수됐지만 Isaac 실행까지 도달하지 못했다. 한 자원은 기동 대기 상한까지 `STOPPED`였고, 다른 자원의 상태 조회는 control-plane hostname과 다른 인증서를 받아 TLS 검증에 실패했다. 인증서 검증을 끄지 않았다. 공급자의 기동 지연·인증서 배치 원인은 확인되지 않았다.
- Stop 요청은 접수됐으나 첫 정지 관찰 한도 안에는 `STOPPING`이었다. 이후 목록 조회로 두 자원의 `STOPPED`를 확인했고 Stop을 추가로 보내지 않았다. 학습·녹화·모델 호출은 0건이며 실패 기록과 원문 해시는 보존한다.
- [후속 수집 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v2.json)은 기존 VM 하나에서 두 job을 순차 실행한다. 상태 조회만 최대 3회 재시도하며 자원 신원 불일치는 즉시 거부한다. Start·실행·복사 요청을 자동 재전송하지 않고, 기동·회수·정지 관찰을 포함해 비용 상한을 계산한다.

### OPS-62: VM별 cache 경로와 원격 비영 종료의 분리

- [순차 수집 v2](../evals/results/agentic_motor_capture_20261028_v2/summary.json)는 SSH 접속 뒤 cache 존재 조건에서 중단됐다. 설정은 첫 VM의 repo 배치를 가정했지만, 다른 VM의 이전 성공 기록에는 별도 repo·Lab 위치가 명시돼 있었다. 당시 조건은 어느 경로가 없었는지 구분하지 않았으므로 현재 환경 소실 여부는 확인되지 않았다. Isaac·학습·녹화는 시작되지 않았다.
- 같은 traceback이 Brev CLI의 연결 재확인 전후에 두 번 기록됐다. [v3 등록](../bench/protocols/agentic_motor_capture_phase_20261028_v3.json)은 원격 본문이 비영 종료해도 외부 envelope에서 종료 코드를 기록하고, host가 단일 최종 marker를 확인한다. 원자적인 run·label 디렉터리가 같은 본문의 재실행을 막는다. 전송 exit 0만으로 본문 성공을 판단하지 않는다.
- v3는 VM별 HOME 상대 layout을 등록하고 실제 경로·존재 상태를 판정 전에 출력한다. lexical venv 경로와 source/runtime pin을 유지하며, dirty checkout·없는 cache는 보존하고 중단한다. 재설치나 자동 후보 탐색으로 측정 환경을 바꾸지 않는다. CPU에서는 Python 3.10과 실제 Bash 실행, WSL에서는 symlink를 포함한 97개 검사를 통과했다.

### OPS-63: Brev CLI의 성공 marker 뒤 자원 이름 출력

- [v3 boot](../evals/results/agentic_motor_capture_20261028_v3/summary.json)는 등록 cache·소스·Python 패키지 pin을 확인한 뒤 `AGENTIC_REMOTE_EXIT=0`을 남겼다. Brev CLI가 그 뒤에 자원 이름을 한 줄 붙였고, host는 marker가 마지막 줄이 아니라는 이유로 거부했다. Isaac·녹화·학습·모델 호출은 시작되지 않았다. 정지 완료와 원문 해시는 보존했다.
- [v4](../bench/protocols/agentic_motor_capture_phase_20261028_v4.json)는 등록된 자원 이름과 정확히 같은 마지막 한 줄만 선택적으로 제외한다. 다른 이름·임의 뒷줄·중복 이름·중복 marker·비영 종료는 계속 거부한다. 실제 v3 stdout을 이용한 회귀와 WSL의 103개 검사를 통과했다. 환경 재설치는 수행하지 않았다.

### OPS-64: 클라우드 cache alias와 OS 잠금 경로

- [v4 실행](../evals/results/agentic_motor_capture_20261028_v4/summary.json)은 boot·pin 확인과 네 checkpoint 복사를 완료했다. 상태 smoke가 자원별 잠금을 취득할 때 고정 cache 경로의 조상 symlink/reparse 검사가 거부했다. GPU 입장과 Isaac 실행 전이며 실제 링크 대상은 아직 확인하지 않았다.
- alias를 허용하려고 경로 검증을 끄거나 실행별 잠금 위치를 만들면 같은 GPU의 상호 배제가 깨질 수 있다. 새 adapter에서 고정 alias를 canonical 경로에 결합하고, 부모와 모든 자식이 같은 lock inode·owner start identity·heartbeat를 확인하도록 준비한다. CPU 회귀 뒤 새 source와 프로토콜로 실행하며 기존 실패 기록은 유지한다.
- 새 v2 어댑터는 고정 alias를 resolve한 경로에서 기존 잠금 코드를 사용하고, SDK·AABA 자식을 새 entry로 연결한다. 실제 WSL 회귀 25개에서 링크 재현·동일 inode 경쟁·자식의 부모 잠금 검증을 통과했다. 새 호스트는 입력 복사와 App 실행 전에 실제 OS 잠금만 취득·검증·해제하는 검사를 수행한다. 이 검사가 실패하면 후속 촬영으로 넘어가지 않는다.
- 원격 부분 원문을 회수하고 자원의 `STOPPED`를 확인했다. 이 시도의 경과 시간과 견적을 곱한 값은 약 $0.5592이며 공급자 청구액은 아니다. 새 영상·학습·모델 호출은 0건이다.

### OPS-65: 승인 JSON과 승인·소비 원장의 결합

- 새 단일 검사 실행기의 첫 CPU 판은 승인 JSON의 SHA·checkpoint·probe·인자를 검사했지만 승인·소비 원장을 다시 읽지 않았다. 구조가 맞는 수동 승인 파일이 통과할 수 있음을 코드 검수에서 확인했다. 이 판의 실제 GPU 실행은 없었다.
- 수정한 새 파일은 기존 controller의 승인 검증을 재사용해 `proposed → approved → consumed` 이벤트·request·arm/epoch·인자와 등록 prefix SHA를 확인한다. 완료 조회도 원래 승인과 running 또는 해당 완료 receipt의 done 상태를 다시 검사한다. 원장 누락·미승인·미소비·변조·다른 request의 거부를 포함한 관련 Python 3.10 회귀 98개를 통과했다.
- 기존 controller가 완료 reference를 요구하는 순서는 유지한다. 첫 대조 검사와 다른 OS의 승인 운반을 임의 파일로 채우지 않으며, 별도 pair 승인 연결 전에는 HOLD로 둔다. CPU 준비와 실제 검사 수용을 구분한다.

### OPS-66: 공급자 조회 시각과 현재 임대 검증 시각

- [v5](../evals/results/agentic_motor_capture_20261028_v5/summary.json)는 canonical 경로에서 두 실제 POSIX 잠금을 검증했다. 다음 입장 검사에서는 잠금 취득 전 공급자 조회 시각을 `verify_lease(now=...)`에 넘겼다. 현재 임대가 활성이어도 그 과거 시각은 취득보다 앞서므로 유효기간 검사에서 거부됐다. GPU query 원문은 남았으며 App과 촬영은 시작하지 않았다.
- 새 unit v3는 공급자 관찰의 30초 신선도를 별도로 검사하고 GPU query 뒤 현재시각에서 임대와 proof를 확인한다. 조회·query·임대 취득·입장 시각을 분리해 기록하고 기존 유효기간과 허용 오차는 유지한다. WSL 회귀 10개는 이전 실패의 재현과 미래·오래된 조회·query 중 노화·원문 변경·만료 임대·query 실패의 거부를 확인했다.
- 새 호스트의 생성된 원격 program도 실제 fixture 임대와 fake GPU query로 실행해 proof 소비까지 확인했다. 호스트 WSL 회귀 105개를 통과했으며 실제 클라우드 실행 결과는 별도로 판정한다.

### OPS-67: 상태 봉인 완료와 SDK 자식의 종료 코드

- [v6](../evals/results/agentic_motor_capture_20261028_v6/summary.json)는 전체 입장 검사 후 실제 상태 1프레임·manifest·`dry_run_result.completed=true`를 남겼다. 자식은 이후 `-11`(SIGSEGV)로 끝났으며 소유 process 정리는 확인됐다. 부모는 이를 상태 시험 성공으로 수용하지 않았다. 세션 close와 앱 close 중 충돌 위치는 단계 기록이 없어 미확정이다.
- 새 상태 child는 기존 녹화기와 같은 순서로 봉인된 terminal과 세션 정상 반환을 검증한 뒤 SDK fast shutdown을 적용한다. 세션·앱 close의 진입·반환·오류, 설정 전후와 근거 SHA를 별도로 기록한다. 부모는 종료 증거의 PID와 소유 자식 PID도 대조한다.
- 실제 child 생산기와 기존 부모를 합친 CPU 경로에서 성공과 봉인 후 `-11` 거부를 확인했다. 종료 정책 helper 누락 및 격리된 사본의 변조도 실행 전에 거부하며, 관련 WSL 회귀 39개를 통과했다. 이 CPU 통과를 SDK 내부 원인 규명이나 실제 충돌 해소로 쓰지 않는다.

### OPS-68: 입력 전송의 인증 응답과 원격 cache 재사용

- [v7](../evals/results/agentic_motor_capture_20261028_v7/summary.json)은 두 전체 입장 검사 뒤 입력 업로드에서 HTTP 401을 관측했다. 같은 실행의 목록 조회·부분 원문 회수·Stop은 성공했으므로 인증 실패의 지속 여부와 원인은 미확정이다. 인증 검증을 우회하거나 실패한 업로드를 재전송하지 않았다. App·상태 frame·전체 영상·학습·모델 API는 0건이다.
- 새 판은 이전에 올린 같은 입력 archive의 등록 HOME 상대 경로·일반 파일·단일 링크·SHA를 확인해 새 출력에 보존한다. 누락·변조·경로 이탈·조상 symlink와 복사 뒤 hardlink는 거부하고 원본은 갱신하지 않는다. 회수 경로와 보관 부모는 Start 전에 확인한다. 생성한 원격 program을 포함한 WSL 회귀 122개를 통과했다.
- 부분 원문 39파일의 사본 해시와 `STOPPED`를 확인했다. 경과 시간과 견적을 곱한 값은 약 $0.4951이며 공급자 청구액은 아니다. 이 실행에서는 이전 SDK 종료 수정의 효과를 측정하지 못했다.

### OPS-69: 임대 검사 래퍼가 가린 GPU 관찰 원문 참조

- [v8](../evals/results/agentic_motor_capture_20261028_v8/summary.json)의 상태 시험은 실제 봉인과 세션 종료 뒤 SDK fast shutdown으로 자연 exit 0을 반환했다. 소유 자식 PID와 정리도 확인돼 OPS-67의 새 종료 경로가 이 한 건에서 통과했다. 이후 AABA 자식도 exit 0으로 끝났지만 부모의 원문 검증이 `prepared GPU 관찰 원문 참조가 없다`로 중단됐다. 녹화 자식은 시작하지 않았다.
- 원인은 기존 `_leased_reader`가 `StrictRemoteGpuReader`를 함수 closure로 감싸 호출 기능만 남기고 `last_receipt` 속성을 가린 것이다. 새 runner의 callable proxy는 호출과 속성 접근 모두 임대를 먼저 확인하고, 현재 underlying 참조를 캐시·재조립 없이 전달한다. 기존 원문 SHA·자원·GPU 검증과 live 판정은 유지한다.
- 실제 reader의 파일·receipt 기록과 원래 `_gpu_reference`를 사용하는 CPU 경로에서 이전 실패를 재현하고 새 경로의 통과를 확인했다. 새 경계 WSL 회귀 23개와 작성자 관련 Python 3.10 회귀 175개가 통과했다. 부분 원문 122파일의 C/D 해시와 자원 정지를 확인했으며, 전체 영상과 시각 기여는 아직 판정하지 않는다.
- 후속 [v9](../evals/results/agentic_motor_capture_20261028_v9/summary.json)는 두 상태 시험과 정상·F2 네 정책의 native·sidecar·trace·playback 결합을 통과했다. 회수본 7,895파일을 다시 해시 대조해 네 정책을 기술 수용했다. 원래 실패 원문은 보존했으며 이 결과로 시각 정보의 판단 기여를 주장하지 않는다.

### OPS-70: 카메라 추적과 articulation 렌더 상태 갱신

- v9의 정상 로봇은 뒤쪽 검사 프레임에서 보이지 않았다. [후속 CPU 검사](../evals/results/agentic_motor_capture_20261028_v9/render_visibility_followup.json)는 CameraParams의 카메라 위치와 trace 위치가 일치하는 반면, 밝은 무채색 mask가 현재 root보다 초기 root 투영에 가까운 것을 확인했다. 네 영상의 모델 입력 사용은 보류했다. 파일·타이밍 결합 통과만으로 렌더 형상의 상태 반영을 보장할 수 없다.
- 확인된 코드 경로에서 capture는 Lab 센서 없이 render product와 orchestrator를 사용했다. Lab의 `forward()`는 articulation kinematics와 fabric 갱신을 수행한다. 해당 갱신 누락은 원인 후보이며 원격 파일 대조와 GPU 시험 전까지 확정하지 않는다. 카메라 profile과 물리 step은 유지하고, 매 capture 전 `forward()` 및 물리 snapshot 불변 검사를 추가한 새 판을 준비한다. 초기 영상과 결과는 덮어쓰지 않는다.
- [이전 영상 감사](../evals/results/vision_render_integrity_audit_20261028_v1/summary.json)는 정상 source 2220·2221과 교정 두 영상에서도 같은 렌더 불일치 정황을 확인했다. 수평 이동이 작은 F1 정책의 자세 반영은 이 위치 검사로 가르지 못했다. 관련 프레임·발표 10쪽 원본의 새 시각 근거 사용을 보류하고, 시뮬레이션 상태의 평가·회복 기록은 유지한다.
- [사실] 새 [v11 정상 한 정책 시험](../evals/results/agentic_motor_render_preflight_20261028_v11/summary.json)은 capture 전 `forward()`를 625회 수행했고 매번 물리 snapshot이 같았다. 고정한 화면 26개 중 초기·현재 위치가 충분히 분리된 23개 모두 현재 root 투영 30픽셀 안에 형상이 있었고, 초기 root 근처는 0개였다. native·상태·원문 사본 결합과 `STOPPED`도 확인했다. 같은 checkpoint의 이전 v9에서는 현재 root 근처가 0/23이었다.
- [추론] 이 관찰은 capture 시 articulation·fabric 갱신 누락 가설과 부합한다. 두 실행의 GPU가 달라 전체 궤적 동등성이나 SDK 내부의 다른 영향을 배제한 인과 A/B로 쓰지 않는다. 카메라·물리 step·화면 상한은 유지했고 기존 네 영상은 덮어쓰지 않았다. 다음 확인은 결함 정책의 같은 경로 재녹화다.

### OPS-71: 기존 VM의 Start 응답과 실제 준비 상태

- [v10 렌더 시험](../evals/results/agentic_motor_render_preflight_20261028_v10/summary.json)은 Start 응답을 받았지만 READY를 관측하지 못했다. `STARTING`·`UNHEALTHY` 뒤 호스트 Stop intent 전 `STOPPED`를 관측했고 등록된 기동 대기 상한에서 종료했다. 원격 boot·Isaac·영상·학습·모델 호출은 모두 0건이다. VM 내부나 공급자 원인은 미확정이다.
- 호스트는 `STOPPED`를 재확인해 Stop 요청을 추가로 보내지 않았다. 실제 경과와 compute 견적을 곱한 값은 약 $0.7533이며 청구액은 아니다. 같은 Start를 반복하지 않고 새 자원의 환경 복원과 다음 렌더 시험을 준비한다. 이전 결과·원문·소비는 보존한다.

### OPS-72: 검사 raw 디렉터리와 부모 metadata의 서로 다른 범위

- 실제 Parent 배치는 `unit/raw/`에 검사 원문을, `unit/`에 outcome·evidence·request를 쓴다. 첫 운반기의 E13 roots만으로는 부모 파일 여섯 경로를 찾을 수 없었다. CPU 등록 준비에서 이를 확인해 실제 실행 전에 차단했다.
- 새 운반기는 E13 roots를 넓히지 않고 별도 metadata layer와 두 map의 SHA를 고정한다. 등록 request는 양 arm의 참조를 포함하므로 raw SHA를 검증한 leaf로 보존한다. 이를 한 arm의 resolver에 재귀로 넣던 첫 CPU 실패도 남겼다. 실제 Parent 폴더 구조를 사용한 export→unpack→host metadata→E13 projection의 WSL 회귀 28개를 통과했다. 실제 클라우드 검사는 별도 결과로 판정한다.

### OPS-73: 생성기와 소비자의 명령 canonical hash 규약

- [실제 등록 최종검사](../evals/results/agentic_runtime_text_pair_registration_20261028_v1/summary.json)는 모델이 학습 기록을 읽고 `P_reward`를 고른 뒤 GPU 실행 전에 차단됐다. 원인 판별에서 자원·순서·seed·probe·소스 조건은 같았고, `commands_sha256`만 달랐다. 생성기는 `E.digest`를, 동결 Host는 `P.PAIR.digest`를 사용했다. 모델 두 판단은 유효했고 추가 API 호출과 GPU 검사는 0건이었다.
- 새 준비기는 기존 소비자의 명령 hash 함수를 사용한다. 생성한 pair 전체와 승인 원문을 동결 Host의 `_binding`으로 소비하는 CPU 회귀 2개가 통과했다. 기존 등록·소비 prefix와 원문은 보존하며 새 P 판과 tag로 같은 모델 선택을 재사용한다. hash 규약을 맞춘 CPU 통과와 실제 검사쌍 완료를 구분한다.

### OPS-74: 등록 결합 통과 뒤 Brev 재시작 거부

- [텍스트 고리의 실제 검사쌍](../evals/results/agentic_runtime_text_pair_start_20261028_v1/summary.json)은 수정된 Parent 등록과 lifecycle 생성자 검사를 통과했다. 그 뒤 기존 VM의 Start 요청이 `Bad Request`와 exit 1을 반환했고, 후속 목록에서 같은 자원이 `STOPPED`임을 확인했다. 원격 boot·검사·추가 API 호출은 시작하지 않았다.
- 응답에는 상세 공급자 사유가 없으므로 용량 부족이나 모델 실패로 단정하지 않는다. Start를 반복하지 않고 원문·기존 모델 응답·승인 소비를 보존한다. 다른 자원에서 이어갈 때는 실제 provider·인스턴스명·GPU·소스·예산을 새 판으로 결합한다.
- AWS `g6e.4xlarge`는 실제 L40S를 제공해도 이름에 `l40s`가 없어 기존 녹화 계약이 거부했다. 새 계약은 등록된 정확한 유형만 추가하고 생성 protocol과 견적 원문의 AWS·단일 L40S 48GB·가격·SHA를 대조한다. 실제 GPU admission·임대·소유 child·렌더 검증은 유지하며 메인 WSL 회귀 23개가 통과했다.


### OPS-75: Iray 의존성 통과 뒤 MaterialX 라이브러리 누락

- [AWS F1 촬영 첫 실행](../evals/results/agentic_f1_render_capture_20261028_v1/summary.json)은 임대·입장·상태 1프레임을 통과했다. 이후 AABA의 앱 생성 중 Hydra engine에서 `-11`이 발생했고 소유 자식 정리는 확인됐다. 같은 stderr에는 `libMaterialXRenderHw.so`, `libMaterialXRenderGlsl.so`, `libusdBakeMtlx.so`가 `libXt.so.6`를 찾지 못했다는 기록이 있다. 전체 녹화와 모델 호출은 0건이다.
- 기존 CPU 검사는 `libneuray.so` 하나만 읽었으므로 이 세 라이브러리의 누락을 검사하지 않았다. 새 검사는 네 대상의 단일 파일·raw SHA·`ldd` exit·stdout/stderr의 미해결 의존성·timeout을 확인한다. Iray만 통과하는 경우도 거부하는 회귀를 포함해 12개를 통과했다.
- [Ubuntu Jammy 공식 파일 목록](https://packages.ubuntu.com/jammy/amd64/libxt6/filelist)은 `libXt.so.6`를 `libxt6`에 포함한다. 후속 판은 이 패키지의 복구 전후 원문과 같은 렌더 조건을 비교한다. 누락과 SIGSEGV가 함께 관측됐다는 사실만으로 단일 원인을 확정하지 않는다. 원문 92파일의 사본 해시와 `STOPPED`를 확인했고 이전 실행은 보존했다.


### OPS-76: 재시작 요청 접수와 실제 GPU 준비의 구분

- [F1 재시작 기록](../evals/results/agentic_f1_render_restart_20261028_v1/summary.json)은 Start 요청 exit 0 뒤에도 `READY`에 도달하지 못했다. 공급자 상태는 `STARTING`에서 `STOPPED`로 바뀌었고, 호스트는 대기 상한에서 종료했다. 원격 boot·패키지 복구·GPU·녹화·학습·모델 호출은 0건이다. 추가 Start나 Stop을 보내지 않았으며 원문 385파일의 사본 SHA를 확인했다.
- [Brev 공식 lifecycle](https://docs.nvidia.com/brev/concepts/gpu-instances)은 Stop 때 GPU를 공급자 풀로 반환하고 재시작 때 같은 공급자·리전·유형의 용량을 다시 확보한다고 설명한다. [CLI의 detached 옵션](https://docs.nvidia.com/brev/cli/instance-management)은 상태 polling 없이 반환하므로 요청 접수만으로 준비 완료를 판단하지 않는다. 현 기록은 용량·공급자·bootstrap 원인을 가르지 못하며, 앞선 다른 자원의 HTTP 400과 같은 원인으로 묶지 않는다.
- [새 환경 등록](../bench/protocols/fresh_brev_l40s_continuous_environment_20261028e.json)은 한 번 생성한 환경에서 등록 작업을 연속 수행하고 종료 때 회수·정지를 확인한다. 최초 생성부터 설치·대기·실행·정지까지의 비용과 시간 상한을 유지하며, 이전 실패 기록과 소비는 보존한다. 새 환경 생성은 검사·회복·영상 성공을 뜻하지 않는다.

### OPS-77: 등록된 잠금 경로와 새 환경 부트스트랩

- [실제 검사쌍 P5](../evals/results/agentic_runtime_text_pair_20261029_v1/summary.json)는 원격 helper의 `prepare`에서 중단됐다. 외부 응답은 `unknown`과 일반적인 schema 오류 문구였지만, 내부 원문은 등록된 canonical lease 디렉터리가 없어 strict 경로 해석이 실패한 `FileNotFoundError`였다. 원격 helper는 정상 종료했고 검사·학습·C 승인·추가 모델 호출은 시작되지 않았다.
- 입력과 registration 디렉터리만 만든 부트스트랩에 canonical lease 경로 준비가 빠져 있었다. 새 준비기는 등록된 빈 디렉터리만 만들고 모든 선언 root를 같은 `Resolver`로 점검한다. 기존 lease·lock·heartbeat를 복사하지 않으며 다른 경로와 링크는 거부한다. Python 3.10 CPU 회귀는 11개를 통과했다. 기존 `unknown` 응답과 승인 소비는 보존하며 재시도는 새 작업 폴더와 tag를 쓴다.

### OPS-78: 렌더 라이브러리 복구 뒤에도 남은 Hydra 초기화 충돌

- [후속 F1 촬영](../evals/results/agentic_f1_render_capture_20261029_v2/summary.json)은 네 렌더 라이브러리의 readback과 상태 1프레임을 통과했다. 이전 `libXt.so.6` 누락 오류는 없었으나 정상 정책의 AABA 자식이 새 stage의 viewport Hydra engine을 만들다가 `-11`로 종료됐다. 정책 재생·native 녹화·F1 자식은 시작되지 않았고 소유 자식 정리는 확인됐다.
- 성공한 v11과 이번 실행에는 같은 CUDA 가시성·GLFW·display 경고가 있었다. 이번 AWS 환경은 driver 595.112.07·Ubuntu 22.04 계열, 이전 Nebius 성공 환경은 driver 580.173.02·Ubuntu 24.04 계열이다. 이 차이로 드라이버 하나를 원인으로 확정할 수 없다. 다음 확인은 같은 고정 SDK와 최소 카메라 초기화 경로를 다른 공급자 환경에서 비교하는 것이다.
- [카메라 초기화 child](../bench/probe_isaac_camera_startup_v1.py)는 등록된 자원 proof·현재 임대·소스를 검증한 뒤 App 생성·update 한 번·close만 실행한다. 정책·물리 환경·영상 기록을 제외해 충돌 단계를 나눈다. CPU 회귀 14개를 통과했으며 부모가 실제 exit 0과 소유 process 정리를 확인하기 전에는 성공으로 쓰지 않는다. GPU에서의 초기화 개선은 아직 미측정이다.
- 원문 109파일의 C/D 해시와 `STOPPED`를 확인했다. 최초 생성부터 정지 관찰까지의 견적은 약 $8.54이며 청구액은 아니다. 이 비용은 같은 VM에서 진행한 검사 준비·설치·대기·촬영 시도를 모두 포함하므로 각 단계에 중복 합산하지 않는다. 새 영상·회복·시각 기여 성공으로 집계하지 않는다.

### OPS-79: 생성자 검증과 정지 경로의 필수 설정

- 상위 lifecycle 생성자와 boot가 통과해도 최종 `Driver.stop()`에 필요한 설정이 모두 검증된 것은 아니었다. 촬영 controller는 Stop 한 번과 `STOPPED` 관찰을 끝냈지만, 상위 `finally`는 environment의 `stop_observation_s` 누락으로 `KeyError`를 남겼다. 추가 Stop RPC는 없었고 후속 읽기 전용 조회에서도 정지를 확인했다. 원래 렌더 실패와 이 정지 설정 오류를 별개로 보존한다.
- 후속 환경 등록 전에는 실제 동결 Driver와 실제 phase 형상으로 정지 경로를 CPU에서 실행한다. 이미 정지된 자원, 실행 중 자원의 단일 Stop, 일시 조회 실패, 불명확한 응답, 자원 신원 불일치를 확인한다. 새 공개 phase에 관찰 상한과 재시도 간격을 명시하고, 종료 시 원래 오류가 정지 오류에 가려지지 않도록 각각 기록한다.

### OPS-80: 공개 GPU 조회 목록과 계정별 생성 목록의 차이

- [다른 공급자 생성 시도](../evals/results/fresh_brev_render_creation_20261029f_v1/summary.json)에서 Brev CLI 0.6.335의 `search`는 `massedcompute_L40S`를 반환했지만 `create`는 같은 유형을 인식하지 못해 종료했다. 새 VM은 생기지 않았고 같은 요청을 재전송하지 않았다. GPU·학습·모델 호출도 없었다.
- 상류 [PR #435](https://github.com/brevdev/brev-cli/pull/435)는 공개 search와 조직별 create 목록의 차이로 Shadeform 유형이 거부되는 같은 경로를 설명한다. 이 PR의 수정은 병합되지 않았다. 조회 결과만으로 현재 계정의 생성 가능성을 보장하지 않으며, 접근 필터나 cloud credential 검사를 우회하지 않는다. 후속 헤드리스 고리는 생성이 확인된 AWS 유형을 별도 등록한다.


### OPS-81: 실행기 어댑터와 실제 호출기의 인자 계약

- [R6 검사쌍 실행](../evals/results/agentic_runtime_text_pair_20261029_v2/summary.json)은 서버 신원 확인과 boot·런타임 pin 검증을 통과했다. 그다음 실제 호출기가 위치 인자 두 개를 넘겼으나 어댑터가 키워드 인자만 받아 TypeError가 발생했다. 검사·학습·새 모델 호출은 시작되지 않았다.
- 같은 오류를 실제 동결 호출 방식으로 CPU에서 재현했다. 후속 회귀는 어댑터 모양을 흉내 내는 fixture를 넘어 동결 R48 실행 함수가 첫 전송 경계까지 도달하는지 확인한다. 후속 교정의 adoption helper도 같은 서버 등록으로 묶고 scope 종료 뒤 원복한다.
- C/D 원문 해시 일치와 STOPPED를 확인했다. [R7 재개 등록](../bench/protocols/agentic_runtime_text_s2221_20261029_v7_resume.json)은 같은 설치 환경에 Start 한 번만 허용하고 이전 생성 시각·비용·절대 종료 기한을 유지한다. 이전 승인 소비와 실패 결과를 덮어쓰지 않으며 모델 실패나 회복 성과로 집계하지 않는다.

### OPS-82: 등록 worker와 backend worker의 명령 불일치

- [R7 검사쌍](../evals/results/agentic_runtime_text_pair_20261029_v3/summary.json)은 서버 adoption·boot·runtime pin과 IO service 기동을 통과했지만 검사 child를 시작하기 전에 HOLD로 끝났다. 등록 명령은 `bench/brev_recovery_worker_v1.py`를 승인했고, 실제 backend는 `bench/brev_recovery_worker_v2.py`를 선택했다. `BoundedPopen`의 정확한 명령 일치 검사가 이 차이를 거부했다.
- pair 원문에는 `BackendError`와 `ValueError`만 남았고 정확한 예외 문구는 기록되지 않았다. `launch` 기록은 없고 stdout·stderr도 비어 있다. 같은 명령 불일치를 CPU에서 재현했으며, 이 재현 문구를 원격 원문으로 바꾸어 쓰지 않는다. 검사·학습·새 모델 호출은 0건이고 pair 임대 해제를 확인했다.
- 다음 판은 등록 worker를 선택하는 범위만 새 Parent에 묶는다. 승인 명령 검사를 느슨하게 하거나 허용 목록을 넓히지 않으며, 기존 R7 결과와 승인 소비를 보존한다.
- R7 로컬 원문과 범위를 제한한 원격 snapshot 575파일의 C/D 해시가 일치했다. 이 수치는 원격 전체 namespace 수집을 뜻하지 않으며, 카메라 A의 별도 84파일 보관 기록과 합치지 않는다.

### OPS-83: pair HOLD를 pending으로 보인 상태 투영

- R7의 원격 pair는 이미 `HOLD`였지만 status 응답의 최상위 값은 `pending`, 하위 `pair.status`만 `hold`였다. arm outcome이 없을 때 최상위 상태를 pending으로 만드는 변환과 최상위 상태만 읽는 polling이 결합해 종료된 pair를 계속 기다렸다.
- 후속 원격 상태 판은 pair HOLD를 terminal 상태로 올려 보낸다. Parent의 임대 해제 검증은 유지하며, 상태 표현 수정이 실패한 검사를 완료로 바꾸지는 않는다.

### OPS-84: handoff heredoc 끝 개행과 once 종료 코드 75

- pair HOLD 뒤 읽기 전용 handoff 수집은 remote exit 75로 실패했다. 생성된 셸에서 heredoc 종결자 `PY`와 닫는 괄호가 `PY)` 한 줄로 붙어 셸 구문 오류가 났다. once 디렉터리는 그보다 먼저 만들어져 후속 호출에서 75를 반환할 수 있었다.
- 확인된 원인은 생성 셸의 끝 개행 누락이다. stderr의 연결 재확인 문구는 전송 계층의 후속 동작과 부합하지만 재시도 횟수는 확정하지 않는다. 다음 판은 본문 끝 개행과 `bash -n`을 검사하고 새 nonce를 쓰며 기존 once marker는 지우지 않는다.

### OPS-85: 최소 카메라 초기화에서 남은 Hydra 충돌

- [카메라 초기화 A](../evals/results/isaac_camera_startup_20261029g_v1/summary.json)의 첫 실행은 비대화형 환경에서 EULA 입력을 요청해 exit 1로 끝났다. 등록된 EULA 설정을 복원한 두 번째 실행은 App startup 진입 뒤 234.032초에 `-11`로 종료됐다. startup 반환·update·close marker는 없고 소유 process 정리는 확인됐다.
- 두 번째 stderr의 마지막 Python frame은 viewport texture의 `__enable_hydra_engine`이다. Kit 로그에는 L40S와 CUDA 장치 사용, RTX/Hydra 초기화가 기록돼 있다. 따라서 정책·학습·모델 API는 모두 0건이지만 SDK startup의 GPU 사용까지 0건으로 쓰지 않는다.
- driver 하나를 원인으로 확정할 수 없다. SDK, headless display, CUDA 가시성, driver 중 어느 항목이 충돌을 만들었는지는 남아 있다. 원문 84파일의 C/D 해시를 대조했고, A가 성공 조건을 충족하지 못해 A2는 실행하지 않았다.

### OPS-86: 실행 스크립트 폴더가 표준 `operator` 모듈을 가린 경우

- [R10 검사쌍](../evals/results/agentic_runtime_text_pair_20261029_v4/summary.json)은 adoption·boot·runtime pin과 입력 해제를 통과한 뒤 Windows IO child import에서 중단됐다. `argparse → re → enum`이 표준 `operator`를 불러올 때 스크립트 폴더의 `operator.py`를 먼저 선택했고, 순환 import 중인 `re.compile` 접근이 실패했다. 원격 Parent·검사·학습·C 승인·새 모델 호출은 시작되지 않았다.
- 기존 함수 import fixture는 Python이 파일을 실행할 때 만드는 `sys.path[0]`을 재현하지 못했다. 새 판은 stdlib import 전에 스크립트 폴더만 제거하고 로컬 `operator.py`는 명시 경로로 불러온다. `serve_mailbox.py`, `prepare_operator.py`, `run_continuous_main.py` 세 진입점을 Python 3.10.20과 3.12.13의 실제 subprocess `--help`로 각각 실행해 여섯 건 모두 exit 0을 확인했다.
- Stop 한 번 뒤 2026-10-28 21:04:17 UTC에 `STOPPED`를 확인했고 stop 오류는 없었다. 원문 248파일의 C/D 해시가 일치했다. 최초 g 생성부터 계산한 $12.2049738302는 상한 견적이며 청구액이 아니다. 앞서 보고한 R6·R7·카메라 구간을 포함하므로 해당 비용과 다시 합산하지 않는다.


### OPS-87: 실제 자식 인자와 설정 schema를 함께 검증하기

- [R11 검사쌍](../evals/results/agentic_runtime_text_pair_20261029_v5/summary.json)은 boot와 입력 해제를 통과했지만 IO service가 준비되기 전에 끝났다. 동결 호출기는 `--config`로 pair 설정을 넘겼고, 새 service는 같은 인자를 상위 typed 설정으로 해석했다. 원격 Parent·검사·학습·새 모델 호출은 시작되지 않았다.
- `--help`와 함수 import만으로는 실행 인자에 담긴 설정의 의미를 확인하지 못했다. 후속 판은 typed 설정과 실제 전달된 pair 설정의 경로·원문 SHA를 함께 확인한 뒤 service 생성자까지 실행하는 cold check를 VM 채택 전에 수행한다. 등록 밖 설정을 허용하거나 승인·receipt 경계를 바꾸지 않는다.
- 정지 오류 없이 21:46:00 UTC `STOPPED`를 확인했다. 닫힌 로컬 실행 기록과 이미 회수한 원격 snapshot 331파일의 C/D 해시가 일치했으며, 원격 전체 namespace 수집을 뜻하지 않는다.

### OPS-88: 호스트 Python과 원격 Isaac Python의 표준 라이브러리 차이

- [R12 검사쌍](../evals/results/agentic_runtime_text_pair_20261029_v6/summary.json)은 IO ready까지 도달한 뒤 `unknown`으로 끝났다. RPC 원문에는 원격 helper의 `from datetime import UTC, datetime`에서 발생한 ImportError가 남았다. 실행 대상인 Python 3.10에는 `datetime.UTC`가 없어 Parent prepare 전에 종료됐다. 검사·학습·새 모델 호출은 0건이다.
- 후속 [Remote5](../bench/agentic_brev_probe_transport_remote_v5.py)는 `timezone.utc`를 사용하고 실제 Python 3.10 subprocess 진입을 확인한다. [Service5](../evals/agentic_brev_mailbox_service_v5.py)와 [Transport2](../evals/agentic_brev_mailbox_transport_v2.py)는 제한된 인프라 오류 코드와 RPC 원문 참조를 전달한다. 인프라 실패의 `unknown` 상태를 정상·이상 판정으로 바꾸지 않으며, 모델 실패를 재시도하는 권한으로 사용하지 않는다.
- 정지 오류 없이 22:14:16 UTC `STOPPED`를 확인했다. 닫힌 로컬 실행 기록과 이미 회수한 원격 snapshot 324파일의 C/D 해시가 일치했다. 누적 견적은 최초 g 생성부터의 상한이며 R11 등 앞선 구간을 다시 더하지 않는다. 새 검사·회복 성과로 집계하지 않는다.

### OPS-89: repair transport와 lifecycle 판의 결합

- R13의 등록 소비는 원격 Parent 시작 전에 `registered repair IO/phase contract`에서 거부됐다. repair transport가 이전 lifecycle SHA를 상속해 Runtime·IO의 현재 판과 달랐던 것이 확인된 원인이다.
- 후속 판은 Runtime, IO, repair transport의 lifecycle SHA를 하나로 맞췄고 R14의 `E.registration`과 `T.registered` 소비가 통과했다. 이 통과는 원격 검사나 회복 결과가 아니라 등록 결합 경계의 검증이다.
- 다음 실측은 같은 결합을 사용하는 Parent와 검사 child가 실행되는지 확인한다. R13의 실패 기록과 소비는 보존한다.

### OPS-90: transport class와 등록 bound method의 source 일치

- P14 최종화는 원격 Parent 시작 전에 `transport must use registered bound methods`로 거부됐다. Transport2의 class source와 Transport1에서 상속한 다섯 method의 source 파일이 달라 Host source 검증을 통과하지 못했다.
- Transport3는 다섯 등록 method를 같은 class와 source 파일에서 명시적으로 정의하고 기존 동작을 위임한다. P16의 authority 발급 전 `Host._sources` 검사가 통과했다.
- 이 검증은 source 결합만 확인한다. 실제 원격 Parent와 검사 child 결과는 후속 실행에서 판정한다.

### OPS-91: 공개 상대경로와 private native 경로의 역할 분리

- R15 공개판은 Transport3 source를 결합했지만 private IO의 lifecycle ref가 절대경로여서 Service5의 저장소 상대경로 계약에 거부됐다. authority는 발급되지 않았다.
- 후속 준비기는 공개 ref를 상대경로·raw SHA·commit blob으로 검증하고, native 해석은 private 소비 경계 안으로 제한했다. R16의 E/T 소비와 P16의 authority 발급 전 source 검사가 통과했다.
- 다음 실측은 등록된 service의 예산 검사와 Parent 진입이다. 공개 결과에는 사용자 경로나 private 경로를 포함하지 않는다.

### OPS-92: 공개 lifecycle ref와 Service 예산 검사의 native 해석

- R16의 reference 응답은 `canonical native absolute path required`와 `unknown`을 기록했다. Service5가 공개 저장소 상대경로를 native 절대경로 resolver에 그대로 넘긴 것이 확인된 원인이다.
- Service6는 상대경로, 일반 파일, raw SHA와 committed blob을 확인한 뒤 예산 검사 동안에만 native ref를 사용하고 원래 계약을 복원한다. R17이 이 경계를 지나 Parent launch에 도달한 사실로 다음 경계까지의 진행을 확인했다.
- `unknown`을 정상이나 결함 판정으로 바꾸지 않는다. 검사 child 진입과 모델 판단은 별도 결과에서 확인한다.

### OPS-93: 원격 Parent의 저장소 경로 초기화 순서

- R17은 service budget을 지나 Parent를 시작했지만, Parent3가 저장소의 `bench`, `evals`, `src` 경로를 추가하기 전에 `brev_recovery_backend_v4`를 import해 `ModuleNotFoundError`로 종료됐다. Parent PID는 GPU 진입 전에 끝났고, 관찰된 GPU process는 0건이었다.
- Parent4는 외부 의존 import 전에 등록 저장소 경로를 추가한다. worker 선택, `BoundedPopen`, 판정 조건은 유지한다. 로컬 공개 회귀 9건과 원격 Python 3.10 진입 검사는 통과했지만 검사 child의 GPU 결과는 아직 별도 판정 대상이다.
- R17 결과는 `infrastructure_hold`이며 새 모델 호출과 학습은 0건이다. 로컬 원문과 범위를 제한한 원격 snapshot 575파일의 C/D 해시가 일치했고, 이는 원격 전체 namespace 수집을 뜻하지 않는다. R18 결과는 이 기록에 포함하지 않는다.

### OPS-94: Camera V8 종료 증거 보관 완료

- V8은 startup 반환, update 한 번, terminal 봉인, close 진입을 기록했다. `close_returned`는 false였지만 부모가 소유한 자식 PID 89865는 exit 0으로 끝났고 process 정리도 확인됐다. 필수 marker 6개와 terminal·shutdown policy SHA, 물리 FLOCK, inactive heartbeat, GPU process 0, READY 상태를 함께 확인해 SDK startup 1회를 수용했다.
- V8 원문 79개, 102,511바이트를 별도 보관하고 원본과 사본의 파일별 SHA를 다시 계산했다. 모든 파일이 일치했고 inventory SHA는 `798fcc73cd199543e01418a9e3d86669dd626f496247d15c2116862cba273250`이다. 앞서 보관한 V4~V7 원문 252개와 상대경로가 겹치는 파일은 없다.
- 수용 범위는 카메라 startup·update·종료 경계다. 정책·물리 환경·Replicator는 실행하지 않았고 native 정책 영상, 학습, 회복, 모델 판단 기여는 확인하지 않았다. V7의 exit `-11` 기록도 그대로 남긴다.
- V7과 V8은 종료 계약과 실행 시점의 cache·환경 조건이 다르므로 153.330초와 10.174초를 속도 개선 수치로 비교하지 않는다. driver 하나를 원인으로 확정하지도 않는다.

결과: [등록·실행 경계 원문 요약](../evals/results/agentic_runtime_infrastructure_failures_20261029_v1/summary.json), [카메라 startup·종료 수용](../evals/results/isaac_camera_startup_h_v8_20261029_v2/summary.json).

### OPS-95: leaf lease의 source map과 commit blob 불일치

- 관찰: R18 원격 pair는 HOLD였고, 같은 오류를 실제 JOB4 CPU 소비자에서 재현했다. 등록 source map 372개 중 14개 파일의 blob이 leaf lease가 가리킨 commit에 없었다.
- 확인 원인: leaf lease의 source_commit이 새 파일들을 포함하지 않았다. Parent의 공개 anchor와 Runtime source는 서로 다른 역할이므로 모든 anchor를 같은 SHA로 합치는 방식으로 고치지 않는다.
- 남은 판정: lease16의 소비 통과 이후 원격 검사 결과는 R19 원문 회수 범위에서 판정한다.
- 수정·검증: 후속 ENV16/lease16은 source map 376개를 결합했고 같은 실제 CPU leaf가 통과했다. R18 읽기 전용 회수 자료는 8파일이며 당시 parent PID 부재와 GPU process 빈 관측을 기록했다.
- 다음 조건: 각 anchor의 역할을 유지한 등록에서 source map의 실제 blob과 production leaf 소비를 확인한다.

근거: `r18-diagnosis` SHA256 `b4bf0b4c696d5acd7f8a72caae0ea9b11745b4b1998d641fe64aeec68db813cb`; `r18-raw-01` SHA256 `a1f749f2fd5a4f3fa237993f609e0000a8ad69a230545fdcde1e72b5a73124f5`; `r19-lease-leaf` SHA256 `7bc222db9a9839fe79f4e549c512b1409dbcb2e1f469604a349f26f5cb2dc4e4`.

### OPS-96: 원격 pending 뒤 Windows WSL 제어 연결 실패

- 관찰: R19 마지막 원격 응답의 pair 상태는 pending이었다. 이후 Windows의 새 WSL 진입은 `Wsl/Service/0x80072747`로 실패했고, Main은 failed/stop_required로 종료됐다.
- 확인 원인: 원문은 WSL 소켓·버퍼 오류와 inventory/handoff 제어 실패를 기록한다. 어떤 자원 고갈이 이를 만들었는지는 확정되지 않았다.
- 남은 판정: 원격 검사 완료, 학습, 회복과 새 모델 호출 수는 원문 회수 전 미확정이다. pending을 최종 검사 결과로 바꾸지 않는다.
- 정리·대조: 별도 watchdog의 결합 inventory에서 g STOPPED를 확인했다. Main 실패 영수증과 watchdog 종료 영수증은 별개이며, 로컬 제어 실패만으로 원격 실행이 어디까지 진행됐는지 판정하지 않는다.
- 다음 조건: 기존 결과를 보존한 채 원격 pair/worker/lease 원문을 읽기 전용으로 회수해 판정한다.

근거: `r19-last-status-stdout` SHA256 `b841694d1418a915c788eba01633b45c7d5bce9c818f273426e673b099051b07`; `r19-failed-inventory-stdout` SHA256 `7fe0cfd01899795edc4ae8d690c4eb5a717d1ed6034218e0ee349f811526ed6a`; `r19-main-result` SHA256 `200f1f2b7e4f2762b4f18ab9c4dfb1e6b8f48ed9860b0646582dfedbab94c627`; `r19-watchdog-finished` SHA256 `120c3c38c95674f92ab9fd967aba3b729e26a0183d2ca901908d5b2edf8ea40a`.

### OPS-97: library ready 뒤 renderer 생성 본문의 의존 함수 누락

- 관찰: H5 library checker는 ready였고 네 library의 ldd 종료 코드는 모두 0이었다. 이어 실행한 renderer heredoc에서 `NameError: name 'require' is not defined`가 발생했다.
- 확인 원인: 생성 Python이 validator의 `require` 의존 함수를 포함하지 않았다. 실패는 GPU child 전 단계였으며 library ready는 renderer 초기화 통과를 뜻하지 않는다.
- 남은 판정: H6 전체 생성 본문의 CPU fixture 통과와 실제 GPU 영상 측정은 서로 다른 범위다. H6의 실제 영상 결과는 이 기록에 없다.
- 수정·검증: H6는 `require`와 validator를 같은 본문에 포함한다. 동결 기록은 Windows 6 passed, Python 3.12 WSL 선택 회귀 5 passed/1 deselected를 담고 있으며, source·test·등록 초안 5개 SHA를 원문과 대조했다. H5 회수 기록은 사본 5파일 해시 일치와 STOPPED를 기록한다.
- 다음 조건: 의존 함수가 포함된 동결 본문을 등록하고 실제 renderer·영상 원문을 별도 측정한다.

근거: `h5-renderer-stderr` SHA256 `1aa63f0d2e2b8c16076b8f972695956691ec8e92a336dccbf3be40a68e0f42d4`; `h5-checker-report` SHA256 `70f35f352b4d5e95472d9301bf947bb85d6b1d92f070a9369db8a10abec32c83`; `h6-freeze` SHA256 `fde9dd3b556d1975dce7c64d0dcbe171aac139f5b976475451e4cd39552ff10f`; `h5-stopped` SHA256 `7081bb90c02d075c25989fbce0756bff08cfc6ef5bbef997818a3b42ae800617`.

### OPS-98: 같은 VM의 interop host 관찰과 영구 셸 복구

- 관찰: Windows 스냅샷의 wslhost 335개는 모두 같은 VM ID를 사용했다. 이 중 Windows 부모가 끝난 것은 331개였지만, 부모 소멸만으로 host를 정리하지 않았다.
- 확인 범위: 복구 영수증에는 불필요해진 자체 H watchdog이 정상 종료한 뒤 영구 Bash 연결이 열렸다고 기록돼 있다. WSL 재시작·UNC 접근·타 프로젝트 프로세스 종료는 수행하지 않았으며, 결합 inventory는 두 소유 자원 STOPPED를 기록한다.
- 남은 가설: same-VM interop host 누적과 커널 자원 압박의 기여 가능성은 남아 있다. 단일 자원 고갈을 원인으로 확정하지 않는다.
- 검증: 복구 영수증의 inventory·g watchdog 종료·Main R19 결과 SHA를 실제 읽은 원문과 대조했다. 이 복구는 로컬 제어 연결의 상태를 기록한다.
- 다음 조건: 열린 제어 연결에서 R19 원문을 회수하고 원격 검사·학습·회복 판정을 이어간다.

근거: `wsl-diagnosis` SHA256 `feeb62771d16286a9aef6087d22e56449de3a448c61a4e507e500dd592060709`; `wsl-recovery` SHA256 `98b2eb0a352fd2b5efacf55908be96511ab6e3a24a7f5fd4f79760a115570421`; `wsl-recovery-inventory` SHA256 `109e913392a04de797f4bbd42b7cb4ede85a4fb38bbd3bed4e324e7084178728`.

[실행 인프라 결과와 artifact 목록](../evals/results/agentic_execution_infrastructure_20261029_v2/summary.json).

### OPS-99: 업로드 캐시와 추출 캐시의 namespace 불일치

- 관찰: 업로드와 SHA 검증 뒤 추출 단계에서 `cache source가 일반 단일 링크 파일이 아니다`로 끝났다. 업로드는 새 namespace의 `input-cache/registered-inputs.tar`를 썼지만 추출 설정은 이전 캐시 경로를 읽었다. 이 오류 문구만으로 이전 경로의 파일 상태를 단정하지 않는다.
- 확인·수정: [이전 phase](../bench/protocols/agentic_f1_render_capture_phase_20261029h_v9.json)와 [수정 phase](../bench/protocols/agentic_f1_render_capture_phase_20261029h_v10.json)의 namespace·cache 필드를 비교했다. 업로드·추출이 같은 경로를 소비하도록 묶고 실제 두 함수가 생성한 Bash 본문으로 이전 경로 거부와 새 경로 추출을 확인했다. 기존 실패 결과는 보존했다.
- 결과: 정상·F1 각 625프레임, sidecar·render-sync 결합, C/D 3,959파일 일치와 CPU 입력 준비를 완료했다. 이는 이미 본 seed 2221의 녹화·입력 결합 결과다. [기술 결과](../evals/results/agentic_f1_fresh_render_dev_20261029h_v1/summary.json)

### OPS-100: 운영체제에 따른 이력 경로 표현과 snapshot 결합

- 관찰: Windows에서 준비한 context를 Linux의 기존 검증기에 넣으면 `변조 또는 오래된 snapshot`으로 거부됐다. 유료 모델 요청 전에 발견했다.
- 확인: snapshot 생성기가 이력 경로에 `Path.resolve()`를 적용한다. 같은 파일의 Windows 표현과 POSIX 표현 때문에 host context가 달라졌으며, 모델 view·입력 수치·이력·이미지는 같았다.
- 수정: 원본 prepared 파일은 보존하고 같은 파일을 가리키는 Linux 경로와 context digest만 새 산출물에 결합했다. 메시지·params·model view·이력의 동일성을 검사하고 [별도 사전등록](../bench/protocols/fresh_frame_linux_host_projection_s2221_20261029_v1.json)에 변경 범위를 고정했다. 검증기의 기존 판정 규칙은 바꾸지 않았다.
- 결과: 기존 caller·provider parser·host 선택 변환을 모두 통과했다. 이후 새 프레임 여섯 장을 사용한 첫 실제 호출은 유효한 `P_reward` 선택을 반환했다. 인용 근거는 telemetry 여섯 개로, 이미지의 선택 기여나 새 회복을 뜻하지 않는다. [모델 결과](../evals/results/fresh_frame_probe_choice_dev_s2221_20261029_v1/summary.json)

### OPS-101: Windows loopback 연결과 기존 Linux 세션의 전송 범위

- 관찰: Windows에서 WSL의 loopback 연결을 시도하면 `WinError 10054`가 발생했다. 원인은 확정하지 않았다. Linux 안의 같은 서버에 접속한 Brev 조회는 통과했다.
- 처리: Windows forwarding을 완료 근거로 쓰지 않고, 기존 Linux 세션 안의 client·server 경로로 첫 모델 호출을 실행했다. API 키는 Linux에 유지했고 새 WSL 진입과 GPU 작업은 추가하지 않았다. Windows loopback은 계속 미확인 상태다.
- 결과: 원래 요청 9건을 보존하고 새 요청 한 건을 정산해 10/12, pending 0이 됐다. 이 전송은 GPU 승인이나 새 자원 생성 권한을 대신하지 않는다. [전송 등록](../bench/protocols/persistent_brev_provider_transport_20261029_v1.json)

### OPS-102: 녹화 소스와 원격 checkout 커밋의 혼동

- 관찰: H 재시작은 READY에 도달했지만 metadata 관측이 예상 HEAD 불일치로 멈췄다. 실제 저장소는 clean 상태의 `6f05df1a0054117257d8ac3c4e54de933ebfa8f5`였고, 관측 등록은 녹화 소스 기준인 `694df50e6c2a196da25b9b40c8788921caebd6a8`을 HEAD로 요구했다. GPU 검사와 새 API 요청은 시작하지 않았다.
- 확인: 이전 녹화의 boot 계획은 phase 등록 커밋을 checkout하며, 녹화에 필요한 소스는 그 조상 커밋과 파일 결합으로 검사한다. 두 커밋의 역할을 구분해야 한다.
- 수정: [관측 등록 v4](../bench/protocols/agentic_h_readiness_time5_20261029_v4.json)는 예상 원격 HEAD와 녹화 소스 기준을 분리하고, 유료 Start 전에 이전 boot 계획의 실제 checkout 값을 읽어 대조한다. 이전 실패 원문과 소비 기록은 보존한다.
- 검증 범위: 이전 boot 계획을 소비하는 CPU 검사와 잘못된 예상 HEAD 거부를 확인했다. v4의 실제 metadata 수집과 후속 GPU 검사는 별도 실행 결과로 기록한다.


### OPS-103: 생성자 논리 경로와 Linux 실행 경로의 혼재

- 관찰: 실제 등록 소비 코드가 `path outside registered root map` 또는 `provider path projection`으로 거부했다. GPU와 새 API는 시작 전이었다.
- 확인: Windows의 producer 논리 경로는 참조 키로 유지해야 하는 반면, host의 local root와 계층별 host 경로는 Linux 실제 경로여야 한다. 두 역할을 같은 방식으로 바꾸면 경로 투영이 깨진다.
- 수정·검증: 새 등록 판에서 참조 키와 local root를 구분했다. 실제 `E.registration`, `T.registered`, repair contract 소비를 모두 통과했다. 이전 판은 보존했다.

### OPS-104: 보관 이력과 새 모델 응답 참조의 고리 계약

- 관찰: 실제 `Runtime.advance()`가 먼저 `fresh host-preloaded context differs`, 수정 뒤 `KeyError: capsule`로 멈췄다. prefix validation과 constructor만 확인했을 때는 발견되지 않았다.
- 확인: Runtime은 이력을 별도 사본에 보관하므로 경로와 context digest가 달라진다. 새 모델 호출은 실제 응답·journal을 갖지만 기존 Runtime은 옛 CALL capsule을 요구한다.
- 수정: [새 prefix 판](../evals/agentic_fresh_selection_prefix_v2.py)은 같은 초기 이력 bytes와 나머지 context의 일치를 확인한다. 실행 이력은 별도 ledger에 쌓는다. [새 Runtime 판](../evals/agentic_investigation_runtime_v2.py)은 실제 응답 참조를 CALL capsule과 구분한다.
- 검증: [실제 상태 전이](../evals/results/fresh_frame_runtime_operation_dev_s2221_20261029_v2/summary.json)가 API10의 `P_reward` 선택으로 pending operation 한 건을 만들었다. 새 API·승인·dispatch는 0이다. producer·remote parent·Host의 같은 참조 계약은 후속 구현 대상이며 GPU 검사나 새 회복 결과로 계산하지 않는다.

### OPS-105: 상속된 시간 판과 첫 원격 단계의 여유 검사

- 관찰: F2 녹화 controller는 먼저 이전 time5의 고정 lifecycle 값 때문에 CPU에서 거부됐다. 새 time 판의 전체 값을 소비하도록 고친 뒤에는 warm 입장 후 첫 원격 작업 전에 비용·시간 여유 검사로 거부됐다. 실제 녹화·학습·모델 호출은 0이었다.
- 확인: constructor 통과만으로 각 원격 단계 timeout과 회수·종료 reserve의 합이 실행 창에 들어간다고 볼 수 없다. `--check-plan`이 만든 빈 output 폴더도 실행 출력과 분리해 보존했다.
- 다음 조건: 원래 비용 기산점과 누적 사용량을 유지한 새 실행 창을 등록하고, 실제 `budget()`으로 단계별 timeout·회수·종료 여유를 유료 실행 전에 확인한다. 기존 실패 원문과 Stop 기록은 보존한다.

### OPS-106: Main과 별도 IO의 Host·Parent 선택 불일치

- 관찰: fresh 응답용 Main을 선택해도 별도 IO는 옛 Service를 생성했다. Host의 증거 검증과 원격 CLI에도 이전 Parent 선택이 남아 있었다. 생성자·source closure 통과만으로 실제 요청·응답 경로의 판 일치를 보장할 수 없었다.
- 확인: Host5의 상속된 증거 검증은 옛 Parent schema를 읽었고, Service의 host-intent 재검증은 Host1을 생성했다. 원격 전송의 Parent 선택과 import 시 계산한 provider packet schema도 함께 바꿔야 했다.
- 수정: [Host6](../evals/agentic_brev_runtime_host_v6.py)·[Service9](../evals/agentic_brev_mailbox_service_v9.py)·[Remote8](../bench/agentic_brev_probe_transport_remote_v8.py)·[Parent8](../bench/run_agentic_brev_probe_parent_v8.py)을 같은 계약으로 연결했다. 등록 prefix 밖의 검증된 후속 검사 요청은 예외 대신 HOLD를 반환하며, 변조·미등록 요청은 계속 거부한다. 기존 판과 결과는 보존했다.
- 검증 범위: CPU 회귀는 Host 선택, Parent/provider schema 일치, CLI 실행, 원래 비용 기산점과 Stop reserve 검사를 다룬다. 새 GPU 검사·회복 결과는 아직 없다. 유료 실행 전에는 최종 등록의 실제 참조로 Main과 IO가 같은 소스를 소비하는지 확인한다.

### OPS-107: 바깥 실행 창과 내부 교정 lease의 기한 불일치

- 관찰: 새 실행 입력을 준비하던 중 repair layout이 참조한 기존 lease basis의 기한이 바깥 실행 창보다 먼저 끝난 상태임을 확인했다. GPU 실행 전 발견했으며, 이 불일치로 학습이 실패한 결과를 추가하지 않는다.
- 수정: [새 lease basis](../bench/protocols/agentic_runtime_text_s2221_20261029_v31_lease_basis.json)는 같은 H 자원과 source 체크포인트를 유지하고, Time9의 기한·예산·현재 소스 결합을 사용한다. repair transport·lifecycle·Runtime도 같은 기한과 소스 해시로 다시 생성했다. 최초 생성 시각과 기존 소비는 초기화하지 않는다.
- 검증: 기존 Linux 셸의 실제 `agentic_play_job_v4.validate_protocol`이 공개 등록 파일의 419개 소스 결합과 유효 기한을 통과했다. 회복 기준·교정·학습량은 바꾸지 않았고, 실제 GPU 입장과 실행 결과는 별도로 확인한다.

### OPS-108: 압축 해제 위치와 worker 입력 경로의 불일치

- 관찰: Runtime43·Parent12의 정상 대조 검사는 worker가 1.57초 만에 exit 1로 끝나 `InvalidExecutionEvidence`가 됐다. 시간 초과는 없었고, 자식 정리와 실행 전후 소스 일치는 통과했다. worker stderr는 `ValueError`만 남겼다.
- 확인: 회수한 `job.json`을 기존 worker 검증 함수에 넣어 `입력 경로가 허용된 workspace를 벗어났다`를 재현했다. 입력 archive는 현재 저장소의 등록 폴더에 풀렸지만 source root-map의 local 값이 과거 workspace를 가리켜, checkpoint와 학습 manifest가 worker의 허용 root 밖 경로로 전달됐다.
- 수정: [Runtime44](../bench/protocols/agentic_runtime_text_s2221_20261030_v44.json)·[Parent13](../bench/protocols/agentic_runtime_text_s2221_pair_fresh_20261030_v13.json)은 source의 논리 경로·SHA·원문을 유지하면서 local 배치를 등록된 압축 해제 위치와 맞춘다. 새 실제 경로도 같은 원문으로 해석하도록 evidence root-map을 결합한다. 교정 단계의 lease basis는 현재 Time10 판으로 연결한다. worker의 경로 검증, 검사 크기와 판정 기준은 바꾸지 않는다.
- 검증: 기존 job 거부와 수정 배치 후 worker 통과를 CPU에서 확인했다. 새 graph의 두 정책 checkpoint·manifest 네 개의 SHA와 두 `E._source` 검증도 통과했다. 이 확인은 새 GPU 검사나 회복 결과를 뜻하지 않는다. 이전 실패 기록은 보존하며, 실패 원문은 같은 실행 수명에서 회수하도록 준비했다.

### OPS-109: 입력 배치 변경과 실제 업로더의 계약 결합

- 관찰: Parent13은 원격 환경 확인 뒤 `registered historical source namespace changed`로 중단됐다. 검사·학습·새 모델 호출 전 실패였으며 이전 원문은 보존했다.
- 확인: Main에 주입된 업로더가 과거 source root의 local·remote 경로가 같아야 한다고 검사했다. 생성자·Host 검증과 worker 입력 검증만으로는 이 업로드 함수의 조건을 확인할 수 없었다.
- 수정: [Runtime45](../bench/protocols/agentic_runtime_text_s2221_20261030_v45.json)·[Parent14](../bench/protocols/agentic_runtime_text_s2221_pair_fresh_20261030_v14.json)은 현재 저장소 안의 등록 입력 배치를 업로더와 함께 결합한다. source archive는 그 위치 한 곳에 설치하고, 논리 참조·원문 SHA·경로와 링크 검증을 유지한다.
- 검증: 실제 등록 파일과 operator의 참조 읽기·archive 생성·복사 경로 함수를 사용해 업로더 전체의 script 3건·copy 2건을 CPU에서 확인했다. 전송을 수행하지 않는 driver로 생성 Python 본문까지 검사했다. GPU 실행 전 같은 준비 함수가 실제로 소비되는지 확인하며, 이 계획 검증을 원격 실행 성공으로 계산하지 않는다.

### OPS-110: 실제 실행 환경 서명과 등록 기대값의 불일치

- 관찰: Runtime45의 정상 대조 `P_reward` worker는 31.97초·exit 0으로 완료됐고, source 전후 일치와 자식 정리도 통과했다. 상위 고리는 완료 증거를 봉인하지 못해 후속 교정을 시작하지 않았다.
- 확인: 실제 source snapshot에는 CUDA 가용성·장치 수가 있지만 기대값에는 두 항목이 없었다. Python의 `base_prefix`도 등록된 시스템 경로와 실측 uv 경로가 달랐다. 전체 snapshot 일치를 요구하는 production 조건이 이 입력을 거부함을 재계산했다. Parent outcome 본문은 회수하지 못했으므로 유일한 최종 원인이라고 단정하지 않는다.
- 수정: [Runtime46](../bench/protocols/agentic_runtime_text_s2221_20261030_v46.json)·[Parent15](../bench/protocols/agentic_runtime_text_s2221_pair_fresh_20261030_v15.json)에 실측 메타데이터 세 항목만 다시 결합한다. 검사와 교정·평가의 기대 snapshot을 함께 맞추고, 다른 버전·패키지·소스·체크포인트 변경은 거부한다. 실행 시에는 별도의 fresh GPU proof가 필요하다.
- 검증: 원문 재계산·부정 변형 검사 3건이 통과했다. 새 검사 기대값, 새 회복 기대값, 실제 실행 전후 snapshot이 같은지도 확인했다. 검증 조건을 완화하거나 이 환경 서명 후보를 실행 권한으로 쓰지 않는다.

### OPS-111: 검사 뒤 GPU 관측 영수증의 root-map 누락

- 관찰: Runtime46의 정상 대조 검사는 exit 0으로 끝났지만 Parent outcome은 `path outside registered root-map`으로 HOLD가 됐다. 결함 검사·후속 모델 판단·교정은 시작하지 않았다.
- 확인: Parent는 각 operation의 `gpu-observations` 폴더에 실행 전후 영수증과 GPU/process CSV를 쓴다. 생성된 검증 plan에는 형제 `raw` 폴더만 등록돼 있어, 전체 source snapshot 일치를 통과한 뒤 영수증 경로를 해석하지 못했다.
- 수정: [Runtime47](../bench/protocols/agentic_runtime_text_s2221_20261030_v47.json)·[Parent16](../bench/protocols/agentic_runtime_text_s2221_pair_fresh_20261030_v16.json)은 operation별 observation 폴더를 parent와 host의 evidence map에 함께 등록한다. 전체 execution root를 허용하거나 기존 경로·SHA·GPU admission 검증을 완화하지 않는다.
- 검증: 실제 이전 binding으로 parent plan의 54개 논리 경로를 재계산했다. 빠진 12개 경로는 observation 영수증과 CSV였으며, 정확한 두 root를 추가한 뒤 누락·중복 mapping은 0이었다. 미회수 영수증의 내용·SHA나 새 고리 실행 완료까지 검증한 결과로 해석하지 않는다. 이전 HOLD와 Stop 기록은 보존한다.

### OPS-112: IO 서비스의 공개 lifecycle 참조 형식

- 관찰: Runtime47의 유료 실행 전 CPU 확인에서 Main·Host·bootstrap은 통과했지만, IO 서비스의 budget 소비가 `public lifecycle repo-relative path required`로 거부됐다. 새 GPU 검사·모델 호출은 없었다.
- 확인: 새 기한을 결합하면서 IO contract에 절대 파일 경로를 넣었다. 이 필드는 공개 등록 파일의 저장소 상대 경로와 SHA를 요구하며, 실제 서비스가 native 경로로 변환한 뒤 등록 커밋의 blob을 검증한다.
- 수정: [Runtime48](../bench/protocols/agentic_runtime_text_s2221_20261030_v48.json)·[Parent17](../bench/protocols/agentic_runtime_text_s2221_pair_fresh_20261030_v17.json)은 같은 lifecycle 파일을 상대 참조로 결합한다. 원래 파일·기한·소스 검증 조건은 유지한다. Runtime47은 미실행 판으로 보존한다.
- 검증: 실제 `_native_lifecycle_reference` 함수로 절대 참조 거부, 등록된 상대 참조 통과, 잘못된 SHA 거부를 확인했다. 전체 실제 소비 확인과 원격 실행은 별도 결과로 남긴다.

### OPS-113: 준비 단계의 root 해석과 실행 폴더의 배타 생성 충돌

- 관찰: Runtime48은 원격 준비 단계의 `Resolver`가 아직 없는 execution 경로를 해석하다 `FileNotFoundError`로 중단됐다. 새 검사·모델 요청·학습 전 실패이며 이전 원문과 Stop 기록은 보존했다.
- 확인: 관측 경로를 root-map에 등록하는 것만으로 충분하지 않았다. 준비 단계는 root가 존재해야 하지만, 자식 실행기는 같은 operation 폴더를 배타 생성한다. 임시로 폴더를 만들었다 지우면 후속 plan 조회에서 다시 실패한다.
- 수정: [Parent10](../bench/run_agentic_brev_probe_parent_v10.py)은 관측 원문을 등록 폴더 안의 영속 observation root에 보관한다. 논리 참조는 operation별 경로를 유지하고, [Remote10](../bench/agentic_brev_probe_transport_remote_v10.py)과 [Service11](../evals/agentic_brev_mailbox_service_v11.py)이 반출·해석 경로를 같은 mapping으로 연결한다. 실행 폴더의 배타 생성, 영수증 SHA, GPU 입장 검사는 유지한다.
- 검증: 실제 backend와 Resolver의 관측 영수증 소비, prepare 뒤 unit 생성·후속 조회, CSV 변경 거부를 CPU에서 확인했다. 공개 회귀의 비공개 binding 의존도 synthetic fixture로 제거해 비공개 폴더 없는 사본에서 관련 7개 검사가 통과했다. 이 확인은 새 GPU 고리 완료나 회복 결과를 뜻하지 않는다.

### OPS-114: 재사용 입력의 영수증 이름과 생성기 인터페이스 불일치

- 관찰: 새 실행 입력을 준비할 때 상속한 projection receipt와 새 receipt의 이름이 같아 `FileExistsError`가 발생했다. 이후 CPU 소비에서는 이전 public protocol 파일명 잔존과 wrapper의 `write` 인터페이스 누락도 발견됐다. 유료 실행 전 발견한 오류이며 모델 진단 실패로 계산하지 않는다.
- 수정: 새 입력 사본에서 상속 영수증을 provenance 이름으로 보존하고 새 영수증은 배타 생성한다. protocol 파일명을 명시적으로 결합하고, preflight의 scoped writer가 실제 함수의 namespace에 전달되도록 맞춘다. 원본 입력과 실패한 부분 출력은 덮어쓰지 않는다.
- 재발 방지: import·도움말·생성자 통과와 최종 입력의 실제 소비를 구분한다. 유료 Start 전에는 같은 최종 등록으로 finalize·operator·repair·preflight·시간 및 비용 예약까지 실행한다. 이미 통과한 CPU 단계는 원문 SHA를 대조한 뒤 재사용하며, 새 실행 창이 필요한 경우 원래 비용 원점과 소비를 보존한 별도 등록을 사용한다.

### OPS-115: 정상 대조군과 교정 전용 종료 검증의 arm 불일치

- 관찰: 정상 NONE 대조군 후보가 CORRECTED만 허용하는 종료 검증기를 재사용하고 있었다. 등록 생성기에서도 CLI 상대경로와 고정 절대경로를 다르게 비교했다. 두 문제는 GPU 실행 전 발견했다.
- 수정: [NONE 종료 검증](../bench/local_same_seed_none_terminal_v1.py)을 새로 두고, manifest의 NONE·seed·학습량·재개하지 않은 설정·결함 없음과 완료 영수증을 확인한다. [등록 생성기](../bench/finalize_local_same_seed_none_control_v1.py)는 경로를 resolve한 뒤 비교한다. 기존 교정 전용 계약은 바꾸지 않았다.
- 재발 방지: 같은 seed 독립 학습과 이전 교정 산출물의 바이트 일치는 복사의 증거로 쓰지 않는다. 별도 실행의 실제 learn 반환과 no-resume manifest를 확인하고 바이트 일치는 따로 보고한다. 이 정책은 실행 전에 등록했다.
- 검증: 관련 Python 3.10 테스트 15개와 Ruff가 통과했다. 실제 source 결합 검사 뒤 NONE 300회 학습·26조건 평가가 exit 0으로 끝났다. [정상 대조 결과](SAME-SEED-RECOVERY-CONTROL.md)는 기존 교정 정책과 같은 checkpoint와 지표를 재현했으며, 기존 미회복 판정을 덮어쓰지 않는다.

### OPS-116: Windows 줄바꿈과 공개 파일 SHA의 차이

- 관찰: 원문 labels·결합 JSON에 CRLF가 들어 있었고 Git의 텍스트 정규화가 이를 LF로 저장하려 했다. 작업 폴더 검사는 통과했지만, 커밋 전 staging blob 대조에서 사전등록된 labels SHA와 달라짐을 찾았다.
- 수정: 해당 결과 폴더와 그림의 결합 JSON만 `.gitattributes`의 `-text`로 지정해 원문 바이트를 보존했다. 실험 수치·사전등록 SHA·모델 원문은 바꾸지 않았다.
- 재발 방지: 공개 manifest는 작업 폴더뿐 아니라 staging/commit blob의 크기와 SHA도 대조한다. [V1](../evals/results/visual_integrity_dev_20261030_v1/manifest.json)·[V2](../evals/results/visual_localization_dev_20261030_v2/manifest.json)의 119개 항목이 일치한 뒤 push했다. CSV를 새로 만들 때는 줄 끝을 LF로 고정한다.

### OPS-117: source 학습 사이 GPU 프로세스 신원 확인 실패

- 관찰: 등록 source 13개 완료 뒤 다음 학습의 GPU 입장 검사에서 실행 경로를 확인할 수 없는 프로세스가 나타났다. 해당 stage에는 GPU 관측만 남았고 학습 자식은 시작되지 않았다. 이후 해당 프로세스가 사라졌으며 같은 입장 규칙으로 다시 확인한 두 관측은 통과했다. 프로세스의 생성·종료 원인은 미확정이다.
- 처리: 완료 source를 재학습하지 않고 [새 재개 등록](../bench/protocols/lineage_source_continuation_20261031_v1.json)과 [재개 실행기](../evals/lineage_source_continuation_v1.py)로 남은 원래 job만 실행한다. 기존 실패 원문과 완료 증거는 보존하며 GPU 입장 기준은 바꾸지 않는다.
- 재발 방지: 학습 전 입장 거부와 학습 실패를 구분한다. 완료 prefix의 체크포인트·accepted·종료 증거를 검증하고, 새 출력 경로와 별도 실행 anchor에서 남은 작업만 허용한다. CPU fixture 통과와 실제 재개 완료는 따로 기록한다.
- 후속 관찰: 재개 실행에서 3개 source를 더 완료한 뒤 사진 앱의 GPU 문맥이 허용목록에 없어 다음 학습 시작 전에 다시 멈췄다. 완료 16개를 묶는 [재개 v2](../evals/lineage_source_continuation_v2.py)와 [입장 대기](../evals/lineage_gpu_wait_v1.py)를 추가했다. 허용목록·메모리·compute 차단 기준은 바꾸지 않고, 제한 시간 안에 연속 clear 관측을 기다린다. 이 대기는 학습 재시도가 아니다. 고정 평가도 [v2 실행기](../evals/lineage_fixed_eval_batch_v2.py)에서 같은 경계를 쓴다.

### OPS-118: 확인 검사의 보조 입력과 모델 송신 중복 경계

- 관찰: 실행 전 consumer 대조에서 `P_value`가 최종 체크포인트 외에 형제 `model_0.pt`도 읽는 것을 확인했다. 새 backend가 최종 파일만 묶으면 보조 파일의 누락·교체를 놓칠 수 있었다. NIM queue에는 응답 파일 생성 전에 송신이 불명확하게 끝난 요청을 재송신하지 않는 영구 표식도 필요했다. 두 항목은 실제 새 검사·API 실행 전에 보완했다.
- 수정: [paired backend](../evals/lineage_probe_backend_v1.py)는 보조 체크포인트의 경로·SHA·크기를 plan·claim·receipt에 기록하고 실행 전후 검증한다. [NIM transport](../evals/lineage_nim_transport_v1.py)는 provider 호출 전에 write-once dispatch claim을 만들고, claim·failure가 남은 요청의 재전송을 거부한다. 종료된 queue와 미확정 요청은 보존한다.
- 검증: 보조 파일 삭제·실행 중 변조, 한 role의 중복 실행, 송신 예외 뒤 재호출, worker deadline·중복 worker·queue 종료를 CPU에서 확인했다. 이 확인은 새 source의 원인 확정이나 회복 결과가 아니다.

### OPS-119: GPU 입장 차단이 반복된 비교와 오류 근거 누락

- 관찰: 첫 30행 진단 비교에서 26역할이 같은 미등록 `svchost.exe`의 C+G 컨텍스트로 입장 차단됐다. 해당 역할의 Isaac 자식은 시작되지 않았다. 기존 실행기가 오류 형식만 남기고 다음 역할·행을 계속해 `unknown`이 누적됐다. 정확한 Windows 서비스 신원과 실제 compute 작업은 원문으로 확인하지 못했다.
- 수정: [입장 어댑터](../evals/lineage_probe_admission_v2.py)는 plan 준비 전과 실행 claim 전에 기존 GPU 기준으로 연속 clear 두 번을 기다린다. 대기 한도를 넘으면 epoch·배치를 중단하고 오류 메시지·traceback·대기 영수증 SHA를 남긴다. 서비스 전체 허용이나 차단 기준 완화는 없다. 운영 사용에는 별도 공개 protocol과 source anchor가 필요하다.
- 검증: CPU에서 busy→clear→clear 후 한 번 실행, 영구 busy의 claim·자식·후속 역할 0, lease 변경·source 변조 거부, scope 복원을 확인했다. 기존 실패 원문과 HOLD는 유지한다. 같은 비교에서 발생한 별도 `PermissionError` 한 건은 상세 예외가 없어 원인 미확정이다.
- 공개 집계: [30행 결과](../evals/results/lineage_diagnosis_dev_20261031_v1/summary.json)는 영수증 없는 부모 예외 26건과 완전한 probe 영수증을 구분한다. [내보내기](../evals/lineage_comparison_export_v1.py)는 측정값 없는 기록을 `unknown`으로 보존하며 정상 측정으로 바꾸지 않는다.

### OPS-120: 첫 교정 사전 실행의 owned 폴더 생성 계약

- 관찰: 3회 갱신 사전 실행이 GPU clear 두 번을 통과한 뒤 `owned/launch.json` 생성에서 `FileNotFoundError`로 멈췄다. Isaac 학습 자식과 모델 호출은 0이며 임대는 실패 종료 후 해제됐다.
- 원인: `run_owned`는 호출자가 만든 폴더에 launch를 배타 생성한다. 첫 교정 부모가 학습·평가의 owned 폴더를 만들지 않은 채 이 함수를 호출했다.
- 수정: [v2 어댑터](../evals/lineage_recovery_execution_v2.py)는 호출 직전 폴더를 배타 생성하고 기존 실행기를 재사용한다. 원래 v1 소스·등록·실패 기록은 보존하며, 새 source와 protocol에 결합한 새 출력 경로로 확인한다.
- 검증: 실제 `run_owned`에서 v1의 폴더 누락을 재현하고, Windows Job·spawn만 대체한 CPU 회귀로 launch·stdout·stderr 생성과 학습·평가 폴더의 재사용 거부를 확인했다. CPU 확인과 실제 3회 갱신 완료는 별도 상태다.

### OPS-121: 학습 완료 뒤 임대 메타데이터 읽기에서 평가 중단

- 관찰: v2는 실제 3회 갱신 사전 실행과 300회 교정 학습을 완료했다. 이후 고정 평가 입장에서 `gpu-lease.json`을 읽는 중 `PermissionError`가 발생했다. 오류 파일명과 호출 경로를 확보했으며 평가 자식은 시작되지 않았다. 메타데이터의 원자 교체와 겹쳤을 가능성은 있으나 파일 접근 거부의 세부 원인은 확정하지 않았다.
- 수정: [lease reader v2](../evals/lineage_lease_context_v2.py)는 canonical 메타데이터 파일의 `EACCES`와 같은 파일명에만 2초·최대 20회 안에서 새 읽기를 시도한다. 다른 파일, JSON 오류, 잘못된 소유자·기간·lock은 재시도하지 않는다. 원래 PID 계층·heartbeat·배타 lock 검증을 그대로 수행하며 과거 메타데이터를 캐시해 수용하지 않는다.
- 재개: [평가 전용 실행기 v4](../evals/lineage_recovery_continuation_v4.py)는 완료된 학습·체크포인트·텔레메트리와 consumed 원장을 검증해 재사용한다. 원래 학습 anchor와 새 등록 anchor를 각각 검증하고 조상 관계를 확인한다. 원래 실패와 원장은 보존하고, 새 원장 사본에 평가 결과를 추가한다. 새 학습·새 모델 호출·새 교정 승인은 없다. 실제 26조건 평가가 exit0으로 완료돼 [회복 결과](../evals/results/lineage_first_recovery_s2303_20261031_v1/summary.json)를 남겼다.
- 재발 진단: [probe 예외 기록](../evals/lineage_probe_failure_evidence_v3.py)은 원래 판정을 바꾸지 않고 message·traceback·errno·winerror·filename을 별도 파일에 남긴다. 과거 상세 정보가 없는 오류를 같은 원인으로 소급 확정하지 않는다.
