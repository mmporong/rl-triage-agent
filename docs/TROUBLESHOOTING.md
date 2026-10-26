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
  CPU 회귀를 통과했으며 실제 checkpoint 재검사는 별도 사전등록 뒤 수행한다.
- 근거: [실패 요약](../evals/results/vision_checkpoint_smoke_dev_20261026y/summary.json),
  [새 smoke 실행기](../bench/run_vision_checkpoint_smoke_v2.py),
  [새 입력 검증기](../evals/vision_record_source_readiness_v2.py).

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
