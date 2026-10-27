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
