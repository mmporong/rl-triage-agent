# Brev에서 영상과 운동 기록 수집

새 수집 경로는 같은 Play 실행의 영상·명령·운동·카메라·시각을 묶는다.
학습과 녹화의 원장은 재개할 수 있으며, 완료 산출물은 해시를 다시 확인해 반환한다.
CPU 계약과 회귀를 검증했고, 실제 GPU 실행은 환경 준비와 촬영 준비 장벽까지 통과했다.
새 source 배치와 전체 renderer 동기화 검사는 아직 완료되지 않았다.
실패 원인과 재발 방지 검사는 [트러블슈팅 기록](TROUBLESHOOTING.md#클라우드-영상-수집)에 남긴다.

## 실행 순서와 상한

[`run_vision_brev_batch_v1.py`](../bench/run_vision_brev_batch_v1.py)는 준비된 Brev Linux 환경에서
학습 네 건, 정상·결함 짝 평가 두 건, Play 녹화 네 건을 한 단계씩 진행한다.
VM 임대와 결과 회수·종료는 바깥 실행자가 맡는다.

학습 원장에는 서로 다른 source seed 두 개와 각 seed의 `NONE`·`F1`·`CORRECTED`만 등록한다.
각 학습은 기존 조건인 환경 4,096개와 300회 갱신을 유지한다. source 단계는 정상·결함 네 건만
실행하며, 교정 두 건은 후속 고리의 교정 승인이 소비된 경우에만 허용한다.
같은 seed의 세 조건과 한 정책에서 자른 시간 구간을 독립 사례로 세지 않는다.

실행 전 별도 source protocol에 seed, 고정 평가, 소스·패키지 버전과 해시, 녹화 설정,
소유 자원, 가격·비용·시간 상한을 고정하고 커밋해야 한다. 현재 잔액을 확인한 예산 기록과
원시 GPU 기록은 비공개에 보관하며, 실행기는 원본 파일 해시와 자원·GPU·protocol 결합을 대조한다.
GPU를 사용하는 학습·검사·평가·녹화는 같은 잠금을 공유한다. GPU가 사용 중이면 새 학습 예약을 만들지 않는다.
잠금은 OS가 관리하며, 소유 프로세스가 종료돼도 잠금 파일 때문에 다음 입장이 막히지 않는다.
클라우드 목록 원문과 실행 전후 GPU UUID·PID 조회 원문을 보관하고, `Unknown`·`N/A`나
불완전한 행을 비어 있는 GPU로 해석하지 않는다. 완료된 학습·평가·녹화를 다시 읽을 때도
관측 영수증과 원문 파일의 해시를 확인한다.

완료된 학습은 checkpoint뿐 아니라 telemetry·manifest·readback·실행 기록까지 다시 확인한다.
교정 정책은 원본 결함 정책과 승인 근거도 확인한다. 완료 여부가 불명확한 요청은 `HOLD`로 남기고
새 실행으로 추정해 반복하지 않는다.

## 같은 Play에서 얻는 자료

[`record_vision_evidence_v1.py`](../bench/record_vision_evidence_v1.py)와
[`vision_play_worker_v1.py`](../bench/vision_play_worker_v1.py)는 고정 Go2 Play task에서
환경 하나를 12.5초 동안 기록한다. 영상은 640×480, 초당 50프레임이며 명령·수치 자막을 넣지 않는다.
몸체 좌표계의 일정한 전진 명령을 사용하고, 카메라는 월드 축 방향을 유지하며 로봇 위치를 따라간다.
정상·결함·교정 정책에 같은 설정을 적용한다.

각 프레임에는 실제 simulation time과 step, 명령·속도·자세, 카메라 위치를 남긴다.
RGB와 `IsaacReadTimes`·`ReferenceTime`은 같은 render product에서 읽는다.
녹화와 동기화 검사는 모두 `rep.orchestrator.step(rt_subframes=1, pause_timeline=False, delta_time=0.0)`을
사용한다. 추가 물리 step 없이 픽셀과 상태 시각을 대조하며, 인코딩 후에는 실제 영상 PTS와 기록을 다시 맞춘다.

출력에는 원본 trace와 두 시간 구간의 입력 명세·운동 기록이 들어간다.
각 구간의 여섯 프레임은 [영상 입력 계약](VISION-EVIDENCE-CONTRACT.md)으로 준비한다.
다른 실행의 고정 평가 수치를 영상 시점의 운동 기록으로 붙이지 않는다.

## 동기화 검사와 모델 사용 경계

CPU에서 시각·파일 결합이 맞아도 렌더링된 픽셀이 최신 상태라는 실측 근거가 되지는 않는다.
별도 `render_sync_smoke`는 warmup 뒤 정지한 물리 상태에서 카메라 A→B→A 캡처를 수행하고
이미지·상태·renderer 시각을 대조한다. 통과 결과의 범위는 해당 환경과 renderer 조합이다.

녹화 완료만으로 `model_use_ready=true`를 내지 않는다. 원본 녹화는
`capture_complete_gpu_smoke_pending`으로 남기고, 실제 GPU 동기화 검사를 연결하기 전에는
선택 비교 입력으로 사용하지 않는다. 영상 관찰·검사 선택·원인 확인·회복은 각각 별도로 판정한다.

새 준비 분리 경로는 한 Writer를 붙인 채 준비 출력 한 번을 검증하고, 측정 네 장을 수집한 뒤
Writer·render product·환경을 정리하고 원문을 봉인한다. 준비 출력은 측정 journal에서 제외한다.
[GPU 개발 확인](../evals/results/vision_prepared_capture_dev_20261026t/summary.json)에서는
준비 frame 32 뒤 측정 frame 33~36이 왔고, callback 다섯 개에서 측정 journal 네 개를 만들었다.
기존 marker·frame/rational/reference 시각·A-A-B-A 픽셀·물리 불변 기준과 정리 검증을 통과했다.
회수한 원문에서도 journal과 픽셀을 다시 대조했다.

이 경로는 앞선 초기 빈 출력·복수 callback을 본 뒤 설계한 개발 확인이다.
이번 실행에서는 빈 출력이 관찰되지 않았으며, 이전 SDK 출력의 원인을 확정하지 않는다.
이 네 프레임 실행의 학습·모델 API·정책 전체 녹화는 각각 0회다.
결과는 이 renderer 조합의 정지 장면 동기화 범위다.

후속 [정책 녹화 판](../bench/protocols/vision_prepared_record_20261026_v2.json)은 같은 Writer의 준비 출력을
측정에서 제외하고 정책을 움직이며 625프레임을 기록했다. 각 캡처 안에서는 물리·시각 불변을 검사하고,
정책 step 사이에는 물리 진행을 허용한다. 원문은 프레임마다 검증해 journal로 저장한다.
영상의 실제 PTS와 같은 Play 운동·명령 기록, 두 구간의 측정·입력 명세를 함께 확인한다.

[개발 결과](../evals/results/vision_prepared_record_dev_20261026v/summary.json)는 준비 포함 콜백 626개에서
측정 625개를 만들었고, Writer detach와 환경 정리 뒤 봉인한 산출물 여덟 개를 로컬에서도 재검증했다.
기존 20회 학습 checkpoint의 기술 녹화이므로 정상·실패·회복 라벨이나 모델 비교 준비 완료를 부여하지 않는다.
이 경로도 앞선 SDK 출력을 본 뒤 설계했다. 새 정상·결함 정책의 고정 평가와 모델 비교는 별도로 진행한다.

## 비용과 종료

실행기는 다음 action의 최대 시간·compute 비용을 시작 전에 예약하고, 완료 뒤 측정값을 검사한다.
VM compute 비용은 기록된 시작 시각부터 대기·준비·회수 시간을 포함해 계산한다.
저장소 과금과 공급자의 실제 청구액은 이 견적에 포함하지 않는다.
VM 종료 전에는 생명주기 비용 상태를 `open`으로 두며 최종 비용 완료를 주장하지 않는다.

바깥 실행자는 단계마다 자료를 로컬로 회수하고 파일별 해시를 확인한 뒤 소유 인스턴스의 정확한 ID만
Stop 또는 삭제해야 한다. 다른 프로젝트 자원을 종료하지 않는다.
환경 복원과 새 VM 생성의 실제 검증 상태는 별도 실행 기록으로 남긴다.

[환경 재사용과 복원 자료](VISION-BREV-ENVIRONMENT.md)는 주요 패키지와 소스가 맞는 기존 환경의
재설치를 생략하고, 새 VM에 사용할 설치 recipe와 명시한 cache 파일 목록을 로컬에 보관한다.
CPU 검증과 실제 환경 복원·Isaac 실행의 결과를 구분한다.

## CPU 검증

```powershell
cd "$HOME/rl-triage-agent"
python -m pytest -q tests/test_vision_brev_backend_v1.py tests/test_run_vision_brev_batch_v1.py tests/test_vision_remote_gpu_v1.py tests/test_record_vision_evidence_v1.py tests/test_vision_gpu_admission_v1.py tests/test_brev_recovery_backend_v4.py
python bench/run_vision_brev_batch_v1.py --help
```

POSIX process group 정리 검사는 Linux에서도 실행한다. 현재 CPU 검증을 실제 Isaac 학습·렌더링의
통과나 영상 선택 효과로 합산하지 않는다.
