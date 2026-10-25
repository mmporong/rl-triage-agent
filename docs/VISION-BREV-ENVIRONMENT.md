# Brev 환경 재사용과 복원 자료

설치된 환경의 주요 패키지와 Isaac Lab commit이 맞으면 재설치를 생략한다.
새 VM에는 같은 버전을 지정한 설치 스크립트를 사용하고, 단계별 결과를 확인해 재개한다.
현재 확인 범위는 CPU에서의 bundle 생성·검증과 설치 제어 흐름이다.
이 경로로 새 VM을 복원하거나 Isaac을 실행한 결과는 아직 없다.

## 고정하는 항목

| 항목 | 버전 또는 소스 |
| --- | --- |
| Python | 3.10.12 |
| pip | 26.2.1 |
| uv bootstrap | 0.13.5 |
| PyTorch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| Isaac Sim | 4.5.0.0 |
| Isaac Lab | commit `90b79bb2d44feb8d833f260f2bf37da3487180ba`, package 0.41.3 |
| isaaclab_rl | 0.1.8 |
| RSL-RL / Gymnasium | 2.3.3 / 1.2.0 |

이 값은 기존 Brev 실행에서 회수한 설치 보고서와 소스 결합을 따른다.
OS image digest는 확보되지 않아 `null`로 남긴다. 다른 wheel 전체와 드라이버까지
같은 환경이라는 뜻은 아니며, 새 자원의 CUDA·Isaac 시작·영상 한 프레임은 별도로 확인해야 한다.
uv bootstrap wheel은 [공식 PyPI metadata](https://pypi.org/pypi/uv/0.13.5/json)의
Linux x86-64 파일 SHA-256을 recipe에 고정한다.

## 로컬에서 자료 준비

[`vision_brev_environment_v1.py`](../bench/vision_brev_environment_v1.py)는 설치나 다운로드 없이
새 폴더에 pins, 설치 스크립트, 검증 스크립트, constraints와 파일별 해시를 만든다.
생성 중에는 torch를 import하거나 GPU를 조회하지 않는다.

```powershell
cd "$HOME/rl-triage-agent"
New-Item -ItemType Directory -Force '.triage-runtime/vision-environment' | Out-Null
python bench/vision_brev_environment_v1.py build --output .triage-runtime/vision-environment/bundle-v1
python bench/vision_brev_environment_v1.py verify-bundle --bundle .triage-runtime/vision-environment/bundle-v1
(Get-FileHash -Algorithm SHA256 '.triage-runtime/vision-environment/bundle-v1/bundle-manifest.json').Hash.ToLowerInvariant()
```

출력 경로가 이미 있으면 덮어쓰지 않는다. `bundle-manifest.json`은 각 recipe 파일의
bytes·SHA-256과 LF 해시, pins와 cache manifest의 해시를 묶는다.
VM에 옮긴 뒤에도 같은 파일 집합과 해시를 대조하고 로컬 원본을 보존한다.
로컬에서 확인한 `bundle-manifest.json`의 SHA-256도 따로 보관한다.
설치·재사용 명령의 `--expected-manifest-sha256`에는 이 값을 전달한다.
원격 사본에서 새로 계산한 값을 승인된 원본 해시 대신 사용하지 않는다.

자산·wheel cache를 기록하려면 `build --cache-spec <로컬 JSON 경로>`를 사용한다.
입력은 `logical_id`, `root`, `files`를 가진 객체의 배열이며, `files`에는 root 안의
상대 경로를 하나씩 명시한다. 출력에는 root의 절대 경로를 넣지 않는다.
홈 폴더를 훑어 cache나 자격증명을 찾지 않으며 symlink·hardlink·경로 이탈을 거부한다.

cache manifest는 파일 목록과 해시다. 자산 bytes를 옮기거나 복원을 마쳤다는 증거가 아니다.
cache 입력을 생략한 bundle은 빈 목록을 기록한다. 실제 필요한 자산은 실행 전에 확보하고
회수 inventory에도 포함해야 하며, 자격증명은 이 자료에 넣지 않는다.

## 설치된 환경 검사

현재 잔액·견적·소유 자원과 실행 사전등록을 확인한 뒤, 준비된 Brev Linux에서 검사한다.
예를 들어 기존 설치 폴더를 읽는 명령은 다음과 같다. bundle 경로는 복사한 로컬 원본과
해시 대조를 마친 경로로 정한다.

```bash
cd "$HOME/workspace/rl-triage-vision-bundle-v1"
bash install.sh \
  --expected-manifest-sha256 "$VISION_BUNDLE_MANIFEST_SHA256" \
  --existing-environment-root "$HOME/workspace/rl-triage-brev-e2e-20261024" \
  --reuse-receipt "$HOME/workspace/vision-existing-environment-reuse-v1.json"
```

이 모드는 대상 환경을 수정하거나 ownership marker를 새로 만들지 않는다.
주요 패키지 버전과 Python·Isaac Lab HEAD가 맞고 추적 파일에 변경이 없을 때만 `reusable`을 기록한다.
맞지 않거나 interpreter가 없으면 해당 상태와 실패 코드를 남기며 설치로 전환하지 않는다.

## 새 환경 설치와 재개

등록된 자원·예산 안에서 새 환경 설치가 필요한 경우에만 같은 bundle의 설치 경로를 실행한다.

```bash
cd "$HOME/workspace/rl-triage-vision-bundle-v1"
bash install.sh --expected-manifest-sha256 "$VISION_BUNDLE_MANIFEST_SHA256"
```

기본 대상은 `$HOME/workspace/rl-triage-vision-v1`이다.
작업 전용 ownership marker가 없는 기존 비어 있지 않은 폴더는 거부한다.
단계 완료 기록이 있어도 현재 버전·소스 검사를 통과해야 해당 단계를 생략한다.
모든 항목이 이미 맞으면 설치 명령을 다시 실행하지 않고 재사용 결과를 남긴다.
실패 코드는 유지하고 attempt별 결과를 보관하며, 부분 설치를 완료로 세지 않는다.

설치 결과는 패키지 준비 범위다. 실제 자원별 GPU 입장, 짧은 학습, renderer 동기화와
영상 확인은 [Brev 수집 경로](VISION-BREV-DATA.md)의 별도 실행 조건이다.
실제 복원 시험을 수행하면 설치·자산 준비·검증을 포함한 시간을 기록하고,
단계별 자료를 로컬로 회수한 뒤 소유 VM을 종료한다.

## 학습·렌더 실행 전 점검

[`run_vision_brev_preflight_v1.py`](../bench/run_vision_brev_preflight_v1.py)는
별도 seed의 `NONE` 정책을 환경 4,096개에서 20회 학습한 뒤,
생성된 checkpoint를 결합한 기존 A→B→A 렌더 동기화 검사를 실행한다.
소스 수집과 같은 runtime·GPU 잠금·자원 증거를 사용하며, 한 번의 `advance`는 한 단계만 진행한다.

점검은 별도 원장에 남긴다. 정상·결함·교정 각 두 건의 300회 학습 계획과
소스 수집 10개 action, checkpoint별 렌더 검사 네 건은 유지한다.
점검의 최대 시간은 학습 1,800초·렌더 900초다. 사전등록은 기존 소스·렌더 예약액에
이 2,700초의 비용을 더해 전체 compute 상한을 검사한다.
실행 전과 단계 사이에도 현재 임대·기한·GPU 점유·비용 여유를 다시 확인한다.

학습의 실제 갱신 수·checkpoint·학습 기록·자식 종료 증거와 렌더 실행의
환경·소스·동기화 원문이 모두 유효해야 `ready_for_source_batch=true`를 반환한다.
그때도 첫 소스 action에 필요한 1,800초의 유효 시간이 남아 있어야 한다.
완료 자료는 새 GPU 실행 없이 재검증할 수 있으며, 불명확한 pending은 자동 반복하지 않는다.

렌더 점검은 checkpoint 바이트를 확인하지만 정책을 로드하거나 물리 step을 진행하지 않는다.
따라서 이 결과는 짧은 학습과 해당 렌더 조합의 실행 근거다.
정책 재생·전체 녹화·회복이나 모델 판단의 근거로 사용하지 않으며
`model_use_ready`는 항상 `false`다. 이 실행 경로는 CPU 회귀로 확인하며,
실제 Brev 실행 결과는 별도 기록으로 구분한다.

## 렌더 시스템 의존성

학습에 필요한 패키지 버전이 맞아도 카메라 렌더러의 시스템 라이브러리가 없을 수 있다.
[`check_vision_renderer_libraries_v1.py`](../bench/check_vision_renderer_libraries_v1.py)는
Isaac Sim의 `libneuray.so` 한 개를 찾아 CPU에서 `ldd` 종료 코드와 누락 의존성을 검사한다.
라이브러리·검사 명령이 없거나, 여러 파일이 발견되거나, 검사 오류·시간 초과·누락이 있으면 `not_ready`를 기록한다.
출력은 새 JSON 파일로 보존하며 GPU나 Isaac Sim 앱을 시작하지 않는다.

Ubuntu 22.04에서 `libGLU.so.1`이 없을 때 필요한 런타임 패키지는
[`libglu1-mesa`](https://packages.ubuntu.com/jammy/libglu1-mesa)다.
설치 뒤 이 CPU 검사를 통과하더라도 실제 렌더 동기화 검사를 별도로 통과해야 한다.
이 검사의 `model_use_ready`는 항상 `false`다.

## 시각 정보 노드 등록

[`vision_play_worker_v2.py`](../bench/vision_play_worker_v2.py)는 앱을 시작한 뒤
`isaacsim.core.nodes` 확장을 활성화하고 한 번 갱신한다. 이어서 `rgb`,
`IsaacReadTimes`, `ReferenceTime` 등록을 확인해야 기존 녹화·동기화 검사로 들어간다.
캡처 전에 [준비 함수](../bench/vision_replicator_startup_v1.py)가 Replicator Orchestrator를
켜고 capture-on-play를 끈 뒤 설정값을 다시 읽는다. 수동 캡처는 허용하고 timeline 재생에
따른 자동 캡처는 막는다. 설정이 일치하지 않으면 녹화에 들어가지 않는다.
활성화나 등록 확인이 실패하면 실패 원문을 남기며 대체 시각 값을 만들지 않는다.
이 순서는 Isaac Sim 4.5의 [확장 활성화 API](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/python_scripting/manual_standalone_python.html)를 따른다.

등록 결과와 캡처 준비 전후 설정은 `renderer_registration.json` 보조 파일로 보존한다.
캡처에 들어가기 전에 job 파일 옆에도 `job.renderer_registration.json`을 기록하므로
첫 캡처가 시간 제한으로 종료되어도 준비 증거가 남는다.
기존 녹화 결과·운동 trace·동기화 기준과 모델 사용 조건은 유지한다.
새 실행 진입점은 기존 프로세스 소유권·시간 제한·GPU 잠금·종료 확인을 재사용하며,
새 worker와 실행기 소스 해시를 포함한 사전등록에서만 이 경로를 선택한다.

## 반환 시각과 리소스 종료

[`vision_timing_adapter_v1.py`](../bench/vision_timing_adapter_v1.py)는 SDK가 반환한
필드 이름·유형·값과 RGB 해시를 검증 전에 보관한다. 동기화 점검은 RGB 원문도
함께 보존한다. 알 수 없는 값 유형은 유형과 미지원 표시를 남기며 시각을 추정하지 않는다.

시각 정규화는 등록된 11필드 또는 그 필드에 `execOut` 실행 포트가 추가된 경우만
허용한다. 실행 포트는 메타데이터로 분리하고 기존 시각 검증을 적용한다.
다른 추가 필드나 누락은 거부하며 물리 상태·시각 허용 오차·A→B→A 픽셀 기준은 유지한다.
원문 보존은 실패 진단을 위한 것이며 모델 입력 승인이나 동기화 통과를 뜻하지 않는다.

캡처가 끝나면 annotator를 분리하고 render product를 해제한 뒤 환경과 앱을 닫는다.
결과 또는 실패 기록은 앱 종료 전에 남긴다. 앱 종료가 반환하지 않으면 부모 실행기의
시간 제한과 GPU 프로세스 종료 검사를 거치며 정상 종료로 처리하지 않는다.

## 자산 준비와 SDK 종료 경계

[`vision_capture_readiness_v1.py`](../bench/vision_capture_readiness_v1.py)는 환경 생성 뒤
USD 자산 loading·streaming 상태를 확인하고 제한된 앱 갱신으로 준비를 기다린다.
물리 상태나 시각이 바뀌면 실패한다. 카메라 위치를 적용한 뒤 SDK의 렌더 누적 초기화
API를 호출하며, 기존 시각·물리 상태·정확한 A→B→A 화면 일치 기준을 유지한다.
준비 기록은 렌더 동기화 성공이나 모델 입력 승인을 대신하지 않는다.

[`vision_play_worker_v3.py`](../bench/vision_play_worker_v3.py)는 SDK 실행과 종료 판정을
분리한다. 시작 인자는 기존 profile의 `fast_shutdown=False`를 유지한다. 결과 또는 실패를
기록한 뒤 별도 종료 정책이 SDK의 빠른 종료를 켜고, 전후 설정과 결과 해시를 영수증에 남긴다.
CPU supervisor는 child가 끝난 뒤 종료 코드, 원본 job, 산출물 해시, 종료 영수증과 실패 기록을
확인한다. child의 종료 코드가 0이어도 실패 기록이나 불완전한 파일이 있으면 실패다.
그 뒤 기존 backend가 의미 검증·시간 제한·프로세스 소유권·GPU 해제를 다시 확인한다.
이 경로가 실제 자원에서 정상 종료와 같은 상태의 화면 재현을 보장하는지는 별도 점검 대상이다.

준비·초기화·시각 원문·종료 영수증은 하나의 manifest에 묶인다. 새 실행 영수증이
manifest의 해시를, attestation이 실행 영수증의 해시를 보관한다. 회수한 파일에서도
같은 연결과 캡처별 시각·RGB를 다시 대조한다. 원문이 누락되거나 달라지면 모델 입력으로
넘기지 않는다. 기존 실행 영수증은 기존 schema로 검증한다.
