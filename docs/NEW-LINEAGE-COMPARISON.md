# 새 정책 계보 비교를 위한 source 수집

이 실행기는 Go2 평지 과제에서 새 seed의 정책 10개와 같은 seed의 정상 참조를 수집한다. 진단 비교에 앞서 학습 입력을 고정하는 단계다. 모델 호출·검사 선택·교정·회복 판정은 실행하지 않는다.

## 등록 범위

| 사례 | seed 후보 | source | 정상 참조 |
|---|---|---|---|
| 정상 2개 | 2301·2302 | NONE | 자신 |
| 보상 2개 | 2303·2304 | F1 | 각 seed의 NONE |
| 모터 2개 | 2305·2306 | F2 | 각 seed의 NONE |
| 보상과 PPO epoch 2개 | 2307·2308 | F1+R1 | 각 seed의 NONE |
| 등록 교정 밖 2개 | 2309·2310 | F5·F3 | 각 seed의 NONE |

총 18회 학습이며 모두 4096개 환경, 300회 갱신, 새 초기화를 쓴다. 결함 강도와 동결 결함 구현은 바꾸지 않는다. seed 2399의 NONE 1회 갱신은 실행 계층 사전 확인용이며 비교 사례에 합산하지 않는다.

결함 유형은 개발에 사용한 범주다. 새 seed는 새 결함 유형을 뜻하지 않는다. 실행 전 검사 범위는 현재 저장소의 실행 기록과 지정 Isaac Lab 로그이며, 다른 장비의 미보관 기록까지 미노출을 증명하지 않는다. 설계자가 기존 비교와 같은 seed 대조 결과를 본 뒤 만든 실행기다.

같은 seed 정상 참조를 두는 이유는 [3221 대조 실험](SAME-SEED-RECOVERY-CONTROL.md)에 있다. 교정 뒤 정책이 원래 대조군의 기준을 넘지 못했으나, 같은 seed의 정상 학습도 같은 체크포인트와 평가값을 만들었다. 이 수집기는 seed가 다른 대조군에만 의존하는 비교를 피한다.

## 실행

Windows PowerShell과 설치된 Isaac Lab을 사용한다. 공용 lease는 자식 실행 전후에 소유자·PID·heartbeat·배타 lock을 확인한다. 각 자식은 기존 Windows Job 실행기로 관리하며 실패하면 다음 학습을 시작하지 않는다.

등록 v2는 Windows 가상환경 Python 중계 프로세스를 실제 부모 계층으로 확인한다. GPU 표시 앱에는 경로가 확인된 Windows 설정 창 두 개와 Codex 표시 도우미를 추가했으며 계산 작업·엔진·메모리 차단 기준은 유지한다. v1은 CPU 입장 확인 뒤 학습·모델 호출 없이 보존했다.

```powershell
cd "$HOME/rl-triage-agent"
$anchor = git rev-parse HEAD
$evidence = Join-Path $HOME 'rl-triage-source-evidence-v1'

# 실제 등록 source·protocol·Isaac 파일을 CPU에서 확인
& ./.venv/Scripts/python.exe -B evals/lineage_source_batch_v1.py verify --lab "$HOME/IsaacLab" --anchor $anchor

# 사전 실행: 새 폴더만 허용하며 GPU lease를 취득·해제
& ./bench/run_lineage_source_batch_v1.ps1 -Anchor $anchor -Output "$evidence/smoke" -Mode smoke -Execute

# 완료 원문을 재검증해 사전 실행을 반복하지 않고 source 18개 수집
& ./bench/run_lineage_source_batch_v1.ps1 -Anchor $anchor -Output "$evidence/sources" -Mode all -SmokeResult "$evidence/smoke" -Execute
```

등록 파일과 묶인 소스가 해당 커밋에 있어야 한다. 결합 밖의 후속 커밋은 허용하며 결합 파일이 달라지면 멈춘다. 위 경로에 이미 결과가 있으면 새 출력 경로가 필요하다. 소비된 seed를 같은 tag로 다시 학습하는 자동 재시도는 없다.

학습 자식은 실제 `PPO.update()` 반환 횟수와 `learn()` 반환을 확인하고 체크포인트 SHA를 앱 종료 전에 봉인한다. `Runner.load()` 호출을 거부해 기존 가중치의 이어 학습을 막는다. 부모는 종료 코드 0뿐 아니라 완료 증거·체크포인트 바이트·소유 프로세스 정리를 검증한다. 체크포인트 바이트 중복은 보고하며 다른 seed로 바꾸지 않는다.

상한은 학습별 부모 벽시계 1800초, 배치 전체 21600초다. 예상 소요 시간이나 GPU 커널 시간 측정값을 뜻하지 않는다. 이 판의 유료 클라우드 GPU Start와 모델 API 예산은 모두 0이다.

## 후속 평가와 분리

수집 후에는 동결된 26조건 평가와 기존 밴드로 정상 참조의 유효성과 결함의 실패 존재를 확인해야 한다. 정상 실패·결함 실패 부재가 있으면 해당 사례와 제외 이유를 보존하고 대체 seed를 찾지 않는다. 계획 10개와 유효 비교 짝 수를 함께 보고한다.

그다음 등록 순서·고정 실행 가능 순서·Nemotron 3.5 수치 입력을 같은 검사·판단 예산에서 비교하는 별도 판을 등록한다. 영상으로 더 나은 검사를 골랐다는 개발 근거가 아직 없어 시각 진단 arm은 보류한다. [VL 렌더 무결성 검사](VISUAL-INTEGRITY-OBSERVER.md)의 결과는 영상 입력을 가리는 기능의 근거이며 진단 선택 이득의 근거로 합산하지 않는다.

후속 API 예산은 아직 배정하지 않았다. 이전 초안의 72회는 사례 10개·모델 arm 하나·판단 2회·판단당 최대 3요청일 때만 상한 60회를 덮는다. 판단을 8회까지 허용하면 상한은 240회여서 별도 배정이 필요하다. 이 수집 판이 해당 요청을 허용하지 않는다.

현재 학습 수집 구현과 CPU 검증은 완료됐으며 실제 source 수집과 후속 평가 결과는 실행별 원문으로 확인한다. 수집기의 완료를 검출률·회복률·모델 우위로 읽지 않는다.
