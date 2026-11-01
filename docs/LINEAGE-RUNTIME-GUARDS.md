# Lineage runtime guard 공개 결과

이 문서는 실제 lineage source에 연결된 입력·품질·busy·transport guard의 CPU 검증 결과를
설명한다. 공개 JSON은 `evals/results/lineage_runtime_guards_20261101_v1/`에 있다.

## 결과 범위

공개 matrix는 10개 경계를 포함한다.

| 근거 종류 | 행 수 | 내용 |
| --- | ---: | --- |
| 기존 actual model/SDK receipt | 1 | case-03 F1 진단 입력의 기존 call0005를 현재 NIM verifier로 읽기 전용 검증 |
| 통제 perturbation | 7 | 등록 입력 누락·권한, blur 격리, 시각·수치 충돌과 unknown, typed timeout, pending/no-resend |
| 통제 injection | 1 | GPU 실측이 아닌 busy callback 주입으로 dispatch 전 HOLD 확인 |
| fixture transport | 1 | CPU fixture send 1회 뒤 완료 receipt 재사용, 추가 send 0, 자동 retry 0 |

10행 중 8행은 HOLD, 2행은 verified다. 모든 행에서 승인, dispatch 권한, 자동 retry는
`false`다. blur 행은 constant PGM의 Laplacian variance 경계로 `image_count=0`과
`frame_refs_count=0`을 확인한다. source frame의 의미적 blur 인식 결과가 아니다.

## 서로 다른 두 actual receipt 맥락

`summary.json`은 기존 모델 응답 두 건을 구분한다.

1. `diagnosis_input`: case-03의 F1 source를 `logs_only` 정책에 넣어 이미 생성된 call0005다.
   matrix는 receipt를 현재 verifier로 다시 읽었고 새 요청을 보내지 않았다.
2. `cached_normal_shadow`: case-01의 NONE 정상 shadow를 `logs_only` 정책에 넣어 이미 생성된
   call0001이다. 완료 receipt 재호출 CPU 증거에서 반환 digest가 같았고 sender callback과 새 HTTP는
   모두 0이었다.

두 맥락은 같은 결과 재사용 여부와 입력 역할이 다르다. call0005는 F1 진단 입력 receipt 검증이고,
call0001은 정상 shadow의 cached 완료 결과 재사용 검증이다. fixture transport의 send 1회는 이 두
actual receipt와 분리되며 모델 요청 수로 세지 않는다.

## 공개 파일

- `summary.json`: 행 수, 근거 종류, 두 actual receipt 맥락, 해석 제한
- `cases.json`: 10행의 원래 상태와 evidence class, HOLD 사유, 승인·dispatch·retry 상태
- `provenance.json`: 공개 protocol 상대경로와 source/index/checkpoint/receipt/input SHA
- `manifest.json`: 공개 파일 목록, byte size, SHA-256

source와 index의 물리 경로, checkpoint 원문, provider 응답 원문, request messages, private metadata,
사용자 경로, PID, credential은 포함하지 않는다. protocol만 저장소의 공개 상대경로로 기록한다.

## 해석 제한

이 결과는 API 장애를 새로 일으키거나 장애 원인을 측정한 실험이 아니다. 원인 진단 정확도,
모델 효과, GPU busy 빈도, 의미적 blur 인식 성능도 측정하지 않았다. matrix의 busy와 timeout은 통제
입력이고, fixture transport는 actual backend 결과로 승격하지 않는다. 동시에 진행한 다른 live
call까지 포함한 global 원장 불변도 주장하지 않는다.

## 공개 결과 검증

```powershell
Set-Location "$HOME\rl-triage-agent"
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B evals/export_lineage_runtime_guards_v1.py verify `
  --out evals/results/lineage_runtime_guards_20261101_v1
```

검증기는 정확히 네 공개 파일만 허용하고 canonical JSON bytes, manifest SHA와 byte size, privacy
표식, 10행 순서, fixture 비승격을 확인한다.
