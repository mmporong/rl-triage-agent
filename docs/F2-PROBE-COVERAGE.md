# F2 P_torque probe coverage control

이 공개본은 `case-05`와 `case-06`의 F2 결함에 대해 `P_torque`가 실제 실행에서 어떤 outcome을 관측하는지 확인하는 사후 개발 대조다. 원 50행 정책 비교의 선택 성과가 아니며 진단 성공, 회복 또는 영상 성과에 합산하지 않는다.

## 판정

| 실제 outcome | 공개 coverage |
| --- | --- |
| `abnormal` | `confirmed_signature` |
| `normal` | `signature_not_observed` |
| `unknown` | `unknown_HOLD` |

`normal`은 F2 결함이 없다는 뜻이 아니다. 이 대조에서 `P_torque` signature가 관측되지 않았다는 뜻만 가진다. `unknown`은 재전송하지 않는다.

## 비용과 재사용

- 원 RI410 결과에 검증된 과거 `P_torque` 영수증이 있으면 그 영수증을 다시 검증하며 신규 SDK 비용은 0이다.
- 과거 영수증이 없는 target만 한 pair를 실행한다. 전체 신규 상한은 2 pair, 4 SDK role, API 0, wall 3600초다.
- 공개 `new_cost`는 신규 예약 SDK role, 실제 관측된 SDK role, 확인 가능한 owned wall만 분리해 기록한다. 과거 실행 비용을 신규 비용에 더하지 않는다.

## 공개 범위

`evals/export_f2_probe_coverage_v1.py`는 등록된 overlay와 private plan, 원 complete 50행 batch, source 결합, fresh pair 또는 historical receipt를 다시 검증한다. 공개 JSON에는 다음만 포함한다.

- registration anchor에 등록된 exporter source closure의 구성요소 이름과 LF SHA
- overlay, plan, 원 batch, pair result의 SHA
- 등록 commit anchor
- 두 target의 outcome, coverage, 실행 방식
- 신규 pair/SDK/API/확인 가능한 wall 합계
- 원 비교 불변과 모든 성과 credit 0 표식

checkpoint 경로, 입력 경로, raw receipt 경로, lease 식별자와 private 경로는 포함하지 않는다.

## CLI

실제 RI410 실행이 끝나고 필요한 파일이 등록된 뒤 사용한다.

```powershell
cd $HOME\rl-triage-agent
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'

& .venv\Scripts\python.exe -B evals\export_f2_probe_coverage_v1.py export `
  --result '<RI410-run>\result.json' `
  --protocol 'bench\protocols\<registered-overlay>.json' `
  --registration-anchor '<40-char-commit>' `
  --plan '<RI410-private-plan>.json' `
  --lab '<IsaacLab-root>' `
  --out 'docs\data\<f2-probe-coverage-export>.json'

& .venv\Scripts\python.exe -B evals\export_f2_probe_coverage_v1.py verify `
  --result '<RI410-run>\result.json' `
  --protocol 'bench\protocols\<registered-overlay>.json' `
  --registration-anchor '<40-char-commit>' `
  --plan '<RI410-private-plan>.json' `
  --lab '<IsaacLab-root>' `
  --export 'docs\data\<f2-probe-coverage-export>.json'
```

현재 문서는 exporter 계약만 설명한다. 원 R3 50행과 RI410 production 결과가 완료되기 전에는 공개 JSON을 생성하지 않는다.
