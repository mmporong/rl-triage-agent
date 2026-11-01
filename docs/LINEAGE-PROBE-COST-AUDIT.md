# Lineage probe 비용 감사

R3 five-policy 실행은 각 paired probe의 case와 reference 역할을 별도 owned SDK 작업으로 실행한다.
기존 R3 판정과 summary는 그대로 유지하며, 이 감사기는 raw receipt와 `owned-result.json`을 다시 묶어
역할별 wall time을 별도 후처리한다.

## 왜 별도 감사가 필요한가

runtime receipt verifier는 raw receipt의 `execution`과 owned 결과가 같은지 검사한다. 다만 verifier가
반환하는 projection에는 `execution`이 포함되지 않는다. paired host가 이 projection에서
`execution.wall_s`를 읽으면 완료된 역할도 unknown으로 기록될 수 있다. 기존 결과의
`owned_child_wall_s=null`, known 합계와 unknown 역할은 원문 필드로 보존한다. 보충 시간은
`timing_source=receipt_reconstruction`으로 분리한다.

## 검증 범위

완료된 10 case × 5 policy, 총 50행만 공개 export할 수 있다. 감사기는 다음을 확인한다.

- 기존 runner preflight의 protocol, execution config, anchor, 10개 source 결합
- 기존 summary `_grid`의 actual complete 50행과 result ref/SHA
- source-v2 bridge의 격리 `verify_pair`
- 각 step의 historical snapshot, public choice, ledger prefix, pair result ref/SHA
- case/reference plan, authorization request, checkpoint, claim, receipt SHA 결합
- raw receipt의 inline execution과 같은 role `owned-result.json`의 byte-level JSON 일치
- exit code 0, natural exit, timeout false, cleanup verified
- wall scope `parent_launch_to_exit_and_owned_job_empty_verification`

inline execution과 owned 결과가 다르거나 role·plan·request·checkpoint가 바뀌면 거부한다. receipt가
명시적으로 없으면 wall time은 `null`로 유지하며 0으로 채우지 않는다. 실제 완료 wall time이
`0.0`이면 known 값 0으로 보존해 missing과 구분한다.

## 집계

공개 결과는 50개 계획행을 모두 유지하고 다음 수치를 case별·policy별·전체로 나눈다.

- 중복 제거한 실제 SDK 역할 수
- receipt에서 복원한 known owned wall 합계와 unknown 역할 수
- paired probe parent wall 합계
- driver row parent wall 합계
- paired probe 0건인 정상 행 수

같은 receipt SHA가 여러 step이나 role에서 참조되면 최초 참조만 합산하고 중복 위치와 원인을 별도로
기록한다. paired probe 0건은 `no_probe_executed`이며 unknown receipt와 다르다.
pair result가 없더라도 한 역할 receipt가 완료된 interruption은 `orphan_role_cost`로 검증하고 전체
consumed SDK 역할 수와 known wall에 한 번만 합산한다. 이 값은 정책 진단 step으로 승격하지 않는다.
나머지 역할이 실행 승인 전에 멈췄다는 명시적 plan·폴더 inventory·오류 단계가 있을 때만
`not_executed`로 기록한다. pair result가 없다는 사실만으로 두 역할 모두 SDK 0이라고 추론하지 않으며,
실행한 역할의 missing wall을 0으로 채우지 않는다.

intent 회계는 reserved 역할, verified timed 역할, orphan 역할, 명시적 미실행 역할, 아직 해소되지 않은
reserved 역할을 나눈다. 동일 orphan receipt가 completed step이나 다른 interruption에 다시 나타나면
SHA 중복 제거가 적용된다.

SDK role wall은 GPU kernel 시간이나 GPU busy 시간이 아니다. parent launch부터 프로세스 종료와 owned
job 비어 있음 확인까지의 wall 범위다. 이 감사는 진단 효능, 모델 비교, 새 GPU 감소 효과를 측정하지
않는다.

## 완료 batch 공개 export

R3 `result.json`이 complete 50행이 된 뒤 다음처럼 실행한다. production CLI에는 fixture나 validator
우회 옵션이 없다.

```powershell
Set-Location "$HOME\rl-triage-agent"
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B evals/lineage_probe_cost_audit_v1.py export `
  --batch $BATCH_RESULT `
  --protocol bench/protocols/five_type_diagnosis_s2550_s2559_20261101_v1_five_policy.json `
  --config $EXECUTION_CONFIG `
  --anchor a8f6dbe8679e1f9b54a47c4d7db415b35fd8c55e `
  --out evals/results/lineage_probe_cost_audit_20261102_v1
```

현재 R3 batch가 진행 중이므로 이 공개 tag는 아직 만들지 않았다.
