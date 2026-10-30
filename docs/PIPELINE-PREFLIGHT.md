# GPU Start 전 실행 계층 preflight

[`evals/agentic_pipeline_preflight_v1.py`](../evals/agentic_pipeline_preflight_v1.py)는 Remote10→Parent10(Parent6 prepare)·Service11 경로의 등록 입력을 GPU Start 전에 CPU에서 읽기 전용으로 점검한다. OPS-108~112처럼 유료 실행 뒤에야 드러난 실행 계층 실패를 먼저 찾는 것이 목적이다.

판정은 기존 생산 함수가 내린다. 이 모듈은 등록 입력을 그 함수들에 실행과 같은 순서로 넘기고 첫 실패 단계를 보고한다. 디렉터리를 만들지 않고, 실행·GPU Start 권한도 발급하지 않는다(`execution_permitted`, `gpu_start_authorized`는 항상 `false`).

## 단계와 생산 검증

| 단계 | 확인 내용 | 판정 함수 | 대응 기록 |
|---|---|---|---|
| `registration` | request·protocol·binding 참조 SHA, schema·fixture 명시, 공개 경로, Remote10 진입 closure, anchor blob | `U.read_ref`, `V1._public`, `U.verify_sources`, Remote8 `_prepare`와 같은 `REMOTE.SOURCES` 조건 | OPS-106 |
| `root_map` | operation별 GPU 관측 root 등록 | Parent10 `registered_observation_roots` | OPS-111 |
| `prepare_layout` | 준비 시점 root 존재, Parent10이 먼저 만드는 관측 root와 이미 있는 root가 독점 pair/unit claim 밖에 있는지, claim 대상이 비어 있는지 | `E.Resolver`(준비 시점 root만), Parent6·Parent1의 경로 규칙 | 필수 회귀 2·3 |
| `backend_args` | backend protocol 결합과 worker job의 파일 인자 | `B.validate_protocol`, `SafeRecoveryBackend.probe`가 만든 job(실행 직전에 가로챔), worker1 `validate_job` | OPS-108 |
| `lifecycle` | IO contract의 lifecycle 참조가 저장소 상대 경로·raw SHA·anchor blob인지 | Service6 `_native_lifecycle_reference` | OPS-112 |
| `budget` | contract·lifecycle 기한 일치, 기산점, 요청 작업+Stop reserve 여유 | Service9 `_validate_budget_origin` | OPS-105·107 |

`backend_args`는 `SafeRecoveryBackend`의 `_execute`만 job을 돌려주는 함수로 바꿔 stage 생성·child 실행 없이 실제 `probe`가 조립한 job을 얻는다. 작업 초는 `--work-seconds`, 없으면 binding의 `budget_gpu_s`, 둘 다 없으면 0이다.

## 실행

```bash
PYTHONUTF8=1 PYTHONPATH=src python evals/agentic_pipeline_preflight_v1.py --check \
  --request <parent request.json> --request-sha256 <sha256> \
  --contract <IO transport contract.json> --contract-sha256 <sha256> \
  --runtime-anchor <lifecycle blob이 있는 commit>
```

- request가 없으면 `--protocol`·`--protocol-sha256`·`--source-anchor`로 등록 단계만 본다. protocol의 `source_commit`은 protocol 파일보다 앞선 커밋이라 anchor로 쓰지 않는다.
- `--home`은 HOME 배치를 바꾼다. 실제 판은 layout repo가 실행 저장소 ROOT와 같아야 한다.
- fixture binding(`cpu_fixture`)은 `--allow-fixture`가 있어야 받는다.

출력은 한 줄 JSON이다. `status`는 `PASS`·`FAIL`·`PARTIAL`(일부 단계 입력 없음)·`UNSUPPORTED`(검증한 단계 없음)이고, `failed_stage`·`error_type`·`reason`·`inputs`·`scope`(단계별 verified/failed/unsupported/not_run)·`next`(다음 조건)를 담는다. 종료 코드는 PASS 0, FAIL 1, 그 밖 2다. 모르는 request·protocol schema는 실패가 아니라 `unsupported`로 보고한다.

## 2026-10-30 실제 등록 확인

공개 Parent 등록 [v17](../bench/protocols/agentic_runtime_text_s2221_pair_fresh_20261030_v17.json)을 등록 커밋 `21bc54e` anchor로 넣으면 `registration`에서 FAIL이다. closure에 Remote10·Parent10·Service11과 그 테스트 6개가 없다. Remote10 경로로 실행하면 Remote8 `_prepare`가 같은 조건으로 거부한다. 같은 anchor의 429개 source blob 검증 자체는 통과했다. 실제 binding·IO contract는 이 checkout에 없어 나머지 단계는 실제 입력으로 실행하지 않았다.

[테스트](../tests/test_agentic_pipeline_preflight_v1.py)는 `tmp_path` 안의 synthetic_cpu 등록을 쓴다. 코드 closure와 lifecycle은 실제 저장소 blob과 공개 [v47 lifecycle](../bench/protocols/agentic_runtime_text_s2221_20261030_v47_lifecycle.json)을 쓰고, checkpoint·manifest·backend protocol 본문은 합성값이다.

필수 회귀는 root-map 누락, 아직 없는 root의 준비 단계 접근, 관측 폴더 선생성과 `exist_ok=False` unit claim 충돌, 절대 lifecycle 참조, worker workspace 밖 backend 파일 인자, 기한·Stop reserve 불일치를 다룬다. 수정 fixture에서는 같은 scratch에서 Parent10 `ensure_observation_roots` → 준비 시점 Resolver → pair/unit claim → stage 생성 → plan root-map 재조회가 통과하며, 미래 stage root를 준비 시점 Resolver에 넣으면 거부된다. Fixture의 `brev_recovery_backend_v4.ROOT`는 "layout repo = 실행 저장소" 등식을 tmp에서 만들려고 tmp repo로 바꾼다. Windows 경로 길이 제한 때문에 fixture ID는 24자이며 Parent6의 64자 operation ID 형식 검사는 이 모듈 범위 밖이다. 이 PASS는 CPU 계약 점검으로, Parent6 prepare 전체(선택·자원·임대·authority), 원격 POSIX 실행, GPU 관측, 회복 결과를 대신하지 않는다.
