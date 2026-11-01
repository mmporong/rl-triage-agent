# pyright: reportMissingImports=false
"""중단된 five-policy v1 배치를 원문 불변 상태에서 한 번 이어 실행한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import lineage_five_policy_driver_v1 as DRIVER
import lineage_five_policy_nim_transport_v1 as NIM
import lineage_five_policy_summary_v1 as SUMMARY
import lineage_gpu_admission_v1 as GPU
import lineage_lease_context_v2 as LEASE_V2
import lineage_source_batch_v1 as LEASE
import lineage_source_probe_inputs_v1 as HOST
import run_lineage_five_policy_v1 as BASE
from lease_timestamps import parse_lease_timestamp

B = DRIVER.BASE
PLAN_SCHEMA = "lineage_five_policy_continuation_plan_v1"
PROTOCOL_SCHEMA = "lineage_five_policy_continuation_protocol_v1"
PARTIAL_CASE = "case-04"
PARTIAL_ARM = "fixed_eligible"
SOURCE_FILES = (
    "bench/resume_lineage_five_policy_v1.py",
    "tests/test_resume_lineage_five_policy_v1.py",
    *LEASE_V2.SOURCE_FILES,
)
_LEASE = LEASE_V2


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True
    )
    return completed.stdout


def _lf_sha(path: Path) -> str:
    return hashlib.sha256(
        Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _ordered(protocol: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (case_id, arm)
        for case_id in protocol["case_order"]
        for arm in protocol["arm_order"][case_id]
    ]


def _comparison_status(unavailable_rows: int) -> str:
    B.require(type(unavailable_rows) is int and unavailable_rows >= 0,
              "input unavailable row 수")
    return ("inconclusive_input_unavailable" if unavailable_rows
            else "eligible_for_registered_summary")


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    base = Path(root).resolve(strict=True)
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError("previous run inventory에 symlink가 있다")
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        B.require(
            PurePosixPath(relative).as_posix() == relative
            and ".." not in PurePosixPath(relative).parts,
            "previous inventory 상대 경로",
        )
        rows[relative] = {"sha256": B.sha(path), "size_bytes": path.stat().st_size}
    return rows


def _inventory_digest(value: Mapping[str, Any]) -> str:
    return B.digest(dict(value))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        B.require(isinstance(value, dict), "ledger JSON object")
        rows.append(value)
    return rows


def _utc(value: object) -> datetime:
    B.require(isinstance(value, str), "실행 시간 원문")
    return parse_lease_timestamp(cast(str, value))


def _completed_row(path: Path, case_id: str, arm: str, *, allow_fixture: bool) -> dict:
    value = B.read(path)
    B.require(
        value.get("schema") == DRIVER.RESULT_SCHEMA
        and value.get("case_id") == case_id
        and value.get("arm") == arm
        and value.get("scope") == ("cpu_fixture" if allow_fixture else "actual_backend"),
        "완료 row identity/scope 불일치",
    )
    if not allow_fixture:
        SUMMARY._row_metrics(path, value, allow_fixture=False)
    return {
        "case_id": case_id,
        "arm": arm,
        "result": B.ref(path),
        "end_state": value.get("end_state"),
        "scoring": deepcopy(value.get("scoring")),
    }


def _partial_role_interruption(root: Path, second_spec: Mapping[str, Any]) -> dict[str, Any]:
    stage = root / "step02/probe/probe"
    authorization_path = stage / "authorization.json"
    authorization = B.read(authorization_path)
    reference = stage / "reference"
    case = stage / "case"
    reference_plan_path = reference / "plan.json"
    reference_receipt_path = reference / "probe_receipt.json"
    reference_owned_path = reference / "owned-result.json"
    reference_plan = HOST._PROBE._verify_plan(reference_plan_path)
    reference_receipt = B.read(reference_receipt_path)
    reference_owned = B.read(reference_owned_path)
    claim = reference_receipt.get("claim")
    B.require(
        authorization.get("schema") == HOST.AUTH_SCHEMA
        and authorization.get("request_spec", {}).get("probe") == second_spec.get("probe")
        and authorization.get("plan_sha256")
        == {"reference": B.sha(reference_plan_path), "case": B.sha(case / "plan.json")},
        "부분 row step02 authorization/plan 결합",
    )
    B.require(
        reference_plan.get("role") == "reference"
        and reference_plan.get("probe") == second_spec.get("probe")
        and reference_receipt.get("status") == "complete"
        and reference_receipt.get("role") == "reference"
        and reference_receipt.get("probe") == second_spec.get("probe")
        and reference_receipt.get("plan") == B.ref(reference_plan_path)
        and reference_receipt.get("authorization")
        == {**B.ref(authorization_path), "request_id": authorization.get("request_id")}
        and reference_receipt.get("execution") == reference_owned
        and reference_owned.get("exit_code") == 0
        and reference_owned.get("natural_exit") is True
        and reference_owned.get("cleanup_verified") is True,
        "부분 row step02 reference 실제 완료 증거",
    )
    with HOST._PROBE._backend_scope():
        measurement = HOST._PROBE.P1._legacy_measurement(reference_plan)
    expected_raw = {
        name: HOST._PROBE._file_ref(Path(path))
        for name, path in reference_plan["raw_files"].items()
        if Path(path).is_file()
    }
    expected_raw["owned_result"] = HOST._PROBE._file_ref(reference_owned_path)
    B.require(
        reference_receipt.get("measurement") == measurement
        and reference_receipt.get("raw_refs") == expected_raw,
        "부분 row step02 reference measurement/raw ref",
    )
    B.require(
        isinstance(claim, dict) and set(claim) == {"path", "sha256", "body"},
        "부분 row step02 reference claim 형식",
    )
    claim_path = Path(claim["path"]).resolve(strict=True)
    claim_body = B.read(claim_path)
    B.require(
        claim.get("sha256") == B.sha(claim_path)
        and claim.get("body") == claim_body
        and claim_body.get("role") == "reference"
        and claim_body.get("request_id") == authorization.get("request_id")
        and claim_body.get("plan_sha256") == B.sha(reference_plan_path),
        "부분 row step02 reference claim 결합",
    )
    case_plan_path = case / "plan.json"
    case_plan = HOST._PROBE._verify_plan(case_plan_path)
    B.require(
        case_plan.get("role") == "case"
        and case_plan.get("probe") == second_spec.get("probe")
        and {path.name for path in case.iterdir()} == {"jobs.json", "plan.json"}
        and not (case / "probe_receipt.json").exists()
        and not (case / "owned-result.json").exists()
        and not (case / "owned/launch.json").exists()
        and not any(
            path.name == "case.json"
            for path in (root / "step02/probe/backend/.lineage_probe_claims").rglob("*.json")
        ),
        "부분 row step02 case 미실행 경계",
    )
    return {
        "schema": "lineage_five_policy_partial_role_interruption_v1",
        "stage": "paired_probe_role_execution",
        "case_id": PARTIAL_CASE,
        "arm": PARTIAL_ARM,
        "step": 2,
        "probe": second_spec["probe"],
        "pair_result": None,
        "authorization": B.ref(authorization_path),
        "roles": {
            "reference": {
                "status": "complete",
                "plan": B.ref(reference_plan_path),
                "receipt": B.ref(reference_receipt_path),
                "owned_result": B.ref(reference_owned_path),
                "claim": B.ref(claim_path),
            },
            "case": {
                "status": "not_executed",
                "plan": B.ref(case_plan_path),
                "receipt": None,
                "owned_result": None,
                "claim": None,
                "reason": "lease_read_permission_error_before_case_execute_approval",
            },
        },
    }


def _partial(
    previous: Path,
    source: Mapping[str, Any],
    protocol_path: Path,
    *,
    allow_fixture: bool,
) -> tuple[dict[str, Any], float]:
    root = previous / PARTIAL_CASE / PARTIAL_ARM
    B.require(root.is_dir() and not (root / "result.json").exists(), "부분 row 경계")
    B.require(
        {path.name for path in root.iterdir()} == {"step01", "step02", "probes.ledger.jsonl"},
        "부분 row에 등록되지 않은 파일 또는 단계가 있다",
    )
    first_context = B.read(root / "step01/context.json")
    second_context = B.read(root / "step02/context.json")
    first_snapshot = first_context.get("snapshot")
    second_snapshot = second_context.get("snapshot")
    B.require(isinstance(first_snapshot, dict) and isinstance(second_snapshot, dict),
              "부분 row context snapshot")
    first_choice = B.read(root / "step01/choice.json")
    second_choice = B.read(root / "step02/choice.json")
    first_spec = B.read(root / "step01/request-spec.json")
    second_spec = B.read(root / "step02/request-spec.json")
    if not allow_fixture:
        ledger_rows = _jsonl(root / "probes.ledger.jsonl")

        def replay_decision(
            saved: dict, prefix: list[dict[str, Any]]
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            with tempfile.TemporaryDirectory(prefix="five-policy-partial-ledger-") as folder:
                ledger_path = Path(folder) / "probes.ledger.jsonl"
                ledger_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in prefix), encoding="utf-8"
                )
                inputs = {**saved["inputs"], "ledger_path": str(ledger_path)}
                fresh = DRIVER.C.V.build_snapshot(**inputs)
                normalized = deepcopy(fresh)
                normalized["inputs"]["ledger_path"] = saved["inputs"]["ledger_path"]
                B.require(normalized == saved, "부분 row snapshot/ledger prefix 불일치")
                choice = DRIVER.C.choose(fresh, PARTIAL_ARM)
                return choice, DRIVER.C.request_spec(fresh, choice)

        first_replay_choice, first_replay_spec = replay_decision(first_snapshot, [])
        second_replay_choice, second_replay_spec = replay_decision(
            second_snapshot, ledger_rows[:4]
        )
        B.require(first_replay_choice == first_choice,
                  "부분 row step01 choice 재계산 불일치")
        B.require(first_replay_spec == first_spec, "부분 row step01 spec 재계산 불일치")
        B.require(second_replay_choice == second_choice,
                  "부분 row step02 choice 재계산 불일치")
        B.require(second_replay_spec == second_spec, "부분 row step02 spec 재계산 불일치")
        original_result = B.read(root / "step01/probe/result.json")
        B.require(
            original_result.get("public_snapshot_sha256") == B.digest(first_snapshot)
            and original_result.get("public_choice") == first_choice
            and original_result.get("public_choice_sha256") == B.digest(first_choice),
            "부분 row step01 원래 HOST snapshot/choice 결합",
        )
        public_ledger = original_result.get("public_ledger", {})
        prefix = public_ledger.get("prefix", {})
        prefix_length = prefix.get("byte_length")
        original_ledger_path = Path(public_ledger.get("path", "")).resolve(strict=True)
        B.require(
            original_ledger_path == Path(first_snapshot["inputs"]["ledger_path"]).resolve()
            and type(prefix_length) is int
            and prefix_length >= 0,
            "부분 row step01 ledger prefix 결합",
        )
        raw_prefix = original_ledger_path.read_bytes()[:prefix_length]
        B.require(
            len(raw_prefix) == prefix_length
            and hashlib.sha256(raw_prefix).hexdigest() == prefix.get("sha256"),
            "부분 row step01 ledger prefix SHA",
        )
        with tempfile.TemporaryDirectory(prefix="five-policy-partial-host-") as folder:
            historical_ledger = Path(folder) / "probes.ledger.jsonl"
            historical_ledger.write_bytes(raw_prefix)
            historical_snapshot = deepcopy(first_snapshot)
            historical_snapshot["inputs"]["ledger_path"] = str(historical_ledger)
            historical_result = deepcopy(original_result)
            historical_result["public_ledger"]["path"] = str(historical_ledger)
            historical_result["public_snapshot_sha256"] = B.digest(historical_snapshot)
            historical_result_path = Path(folder) / "result.json"
            B.write(historical_result_path, historical_result)
            verified = HOST.verify_pair(
                historical_result_path, historical_snapshot, first_choice
            )
    else:
        verified = B.read(root / "step01/probe/result.json")
    B.require(
        verified.get("probe") == first_spec.get("probe")
        and verified.get("outcome") in {"normal", "abnormal", "unknown"},
        "부분 row step01 receipt 불일치",
    )
    observations = second_snapshot.get("view", {}).get("observations")
    B.require(
        observations == [{"probe": first_spec["probe"], "outcome": verified["outcome"]}],
        "부분 row step01 이후 snapshot 불일치",
    )
    busy = B.read(root / "step02/busy-gate.json")
    B.require(
        busy.get("decision") == "continue"
        and busy.get("dispatch_granted") is False
        and busy.get("automatic_retry") is False,
        "부분 row step02 busy gate 불일치",
    )
    ledger = _jsonl(root / "probes.ledger.jsonl")
    B.require(
        [row.get("event") for row in ledger]
        == ["proposed", "approved", "consumed", "receipt", "proposed", "approved", "consumed"],
        "부분 row ledger는 완료1+미완료1이어야 한다",
    )
    B.require(
        len({row["request_id"] for row in ledger[:4]}) == 1
        and len({row["request_id"] for row in ledger[4:]}) == 1
        and ledger[0]["request_id"] != ledger[4]["request_id"],
        "부분 row ledger request identity",
    )
    intent = B.read(previous / "paired-intents/0018.json")
    B.require(
        intent.get("case_id") == PARTIAL_CASE
        and intent.get("arm") == PARTIAL_ARM
        and intent.get("source_digest") == B.digest(dict(source))
        and intent.get("choice_digest") == B.digest(second_choice)
        and intent.get("snapshot_digest") == B.digest(second_snapshot)
        and intent.get("sdk_roles_reserved") == 2
        and intent.get("automatic_retry") is False,
        "부분 row paired intent 18 결합 불일치",
    )
    B.require(
        not (root / "step02/probe/result.json").exists(),
        "부분 row step02 pair 결과가 존재한다",
    )
    interruption = _partial_role_interruption(root, second_spec) if not allow_fixture else None
    parent_wall_s = (_utc(ledger[-1]["at"]) - _utc(ledger[0]["at"])).total_seconds()
    B.require(parent_wall_s >= 0, "부분 row 원문 시간이 뒤로 움직였다")
    qualification = source.get("qualification")
    B.require(isinstance(qualification, dict), "부분 row qualification")
    result = {
        "schema": DRIVER.RESULT_SCHEMA,
        "arm": PARTIAL_ARM,
        "case_id": PARTIAL_CASE,
        "run_id": "case_04_fixed_eligible",
        "scope": "cpu_fixture" if allow_fixture else "actual_backend",
        "selector_policy": PARTIAL_ARM,
        "host_storage_policy": DRIVER._storage_policy(PARTIAL_ARM),
        "host_storage_alias": PARTIAL_ARM != DRIVER._storage_policy(PARTIAL_ARM),
        "source_digest": B.digest(dict(source)),
        "source_index": deepcopy(source.get("index")),
        "checkpoint_bindings": deepcopy(source.get("packet", {}).get("input_bindings")),
        "steps": [{
            "step": 1,
            "selected_by": first_choice.get("selected_by"),
            "selector_policy": PARTIAL_ARM,
            "host_storage_policy": DRIVER._storage_policy(PARTIAL_ARM),
            "probe": first_spec["probe"],
            "outcome": verified["outcome"],
            "fact_ids": deepcopy(first_choice.get("fact_ids", [])),
            "probe_result": B.ref(root / "step01/probe/result.json"),
            "model_call": None,
            "before_view_digest": first_snapshot.get("view_digest"),
            "after_view_digest": second_snapshot.get("view_digest"),
            "owned_child_wall_s": verified.get("owned_child_wall_s"),
        }],
        "model_calls": 0,
        "native_normal_intervention_proposal": None,
        "hold_reason": "lease_metadata_read_failed",
        "input_unavailable": 0,
        "semantic_invalid": 0,
        "infra_unknown": 1,
        "provider_unknown": 0,
        "frame_fact_citations": sum(
            str(fact).startswith("vision:frame:") for fact in first_choice.get("fact_ids", [])
        ),
        "repair": False,
        "recovery": False,
        "heldout": False,
        "qualification": deepcopy(qualification),
        "end_state": "HOLD",
        "confirmed_subtype": None,
        "final_view_digest": second_snapshot.get("view_digest"),
        "remaining_candidates": deepcopy(second_snapshot.get("view", {}).get("remaining_candidates")),
        "scoring": DRIVER._score(qualification, None),
        "parent_wall_s": parent_wall_s,
        "parent_wall_basis": "original_ledger_first_proposal_to_unfinished_second_consumed",
        "parent_wall_is_observed_lower_bound": True,
        "parent_wall_measurement_complete": False,
        "parent_wall_comparable_for_policy_cost": False,
        "interruption": interruption,
    }
    return result, parent_wall_s


def prepare(
    protocol_path: Path,
    anchor: str,
    config_path: Path,
    previous: Path,
    stopped_lease: Path,
    out: Path,
    *,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    previous = Path(previous).resolve(strict=True)
    stopped_lease = Path(stopped_lease).resolve(strict=True)
    B.require(not (previous / "result.json").exists(), "이미 완결된 batch는 이어 실행할 수 없다")
    protocol, _config, sources = BASE.preflight(protocol_path, anchor, config_path)
    before = _inventory(previous)
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, str]] = []
    partial_result = None
    result_paths: set[str] = set()
    for case_id, arm in _ordered(protocol):
        arm_root = previous / case_id / arm
        result_path = arm_root / "result.json"
        if result_path.is_file():
            relative = result_path.relative_to(previous).as_posix()
            B.require(relative not in result_paths, "완료 result 중복")
            result_paths.add(relative)
            completed.append(_completed_row(result_path, case_id, arm, allow_fixture=allow_fixture))
        elif (case_id, arm) == (PARTIAL_CASE, PARTIAL_ARM):
            partial_result, _partial_wall = _partial(
                previous, sources[case_id]["source"], protocol_path,
                allow_fixture=allow_fixture,
            )
        else:
            B.require(not arm_root.exists(), "미시작 row에 알 수 없는 부분 출력이 있다")
            pending.append({"case_id": case_id, "arm": arm})
    B.require(
        len(completed) == 18 and partial_result is not None and len(pending) == 31,
        "continuation 경계는 완료18+부분1+미시작31이어야 한다",
    )
    failure = B.read(previous / "failure.json")
    B.require(
        failure.get("error_type") == "PermissionError"
        and failure.get("automatic_retry") is False,
        "원래 failure가 lease metadata read 실패가 아니다",
    )
    old_transport = NIM.status(previous / "nim")
    old_paired = len(list((previous / "paired-intents").glob("*.json")))
    B.require(old_transport["reservations_charged"] == 7 and old_paired == 18,
              "원래 API/paired 예약 수가 다르다")
    lease_initial = B.read(previous / "lease-initial.json")
    stopped = B.read(stopped_lease)
    B.require(
        stopped.get("leaseId") == lease_initial.get("leaseId")
        and stopped.get("owner") == lease_initial.get("owner")
        and stopped.get("wrapperPid") == lease_initial.get("wrapperPid")
        and stopped.get("childPid") == lease_initial.get("childPid")
        and stopped.get("acquiredUtc") == lease_initial.get("acquiredUtc")
        and stopped.get("state") == "failed"
        and stopped.get("exitCode") == 1
        and isinstance(stopped.get("completedUtc"), str),
        "동결 stopped lease가 원래 failed lease와 다르다",
    )
    old_wall = (_utc(stopped["completedUtc"]) - _utc(lease_initial["acquiredUtc"])).total_seconds()
    B.require(old_wall >= 0, "원래 lease 완료 시간이 뒤로 움직였다")
    after = _inventory(previous)
    B.require(before == after, "prepare 검증 중 previous run bytes가 바뀌었다")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "prepared",
        "original_protocol": B.ref(protocol_path),
        "original_anchor": anchor,
        "execution_config": B.ref(config_path),
        "previous_run": str(previous),
        "previous_inventory": after,
        "previous_inventory_sha256": _inventory_digest(after),
        "previous_failure": B.ref(previous / "failure.json"),
        "previous_lease_initial": B.ref(previous / "lease-initial.json"),
        "previous_stopped_lease": B.ref(stopped_lease),
        "completed_rows": completed,
        "partial_identity": {"case_id": PARTIAL_CASE, "arm": PARTIAL_ARM},
        "partial_result": partial_result,
        "pending_rows": pending,
        "old_model_reservations": old_transport["reservations_charged"],
        "old_paired_intents": old_paired,
        "old_sdk_roles_reserved": old_paired * 2,
        "old_sdk_roles_executed": (old_paired - 1) * 2 + 1,
        "old_parent_wall_s": old_wall,
        "old_parent_wall_basis": "immutable_failed_lease_acquiredUtc_to_completedUtc",
        "caps": {"model_reservations": 72, "paired_intents": 150,
                 "sdk_roles": 300, "parent_wall_s": 144000},
        "automatic_retry": False,
        "requests_replayed": 0,
        "gpu_executions": 0,
        "model_api_calls": 0,
    }
    B.write(Path(out).resolve(), plan)
    return deepcopy(plan)


def continuation_protocol_value(plan: Mapping[str, Any]) -> dict[str, Any]:
    B.require(plan.get("schema") == PLAN_SCHEMA, "continuation plan schema")
    return {
        "schema": PROTOCOL_SCHEMA,
        "original_protocol_sha256": plan["original_protocol"]["sha256"],
        "original_anchor": plan["original_anchor"],
        "execution_config_sha256": plan["execution_config"]["sha256"],
        "stopped_lease_sha256": plan["previous_stopped_lease"]["sha256"],
        "previous_inventory_sha256": plan["previous_inventory_sha256"],
        "prepared_plan_digest": B.digest(dict(plan)),
        "completed_rows": 18,
        "partial_rows": 1,
        "pending_rows": 31,
        "old_model_reservations": 7,
        "old_paired_intents": 18,
        "old_sdk_roles_executed": 35,
        "old_sdk_roles_reserved": 36,
        "caps": deepcopy(plan["caps"]),
        "sources_lf": {name: _lf_sha(ROOT / name) for name in SOURCE_FILES},
        "interrupted_policy": "seal_technical_hold_without_step2_retry",
        "automatic_infra_retry": 0,
        "request_replay": 0,
    }


def _verify_continuation_protocol(
    path: Path, anchor: str, plan: Mapping[str, Any], *, allow_fixture: bool
) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    value = B.read(path)
    expected = continuation_protocol_value(plan)
    B.require(value == expected, "public continuation protocol이 prepared plan과 다르다")
    if allow_fixture:
        return value
    B.require(_hex(anchor, 40), "continuation anchor 형식")
    relative = path.relative_to(ROOT).as_posix()
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    for name, expected_sha in {relative: _lf_sha(path), **value["sources_lf"]}.items():
        current = ROOT / name
        B.require(current.is_file() and _lf_sha(current) == expected_sha,
                  "continuation source SHA 불일치: " + name)
        committed = _git("show", f"{anchor}:{name}")
        normalized = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        B.require(hashlib.sha256(normalized).hexdigest() == expected_sha,
                  "continuation anchor source 불일치: " + name)
    return value


def _verify_plan(plan_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    plan_path = Path(plan_path).resolve(strict=True)
    plan = B.read(plan_path)
    B.require(plan.get("schema") == PLAN_SCHEMA and plan.get("status") == "prepared",
              "prepared plan schema/status")
    protocol_path = B._bound_file(plan["original_protocol"])
    config_path = B._bound_file(plan["execution_config"])
    previous = Path(plan["previous_run"]).resolve(strict=True)
    current = _inventory(previous)
    B.require(
        current == plan["previous_inventory"]
        and _inventory_digest(current) == plan["previous_inventory_sha256"],
        "previous run inventory가 prepare 뒤 바뀌었다",
    )
    B._bound_file(plan["previous_failure"])
    B._bound_file(plan["previous_lease_initial"])
    B._bound_file(plan["previous_stopped_lease"])
    return plan, protocol_path, config_path, previous


def execute(
    plan_path: Path,
    continuation_protocol: Path,
    continuation_anchor: str,
    output: Path,
    *,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    plan, protocol_path, config_path, previous = _verify_plan(plan_path)
    _verify_continuation_protocol(
        continuation_protocol, continuation_anchor, plan, allow_fixture=allow_fixture
    )
    protocol, config, sources = BASE.preflight(
        protocol_path, plan["original_anchor"], config_path
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    new_paired = 0
    first_lease = _LEASE.lease_context(evidence_dir=None)
    old_lease = B.read(previous / "lease-initial.json")
    B.require(
        first_lease["owner"] == LEASE.OWNER
        and first_lease["leaseId"] != old_lease["leaseId"],
        "continuation은 같은 owner의 새 lease여야 한다",
    )
    B.write(output / "lease-initial.json", first_lease)
    NIM.initialize(
        output / "nim", protocol_path=protocol_path,
        execution_anchor=plan["original_anchor"], api_cap=72,
    )
    partial_path = output / PARTIAL_CASE / PARTIAL_ARM / "result.json"
    B.write(partial_path, plan["partial_result"])
    completed_by_key = {
        (row["case_id"], row["arm"]): row for row in plan["completed_rows"]
    }
    pending = {(row["case_id"], row["arm"]) for row in plan["pending_rows"]}
    results = []

    def context(_request=None):
        elapsed = time.monotonic() - started
        B.require(plan["old_parent_wall_s"] + elapsed < plan["caps"]["parent_wall_s"],
                  "aggregate diagnosis wall cap")
        now = _LEASE.lease_context(evidence_dir=None)
        B.require(now["leaseId"] == first_lease["leaseId"] and now["owner"] == first_lease["owner"],
                  "continuation lease identity changed")
        B.require(B.sha(config_path) == plan["execution_config"]["sha256"],
                  "execution config changed during continuation")
        for name, expected_sha in protocol["sources_lf"].items():
            B.require(B.sha(ROOT / name, lf=True) == expected_sha,
                      "original registered source changed: " + name)
        return {"approved": True}

    def busy(operation_id, _intent):
        context()
        stage = output / "gpu-observations" / operation_id
        try:
            GPU.snapshot(stage)
        except RuntimeError:
            receipt = B.read(stage / "gpu_before.json")
            B.require(receipt.get("bounded_resample", {}).get("accepted") is False,
                      "GPU exception without bound rejection evidence")
            return {"status": "busy", "observation_id": B.sha(stage / "gpu_before.json")}
        return {"status": "available", "observation_id": B.sha(stage / "gpu_before.json")}

    for case_id, arm in _ordered(protocol):
        key = (case_id, arm)
        if key in completed_by_key:
            results.append(deepcopy(completed_by_key[key]))
            continue
        if key == (PARTIAL_CASE, PARTIAL_ARM):
            value = plan["partial_result"]
            results.append({"case_id": case_id, "arm": arm, "result": B.ref(partial_path),
                            "end_state": value["end_state"], "scoring": deepcopy(value["scoring"])})
            continue
        B.require(key in pending, "실행 대상이 prepared pending 집합과 다르다")
        item = sources[case_id]
        source = item["source"]
        if source is None:
            value = BASE._missing_source_result(
                case_id, arm, protocol["source_digests"][case_id], config["source_index"],
                item["input_hold"], output / case_id / arm,
            )
        else:
            def dispatch(snapshot, choice, *, out, _source=source, _item=item,
                         _case=case_id, _arm=arm):
                nonlocal new_paired
                context()
                B.require(plan["old_paired_intents"] + new_paired < plan["caps"]["paired_intents"],
                          "aggregate paired probe cap")
                B.require(plan["old_sdk_roles_reserved"] + (new_paired + 1) * 2
                          <= plan["caps"]["sdk_roles"], "aggregate SDK role cap")
                new_paired += 1
                B.write(output / "paired-intents" / f"{new_paired:04d}.json", {
                    "case_id": _case, "arm": _arm, "source_digest": B.digest(_source),
                    "choice_digest": B.digest(choice), "snapshot_digest": B.digest(snapshot),
                    "sdk_roles_reserved": 2, "automatic_retry": False,
                })
                return HOST.execute_pair(
                    snapshot, choice, backend_packet=_source["backend_packet"],
                    case_input=_item["inputs"]["case"], reference_input=_item["inputs"]["reference"],
                    probe_protocol=ROOT / protocol["tool_protocols"]["paired_probe"]["path"],
                    view_protocol=ROOT / protocol["tool_protocols"]["view"]["path"],
                    anchor=plan["original_anchor"], lab=Path(config["lab"]), out=out,
                    verify_context=context,
                )

            def complete(request):
                context()
                new_status = NIM.status(output / "nim")
                B.require(plan["old_model_reservations"] + new_status["reservations_charged"]
                          < plan["caps"]["model_reservations"], "aggregate model reservation cap")
                return NIM.complete(
                    output / "nim", request,
                    send=lambda body, call: BASE._WORKER._worker_send(
                        output / "nim", protocol_path, plan["original_anchor"], body, call
                    ),
                )

            value = DRIVER.run_arm(
                source, arm=arm, run_id=case_id.replace("-", "_") + "_" + arm,
                output=output / case_id / arm, protocol_path=protocol_path,
                execution_anchor=plan["original_anchor"],
                complete=complete if arm in DRIVER.MODEL_ARMS else None,
                execute_probe=dispatch, verify_probe=HOST.verify_pair, verify_call=NIM.verify_call,
                busy_reader=busy, registered_input_roots=item["registered_input_roots"],
                allow_fixture=allow_fixture,
            )
        results.append({"case_id": case_id, "arm": arm,
                        "result": B.ref(output / case_id / arm / "result.json"),
                        "end_state": value["end_state"], "scoring": deepcopy(value["scoring"])})
    B.require(len(results) == 50, "continuation final batch는 50행이어야 한다")
    new_transport = NIM.status(output / "nim")
    elapsed = time.monotonic() - started
    unavailable_rows = sum(row["scoring"] is None for row in results)
    result = {
        "schema": "lineage_five_policy_batch_result_v1",
        "status": "complete",
        "execution_protocol": B.ref(protocol_path),
        "execution_anchor": plan["original_anchor"],
        "execution_config": B.ref(config_path),
        "planned_cases": 10,
        "planned_rows": 50,
        "rows": results,
        "input_unavailable_rows": unavailable_rows,
        "comparison_status": _comparison_status(unavailable_rows),
        "paired_intents": plan["old_paired_intents"] + new_paired,
        "sdk_roles_reserved": plan["old_sdk_roles_reserved"] + new_paired * 2,
        "sdk_roles_executed": plan["old_sdk_roles_executed"] + new_paired * 2,
        "transport": {
            "old_reservations": plan["old_model_reservations"],
            "new": new_transport,
            "aggregate_reservations": plan["old_model_reservations"]
            + new_transport["reservations_charged"],
            "cap": plan["caps"]["model_reservations"],
        },
        "parent_wall_s": plan["old_parent_wall_s"] + elapsed,
        "lease_id": first_lease["leaseId"],
        "repair": False,
        "recovery": False,
        "heldout": False,
        "model_version_gain": False,
        "continuation": {
            "schema": "lineage_five_policy_continuation_provenance_v1",
            "plan": B.ref(Path(plan_path).resolve()),
            "public_protocol": B.ref(Path(continuation_protocol).resolve()),
            "continuation_anchor": continuation_anchor,
            "previous_failure": deepcopy(plan["previous_failure"]),
            "previous_inventory_sha256": plan["previous_inventory_sha256"],
            "new_lease": B.ref(output / "lease-initial.json"),
            "completed_rows_reused": 18,
            "partial_rows_sealed": 1,
            "pending_rows_executed": 31,
            "partial_step2_retried": False,
            "partial_orphan_reference_roles": 1,
            "requests_replayed": 0,
            "automatic_infra_retry": 0,
            "parent_wall": {"old_s": plan["old_parent_wall_s"], "new_s": elapsed,
                            "aggregate_s": plan["old_parent_wall_s"] + elapsed,
                            "old_basis": plan["old_parent_wall_basis"]},
            "policy_cost_comparison_complete": False,
            "policy_cost_limitation": "interrupted_row_parent_wall_is_lower_bound_and_mixed_scope",
            "missing_measurement_filled_with_zero": False,
            "caps": deepcopy(plan["caps"]),
        },
    }
    B.write(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--protocol", type=Path, required=True)
    prep.add_argument("--anchor", required=True)
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--previous", type=Path, required=True)
    prep.add_argument("--stopped-lease", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--continuation-protocol", type=Path, required=True)
    run.add_argument("--continuation-anchor", required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.protocol, args.anchor, args.config, args.previous,
                    args.stopped_lease, args.out)
        else:
            execute(args.plan, args.continuation_protocol, args.continuation_anchor, args.out)
    except Exception as error:
        target = getattr(args, "out", None)
        if target is not None:
            target_path = Path(target).resolve()
            failure_path = target_path / "failure.json" if target_path.is_dir() else target_path.with_suffix(".failure.json")
            if not failure_path.exists():
                B.write(failure_path, {"status": "invalid", "error_type": type(error).__name__,
                                       "message": str(error), "automatic_retry": False})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
