# pyright: reportArgumentType=false, reportMissingImports=false
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""다섯 정책 actual 진단에서 첫 등록 교정 후보를 검증하고 승인 뒤 실행한다."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_evaluation_gate_v1 as GATE
import lineage_five_policy_driver_v1 as DRIVER
import lineage_five_policy_nim_transport_v1 as TRANSPORT
import lineage_recovery_execution_v1 as OLD_RECOVERY
import lineage_source_batch_v2 as SOURCE_BATCH
import lineage_source_probe_inputs_v1 as SOURCE_INPUT
import run_lineage_five_policy_v1 as RUNNER
import runtime_observation_training_v1 as TRAINING

SCHEMA = "five_policy_recovery_candidate_v1"
PROTOCOL_SCHEMA = "five_policy_recovery_protocol_v1"
APPROVAL_SCHEMA = "five_policy_recovery_approval_v1"
RECEIPT_SCHEMA = "five_policy_recovery_receipt_v1"
RESULT_SCHEMA = "five_policy_recovery_result_v1"
EPOCH2_SCHEMA = "five_policy_recovery_epoch2_input_v1"
CONDITIONAL_DESIGN_RELATIVE = (
    "bench/protocols/five_policy_recovery_conditional_s2552_s2556_20261101_v1.json"
)
TARGETS = {
    ("case-03", "fixed_eligible"): "F1",
    ("case-03", "logs_only"): "F1",
    ("case-07", "fixed_eligible"): "COMPOUND",
    ("case-07", "logs_only"): "COMPOUND",
}
CORRECTIONS = {
    "reward": "restore_body_frame_reward",
    "optimizer_update_count": "restore_configured_ppo_epochs",
}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_FILES = tuple(sorted({
    *DRIVER.SOURCE_FILES, *SOURCE_INPUT.SOURCE_FILES,
    *TRAINING.TRAIN.BOUND_SOURCES, *TRAINING.EXEC2.SOURCE_FILES,
    "bench/run_five_policy_recovery_loop_v1.py",
    "tests/test_run_five_policy_recovery_loop_v1.py",
    "evals/runtime_observation_training_v1.py",
    "tests/test_runtime_observation_training_v1.py",
    "evals/five_policy_recovery_candidate_v1.py",
    "tests/test_five_policy_recovery_candidate_v1.py",
    "evals/export_five_policy_recovery_v1.py",
    "tests/test_export_five_policy_recovery_v1.py",
}))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"비유한 JSON 숫자: {token}")))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    require(not path.exists(), f"{path.name}은 fresh 파일이어야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": sha(path)}


def bound(value: object, name: str) -> Path:
    require(isinstance(value, dict) and set(value) >= {"path", "sha256"}, f"{name} ref 형식")
    item = cast(dict[str, Any], value)
    path = Path(str(item["path"])).resolve(strict=True)
    require(path.is_file() and HEX64.fullmatch(str(item["sha256"])) is not None
            and sha(path) == item["sha256"], f"{name} SHA가 바뀌었다")
    return path


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("protocol component는 저장소 내부여야 한다") from error


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git anchor/blob 결합을 확인하지 못했다")
    return result.stdout


def conditional_design(path: Path, *, first_seeds: Mapping[str, int] | None = None,
                       second_seeds: Mapping[str, int] | None = None) -> dict[str, Any]:
    """첫 결과 전에 고정한 두 epoch의 seed·분기·상한을 함께 검증한다."""
    path = Path(path).resolve(strict=True)
    relative = _relative(path)
    require(relative.startswith("bench/protocols/"), "conditional design 공개 경로")
    value = read(path)
    keys = {f"{case}:{policy}" for case, policy in TARGETS}
    first, second = value.get("first_seeds"), value.get("second_seeds")
    require(value.get("schema") == "five_policy_recovery_conditional_design_v1"
            and value.get("status") == "preregistered_conditional_design"
            and isinstance(first, dict) and isinstance(second, dict)
            and set(first) == keys and set(second) == keys
            and all(type(seed) is int and seed >= 0 for seed in [*first.values(), *second.values()])
            and len({*first.values(), *second.values()}) == 8,
            "conditional design 두 epoch seed 계약")
    require(value.get("first_gpu_phase") == {"sdk_cap": 8, "api_cap": 0, "wall_cap_s": 14400}
            and value.get("second_gpu_phase") == {"sdk_cap": 32, "api_cap": 32, "wall_cap_s": 43200}
            and value.get("second_requires_valid_not_recovered_receipt") is True
            and value.get("source_case_denominator") == 2
            and value.get("pipeline_denominator") == 4
            and value.get("training_mode") == "fresh_restart_300"
            and value.get("max_distinct_corrections") == 2
            and value.get("case_replacement") is False and value.get("epoch3") is False
            and value.get("model_result_dependent_seed_selection") is False
            and value.get("execution_freshness_required") is True
            and value.get("execution_authorized") is False,
            "conditional design 분기·실행 한도")
    study = value.get("study_design")
    require(isinstance(study, dict) and set(study) == {"path", "sha256_lf"},
            "conditional design study ref")
    study_path = TRANSPORT.BASE._source_path(ROOT, study["path"])
    require(sha(study_path, lf=True) == study["sha256_lf"], "conditional design study SHA")
    if first_seeds is not None:
        require(dict(first_seeds) == first, "first seed가 conditional design과 다르다")
    if second_seeds is not None:
        require(dict(second_seeds) == second, "second seed가 conditional design과 다르다")
    return deepcopy(value)


def component_conditional_design(component: Mapping[str, Any], *,
                                 second_seeds: Mapping[str, int] | None = None) -> tuple[Path, dict[str, Any]]:
    """첫 교정 component가 결합한 조건부 설계를 다음 epoch에서도 소비한다."""
    components = component.get("components")
    require(isinstance(components, dict) and isinstance(component.get("recovery_seeds"), dict),
            "first component seed·components 계약")
    reference = components.get("conditional_design")
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256_lf"},
            "first component conditional design ref")
    path = TRANSPORT.BASE._source_path(ROOT, reference["path"])
    require(sha(path, lf=True) == reference["sha256_lf"], "conditional design SHA가 바뀌었다")
    value = conditional_design(path, first_seeds=component.get("recovery_seeds"),
                               second_seeds=second_seeds)
    return path, value


def prepare_protocol(five_policy_protocol: Path, source_protocol: Path,
                     recovery_seeds: Mapping[str, int],
                     fixed_protocol: Path = OLD_RECOVERY.FIXED_PROTOCOL, *,
                     conditional_design_path: Path | None = None) -> dict[str, Any]:
    """R3/source/fixed 계약과 네 pipeline seed를 공개 recovery protocol로 만든다."""
    five_path = Path(five_policy_protocol).resolve(strict=True)
    source_path = Path(source_protocol).resolve(strict=True)
    fixed_path = Path(fixed_protocol).resolve(strict=True)
    five = read(five_path)
    source = read(source_path)
    require(five.get("schema") == DRIVER.SCHEMA, "R3 five-policy protocol schema가 다르다")
    require(source.get("schema") == SOURCE_BATCH.SCHEMA, "source-v2 protocol schema가 다르다")
    require(fixed_path == OLD_RECOVERY.FIXED_PROTOCOL.resolve(), "동결 fixed26 protocol만 허용한다")
    expected_keys = {f"{case}:{policy}" for case, policy in TARGETS}
    require(set(recovery_seeds) == expected_keys
            and all(type(seed) is int and seed >= 0 for seed in recovery_seeds.values()),
            "네 pipeline의 등록 recovery seed가 필요하다")
    design_path = (ROOT / CONDITIONAL_DESIGN_RELATIVE if conditional_design_path is None
                   else Path(conditional_design_path).resolve(strict=True))
    design = conditional_design(design_path, first_seeds=recovery_seeds)
    require(five.get("study_design") == design["study_design"],
            "R3와 conditional design study가 다르다")
    tools = five.get("tool_protocols")
    require(isinstance(tools, dict) and "paired_probe" in tools, "R3 paired probe protocol이 없다")
    paired_wrapper = ROOT / tools["paired_probe"]["path"]
    wrapper = read(paired_wrapper)
    component = wrapper.get("component_probe_protocol")
    require(isinstance(component, dict), "paired probe component가 없다")
    base_probe = ROOT / component["path"]
    base = read(base_probe)
    runtime = base.get("runtime_source_sha256_lf")
    require(isinstance(runtime, dict) and set(runtime) == SOURCE_INPUT.BASE_PROBE.RUNTIME_PINS,
            "recovery runtime source pin은 기존 9개여야 한다")
    components = {
        "conditional_design": {"path": _relative(design_path), "sha256_lf": sha(design_path, lf=True)},
        "five_policy": {"path": _relative(five_path), "sha256_lf": sha(five_path, lf=True)},
        "source_collection": {"path": _relative(source_path), "sha256_lf": sha(source_path, lf=True)},
        "fixed_evaluation": {"path": _relative(fixed_path), "sha256_lf": sha(fixed_path, lf=True)},
        "paired_probe": {"path": _relative(paired_wrapper), "sha256_lf": sha(paired_wrapper, lf=True)},
        "base_probe": {"path": _relative(base_probe), "sha256_lf": sha(base_probe, lf=True)},
        "bands": {"path": _relative(OLD_RECOVERY.BANDS), "sha256": sha(OLD_RECOVERY.BANDS)},
    }
    source_names = set(SOURCE_FILES) | set(TRAINING.TRAIN.BOUND_SOURCES)
    return {
        "schema": PROTOCOL_SCHEMA, "phase": "first_registered_correction_only",
        "targets": [{"case_id": case, "policy": policy, "source_arm": arm}
                    for (case, policy), arm in TARGETS.items()],
        "recovery_seeds": dict(sorted(recovery_seeds.items())),
        "corrections": deepcopy(CORRECTIONS),
        "training": {"arm": "FIRST", "mode": "fresh_restart", "updates": 300,
                     "source_checkpoint": "provenance_only", "runner_load_calls": 0},
        "evaluation": {"conditions": 26, "first_recovered_stops": True,
                       "not_recovered_requires_new_epoch2_protocol": True},
        "components": components, "runtime_source_sha256_lf": deepcopy(runtime),
        "source_sha256_lf": {name: sha(ROOT / name, lf=True) for name in sorted(source_names)},
        "authority": {"candidate_approval": False, "training": False, "gpu": False,
                      "second_epoch": False},
    }


def verify_protocol(path: Path, *, anchor: str, lab: Path) -> dict[str, Any]:
    path, lab = Path(path).resolve(strict=True), Path(lab).resolve(strict=True)
    require(HEX40.fullmatch(anchor) is not None, "recovery anchor 형식")
    value = read(path)
    components = value.get("components")
    require(isinstance(components, dict), "recovery components가 없다")
    expected = prepare_protocol(
        ROOT / components["five_policy"]["path"],
        ROOT / components["source_collection"]["path"],
        value.get("recovery_seeds", {}), ROOT / components["fixed_evaluation"]["path"],
        conditional_design_path=ROOT / components["conditional_design"]["path"])
    require(value == expected, "recovery protocol 내용이 현재 source와 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bindings = {_relative(path): sha(path, lf=True), **value["source_sha256_lf"],
                **{row["path"]: row["sha256_lf"] for row in components.values()
                   if "sha256_lf" in row}}
    for name, expected_sha in bindings.items():
        current = ROOT / name
        require(current.is_file() and sha(current, lf=True) == expected_sha,
                "현재 recovery source/component가 바뀌었다: " + name)
        blob = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        require(hashlib.sha256(blob).hexdigest() == expected_sha,
                "anchor recovery source/component가 바뀌었다: " + name)
    for name, expected_sha in value["runtime_source_sha256_lf"].items():
        require((lab / name).is_file() and sha(lab / name, lf=True) == expected_sha,
                "Isaac Lab runtime source가 바뀌었다: " + name)
    return deepcopy(value)


def _ledger_events(raw: bytes) -> list[dict[str, Any]]:
    events = []
    for line in raw.decode("utf-8").splitlines():
        value = json.loads(line)
        require(isinstance(value, dict), "ledger prefix event 형식")
        events.append(value)
    return events


def _rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proposed = {row["request_id"]: row for row in events if row.get("event") == "proposed"}
    rows = []
    for event in events:
        if event.get("event") != "receipt":
            continue
        request = proposed.get(event.get("request_id"))
        require(isinstance(request, dict), "ledger receipt의 proposed event가 없다")
        rows.append({"probe": request["probe"], "outcome": event["outcome"],
                     "measurement": deepcopy(event["measurement"]), "gpu_s": event["gpu_s"]})
    return rows


def _snapshot(packet: dict[str, Any], result: dict[str, Any], ledger_path: Path,
              events: list[dict[str, Any]]) -> dict[str, Any]:
    policy = result["host_storage_policy"]
    protocol_sha = DRIVER.C.V.file_digest(DRIVER.C.V.PROTOCOL)
    binding = DRIVER.C.V.epoch_binding(packet, result["run_id"], policy, protocol_sha, None)
    rows = _rows(events)
    view = DRIVER.C.V._view(packet, rows, None, False)
    pre = {"binding": binding, "prior_observations": rows[:-1],
           "executed_probe": rows[-1]["probe"] if rows else None}
    inputs = {"packet": packet, "run_id": result["run_id"], "policy": policy,
              "protocol_sha256": protocol_sha, "ledger_path": str(ledger_path),
              "parent": None, "masked": False}
    return {"schema": "selection_snapshot_v1", "inputs": inputs, "binding": binding,
            "ledger_digest": DRIVER.C.L.digest(events),
            "pre_receipt_anchor": DRIVER.C.L.digest(pre), "view": view,
            "view_digest": DRIVER.C.L.digest(view), "observation_masked_control": False,
            "proposal_allowed": view["status"] == "open", "execution_authorized": False,
            "approval_granted": False}


def _verify_steps(result_path: Path, result: dict[str, Any], source: dict[str, Any],
                  *, host_verify: Callable = SOURCE_INPUT.verify_pair,
                  call_verify: Callable = TRANSPORT.verify_call) -> dict[str, Any]:
    root = result_path.parent
    ledger_path = root / "probes.ledger.jsonl"
    steps = result.get("steps")
    require(isinstance(steps, list), "R3 result steps 형식")
    if not steps:
        require(not ledger_path.exists() or ledger_path.read_bytes() == b"", "step 없는 행의 ledger가 비지 않았다")
        return {"steps": [], "ledger": None, "final_snapshot": None}
    require(ledger_path.is_file(), "R3 public ledger가 없다")
    ledger_raw = ledger_path.read_bytes()
    prior_length = 0
    evidence = []
    for index, step in enumerate(steps, 1):
        require(step.get("step") == index, "R3 step 순서")
        directory = root / f"step{index:02d}"
        context_path, choice_path = directory / "context.json", directory / "choice.json"
        context, choice = read(context_path), read(choice_path)
        probe_result_path = bound(step.get("probe_result"), "paired probe result")
        host_result = read(probe_result_path)
        public_ledger = host_result.get("public_ledger")
        require(isinstance(public_ledger, dict)
                and Path(public_ledger.get("path", "")).resolve() == ledger_path,
                "host result public ledger 경로")
        prefix = public_ledger.get("prefix")
        require(isinstance(prefix, dict) and type(prefix.get("byte_length")) is int,
                "host result ledger prefix")
        length = prefix["byte_length"]
        require(prior_length < length <= len(ledger_raw)
                and hashlib.sha256(ledger_raw[:length]).hexdigest() == prefix.get("sha256"),
                "host result ledger prefix SHA")
        prior_events = _ledger_events(ledger_raw[:prior_length])
        snapshot = _snapshot(source["packet"], result, ledger_path, prior_events)
        require(context.get("snapshot") == snapshot and host_result.get("public_choice") == choice
                and host_result.get("public_snapshot_sha256") == DRIVER.BASE.digest(snapshot),
                "R3 context/choice/historical snapshot 결합")
        verified = host_verify(probe_result_path, snapshot, choice)
        require(verified.get("outcome") == step.get("outcome")
                and verified.get("probe") == step.get("probe"),
                "R3 paired receipt 재검증 결과")
        model_ref = step.get("model_call")
        if result["arm"] in DRIVER.MODEL_ARMS:
            call_path = Path(str(cast(dict[str, Any], model_ref)["path"])).resolve(strict=True)
            checked_call = call_verify(call_path)
            require(checked_call.get("call_ref") == model_ref and checked_call.get("live") is True,
                    "R3 model call 원문 검증")
        else:
            require(model_ref is None, "rule policy에 model call이 있다")
        after_events = _ledger_events(ledger_raw[:length])
        require(len(after_events) == len(prior_events) + 4
                and [row.get("event") for row in after_events[-4:]]
                == ["proposed", "approved", "consumed", "receipt"],
                "R3 step ledger lifecycle")
        evidence.append({"step": index, "context": ref(context_path), "choice": ref(choice_path),
                         "probe_result": ref(probe_result_path), "ledger_prefix": deepcopy(prefix)})
        prior_length = length
    require(prior_length == len(ledger_raw), "R3 final ledger에 결과 밖 suffix가 있다")
    final = DRIVER.C.V.build_snapshot(
        source["packet"], result["run_id"], result["host_storage_policy"],
        DRIVER.C.V.file_digest(DRIVER.C.V.PROTOCOL), ledger_path)
    require(final["view_digest"] == result.get("final_view_digest"), "R3 final view digest")
    return {"steps": evidence, "ledger": {**ref(ledger_path), "final_prefix": {
                "byte_length": len(ledger_raw), "sha256": hashlib.sha256(ledger_raw).hexdigest()}},
            "final_snapshot": final}


def _completed_batch(path: Path) -> dict[str, Any]:
    value = read(path)
    rows = value.get("rows")
    require(value.get("schema") == "lineage_five_policy_batch_result_v1"
            and value.get("status") == "complete" and value.get("planned_cases") == 10
            and value.get("planned_rows") == 50 and isinstance(rows, list) and len(rows) == 50
            and value.get("repair") is False and value.get("recovery") is False,
            "완료된 R3 actual 50-row result가 아니다")
    return value


def prepare_candidate(batch_result_path: Path, *, protocol_path: Path, config_path: Path,
                      anchor: str, case_id: str, policy: str,
                      preflight: Callable = RUNNER.preflight,
                      host_verify: Callable = SOURCE_INPUT.verify_pair,
                      call_verify: Callable = TRANSPORT.verify_call) -> dict[str, Any]:
    """R3 actual row를 재검증하고 승인 false인 첫 교정 후보 또는 HOLD를 반환한다."""
    target = (case_id, policy)
    require(target in TARGETS, "사전 고정된 case/policy만 후보가 될 수 있다")
    _protocol, _config, sources = preflight(protocol_path, anchor, config_path)
    batch_path = Path(batch_result_path).resolve(strict=True)
    batch = _completed_batch(batch_path)
    matches = [row for row in batch["rows"]
               if row.get("case_id") == case_id and row.get("arm") == policy]
    require(len(matches) == 1, "R3 target row가 유일하지 않다")
    row = matches[0]
    result_path = bound(row.get("result"), "R3 row result")
    result = read(result_path)
    source = sources[case_id]["source"]
    require(result.get("schema") == DRIVER.RESULT_SCHEMA and result.get("scope") == "actual_backend"
            and result.get("case_id") == case_id and result.get("arm") == policy
            and result.get("source_digest") == DRIVER.BASE.digest(source)
            and result.get("qualification") == source["qualification"],
            "R3 row result/source 결합")
    verified = _verify_steps(result_path, result, source, host_verify=host_verify, call_verify=call_verify)

    role_inputs = sources[case_id]["inputs"]
    case_bridge, case_raw = SOURCE_INPUT.checked(role_inputs["case"])
    reference_bridge, reference_raw = SOURCE_INPUT.checked(role_inputs["reference"])
    require(case_bridge["index"] == reference_bridge["index"] == source["index"]
            and case_raw["manifest"]["arm"] == TARGETS[target]
            and reference_raw["manifest"]["arm"] == "NONE",
            "R3 source index/case/reference original fault 결합")
    original_faults = deepcopy(SOURCE_INPUT.SOURCE_FAULTS[TARGETS[target]])
    hold_reason = result.get("hold_reason")
    family = result.get("confirmed_subtype")
    eligible = (result.get("end_state") == "confirmed" and hold_reason is None
                and family in CORRECTIONS and result.get("scoring", {}).get("correct_any_family") is True
                and result.get("scoring", {}).get("wrong_confirmation") is False)
    base = {
        "schema": SCHEMA, "case_id": case_id, "policy": policy,
        "source_arm": TARGETS[target], "source_seed": case_raw["manifest"]["seed"],
        "original_faults": original_faults, "source_checkpoint_sha256":
            case_bridge["binding"]["checkpoint_sha256"],
        "source_checkpoint_role": "provenance_only_never_loaded",
        "inputs": {"batch_result": ref(batch_path), "row_result": ref(result_path),
                   "five_policy_protocol": ref(Path(protocol_path)),
                   "execution_config": ref(Path(config_path)), "anchor": anchor,
                   "source_index": deepcopy(source["index"]),
                   "source_protocol": deepcopy(case_raw["raw"]["protocol"]),
                   "case_probe_input": ref(role_inputs["case"]),
                   "reference_probe_input": ref(role_inputs["reference"])},
        "diagnosis_evidence": verified,
        "approval_granted": False, "execution_allowed": False,
        "training_executed": False, "correction_executed": False,
        "replacement_allowed": False, "replacement_used": False,
    }
    if not eligible:
        reason = str(hold_reason or "wrong_or_unsupported_or_unconfirmed_family")
        return {**base, "status": "HOLD", "hold_reason": reason,
                "confirmed_family": family, "correction": None}
    correction = CORRECTIONS[cast(str, family)]
    final = verified["final_snapshot"]
    require(isinstance(final, dict), "confirmed row의 final snapshot이 없다")
    component, registered = DRIVER.C.registered_correction(final, applied=[])
    require(component == family and registered == correction,
            "confirmed family와 registry correction이 다르다")
    return {**base, "status": "validated_candidate", "hold_reason": None,
            "confirmed_family": family, "correction": correction}


def checked_candidate(path: Path, *, preflight: Callable = RUNNER.preflight,
                      host_verify: Callable = SOURCE_INPUT.verify_pair,
                      call_verify: Callable = TRANSPORT.verify_call) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    stored = read(path)
    inputs = stored.get("inputs")
    require(isinstance(inputs, dict), "candidate inputs가 없다")
    rebuilt = prepare_candidate(
        bound(inputs["batch_result"], "batch result"),
        protocol_path=bound(inputs["five_policy_protocol"], "five-policy protocol"),
        config_path=bound(inputs["execution_config"], "execution config"),
        anchor=inputs["anchor"], case_id=stored.get("case_id"), policy=stored.get("policy"),
        preflight=preflight, host_verify=host_verify, call_verify=call_verify)
    require(stored == rebuilt, "candidate 또는 R3 원문이 바뀌었다")
    return deepcopy(stored)


def _approval(path: Path, candidate_path: Path, protocol_path: Path, anchor: str,
              registered_seed: int, lease: dict[str, Any]) -> dict[str, Any]:
    value = read(path)
    require(value.get("schema") == APPROVAL_SCHEMA and value.get("decision") == "approve"
            and isinstance(value.get("approver"), str) and bool(value["approver"].strip())
            and value.get("candidate") == ref(candidate_path)
            and value.get("protocol") == ref(protocol_path) and value.get("anchor") == anchor
            and value.get("registered_seed") == registered_seed
            and value.get("lease_id") == lease.get("leaseId"),
            "operator approval/candidate/protocol/seed/lease 결합")
    return value


def execute_candidate(candidate_path: Path, protocol_path: Path, approval_path: Path, *,
                      anchor: str, lab: Path, isaac_python: Path, registered_seed: int,
                      out: Path, runner: Callable, wait: Callable, lease_reader: Callable,
                      preflight: Callable = RUNNER.preflight,
                      host_verify: Callable = SOURCE_INPUT.verify_pair,
                      call_verify: Callable = TRANSPORT.verify_call) -> dict[str, Any]:
    """승인된 첫 correction을 fresh FIRST 300→fixed26→등록 회복 판정으로 실행한다."""
    candidate_path, protocol_path = Path(candidate_path).resolve(), Path(protocol_path).resolve()
    out, lab, isaac_python = Path(out).resolve(), Path(lab).resolve(), Path(isaac_python).resolve()
    require(not out.exists(), "recovery output은 fresh 경로여야 한다")
    candidate = checked_candidate(candidate_path, preflight=preflight,
                                  host_verify=host_verify, call_verify=call_verify)
    require(candidate.get("status") == "validated_candidate", "HOLD 후보는 실행할 수 없다")
    protocol = verify_protocol(protocol_path, anchor=anchor, lab=lab)
    key = f"{candidate['case_id']}:{candidate['policy']}"
    require(type(registered_seed) is int and protocol["recovery_seeds"].get(key) == registered_seed,
            "실행 seed가 공개 recovery protocol과 다르다")
    require(candidate["inputs"]["source_protocol"] == {
                "path": str((ROOT / protocol["components"]["source_collection"]["path"]).resolve()),
                "sha256": sha(ROOT / protocol["components"]["source_collection"]["path"])},
            "candidate source protocol과 recovery protocol이 다르다")
    lease = lease_reader()
    require(isinstance(lease, dict) and isinstance(lease.get("leaseId"), str)
            and bool(lease["leaseId"]), "공용 GPU lease가 없다")
    _approval(approval_path, candidate_path, protocol_path, anchor, registered_seed, lease)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "intent.json", {"schema": RESULT_SCHEMA + "_intent",
              "candidate": ref(candidate_path), "protocol": ref(protocol_path),
              "approval": ref(approval_path), "lease_id": lease["leaseId"],
              "registered_seed": registered_seed, "automatic_retry": False})
    try:
        training_stage = out / "training"
        manifest = TRAINING.build_manifest(
            protocol_path, protocol, anchor, stage=training_stage, seed=registered_seed,
            arm="FIRST", original_faults=candidate["original_faults"],
            applied_corrections=[candidate["correction"]],
            source_checkpoint_sha256=candidate["source_checkpoint_sha256"],
            run_name=f"five_policy_recovery_{candidate['case_id'].replace('-', '_')}_{candidate['policy']}")
        training = TRAINING.run_training(
            training_stage, manifest, lab=lab, isaac_python=isaac_python,
            runner=runner, wait=wait)
        current_lease = lease_reader()
        require(current_lease.get("leaseId") == lease["leaseId"], "recovery 중 GPU lease가 바뀌었다")
        evidence = read(bound(training["evidence"], "recovery training evidence"))
        require(evidence.get("fresh_initialization") is True
                and evidence.get("checkpoint_loaded") is False
                and evidence.get("resume_requested") is False
                and evidence.get("source_checkpoint_sha256") == candidate["source_checkpoint_sha256"],
                "fresh FIRST training evidence가 다르다")
        corrected, evaluation = TRAINING.run_fixed(
            out / "evaluation", evidence,
            name=f"recovery_{candidate['case_id'].replace('-', '_')}_{candidate['policy']}",
            lab=lab, isaac_python=isaac_python, runner=runner, wait=wait)
        reference_bridge, reference_raw = SOURCE_INPUT.checked(
            bound(candidate["inputs"]["reference_probe_input"], "reference probe input"))
        del reference_bridge
        fixed_result = read(Path(reference_raw["raw"]["fixed_result"]["path"]))
        reference_report = fixed_result.get("report")
        require(isinstance(reference_report, dict), "source reference fixed26 report가 없다")
        verdict = GATE.RESIDUAL.recovery(corrected, reference_report,
                                         read(OLD_RECOVERY.BANDS)["bands"])["verdict"]
        outcome = verdict.get("outcome")
        status = "HOLD" if outcome == "unknown" else outcome
        require(status in {"recovered", "not_recovered", "HOLD"}, "등록 recovery 판정이 아니다")
        receipt = {
            "schema": RECEIPT_SCHEMA, "status": status,
            "case_id": candidate["case_id"], "policy": candidate["policy"],
            "registered_seed": registered_seed, "source_seed": candidate["source_seed"],
            "candidate": ref(candidate_path), "approval": ref(approval_path),
            "protocol": ref(protocol_path), "lease_id": lease["leaseId"],
            "confirmed_family": candidate["confirmed_family"],
            "correction": candidate["correction"], "training_mode": "fresh_restart",
            "source_checkpoint_sha256": candidate["source_checkpoint_sha256"],
            "source_checkpoint_loaded": False, "training": deepcopy(training),
            "new_checkpoint": deepcopy(training["checkpoint"]),
            "fixed26": {"conditions": 26, "corrected_report": deepcopy(corrected),
                        "reference_report_sha256": digest(reference_report),
                        "execution": deepcopy(evaluation)},
            "recovery": deepcopy(verdict), "automatic_retry": False,
            "second_epoch_registered": False,
            "next_step": "second_diagnosis_required" if status == "not_recovered" else "stop",
        }
        receipt_path = out / "receipt.json"
        write_new(receipt_path, receipt)
        epoch2 = None
        if status == "not_recovered":
            epoch2 = {
                "schema": EPOCH2_SCHEMA, "status": "input_only",
                "case_id": candidate["case_id"], "policy": candidate["policy"],
                "prior_receipt": ref(receipt_path), "candidate": ref(candidate_path),
                "checkpoint": deepcopy(training["checkpoint"]),
                "collected": deepcopy(training["collected"]),
                "runtime_facts": deepcopy(training["runtime_facts"]),
                "remaining_faults": deepcopy(evidence["remaining_faults"]),
                "new_protocol_required": True, "approval_granted": False,
                "execution_allowed": False, "automatic_second_execution": False,
            }
            write_new(out / "epoch2-input.json", epoch2)
        result = {"schema": RESULT_SCHEMA, "status": "complete", "receipt": ref(receipt_path),
                  "outcome": status, "epoch2_input": ref(out / "epoch2-input.json") if epoch2 else None,
                  "overall_goal_complete": False, "automatic_retry": False}
        write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as error:
        if not (out / "failure.json").exists():
            write_new(out / "failure.json", {"status": "invalid", "error_type": type(error).__name__,
                      "error": str(error), "automatic_retry": False})
        raise
