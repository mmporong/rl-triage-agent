# pyright: reportArgumentType=false, reportMissingImports=false
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""첫 교정 비회복 receipt를 epoch-2 혼합 source 진단과 두 번째 교정에 연결한다."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import five_policy_recovery_candidate_v1 as FIRST
import five_policy_recovery_nim_transport_v1 as TRANSPORT
import lineage_five_policy_driver_v1 as DRIVER
import lineage_source_probe_inputs_v1 as SOURCE_INPUT
import local_pair_probe_host_v1 as BASE_HOST
import optimizer_runtime_facts_v1 as RUNTIME
import runtime_observation_probe_v1 as BASE_PROBE
import runtime_observation_training_v1 as TRAINING
import runtime_training_inputs_v1 as TRAINING_INPUT

SOURCE_SCHEMA = "five_policy_recovery_epoch2_source_v1"
SECOND_CANDIDATE_SCHEMA = "five_policy_recovery_second_candidate_v1"
SECOND_APPROVAL_SCHEMA = "five_policy_recovery_second_approval_v1"
SECOND_RECEIPT_SCHEMA = "five_policy_recovery_second_receipt_v1"
SECOND_RESULT_SCHEMA = "five_policy_recovery_second_result_v1"
DIAGNOSIS_RESULT_SCHEMA = "five_policy_recovery_epoch2_diagnosis_v1"
LOOP_SCHEMA = TRANSPORT.PROTOCOL_SCHEMA
POLICIES = ("fixed_eligible", "logs_only")
TARGETS = tuple(FIRST.TARGETS)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FAULT_FOR_CORRECTION = {
    "restore_body_frame_reward": "F1",
    "restore_configured_ppo_epochs": "R1",
}
SOURCE_FILES = tuple(sorted({
    *DRIVER.SOURCE_FILES, *SOURCE_INPUT.SOURCE_FILES, *TRANSPORT.SOURCE_FILES,
    *FIRST.SOURCE_FILES, *BASE_PROBE.SOURCE_FILES, *BASE_HOST.SOURCE_FILES,
    *TRAINING.TRAIN.BOUND_SOURCES,
    "evals/five_policy_recovery_candidate_v1.py",
    "tests/test_five_policy_recovery_candidate_v1.py",
    "evals/five_policy_recovery_nim_transport_v1.py",
    "tests/test_five_policy_recovery_nim_transport_v1.py",
    "evals/five_policy_recovery_epoch2_v1.py",
    "tests/test_five_policy_recovery_epoch2_v1.py",
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
    target = Path(path).resolve()
    require(not target.exists(), f"{target.name}은 fresh 파일이어야 한다")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path: Path) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    require(target.is_file() and not target.is_symlink(), "안전한 파일 ref가 아니다")
    return {"path": str(target), "sha256": sha(target)}


def bound(value: object, name: str) -> Path:
    require(isinstance(value, dict) and set(value) >= {"path", "sha256"}, f"{name} ref 형식")
    item = cast(dict[str, Any], value)
    target = Path(str(item["path"])).resolve(strict=True)
    require(target.is_file() and not target.is_symlink()
            and HEX64.fullmatch(str(item["sha256"])) is not None
            and sha(target) == item["sha256"], f"{name} SHA가 바뀌었다")
    return target


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("protocol component는 저장소 내부여야 한다") from error


def _isolated(module: ModuleType, name: str) -> ModuleType:
    path = Path(str(module.__file__)).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    clone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clone)
    return clone


def _mixed_source(path: Path, lab: Path, role: str) -> tuple[dict, dict, dict]:
    """schema를 바꾸지 않고 FIRST collected와 source-v2 reference를 분기한다."""
    schema = read(path).get("schema")
    if role == "case" and schema == TRAINING_INPUT.SCHEMA:
        return cast(tuple[dict, dict, dict], BASE_PROBE._source(path, lab, role))
    if role == "reference" and schema == SOURCE_INPUT.SCHEMA:
        return cast(tuple[dict, dict, dict], SOURCE_INPUT._source(path, lab, role))
    raise ValueError("epoch-2 case는 FIRST collected, reference는 source-v2 bridge여야 한다")


_PROBE = _isolated(BASE_PROBE, "_five_policy_recovery_epoch2_probe_v1")
_PROBE.__dict__["SOURCE_FILES"] = SOURCE_FILES
_PROBE.__dict__["_source"] = _mixed_source
# 실제 remaining_faults/applied_corrections/readback을 소비하는 BASE 구현을 유지한다.
_PROBE.__dict__["_effective_fault"] = BASE_PROBE._effective_fault
_PROBE.__dict__["_residual_manifest"] = BASE_PROBE._residual_manifest
_HOST = _isolated(BASE_HOST, "_five_policy_recovery_epoch2_host_v1")
_HOST.__dict__["OBS"] = _PROBE


def prepare_probe_protocol(base_probe_protocol: Path) -> dict[str, Any]:
    """혼합 source closure를 기존 runtime observation component 위에 공개 등록한다."""
    return cast(dict[str, Any], _PROBE.prepare_protocol(base_probe_protocol))


def execute_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return cast(dict[str, Any], _HOST.execute_pair(*args, **kwargs))


def verify_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return cast(dict[str, Any], _HOST.verify_pair(*args, **kwargs))


def protocol_value(first_component: Path, probe_protocol: Path,
                   sources: list[dict[str, Any]], second_seeds: Mapping[str, int]) -> dict[str, Any]:
    """RI382 cap32 protocol에 epoch-2 component/source/seed 결합을 추가한다."""
    first_path = Path(first_component).resolve(strict=True)
    probe_path = Path(probe_protocol).resolve(strict=True)
    expected_keys = {f"{case}:{policy}" for case, policy in TARGETS}
    require(set(second_seeds) == expected_keys
            and all(type(seed) is int and seed >= 0 for seed in second_seeds.values()),
            "네 pipeline의 등록 second seed가 필요하다")
    first = read(first_path)
    require(first.get("schema") == FIRST.PROTOCOL_SCHEMA,
            "first correction component schema가 다르다")
    design_path, _design = FIRST.component_conditional_design(first, second_seeds=second_seeds)
    require(read(probe_path).get("schema") == BASE_PROBE.PROTOCOL_SCHEMA,
            "mixed probe protocol schema가 다르다")
    source_digests: dict[str, str] = {}
    for source in sources:
        checked = checked_source(source)
        key = f"{checked['case_id']}:{checked['policy']}"
        require(key in expected_keys and key not in source_digests, "epoch-2 source target 중복/범위")
        source_digests[key] = digest(checked)
    pins = {name: sha(ROOT / name, lf=True) for name in SOURCE_FILES}
    return {
        "schema": LOOP_SCHEMA, "purpose": "R4", "api_cap": TRANSPORT.API_CAP,
        "model": TRANSPORT.CALL.PARAMS["model"],
        "accepted_provider_models": sorted(TRANSPORT.MODEL_CALL.ACCEPTED_PROVIDER_MODELS),
        "base_params": deepcopy(TRANSPORT.CALL.PARAMS), "sources_lf": pins,
        "backend0prefix": 0, "authority": deepcopy(TRANSPORT.AUTHORITY),
        "automatic_retries": 0, "policies": list(POLICIES), "max_probes_per_arm": 3,
        "targets": [f"{case}:{policy}" for case, policy in TARGETS],
        "second_seeds": dict(sorted(second_seeds.items())),
        "source_digests": source_digests,
        "first_correction_component": {
            "path": _relative(first_path), "sha256_lf": sha(first_path, lf=True)},
        "conditional_design": {"path": _relative(design_path), "sha256_lf": sha(design_path, lf=True)},
        "probe_protocol": {"path": _relative(probe_path), "sha256_lf": sha(probe_path, lf=True)},
        "view_protocol": {"path": _relative(DRIVER.C.V.PROTOCOL),
                          "sha256_lf": sha(DRIVER.C.V.PROTOCOL, lf=True)},
        "denominator_changed": False, "new_cases_added": False,
        "missing_target_success_credit": False,
        "first_component_authority_inherited": False,
    }


def verify_loop_protocol(path: Path, anchor: str, *, lab: Path | None = None) -> dict[str, Any]:
    value = TRANSPORT._verify_protocol(path, anchor)
    expected_targets = [f"{case}:{policy}" for case, policy in TARGETS]
    require(value.get("policies") == list(POLICIES) and value.get("max_probes_per_arm") == 3
            and value.get("targets") == expected_targets
            and set(value.get("second_seeds", {})) == set(expected_targets)
            and all(type(seed) is int and seed >= 0 for seed in value["second_seeds"].values())
            and isinstance(value.get("source_digests"), dict)
            and value.get("denominator_changed") is False
            and value.get("new_cases_added") is False
            and value.get("missing_target_success_credit") is False
            and value.get("first_component_authority_inherited") is False,
            "epoch-2 loop 등록 범위가 다르다")
    first_row, probe_row, view_row = (
        value.get("first_correction_component"), value.get("probe_protocol"),
        value.get("view_protocol"))
    for row, schema, label in (
        (first_row, FIRST.PROTOCOL_SCHEMA, "first component"),
        (probe_row, BASE_PROBE.PROTOCOL_SCHEMA, "probe protocol"),
    ):
        require(isinstance(row, dict), f"{label} ref가 없다")
        component = ROOT / row["path"]
        require(component.is_file() and sha(component, lf=True) == row.get("sha256_lf")
                and read(component).get("schema") == schema, f"{label} 결합이 다르다")
        committed = TRANSPORT._git(ROOT, "show", f"{anchor}:{row['path']}")
        normalized = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        require(hashlib.sha256(normalized).hexdigest() == row["sha256_lf"],
                f"anchor {label} blob이 다르다")
    design_path, _design = FIRST.component_conditional_design(
        read(ROOT / first_row["path"]), second_seeds=value["second_seeds"])
    design_row = {"path": _relative(design_path), "sha256_lf": sha(design_path, lf=True)}
    require(value.get("conditional_design") == design_row, "loop conditional design 결합")
    design_blob = TRANSPORT._git(ROOT, "show", f"{anchor}:{design_row['path']}")
    design_blob = design_blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    require(hashlib.sha256(design_blob).hexdigest() == design_row["sha256_lf"],
            "anchor conditional design blob이 다르다")
    require(isinstance(view_row, dict) and ROOT / view_row["path"] == DRIVER.C.V.PROTOCOL
            and sha(DRIVER.C.V.PROTOCOL, lf=True) == view_row.get("sha256_lf"),
            "selection view protocol 결합이 다르다")
    committed_view = TRANSPORT._git(ROOT, "show", f"{anchor}:{view_row['path']}")
    normalized_view = committed_view.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    require(hashlib.sha256(normalized_view).hexdigest() == view_row["sha256_lf"],
            "anchor selection view protocol blob이 다르다")
    if lab is not None:
        _PROBE.verify_protocol(ROOT / probe_row["path"], Path(lab).resolve(), anchor)
    return deepcopy(value)


def _first_material(epoch2_path: Path, *, lab: Path,
                    candidate_checker: Callable = FIRST.checked_candidate) -> dict[str, Any]:
    epoch2_path = Path(epoch2_path).resolve(strict=True)
    value = read(epoch2_path)
    require(value.get("schema") == FIRST.EPOCH2_SCHEMA and value.get("status") == "input_only"
            and value.get("new_protocol_required") is True
            and value.get("approval_granted") is False
            and value.get("execution_allowed") is False
            and value.get("automatic_second_execution") is False,
            "epoch-2 input 권한/shape가 다르다")
    receipt_path = bound(value.get("prior_receipt"), "first receipt")
    receipt = read(receipt_path)
    candidate_path = bound(value.get("candidate"), "first candidate")
    candidate = candidate_checker(candidate_path)
    require(receipt.get("schema") == FIRST.RECEIPT_SCHEMA
            and receipt.get("status") == "not_recovered"
            and receipt.get("recovery", {}).get("outcome") == "not_recovered"
            and receipt.get("candidate") == ref(candidate_path)
            and receipt.get("case_id") == candidate["case_id"]
            and receipt.get("policy") == candidate["policy"]
            and receipt.get("correction") == candidate["correction"]
            and receipt.get("new_checkpoint") == value.get("checkpoint")
            and receipt.get("training", {}).get("checkpoint") == value.get("checkpoint")
            and receipt.get("training", {}).get("collected") == value.get("collected")
            and receipt.get("training", {}).get("runtime_facts") == value.get("runtime_facts")
            and receipt.get("automatic_retry") is False
            and receipt.get("second_epoch_registered") is False,
            "first not_recovered receipt/epoch2 input 결합이 다르다")
    first_protocol_path = bound(receipt.get("protocol"), "first recovery protocol")
    FIRST.verify_protocol(
        first_protocol_path, anchor=candidate["inputs"]["anchor"], lab=Path(lab).resolve())
    checkpoint_path = bound(value.get("checkpoint"), "FIRST checkpoint")
    collected_path = bound(value.get("collected"), "FIRST collected")
    training_collected, _training_meta = TRAINING_INPUT.checked(collected_path)
    checkpoint_sha = sha(checkpoint_path)
    require(training_collected["source_binding"]["checkpoint_sha256"] == checkpoint_sha,
            "FIRST collected/checkpoint SHA가 다르다")
    runtime_rows = value.get("runtime_facts")
    require(isinstance(runtime_rows, dict) and set(runtime_rows) == {"case", "reference"},
            "FIRST runtime fact refs가 다르다")
    case_runtime_path = bound(runtime_rows["case"], "FIRST case runtime facts")
    _case_envelope, case_runtime = RUNTIME.checked(
        case_runtime_path, expected_checkpoint_sha256=checkpoint_sha)
    reference_runtime_path = bound(runtime_rows["reference"], "FIRST reference runtime facts")
    RUNTIME.checked(reference_runtime_path, expected_checkpoint_sha256=checkpoint_sha)
    evidence = read(bound(receipt["training"]["evidence"], "FIRST training evidence"))
    require(evidence.get("fresh_initialization") is True
            and evidence.get("checkpoint_loaded") is False
            and evidence.get("resume_requested") is False
            and evidence.get("source_checkpoint_sha256") == candidate["source_checkpoint_sha256"]
            and evidence.get("remaining_faults") == value.get("remaining_faults")
            and receipt.get("source_checkpoint_sha256") == candidate["source_checkpoint_sha256"]
            and receipt.get("source_checkpoint_loaded") is False,
            "FIRST fresh training/provenance evidence가 다르다")
    case_collected, case_meta, case_source = _mixed_source(collected_path, lab, "case")
    require(case_collected == training_collected
            and case_source["checkpoint"]["sha256"] == checkpoint_sha
            and case_source["runtime_facts"] == case_runtime
            and case_source["original_faults"] == candidate["original_faults"]
            and case_source["applied_corrections"] == [candidate["correction"]]
            and case_source["remaining_faults"] == value.get("remaining_faults"),
            "FIRST 실제 readback lineage가 epoch-2 입력과 다르다")
    reference_path = bound(candidate["inputs"]["reference_probe_input"], "source-v2 reference")
    reference_collected, reference_meta, reference_source = _mixed_source(
        reference_path, lab, "reference")
    require(reference_source["source_arm"] == "NONE"
            and reference_source["original_faults"] == []
            and reference_source["applied_corrections"] == []
            and reference_source["remaining_faults"] == []
            and reference_source["runtime_facts"] == RUNTIME.unavailable_facts("reference"),
            "source-v2 NONE reference 계약이 다르다")
    return {
        "input": value, "input_path": epoch2_path, "receipt": receipt,
        "receipt_path": receipt_path, "receipt_sha256": sha(receipt_path),
        "first_protocol_path": first_protocol_path,
        "candidate": candidate, "candidate_path": candidate_path,
        "checkpoint_sha256": checkpoint_sha, "collected_path": collected_path,
        "case_collected": case_collected, "case_metadata": case_meta, "case_source": case_source,
        "reference_path": reference_path, "reference_collected": reference_collected,
        "reference_metadata": reference_meta, "reference_source": reference_source,
    }


def _prepare_source_value(epoch2_input: Path, *, lab: Path,
                          candidate_checker: Callable = FIRST.checked_candidate) -> dict[str, Any]:
    material = _first_material(epoch2_input, lab=lab, candidate_checker=candidate_checker)
    case_meta, reference_meta = material["case_metadata"], material["reference_metadata"]
    case_facts, _ = TRAINING_INPUT.INPUT._telemetry_projection(
        "case", read(Path(case_meta["telemetry"])))
    reference_facts, _ = TRAINING_INPUT.INPUT._telemetry_projection(
        "reference", read(Path(reference_meta["raw"]["telemetry"]["path"])))
    public_facts = {**case_facts, **reference_facts}
    runtime_facts = {**material["case_source"]["runtime_facts"],
                     **material["reference_source"]["runtime_facts"]}
    candidate = material["candidate"]
    prior = {
        "verdict": "not_recovered", "correction_id": candidate["correction"],
        "result_sha256": material["receipt_sha256"],
        "checkpoint_sha256": material["checkpoint_sha256"],
    }
    bindings = {
        "case": {"checkpoint_sha256": material["checkpoint_sha256"],
                 "telemetry_sha256": material["case_source"]["telemetry_sha256"]},
        "reference": {"checkpoint_sha256": material["reference_source"]["checkpoint"]["sha256"],
                      "telemetry_sha256": material["reference_source"]["telemetry_sha256"]},
    }
    public = DRIVER.C.V.S.build_packet(
        epoch=2, applied_corrections=[candidate["correction"]], prior_receipt=prior,
        facts=public_facts, input_bindings=bindings, budget_left=3)
    backend = DRIVER.C.V.S.build_packet(
        epoch=2, applied_corrections=[candidate["correction"]], prior_receipt=prior,
        facts={**public_facts, **runtime_facts}, input_bindings=bindings, budget_left=3)
    BASE_HOST._packets(public, backend)
    source = {
        "schema": SOURCE_SCHEMA, "case_id": candidate["case_id"],
        "policy": candidate["policy"], "packet": public, "backend_packet": backend,
        "qualification": {"arm": candidate["source_arm"], "label": "eligible",
                          "diagnosis_eligible": True},
        "source_binding": {
            "epoch2_input_sha256": sha(material["input_path"]),
            "first_receipt_sha256": material["receipt_sha256"],
            "first_component_sha256_lf": sha(material["first_protocol_path"], lf=True),
            "case_checkpoint_sha256": material["checkpoint_sha256"],
            "reference_checkpoint_sha256": material["reference_source"]["checkpoint"]["sha256"],
        },
        "index": deepcopy(candidate["inputs"]["source_index"]),
        "inputs": {"epoch2_input": ref(material["input_path"]),
                   "case_collected": ref(material["collected_path"]),
                   "reference_source_v2": ref(material["reference_path"]),
                   "first_candidate": ref(material["candidate_path"]),
                   "first_receipt": ref(material["receipt_path"]),
                   "first_protocol": ref(material["first_protocol_path"])},
        "lab": str(Path(lab).resolve()), "first_correction": candidate["correction"],
        "original_faults": deepcopy(candidate["original_faults"]),
        "remaining_faults": deepcopy(material["case_source"]["remaining_faults"]),
        "providers": None, "visual_evidence": None,
        "approval_granted": False, "api_authorized": False,
    }
    return source


def checked_source(source: dict[str, Any], *,
                   candidate_checker: Callable = FIRST.checked_candidate) -> dict[str, Any]:
    require(source.get("schema") == SOURCE_SCHEMA and source.get("policy") in POLICIES
            and (source.get("case_id"), source.get("policy")) in FIRST.TARGETS
            and source.get("approval_granted") is False
            and source.get("api_authorized") is False
            and source.get("providers") is None and source.get("visual_evidence") is None,
            "epoch-2 source contract")
    inputs = source.get("inputs")
    require(isinstance(inputs, dict), "epoch-2 source inputs가 없다")
    rebuilt = _prepare_source_value(
        bound(inputs["epoch2_input"], "epoch2 input"), lab=Path(source["lab"]),
        candidate_checker=candidate_checker)
    require(source == rebuilt, "epoch-2 source 또는 원문이 바뀌었다")
    return deepcopy(source)


def prepare_source(epoch2_input: Path, *, lab: Path,
                   candidate_checker: Callable = FIRST.checked_candidate) -> dict[str, Any]:
    value = _prepare_source_value(
        epoch2_input, lab=lab, candidate_checker=candidate_checker)
    return checked_source(value, candidate_checker=candidate_checker)


def _driver_protocol(path: Path, anchor: str) -> dict[str, Any]:
    return verify_loop_protocol(path, anchor)


def _driver_checked_source(source: dict[str, Any], *, fixture: bool = False) -> dict[str, Any]:
    """root driver fixture ABI를 받되 R4 원문 검증은 완화하지 않는다."""
    require(type(fixture) is bool, "fixture flag 형식")
    return checked_source(source)


def _driver_source_registration_key(source: dict[str, Any], arm: str) -> str:
    checked = checked_source(source)
    require(arm in POLICIES and checked["policy"] == arm, "R4 source/policy 등록 key 불일치")
    return f"{checked['case_id']}:{arm}"


def _driver_context(source: dict, snapshot: dict, arm: str, remaining: int) -> tuple[dict, list]:
    checked_source(source)
    require(arm in POLICIES and source.get("providers") is None, "R4 context source/arm 범위")
    context = DRIVER.A.prepare(snapshot, goal=DRIVER.BASE.P.PAIR_GOAL, decisions_left=remaining)
    return context, DRIVER.CALL._messages(context, DRIVER.A.messages(context))


_DRIVER = _isolated(DRIVER, "_five_policy_recovery_epoch2_driver_v1")
_DRIVER.__dict__["checked_source"] = _driver_checked_source
_DRIVER.__dict__["verify_protocol"] = _driver_protocol
_DRIVER.__dict__["source_registration_key"] = _driver_source_registration_key
_DRIVER.__dict__["context_for"] = _driver_context
_DRIVER.__dict__["TRANSPORT"] = TRANSPORT
_DRIVER.__dict__["RESULT_SCHEMA"] = DIAGNOSIS_RESULT_SCHEMA


def run_diagnosis(source: dict[str, Any], *, arm: str, run_id: str, output: Path,
                  protocol_path: Path, execution_anchor: str, lab: Path,
                  complete: Callable | None, execute_probe: Callable,
                  busy_reader: Callable) -> dict[str, Any]:
    """고정/로그 정책 하나를 최대 세 검사로 actual epoch-2 진단한다."""
    require(arm in POLICIES and arm == source.get("policy"), "등록 policy instance만 실행한다")
    checked = checked_source(source)
    protocol = verify_loop_protocol(protocol_path, execution_anchor, lab=lab)
    key = f"{checked['case_id']}:{checked['policy']}"
    first_component = protocol["first_correction_component"]
    require(protocol["source_digests"].get(key) == digest(checked)
            and first_component["sha256_lf"]
            == checked["source_binding"]["first_component_sha256_lf"],
            "epoch-2 source/first component가 미등록이다")
    return cast(dict[str, Any], _DRIVER.run_arm(
        checked, arm=arm, run_id=run_id, output=output, protocol_path=protocol_path,
        execution_anchor=execution_anchor, complete=complete, execute_probe=execute_probe,
        verify_probe=verify_pair, verify_call=TRANSPORT.verify_call,
        busy_reader=busy_reader, allow_fixture=False))


def prepare_second_candidate(source_path: Path, diagnosis_result_path: Path, *,
                             protocol_path: Path, anchor: str, lab: Path) -> dict[str, Any]:
    source_path = Path(source_path).resolve(strict=True)
    source = checked_source(read(source_path))
    protocol = verify_loop_protocol(protocol_path, anchor, lab=lab)
    key = f"{source['case_id']}:{source['policy']}"
    require(protocol["source_digests"].get(key) == digest(source), "epoch-2 source가 미등록이다")
    result_path = Path(diagnosis_result_path).resolve(strict=True)
    result = read(result_path)
    require(result.get("schema") == DIAGNOSIS_RESULT_SCHEMA
            and result.get("scope") == "actual_backend"
            and result.get("case_id") == source["case_id"]
            and result.get("arm") == source["policy"]
            and result.get("source_digest") == digest(source),
            "epoch-2 actual diagnosis/source 결합")
    verified = FIRST._verify_steps(
        result_path, result, source, host_verify=verify_pair, call_verify=TRANSPORT.verify_call)
    family = result.get("confirmed_subtype")
    eligible = (result.get("end_state") == "confirmed" and result.get("hold_reason") is None
                and family in FIRST.CORRECTIONS
                and result.get("scoring", {}).get("correct_any_family") is True
                and result.get("scoring", {}).get("wrong_confirmation") is False)
    base = {
        "schema": SECOND_CANDIDATE_SCHEMA, "case_id": source["case_id"],
        "policy": source["policy"], "source_arm": source["qualification"]["arm"],
        "first_correction": source["first_correction"],
        "original_faults": deepcopy(source["original_faults"]),
        "remaining_faults": deepcopy(source["remaining_faults"]),
        "source_checkpoint_sha256": source["packet"]["input_bindings"]["case"]["checkpoint_sha256"],
        "source_checkpoint_role": "provenance_only_never_loaded",
        "inputs": {"source": ref(source_path), "diagnosis_result": ref(result_path),
                   "loop_protocol": ref(Path(protocol_path)), "anchor": anchor,
                   "reference_source_v2": deepcopy(source["inputs"]["reference_source_v2"])},
        "diagnosis_evidence": verified, "approval_granted": False,
        "execution_allowed": False, "training_executed": False,
        "replacement_allowed": False, "replacement_used": False,
    }
    if not eligible:
        return {**base, "status": "HOLD", "hold_reason": str(
            result.get("hold_reason") or "wrong_or_unsupported_or_unconfirmed_family"),
            "confirmed_family": family, "second_correction": None}
    final = verified["final_snapshot"]
    component, correction = DRIVER.C.registered_correction(final, [source["first_correction"]])
    required_fault = FAULT_FOR_CORRECTION.get(correction)
    require(component == family and correction != source["first_correction"]
            and required_fault in source["original_faults"]
            and required_fault in source["remaining_faults"],
            "두 번째 교정이 original/remaining fault 또는 중복 제한을 벗어났다")
    confirming_probe = "P_reward" if family == "reward" else DRIVER.C.RC.NEW_PROBE
    require(any(row.get("probe") == confirming_probe and row.get("outcome") == "abnormal"
                for row in result["steps"]), "두 번째 교정의 actual abnormal receipt가 없다")
    return {**base, "status": "validated_candidate", "hold_reason": None,
            "confirmed_family": family, "second_correction": correction}


def checked_second_candidate(path: Path) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    stored = read(target)
    inputs = stored.get("inputs")
    require(isinstance(inputs, dict), "second candidate inputs가 없다")
    source_path = bound(inputs["source"], "epoch2 source")
    source = read(source_path)
    rebuilt = prepare_second_candidate(
        source_path, bound(inputs["diagnosis_result"], "epoch2 diagnosis"),
        protocol_path=bound(inputs["loop_protocol"], "loop protocol"),
        anchor=inputs["anchor"], lab=Path(source["lab"]))
    require(stored == rebuilt, "second candidate 또는 원문이 바뀌었다")
    return deepcopy(stored)


def _approval(path: Path, candidate_path: Path, protocol_path: Path, anchor: str,
              seed: int, lease: dict[str, Any]) -> dict[str, Any]:
    value = read(path)
    require(value.get("schema") == SECOND_APPROVAL_SCHEMA and value.get("decision") == "approve"
            and isinstance(value.get("approver"), str) and bool(value["approver"].strip())
            and value.get("candidate") == ref(candidate_path)
            and value.get("protocol") == ref(protocol_path) and value.get("anchor") == anchor
            and value.get("registered_seed") == seed
            and value.get("lease_id") == lease.get("leaseId"),
            "second approval/candidate/protocol/seed/lease 결합")
    return value


def execute_second_candidate(candidate_path: Path, protocol_path: Path, approval_path: Path, *,
                             anchor: str, lab: Path, isaac_python: Path, registered_seed: int,
                             out: Path, runner: Callable, wait: Callable,
                             lease_reader: Callable) -> dict[str, Any]:
    """명시 승인된 서로 다른 두 교정을 BOTH fresh300/fixed26으로 평가한다."""
    candidate_path, protocol_path = Path(candidate_path).resolve(), Path(protocol_path).resolve()
    out, lab = Path(out).resolve(), Path(lab).resolve()
    require(not out.exists(), "second recovery output은 fresh 경로여야 한다")
    candidate = checked_second_candidate(candidate_path)
    require(candidate.get("status") == "validated_candidate", "HOLD second candidate는 실행 불가")
    protocol = verify_loop_protocol(protocol_path, anchor, lab=lab)
    key = f"{candidate['case_id']}:{candidate['policy']}"
    require(type(registered_seed) is int and protocol["second_seeds"].get(key) == registered_seed,
            "second seed가 공개 loop protocol과 다르다")
    lease = lease_reader()
    require(isinstance(lease, dict) and isinstance(lease.get("leaseId"), str)
            and bool(lease["leaseId"]), "공용 GPU lease가 없다")
    _approval(approval_path, candidate_path, protocol_path, anchor, registered_seed, lease)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "intent.json", {
        "schema": SECOND_RESULT_SCHEMA + "_intent", "candidate": ref(candidate_path),
        "protocol": ref(protocol_path), "approval": ref(approval_path),
        "lease_id": lease["leaseId"], "registered_seed": registered_seed,
        "automatic_retry": False})
    try:
        corrections = [candidate["first_correction"], candidate["second_correction"]]
        manifest = TRAINING.build_manifest(
            protocol_path, protocol, anchor, stage=out / "training", seed=registered_seed,
            arm="BOTH", original_faults=candidate["original_faults"],
            applied_corrections=corrections,
            source_checkpoint_sha256=candidate["source_checkpoint_sha256"],
            run_name=f"five_policy_second_{candidate['case_id'].replace('-', '_')}_{candidate['policy']}")
        training = TRAINING.run_training(
            out / "training", manifest, lab=lab, isaac_python=Path(isaac_python),
            runner=runner, wait=wait)
        require(lease_reader().get("leaseId") == lease["leaseId"], "second recovery 중 lease 변경")
        evidence = read(bound(training["evidence"], "second training evidence"))
        require(evidence.get("fresh_initialization") is True
                and evidence.get("checkpoint_loaded") is False
                and evidence.get("resume_requested") is False
                and evidence.get("source_checkpoint_sha256") == candidate["source_checkpoint_sha256"],
                "BOTH fresh training evidence가 다르다")
        corrected, evaluation = TRAINING.run_fixed(
            out / "evaluation", evidence,
            name=f"second_{candidate['case_id'].replace('-', '_')}_{candidate['policy']}",
            lab=lab, isaac_python=Path(isaac_python), runner=runner, wait=wait)
        _bridge, reference_raw = SOURCE_INPUT.checked(
            bound(candidate["inputs"]["reference_source_v2"], "reference source-v2"))
        reference_report = read(Path(reference_raw["raw"]["fixed_result"]["path"])).get("report")
        require(isinstance(reference_report, dict), "source reference fixed26 report가 없다")
        verdict = FIRST.GATE.RESIDUAL.recovery(
            corrected, reference_report, read(FIRST.OLD_RECOVERY.BANDS)["bands"])["verdict"]
        outcome = verdict.get("outcome")
        status = "HOLD" if outcome == "unknown" else outcome
        require(status in {"recovered", "not_recovered", "HOLD"}, "등록 recovery 판정이 아니다")
        receipt = {
            "schema": SECOND_RECEIPT_SCHEMA, "status": status,
            "case_id": candidate["case_id"], "policy": candidate["policy"],
            "candidate": ref(candidate_path), "approval": ref(Path(approval_path)),
            "protocol": ref(protocol_path), "registered_seed": registered_seed,
            "lease_id": lease["leaseId"], "corrections": corrections,
            "training_mode": "fresh_restart", "source_checkpoint_loaded": False,
            "source_checkpoint_sha256": candidate["source_checkpoint_sha256"],
            "training": deepcopy(training), "fixed26": {
                "conditions": 26, "corrected_report": deepcopy(corrected),
                "reference_report_sha256": digest(reference_report),
                "execution": deepcopy(evaluation)},
            "recovery": deepcopy(verdict), "automatic_retry": False,
            "next_step": "stop_no_epoch3_implemented",
        }
        receipt_path = out / "receipt.json"
        write_new(receipt_path, receipt)
        result = {"schema": SECOND_RESULT_SCHEMA, "status": "complete",
                  "receipt": ref(receipt_path), "outcome": status,
                  "automatic_retry": False, "epoch3_allowed": False,
                  "overall_goal_complete": status == "recovered"}
        write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as error:
        if not (out / "failure.json").exists():
            write_new(out / "failure.json", {"status": "invalid",
                      "error_type": type(error).__name__, "error": str(error),
                      "automatic_retry": False})
        raise
