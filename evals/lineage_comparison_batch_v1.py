"""고정 10-case × 3-policy lineage epoch-1 비교 batch controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_diagnosis_epoch_v1 as E
import lineage_evaluation_gate_v1 as GATE
import lineage_nim_transport_v1 as T
import lineage_probe_backend_v1 as P
import lineage_selection_inputs_v1 as INPUT
import lineage_source_batch_v1 as SOURCE

SCHEMA = "lineage_comparison_batch_protocol_v1"
MANIFEST_SCHEMA = "lineage_comparison_batch_manifest_v1"
RESULT_SCHEMA = "lineage_comparison_batch_result_v1"
POLICIES = ("registry_next", "fixed_eligible", "model_adaptive")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_FILES = (
    "evals/lineage_comparison_batch_v1.py",
    "tests/test_lineage_comparison_batch_v1.py",
    "evals/lineage_diagnosis_epoch_v1.py",
    "evals/lineage_evaluation_gate_v1.py",
    "evals/lineage_selection_inputs_v1.py",
    "evals/lineage_source_batch_v1.py",
)
ARM_EXPECTED = {
    "NONE": frozenset(), "F1": frozenset({"reward"}), "F2": frozenset({"actuator"}),
    "F3": frozenset({"exploration"}), "F5": frozenset({"physics"}),
    "COMPOUND": frozenset({"reward", "optimizer_update_count"}),
}
EXECUTION_REQUEST_BUDGET_S = 1200


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def validate_arm_expected() -> dict[str, frozenset[str]]:
    """집계용 정답 subtype이 실제 동결 registry 밖으로 벗어나지 않게 한다."""
    registered = set(E.C.RC.registry()["mechanisms"])
    found = set().union(*ARM_EXPECTED.values())
    _require(found <= registered, "ARM_EXPECTED에 registry 범위 밖 subtype이 있다")
    _require(ARM_EXPECTED["COMPOUND"] == frozenset({"reward", "optimizer_update_count"}),
             "COMPOUND의 R1 canonical subtype이 다르다")
    return deepcopy(ARM_EXPECTED)


def validate_paired_watchdogs(probe_protocol: dict) -> dict[str, int]:
    """두 role child의 최악 watchdog 합을 논리 request 상한 안에 묶는다."""
    watchdogs = probe_protocol.get("watchdogs_s")
    _require(isinstance(watchdogs, dict) and set(watchdogs) == {"legacy", "optimizer"},
             "paired probe watchdogs_s 필드가 다르다")
    for backend, seconds in watchdogs.items():
        _require(type(seconds) is int and seconds > 0,
                 f"{backend} paired role watchdog는 양의 정수여야 한다")
        _require(2 * seconds <= EXECUTION_REQUEST_BUDGET_S,
                 f"{backend} case/reference watchdog 합이 논리 request 1200초 상한을 넘는다")
    return deepcopy(watchdogs)


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    try:
        value = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("공개 protocol component는 repo 내부여야 한다") from error
    _require(".." not in Path(value).parts and not Path(value).is_absolute(), "안전하지 않은 상대경로")
    return value


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git commit/blob 결합을 확인하지 못했다")
    return result.stdout


def _component_source_bindings(values: dict[str, dict]) -> dict[str, str]:
    """실행 component protocol이 고정한 repo source closure를 현재 파일과 대조한다."""
    bound: dict[str, str] = {}
    for component, value in values.items():
        sources = value.get("source_sha256_lf")
        _require(isinstance(sources, dict) and sources,
                 f"{component} component의 source closure가 없다")
        for name, expected in sources.items():
            path = ROOT / name
            _require(HEX64.fullmatch(str(expected)) is not None and path.is_file()
                     and _sha(path, lf=True) == expected,
                     f"{component} component source가 바뀌었다: {name}")
            prior = bound.setdefault(name, expected)
            _require(prior == expected, f"component 사이 source SHA가 충돌한다: {name}")
    return bound


def latin_plan(cases: list[dict]) -> list[dict]:
    _require(len(cases) == 10 and len({row["case_id"] for row in cases}) == 10,
             "Latin plan은 고정 10 case가 필요하다")
    rows = []
    for case_index, case in enumerate(cases):
        order = POLICIES[case_index % 3:] + POLICIES[:case_index % 3]
        for policy_index, policy in enumerate(order):
            rows.append({"order": len(rows) + 1, "case_index": case_index + 1,
                         "policy_index": policy_index + 1, "case_id": case["case_id"],
                         "policy": policy})
    return rows


def transport_plan(cases: list[dict]) -> dict:
    slots = [{"slot_id": f"{case['case_id']}.model.{decision}", "case_id": case["case_id"]}
             for case in cases for decision in range(1, 4)]
    return {"case_ids": [row["case_id"] for row in cases], "logical_slots": slots,
            "global_request_cap": 90, "per_slot_max": 3}


def prepare_protocol(source_protocol: Path, transport_protocol: Path,
                     probe_protocol: Path) -> dict:
    """절대경로와 자기 commit 필드가 없는 공개 비교 protocol을 만든다."""
    source_path, transport_path = Path(source_protocol).resolve(), Path(transport_protocol).resolve()
    validate_arm_expected()
    probe_path = Path(probe_protocol).resolve()
    source = SOURCE.validate_protocol(_read(source_path))
    transport, probe = _read(transport_path), _read(probe_path)
    paired_watchdogs = validate_paired_watchdogs(probe)
    view_protocol = _read(E.C.V.PROTOCOL)
    controller_protocol = _read(E.C.PROTOCOL)
    _component_source_bindings({"nim_transport": transport, "paired_probe": probe,
                                "selection_view": view_protocol,
                                "selection_controller": controller_protocol})
    cases = deepcopy(source["cases"])
    expected_transport = transport_plan(cases)
    _require(transport.get("schema") == T.SCHEMA and transport.get("plan") == expected_transport,
             "별도 transport protocol의 30-slot/90-POST 계획이 다르다")
    _require(probe.get("schema") == P.PROTOCOL_SCHEMA,
             "paired probe component protocol schema가 다르다")
    components = {
        "source_collection": {"path": _relative(source_path), "sha256_lf": _sha(source_path, lf=True)},
        "nim_transport": {"path": _relative(transport_path), "sha256_lf": _sha(transport_path, lf=True)},
        "paired_probe": {"path": _relative(probe_path), "sha256_lf": _sha(probe_path, lf=True)},
        "selection_view": {"path": _relative(E.C.V.PROTOCOL),
                           "sha256_lf": _sha(E.C.V.PROTOCOL, lf=True)},
        "selection_controller": {"path": _relative(E.C.PROTOCOL),
                                 "sha256_lf": _sha(E.C.PROTOCOL, lf=True)},
    }
    value = {
        "schema": SCHEMA, "phase": "epoch1_diagnosis_confirmation_only",
        "cases": cases, "policies": list(POLICIES), "latin_execution_plan": latin_plan(cases),
        "budget": {"max_decisions_per_policy_case": 3, "execution_request_budget_s": 1200,
                   "paired_probe_watchdogs_s": paired_watchdogs,
                   "paired_roles_per_logical_request": 2,
                   "model_logical_slots_per_case": 3, "model_logical_slots_total": 30,
                   "model_post_cap_per_slot": 3, "model_post_cap_total": 90,
                   "api_cap_scope": "new_independent_batch_not_combined_with_prior_72_or_120_drafts",
                   "replacement_seeds": 0, "selective_retries": 0},
        "component_protocols": components,
        "runtime_source_sha256_lf": deepcopy(source["runtime_source_sha256_lf"]),
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "transport_dag": "batch -> transport protocol; transport protocol does not hash batch protocol",
        "error_rules": {"model_invalid_unknown_hold_rule_fallback": False,
                        "pending_auto_resend": False, "resume_supported": False,
                        "output": "fresh_write_once_fail_closed"},
        "aggregation": {"denominator_cases_per_policy": 10,
                        "excluded_cases_remain_in_denominator": True,
                        "compound_confirmation": "first_registered_component_is_partial_not_full_cause"},
        "claims": {"correction": False, "training": False, "epoch2": False,
                   "causality": False, "generalization": False},
        "outcome_wording": {
            "complete": "고정 10 case의 epoch1 진단·확인 결과를 세 정책의 같은 입력 짝으로 보존했다.",
            "invalid": "입력·계보·pending 또는 실행 계약이 불완전해 자동 재전송 없이 중단했다.",
            "excluded": "고정 분모 10에는 남기되 gate 사유와 함께 진단 실행에서 제외했다.",
        },
    }
    _require("source_commit" not in value and not any(Path(row["path"]).is_absolute()
             for row in components.values()), "공개 protocol에 commit 또는 절대경로가 들어갔다")
    return value


def verify_protocol(protocol_path: Path, anchor: str) -> dict:
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "execution anchor 형식이 잘못됐다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    components = value.get("component_protocols", {})
    _require(isinstance(components, dict), "component protocol 표가 없다")
    expected = prepare_protocol(*[ROOT / components[name]["path"]
                                  for name in ("source_collection", "nim_transport", "paired_probe")])
    _require(value == expected, "public batch protocol 또는 component SHA가 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"],
             **{row["path"]: row["sha256_lf"] for row in components.values()}}
    component_sources = _component_source_bindings({
        name: _read(ROOT / components[name]["path"])
        for name in ("nim_transport", "paired_probe", "selection_view", "selection_controller")})
    for name, expected_sha in component_sources.items():
        _require(name not in bound or bound[name] == expected_sha,
                 f"batch/component source SHA가 충돌한다: {name}")
        bound[name] = expected_sha
    for name, expected_sha in bound.items():
        current = ROOT / name
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 batch source/component가 protocol과 다르다: {name}")
        committed = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected_sha,
                 f"execution anchor blob이 protocol과 다르다: {name}")
    return value


def _source_records(source_result: dict, source_protocol: dict) -> dict[str, dict]:
    jobs = {row["job_id"]: row for row in source_protocol["training_jobs"]}
    rows = [row for row in source_result.get("records", [])
            if isinstance(row, dict) and row.get("purpose") == "source"]
    by_job = {row.get("job_id"): row for row in rows}
    _require(source_result.get("status") == "complete"
             and source_result.get("source_jobs_completed") == 18
             and len(rows) == 18 and set(by_job) == set(jobs),
             "source 완료 result의 18개 job이 다르다")
    for job_id, job in jobs.items():
        checkpoint = by_job[job_id].get("checkpoint", {})
        _require(by_job[job_id].get("arm") == job["arm"]
                 and by_job[job_id].get("seed") == job["seed"]
                 and HEX64.fullmatch(str(checkpoint.get("sha256", ""))) is not None,
                 f"source result job/seed/checkpoint가 다르다: {job_id}")
    return by_job


def _collected(index: dict, source_protocol: dict, records: dict[str, dict]) -> dict[str, dict]:
    jobs = {row["job_id"]: row for row in source_protocol["training_jobs"]}
    entries = index.get("jobs")
    _require(index.get("schema") == "lineage_collected_index_v1"
             and isinstance(entries, dict) and set(entries) == set(jobs),
             "collected index는 고정 18 job을 모두 가져야 한다")
    checked = {}
    fields = {"path", "sha256", "job_id", "arm", "seed", "checkpoint_sha256", "telemetry_sha256"}
    for job_id, job in jobs.items():
        entry = entries[job_id]
        path = Path(entry.get("path", "")).resolve() if isinstance(entry, dict) else Path()
        _require(isinstance(entry, dict) and set(entry) == fields and path.is_file()
                 and _sha(path) == entry["sha256"] and entry["job_id"] == job_id
                 and entry["arm"] == job["arm"] and entry["seed"] == job["seed"],
                 f"collected index 파일/job 결합이 다르다: {job_id}")
        value, _telemetry = INPUT._checked_collected(path)
        source = value["private_source_map"]
        binding = value["source_binding"]
        _require(source["job_id"] == job_id and source["arm"] == job["arm"]
                 and source["seed"] == job["seed"]
                 and binding["checkpoint_sha256"] == records[job_id]["checkpoint"]["sha256"]
                 == entry["checkpoint_sha256"]
                 and binding["telemetry_sha256"] == entry["telemetry_sha256"],
                 f"collected source/CP/telemetry 결합이 다르다: {job_id}")
        checked[job_id] = {**deepcopy(entry), "path": str(path)}
    return checked


def _inputs(source_protocol: dict, source_result_path: Path, gate_path: Path,
            index_path: Path) -> tuple[dict, dict, dict]:
    source_result, gate, index = _read(source_result_path), _read(gate_path), _read(index_path)
    records = _source_records(source_result, source_protocol)
    collected = _collected(index, source_protocol, records)
    cases = source_protocol["cases"]
    gate_rows = {row.get("case_id"): row for row in gate.get("cases", []) if isinstance(row, dict)}
    _require(gate.get("schema") == GATE.SCHEMA and gate.get("status") == "complete"
             and set(gate_rows) == {row["case_id"] for row in cases},
             "고정 평가 gate의 10 case가 source protocol과 다르다")
    for case in cases:
        row = gate_rows[case["case_id"]]
        _require(all(row.get(key) == case[key] for key in
                     ("case_id", "seed", "arm", "source_job", "reference_job")),
                 f"gate와 source case가 다르다: {case['case_id']}")
    return records, collected, gate_rows


def prepare(out: Path, protocol_path: Path, source_result: Path, gate_path: Path,
            collected_index: Path, *, lab: Path, source_collection_anchor: str,
            execution_anchor: str, transport_root: Path, probe_protocol: Path,
            transport_protocol: Path) -> dict:
    """실제 source·gate·18 collected·component 경로를 private manifest로 봉인한다."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = verify_protocol(Path(protocol_path), execution_anchor)
    source_protocol_path = ROOT / protocol["component_protocols"]["source_collection"]["path"]
    source_protocol = SOURCE.validate_protocol(_read(source_protocol_path))
    _git("merge-base", "--is-ancestor", source_collection_anchor, execution_anchor)
    _require(HEX40.fullmatch(source_collection_anchor) is not None
             and source_collection_anchor != execution_anchor,
             "source collection anchor와 이후 execution anchor를 분리해야 한다")
    source_result, gate_path, collected_index = (Path(source_result).resolve(), Path(gate_path).resolve(),
                                                  Path(collected_index).resolve())
    _inputs(source_protocol, source_result, gate_path, collected_index)
    source_intent = _read(source_result.parent / "intent.json")
    _require(source_intent.get("anchor") == source_collection_anchor
             and source_intent.get("protocol_sha256_lf") == _sha(source_protocol_path, lf=True),
             "source 완료 result의 collection anchor/protocol intent가 다르다")
    _require(SOURCE.verify(Path(lab).resolve(), source_collection_anchor) == source_protocol,
             "source collection runtime/anchor 재검증이 다르다")
    registered_transport = ROOT / protocol["component_protocols"]["nim_transport"]["path"]
    registered_probe = ROOT / protocol["component_protocols"]["paired_probe"]["path"]
    _require(Path(transport_protocol).resolve() == registered_transport.resolve()
             and Path(probe_protocol).resolve() == registered_probe.resolve(),
             "private component 경로가 public protocol 등록 파일과 다르다")
    P.verify_protocol(registered_probe, Path(lab).resolve(), execution_anchor)
    transport_root = Path(transport_root).resolve()
    session = T._session(transport_root)
    _require(session["plan"] == transport_plan(source_protocol["cases"])
             and session["protocol_sha256_lf"] == _sha(Path(transport_protocol), lf=True)
             and session["registration_anchor"] == execution_anchor,
             "transport session의 30-slot/90-cap protocol이 다르다")
    manifest = {
        "schema": MANIFEST_SCHEMA, "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": _sha(Path(protocol_path)), "source_result_path": str(source_result),
        "source_result_sha256": _sha(source_result), "gate_path": str(gate_path),
        "gate_sha256": _sha(gate_path), "collected_index_path": str(collected_index),
        "collected_index_sha256": _sha(collected_index), "lab": str(Path(lab).resolve()),
        "source_collection_anchor": source_collection_anchor, "execution_anchor": execution_anchor,
        "transport_root": str(transport_root), "transport_protocol": str(Path(transport_protocol).resolve()),
        "transport_protocol_sha256_lf": _sha(Path(transport_protocol), lf=True),
        "probe_protocol": str(Path(probe_protocol).resolve()),
        "probe_protocol_sha256_lf": _sha(Path(probe_protocol), lf=True),
        "output_root": str(out), "max_decisions": 3, "execution_budget_s": 1200,
        "execution_plan": protocol["latin_execution_plan"], "transport_plan": session["plan"],
        "resume_supported": False, "correction_training_epoch2": False,
    }
    _write_new(out / "manifest.json", manifest)
    _write_new(out / "intent.json", {"mode": "epoch1_comparison", "api_executed": False,
                                      "gpu_executed": False, "resume_supported": False})
    return deepcopy(manifest)


def _validate_manifest(path: Path) -> tuple[dict, dict, dict, dict]:
    manifest = _read(path)
    required = {"schema", "protocol_path", "protocol_sha256", "source_result_path",
                "source_result_sha256", "gate_path", "gate_sha256", "collected_index_path",
                "collected_index_sha256", "lab", "source_collection_anchor", "execution_anchor",
                "transport_root", "transport_protocol", "transport_protocol_sha256_lf",
                "probe_protocol", "probe_protocol_sha256_lf", "output_root", "max_decisions",
                "execution_budget_s", "execution_plan", "transport_plan", "resume_supported",
                "correction_training_epoch2"}
    _require(set(manifest) == required and manifest.get("schema") == MANIFEST_SCHEMA,
             "private batch manifest 필드가 다르다")
    for field, sha_field, lf in (
            ("protocol_path", "protocol_sha256", False),
            ("source_result_path", "source_result_sha256", False),
            ("gate_path", "gate_sha256", False),
            ("collected_index_path", "collected_index_sha256", False),
            ("transport_protocol", "transport_protocol_sha256_lf", True),
            ("probe_protocol", "probe_protocol_sha256_lf", True)):
        file = Path(manifest[field])
        _require(file.is_file() and _sha(file, lf=lf) == manifest[sha_field],
                 f"manifest 입력 SHA가 바뀌었다: {field}")
    protocol = verify_protocol(Path(manifest["protocol_path"]), manifest["execution_anchor"])
    _require(manifest["max_decisions"] == 3 and manifest["execution_budget_s"] == 1200
             and manifest["execution_plan"] == protocol["latin_execution_plan"]
             and manifest["transport_plan"] == transport_plan(protocol["cases"])
             and manifest["resume_supported"] is False
             and manifest["correction_training_epoch2"] is False,
             "manifest 실행 계획·예산·범위가 다르다")
    source_protocol = SOURCE.validate_protocol(_read(
        ROOT / protocol["component_protocols"]["source_collection"]["path"]))
    _records, collected, gate_rows = _inputs(source_protocol, Path(manifest["source_result_path"]),
                                              Path(manifest["gate_path"]),
                                              Path(manifest["collected_index_path"]))
    return manifest, protocol, collected, gate_rows


def _initial_view(case: dict, collected: dict, manifest: dict, output: Path) -> str:
    envelope = INPUT.make_selection_packet(Path(collected[case["source_job"]]["path"]),
                                           Path(collected[case["reference_job"]]["path"]),
                                           budget_left=3, execution_budget_s=1200)
    digests = []
    for policy in POLICIES:
        snapshot = E.C.V.build_snapshot(envelope["packet"],
                                        f"view_{case['case_id'].replace('-', '_')}_{policy}", policy,
                                        E.C.V.file_digest(E.C.V.PROTOCOL),
                                        output / "view-ledgers" / f"{case['case_id']}_{policy}.jsonl")
        digests.append(snapshot["view_digest"])
    _require(len(set(digests)) == 1, "세 정책의 fresh public view가 다르다")
    return digests[0]


def _classification(case: dict, result: dict) -> dict:
    validate_arm_expected()
    subtype = result.get("confirmed_subtype")
    expected = ARM_EXPECTED[case["arm"]]
    confirmed = isinstance(subtype, str) and bool(subtype)
    partial = case["arm"] == "COMPOUND" and subtype in expected
    return {"confirmed": confirmed, "correct_confirmation": confirmed and subtype in expected,
            "false_confirmation": confirmed and subtype not in expected,
            "compound_partial": partial,
            "full_cause_confirmed": confirmed and subtype in expected and case["arm"] != "COMPOUND",
            "none_intervention": case["arm"] == "NONE" and result.get("decision_count", 0) > 0}


def _sum_known(results: list[dict], path: tuple[str, ...]) -> dict:
    values = []
    for result in results:
        value: Any = result
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        values.append(value)
    known = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {"total": sum(known) if len(known) == len(values) else None,
            "known_subtotal": sum(known), "unknown_rows": len(values) - len(known)}


def _partial_cost(results: list[dict], total_path: tuple[str, ...],
                  subtotal_path: tuple[str, ...]) -> dict:
    totals, subtotals = [], []
    for result in results:
        total: Any = result
        subtotal: Any = result
        for key in total_path:
            total = total.get(key) if isinstance(total, dict) else None
        for key in subtotal_path:
            subtotal = subtotal.get(key) if isinstance(subtotal, dict) else None
        totals.append(total)
        subtotals.append(subtotal if isinstance(subtotal, (int, float)) and not isinstance(subtotal, bool)
                         else total)
    known_totals = [value for value in totals
                    if isinstance(value, (int, float)) and not isinstance(value, bool)]
    known_subtotals = [value for value in subtotals
                       if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {"total": sum(known_totals) if len(known_totals) == len(totals) else None,
            "known_subtotal": sum(known_subtotals),
            "unknown_rows": len(totals) - len(known_totals)}


def _paired_stats(result: dict) -> dict:
    stats = {"attempted": 0, "succeeded": 0, "unknown": 0, "unverified": 0}
    for step in result.get("steps", []):
        receipts = step.get("paired_receipts", {}) if isinstance(step, dict) else {}
        for role in ("reference", "case"):
            receipt = receipts.get(role)
            if not isinstance(receipt, dict):
                stats["unknown"] += 1
                stats["unverified"] += 1
                continue
            attempted = isinstance(receipt.get("claim"), dict) or bool(receipt.get("execution"))
            if attempted:
                stats["attempted"] += 1
            else:
                stats["unverified"] += 1
            if attempted and receipt.get("status") == "complete":
                stats["succeeded"] += 1
            else:
                stats["unknown"] += 1
    return stats


def _owned_wall(results: list[dict]) -> dict:
    known = []
    unknown_roles = 0
    for result in results:
        wall = result.get("wall", {})
        value = wall.get("probe_child_known_s")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            known.append(value)
        unknown_roles += len(wall.get("probe_child_unknown_roles", []))
    return {"total": sum(known) if len(known) == len(results) and unknown_roles == 0 else None,
            "known_subtotal": sum(known), "unknown_rows": len(results) - len(known),
            "unknown_roles": unknown_roles}


def _summary(rows: list[dict]) -> dict:
    results = [row["result"] for row in rows]
    by_policy = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        paired = [_paired_stats(row["result"]) for row in selected]
        by_policy[policy] = {
            "denominator_cases": 10,
            "excluded": sum(row["gate_label"] in E.STOP_GATE_LABELS for row in selected),
            "excluded_reasons": {label: sum(row["gate_label"] == label for row in selected)
                                 for label in sorted(E.STOP_GATE_LABELS)},
            "confirmed": sum(row["classification"]["confirmed"] for row in selected),
            "correct_confirmations": sum(row["classification"]["correct_confirmation"] for row in selected),
            "false_confirmations": sum(row["classification"]["false_confirmation"] for row in selected),
            "compound_partial": sum(row["classification"]["compound_partial"] for row in selected),
            "none_interventions": sum(row["classification"]["none_intervention"] for row in selected),
            "hold": sum(row["result"].get("end_state") == "HOLD" for row in selected),
            "unknown": sum(row["result"].get("hold_reason") == "probe_outcome_unknown"
                           or any(state in {"unknown", "infra_error"}
                                  for state in row["result"].get("model_validity", []))
                           for row in selected),
            "invalid": sum("invalid" in row["result"].get("model_validity", []) for row in selected),
            "repair_unsupported": sum(row["result"].get("repair_supported") is False
                                      for row in selected),
            "logical_probes": sum(row["result"].get("decision_count", 0) for row in selected),
            "paired_backend_attempts": sum(row["attempted"] for row in paired),
            "paired_backend_successes": sum(row["succeeded"] for row in paired),
            "paired_backend_unknown": sum(row["unknown"] for row in paired),
            "paired_backend_unverified": sum(row["unverified"] for row in paired),
            "api_requests": sum(row["result"].get("requests", 0) for row in selected),
        }
    return {"rows": len(rows), "denominator_case_policy_pairs": 30, "by_policy": by_policy,
            "costs": {
                "api_requests": sum(row.get("requests", 0) for row in results),
                "tokens": _partial_cost(results, ("tokens", "total_tokens"),
                                        ("tokens", "known_subtotal", "total_tokens")),
                "model_wall_s": _partial_cost(results, ("wall", "model_wall_s"),
                                               ("wall", "known_model_wall_subtotal_s")),
                "owned_wall_s": _owned_wall(results),
                "parent_wall_s": _sum_known(results, ("wall", "probe_parent_known_s")),
            }}


def execute(manifest_path: Path, *, epoch_runner=E.run_epoch,
            probe_executor=None, complete_factory=None) -> dict:
    """사전등록 30행을 단회 실행한다. 중단된 output의 resume는 명시적으로 거부한다."""
    manifest, protocol, collected, gate_rows = _validate_manifest(Path(manifest_path))
    root = Path(manifest["output_root"])
    execution = root / "execution"
    if execution.exists():
        raise RuntimeError("resume_not_supported: 기존 중간 결과를 보존하고 자동 재전송하지 않는다")
    execution.mkdir()
    _write_new(execution / "started.json", {"status": "running", "resume_supported": False,
                                             "planned_rows": 30})
    synthetic = epoch_runner is not E.run_epoch or probe_executor is not None or complete_factory is not None
    rows = []
    try:
        cases = {row["case_id"]: row for row in protocol["cases"]}
        views = {case_id: _initial_view(case, collected, manifest, execution)
                 for case_id, case in cases.items()}
        for planned in manifest["execution_plan"]:
            case, policy = cases[planned["case_id"]], planned["policy"]
            row_root = execution / "runs" / f"{planned['order']:02d}_{case['case_id']}_{policy}"
            slots = tuple(f"{case['case_id']}.model.{index}" for index in range(1, 4))
            result = epoch_runner(
                row_root, Path(collected[case["source_job"]]["path"]),
                Path(collected[case["reference_job"]]["path"]), policy,
                run_id=f"batch_{case['case_id'].replace('-', '_')}_{policy}",
                lab=Path(manifest["lab"]), probe_protocol=Path(manifest["probe_protocol"]),
                anchor=manifest["execution_anchor"], gate_path=Path(manifest["gate_path"]),
                gate_sha256=manifest["gate_sha256"], case_id=case["case_id"],
                transport_root=Path(manifest["transport_root"]), slot_ids=slots,
                max_decisions=3, execution_budget_s=1200, probe_executor=probe_executor,
                complete_factory=complete_factory)
            result_path = row_root / "result.json"
            _require(result_path.is_file() and _read(result_path) == result,
                     "epoch runner result 파일이 반환값과 다르다")
            row = {**deepcopy(planned), "arm": case["arm"],
                   "gate_label": gate_rows[case["case_id"]]["label"],
                   "fresh_view_digest": views[case["case_id"]], "result": deepcopy(result),
                   "result_path": str(result_path), "result_sha256": _sha(result_path),
                   "classification": _classification(case, result)}
            rows.append(row)
            _write_new(execution / "rows" / f"{planned['order']:02d}.json", row)
        for case in protocol["cases"]:
            paired = [row for row in rows if row["case_id"] == case["case_id"]]
            _require(len(paired) == 3 and len({row["fresh_view_digest"] for row in paired}) == 1,
                     "case별 세 정책의 fresh input 짝이 불완전하다")
            _write_new(execution / "case-pairs" / f"{case['case_id']}.json", {
                "case_id": case["case_id"], "fresh_view_digest": paired[0]["fresh_view_digest"],
                "policies": {row["policy"]: {"result_path": row["result_path"],
                                               "result_sha256": row["result_sha256"]}
                             for row in paired}})
        result = {"schema": RESULT_SCHEMA, "status": "complete",
                  "evidence_scope": "synthetic_cpu_fixture" if synthetic else "actual_backend",
                  "rows": rows, "summary": _summary(rows), "correction_executed": False,
                  "training_executed": False, "epoch2_registered": False,
                  "resume_supported": False}
        _write_new(execution / "result.json", result)
        _write_new(execution / "completed.json", {"status": "complete", "rows": 30,
                                                   "result_sha256": _sha(execution / "result.json")})
        return deepcopy(result)
    except BaseException as error:
        failure = execution / "failed.json"
        if not failure.exists():
            _write_new(failure, {"status": "invalid", "completed_rows": len(rows),
                                 "error_type": type(error).__name__, "error": str(error),
                                 "resume_supported": False, "automatic_resend": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    public = commands.add_parser("prepare-protocol")
    public.add_argument("--source-protocol", type=Path, required=True)
    public.add_argument("--transport-protocol", type=Path, required=True)
    public.add_argument("--probe-protocol", type=Path, required=True)
    public.add_argument("--out", type=Path, required=True)
    prep = commands.add_parser("prepare")
    for name in ("protocol", "source-result", "gate", "collected-index", "lab", "transport-root",
                 "probe-protocol", "transport-protocol", "out"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--source-collection-anchor", required=True)
    prep.add_argument("--execution-anchor", required=True)
    run = commands.add_parser("execute")
    run.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.source_protocol, args.transport_protocol,
                                               args.probe_protocol))
    elif args.command == "prepare":
        prepare(args.out, args.protocol, args.source_result, args.gate, args.collected_index,
                lab=args.lab, source_collection_anchor=args.source_collection_anchor,
                execution_anchor=args.execution_anchor, transport_root=args.transport_root,
                probe_protocol=args.probe_protocol, transport_protocol=args.transport_protocol)
    else:
        execute(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
