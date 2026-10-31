"""등록된 전향 source를 세 진단 정책으로 비교하는 CPU parent 실행기다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import evidence_guided_probe_host_v1 as HOST
import lineage_lease_context_v2 as LEASE
import lineage_nim_transport_v4 as T4
import prospective_agentic_comparison_v1 as COMP
import prospective_diagnosis_inputs_v2 as INPUT
import prospective_diagnosis_training_v1 as TRAIN
import run_prospective_diagnosis_training_v1 as SOURCE
import runtime_observation_probe_v1 as OBS
import runtime_observation_view_v1 as RV
import selection_comparison_controller_v1 as C

SCHEMA = "prospective_agentic_comparison_protocol_v1"
PREFLIGHT_SCHEMA = "prospective_agentic_comparison_preflight_v1"
RESULT_SCHEMA = "prospective_agentic_comparison_result_v1"
OWNER = SOURCE.OWNER
BANDS = ROOT / C.D1.BANDS
BANDS_SHA256 = "7589a132d4e1a55e159527389a8b635a1766472ecde558da0a1a704c286f3f23"
MAX_WALL_S = 14400
MAX_REQUESTS = 96
MAX_PAIRED_PROBES = 72
MAX_LEGACY_ROLE_EXECUTIONS = 144

SOURCE_FILES = (
    "evals/selection_comparison_controller_v1.py",
    "evals/runtime_observation_view_v1.py",
    "evals/runtime_observation_probe_v1.py",
    "evals/evidence_guided_probe_host_v1.py",
    "evals/prospective_diagnosis_inputs_v1.py",
    "evals/prospective_diagnosis_inputs_v2.py",
    "evals/evidence_guided_selector_v1.py",
    "tests/test_evidence_guided_selector_v1.py",
    "evals/diagnostic_fact_ledger_v1.py",
    "tests/test_diagnostic_fact_ledger_v1.py",
    "evals/recorded_optimizer_probe_v1.py",
    "tests/test_recorded_optimizer_probe_v1.py",
    "evals/prospective_agentic_comparison_v1.py",
    "tests/test_prospective_agentic_comparison_v1.py",
    "evals/run_prospective_agentic_comparison_v1.py",
    "tests/test_run_prospective_agentic_comparison_v1.py",
    "evals/lineage_nim_transport_v1.py",
    "evals/lineage_nim_transport_v2.py",
    "evals/lineage_nim_transport_v3.py",
    "evals/lineage_nim_transport_v4.py",
    "evals/core_seed_guard_v1.py",
    "evals/lineage_lease_context_v2.py",
    "evals/immutable_git_proof_cache_v1.py",
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _write(path: Path, value: object) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def _public_ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("공개 protocol 입력은 저장소 안에 있어야 한다") from error
    return {"path": relative, "sha256_lf": _sha(path, lf=True)}


def _git(*args: str) -> bytes:
    return subprocess.run(("git", *args), cwd=ROOT, check=True,
                          capture_output=True).stdout


def _study(study_path: Path) -> dict[str, Any]:
    study = _read(study_path)
    comparison = study.get("comparison")
    _require(study.get("schema") == "evidence_guided_study_v1", "study schema 불일치")
    _require(study.get("seeds") == [2320, 2321]
             and study.get("categories") == ["NONE", "F1", "R1_ONLY", "COMPOUND"],
             "등록 seed/category 불일치")
    _require(isinstance(comparison, dict)
             and comparison.get("policies") == list(COMP.POLICIES)
             and comparison.get("paired_probe_budget") == COMP.PROBE_BUDGET
             and comparison.get("model_action_budget") == COMP.MODEL_ACTION_BUDGET
             and comparison.get("semantic_feedback_retries_per_action") == 1
             and comparison.get("global_api_request_cap") == MAX_REQUESTS,
             "study 비교 상한 불일치")
    _require(study.get("repair_followup", {}).get("automatic_repair") is False,
             "비교 실행은 교정을 허용하지 않는다")
    return study


def prepare_protocol(study_path: Path, training_protocol_path: Path,
                     probe_protocol_path: Path, view_protocol_path: Path,
                     transport_protocol_path: Path) -> dict[str, Any]:
    """현재 source closure와 다섯 등록 protocol을 공개 후보로 봉인한다."""
    study = _study(study_path)
    training = _read(training_protocol_path)
    _require(training.get("schema") == TRAIN.SCHEMA, "training protocol schema 불일치")
    _require(training.get("seeds") == study["seeds"]
             and training.get("categories") == study["categories"], "study/source 설계 불일치")
    transport = _read(transport_protocol_path)
    _require(transport.get("schema") == T4.SCHEMA, "transport V4 protocol이 아니다")
    plan = transport.get("plan")
    slots = [f"c{case:02d}.{policy}.d{action}"
             for case in range(1, 9)
             for policy in COMP.POLICIES[:2]
             for action in range(1, COMP.MODEL_ACTION_BUDGET + 1)]
    _require(isinstance(plan, dict) and plan.get("case_ids") == [f"c{i:02d}" for i in range(1, 9)]
             and plan.get("global_request_cap") == MAX_REQUESTS and plan.get("per_slot_max") == 3
             and [row.get("slot_id") for row in plan.get("logical_slots", [])] == slots,
             "transport V4 64-slot/global-cap 계획 불일치")
    _require(_read(probe_protocol_path).get("schema") == OBS.PROTOCOL_SCHEMA,
             "paired probe protocol schema 불일치")
    _require(_read(view_protocol_path).get("schema") == RV.SCHEMA,
             "runtime view protocol schema 불일치")
    training_ref = _public_ref(training_protocol_path)
    _require(study.get("source_protocol") == training_ref["path"],
             "study source_protocol 결합 불일치")
    for relative in SOURCE_FILES:
        _require((ROOT / relative).is_file(), f"source closure 파일이 없다: {relative}")
    return {
        "schema": SCHEMA,
        "purpose": "prospective_registered_evidence_guided_system_comparison",
        "protocols": {
            "study": _public_ref(study_path),
            "training": training_ref,
            "paired_probe": _public_ref(probe_protocol_path),
            "view": _public_ref(view_protocol_path),
            "transport_v4": _public_ref(transport_protocol_path),
        },
        "source_sha256_lf": {relative: _sha(ROOT / relative, lf=True)
                              for relative in SOURCE_FILES},
        "runtime_source_sha256_lf": deepcopy(training["runtime_source_sha256_lf"]),
        "design": {
            "opaque_cases": [f"c{index:02d}" for index in range(1, 9)],
            "policies": list(COMP.POLICIES),
            "policy_order": [list(COMP.policy_order(index)) for index in range(8)],
            "same_seed_none_reference": True,
            "healthy_gate_model_requests": 0,
            "healthy_gate_paired_probes": 0,
            "automatic_repair": False,
        },
        "bounds": {"total_wall_s": MAX_WALL_S, "global_api_requests": MAX_REQUESTS,
                   "paired_probes": MAX_PAIRED_PROBES,
                   "legacy_role_executions_worst_case": MAX_LEGACY_ROLE_EXECUTIONS},
        "bands": {"path": BANDS.relative_to(ROOT).as_posix(), "sha256": BANDS_SHA256},
        "claims": {"model_version_advantage": False, "recovery": False,
                   "unseen_family_generalization": False, "gpu_started_by_parent": False},
    }


def verify_protocol(path: Path, anchor: str, lab: Path) -> dict[str, Any]:
    """현재 bytes, immutable anchor blobs, runtime pins를 함께 재검증한다."""
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", anchor)), "anchor 형식 불일치")
    value = _read(path)
    refs = value.get("protocols")
    _require(value.get("schema") == SCHEMA and isinstance(refs, dict), "비교 protocol schema 불일치")
    refs = cast(dict[str, dict[str, str]], refs)
    paths = {name: ROOT / row["path"] for name, row in refs.items()}
    expected = prepare_protocol(paths["study"], paths["training"], paths["paired_probe"],
                                paths["view"], paths["transport_v4"])
    _require(value == expected, "비교 protocol이 현재 source closure와 다르다")
    _require(_sha(BANDS) == BANDS_SHA256, "fixed26 bands SHA 불일치")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    for row in refs.values():
        blob = _git("show", f"{anchor}:{row['path']}").replace(b"\r\n", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == row["sha256_lf"],
                 f"anchor protocol blob 불일치: {row['path']}")
    for relative, digest in value["source_sha256_lf"].items():
        blob = _git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == digest, f"anchor source blob 불일치: {relative}")
    TRAIN.verify_protocol(paths["training"], anchor, Path(lab).resolve())
    T4.verify_protocol(paths["transport_v4"], anchor)
    return deepcopy(value)


def _checked_ref(row: object, root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(row, dict) and set(row) == {"path", "sha256"}, "raw ref shape 불일치")
    row = cast(dict[str, str], row)
    path = Path(row["path"]).resolve()
    if root is not None:
        _require(path.is_relative_to(Path(root).resolve()), "raw ref가 source root를 이탈했다")
    _require(path.is_file() and _sha(path) == row["sha256"], "raw ref SHA 불일치")
    return path, _read(path)


def _source_cases(source_batch_result: Path, protocol: dict[str, Any], anchor: str
                  ) -> list[dict[str, Any]]:
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", anchor)), "comparison anchor 형식 불일치")
    batch_path = Path(source_batch_result).resolve()
    batch = _read(batch_path)
    _require(batch.get("schema") == SOURCE.BATCH_SCHEMA and batch.get("status") == "complete",
             "source batch가 complete가 아니다")
    source_anchor = batch.get("anchor")
    _require(isinstance(source_anchor, str) and bool(re.fullmatch(r"[0-9a-f]{40}", source_anchor))
             and batch.get("model_requests") == 0 and batch.get("repairs") == 0
             and batch.get("automatic_retry") is False,
             "source batch 실행 경계 불일치")
    study = _study(ROOT / protocol["protocols"]["study"]["path"])
    training_protocol = ROOT / protocol["protocols"]["training"]["path"]
    _require(batch.get("protocol") == _ref(training_protocol),
             "source batch training protocol ref 불일치")
    _require(batch.get("seeds") == study["seeds"] and batch.get("categories") == study["categories"],
             "source batch seed/category 불일치")
    stages = batch.get("stages")
    _require(isinstance(stages, list) and len(stages) == 9 and stages[0].get("stage") == "smoke",
             "source batch 17-job 결합이 완전하지 않다")
    stages = cast(list[dict[str, Any]], stages)
    source_root = batch_path.parent
    _, smoke = _checked_ref(stages[0].get("result"), source_root)
    _require(smoke.get("schema") == "prospective_diagnosis_smoke_v1"
             and smoke.get("status") == "complete", "source smoke result 불일치")
    expected_labels = [f"s{seed}-{category.lower()}" for seed in study["seeds"]
                       for category in study["categories"]]
    _require([row.get("stage") for row in stages[1:]] == expected_labels,
             "source stage 순서가 등록 설계와 다르다")
    cases: list[dict[str, Any]] = []
    for index, (stage, label) in enumerate(zip(stages[1:], expected_labels, strict=True), 1):
        result_path, result = _checked_ref(stage.get("result"), source_root)
        input_path, _ = _checked_ref(stage.get("input"), source_root)
        collected, metadata = INPUT.checked(input_path)
        _require(result.get("schema") == TRAIN.RESULT_SCHEMA and result.get("status") == "complete",
                 f"{label} case result 불일치")
        _require(result.get("claims") == {"recovery": False, "diagnosis": False,
                                          "model_advantage": False}, "source claim 불일치")
        _, training_result = _checked_ref(result.get("training"), source_root)
        _, manifest = _checked_ref(training_result.get("manifest"), source_root)
        _require(manifest.get("source_commit_sha") == source_anchor,
                 "source batch anchor와 training manifest가 다르다")
        _owned_execution(training_result.get("execution"), "training")
        category = study["categories"][(index - 1) % len(study["categories"])]
        seed = study["seeds"][(index - 1) // len(study["categories"])]
        _require(result.get("seed") == seed and result.get("category") == category,
                 "source private evaluator binding 불일치")
        checkpoint = result.get("checkpoint")
        _require(isinstance(checkpoint, dict), "source checkpoint shape 불일치")
        checkpoint = cast(dict[str, Any], checkpoint)
        _require(stage.get("checkpoint") == checkpoint
                 and collected.get("source_binding", {}).get("checkpoint_sha256")
                 == checkpoint.get("sha256"), "source checkpoint 결합 불일치")
        _require(stage.get("input_binding") == collected.get("source_binding"),
                 "source input binding 불일치")
        fixed_path, fixed = _checked_ref(result.get("fixed_report"), source_root)
        _require(result.get("report") == fixed, "fixed26 embedded report 불일치")
        _owned_execution(result.get("fixed_execution"), "fixed26")
        cases.append({"case_id": f"c{index:02d}", "seed": seed, "category": category,
                      "input": input_path, "result": result_path, "fixed": fixed_path,
                      "fixed_report": fixed, "checkpoint": deepcopy(checkpoint),
                      "metadata": metadata})
    return cases


def _gate(case: dict[str, Any], reference: dict[str, Any], bands: dict[str, Any]) -> dict[str, Any]:
    reference_healthy = (C.D1.V.R1.U.walks(reference["fixed_report"])
                         and C.D1.recovery(reference["fixed_report"],
                                            reference["fixed_report"], bands)["outcome"] == "recovered")
    source_outcome = C.D1.recovery(case["fixed_report"], reference["fixed_report"], bands)["outcome"]
    if not reference_healthy:
        return {"run": False, "reason": "unhealthy_reference", "source_outcome": source_outcome}
    if source_outcome == "recovered":
        return {"run": False, "reason": "healthy_gate", "source_outcome": source_outcome,
                "analysis_exclusion": (None if case["category"] == "NONE"
                                       else "source_failure_absent")}
    if source_outcome == "unknown":
        return {"run": False, "reason": "source_failure_absent", "source_outcome": source_outcome}
    return {"run": True, "reason": None, "source_outcome": source_outcome}


def _score(category: str, epoch: dict[str, Any]) -> dict[str, Any]:
    family = epoch.get("confirmed_family")
    expected = {"F1": {"reward"}, "R1_ONLY": {"optimizer_update_count"},
                "COMPOUND": {"reward", "optimizer_update_count"}}.get(category, set())
    correct = family in expected and epoch.get("decision_outcome") == "confirmed"
    partial = category == "COMPOUND" and correct
    compound_complete = False
    return {"expected_family": sorted(expected), "confirmed_family": family,
            "correct_confirmed_family": correct,
            "correct_complete_family": (correct if len(expected) == 1 else compound_complete),
            "compound_partial": partial, "compound_complete": compound_complete}


def _owned_execution(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and value.get("exit_code") == 0
             and value.get("natural_exit") is True
             and value.get("cleanup_verified") is True
             and value.get("timed_out") is False, f"{label} owned cleanup 불일치")
    return cast(dict[str, Any], value)


def _verify_current_protocol(protocol_path: Path, protocol: dict[str, Any], lab: Path) -> None:
    _require(_read(protocol_path) == protocol, "실행 중 comparison protocol이 바뀌었다")
    refs = cast(dict[str, dict[str, str]], protocol["protocols"])
    paths = {name: ROOT / row["path"] for name, row in refs.items()}
    _require(prepare_protocol(paths["study"], paths["training"], paths["paired_probe"],
                              paths["view"], paths["transport_v4"]) == protocol,
             "실행 중 source/protocol closure가 바뀌었다")
    for relative, expected in protocol["runtime_source_sha256_lf"].items():
        target = Path(lab).resolve() / relative
        _require(target.is_file() and _sha(target, lf=True) == expected,
                 f"실행 중 runtime pin이 바뀌었다: {relative}")
    _require(_sha(BANDS) == BANDS_SHA256, "실행 중 fixed26 bands가 바뀌었다")


def _context_verifier(
    *, protocol_path: Path, protocol: dict[str, Any], lab: Path,
    source_batch_result: Path, source_sha: str, lease_id: object,
    transport_root: Path, active_inputs: dict[str, dict[str, Any]], started: float,
) -> Callable[[dict[str, Any] | None], dict[str, Any]]:
    """현재 bytes와 role별 checkpoint를 모두 소비한 호출만 host에 승인한다."""
    def verify(context: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - started
        _require(math.isfinite(elapsed) and 0 <= elapsed < MAX_WALL_S, "comparison wall 상한 종료")
        _verify_current_protocol(protocol_path, protocol, lab)
        _require(_sha(source_batch_result) == source_sha, "실행 중 source batch가 바뀌었다")
        current = _lease()
        _require(current.get("lease_id") == lease_id, "실행 중 lease identity 변경")
        status = T4.status(Path(transport_root).resolve())
        requests = status.get("request_count", status.get("requests", 0))
        _require(type(requests) is int and 0 <= requests <= MAX_REQUESTS,
                 "transport global request cap 불일치")
        checked_role = None
        if context is not None:
            _require(context.get("schema") == "evidence_guided_probe_host_context_v1",
                     "host context schema 불일치")
            role = context.get("role")
            _require(role in ("case", "reference") and role in active_inputs,
                     "host context role이 현재 pair와 다르다")
            checked_role = cast(str, role)
            expected = active_inputs[checked_role]
            public, metadata = INPUT.checked(Path(expected["input"]))
            binding = context.get("request_spec", {}).get("source_binding", {}).get(checked_role)
            _require(isinstance(binding, dict)
                     and binding.get("checkpoint_sha256") == expected["checkpoint_sha256"]
                     == public["source_binding"]["checkpoint_sha256"],
                     f"{checked_role} context checkpoint 불일치")
            runtime_ref = metadata.get("runtime_facts", {}).get(checked_role)
            _require(isinstance(runtime_ref, dict)
                     and Path(runtime_ref["path"]).resolve() == Path(expected["runtime"]).resolve()
                     and _sha(Path(runtime_ref["path"])) == runtime_ref.get("sha256"),
                     f"{checked_role} context runtime ref 불일치")
            envelope, _facts = HOST.RECORDED.FACT.checked(
                Path(runtime_ref["path"]), expected_checkpoint_sha256=expected["checkpoint_sha256"])
            _require(envelope["source_binding"]["binding_digest"] == binding.get("binding_digest"),
                     f"{checked_role} context runtime binding 불일치")
        return {"approved": True, "elapsed_s": elapsed, "requests": requests,
                "checked_role": checked_role, "source_sha256": source_sha}
    return verify


def _validity(epoch_out: Path, epoch: dict[str, Any]) -> dict[str, int]:
    first, final, feedback = 0, 0, 0
    for index, step in enumerate(epoch.get("steps", []), 1):
        if "dialogue_sha256" not in step:
            continue
        path = Path(epoch_out) / f"step-{index:02d}" / "dialogue.json"
        _require(path.is_file() and _sha(path) == step["dialogue_sha256"],
                 "dialogue raw SHA 불일치")
        value = _read(path)
        calls = value.get("calls")
        _require(isinstance(calls, list) and 1 <= len(calls) <= 2, "dialogue call 수 불일치")
        calls = cast(list[dict[str, Any]], calls)
        first += int(calls[0].get("review", {}).get("status") == "valid")
        final += int(value.get("model_valid") is True)
        feedback += int(value.get("semantic_feedback_used") is True)
    return {"first_pass_valid_actions": first, "post_feedback_valid_actions": final,
            "semantic_feedback_actions": feedback}


def _lease() -> dict[str, Any]:
    lease = LEASE.lease_context()
    _require(lease.get("owner") == OWNER, "공용 lease owner 불일치")
    return lease


def preflight(protocol_path: Path, anchor: str, source_batch_result: Path, lab: Path,
              out: Path, transport_root: Path) -> dict[str, Any]:
    """실제 source를 소비하되 모델 요청과 probe 실행 없이 준비 상태만 기록한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "preflight out은 fresh 경로여야 한다")
    protocol = verify_protocol(protocol_path, anchor, lab)
    source_sha = _sha(source_batch_result)
    cases = _source_cases(source_batch_result, protocol, anchor)
    _require(_sha(source_batch_result) == source_sha, "preflight 중 source batch가 바뀌었다")
    source_anchor = _read(source_batch_result)["anchor"]
    status = T4.status(Path(transport_root).resolve())
    lease = _lease()
    value = {"schema": PREFLIGHT_SCHEMA, "status": "ready", "protocol": _ref(protocol_path),
             "source_batch": _ref(source_batch_result), "anchor": anchor,
             "source_anchor": source_anchor,
             "opaque_cases": [row["case_id"] for row in cases], "model_requests": 0,
             "paired_probes": 0, "gpu_started": False, "transport_status": status,
             "lease_id": lease.get("lease_id")}
    _write(out / "preflight.json", value)
    return value


def execution(protocol_path: Path, anchor: str, source_batch_result: Path, lab: Path,
              out: Path, transport_root: Path, *, run_epoch: Callable = COMP.run_epoch,
              execute_host: Callable = HOST.execute_pair) -> dict[str, Any]:
    """8개 source의 등록 순서를 유지해 세 정책을 Latin 순서로 실행한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "comparison out은 fresh 경로여야 한다")
    protocol = verify_protocol(protocol_path, anchor, lab)
    source_sha = _sha(source_batch_result)
    cases = _source_cases(source_batch_result, protocol, anchor)
    _require(_sha(source_batch_result) == source_sha, "준비 중 source batch가 바뀌었다")
    source_anchor = _read(source_batch_result)["anchor"]
    references = {row["seed"]: row for row in cases if row["category"] == "NONE"}
    _require(set(references) == {2320, 2321}, "same-seed NONE reference가 완전하지 않다")
    bands = _read(BANDS)["bands"]
    lease = _lease()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    probe_counter = 0
    complete_factory = COMP.transport_factory(Path(transport_root).resolve())
    active_inputs: dict[str, dict[str, Any]] = {}
    verify_context = _context_verifier(
        protocol_path=Path(protocol_path).resolve(), protocol=protocol, lab=Path(lab).resolve(),
        source_batch_result=Path(source_batch_result).resolve(), source_sha=source_sha,
        lease_id=lease.get("lease_id"), transport_root=Path(transport_root).resolve(),
        active_inputs=active_inputs, started=started)

    try:
        for case_index, case in enumerate(cases):
            reference = references[case["seed"]]
            active_inputs.clear()
            active_inputs.update({
                "case": {"input": case["input"],
                         "runtime": case["metadata"]["runtime_facts"]["case"]["path"],
                         "checkpoint_sha256": case["checkpoint"]["sha256"]},
                "reference": {"input": reference["input"],
                              "runtime": reference["metadata"]["runtime_facts"]["reference"]["path"],
                              "checkpoint_sha256": reference["checkpoint"]["sha256"]},
            })
            gate = _gate(case, reference, bands)
            for policy in COMP.policy_order(case_index):
                base = {"case_id": case["case_id"], "policy": policy,
                        "checkpoint_sha256": case["checkpoint"]["sha256"],
                        "reference_checkpoint_sha256": reference["checkpoint"]["sha256"],
                        "gate": deepcopy(gate)}
                if not gate["run"]:
                    excluded = gate["reason"] != "healthy_gate" or gate.get("analysis_exclusion") is not None
                    rows.append({**base, "status": "excluded" if excluded else "complete",
                                 "decision_outcome": "HOLD", "model_requests": 0,
                                 "model_actions": 0, "first_pass_valid_actions": 0,
                                 "post_feedback_valid_actions": 0, "semantic_feedback_actions": 0,
                                 "paired_probes": 0, "correct_confirmed_family": False,
                                 "correct_complete_family": False,
                                 "compound_partial": False, "compound_complete": False})
                    continue

                def pair(state: dict, action: dict, stage: Path, *, _case=case,
                         _reference=reference) -> dict[str, Any]:
                    nonlocal probe_counter
                    _require(probe_counter < MAX_PAIRED_PROBES,
                             "paired probe global cap 종료")
                    result = execute_host(
                        state, action, case_input=_case["input"], reference_input=_reference["input"],
                        case_runtime=Path(_case["metadata"]["runtime_facts"]["case"]["path"]),
                        reference_runtime=Path(_reference["metadata"]["runtime_facts"]["reference"]["path"]),
                        probe_protocol=ROOT / protocol["protocols"]["paired_probe"]["path"],
                        view_protocol=ROOT / protocol["protocols"]["view"]["path"], anchor=anchor,
                        lab=Path(lab).resolve(), out=stage, verify_context=verify_context)
                    probe_counter += 1
                    return result

                epoch_out = out / case["case_id"] / policy
                epoch = run_epoch(case["input"], reference["input"], policy=policy,
                                  run_id=f"{case['case_id']}.{policy}",
                                  out=epoch_out,
                                  verify_context=verify_context,
                                  complete_factory=None if policy == "strong_registered_rule" else complete_factory,
                                  execute_pair=pair)
                score = _score(case["category"], epoch)
                validity = _validity(epoch_out, epoch)
                rows.append({**base, "status": epoch["status"],
                             "decision_outcome": epoch["decision_outcome"],
                             "model_requests": epoch["request_count"],
                             "model_actions": epoch["model_actions"], **validity,
                             "paired_probes": epoch["probe_count"],
                             "parent_probe_wall_s": epoch["parent_probe_wall_s"],
                             "owned_child_wall_s": epoch["owned_child_wall_s"], **score})
        verify_context(None)
        _require(len(rows) == 24, "계획된 8×3 분모가 보존되지 않았다")
        requests = sum(row["model_requests"] for row in rows)
        probes = sum(row["paired_probes"] for row in rows)
        _require(requests <= MAX_REQUESTS and probes <= MAX_PAIRED_PROBES, "비교 전역 상한 초과")
        value = {"schema": RESULT_SCHEMA, "status": "complete", "protocol": _ref(protocol_path),
                 "source_batch": _ref(source_batch_result), "anchor": anchor, "rows": rows,
                 "source_anchor": source_anchor,
                 "planned_cases": 8, "planned_policy_rows": 24,
                 "model_requests": requests, "paired_probes": probes,
                 "legacy_role_executions_upper_bound": probes * 2,
                 "cpu_parent_wall_s": time.monotonic() - started,
                 "coverage": {policy: {
                     "planned": 8,
                     "correct_complete_family": sum(
                         row.get("correct_complete_family") is True for row in rows
                         if row["policy"] == policy),
                     "correct_confirmed_family": sum(
                         row.get("correct_confirmed_family") is True for row in rows
                         if row["policy"] == policy),
                     "compound_partial": sum(row.get("compound_partial") is True for row in rows
                                             if row["policy"] == policy),
                     "compound_complete": sum(row.get("compound_complete") is True for row in rows
                                              if row["policy"] == policy),
                     "excluded": sum(row["status"] == "excluded" for row in rows
                                     if row["policy"] == policy),
                     "first_pass_valid_actions": sum(row["first_pass_valid_actions"] for row in rows
                                                     if row["policy"] == policy),
                     "post_feedback_valid_actions": sum(
                         row["post_feedback_valid_actions"] for row in rows
                         if row["policy"] == policy),
                     "model_requests": sum(row["model_requests"] for row in rows
                                           if row["policy"] == policy),
                     "paired_probes": sum(row["paired_probes"] for row in rows
                                          if row["policy"] == policy),
                 } for policy in COMP.POLICIES},
                 "claims": deepcopy(protocol["claims"]), "automatic_repair": False}
        _write(out / "result.json", value)
        return value
    except BaseException as error:
        _write(out / "failure.json", {"schema": RESULT_SCHEMA, "status": "invalid",
               "error_type": type(error).__name__, "error": str(error), "rows": rows,
               "automatic_retry": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    for name in ("study", "training-protocol", "probe-protocol", "view-protocol",
                 "transport-protocol", "out"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    for command in ("preflight", "execute"):
        run = sub.add_parser(command)
        for name in ("protocol", "source-batch-result", "lab", "out", "transport-root"):
            run.add_argument(f"--{name}", type=Path, required=True)
        run.add_argument("--anchor", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        value = prepare_protocol(args.study, args.training_protocol, args.probe_protocol,
                                 args.view_protocol, args.transport_protocol)
        _write(args.out, value)
    elif args.command == "preflight":
        preflight(args.protocol, args.anchor, args.source_batch_result,
                  args.lab, args.out, args.transport_root)
    else:
        execution(args.protocol, args.anchor, args.source_batch_result,
                  args.lab, args.out, args.transport_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
