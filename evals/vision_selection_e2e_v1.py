"""V05.2 두 선택 입력의 검사·교정·후속 관찰 CPU 실행 계약."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as COMPARISON
import vision_selection_worker_v1 as WORKER

SCHEMA = "vision_selection_e2e_plan_v1"
API_CAP = 32
TARGETS = (
    {"slot_id": "input-03", "seed_group": "source-01", "source_role": "registered-f1",
     "window": "window-01", "view_kind": "text_and_video"},
    {"slot_id": "input-07", "seed_group": "source-02", "source_role": "registered-f1",
     "window": "window-01", "view_kind": "text_and_video"},
)
APPROVAL_FIELDS = frozenset({
    "source_checkpoint_sha256", "source_job_id", "seed", "source_protocol_sha256_lf",
    "selection_terminal_sha256", "activation_bridge_sha256", "probe_authorization_sha256",
    "probe_seal_sha256",
})
PROBE_RECEIPT_FIELDS = frozenset({
    "status", "outcome", "seed", "arm", "checkpoint_sha256", "job_sha256",
    "worker_sha256", "execution_sha256", "report_jobs_sha256", "report_sha256",
    "gpu_before_sha256", "gpu_after_sha256", "authorization_sha256", "seal_sha256",
})


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name}은 JSON object여야 한다")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _capsule_ref(repo: Path, value: Any, expected_sha: Any) -> dict[str, str]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "status"}
            or value.get("status") != "frozen" or value.get("sha256") != expected_sha):
        raise ValueError("대상 capsule SHA가 protocol과 다르다")
    raw = Path(str(value.get("path")))
    path = (raw if raw.is_absolute() else repo / raw).resolve(strict=True)
    path.relative_to(repo)
    if path.is_symlink() or _sha(path) != expected_sha:
        raise ValueError("대상 capsule 원문이 저장소 결합과 다르다")
    return {"path": str(path), "sha256": str(expected_sha), "status": "frozen"}


def _artifact(repo: Path, value: Any, label: str, *, frozen: bool = False,
              lf: bool = False) -> tuple[Path, dict[str, Any]]:
    fields = {"path", "sha256", "status"} if frozen else {"path", "sha256"}
    if (not isinstance(value, dict) or set(value) != fields
            or (frozen and value.get("status") != "frozen")):
        raise ValueError(f"{label} reference 형식이 다르다")
    raw = Path(str(value.get("path")))
    path = (raw if raw.is_absolute() else repo / raw).resolve(strict=True)
    path.relative_to(repo)
    digest = _sha_lf(path) if lf else _sha(path)
    if path.is_symlink() or digest != value.get("sha256"):
        raise ValueError(f"{label} SHA가 다르다")
    return path, _read(path)


def _training_jobs(repo: Path, location: Mapping[str, Any], source: Mapping[str, Any],
                   row: Mapping[str, Any], expected_manifest_sha: str) -> dict[str, str]:
    manifest_path, manifest = _artifact(repo, location.get("input_manifest"), "input manifest", frozen=True)
    if _sha(manifest_path) != expected_manifest_sha:
        raise ValueError("production verifier의 input manifest SHA가 다르다")
    readiness_path, readiness = _artifact(repo, manifest.get("model_readiness"), "model readiness")
    del readiness_path
    training_action = readiness.get("training_action")
    record_refs = readiness.get("record")
    if not isinstance(training_action, Mapping) or not isinstance(record_refs, Mapping):
        raise TypeError("readiness의 training/record 원문 위치가 없다")
    _, training_raw = _artifact(repo, training_action.get("raw"), "training raw")
    _, record_job = _artifact(repo, record_refs.get("job"), "record job")
    source_job = training_raw.get("job_id")
    record_job_id = record_job.get("job_id")
    plan = source.get("training_plan")
    if not isinstance(source_job, str) or not isinstance(record_job_id, str) or not isinstance(plan, list):
        raise TypeError("readiness의 source/record job ID가 없다")
    source_matches = [item for item in plan if isinstance(item, dict)
                      and item.get("job_id") == source_job and item.get("seed") == row.get("source_seed")
                      and item.get("arm") == "F1"]
    corrected = [item for item in plan if isinstance(item, dict)
                 and item.get("seed") == row.get("source_seed") and item.get("arm") == "CORRECTED"
                 and item.get("source_job_id") == source_job]
    if len(source_matches) != 1 or len(corrected) != 1 or not isinstance(corrected[0].get("job_id"), str):
        raise ValueError("등록 source/CORRECTED training job이 유일하지 않다")
    return {"source_training_job_id": source_job, "record_job_id": record_job_id,
            "corrected_job_id": corrected[0]["job_id"]}


def _target_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    method = protocol.get("method_protocol")
    source = protocol.get("source_protocol")
    if not isinstance(method, Mapping) or not isinstance(source, Mapping):
        raise TypeError("actual materializer protocol reference가 없다")
    raw_rows = protocol.get("rows")
    method_sha, source_sha = method.get("sha256_lf"), source.get("sha256_lf")
    if not isinstance(raw_rows, list) or not isinstance(method_sha, str) or not isinstance(source_sha, str):
        raise TypeError("actual materializer row 또는 protocol SHA 형식이 다르다")
    rows = COMPARISON.validate_inputs(
        raw_rows, method_protocol_sha256=method_sha, source_protocol_sha256=source_sha,
    )
    if any(set(row) != COMPARISON.INPUT_FIELDS for row in rows):
        raise ValueError("actual materializer row 필드가 comparison 입력 계약과 다르다")
    selected = []
    for target in TARGETS:
        matches = [row for row in rows if isinstance(row, dict)
                   and all(row.get(key) == value for key, value in target.items())]
        if len(matches) != 1:
            raise ValueError("등록 e2e 대상 row가 유일하지 않다")
        selected.append(matches[0])
    return selected


def prepare_plan(
    repo: Path, protocol_path: Path, input_locations_path: Path, output_path: Path,
    *, execution_id: str, source_anchor: str, protocol_sha256: str,
    production_prepare: Callable[..., dict[str, Any]] = WORKER.prepare,
    initialize: Callable[..., dict[str, str]] = CALL.initialize,
) -> dict[str, Any]:
    """production comparison 검증 뒤 결과가 없을 때 exact 두 capsule을 봉인한다."""
    repository = Path(repo).resolve(strict=True)
    protocol_path = Path(protocol_path).resolve(strict=True)
    locations_path = Path(input_locations_path).resolve(strict=True)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    prepared = production_prepare(
        protocol_path, protocol_sha256, source_anchor, locations_path, root=repository,
    )
    if not isinstance(prepared, dict) or set(prepared) != {"protocol", "source", "inputs", "plan"}:
        raise ValueError("production worker 준비 결과 형식이 다르다")
    protocol = prepared["protocol"]
    if not isinstance(protocol, dict):
        raise TypeError("production protocol이 객체가 아니다")
    comparison_repo, comparison_stage, comparison_manifest = CALL._plan(prepared["plan"])
    if (comparison_repo != repository or comparison_manifest.get("phase") != "comparison"
            or comparison_manifest.get("execution_id") != protocol.get("execution_id")):
        raise ValueError("comparison execution/root 결합이 다르다")
    comparison_output = comparison_stage / "comparison-output"
    result_paths = (comparison_output / "choice_results.json", comparison_output / "comparison_summary.json")
    terminal_paths = tuple(comparison_stage.glob("decisions/*/terminal.json"))
    if terminal_paths or any(path.exists() or path.is_symlink() for path in result_paths):
        raise ValueError("comparison 결과를 본 뒤 e2e 입력을 고정할 수 없다")
    targets = _target_rows(protocol)
    verified_inputs = prepared["inputs"]
    input_rows = verified_inputs.get("rows") if isinstance(verified_inputs, dict) else None
    if not isinstance(input_rows, list) or len(input_rows) != 24:
        raise ValueError("production verifier의 input 24행이 필요하다")
    by_call = {row.get("call_id"): row for row in input_rows if isinstance(row, dict)}
    if len(by_call) != 24:
        raise ValueError("production verifier call_id가 중복됐다")
    locations = _read(locations_path)
    location_rows = locations.get("rows")
    if (_sha(locations_path) != verified_inputs.get("input_locations_sha256")
            or not isinstance(location_rows, list) or len(location_rows) != 24):
        raise ValueError("production verifier 뒤 input locations가 변경됐다")
    by_location = {row.get("call_id"): row for row in location_rows if isinstance(row, dict)}
    manifest_hashes = verified_inputs.get("input_manifest_sha256")
    if len(by_location) != 24 or not isinstance(manifest_hashes, dict):
        raise ValueError("production verifier의 input manifest map이 다르다")
    source_ref = protocol.get("source_protocol")
    if not isinstance(source_ref, dict):
        raise TypeError("source protocol reference가 없다")
    _, source_protocol = _artifact(repository, {
        "path": source_ref.get("path"), "sha256": source_ref.get("sha256_lf"),
    }, "source protocol", lf=True)
    calls = []
    frozen_rows = []
    for target, row in zip(TARGETS, targets, strict=True):
        verified = by_call.get(row.get("call_id"))
        if not isinstance(verified, dict) or set(verified) != {"call_id", "run_id", "capsule"}:
            raise ValueError("production verifier 대상 capsule이 없다")
        capsule = _capsule_ref(repository, verified["capsule"], row.get("capsule_sha256"))
        location = by_location.get(row.get("call_id"))
        expected_manifest_sha = manifest_hashes.get(row.get("call_id"))
        if not isinstance(location, dict) or not isinstance(expected_manifest_sha, str):
            raise TypeError("대상 input manifest 결합이 없다")
        jobs = _training_jobs(repository, location, source_protocol, row, expected_manifest_sha)
        frozen = CALL.prepared(capsule)
        if frozen.get("view_kind") != "text_and_video" or frozen.get("checkpoint_sha256") != row.get("checkpoint_sha256"):
            raise ValueError("대상 capsule 내용이 등록 row와 다르다")
        eligible = frozen.get("bundle", {}).get("selection_view", {}).get("eligible_probes")
        if (not isinstance(eligible, list) or not eligible
                or any(not isinstance(probe, str) or not probe for probe in eligible)):
            raise ValueError("대상 capsule 등록 probe 목록이 없다")
        call_id = f"e2e-{target['seed_group']}"
        references = [candidate for candidate in protocol["rows"]
                      if candidate.get("seed_group") == target["seed_group"]
                      and candidate.get("source_role") == "normal"
                      and candidate.get("window") == "window-01"
                      and candidate.get("view_kind") == "text_and_video"]
        if len(references) != 1:
            raise ValueError("NONE reference row가 유일하지 않다")
        calls.append({"call_id": call_id, "run_id": f"vision-selection-{execution_id}-{call_id}",
                      "capsule": capsule})
        frozen_rows.append({"target": dict(target), "call_id": call_id,
                            "source_call_id": row["call_id"], "source_seed": row["source_seed"],
                            "source_job_id": jobs["source_training_job_id"],
                            "record_job_id": jobs["record_job_id"],
                            "source_lineage_id": row["source_lineage_id"],
                            "corrected_job_id": jobs["corrected_job_id"],
                            "checkpoint_sha256": row["checkpoint_sha256"],
                            "reference_checkpoint_sha256": references[0]["checkpoint_sha256"],
                            "capsule_sha256": row["capsule_sha256"],
                            "eligible_probes": list(eligible),
                            "source_protocol_sha256_lf": row["source_protocol_sha256"]})
    reference = initialize(repository, execution_id, "e2e", source_anchor, protocol_sha256, calls)
    body = {"schema": SCHEMA, "api_cap": API_CAP, "execution_id": execution_id,
            "source_anchor": source_anchor, "protocol_sha256": protocol_sha256,
            "input_locations_sha256": _sha(locations_path), "selection_plan": reference,
            "comparison": {
                "selection_plan": prepared["plan"],
                "execution_root": comparison_stage.relative_to(repository).as_posix(),
                "output_root": comparison_output.relative_to(repository).as_posix(),
            },
            "rows": frozen_rows, "correction": {"id": "restore_body_frame_reward",
                                                  "recovery_mode": "restart"},
            "fixed_eval_decides_recovery": True,
            "post_observation_used_for_decision": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return body


def _row(plan: Mapping[str, Any], group: str) -> dict[str, Any]:
    matches = [row for row in plan.get("rows", []) if row.get("target", {}).get("seed_group") == group]
    if len(matches) != 1:
        raise ValueError("e2e seed group이 유일하지 않다")
    return matches[0]


def approval_candidate(plan: Mapping[str, Any], group: str, selection: Mapping[str, Any],
                       pair: Mapping[str, Any]) -> dict[str, Any] | None:
    """유효 P_reward abnormal 쌍에만 checkpoint-bound 교정 승인 후보를 만든다."""
    if (selection.get("model_status") != "valid" or selection.get("action") != "continue"
            or selection.get("probe") != "P_reward"):
        return None
    if set(pair) != {"probe", "none", "f1"} or pair.get("probe") != "P_reward":
        return None
    row = _row(plan, group)
    none, f1 = pair["none"], pair["f1"]
    if any(not isinstance(item, Mapping) for item in (none, f1)):
        return None
    if (set(none) != PROBE_RECEIPT_FIELDS or set(f1) != PROBE_RECEIPT_FIELDS
            or none.get("status") != "complete" or f1.get("status") != "complete"
            or none.get("outcome") != "normal" or f1.get("outcome") != "abnormal"
            or none.get("arm") != "NONE" or f1.get("arm") != "F1"
            or none.get("checkpoint_sha256") != row["reference_checkpoint_sha256"]
            or f1.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or none.get("seed") != f1.get("seed") or f1.get("seed") != row["source_seed"]):
        return None
    required = {"terminal_sha256", "activation_bridge_sha256"}
    digest_fields = PROBE_RECEIPT_FIELDS - {"status", "outcome", "seed", "arm"}
    if (not required <= set(selection)
            or any(not isinstance(item.get(field), str) or len(item[field]) != 64
                   for item in (none, f1) for field in digest_fields)):
        return None
    return {
        "source_checkpoint_sha256": row["checkpoint_sha256"],
        "source_job_id": row["source_job_id"], "seed": row["source_seed"],
        "source_protocol_sha256_lf": row["source_protocol_sha256_lf"],
        "selection_terminal_sha256": selection["terminal_sha256"],
        "activation_bridge_sha256": selection["activation_bridge_sha256"],
        "probe_authorization_sha256": f1["authorization_sha256"],
        "probe_seal_sha256": f1["seal_sha256"],
    }


def commands(plan: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """현재 증거에서 허용되는 다음 provider/GPU 작업만 순서대로 반환한다."""
    selections = state.get("selections", {})
    probes = state.get("probes", {})
    approvals = state.get("approvals", {})
    corrections = state.get("corrections", {})
    observations = state.get("observations", {})
    api_used = state.get("api_used", 0)
    if type(api_used) is not int or not 0 <= api_used <= API_CAP:
        raise ValueError("e2e API 원장이 cap 32를 벗어났다")
    actions = []
    for row in plan.get("rows", []):
        group = row["target"]["seed_group"]
        selection = selections.get(group)
        if selection is None:
            actions.append({"kind": "initial_selection", "group": group,
                            "call_id": row["call_id"], "api_cap": API_CAP})
            break
        if (selection.get("model_status") != "valid" or selection.get("action") != "continue"):
            continue
        probe = selection.get("probe")
        if probe not in row["eligible_probes"]:
            continue
        pair = probes.get(group)
        if pair is None:
            actions.append({"kind": "probe_pair", "group": group, "probe": probe,
                            "arms": ["NONE", "F1"], "checkpoint_sha256": row["checkpoint_sha256"]})
            continue
        candidate = approval_candidate(plan, group, selection, pair)
        if candidate is None:
            continue
        approval = approvals.get(group)
        if approval is None:
            actions.append({"kind": "propose_correction", "group": group,
                            "correction_id": "restore_body_frame_reward", "recovery_mode": "restart",
                            "evidence": candidate})
            continue
        if set(approval.get("evidence", {})) != APPROVAL_FIELDS or approval.get("evidence") != candidate:
            raise ValueError("교정 승인이 현재 checkpoint/probe 증거와 다르다")
        if approval.get("state") == "approved":
            actions.append({"kind": "consume_and_run_correction", "group": group,
                            "job_id": row["corrected_job_id"],
                            "correction_id": "restore_body_frame_reward", "recovery_mode": "restart"})
            continue
        correction = corrections.get(group)
        if approval.get("state") != "consumed" or correction is None:
            continue
        fixed = correction.get("fixed_eval")
        recovered = isinstance(fixed, Mapping) and fixed.get("none") == "healthy" and fixed.get("corrected") == "healthy"
        if correction.get("status") != "complete" or not recovered:
            continue
        if group not in observations:
            actions.append({"kind": "post_observation", "group": group,
                            "checkpoint_sha256": correction.get("checkpoint_sha256"),
                            "used_for_decision": False, "shared_api_cap": API_CAP})
    model_kinds = {"initial_selection", "post_observation"}
    remaining = API_CAP - api_used
    outputs = []
    model_emitted = False
    for action in actions:
        if action["kind"] in model_kinds:
            if remaining <= 0 or model_emitted:
                continue
            remaining -= 1
            model_emitted = True
        outputs.append(action)
    return outputs
