"""Audit active-diagnosis evidence chains without invoking a model or simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, TypeGuard

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
from rl_triage import active_diagnosis as D, probe_loop as L  # noqa: E402
import brev_probe_backend as B  # noqa: E402

TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PROOF_GATES = (
    "leader_changed",
    "latest_receipt_cited",
    "changed_leader_rejected_by_actual_observation",
    "after_leader_survives_actual_observation",
    "ablation_preserves_before_leader",
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path, *, raw: bool = False) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data if raw else data.replace(b"\r\n", b"\n")).hexdigest()


def _path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"공개 상대 경로가 아니다: {relative!r}")
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"저장소 밖 경로다: {relative!r}")
    return path


def _load(path: Path, errors: list[str], label: str):
    try:
        return _read(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label} 누락 또는 JSON 오류: {exc}")
        return None


def _sanitize_errors(root: Path, errors: list[str]) -> list[str]:
    root_text = str(root.resolve())
    variants = (root_text, root_text.replace("\\", "\\\\"), root_text.replace("\\", "/"))
    sanitized = []
    for error in errors:
        value = error
        for prefix in variants:
            value = value.replace(prefix, ".")
        value = re.sub(r"(?i)[a-z]:(?:\\\\|\\|/)[^\s'\"\])}]+", "<absolute-path>", value)
        value = re.sub(r"/(?:home|Users)/[^\s'\"\])}]+", "<absolute-path>", value)
        sanitized.append(value)
    return sanitized


def _contains_host_absolute_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.search(r"(?i)[a-z]:(?:\\\\|\\|/)+|/(?:home|Users|tmp|var/tmp)/", value))


def _nonnegative_number(value: object) -> TypeGuard[int | float]:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def code_integrity_errors(root: Path, protocol: dict) -> list[str]:
    """Verify the exact frozen executable sources before producing evidence verdicts."""
    from run_active_diagnosis import CODE

    mapping = protocol.get("code_sha256_lf")
    if not isinstance(mapping, dict) or set(mapping) != set(CODE):
        return ["사전등록 code SHA 목록 불일치"]
    errors = []
    for relative, expected in mapping.items():
        try:
            path = _path(root, relative)
            if path.is_symlink() or _sha(path) != expected:
                errors.append(f"사전등록 이후 코드 변경: {relative}")
        except (OSError, ValueError):
            errors.append(f"동결 코드 누락/경로 오류: {relative}")
    return errors


def _strip_fence(text: str) -> str:
    value = text.strip()
    return value[7:value.rfind("```")].strip() if value.startswith("```json") and value.endswith("```") else value


def _spec(protocol: dict, case: str) -> dict | None:
    return next((item for item in protocol.get("cases", []) if isinstance(item, dict) and item.get("case") == case), None)


def _cloud_binding_errors(root: Path, folder: Path, tag: str, case: str, spec: dict,
                          protocol: dict, index: int, probe: str, binding: dict,
                          backend_binding: object, expected_watchdog_s: object = None) -> list[str]:
    errors = []
    backend_config = protocol.get("probe_backend")
    if not isinstance(backend_config, dict) or not isinstance(backend_binding, dict):
        return ["cloud backend preflight binding 불일치"]
    preflight = _load(folder / f"backend_preflight_{index}.json", errors, "backend preflight")
    gpu = _load(folder / f"gpu_preflight_{index}.json", errors, "GPU preflight")
    try:
        report = _read(root / "evals/results" / tag / "probes" / f"{case}__{probe}.json")
        report_name = report.get("checkpoint", {}).get("file")
    except (OSError, json.JSONDecodeError, AttributeError):
        report_name = None
    expected_checkpoints = [{"role": "last", "file": report_name,
                             "sha256": spec.get("checkpoint_sha256")}]
    dependency = spec.get("checkpoint_dependencies", {}).get(probe)
    if isinstance(dependency, dict):
        expected_checkpoints.append({"role": probe + ":from", **dependency})
    environment = backend_binding.get("environment")
    ok = (
        isinstance(preflight, dict) and preflight == backend_binding
        and isinstance(gpu, dict) and backend_binding.get("gpu_snapshot_sha256_lf") == L.digest(gpu)
        and backend_binding.get("name") == "brev"
        and backend_binding.get("instance") == backend_config.get("instance")
        and backend_binding.get("workspace_basename") == backend_config.get("workspace_basename")
        and backend_binding.get("source_sha256_lf") == protocol.get("code_sha256_lf")
        and backend_binding.get("source_set_digest") == L.digest(protocol.get("code_sha256_lf"))
        and backend_binding.get("owner_binding_digest") == L.digest(binding)
        and backend_binding.get("checkpoint_set") == expected_checkpoints
        and isinstance(backend_binding.get("jobs_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", backend_binding["jobs_sha256"]) is not None
        and B.environment_matches(environment, backend_config.get("environment"))
        and isinstance(environment, dict) and environment.get("owner_marker_sha256")
            == backend_config.get("owner_marker_sha256")
        and backend_binding.get("environment_digest") == L.digest(environment)
        and _nonnegative_number(backend_binding.get("preapproval_transport_wall_s"))
        and _nonnegative_number(backend_binding.get("launcher_watchdog_s"))
        and (expected_watchdog_s is None
             or backend_binding.get("launcher_watchdog_s") == expected_watchdog_s)
    )
    if not ok:
        errors.append("cloud backend preflight binding 불일치")
    return errors


OUTCOME_STATUSES = {"valid", "semantic_invalid", "infra_exhausted", "harness_fatal"}
OUTCOME_KEYS = {"stage", "status", "packet_sha256_lf", "attempts", "decision",
                "decision_sha256_lf", "error"}
PAIR_KEYS = {"stage", "order", "actual", "control", "contrast", "proof_eligible"}


def _attempts(folder: Path, stage: str, errors: list[str]) -> list[tuple[Path, dict]]:
    paths = sorted((folder / "model").glob(f"{stage}.attempt*.json"))
    numbers = []
    records = []
    for path in paths:
        match = re.fullmatch(re.escape(stage) + r"\.attempt([1-9][0-9]*)\.json", path.name)
        if not match:
            errors.append(f"{stage}: 허용되지 않은 attempt 파일 {path.name}")
            continue
        numbers.append(int(match.group(1)))
        record = _load(path, errors, path.name)
        if isinstance(record, dict):
            records.append((path, record))
    if numbers != list(range(1, len(numbers) + 1)):
        errors.append(f"{stage}: attempt가 1부터 연속된 기록이 아니다")
    return records


def _model_outcome(folder: Path, stage: str, protocol: dict,
                   errors: list[str]) -> tuple[dict | None, dict | None, list[dict]]:
    packet_path = folder / "model" / f"{stage}.packet.json"
    decision_path = folder / "model" / f"{stage}.decision.json"
    outcome_path = folder / "model" / f"{stage}.outcome.json"
    packet = _load(packet_path, errors, f"{stage} packet")
    outcome = _load(outcome_path, errors, f"{stage} outcome")
    if not isinstance(packet, dict) or not isinstance(outcome, dict):
        return None, None, []
    if set(outcome) != OUTCOME_KEYS or outcome.get("stage") != stage or outcome.get("status") not in OUTCOME_STATUSES:
        errors.append(f"{stage}: outcome schema가 다르다")
    if outcome.get("packet_sha256_lf") != _sha(packet_path):
        errors.append(f"{stage}: outcome packet SHA가 다르다")
    packet_sha = _sha(packet_path)
    message_sha = L.digest(D.decision_messages(packet))
    records = _attempts(folder, stage, errors)
    refs = outcome.get("attempts")
    expected_refs = [{"path": path.relative_to(folder).as_posix(), "sha256_lf": _sha(path)}
                     for path, _ in records]
    if refs != expected_refs:
        errors.append(f"{stage}: outcome attempt 링크가 실제 파일과 다르다")
    successful: list[dict] = []
    semantic_failures = 0
    infra_failures = 0
    harness_failures = 0
    attempt_kinds: list[str] = []
    for path, record in records:
        if record.get("packet_sha256") != packet_sha:
            errors.append(f"{path.name}: packet SHA가 다르다")
        if record.get("messages_sha256") != message_sha:
            errors.append(f"{path.name}: 실제 prompt messages SHA가 다르다")
        request = record.get("request")
        expected_request = {"model": protocol.get("model"), "temperature": 0,
                            "max_tokens": 16384, "stream": False,
                            "extra_body": {"reasoning_effort": "none"}}
        if request != expected_request:
            errors.append(f"{path.name}: provider 요청 계약이 다르다")
        if record.get("saved_response_cached") is not False or record.get("client_cache_used") is not False:
            errors.append(f"{path.name}: fresh 호출 provenance가 아니다")
        if not all(key in record for key in ("response_id", "finish_reason", "reasoning_content")):
            errors.append(f"{path.name}: response provenance 필드가 누락됐다")
        if record.get("infra_error"):
            infra_failures += 1
            attempt_kinds.append("infra")
            if record.get("retryable") is not True:
                errors.append(f"{path.name}: infra 오류가 retryable로 기록되지 않았다")
            continue
        if record.get("error_type") or record.get("worker_cleanup_failed"):
            harness_failures += 1
            attempt_kinds.append("harness")
            continue
        if record.get("exit_code") != 0 or not isinstance(record.get("response_id"), str):
            harness_failures += 1
            attempt_kinds.append("harness")
            continue
        if any(record.get(key) is None for key in ("input_tokens", "output_tokens", "total_tokens")):
            errors.append(f"{path.name}: 성공 호출 usage가 없다")
        try:
            parsed = json.loads(_strip_fence(record.get("content") or ""))
            validated = D.validate_decision(parsed, packet)
        except (json.JSONDecodeError, TypeError, ValueError):
            semantic_failures += 1
            attempt_kinds.append("semantic")
            continue
        successful.append(validated)
        attempt_kinds.append("valid")

    status = outcome.get("status")
    decision = outcome.get("decision")
    typed_error = outcome.get("error")
    if isinstance(typed_error, dict) and _contains_host_absolute_path(typed_error.get("message")):
        errors.append(f"{stage}: raw outcome 오류에 호스트 절대 경로가 있다")
    max_attempts = protocol.get("max_infra_attempts")
    if (isinstance(max_attempts, bool) or not isinstance(max_attempts, int)
            or not 1 <= len(records) <= max_attempts):
        errors.append(f"{stage}: outcome attempt 수가 사전등록 한도를 벗어났다")
    if status == "valid":
        if typed_error is not None or len(successful) != 1 or successful[-1] != decision:
            errors.append(f"{stage}: valid outcome과 재생한 모델 판단이 다르다")
        saved = _load(decision_path, errors, f"{stage} decision")
        decision_sha = _sha(decision_path) if decision_path.is_file() else None
        if saved != decision or outcome.get("decision_sha256_lf") != decision_sha:
            errors.append(f"{stage}: decision 파일/hash가 outcome과 다르다")
        if semantic_failures or harness_failures:
            errors.append(f"{stage}: semantic/harness 실패 뒤 valid로 세탁됐다")
        if not attempt_kinds or attempt_kinds[-1] != "valid" or any(
                kind != "infra" for kind in attempt_kinds[:-1]):
            errors.append(f"{stage}: valid 이전 시도는 infra 오류만 허용된다")
    else:
        if decision is not None or outcome.get("decision_sha256_lf") is not None or decision_path.exists():
            errors.append(f"{stage}: 실패 outcome에 decision이 남아 있다")
        if (not isinstance(typed_error, dict) or set(typed_error) != {"type", "message"}
                or not all(isinstance(typed_error.get(key), str) and typed_error[key] for key in ("type", "message"))):
            errors.append(f"{stage}: 실패 outcome typed error가 없다")
        expected_type = {"semantic_invalid": "ModelSemanticError",
                         "infra_exhausted": "ModelInfraError",
                         "harness_fatal": "ModelHarnessError"}.get(status if isinstance(status, str) else "")
        if isinstance(typed_error, dict) and typed_error.get("type") != expected_type:
            errors.append(f"{stage}: outcome 오류 분류가 다르다")
        if status == "semantic_invalid" and (semantic_failures != 1 or successful or harness_failures
                                               or not records or semantic_failures + infra_failures != len(records)):
            errors.append(f"{stage}: semantic invalid 재생 결과가 다르다")
        if status == "semantic_invalid" and (not attempt_kinds or attempt_kinds[-1] != "semantic" or any(
                kind != "infra" for kind in attempt_kinds[:-1])):
            errors.append(f"{stage}: semantic 응답 뒤 재시도했거나 이전 시도가 infra 오류가 아니다")
        if status == "infra_exhausted" and (infra_failures != len(records)
                                             or len(records) != protocol.get("max_infra_attempts")):
            errors.append(f"{stage}: infra 재시도 한도 기록이 다르다")
        if status == "harness_fatal" and (successful or semantic_failures or harness_failures > 1):
            errors.append(f"{stage}: harness fatal 재생 결과가 다르다")
        if status == "harness_fatal" and (not attempt_kinds or attempt_kinds[-1] != "harness" or any(
                kind != "infra" for kind in attempt_kinds[:-1])):
            errors.append(f"{stage}: harness fatal 이전 시도는 infra 오류만 허용된다")
    if successful and records and successful[-1] != decision:
        errors.append(f"{stage}: 마지막 유효 모델 응답과 outcome decision이 다르다")
    return packet, outcome, [record for _, record in records]


def _masked_packet(actual: dict, latest: dict) -> dict:
    prefix = f"receipt:{latest['request_id']}:"
    return {
        **actual,
        "observed": actual["observed"][:-1],
        "facts": {key: value for key, value in actual["facts"].items() if not key.startswith(prefix)},
    }


def _source_checks(root: Path, spec: dict, errors: list[str]) -> None:
    for source in spec.get("sources", {}).values():
        if not isinstance(source, dict) or set(("path", "sha256_lf")) - set(source):
            errors.append("등록 source 형식이 잘못됐다")
            continue
        try:
            if "private" in Path(source["path"]).parts:
                errors.append(f"비공개 source는 감사 입력이 될 수 없다: {source['path']}")
                continue
            path = _path(root, source["path"])
            if not path.is_file() or _sha(path) != source["sha256_lf"]:
                errors.append(f"등록 source SHA가 다르다: {source['path']}")
        except ValueError as exc:
            errors.append(str(exc))


def _ledger_checks(root: Path, folder: Path, tag: str, result: dict, spec: dict,
                   protocol: dict, decisions: list[dict], errors: list[str]) -> None:
    observed = result.get("observed", [])
    if not isinstance(observed, list) or any(not isinstance(item, dict) for item in observed):
        errors.append("result.observed가 receipt 객체 목록이 아니다")
        return
    ledger_path = folder / "ledger.jsonl"
    if not ledger_path.is_file() and not observed:
        return
    try:
        events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"ledger 누락 또는 JSON 오류: {exc}")
        return
    if any(not isinstance(event, dict) for event in events):
        errors.append("ledger event가 JSON 객체가 아니다")
        return
    by_id: dict[str, list[dict]] = {}
    for event in events:
        by_id.setdefault(event.get("request_id"), []).append(event)
    if set(by_id) != {item.get("request_id") for item in observed}:
        errors.append("ledger 요청 집합과 result receipt 집합이 다르다")
    prior_digests = []
    prior_canonical_gpu_s = 0.0
    protocol_sha = protocol.get("_audit_sha256_lf")
    controller = result.get("controller")
    for index, augmented in enumerate(observed, 1):
        if not isinstance(augmented, dict):
            continue
        rid, probe = augmented.get("request_id"), augmented.get("probe")
        if (not isinstance(rid, str) or probe not in L.PROBES
                or (controller == "model" and index > len(decisions))):
            errors.append(f"{rid}: 알 수 없는 probe 또는 연결할 raw decision이 없다")
            continue
        chain = by_id.get(rid, [])
        if [event.get("event") for event in chain] != ["proposed", "approved", "consumed", "receipt"]:
            errors.append(f"{rid}: exactly-once 원장 순서가 아니다")
            continue
        proposed, approved, _, canonical = chain
        envelope = _load(folder / "approvals" / f"{rid}.json", errors, f"{rid} approval envelope")
        if not isinstance(envelope, dict):
            continue
        expected_stage = "initial" if index == 1 else f"after_{index - 1}"
        decision_path = folder / "model" / f"{expected_stage}.decision.json"
        packet_path = folder / "model" / f"{expected_stage}.packet.json"
        attempts = sorted((folder / "model").glob(f"{expected_stage}.attempt*.json"))
        binding = envelope.get("binding", {})
        if not isinstance(binding, dict):
            errors.append(f"{rid}: binding이 객체가 아니다")
            continue
        if controller == "model":
            expected_binding = {
                "controller": "model",
                "packet_sha256_lf": _sha(packet_path),
                "model_call_sha256_lf": _sha(attempts[-1]) if attempts else None,
                "decision_sha256_lf": _sha(decision_path),
                "protocol_sha256_lf": protocol_sha,
            }
            expected_probe = decisions[index - 1].get("probe")
        else:
            expected_binding = {
                "controller": controller,
                "initial_packet_sha256_lf": _sha(folder / "initial_packet.json"),
                "protocol_sha256_lf": protocol_sha,
            }
            if controller == "fixed_order":
                prior_probes = [item.get("probe") for item in observed[:index - 1]]
                expected_probe = next((item for item in L.FIXED_ORDER if item not in prior_probes), None)
            else:
                budget_left = protocol.get("max_probes", 0) - (index - 1)
                state = D.evidence_state(observed[:index - 1], budget_left)
                expected_probe = L.next_probe(
                    state["hypotheses"],
                    {item["probe"]: item["outcome"] for item in observed[:index - 1]})
        checks = (
            (proposed.get("prereg_digest") == L.digest(envelope), "envelope digest"),
            (proposed.get("args_digest") == L.digest(L.PROBES.get(probe, {}).get("args")), "args digest"),
            (proposed.get("probe") == probe == expected_probe == envelope.get("probe"), "selected probe"),
            (envelope.get("args") == L.PROBES.get(probe, {}).get("args"), "canonical args"),
            (envelope.get("checkpoint_sha256") == spec.get("checkpoint_sha256"), "checkpoint"),
            (envelope.get("reference_source") == spec.get("sources", {}).get("probe_reference"), "reference source"),
            (envelope.get("prior_receipt_digests") == prior_digests, "prior receipt digests"),
            (binding == expected_binding, "execution binding"),
            (approved.get("approver") == protocol.get("approver") == result.get("approval_actor"), "approver"),
        )
        for ok, label in checks:
            if not ok:
                errors.append(f"{rid}: {label} 불일치")
        if augmented.get("receipt_digest") != L.digest(canonical):
            errors.append(f"{rid}: canonical receipt digest 불일치")
        for field in ("outcome", "measurement", "gpu_s", "exit_code", "over_budget"):
            if augmented.get(field) != canonical.get(field):
                errors.append(f"{rid}: ledger/result {field} 불일치")
        if envelope.get("protocol_sha256_lf") != protocol_sha:
            errors.append(f"{rid}: envelope protocol SHA 불일치")
        if proposed.get("budget_gpu_s") != protocol.get("probe_execution_budget_s"):
            errors.append(f"{rid}: 실행 예산 불일치")
        backend_binding = envelope.get("probe_backend")
        backend_config = protocol.get("probe_backend", {"name": "native_windows"})
        if "probe_backend" not in protocol:
            backend_binding = None
        elif not isinstance(backend_config, dict):
            errors.append(f"{rid}: protocol backend 형식 오류")
            backend_binding = None
        elif backend_config.get("name") == "native_windows":
            if backend_binding != {"name": "native_windows"}:
                errors.append(f"{rid}: native backend binding 불일치")
        else:
            case_wall_limit = protocol.get("per_case_isaac_wall_limit_s")
            expected_watchdog = (case_wall_limit - prior_canonical_gpu_s
                                 if _nonnegative_number(case_wall_limit) else None)
            errors.extend(f"{rid}: {error}" for error in _cloud_binding_errors(
                root, folder, tag, result["case"], spec, protocol, index, probe, binding,
                backend_binding, expected_watchdog))
        prereg_digest, args_digest = proposed.get("prereg_digest"), proposed.get("args_digest")
        if not isinstance(prereg_digest, str) or not isinstance(args_digest, str):
            errors.append(f"{rid}: 원장 digest 형식 오류")
            expected_effect = None
        else:
            expected_effect = L.Ledger.effect_key(prereg_digest, probe, args_digest)
        if expected_effect is None or proposed.get("effect_key") != expected_effect:
            errors.append(f"{rid}: effect key 불일치")
        _artifact_checks(root, tag, result["case"], probe, augmented, canonical, spec,
                         proposed.get("budget_gpu_s"), backend_binding, errors)
        canonical_gpu_s = canonical.get("gpu_s")
        if _nonnegative_number(canonical_gpu_s):
            prior_canonical_gpu_s += canonical_gpu_s
        prior_digests.append(augmented.get("receipt_digest"))


def _artifact_checks(root: Path, tag: str, case: str, probe: str, augmented: dict,
                     canonical: dict, spec: dict, budget_gpu_s: object,
                     backend_binding: object, errors: list[str]) -> None:
    base = root / "evals" / "results" / tag
    report_path = base / "probes" / f"{case}__{probe}.json"
    report = _load(report_path, errors, f"{probe} report")
    execution = _load(base / "execution" / f"{case}__{probe}.json", errors, f"{probe} execution")
    resources = _load(base / "resources" / f"{case}__{probe}.json", errors, f"{probe} resources")
    for label, value in (("report", report), ("execution", execution), ("resources", resources)):
        if not isinstance(value, dict):
            errors.append(f"{probe}: {label} 산출물이 JSON 객체가 아니다")
    if not isinstance(report, dict) or not isinstance(execution, dict) or not isinstance(resources, dict):
        return
    checkpoint, report_sha = spec.get("checkpoint_sha256"), _sha(report_path)
    report_checkpoint = report.get("checkpoint")
    if not isinstance(report_checkpoint, dict):
        errors.append(f"{probe}: report checkpoint 형식 오류")
        return
    probe_cells = report.get("probes")
    if not isinstance(probe_cells, dict) or not isinstance(probe_cells.get(probe), dict):
        errors.append(f"{probe}: report probe cell 형식 오류")
        return
    cell = probe_cells[probe]
    sources = spec.get("sources")
    if not isinstance(sources, dict):
        errors.append("등록 source 목록 형식 오류")
        return
    reference_source = sources.get("probe_reference")
    if not isinstance(reference_source, dict):
        errors.append("probe reference source 형식 오류")
        return
    try:
        reference_path = _path(root, reference_source.get("path", ""))
    except ValueError as exc:
        errors.append(str(exc))
        return
    reference_report = _load(reference_path, errors, "probe reference")
    if not isinstance(reference_report, dict):
        errors.append("probe reference report 형식 오류")
        return
    reference_cells = reference_report.get("probes")
    reference_cell = reference_cells.get(probe) if isinstance(reference_cells, dict) else None
    if not isinstance(reference_cell, dict):
        errors.append(f"{probe}: reference probe cell 형식 오류")
        return
    measurement, reference = cell.get("measurement"), reference_cell.get("measurement")
    canonical_measurement = canonical.get("measurement")
    if not all(isinstance(value, dict) for value in (measurement, reference, canonical_measurement)):
        errors.append(f"{probe}: measurement 형식 오류")
        return
    numeric_values = {
        "cell exit": cell.get("exit"), "canonical exit_code": canonical.get("exit_code"),
        "execution exit_code": execution.get("exit_code"), "execution gpu_wall_s": execution.get("gpu_wall_s"),
        "execution execution_wall_s": execution.get("execution_wall_s"), "execution cpu_s": execution.get("cpu_s"),
        "execution parent_cpu_s": execution.get("parent_cpu_s", 0), "resources cpu_s": resources.get("cpu_s"),
        "canonical gpu_s": canonical.get("gpu_s"),
    }
    for label, value in numeric_values.items():
        if not _nonnegative_number(value):
            errors.append(f"{probe}: {label} 형식 오류")
            return
    execution_steps, resource_steps = execution.get("simulator_steps"), resources.get("simulator_steps")
    if (isinstance(execution_steps, bool) or not isinstance(execution_steps, int) or execution_steps < 0
            or isinstance(resource_steps, bool) or not isinstance(resource_steps, int) or resource_steps < 0):
        errors.append(f"{probe}: simulator_steps 형식 오류")
        return
    if not _nonnegative_number(budget_gpu_s):
        errors.append(f"{probe}: 실행 예산 형식 오류")
        return
    try:
        expected_outcome = L.classify(probe, measurement, reference) if cell.get("exit") == 0 else "unknown"
    except (KeyError, TypeError, ValueError):
        errors.append(f"{probe}: measurement를 분류할 수 없다")
        return
    over = canonical["gpu_s"] > budget_gpu_s
    if canonical.get("exit_code") != 0 or over:
        expected_outcome = "unknown"
    checks = (
        (report_checkpoint.get("sha256") == checkpoint == augmented.get("checkpoint_sha256"), "checkpoint SHA"),
        (augmented.get("report_sha256_lf") == report_sha, "report SHA"),
        (cell.get("measurement") == canonical.get("measurement"), "measurement"),
        (cell.get("exit") == canonical.get("exit_code"), "exit code"),
        (canonical.get("over_budget") == over and canonical.get("outcome") == expected_outcome, "classification/over-budget"),
        (execution.get("gpu_wall_s") == canonical.get("gpu_s") == execution.get("execution_wall_s"), "execution wall"),
        (execution.get("exit_code") == canonical.get("exit_code"), "execution exit code"),
        (execution.get("case") == case == resources.get("case") and execution.get("probe") == probe == resources.get("probe"), "artifact case/probe"),
        (execution.get("checkpoint_sha256") == checkpoint and execution.get("report_sha256_lf") == report_sha, "execution binding"),
        (resources.get("checkpoint_sha256") == checkpoint and resources.get("report_sha256_lf") == report_sha, "resource binding"),
        (execution["cpu_s"] == resources["cpu_s"] + execution.get("parent_cpu_s", 0)
         and execution_steps == resource_steps, "resource totals"),
    )
    if isinstance(backend_binding, dict) and backend_binding.get("name") == "brev":
        backend_digest = L.digest(backend_binding)
        meter_path = base / "meters" / f"{case}__{probe}.json"
        meter = _load(meter_path, errors, f"{probe} meter")
        meter_sha = _sha(meter_path) if meter_path.is_file() else None
        meter_rows = meter.get("measurements") if isinstance(meter, dict) else None
        meter_row = meter_rows[0] if isinstance(meter_rows, list) and len(meter_rows) == 1 else None
        meter_binding_ok = (
            isinstance(meter, dict) and isinstance(meter_row, dict)
            and resources.get("meter_sha256_lf") == meter_sha
            and meter.get("schema") == "probe_process_metrics_v1"
            and meter.get("completed") is True
            and meter.get("phase") in {"before_app_close", "after_main"}
            and meter.get("cpu_s") == resources.get("cpu_s")
            and meter.get("cpu_scope") == resources.get("isaac_cpu_scope")
            and (meter.get("phase") == "after_main") == resources.get("app_close_cpu_included")
            and meter.get("simulator_steps_scope") == resources.get("simulator_steps_scope")
            and meter.get("probes_sha256_lf") == resources.get("probes_sha256_lf")
            and meter.get("jobs_sha256") == resources.get("jobs_sha256")
                == backend_binding.get("jobs_sha256")
            and all(meter_row.get(field) == resources.get(field) for field in (
                "probe", "checkpoint_sha256", "num_envs", "successful_step_calls",
                "simulator_steps", "completed"))
        )
        remote_launcher_wall = execution.get("remote_launcher_wall_s")
        rounded_launcher_wall = (round(remote_launcher_wall, 1)
                                 if _nonnegative_number(remote_launcher_wall) else None)
        launcher_watchdog = backend_binding.get("launcher_watchdog_s")
        cleanup_wall = execution.get("cleanup_wall_s")
        timed_out = execution.get("timed_out")
        exit_code = execution.get("exit_code")
        launcher_timing_ok = (
            _nonnegative_number(remote_launcher_wall)
            and _nonnegative_number(launcher_watchdog)
            and ((timed_out is False and exit_code != 124 and remote_launcher_wall <= launcher_watchdog)
                 or (timed_out is True and exit_code == 124 and remote_launcher_wall >= launcher_watchdog))
        )
        cleanup_evidence_ok = (
            _nonnegative_number(cleanup_wall)
            and execution.get("cleanup_scope") == "owned_process_group_and_post_gpu_snapshot"
            and execution.get("cleanup_group_alive") is False
            and execution.get("cleanup_compute_pids") == []
            and isinstance(execution.get("cleanup_memory_mib"), int)
            and not isinstance(execution.get("cleanup_memory_mib"), bool)
            and 0 <= execution["cleanup_memory_mib"] <= 2500
            and execution.get("cleanup_check_error") is False
        )
        cloud_numbers = (remote_launcher_wall, cleanup_wall, execution.get("transport_wall_s"),
                         execution.get("retrieval_wall_s"), execution.get("end_to_end_wall_s"))
        preapproval_wall = backend_binding.get("preapproval_transport_wall_s")
        transport_wall = execution.get("transport_wall_s")
        retrieval_wall = execution.get("retrieval_wall_s")
        end_to_end_wall = execution.get("end_to_end_wall_s")
        cost_timing_ok = (
            _nonnegative_number(preapproval_wall) and _nonnegative_number(transport_wall)
            and _nonnegative_number(retrieval_wall) and _nonnegative_number(end_to_end_wall)
            and _nonnegative_number(remote_launcher_wall) and _nonnegative_number(cleanup_wall)
            and transport_wall + 1e-6 >= preapproval_wall
            and end_to_end_wall + 1e-6 >= transport_wall + remote_launcher_wall + cleanup_wall + retrieval_wall
        )
        if (any(not _nonnegative_number(value) for value in cloud_numbers)
                or execution.get("backend_binding_sha256") != backend_digest
                or resources.get("backend_binding_sha256") != backend_digest
                or execution.get("jobs_sha256") != backend_binding.get("jobs_sha256")
                or resources.get("jobs_sha256") != backend_binding.get("jobs_sha256")
                or execution.get("gpu_wall_s") != rounded_launcher_wall
                or execution.get("execution_wall_s") != execution.get("gpu_wall_s")
                or execution.get("launcher_watchdog_s") != backend_binding.get("launcher_watchdog_s")
                or not launcher_timing_ok
                or not cleanup_evidence_ok
                or not meter_binding_ok
                or not cost_timing_ok
                or execution.get("cleanup_confirmed") is not True):
            errors.append(f"{augmented.get('request_id')}: cloud artifact binding 불일치")
    for ok, label in checks:
        if not ok:
            errors.append(f"{augmented.get('request_id')}: {label} 불일치")


def _cost_checks(folder: Path, result: dict, errors: list[str]) -> bool:
    error_count = len(errors)
    records = []
    for path in sorted((folder / "model").glob("*.attempt*.json")):
        value = _load(path, errors, path.name)
        if isinstance(value, dict):
            records.append(value)
    observed = result.get("observed", [])
    case = result.get("case")
    if (not isinstance(observed, list) or any(not isinstance(item, dict) for item in observed)
            or not isinstance(case, str)):
        errors.append("cost 재계산용 case/observed 형식 오류")
        return True
    resource_rows = []
    for receipt in observed:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("probe"), str):
            errors.append("cost 재계산용 receipt probe 형식 오류")
            return True
        resource_rows.append(_load(folder.parents[1] / "resources" / f"{case}__{receipt['probe']}.json",
                                   errors, "resource"))
    execution_rows = [_load(folder.parents[1] / "execution" / f"{case}__{receipt['probe']}.json", errors, "execution")
                      for receipt in observed]
    cpu_values = [row.get("cpu_s") if isinstance(row, dict) else None for row in execution_rows]
    step_values = [row.get("simulator_steps") if isinstance(row, dict) else None for row in resource_rows]
    gpu_values = [row.get("gpu_s") for row in observed if isinstance(row, dict)]

    def token_sum(field: str) -> int | None:
        values = [record.get(field) for record in records]
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
               for value in values):
            errors.append(f"model attempt {field}가 음이 아닌 정수 또는 null이 아니다")
            return None
        return sum(value for value in values if isinstance(value, int) and not isinstance(value, bool)) if all(
            isinstance(value, int) and not isinstance(value, bool) for value in values) else None

    def number_sum(values: list[Any], label: str, *, empty_zero: bool = False) -> int | float | None:
        if any(value is not None and not _nonnegative_number(value) for value in values):
            errors.append(f"{label}가 유한한 음이 아닌 수 또는 null이 아니다")
            return None
        if not values:
            return 0 if empty_zero else None
        if any(value is None for value in values):
            return None
        return sum(value for value in values if _nonnegative_number(value))

    input_tokens = token_sum("input_tokens")
    output_tokens = token_sum("output_tokens")
    for record in records:
        total = record.get("total_tokens")
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
            errors.append("model attempt total_tokens가 음이 아닌 정수 또는 null이 아니다")
        elif (isinstance(record.get("input_tokens"), int) and not isinstance(record.get("input_tokens"), bool)
              and isinstance(record.get("output_tokens"), int) and not isinstance(record.get("output_tokens"), bool)
              and total != record["input_tokens"] + record["output_tokens"]):
            errors.append("model attempt total_tokens 합계가 다르다")
    expected = {
        "model_calls": len(records),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_wall_s": number_sum([r.get("execution_wall_s") for r in records], "model execution_wall_s",
                                   empty_zero=True),
        "model_cpu_s": number_sum([r.get("cpu_s") for r in records], "model cpu_s", empty_zero=True),
        "isaac_wall_s": number_sum(gpu_values, "receipt gpu_s", empty_zero=True),
        "isaac_cpu_s": number_sum(cpu_values, "execution cpu_s", empty_zero=True),
        "isaac_simulator_steps": number_sum(step_values, "resource simulator_steps", empty_zero=True),
        "human_review_s": None,
        "gpu_device_busy_s": None,
    }
    costs = result.get("costs")
    if not isinstance(costs, dict):
        errors.append("result.costs가 객체가 아니다")
        return True
    for key, value in expected.items():
        actual = costs.get(key)
        if key == "model_calls":
            valid_actual = not isinstance(actual, bool) and isinstance(actual, int) and actual >= 0
        elif key in ("input_tokens", "output_tokens"):
            valid_actual = (actual is None or
                            (not isinstance(actual, bool) and isinstance(actual, int) and actual >= 0))
        elif key in ("human_review_s", "gpu_device_busy_s"):
            valid_actual = actual is None
        else:
            valid_actual = actual is None or _nonnegative_number(actual)
        if not valid_actual:
            errors.append(f"costs.{key} 형식이 잘못됐다")
            continue
        same = (math.isclose(actual, value, rel_tol=0, abs_tol=1e-9)
                if _nonnegative_number(actual) and _nonnegative_number(value) else actual == value)
        if not same:
            errors.append(f"costs.{key}가 전체 시도/receipt 합계와 다르다")
    return len(errors) > error_count


def _gpu_wait_checks(folder: Path, stage: int, errors: list[str]) -> None:
    path = folder / f"gpu_wait_{stage}.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"GPU wait stage {stage} 누락 또는 JSON 오류: {exc}")
        return
    if not rows or any(not isinstance(row, dict) for row in rows):
        errors.append(f"GPU wait stage {stage} snapshot 목록이 비었다")
        return
    for index, row in enumerate(rows):
        ready, reasons = row.get("ready"), row.get("busy_reasons")
        if not isinstance(ready, bool) or not isinstance(reasons, list) or any(not isinstance(x, str) for x in reasons):
            errors.append(f"GPU wait stage {stage} snapshot 형식 오류")
            continue
        if ready != (not reasons):
            errors.append(f"GPU wait stage {stage} ready/reason 불일치")
        if index < len(rows) - 1 and ready:
            errors.append(f"GPU wait stage {stage} ready 뒤 추가 snapshot이 있다")
    if rows[-1].get("ready") is not True:
        errors.append(f"GPU wait stage {stage}가 ready로 끝나지 않았다")
    preflight = _load(folder / f"gpu_preflight_{stage}.json", errors, f"GPU preflight stage {stage}")
    if preflight != rows[-1]:
        errors.append(f"GPU wait stage {stage} 최종 snapshot과 preflight가 다르다")


def _result_state_checks(result: dict, initial_outcome: dict | None, pairs: list[dict],
                         decisions: list[dict], observed: list[dict], cost_invalid: bool,
                         errors: list[str]) -> None:
    if cost_invalid and result.get("stop_reason") == "cost_artifact_invalid":
        if (result.get("status") != "hold" or result.get("error_type") != "ModelHarnessError"
                or result.get("error") != "Isaac cost sidecar validation failed"):
            errors.append("cost artifact HOLD 요약이 안전한 harness 오류 계약과 다르다")
        return
    expected_status = "completed"
    expected_stop = "probe_budget_or_frozen_stop"
    expected_error: dict | None = None
    if isinstance(initial_outcome, dict) and initial_outcome.get("status") != "valid":
        expected_status, expected_stop = "hold", "initial_model_invalid"
        expected_error = initial_outcome.get("error") if isinstance(initial_outcome.get("error"), dict) else None
    else:
        fatal_outcome = next((outcome for pair in pairs for outcome in (pair.get("actual"), pair.get("control"))
                              if isinstance(outcome, dict) and outcome.get("status") == "harness_fatal"), None)
        invalid_actual = next((pair.get("actual") for pair in pairs
                               if isinstance(pair.get("actual"), dict)
                               and pair["actual"].get("status") in ("semantic_invalid", "infra_exhausted")), None)
        if isinstance(fatal_outcome, dict):
            expected_status, expected_stop = "hold", None
            expected_error = fatal_outcome.get("error") if isinstance(fatal_outcome.get("error"), dict) else None
        elif isinstance(invalid_actual, dict):
            expected_status, expected_stop = "hold", "actual_model_invalid"
            expected_error = invalid_actual.get("error") if isinstance(invalid_actual.get("error"), dict) else None
        elif any(receipt.get("outcome") == "unknown" for receipt in observed):
            expected_status, expected_stop = "hold", "unknown_receipt_metrics_not_used_for_decision"
        elif decisions and decisions[-1].get("action") == "hold":
            expected_status, expected_stop = "hold", "model_hold"
        elif decisions and decisions[-1].get("action") == "confirm":
            expected_stop = "model_confirm"
    if result.get("status") != expected_status or result.get("stop_reason") != expected_stop:
        errors.append("result status/stop_reason이 raw outcome과 마지막 판단에서 재계산한 값과 다르다")
    if expected_error is not None:
        if (result.get("error_type") != expected_error.get("type")
                or result.get("error") != expected_error.get("message")):
            errors.append("result 오류가 중단을 일으킨 raw outcome 오류와 다르다")
    elif result.get("error_type") is not None or result.get("error") is not None:
        errors.append("정상 종료 result에 raw outcome과 결속되지 않은 오류가 있다")


def _baseline_state_checks(result: dict, observed: list[dict], cost_invalid: bool,
                           errors: list[str]) -> None:
    if cost_invalid and result.get("stop_reason") == "cost_artifact_invalid":
        if (result.get("status") != "hold" or result.get("error_type") != "ModelHarnessError"
                or result.get("error") != "Isaac cost sidecar validation failed"):
            errors.append("baseline cost artifact HOLD 요약이 안전한 harness 오류 계약과 다르다")
        return
    expected_status = "hold" if any(item.get("outcome") == "unknown" for item in observed) else "completed"
    expected_stop = ("unknown_receipt_metrics_not_used_for_decision"
                     if expected_status == "hold" else "probe_budget_or_frozen_stop")
    if result.get("status") != expected_status or result.get("stop_reason") != expected_stop:
        errors.append("baseline result status/stop_reason 재계산 불일치")
    if result.get("error_type") is not None or result.get("error") is not None:
        errors.append("정상 baseline result에 오류가 있다")


def audit_case(root: Path, tag: str, case: str, protocol: dict) -> dict:
    """Return a fail-closed audit record for one registered case."""
    root, errors = Path(root).resolve(), []
    errors.extend(code_integrity_errors(root, protocol))
    spec = _spec(protocol, case)
    if spec is None:
        return {"case": case, "valid": False, "errors": ["protocol에 등록되지 않은 사례"],
                "proof_changed": False, "verified_stage_indices": [], "result_sha256_lf": None}
    _source_checks(root, spec, errors)
    folder = root / "evals" / "results" / tag / "cases" / case
    result = _load(folder / "result.json", errors, "result")
    if not isinstance(result, dict):
        return {"case": case, "valid": False, "errors": _sanitize_errors(root, errors),
                "proof_changed": False, "verified_stage_indices": [], "result_sha256_lf": None}
    result_sha = _sha(folder / "result.json")
    if result.get("case") != case or result.get("tag") != tag or result.get("schema") != "active_diagnosis_case_v2":
        errors.append("result schema/case/tag 불일치")
    controller = result.get("controller")
    if controller not in {"model", "fixed_order", "cheapest_discrimination"}:
        errors.append("active 감사 대상 controller가 등록값이 아니다")
    if result.get("protocol_sha256_lf") != protocol.get("_audit_sha256_lf"):
        errors.append("result protocol SHA 불일치")
    if result.get("approval_identity_attested") is not False or result.get("human_review_s") is not None:
        errors.append("승인 신원 미확인과 human review 미측정 상태가 명시되지 않았다")
    if _contains_host_absolute_path(result.get("error")):
        errors.append("raw result 오류에 호스트 절대 경로가 있다")
    decisions, observed = result.get("decisions", []), result.get("observed", [])
    if not isinstance(decisions, list) or any(not isinstance(item, dict) for item in decisions):
        errors.append("result.decisions가 decision 객체 목록이 아니다")
        decisions = []
    if not isinstance(observed, list) or any(not isinstance(item, dict) for item in observed):
        errors.append("result.observed가 receipt 객체 목록이 아니다")
        observed = []
    budget_left = protocol.get("max_probes")
    if isinstance(budget_left, bool) or not isinstance(budget_left, int) or budget_left < len(observed):
        errors.append("protocol max_probes와 observed 수가 맞지 않는다")
        budget_left = len(observed)
    if result.get("evidence_state") != D.evidence_state(observed, budget_left - len(observed)):
        errors.append("result evidence_state 재계산 불일치")
    expected_action = decisions[-1].get("action") if decisions else None
    if result.get("model_action") != expected_action:
        errors.append("result model_action이 마지막 raw decision과 다르다")
    if controller in {"fixed_order", "cheapest_discrimination"}:
        initial_packet = _load(folder / "initial_packet.json", errors, "initial packet")
        if (not isinstance(initial_packet, dict) or initial_packet.get("phase") != "initial"
                or initial_packet.get("observed") != []
                or initial_packet.get("budget_left") != protocol.get("max_probes")
                or initial_packet.get("execution_budget_s") != protocol.get("probe_execution_budget_s")):
            errors.append("baseline initial packet 계약 불일치")
        for field in ("pairs", "without_receipt", "contrasts"):
            if result.get(field) != []:
                errors.append(f"baseline result.{field}가 빈 목록이 아니다")
        if decisions or result.get("initial_outcome") is not None or result.get("model_action") is not None:
            errors.append("baseline result에 모델 판단 상태가 있다")
        model_files = list((folder / "model").glob("*")) if (folder / "model").exists() else []
        if model_files:
            errors.append("baseline case에 모델 산출물이 있다")
        for stage in range(1, len(observed) + 1):
            _gpu_wait_checks(folder, stage, errors)
        _ledger_checks(root, folder, tag, result, spec, protocol, decisions, errors)
        cost_invalid = _cost_checks(folder, result, errors)
        _baseline_state_checks(result, observed, cost_invalid, errors)
        valid = not errors
        return {"case": case, "valid": valid, "errors": _sanitize_errors(root, errors),
                "proof_changed": False, "verified_stage_indices": [],
                "result_sha256_lf": result_sha}
    verified: list[int] = []
    reconstructed: list[dict] = []
    expected_stages: set[str] = set()
    _, initial_outcome, _ = _model_outcome(folder, "initial", protocol, errors)
    expected_stages.add("initial")
    if isinstance(initial_outcome, dict) and initial_outcome.get("status") == "valid":
        decision = initial_outcome.get("decision")
        if isinstance(decision, dict):
            reconstructed.append(decision)
            verified.append(0)
    pairs = result.get("pairs")
    shadows = result.get("without_receipt")
    contrasts = result.get("contrasts")
    if not isinstance(pairs, list) or any(not isinstance(item, dict) for item in pairs):
        errors.append("result.pairs가 pair 객체 목록이 아니다")
        pairs = []
    if not isinstance(shadows, list) or len(shadows) != len(pairs):
        errors.append("result.without_receipt가 pair와 정렬된 목록이 아니다")
        shadows = []
    if not isinstance(contrasts, list) or len(contrasts) != len(pairs):
        errors.append("result.contrasts가 pair와 정렬된 목록이 아니다")
        contrasts = []
    previous = reconstructed[0] if reconstructed else None
    for position, pair in enumerate(pairs, 1):
        if set(pair) != PAIR_KEYS:
            errors.append(f"pair {position}: schema가 다르다")
        stage = pair.get("stage")
        if stage != position or stage > len(observed):
            errors.append(f"pair {position}: stage가 receipt 순서와 다르다")
            continue
        order = pair.get("order")
        if order not in (["observed", "masked"], ["masked", "observed"]):
            errors.append(f"pair {stage}: arm 순서가 잘못됐다")
        pair_path = folder / "model" / f"pair_{stage}.json"
        saved_pair = _load(pair_path, errors, f"pair {stage}")
        if saved_pair != pair:
            errors.append(f"pair {stage}: 파일과 result가 다르다")
        actual_stage, control_stage = f"after_{stage}", f"without_receipt_{stage}"
        actual_packet = control_packet = None
        actual_outcome = loaded_control = None
        if pair.get("actual") is not None:
            expected_stages.add(actual_stage)
            actual_packet, actual_outcome, _ = _model_outcome(folder, actual_stage, protocol, errors)
        elif (folder / "model" / f"{actual_stage}.outcome.json").exists():
            errors.append(f"pair {stage}: null actual인데 산출물이 존재한다")
        control_outcome = pair.get("control")
        if control_outcome is not None:
            expected_stages.add(control_stage)
            control_packet, loaded_control, _ = _model_outcome(folder, control_stage, protocol, errors)
        elif (folder / "model" / f"{control_stage}.outcome.json").exists():
            errors.append(f"pair {stage}: null control인데 산출물이 존재한다")
        if pair.get("actual") is None or control_outcome is None:
            first = pair.get("order", [None])[0] if isinstance(pair.get("order"), list) else None
            called = pair.get("actual") if first == "observed" else pair.get("control")
            if not isinstance(called, dict) or called.get("status") != "harness_fatal":
                errors.append(f"pair {stage}: 미호출 arm은 첫 arm harness fatal에서만 허용된다")
        if pair.get("actual") != actual_outcome or control_outcome != loaded_control:
            errors.append(f"pair {stage}: outcome과 pair가 다르다")
        receipt = observed[stage - 1]
        if isinstance(actual_packet, dict) and isinstance(control_packet, dict):
            if control_packet != _masked_packet(actual_packet, receipt):
                errors.append(f"pair {stage}: masked packet이 latest receipt만 제거한 대조가 아니다")
        actual_valid = isinstance(actual_outcome, dict) and actual_outcome.get("status") == "valid"
        control_valid = isinstance(loaded_control, dict) and loaded_control.get("status") == "valid"
        eligible = actual_valid and control_valid
        if pair.get("proof_eligible") is not eligible:
            errors.append(f"pair {stage}: proof_eligible가 outcome 상태와 다르다")
        actual_decision = actual_outcome.get("decision") if actual_valid and isinstance(actual_outcome, dict) else None
        control_decision = loaded_control.get("decision") if control_valid and isinstance(loaded_control, dict) else None
        recorded = pair.get("contrast")
        recomputed = (D.contrast(previous, actual_decision, control_decision, observed[:stage])
                      if (eligible and isinstance(previous, dict) and isinstance(actual_decision, dict)
                          and isinstance(control_decision, dict)) else None)
        if recorded != recomputed:
            errors.append(f"pair {stage}: contrast 재계산 불일치")
        if position <= len(shadows) and shadows[position - 1] != control_decision:
            errors.append(f"pair {stage}: without_receipt 정렬 불일치")
        if position <= len(contrasts) and contrasts[position - 1] != recorded:
            errors.append(f"pair {stage}: contrasts 정렬 불일치")
        if actual_valid and isinstance(actual_decision, dict):
            reconstructed.append(actual_decision)
            verified.append(stage)
            previous = actual_decision
    if decisions != reconstructed:
        errors.append("result.decisions가 valid actual outcome 순서와 다르다")
    if result.get("initial_outcome") != initial_outcome:
        errors.append("result.initial_outcome이 outcome 파일과 다르다")
    invalid_actual = (not isinstance(initial_outcome, dict) or initial_outcome.get("status") != "valid") or any(
        not isinstance(pair.get("actual"), dict) or pair["actual"].get("status") != "valid" for pair in pairs)
    if invalid_actual and result.get("status") != "hold":
        errors.append("초기/actual invalid인데 result가 HOLD가 아니다")
    for stage in range(1, len(observed) + 1):
        _gpu_wait_checks(folder, stage, errors)
    for path in (folder / "model").glob("*.attempt*.json"):
        if path.name.split(".attempt", 1)[0] not in expected_stages:
            errors.append(f"result에 연결되지 않은 모델 attempt: {path.name}")
    _ledger_checks(root, folder, tag, result, spec, protocol, decisions, errors)
    cost_invalid = _cost_checks(folder, result, errors)
    _result_state_checks(result, initial_outcome, pairs, reconstructed, observed, cost_invalid, errors)
    valid = not errors
    proof = bool(
        valid and result.get("status") == "completed"
        and any(isinstance(contrast, dict) and all(contrast.get(gate) is True for gate in PROOF_GATES)
                for contrast in contrasts)
    )
    return {"case": case, "valid": valid, "errors": _sanitize_errors(root, errors),
            "proof_changed": proof, "verified_stage_indices": verified,
            "result_sha256_lf": result_sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not TAG.fullmatch(args.tag):
            raise ValueError("--tag 형식 오류")
        protocol = _read(args.protocol)
        if not isinstance(protocol, dict):
            raise ValueError("protocol은 JSON 객체여야 한다")
        if protocol.get("schema") != "active_diagnosis_protocol_v2":
            raise ValueError("protocol schema 오류")
        raw_cases = protocol.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("protocol.cases는 목록이어야 한다")
        cases = [item.get("case") for item in raw_cases if isinstance(item, dict)]
        if (len(cases) != 8 or any(not isinstance(case, str) or not case for case in cases)
                or len(set(cases)) != 8):
            raise ValueError("protocol은 중복 없는 등록 사례 8개여야 한다")
        case_names = [case for case in cases if isinstance(case, str)]
        bound = {**protocol, "_audit_sha256_lf": _sha(args.protocol)}
        output = {"schema": "active_diagnosis_audit_v2", "tag": args.tag,
                  "protocol_sha256_lf": bound["_audit_sha256_lf"],
                  "approval_identity_attested": False,
                  "cases": [audit_case(ROOT, args.tag, case, bound) for case in case_names]}
        output["valid"] = all(item["valid"] for item in output["cases"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"active audit 입력 오류: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out), "valid": output["valid"],
                      "proof_changed": sum(item["proof_changed"] for item in output["cases"])}, ensure_ascii=False))
    return 0 if output["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
