"""Audit active-diagnosis evidence chains without invoking a model or simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
from rl_triage import active_diagnosis as D, probe_loop as L  # noqa: E402

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


def _attempts(folder: Path, stage: str, errors: list[str]) -> list[tuple[Path, dict]]:
    paths = sorted((folder / "model").glob(f"{stage}.attempt*.json"))
    numbers = []
    records = []
    for path in paths:
        match = re.fullmatch(re.escape(stage) + r"\.attempt([12])\.json", path.name)
        if not match:
            errors.append(f"{stage}: 허용되지 않은 attempt 파일 {path.name}")
            continue
        numbers.append(int(match.group(1)))
        record = _load(path, errors, path.name)
        if isinstance(record, dict):
            records.append((path, record))
    if not records or numbers != list(range(1, len(numbers) + 1)):
        errors.append(f"{stage}: attempt가 1부터 연속된 1~2회 기록이 아니다")
    return records


def _model_stage(folder: Path, stage: str, expected: dict, protocol: dict,
                 errors: list[str]) -> tuple[dict | None, list[dict]]:
    packet_path = folder / "model" / f"{stage}.packet.json"
    decision_path = folder / "model" / f"{stage}.decision.json"
    packet = _load(packet_path, errors, f"{stage} packet")
    saved = _load(decision_path, errors, f"{stage} decision")
    if not isinstance(packet, dict) or not isinstance(saved, dict):
        return None, []
    if saved != expected:
        errors.append(f"{stage}: decision 파일과 result가 다르다")
    packet_raw_sha = _sha(packet_path, raw=True)
    message_sha = L.digest(D.decision_messages(packet))
    records = _attempts(folder, stage, errors)
    successful = []
    for path, record in records:
        if record.get("packet_sha256") != packet_raw_sha:
            errors.append(f"{path.name}: packet SHA가 다르다")
        if record.get("messages_sha256") != message_sha:
            errors.append(f"{path.name}: 실제 prompt messages SHA가 다르다")
        request = record.get("request", {})
        if request.get("model") != protocol.get("model") or request.get("temperature") != 0:
            errors.append(f"{path.name}: provider model 또는 temperature가 다르다")
        if record.get("saved_response_cached") is not False or record.get("client_cache_used") is not False:
            errors.append(f"{path.name}: fresh 호출 provenance가 아니다")
        if not all(key in record for key in ("response_id", "finish_reason", "reasoning_content")):
            errors.append(f"{path.name}: response provenance 필드가 누락됐다")
        if record.get("infra_error"):
            continue
        if record.get("exit_code") != 0 or not isinstance(record.get("response_id"), str):
            errors.append(f"{path.name}: 성공 response id/exit 기록이 없다")
            continue
        if any(record.get(key) is None for key in ("input_tokens", "output_tokens", "total_tokens")):
            errors.append(f"{path.name}: 성공 호출 usage가 없다")
        try:
            parsed = json.loads(_strip_fence(record.get("content") or ""))
            validated = D.validate_decision(parsed, packet)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path.name}: 모델 JSON을 계약대로 재생할 수 없다: {exc}")
            continue
        if validated != saved:
            errors.append(f"{path.name}: raw 모델 응답과 저장 decision이 다르다")
        else:
            successful.append(path)
    if len(successful) != 1:
        errors.append(f"{stage}: 저장 decision에 연결된 성공 호출이 정확히 하나가 아니다")
    return packet, [record for _, record in records]


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
    try:
        events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"ledger 누락 또는 JSON 오류: {exc}")
        return
    by_id: dict[str, list[dict]] = {}
    for event in events:
        by_id.setdefault(event.get("request_id"), []).append(event)
    if set(by_id) != {item.get("request_id") for item in observed}:
        errors.append("ledger 요청 집합과 result receipt 집합이 다르다")
    prior_digests = []
    protocol_sha = protocol.get("_audit_sha256_lf")
    for index, augmented in enumerate(observed, 1):
        if not isinstance(augmented, dict):
            continue
        rid, probe = augmented.get("request_id"), augmented.get("probe")
        if not isinstance(rid, str) or probe not in L.PROBES or index > len(decisions):
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
        expected_binding = {
            "controller": "model",
            "packet_sha256_lf": _sha(packet_path),
            "model_call_sha256_lf": _sha(attempts[-1]) if attempts else None,
            "decision_sha256_lf": _sha(decision_path),
            "protocol_sha256_lf": protocol_sha,
        }
        checks = (
            (proposed.get("prereg_digest") == L.digest(envelope), "envelope digest"),
            (proposed.get("args_digest") == L.digest(L.PROBES.get(probe, {}).get("args")), "args digest"),
            (proposed.get("probe") == probe == decisions[index - 1].get("probe") == envelope.get("probe"), "selected probe"),
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
        prereg_digest, args_digest = proposed.get("prereg_digest"), proposed.get("args_digest")
        if not isinstance(prereg_digest, str) or not isinstance(args_digest, str):
            errors.append(f"{rid}: 원장 digest 형식 오류")
            expected_effect = None
        else:
            expected_effect = L.Ledger.effect_key(prereg_digest, probe, args_digest)
        if expected_effect is None or proposed.get("effect_key") != expected_effect:
            errors.append(f"{rid}: effect key 불일치")
        _artifact_checks(root, tag, result["case"], probe, augmented, canonical, spec,
                         proposed.get("budget_gpu_s"), errors)
        prior_digests.append(augmented.get("receipt_digest"))


def _artifact_checks(root: Path, tag: str, case: str, probe: str, augmented: dict,
                     canonical: dict, spec: dict, budget_gpu_s: object, errors: list[str]) -> None:
    base = root / "evals" / "results" / tag
    report_path = base / "probes" / f"{case}__{probe}.json"
    report = _load(report_path, errors, f"{probe} report")
    execution = _load(base / "execution" / f"{case}__{probe}.json", errors, f"{probe} execution")
    resources = _load(base / "resources" / f"{case}__{probe}.json", errors, f"{probe} resources")
    if not isinstance(report, dict) or not isinstance(execution, dict) or not isinstance(resources, dict):
        return
    checkpoint, report_sha = spec.get("checkpoint_sha256"), _sha(report_path)
    probe_cells = report.get("probes")
    if not isinstance(probe_cells, dict) or not isinstance(probe_cells.get(probe), dict):
        errors.append(f"{probe}: report probe cell 형식 오류")
        return
    cell = probe_cells[probe]
    reference_source = spec.get("sources", {}).get("probe_reference", {})
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
    reference = reference_cell.get("measurement")
    expected_outcome = L.classify(probe, cell.get("measurement"), reference) if cell.get("exit") == 0 else "unknown"
    over = (canonical.get("gpu_s") is not None and isinstance(budget_gpu_s, (int, float))
            and canonical["gpu_s"] > budget_gpu_s)
    if canonical.get("exit_code") != 0 or over:
        expected_outcome = "unknown"
    checks = (
        (report.get("checkpoint", {}).get("sha256") == checkpoint == augmented.get("checkpoint_sha256"), "checkpoint SHA"),
        (augmented.get("report_sha256_lf") == report_sha, "report SHA"),
        (cell.get("measurement") == canonical.get("measurement"), "measurement"),
        (cell.get("exit") == canonical.get("exit_code"), "exit code"),
        (canonical.get("over_budget") == over and canonical.get("outcome") == expected_outcome, "classification/over-budget"),
        (execution.get("gpu_wall_s") == canonical.get("gpu_s") == execution.get("execution_wall_s"), "execution wall"),
        (execution.get("exit_code") == canonical.get("exit_code"), "execution exit code"),
        (execution.get("case") == case == resources.get("case") and execution.get("probe") == probe == resources.get("probe"), "artifact case/probe"),
        (execution.get("checkpoint_sha256") == checkpoint and execution.get("report_sha256_lf") == report_sha, "execution binding"),
        (resources.get("checkpoint_sha256") == checkpoint and resources.get("report_sha256_lf") == report_sha, "resource binding"),
        (execution.get("cpu_s") == resources.get("cpu_s", 0) + execution.get("parent_cpu_s", 0)
         and execution.get("simulator_steps") == resources.get("simulator_steps"), "resource totals"),
    )
    for ok, label in checks:
        if not ok:
            errors.append(f"{augmented.get('request_id')}: {label} 불일치")


def _cost_checks(folder: Path, result: dict, errors: list[str]) -> None:
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
        return
    resource_rows = []
    for receipt in observed:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("probe"), str):
            errors.append("cost 재계산용 receipt probe 형식 오류")
            return
        resource_rows.append(_load(folder.parents[1] / "resources" / f"{case}__{receipt['probe']}.json",
                                   errors, "resource"))
    execution_rows = [_load(folder.parents[1] / "execution" / f"{case}__{receipt['probe']}.json", errors, "execution")
                      for receipt in observed]
    cpu_values = [row.get("cpu_s") if isinstance(row, dict) else None for row in execution_rows]
    step_values = [row.get("simulator_steps") if isinstance(row, dict) else None for row in resource_rows]
    gpu_values = [row.get("gpu_s") for row in observed if isinstance(row, dict)]
    expected = {
        "model_calls": len(records),
        "input_tokens": sum(r["input_tokens"] for r in records) if all(r.get("input_tokens") is not None for r in records) else None,
        "output_tokens": sum(r["output_tokens"] for r in records) if all(r.get("output_tokens") is not None for r in records) else None,
        "model_wall_s": sum(r.get("execution_wall_s", 0) for r in records),
        "model_cpu_s": sum(r.get("cpu_s", 0) for r in records) if all(r.get("cpu_s") is not None for r in records) else None,
        "isaac_wall_s": sum(value for value in gpu_values if isinstance(value, (int, float))) if all(isinstance(value, (int, float)) for value in gpu_values) else None,
        "isaac_cpu_s": sum(value for value in cpu_values if isinstance(value, (int, float))) if cpu_values and all(isinstance(value, (int, float)) for value in cpu_values) else (0 if not observed else None),
        "isaac_simulator_steps": sum(value for value in step_values if isinstance(value, (int, float))) if step_values and all(isinstance(value, (int, float)) for value in step_values) else (0 if not observed else None),
        "human_review_s": None,
        "gpu_device_busy_s": None,
    }
    for key, value in expected.items():
        actual = result.get("costs", {}).get(key)
        same = (math.isclose(actual, value, rel_tol=0, abs_tol=1e-9)
                if isinstance(actual, (int, float)) and isinstance(value, (int, float)) else actual == value)
        if not same:
            errors.append(f"costs.{key}가 전체 시도/receipt 합계와 다르다")


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
        return {"case": case, "valid": False, "errors": errors,
                "proof_changed": False, "verified_stage_indices": [], "result_sha256_lf": None}
    result_sha = _sha(folder / "result.json")
    if result.get("case") != case or result.get("tag") != tag or result.get("schema") != "active_diagnosis_case_v1":
        errors.append("result schema/case/tag 불일치")
    if result.get("controller") != "model":
        errors.append("active 감사 대상 controller가 model이 아니다")
    if result.get("protocol_sha256_lf") != protocol.get("_audit_sha256_lf"):
        errors.append("result protocol SHA 불일치")
    if result.get("approval_identity_attested") is not False or result.get("human_review_s") is not None:
        errors.append("승인 신원 미확인과 human review 미측정 상태가 명시되지 않았다")
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
    verified = []
    for index, decision in enumerate(decisions):
        stage = "initial" if index == 0 else f"after_{index}"
        error_count = len(errors)
        packet, records = _model_stage(folder, stage, decision, protocol, errors)
        if packet is not None and len(errors) == error_count:
            verified.append(index)
    shadows = result.get("without_receipt", [])
    contrasts = result.get("contrasts", [])
    if not isinstance(shadows, list) or any(not isinstance(item, dict) for item in shadows):
        errors.append("result.without_receipt가 decision 객체 목록이 아니다")
        shadows = []
    if not isinstance(contrasts, list) or any(not isinstance(item, dict) for item in contrasts):
        errors.append("result.contrasts가 contrast 객체 목록이 아니다")
        contrasts = []
    for index, (receipt, shadow, recorded) in enumerate(zip(observed, shadows, contrasts), 1):
        actual = _load(folder / "model" / f"after_{index}.packet.json", errors, f"after_{index} packet")
        masked, records = _model_stage(folder, f"without_receipt_{index}", shadow, protocol, errors)
        if isinstance(actual, dict) and masked != _masked_packet(actual, receipt):
            errors.append(f"stage {index}: masked packet이 latest receipt만 제거한 대조가 아니다")
        if index < len(decisions):
            recomputed = D.contrast(decisions[index - 1], decisions[index], shadow, observed[:index])
            if recorded != recomputed:
                errors.append(f"stage {index}: contrast 재계산 불일치")
    usable_receipts = sum(D.receipt_facts(receipt) != {} for receipt in observed)
    if not (len(shadows) == len(contrasts) == usable_receipts):
        errors.append("post/masked/contrast 단계 수가 다르다")
    expected_stages = {"initial", *(f"after_{index}" for index in range(1, len(decisions))),
                       *(f"without_receipt_{index}" for index in range(1, len(shadows) + 1))}
    for path in (folder / "model").glob("*.attempt*.json"):
        if path.name.split(".attempt", 1)[0] not in expected_stages:
            errors.append(f"result에 연결되지 않은 모델 attempt: {path.name}")
    _ledger_checks(root, folder, tag, result, spec, protocol, decisions, errors)
    _cost_checks(folder, result, errors)
    valid = not errors
    proof = bool(
        valid and result.get("status") == "completed"
        and any(all(contrast.get(gate) is True for gate in PROOF_GATES) for contrast in contrasts)
    )
    return {"case": case, "valid": valid, "errors": errors,
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
        if protocol.get("schema") != "active_diagnosis_protocol_v1":
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
        output = {"schema": "active_diagnosis_audit_v1", "tag": args.tag,
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
