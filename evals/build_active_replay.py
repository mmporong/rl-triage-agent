"""공개 active diagnosis 결과로 오프라인 심사 리플레이 HTML과 요약 JSON을 만든다.

모델·네트워크·GPU를 사용하지 않는다. 결과의 완료 표식만 믿지 않고 공개 원장과
실행 산출물을 대조한다. 검증할 수 없는 사례는 화면에서 HOLD로 남긴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from audit_active_diagnosis import code_integrity_errors  # noqa: E402

PROTOCOL = Path("bench/protocols/active_diagnosis_v1.json")
TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CASE_SCHEMA = "active_diagnosis_case_v1"
REQUIRED_COSTS = (
    "model_calls", "input_tokens", "output_tokens", "model_wall_s", "model_cpu_s",
    "isaac_wall_s", "isaac_cpu_s", "isaac_simulator_steps", "human_review_s",
    "gpu_device_busy_s",
)
CONTRAST_GATES = (
    "leader_changed", "latest_receipt_cited",
    "changed_leader_rejected_by_actual_observation", "after_leader_survives_actual_observation",
    "ablation_preserves_before_leader",
)


class BuildError(ValueError):
    """입력 경로나 공개 스냅샷 계약이 안전하지 않을 때 발생한다."""


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def canonical_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BuildError(f"저장소 상대 POSIX 경로가 아니다: {value!r}")
    candidate = Path(value)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise BuildError(f"저장소 경계를 벗어난 경로: {value!r}")
    resolved = (ROOT / candidate).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise BuildError(f"저장소 경계를 벗어난 경로: {value!r}")
    return resolved


class Snapshot:
    def __init__(self) -> None:
        self.sources: dict[str, str] = {}

    def read_json(self, relative: str, *, required: bool = True) -> dict | list | None:
        path = _relative_path(relative)
        if not path.is_file():
            if required:
                raise BuildError(f"공개 산출물이 없다: {relative}")
            return None
        raw = path.read_bytes()
        self.sources[relative] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BuildError(f"JSON을 읽을 수 없다: {relative} ({exc})") from None

    def read_jsonl(self, relative: str) -> list[dict]:
        path = _relative_path(relative)
        if not path.is_file():
            raise BuildError(f"공개 원장이 없다: {relative}")
        self.sources[relative] = sha256_lf(path)
        rows: list[dict] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f"{relative}:{number} JSON 오류 ({exc.msg})") from None
            if not isinstance(row, dict):
                raise BuildError(f"{relative}:{number} 원장 행이 객체가 아니다")
            rows.append(row)
        return rows

    def record_file(self, relative: str) -> Path:
        path = _relative_path(relative)
        if not path.is_file():
            raise BuildError(f"공개 산출물이 없다: {relative}")
        self.sources[relative] = sha256_lf(path)
        return path


def _case_names(protocol: dict) -> list[str]:
    raw = protocol.get("cases")
    if not isinstance(raw, list):
        raise BuildError("protocol.cases가 목록이 아니다")
    names: list[str] = []
    for item in raw:
        name = item.get("case") if isinstance(item, dict) else None
        if not isinstance(name, str) or not TAG_RE.fullmatch(name):
            raise BuildError("protocol에는 이름이 있는 사례 8개가 필요하다")
        names.append(name)
    if len(names) != 8:
        raise BuildError("protocol에는 이름이 있는 사례 8개가 필요하다")
    if len(set(names)) != 8:
        raise BuildError("protocol 사례 이름이 중복됐다")
    return names


def _case_spec(protocol: dict, case: str) -> dict:
    for item in protocol["cases"]:
        if isinstance(item, dict) and item.get("case") == case:
            return item
    return {"case": case}


def _verify_registered_sources(spec: dict, snap: Snapshot) -> list[str]:
    failures: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if set(("path", "sha256_lf")) <= value.keys():
                _relative_path(value["path"])
                try:
                    path = snap.record_file(value["path"])
                    if sha256_lf(path) != value["sha256_lf"]:
                        failures.append(f"사전등록 입력 해시 불일치: {value['path']}")
                except BuildError as exc:
                    failures.append(str(exc))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(spec.get("sources", {}))
    return failures


def _strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        return value[7:-3].strip()
    return value


def _verify_model_chain(case_dir: str, result: dict, snap: Snapshot) -> list[str]:
    failures: list[str] = []
    decisions = result.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return ["판단 기록이 없다"]

    def verify_stage(stage: str, expected: dict) -> None:
        packet_rel = f"{case_dir}/model/{stage}.packet.json"
        decision_rel = f"{case_dir}/model/{stage}.decision.json"
        try:
            packet_path = snap.record_file(packet_rel)
            actual_decision = snap.read_json(decision_rel)
            if actual_decision != expected:
                failures.append(f"{stage} 파싱 판단이 result와 다르다")
                return
            attempts = sorted(_relative_path(f"{case_dir}/model").glob(f"{stage}.attempt*.json"))
            if not attempts:
                failures.append(f"{stage} 모델 호출 기록이 없다")
                return
            successful = False
            packet_sha = hashlib.sha256(packet_path.read_bytes()).hexdigest()
            for attempt_path in attempts:
                rel = attempt_path.relative_to(ROOT).as_posix()
                call = snap.read_json(rel)
                if not isinstance(call, dict) or call.get("packet_sha256") != packet_sha:
                    failures.append(f"{rel} 입력 패킷 해시가 다르다")
                    continue
                if not call.get("infra_error") and call.get("exit_code") == 0:
                    try:
                        parsed = json.loads(_strip_json_fence(call.get("content") or ""))
                    except json.JSONDecodeError:
                        failures.append(f"{rel} 응답을 JSON으로 재생할 수 없다")
                    else:
                        successful = successful or parsed == actual_decision
            if not successful:
                failures.append(f"{stage} 성공 응답과 파싱 판단이 연결되지 않는다")
        except BuildError as exc:
            failures.append(str(exc))

    for index, expected in enumerate(decisions):
        verify_stage("initial" if index == 0 else f"after_{index}", expected)
    shadows = result.get("without_receipt")
    if not isinstance(shadows, list) or len(shadows) != max(0, len(decisions) - 1):
        failures.append("receipt 제거 대조 판단 수가 실제 판단 단계와 다르다")
    else:
        for index, expected in enumerate(shadows, 1):
            verify_stage(f"without_receipt_{index}", expected)
    return failures


def _verify_ledger(case_dir: str, result: dict, protocol: dict, snap: Snapshot) -> list[str]:
    failures: list[str] = []
    observed = result.get("observed")
    if not isinstance(observed, list) or not observed:
        return ["실제 관측 receipt가 없다"]
    try:
        snap.read_json(f"{case_dir}/prereg.json")
        events = snap.read_jsonl(f"{case_dir}/ledger.jsonl")
    except BuildError as exc:
        return [str(exc)]
    checkpoint = result.get("checkpoint_sha256")
    spec = _case_spec(protocol, str(result.get("case")))
    by_request: dict[str, list[dict]] = {}
    for event in events:
        rid = event.get("request_id")
        if isinstance(rid, str):
            by_request.setdefault(rid, []).append(event)
        else:
            failures.append("원장에 request_id가 없는 event가 있다")
    observed_ids = {
        receipt.get("request_id") for receipt in observed
        if isinstance(receipt, dict) and isinstance(receipt.get("request_id"), str)
    }
    if set(by_request) != observed_ids:
        failures.append("원장 전체 요청 집합과 result receipt 집합이 다르다")
    seen: set[str] = set()
    for receipt_index, receipt in enumerate(observed):
        if not isinstance(receipt, dict) or not isinstance(receipt.get("request_id"), str):
            failures.append("관측 receipt 형식이 잘못됐다")
            continue
        rid = receipt["request_id"]
        if rid in seen:
            failures.append(f"중복 관측 request_id: {rid}")
            continue
        seen.add(rid)
        chain = by_request.get(rid, [])
        if [event.get("event") for event in chain] != ["proposed", "approved", "consumed", "receipt"]:
            failures.append(f"{rid} 원장 순서가 단일 proposed→approved→consumed→receipt가 아니다")
            continue
        proposed, approved, _, ledger_receipt = chain
        try:
            envelope = snap.read_json(f"{case_dir}/approvals/{rid}.json")
        except BuildError as exc:
            failures.append(str(exc))
            continue
        if not isinstance(envelope, dict):
            failures.append(f"{rid} 승인 envelope가 JSON 객체가 아니다")
            continue
        if proposed.get("prereg_digest") != canonical_digest(envelope) or proposed.get("probe") != receipt.get("probe"):
            failures.append(f"{rid} 승인 대상과 관측 대상이 다르다")
        probe = receipt.get("probe")
        probe_arguments = protocol.get("probe_arguments")
        canonical_args = probe_arguments.get(probe) if isinstance(probe_arguments, dict) else None
        args_digest = canonical_digest(canonical_args)
        effect_key = canonical_digest({"prereg": proposed.get("prereg_digest"),
                                       "probe": probe, "args": args_digest})[:16]
        if proposed.get("args_digest") != args_digest or proposed.get("effect_key") != effect_key:
            failures.append(f"{rid} 원장 args digest 또는 effect key가 protocol과 다르다")
        if proposed.get("budget_gpu_s") != protocol.get("probe_execution_budget_s"):
            failures.append(f"{rid} 원장 실행 예산이 protocol과 다르다")
        if (envelope.get("case") != result.get("case")
                or envelope.get("checkpoint_sha256") != checkpoint
                or envelope.get("protocol_sha256_lf") != result.get("protocol_sha256_lf")):
            failures.append(f"{rid} 승인 envelope가 사례·체크포인트·protocol과 다르다")
        prior = [item.get("receipt_digest") for item in observed[:receipt_index]]
        if envelope.get("prior_receipt_digests") != prior:
            failures.append(f"{rid} 승인 envelope의 이전 receipt 연결이 다르다")
        if (envelope.get("probe") != probe or envelope.get("args") != canonical_args
                or envelope.get("thresholds") != protocol.get("thresholds")
                or envelope.get("budget_probes") != protocol.get("max_probes")
                or envelope.get("reference_source") != spec.get("sources", {}).get("probe_reference")):
            failures.append(f"{rid} 승인 envelope의 probe 조건이 protocol과 다르다")
        binding = envelope.get("binding", {})
        if not isinstance(binding, dict):
            failures.append(f"{rid} 승인 binding이 JSON 객체가 아니다")
            binding = {}
        if binding.get("controller") != result.get("controller"):
            failures.append(f"{rid} 승인 binding의 controller가 다르다")
        for key in ("packet_sha256_lf", "model_call_sha256_lf", "decision_sha256_lf"):
            if result.get("controller") == "model" and key not in binding:
                failures.append(f"{rid} 승인 binding에 {key}가 없다")
        if result.get("controller") == "model":
            stage = "initial" if receipt_index == 0 else f"after_{receipt_index}"
            model_dir = _relative_path(f"{case_dir}/model")
            calls = sorted(model_dir.glob(f"{stage}.attempt*.json"))
            bound_files = {
                "packet_sha256_lf": model_dir / f"{stage}.packet.json",
                "decision_sha256_lf": model_dir / f"{stage}.decision.json",
                "model_call_sha256_lf": calls[-1] if calls else model_dir / f"{stage}.attempt-missing.json",
            }
            for key, path in bound_files.items():
                try:
                    rel = path.relative_to(ROOT).as_posix()
                    actual = sha256_lf(snap.record_file(rel))
                except (BuildError, ValueError) as exc:
                    failures.append(str(exc))
                else:
                    if binding.get(key) != actual:
                        failures.append(f"{rid} 승인 binding의 {key}가 실제 파일과 다르다")
        else:
            try:
                initial = snap.record_file(f"{case_dir}/initial_packet.json")
                if binding.get("initial_packet_sha256_lf") != sha256_lf(initial):
                    failures.append(f"{rid} baseline 승인 binding의 initial packet 해시가 다르다")
            except BuildError as exc:
                failures.append(str(exc))
        if approved.get("approver") != result.get("approval_actor") or approved.get("approver") != protocol.get("approver"):
            failures.append(f"{rid} 승인 주체가 result와 다르다")
        for field in ("outcome", "measurement", "gpu_s", "exit_code", "over_budget"):
            if ledger_receipt.get(field) != receipt.get(field):
                failures.append(f"{rid} 원장 receipt의 {field}가 result와 다르다")
        if receipt.get("receipt_digest") != canonical_digest(ledger_receipt):
            failures.append(f"{rid} receipt digest가 원장 행과 다르다")
        report_rel = f"evals/results/{result.get('tag')}/probes/{result.get('case')}__{probe}.json"
        execution_rel = f"evals/results/{result.get('tag')}/execution/{result.get('case')}__{probe}.json"
        resources_rel = f"evals/results/{result.get('tag')}/resources/{result.get('case')}__{probe}.json"
        try:
            report_path = snap.record_file(report_rel)
            execution = snap.read_json(execution_rel)
            resources = snap.read_json(resources_rel)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_sha = sha256_lf(report_path)
            if receipt.get("report_sha256_lf") != report_sha:
                failures.append(f"{rid} report SHA256이 다르다")
            for artifact, label in ((execution, "execution"), (resources, "resources")):
                if not isinstance(artifact, dict) or artifact.get("case") != result.get("case") or artifact.get("probe") != probe:
                    failures.append(f"{rid} {label} 사례 또는 probe가 다르다")
                elif artifact.get("checkpoint_sha256") != checkpoint or artifact.get("report_sha256_lf") != report_sha:
                    failures.append(f"{rid} {label} 체크포인트 또는 report 해시가 다르다")
            if isinstance(execution, dict) and isinstance(resources, dict):
                if execution.get("gpu_wall_s") != receipt.get("gpu_s") or execution.get("exit_code") != receipt.get("exit_code"):
                    failures.append(f"{rid} execution wall 또는 exit가 receipt와 다르다")
                parent_cpu, resource_cpu = execution.get("parent_cpu_s"), resources.get("cpu_s")
                if not isinstance(parent_cpu, (int, float)) or not isinstance(resource_cpu, (int, float)):
                    failures.append(f"{rid} CPU 측정값이 없다")
                elif execution.get("cpu_s") != parent_cpu + resource_cpu:
                    failures.append(f"{rid} execution CPU가 parent와 resource 합계가 아니다")
                if execution.get("simulator_steps") != resources.get("simulator_steps"):
                    failures.append(f"{rid} simulator step 합계가 다르다")
            report_cell = report.get("probes", {}).get(probe, {})
            if (report_cell.get("measurement") != receipt.get("measurement")
                    or report_cell.get("exit") != receipt.get("exit_code")):
                failures.append(f"{rid} report 측정값 또는 exit가 receipt와 다르다")
            if report.get("checkpoint", {}).get("sha256") != checkpoint or receipt.get("checkpoint_sha256") != checkpoint:
                failures.append(f"{rid} 체크포인트 SHA256이 다르다")
        except (BuildError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
    return failures


def _audit_records(audit: dict | None) -> dict[str, dict]:
    if not isinstance(audit, dict):
        return {}
    cases = audit.get("cases")
    if isinstance(cases, dict):
        return {str(case): row for case, row in cases.items() if isinstance(row, dict)}
    if isinstance(cases, list):
        return {row["case"]: row for row in cases
                if isinstance(row, dict) and isinstance(row.get("case"), str)}
    return {}


def _contrast(result: dict) -> tuple[int | None, dict]:
    contrasts = result.get("contrasts")
    if isinstance(contrasts, list) and contrasts:
        for index, contrast in enumerate(contrasts):
            if isinstance(contrast, dict) and all(contrast.get(gate) is True for gate in CONTRAST_GATES):
                return index, contrast
        return len(contrasts) - 1, contrasts[-1] if isinstance(contrasts[-1], dict) else {}
    raw_legacy = result.get("contrast")
    legacy: dict = raw_legacy if isinstance(raw_legacy, dict) else {}
    return (0 if legacy else None), legacy


def _public_case(case: str, tag: str, protocol: dict, snap: Snapshot,
                 audit: dict | None, audits: dict[str, dict]) -> dict:
    case_dir = f"evals/results/{tag}/cases/{case}"
    result_rel = f"{case_dir}/result.json"
    reasons = _verify_registered_sources(_case_spec(protocol, case), snap)
    result = snap.read_json(result_rel, required=False)
    if not isinstance(result, dict):
        return {"case": case, "state": "hold", "proof": False,
                "reasons": reasons + ["active result.json이 없다"], "result": None,
                "baselines": {}}
    if result.get("schema") != CASE_SCHEMA:
        reasons.append("active 결과 schema가 다르다")
    if result.get("case") != case or result.get("tag") != tag:
        reasons.append("active 결과의 case 또는 tag가 경로와 다르다")
    if result.get("status") != "completed":
        reasons.append("active 실행 상태가 completed가 아니다")
    if result.get("controller") != "model":
        reasons.append("active 결과 controller가 model이 아니다")
    for field in ("approval_actor", "human_review_s", "gpu_device_busy_s"):
        if field not in result:
            reasons.append(f"{field}가 누락됐다")
    costs = result.get("costs")
    if not isinstance(costs, dict):
        reasons.append("비용 객체가 없다")
    else:
        for field in REQUIRED_COSTS:
            if field not in costs:
                reasons.append(f"비용 필드 {field}가 누락됐다")
    decisions = result.get("decisions")
    if isinstance(decisions, list):
        for number, decision in enumerate(decisions, 1):
            if not isinstance(decision, dict) or not isinstance(decision.get("ranking"), list) or len(decision["ranking"]) != 6:
                reasons.append(f"판단 {number}의 6개 가설 순위가 없다")
    contrast_index, contrast = _contrast(result)
    for gate in CONTRAST_GATES:
        if contrast.get(gate) is not True:
            reasons.append(f"대조 gate {gate}가 true가 아니다")
    protocol_sha = snap.sources[PROTOCOL.as_posix()]
    if result.get("protocol_sha256_lf") != protocol_sha:
        reasons.append("결과가 현재 사전등록 protocol에 묶이지 않았다")
    reasons += _verify_ledger(case_dir, result, protocol, snap)
    reasons += _verify_model_chain(case_dir, result, snap)
    audit_case = audits.get(case)
    if audit_case is None:
        reasons.append("독립 auditor 사례 기록이 없다")
    else:
        if audit_case.get("valid") is not True:
            reasons.append("독립 auditor가 사례 산출물을 유효하다고 판정하지 않았다")
        if audit_case.get("proof_changed") is not True:
            reasons.append("독립 auditor가 판단 변경 증명을 확인하지 않았다")
        verified = audit_case.get("verified_stage_indices")
        after_decision_index = None if contrast_index is None else contrast_index + 1
        if after_decision_index is None or not isinstance(verified, list) or after_decision_index not in verified:
            reasons.append("선택한 판단 변경 단계가 auditor 검증 목록에 없다")
        if audit_case.get("result_sha256_lf") != snap.sources[result_rel]:
            reasons.append("독립 auditor의 result SHA256이 다르다")
    if isinstance(audit, dict):
        if not str(audit.get("schema", "")).startswith("active_diagnosis_audit"):
            reasons.append("독립 auditor schema가 다르다")
        if audit.get("tag") not in (None, tag):
            reasons.append("독립 auditor tag가 다르다")
        if audit.get("protocol_sha256_lf") != protocol_sha:
            reasons.append("독립 auditor의 protocol SHA256이 다르다")
    reasons = list(dict.fromkeys(reasons))
    return {"case": case, "state": "verified" if not reasons else "hold", "proof": not reasons,
            "reasons": reasons, "result": result, "contrast": contrast,
            "contrast_index": contrast_index, "baselines": {}}


def _baseline(case: str, tag: str, controller: str, protocol: dict, snap: Snapshot) -> dict:
    rel = f"evals/results/{tag}/cases/{case}/result.json"
    result = snap.read_json(rel, required=False)
    reasons: list[str] = []
    if not isinstance(result, dict):
        return {"state": "hold", "reasons": [f"{controller} result.json이 없다"], "result": None}
    if result.get("schema") != CASE_SCHEMA or result.get("case") != case or result.get("tag") != tag:
        reasons.append("baseline schema, case 또는 tag가 경로와 다르다")
    if result.get("controller") != controller:
        reasons.append("baseline controller가 다르다")
    if result.get("status") != "completed":
        reasons.append("baseline 실행 상태가 completed가 아니다")
    costs = result.get("costs")
    if result.get("decisions") != [] or not isinstance(costs, dict) or costs.get("model_calls") != 0:
        reasons.append("baseline이 모델 호출 없는 규칙 제어기 조건을 만족하지 않는다")
    for field in REQUIRED_COSTS:
        if not isinstance(costs, dict) or field not in costs:
            reasons.append(f"baseline 비용 필드 {field}가 누락됐다")
    if result.get("protocol_sha256_lf") != snap.sources[PROTOCOL.as_posix()]:
        reasons.append("baseline이 현재 사전등록 protocol에 묶이지 않았다")
    reasons += _verify_registered_sources(_case_spec(protocol, case), snap)
    case_dir = f"evals/results/{tag}/cases/{case}"
    reasons += _verify_ledger(case_dir, result, protocol, snap)
    model_dir = _relative_path(f"{case_dir}/model")
    if model_dir.is_dir() and any(model_dir.glob("*.attempt*.json")):
        reasons.append("baseline에 모델 호출 파일이 있다")
    if isinstance(costs, dict):
        for field in ("model_calls", "input_tokens", "output_tokens", "model_wall_s", "model_cpu_s"):
            if costs.get(field) != 0:
                reasons.append(f"baseline 비용 {field}가 0이 아니다")
        observed = result.get("observed")
        if isinstance(observed, list):
            execution_rows: list[dict] = []
            resource_rows: list[dict] = []
            for receipt in observed:
                if not isinstance(receipt, dict) or not isinstance(receipt.get("probe"), str):
                    continue
                probe = receipt["probe"]
                execution = snap.read_json(f"evals/results/{tag}/execution/{case}__{probe}.json", required=False)
                resources = snap.read_json(f"evals/results/{tag}/resources/{case}__{probe}.json", required=False)
                if isinstance(execution, dict):
                    execution_rows.append(execution)
                if isinstance(resources, dict):
                    resource_rows.append(resources)

            def measured_sum(rows: list[dict], field: str) -> int | float | None:
                if len(rows) != len(observed):
                    return None
                total: int | float = 0
                for row in rows:
                    value = row.get(field)
                    if not isinstance(value, (int, float)):
                        return None
                    total += value
                return total

            expected = {
                "isaac_wall_s": measured_sum(execution_rows, "gpu_wall_s"),
                "isaac_cpu_s": measured_sum(execution_rows, "cpu_s"),
                "isaac_simulator_steps": measured_sum(resource_rows, "simulator_steps"),
                "human_review_s": None,
                "gpu_device_busy_s": None,
            }
            for field, value in expected.items():
                if costs.get(field) != value:
                    reasons.append(f"baseline 비용 {field}가 실측 산출물 합계와 다르다")
    return {"state": "verified" if not reasons else "hold", "reasons": reasons, "result": result}


def _audit_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise BuildError("--audit은 저장소 안의 공개 산출물이어야 한다")
    return resolved.relative_to(ROOT).as_posix()


def build_summary(tag: str, audit_path: Path | None = None) -> dict:
    if not TAG_RE.fullmatch(tag):
        raise BuildError(f"--tag 형식이 잘못됐다: {tag!r}")
    snap = Snapshot()
    protocol = snap.read_json(PROTOCOL.as_posix())
    if not isinstance(protocol, dict) or protocol.get("schema") != "active_diagnosis_protocol_v1":
        raise BuildError("active_diagnosis_v1 protocol schema가 아니다")
    integrity_errors = code_integrity_errors(ROOT, protocol)
    if integrity_errors:
        raise BuildError("; ".join(integrity_errors))
    cases = _case_names(protocol)
    audit_rel = _audit_relative(audit_path)
    raw_audit = snap.read_json(audit_rel) if audit_rel else None
    if raw_audit is not None and not isinstance(raw_audit, dict):
        raise BuildError("독립 auditor 산출물이 JSON 객체가 아니다")
    audit: dict | None = raw_audit
    audits = _audit_records(audit)
    rows = []
    for case in cases:
        row = _public_case(case, tag, protocol, snap, audit, audits)
        row["baselines"] = {
            "fixed_order": _baseline(case, f"{tag}_fixed", "fixed_order", protocol, snap),
            "cheapest_discrimination": _baseline(
                case, f"{tag}_cheap", "cheapest_discrimination", protocol, snap
            ),
        }
        rows.append(row)
    spotlight = next((row["case"] for row in rows if row["proof"]), cases[0])
    return {
        "schema": "active_replay_summary_v1", "tag": tag,
        "claim": "검사 결과를 근거로 우선 가설을 바꾼 실행 기록",
        "scope": protocol.get("scope"), "registered_order": cases, "spotlight_case": spotlight,
        "spotlight_verified": any(row["proof"] for row in rows), "audit": audit_rel, "cases": rows,
        "disclaimers": [
            "한 사례의 판단 변경은 정확도나 일반화 성능을 뜻하지 않는다.",
            "approval_actor는 실행 승인 주체이며 사람 검토 시간을 뜻하지 않는다.",
            "wall time과 GPU device busy time은 다른 값이다. 미측정값은 0으로 바꾸지 않았다.",
            "같은 조건의 새 보류 세트 비교 전에는 AI 우월성이나 비용 절감을 주장하지 않는다.",
        ],
        "sources": [{"path": path, "sha256_lf": digest} for path, digest in sorted(snap.sources.items())],
    }


def _json_for_script(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_html(summary: dict) -> str:
    data = _json_for_script(summary)
    return f"""<!doctype html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Active diagnosis evidence replay</title>
<style>
:root{{--ink:#17202a;--muted:#627184;--line:#dce3ea;--soft:#f5f8fb;--blue:#1769e0;--hold:#a75b00;--ok:#087a55}}
*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 24px 56px}}h1{{font-size:clamp(30px,5vw,54px);line-height:1.08;margin:8px 0 14px;letter-spacing:-.035em}}
h2{{font-size:24px;margin:38px 0 14px}}p{{margin:7px 0}}.eyebrow{{color:var(--blue);font-weight:750}}.lede{{font-size:19px;max-width:850px;color:#3e4b59}}
.notice{{border-left:4px solid var(--blue);background:var(--soft);padding:13px 16px;margin:22px 0}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card,.stage,.panel{{border:1px solid var(--line);border-radius:14px;background:#fff;padding:16px}}.kpi b{{display:block;font-size:26px;margin-top:6px}}.muted{{color:var(--muted)}}
.casebar,.controls,.actions{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}}button{{border:1px solid #b9c5d1;background:#fff;color:var(--ink);border-radius:9px;padding:9px 12px;font:inherit;cursor:pointer}}
button[aria-pressed=\"true\"],button:hover{{border-color:var(--blue);color:var(--blue);background:#f2f7ff}}.stage{{min-height:255px}}.stage h3{{margin:0 0 12px}}.badge{{display:inline-block;border-radius:999px;padding:3px 9px;font-weight:750;font-size:13px;background:#fff4e7;color:var(--hold)}}
.badge.ok{{background:#eaf8f2;color:var(--ok)}}ol{{padding-left:22px}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:7px 14px}}dt{{color:var(--muted)}}dd{{margin:0;overflow-wrap:anywhere}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:var(--soft)}}.tablewrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}}
.foot{{margin-top:28px;color:var(--muted);font-size:14px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:520px){{.grid{{grid-template-columns:1fr}}main{{padding:24px 14px}}}}
</style></head><body><main>
<p class=\"eyebrow\">OFFLINE EVIDENCE REPLAY</p><h1>검사를 고르고, 실제 측정을 읽고, 판단을 갱신한 기록</h1>
<p class=\"lede\">저장된 모델 판단과 Isaac 실행 receipt를 한 화면에서 재생합니다. 검증할 수 없는 연결은 PASS로 꾸미지 않고 HOLD로 표시합니다.</p>
<div class=\"notice\" id=\"scope\"></div><section class=\"grid\" id=\"kpis\"></section>
<h2>사례 재생</h2><div class=\"casebar\" id=\"casebar\"></div><div class=\"controls\"><button id=\"prev\">이전 단계</button><button id=\"next\">다음 단계</button><span class=\"muted\" id=\"position\"></span></div>
<section class=\"stage\" id=\"stage\"></section><div class=\"actions\"><span class=\"muted\">개념 재생:</span><button data-action=\"검사 승인\">검사 승인</button><button data-action=\"보류\">보류</button><button data-action=\"추가 근거 요청\">추가 근거 요청</button><span id=\"actionState\" class=\"muted\">실제 실행은 일어나지 않습니다.</span></div>
<h2>전체 8개 사례와 공정 baseline</h2><div class=\"tablewrap\"><table><thead><tr><th>사례</th><th>Active</th><th>모델 호출 / wall</th><th>Active Isaac wall</th><th>Fixed order / Isaac wall</th><th>Cheapest discrimination / Isaac wall</th><th>GPU busy</th></tr></thead><tbody id=\"cohort\"></tbody></table></div>
<div class=\"foot\" id=\"notes\"></div>
<script id=\"snapshot\" type=\"application/json\">{data}</script><script>
"use strict";
const S=JSON.parse(document.getElementById("snapshot").textContent);let ci=Math.max(0,S.registered_order.indexOf(S.spotlight_case)),si=0;
const $=id=>document.getElementById(id),el=(tag,text,cls)=>{{const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n}};
const fmt=v=>v===null||v===undefined?"미측정":typeof v==="number"?new Intl.NumberFormat("ko-KR",{{maximumFractionDigits:2}}).format(v):String(v);
const sec=v=>v===null||v===undefined?"미측정":fmt(v)+" s";
const state=b=>b&&b.state==="verified"?"VERIFIED":"HOLD";const current=()=>S.cases[ci];
function renderKpis(){{const box=$("kpis");box.replaceChildren();const verified=S.cases.filter(x=>x.proof).length,calls=S.cases.reduce((n,x)=>n+(x.result?.costs?.model_calls||0),0);[["검증 사례",verified+" / 8"],["HOLD",8-verified],["기록된 모델 호출",calls],["spotlight",S.spotlight_verified?S.spotlight_case:"검증 사례 없음"]].forEach(([a,b])=>{{const c=el("div",undefined,"card kpi");c.append(el("span",a,"muted"),el("b",b));box.append(c)}})}}
function renderCases(){{const bar=$("casebar");bar.replaceChildren();S.cases.forEach((row,i)=>{{const b=el("button",row.case+" · "+state(row));b.setAttribute("aria-pressed",String(i===ci));b.addEventListener("click",()=>{{ci=i;si=0;render()}});bar.append(b)}})}}
function pair(dl,k,v){{dl.append(el("dt",k),el("dd",fmt(v)))}}function list(items){{const ol=el("ol");(items?.length?items:["기록 없음"]).forEach(x=>ol.append(el("li",x)));return ol}}
function renderStage(){{const row=current(),r=row.result||{{}},d=r.decisions||[],o=r.observed||[],ix=row.contrast_index??0,before=d[ix],after=d[ix+1],obs=o[ix],box=$("stage");box.replaceChildren();const badge=el("span",state(row),"badge "+(row.proof?"ok":""));box.append(badge);const titles=["1. AI 검사 선택","2. 승인된 실행","3. 실제 측정","4. 판단 갱신"];box.append(el("h3",titles[si]));const dl=el("dl");
if(si===0){{pair(dl,"측정 전 1위",before?.ranking?.[0]);pair(dl,"선택 검사",before?.probe);pair(dl,"행동",before?.action);pair(dl,"이유",before?.reason)}}
if(si===1){{pair(dl,"승인 주체",r.approval_actor);pair(dl,"승인 신원 검증",r.approval_identity_attested);pair(dl,"사람 검토 시간",r.human_review_s);pair(dl,"request",obs?.request_id);pair(dl,"실행 결과 코드",obs?.exit_code)}}
if(si===2){{pair(dl,"검사",obs?.probe);pair(dl,"판정",obs?.outcome);pair(dl,"측정",obs?.measurement?JSON.stringify(obs.measurement):null);pair(dl,"Isaac wall (s)",r.costs?.isaac_wall_s);pair(dl,"GPU device busy (s)",r.costs?.gpu_device_busy_s)}}
if(si===3){{pair(dl,"갱신 1위",after?.ranking?.[0]);pair(dl,"행동",after?.action);pair(dl,"receipt 인용",row.contrast?.latest_receipt_cited);pair(dl,"관측으로 이전 1위 기각",row.contrast?.changed_leader_rejected_by_actual_observation);box.append(el("p",row.proof?S.claim:"이 사례는 증명 조건을 충족하지 않아 HOLD입니다.","notice"));if(!row.proof)box.append(list(row.reasons))}}
box.append(dl);$("position").textContent=(si+1)+" / 4 · "+row.case}}
function renderTable(){{const body=$("cohort");body.replaceChildren();S.cases.forEach(row=>{{const r=row.result||{{}},f=row.baselines.fixed_order,c=row.baselines.cheapest_discrimination,tr=el("tr");[row.case,state(row),fmt(r.costs?.model_calls)+" / "+sec(r.costs?.model_wall_s),sec(r.costs?.isaac_wall_s),state(f)+" / "+sec(f.result?.costs?.isaac_wall_s),state(c)+" / "+sec(c.result?.costs?.isaac_wall_s),sec(r.costs?.gpu_device_busy_s)].forEach(v=>tr.append(el("td",v)));body.append(tr)}})}}
function render(){{renderCases();renderStage()}}$("prev").addEventListener("click",()=>{{si=(si+3)%4;renderStage()}});$("next").addEventListener("click",()=>{{si=(si+1)%4;renderStage()}});document.querySelectorAll("[data-action]").forEach(b=>b.addEventListener("click",()=>$("actionState").textContent=b.dataset.action+"을 선택했습니다. 개념 재생이며 실행 요청은 보내지 않았습니다."));
$("scope").textContent="표시 범위: "+fmt(S.scope)+" · "+S.claim;const notes=$("notes");S.disclaimers.forEach(x=>notes.append(el("p",x)));renderKpis();renderTable();render();
</script></main></body></html>"""


def write_outputs(summary: dict, out: Path) -> tuple[Path, Path]:
    out = out.resolve()
    if out.suffix.lower() != ".html":
        raise BuildError("--out은 .html 파일이어야 한다")
    summary_path = out.with_suffix(".json")
    if out.exists() or summary_path.exists():
        raise BuildError("HTML 또는 요약 JSON이 이미 있다. 기존 스냅샷은 덮어쓰지 않는다")
    out.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        with out.open("x", encoding="utf-8") as handle:
            handle.write(render_html(summary))
    except Exception:
        summary_path.unlink(missing_ok=True)
        raise
    return out, summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", type=Path, help="독립 auditor의 공개 JSON 산출물")
    args = parser.parse_args(argv)
    try:
        summary = build_summary(args.tag, args.audit)
        html_path, json_path = write_outputs(summary, args.out)
    except (BuildError, OSError) as exc:
        print(f"active replay 입력 오류: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"html": str(html_path), "summary": str(json_path),
                      "verified": sum(case["proof"] for case in summary["cases"]),
                      "holds": sum(not case["proof"] for case in summary["cases"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
